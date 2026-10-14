"""
Teach the text reader to read pill imprints: a LoRA adapter on Florence-2-large (core/imprint/lora.py), switched on
for imprint reads only. Zero-shot it misses the imprint on about half of the photos (engraved text, odd angles).

Data: C3PI everyday photos of the *training* medicines (the split's `train`), framed as the API frames them (window
1.55, the central 92%), whose imprint C3PI rated Clear or Partial: about 19,000 photos of 1,740 medicines. Validation
and test medicines are never used, so the reads measured on them are on medicines the reader never saw.
The recorded imprint covers both sides, and a photo shows one side at any angle, so the target isn't known exactly.
Each step therefore scores, without gradients, every rotation (0/90/180/270) against every run of consecutive
imprint tokens ("TEVA", "5729", "TEVA\n5729"), keeps the best-fitting pair, and trains on that pair (hard EM), with
a small extra rotation and colour jitter.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\train_imprint_reader.py --prepare      extract the framed crops once (~5 min)
  .venv\\Scripts\\python scripts\\train_imprint_reader.py [--epochs 2]   train (~3 h for 2 epochs on an RTX 4080)
  .venv\\Scripts\\python scripts\\train_imprint_reader.py --export --name <name>   copy the adapter to
      outputs/serving/imprint_adapter.pt, which scripts\\deploy_hf.py assets uploads with the model
Output: outputs/imprint_reader/<name>/adapter.pt (+ train_log.csv, eval.json: own and other imprint reads on 400
validation photos, with and without the adapter).
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.imprint.lora import SCOPE, add_lora, lora_state, set_lora  # noqa: E402
from epills.core.imprint.match import ImprintIndex, compact, read_hits  # noqa: E402
from epills.core.imprint.ocr import GPU_MODEL, READ_CENTRE, ROTATIONS, ImprintReader  # noqa: E402

FT = EPILLS_ROOT / "data/interim/imprint_ft"
RAW = EPILLS_ROOT / "data/raw/c3pi/images"
NL, EOS, START, BOS, PAD = 50118, 2, 2, 0, 1  # Florence-2's newline, end, decoder start, <s>, pad tokens
EVAL_PHOTOS = 400


def table():
    """
    Training photos with a readable imprint (Clear or Partial), training photos with nothing to read (C3PI's
    "Can not Identify", or no imprint: the reader should write nothing), and validation photos; with their boxes.
    """
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    meta = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/metadata.csv", dtype=str, keep_default_na=False,
                       usecols=["path", "rating_imprint"])
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    boxes = crops.drop_duplicates("crop_path")[["crop_path", "width", "height", "x0", "y0", "x1", "y1"]]
    m = m.merge(meta, on="path", how="left").merge(boxes, on="crop_path", how="left")
    train = m[(m.split == "train") & (m.imprint != "") & m.rating_imprint.isin(["Clear", "Partial"])]
    blank = m[(m.split == "train") & ((m.rating_imprint == "Can not Identify") | (m.imprint == ""))]
    val = m[m.split.isin(["val_enrolled", "val_unknown"]) & (m.imprint != "")]
    return train.reset_index(drop=True), blank.reset_index(drop=True), val.reset_index(drop=True)


def _extract(row):
    from epills.core.inference.preprocess import crop_window
    out = FT / "crops" / Path(row["crop_path"]).with_suffix(".jpg")
    if out.exists():
        return
    img = Image.open(RAW / row["path"]).convert("RGB")
    size = (int(row["width"]), int(row["height"]))
    if img.size != size and img.size[::-1] == size:
        img = img.transpose(Image.Transpose.ROTATE_90)
    out.parent.mkdir(parents=True, exist_ok=True)
    crop_window(img, tuple(float(row[k]) for k in ("x0", "y0", "x1", "y1")), 1.55).save(out, quality=95)


def prepare():
    train, blank, val = table()
    rows = pd.concat([train, blank, val]).to_dict("records")
    with ProcessPoolExecutor(12) as pool:
        for i, _ in enumerate(pool.map(_extract, rows, chunksize=32)):
            if i % 1000 == 999:
                print(f"\r{i + 1:,} / {len(rows):,}", end="", flush=True)
    print(f"\n{len(train):,} training, {len(blank):,} nothing-to-read and {len(val):,} validation crops in "
          f"{FT / 'crops'}")


def crop_of(row):
    return Image.open(FT / "crops" / Path(row.crop_path).with_suffix(".jpg")).convert("RGB")


def runs(imprint):
    """Runs of consecutive recorded tokens, one per line: 'TEVA;5729' -> TEVA, 5729, TEVA\\n5729."""
    toks = [compact(t) for t in str(imprint).split(";") if compact(t)]
    out = {"\n".join(toks[i:j]) for i in range(len(toks)) for j in range(i + 1, min(len(toks), i + 3) + 1)}
    return sorted(s for s in out if len(s.replace("\n", "")) >= 2)


class Trainer:
    def __init__(self, r, alpha, lr, dtype=torch.bfloat16):
        self.reader = ImprintReader(GPU_MODEL)
        self.model, self.proc = self.reader.model.to(dtype), self.reader.processor
        self.reader.dtype = self.dtype = dtype
        self.tok = self.proc.tokenizer
        add_lora(self.model, r, alpha, dropout=0.05)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.params:
            p.data = p.data.float()
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.dev = self.reader.device

    def encode(self, images):
        x = self.proc(text=["<OCR>"] * len(images), images=images, return_tensors="pt")
        ids, px = x["input_ids"].to(self.dev), x["pixel_values"].to(self.dev, self.dtype)
        inner = self.model.model
        e = inner.get_input_embeddings()(ids)
        f = inner.get_image_features(px).pooler_output.to(e.dtype)
        e = e.masked_scatter(inner.get_placeholder_mask(ids, inputs_embeds=e, image_features=f), f)
        return inner.language_model.encoder(inputs_embeds=e).last_hidden_state

    def nll(self, enc, texts):
        """Mean token NLL of each text (+ newline, end) given encoder states (one row per text)."""
        from transformers.modeling_outputs import BaseModelOutput
        seqs = [self.tok(t, add_special_tokens=False).input_ids + [NL, EOS] for t in texts]
        L = max(len(s) for s in seqs)
        dec = torch.full((len(seqs), L + 1), PAD, dtype=torch.long)
        lab = torch.full((len(seqs), L + 1), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            dec[i, :len(s) + 1] = torch.tensor([START, BOS] + s[:-1])
            lab[i, 1:len(s) + 1] = torch.tensor(s)
        dec, lab = dec.to(self.dev), lab.to(self.dev)
        out = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc), decoder_input_ids=dec,
                         decoder_attention_mask=(dec != PAD).long())
        lp = torch.log_softmax(out.logits.float(), -1)
        mask = lab != -100
        g = lp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        return -(g.sum(1) / mask.sum(1))

    def step(self, crops, imprints, negatives=0.0, blanks=()):
        """negatives > 0: also train one wrong rotation per photo, and the `blanks` (photos with nothing readable, at
        any rotation), to read nothing (weight `negatives`), so the reader doesn't write an imprint it remembers."""
        regions = [self.reader.pill_region(c, READ_CENTRE) for c in crops]
        cands = [runs(i) for i in imprints]
        # E-step: the rotation and token run that fit each photo best (a mild preference for longer runs)
        with torch.no_grad():
            set_lora(self.model, True)
            enc = self.encode([r.rotate(a, expand=True) for r in regions for a in ROTATIONS])
            pairs = [(b, k, t) for b, cs in enumerate(cands) for k in range(len(ROTATIONS)) for t in cs]
            scores = []
            for s in range(0, len(pairs), 96):
                chunk = pairs[s:s + 96]
                rows = torch.tensor([b * len(ROTATIONS) + k for b, k, _ in chunk], device=self.dev)
                scores.append(self.nll(enc[rows], [t for _, _, t in chunk]).cpu())
            scores = torch.cat(scores) - 0.05 * torch.tensor([t.count("\n") + 1 for _, _, t in pairs])
        best = {}
        for (b, k, t), s in zip(pairs, scores.tolist()):
            if b not in best or s < best[b][0]:
                best[b] = (s, k, t)
        # M-step: train on the chosen pair, with a small extra rotation and colour jitter
        def jitter(img, angle):
            img = img.rotate(angle + random.uniform(-12, 12), expand=True, fillcolor=(128, 128, 128))
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            return ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

        images, texts, wrong = [], [], []
        for b, (_, k, t) in sorted(best.items()):
            images.append(jitter(regions[b], ROTATIONS[k]))
            texts.append(t)
            if negatives:
                wrong.append(jitter(regions[b], ROTATIONS[(k + random.choice((1, 2, 3))) % len(ROTATIONS)]))
        if negatives:
            wrong += [jitter(self.reader.pill_region(c, READ_CENTRE), random.choice(ROTATIONS)) for c in blanks]
        self.model.train()
        loss = self.nll(self.encode(images), texts).mean()
        if wrong:  # the empty reading: just the newline and the end
            loss = loss + negatives * self.nll(self.encode(wrong), [""] * len(wrong)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.model.eval()
        return float(loss), float(np.mean([v[0] for v in best.values()]))


class Evaluator:
    """
    Checks on n validation photos the catalogue doesn't hold, with and without the adapter: own and another pill's
    imprint read (counted as served), and what decides it, the API's catalogue identify: how often the first
    candidate is right with those reads, at a few re-ranking weights (large_lora_v1 read more imprints but made
    this worse, by writing imprints it remembered).
    """
    ALPHAS = (0.05, 0.1, 0.2, 0.3)

    def __init__(self, val, weight, n=EVAL_PHOTOS, seed=0):
        from epills.core.catalogue.store import Catalogue
        self.cat, self.weight = Catalogue(EPILLS_ROOT / "data/processed/catalogue"), weight
        m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
        m["pill_id"] = m.ndc9 + "_" + m.part
        self.group_of = m.groupby("pill_id").label.first().to_dict()
        run = EPILLS_ROOT / "outputs/runs/stage2_run4"
        cal = json.loads((run / "calibration.json").read_text())
        z = np.load(run / f"val_embeddings_w{cal['window']:g}.npz")
        order = pd.concat([m[m.split == "val_enrolled"], m[m.split == "val_unknown"]]).crop_path.tolist()
        emb = np.concatenate([z["known_emb"], z["unknown_emb"]])
        crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
        crops = crops[(crops.found == "1") & (crops.crop_path != "")].copy()
        crops["pill_id"] = crops.ndc9 + "_" + crops.part
        crops = crops.sample(frac=1, random_state=0)  # as build_catalogue.py: the catalogue's own photos
        crops["rank"] = crops.groupby("pill_id").cumcount()
        held = set(crops[crops["rank"] < 8].crop_path)
        ok = val[~val.crop_path.isin(held) & (val.ndc9 + "_" + val.part).isin(self.cat.attrs)]
        self.sample = ok.sample(n, random_state=seed)
        pos = {p: i for i, p in enumerate(order)}
        self.emb = [emb[pos[p]] for p in self.sample.crop_path]
        rng = np.random.default_rng(seed)
        self.others = [rng.choice(val.imprint[val.label != lab].to_numpy()) for lab in self.sample.label]
        self.zero = None

    def reads(self, trainer, adapter):
        set_lora(trainer.model, adapter)
        trainer.reader.dtype, trainer.reader.beams = trainer.dtype, 3
        rows = list(self.sample.itertuples())
        out = [rd for s in range(0, len(rows), 16) for rd in trainer.reader.read([crop_of(r) for r in rows[s:s + 16]])]
        set_lora(trainer.model, True)
        return out

    def score(self, reads):
        rows = list(self.sample.itertuples())
        res = {"own_read": float(np.mean([bool(read_hits(rd, {0: r.imprint}, self.weight, 0.5))
                                          for r, rd in zip(rows, reads)])),
               "other_read": float(np.mean([bool(read_hits(rd, {0: o}, self.weight, 0.5))
                                            for rd, o in zip(reads, self.others)]))}
        for a in (0.0,) + self.ALPHAS:
            right = []
            for r, e, rd in zip(rows, self.emb, reads):
                top = self.cat.identify(e[None], top_k=1, reads=rd if a else None, alpha=a)[0][0]["pill_id"]
                right.append(self.group_of.get(top, top) == self.group_of.get(r.ndc9 + "_" + r.part))
            res[f"identify_top1_alpha_{a:g}"] = float(np.mean(right))
        return res

    def __call__(self, trainer):
        if self.zero is None:
            self.zero = self.score(self.reads(trainer, False))
        out = {"zero_shot": self.zero, "adapter": self.score(self.reads(trainer, True))}
        for name, r in out.items():
            best = max(self.ALPHAS, key=lambda a: r[f"identify_top1_alpha_{a:g}"])
            print(f"{name}: own read {r['own_read']:.1%}, another's {r['other_read']:.1%}; identify first right "
                  f"{r[f'identify_top1_alpha_{best:g}']:.1%} (alpha {best}; look only "
                  f"{r['identify_top1_alpha_0']:.1%})", flush=True)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--name", default="large_lora_v1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="train on this many photos only (smoke test)")
    parser.add_argument("--negatives", type=float, default=0.0,
                        help="weight of reading nothing at a wrong rotation (0: off, as large_lora_v1)")
    parser.add_argument("--blanks", type=float, default=0.0,
                        help="nothing-to-read photos per readable one in each batch (needs --negatives)")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.export:
        saved = torch.load(EPILLS_ROOT / "outputs/imprint_reader" / args.name / "adapter.pt", weights_only=True)
        saved["config"]["name"] = args.name
        target = EPILLS_ROOT / "outputs/serving/imprint_adapter.pt"
        torch.save(saved, target)
        print(f"exported {args.name} to {target}")
        return
    random.seed(0), torch.manual_seed(0)
    train, blank, val = table()
    if args.limit:
        train = train.sample(args.limit, random_state=0)
    blank = blank.sample(frac=1, random_state=0)
    n_blank = round(args.batch * args.blanks)
    out = EPILLS_ROOT / "outputs/imprint_reader" / args.name
    out.mkdir(parents=True, exist_ok=True)
    import sqlite3
    con = sqlite3.connect(EPILLS_ROOT / "data/processed/catalogue/catalogue.sqlite")
    weight = ImprintIndex([r[0] or "" for r in con.execute("select imprint from products")]).weight
    trainer = Trainer(args.r, 2 * args.r, args.lr)
    evaluator = Evaluator(val, weight)
    steps = args.epochs * (len(train) // args.batch)
    warm = max(1, steps // 20)  # 5% warm-up, then cosine decay to 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        trainer.opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, i / max(1, steps)))))
    log, t0, n, results = [], time.time(), 0, {}
    print(f"{len(train):,} training photos of {train.pill_type.nunique()} medicines; {steps} steps", flush=True)
    for epoch in range(args.epochs):
        order = train.sample(frac=1, random_state=epoch)
        for s in range(0, len(order) - args.batch + 1, args.batch):
            chunk = order.iloc[s:s + args.batch]
            blanks = [crop_of(blank.iloc[(n * n_blank + j) % len(blank)]) for j in range(n_blank)]
            loss, fit = trainer.step([crop_of(r) for r in chunk.itertuples()], chunk.imprint.tolist(), args.negatives,
                                     blanks)
            sched.step()
            n += 1
            log.append({"step": n, "epoch": epoch, "loss": loss, "best_fit_nll": fit})
            if n % 50 == 0:
                recent = pd.DataFrame(log[-50:])
                print(f"epoch {epoch} step {n}/{steps}: loss {recent.loss.mean():.3f}, best-fit NLL "
                      f"{recent.best_fit_nll.mean():.3f}, {(time.time() - t0) / n:.2f} s/step", flush=True)
        saved = {"config": {"r": args.r, "alpha": 2 * args.r, "scope": list(SCOPE), "base": GPU_MODEL,
                            "epochs": epoch + 1, "negatives": args.negatives, "blanks": args.blanks},
                 "state": lora_state(trainer.model)}
        torch.save(saved, out / f"adapter_e{epoch + 1}.pt")
        torch.save(saved, out / "adapter.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        results[f"epoch_{epoch + 1}"] = evaluator(trainer)
        (out / "eval.json").write_text(json.dumps({"photos": EVAL_PHOTOS, **results, "train_photos": len(train),
                                                   "epochs": args.epochs, "lr": args.lr, "r": args.r,
                                                   "negatives": args.negatives, "blanks": args.blanks}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
