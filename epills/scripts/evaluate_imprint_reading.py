"""
How well the text reader reads pill imprints, and which reading settings read more (imprint accuracy work).

1. Failure analysis (cached reads of all validation crops): share of photos where a token of the pill's own imprint
   is read, by imprint type (printed, debossed), C3PI's imprint rating, shape and size.
2. Variants on one fixed sample of validation crops (never test data), each scored by
     own_read    a token of the pill's own imprint is read (what we want up)
     other_read  a token of a random other pill's imprint is read (false evidence: must stay low)
     ms_per_pill reading time on this GPU
   Variants: the served setting (centre 70%, 4 rotations, 3 beams), greedy decoding, contrast enhancement (CLAHE),
   CLAHE in grey, 8 rotations, a tighter or wider pill region, and Florence-2-large.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_imprint_reading.py [--sample 400] [--variants base,greedy,clahe]
Output: data/interim/ocr/imprint_reading_variants[_served].json (variants from earlier runs are kept; --analysis
large_c92 analyses the large model's cached reads, scripts/read_imprints_cache.py, framed as served)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.imprint.match import imprint_tokens, read_hits  # noqa: E402

CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
CACHE = EPILLS_ROOT / "data/interim/ocr/Florence-2-base/validation_all.json"
VARIANTS = {  # name: (model, centre, rotations, beams, preprocessing)
    "base": ("base", 0.7, (0, 90, 180, 270), 3, None),
    "greedy": ("base", 0.7, (0, 90, 180, 270), 1, None),
    "clahe": ("base", 0.7, (0, 90, 180, 270), 3, "clahe"),
    "clahe_grey": ("base", 0.7, (0, 90, 180, 270), 3, "clahe_grey"),
    "rot8": ("base", 0.7, tuple(range(0, 360, 45)), 3, None),
    "tight": ("base", 0.55, (0, 90, 180, 270), 3, None),
    "wide": ("base", 0.85, (0, 90, 180, 270), 3, None),
    "clahe_greedy": ("base", 0.7, (0, 90, 180, 270), 1, "clahe"),
    "large": ("large", 0.7, (0, 90, 180, 270), 3, None),
    "c78": ("base", 0.78, (0, 90, 180, 270), 3, None),
    "c92": ("base", 0.92, (0, 90, 180, 270), 3, None),
    "c100": ("base", 1.0, (0, 90, 180, 270), 3, None),
    "c85_greedy": ("base", 0.85, (0, 90, 180, 270), 1, None),
    "large_c85": ("large", 0.85, (0, 90, 180, 270), 3, None),
    "large_c92": ("large", 0.92, (0, 90, 180, 270), 3, None),
    "c92_greedy": ("base", 0.92, (0, 90, 180, 270), 1, None),
    "mosaic_c92": ("base", 0.92, (0, 90, 180, 270), 3, "mosaic"),
    "mosaic_c92_greedy": ("base", 0.92, (0, 90, 180, 270), 1, "mosaic"),
    "large_mosaic_c92": ("large", 0.92, (0, 90, 180, 270), 3, "mosaic"),
}


def enhance(img, how):
    """CLAHE on the lightness channel: brings out debossed (engraved, same-colour) imprints."""
    if how is None:
        return img
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    if how == "clahe_grey":
        g = lab[..., 0]
        return Image.fromarray(np.stack([g, g, g], axis=2))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class Reader:
    def __init__(self, size):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        name = f"florence-community/Florence-2-{size}"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Florence2ForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def read_mosaic(self, crops, centre, beams, batch=16):
        """The 4 rotations of each pill tiled 2 x 2 into one image: one encoder pass instead of four. Text regions are
        mapped back to their tile, so reads of different rotations never run together."""
        images = []
        for c in crops:
            w, h = c.size
            dx, dy = w * (1 - centre) / 2, h * (1 - centre) / 2
            region = c.convert("RGB").crop((int(dx), int(dy), int(w - dx), int(h - dy))).resize((384, 384))
            tile = Image.new("RGB", (768, 768), region.getpixel((0, 0)))
            for k, a in enumerate((0, 90, 180, 270)):
                tile.paste(region.rotate(a), ((k % 2) * 384, (k // 2) * 384))
            images.append(tile)
        out = []
        for i in range(0, len(images), batch):
            x = self.processor(text=["<OCR_WITH_REGION>"] * len(images[i:i + batch]), images=images[i:i + batch],
                               return_tensors="pt").to("cuda", torch.float16)
            ids = self.model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=64,
                                      num_beams=beams)
            for raw in self.processor.batch_decode(ids, skip_special_tokens=False):
                r = self.processor.post_process_generation(raw, task="<OCR_WITH_REGION>", image_size=(768, 768))
                r = r["<OCR_WITH_REGION>"]
                per_tile = ["", "", "", ""]
                for quad, text in zip(r["quad_boxes"], r["labels"]):
                    cx, cy = sum(quad[0::2]) / 4, sum(quad[1::2]) / 4
                    k = int(cx >= 384) + 2 * int(cy >= 384)
                    per_tile[k] = (per_tile[k] + " " + text.replace("</s>", "")).strip()
                out.append(per_tile)
        return out

    @torch.no_grad()
    def read(self, crops, centre, rotations, beams, prep, batch=32):
        if prep == "mosaic":
            return self.read_mosaic(crops, centre, beams)
        images = []
        for c in crops:
            w, h = c.size
            dx, dy = w * (1 - centre) / 2, h * (1 - centre) / 2
            region = enhance(c.convert("RGB").crop((int(dx), int(dy), int(w - dx), int(h - dy))), prep)
            images += [region.rotate(a, expand=True, fillcolor=region.getpixel((0, 0))) for a in rotations]
        texts = []
        for i in range(0, len(images), batch):
            x = self.processor(text=["<OCR>"] * len(images[i:i + batch]), images=images[i:i + batch],
                               return_tensors="pt").to("cuda", torch.float16)
            out = self.model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=20,
                                      num_beams=beams)
            texts += self.processor.batch_decode(out, skip_special_tokens=True)
        n = len(rotations)
        return [texts[i:i + n] for i in range(0, len(texts), n)]


def score(reads, imprints, others):
    own = np.mean([bool(read_hits(r, {0: i})) for r, i in zip(reads, imprints)])
    other = np.mean([bool(read_hits(r, {0: o})) for r, o in zip(reads, others)])
    return float(own), float(other)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--served-framing", action="store_true", help="crop as the API does (window 1.55)")
    parser.add_argument("--analysis", default="base_c70", help="cached reads for part 1: base_c70 or large_c92")
    args = parser.parse_args()

    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    v = m[m.split.isin(["val_enrolled", "val_unknown"]) & (m.imprint != "")]
    v = v.merge(crops[["crop_path", "imprint_type", "rating_imprint"]], on="crop_path", how="left")
    v = v[[len(imprint_tokens(i)) > 0 for i in v.imprint]].reset_index(drop=True)

    # 1. failure analysis on the cached reads
    cache = json.loads((CACHE if args.analysis == "base_c70" else
                        EPILLS_ROOT / "data/interim/ocr" / args.analysis / "validation_all.json").read_text())
    v["reads"] = [cache.get(str(CROPS / p)) for p in v.crop_path]
    a = v[v.reads.notna()].copy()
    a["hit"] = [bool(read_hits(r, {0: i})) for r, i in zip(a.reads, a.imprint)]
    a["size_band"] = pd.cut(pd.to_numeric(a.size_mm, errors="coerce"), [0, 7, 10, 14, 40],
                       labels=["< 7 mm", "7-10 mm", "10-14 mm", "> 14 mm"])
    print(f"cached validation reads: {len(a):,} photos with a readable-token imprint; own token read "
          f"{a.hit.mean():.1%}")
    analysis = {"photos": int(len(a)), "own_read": float(a.hit.mean())}
    for col in ("imprint_type", "rating_imprint", "shape", "size_band"):
        t = a.groupby(col, observed=True).hit.agg(["mean", "size"])
        t = t[t["size"] >= 30].sort_values("size", ascending=False)
        analysis[col] = {str(k): {"own_read": round(float(r["mean"]), 3), "photos": int(r["size"])}
                         for k, r in t.iterrows()}
        print(f"\nby {col}:\n" + t.round(3).to_string())

    # 2. variants on one sample
    rng = np.random.default_rng(0)
    s = v.sample(min(args.sample, len(v)), random_state=0).reset_index(drop=True)
    others = [s.imprint[j] for j in rng.permutation(len(s))]
    if args.served_framing:  # as the API frames a pill: from the photo, a window of 1.55 x the pill box's long side
        from PIL import ImageOps
        from epills.core.inference.preprocess import crop_window
        boxes = crops.set_index("crop_path")[["path", "x0", "y0", "x1", "y1", "width", "height"]]
        images = []
        for p in s.crop_path:
            b = boxes.loc[p]
            img = Image.open(EPILLS_ROOT / "data/raw/c3pi/images" / b.path).convert("RGB")
            if img.size != (int(b.width), int(b.height)) and img.size[::-1] == (int(b.width), int(b.height)):
                img = img.transpose(Image.Transpose.ROTATE_90)
            images.append(crop_window(img, tuple(float(b[k]) for k in ("x0", "y0", "x1", "y1")), 1.55))
    else:
        images = [Image.open(CROPS / p) for p in s.crop_path]
    results, readers = {}, {}
    for name in args.variants.split(","):
        size, centre, rotations, beams, prep = VARIANTS[name]
        readers.setdefault(size, Reader(size))
        readers[size].read(images[:8], centre, rotations, beams, prep)  # warm-up
        torch.cuda.synchronize()
        t = time.time()
        reads = readers[size].read(images, centre, rotations, beams, prep)
        torch.cuda.synchronize()
        ms = (time.time() - t) / len(images) * 1000
        own, other = score(reads, s.imprint, others)
        by_type = {k: round(float(np.mean([bool(read_hits(r, {0: i})) for r, i, ty in
                                           zip(reads, s.imprint, s.imprint_type) if ty == k])), 3)
                   for k in ("PRINTED", "DEBOSSED") if (s.imprint_type == k).any()}
        results[name] = {"own_read": round(own, 3), "other_read": round(other, 3), "ms_per_pill": round(ms, 1),
                         "by_type": by_type, "setting": {"model": size, "centre": centre,
                                                         "rotations": len(rotations), "beams": beams, "prep": prep}}
        print(f"{name:13s} own {own:.1%}  other {other:.1%}  {ms:6.0f} ms/pill  {by_type}", flush=True)
    out = EPILLS_ROOT / f"data/interim/ocr/imprint_reading_variants{'_served' if args.served_framing else ''}.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    analyses = old.get("analyses", {}) | {args.analysis: analysis}
    out.write_text(json.dumps({"analyses": analyses, "sample": int(len(s)),
                               "variants": old.get("variants", {}) | results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
