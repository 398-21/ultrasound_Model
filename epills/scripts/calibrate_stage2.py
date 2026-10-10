"""
P5: open-set calibration of a trained Stage 2 model, on validation data only (never on a test set).

Simulates patients (epills.training.eval.openset): galleries of G medications from val_enrolled with k enrolment
photos each; val_unknown photos play pills that aren't the patient's. Chosen here:
  threshold  the 95th percentile of the unknown pills' s1, so 95% of unknown pills are rejected (the plan's target)
  margin     the largest margin that asks for a flip on at most --max-flip of the patient's own pills; it turns
             close calls (most wrong matches) into flip requests

With --windows, the crop contract's window is chosen here too, on validation: every validation crop is re-cut the
way the API cuts a photo (a tight pill box from the stand-in finder, then a square window w x its long side), for
each w; the window with the most correct matches at 95% unknown rejection is kept, and the threshold and margin are
calibrated on crops cut with it. The window is saved with the calibration and travels with the model (export).

Outputs, in the run folder:
  calibration.json        threshold, margin and the operating point (the API loads this next to best.pt)
  calibration_table.csv   every margin tried, for gallery sizes 5, 10, 15 and k = 3 and 5 enrolment photos
  val_embeddings*.npz     the validation embeddings (cache; one per window)

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\calibrate_stage2.py --run outputs\\runs\\stage2_run2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.inference.pill_finder import find_pills  # noqa: E402
from epills.core.inference.preprocess import crop_window  # noqa: E402
from epills.core.models.hybrid import HybridPillNet  # noqa: E402
from epills.training.datasets.pills import PillCrops, Vocab, eval_transform, load_manifest  # noqa: E402
from epills.training.eval.openset import operating_point, simulate  # noqa: E402
from epills.training.eval.retrieval import embed  # noqa: E402

UNKNOWN_REJECTION = 0.95
MARGINS = np.round(np.arange(0, 0.3001, 0.005), 3)


class Reframed(Dataset):
    """
    Validation crops re-cut like the API cuts a photo: the stand-in finder's tight pill box (these crops have plain
    backgrounds), then the crop contract with `window`. If the finder fails or its box touches the crop's edge, the
    crop keeps its original framing.
    """

    def __init__(self, df, images, window):
        self.paths = [images / p for p in df.crop_path]
        index = {g: i for i, g in enumerate(sorted(df.label.unique()))}
        self.labels = df.label.map(index).to_numpy()
        self.window, self.transform = window, eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        boxes, n = find_pills(img), img.size[0]
        if boxes and min(boxes[0][:2]) > 2 and max(boxes[0][2:]) < n - 2:
            img = crop_window(img, boxes[0], self.window)
        return {"image": self.transform(img), "label": int(self.labels[i])}


def val_embeddings(run, window, state):
    """(known_emb, known_lab, unknown_emb) for validation, as cropped (window None) or re-cut with the window."""
    cache = run / ("val_embeddings.npz" if window is None else f"val_embeddings_w{window:g}.npz")
    if cache.exists():
        z = np.load(cache)
        return z["known_emb"], z["known_lab"], z["unknown_emb"]
    if "model" not in state:
        state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state["model"], ckpt = HybridPillNet.from_checkpoint(run / "best.pt", state["device"])
        cfg = ckpt["config"]
        state["manifest"] = load_manifest(EPILLS_ROOT / cfg["data"]["manifest"])
        state["images"] = EPILLS_ROOT / cfg["data"]["images"]
    m, images, model, device = state["manifest"], state["images"], state["model"], state["device"]
    vocab = Vocab(m[m.split == "train"])
    kw = dict(batch_size=128, num_workers=8)
    out = []
    for split in ("val_enrolled", "val_unknown"):
        df = m[m.split == split]
        ds = PillCrops(df, images, eval_transform(), vocab) if window is None else Reframed(df, images, window)
        out.append(embed(model, DataLoader(ds, **kw), device))
    known_emb, known_lab = out[0]
    np.savez(cache, known_emb=known_emb, known_lab=known_lab, unknown_emb=out[1][0])
    return known_emb, known_lab, out[1][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run folder with best.pt")
    parser.add_argument("--gallery-size", type=int, default=10)
    parser.add_argument("--k-enrol", type=int, default=3)
    parser.add_argument("--max-flip", type=float, default=0.10, help="largest share of own pills asked to flip")
    parser.add_argument("--galleries", type=int, default=1000)
    parser.add_argument("--unknown-per-gallery", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", type=float, nargs="+",
                        help="choose the crop contract's window among these, on validation (e.g. 1.4 1.55 1.7 1.85)")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run

    state, sweep, window = {}, [], None
    for w in args.windows or []:
        k_emb, k_lab, u_emb = val_embeddings(run, w, state)
        kn, un = simulate(k_emb, k_lab, u_emb, args.gallery_size, args.k_enrol, args.galleries,
                          args.unknown_per_gallery, args.seed)
        t = float(np.quantile(un.s1, UNKNOWN_REJECTION))
        sweep.append({"window": w, "threshold": t, **operating_point(kn, un, t, 0.0)})
        print(f"window {w:g}: threshold {t:.4f}, correct matches {sweep[-1]['correct_match']:.1%}, "
              f"wrong {sweep[-1]['wrong_match']:.2%} (margin 0)", flush=True)
    if sweep:
        window = max(sweep, key=lambda r: r["correct_match"])["window"]
        print(f"chosen window: {window:g}", flush=True)
    known_emb, known_lab, unknown_emb = val_embeddings(run, window, state)

    sim = {}
    for size in (5, 10, 15):
        for k in (3, 5):
            sim[size, k] = simulate(known_emb, known_lab, unknown_emb, size, k, args.galleries,
                                    args.unknown_per_gallery, args.seed)
    known, unknown = sim[args.gallery_size, args.k_enrol]
    threshold = float(np.quantile(unknown.s1, UNKNOWN_REJECTION))

    rows = []
    for (size, k), (kn, un) in sim.items():
        for margin in MARGINS:
            rows.append({"gallery_size": size, "k_enrol": k, "margin": margin,
                         **operating_point(kn, un, threshold, margin)})
    table = pd.DataFrame(rows)
    table.to_csv(run / "calibration_table.csv", index=False)

    main_rows = table[(table.gallery_size == args.gallery_size) & (table.k_enrol == args.k_enrol)]
    margin = float(main_rows[main_rows.own_flip <= args.max_flip].margin.max())
    point = operating_point(known, unknown, threshold, margin)
    others = {f"gallery_{s}_k_{k}": operating_point(*sim[s, k], threshold, margin) for s, k in sim}
    result = {
        "threshold": threshold, "margin": margin, "window": window,
        "rule": f"threshold: {UNKNOWN_REJECTION:.0%} of unknown pills rejected; margin: largest with own-pill flip "
                f"requests <= {args.max_flip:.0%}",
        "calibrated_on": {"split": "val_enrolled + val_unknown", "gallery_size": args.gallery_size,
                          "k_enrol": args.k_enrol, "galleries": args.galleries, "seed": args.seed},
        "operating_point": point, "at_other_settings": others, "window_sweep": sweep,
        "model": str((run / "best.pt").relative_to(EPILLS_ROOT)),
    }
    (run / "calibration.json").write_text(json.dumps(result, indent=2))

    pct = lambda d: " | ".join(f"{k} {v:6.1%}" for k, v in d.items())  # noqa: E731
    print(f"threshold {threshold:.4f} | margin {margin:.3f}")
    print(f"gallery {args.gallery_size}, k={args.k_enrol}: {pct(point)}")
    print(f"same threshold, margin 0:   {pct(operating_point(known, unknown, threshold, 0.0))}")
    for name, d in others.items():
        print(f"  {name:16s} {pct(d)}")


if __name__ == "__main__":
    main()
