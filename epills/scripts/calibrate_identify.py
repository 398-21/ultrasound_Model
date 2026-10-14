"""
Honest confidence for /v1/identify (blueprint job B4): raw similarities are not probabilities (pills that aren't in
the catalogue still score about 0.93), so the API reports, per answer:
  confidence      the chance the first candidate is the right product, if the pill is a US catalogue product:
                  logistic fit of (top-1 similarity, gap to the runner-up, imprint read on the top-1)
  catalogue_fit   good / weak / poor: how well the best match fits at all (top-1 similarity bands). poor: below what
                  95% of rightly identified catalogue pills reach; good: above what 90% of non-catalogue pills reach
  verdict         no_confident_match (poor fit, or confidence < LOW), possible_match (good fit and confidence >= HIGH),
                  uncertain (the rest)
Data, run exactly through Catalogue.identify as the API does (imprint reads included):
  catalogue pills   C3PI everyday crops that are NOT in the catalogue (the catalogue keeps 8 per product; the rest
                    are held out by build_catalogue.py): validation types to fit, epillid_holdout types to test.
                    "Right" = the top-1 is the same appearance group (identical-looking products of other labelers).
  other pills       MEDISEG (32 Hong Kong products, none in the catalogue): half the products to fit, half to test;
                    up to 800 crops per half, imprints read by OCR (cached).
Output: <run>/eval/identify_confidence.json, and the constants for src/epills/core/catalogue/confidence.py.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\calibrate_identify.py --run outputs\\runs\\stage2_run4
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.catalogue.store import Catalogue  # noqa: E402
from epills.core.inference.preprocess import crop_window  # noqa: E402

spec = importlib.util.spec_from_file_location("ev", EPILLS_ROOT / "scripts" / "evaluate_stage2.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
OCR = EPILLS_ROOT / "data/interim/ocr/Florence-2-base"
LOW, HIGH = 0.2, 0.5


def features(cat, emb, reads):
    rows = []
    for e, r in zip(emb, reads):
        c, _ = cat.identify(e[None], top_k=2, reads=r or None)
        rows.append({"top1": c[0]["pill_id"], "s1": c[0]["similarity"], "gap": c[0]["similarity"] - c[1]["similarity"],
                     "r1": c[0]["imprint_read"]})
    return pd.DataFrame(rows)


def design(df):
    return np.column_stack([np.ones(len(df)), df.s1, df.gap, df.r1])


def fit_logistic(X, y, l2=1e-3, steps=50):
    """Newton's method (IRLS) with a little ridge, so it always converges."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, X.T @ (p - y) + l2 * w)
    return w


def predict(w, df):
    return 1 / (1 + np.exp(-design(df) @ w))


def reliability(p, y, bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            out.append({"confidence": f"{lo:.1f}-{hi:.1f}", "pills": int(m.sum()), "stated": round(float(p[m].mean()), 3),
                        "observed": round(float(y[m].mean()), 3)})
    return out


def verdicts(p, s1, t_poor, t_good):
    fit = np.where(s1 < t_poor, "poor", np.where(s1 >= t_good, "good", "weak"))
    v = np.where((fit == "poor") | (p < LOW), "no_confident_match",
                 np.where((fit == "good") & (p >= HIGH), "possible_match", "uncertain"))
    return fit, v


def mediseg_reads(items, cache_file, model=None, centre=0.7, adapter=None):
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = [f"{p}|{','.join(f'{v:.1f}' for v in b)}" for p, b in zip(items.path, items.box)]
    todo = [i for i, k in enumerate(key) if k not in cache]
    if todo:
        from epills.core.imprint.ocr import ImprintReader
        reader = ImprintReader(model, adapter=adapter) if model else ImprintReader()
        for start in range(0, len(todo), 64):
            chunk = todo[start:start + 64]
            crops = [crop_window(ImageOps.exif_transpose(Image.open(items.path.iloc[i])).convert("RGB"),
                                 items.box.iloc[i], 1.55) for i in chunk]
            for i, r in zip(chunk, reader.read(crops, centre=centre)):
                cache[key[i]] = r
            cache_file.write_text(json.dumps(cache))
            print(f"\rOCR MEDISEG: {min(start + 64, len(todo)):,} / {len(todo):,}", end="", flush=True)
        print(flush=True)
    return [cache[k] for k in key]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--served", action="store_true", help="score the served weights and thresholds "
                        "(core/catalogue/confidence.py) instead of fitting new ones")
    parser.add_argument("--ocr", default="base_c70", help="reads to use: base_c70, or a preset cached by "
                                                           "scripts/read_imprints_cache.py (large_c92)")
    args = parser.parse_args()
    from epills.core.imprint.ocr import PRESETS
    ocr_dir = OCR if args.ocr == "base_c70" else EPILLS_ROOT / "data/interim/ocr" / args.ocr
    preset = PRESETS[args.ocr]
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cat = Catalogue(EPILLS_ROOT / "data/processed/catalogue")
    cal = json.loads((run / "calibration.json").read_text())

    # which everyday crops the catalogue holds (as build_catalogue.py, seed 0, 8 per product)
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    crops = crops[(crops.found == "1") & (crops.crop_path != "")].copy()
    crops["pill_id"] = crops.ndc9 + "_" + crops.part
    crops = crops.sample(frac=1, random_state=0)
    crops["rank"] = crops.groupby("pill_id").cumcount()
    in_catalogue = set(crops[crops["rank"] < 8].crop_path)

    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    m["pill_id"] = m.ndc9 + "_" + m.part
    group_of = m.groupby("pill_id").label.first().to_dict()
    z = np.load(run / f"val_embeddings_w{cal['window']:g}.npz")
    h = np.load(run / "eval/c3pi_holdout_embeddings.npz")
    val = pd.concat([m[m.split == "val_enrolled"], m[m.split == "val_unknown"]])
    val_emb = np.concatenate([z["known_emb"], z["unknown_emb"]])
    test = m[m.split == "epillid_holdout"]
    r_val = json.loads((ocr_dir / "validation_all.json").read_text())
    r_test = json.loads((ocr_dir / "c3pi_holdout.json").read_text())

    def c3pi(rows, emb, reads):
        keep = ~rows.crop_path.isin(in_catalogue).to_numpy() & rows.pill_id.isin(cat.attrs).to_numpy()
        rows, emb = rows[keep], emb[keep]
        df = features(cat, emb, [reads.get(str(CROPS / p)) for p in rows.crop_path])
        df["right"] = [group_of.get(t, t) == group_of.get(p, p) for t, p in zip(df.top1, rows.pill_id)]
        return df

    print("C3PI validation ...", flush=True)
    fit_c3pi = c3pi(val, val_emb, r_val)
    print("C3PI test ...", flush=True)
    test_c3pi = c3pi(test, h["emb"], r_test)

    items = ev.coco_items(ev.MEDISEG / "MEDISEG/32pills/annotations.json", ev.MEDISEG / "MEDISEG/32pills/images")
    med_emb = np.load(run / "eval/mediseg_embeddings.npz")["emb"]
    products = sorted(items.label.unique())
    rng = np.random.default_rng(0)
    half_a = set(rng.permutation(products)[:len(products) // 2])
    med = {}
    for name, in_half in (("fit", True), ("test", False)):
        idx = np.flatnonzero(items.label.isin(half_a).to_numpy() == in_half)
        idx = rng.choice(idx, min(800, len(idx)), replace=False)
        reads = mediseg_reads(items.iloc[idx], ocr_dir / "mediseg.json", preset["model"], preset["centre"],
                              EPILLS_ROOT / preset["adapter"] if preset.get("adapter") else None)
        print(f"MEDISEG {name} ...", flush=True)
        med[name] = features(cat, med_emb[idx], reads)
        med[name]["right"] = False

    if args.served:  # the served rule as it is, e.g. fitted on the GPU reader's reads, scored on the CPU reader's
        from epills.core.catalogue.confidence import T_GOOD, T_POOR, WEIGHTS
        w, t_poor, t_good = np.array(WEIGHTS), T_POOR, T_GOOD
    else:
        w = fit_logistic(design(fit_c3pi), fit_c3pi.right.to_numpy().astype(float))
        t_poor = float(np.quantile(fit_c3pi.s1[fit_c3pi.right], 0.05))
        t_good = float(np.quantile(med["fit"].s1, 0.90))
    report = {"protocol": __doc__.strip().split("\n\n")[0], "weights": w.round(4).tolist(),
              "feature_order": ["1", "s1", "gap", "r1"], "t_poor": round(t_poor, 4), "t_good": round(t_good, 4),
              "low": LOW, "high": HIGH, "model_version": cat.model_version}
    for name, c, o in (("validation", fit_c3pi, med["fit"]), ("test", test_c3pi, med["test"])):
        pc, po = predict(w, c), predict(w, o)
        fc, vc = verdicts(pc, c.s1.to_numpy(), t_poor, t_good)
        fo, vo = verdicts(po, o.s1.to_numpy(), t_poor, t_good)
        y = c.right.to_numpy()
        report[name] = {
            "catalogue_pills": int(len(c)), "top1_right": round(float(y.mean()), 3),
            "reliability": reliability(pc, y),
            "catalogue_pills_by_verdict": {v: {"share": round(float((vc == v).mean()), 3),
                                               "top1_right": round(float(y[vc == v].mean()), 3) if (vc == v).any() else None}
                                           for v in ("possible_match", "uncertain", "no_confident_match")},
            "other_pills": int(len(o)),
            "other_pills_by_verdict": {v: round(float((vo == v).mean()), 3)
                                       for v in ("possible_match", "uncertain", "no_confident_match")},
            "other_pills_mean_confidence": round(float(po.mean()), 3),
        }
    report["ocr"] = args.ocr
    out = run / (("eval/identify_confidence" if args.ocr == "base_c70" else f"eval/identify_confidence_{args.ocr}") +
                 ("_served.json" if args.served else ".json"))
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "protocol"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
