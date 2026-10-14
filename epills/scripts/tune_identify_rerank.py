"""
The imprint re-ranking of /v1/identify, chosen for what the API does: name the right catalogue product first.

IMPRINT_ALPHA and IMPRINT_TOP_K were first chosen on the ePillID benchmark (scripts/evaluate_imprint.py: MAP over
ePillID's 4,902 pill types). The API ranks the 4,936-product catalogue instead, where reads that match another
product's imprint pull it up more often, so the weight is chosen again here, on the catalogue itself:
  validation  C3PI validation photos the catalogue doesn't hold (as calibrate_identify.py): the weight (alpha) and
              how many of the closest products it may reorder (top_k) that put the right product first most often
  test, once  the held-out medicines' photos: the served setting against the chosen one
Reads: an OCR cache (scripts/read_imprints_cache.py). The identify confidence must be refitted afterwards
(scripts/calibrate_identify.py), since it uses the re-ranked scores.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\tune_identify_rerank.py --run outputs\\runs\\stage2_run4 --ocr large_c92
Output: <run>/eval/identify_rerank_<ocr>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.catalogue.store import IMPRINT_ALPHA, IMPRINT_TOP_K, Catalogue  # noqa: E402

CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
ALPHAS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
TOP_KS = [10, 20, 50, 100, 200, 500]
DEPTH = max(TOP_KS)


def prepare(cat, emb, rows, reads, group_of):
    """Per photo: similarities of its DEPTH closest products (best first), their imprint scores, which are right."""
    out = []
    for e, (_, r) in zip(emb, rows.iterrows()):
        q = e / np.linalg.norm(e)
        per_product = np.maximum.reduceat(q[None] @ cat.emb.T, cat.starts, axis=1)[0]
        idx = np.argsort(-per_product)[:DEPTH]
        rd = reads.get(str(CROPS / r.crop_path)) or []
        imp = cat.imprint_index.scores(rd, idx) if rd else np.zeros(len(idx), np.float32)
        truth = group_of.get(r.pill_id, r.pill_id)
        right = np.array([group_of.get(p, p) == truth for p in cat.segment_pill[idx]])
        out.append((per_product[idx], imp, right))
    return out


def rates(prepared, alpha, k):
    top1 = top5 = 0
    for sims, imp, right in prepared:
        score = sims.copy()
        score[:k] += alpha * imp[:k]
        order = np.argsort(-score)
        top1 += right[order[0]]
        top5 += right[order[:5]].any()
    return top1 / len(prepared), top5 / len(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ocr", default="large_c92")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cat = Catalogue(EPILLS_ROOT / "data/processed/catalogue")
    cal = json.loads((run / "calibration.json").read_text())
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    crops = crops[(crops.found == "1") & (crops.crop_path != "")].copy()
    crops["pill_id"] = crops.ndc9 + "_" + crops.part
    crops = crops.sample(frac=1, random_state=0)  # as build_catalogue.py: the catalogue's own everyday photos
    crops["rank"] = crops.groupby("pill_id").cumcount()
    held = set(crops[crops["rank"] < 8].crop_path)
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    m["pill_id"] = m.ndc9 + "_" + m.part
    group_of = m.groupby("pill_id").label.first().to_dict()
    z = np.load(run / f"val_embeddings_w{cal['window']:g}.npz")
    h = np.load(run / "eval/c3pi_holdout_embeddings.npz")
    ocr = EPILLS_ROOT / "data/interim/ocr" / args.ocr
    sets = {"validation": (pd.concat([m[m.split == "val_enrolled"], m[m.split == "val_unknown"]]),
                           np.concatenate([z["known_emb"], z["unknown_emb"]]), ocr / "validation_all.json"),
            "test": (m[m.split == "epillid_holdout"], h["emb"], ocr / "c3pi_holdout.json")}
    prepared = {}
    for name, (rows, emb, cache) in sets.items():
        rows = rows.reset_index(drop=True)
        keep = (~rows.crop_path.isin(held) & rows.pill_id.isin(cat.attrs)).to_numpy()
        print(f"{name}: {keep.sum():,} photos ...", flush=True)
        prepared[name] = prepare(cat, emb[keep], rows[keep], json.loads(cache.read_text()), group_of)

    grid = pd.DataFrame([{"alpha": a, "top_k": k, **dict(zip(("top1", "top5"), rates(prepared["validation"], a, k)))}
                         for a in ALPHAS for k in TOP_KS])
    best = grid.sort_values(["top1", "top5", "alpha", "top_k"], ascending=[False, False, True, True]).iloc[0]
    alpha, top_k = float(best.alpha), int(best.top_k)
    print("validation, right product first:")
    print(grid.pivot(index="alpha", columns="top_k", values="top1").round(4).to_string())
    print(f"chosen: alpha {alpha}, top_k {top_k}")
    test = {label: dict(zip(("top1", "top5"), rates(prepared["test"], a, k)))
            for label, (a, k) in (("look only", (0.0, 1)), ("served before", (IMPRINT_ALPHA, IMPRINT_TOP_K)),
                                  ("chosen", (alpha, top_k)))}
    for label, r in test.items():
        print(f"test, {label}: first right {r['top1']:.1%}, top 5 {r['top5']:.1%}")
    result = {"protocol": __doc__.strip().split("\n\n")[1], "ocr": args.ocr,
              "photos": {k: len(v) for k, v in prepared.items()},
              "served_before": {"alpha": IMPRINT_ALPHA, "top_k": IMPRINT_TOP_K},
              "chosen": {"alpha": alpha, "top_k": top_k}, "validation_grid": grid.to_dict("records"), "test": test}
    out = run / f"eval/identify_rerank_{args.ocr}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
