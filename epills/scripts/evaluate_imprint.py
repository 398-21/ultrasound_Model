"""
ePillID with imprint re-ranking: the run's embedding ranks the 4,902 pill types, then the imprint read by OCR
(epills.core.imprint) reorders the top candidates. A separate variant from the plain ePillID result: it uses extra
information (each type's imprint text from C3PI's metadata, matched by NDC9), which the paper's model did not.

1. OCR reads (Florence-2, 4 rotations) of the test photos (ePillID fold 4) and of a validation sample; cached in
   data/interim/ocr/<model>/ and reused for every run (they don't depend on the embedding model).
2. Tuning on validation only: validation everyday photos of types that are in ePillID, against the references of all
   4,902 types (as the benchmark; runs 5-6 never trained on validation types' references). alpha (weight of the
   imprint score) and top_k (candidates re-ranked) are chosen by MAP.
3. The test, once, with the chosen alpha and top_k: two-sided pairs as published (similarities averaged over the
   two sides, reads of both sides pooled) and single photos.
Needs the run's ePillID embeddings (scripts/evaluate_stage2.py --sets epillid) and val_embeddings.npz
(scripts/calibrate_stage2.py). Output: <run>/eval/epillid_imprint.json.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_imprint.py --run outputs\\runs\\stage2_run5_official
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.imprint.match import ImprintIndex, rerank  # noqa: E402
from epills.core.inference.matching import normalise  # noqa: E402

spec = importlib.util.spec_from_file_location("evaluate_stage2", EPILLS_ROOT / "scripts" / "evaluate_stage2.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

OCR_MODEL = "florence-community/Florence-2-base"
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
TOP_KS = [10, 20, 50, 100, 200, 500]
VAL_SAMPLE = 2500


def reads_for(paths, centre, cache_file, reader_box):
    """OCR reads per image path, cached by path."""
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo = [p for p in paths if str(p) not in cache]
    if todo and cache_file.parent != EPILLS_ROOT / "data/interim/ocr" / OCR_MODEL.split("/")[-1]:
        sys.exit(f"{cache_file} lacks {len(todo):,} reads: finish it with scripts/read_imprints_cache.py --preset "
                 f"{cache_file.parent.name} (reading here would mix in another reader)")
    if todo:
        if reader_box[0] is None:
            from epills.core.imprint.ocr import ImprintReader
            reader_box[0] = ImprintReader(OCR_MODEL)
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            for p, r in zip(chunk, reader_box[0].read([Image.open(p) for p in chunk], centre=centre)):
                cache[str(p)] = r
            cache_file.write_text(json.dumps(cache))
            print(f"\rOCR {cache_file.stem}: {min(i + 64, len(todo)):,} / {len(todo):,}", end="", flush=True)
        print(flush=True)
    return [cache[str(p)] for p in paths]


def class_sims(q_emb, ref_emb, ref_cls, n_classes):
    pw = torch.from_numpy(normalise(q_emb) @ normalise(ref_emb).T)
    return torch.full((len(q_emb), n_classes), -2.0).scatter_reduce(
        1, torch.from_numpy(ref_cls).expand(len(q_emb), -1), pw, reduce="amax").numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ocr", default="base_c70", help="reads to use: base_c70 (cached here), or a preset cached by "
                                                           "scripts/read_imprints_cache.py (large_c92)")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cache_dir = EPILLS_ROOT / "data/interim/ocr" / (OCR_MODEL.split("/")[-1] if args.ocr == "base_c70" else args.ocr)
    val_cache = "validation.json" if args.ocr == "base_c70" else "validation_all.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reader = [None]

    # classes = ePillID pill types; their imprints from C3PI metadata (by NDC9)
    labels = pd.read_csv(ev.EPILLID / "all_labels.csv")
    refs = labels[labels.is_ref].reset_index(drop=True)
    classes = np.array(sorted(refs.pilltype_id.unique()))
    ref_cls = np.searchsorted(classes, refs.pilltype_id.to_numpy())
    ndc_of = (refs.drop_duplicates("pilltype_id").set_index("pilltype_id")
              .apply(lambda r: f"{int(r.label_code_id):05d}{int(r.prod_code_id):04d}", axis=1))
    meta = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/metadata.csv", dtype=str, keep_default_na=False,
                       usecols=["ndc9", "imprint"])
    imprint_of = meta[meta.imprint != ""].groupby("ndc9").imprint.first()
    index = ImprintIndex([imprint_of.get(ndc_of[c], "") for c in classes])
    known = sum(1 for t in index.tokens if t)
    print(f"{len(classes):,} classes, {known:,} with a known imprint", flush=True)

    ref_emb = np.load(run / "eval/epillid_ref_embeddings.npz")["emb"]
    q4_emb = np.load(run / "eval/epillid_fold4_embeddings.npz")["emb"]

    # 2. tuning on validation
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    val = pd.concat([m[m.split == "val_enrolled"], m[m.split == "val_unknown"]]).reset_index(drop=True)
    window = json.loads((run / "calibration.json").read_text()).get("window")
    z = np.load(run / ("val_embeddings.npz" if window is None else f"val_embeddings_w{window:g}.npz"))
    val_emb = np.concatenate([z["known_emb"], z["unknown_emb"]])
    type_of_ndc = {n: c for c, n in ndc_of.items()}
    val["cls"] = val.ndc9.map(lambda n: type_of_ndc.get(n))
    keep = val[val.cls.notna()].sample(frac=1, random_state=0).head(VAL_SAMPLE)
    v_sims = class_sims(val_emb[keep.index.to_numpy()], ref_emb, ref_cls, len(classes))
    v_right = np.searchsorted(classes, keep.cls.to_numpy())
    v_reads = reads_for([EPILLS_ROOT / "data/interim/c3pi/crops/images" / p for p in keep.crop_path], 0.7,
                        cache_dir / val_cache, reader)
    grid = []
    for k in TOP_KS:
        for a in ALPHAS:
            mtr = ev.epillid_metrics(rerank(v_sims, v_reads, index, a, k), v_right)
            grid.append({"top_k": k, "alpha": a, "MAP": mtr["MAP"], "MAP@1": mtr["MAP@1"]})
    grid = pd.DataFrame(grid)
    best = grid.sort_values(["MAP", "alpha"], ascending=[False, True]).iloc[0]
    alpha, top_k = float(best.alpha), int(best.top_k)
    print(f"validation ({len(keep):,} photos of {keep.cls.nunique()} types):")
    print(grid.pivot(index="alpha", columns="top_k", values="MAP").round(4).to_string())
    print(f"chosen: alpha {alpha}, top_k {top_k}", flush=True)

    # 3. the test, once
    f4 = pd.read_csv(ev.EPILLID / "folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/"
                                  "pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    t_reads = reads_for([ev.EPILLID / "classification_data" / p for p in f4.image_path], 1.0,
                        cache_dir / "epillid_fold4.json", reader)
    q_cls = np.searchsorted(classes, f4.pilltype_id.to_numpy())
    sims = class_sims(q4_emb, ref_emb, ref_cls, len(classes))
    front = f4.is_front.to_numpy().astype(bool)
    pairs = np.array([(i, j) for i in range(len(f4)) for j in range(i + 1, len(f4))
                      if q_cls[i] == q_cls[j] and front[i] != front[j]])
    two = (sims[pairs[:, 0]] + sims[pairs[:, 1]]) / 2
    pair_reads = [t_reads[i] + t_reads[j] for i, j in pairs]
    result = {
        "variant": "ePillID official + imprint re-ranking (Florence-2 OCR; imprints from C3PI metadata by NDC9)",
        "chosen_on_validation": {"alpha": alpha, "top_k": top_k, "validation_photos": int(len(keep))},
        "validation_grid": grid.to_dict(orient="records"),
        "two_sided": {"plain": ev.epillid_metrics(two, q_cls[pairs[:, 0]]),
                      "imprint": ev.epillid_metrics(rerank(two, pair_reads, index, alpha, top_k), q_cls[pairs[:, 0]])},
        "single_photo": {"plain": ev.epillid_metrics(sims, q_cls),
                         "imprint": ev.epillid_metrics(rerank(sims, t_reads, index, alpha, top_k), q_cls)},
        "test_reads_with_own_token": float(np.mean([index.scores(r, [c])[0] > 0 for r, c in zip(t_reads, q_cls)])),
        "paper_best_two_sided": {"MAP": 0.9576, "GAP": 0.8120, "MAP@1": 0.9201, "GAP@1": 0.9119},
    }
    result["ocr"] = args.ocr
    (run / ("eval/epillid_imprint.json" if args.ocr == "base_c70" else f"eval/epillid_imprint_{args.ocr}.json")
     ).write_text(json.dumps(result, indent=2))
    for side in ("two_sided", "single_photo"):
        for how in ("plain", "imprint"):
            r = result[side][how]
            print(f"{side:12s} {how:8s} MAP {r['MAP']:.3f}  MAP@1 {r['MAP@1']:.3f}  GAP {r['GAP']:.3f}  "
                  f"GAP@1 {r['GAP@1']:.3f}  top5 {r['top5']:.3f}")
    print(f"test photos whose read contains a token of their own imprint: {result['test_reads_with_own_token']:.1%}")


if __name__ == "__main__":
    main()
