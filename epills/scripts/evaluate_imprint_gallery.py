"""
F7's imprint check in the app scenario, measured: does reading the imprint name more of the patient's pills correctly
without naming more of them wrongly?

Patients with 10 medications, 3 enrolment photos each, at the run's served threshold and margin (calibration.json),
as in calibrate_stage2.py; then each pill's imprint, read by OCR (Florence-2, 4 rotations, the central 70% of the
crop), checks the decision exactly as the API does: core.inference.matching.check_imprint, with hits from
core.imprint.match.read_hits and the served catalogue's token weights. A medication's recorded imprint is the most
common one among its photos in C3PI's metadata.
  validation   val_enrolled = the patients' medications, val_unknown = other pills; chooses MIN_WEIGHT, the least
               token rarity that counts as a read, as the one naming the most pills correctly while adding at most
               0.1 percentage point to either wrong rate (own pill named wrongly, other pill accepted)
  test, once   the C3PI held-out medications (epillid_holdout); unknown pills from those outside each gallery
OCR reads are cached in data/interim/ocr/Florence-2-base/ (~0.3 s per crop on the GPU).

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_imprint_gallery.py --run outputs\\runs\\stage2_run4 [--ocr-only]
Output: <run>/eval/imprint_gallery.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.catalogue.store import Catalogue  # noqa: E402
from epills.core.imprint.match import GALLERY_CONFIRM_DISTINCT, read_hits  # noqa: E402
from epills.core.inference.matching import Decision, check_imprint, normalise  # noqa: E402

OCR_MODEL = "florence-community/Florence-2-base"
CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
CACHE = EPILLS_ROOT / "data/interim/ocr" / OCR_MODEL.split("/")[-1]  # --ocr base_c70; other presets: ocr/<preset>
GALLERY, K_ENROL, GALLERIES, UNKNOWN_PER_GALLERY = 10, 3, 1000, 200
MIN_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TOLERANCE = 0.001  # at most +0.1 percentage point on either wrong rate


def reads_for(paths, cache_file, seed_file=None):
    """OCR reads per crop, cached by path (seed_file: an earlier cache to reuse, e.g. evaluate_imprint.py's)."""
    cache = json.loads(seed_file.read_text()) if seed_file and seed_file.exists() else {}
    if cache_file.exists():
        cache.update(json.loads(cache_file.read_text()))
    todo = [p for p in paths if str(p) not in cache]
    if todo and cache_file.parent != EPILLS_ROOT / "data/interim/ocr" / OCR_MODEL.split("/")[-1]:
        sys.exit(f"{cache_file} lacks {len(todo):,} reads: finish it with scripts/read_imprints_cache.py --preset "
                 f"{cache_file.parent.name} (reading here would mix in another reader)")
    if todo:
        from epills.core.imprint.ocr import ImprintReader
        reader = ImprintReader(OCR_MODEL)
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            for p, r in zip(chunk, reader.read([Image.open(p) for p in chunk], centre=0.7)):
                cache[str(p)] = r
            cache_file.write_text(json.dumps(cache))
            print(f"\rOCR {cache_file.stem}: {min(i + 64, len(todo)):,} / {len(todo):,}", end="", flush=True)
        print(flush=True)
    return [cache[str(p)] for p in paths]


def simulate(emb, lab, unknown_emb, unknown_lab, seed=0):
    """
    As training.eval.openset.simulate, keeping identities: per query its photo, whether it is the patient's, its true
    medication, its gallery and the two closest gallery medications with their similarities. unknown_emb None:
    unknown pills are photos of the medications outside each gallery.
    """
    rng = np.random.default_rng(seed)
    emb = normalise(emb)
    from_rest = unknown_emb is None
    unknown_emb, unknown_lab = (emb, lab) if from_rest else (normalise(unknown_emb), unknown_lab)
    by_label = {lab_: np.flatnonzero(lab == lab_) for lab_ in np.unique(lab)}
    eligible = np.array([lab_ for lab_, idx in by_label.items() if len(idx) > K_ENROL])
    rows, galleries = [], []
    for g in range(GALLERIES):
        labels = rng.choice(eligible, GALLERY, replace=False)
        protos, q_idx = [], []
        for lab_ in labels:
            enrol = rng.choice(by_label[lab_], K_ENROL, replace=False)
            protos.append(normalise(emb[enrol].mean(axis=0)))
            q_idx.append(np.setdiff1d(by_label[lab_], enrol))
        protos, q_idx = np.stack(protos), np.concatenate(q_idx)
        pool = np.flatnonzero(~np.isin(unknown_lab, labels)) if from_rest else np.arange(len(unknown_emb))
        u_idx = rng.choice(pool, min(UNKNOWN_PER_GALLERY, len(pool)), replace=False)
        for known, e, idx, true in ((True, emb[q_idx], q_idx, lab[q_idx]),
                                    (False, unknown_emb[u_idx], u_idx, unknown_lab[u_idx])):
            sims = e @ protos.T
            order = np.argsort(-sims, axis=1)[:, :2]
            n = np.arange(len(sims))
            rows.append(pd.DataFrame({"gallery": g, "known": known, "photo": idx, "true": true,
                                      "best": labels[order[:, 0]], "second": labels[order[:, 1]],
                                      "s1": sims[n, order[:, 0]], "s2": sims[n, order[:, 1]]}))
        galleries.append(labels)
    return pd.concat(rows, ignore_index=True), galleries


def outcomes(df, galleries, hits, imprint_of, cal, min_weight, distinct_hits=None):
    """The app's decisions, with the imprint check at min_weight (None: without it). hits[i] = {medication of row i's
    gallery: weight of its best imprint token read on row i's photo} (gallery_hits); distinct_hits: the same with
    distinct tokens, for check_imprint's distinct_hits (a match confirmed or contradicted by those only)."""
    thr, margin = cal["threshold"], cal["margin"]
    imprinted = [{m for m in labels if imprint_of.get(m)} for labels in galleries]
    dec, med, status = [], [], []
    for i, r in enumerate(df.itertuples(index=False)):
        verdict = "unknown" if r.s1 < thr else "ambiguous" if r.s1 - r.s2 < margin else "match"
        d, st = Decision(r.best, r.s1, r.second, r.s2, verdict), "off"
        if min_weight is not None:
            found = {m for m, w in hits[i].items() if w >= min_weight}
            dh = None if distinct_hits is None else {m for m, w in distinct_hits[i].items() if w >= min_weight}
            d, st = check_imprint(d, found, imprinted[r.gallery], thr, dh)
        dec.append(d.decision), med.append(d.medication_id), status.append(st)
    out = df.assign(decision=dec, medication=med, status=status)
    k, u = out[out.known], out[~out.known]
    km = k.decision == "match"
    return {
        "correct_match": float((km & (k.medication == k.true)).mean()),
        "wrong_match": float((km & (k.medication != k.true)).mean()),
        "own_flip": float((k.decision == "ambiguous").mean()),
        "own_rejected": float((k.decision == "unknown").mean()),
        "unknown_rejected": float((u.decision == "unknown").mean()),
        "unknown_flip": float((u.decision == "ambiguous").mean()),
        "unknown_matched": float((u.decision == "match").mean()),
        "status_own": k.status.value_counts(normalize=True).round(4).to_dict(),
        "status_unknown": u.status.value_counts(normalize=True).round(4).to_dict(),
    }


def gallery_hits(df, galleries, reads_known, reads_unknown, imprint_of, weight, distinct=False):
    """Per row: read_hits of its photo's reads against its gallery's imprints, exactly as the API calls it (with
    distinct, counting only tokens that tell the gallery apart)."""
    imprints = [{m: imprint_of[m] for m in labels if imprint_of.get(m)} for labels in galleries]
    return [read_hits((reads_known if r.known else reads_unknown)[r.photo], imprints[r.gallery], weight,
                      distinct=distinct) for r in df.itertuples(index=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ocr-only", action="store_true", help="read and cache the imprints, then stop")
    parser.add_argument("--distinct", action=argparse.BooleanOptionalAction, default=GALLERY_CONFIRM_DISTINCT,
                        help="a match is confirmed or contradicted only by tokens that tell the gallery apart "
                             "(default: as served)")
    parser.add_argument("--ocr", default="base_c70", help="reads to use: base_c70 (cached here), or a preset cached by "
                                                           "scripts/read_imprints_cache.py (large_c92)")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    global CACHE
    if args.ocr != "base_c70":
        CACHE = EPILLS_ROOT / "data/interim/ocr" / args.ocr
    CACHE.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    val_k, val_u = m[m.split == "val_enrolled"], m[m.split == "val_unknown"]
    test = m[m.split == "epillid_holdout"]
    r_val = reads_for([CROPS / p for p in pd.concat([val_k, val_u]).crop_path], CACHE / "validation_all.json",
                      CACHE / "validation.json")
    r_test = reads_for([CROPS / p for p in test.crop_path], CACHE / "c3pi_holdout.json")
    if args.ocr_only:
        return
    r_val_k, r_val_u = r_val[:len(val_k)], r_val[len(val_k):]

    imprint_of = (m[m.imprint != ""].groupby("label").imprint.agg(lambda s: s.mode().iloc[0]).to_dict())
    weight = Catalogue(EPILLS_ROOT / "data/processed/catalogue").imprint_index.weight
    cal = json.loads((run / "calibration.json").read_text())
    z = np.load(run / f"val_embeddings_w{cal['window']:g}.npz")
    h = np.load(run / "eval/c3pi_holdout_embeddings.npz")
    # the validation embeddings' labels are class indices; the manifest's are the medication labels
    assert len(z["known_emb"]) == len(val_k) and len(z["unknown_emb"]) == len(val_u) and len(h["emb"]) == len(test)
    for cached, labels in ((z["known_lab"], val_k.label), (h["lab"], test.label)):  # same rows, same order
        assert pd.Series(cached).groupby(labels.to_numpy()).nunique().max() == 1

    result = {"protocol": __doc__.strip().split("\n\n")[1], "threshold": cal["threshold"], "margin": cal["margin"],
              "gallery_size": GALLERY, "k_enrol": K_ENROL, "galleries": GALLERIES}
    sets = {"validation": (z["known_emb"], val_k.label.to_numpy(), z["unknown_emb"], val_u.label.to_numpy(),
                           r_val_k, r_val_u),
            "test": (h["emb"], test.label.to_numpy(), None, None, r_test, r_test)}
    chosen = None
    for name, (emb, lab, u_emb, u_lab, rk, ru) in sets.items():
        df, galleries = simulate(emb, lab, u_emb, u_lab)
        hits = gallery_hits(df, galleries, rk, ru, imprint_of, weight)
        dhits = gallery_hits(df, galleries, rk, ru, imprint_of, weight, distinct=True) if args.distinct else None
        own = np.mean([read_hits(r, {0: imprint_of.get(l, "")}, weight) != {} for r, l in zip(rk, lab)])
        base = outcomes(df, galleries, hits, imprint_of, cal, None)
        entry = {"queries_own": int(df.known.sum()), "queries_other": int((~df.known).sum()),
                 "photos_with_own_imprint_read": float(own), "without_check": base}
        if name == "validation":
            grid = {w: outcomes(df, galleries, hits, imprint_of, cal, w, dhits) for w in MIN_WEIGHTS}
            ok = [w for w, g in grid.items() if g["wrong_match"] <= base["wrong_match"] + TOLERANCE
                  and g["unknown_matched"] <= base["unknown_matched"] + TOLERANCE]
            chosen = max(ok, key=lambda w: (grid[w]["correct_match"], w)) if ok else None
            entry["grid"] = {str(w): g for w, g in grid.items()}
            entry["chosen_min_weight"] = chosen
        if chosen is not None:
            entry["with_check"] = outcomes(df, galleries, hits, imprint_of, cal, chosen, dhits)
        result[name] = entry
        print(f"\n{name}: {entry['queries_own']:,} own pills, {entry['queries_other']:,} other pills; "
              f"own imprint read on {own:.1%} of photos")
        rows = {"without": base, **({"with": entry["with_check"]} if "with_check" in entry else {})}
        if name == "validation":
            rows = {"without": base, **{f"w>={w}": g for w, g in grid.items()}}
        cols = ["correct_match", "wrong_match", "own_flip", "own_rejected", "unknown_rejected", "unknown_flip",
                "unknown_matched"]
        print(pd.DataFrame(rows).T[cols].astype(float).mul(100).round(2).to_string())
        if name == "validation":
            print(f"chosen min_weight: {chosen}")
    out = run / (("eval/imprint_gallery" if args.ocr == "base_c70" else f"eval/imprint_gallery_{args.ocr}") +
                 ("_distinct.json" if args.distinct else ".json"))
    result["ocr"], result["distinct"] = args.ocr, args.distinct
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
