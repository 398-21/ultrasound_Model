"""
Look-alike strengths (blueprint job B12): does the model tell one medicine's strengths and release forms apart?

Sibling pairs: two catalogue products with the same active ingredients but a different strength (metformin 500 vs
850 mg), or the same strength in another release form (IR vs ER), both with at least 4 everyday photos among the
validation and held-out C3PI photos (no product here was trained on). A stress test: the patient's gallery holds both
siblings (3 enrolment photos each, 20 random draws per pair); every other photo of either is a query. Decisions as the
API makes them (served threshold and margin), then with the imprint check (core.inference.matching.check_imprint, OCR
reads cached by evaluate_imprint_gallery.py). The dangerous outcome is "named as the sibling".
"Look-alike": both siblings have the same recorded shape and colour.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_lookalike_strengths.py --run outputs\\runs\\stage2_run4 [--ocr large_c92]
Output: <run>/eval/lookalike_strengths.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.catalogue.identity import product_identity  # noqa: E402
from epills.core.catalogue.store import Catalogue  # noqa: E402
from epills.core.imprint.match import GALLERY_CONFIRM_DISTINCT, GALLERY_MIN_WEIGHT, read_hits  # noqa: E402
from epills.core.inference.matching import Decision, check_imprint, normalise  # noqa: E402

CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
OCR = EPILLS_ROOT / "data/interim/ocr/Florence-2-base"
K, DRAWS = 3, 20
ORDER = ["named correctly", "named as the other", "ambiguous", "unknown"]


def load(run, ocr="base_c70"):
    """
    The served calibration, the catalogue, and every validation and held-out C3PI photo: rows (manifest), normalised
    embeddings, cached OCR reads, and pill_id -> photo indices for products with at least K + 1 photos.
    """
    cal = json.loads((run / "calibration.json").read_text())
    cat = Catalogue(EPILLS_ROOT / "data/processed/catalogue")
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    m["pill_id"] = m.ndc9 + "_" + m.part
    z = np.load(run / f"val_embeddings_w{cal['window']:g}.npz")
    h = np.load(run / "eval/c3pi_holdout_embeddings.npz")
    rows = pd.concat([m[m.split == "val_enrolled"], m[m.split == "val_unknown"], m[m.split == "epillid_holdout"]])
    emb = normalise(np.concatenate([z["known_emb"], z["unknown_emb"], h["emb"]]))
    ocr_dir = OCR if ocr == "base_c70" else EPILLS_ROOT / "data/interim/ocr" / ocr
    reads = {**json.loads((ocr_dir / "validation_all.json").read_text()),
             **json.loads((ocr_dir / "c3pi_holdout.json").read_text())}
    rows = rows.reset_index(drop=True)
    photos = {pid: idx.to_numpy() for pid, idx in rows.groupby("pill_id").groups.items() if len(idx) >= K + 1}
    return cal, cat, rows, emb, reads, photos


def play(pairs, imprints, cal, weight, rows, emb, reads, photos, draws=DRAWS, seed=0,
         distinct=GALLERY_CONFIRM_DISTINCT):
    """
    pairs: [(a, b, {extra columns})]; imprints: pill_id -> recorded imprint. Per pair, `draws` galleries holding both
    (K enrolment photos each); every other photo of either is a query, decided as the API does, then with the imprint
    check. One row per query and check: outcome (ORDER) and the imprint check's status.
    """
    thr, margin = cal["threshold"], cal["margin"]
    rng = np.random.default_rng(seed)
    out = []
    for a, b, extra in pairs:
        gallery = {a: imprints[a], b: imprints[b]}
        imprinted = {k for k, v in gallery.items() if v}
        for _ in range(draws):
            enrol = {pid: rng.choice(photos[pid], K, replace=False) for pid in (a, b)}
            protos = {pid: normalise(emb[idx].mean(axis=0)) for pid, idx in enrol.items()}
            for true in (a, b):
                for q in np.setdiff1d(photos[true], enrol[true]):
                    sims = {pid: float(emb[q] @ protos[pid]) for pid in (a, b)}
                    best, second = sorted(sims, key=sims.get, reverse=True)
                    s1, s2 = sims[best], sims[second]
                    verdict = "unknown" if s1 < thr else "ambiguous" if s1 - s2 < margin else "match"
                    d = Decision(best, s1, second, s2, verdict)
                    r = reads.get(str(CROPS / rows.crop_path[q]), [])
                    hits = set(read_hits(r, gallery, weight, GALLERY_MIN_WEIGHT))
                    dh = set(read_hits(r, gallery, weight, GALLERY_MIN_WEIGHT, distinct=True)) if distinct else None
                    d2, status = check_imprint(d, hits, imprinted, thr, dh)
                    for check, dd, st in (("look only", d, None), ("with imprint", d2, status)):
                        outcome = (dd.decision if dd.decision != "match" else
                                   "named correctly" if dd.medication_id == true else "named as the other")
                        out.append({"pair": f"{a}|{b}", "check": check, "outcome": outcome, "status": st, **extra})
    return pd.DataFrame(out)


def table(sub, order=ORDER):
    """% of queries per outcome, one row per check."""
    t = sub.groupby("check").outcome.value_counts(normalize=True).unstack(fill_value=0.0)
    return t.reindex(columns=order, fill_value=0.0).mul(100).round(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ocr", default="base_c70", help="reads to use: base_c70, or a preset cached by "
                                                          "read_imprints_cache.py (e.g. large_c92)")
    parser.add_argument("--distinct", action=argparse.BooleanOptionalAction, default=GALLERY_CONFIRM_DISTINCT,
                        help="a match is confirmed or contradicted only by tokens that tell the gallery apart "
                             "(default: as served)")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cal, cat, rows, emb, reads, photos = load(run, args.ocr)

    info = {}
    for pid in photos:
        p = cat.product(pid)
        if p is None:
            continue
        ident = product_identity(p["rxnorm_name"])
        ingredients = frozenset(i["rxcui"] for i in p["ingredients"] if i["rxcui"])
        if ingredients and ident["strength"] and ident["dose_form"] != "Pack":
            info[pid] = {"ingredients": ingredients, "strength": ident["strength"], "release": ident["release"],
                         "shape": p["shape"], "colour": p["colour"], "imprint": p["imprint"] or "",
                         "name": p["rxnorm_name"]}
    pairs = []
    for a, b in combinations(sorted(info), 2):
        A, B = info[a], info[b]
        if A["ingredients"] != B["ingredients"]:
            continue
        kind = ("strength" if A["strength"] != B["strength"] else
                "release" if A["release"] != B["release"] else None)
        if kind:
            pairs.append((a, b, {"kind": kind, "alike": A["shape"] == B["shape"] and A["colour"] == B["colour"]}))
    print(f"{len(info)} products with photos; {len(pairs)} sibling pairs "
          f"({sum(p[2]['kind'] == 'strength' for p in pairs)} strength, "
          f"{sum(p[2]['kind'] == 'release' for p in pairs)} release; {sum(p[2]['alike'] for p in pairs)} look alike)",
          flush=True)

    df = play(pairs, {pid: v["imprint"] for pid, v in info.items()}, cal, cat.imprint_index.weight,
              rows, emb, reads, photos, distinct=args.distinct)
    df["outcome"] = df.outcome.replace({"named as the other": "named as the sibling"})
    order = [o.replace("the other", "the sibling") for o in ORDER]
    result = {"protocol": __doc__.strip().split("\n\n")[1], "ocr": args.ocr, "distinct": args.distinct,
              "pairs": len(pairs),
              "pairs_by_kind": {k: int(sum(p[2]["kind"] == k for p in pairs)) for k in ("strength", "release")},
              "look_alike_pairs": int(sum(p[2]["alike"] for p in pairs)), "queries": int(len(df) // 2)}
    for name, sub in (("all", df), ("strength", df[df.kind == "strength"]), ("release", df[df.kind == "release"]),
                      ("look_alike", df[df.alike])):
        if len(sub):
            t = table(sub, order)
            result[name] = t.to_dict(orient="index")
            print(f"\n{name} ({sub.pair.nunique()} pairs, {len(sub) // 2:,} queries), %:")
            print(t.to_string())
    out = run / (("eval/lookalike_strengths" if args.ocr == "base_c70" else f"eval/lookalike_strengths_{args.ocr}") +
                 ("_distinct.json" if args.distinct else ".json"))
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
