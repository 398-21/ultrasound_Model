"""
/v1/identify's top-k on held-out medicines (test case I01: "C3PI ground truth with known imprint: top-k accuracy
measured and recorded"): where the right product lands in the list, with and without the imprint read, and how often
the first candidate is at least the same medicine (ingredients, strength, release) from another maker.

Photos: C3PI everyday photos of the held-out medicines (epillid_holdout) that the catalogue doesn't hold, run through
Catalogue.identify exactly as the API does. "Right" = the same appearance group (identical-looking products of other
labelers count). Reads: an OCR cache (scripts/read_imprints_cache.py).

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_identify_topk.py --run outputs\\runs\\stage2_run4 --ocr large_c92
Output: <run>/eval/identify_topk_<ocr>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.catalogue.identity import product_identity  # noqa: E402
from epills.core.catalogue.store import Catalogue  # noqa: E402
from epills.core.imprint.match import read_hits  # noqa: E402

CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
K = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ocr", default="large_c92")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cat = Catalogue(EPILLS_ROOT / "data/processed/catalogue")
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    crops = crops[(crops.found == "1") & (crops.crop_path != "")].copy()
    crops["pill_id"] = crops.ndc9 + "_" + crops.part
    crops = crops.sample(frac=1, random_state=0)  # as build_catalogue.py: 8 everyday photos per product
    crops["rank"] = crops.groupby("pill_id").cumcount()
    in_catalogue = set(crops[crops["rank"] < 8].crop_path)
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    m["pill_id"] = m.ndc9 + "_" + m.part
    group_of = m.groupby("pill_id").label.first().to_dict()
    test = m[m.split == "epillid_holdout"].reset_index(drop=True)
    emb = np.load(run / "eval/c3pi_holdout_embeddings.npz")["emb"]
    reads = json.loads((EPILLS_ROOT / "data/interim/ocr" / args.ocr / "c3pi_holdout.json").read_text())
    keep = (~test.crop_path.isin(in_catalogue) & test.pill_id.isin(cat.attrs)).to_numpy()
    test, emb = test[keep].reset_index(drop=True), emb[keep]

    def medicine(pid):
        p = cat.product(pid)
        ident = product_identity(p["rxnorm_name"])
        return frozenset(i["rxcui"] or i["name"] for i in p["ingredients"]), ident["strength"], ident["release"]

    rows = []
    for i, r in test.iterrows():
        rd = reads.get(str(CROPS / r.crop_path)) or None
        out = {"pill_id": r.pill_id}
        for name, use in (("look", None), ("imprint", rd)):
            c, _ = cat.identify(emb[i][None], top_k=K, reads=use)
            groups = [group_of.get(x["pill_id"], x["pill_id"]) for x in c]
            g = group_of.get(r.pill_id, r.pill_id)
            out[f"rank_{name}"] = groups.index(g) + 1 if g in groups else K + 1
            out[f"same_medicine_{name}"] = medicine(c[0]["pill_id"]) == medicine(r.pill_id)
        out["own_read"] = bool(rd) and bool(read_hits(rd, {0: cat.product(r.pill_id)["imprint"] or ""}))
        rows.append(out)
    df = pd.DataFrame(rows)

    def topk(ranks):
        return {f"top{k}": round(float(np.mean(ranks <= k)), 4) for k in (1, 3, 5, 10, 20)}

    result = {"protocol": __doc__.strip().split("\n\n")[1], "ocr": args.ocr, "photos": len(df),
              "products": int(df.pill_id.nunique()), "own_imprint_read": round(float(df.own_read.mean()), 4)}
    for name in ("look", "imprint"):
        result[name] = {**topk(df[f"rank_{name}"]),
                        "first_is_same_medicine": round(float(df[f"same_medicine_{name}"].mean()), 4)}
    result["imprint_by_own_read"] = {str(k): {"photos": len(s), **topk(s.rank_imprint)}
                                     for k, s in df.groupby("own_read")}
    per = df.groupby("pill_id").rank_imprint.apply(lambda s: np.mean(s <= 5))
    result["products_in_top5_under_half_the_time"] = round(float(np.mean(per < 0.5)), 4)
    out = run / f"eval/identify_topk_{args.ocr}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
