"""
Same-maker imprints (blueprint job B27): are two different medicines from one maker, whose imprints share a token
(Teva's "93", "TEVA"), confused with each other?

Pairs: two catalogue products from the same maker with no active ingredient in common whose recorded imprints share a
token of 2+ characters, both with at least 4 everyday photos among the validation and held-out C3PI photos. The
patient's gallery holds both (3 enrolment photos each, 10 random draws per pair); every other photo of either is a
query, decided as the API does, then with the imprint check. The dangerous outcome is "named as the other".
A shared token "counts" when its rarity reaches GALLERY_MIN_WEIGHT: reading it alone is then a read of both
medicines, so it can't settle a tie, and it makes a wrong look-match "confirmed" instead of a "conflict".
"Look-alike": both have the same recorded shape and colour.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_same_maker_imprints.py --run outputs\\runs\\stage2_run4 --ocr large_c92
Output: <run>/eval/same_maker_imprints[_<ocr>].json
"""
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.imprint.match import GALLERY_CONFIRM_DISTINCT, GALLERY_MIN_WEIGHT, imprint_tokens  # noqa: E402

spec = importlib.util.spec_from_file_location("lookalike", EPILLS_ROOT / "scripts/evaluate_lookalike_strengths.py")
lookalike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lookalike)
DRAWS = 10


def maker(name):
    """'TEVA PHARM.', 'Teva' -> 'TEVA'; 'SANDOZ, INC' -> 'SANDOZ': the first word of the labeller's name."""
    words = re.sub(r"[^A-Z ]", " ", str(name or "").upper()).split()
    return words[0] if words else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ocr", default="large_c92", help="reads to use: base_c70, or a preset cached by "
                                                           "read_imprints_cache.py")
    parser.add_argument("--distinct", action=argparse.BooleanOptionalAction, default=GALLERY_CONFIRM_DISTINCT,
                        help="a match is confirmed or contradicted only by tokens that tell the gallery apart "
                             "(default: as served)")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cal, cat, rows, emb, reads, photos = lookalike.load(run, args.ocr)
    weight = cat.imprint_index.weight

    info = {}
    for pid in photos:
        p = cat.product(pid)
        if p is None:
            continue
        ingredients = frozenset(i["rxcui"] for i in p["ingredients"] if i["rxcui"])
        if ingredients and maker(p["manufacturer"]) and imprint_tokens(p["imprint"]):
            info[pid] = {"maker": maker(p["manufacturer"]), "ingredients": ingredients, "imprint": p["imprint"],
                         "tokens": imprint_tokens(p["imprint"]), "shape": p["shape"], "colour": p["colour"]}
    pairs = []
    for a, b in combinations(sorted(info), 2):
        A, B = info[a], info[b]
        shared = A["tokens"] & B["tokens"]
        if A["maker"] != B["maker"] or A["ingredients"] & B["ingredients"] or not shared:
            continue
        top = max(shared, key=lambda t: weight.get(t, 1.0))
        pairs.append((a, b, {"maker": A["maker"], "shared": top,
                             "counts": weight.get(top, 1.0) >= GALLERY_MIN_WEIGHT,
                             "alike": A["shape"] == B["shape"] and A["colour"] == B["colour"]}))
    print(f"{len(info)} products with photos, a maker and an imprint; {len(pairs)} same-maker pairs sharing a token "
          f"({sum(p[2]['counts'] for p in pairs)} where the shared token alone counts as a read; "
          f"{sum(p[2]['alike'] for p in pairs)} look alike)", flush=True)

    df = lookalike.play(pairs, {pid: v["imprint"] for pid, v in info.items()}, cal, weight, rows, emb, reads, photos,
                        draws=DRAWS, distinct=args.distinct)
    shared_tokens = Counter(p[2]["shared"] for p in pairs)
    result = {"protocol": __doc__.strip().split("\n\n")[1], "ocr": args.ocr, "distinct": args.distinct,
              "gallery_min_weight": GALLERY_MIN_WEIGHT,
              "pairs": len(pairs), "queries": int(len(df) // 2),
              "pairs_by_maker": dict(Counter(p[2]["maker"] for p in pairs).most_common(10)),
              "pairs_by_shared_token": {t: {"pairs": n, "weight": round(weight.get(t, 1.0), 3)}
                                        for t, n in shared_tokens.most_common(10)}}
    groups = [("all", df), ("shared token counts as a read", df[df.counts]),
              ("shared token too common to count", df[~df.counts]), ("look_alike", df[df.alike])]
    groups += [(f"shared '{t}'", df[df.shared == t]) for t, _ in shared_tokens.most_common(3)]
    for name, sub in groups:
        if not len(sub):
            continue
        t = lookalike.table(sub)
        imp = sub[sub.check == "with imprint"]
        wrong = imp[imp.outcome == "named as the other"]
        look_wrong = sub[(sub.check == "look only") & (sub.outcome == "named as the other")]
        result[name] = {**t.to_dict(orient="index"), "pairs": int(sub.pair.nunique()), "queries": int(len(sub) // 2),
                        "wrong_names_confirmed_by_imprint_%":
                            round(100 * float((wrong.status == "confirmed").mean()), 2) if len(wrong) else None,
                        "wrong_names": {"look only": int(len(look_wrong)), "with imprint": int(len(wrong))}}
        print(f"\n{name} ({sub.pair.nunique()} pairs, {len(sub) // 2:,} queries), %:")
        print(t.to_string())
        print(f"  wrong names with the imprint check: {len(wrong):,}, of which 'confirmed': "
              f"{result[name]['wrong_names_confirmed_by_imprint_%']}% (look only: {len(look_wrong):,})")
    imp = df[df.check == "with imprint"]
    rate = imp.assign(wrong=imp.outcome == "named as the other").groupby("pair").wrong.agg(["mean", "sum", "size"])
    result["worst_pairs"] = [
        {"pair": [cat.product(pid)["rxnorm_name"] + f" ({info[pid]['imprint']})" for pid in pair.split("|")],
         "named_as_the_other_%": round(100 * r["mean"], 2), "queries": int(r["size"])}
        for pair, r in rate[rate["sum"] > 0].sort_values("mean", ascending=False).head(8).iterrows()]
    print("\nworst pairs (with the imprint check):")
    for w in result["worst_pairs"]:
        print(f"  {w['named_as_the_other_%']:5.2f}% of {w['queries']:,}: {' | '.join(w['pair'])}")
    out = run / (("eval/same_maker_imprints" if args.ocr == "base_c70" else f"eval/same_maker_imprints_{args.ocr}") +
                 ("_distinct.json" if args.distinct else ".json"))
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
