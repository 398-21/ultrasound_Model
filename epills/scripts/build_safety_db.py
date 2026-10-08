"""
Build the Stage 3 interaction database (data/processed/safety/safety.sqlite) from the DDInter 2.0 CSVs.

Steps:
  1. Read the 14 DDInter files (data/safety/ddinter/): drugs and interacting pairs with their level.
  2. Map every DDInter drug to RxNorm ingredient codes (IN), so medications entered by brand, generic or NDC can be
     looked up: the name without its qualifier ("Morphine (liposomal)" -> "Morphine") is searched in RxNorm, then
     reduced to its ingredients. Responses are cached in data/safety/rxnorm_cache/, so reruns are fast.
  3. Write the SQLite file (schema in src/epills/core/safety/db.py) and a mapping report for review:
     data/processed/safety/ddinter_rxnorm_map.csv and build_report.json.

The file is what the hosted API reads; it is uploaded to the Hugging Face data repo, never committed to git.
DDInter is CC BY-NC-SA 4.0: non-commercial use, with attribution, and derived data under the same licence.

Run from the epills folder:  .venv\\Scripts\\python scripts\\build_safety_db.py
"""
import argparse
import csv
import datetime
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EPILLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS / "src"))
from epills.core.safety.db import SafetyDB, split_qualifier, write_db  # noqa: E402
from epills.core.safety.rxnorm import RxNormClient  # noqa: E402


def read_ddinter(folder):
    drugs, pairs, files = {}, {}, sorted(Path(folder).glob("ddinter_downloads_code_*.csv"))
    if not files:
        sys.exit(f"No DDInter CSVs in {folder}; run scripts/download_small_sources.py first.")
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                drugs[r["DDInterID_A"]] = r["Drug_A"]
                drugs[r["DDInterID_B"]] = r["Drug_B"]
                key = tuple(sorted((r["DDInterID_A"], r["DDInterID_B"])))
                if key in pairs and pairs[key] != r["Level"]:
                    sys.exit(f"Conflicting levels for {key}: {pairs[key]} vs {r['Level']}")
                pairs[key] = r["Level"]
    return drugs, pairs, [f.name for f in files]


def map_drug(client, ddinter_id, name):
    base, qualifier = split_qualifier(name)
    for query in dict.fromkeys([base, name.strip()]):  # base name first, then the full DDInter name
        rxcui = client.rxcui_by_name(query)
        if rxcui:
            ingredients = client.ingredients(rxcui)
            if ingredients:
                return dict(ddinter_id=ddinter_id, name=name, qualifier=qualifier, query=query, rxcui=rxcui,
                            ingredients=ingredients)
    return dict(ddinter_id=ddinter_id, name=name, qualifier=qualifier, query=base, rxcui=None, ingredients=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ddinter", type=Path, default=EPILLS / "data" / "safety" / "ddinter")
    parser.add_argument("--cache", type=Path, default=EPILLS / "data" / "safety" / "rxnorm_cache")
    parser.add_argument("--out", type=Path, default=EPILLS / "data" / "processed" / "safety" / "safety.sqlite")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    drugs, pairs, files = read_ddinter(args.ddinter)
    print(f"DDInter: {len(drugs):,} drugs, {len(pairs):,} pairs from {len(files)} files", flush=True)

    client = RxNormClient(cache_dir=args.cache)
    with ThreadPoolExecutor(args.workers) as pool:
        mapped = list(pool.map(lambda item: map_drug(client, *item), sorted(drugs.items())))
    ok = [m for m in mapped if m["ingredients"]]
    print(f"RxNorm: {len(ok):,} of {len(mapped):,} DDInter drugs mapped to ingredients", flush=True)

    ingredient_rows = [(m["ddinter_id"], r) for m in ok for r, _ in m["ingredients"]]
    meta = {
        "source": "DDInter 2.0, https://ddinter2.scbdd.com/ (bulk CSV downloads)",
        "licence": "CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike)",
        "rxnorm": "RxNav REST API (NLM), ingredient (IN) concepts",
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ddinter_files": ",".join(files),
        "drugs": len(drugs), "pairs": len(pairs), "drugs_mapped_to_rxnorm": len(ok),
    }
    write_db(args.out, sorted(drugs.items()), ingredient_rows, ((a, b, lv) for (a, b), lv in pairs.items()), meta)

    # review files
    report_dir = args.out.parent
    with open(report_dir / "ddinter_rxnorm_map.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ddinter_id", "ddinter_name", "qualifier", "rxnorm_query", "rxcui", "ingredient_rxcuis",
                    "ingredient_names"])
        for m in mapped:
            w.writerow([m["ddinter_id"], m["name"], m["qualifier"], m["query"], m["rxcui"] or "",
                        ";".join(r for r, _ in m["ingredients"]), ";".join(n for _, n in m["ingredients"])])
    plain_by_rxcui = Counter(r for m in ok if not m["qualifier"] for r, _ in m["ingredients"])
    shared = {r: n for r, n in plain_by_rxcui.items() if n > 1}
    levels = Counter(pairs.values())
    report = {**meta, "levels": dict(levels),
              "unmapped": [m["name"] for m in mapped if not m["ingredients"]],
              "ingredients_with_several_plain_ddinter_drugs": len(shared)}
    (report_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    db = SafetyDB(args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB). Levels: {dict(levels)}")
    print(f"Unmapped DDInter drugs: {len(report['unmapped'])} (listed in build_report.json)")
    print("Spot check:", {pair: db.level(*[db.drugs_for_ingredient(name=n)[0][0] for n in pair])
                          for pair in (("Warfarin", "Ibuprofen"), ("Atorvastatin", "Clarithromycin"))})


if __name__ == "__main__":
    main()
