"""
Download the small data sources into epills/data/ and check them (C3PI has its own script, download_c3pi.py).

  ePillID   data/raw/epillid/ePillID_data.zip (160.7 MB, MIT), checked against the SHA-256 recorded in
            docs/datasets.md, then unzipped next to it
  DDInter   data/safety/ddinter/ddinter_downloads_code_<X>.csv, all 14 ATC-letter files (CC BY-NC-SA 4.0)

Finished files are skipped, so the script can be rerun. Usage, from the epills folder:
  python scripts/download_small_sources.py              # both
  python scripts/download_small_sources.py --only epillid
"""
import argparse
import csv
import sys
import zipfile
from pathlib import Path

from download_c3pi import fetch, file_digest

EPILLS_ROOT = Path(__file__).resolve().parents[1]

EPILLID_URL = "https://github.com/usuyama/ePillID-benchmark/releases/download/ePillID_data_v1.0/ePillID_data.zip"
EPILLID_SIZE = 160_732_055
EPILLID_SHA256 = "b960b3b1f0ee145ae2be7cfe188de1feaba4286b5d2fb7388004d151ad7311ac"

DDINTER_URL = "https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{}.csv"
DDINTER_CODES = "ABCDGHJLMNPRSV"  # the download page links 8 of these; all 14 are served at the same pattern


def download_epillid(data_root):
    folder = data_root / "raw" / "epillid"
    zip_path = folder / "ePillID_data.zip"
    if not zip_path.exists():
        print("Downloading ePillID (160.7 MB) ...", flush=True)
        fetch(EPILLID_URL, zip_path, size=EPILLID_SIZE, digest=EPILLID_SHA256, algorithm="sha256", resume=True)
    elif file_digest(zip_path, "sha256") != EPILLID_SHA256:
        sys.exit(f"{zip_path} doesn't match the SHA-256 in docs/datasets.md; delete it and rerun.")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not all((folder / name).exists() for name in names):
            archive.extractall(folder)

    labels_path = next(folder.rglob("all_labels.csv"))
    labels = list(csv.DictReader(open(labels_path, encoding="utf-8")))
    consumer = [row for row in labels if row["is_ref"] == "False"]
    folds = sorted(p.name for p in folder.rglob("*_[0-9].csv") if "folds" in p.parts)
    images = sum(1 for name in names if name.lower().endswith((".jpg", ".png")))
    print(f"ePillID: {len(labels):,} labelled images ({len(consumer):,} consumer, {len(labels) - len(consumer):,} reference), "
          f"{len({row['pilltype_id'] for row in labels}):,} pill types, {len({row['pilltype_id'] for row in consumer}):,} with "
          f"consumer images; {images:,} image files in the zip; {len(folds)} fold files. SHA-256 checked.", flush=True)


def download_ddinter(data_root):
    folder = data_root / "safety" / "ddinter"
    pairs, drugs, levels = set(), set(), {}
    for code in DDINTER_CODES:
        dest = folder / f"ddinter_downloads_code_{code}.csv"
        if not dest.exists():
            fetch(DDINTER_URL.format(code), dest)
        rows = list(csv.DictReader(open(dest, encoding="utf-8")))
        assert rows and set(rows[0]) >= {"DDInterID_A", "DDInterID_B", "Level"}, f"{dest.name}: unexpected columns"
        for row in rows:
            pair = tuple(sorted((row["DDInterID_A"], row["DDInterID_B"])))
            if pair not in pairs:
                pairs.add(pair)
                levels[row["Level"]] = levels.get(row["Level"], 0) + 1
            drugs.update(pair)
    print(f"DDInter: {len(DDINTER_CODES)} files, {len(pairs):,} unique pairs over {len(drugs):,} drugs; levels "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(levels.items(), key=lambda kv: -kv[1])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EPILLS_ROOT / "data", help="default: epills/data")
    parser.add_argument("--only", choices=("epillid", "ddinter"))
    args = parser.parse_args()
    if args.only in (None, "epillid"):
        download_epillid(args.data_root)
    if args.only in (None, "ddinter"):
        download_ddinter(args.data_root)


if __name__ == "__main__":
    main()
