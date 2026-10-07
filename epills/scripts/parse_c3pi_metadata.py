"""
Parse C3PI's 110 per-disc metadata files (data/raw/c3pi/ALLXML/) into one table, one row per image:
data/interim/c3pi/metadata.csv.

Columns: disc, path (as in the archive and under data/raw/c3pi/images/), image_type (Class for consumer and
studio photos, Layout for the processed reference renderings), the pill's identity and attributes, the capture
conditions and clarity ratings (consumer photos only), file size and SHA-1. `color` keeps both colours of a
two-tone pill, joined by ";" (the XML repeats the Color element). `downloaded` says whether the image is under
data/raw/c3pi/images/.

Also prints the attribute coverage for the downloaded consumer photos, per pill type (NDC9), for gate G3.
Standard library only. Usage, from the epills folder:  python scripts/parse_c3pi_metadata.py
"""
import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

EPILLS_ROOT = Path(__file__).resolve().parents[1]
CONSUMER_TYPE = "C3PI_Test"

# (column, XML path) in output order; Color is handled separately (it can repeat)
FIELDS = (
    ("ndc9", "NDC9"), ("ndc11", "NDC11"), ("part", "Part"), ("parts", "Parts"), ("labeled_by", "LabeledBy"),
    ("proprietary_name", "ProprietaryName"), ("generic_name", "GenericName"),
    ("shape", "Shape"), ("size_mm", "Size"), ("imprint", "Imprint"), ("imprint_type", "ImprintType"),
    ("imprint_color", "ImprintColor"), ("score", "Score"), ("symbol", "Symbol"),
    ("camera", "Camera"), ("illumination", "Illumination"), ("background", "Background"), ("polarizer", "Polarizer"),
    ("rating_imprint", "RatingImprint"), ("rating_shape", "RatingShape"), ("rating_color", "RatingColor"),
    ("rating_shadow", "RatingShadow"), ("rating_background", "RatingBackground"),
    ("acquisition_date", "AcquisitionDate"), ("medicos_id", "MedicosConsultantsID"),
    ("file_type", "File/Type"), ("file_size", "File/Size"), ("sha1", "File/Sha1"),
)
COLUMNS = ("disc", "path", "image_type") + tuple(c for c, _ in FIELDS[:8]) + ("color",) + \
          tuple(c for c, _ in FIELDS[8:]) + ("downloaded",)


def text(element, path):
    value = element.findtext(path)
    return value.strip() if value else ""


def parse(raw):
    """One row per image file. The XML lists 316 consumer files twice, with identical entries; the repeats are dropped."""
    rows, seen, repeats = [], set(), 0
    for xml_path in sorted((raw / "ALLXML").glob("PillProjectDisc*.xml"), key=lambda p: int(p.stem[15:])):
        disc = xml_path.stem
        for image in ET.parse(xml_path).getroot().iter("Image"):
            row = {"disc": disc, "path": f"{disc}/images/{text(image, 'File/Name')}",
                   "image_type": text(image, "Class") or text(image, "Layout")}
            row.update({column: text(image, path) for column, path in FIELDS})
            row["color"] = ";".join(c.text.strip() for c in image.findall("Color") if c.text and c.text.strip())
            if row["path"].upper() in seen:
                repeats += 1
                continue
            seen.add(row["path"].upper())
            row["downloaded"] = int((raw / "images" / row["path"]).exists())
            rows.append(row)
    return rows, repeats


def coverage(rows):
    """Share of downloaded consumer pill types (NDC9) whose photos carry each attribute."""
    consumer = [r for r in rows if r["image_type"] == CONSUMER_TYPE and r["downloaded"]]
    by_type = defaultdict(list)
    for r in consumer:
        by_type[r["ndc9"]].append(r)
    print(f"\nDownloaded consumer photos: {len(consumer):,} of {len(by_type):,} pill types (NDC9)")
    for column in ("shape", "color", "size_mm", "imprint", "imprint_type", "score"):
        has = sum(any(r[column] not in ("", "0") for r in photos) for photos in by_type.values())
        conflicting = sum(len({r[column] for r in photos if r[column]}) > 1 for photos in by_type.values())
        print(f"  {column:13s} present for {has:5,} types ({has / len(by_type):6.1%}); "
              f"types with conflicting values: {conflicting}")
    two_tone = sum(any(";" in r["color"] for r in photos) for photos in by_type.values())
    print(f"  two-tone colour (2 values) for {two_tone:,} types")
    for column in ("shape", "color"):
        values = Counter(r[column] for r in consumer)
        print(f"  {column} values (photos): " + ", ".join(f"{v or '(empty)'} {n:,}" for v, n in values.most_common(14)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EPILLS_ROOT / "data", help="default: epills/data")
    args = parser.parse_args()
    raw = args.data_root / "raw" / "c3pi"
    rows, repeats = parse(raw)
    out = args.data_root / "interim" / "c3pi" / "metadata.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    types = Counter(r["image_type"] for r in rows)
    print(f"{len(rows):,} images from {len({r['disc'] for r in rows})} metadata files -> {out} "
          f"({repeats} repeated entries dropped)")
    print("  by image type: " + ", ".join(f"{t} {n:,}" for t, n in types.most_common()))
    print(f"  downloaded: {sum(r['downloaded'] for r in rows):,}")
    coverage(rows)


if __name__ == "__main__":
    main()
