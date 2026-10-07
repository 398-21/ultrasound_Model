"""
Download the selected parts of NLM's C3PI pill-image archive into data/raw/c3pi/.

What it downloads (details in docs/datasets.md, "NLM C3PI"):
  1. Metadata: the archive's README files, the image directory and the reference-image list (meta/), and the 110
     per-disc metadata files (ALLXML/, 130 MB), which give every image's size and SHA-1.
  2. Images of the selected types (default about 89 GB), saved under images/ with the archive's own paths,
     e.g. images/PillProjectDisc10/images/<name>.JPG:
       C3PI_Test           40,882 consumer-grade photos (JPG only)   84.0 GB
       MC_CHALLENGE_V1.0    4,408 reference images (2016 challenge)   2.2 GB
       MC_API_RXNAV_V1.3    4,708 RxImage reference images            2.7 GB
  3. Optionally rximage.zip (7.3 GB, checked against rximage.MD5).

Every image is checked against the SHA-1 in its disc's metadata file before it is kept, and a partly downloaded
file never gets its final name. The script can be stopped at any time and rerun: finished files are skipped.
Reference images come first. Consumer photos follow in rounds (one photo of every pill type, then a second, ...),
so a partial download still covers every pill type evenly.

Usage, from the epills folder:
  python scripts/download_c3pi.py                    # metadata + the three image types above
  python scripts/download_c3pi.py --limit 200        # a quick test: the first 200 images in download order
  python scripts/download_c3pi.py --metadata-only    # metadata only
  python scripts/download_c3pi.py --with-rximage     # also rximage.zip
  python scripts/download_c3pi.py --verify           # re-check the SHA-1 of every image already downloaded

Outputs besides the raw files (data/interim/c3pi/): download_manifest.csv (every selected image with its pill
type, size and SHA-1), download_failures.csv (only if something failed) and download_summary.json.
"""
import argparse
import csv
import hashlib
import http.client
import json
import os
import shutil
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://data.lhncbc.nlm.nih.gov/public/Pills"
USER_AGENT = "epills-c3pi-download/1.0 (research use)"
EPILLS_ROOT = Path(__file__).resolve().parents[1]

META_FILES = ("AAREADME", "README", "directory_consumer_grade_images.txt", "directory_reference_images.xls",
              "RxImageImageLayouts.docx", "rximage.MD5")
DIRECTORY = "directory_consumer_grade_images.txt"
DIRECTORY_SHA256 = "69ac367991b2cfa204219e5bd3fd7016f2be280771a5517d4fbc3cfe56b84302"  # checked 2026-10-06
N_DISCS = 110
DEFAULT_TYPES = ("MC_CHALLENGE_V1.0", "MC_API_RXNAV_V1.3", "C3PI_Test")  # download order: references first
CONSUMER_TYPE = "C3PI_Test"
IMAGE_EXTENSION = ".JPG"
MANIFEST_COLUMNS = ("order", "ndc11", "part", "image_type", "name", "path", "size", "sha1")


class CheckFailed(Exception):
    """A downloaded file has the wrong size or checksum."""


def fetch(url, dest, size=None, digest=None, algorithm="sha1", resume=False, attempts=5, timeout=60):
    """
    Download url to dest via dest.part, check size and digest, then give it its final name. Retries with backoff.
    With resume=True an existing .part file is continued (HTTP Range); otherwise every attempt starts over.
    Returns the number of bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            start = part.stat().st_size if resume and part.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if start:
                headers["Range"] = f"bytes={start}-"
            hasher = hashlib.new(algorithm)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                if start and response.status != 206:
                    start = 0  # the server ignored the range: start over
                with open(part, "ab" if start else "wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        hasher.update(chunk)
            got = part.stat().st_size
            if size is not None and got != size:
                raise CheckFailed(f"{got:,} bytes, expected {size:,}")
            if digest is not None:
                actual = file_digest(part, algorithm) if start else hasher.hexdigest()
                if actual != digest.lower():
                    raise CheckFailed(f"{algorithm.upper()} {actual}, expected {digest.lower()}")
            os.replace(part, dest)
            return got
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in (403, 404, 410):
                break  # retrying won't help
        except (CheckFailed, urllib.error.URLError, http.client.HTTPException, OSError) as error:
            last_error = error
            if isinstance(error, CheckFailed):
                part.unlink(missing_ok=True)  # a resumed file that fails its check must start over
        if not resume:
            part.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(2 ** attempt)
    if not resume:
        part.unlink(missing_ok=True)
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def file_digest(path, algorithm="sha1"):
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_metadata(raw):
    """README files, the image directory, the reference list and the 110 per-disc metadata files."""
    for name in META_FILES:
        dest = raw / "meta" / name
        if not dest.exists():
            fetch(f"{BASE_URL}/{name}", dest)
    actual = file_digest(raw / "meta" / DIRECTORY, "sha256")
    if actual != DIRECTORY_SHA256:
        sys.exit(f"meta/{DIRECTORY} has SHA-256 {actual}, not the {DIRECTORY_SHA256} checked on 2026-10-06. "
                 "The archive has changed: update docs/datasets.md and this script before downloading.")
    for disc in range(1, N_DISCS + 1):
        dest = raw / "ALLXML" / f"PillProjectDisc{disc}.xml"
        for attempt in (1, 2):
            if not dest.exists():
                fetch(f"{BASE_URL}/ALLXML/{dest.name}", dest)
            try:
                ET.parse(dest)  # a complete, well-formed file
                break
            except ET.ParseError:
                dest.unlink()
                if attempt == 2:
                    raise
    print(f"Metadata in place: {raw / 'meta'} and {N_DISCS} files in {raw / 'ALLXML'}", flush=True)


def disc_files(xml_path):
    """{file name (upper case): (size, sha1)} for every image a disc's metadata file lists."""
    files = {}
    for image in ET.parse(xml_path).getroot().iter("Image"):
        name = image.findtext("File/Name")
        if name:
            files[name.upper()] = (int(image.findtext("File/Size")), image.findtext("File/Sha1").lower())
    return files


def build_manifest(raw, image_types):
    """
    Every selected image once, in download order: reference types first, then consumer photos in rounds (the
    k-th photo of every pill type before any type's (k+1)-th). A pill type is NDC11 + part.
    """
    rows, seen = [], set()
    with open(raw / "meta" / DIRECTORY, encoding="latin-1", newline="") as fh:
        for ndc11, part, path, image_type, *name in csv.reader(fh, delimiter="|", quoting=csv.QUOTE_NONE):
            name = "|".join(name)
            if image_type in image_types and path.upper().endswith(IMAGE_EXTENSION) and path not in seen:
                seen.add(path)
                rows.append({"ndc11": ndc11, "part": part, "image_type": image_type, "name": name, "path": path})

    checks = {}
    for disc in sorted({row["path"].split("/")[0] for row in rows}):
        checks.update({(disc, file): value for file, value in disc_files(raw / "ALLXML" / f"{disc}.xml").items()})
    missing = 0
    for row in rows:
        disc, _, file = row["path"].split("/")
        row["size"], row["sha1"] = checks.get((disc, file.upper()), (None, None))
        missing += row["size"] is None
    if missing:
        print(f"Warning: {missing} selected images have no entry in their disc's metadata file; they are checked "
              "against nothing but the download itself.", flush=True)

    rank = Counter()
    for row in sorted(rows, key=lambda r: r["path"]):
        row["round"] = rank[(row["ndc11"], row["part"])] if row["image_type"] == CONSUMER_TYPE else -1
        rank[(row["ndc11"], row["part"])] += row["image_type"] == CONSUMER_TYPE
    type_order = {t: i for i, t in enumerate(image_types)}
    rows.sort(key=lambda r: (r["round"], type_order[r["image_type"]], r["ndc11"], r["part"], r["path"]))
    for order, row in enumerate(rows):
        row["order"] = order
    return rows


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def local_path(raw, row):
    return raw / "images" / row["path"]


def is_done(raw, row):
    """Finished files have their final name and the right size (they only get that name after the checks)."""
    path = local_path(raw, row)
    return path.exists() and (row["size"] is None or path.stat().st_size == row["size"])


def download_images(raw, rows, workers):
    """Download the rows not yet present, several at a time. Returns the rows that failed, with the reason."""
    todo = [row for row in rows if not is_done(raw, row)]
    total_bytes = sum(row["size"] or 0 for row in todo)
    free = shutil.disk_usage(raw).free
    print(f"{len(rows) - len(todo):,} of {len(rows):,} images already present; downloading {len(todo):,} "
          f"({total_bytes / 1e9:.1f} GB) with {workers} parallel downloads. Free disk: {free / 1e9:.0f} GB.", flush=True)
    if total_bytes > free - 5e9:
        sys.exit("Not enough free disk space (need the download size plus 5 GB spare).")

    lock, failures = threading.Lock(), []
    done = {"files": 0, "bytes": 0}
    started = last_report = time.time()

    def one(row):
        url = f"{BASE_URL}/{urllib.request.quote(row['path'])}"
        return fetch(url, local_path(raw, row), row["size"], row["sha1"])

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(one, row): row for row in todo}
        for future in as_completed(futures):
            row = futures[future]
            try:
                size = future.result()
            except RuntimeError as error:
                failures.append({**row, "error": str(error)})
                size = 0
            with lock:
                done["files"] += 1
                done["bytes"] += size
                now = time.time()
                if now - last_report >= 30 or done["files"] == len(todo):
                    last_report = now
                    rate = done["bytes"] / max(now - started, 1e-9)
                    eta = (total_bytes - done["bytes"]) / rate if rate else float("inf")
                    print(f"  {done['files']:,}/{len(todo):,} files, {done['bytes'] / 1e9:.1f}/{total_bytes / 1e9:.1f} GB, "
                          f"{rate / 1e6:.1f} MB/s, about {eta / 60:.0f} min left, {len(failures)} failed", flush=True)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nStopped. Finished files are kept; run the same command again to continue.", flush=True)
        raise SystemExit(130)
    pool.shutdown()
    return failures


def download_rximage(raw):
    """rximage.zip (7.3 GB), resumable, checked against the archive's MD5."""
    dest = raw / "rximage.zip"
    md5 = (raw / "meta" / "rximage.MD5").read_text().split()[0].lower()
    if dest.exists() and file_digest(dest, "md5") == md5:
        print("rximage.zip already present and checked.", flush=True)
        return
    print("Downloading rximage.zip (7.3 GB) ...", flush=True)
    fetch(f"{BASE_URL}/rximage.zip", dest, digest=md5, algorithm="md5", resume=True, attempts=20, timeout=120)
    print("rximage.zip downloaded and checked against rximage.MD5.", flush=True)


def verify_images(raw, rows):
    """Re-check the SHA-1 of every downloaded image; a corrupt file is deleted so the next run fetches it again."""
    present = [row for row in rows if local_path(raw, row).exists()]
    bad = [row for row in present if row["sha1"] and file_digest(local_path(raw, row)) != row["sha1"]]
    for row in bad:
        local_path(raw, row).unlink()
    print(f"Verified {len(present):,} images: {len(bad)} failed the SHA-1 check"
          + (" and were deleted; rerun the download to fetch them again." if bad else "."), flush=True)
    return bad


def summarise(raw, rows, failures, image_types):
    """Counts per image type and photos per pill type for what is on disk now."""
    summary = {"checked_at": time.strftime("%Y-%m-%d %H:%M"), "image_types": {}, "failed": len(failures)}
    for image_type in image_types:
        selected = [row for row in rows if row["image_type"] == image_type]
        present = [row for row in selected if is_done(raw, row)]
        summary["image_types"][image_type] = {
            "selected": len(selected), "present": len(present),
            "gb_present": round(sum(local_path(raw, r).stat().st_size for r in present) / 1e9, 2),
            "pill_types_present": len({(r["ndc11"], r["part"]) for r in present}),
        }
    consumer = Counter((r["ndc11"], r["part"]) for r in rows if r["image_type"] == CONSUMER_TYPE and is_done(raw, r))
    if consumer:
        counts = sorted(consumer.values())
        summary["consumer_photos_per_pill_type"] = {
            "min": counts[0], "median": statistics.median(counts), "max": counts[-1],
            "types_with_at_least_4": sum(c >= 4 for c in counts)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EPILLS_ROOT / "data", help="default: epills/data")
    parser.add_argument("--types", default=",".join(DEFAULT_TYPES), help="comma-separated image types")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads (default 16)")
    parser.add_argument("--limit", type=int, help="only the first N images in download order (a test run)")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--with-rximage", action="store_true", help="also download rximage.zip (7.3 GB)")
    parser.add_argument("--verify", action="store_true", help="re-check the SHA-1 of downloaded images, no download")
    args = parser.parse_args()

    raw = args.data_root / "raw" / "c3pi"
    interim = args.data_root / "interim" / "c3pi"
    image_types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    raw.mkdir(parents=True, exist_ok=True)

    download_metadata(raw)
    if args.metadata_only:
        return
    rows = build_manifest(raw, image_types)
    write_csv(interim / "download_manifest.csv", rows, MANIFEST_COLUMNS)
    per_type = Counter(row["image_type"] for row in rows)
    print("Selected: " + ", ".join(f"{t} {per_type[t]:,}" for t in image_types)
          + f"; {sum(r['size'] or 0 for r in rows) / 1e9:.1f} GB in total. Manifest: {interim / 'download_manifest.csv'}",
          flush=True)

    if args.verify:
        failures = [{**row, "error": "SHA-1 mismatch"} for row in verify_images(raw, rows)]
    else:
        failures = download_images(raw, rows if args.limit is None else rows[:args.limit], args.workers)
        if args.with_rximage:
            download_rximage(raw)

    failures_path = interim / "download_failures.csv"
    if failures:
        write_csv(failures_path, failures, MANIFEST_COLUMNS + ("error",))
    elif failures_path.exists():
        failures_path.unlink()
    summary = summarise(raw, rows, failures, image_types)
    (interim / "download_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary (also in download_summary.json):\n" + json.dumps(summary, indent=2), flush=True)
    if failures:
        print(f"\n{len(failures)} images failed (listed in {failures_path}). Rerun the same command to retry them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
