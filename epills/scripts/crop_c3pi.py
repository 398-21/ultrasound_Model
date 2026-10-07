"""
Crop every downloaded C3PI consumer photo (C3PI_Test) around its pill: data/interim/c3pi/crops/.

  1. Calibrate: find the pill without a size prior in a random sample of photos, and estimate pixels per mm for
     each camera and image width (scale_table.csv). Skipped when the table exists, unless --recalibrate.
  2. Crop: find the pill in every photo using the expected pill length (px/mm x the metadata size), and save a
     384 x 384 crop as images/<disc>/<file>.jpg. Results are appended to crops_progress.csv as they come, so the
     run can be stopped and resumed; at the end they are written to crops.csv.
  3. Report: summary.json (counts per background, crops per pill type) and a numbered review sample of 200 random
     crops (review_<seed>_0.jpg, _1.jpg, review_<seed>.csv) to check by eye.

Needs the data environment (requirements-data.txt). Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\crop_c3pi.py                 # everything
  .venv\\Scripts\\python scripts\\crop_c3pi.py --limit 300     # a quick test on the first 300 photos
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.training.datasets.c3pi_crop import expected_length, find_pill, scale_table, square_crop  # noqa: E402

CROP_SIZE = 384
RESULT_COLUMNS = ("path", "crop_path", "found", "width", "height", "x0", "y0", "x1", "y1", "side", "expected_len_px",
                  "det_score", "cnr", "solidity", "centre_dist", "size_ratio", "area_ratio", "widened", "second_pill",
                  "runner_up", "n_candidates")


def _init_worker():
    cv2.setNumThreads(1)


def calibrate_one(job):
    raw, path = job
    img = cv2.imread(str(raw / "images" / path))
    box, _, info = find_pill(img)
    long_px = max(box[2] - box[0], box[3] - box[1]) if box else math.nan
    return dict(path=path, width=img.shape[1], score=info.get("score", 0.0), long_px=long_px)


def crop_one(job):
    raw, out_dir, path, camera, size_mm, shape, colour, table = job
    img = cv2.imread(str(raw / "images" / path))
    if img is None:
        return dict(path=path, found=0, crop_path="")
    height, width = img.shape[:2]
    exp_len, spread = expected_length(table, camera, width, size_mm)
    box, side, info = find_pill(img, exp_len, spread, shape, colour)
    row = dict(path=path, width=width, height=height, expected_len_px=exp_len, found=int(box is not None), crop_path="")
    row.update({k: info.get(k) for k in RESULT_COLUMNS if k in info})
    row["det_score"] = info.get("score", 0.0)
    if box is not None:
        crop_path = Path(path).with_suffix(".jpg").as_posix()
        dest = out_dir / "images" / crop_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), square_crop(img, box, side, CROP_SIZE), [cv2.IMWRITE_JPEG_QUALITY, 95])
        row.update(crop_path=crop_path, x0=round(box[0], 1), y0=round(box[1], 1), x1=round(box[2], 1),
                   y1=round(box[3], 1), side=round(side, 1))
    return row


def run_pool(fn, jobs, workers, label):
    results, started = [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if n % 2000 == 0 or n == len(jobs):
                rate = n / (time.time() - started)
                print(f"  {label}: {n:,}/{len(jobs):,} photos, {rate:.0f}/s, about {(len(jobs) - n) / rate / 60:.0f} min left",
                      flush=True)
            yield results[-1]


def review_sheets(out_dir, crops, seed, n=200):
    """Numbered grid of n random crops (100 per sheet) for checking by eye, plus the list of what each number is."""
    sample = crops[crops.found == 1].sample(min(n, int((crops.found == 1).sum())), random_state=seed).reset_index(drop=True)
    tiles = []
    for k, row in sample.iterrows():
        tile = np.full((172, 154, 3), 255, np.uint8)
        img = cv2.imread(str(out_dir / "images" / row.crop_path))
        tile[:150, 2:152] = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, str(k), (4, 167), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 10:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
    for part, start in enumerate(range(0, len(rows), 10)):
        cv2.imwrite(str(out_dir / f"review_{seed}_{part}.jpg"), np.vstack(rows[start:start + 10]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    sample[["path", "crop_path", "background", "camera", "shape", "widened", "second_pill", "det_score"]].to_csv(
        out_dir / f"review_{seed}.csv", index_label="number")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EPILLS_ROOT / "data", help="default: epills/data")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--calibration-sample", type=int, default=4000)
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--limit", type=int, help="only the first N photos (a test run)")
    parser.add_argument("--review-seed", type=int, default=2026)
    args = parser.parse_args()

    raw = args.data_root / "raw" / "c3pi"
    out_dir = args.data_root / "interim" / "c3pi" / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.data_root / "interim" / "c3pi" / "metadata.csv", dtype=str, keep_default_na=False)
    photos = meta[(meta.image_type == "C3PI_Test") & (meta.downloaded == "1") & meta.path.str.upper().str.endswith(".JPG")]
    photos = photos.sort_values("path").reset_index(drop=True)
    if args.limit:
        photos = photos.head(args.limit)
    print(f"{len(photos):,} consumer photos to crop", flush=True)

    table_path = out_dir / "scale_table.csv"
    if args.recalibrate or not table_path.exists():
        sample = photos.sample(min(args.calibration_sample, len(photos)), random_state=0)
        found = pd.DataFrame(list(run_pool(calibrate_one, [(raw, p) for p in sample.path], args.workers, "calibrate")))
        found = found.merge(sample[["path", "camera", "size_mm"]], on="path")
        found["size_mm"] = pd.to_numeric(found.size_mm, errors="coerce")
        scale_table(found).to_csv(table_path)
    table = pd.read_csv(table_path).set_index(["camera", "width"])
    print("Pixels per mm by camera and width:\n" + table.round(2).to_string(), flush=True)

    progress_path = out_dir / "crops_progress.csv"
    done = set(pd.read_csv(progress_path, usecols=["path"]).path) if progress_path.exists() else set()
    todo = photos[~photos.path.isin(done)]
    print(f"{len(done):,} already cropped; cropping {len(todo):,} with {args.workers} processes", flush=True)
    jobs = [(raw, out_dir, r.path, r.camera, r.size_mm, r.shape, r.color, table) for r in todo.itertuples()]
    with open(progress_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if not done:
            writer.writeheader()
        for row in run_pool(crop_one, jobs, args.workers, "crop"):
            writer.writerow(row)

    results = pd.read_csv(progress_path).drop_duplicates("path", keep="last")
    crops = photos.merge(results, on="path", how="left")
    crops.to_csv(out_dir / "crops.csv", index=False)

    found = crops[crops.found == 1]
    per_type = found.groupby("ndc9").size()
    by_background = crops.groupby("background").agg(photos=("path", "size"), found=("found", "sum"),
                                                    widened=("widened", lambda s: int((s == True).sum())))  # noqa: E712
    summary = {
        "photos": len(crops), "cropped": int(crops.found.sum()), "not_found": int((crops.found != 1).sum()),
        "widened_to_expected_size": int((found.widened == True).sum()),  # noqa: E712
        "second_pill_seen": int((found.second_pill == True).sum()),  # noqa: E712
        "pill_types_ndc9": int(per_type.size), "pill_types_with_4_or_more_crops": int((per_type >= 4).sum()),
        "crops_per_pill_type": {"min": int(per_type.min()), "median": float(per_type.median()), "max": int(per_type.max())},
        "by_background": by_background.reset_index().to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    review_sheets(out_dir, crops, args.review_seed)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_background"}, indent=2))
    print(f"Crops: {out_dir / 'images'}; table: {out_dir / 'crops.csv'}; review: {out_dir}/review_{args.review_seed}_*.jpg")


if __name__ == "__main__":
    main()
