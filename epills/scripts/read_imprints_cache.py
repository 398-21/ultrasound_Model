"""
Imprint reads with the served reader, cached for the evaluation scripts (evaluate_imprint.py,
evaluate_imprint_gallery.py, calibrate_identify.py --ocr <preset>).

Presets (core.imprint.ocr): base_c70 is the reader the imprint settings were first tuned with (crops on disk);
large_c92 is the reader served on the GPU since 13 Oct: Florence-2-large on the central 92% of a crop framed as the
API frames it (a window of 1.55 x the pill box's long side, cut from the photo).
Writes data/interim/ocr/<preset>/{validation_all,c3pi_holdout,epillid_fold4}.json, keyed like the older caches.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\read_imprints_cache.py --preset large_c92
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.imprint.ocr import PRESETS, ImprintReader  # noqa: E402
from epills.core.inference.preprocess import crop_window  # noqa: E402

CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
RAW = EPILLS_ROOT / "data/raw/c3pi/images"


def served_crop(row):
    img = Image.open(RAW / row.path).convert("RGB")
    if img.size != (int(row.width), int(row.height)) and img.size[::-1] == (int(row.width), int(row.height)):
        img = img.transpose(Image.Transpose.ROTATE_90)
    return crop_window(img, tuple(float(row[k]) for k in ("x0", "y0", "x1", "y1")), 1.55)


def build(name, keys, load, reader, centre, out_dir, chunk_size=64):
    """Reads per key, cached; the next chunk's photos are decoded in threads while the GPU reads this one."""
    path = out_dir / f"{name}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [i for i, k in enumerate(keys) if k not in cache]
    chunks = [todo[s:s + chunk_size] for s in range(0, len(todo), chunk_size)]
    with ThreadPoolExecutor(8) as pool:
        pending = pool.map(load, chunks[0]) if chunks else None
        for n, chunk in enumerate(chunks):
            images = list(pending)
            if n + 1 < len(chunks):
                pending = pool.map(load, chunks[n + 1])
            for i, r in zip(chunk, reader.read(images, centre=centre)):
                cache[keys[i]] = r
            if n % 5 == 4 or n + 1 == len(chunks):
                path.write_text(json.dumps(cache))
            print(f"\r{name}: {min((n + 1) * chunk_size, len(todo)):,} / {len(todo):,}", end="", flush=True)
    path.write_text(json.dumps(cache))
    print(f"\r{name}: {len(keys):,} reads cached", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="large_c92", choices=sorted(PRESETS))
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    out_dir = EPILLS_ROOT / "data/interim/ocr" / args.preset
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = ImprintReader(preset["model"], adapter=EPILLS_ROOT / preset["adapter"] if preset.get("adapter") else None)

    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    boxes = crops.drop_duplicates("crop_path").set_index("crop_path")
    for name, splits in (("validation_all", ["val_enrolled", "val_unknown"]), ("c3pi_holdout", ["epillid_holdout"])):
        rows = pd.concat([m[m.split == s] for s in splits])
        keys = [str(CROPS / p) for p in rows.crop_path]
        if preset["served_framing"]:
            load = lambda i, rows=rows: served_crop(boxes.loc[rows.crop_path.iloc[i]])  # noqa: E731
        else:
            load = lambda i, keys=keys: Image.open(keys[i])  # noqa: E731
        build(name, keys, load, reader, preset["centre"], out_dir)

    spec = importlib.util.spec_from_file_location("ev", EPILLS_ROOT / "scripts" / "evaluate_stage2.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    f4 = pd.read_csv(ev.EPILLID / "folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/"
                                  "pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    keys = [str(ev.EPILLID / "classification_data" / p) for p in f4.image_path]
    build("epillid_fold4", keys, lambda i: Image.open(keys[i]), reader, 1.0, out_dir)  # ePillID photos are tight


if __name__ == "__main__":
    main()
