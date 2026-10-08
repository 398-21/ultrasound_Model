"""
Decision D7: crop C3PI's studio reference photos and add them to the training split as extra photos of their group.

Only reference photos of pill types whose appearance group is in the *train* split are used: references of
validation or test pills would leak their identity into training. Reference-only pill types (no everyday photos,
so no group and no look-alike check) are left out, and so are photos the cropper flagged (several blobs, or two
pill sides touching): on review, most of those 13 crops were wrong.

Writes:
  data/interim/c3pi/crops/images/references/<photo>_<n>.jpg   the crops (same root as the everyday crops)
  data/interim/c3pi/crops/references.csv                     one row per crop, with the cropper's note
  data/interim/c3pi/crops/references_review.jpg              40 random crops, for a look before training
  data/processed/c3pi_splits/manifest_with_refs.csv          manifest.csv + the reference crops (split train,
                                                             source "reference"); everyday rows get source "everyday"

Usage, from the epills folder:  .venv\\Scripts\\python scripts\\crop_c3pi_refs.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.training.datasets.c3pi_ref_crop import crop_challenge, crop_rxnav  # noqa: E402

RAW = EPILLS_ROOT / "data/raw/c3pi/images"
CROPS = EPILLS_ROOT / "data/interim/c3pi/crops"
SPLITS = EPILLS_ROOT / "data/processed/c3pi_splits"
TYPES = {"MC_CHALLENGE_V1.0": crop_challenge, "MC_API_RXNAV_V1.3": crop_rxnav}


def crop_one(row):
    path, image_type = row
    img = cv2.imread(str(RAW / path))
    if img is None:
        return [], "unreadable"
    crops, note = TYPES[image_type](img)
    out = []
    for n, crop in enumerate(crops):
        rel = Path("references") / f"{Path(path).with_suffix('').as_posix()}_{n}.jpg"
        dest = CROPS / "images" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(rel.as_posix())
    return out, note


def main():
    meta = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/metadata.csv", dtype=str, keep_default_na=False)
    groups = pd.read_csv(SPLITS / "groups.csv", dtype=str, keep_default_na=False)
    manifest = pd.read_csv(SPLITS / "manifest.csv", dtype=str, keep_default_na=False)

    refs = meta[meta.image_type.isin(TYPES) & (meta.downloaded == "1")].copy()
    refs["pill_type"] = refs.ndc9 + "_" + refs.part
    refs = refs.merge(groups[["pill_type", "group", "split"]], on="pill_type", how="left")
    print("reference photos by split of their pill type:\n",
          pd.crosstab(refs.image_type, refs.split.replace("", "none").fillna("no everyday photos")), flush=True)
    refs = refs[refs.split == "train"].reset_index(drop=True)

    with Pool(12) as pool:
        results = pool.map(crop_one, list(zip(refs.path, refs.image_type)), chunksize=8)
    rows = []
    for (_, r), (paths, note) in zip(refs.iterrows(), results):
        for n, p in enumerate(paths):
            rows.append({**r.to_dict(), "crop_path": p, "side_index": n, "note": note})
        if not paths:
            rows.append({**r.to_dict(), "crop_path": "", "side_index": -1, "note": note})
    out = pd.DataFrame(rows)
    out.to_csv(CROPS / "references.csv", index=False)
    print("\ncropper notes:\n", out.drop_duplicates("path").groupby("image_type").note.value_counts(), flush=True)

    ok = out[(out.crop_path != "") & (out.note == "ok")]  # flagged photos (0.3%) were mostly wrong on review
    add = pd.DataFrame({c: ok[c] if c in ok else "" for c in manifest.columns})
    add["split"], add["label"] = "train", ok.group.to_numpy()
    full = pd.concat([manifest.assign(source="everyday"), add.assign(source="reference")], ignore_index=True)
    full.to_csv(SPLITS / "manifest_with_refs.csv", index=False)
    train = full[full.split == "train"]
    print(f"\nadded {len(add):,} reference crops of {add.label.nunique():,} training groups "
          f"({train.label.nunique():,} groups in train); train photos {len(manifest[manifest.split == 'train']):,} "
          f"-> {len(train):,}", flush=True)

    # review sheet: 40 random crops
    sample = ok.sample(min(40, len(ok)), random_state=0)
    t = 128
    sheet = Image.new("RGB", (10 * t, 4 * t), "white")
    for i, p in enumerate(sample.crop_path):
        sheet.paste(Image.open(CROPS / "images" / p).convert("RGB").resize((t, t)), ((i % 10) * t, (i // 10) * t))
    sheet.save(CROPS / "references_review.jpg", quality=90)


if __name__ == "__main__":
    main()
