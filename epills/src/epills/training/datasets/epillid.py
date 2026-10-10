"""
ePillID reference photos as crops in our framing, and the mapping between ePillID pill types and NDC9.

ePillID's references are tight 224 px crops on black; pad_crop pads them to the reference-crop framing (window 1.4x,
border colour) at 384 px, stored under <crops root>/epillid_refs/. Used by scripts/add_epillid_refs.py (training
data, official protocol) and by the training loop's benchmark-aligned validation metric (all 4,902 types).
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EPILLID = Path(__file__).resolve().parents[4] / "data/raw/epillid/ePillID_data"
WINDOW, SIZE = 1.4, 384


def pad_crop(path):
    """A tight crop -> square window WINDOW x its long side, filled with the median colour of its border."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    ring = np.concatenate([img[:2].reshape(-1, 3), img[-2:].reshape(-1, 3), img[:, :2].reshape(-1, 3),
                           img[:, -2:].reshape(-1, 3)])
    f = SIZE / (WINDOW * max(h, w))
    M = np.float32([[f, 0, SIZE / 2 - w / 2 * f], [0, f, SIZE / 2 - h / 2 * f]])
    return cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=tuple(float(v) for v in np.median(ring, axis=0)))


def references(crops_root=None):
    """All 9,804 ePillID references: pilltype_id, ndc9, image_path, crop_path (crops written if missing)."""
    refs = pd.read_csv(EPILLID / "all_labels.csv")
    refs = refs[refs.is_ref].reset_index(drop=True)
    refs["ndc9"] = (refs.label_code_id.astype(int).map("{:05d}".format)
                    + refs.prod_code_id.astype(int).map("{:04d}".format))
    refs["crop_path"] = "epillid_refs/" + refs.image_path.map(lambda p: Path(p).name)
    if crops_root is not None:
        for rel, src in zip(refs.crop_path, refs.image_path):
            dest = Path(crops_root) / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), pad_crop(EPILLID / "classification_data" / src), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return refs
