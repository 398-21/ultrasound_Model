"""
ePillID official protocol (Usuyama et al. 2020): models may train on the reference photos of *all* 4,902 pill types,
including the holdout (fold 4) types, whose references are also the test gallery. This builds the training manifest
for a benchmark run under that protocol: manifest_with_refs.csv (runs 3-4) + ePillID's own reference photos.

  pill types in our train or epillid_holdout groups   added with that group's label (holdout = the official
                                                      protocol's point: their references are trained on)
  pill types with no C3PI group (reference-only)      added as new classes "ep:<pilltype_id>", unless C3PI's
                                                      metadata shows they look like a validation pill
  pill types in validation groups, or ambiguous       left out, so validation stays clean (stricter than the
                                                      paper, where every reference is trained on)

A model trained on this manifest is a **benchmark model only**: it has seen the test types' references, so the
C3PI held-out test and the unknown-pill tests are not valid for it (validation and MEDISEG still are). The app
keeps the strict model. ePillID consumer photos are never added (fold 4's are the test queries).

Crops: ePillID's references are tight 224 px crops on black; they are padded to the reference-crop framing (window
1.4x, border colour) and stored at 384 px under data/interim/c3pi/crops/images/epillid_refs/.

Usage, from the epills folder:  .venv\\Scripts\\python scripts\\add_epillid_refs.py
"""
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.training.datasets.epillid import EPILLID, pad_crop  # noqa: E402

SPLITS = EPILLS_ROOT / "data/processed/c3pi_splits"
CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"


def appearance_key(r):
    """As scripts/make_splits.py: imprint + shape + colour + size."""
    return f"{re.sub(r'[^A-Z0-9;]', '', str(r.imprint).upper())}|{r['shape']}|{r.color}|{r.size_mm}"


def main():
    refs = pd.read_csv(EPILLID / "all_labels.csv")
    refs = refs[refs.is_ref].copy()
    refs["ndc9"] = (refs.label_code_id.astype(int).map("{:05d}".format)
                    + refs.prod_code_id.astype(int).map("{:04d}".format))
    groups = pd.read_csv(SPLITS / "groups.csv", dtype=str, keep_default_na=False)
    by_ndc = groups.groupby("ndc9").agg(group=("group", lambda s: sorted(set(s))),
                                        split=("split", lambda s: sorted(set(s))))
    refs = refs.join(by_ndc, on="ndc9")

    meta = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/metadata.csv", dtype=str, keep_default_na=False)
    first = meta.drop_duplicates("ndc9").set_index("ndc9")
    val_keys = set(groups[groups.split.isin(["val_enrolled", "val_unknown"]) & (groups.imprint_norm != "")]
                   .appearance_key)

    rows, skipped = [], {}
    for _, r in refs.iterrows():
        if isinstance(r.group, list):
            if r.split == ["epillid_holdout"] and len(r.group) > 1:
                # a multi-part pack among the test types: which part this reference shows is unknown, so it gets its
                # own class (the test types' everyday photos are never trained on, so nothing clashes)
                label = f"ep:{r.pilltype_id}"
            elif len(r.group) > 1:
                skipped["several C3PI groups"] = skipped.get("several C3PI groups", 0) + 1
                continue
            elif r.split[0] not in ("train", "epillid_holdout"):
                skipped["validation pill"] = skipped.get("validation pill", 0) + 1
                continue
            else:
                label = r.group[0]
        else:
            label = f"ep:{r.pilltype_id}"
            if r.ndc9 in first.index and appearance_key(first.loc[r.ndc9]) in val_keys:
                skipped["looks like a validation pill"] = skipped.get("looks like a validation pill", 0) + 1
                continue
        m = first.loc[r.ndc9] if r.ndc9 in first.index else None
        crop_path = f"epillid_refs/{Path(r.image_path).name}"
        dest = CROPS / crop_path
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), pad_crop(EPILLID / "classification_data" / r.image_path),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"crop_path": crop_path, "path": r.image_path, "split": "train", "label": label, "group": label,
                     "pill_type": r.pilltype_id, "ndc9": r.ndc9,
                     "shape": m["shape"] if m is not None else "", "color": m.color if m is not None else "",
                     "source": "epillid_ref"})
    add = pd.DataFrame(rows)
    manifest = pd.read_csv(SPLITS / "manifest_with_refs.csv", dtype=str, keep_default_na=False)
    full = pd.concat([manifest, add.reindex(columns=manifest.columns, fill_value="")], ignore_index=True)
    full.to_csv(SPLITS / "manifest_epillid_official.csv", index=False)
    train = full[full.split == "train"]
    holdout = add[add.label.isin(set(groups[groups.split == "epillid_holdout"].group))]
    print(f"added {len(add):,} ePillID reference crops ({add.label.nunique():,} labels, of which "
          f"{add.label.str.startswith('ep:').sum():,} photos in new reference-only classes; "
          f"{len(holdout)} photos of holdout types); skipped {skipped}")
    print(f"train: {len(train):,} photos, {train.label.nunique():,} labels "
          f"(manifest_with_refs: {(manifest.split == 'train').sum():,} photos)")


if __name__ == "__main__":
    main()
