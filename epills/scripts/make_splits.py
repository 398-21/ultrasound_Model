"""
Splits for Stage 2 training: C3PI consumer crops -> train / val_enrolled / val_unknown, with the ePillID holdout
removed, plus a leakage report. See ROADMAP §1.3.

Units:
  pill type         NDC9 + part. One product code can hold several different pills (multi-pill packs), so the part
                    is needed to tell them apart. This is also how ePillID defines its classes.
  appearance group  pill types with the same imprint, shape, colour and size: the same physical pill sold under
                    several product codes (repackagers, brand vs generic from one maker). Splits are made by group,
                    and training uses the group as the label, because two classes with identical photos would only
                    confuse the metric loss.

Steps:
  1. Group pill types by appearance (union-find; a pill type without an imprint stays its own group).
  2. Remove every group that contains an ePillID holdout pill type (fold 4, matched on NDC9). These are the official
     test medications, so no photo of them, or of a look-alike, may be seen in training.
  3. Split the remaining groups at random (seeded) into train / val_enrolled / val_unknown (default 80 / 10 / 10).
  4. Leakage checks: shared groups and NDC9s across splits, identical files (SHA-1), and near-identical crops
     (look-alikes by 32x32 thumbnail correlation, then copies by centre texture and pixel difference).

Outputs in data/processed/c3pi_splits/: manifest.csv (one row per crop), groups.csv, leakage_report.md,
summary.json, near_duplicates_across_splits.csv. Run from the epills folder: python scripts/make_splits.py
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EPILLS_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_FOLD = 4  # ePillID's official test fold (train_cv.py holds out the last sorted fold file)
THUMB = 32
NEAR_DUP_CORR = 0.97  # thumbnail correlation for look-alike candidates
TEXTURE_DUP_CORR = 0.8  # centre-texture correlation of a copy (JPEG-70 re-encoded, 1 px shifted copies: 0.90-0.97)
PIXEL_DUP_DIFF = 3.0  # and mean absolute pixel difference of the centre (copies: median 1.8; look-alikes: >= 3.9)


def norm_imprint(text):
    return re.sub(r"[^A-Z0-9;]", "", str(text).upper())


def appearance_groups(types):
    """Union-find over pill types that share imprint + shape + colour + size. Returns {pill_type: group id}."""
    parent = {t: t for t in types.pill_type}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keyed = types[types.imprint_norm != ""]
    for _, members in keyed.groupby("appearance_key").pill_type:
        members = list(members)
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    roots = {t: find(t) for t in parent}
    ids = {root: f"g{i:04d}" for i, root in enumerate(sorted(set(roots.values())))}
    return {t: ids[r] for t, r in roots.items()}


def thumbnail(path):
    """32x32 colour thumbnail of a crop, flattened and normalised to zero mean and unit length (for correlation)."""
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros(THUMB * THUMB * 3, np.float32)
    v = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def centre_pixels(path, n=128):
    """The crop's central 60% at n x n, as float colour pixels."""
    g = cv2.imread(str(path))
    h = g.shape[0]
    return cv2.resize(g[int(0.2 * h):int(0.8 * h), int(0.2 * h):int(0.8 * h)], (n, n),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def texture(path, n=128):
    """Fine texture (image minus its blur) of the crop's central 60%, normalised. Copies of one photo keep the same
    sensor noise and paper grain (correlation ~0.97 even after re-encoding and a 1 px shift); different photos don't.
    The centre avoids the edge stripes of crops padded past the photo border, which all look alike."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h = g.shape[0]
    g = cv2.resize(g[int(0.2 * h):int(0.8 * h), int(0.2 * h):int(0.8 * h)], (n, n), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    v = (g - cv2.GaussianBlur(g, (0, 0), 3)).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def near_duplicates(vectors, split, threshold=NEAR_DUP_CORR, chunk=2048):
    """Candidates: for every crop, its most correlated thumbnail in a different split, above threshold (each pair
    once). Plain pills on the same background pass this easily; texture() then tells copies from look-alikes."""
    pairs = []
    for start in range(0, len(vectors), chunk):
        block = vectors[start:start + chunk] @ vectors.T
        other = split[start:start + chunk, None] != split[None, :]
        block[~other] = -1
        best = block.argmax(axis=1)
        corr = block[np.arange(len(best)), best]
        for k in np.flatnonzero(corr >= threshold):
            i, j = start + k, int(best[k])
            pairs.append((min(i, j), max(i, j), float(corr[k])))
    return sorted(set(pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EPILLS_ROOT / "data")
    parser.add_argument("--val-fraction", type=float, default=0.10, help="share of groups for val_enrolled and for val_unknown each")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    crops_dir = args.data_root / "interim" / "c3pi" / "crops"
    out_dir = args.data_root / "processed" / "c3pi_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = pd.read_csv(crops_dir / "crops.csv", dtype=str, keep_default_na=False)
    crops = crops[crops.found.isin(["1", "True"])].copy()
    crops["pill_type"] = crops.ndc9 + "_" + crops.part
    crops["imprint_norm"] = crops.imprint.map(norm_imprint)
    crops["appearance_key"] = crops.imprint_norm + "|" + crops["shape"] + "|" + crops.color + "|" + crops.size_mm

    # 1. appearance groups (one row per pill type; the first metadata value of each type)
    types = crops.groupby("pill_type").agg(ndc9=("ndc9", "first"), part=("part", "first"),
                                           imprint_norm=("imprint_norm", "first"),
                                           appearance_key=("appearance_key", "first"),
                                           name=("proprietary_name", "first"), photos=("path", "size")).reset_index()
    group_of = appearance_groups(types)
    types["group"] = types.pill_type.map(group_of)
    crops["group"] = crops.pill_type.map(group_of)

    # 2. ePillID holdout
    fold_dir = next((args.data_root / "raw" / "epillid").rglob("pilltypeid_nih_sidelbls0.01_metric_5folds/base"))
    holdout = pd.read_csv(next(fold_dir.glob(f"*_5folds_{HOLDOUT_FOLD}.csv")), dtype=str)
    holdout_ndc9 = set(holdout.label_code_id.str.zfill(5) + holdout.prod_code_id.str.zfill(4))
    holdout_groups = set(types.loc[types.ndc9.isin(holdout_ndc9), "group"])
    lookalike_extra = types[types.group.isin(holdout_groups) & ~types.ndc9.isin(holdout_ndc9)]

    # 3. split the remaining groups
    groups = sorted(set(types.group) - holdout_groups)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    n_val = round(args.val_fraction * len(groups))
    split_of = {g: "val_enrolled" for g in groups[:n_val]}
    split_of.update({g: "val_unknown" for g in groups[n_val:2 * n_val]})
    split_of.update({g: "train" for g in groups[2 * n_val:]})
    split_of.update({g: "epillid_holdout" for g in holdout_groups})
    crops["split"] = crops.group.map(split_of)
    types["split"] = types.group.map(split_of)

    # 4. leakage checks
    report = {}
    report["groups_in_several_splits"] = int((crops.groupby("group").split.nunique() > 1).sum())
    report["ndc9_in_several_splits"] = int((crops.groupby("ndc9").split.nunique() > 1).sum())
    sha_splits = crops.groupby("sha1").split.nunique()
    report["identical_files_across_splits"] = int((sha_splits > 1).sum())
    report["identical_files_total_repeats"] = int((crops.sha1.duplicated()).sum())
    paths = [crops_dir / "images" / p for p in crops.crop_path]
    with ProcessPoolExecutor(args.workers) as pool:
        vectors = np.stack(list(pool.map(thumbnail, paths, chunksize=256)))
    split_arr, group_arr = crops.split.to_numpy(), crops.group.to_numpy()
    candidates = near_duplicates(vectors, split_arr)
    rows = []
    for i, j, c in candidates:
        tex = float(texture(paths[i]) @ texture(paths[j]))
        diff = float(np.abs(centre_pixels(paths[i]) - centre_pixels(paths[j])).mean())
        rows.append(dict(texture_corr=round(tex, 4), pixel_diff=round(diff, 2), thumb_corr=round(c, 4),
                         duplicate=tex >= TEXTURE_DUP_CORR and diff <= PIXEL_DUP_DIFF,
                         a=crops.crop_path.iat[i], b=crops.crop_path.iat[j], split_a=split_arr[i], split_b=split_arr[j],
                         name_a=crops.proprietary_name.iat[i], name_b=crops.proprietary_name.iat[j],
                         imprint_a=crops.imprint.iat[i], imprint_b=crops.imprint.iat[j]))
    cand = pd.DataFrame(rows, columns=["texture_corr", "pixel_diff", "thumb_corr", "duplicate", "a", "b", "split_a", "split_b",
                                       "name_a", "name_b", "imprint_a", "imprint_b"])
    cand.sort_values("texture_corr", ascending=False).to_csv(out_dir / "near_duplicates_across_splits.csv", index=False)
    report["lookalike_candidates_across_splits"] = len(cand)
    report["max_texture_corr_across_splits"] = round(float(cand.texture_corr.max()), 3) if len(cand) else None
    report["min_pixel_diff_across_splits"] = round(float(cand.pixel_diff.min()), 2) if len(cand) else None
    report["near_duplicate_pairs_across_splits"] = int(cand.duplicate.sum())

    # outputs
    crops["label"] = crops.group  # training label = appearance group
    keep = ["crop_path", "path", "split", "label", "group", "pill_type", "ndc9", "ndc11", "part", "proprietary_name",
            "generic_name", "shape", "color", "size_mm", "imprint", "camera", "background", "illumination", "sha1"]
    crops[keep].to_csv(out_dir / "manifest.csv", index=False)
    types.to_csv(out_dir / "groups.csv", index=False)

    per_split = crops.groupby("split").agg(photos=("path", "size"), groups=("group", "nunique"),
                                           pill_types=("pill_type", "nunique"), ndc9=("ndc9", "nunique"))
    photos_per_group = crops.groupby(["split", "group"]).size().groupby("split").agg(["min", "median", "max"])
    per_split = per_split.join(photos_per_group)
    summary = dict(seed=args.seed, holdout_fold=HOLDOUT_FOLD,
                   pill_types=int(types.shape[0]), appearance_groups=int(types.group.nunique()),
                   pill_types_merged_into_shared_groups=int((types.groupby("group").pill_type.transform("size") > 1).sum()),
                   holdout_ndc9_in_epillid=len(holdout_ndc9),
                   holdout_ndc9_found_in_c3pi=int(types.ndc9.isin(holdout_ndc9).sum()),
                   lookalike_pill_types_also_removed=int(len(lookalike_extra)),
                   splits={k: {c: int(v) for c, v in row.items()} for k, row in per_split.iterrows()},
                   leakage=report)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# C3PI split: leakage report",
        "",
        f"Seed {args.seed}; ePillID holdout = fold {HOLDOUT_FOLD}; made by `scripts/make_splits.py`.",
        "",
        "## Splits",
        "",
        "| split | " + " | ".join(per_split.columns) + " |",
        "|---" * (len(per_split.columns) + 1) + "|",
        *("| " + split + " | " + " | ".join(f"{v:,.0f}" for v in row) + " |" for split, row in per_split.iterrows()),
        "",
        "## Grouping",
        f"- {summary['pill_types']:,} pill types (NDC9 + part) form {summary['appearance_groups']:,} appearance groups; "
        f"{summary['pill_types_merged_into_shared_groups']} pill types share a group with a look-alike.",
        f"- ePillID holdout: {len(holdout_ndc9)} NDC9s, of which {summary['holdout_ndc9_found_in_c3pi']} pill types are in C3PI. "
        f"{len(lookalike_extra)} further pill types were removed because they look identical to a holdout pill"
        + (": " + "; ".join(sorted(set(lookalike_extra.name))) if len(lookalike_extra) else "."),
        "",
        "## Checks",
        f"- Appearance groups in more than one split: {report['groups_in_several_splits']}",
        f"- NDC9s in more than one split: {report['ndc9_in_several_splits']} "
        "(non-zero only if a multi-pill pack has parts that look like pills in another split; listed in groups.csv)",
        f"- Identical files (SHA-1) in more than one split: {report['identical_files_across_splits']}",
        f"- Copies of the same photo in different splits: {report['near_duplicate_pairs_across_splits']}. "
        f"{report['lookalike_candidates_across_splits']:,} look-alike pairs (32x32 thumbnail correlation >= {NEAR_DUP_CORR}) "
        f"were checked in the crop centre. A copy needs matching fine texture (>= {TEXTURE_DUP_CORR}) and nearly equal pixels "
        f"(mean difference <= {PIXEL_DUP_DIFF} of 255; JPEG-70 re-encoded, 1 px shifted copies have a median of 1.8). "
        f"The smallest pixel difference among the candidates was {report['min_pixel_diff_across_splits']}. "
        "All pairs are in near_duplicates_across_splits.csv, most similar texture first.",
    ]
    (out_dir / "leakage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
