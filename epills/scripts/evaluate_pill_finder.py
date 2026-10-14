"""
The stand-in pill finder (core/inference/pill_finder.py) against known pill positions: blueprint job B1 ("1 pill -> 1
box, 2 pills -> 2 boxes").

Photos: C3PI consumer photos, one pill each (those the cropper saw a second pill in are left out), with the pill's box from the reviewed training cropper
(data/interim/c3pi/crops/crops.csv). Two- and three-pill photos are made from them by copying the pill's patch (with
its own background around it) to free places in the same photo, so every true box is known.
A pill counts as found when a returned box overlaps its true box with IoU >= 0.5.
The residual that matters for dose logging (blueprint B24, test case D01) is over-counting while the API says the
count is reliable (count_reliable: boxes found and no count_uncertain flag): an extra pill the app would record.
Output: outputs/eval/pill_finder.json (cited by /v1/model-info).

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_pill_finder.py [--photos 1000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.inference.pill_finder import detect_pills  # noqa: E402

RAW = EPILLS_ROOT / "data/raw/c3pi/images"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load(row):
    """The photo as the cropper saw it (its recorded width and height), and the true box."""
    img = Image.open(RAW / row.path).convert("RGB")
    if img.size != (int(row.width), int(row.height)):
        img = img.transpose(Image.Transpose.ROTATE_90) if img.size[::-1] == (int(row.width), int(row.height)) else img
    return img, tuple(float(v) for v in (row.x0, row.y0, row.x1, row.y1))


def feather(size):
    """A paste mask that fades the patch's own background into the photo's (no rectangular seam)."""
    w, h = size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse((w * 0.08, h * 0.08, w * 0.92, h * 0.92), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(min(w, h) * 0.06))


def with_copies(img, box, n, rng):
    """The photo with n - 1 copies of the pill's patch pasted where they don't overlap anything; None if no room."""
    x0, y0, x1, y1 = box
    pad = 0.25 * max(x1 - x0, y1 - y0)
    px0, py0, px1, py1 = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(img.width, x1 + pad)),
                          int(min(img.height, y1 + pad)))
    patch = img.crop((px0, py0, px1, py1))
    if patch.width * 2 > img.width or patch.height * 2 > img.height:
        return None, None  # a close-up: no room for another pill
    out, boxes, taken = img.copy(), [box], [(px0, py0, px1, py1)]
    for _ in range(n - 1):
        for _try in range(50):
            nx = int(rng.integers(0, img.width - patch.width))
            ny = int(rng.integers(0, img.height - patch.height))
            cand = (nx, ny, nx + patch.width, ny + patch.height)
            if all(iou(cand, t) == 0 and (cand[2] < t[0] - pad or cand[0] > t[2] + pad or
                                          cand[3] < t[1] - pad or cand[1] > t[3] + pad) for t in taken):
                out.paste(patch, (nx, ny), feather(patch.size))
                taken.append(cand)
                boxes.append((nx + x0 - px0, ny + y0 - py0, nx + x1 - px0, ny + y1 - py0))
                break
        else:
            return None, None
    return out, boxes


def score(found, truth):
    hit = [any(iou(f, t) >= 0.5 for f in found) for t in truth]
    return {"exact_count": len(found) == len(truth), "all_found": all(hit), "boxes": len(found),
            "right": len(found) == len(truth) and all(hit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=int, default=1000)
    args = parser.parse_args()
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    crops = crops[(crops.found == "1") & (crops.image_type == "C3PI_Test") & (crops.second_pill != "True")]
    sample = crops.drop_duplicates("path").sample(args.photos, random_state=1)
    rng = np.random.default_rng(0)
    rows = []
    for r in sample.itertuples():
        img, box = load(r)
        for n in (1, 2, 3):
            photo, truth = (img, [box]) if n == 1 else with_copies(img, box, n, rng)
            if photo is None:
                continue
            det = detect_pills(photo)
            rows.append({"pills": n, **score(det.boxes, truth), "flagged": bool(det.flags),
                         "reliable": bool(det.boxes) and "count_uncertain" not in det.flags})
    df = pd.DataFrame(rows)
    table = df.groupby("pills").agg(photos=("right", "size"), right=("right", "mean"), exact_count=("exact_count", "mean"),
                                    all_found=("all_found", "mean"), mean_boxes=("boxes", "mean"),
                                    flagged=("flagged", "mean"))
    print("right = exactly as many boxes as pills, each pill's true box matched (IoU >= 0.5)")
    print(table.round(3).to_string())
    wrong_unflagged = df[~df.right & ~df.flagged]
    print(f"wrong and not flagged: {len(wrong_unflagged)} of {len(df)} ({len(wrong_unflagged) / len(df):.1%})")
    out = {"protocol": __doc__.strip().split("\n\nUsage")[0], "photos_sampled": args.photos, "by_pills": {}}
    for n, g in df.groupby("pills"):
        over = g.boxes > n
        out["by_pills"][str(n)] = {
            "photos": int(len(g)), "right": round(float(g.right.mean()), 4),
            "exact_count": round(float(g.exact_count.mean()), 4), "no_boxes": round(float((g.boxes == 0).mean()), 4),
            "under_count": round(float((g.boxes < n).mean()), 4), "over_count": round(float(over.mean()), 4),
            "over_count_while_reliable": round(float((over & g.reliable).mean()), 4),
            "wrong_unflagged": round(float((~g.right & ~g.flagged).mean()), 4)}
    path = EPILLS_ROOT / "outputs/eval/pill_finder.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["by_pills"], indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
