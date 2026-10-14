"""
A stand-in for the Stage 1 detector: finds pills on a plain background (a mat, a table) in a full photo.

Works when the background is plain (one colour, possibly unevenly lit) and pills lie apart. Anything else needs the
real detector, whose boxes the API accepts instead.
1. Background: each pixel is compared with the mat's colour *at that place*: a smooth surface (degree-3 polynomial
   per Lab channel) fitted robustly, so vignetting and lighting gradients don't turn corners into "pills". Pills are
   outliers to that fit and don't move it.
2. A pill is a blob that differs from it (Otsu threshold, at least MIN_DISTANCE), after a morphological clean-up and
   hole filling, with a pill-like shape: compact (solidity), not too elongated, not a sliver along the photo's edge.
3. Pieces of one pill (a light pill on a light background splits along its score line or imprint) are merged when
   they nearly touch; small crumbs next to a pill are dropped.
Every box carries quality flags; any flag, or a dropped crumb, sets the photo flag `count_uncertain`, so a pill count
from this finder is never trusted blindly (blueprint job B1).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

WORK = 768             # long side of the working image
MIN_DISTANCE = 12.0    # Lab distance from the local background below which a pixel is background
MIN_AREA = 0.0008      # smallest blob, as a share of the photo
MAX_AREA = 0.25        # largest blob
MIN_SOLIDITY = 0.80    # blob area / its convex hull: pills are convex, lighting patches and shadows aren't
MAX_ASPECT = 4.5       # long / short side of the blob's rotated rectangle
MERGE_GAP = 0.12       # pieces closer than this x the larger piece's long side are one pill
CRUMB_RATIO = 0.2      # a blob smaller than this x a neighbouring pill's area ...
CRUMB_GAP = 0.6        # ... within this x its long side is a crumb (shadow edge, chip), dropped
TEXTURE_RATIO = 0.08   # anywhere, a blob below this x the largest pill's area is a speck of texture, dropped
LOW_FACTOR = 0.5       # hysteresis: the faint threshold, as a share of the clear one


@dataclass
class Detection:
    boxes: list = field(default_factory=list)       # (x0, y0, x1, y1) in the photo's pixels, largest first
    box_flags: list = field(default_factory=list)   # per box: "touches_edge", "merged", "irregular", "small"
    flags: list = field(default_factory=list)       # per photo: "no_pills", "count_uncertain", "too_many"
    dropped: int = 0                                # crumbs and non-pill blobs left out

    @property
    def pill_count(self):
        return len(self.boxes)


def _gap(a, b):
    """Distance between two boxes (0 when they overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def _long(b):
    return max(b[2] - b[0], b[3] - b[1])


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _terms(x, y):
    """Polynomial terms up to degree 3 in x and y (in -1..1): enough for vignetting and lighting gradients."""
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3], axis=-1)


def _background(lab):
    """The mat's colour at every pixel: a smooth surface per Lab channel, fitted robustly (pills are outliers)."""
    sh, sw = lab.shape[:2]
    tiny = cv2.resize(lab, (64, max(8, round(64 * sh / sw))), interpolation=cv2.INTER_AREA).astype(np.float32)
    th, tw = tiny.shape[:2]
    gy, gx = np.mgrid[0:th, 0:tw]
    A = _terms(gx / (tw - 1) * 2 - 1, gy / (th - 1) * 2 - 1).reshape(-1, 10)
    Y = tiny.reshape(-1, 3)
    keep = np.ones(len(Y), bool)
    for _ in range(4):  # refit without the pixels that don't look like mat
        coef, *_ = np.linalg.lstsq(A[keep], Y[keep], rcond=None)
        resid = np.linalg.norm(Y - A @ coef, axis=1)
        keep = resid <= max(2.5 * np.median(resid[keep]), 3.0)
    gy, gx = np.mgrid[0:sh, 0:sw]
    return (_terms(gx / (sw - 1) * 2 - 1, gy / (sh - 1) * 2 - 1) @ coef).astype(np.float32)


def _mask(lab):
    """Pixels that differ from the mat's colour at that place."""
    sh, sw = lab.shape[:2]
    d = np.linalg.norm(lab.astype(np.float32) - _background(lab), axis=2)
    otsu, _ = cv2.threshold(np.clip(d * 2, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(otsu / 2, MIN_DISTANCE)
    k = max(3, int(0.01 * min(sh, sw)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    strong = cv2.morphologyEx((d > high).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # hysteresis: a pill's faint parts (the white half of a capsule) join the clearly different part they touch
    weak = cv2.morphologyEx((d > max(high * LOW_FACTOR, MIN_DISTANCE * 0.75)).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[strong > 0])] = True
    keep[0] = False
    mask = cv2.morphologyEx(keep[labels].astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    flood = np.pad(mask, 1)
    cv2.floodFill(flood, np.zeros((sh + 4, sw + 4), np.uint8), (0, 0), 1)
    return mask | (1 - flood[1:-1, 1:-1])


def detect_pills(image, max_pills=20):
    """PIL photo -> Detection: pill boxes in the photo's pixels with quality flags."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    f = WORK / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    mask = _mask(cv2.cvtColor(small, cv2.COLOR_RGB2LAB))
    sh, sw = mask.shape

    blobs, dropped = [], 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        share = area / (sh * sw)
        if not MIN_AREA <= share <= MAX_AREA:
            dropped += share > MAX_AREA
            continue
        contours, _ = cv2.findContours((labels[y:y + bh, x:x + bw] == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull if hull > 0 else 0.0
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        edges = (x <= 1) + (y <= 1) + (x + bw >= sw - 1) + (y + bh >= sh - 1)
        if edges >= 2 or aspect > MAX_ASPECT or (edges and solidity < 0.9) or solidity < MIN_SOLIDITY - 0.15:
            dropped += 1  # lighting patches, mat edges, slivers: not pills
            continue
        flags = set()
        if edges:
            flags.add("touches_edge")
        if solidity < MIN_SOLIDITY:
            flags.add("irregular")
        blobs.append({"box": [float(x), float(y), float(x + bw), float(y + bh)], "area": float(area), "flags": flags})

    merged = True
    while merged:  # pieces of one pill
        merged = False
        for a in range(len(blobs)):
            for b in range(a + 1, len(blobs)):
                A, B = blobs[a], blobs[b]
                if _gap(A["box"], B["box"]) <= MERGE_GAP * max(_long(A["box"]), _long(B["box"])):
                    A["box"] = [min(A["box"][0], B["box"][0]), min(A["box"][1], B["box"][1]),
                                max(A["box"][2], B["box"][2]), max(A["box"][3], B["box"][3])]
                    A["area"] += B["area"]
                    A["flags"] |= B["flags"] | {"merged"}
                    del blobs[b]
                    merged = True
                    break
            if merged:
                break

    if any("touches_edge" not in b["flags"] for b in blobs):  # whole pills found: blobs cut by the edge are mat
        dropped += sum("touches_edge" in b["flags"] for b in blobs)
        blobs = [b for b in blobs if "touches_edge" not in b["flags"]]
    blobs.sort(key=lambda t: -t["area"])
    kept = []
    for blob in blobs:  # crumbs next to a bigger pill, specks of texture anywhere
        if kept and (blob["area"] < TEXTURE_RATIO * kept[0]["area"] or any(
                blob["area"] < CRUMB_RATIO * k["area"] and _gap(blob["box"], k["box"]) < CRUMB_GAP * _long(k["box"])
                for k in kept)):
            dropped += 1
            continue
        kept.append(blob)
    if len(kept) >= 2:
        median = float(np.median([k["area"] for k in kept]))
        for k in kept:
            if k["area"] < 0.15 * median:
                k["flags"].add("small")

    det = Detection(dropped=dropped)
    for k in kept[:max_pills]:
        det.boxes.append(tuple(round(v / f, 1) for v in k["box"]))
        det.box_flags.append(sorted(k["flags"]))
    if not det.boxes:
        det.flags.append("no_pills")
    if len(kept) > max_pills:
        det.flags.append("too_many")
    if any(det.box_flags) or len(kept) > max_pills or any(
            "merged" in fl for fl in det.box_flags):
        det.flags.append("count_uncertain")
    return det


def find_pills(image, max_pills=20):
    """PIL photo -> boxes (x0, y0, x1, y1) in the photo's pixels, largest first (see detect_pills for flags)."""
    return detect_pills(image, max_pills).boxes
