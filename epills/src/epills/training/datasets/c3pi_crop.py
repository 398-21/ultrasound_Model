"""
Find the pill in a C3PI consumer photo and cut a square crop around it.

Each photo shows one pill (sometimes two of the same kind) on a surface, small in the frame. The finder works on a
copy shrunk to WORK px:
  1. Background: a robust (Tukey-weighted) quadratic fit of each Lab channel over the whole image, so lighting
     gradients and vignetting are modelled and the pill's pixels end up as outliers; and, when the pill's size is
     known, a local median over ~3x its length, which copes with steps such as a counter edge.
  2. Candidates: blobs of the colour difference from either background, over a sweep of thresholds.
  3. Choice: the blob that stands out most from its own surroundings (contrast-to-noise against a ring around it),
     preferring convex blobs near the centre and, when the expected pill length is known, of that length.
  4. Shadows: a white pill on white paper can differ less from the paper than its own shadow does. If the winner
     looks like a shadow (darker, colourless, no brighter pixels) and a pill-like blob sits beside it, both are taken.
  5. Refinement: GrabCut around the chosen blob, so a pale pill's full outline is found, not only its rim or shadow.
  6. Window: a square 1.4x the pill's long side, at least 1.6x the expected length when the outline came out short.
     When only part of the pill (or, for a white or grey pill, only its shadow) was found, the window grows to 2.2x
     the expected length around it; a pale pill merged with its shadow gets at least 1.8x.

The expected length comes from scale_table(): per camera and image width, the median pixels per mm of confident
detections, times the pill's size in mm from the metadata.
"""
import math

import cv2
import numpy as np

WORK = 640                 # long side of the detection image
ROUND_SHAPES = {"ROUND"}   # metadata shapes whose area is pi/4 L^2; elongated pills are at least ~0.4 L wide


def _poly_terms(x, y, degree):
    terms = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            terms.append(x ** (d - i) * y ** i)
    return np.stack(terms, 1)


def background_surface(lab, degree=2, iterations=5):
    """Robust low-order polynomial fit of each Lab channel; the pill's pixels end up as outliers."""
    h, w = lab.shape[:2]
    step = max(1, int(math.sqrt(h * w / 8000)))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    A = _poly_terms(xs.ravel() / w - 0.5, ys.ravel() / h - 0.5, degree)
    samples = lab[ys.ravel(), xs.ravel()].astype(np.float64)
    weights = np.ones(len(A))
    for _ in range(iterations):
        coef = np.linalg.lstsq(A * weights[:, None], samples * weights[:, None], rcond=None)[0]
        resid = np.linalg.norm(samples - A @ coef, axis=1)
        u = resid / (4.685 * (1.4826 * np.median(resid) + 1e-6))
        weights = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    yy, xx = np.mgrid[0:h, 0:w]
    return (_poly_terms(xx.ravel() / w - 0.5, yy.ravel() / h - 0.5, degree) @ coef).reshape(h, w, 3).astype(np.float32)


def _component_stats(labels, i, de, stats, diff):
    """Contrast-to-noise against a surrounding ring, solidity, position, shape and mean colour change of one blob."""
    h, w = de.shape
    x, y, bw, bh, area = stats[i]
    pad = max(4, int(0.35 * math.sqrt(area)))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    roi = labels[y0:y1, x0:x1] == i
    ring = cv2.dilate(roi.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    ring &= ~roi
    d = de[y0:y1, x0:x1]
    inside, around = d[roi], d[ring]
    if around.size < 10:
        return None
    contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    noise = 1.4826 * float(np.median(np.abs(around - np.median(around)))) + 2.0
    change = diff[y0:y1, x0:x1][roi]
    return dict(
        d_lightness=float(change[:, 0].mean()),
        d_chroma=float(np.hypot(change[:, 1], change[:, 2]).mean()),
        bright_frac=float((change[:, 0] > 4).mean()),
        cnr=(float(np.median(inside)) - float(np.median(around))) / noise,
        solidity=cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1),
        touches=bool(x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1),
        centre_dist=math.hypot((x + bw / 2 - w / 2) / w, (y + bh / 2 - h / 2) / h),
        area_frac=area / (h * w),
        aspect=max(bw, bh) / max(1, min(bw, bh)),
        box=(x, y, x + bw, y + bh),
    )


def _candidates(lab_u8, expected_small=None):
    """
    Blob candidates over a threshold sweep, against two background models: the global robust surface (smooth
    lighting, large pills) and, when the pill's size is known, a local median over ~3x its length (steps such as a
    counter edge, uneven surfaces).
    """
    lab = lab_u8.astype(np.float32)
    backgrounds = [background_surface(lab)]
    if expected_small:
        k = int(min(255, max(31, 3 * expected_small))) | 1
        backgrounds.append(cv2.medianBlur(lab_u8, k).astype(np.float32))
    h, w = lab.shape[:2]
    k = max(3, round(min(h, w) / 160)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    for background in backgrounds:
        diff = lab - background
        de = cv2.GaussianBlur(np.sqrt((diff ** 2).sum(axis=2)), (0, 0), 1.5)
        med = float(np.median(de))
        mad = 1.4826 * float(np.median(np.abs(de - med))) + 0.5
        for mult in (3, 4.5, 6, 8, 11, 15, 20, 28):
            thr = max(6.0, med + mult * mad)
            if thr >= de.max():
                break
            mask = cv2.morphologyEx((de > thr).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                if not 0.0002 <= stats[i, 4] / (h * w) <= 0.4:
                    continue
                info = _component_stats(labels, i, de, stats, diff)
                if info is not None:
                    out.append((info, labels == i))
    return out


def _shadow_like(info):
    """
    Darker than the background, with little colour change and almost no brighter pixels: a shadow on its own. (A
    pill together with its shadow has brighter pixels; a grey or black pill can look like this too.)
    """
    return info["d_lightness"] < -8 and info["d_chroma"] < 0.5 * abs(info["d_lightness"]) and info["bright_frac"] < 0.1


def _box_gap(a, b):
    """Distance between two boxes (0 when they overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _score(info, expected_small, spread):
    s = (max(info["cnr"], 0) * info["solidity"] ** 3 * math.exp(-(info["centre_dist"] / 0.3) ** 2)
         * (0.3 if info["touches"] else 1.0) * min(1.0, info["area_frac"] / 0.0015) * (0.5 if info["aspect"] > 5 else 1.0))
    if expected_small:
        x0, y0, x1, y1 = info["box"]
        ratio = max(x1 - x0, y1 - y0) / expected_small
        s *= math.exp(-0.5 * (math.log(max(ratio, 1e-3)) / (spread + 0.25)) ** 2)
    return s


def _grabcut_refine(img, comp, expected_small):
    """GrabCut with the blob as (probable) foreground and definite background outside ~1.6x the pill's size."""
    h, w = comp.shape
    ys, xs = np.nonzero(comp)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = 0.8 * max(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1, expected_small or 0) + 4
    x0, y0 = int(max(0, cx - 2 * half)), int(max(0, cy - 2 * half))
    x1, y1 = int(min(w, cx + 2 * half)), int(min(h, cy + 2 * half))
    mask = np.full((y1 - y0, x1 - x0), cv2.GC_BGD, np.uint8)
    mask[int(max(0, cy - half - y0)):int(min(y1 - y0, cy + half - y0)),
         int(max(0, cx - half - x0)):int(min(x1 - x0, cx + half - x0))] = cv2.GC_PR_BGD
    local = comp[y0:y1, x0:x1]
    mask[local] = cv2.GC_PR_FGD
    mask[cv2.erode(local.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(np.ascontiguousarray(img[y0:y1, x0:x1]), mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 4,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return comp
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    _, labels = cv2.connectedComponents(fg, connectivity=8)
    keep = np.unique(labels[local & (fg > 0)])
    keep = keep[keep > 0]
    if keep.size == 0:
        return comp
    out = np.zeros_like(comp)
    out[y0:y1, x0:x1] = np.isin(labels, keep)
    return out | comp


PALE_COLOURS = {"WHITE", "GRAY"}  # pills that can differ less from a pale surface than their own shadow does


def find_pill(img, expected_len_px=None, spread=0.25, shape="", colour=""):
    """
    Locate the pill in a BGR photo. Returns (box, side, info): box = (x0, y0, x1, y1) of the found pill region and
    side = the square crop window, both in the photo's pixels, or (None, None, info) when nothing pill-like is found.
    expected_len_px is the pill's expected long side in the photo (see expected_length); spread its relative error;
    shape and colour are the metadata values (colour may hold two, joined by ";").
    """
    pale = bool(set(colour.split(";")) & PALE_COLOURS)
    H, W = img.shape[:2]
    s = WORK / max(H, W)
    small = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(cv2.medianBlur(small, 5), (0, 0), 1.0)
    expected_small = expected_len_px * s if expected_len_px else None
    cands = _candidates(cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB), expected_small)
    if not cands:
        return None, None, dict(score=0.0, n_candidates=0)
    scored = sorted(((_score(info, expected_small, spread), info, comp) for info, comp in cands), key=lambda t: -t[0])
    best_score, info, comp = scored[0]
    # A white pill on white paper can differ less from the paper than its own shadow does. If the winner looks like a
    # shadow and a pill-like blob sits right next to it, take both: the crop then centres on pill + shadow.
    shadow_merged = False
    if _shadow_like(info):
        reach = expected_small or max(info["box"][2] - info["box"][0], info["box"][3] - info["box"][1])
        partners = [c2 for sc, i2, c2 in scored[1:] if sc >= 0.3 * best_score and not _shadow_like(i2)
                    and _box_gap(i2["box"], info["box"]) <= 0.6 * reach
                    and max(i2["box"][2] - i2["box"][0], i2["box"][3] - i2["box"][1]) <= 2 * reach]
        if partners:
            comp = comp | partners[0]
            shadow_merged = True
    refined = _grabcut_refine(smooth, comp, expected_small)
    ys, xs = np.nonzero(refined)
    if expected_small and max(xs.max() - xs.min(), ys.max() - ys.min()) > (2.6 if shadow_merged else 2.2) * expected_small:
        refined = comp  # the refinement ran away into the background
        ys, xs = np.nonzero(refined)
    box = (xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s)
    long_px = max(box[2] - box[0], box[3] - box[1])
    side = 1.4 * long_px
    size_ratio = area_ratio = float("nan")
    partial = False
    if expected_len_px:
        size_ratio = long_px / expected_len_px
        expected_area = (0.785 if shape in ROUND_SHAPES else 0.3) * expected_small ** 2
        area_ratio = float(refined.sum()) / expected_area
        if size_ratio < 0.85:  # shorter than expected: leave room for a faint edge the outline missed
            side = max(side, 1.6 * expected_len_px)
        # only part of the pill was found, or (pale pills) only its shadow, with no pill beside it
        partial = size_ratio < 0.6 or area_ratio < 0.45 or (pale and _shadow_like(info) and not shadow_merged)
        if partial:
            side = max(side, 2.2 * expected_len_px)
        elif pale and shadow_merged:  # the pale pill's far edge can be fainter than the outline found
            side = max(side, 1.8 * expected_len_px)
    # a second pill of the same kind: a strong, separate candidate of similar size
    second = any(sc > 0.3 * best_score and not (c2 & refined).any()
                 and 0.6 < i2["area_frac"] / max(info["area_frac"], 1e-9) < 1.7 for sc, i2, c2 in scored[1:])
    out = {k: v for k, v in info.items() if k != "box"}
    out.update(score=best_score, size_ratio=size_ratio, area_ratio=area_ratio, n_candidates=len(cands),
               widened=partial, second_pill=second, shadow_merged=shadow_merged,
               runner_up=scored[1][0] / best_score if len(scored) > 1 and best_score > 0 else 0.0)
    return box, side, out


def square_crop(img, box, side, size=384):
    """The square window of the given side centred on box, resized to size x size (edges replicated if outside)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    f = size / side
    M = np.float32([[f, 0, size / 2 - cx * f], [0, f, size / 2 - cy * f]])
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def scale_table(detections, min_score=8.0, min_count=5):
    """
    Median pixels per mm and its relative spread (IQR / median) per (camera, width), from detections without a size
    prior. detections: a DataFrame with camera, width, score, long_px and size_mm.
    """
    good = detections[(detections.score > min_score) & (detections.size_mm > 0) & detections.long_px.notna()]
    px_per_mm = (good.long_px / good.size_mm).groupby([good.camera, good.width])
    table = px_per_mm.agg(n="size", px_per_mm="median")
    table["spread"] = (px_per_mm.quantile(0.75) - px_per_mm.quantile(0.25)) / table.px_per_mm
    return table[table.n >= min_count]


def expected_length(table, camera, width, size_mm):
    """(expected long side in px, relative spread) for a photo, or (None, 0.25) when unknown."""
    try:
        size_mm = float(size_mm)
        row = table.loc[(camera, int(width))]
    except (KeyError, TypeError, ValueError):
        return None, 0.25
    if not size_mm > 0:
        return None, 0.25
    return float(row.px_per_mm) * size_mm, float(min(0.5, max(0.2, row.spread)))
