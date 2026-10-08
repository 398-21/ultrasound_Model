"""
Crops of C3PI's studio reference photos (decision D7), framed like the everyday-photo crops (c3pi_crop.py): a square
window 1.4x the pill's long side, resized to 384 px.

  MC_CHALLENGE_V1.0   one side of one pill on plain grey, 2400 x 1600 (the photos ePillID's reference gallery
                      is made from)
  MC_API_RXNAV_V1.3   RxImage: both sides of the pill in one 1024 x 896 photo, side by side or stacked, with mm rulers
                      along the left and bottom edges and an NLM information bar at the bottom -> two crops

The background is plain, so a pill is anything far enough in Lab colour from the background's median. A
morphological opening removes the thin rulers, digits and watermark; holes (imprints, score lines) are filled.
In RxImage photos, the information bar is cut off, and each crop has the other side of the pill, the edge-on
profile view some photos show between them, and the ruler marks painted over with the background colour, so every
crop holds one side of one pill.
"""
import cv2
import numpy as np

from epills.training.datasets.c3pi_crop import square_crop

WINDOW = 1.4
CROP_SIZE = 384
MIN_DISTANCE = 12.0   # Lab distance below which a pixel is background, whatever Otsu says


def background_colour(img, lab, border=0.03):
    """
    The plain background: the most common lightness among low-colour pixels in a frame along all four edges
    (pills can fill most of an RxImage photo and touch its edges; rulers are thin). Returns (Lab, BGR) medians.
    """
    h, w = lab.shape[:2]
    b = max(2, int(border * min(h, w)))
    edge = np.zeros((h, w), bool)
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = True, True, True, True
    grey = edge & (np.abs(lab[..., 1] - 128) < 6) & (np.abs(lab[..., 2] - 128) < 6)
    if grey.sum() < 100:
        grey = edge
    mode = np.bincount(lab[..., 0][grey].astype(int), minlength=256).argmax()
    sel = grey & (np.abs(lab[..., 0] - mode) <= 6)
    return np.median(lab[sel], axis=0), np.median(img[sel], axis=0)


def foreground(img):
    """
    (mask, thin, background BGR): mask = pills (holes filled); thin = the thin structures the opening removed
    (rulers, digits, watermark).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bg_lab, bg_bgr = background_colour(img, lab)
    d = np.linalg.norm(lab - bg_lab, axis=2)
    otsu, _ = cv2.threshold(np.clip(d * 2, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = (d > max(otsu / 2, MIN_DISTANCE)).astype(np.uint8)
    k = max(5, int(0.02 * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    # fill holes: everything not reachable through background from a frame around the image
    flood = np.pad(mask, 1)
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    mask = mask | (1 - flood[1:-1, 1:-1])
    thin = raw.astype(bool) & ~cv2.dilate(mask, kernel).astype(bool)
    return mask, thin, bg_bgr


def components(mask, min_share=0.25):
    """Boxes (x0, y0, x1, y1) of the connected components at least min_share the size of the largest, largest first."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return []
    stats = stats[1:]
    stats = stats[stats[:, cv2.CC_STAT_AREA] >= min_share * stats[:, cv2.CC_STAT_AREA].max()]
    stats = stats[np.argsort(-stats[:, cv2.CC_STAT_AREA])]
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats]


def split_touching(mask, max_iterations=40):
    """Erode a mask until it falls apart into two comparable pills; boxes grown back by the erosion."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = mask.copy()
    for i in range(1, max_iterations + 1):
        m = cv2.erode(m, kernel)
        boxes = components(m, min_share=0.4)
        if len(boxes) >= 2:
            g = 2 * i
            return [(x0 - g, y0 - g, x1 + g, y1 + g) for x0, y0, x1, y1 in boxes[:2]]
        if not m.any():
            break
    return None


def info_bar_top(img):
    """First row of RxImage's light-blue information bar, searched in the bottom fifth (row 770 of 896)."""
    h = img.shape[0]
    low = img[int(0.8 * h):].astype(np.int16)
    blue = ((low[..., 0] - low[..., 2]) > 25) & (low[..., 0] > 120)
    rows = np.flatnonzero(blue.mean(axis=1) > 0.5)
    return int(0.8 * h) + int(rows[0]) if len(rows) else int(0.86 * h)


def crop_box(img, box, fill=None):
    """The C3PI framing; outside the photo: the background colour if given (no smeared edges), else replicated."""
    if fill is None:
        return square_crop(img, box, WINDOW * max(box[2] - box[0], box[3] - box[1]), CROP_SIZE)
    side = WINDOW * max(box[2] - box[0], box[3] - box[1])
    f = CROP_SIZE / side
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    M = np.float32([[f, 0, CROP_SIZE / 2 - cx * f], [0, f, CROP_SIZE / 2 - cy * f]])
    return cv2.warpAffine(img, M, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in fill))


def crop_challenge(img, work=800):
    """MC_CHALLENGE photo -> ([crop], note)."""
    f = work / max(img.shape[:2])
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    mask, _, bg = foreground(small)
    boxes = components(mask)
    if not boxes:
        return [], "no pill found"
    box = tuple(v / f for v in boxes[0])
    return [crop_box(img, box, bg)], "ok" if len(boxes) == 1 else f"{len(boxes)} blobs, largest taken"


def crop_rxnav(img):
    """RxImage photo -> ([one crop per pill side], note); normally two crops."""
    top = info_bar_top(img)
    work = img[:top].copy()
    mask, thin, bg = foreground(work)
    work[thin] = bg  # rulers, digits and watermark
    band = int(0.05 * top)  # the mm ruler right above the information bar
    work[top - band:][mask[top - band:] == 0] = bg
    boxes = components(mask)
    note = "ok"
    if len(boxes) == 1:
        split = split_touching(mask)
        if split is None:
            return [crop_box(work, boxes[0], bg)], "one blob (sides touching?), taken whole"
        return [crop_box(work, b, bg) for b in split], "touching sides split"
    if not boxes:
        return [], "no pill found"
    if len(boxes) > 2:
        note = f"{len(boxes)} blobs, two largest taken"
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    k = max(9, int(0.03 * min(work.shape[:2])) | 1)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    crops = []
    for box in boxes[:2]:
        sub = labels[box[1]:box[3], box[0]:box[2]]
        own = np.bincount(sub[sub > 0]).argmax()  # the component this box was made from
        others = cv2.dilate(((labels > 0) & (labels != own)).astype(np.uint8), grow).astype(bool)
        clean = work.copy()
        clean[others & (labels != own)] = bg
        crops.append(crop_box(clean, box, bg))
    return crops, note
