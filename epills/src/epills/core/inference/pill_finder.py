"""
A stand-in for the Stage 1 detector: finds pills on a plain background (a mat, a table) in a full photo.

Works when the background is one roughly uniform colour and pills don't touch. Anything else needs the real
detector, whose boxes the API accepts instead. The background is the median colour of a frame along the photo's
edges; a pill is a blob that differs from it in Lab colour, after a morphological clean-up and hole filling.
"""
import cv2
import numpy as np

WORK = 768             # long side of the working image
MIN_DISTANCE = 14.0    # Lab distance below which a pixel is background
MIN_AREA = 0.0015      # smallest blob, as a share of the photo
MAX_AREA = 0.35        # largest blob (bigger: probably the mat's edge or a shadow, not a pill)


def find_pills(image, max_pills=20):
    """PIL photo -> boxes (x0, y0, x1, y1) in the photo's pixels, largest first."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    f = WORK / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    sh, sw = lab.shape[:2]
    b = max(2, int(0.04 * min(sh, sw)))
    frame = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(frame, axis=0), axis=2)
    otsu, _ = cv2.threshold(np.clip(d * 2, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (d > max(otsu / 2, MIN_DISTANCE)).astype(np.uint8)
    k = max(3, int(0.01 * min(sh, sw)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    flood = np.pad(mask, 1)
    cv2.floodFill(flood, np.zeros((sh + 4, sw + 4), np.uint8), (0, 0), 1)
    mask = mask | (1 - flood[1:-1, 1:-1])

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for x, y, bw, bh, area in stats[1:]:
        share = area / (sh * sw)
        touches = (x <= 1) + (y <= 1) + (x + bw >= sw - 1) + (y + bh >= sh - 1)
        if MIN_AREA <= share <= MAX_AREA and touches < 2:
            boxes.append((area, (x / f, y / f, (x + bw) / f, (y + bh) / f)))
    boxes.sort(key=lambda t: -t[0])
    return [tuple(round(v, 1) for v in box) for _, box in boxes[:max_pills]]
