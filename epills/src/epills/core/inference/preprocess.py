"""
The crop contract and the model's input transform, shared by training (src/epills/training) and the API.

Crop contract: a pill is cropped as a square window WINDOW x the long side of its *tight* box (the pill only, as a
detector gives it), centred on the box, from the full-resolution photo. Outside the photo the edge pixels are
repeated. A detector (Stage 1) sends boxes; the API crops with crop_window.

Each model's window is chosen on validation by scripts/calibrate_stage2.py --windows and travels with the model
(Embedder.window); run 3: 1.4 (validation: 1.3-1.7 all within 0.6 points). WINDOW = 1.7 is only the default for a
model without one: it reproduces the training crops, where the pill alone fills a median 59% of the crop side
(measured 2026-10-09 on 581 training crops; 10-90%: 41-69%; the C3PI cropper's 1.4 applies to its own boxes,
which include the shadow).
"""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

IMAGE_SIZE = 224     # model input
CROP_SIZE = 384      # stored crops
WINDOW = 1.7         # crop window / long side of a tight pill box


def eval_transform(size=IMAGE_SIZE):
    """PIL crop -> float tensor (3, size, size) in [0, 1]; what the model sees at evaluation and in the API."""
    return T.Compose([T.ToImage(), T.Resize(size, antialias=True), T.CenterCrop(size),
                      T.ToDtype(torch.float32, scale=True)])


def crop_window(image, box, window=WINDOW, out_size=CROP_SIZE):
    """
    The contract crop: a PIL image of the square window `window` x the long side of box = (x0, y0, x1, y1), in
    the photo's pixels, resized to out_size. Parts outside the photo repeat the edge pixels.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty box {box}")
    side = window * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = max(1, int(round(side)))
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    pad = ((max(0, -top), max(0, top + s - h)), (max(0, -left), max(0, left + s - w)), (0, 0))
    if any(p for pair in pad[:2] for p in pair):
        arr = np.pad(arr, pad, mode="edge")
        left, top = left + pad[1][0], top + pad[0][0]
    crop = Image.fromarray(arr[top:top + s, left:left + s])
    return crop.resize((out_size, out_size), Image.Resampling.BICUBIC)
