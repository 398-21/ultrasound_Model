"""Smoke tests for the C3PI pill finder on synthetic photos (run: .venv\\Scripts\\python -m pytest tests)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epills.training.datasets.c3pi_crop import expected_length, find_pill, scale_table, square_crop  # noqa: E402


def synthetic_photo(centre=(1150, 820), axes=(90, 45), colour=(70, 140, 220), size=(1600, 2400), seed=0):
    """An oval 'pill' with a soft shadow on a grey surface with a lighting gradient and sensor noise (BGR)."""
    h, w = size
    rng = np.random.default_rng(seed)
    gradient = np.linspace(150, 190, w)[None, :] + np.linspace(-10, 10, h)[:, None]
    img = np.repeat(gradient[..., None], 3, axis=2)
    shadow = np.zeros((h, w), np.float32)
    cv2.ellipse(shadow, (centre[0] + 12, centre[1] + 12), axes, 0, 0, 360, 1.0, -1)
    img *= 1 - 0.35 * cv2.GaussianBlur(shadow, (0, 0), 8)[..., None]
    cv2.ellipse(img, centre, axes, 0, 0, 360, colour, -1)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_finds_the_pill_and_crops_it_whole():
    img = synthetic_photo()
    box, side, info = find_pill(img)
    assert box is not None
    x0, y0, x1, y1 = box
    # the box covers the pill (1060..1240 x 775..865), allowing for the shadow and the detection scale
    assert x0 < 1070 and x1 > 1230 and y0 < 785 and y1 > 855
    assert x1 - x0 < 260 and y1 - y0 < 160
    crop = square_crop(img, box, side, 384)
    assert crop.shape == (384, 384, 3)
    # the pill's colour (orange-ish, high red) is at the crop centre, and the crop border is background
    assert crop[192, 192, 2] > 180
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    assert np.abs(border.astype(int) - [160, 160, 160]).max(axis=1).mean() < 60


def test_size_prior_widens_the_window_when_only_part_is_found():
    img = synthetic_photo()
    # pretend the pill should be twice as long as what is visible: the window must grow to 2.2x that length
    box, side, info = find_pill(img, expected_len_px=400, spread=0.2, shape="OVAL")
    assert info["widened"] and side >= 2.2 * 400 - 1


def test_scale_table_and_expected_length():
    detections = pd.DataFrame({"camera": ["A"] * 6 + ["B"] * 2, "width": [4000] * 6 + [2048] * 2,
                               "score": [20] * 8, "long_px": [400, 420, 380, 410, 390, 400, 300, 300],
                               "size_mm": [10] * 8})
    table = scale_table(detections)
    assert list(table.index) == [("A", 4000)]          # camera B has too few detections
    length, spread = expected_length(table, "A", 4000, "12")
    assert abs(length - 40 * 12) < 1e-6 and 0.2 <= spread <= 0.5
    assert expected_length(table, "B", 2048, "12") == (None, 0.25)
    assert expected_length(table, "A", 4000, "") == (None, 0.25)
