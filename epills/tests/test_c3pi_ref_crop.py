"""The reference-photo cropper on synthetic studio photos (no data needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.training.datasets.c3pi_ref_crop import CROP_SIZE, crop_challenge, crop_rxnav  # noqa: E402

GREY = (118, 118, 118)


def rxnav_like(touching=False):
    """Two pill sides on grey, mm rulers on the left and bottom, a light-blue information bar."""
    img = np.full((896, 1024, 3), GREY, np.uint8)
    gap = 0 if touching else 60
    cv2.ellipse(img, (300, 380), (220, 140), 0, 0, 360, (60, 90, 200), -1)          # left side (BGR: reddish)
    cv2.ellipse(img, (300 + 440 + gap, 380), (220, 140), 0, 0, 360, (60, 90, 200), -1)
    cv2.putText(img, "E12", (230, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (40, 40, 40), 4)  # imprint: a hole to fill
    for x in range(30, 1000, 25):                                                     # bottom ruler ticks
        cv2.line(img, (x, 740), (x, 760), (230, 230, 230), 2)
    for y in range(20, 740, 25):                                                      # left ruler ticks
        cv2.line(img, (5, y), (20, y), (230, 230, 230), 2)
    img[770:] = (220, 190, 150)                                                       # the bar (BGR: light blue)
    return img


def test_rxnav_gives_two_clean_crops():
    crops, note = crop_rxnav(rxnav_like())
    assert note == "ok" and len(crops) == 2
    for c in crops:
        assert c.shape == (CROP_SIZE, CROP_SIZE, 3)
        # the pill is in the middle, the corners are background grey (no bar, no ruler, no other side)
        assert np.abs(c[CROP_SIZE // 2, CROP_SIZE // 2].astype(int) - (60, 90, 200)).max() < 30
        for y, x in ((5, 5), (5, -5), (-5, 5), (-5, -5)):
            assert np.abs(c[y, x].astype(int) - GREY).max() < 12


def test_touching_sides_are_split():
    crops, note = crop_rxnav(rxnav_like(touching=True))
    assert len(crops) == 2 and note == "touching sides split"


def test_challenge_single_pill_framed_like_everyday_crops():
    img = np.full((1600, 2400, 3), GREY, np.uint8)
    cv2.circle(img, (1200, 800), 150, (200, 200, 230), -1)
    (crop,), note = crop_challenge(img)
    assert note == "ok"
    # pill diameter 300 px in a window 1.4 x 300: it spans ~1/1.4 of the crop
    row = np.abs(crop[CROP_SIZE // 2].astype(int) - GREY).max(axis=1) > 30
    assert abs(row.sum() / CROP_SIZE - 1 / 1.4) < 0.03
