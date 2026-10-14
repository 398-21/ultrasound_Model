"""The stand-in pill finder (blueprint job B1): one pill -> one box, two pills -> two boxes, uneven light ignored."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.core.inference.pill_finder import detect_pills, find_pills  # noqa: E402


def vignetted(w=1200, h=900, colour=(150, 150, 155)):
    """A plain grey mat lit unevenly: bright centre, dark corners (what turned corners into 'pills' before)."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / w, (yy - h / 2) / h)
    shade = (1.15 - 1.6 * r * r)[..., None]  # smooth fall-off to ~55% in the corners, as lenses do
    rng = np.random.default_rng(0)
    img = np.clip(np.array(colour, np.float32) * shade + rng.normal(0, 2, (h, w, 1)), 0, 255)
    return Image.fromarray(img.astype(np.uint8))


def capsule(draw, x, y, length=180, width=70, colours=((200, 40, 40), (235, 235, 230))):
    """A two-tone capsule: one half red, the other near-white (only the red half used to be found)."""
    half = length // 2
    draw.rounded_rectangle((x, y, x + half + 20, y + width), radius=width // 2, fill=colours[0])
    draw.rounded_rectangle((x + half - 20, y, x + length, y + width), radius=width // 2, fill=colours[1])
    return (x, y, x + length, y + width)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union


def test_one_two_tone_capsule_on_an_unevenly_lit_mat_is_one_box():
    img = vignetted()
    truth = capsule(ImageDraw.Draw(img), 500, 400)
    det = detect_pills(img)
    assert len(det.boxes) == 1 and iou(det.boxes[0], truth) > 0.6
    assert find_pills(img) == det.boxes


def test_two_pills_are_two_boxes_and_an_empty_mat_is_none():
    img = vignetted()
    d = ImageDraw.Draw(img)
    truths = [capsule(d, 200, 300), (800, 380, 900, 480)]
    d.ellipse(truths[1], fill=(240, 200, 60))
    det = detect_pills(img)
    assert len(det.boxes) == 2 and all(max(iou(b, t) for b in det.boxes) > 0.6 for t in truths)
    empty = detect_pills(vignetted())
    assert empty.boxes == [] and "no_pills" in empty.flags


def test_a_pill_cut_by_the_photo_edge_is_flagged_and_the_count_is_uncertain():
    img = vignetted()
    ImageDraw.Draw(img).ellipse((-40, 400, 60, 500), fill=(240, 200, 60))  # half outside the photo
    det = detect_pills(img)
    assert det.boxes and "touches_edge" in det.box_flags[0] and "count_uncertain" in det.flags
