"""The crop contract, the stand-in pill finder and the pill catalogue (identify, search), on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.core.catalogue.store import Catalogue, imprint_matches, write_catalogue  # noqa: E402
from epills.core.inference.pill_finder import find_pills  # noqa: E402
from epills.core.inference.preprocess import WINDOW, crop_window  # noqa: E402

MAT = (205, 190, 160)


def mat_photo(pills, size=(1200, 900)):
    """A photo of pills (centre, radii, colour) on a plain beige mat."""
    img = Image.new("RGB", size, MAT)
    d = ImageDraw.Draw(img)
    for (cx, cy), (rx, ry), colour in pills:
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=colour)
    return img


def test_crop_window_frames_the_pill_and_repeats_edges():
    img = mat_photo([((600, 450), (100, 60), (200, 30, 30))])
    crop = crop_window(img, (500, 390, 700, 510), out_size=280)
    row = np.asarray(crop)[140]
    red = (row[:, 0] > 150) & (row[:, 1] < 100)
    assert abs(red.sum() / 280 - 1 / WINDOW) < 0.02  # the pill's long side is 1/1.4 of the window
    # a box at the photo's corner: the window runs outside, filled with edge pixels (the mat), never black
    corner = np.asarray(crop_window(img, (0, 0, 80, 80), out_size=100))
    assert corner.min() > 100


def test_pill_finder_on_a_plain_mat():
    pills = [((300, 300), (90, 50), (240, 240, 240)), ((800, 400), (60, 60), (200, 120, 40)),
             ((550, 700), (110, 45), (40, 90, 200))]
    boxes = find_pills(mat_photo(pills))
    assert len(boxes) == 3
    for (cx, cy), (rx, ry), _ in pills:
        assert any(abs((b[0] + b[2]) / 2 - cx) < 10 and abs((b[1] + b[3]) / 2 - cy) < 10 and
                   abs((b[2] - b[0]) - 2 * rx) < 15 for b in boxes)


def test_imprint_matching():
    assert imprint_matches("m 547", "M;547")
    assert imprint_matches("M547", "M;547")
    assert imprint_matches("547", "M;547")
    assert imprint_matches("E12", "E;12;E;12")
    assert not imprint_matches("M 548", "M;547")
    assert not imprint_matches("", "M;547")


def unit(*v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


PRODUCTS = [
    {"pill_id": "A_1", "name": "GLUCOPHAGE 500 MG", "generic_name": "METFORMIN HYDROCHLORIDE", "shape": "ROUND",
     "colour": "WHITE", "imprint": "M;547", "rxcui": "861007", "rxnorm_name": "metformin hydrochloride 500 MG Oral Tablet"},
    {"pill_id": "B_1", "name": "JANUMET 50-500", "generic_name": "SITAGLIPTIN AND METFORMIN", "shape": "CAPSULE",
     "colour": "PINK", "imprint": "575", "rxnorm_name": "sitagliptin 50 MG / metformin hydrochloride 500 MG"},
    {"pill_id": "C_1", "name": "LIPITOR 10MG", "generic_name": "ATORVASTATIN CALCIUM", "shape": "OVAL",
     "colour": "WHITE", "imprint": "PD;155", "rxnorm_name": "atorvastatin 10 MG Oral Tablet [Lipitor]"},
]
INGREDIENTS = [("A_1", "6809", "metformin"), ("B_1", "6809", "metformin"), ("B_1", "593411", "sitagliptin"),
               ("C_1", "83367", "atorvastatin")]


class FakeRxNorm:
    names = {"glucophage": "861007", "metformin": "6809", "lipitor": "153165"}
    ingredient_of = {"861007": [("6809", "metformin")], "6809": [("6809", "metformin")],
                     "153165": [("83367", "atorvastatin")]}

    def rxcui_by_name(self, name):
        return self.names.get(name.lower())

    def approximate(self, term):
        return [("861007", "glucophage")] if "gluco" in term.lower() else []

    def ingredients(self, rxcui):
        return self.ingredient_of.get(rxcui, [])


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    folder = tmp_path_factory.mktemp("catalogue")
    (folder / "images").mkdir()
    Image.new("RGB", (8, 8), "white").save(folder / "images" / "A_1_0.jpg")
    # A: front and back look different; B and C one picture each
    emb = [unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0), unit(0, 0, 0, 1)]
    write_catalogue(folder, PRODUCTS, INGREDIENTS, [("A_1", 0, "rximage", "A_1_0.jpg")], emb,
                    ["A_1", "A_1", "B_1", "C_1"], ["reference", "photo", "reference", "reference"],
                    {"model_version": "test-1"})
    return Catalogue(folder)


def test_identify_uses_the_best_view_of_each_product(catalogue):
    (top, *_), notes = catalogue.identify(unit(0.1, 1, 0, 0.2)[None], top_k=3)
    assert top["pill_id"] == "A_1" and notes == []  # matches A's back
    both_sides = catalogue.identify(np.stack([unit(1, 0, 0, 0), unit(0, 1, 0, 0)]), top_k=1)[0]
    assert both_sides[0]["pill_id"] == "A_1" and both_sides[0]["similarity"] == pytest.approx(1.0, abs=1e-3)


def test_identify_filters(catalogue):
    query = unit(0, 0, 1, 0.1)[None]  # looks like B
    assert catalogue.identify(query, imprint="PD 155")[0][0]["pill_id"] == "C_1"
    assert catalogue.identify(query, shape="round")[0][0]["pill_id"] == "A_1"
    candidates, notes = catalogue.identify(query, imprint="ZZZ 999")
    assert candidates[0]["pill_id"] == "B_1" and "imprint" in notes[0]


def test_search_by_brand_and_generic(catalogue):
    products, resolution = catalogue.search("Glucophage", FakeRxNorm())
    assert [(p["pill_id"], p["match"]) for p in products] == [("A_1", "same"), ("B_1", "contains")]
    assert resolution["ingredients"] == [{"rxcui": "6809", "name": "metformin"}]
    products, resolution = catalogue.search("glucofage", FakeRxNorm())  # typo: approximate match
    assert products[0]["pill_id"] == "A_1"
    products, resolution = catalogue.search("lipitor", None)  # no RxNorm: names only
    assert [p["pill_id"] for p in products] == ["C_1"] and resolution["rxnorm"] == "not used"
    assert [p["pill_id"] for p in catalogue.search("metformin", FakeRxNorm(), strength="50 MG")[0]] == ["B_1"]


def test_product_and_image(catalogue):
    p = catalogue.product("A_1")
    assert p["ingredients"] == [{"rxcui": "6809", "name": "metformin"}] and p["images"][0]["kind"] == "rximage"
    assert catalogue.image_file("A_1", 0).exists() and catalogue.image_file("A_1", 5) is None
    assert catalogue.product("nope") is None
