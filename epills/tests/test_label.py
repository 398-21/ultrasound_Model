"""Box and label reading (epills.core.label): OCR lines -> medicines, with a small stand-in for RxNorm."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.core.label import STRENGTH, LabelReader, _strength_text, dose_form  # noqa: E402

CONCEPTS = [("161", "IN", "acetaminophen"), ("1886", "IN", "caffeine"), ("83367", "IN", "atorvastatin"),
            ("83366", "PIN", "atorvastatin calcium"), ("153165", "BN", "Lipitor"),
            ("1872070", "BN", "Panadol Extra"), ("11295", "IN", "water"), ("6809", "IN", "metformin")]
INGREDIENTS = {"161": [("161", "acetaminophen")], "1886": [("1886", "caffeine")],
               "83367": [("83367", "atorvastatin")], "83366": [("83367", "atorvastatin")],
               "153165": [("83367", "atorvastatin")], "1872070": [("161", "acetaminophen"), ("1886", "caffeine")],
               "6809": [("6809", "metformin")], "617314": [("83367", "atorvastatin")],
               "617312": [("83367", "atorvastatin")]}
PRODUCTS = {"617312": {"tty": "SCDC", "name": "atorvastatin 10 MG"},
            "617314": {"tty": "SBD", "name": "atorvastatin 10 MG Oral Tablet [Lipitor]"}}


class FakeRxNorm:
    def __init__(self):
        self.asked = []

    def get(self, path, **params):
        assert path == "allconcepts.json" and params["tty"] == "IN PIN BN"
        return {"minConceptGroup": {"minConcept": [{"rxcui": r, "tty": t, "name": n} for r, t, n in CONCEPTS]}}

    def rxcui_by_name(self, name):
        return next((r for r, _, n in CONCEPTS if n == name), None)

    def ingredients(self, rxcui):
        return INGREDIENTS.get(rxcui, [])

    def properties(self, rxcui):
        return PRODUCTS.get(rxcui)

    def approximate(self, term):
        self.asked.append(term)
        return [("617312", None), ("617314", None)] if term.lower().startswith("lipitor 10 mg") else []


def test_strengths_and_dose_forms():
    def strength(text):
        return _strength_text(STRENGTH.search(text))
    assert [strength(t) for t in ("500MG", "0,5 mg", "250 mg/5 ml", "1%", "100 IU")] == \
        ["500 mg", "0.5 mg", "250 mg/5 ml", "1%", "100 iu"]
    assert STRENGTH.search("Take 2 tablets") is None
    assert dose_form(["Lipitor", "30 film-coated tablets"]) == "tablet" and dose_form(["Lipitor"]) is None


def test_brand_and_generic_merge_and_the_product_needs_the_printed_form():
    rx = FakeRxNorm()
    [c] = LabelReader(rx).read(["Lipitor", "atorvastatin calcium", "10 mg", "30 film-coated tablets", "Pfizer"])
    assert (c["name"], c["brand"], c["strength"]) == ("Lipitor 10 mg", "Lipitor", "10 mg")
    assert (c["rxcui"], c["rxnorm_name"]) == ("617314", "atorvastatin 10 MG Oral Tablet [Lipitor]")  # not the SCDC
    assert [i["name"] for i in c["ingredients"]] == ["atorvastatin"]
    assert rx.asked == ["Lipitor 10 mg tablet"]


def test_international_names_combinations_and_instructions():
    reader = LabelReader(FakeRxNorm())
    [c] = reader.read(["EXAMPLE CLINIC Tel: 6123 4567", "PARACETAMOL 500MG TABLET", "Take 2 tablets 4 times a day",
                       "Drink water"])
    assert (c["name"], c["rxcui"], c["rxnorm_name"]) == ("Paracetamol 500 mg", "161", None)  # 'water' is skipped
    assert c["found_in"] == ["PARACETAMOL 500MG TABLET"]  # only the line naming the medicine comes back
    [c] = reader.read(["Panadol Extra", "Paracetamol 500mg Caffeine 65mg"])
    assert c["brand"] == "Panadol Extra" and c["strength"] == "500 mg, 65 mg"
    assert [i["name"] for i in c["ingredients"]] == ["acetaminophen", "caffeine"]


def test_misread_names_get_their_real_spelling_and_nothing_is_invented():
    reader = LabelReader(FakeRxNorm())
    [c] = reader.read(["Atorvastatln 20 mg"])
    assert c["name"] == "Atorvastatin 20 mg"
    assert reader.read(["Keep out of reach of children", "Store below 30C"]) == []


def test_a_prescription_gives_each_medicine_its_own_directions_and_never_personal_lines():
    from epills.core.label import compare_with_list, document_fields
    reader = LabelReader(FakeRxNorm())
    lines = ["EXAMPLE CLINIC", "Name: MR A TAN", "Date: 12 Oct 2026", "1. Lipitor 10 mg tablet",
             "Take 1 tablet at night", "Qty: 30 tablets", "2. Paracetamol 500mg tablet",
             "Take 2 tablets 4 times a day", "when necessary for pain", "Exp: 03/2028"]
    lipitor, para = sorted(document_fields(lines, reader.read(lines)), key=lambda c: c["first_line"])
    assert lipitor["fields"]["directions"]["value"] == "Take 1 tablet at night"
    assert lipitor["fields"]["quantity"]["value"] == "30 tablets" and lipitor["fields"]["date"]["value"] == "12 Oct 2026"
    assert para["fields"]["directions"]["value"] == "Take 2 tablets 4 times a day when necessary for pain"
    assert para["fields"]["quantity"] is None and para["fields"]["expiry_date"]["value"] == "03/2028"
    assert "MR A TAN" not in str(lipitor["fields"]) + str(para["fields"])
    meds = [{"id": "a", "name": "Lipitor 10 mg", "ingredients": {"83367"}, "strength": "10 mg"}]
    assert compare_with_list(lipitor, meds)["status"] == "same_medicine"
    assert compare_with_list(para, meds)["status"] == "not_on_list"
