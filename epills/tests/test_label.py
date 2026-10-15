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
    assert rx.asked == ["Lipitor 10 MG Oral Tablet"]  # one search when RxNorm-style phrasing finds it


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


def test_a_singapore_dispensing_label_gives_every_part_of_the_sticker():
    from epills.core.label import document_fields
    reader = LabelReader(FakeRxNorm())
    lines = ["SINGHEALTH POLYCLINICS - BEDOK", "Name: TAN AH KOW", "NRIC: S1234567D", "Date: 14/10/2026",
             "METFORMIN 500MG TABLET", "Qty: 60 TAB", "Take ONE tablet TWICE a day", "AFTER FOOD", "For diabetes",
             "Swallow whole. Do not crush.", "Exp: 10/2027"]
    (met,) = document_fields(lines, reader.read(lines))
    f = met["fields"]
    assert met["strength"] == "500 mg" and f["dose_form"]["value"] == "tablet"
    assert (f["quantity"]["amount"], f["quantity"]["unit"], f["quantity"]["line"]) == (60, "tablet", "Qty: 60 TAB")
    i = f["instructions"]
    assert i["text"] == "Take ONE tablet TWICE a day AFTER FOOD" and i["line"] == "Take ONE tablet TWICE a day / AFTER FOOD"
    assert (i["dose"], i["times_per_day"], i["food"], i["as_needed"]) == \
        ({"amount": 1, "amount_max": None, "unit": "tablet"}, 2, "after_food", False)
    assert f["indication"] == {"value": "diabetes", "line": "For diabetes"}
    assert [c["value"] for c in f["cautions"]] == ["Swallow whole. Do not crush."]
    assert f["date"]["value"] == "14/10/2026" and f["expiry_date"]["value"] == "10/2027"
    assert "TAN AH KOW" not in str(met) and "S1234567D" not in str(met)  # personal lines are never used


def test_as_needed_limits_and_shorthand_on_labels():
    from epills.core.label import document_fields
    reader = LabelReader(FakeRxNorm())
    lines = ["Paracetamol 500mg tab", "Quantity: 20 tablets",
             "Take 1-2 tablets every 6 hours when necessary for pain or fever",
             "Do not take more than 8 tablets in 24 hours", "Exp 01/2028"]
    (para,) = document_fields(lines, reader.read(lines))
    i = para["fields"]["instructions"]
    assert i["dose"] == {"amount": 1, "amount_max": 2, "unit": "tablet"} and i["every_hours"] == 6
    assert i["times_per_day"] is None and i["as_needed"] and i["max_per_day"] == "8 tablets in 24 hours"
    assert para["fields"]["indication"]["value"] == "pain or fever"
    lines = ["ATORVASTATIN 20MG TAB", "1 TAB ON", "To lower cholesterol", "Avoid grapefruit juice", "Qty: 30 TAB"]
    (ator,) = document_fields(lines, reader.read(lines))
    i = ator["fields"]["instructions"]
    assert (i["dose"]["amount"], i["times_per_day"], i["times_of_day"]) == (1, 1, ["night"])
    assert ator["fields"]["indication"]["value"] == "lower cholesterol"
    assert [c["value"] for c in ator["fields"]["cautions"]] == ["Avoid grapefruit juice"]
    lines = ["METFORMIN 500MG TABLET      Qty: 60 TAB", "Take 1 tablet daily", "Avoid caffeine drinks"]
    (met,) = document_fields(lines, reader.read(lines))  # a name in a caution isn't a dispensed medicine
    assert met["fields"]["quantity"]["amount"] == 60 and met["fields"]["cautions"][0]["value"] == "Avoid caffeine drinks"
    lines = ["PARACETAMOL 500MG TABLET", "Dosage: 2 tablets 3x/day when needed"]  # other wordings for directions
    (para,) = document_fields(lines, reader.read(lines))
    i = para["fields"]["instructions"]
    assert (i["dose"]["amount"], i["times_per_day"], i["as_needed"]) == (2, 3, True)
    lines = ["Lipitor", "atorvastatin calcium", "10 mg", "30 film-coated tablets"]  # a box: nothing for the patient
    (box,) = document_fields(lines, reader.read(lines))
    assert box["fields"]["instructions"] is None and box["fields"]["indication"] is None


def test_instructions_are_parsed_from_what_is_printed_only():
    from epills.core.label import indication_of, parse_instructions
    p = parse_instructions("Take half tablet once daily in the morning before breakfast")
    assert (p["dose"]["amount"], p["times_per_day"], p["times_of_day"], p["food"]) == \
        (0.5, 1, ["morning"], "before_food")
    p = parse_instructions("Take 2 capsules 3 times a day with or without food for 7 days")
    assert (p["dose"]["unit"], p["times_per_day"], p["food"], p["duration"]) == \
        ("capsule", 3, "with_or_without_food", "7 days")
    assert parse_instructions("1 TAB BD")["times_per_day"] == 2
    p = parse_instructions("Take 1 tablet(s) every 8 hours")
    assert p["every_hours"] == 8 and p["times_per_day"] is None  # never turned into times a day
    p = parse_instructions("Use 2 puffs when required")
    assert p["dose"] == {"amount": 2, "amount_max": None, "unit": "puff"} and p["as_needed"]
    p = parse_instructions("Take 10 ml three times daily after meals")
    assert (p["dose"]["amount"], p["dose"]["unit"], p["times_per_day"], p["food"]) == (10, "ml", 3, "after_food")
    p = parse_instructions("Take one tablet at night")
    assert p["times_per_day"] is None and p["times_of_day"] == ["night"] and p["food"] is None
    p = parse_instructions("Sig: 1 tab 2x/day after meals")
    assert (p["dose"]["amount"], p["dose"]["unit"], p["times_per_day"], p["food"]) == (1, "tablet", 2, "after_food")
    assert indication_of("Take 1 tablet daily for 5 days") is None
    assert indication_of("For external use only") is None
    assert indication_of("Indication: High blood pressure") == "High blood pressure"


MOH_LABEL = ["KEEP AWAY FROM CHILDREN", "NIFEDIPINE LA 60MG TAB", "(30 TAB)", "TAKE 1 TABLET ONCE EVERY MORNING",
             "May be taken with or without food.", "Swallow the tablet whole. Do not crush or chew.",
             "for blood pressure", "TAN AH SENG", "SXXXX123A 10/08/2021", "KNOW YOUR MEDICINE GENERAL HOSPITAL PHARMACY",
             "MEDICINE ROAD S(123456) TEL: 12345678"]  # MOH's "Know about medication label" example, as read


def test_the_moh_example_label_in_moh_s_sections():
    from epills.core.label import dispensed_by, document_fields
    CONCEPTS.append(("7417", "IN", "nifedipine"))
    INGREDIENTS["7417"] = [("7417", "nifedipine")]
    try:
        reader = LabelReader(FakeRxNorm())
        (nif,) = document_fields(MOH_LABEL, reader.read(MOH_LABEL))
    finally:
        CONCEPTS.pop(), INGREDIENTS.pop("7417")
    f, i = nif["fields"], nif["fields"]["instructions"]
    assert (nif["name"], nif["strength"], nif["release"], nif["release_printed"]) == ("Nifedipine LA 60 mg", "60 mg",
                                                                                      "ER", "LA")  # what
    assert (f["quantity"]["amount"], f["quantity"]["unit"]) == (30, "tablet")
    assert (i["dose"]["amount"], i["times_per_day"], i["times_of_day"], i["food"]) == \
        (1, 1, ["morning"], "with_or_without_food")  # how much, how often, when
    assert f["indication"] == {"value": "blood pressure", "line": "for blood pressure"}  # why
    assert {c["value"]: c["kind"] for c in f["cautions"]} == {
        "KEEP AWAY FROM CHILDREN": "storage", "Swallow the tablet whole. Do not crush or chew.": "how_to_take"}
    assert f["date"] == {"value": "10/08/2021", "line": "10/08/2021"}  # beside the NRIC: the date alone
    assert "SXXXX123A" not in str(nif) and "TAN AH SENG" not in str(nif) and "12345678" not in str(nif)
    assert dispensed_by(MOH_LABEL, [nif])["value"] == "KNOW YOUR MEDICINE GENERAL HOSPITAL PHARMACY"


def test_release_printed_and_a_product_of_another_release_is_never_given():
    from epills.core.label import release_printed
    assert release_printed(["NIFEDIPINE LA 60MG TAB"]) == ("ER", "LA")
    assert release_printed(["Metformin XR 500mg"]) == ("ER", "XR")
    assert release_printed(["Aspirin 100mg enteric-coated tablet"]) == ("DR", "enteric-coated")
    assert release_printed(["Lipitor 10 mg"]) == (None, None)
    reader = LabelReader(FakeRxNorm())
    # the only atorvastatin 10 mg products here are immediate release, so an extended-release label gets none
    assert reader._product("Lipitor", "10 mg", "tablet", [("83367", "atorvastatin")], "ER") is None
    assert reader._product("Lipitor", "10 mg", "tablet", [("83367", "atorvastatin")])[1] == \
        "atorvastatin 10 MG Oral Tablet [Lipitor]"
