"""Stage 3 rules on a tiny made-up database and a fake RxNorm (no network)."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.core.safety.db import SafetyDB, write_db  # noqa: E402
from epills.core.safety.rules import PHARMACIST, check_dose, check_medications, resolve_medication  # noqa: E402

# made-up ingredients and codes; the levels follow DDInter's for these pairs
DRUGS = [("D1", "Warfarin"), ("D2", "Ibuprofen"), ("D3", "Atorvastatin"), ("D4", "Clarithromycin"),
         ("D5", "Acetylsalicylic acid"), ("D6", "Morphine"), ("D7", "Morphine (liposomal)"), ("D8", "Acetaminophen")]
INGREDIENTS = [("D1", "11289"), ("D2", "5640"), ("D3", "83367"), ("D4", "21212"), ("D5", "1191"),
               ("D6", "7052"), ("D7", "7052"), ("D8", "161")]
PAIRS = [("D1", "D2", "Major"), ("D3", "D4", "Major"), ("D2", "D8", "Unknown"), ("D1", "D5", "Major"),
         ("D2", "D5", "Moderate"), ("D6", "D2", "Minor"), ("D7", "D2", "Major")]


class FakeRxNorm:
    """Names, NDCs and ingredients for the made-up medications."""
    names = {"warfarin": "11289", "ibuprofen": "5640", "advil": "153008", "lipitor": "153165",
             "atorvastatin": "83367", "clarithromycin": "21212", "aspirin": "1191", "acetylsalicylic acid": "1191",
             "morphine": "7052", "tylenol": "202433", "acetaminophen": "161", "paracetamol": "161"}
    ingredient_of = {"11289": [("11289", "warfarin")], "5640": [("5640", "ibuprofen")],
                     "153008": [("5640", "ibuprofen")], "153165": [("83367", "atorvastatin")],
                     "617314": [("83367", "atorvastatin")], "83367": [("83367", "atorvastatin")],
                     "21212": [("21212", "clarithromycin")], "1191": [("1191", "aspirin")],
                     "7052": [("7052", "morphine")], "202433": [("161", "acetaminophen")],
                     "161": [("161", "acetaminophen")]}

    def rxcui_by_name(self, name):
        return self.names.get(name.lower().strip())

    def rxcui_by_ndc(self, ndc):
        return {"00071015523": "617314"}.get("".join(c for c in ndc if c.isdigit()))

    def approximate(self, term):
        # like RxNorm: the top candidate can be retired (no ingredients), so callers must try the next one
        if "lipitor" in term.lower():
            return [("999999", "retired concept"), ("153165", "atorvastatin 10 MG [Lipitor]")]
        return []

    def properties(self, rxcui):
        return {"617314": {"rxcui": "617314", "name": "atorvastatin 10 MG Oral Tablet [Lipitor]", "tty": "SBD"}
                }.get(rxcui)

    def ingredients(self, rxcui):
        return self.ingredient_of.get(rxcui, [])


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("safety") / "safety.sqlite"
    write_db(path, DRUGS, INGREDIENTS, PAIRS, {"source": "test"})
    return SafetyDB(path)


def screen(meds, db):
    resolved = [resolve_medication(m, FakeRxNorm()) for m in meds]
    return check_medications(resolved, db)


def test_brand_name_resolves_and_unknown_name_is_reported(db):
    alerts, per_med = screen([{"id": "a", "name": "Coumadin warfarin"}, {"id": "b", "name": "Advil"}], db)
    assert per_med[1]["ingredients"] == [{"rxcui": "5640", "name": "ibuprofen"}]
    assert [a.type for a in alerts] == ["unresolved"]  # "Coumadin warfarin" isn't a known name in the fake


def test_interaction_by_ndc_and_name(db):
    alerts, per_med = screen([{"id": "a", "ndc": "00071-0155-23", "name": "Lipitor 10 mg"},
                              {"id": "b", "name": "Clarithromycin"}], db)
    assert per_med[0]["resolved_by"] == "ndc"
    (alert,) = alerts
    assert (alert.type, alert.severity, alert.colour, alert.speak, alert.notify_caregiver) == \
           ("interaction", "major", "red", True, True)
    assert alert.medication_ids == ["a", "b"]


def test_code_only_medication_gets_a_readable_name(db):
    _, per_med = screen([{"id": "a", "ndc": "00071-0155-23"}], db)
    assert per_med[0]["name"] == "atorvastatin 10 MG Oral Tablet [Lipitor]"


def test_aspirin_matches_ddinter_acetylsalicylic_acid(db):
    alerts, _ = screen([{"id": "a", "name": "warfarin"}, {"id": "b", "name": "aspirin"}], db)
    assert [(a.type, a.severity) for a in alerts] == [("interaction", "major")]


def test_duplication_brand_and_generic(db):
    alerts, _ = screen([{"id": "a", "name": "Lipitor"}, {"id": "b", "name": "atorvastatin"}], db)
    (alert,) = alerts
    assert (alert.type, alert.severity, alert.colour) == ("duplication", "moderate", "amber")
    assert alert.details["shared_ingredients"] == ["atorvastatin"]


def test_ungraded_pair_is_on_screen_only(db):
    """Decision D6: DDInter's "Unknown" level is shown, but not spoken and not amber."""
    alerts, _ = screen([{"id": "a", "name": "ibuprofen"}, {"id": "b", "name": "paracetamol"}], db)
    (alert,) = alerts
    assert (alert.severity, alert.severity_graded, alert.colour, alert.speak) == ("ungraded", False, "grey", False)
    assert "isn't graded" in alert.message


def test_plain_ddinter_entry_preferred_over_qualified(db):
    # Morphine-ibuprofen is Minor; the liposomal form (Major) must not be used for a plain morphine pill
    alerts, _ = screen([{"id": "a", "name": "morphine"}, {"id": "b", "name": "ibuprofen"}], db)
    (alert,) = alerts
    assert alert.severity == "minor" and alert.colour == "grey" and not alert.speak


def test_ingredients_given_for_non_us_medicine(db):
    alerts, per_med = screen([{"id": "a", "name": "Panadol Extra (AU)", "ingredients": ["paracetamol"]},
                              {"id": "b", "name": "warfarin"}], db)
    assert per_med[0]["resolved_by"] == "ingredients"
    assert alerts == []  # no listed interaction in the made-up data


def test_approximate_name_is_reported(db):
    _, per_med = screen([{"id": "a", "name": "lipitor 10mg tabs"}], db)
    assert per_med[0]["resolved_by"] == "approximate"
    assert per_med[0]["matched_name"] == "atorvastatin 10 MG [Lipitor]"


def test_unresolved_and_failed_lookup_are_alerts_not_silence(db):
    class Broken(FakeRxNorm):
        def rxcui_by_name(self, name):
            raise ConnectionError("RxNav down")
    resolved = [resolve_medication({"id": "a", "name": "warfarin"}, Broken()),
                resolve_medication({"id": "b", "name": "Mystery pill"}, FakeRxNorm())]
    alerts, _ = check_medications(resolved, db)
    assert [a.type for a in alerts] == ["unresolved", "unresolved"]
    assert "RxNav down" in alerts[0].details["reason"]


def test_dose_check_counts():
    schedule = [{"medication_id": "met", "count": 1}, {"medication_id": "aml", "count": 1},
                {"medication_id": "ator", "count": 1}]
    pills = [{"medication_id": "met", "decision": "match"}, {"medication_id": "met", "decision": "match"},
             {"medication_id": "aml", "decision": "match"}, {"medication_id": "vitd", "decision": "match"},
             {"medication_id": None, "decision": "unknown"}, {"medication_id": "aml", "decision": "ambiguous"}]
    names = {"met": "metformin", "aml": "amlodipine", "ator": "atorvastatin", "vitd": "vitamin D"}
    alerts, summary = check_dose(schedule, pills, names)
    types = {a.type: a for a in alerts}
    assert set(types) == {"unknown_pill", "double_dose", "not_scheduled", "not_seen", "unsure_pill"}
    assert types["double_dose"].message.startswith("This photo shows two pills that look like your metformin instead "
                                                   "of one. Please count them: if there are two,")
    assert types["double_dose"].details["confirm_count"] and not types["double_dose"].notify_caregiver
    assert types["not_seen"].message.startswith("I can't see your atorvastatin in this photo.")  # never "missed"
    assert types["unknown_pill"].severity == "major" and types["not_seen"].severity == "moderate"
    assert summary["pills_seen"] == 6 and not summary["all_ok"] and not summary["all_visually_matched"]


def test_dose_check_all_fine():
    alerts, summary = check_dose([{"medication_id": "met", "count": 2}],
                                 [{"medication_id": "met", "decision": "match"}] * 2, {"met": "metformin"})
    assert alerts == [] and summary["all_visually_matched"] and \
        summary["text"] == "I see two pills. They all look like the medicines of this dose."


def test_every_message_follows_the_wording_rules(db):
    """Every alert ends with the pharmacist line; none tells anyone to take or stop a medication."""
    alerts, _ = screen([{"id": "a", "name": "warfarin"}, {"id": "b", "name": "aspirin"},
                        {"id": "c", "name": "Lipitor"}, {"id": "d", "name": "atorvastatin"},
                        {"id": "e", "name": "ibuprofen"}, {"id": "f", "name": "paracetamol"},
                        {"id": "g", "name": "Mystery"}], db)
    dose_alerts, _ = check_dose([{"medication_id": "a", "count": 1}, {"medication_id": "b", "count": 1}],
                                [{"medication_id": "a", "decision": "match"}] * 2 +
                                [{"medication_id": "c", "decision": "match"},
                                 {"medication_id": None, "decision": "unknown"},
                                 {"medication_id": None, "decision": "ambiguous"}])
    messages = [a.message for a in alerts + dose_alerts]
    assert len(messages) >= 8
    for m in messages:
        assert m.endswith(PHARMACIST), m
        assert not re.search(r"\b(take|taking|stop|stopping)\b", m, re.I), m
