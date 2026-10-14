"""
Stage 3 rules: fixed logic on curated data, no trained model (ROADMAP §4).

  check_medications  when a medication is added or changed: drug-drug interactions (DDInter 2.0) and therapeutic
                     duplication (two medications sharing an active ingredient, via RxNorm)
  check_dose         at every dose check: extra pills, pills not part of this dose, scheduled medicines not seen, unknown and
                     unsure pills (from Stage 2's matching)

Alert rules from the project plan:
  - Major: red, spoken, caregiver notified. Moderate: amber, spoken. Minor: shown on screen only.
  - Every alert ends with "Check with your pharmacist." The app never tells anyone to take or stop a medication.
  - DDInter's "Unknown" level (listed, but not graded) gets its own severity, "ungraded": on screen only, not spoken
    (decision D6, 2026-10-08), so common medication lists don't drown the graded alerts in noise.
  - A medication that can't be identified is reported, never silently skipped: no alert must not mean "no
    interaction" when the check couldn't run.
"""
from dataclasses import asdict, dataclass, field
from itertools import combinations

from epills.core.safety.db import normalise_name

PHARMACIST = "Check with your pharmacist."
UNKNOWN_LEVEL_AS = "ungraded"  # decision D6
SOURCE_DDINTER = "DDInter 2.0 (CC BY-NC-SA 4.0), interaction severity"
SOURCE_RXNORM = "RxNorm (NLM), active ingredients"
SOURCE_SCHEDULE = "The patient's schedule and the pill check"

STYLE = {  # severity -> how the app shows it (the plan's severity mapping)
    "major": {"colour": "red", "speak": True, "notify_caregiver": True},
    "moderate": {"colour": "amber", "speak": True, "notify_caregiver": False},
    "ungraded": {"colour": "grey", "speak": False, "notify_caregiver": False},
    "minor": {"colour": "grey", "speak": False, "notify_caregiver": False},
}
RANK = {"major": 4, "moderate": 3, "ungraded": 2, "minor": 1}


@dataclass
class Alert:
    type: str
    severity: str
    message: str
    medication_ids: list
    source: str
    severity_graded: bool = True
    colour: str = ""
    speak: bool = False
    notify_caregiver: bool = False
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        style = STYLE[self.severity]
        self.colour, self.speak, self.notify_caregiver = style["colour"], style["speak"], style["notify_caregiver"]
        if not self.message.endswith(PHARMACIST):
            self.message = f"{self.message} {PHARMACIST}"

    def to_dict(self):
        return asdict(self)


@dataclass
class Ingredient:
    rxcui: str | None
    name: str


@dataclass
class ResolvedMedication:
    id: str
    display_name: str
    ingredients: list
    resolved_by: str  # ingredients / ndc / rxcui / name / approximate / unresolved
    matched_name: str | None = None
    note: str | None = None


def resolve_medication(med, rxnorm):
    """
    The active ingredients of one medication. med: dict with id and any of name, rxcui, ndc, ingredients (names
    printed on the label, for medicines without US codes). Order of trust: explicit ingredients, NDC, RxCUI, exact
    name, then approximate name (reported, so the app can ask "did you mean ...?").
    """
    mid, name = str(med["id"]), med.get("name") or ""
    display = name or ", ".join(med.get("ingredients") or []) or mid
    try:
        if med.get("ingredients"):
            found = []
            for ing in med["ingredients"]:
                rxcui = rxnorm.rxcui_by_name(ing) if rxnorm else None
                ins = rxnorm.ingredients(rxcui) if rxcui else []
                found += [Ingredient(r, n) for r, n in ins] or [Ingredient(None, normalise_name(ing))]
            return ResolvedMedication(mid, display, found, "ingredients")
        if rxnorm is None:
            return ResolvedMedication(mid, display, [], "unresolved", note="RxNorm lookups are switched off")
        for how, rxcui in (("ndc", med.get("ndc") and rxnorm.rxcui_by_ndc(med["ndc"])),
                           ("rxcui", med.get("rxcui")),
                           ("name", name and rxnorm.rxcui_by_name(name))):
            if rxcui:
                ins = rxnorm.ingredients(rxcui)
                if ins:
                    if not name:  # entered by code only: show RxNorm's name, not the app's id
                        display = (rxnorm.properties(rxcui) or {}).get("name") or display
                    return ResolvedMedication(mid, display, [Ingredient(r, n) for r, n in ins], how)
        if name:
            for rxcui, matched in rxnorm.approximate(name):  # best first; skip retired concepts
                ins = rxnorm.ingredients(rxcui)
                if ins:
                    return ResolvedMedication(mid, display, [Ingredient(r, n) for r, n in ins], "approximate",
                                              matched_name=matched)
    except Exception as error:  # RxNorm unreachable: report it, don't guess
        return ResolvedMedication(mid, display, [], "unresolved", note=f"lookup failed: {error}")
    return ResolvedMedication(mid, display, [], "unresolved", note="no match in RxNorm")


def _ingredient_list(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def check_medications(resolved, db):
    """
    Interactions and duplications across a patient's medication list (list of ResolvedMedication).
    Returns (alerts, per_medication), where per_medication says how each one was identified and checked.
    """
    alerts, per_med = [], []
    ddinter = {}  # medication id -> [(ingredient, [ddinter rows])]
    for m in resolved:
        rows = [(ing, db.drugs_for_ingredient(ing.rxcui, ing.name)) for ing in m.ingredients]
        ddinter[m.id] = rows
        not_listed = [ing.name for ing, r in rows if not r]
        per_med.append({
            "id": m.id, "name": m.display_name, "resolved_by": m.resolved_by, "matched_name": m.matched_name,
            "ingredients": [{"rxcui": i.rxcui, "name": i.name} for i in m.ingredients],
            "not_in_interaction_database": not_listed, "note": m.note,
        })
        if m.resolved_by == "unresolved":
            alerts.append(Alert(
                "unresolved", "moderate",
                f"I couldn't identify {m.display_name}, so it was not checked for interactions. "
                "Please add its active ingredient from the label.",
                [m.id], SOURCE_RXNORM, details={"reason": m.note}))

    # therapeutic duplication: the same active ingredient in two medications (brand + generic, or a combination
    # pill + a single one). Amber, not red: two strengths of one drug can be prescribed together on purpose.
    for a, b in combinations(resolved, 2):
        shared = sorted({i.name for i in a.ingredients} & {i.name for i in b.ingredients}
                        | {i.name for i in a.ingredients for j in b.ingredients if i.rxcui and i.rxcui == j.rxcui})
        if shared:
            alerts.append(Alert(
                "duplication", "moderate",
                f"{a.display_name} and {b.display_name} both contain {_ingredient_list(shared)}, "
                "so together they may be a double dose.",
                [a.id, b.id], SOURCE_RXNORM, details={"shared_ingredients": shared}))

    # interactions: every ingredient of one medication against every ingredient of another
    for a, b in combinations(resolved, 2):
        found = []
        for ing_a, rows_a in ddinter[a.id]:
            for ing_b, rows_b in ddinter[b.id]:
                if ing_a.rxcui and ing_a.rxcui == ing_b.rxcui:
                    continue  # same ingredient: a duplication, handled above
                for ra in rows_a:
                    for rb in rows_b:
                        level = db.level(ra[0], rb[0])
                        if level:
                            found.append({"ingredient_a": ing_a.name, "ingredient_b": ing_b.name,
                                          "ddinter_a": ra[1], "ddinter_b": rb[1], "level": level})
        if not found:
            continue
        severities = [UNKNOWN_LEVEL_AS if f["level"] == "Unknown" else f["level"].lower() for f in found]
        severity = max(severities, key=RANK.get)
        graded = severity != "ungraded"
        words = {"major": "a serious", "moderate": "a moderate", "minor": "a minor", "ungraded": ""}[severity]
        text = (f"{a.display_name} and {b.display_name} have {words} interaction." if graded else
                f"{a.display_name} and {b.display_name} are listed as interacting, but the severity isn't graded.")
        alerts.append(Alert("interaction", severity, text, [a.id, b.id], SOURCE_DDINTER,
                            severity_graded=graded, details={"pairs": found}))

    alerts.sort(key=lambda al: -RANK[al.severity])
    return alerts, per_med


def _number_word(n):
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return words[n] if 0 <= n < len(words) else str(n)


def _count_words(n, singular, plural=None):
    return f"{_number_word(n)} {singular if n == 1 else (plural or singular + 's')}"


def check_dose(schedule, pills, names=None):
    """
    Dose check after Stage 2's matching.
      schedule: [{"medication_id", "count"}] for this dose (count = pills expected)
      pills:    [{"medication_id", "decision"}], decision = match / unknown / ambiguous, from /v1/match
      names:    optional {medication_id: display name} for messages (enrolled medications)
    Returns (alerts, summary dict).
    """
    names = {str(k): v for k, v in (names or {}).items()}
    expected = {str(s["medication_id"]): int(s.get("count", 1)) for s in schedule}
    matched, unknown, ambiguous = {}, 0, 0
    for p in pills:
        decision = p.get("decision", "unknown")
        if decision == "match" and p.get("medication_id") is not None:
            mid = str(p["medication_id"])
            matched[mid] = matched.get(mid, 0) + 1
        elif decision == "ambiguous":
            ambiguous += 1
        else:
            unknown += 1
    name = lambda mid: names.get(mid, mid)
    alerts = []
    if unknown:
        what = "One pill doesn't" if unknown == 1 else f"{_count_words(unknown, 'pill').capitalize()} don't"
        alerts.append(Alert("unknown_pill", "major",
                            f"{what} look like any of your medicines. Please set {'it' if unknown == 1 else 'them'} "
                            "aside.",
                            [], SOURCE_SCHEDULE, details={"count": unknown}))
    for mid, n in sorted(matched.items()):
        e = expected.get(mid, 0)
        if e == 0:
            alerts.append(Alert("not_scheduled", "major",
                                f"{'One pill looks' if n == 1 else _count_words(n, 'pill').capitalize() + ' look'} like "
                                f"your {name(mid)}, which is not part of this dose. Please set "
                                f"{'it' if n == 1 else 'them'} aside.",
                                [mid], SOURCE_SCHEDULE, details={"found": n, "expected": 0}))
        elif n > e:
            # one pill boxed twice also looks like this (1.4% of one-pill photos with the stand-in finder, blueprint
            # B24): the patient confirms the count before an extra pill is recorded or a caregiver is told
            alert = Alert("double_dose", "major",
                          f"This photo shows {_count_words(n, 'pill')} that look like your {name(mid)} instead of "
                          f"{_number_word(e)}. Please count them: if there are {_number_word(n)}, set the extra "
                          f"{'one' if n - e == 1 else 'ones'} aside.",
                          [mid], SOURCE_SCHEDULE, details={"found": n, "expected": e, "confirm_count": True})
            alert.notify_caregiver = False
            alerts.append(alert)
    for mid, e in sorted(expected.items()):
        n = matched.get(mid, 0)
        if n < e:  # not seen in this photo: never a missed dose (the backend knows what was taken)
            text = (f"I can't see your {name(mid)} in this photo." if n == 0 else
                    f"I can see only {_count_words(n, 'pill')} that look like your {name(mid)}, instead of "
                    f"{_number_word(e)}.")
            alerts.append(Alert("not_seen", "moderate", text, [mid], SOURCE_SCHEDULE,
                                details={"found": n, "expected": e}))
    if ambiguous:
        alerts.append(Alert("unsure_pill", "moderate",
                            f"I'm not sure about {'one pill' if ambiguous == 1 else _count_words(ambiguous, 'pill')}. "
                            "Please turn it over and photograph the dose again.", [], SOURCE_SCHEDULE,
                            details={"count": ambiguous}))
    alerts.sort(key=lambda al: -RANK[al.severity])
    total = len(pills)
    summary = {
        "pills_seen": total, "matched": sum(matched.values()), "unknown": unknown, "unsure": ambiguous,
        "all_ok": not alerts,  # deprecated name of all_visually_matched
        "all_visually_matched": not alerts,
        "count_reliable": True,
        "text": (f"I see {_count_words(total, 'pill')}." +
                 (" They all look like the medicines of this dose." if not alerts and total else "")),
    }
    return alerts, summary
