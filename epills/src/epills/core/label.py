"""
Medicines on a box or pharmacy label (feature F8): OCR lines (imprint.ocr.ImprintReader.read_text) -> candidate
medicines with strength and RxNorm codes, to fill the medication list (and so the Stage 3 checks).

1. Names: the lines are searched for RxNorm's ingredient and brand names (IN, PIN, BN: ~23,500, fetched once), plus
   international names RxNorm lacks (INN_TO_US: paracetamol -> acetaminophen). Phrases of 1-4 words match exactly
   (case and punctuation ignored), longest first; single words of 6+ letters also match with one OCR error (two from
   9 letters). Everyday words that happen to be names ("water", "relief") are skipped (COMMON).
2. Strength: amounts with units after the name on its line, else the line's only amount, else the next line's when
   that line names nothing ("10 mg" printed under "atorvastatin calcium").
3. RxNorm: name + strength -> the product concept (e.g. "atorvastatin 10 MG Oral Tablet [Lipitor]"), kept only if it
   has the name's ingredients and its strength; the ingredients always come from RxNorm.
4. Names whose ingredients are the same, or a subset of a brand's (paracetamol and caffeine under Panadol Extra),
   merge into one candidate. Best first: most evidence, then the order on the label.
RxNorm's own match scores are not used to find names: there, "Take 2 tablets 4 times a day" scores higher than
"PARACETAMOL 500MG TABLET".
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# international (INN/BAN, as printed in Singapore) -> RxNorm's US names; a list for combinations
INN_TO_US = {
    "paracetamol": ["acetaminophen"], "salbutamol": ["albuterol"], "adrenaline": ["epinephrine"],
    "noradrenaline": ["norepinephrine"], "frusemide": ["furosemide"], "glibenclamide": ["glyburide"],
    "lignocaine": ["lidocaine"], "amoxycillin": ["amoxicillin"], "cefalexin": ["cephalexin"],
    "clavulanic acid": ["clavulanate"], "potassium clavulanate": ["clavulanate"],
    "chlorphenamine": ["chlorpheniramine"], "mesalazine": ["mesalamine"], "pethidine": ["meperidine"],
    "thyroxine": ["levothyroxine"], "sodium valproate": ["valproate"], "isoprenaline": ["isoproterenol"],
    "aciclovir": ["acyclovir"], "ciclosporin": ["cyclosporine"], "colecalciferol": ["cholecalciferol"],
    "rifampicin": ["rifampin"], "beclometasone": ["beclomethasone"], "dexamfetamine": ["dextroamphetamine"],
    "glyceryl trinitrate": ["nitroglycerin"], "phenobarbitone": ["phenobarbital"], "oestradiol": ["estradiol"],
    "eformoterol": ["formoterol"], "colestyramine": ["cholestyramine"], "clomifene": ["clomiphene"],
    "orciprenaline": ["metaproterenol"], "oxpentifylline": ["pentoxifylline"], "dothiepin": ["dosulepin"],
    "co-amoxiclav": ["amoxicillin", "clavulanate"], "co-trimoxazole": ["sulfamethoxazole", "trimethoprim"],
    "co-codamol": ["acetaminophen", "codeine"],
}
# words on boxes and labels that are also RxNorm names, never a medicine on their own
COMMON = {
    "water", "sugar", "salt", "relief", "extra", "fast", "film", "coated", "tablet", "tablets", "capsule", "capsules",
    "oral", "daily", "night", "day", "care", "plus", "forte", "max", "ultra", "rapid", "soft", "children", "adult",
    "adults", "junior", "baby", "cold", "flu", "pain", "fever", "allergy", "sleep", "cough", "clear", "active",
    "original", "advance", "advanced", "complete", "health", "natural", "gold", "silver", "oxygen", "air", "alcohol",
    "glycerin", "talc", "starch", "lactose", "gelatin", "cellulose", "iron", "calcium", "sodium", "potassium",
    "magnesium", "zinc", "acid", "hydrochloride", "clinic", "pharmacy", "medicine", "tan", "lim", "lee", "ong", "ng",
}
UNIT = r"(?:mg|mcg|µg|ug|g|ml|iu|units?|%)"
STRENGTH = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({UNIT})(?:\s*/\s*(\d+(?:[.,]\d+)?)?\s*({UNIT}|tab|cap)s?)?(?![a-z])",
                      re.I)
# dose forms as printed -> as RxNorm names them
FORMS = {"tablet": "tablet", "tab": "tablet", "caplet": "tablet", "capsule": "capsule", "cap": "capsule",
         "syrup": "solution", "solution": "solution", "suspension": "suspension", "cream": "cream",
         "ointment": "ointment", "gel": "gel", "injection": "injection", "drops": "drops", "inhaler": "inhal",
         "patch": "patch", "spray": "spray", "suppository": "suppository"}


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


INN = {norm(k): v for k, v in INN_TO_US.items()}


def dose_form(lines):
    """The first dose form printed anywhere on the label, as RxNorm names it, or None."""
    for line in lines:
        for word in re.findall(r"[a-z]+", line.lower()):
            word = word[:-1] if word.endswith("s") and word[:-1] in FORMS else word
            if word in FORMS:
                return FORMS[word]
    return None


def _edits(a, b, limit):
    """Levenshtein distance, or limit + 1 once it exceeds limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _strength_text(m):
    amount, unit, per_amount, per_unit = m.groups()
    unit = unit.lower().replace("µg", "mcg").replace("ug", "mcg")
    text = f"{amount.replace(',', '.')}{'' if unit == '%' else ' '}{unit}"
    if per_unit:
        text += f"/{per_amount + ' ' if per_amount else ''}{per_unit.lower()}"
    return text


@dataclass
class Found:
    line: int
    start: int
    end: int
    text: str  # as printed
    rxcui: str
    tty: str  # IN, PIN, BN, or INN for a mapped international name
    us_names: list = field(default_factory=list)
    spelled: str | None = None  # RxNorm's spelling when the printed word was misread (fuzzy match)
    strength: str | None = None


class LabelReader:
    def __init__(self, rxnorm):
        self.rxnorm = rxnorm
        self._vocab = None

    @property
    def vocab(self):
        """normalised name -> (rxcui, tty), and single words by length for the fuzzy match; fetched once."""
        if self._vocab is None:
            concepts = self.rxnorm.get("allconcepts.json", tty="IN PIN BN")["minConceptGroup"]["minConcept"]
            names = {}
            for c in concepts:
                key = norm(c["name"])
                if key and key not in COMMON and len(key) >= 4 and (key not in names or c["tty"] == "IN"):
                    names[key] = (c["rxcui"], c["tty"], c["name"])
            by_len = {}
            for key in names:
                if " " not in key and len(key) >= 6 and not key.isdigit():
                    by_len.setdefault(len(key), []).append(key)
            self._vocab = (names, by_len)
        return self._vocab

    def _lookup(self, phrase):
        names, by_len = self.vocab
        if phrase in INN:
            first = names.get(INN[phrase][0])
            return (first[0] if first else None), "INN", INN[phrase], None
        if phrase in names:
            return names[phrase][0], names[phrase][1], [], None
        if " " not in phrase and len(phrase) >= 6 and not any(ch.isdigit() for ch in phrase):
            limit = 1 if len(phrase) < 9 else 2
            for n in range(len(phrase) - limit, len(phrase) + limit + 1):
                for key in by_len.get(n, ()):
                    if key[0] == phrase[0] and _edits(phrase, key, limit) <= limit:
                        return names[key][0], names[key][1], [], names[key][2]
        return None

    def find_names(self, lines):
        """Each medicine name in the lines, with its strength (steps 1 and 2)."""
        found = []
        for li, line in enumerate(lines):
            words = [(m.start(), m.end(), m.group().lower()) for m in re.finditer(r"[A-Za-z][A-Za-z\-]*", line)]
            i = 0
            while i < len(words):
                for n in range(min(4, len(words) - i), 0, -1):
                    phrase = norm(" ".join(w for _, _, w in words[i:i + n]))
                    hit = self._lookup(phrase) if phrase and phrase not in COMMON else None
                    if hit and hit[0]:
                        start, end = words[i][0], words[i + n - 1][1]
                        found.append(Found(li, start, end, line[start:end], hit[0], hit[1], hit[2], hit[3]))
                        i += n
                        break
                else:
                    i += 1
        for f in found:  # strengths
            same = [f2 for f2 in found if f2.line == f.line and f2.start > f.start]
            stop = min((f2.start for f2 in same), default=None)
            amounts = list(STRENGTH.finditer(lines[f.line]))
            after = [m for m in amounts if m.start() >= f.end and (stop is None or m.start() < stop)]
            if after:
                f.strength = _strength_text(after[0])
            elif len(amounts) == 1 and len([f2 for f2 in found if f2.line == f.line]) == 1:
                f.strength = _strength_text(amounts[0])
            elif f.line + 1 < len(lines) and not any(f2.line == f.line + 1 for f2 in found):
                nxt = list(STRENGTH.finditer(lines[f.line + 1]))
                if nxt:
                    f.strength = _strength_text(nxt[0])
        return found

    def _ingredients(self, f):
        """(rxcui, name) of the active ingredients. An ingredient name found in the list is its own answer (no
        RxNorm call: each costs up to a second from Singapore); brands and salts ask RxNorm."""
        names = self.vocab[0]
        if f.tty == "INN":
            known = [names.get(us) for us in f.us_names]
            if all(k and k[1] == "IN" for k in known):
                return sorted({(k[0], us) for k, us in zip(known, f.us_names)}, key=lambda x: x[1])
            out = []
            for us in f.us_names:
                rxcui = self.rxnorm.rxcui_by_name(us)
                out += self.rxnorm.ingredients(rxcui) if rxcui else []
            return sorted(set(out), key=lambda x: x[1])
        if f.tty == "IN":
            return [(f.rxcui, (f.spelled or f.text).lower())]  # RxNorm's ingredient name, as printed or corrected
        return self.rxnorm.ingredients(f.rxcui)

    def _product(self, name, strength, form, ingredients):
        """RxNorm's concept for name + strength, if it has these ingredients and this strength (step 3)."""
        number = re.match(r"[\d.]+", strength).group()
        want = {r for r, _ in ingredients}
        candidates = [r for r, _ in self.rxnorm.approximate(" ".join(filter(None, (name, strength, form))))]
        with ThreadPoolExecutor(5) as pool:  # each candidate's details in parallel, then judged in RxNorm's order
            props = list(pool.map(lambda r: self.rxnorm.properties(r) or {}, candidates))
            fit = [(r, p) for r, p in zip(candidates, props)
                   if p.get("tty") in ("SCD", "SBD", "SCDC", "SBDC", "GPCK", "BPCK")
                   and re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", p.get("name", ""))]
            wanted_names = [n.lower() for _, n in ingredients]
            # a clinical drug's name lists its ingredients ("acetaminophen 500 MG Oral Tablet"): no lookup needed
            named = [all(n in p["name"].lower() for n in wanted_names) for _, p in fit]
            ingredients_of = list(pool.map(lambda rp: None if rp[1] else self.rxnorm.ingredients(rp[0][0]),
                                           zip(fit, named)))
        fallback = None
        for (rxcui, p), is_named, ins in zip(fit, named, ingredients_of):
            if is_named or want <= {r for r, _ in ins}:
                if not form or form in p["name"].lower():
                    return rxcui, p["name"]
                fallback = fallback or (rxcui, p["name"])  # right medicine and strength, other dose form
        return fallback

    def read(self, lines):
        """OCR lines -> candidate medicines, best first (steps 1-4)."""
        form = dose_form(lines)
        found_all = self.find_names(lines)
        with ThreadPoolExecutor(8) as pool:  # RxNorm lookups in parallel (the client keeps to NLM's rate limit)
            ingredient_lists = list(pool.map(self._ingredients, found_all))
        groups = [{"found": [f], "ingredients": ing} for f, ing in zip(found_all, ingredient_lists) if ing]
        groups.sort(key=lambda g: -len(g["ingredients"]))  # brands with several ingredients absorb their parts
        merged = []
        for g in groups:
            keys = {r for r, _ in g["ingredients"]}
            home = next((m for m in merged if keys <= {r for r, _ in m["ingredients"]}), None)
            if home:
                home["found"] += g["found"]
            else:
                merged.append(g)
        plans = []
        for g in merged:
            found = sorted(g["found"], key=lambda f: (f.line, f.start))
            brand = next((f for f in found if f.tty == "BN"), None)
            strengths = []
            for f in found:
                if f.strength and f.strength not in strengths:
                    strengths.append(f.strength)
            lead = brand or found[0]
            query = (lead.spelled or lead.text) if lead.tty != "INN" else " ".join(lead.us_names)
            plans.append((g, found, brand, strengths, lead, query))
        with ThreadPoolExecutor(8) as pool:  # the exact products, in parallel
            products = list(pool.map(lambda p: self._product(p[5], p[3][0], form, p[0]["ingredients"])
                                     if len(p[3]) == 1 else None, plans))
        out = []
        for (g, found, brand, strengths, lead, _), product in zip(plans, products):
            printed = lead.spelled[:1].upper() + lead.spelled[1:] if lead.spelled else lead.text
            label = printed if not printed.isupper() else printed.capitalize()
            strength = strengths[0] if len(strengths) == 1 else ", ".join(strengths) or None
            out.append({
                "name": f"{label} {strength}" if strength else label,
                "brand": brand.text if brand else None,
                "ingredients": [{"rxcui": r, "name": n} for r, n in g["ingredients"]],
                "strength": strength,
                "rxcui": product[0] if product else (lead.rxcui if lead.tty != "INN" else g["ingredients"][0][0]),
                "rxnorm_name": product[1] if product else None,
                "found_in": [lines[i] for i in sorted({f.line for f in found})],
                "first_line": found[0].line,
                "_rank": (-(bool(brand) + bool(strengths) + bool(product) + len(found)), found[0].line),
            })
        out.sort(key=lambda c: c.pop("_rank"))
        return out


# ---- what else a label or prescription says (blueprint job B7) -------------------------------------------------------
DIRECTION_START = re.compile(r"^\s*(take|apply|use|inhale|instil+|insert|dissolve|chew|swallow|place|spray|give|put|"
                             r"rub|gargle|dilute|one|two|1|2)\b", re.I)
DIRECTION_WORDS = re.compile(r"\b(times? (a|per) day|(once|twice|thrice) (a|per) day|daily|every \d+ hours?|"
                             r"when necessary|as needed|at night|in the "
                             r"morning|at bedtime|before (meals|food|breakfast)|after (meals|food|breakfast)|"
                             r"with (meals|food)|on an empty stomach)\b", re.I)
QUANTITY = re.compile(r"\b(?:qty|quantity|dispensed)\s*[:.]?\s*(\d+)\s*([a-z]*)|\b(\d+)\s+"
                      r"(?:(?!(?:mg|mcg|g|ml|iu)\b)[a-z\-]+\s+)?(tablets?|capsules?|caplets?|tabs?|caps?|sachets?)\b",
                      re.I)
DATE = re.compile(r"\b(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
                  r"[a-z]* \d{4}|\d{1,2}[/.\-]\d{4})\b", re.I)
PERSONAL = re.compile(r"\b(name|patient|nric|fin|address|tel|phone|dob|date of birth)\b", re.I)


def _field(value, line):
    return {"value": value, "line": line} if value else None


def document_fields(lines, candidates):
    """
    Per candidate, what the label prints about it, each value with the line it was read from; None when it isn't
    printed (never inferred). A candidate's block runs from its first line to the next candidate's (the first one
    also gets the header lines above it: the dispensing date). Lines with personal details are never used.
    """
    starts = sorted({c["first_line"] for c in candidates})
    for c in candidates:
        i = starts.index(c["first_line"])
        lo = 0 if i == 0 else starts[i]
        hi = starts[i + 1] if i + 1 < len(starts) else len(lines)
        block = [(n, lines[n]) for n in range(lo, hi) if not PERSONAL.search(lines[n])]
        directions, quantity, date, expiry = [], None, None, None
        for n, line in block:
            is_direction = n >= c["first_line"] and (DIRECTION_START.search(line) and DIRECTION_WORDS.search(line)
                                                    or (directions and directions[-1][0] == n - 1
                                                        and DIRECTION_WORDS.search(line)))
            if is_direction:
                directions.append((n, line))
                continue
            q = QUANTITY.search(line)
            if q and quantity is None and n >= c["first_line"] and line not in c["found_in"]:
                number, unit = (q.group(1), q.group(2)) if q.group(1) else (q.group(3), q.group(4))
                quantity = _field(f"{number} {unit.lower()}".strip(), line)
            d = DATE.search(line)
            if d:
                if re.search(r"\bexp", line, re.I):
                    expiry = expiry or _field(d.group(1), line)
                else:
                    date = date or _field(d.group(1), line)
        form = dose_form([line for n, line in block if n >= c["first_line"]])
        c["fields"] = {
            "directions": _field(" ".join(l for _, l in directions), " / ".join(l for _, l in directions)),
            "quantity": quantity,
            "dose_form": _field(form, next((l for n, l in block if n >= c["first_line"] and form and
                                            re.search(form[:5], l, re.I)), None)),
            "date": date,
            "expiry_date": expiry,
        }
    return candidates


# ---- is this box one of my medicines? (blueprint job B9) -------------------------------------------------------------
def _amounts(strength):
    """'500 mg, 65 mg' or '500 MG / 65 MG' -> {(500.0, 'mg'), (65.0, 'mg')}."""
    return {(float(m.group(1).replace(",", ".")), m.group(2).lower().replace("µg", "mcg").replace("ug", "mcg"))
            for m in STRENGTH.finditer(strength or "")}


def compare_with_list(candidate, medications):
    """
    medications: the patient's list, each {"id", "name", "ingredients": {rxcui}, "strength": printed or None}.
    -> {"status", "medication_id", "message"}: a deterministic comparison of ingredients and strength, no model score.
      same_medicine        same ingredients and strength (or the list gives no strength)
      different_strength   same ingredients, another strength
      contains_listed      a combination that includes a listed medicine's ingredient
      not_on_list          no ingredient in common
    """
    have = {i["rxcui"] for i in candidate["ingredients"] if i.get("rxcui")}
    best = None
    for med in medications:
        common = have & med["ingredients"]
        if not common:
            continue
        if common == have == med["ingredients"]:
            a, b = _amounts(candidate.get("strength")), _amounts(med.get("strength"))
            if a and b and a != b:
                status = "different_strength"
                message = (f"This says {candidate.get('strength')}; your list has {med['name']}. Check with your "
                           "pharmacist.")
            else:
                status = "same_medicine"
                message = f"This matches your {med['name']}: the same medicine" + \
                          (" and strength." if a and b else ". Check the strength printed on it.")
            return {"status": status, "medication_id": med["id"], "message": message}
        best = best or {"status": "contains_listed", "medication_id": med["id"],
                        "message": f"This medicine shares an ingredient with your {med['name']} but is not the same. "
                                   "Check with your pharmacist."}
    return best or {"status": "not_on_list", "medication_id": None,
                    "message": "This medicine isn't on your list. Check with your pharmacist."}
