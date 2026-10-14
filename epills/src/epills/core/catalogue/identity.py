"""
A catalogue product's identity, from its RxNorm name (blueprint job B13): products must never be merged across
strength, dose form or release, so these are returned as fields, not left inside a name.

RxNorm clinical and branded drug names follow one pattern:
  [12 HR | 24 HR] <ingredient> <amount> <unit> [/ <ingredient> <amount> <unit> ...] <dose form> [<Brand>]
e.g. "24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet [Glucophage]".
"""
import re

UNIT = r"(?:MG|MCG|UNT|MEQ|ML|%|MG/ML|MG/MG|MCG/ML|UNT/ML|MEQ/ML|MG/HR|MCG/HR|MG/ACTUAT|MCG/ACTUAT|ACTUAT)"
AMOUNT = re.compile(rf"(\d+(?:\.\d+)?)\s+({UNIT})(?=\s|$)")


def product_identity(rxnorm_name, name=None):
    """-> strength, strengths (per ingredient), dose_form, release (IR / ER / DR), brand, brand_or_generic."""
    out = {"strength": None, "strengths": [], "dose_form": None, "release": None, "brand": None,
           "brand_or_generic": None}
    text = (rxnorm_name or "").strip()
    if not text:
        return out
    brand = re.search(r"\s*\[([^\]]+)\]\s*$", text)
    if brand:
        out["brand"], text = brand.group(1), text[:brand.start()]
    out["brand_or_generic"] = "brand" if brand else "generic"
    if text.startswith("{"):  # a pack, e.g. "{7 (... Oral Tablet) / 7 (... Oral Tablet) } Pack": several pills
        out["dose_form"], out["release"] = "Pack", "IR"
        out["pack"] = [{"count": int(n), **{k: v for k, v in product_identity(re.sub(r"\s*\[[^\]]*\]", "", c)).items()
                                            if k in ("strength", "dose_form", "release")}}
                       for n, c in re.findall(r"(\d+) \(([^()]*)\)", text)]
        return out
    upper = text.upper()
    out["release"] = ("ER" if re.match(r"^\d+ HR ", text) or "EXTENDED RELEASE" in upper else
                      "DR" if "DELAYED RELEASE" in upper else "IR")
    amounts = list(AMOUNT.finditer(text))
    if amounts:
        out["dose_form"] = text[amounts[-1].end():].strip() or None
        body = re.sub(r"^\d+ HR ", "", text[:amounts[-1].end()])
        for part in body.split(" / "):
            m = AMOUNT.search(part)
            if m:
                out["strengths"].append({"ingredient": part[:m.start()].strip(),
                                         "strength": f"{m.group(1)} {m.group(2)}"})
        out["strength"] = " / ".join(s["strength"] for s in out["strengths"]) or None
    return out
