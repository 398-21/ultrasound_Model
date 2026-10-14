"""
Official label text from the openFDA drug label API, shown only when the user asks (the plan's "Official label text").

The text is quoted, never summarised or turned into advice; openFDA itself says its results are unvalidated, so the
response carries that disclaimer. Free; an optional API key (OPENFDA_API_KEY) raises the rate limits.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://api.fda.gov/drug/label.json"
SECTIONS = ("boxed_warning", "contraindications", "warnings_and_cautions", "warnings", "do_not_use",
            "stop_use", "ask_doctor", "ask_doctor_or_pharmacist")
DISCLAIMER = ("Label text from openFDA (U.S. FDA), shown as written. openFDA results are unvalidated and are not "
              "advice. Check with your pharmacist.")


def _search(query, timeout=10, fetch=None):
    params = {"search": query, "limit": 1}
    key = os.environ.get("OPENFDA_API_KEY")
    if key:
        params["api_key"] = key
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    if fetch:
        return fetch(url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:  # openFDA answers 404 when nothing matches
            return {"results": []}
        raise


def label_warnings(rxcui=None, ingredient=None, max_chars=2000, fetch=None):
    """
    The warning sections of one current label, matched by RxCUI first, then by generic (ingredient) name.
    Returns None when no label is found.
    """
    queries = []
    if rxcui:
        queries.append(f'openfda.rxcui:"{rxcui}"')
    if ingredient:
        queries.append(f'openfda.generic_name:"{ingredient}"')
    for q in queries:
        results = _search(q, fetch=fetch).get("results") or []
        if results:
            r = results[0]
            sections = {}
            for name in SECTIONS:
                if r.get(name):
                    text = " ".join(r[name])
                    sections[name] = text[:max_chars] + ("…" if len(text) > max_chars else "")
            of = r.get("openfda", {})
            set_id = r.get("set_id")
            return {
                "brand_name": (of.get("brand_name") or [None])[0],
                "generic_name": (of.get("generic_name") or [None])[0],
                "label_date": r.get("effective_time"),
                "matched_by": q.split(":")[0].replace("openfda.", ""),
                "sections": sections,
                "disclaimer": DISCLAIMER,
                # for citation (the blueprint's Medication Knowledge tool): which label, from where
                "source": "US FDA drug label (structured product labeling), via openFDA",
                "set_id": set_id, "label_id": r.get("id"),
                "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if set_id else None,
            }
    return None
