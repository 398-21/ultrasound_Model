"""
RxNorm (NLM RxNav REST API) lookups: from a medication's name, NDC or RxCUI to its active ingredients.

Ingredients are RxNorm IN concepts, e.g. 'Lipitor 10 mg' -> atorvastatin (83367), 'Tylenol with codeine' ->
acetaminophen (161) + codeine (2670). Free, no key; NLM asks for at most 20 requests per second per IP.
Standard library only (urllib), so the hosted image needs nothing extra. Responses are cached in memory, and
optionally on disk (the offline database build uses that, so a rerun doesn't repeat thousands of requests).
"""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormError(Exception):
    """RxNav couldn't be reached or answered with an error."""


class RxNormClient:
    def __init__(self, base_url=BASE_URL, cache_dir=None, timeout=10, max_per_second=15, fetch=None):
        """fetch(url) -> parsed JSON can be passed in for tests; by default the live API is used."""
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.min_interval = 1.0 / max_per_second
        self._fetch = fetch or self._http_get
        self._memory = {}
        self._lock = threading.Lock()
        self._last = 0.0

    # -- transport ---------------------------------------------------------------------------------------------
    def _http_get(self, url):
        with self._lock:  # stay under NLM's rate limit
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        last_error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                           "User-Agent": "epills-pharmascan/0.1"})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                time.sleep(0.5 * 2 ** attempt)
        raise RxNormError(f"RxNorm request failed: {url} ({last_error})")

    def get(self, path, **params):
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}/{path}" + (f"?{query}" if query else "")
        if url in self._memory:
            return self._memory[url]
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
            if cache_file.exists():
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                self._memory[url] = data
                return data
        data = self._fetch(url)
        self._memory[url] = data
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    # -- lookups -----------------------------------------------------------------------------------------------
    def rxcui_by_name(self, name):
        """Exact or normalised name match (brand, generic or ingredient names); None if not found."""
        ids = (self.get("rxcui.json", name=name, search=2).get("idGroup") or {}).get("rxnormId") or []
        return ids[0] if ids else None

    def rxcui_by_ndc(self, ndc):
        """RxCUI of a National Drug Code (the code on a US pill bottle); None if not found."""
        digits = "".join(ch for ch in str(ndc) if ch.isdigit())
        ids = (self.get("rxcui.json", idtype="NDC", id=digits).get("idGroup") or {}).get("rxnormId") or []
        return ids[0] if ids else None

    def approximate(self, term, max_candidates=5):
        """
        Fuzzy matches for free text such as 'lipitor 10mg' or a typo, best first, as [(rxcui, name or None)] with
        each concept once. Callers try them in order: the top match can be a retired concept with no ingredients.
        """
        candidates = (self.get("approximateTerm.json", term=term, maxEntries=max_candidates * 3)
                      .get("approximateGroup") or {}).get("candidate") or []
        names, order = {}, []
        for c in candidates:
            if c["rxcui"] not in names:
                order.append(c["rxcui"])
                names[c["rxcui"]] = None
            if c.get("name") and not names[c["rxcui"]]:
                names[c["rxcui"]] = c["name"]
        return [(r, names[r]) for r in order[:max_candidates]]

    def properties(self, rxcui):
        """{'rxcui', 'name', 'tty', ...} of a concept, or None."""
        return self.get(f"rxcui/{rxcui}/properties.json").get("properties")

    def ingredients(self, rxcui):
        """Active ingredients (IN) of any concept, as [(rxcui, name)]; an ingredient returns itself."""
        groups = (self.get(f"rxcui/{rxcui}/related.json", tty="IN").get("relatedGroup") or {}).get("conceptGroup") or []
        out = []
        for g in groups:
            for c in g.get("conceptProperties") or []:
                out.append((c["rxcui"], c["name"]))
        if not out:
            props = self.properties(rxcui)
            if props and props.get("tty") == "IN":
                out = [(props["rxcui"], props["name"])]
        return sorted(set(out), key=lambda x: x[1])
