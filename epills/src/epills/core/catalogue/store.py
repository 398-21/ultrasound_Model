"""
The pill catalogue: every pill we have pictures of (features F1b and F2 in docs/features.md).

A catalogue folder (built by scripts/build_catalogue.py) holds
  catalogue.sqlite    products (one per pill type: NDC9 + part), their RxNorm ingredients, display images, meta
  embeddings.npz      embeddings of each product's reference and everyday crops (emb, product, source, pill_ids);
                      they belong to one model version (meta model_version), and must be rebuilt for a new model
  images/             display pictures, served by the API

identify: a pill's embedding(s) -> the closest products. A product's score is its best match over its images
(front and back look different); with several photos of one pill (front + back), the scores are averaged.
search: a medicine name -> products, through RxNorm ingredients (brand or generic) and the names themselves.
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE products (
    pill_id TEXT PRIMARY KEY, ndc9 TEXT, part INTEGER, parts INTEGER, ndc11 TEXT, name TEXT, generic_name TEXT,
    manufacturer TEXT, shape TEXT, colour TEXT, size_mm REAL, imprint TEXT, imprint_type TEXT, imprint_colour TEXT,
    score INTEGER, rxcui TEXT, rxnorm_name TEXT, rxnorm_status TEXT);
CREATE TABLE ingredients (pill_id TEXT, rxcui TEXT, name TEXT);
CREATE TABLE images (pill_id TEXT, n INTEGER, kind TEXT, file TEXT, PRIMARY KEY (pill_id, n));
CREATE INDEX ingredients_rxcui ON ingredients (rxcui);
"""
PRODUCT_COLUMNS = ["pill_id", "ndc9", "part", "parts", "ndc11", "name", "generic_name", "manufacturer", "shape",
                   "colour", "size_mm", "imprint", "imprint_type", "imprint_colour", "score", "rxcui", "rxnorm_name",
                   "rxnorm_status"]
# imprint re-ranking in identify (imprint.match.rerank): the weight of the read imprint and how many of the closest
# products it may reorder, chosen on C3PI validation photos against this catalogue by scripts/tune_identify_rerank.py
# (2026-10-13, GPU reader large_c92). Held-out medicines: right product first 74.8% (0.3, 500, chosen earlier on the
# ePillID benchmark) -> 80.2%, top 5 91.7% -> 94.4% (outputs/runs/stage2_run4/eval/identify_rerank_large_c92.json)
IMPRINT_ALPHA, IMPRINT_TOP_K = 0.05, 100


def imprint_tokens(text):
    """'M;547' or 'm 547' -> ['M', '547']: upper case, letters and digits only."""
    return [t for t in (re.sub(r"[^A-Z0-9]", "", p) for p in re.split(r"[;\s,/\-]+", str(text or "").upper())) if t]


def imprint_matches(query, imprint):
    """A typed imprint matches when each typed token is on the pill, or its letters run together appear on it."""
    q, p = imprint_tokens(query), imprint_tokens(imprint)
    if not q or not p:
        return False
    return all(t in p for t in q) or "".join(q) in "".join(p)


def write_catalogue(folder, products, ingredients, images, emb, emb_pill_ids, emb_sources, meta):
    """products: dicts with PRODUCT_COLUMNS; ingredients: (pill_id, rxcui, name); images: (pill_id, n, kind, file);
    emb: (N, d) embeddings with their pill_id and source ('reference' or 'photo')."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "catalogue.sqlite"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.executemany(f"INSERT INTO products VALUES ({','.join('?' * len(PRODUCT_COLUMNS))})",
                    [[p.get(c) for c in PRODUCT_COLUMNS] for p in products])
    con.executemany("INSERT INTO ingredients VALUES (?, ?, ?)", ingredients)
    con.executemany("INSERT INTO images VALUES (?, ?, ?, ?)", images)
    con.commit()
    con.close()
    pill_ids = np.array(sorted({p["pill_id"] for p in products}))
    owner = np.searchsorted(pill_ids, np.asarray(emb_pill_ids))
    order = np.argsort(owner, kind="stable")
    np.savez(folder / "embeddings.npz", emb=np.asarray(emb, np.float16)[order], product=owner[order].astype(np.int32),
             source=np.asarray([s == "reference" for s in emb_sources], np.int8)[order], pill_ids=pill_ids)


class Catalogue:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.con = sqlite3.connect(f"file:{(self.folder / 'catalogue.sqlite').as_posix()}?mode=ro", uri=True,
                                   check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.meta = {r["key"]: json.loads(r["value"]) for r in self.con.execute("SELECT key, value FROM meta")}
        z = np.load(self.folder / "embeddings.npz")
        emb = z["emb"].astype(np.float32)
        self.emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        self.owner, self.pill_ids = z["product"], z["pill_ids"]
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.owner) != 0])  # segment start per product
        self.segment_pill = self.pill_ids[self.owner[self.starts]]
        rows = self.con.execute("SELECT pill_id, imprint, shape, colour FROM products").fetchall()
        self.attrs = {r["pill_id"]: (r["imprint"], r["shape"], r["colour"]) for r in rows}
        self.model_version = self.meta.get("model_version")
        self._imprint_index = self._siblings = None

    def __len__(self):
        return len(self.attrs)

    @property
    def imprint_index(self):
        """Every product's recorded imprint with its tokens' rarity (imprint.match.ImprintIndex), in product order."""
        if self._imprint_index is None:
            from epills.core.imprint.match import ImprintIndex
            self._imprint_index = ImprintIndex([self.attrs[p][0] or "" for p in self.segment_pill])
        return self._imprint_index

    # ---- identify ---------------------------------------------------------------------------------------------
    def identify(self, queries, top_k=5, imprint=None, shape=None, colour=None, reads=None, alpha=None, rerank_k=None):
        """
        queries: (n, d) unit embeddings of one pill (e.g. front and back). Returns (candidates, notes): candidates =
        [{pill_id, similarity, imprint_match, imprint_read}] best first. Filters (shape, colour, typed imprint) narrow
        the list; when nothing passes a filter, it is dropped and the notes say so. reads: the pill's imprint as read
        by OCR (all rotations, both sides): re-ranks the closest products by how well their recorded imprint matches
        (imprint.match.rerank, IMPRINT_ALPHA and IMPRINT_TOP_K unless alpha / rerank_k are given); imprint_read is
        that match's strength, 0-1.
        """
        q = np.asarray(queries, np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        per_product = np.maximum.reduceat(q @ self.emb.T, self.starts, axis=1).mean(axis=0)
        keep = np.ones(len(self.segment_pill), bool)
        notes = []
        for name, value, test in (("shape", shape, lambda a, v: (a[1] or "").upper() == v.upper()),
                                  ("colour", colour, lambda a, v: v.upper() in (a[2] or "").upper().split(";")),
                                  ("imprint", imprint, lambda a, v: imprint_matches(v, a[0]))):
            if not value:
                continue
            passed = np.array([test(self.attrs[p], value) for p in self.segment_pill])
            if (keep & passed).any():
                keep &= passed
            else:
                notes.append(f"no catalogue pill matches {name} '{value}'; ignored")
        score = per_product
        if reads:
            from epills.core.imprint.match import rerank
            score = rerank(per_product[None], [reads], self.imprint_index, IMPRINT_ALPHA if alpha is None else alpha,
                           IMPRINT_TOP_K if rerank_k is None else rerank_k)[0]
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-score[idx])][:top_k]
        read = self.imprint_index.scores(reads, idx) if reads else np.zeros(len(idx))
        return [{"pill_id": str(self.segment_pill[i]), "similarity": round(float(per_product[i]), 4),
                 "imprint_match": bool(imprint and imprint_matches(imprint, self.attrs[self.segment_pill[i]][0])),
                 "imprint_read": round(float(r), 3)}
                for i, r in zip(idx, read)], notes

    # ---- products -----------------------------------------------------------------------------------------------
    def product(self, pill_id):
        row = self.con.execute("SELECT * FROM products WHERE pill_id = ?", (pill_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["ingredients"] = [dict(r) for r in self.con.execute(
            "SELECT rxcui, name FROM ingredients WHERE pill_id = ? ORDER BY name", (pill_id,))]
        out["images"] = [dict(r) for r in self.con.execute(
            "SELECT n, kind FROM images WHERE pill_id = ? ORDER BY n", (pill_id,))]
        return out

    def look_alike_strengths(self, pill_id):
        """
        Products with the same active ingredients in another strength or release form that share this one's recorded
        shape and colour (test case M02): the photo can't tell them apart, so the strength needs the label.
        """
        if self._siblings is None:
            from epills.core.catalogue.identity import product_identity
            groups, self._siblings = {}, {}
            for r in self.con.execute("SELECT pill_id, rxnorm_name, shape, colour FROM products"):
                ings = frozenset(i["rxcui"] for i in self.con.execute(
                    "SELECT rxcui FROM ingredients WHERE pill_id = ?", (r["pill_id"],)) if i["rxcui"])
                if ings and r["shape"] and r["colour"]:
                    ident = product_identity(r["rxnorm_name"])
                    groups.setdefault((ings, r["shape"], r["colour"]), []).append(
                        (r["pill_id"], ident["strength"], ident["release"]))
            for members in groups.values():
                for pid, strength, release in members:
                    self._siblings[pid] = [{"pill_id": q, "strength": s, "release": rel} for q, s, rel in members
                                           if (s, rel) != (strength, release) and s]
        return self._siblings.get(pill_id, [])

    def image_file(self, pill_id, n):
        row = self.con.execute("SELECT file FROM images WHERE pill_id = ? AND n = ?", (pill_id, n)).fetchone()
        return self.folder / "images" / row["file"] if row else None

    # ---- search by name -----------------------------------------------------------------------------------------
    def search(self, name, rxnorm=None, strength=None, limit=50):
        """
        Products for a medicine name. With an RxNorm client, the name is resolved to its active ingredients (brand
        or generic, typos tolerated): products with exactly those ingredients come first ("same"), then products
        that also contain others ("contains"); plain name matches fill in ("name"). Returns (products, resolution).
        """
        from epills.core.label import INN_TO_US
        us_names = INN_TO_US.get(" ".join(name.lower().split()), [])  # co-amoxiclav: RxNorm only knows US names
        resolution = {"query": name, "ingredients": [], "rxnorm": "not used"}
        if us_names:
            resolution["us_names"] = us_names
        ranked = {}
        if rxnorm is not None:
            try:
                if us_names:
                    rxcuis = [rxnorm.rxcui_by_name(n) for n in us_names]
                    ingredients = [i for r in rxcuis if r for i in rxnorm.ingredients(r)]
                else:
                    rxcui = rxnorm.rxcui_by_name(name)
                    if not rxcui:
                        candidates = rxnorm.approximate(name)
                        rxcui = candidates[0][0] if candidates else None
                    ingredients = rxnorm.ingredients(rxcui) if rxcui else []
                resolution.update(rxnorm="ok" if ingredients else "no match",
                                  ingredients=[{"rxcui": r, "name": n} for r, n in ingredients])
            except Exception as error:  # offline: fall back to the names
                ingredients = []
                resolution["rxnorm"] = f"unavailable ({error})"
            wanted = {r for r, _ in ingredients}
            if wanted:
                rows = self.con.execute(
                    f"SELECT pill_id, rxcui FROM ingredients WHERE pill_id IN (SELECT pill_id FROM ingredients "
                    f"WHERE rxcui IN ({','.join('?' * len(wanted))}))", sorted(wanted)).fetchall()
                have = {}
                for r in rows:
                    have.setdefault(r["pill_id"], set()).add(r["rxcui"])
                for pid, ing in have.items():
                    if wanted <= ing:
                        ranked[pid] = "same" if ing == wanted else "contains"
        for n in [name.strip()] + us_names:
            like = f"%{n}%"
            for r in self.con.execute("SELECT pill_id FROM products WHERE name LIKE ? OR generic_name LIKE ? "
                                      "OR rxnorm_name LIKE ?", (like, like, like)):
                ranked.setdefault(r["pill_id"], "name")
        products = [self.product(pid) | {"match": how} for pid, how in ranked.items()]
        if strength:
            s = strength.lower().replace(" ", "")
            products = [p for p in products
                        if s in f"{p['rxnorm_name'] or ''} {p['name'] or ''}".lower().replace(" ", "")]
        order = {"same": 0, "contains": 1, "name": 2}
        products.sort(key=lambda p: (order[p["match"]], p["rxnorm_name"] or p["name"] or "", p["pill_id"]))
        return products[:limit], resolution
