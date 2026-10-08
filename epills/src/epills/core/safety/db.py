"""
The Stage 3 interaction database: DDInter 2.0 pairs keyed to RxNorm ingredients, in one SQLite file.

Built offline by scripts/build_safety_db.py from the DDInter CSVs; read at run time by the API. This module owns the
schema, so the builder and the reader can't drift apart.

Tables
  meta(key, value)                         source, licence, build date, counts
  drugs(ddinter_id, name, base_name, qualifier)
                                           qualifier is the parenthesised part of a DDInter name, e.g. "topical" in
                                           "Diflorasone (topical)"; empty for the plain drug
  drug_ingredients(ddinter_id, rxcui)      the RxNorm ingredient (IN) codes of each DDInter drug
  pairs(a, b, level)                       one row per interacting pair, a < b; level is Major / Moderate / Minor /
                                           Unknown
"""
import re
import sqlite3
from pathlib import Path

LEVELS = ("Major", "Moderate", "Minor", "Unknown")

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE drugs (ddinter_id TEXT PRIMARY KEY, name TEXT NOT NULL, base_name TEXT NOT NULL, qualifier TEXT NOT NULL);
CREATE TABLE drug_ingredients (ddinter_id TEXT NOT NULL, rxcui TEXT NOT NULL, PRIMARY KEY (ddinter_id, rxcui));
CREATE TABLE pairs (a TEXT NOT NULL, b TEXT NOT NULL, level TEXT NOT NULL, PRIMARY KEY (a, b));
CREATE INDEX drug_ingredients_rxcui ON drug_ingredients (rxcui);
CREATE INDEX drugs_base_name ON drugs (base_name);
"""


def normalise_name(name):
    """Lower case, single spaces; used to match names across sources."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def split_qualifier(name):
    """'Morphine (liposomal)' -> ('morphine', 'liposomal'); 'Warfarin' -> ('warfarin', '')."""
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", str(name).strip())
    if m:
        return normalise_name(m.group(1)), normalise_name(m.group(2))
    return normalise_name(name), ""


def write_db(path, drugs, ingredients, pairs, meta):
    """
    Write a new database. drugs: iterable of (ddinter_id, name); ingredients: iterable of (ddinter_id, rxcui);
    pairs: iterable of (id_a, id_b, level); meta: dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO drugs VALUES (?, ?, ?, ?)",
                    ((i, n, *split_qualifier(n)) for i, n in drugs))
    con.executemany("INSERT OR IGNORE INTO drug_ingredients VALUES (?, ?)", ingredients)
    rows = []
    for a, b, level in pairs:
        if level not in LEVELS:
            raise ValueError(f"unknown DDInter level {level!r}")
        rows.append((min(a, b), max(a, b), level))
    con.executemany("INSERT OR REPLACE INTO pairs VALUES (?, ?, ?)", rows)
    con.executemany("INSERT INTO meta VALUES (?, ?)", ((k, str(v)) for k, v in meta.items()))
    con.commit()
    con.close()
    tmp.replace(path)


class SafetyDB:
    """Read-only access to the interaction database."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"safety database not found: {self.path}")
        # read-only; check_same_thread=False because the API serves requests from a thread pool
        self.con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self.meta = dict(self.con.execute("SELECT key, value FROM meta"))

    def drugs_for_ingredient(self, rxcui=None, name=None):
        """
        DDInter drugs for one ingredient, as [(ddinter_id, name, qualifier)]. Matches the RxNorm code first, then the
        name. The plain entry is preferred: 'Morphine' rather than 'Morphine (liposomal)', since the app checks pills;
        qualified entries (topical, injection, ...) are used only when no plain one exists.
        """
        rows = []
        if rxcui:
            rows = self.con.execute(
                "SELECT d.ddinter_id, d.name, d.qualifier FROM drug_ingredients i JOIN drugs d USING (ddinter_id) "
                "WHERE i.rxcui = ?", (str(rxcui),)).fetchall()
        if not rows and name:
            base, _ = split_qualifier(name)
            rows = self.con.execute("SELECT ddinter_id, name, qualifier FROM drugs WHERE base_name = ?",
                                    (base,)).fetchall()
        plain = [r for r in rows if not r[2]]
        return plain or rows

    def level(self, id_a, id_b):
        """DDInter level for a pair of DDInter drugs, or None if no interaction is listed."""
        row = self.con.execute("SELECT level FROM pairs WHERE a = ? AND b = ?",
                               (min(id_a, id_b), max(id_a, id_b))).fetchone()
        return row[0] if row else None
