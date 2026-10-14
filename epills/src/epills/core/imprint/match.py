"""
Imprint matching and re-ranking: the embedding proposes candidates, the imprint read by OCR (ocr.py) reorders them.

A candidate's imprint is its recorded text (C3PI: sides and lines separated by ';', e.g. 'TEVA;5729'). Its tokens of
2+ characters are looked for inside the OCR reads (all rotations run together, letters and digits only), so a read
'738GG' still finds 'GG' and '738'. A token's weight is its rarity among all candidates' imprints (IDF, scaled to
0-1): reading '3145' is strong evidence, reading '93' (Teva prints it on hundreds of pills) is weak. A candidate's
imprint score is the weight of the best token found (0 when nothing is found or its imprint is unknown).

Re-ranking adds alpha x imprint score to the similarity of the top_k candidates; alpha and top_k are chosen on
validation. A pill whose imprint the OCR can't read keeps the embedding's order.
"""
import math
import re
from collections import Counter

import numpy as np

# the gallery imprint check (inference.matching.check_imprint): least token rarity that counts as a read, chosen on
# validation by scripts/evaluate_imprint_gallery.py (most pills named correctly, wrong rates +0.1 point at most).
# Run 4, galleries of 10, re-chosen 2026-10-13 with the served GPU reader (Florence-2-large, preset large_c92; the
# same 0.5 as with Florence-2-base) and GALLERY_CONFIRM_DISTINCT: validation 78.3% -> 81.5% named correctly; test
# 76.0% -> 78.3%, named wrongly 0.04% -> 0.06%, other pills accepted 3.8% -> 3.5%
# (outputs/runs/stage2_run4/eval/imprint_gallery_large_c92_distinct.json)
GALLERY_MIN_WEIGHT = 0.5
# a match is confirmed or contradicted only by tokens that tell the patient's medications apart (check_imprint's
# distinct_hits, blueprint B27): 'TV' read on a pill no longer confirms the patient's 'TV;7295' next to their 'TV;58'.
# Validation against the plain rule: named wrongly 35 -> 30, other pills accepted 6,158 -> 6,150, named correctly
# 81.49% -> 81.46%; in same-maker pairs no wrong name is 'confirmed' any more (36% were).
GALLERY_CONFIRM_DISTINCT = True


def compact(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def imprint_tokens(imprint):
    """'TEVA;5729' -> {'TEVA', '5729'}: the recorded tokens of 2+ characters."""
    return {t for t in (compact(p) for p in re.split(r"[;\s,/\-\.]+", str(imprint or ""))) if len(t) >= 2}


class ImprintIndex:
    """The candidates' imprints (one per class, '' when unknown) with token rarity weights."""

    def __init__(self, imprints):
        self.tokens = [imprint_tokens(i) for i in imprints]
        df = Counter(t for toks in self.tokens for t in toks)
        n = max(1, sum(1 for toks in self.tokens if toks))
        idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        top = max(idf.values(), default=1.0)
        self.weight = {t: v / top for t, v in idf.items()}

    def scores(self, reads, candidates):
        """reads: OCR strings for one pill (all rotations, both sides); candidates: class indices -> scores."""
        text = [compact(r) for r in reads if compact(r)]
        out = np.zeros(len(candidates), np.float32)
        if not text:
            return out
        for j, c in enumerate(candidates):
            found = [self.weight[t] for t in self.tokens[c] if any(t in s for s in text)]
            out[j] = max(found, default=0.0)
        return out


def rerank(class_sims, reads, index, alpha, top_k=20):
    """
    class_sims: (queries, classes) similarities; reads: per query, its OCR strings. Returns new scores: the top_k
    classes of each query get + alpha x imprint score; the others keep their similarity.
    """
    out = np.array(class_sims, np.float32, copy=True)
    if alpha <= 0:
        return out
    for q in range(len(out)):
        top = np.argpartition(-out[q], min(top_k, out.shape[1] - 1))[:top_k]
        out[q, top] += alpha * index.scores(reads[q], top)
    return out


def distinct_tokens(imprints):
    """
    {key: recorded imprint} -> {key: its tokens that no other key's imprint contains}. 'TV;58' next to 'TV;7295' keeps
    {'58'}: reading 'TV' says nothing about which of the two it is. A token inside another's ('72' vs '7295') is dropped
    too, since reading the longer one also finds it.
    """
    tokens = {key: imprint_tokens(imprint) for key, imprint in imprints.items()}
    return {key: {t for t in own if not any(t in u for other, theirs in tokens.items() if other != key for u in theirs)}
            for key, own in tokens.items()}


def read_hits(reads, imprints, weight=None, min_weight=0.0, distinct=False):
    """
    Which recorded imprints were read on one pill. reads: its OCR strings (all rotations, both sides); imprints:
    {key: recorded imprint}; weight: token -> rarity (an ImprintIndex's .weight over the catalogue; a token it doesn't
    know counts as rare, 1.0); distinct: count only tokens that tell the keys apart (distinct_tokens). Returns {key:
    weight of its best token found}, for hits of at least min_weight.
    """
    text = [compact(r) for r in reads if compact(r)]
    if not text:
        return {}
    weight = weight or {}
    tokens = distinct_tokens(imprints) if distinct else {key: imprint_tokens(i) for key, i in imprints.items()}
    hits = {}
    for key, own in tokens.items():
        found = [weight.get(t, 1.0) for t in own if any(t in s for s in text)]
        if found and max(found) >= min_weight:
            hits[key] = max(found)
    return hits
