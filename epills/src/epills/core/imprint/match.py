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
