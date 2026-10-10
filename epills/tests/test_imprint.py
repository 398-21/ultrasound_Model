"""Imprint matching and re-ranking (epills.core.imprint.match); the OCR model itself isn't loaded here."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.core.imprint.match import ImprintIndex, imprint_tokens, rerank  # noqa: E402

IMPRINTS = ["TEVA;5729", "93;3145", "93;7146", "93;1234", "GG;738", "", "M;547"]


def test_tokens():
    assert imprint_tokens("TEVA;5729") == {"TEVA", "5729"}
    assert imprint_tokens("M;547") == {"547"}  # single characters carry too little evidence
    assert imprint_tokens("") == set()


def test_rare_tokens_weigh_more_and_merged_reads_count():
    index = ImprintIndex(IMPRINTS)
    assert index.weight["93"] < index.weight["3145"]  # '93' is on three pills
    s = index.scores(["", "738GG", "x"], [4, 0])        # 'GG' and '738' read run together
    assert s[0] > 0 and s[1] == 0
    assert index.scores(["93"], [1])[0] < index.scores(["3145"], [1])[0]
    assert (index.scores([], [0, 1]) == 0).all() and index.scores(["TEVA"], [5])[0] == 0  # nothing read / unknown


def test_rerank_moves_the_imprint_match_up_only_among_the_top_k():
    index = ImprintIndex(IMPRINTS)
    sims = np.array([[0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40]], np.float32)
    out = rerank(sims, [["93 3145"]], index, alpha=0.3, top_k=3)
    assert out[0].argmax() == 1                          # the read '3145' lifts candidate 1 above candidate 0
    assert np.allclose(out[0, 3:], sims[0, 3:])          # outside the top 3, nothing changes
    assert np.allclose(rerank(sims, [[]], index, 0.3), sims)  # nothing read: the embedding's order stays
