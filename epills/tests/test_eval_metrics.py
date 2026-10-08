"""The ePillID metrics in scripts/evaluate_stage2.py, against hand-computed values (scikit-learn's definitions)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
spec = importlib.util.spec_from_file_location("evaluate_stage2", ROOT / "scripts" / "evaluate_stage2.py")
evaluate_stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_stage2)


def test_average_precision_matches_sklearn_definition():
    ap = evaluate_stage2.average_precision
    # ranks 1 and 3 are right: (1/1 + 2/3) / 2
    assert abs(ap(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6])) - 5 / 6) < 1e-12
    # a tie is one threshold: precision 1/2 at recall 1
    assert abs(ap(np.array([1, 0]), np.array([0.5, 0.5])) - 0.5) < 1e-12


def test_epillid_metrics():
    sim = np.array([[0.9, 0.1, 0.2],    # right class 0, rank 1
                    [0.8, 0.3, 0.6]])   # right class 1, rank 3
    m = evaluate_stage2.epillid_metrics(sim, np.array([0, 1]))
    assert abs(m["MAP"] - (1 + 1 / 3) / 2) < 1e-12 and m["MAP@1"] == 0.5
    # GAP@1: top-1 predictions 0.9 (right) and 0.8 (wrong) -> AP 1.0
    assert m["GAP@1"] == 1.0
