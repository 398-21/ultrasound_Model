"""Run-time matching decisions, the validation metrics, and the P x K sampler."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epills.core.inference.matching import decide  # noqa: E402


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


GALLERY = {"metformin": np.stack([unit([1, 0, 0]), unit([0.9, 0.1, 0])]),
           "aspirin": np.stack([unit([0, 1, 0])])}


def test_match_unknown_and_ambiguous():
    queries = np.stack([unit([1, 0.05, 0]), unit([0, 0, 1]), unit([1, 1, 0])])
    match, unknown, ambiguous = decide(queries, GALLERY, threshold=0.5, margin=0.1)
    assert (match.decision, match.medication_id) == ("match", "metformin")
    assert unknown.decision == "unknown"            # far from every enrolled medication
    assert ambiguous.decision == "ambiguous"        # halfway between the two: flip and retake


def test_evaluate_on_separable_embeddings():
    from epills.training.eval.retrieval import auroc, evaluate
    rng = np.random.default_rng(0)
    centres = np.eye(8, dtype=np.float32)
    labels = np.repeat(np.arange(6), 6)
    known = centres[labels] + 0.05 * rng.normal(size=(len(labels), 8)).astype(np.float32)
    unknown = centres[[6, 7] * 10] + 0.05 * rng.normal(size=(20, 8)).astype(np.float32)
    m = evaluate(known, labels, unknown, k_enrol=3, gallery_size=4, n_galleries=20)
    assert m["top1_all"] == 1.0 and m["top1_gallery10"] == 1.0 and m["auroc_unknown"] > 0.99
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0


def test_pk_sampler_batches():
    pytest.importorskip("torch")
    from epills.training.datasets.pills import PKSampler
    labels = [0] * 10 + [1] * 2 + [2] * 6 + [3] * 8
    sampler = PKSampler(labels, p=3, k=4, seed=1)
    for batch in sampler:
        assert len(batch) == 12
        counts = np.bincount([labels[i] for i in batch])
        assert sorted(c for c in counts if c) == [4, 4, 4]  # 3 labels x 4 images, small labels with replacement
