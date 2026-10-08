"""The patient simulation behind calibration (P5) and evaluation (P6), on synthetic embeddings."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from epills.training.eval.openset import closed_set_top1, operating_point, simulate  # noqa: E402


def clusters(n_labels=20, per_label=6, dim=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_labels, dim))
    lab = np.repeat(np.arange(n_labels), per_label)
    return centres[lab] + noise * rng.normal(size=(len(lab), dim)), lab


def test_unknown_pills_come_from_outside_the_gallery():
    emb, lab = clusters()
    known, unknown = simulate(emb, lab, None, gallery_size=5, k_enrol=3, n_galleries=50, unknown_per_gallery=30,
                              seed=1)
    for g in range(50):
        assert not set(known[known.gallery == g].group) & set(unknown[unknown.gallery == g].group)


def test_enrolment_only_from_allowed_photos():
    emb, lab = clusters(per_label=4)
    allowed = np.tile([True, True, True, False], 20)  # the 4th photo of each medication can't be enrolled
    known, _ = simulate(emb, lab, None, 5, 3, 20, 10, seed=2, enrol_mask=allowed)
    assert (known.groupby("gallery").size() == 5).all()  # exactly the one non-enrollable photo per medication


def test_decisions_add_up_and_separable_data_is_perfect():
    emb, lab = clusters()
    known, unknown = simulate(emb, lab, None, 5, 3, 50, 30, seed=3)
    p = operating_point(known, unknown, threshold=0.9, margin=0.1)
    assert abs(p["correct_match"] + p["wrong_match"] + p["own_flip"] + p["own_rejected"] - 1) < 1e-9
    assert abs(p["unknown_rejected"] + p["unknown_flip"] + p["unknown_matched"] - 1) < 1e-9
    assert p["correct_match"] == 1.0 and p["unknown_rejected"] == 1.0
    assert closed_set_top1(emb, lab, 3)["top1"] == 1.0
