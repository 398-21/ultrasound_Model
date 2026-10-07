"""
Gallery matching with unknown-pill rejection: the run-time decision, shared by the API and the evaluation code.

A patient's gallery holds, per medication, the embeddings of its enrolment photos. Each medication is represented by
the normalised mean of its embeddings (the plan: "each entry averages the enrolled images"). A pill is assigned to
the medication with the highest cosine similarity, unless
  - that similarity is below the rejection threshold          -> "unknown" (don't take it; ask a pharmacist), or
  - the runner-up is within the ambiguity margin of the best  -> "ambiguous" (flip the pill and retake the photo).
The threshold and margin are chosen on validation data (training/eval), never on test data.
"""
from dataclasses import dataclass

import numpy as np


def normalise(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def gallery_prototypes(embeddings_by_id):
    """{medication id: (n, d) embeddings} -> (ids, (m, d) unit prototypes)."""
    ids = list(embeddings_by_id)
    protos = np.stack([normalise(normalise(embeddings_by_id[i]).mean(axis=0)) for i in ids])
    return ids, protos


def similarities(queries, prototypes):
    """Cosine similarities (q, m) between unit query embeddings and unit prototypes."""
    return normalise(queries) @ prototypes.T


@dataclass
class Decision:
    medication_id: object  # best match (also given for unknown / ambiguous, for display)
    similarity: float
    runner_up_id: object
    runner_up_similarity: float
    decision: str  # "match", "unknown" or "ambiguous"


def decide(queries, embeddings_by_id, threshold, margin=0.0):
    """One Decision per query embedding."""
    ids, protos = gallery_prototypes(embeddings_by_id)
    sims = similarities(queries, protos)
    order = np.argsort(-sims, axis=1)
    out = []
    for row, idx in zip(sims, order):
        best = idx[0]
        second = idx[1] if len(idx) > 1 else None
        s1 = float(row[best])
        s2 = float(row[second]) if second is not None else -1.0
        if s1 < threshold:
            verdict = "unknown"
        elif second is not None and s1 - s2 < margin:
            verdict = "ambiguous"
        else:
            verdict = "match"
        out.append(Decision(ids[best], s1, ids[second] if second is not None else None, s2, verdict))
    return out
