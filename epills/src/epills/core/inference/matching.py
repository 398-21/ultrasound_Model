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


def check_imprint(d, hits, imprinted, threshold, distinct_hits=None):
    """
    The imprint read on the pill checks a Decision. hits: medication ids whose recorded imprint was read on the pill
    (imprint.match.read_hits); imprinted: ids that have a recorded imprint; distinct_hits (optional): the same, counting
    only tokens that tell the gallery's medications apart (read_hits distinct=True).
      match, its own imprint read                     -> match      ("confirmed")
      match, only other medications' imprints read    -> ambiguous  ("conflict": photograph the pill again)
      ambiguous, exactly one of the two closest read  -> match on it ("resolved"; on the runner-up only if it passes
                                                         the threshold)
    With distinct_hits, a match is confirmed or contradicted by distinct tokens only: reading 'TV' on a pill that looks
    like the patient's 'TV;7295' no longer confirms it next to their 'TV;58', and reading '58' makes it a conflict
    (blueprint B27). It can only turn a match into "photograph again"; ties are settled as before.
    An imprint never turns an unknown pill into a match ("not_used"). Other statuses: "not_checked" (no recorded
    imprints in the gallery), "not_read" (none of them read), "unresolved" (the read doesn't tell them apart).
    Returns (Decision, status).
    """
    if not imprinted:
        return d, "not_checked"
    if d.decision == "unknown":
        return d, "not_used"
    if not hits:
        return d, "not_read"
    best, second = d.medication_id, d.runner_up_id
    if d.decision == "match":
        confirm = hits if distinct_hits is None else distinct_hits
        if best in confirm:
            return d, "confirmed"
        if best not in hits or confirm:  # another's imprint read (any token), or another's distinct token
            return Decision(best, d.similarity, second, d.runner_up_similarity, "ambiguous"), "conflict"
        return d, "unresolved"  # its own imprint read, but only a token the medications share
    if best in hits and second not in hits:
        return Decision(best, d.similarity, second, d.runner_up_similarity, "match"), "resolved"
    if second in hits and best not in hits and d.runner_up_similarity >= threshold:
        return Decision(second, d.runner_up_similarity, best, d.similarity, "match"), "resolved"
    return d, "unresolved"
