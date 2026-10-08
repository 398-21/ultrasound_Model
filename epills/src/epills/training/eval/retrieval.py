"""
Validation metrics for Stage 2, simulating enrolment on the validation splits:

  val_enrolled  each group's first k_enrol photos (seeded order) are its enrolment photos; the rest are queries
  val_unknown   never enrolled; every photo is an unknown-pill query

Metrics:
  top1_all          closed set: each query against all val_enrolled groups at once (~214; much harder than a patient)
  top1_gallery10    closed set: random patient-sized galleries of 10 medications (plan experiment E2)
  auroc_unknown     open set: known queries vs unknown queries, scored by the best gallery similarity
  threshold_95      the similarity threshold that rejects 95% of unknown queries (the plan's safety target)
  known_correct_95  share of known queries accepted and matched correctly at that threshold
  wrong_accept_95   share of known queries accepted but matched to the wrong medication at that threshold
  *_gallery10       the open-set metrics above in patient-sized galleries: unknown queries are scored against the same
                    10 medications as the known queries, and the threshold is set per gallery (what the app does)
  embedding_dims    effective number of dimensions the embeddings use (entropy of the variance spectrum); a value
                    near 1-5 means the embeddings have collapsed and can't separate many medications
The matching itself is epills.core.inference.matching, the same code the API runs.
"""
import numpy as np
import torch

from epills.core.inference.matching import gallery_prototypes, similarities


@torch.no_grad()
def embed(model, loader, device, amp=True):
    model.eval()
    out, labels = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            emb = model(batch["image"].to(device, non_blocking=True))["embedding"]
        out.append(emb.float().cpu())
        labels.append(batch["label"])
    return torch.cat(out).numpy(), torch.cat(labels).numpy()


def enrol_split(labels, k_enrol=3, seed=0):
    """Boolean mask of enrolment photos: k_enrol per label (at least one photo per label is left as a query)."""
    rng = np.random.default_rng(seed)
    enrol = np.zeros(len(labels), bool)
    for lab in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == lab))
        enrol[idx[:min(k_enrol, len(idx) - 1)]] = True
    return enrol


def auroc(pos, neg):
    """Area under the ROC curve: probability that a random positive scores above a random negative (ties half)."""
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def effective_dims(emb):
    """exp(entropy) of the share of variance per principal direction: 256 if all used equally, 1 if a single line."""
    x = emb - emb.mean(axis=0)
    var = np.linalg.svd(x, compute_uv=False) ** 2
    var = var[var > 0] / var.sum()
    return float(np.exp(-(var * np.log(var)).sum()))


def evaluate(known_emb, known_labels, unknown_emb, k_enrol=3, gallery_size=10, n_galleries=200, seed=0):
    enrol = enrol_split(known_labels, k_enrol, seed)
    gallery = {lab: known_emb[enrol & (known_labels == lab)] for lab in np.unique(known_labels)}
    ids, protos = gallery_prototypes(gallery)
    ids = np.array(ids)
    q_emb, q_lab = known_emb[~enrol], known_labels[~enrol]

    sims = similarities(q_emb, protos)
    best = sims.argmax(axis=1)
    correct = ids[best] == q_lab
    known_score = sims.max(axis=1)
    unknown_sims = similarities(unknown_emb, protos)
    unknown_score = unknown_sims.max(axis=1)

    rng = np.random.default_rng(seed + 1)
    hits, g_auroc, g_correct, g_wrong, g_threshold = [], [], [], [], []
    for _ in range(n_galleries):
        chosen = rng.choice(len(ids), size=min(gallery_size, len(ids)), replace=False)
        q = np.isin(q_lab, ids[chosen])
        g_sims = sims[q][:, chosen]
        g_hit = ids[chosen][g_sims.argmax(axis=1)] == q_lab[q]
        hits.append(g_hit)
        g_known, g_unknown = g_sims.max(axis=1), unknown_sims[:, chosen].max(axis=1)
        t = float(np.quantile(g_unknown, 0.95))
        g_auroc.append(auroc(g_known, g_unknown))
        g_correct.append(((g_known >= t) & g_hit).mean())
        g_wrong.append(((g_known >= t) & ~g_hit).mean())
        g_threshold.append(t)
    threshold = float(np.quantile(unknown_score, 0.95))
    accepted = known_score >= threshold
    return {
        "top1_all": float(correct.mean()),
        "top1_gallery10": float(np.concatenate(hits).mean()),
        "auroc_unknown": auroc(known_score, unknown_score),
        "threshold_95": threshold,
        "known_correct_95": float((accepted & correct).mean()),
        "wrong_accept_95": float((accepted & ~correct).mean()),
        "auroc_gallery10": float(np.mean(g_auroc)),
        "threshold_95_gallery10": float(np.mean(g_threshold)),
        "known_correct_95_gallery10": float(np.mean(g_correct)),
        "wrong_accept_95_gallery10": float(np.mean(g_wrong)),
        "embedding_dims": effective_dims(known_emb),
        "n_gallery": int(len(ids)), "n_known_queries": int(len(q_lab)), "n_unknown_queries": int(len(unknown_score)),
    }
