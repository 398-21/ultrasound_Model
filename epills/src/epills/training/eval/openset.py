"""
Open-set simulation of patients, shared by calibration (P5, validation) and evaluation (P6, test sets).

A patient has a gallery of G medications, each enrolled with k photos; that medication's other photos are queries,
and photos of medications outside the gallery play pills that aren't the patient's. Per query: s1 = cosine
similarity to the closest gallery medication, s2 = to the runner-up. The app's decision
(epills.core.inference.matching.decide):
  s1 < threshold          -> "unknown"   (set the pill aside; ask a pharmacist)
  s1 - s2 < margin        -> "ambiguous" (flip the pill and retake the photo)
  otherwise               -> "match"
"""
import numpy as np
import pandas as pd

from epills.core.inference.matching import normalise


def simulate(known_emb, known_lab, unknown_emb, gallery_size, k_enrol, n_galleries, unknown_per_gallery, seed,
             unknown_lab=None, enrol_mask=None):
    """
    Per query: gallery, group (the query's true medication), s1, s2 and, for known queries, whether the best match
    is right. Returns (known df, unknown df).
      unknown_emb  photos of medications that are never enrolled (validation); None: each gallery's unknown pills
                   are photos of the medications outside that gallery (test sets with one pool of medications)
      unknown_lab  labels of unknown_emb, for bootstraps by medication
      enrol_mask   photos allowed as enrolment photos (e.g. single-pill photos); default all
    """
    rng = np.random.default_rng(seed)
    known_emb = normalise(known_emb)
    from_rest = unknown_emb is None
    if from_rest:
        unknown_emb, unknown_lab = known_emb, known_lab
    else:
        unknown_emb = normalise(unknown_emb)
        if unknown_lab is None:
            unknown_lab = np.full(len(unknown_emb), -1)
    if enrol_mask is None:
        enrol_mask = np.ones(len(known_lab), bool)
    by_label = {}
    for lab in np.unique(known_lab):
        idx = np.flatnonzero(known_lab == lab)
        can_enrol = idx[enrol_mask[idx]]
        if len(can_enrol) >= k_enrol and len(idx) > k_enrol:
            by_label[lab] = (idx, can_enrol)
    eligible = np.array(list(by_label))
    known_rows, unknown_rows = [], []
    for g in range(n_galleries):
        labels = rng.choice(eligible, size=gallery_size, replace=False)
        protos, queries, q_lab = [], [], []
        for j, lab in enumerate(labels):
            idx, can_enrol = by_label[lab]
            enrol = rng.choice(can_enrol, size=k_enrol, replace=False)
            protos.append(normalise(known_emb[enrol].mean(axis=0)))
            rest = np.setdiff1d(idx, enrol)
            queries.append(rest)
            q_lab.append(np.full(len(rest), j))
        protos = np.stack(protos)
        queries, q_lab = np.concatenate(queries), np.concatenate(q_lab)
        pool = np.flatnonzero(~np.isin(unknown_lab, labels)) if from_rest else np.arange(len(unknown_emb))
        u = rng.choice(pool, size=min(unknown_per_gallery, len(pool)), replace=False)
        for emb, rows, lab, group in ((known_emb[queries], known_rows, q_lab, known_lab[queries]),
                                      (unknown_emb[u], unknown_rows, None, unknown_lab[u])):
            sims = emb @ protos.T
            order = np.argsort(-sims, axis=1)
            s1 = sims[np.arange(len(sims)), order[:, 0]]
            s2 = sims[np.arange(len(sims)), order[:, 1]]
            df = pd.DataFrame({"gallery": g, "group": group, "s1": s1, "s2": s2})
            if lab is not None:
                df["correct"] = order[:, 0] == lab
            rows.append(df)
    return pd.concat(known_rows, ignore_index=True), pd.concat(unknown_rows, ignore_index=True)


def operating_point(known, unknown, threshold, margin):
    """Shares of the patient's own pills (known) and of other pills (unknown) per decision."""
    k_margin, u_margin = known.s1 - known.s2, unknown.s1 - unknown.s2
    k_pass, u_pass = known.s1 >= threshold, unknown.s1 >= threshold
    k_match, u_match = k_pass & (k_margin >= margin), u_pass & (u_margin >= margin)
    return {
        "correct_match": float((k_match & known.correct).mean()),
        "wrong_match": float((k_match & ~known.correct).mean()),
        "own_flip": float((k_pass & (k_margin < margin)).mean()),
        "own_rejected": float((~k_pass).mean()),
        "unknown_rejected": float((~u_pass).mean()),
        "unknown_flip": float((u_pass & (u_margin < margin)).mean()),
        "unknown_matched": float(u_match.mean()),
    }


def closed_set_top1(emb, lab, k_enrol, seed=0, enrol_mask=None):
    """Top-1 accuracy with every medication enrolled at once (k photos each, from enrol_mask), the rest as queries."""
    rng = np.random.default_rng(seed)
    emb = normalise(emb)
    if enrol_mask is None:
        enrol_mask = np.ones(len(lab), bool)
    labels, protos, queries, q_lab = [], [], [], []
    for lab_ in np.unique(lab):
        idx = np.flatnonzero(lab == lab_)
        can_enrol = idx[enrol_mask[idx]]
        if len(can_enrol) < k_enrol or len(idx) <= k_enrol:
            continue
        enrol = rng.choice(can_enrol, size=k_enrol, replace=False)
        protos.append(normalise(emb[enrol].mean(axis=0)))
        rest = np.setdiff1d(idx, enrol)
        queries.append(rest)
        q_lab.append(np.full(len(rest), len(labels)))
        labels.append(lab_)
    queries, q_lab = np.concatenate(queries), np.concatenate(q_lab)
    pred = (emb[queries] @ np.stack(protos).T).argmax(axis=1)
    return {"top1": float((pred == q_lab).mean()), "medications": len(labels), "queries": int(len(queries)),
            "correct_per_query": pred == q_lab, "query_index": queries}


def bootstrap(known, unknown, threshold, margin, n=1000, seed=0):
    """95% intervals for operating_point by resampling medications (groups) with replacement: photos of one
    medication are correlated, so the medication, not the photo, is the unit."""
    rng = np.random.default_rng(seed)
    k_groups = {g: idx for g, idx in known.groupby("group").indices.items()}
    u_groups = {g: idx for g, idx in unknown.groupby("group").indices.items()}
    k_keys, u_keys = np.array(list(k_groups)), np.array(list(u_groups))
    draws = []
    for _ in range(n):
        k_idx = np.concatenate([k_groups[g] for g in rng.choice(k_keys, len(k_keys))])
        u_idx = np.concatenate([u_groups[g] for g in rng.choice(u_keys, len(u_keys))])
        draws.append(operating_point(known.iloc[k_idx], unknown.iloc[u_idx], threshold, margin))
    draws = pd.DataFrame(draws)
    return {k: [float(draws[k].quantile(0.025)), float(draws[k].quantile(0.975))] for k in draws}
