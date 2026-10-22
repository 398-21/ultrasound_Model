"""
PharmaScan: is this my pill? A 10-minute execution-explanation lecture (lectrace).

    .venv\\Scripts\\lectrace serve 01_pharmascan.py      (from this folder; opens http://localhost:7000)

Everything after the two image backbones runs here, in numpy, with the served model's own weights (run 4,
stage2_run4@2026-10-09): capsule fusion, transformer refinement, the embedding, gallery matching, imprint re-ranking
and the confidence. _precompute.py exported the weights and the backbones' outputs; the lecture checks its numpy
forward pass against PyTorch.
"""
import numpy as np
from lectrace import image, note, plot, table, text

from _helpers import (batch_norm, compact, dense, encoder_layer, imprint_scores, load, outcome_chart, relu,
                      similarity_chart, softmax, squash, top_rows, unit)


def main():
    demo, w, back, catalogue = load()  # @hide
    items = demo["items"]  # @hide

    # ---------------------------------------------------------------------------------------------- intro (2-3 min)
    text("# PharmaScan: is this my pill?")
    text("**The problem.** Older patients often take 5–10 medicines. In Singapore, pharmacies and polyclinics "
         "dispense tablets loose in a bag with a printed sticker; once out of the bag, many pills look alike. Taking "
         "the wrong pill is a **medication error**, a leading cause of avoidable harm (WHO, *Medication Without "
         "Harm*, 2017).")
    text("**What we want.** The patient photographs the pill before taking it. The app says *“this looks like your "
         "metformin”*, or *“this is not one of your medicines”*, or *“turn it over and photograph it again”*. "
         "An incorrect accepted name and a request for further checking have different consequences. We measure "
         "**accepted errors and abstention** separately.")
    text("**Why this is hard for a classifier.** Every patient has a different list, and new medicines appear all "
         "the time, so we cannot train one class per medicine. Instead the network learns an **embedding**: pills "
         "of the same medicine land close together, and a new medicine needs only 3 enrolment photos, no retraining. "
         "This is open-set recognition.")
    image("images/fig01_system_overview.png", width=760)
    note("~1 min so far. The system: dose photo → pill finder → embedding → match against the patient's own "
         "medicines, or identify against a 4,936-product US catalogue; an OCR imprint reader checks both.")
    text("**Our model.** We reused our earlier facial-expression network: VGG16 with CBAM attention and a "
         "Vision Transformer in parallel, fused by a capsule block, refined by a small transformer, and trained "
         "with ArcFace on 29,471 NLM C3PI photos of 1,710 pill groups. Thresholds are set on validation medicines "
         "only. In the strict C3PI evaluation, held-out test medicines were never seen in training. "
         "The separate official ePillID protocol permits test-type reference images during training.")
    image("images/fig02_architecture.png", width=760)
    note("~2.5 min. Now four execution states, following two real photos through the served model: an unseen photo "
         "of the patient's own metformin, and a pill from someone else's box.")

    # ---------------------------------------------------------------------------------------- state 1: photo -> two views
    text("## State 1 — A photo becomes a crop and two feature vectors")
    dose = items[12]  # @hide
    image(dose["photo"], width=420)
    pills_found = dose["pills_found"]  # @inspect pills_found
    box = dose["box"]  # @inspect box
    text(f"The stand-in pill finder finds **{pills_found} pill**. The crop is a square **{demo['window']}×** the "
         "box's long side (the framing chosen on validation), resized to 224×224.")
    cnn = back["vgg"][12]  # @hide
    vit = back["vit"][12]  # @hide
    cnn_first_8 = np.round(cnn[:8], 3)  # @inspect cnn_first_8
    vit_first_8 = np.round(vit[:8], 3)  # @inspect vit_first_8
    text(f"The two backbones (≈100M parameters, precomputed) see the crop differently: VGG16 + CBAM gives "
         f"**{cnn.size} numbers** (local texture, colour, edges), and ViT-B/16 gives **{vit.size}** (global layout, "
         "via self-attention between 16×16 patches). The panel shows the first 8 of each.")
    image(dose["crop"], width=200)
    image(dose["vit_attention"], width=200)
    image(dose["cnn_attention"], width=200)
    cnn_gate = np.round(back["cnn_gate"][12], 2)  # @inspect cnn_gate
    note("Left: the crop. Middle: where the ViT's [CLS] token looks in its last layer (blue = more). Right: CBAM's "
         "7×7 spatial gate. The gate is ≈1 over the pill and most of the frame and switches off one corner, so the "
         "CNN branch pools almost the whole crop. The ViT spreads attention over a few patches, some on background "
         "(a known ViT artefact). Don't over-read attention maps; the next state shows how the two views are combined.")

    # ------------------------------------------------------------------------------------ state 2: fusing the two views
    text("## State 2 — Capsule routing decides how much to trust each view")  # @clear cnn_first_8 vit_first_8 cnn_gate pills_found box
    text("Each branch becomes a 256-number **capsule**. Dynamic routing starts undecided (50/50) and moves weight "
         "toward the branch that **agrees** with the combined result. Step into `capsule_fusion` to watch the "
         "coupling coefficients `c` change.")
    fused, couplings = capsule_fusion(cnn, vit, w)  # @inspect couplings
    text("Then a small transformer splits the fused vector into 4 tokens of 64, lets them attend to each other "
         "(2 layers × 4 heads), averages them, and the head maps the result to the final **256-number unit vector**.")
    z, attention = refine_and_embed(fused, w)  # @stepover @inspect attention
    length = round(float(np.linalg.norm(z)), 4)  # @inspect length
    same_as_pytorch = round(float(z @ unit(back["torch_embedding"][12])), 6)  # @inspect same_as_pytorch
    note("couplings: routing moves from 50/50 to about 9% CNN / 91% ViT for this pill, so the fused vector is "
         "mostly the ViT's view. attention: the refinement block's first layer attends uniformly (0.25 everywhere) "
         "and the second almost so. This describes the attention weights for this example; it does not establish "
         "whether the block is a bottleneck or improves recognition. That requires a controlled ablation. same_as_pytorch is the cosine between this numpy forward pass and the served "
         "PyTorch model on the same photo: 1.0 means this lecture really runs the deployed model's weights.")

    # -------------------------------------------------------------------- state 3: the patient's gallery, open-set match
    text("## State 3 — Is it one of *your* medicines?")  # @clear couplings attention length same_as_pytorch
    text("The patient enrolled 4 medicines (C3PI validation medicines, never trained on) from **3 photos each**. "
         "A medicine's **prototype** is the normalised mean of its 3 embeddings.")
    names = ["metformin ER 750 mg", "lisinopril 5 mg", "amlodipine 10 mg", "simvastatin"]  # @hide
    gallery_images = [items[i]["crop"] for i in (0, 3, 6, 9)]  # @hide
    for path in gallery_images:  # @hide
        image(path, width=110)  # @hide
    prototypes = enrol(back, w)  # @stepover
    threshold, margin = demo["threshold"], demo["margin"]  # @inspect threshold margin
    text(f"Decision rule, set on validation: below **τ = {threshold:.3f}** → *unknown* (95% of other medicines are "
         f"rejected); top two closer than **m = {margin}** → *photograph again*; otherwise → the match.")
    decision, sims = decide(z, prototypes, names, threshold, margin)  # @inspect decision
    foreign = items[13]  # @hide
    image(foreign["crop"], width=150)
    text("Now a pill from someone else's box (it is escitalopram 5 mg, an antidepressant; the model doesn't know "
         "that).")
    z2, _ = refine_and_embed(capsule_fusion(back["vgg"][13], back["vit"][13], w)[0], w)  # @stepover
    decision2, sims2 = decide(z2, prototypes, names, threshold, margin)  # @inspect decision2
    rows = [{"dose": d, "medicine": n, "similarity": round(float(s), 3), "passes": bool(s >= threshold)}  # @hide
            for d, ss in (("their metformin", sims), ("someone else's pill", sims2)) for n, s in zip(names, ss)]  # @hide
    plot(similarity_chart(rows, threshold))
    note("Why it matters: the foreign pill is still *closest* to one of the patient's medicines, but that closest "
         "similarity is below the calibrated threshold, so the app says “set it aside” instead of naming it. "
         "A plain classifier would have to pick one of the four.")

    # --------------------------------------------------------------- state 4: identify with the imprint, honest confidence
    text("## State 4 — What is this unknown pill? Look first, then read the imprint")  # @clear threshold margin decision
    look = catalogue.look(z2)  # @stepover
    look_top = top_rows(look, None, catalogue, 5)  # @hide
    table(look_top, caption="By look alone: the 5 nearest catalogue products")
    reads = foreign["reads"]  # @inspect reads
    text("Florence-2 OCR reads the crop at 4 rotations. Each catalogue imprint token has a weight: rare tokens "
         "('EC5') count a lot, common ones ('93', 'TV') little. The top 100 products by look get "
         f"**+ α × imprint score**, α = {demo['imprint_alpha']} (chosen on validation).")
    score = rerank(look, reads, catalogue, demo["imprint_alpha"], demo["imprint_top_k"])
    both_top = top_rows(look, score, catalogue, 5)  # @hide
    table(both_top, caption="After imprint re-ranking")
    verdict = confidence(look, score, reads, catalogue, demo["confidence"])  # @inspect verdict
    note("Why it matters: the look alone puts the right product 4th among near-identical small white tablets; the "
         "imprint puts it 1st. The confidence is a logistic model fitted on validation, because raw similarities "
         "aren't probabilities (pills not in the catalogue still score ≈0.93). The answer is still worded as a "
         "*possible* match: the patient or pharmacist confirms.")

    # -------------------------------------------------------------------------------------- key result (≈1.5 min)
    text("## Key result and what it means for patients")
    plot(outcome_chart([
        {"set": "Own pills, C3PI-style photos", "outcome": "named correctly", "percent": 78.3, "k": 0},
        {"set": "Own pills, C3PI-style photos", "outcome": "photograph again", "percent": 7.2, "k": 1},
        {"set": "Own pills, C3PI-style photos", "outcome": "called unknown", "percent": 14.5, "k": 2},
        {"set": "Own pills, C3PI-style photos", "outcome": "named wrongly", "percent": 0.06, "k": 3},
        {"set": "Own pills, other phones (MEDISEG)", "outcome": "named correctly", "percent": 18.7, "k": 0},
        {"set": "Own pills, other phones (MEDISEG)", "outcome": "photograph again", "percent": 10.2, "k": 1},
        {"set": "Own pills, other phones (MEDISEG)", "outcome": "called unknown", "percent": 70.4, "k": 2},
        {"set": "Own pills, other phones (MEDISEG)", "outcome": "named wrongly", "percent": 0.7, "k": 3},
    ]))
    text("- **200 test medicines never seen in training**, 10 enrolled per patient from 3 photos: **78.3% named "
         "correctly, 0.06% named wrongly** among enrolled queries. Among foreign queries, **94.8% are rejected "
         "and 3.5% are incorrectly accepted**. These rates have different denominators.\n"
         "- Identifying an unknown pill among 4,936 products: right product first **80.2%** (71.6% by look alone), "
         "in the top 5 **94.4%**.\n"
         "- **Healthcare reading:** on MEDISEG, 70.4% of enrolled queries are called unknown and 0.7% receive "
         "an incorrect accepted name. Rejection creates a substantial checking burden, and residual errors remain. "
         "These public-image experiments do not establish clinical safety. Local enrolment and calibration "
         "should be evaluated under the intended acquisition conditions.\n"
         "- Visual evidence only: strength comes from the pharmacy label, and the patient confirms.")
    note("Sources: epills/outputs/runs/stage2_run4/eval (imprint_gallery_large_c92_distinct.json, mediseg.json, "
         "identify_topk_large_c92.json). Live API: https://davidpansx-pharmascan-api.hf.space/docs")


# ==== the steps shown in the trace ===================================================================================
def capsule_fusion(cnn, vit, w):
    u = np.stack([branch_capsule(cnn, w, 0), branch_capsule(vit, w, 1)])  # (2 branches, 256)
    b = np.zeros(2)  # routing logits: undecided
    couplings = []
    v_prev = None
    for i in range(2):  # 2 routing iterations, as trained
        c = softmax(b)  # coupling coefficients: [CNN share, ViT share]
        couplings.append(np.round(c, 3))
        v = squash((c[:, None] * u).sum(axis=0))  # the fused capsule
        if i > 0:
            v = 0.5 * v + 0.5 * v_prev  # residual update
        v_prev = v
        agreement = u @ v  # how well each branch agrees with the fused capsule
        b = b + agreement
    return v, couplings


def branch_capsule(x, w, i):
    h = relu(dense(w, f"fusion.mlp{i}.0", x))
    return batch_norm(w, f"fusion.bn{i}", dense(w, f"fusion.mlp{i}.1", h))


def refine_and_embed(fused, w):
    tokens = dense(w, "refine.proj", fused).reshape(4, 64) + w["refine.pos"]
    attention = []
    for i in range(2):
        tokens, attn = encoder_layer(w, i, tokens)
        attention.append(np.round(attn.mean(axis=0), 2))  # 4x4 token attention, averaged over heads
    z = dense(w, "head", tokens.mean(axis=0))
    return unit(z), attention


def enrol(back, w):
    prototypes = []
    for med in range(4):
        embeddings = [refine_and_embed(capsule_fusion(back["vgg"][k], back["vit"][k], w)[0], w)[0]
                      for k in range(3 * med, 3 * med + 3)]
        prototypes.append(unit(np.mean(embeddings, axis=0)))
    return np.stack(prototypes)


def decide(z, prototypes, names, threshold, margin):
    sims = prototypes @ z  # cosine similarity to each enrolled medicine
    order = np.argsort(-sims)
    best, s1, s2 = names[order[0]], float(sims[order[0]]), float(sims[order[1]])
    if s1 < threshold:
        decision = f"unknown: not one of your medicines (closest: {best}, {s1:.3f} < {threshold:.3f})"
    elif s1 - s2 < margin:
        decision = f"photograph it again (top two only {s1 - s2:.3f} apart)"
    else:
        decision = f"looks like your {best} ({s1:.3f})"
    return decision, sims


def rerank(look, reads, catalogue, alpha, top_k):
    texts = [compact(r) for r in reads if compact(r)]  # 'EC5\n' -> 'EC5'
    top = np.argsort(-look)[:top_k]  # the 100 nearest products by look
    bonus = imprint_scores(texts, catalogue, top)  # per product: weight of its rarest token found in the reads
    nearest_five = {catalogue[j]["imprint"]: round(float(b), 3) for j, b in zip(top[:5], bonus[:5])}
    score = look.copy()
    score[top] += alpha * bonus
    return score


def confidence(look, score, reads, catalogue, cfg):
    order = np.argsort(-score)
    s1, s2 = float(look[order[0]]), float(look[order[1]])  # look similarities of the top two after re-ranking
    imprint_read = float(imprint_scores([compact(r) for r in reads], catalogue, order[:1])[0])
    w0, w1, w2, w3 = cfg["weights"]
    p = 1 / (1 + np.exp(-(w0 + w1 * s1 + w2 * (s1 - s2) + w3 * imprint_read)))
    fit = "poor" if s1 < cfg["t_poor"] else "good" if s1 >= cfg["t_good"] else "weak"
    if fit == "poor" or p < cfg["low"]:
        verdict = "no confident match"
    elif fit == "good" and p >= cfg["high"]:
        verdict = "possible match"
    else:
        verdict = "uncertain"
    return f"{verdict}: {catalogue[order[0]]['name']} (confidence {p:.2f})"



if __name__ == "__main__":
    main()
