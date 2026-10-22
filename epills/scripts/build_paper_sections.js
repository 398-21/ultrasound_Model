// Builds epills/docs/paper/PharmaScan_approach_experiments.docx: the Approach and Experiments sections, the same text,
// figures and tables as docs/paper/latex/approach.tex and experiments.tex, for a teammate to transfer into LaTeX.
// Run: node scripts/build_paper_sections.js (needs: npm install docx@9.5.1)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, Footer, PageNumber,
} = require("docx");

const PAPER = "C:/Users/sheng/Documents/GitHub/ultrasound_Model/epills/docs/paper";
const FIG = path.join(PAPER, "figures");
const W = 9026, FONT = "Times New Roman", MONO = "Consolas", INK2 = "52514E";
// numbering as LaTeX produces it (float order in approach.tex then experiments.tex)
const NUM = Object.fromEntries([
  ...["fig:arch", "fig:calib", "fig:finder", "fig:main", "fig:openset", "fig:training", "fig:identify", "fig:robust",
    "fig:confidence", "fig:epillid", "fig:imprint", "fig:lookalike", "fig:sim", "fig:tsne"].map((l, i) => [l, i + 1]),
  ...["tab:arch", "tab:train", "tab:main", "tab:ablation", "tab:epillid", "tab:negative"].map((l, i) => [l, i + 1])]);

// inline: @fig:x / @tab:x -> number, **bold**, *italic*, `code`, [cite:key], [[TO DO]]
function runs(text, base = {}) {
  text = text.replace(/@((?:fig|tab):[a-z]+)/g, (_, l) => { if (!NUM[l]) throw new Error(l); return String(NUM[l]); });
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[cite:[^\]]+\]|\[\[[^\]]+\]\])/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("*")) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else if (part.startsWith("`")) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: MONO, size: 18 }));
    else if (part.startsWith("[cite:")) out.push(new TextRun({ ...base, text: "[" + part.slice(6, -1) + "]",
      color: "1D5FAE" }));
    else if (part.startsWith("[[")) out.push(new TextRun({ ...base, text: "[" + part.slice(2, -2) + "]", bold: true,
      color: "9A3412", shading: { type: ShadingType.CLEAR, fill: "FDE9DF", color: "auto" } }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = t => new Paragraph({ children: runs(t), spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED });
const PARA = (title, t) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: title + " ", bold: true }), ...runs(t)] });
const LABEL = l => new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: `\\label{${l}}`, font: MONO,
  size: 16, color: "8A8984" })] });
const H1 = (t, label, brk = true) => [new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: brk,
  children: [new TextRun(t)] }), LABEL(label)];
const H2 = (t, label) => [new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] }), LABEL(label)];
const CODE = lines => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l || " ", font: MONO, size: 17 })],
  shading: { type: ShadingType.CLEAR, fill: "F4F3F0", color: "auto" },
  spacing: { after: i === lines.length - 1 ? 160 : 0 }, indent: { left: 120, right: 120 } }));
const EQ = (num, readable, latex, label) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 60 }, children: [
    new TextRun({ text: readable, font: "Cambria Math", size: 22 }), new TextRun({ text: `\t(${num})`, size: 20 })] }),
  ...CODE([`% equation (${num}), \\label{${label}}`, ...latex])];
const NOTE = (t, fill = "FDF3E9", edge = "EB6834") => new Paragraph({
  children: runs(t), spacing: { before: 80, after: 160 }, indent: { left: 160, right: 160 },
  shading: { type: ShadingType.CLEAR, fill, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: edge, space: 6 } } });

// booktabs-like tables: rules above and below the header and at the bottom, no vertical lines
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
function table(label, caption, headers, rows, widths, rules = []) {
  const total = widths.reduce((a, b) => a + b, 0);
  widths = widths.map(w => Math.round(w * W / total));
  widths[widths.length - 1] = W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const cell = (t, i, r, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { left: NONE, right: NONE,
      top: head ? RULE(12) : (r === 0 ? RULE(6) : rules.includes(r) ? RULE(4) : NONE),
      bottom: r === rows.length - 1 ? RULE(12) : NONE },
    margins: { top: 30, bottom: 30, left: 70, right: 70 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { size: 18, bold: head || undefined }) })] });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 60 }, alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: `Table ${NUM[label]}: `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, -1, true)) }),
        ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, ri, false)) }))] }),
    LABEL(label)];
}
function figure(name, label, caption, widthIn = 6.0) {
  const buf = fs.readFileSync(path.join(FIG, name + ".png"));
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20);
  const w = widthIn * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 }, children: [
      new ImageRun({ type: "png", data: buf, transformation: { width: w, height: w * ph / pw },
        altText: { title: name, description: caption, name } })] }),
    new Paragraph({ spacing: { before: 60, after: 20 }, alignment: AlignmentType.JUSTIFIED, children: [
      new TextRun({ text: `Figure ${NUM[label]}: `, bold: true, size: 19 }), ...runs(caption, { size: 19 })] }),
    new Paragraph({ spacing: { after: 220 }, children: [new TextRun({
      text: `\\includegraphics[width=${widthIn >= 5.9 ? "\\linewidth" : (widthIn / 5.5).toFixed(2) + "\\linewidth"}]{${name}.pdf}   \\label{${label}}`,
      font: MONO, size: 16, color: "8A8984" })] })];
}

// ======================================================================================================================
const body = [];
const add = (...xs) => xs.forEach(x => Array.isArray(x) ? body.push(...x) : body.push(x));

add(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "PharmaScan — Approach and Experiments",
    size: 40, bold: true })] }),
  P("The two sections of the NeurIPS paper, ready to transfer into LaTeX. The same text already exists as LaTeX in `epills/docs/paper/latex/` (approach.tex, experiments.tex), which compiles in the NeurIPS 2025 style (preview: preview_sections.pdf); pasting those files is the quickest route."),
  NOTE("**Conventions.** Citations appear as [bibkey] and become `\\citep{bibkey}` (keys in `epills/docs/paper/references.bib`). Figure and table numbers are those LaTeX produces; the grey line under each figure, table and heading gives its `\\label` and file (`figures/*.pdf`), so references become `Figure~\\ref{...}`. Equations come with their LaTeX source. Every figure and table sits next to the text that cites it. All numbers come from the result files in `epills/outputs/runs/`.", "EEF4FC", "2A78D6"));

// ---------------------------------------------------------------------------------------------------- APPROACH
add(H1("Approach", "sec:approach", false),
  P("A detected pill is cropped, embedded by a hybrid CNN–ViT network, and compared by cosine similarity with the patient's enrolled medicines or with a product catalogue. Decision thresholds are calibrated on validation medicines only, and an OCR imprint reader adds a second, independent check."),
  H2("Network", "sec:arch"),
  P("Figure @fig:arch shows our network and Table @tab:arch its layers. A VGG16 branch [cite:simonyan2015vgg] with CBAM channel and spatial attention [cite:woo2018cbam] and a ViT-B/16 branch [cite:dosovitskiy2021vit] see the same 224×224 crop: the CNN contributes local texture, colour and edges, the ViT global layout through self-attention between patches. A capsule block [cite:sabour2017capsules] turns each branch into a 256-d capsule and fuses them by dynamic routing: the coupling coefficients start equal and move towards the branch that agrees with the fused vector. A small transformer encoder [cite:vaswani2017attention] splits the fused vector into four 64-d tokens and refines it; a linear head maps the mean token to the embedding z = f_θ(x), with ‖z‖ = 1 and 256 dimensions. Auxiliary heads predict the pill's shape and colours. All layers after the backbones use Glorot-uniform initialisation."),
  figure("fig02_architecture", "fig:arch", "The hybrid network. VGG16 with CBAM attention and ViT-B/16 see the same crop; a capsule block fuses their features by dynamic routing and a transformer block refines the result into a unit-length 256-d embedding trained with ArcFace, with auxiliary shape and colour heads. Numbers on the arrows are feature sizes."),
  table("tab:arch", "The network's layers (inference mode). Pretrained weights: VGG16 on ImageNet, ViT-B/16 on ImageNet-21k.",
    ["Block", "Details", "Output"], [
      ["CNN branch", "VGG16 conv. blocks → CBAM → global average pool → Dense, ReLU", "256"],
      ["ViT branch", "ViT-B/16 (`vit-base-patch16-224-in21k`), pooler output", "768"],
      ["Capsule fusion", "per branch: Dense 256, ReLU, dropout 0.4, Dense 256, batch norm; dynamic routing, 2 iterations, residual update", "256"],
      ["Refinement", "4 tokens × 64 + learned positions; 2 post-norm encoder layers (4 heads, FFN 128, dropout 0.4); mean pool", "64"],
      ["Embedding head", "Linear, ℓ2 normalisation", "256"],
      ["Auxiliary heads", "shape (softmax), colour (multi-label sigmoid)", "–"]], [1.8, 6, 1]),
  H2("Training", "sec:training"),
  PARA("Objective.", "The embedding is trained with ArcFace [cite:deng2019arcface] over the 1,710 training appearance groups (pill types that share imprint, shape, colour and size form one group), plus auxiliary losses:"),
  EQ(1, "L = −log [ e^{s·cos(θ_y + m)} / ( e^{s·cos(θ_y + m)} + Σ_{j≠y} e^{s·cos θ_j} ) ] + λ_s·L_shape + λ_c·L_colour", [
    "\\mathcal{L}=-\\log\\frac{e^{s\\cos(\\theta_y+m)}}{e^{s\\cos(\\theta_y+m)}+\\sum_{j\\neq y}e^{s\\cos\\theta_j}}",
    "  +\\lambda_s\\mathcal{L}_{\\text{shape}}+\\lambda_c\\mathcal{L}_{\\text{colour}},"], "eq:loss"),
  P("with s = 32, m = 0.3 and λ_s = λ_c = 0.2, where θ_j is the angle between z and the j-th class weight, L_shape is a cross-entropy and L_colour a binary cross-entropy (unknown labels ignored). The margin warms up over the first three epochs (0, 0.1, 0.2), then stays at 0.3."),
  PARA("Optimisation.", "Table @tab:train lists the settings. Training has two phases: in phase 1 the ViT and VGG blocks 1–3 are frozen; in phase 2 everything trains. Batches hold P groups × K = 4 photos each [cite:hermans2017triplet]. AdamW [cite:loshchilov2019adamw] uses three learning-rate groups: pretrained backbones, new layers, and the randomly initialised ArcFace class table. These rates were the decisive choice. Our first run trained the new layers at 10⁻³ and its embedding collapsed to about 3 effective dimensions (footnote: exp of the entropy of the share of variance per principal direction; 256 if all directions are used equally, 1 for a line), enough to encode colour and shape but not imprints. Lower rates (3×10⁻⁵, then 10⁻⁵), a faster rate for the class table and the margin warm-up fixed it (see Ablations, Figure @fig:training)."),
  PARA("Augmentation.", "Colour jitter is kept small (saturation 0.05, hue 0.0125) because colour identifies pills. To mimic detector error the crop window is jittered (×0.75–1.35, centre shift ≤ 10%). For other phones and lighting we add brightness and contrast (±0.3), white balance (±8%), downscaling to 40–160 px (p = 0.3), JPEG quality 30–90 (p = 0.3) and Gaussian noise (σ = 0.03, p = 0.2), and paste a neighbouring pill partly into the frame (p = 0.2)."),
  table("tab:train", "Training settings of the served model (run 4). Selection: top-1 on validation-enrolled medicines (best: phase 2, epoch 35). Seed 2026.",
    ["Setting", "Phase 1", "Phase 2"], [
      ["Trainable", "all but ViT, VGG blocks 1–3", "all"],
      ["Max. epochs; batch (P × K)", "25; 32 × 4", "35; 16 × 4"],
      ["LR: backbones / new layers / ArcFace", "3×10⁻⁵ / 3×10⁻⁵ / 10⁻³", "10⁻⁵ / 10⁻⁵ / 10⁻⁴"],
      ["Optimiser", "AdamW, weight decay 10⁻³", "same"],
      ["LR schedule", "reduce on plateau (×0.5, patience 3)", "same"],
      ["Early stopping", "patience 6, best weights restored", "same"]], [3, 3, 2.4]),
  H2("Open-set decision rule and calibration", "sec:decision"),
  P("A patient enrols each medicine j from k = 3 photos; its prototype p_j is the normalised sum of their embeddings. For a query with similarities s_j = zᵀp_j, sorted s₍₁₎ ≥ s₍₂₎,"),
  EQ(2, "ŷ = UNKNOWN if s₍₁₎ < τ;   RETAKE if s₍₁₎ ≥ τ and s₍₁₎ − s₍₂₎ < m;   argmax_j s_j otherwise", [
    "\\mathbf{p}_j=\\sum_i f_\\theta(x_{j,i})/\\lVert\\sum_i f_\\theta(x_{j,i})\\rVert  % inline, before the equation",
    "\\hat y=\\begin{cases}\\textsc{unknown} & s_{(1)}<\\tau,\\\\",
    "  \\textsc{retake} & s_{(1)}\\ge\\tau \\text{ and } s_{(1)}-s_{(2)}<m,\\\\",
    "  \\arg\\max_j s_j & \\text{otherwise,}\\end{cases}"], "eq:decision"),
  P("where RETAKE asks the patient to turn the pill over and photograph it again. We calibrate on validation only, by simulating 1,000 patients with 10 medicines each. τ = 0.895 rejects 95% of medicines that are not the patient's. As Figure @fig:calib(a) shows, the margin trades correct names for retakes while wrong names are already near zero; we take the largest margin with at most 10% retakes, m = 0.115. The crop window, 1.55× the detected box's long side, is the best of a sweep from 1.3 to 2.0, although all windows lie within one point (Figure @fig:calib(b))."),
  figure("fig06_calibration", "fig:calib", "Settings chosen on validation (1,000 simulated patients, 10 medicines, 3 enrolment photos; τ rejects 95% of other medicines). (a) The margin trades correct names for retakes while wrong names stay near zero. (b) The crop window (note the narrow y-range)."),
  H2("Identification with imprints and calibrated confidence", "sec:identify"),
  P("A pill that is not in the gallery is ranked against 4,936 US catalogue products; a product's look score s is the similarity of its closest reference photo. Florence-2-large [cite:xiao2024florence2] reads the imprint zero-shot at four rotations (Florence-2-base on the CPU path). Each recorded imprint token gets an IDF weight in [0, 1], so a rare token ('EC5') counts a lot and a common one ('93', printed on hundreds of Teva pills) little. A product's imprint score r is the weight of its best token found in the reads, and the top K = 100 products by look are re-ranked by s + αr with α = 0.05, both chosen on validation (Figure @fig:identify(b)). In the patient's gallery, only tokens that distinguish the enrolled imprints from each other confirm or contradict a match. Raw similarities are not probabilities (pills that are not in the catalogue still score about 0.93), so a logistic model fitted on validation, in the spirit of Platt scaling [cite:platt1999probabilistic], gives the stated confidence σ(−12.50 + 12.50·s₁ + 72.13·(s₁ − s₂) + 3.92·r₁). Here s₁, s₂ are the look scores of the top two re-ranked products and r₁ is the imprint score of the first."),
  ...CODE(["% inline LaTeX for the confidence", "$\\sigma(-12.50+12.50\\,s_1+72.13\\,(s_1-s_2)+3.92\\,r_1)$"]),
  H2("Implementation and reproducibility", "sec:impl"),
  P("We use PyTorch [cite:paszke2019pytorch] 2.11 and Hugging Face Transformers 5.19, and train on one NVIDIA RTX 4080 SUPER (16 GB). The model is served as a FastAPI service on a Hugging Face ZeroGPU Space, with 92 automated tests. Each stage is one script in the supplementary code (`make_splits`, `train_stage2`, `calibrate_stage2`, `evaluate_stage2`, `tune_identify_rerank`, `calibrate_identify`, `make_paper_figures`), and every number reported here is written to a JSON file by those scripts."),
  H2("Software 3.0 contribution", "sec:sw3-approach"),
  P("We used an AI coding agent, Claude Code [cite:anthropic_claude_code], throughout, programming in natural language in the sense of Software 3.0 [cite:karpathy2025software3]. It wrote the model, training, calibration and serving code and the evaluation scripts from our instructions; we reviewed and ran them. The effective-dimension measure in the training log located run 1's collapse in the new layers' learning rate, and the fix was proposed and tested in the same workflow. Every proposal was adopted or rejected on validation measurements, not on the agent's word: two imprint-reader fine-tunes it set up were rejected (Table @tab:negative)."),
  NOTE("[[TO DO]] Agree this paragraph's wording as a team; the development log is `epills/PROGRESS_LOG.md`."));

// ------------------------------------------------------------------------------------------------- EXPERIMENTS
add(H1("Experiments", "sec:experiments"),
  H2("Setup", "sec:setup"),
  PARA("Test sets.", "No test medicine appears in training in any form, and every setting is fixed on validation before a test set is scored, once. *C3PI held-out*: 4,005 photos of 200 appearance groups (the medicines of ePillID's official holdout fold and their look-alikes), photographed like the training data. *MEDISEG* [cite:mediseg2026]: 13,973 pills of 32 Hong Kong products, taken with other phones and in cluttered scenes; median pill size 81 px. *ePillID* [cite:usuyama2020epillid]: the published benchmark (fold 4, two-sided queries against references of all 4,902 types)."),
  PARA("Healthcare-relevant metrics.", "We simulate patients with 10 enrolled medicines (3 photos each). For the patient's own pills we report the share *named correctly*, *named wrongly* (the safety-critical error), *retake* and *called unknown*; for pills that are not theirs, the share *rejected*. A wrong name may be acted on, whereas \"unknown\" only costs time, so wrong names matter most. We also report top-k accuracy, AUROC, ePillID's MAP and GAP, and the reliability of the stated confidence. 95% intervals come from resampling medicines."),
  PARA("Baselines.", "(i) Run 1: the same network with a common recipe (new layers at 10⁻³, no margin warm-up). (ii) The published ePillID models [cite:usuyama2020epillid]. (iii) Look only versus look plus imprint, for the gallery and the catalogue. (iv) For the pill finder that feeds the model, an earlier finder that returned exactly one box on only 28% of single-pill photos; ours counts 85–88% of 1-, 2- and 3-pill photos right (Figure @fig:finder)."),
  figure("fig17_pill_finder", "fig:finder", "The pill finder on C3PI photos with known pill counts (2- and 3-pill photos composed from single-pill photos). Small shares are listed after each bar in legend order.", 3.3),
  H2("Main results", "sec:main"),
  P("Table @tab:main gives the test results and Figure @fig:main the three headline measures. In the conditions it was trained for, the served model (run 4) names 76.0% of a patient's own pills correctly and only 0.04% wrongly (95% CI 0.02–0.07%), while rejecting 94.8% of pills that are not the patient's (Figure @fig:openset(a)). The imprint check raises correct names to 78.3% (wrong 0.06%) by settling retakes. Against the catalogue, the right product comes first for 80.2% of held-out pills and is in the top 5 for 94.4%, against 71.6% and 91.7% by look alone. Adding reference photos (run 3) nearly tripled ePillID MAP@1 (13.7 → 37.1%), and robustness augmentation (run 4) gained 6.4 points on MEDISEG at a small cost in C3PI conditions; run 4 is the served model."),
  table("tab:main", "Test results on medicines never seen in training (%; 10-medicine galleries, 3 enrolment photos, calibrated τ and m). Bottom block: the served pipeline with imprint reading (GPU reader; CPU reader in brackets).",
    ["", "Run 2", "Run 3", "Run 4 (served)"], [
      ["C3PI held-out: top-1 among 200 medicines", "82.0", "**84.4**", "83.2"],
      ["   gallery: named correctly / wrongly", "78.6 / 0.1", "**80.4** / 0.0", "76.0 / 0.04"],
      ["   other medicines rejected", "94.2", "94.1", "**94.8**"],
      ["MEDISEG: top-1 among 32 products", "38.9", "41.8", "**48.2**"],
      ["   gallery: named correctly / wrongly", "17.6 / 1.1", "18.7 / 1.0", "18.7 / **0.7**"],
      ["   own called unknown / other rejected", "51.5 / 78.7", "64.2 / 89.3", "70.4 / **93.1**"],
      ["ePillID strict (no test type in training): MAP@1", "13.7", "37.1", "**42.3**"],
      ["C3PI gallery + imprint check: named correctly / wrongly", "", "", "78.3 (77.3) / 0.06"],
      ["   retake / called unknown", "", "", "7.2 / 14.5"],
      ["Catalogue (4,936 products) + imprint: top-1 / top-5", "", "", "80.2 (79.2) / 94.4"],
      ["   look only: top-1 / top-5", "", "", "71.6 / 91.7"]], [4.2, 1.4, 1.4, 1.8], [7]),
  figure("fig07_main_results", "fig:main", "Test results on medicines never seen in training, each scored once. Run 3 adds C3PI studio reference photos; run 4 adds augmentation for other phones and jittered crops."),
  figure("fig08_open_set_outcomes", "fig:openset", "What the patient sees (run 4, embedding only; 10 medicines, 3 enrolment photos). Small shares are listed after each bar in legend order. In C3PI conditions most own pills are named correctly and wrong names are rare; on other phones the threshold transfers badly and most own pills are called unknown, which is safe but unhelpful."),
  H2("Ablations and hyperparameters", "sec:ablations"),
  P("Table @tab:ablation isolates each design decision. The learning rates matter most. With the new layers at 10⁻³ the embedding collapses to about 3 effective dimensions and validation top-1 stalls at 77%, while lower rates keep 12–15 dimensions and reach 98% (Figure @fig:training). The decision settings are smooth (Figure @fig:calib): the crop window moves correct names by under one point, and the margin trades correct names for retakes almost one for one. The imprint weight has a clear optimum (Figure @fig:identify(b)), because a large α lets false OCR reads promote wrong products; the imprint helps most when the pill's own imprint is actually read (Figure @fig:identify(a))."),
  table("tab:ablation", "Ablations (validation unless marked *test*).",
    ["Change", "Effect", "Kept"], [
      ["Run 1 → 2: lower LRs (3×10⁻⁵, 10⁻⁵), separate ArcFace LR, margin warm-up", "top-1 in 10 medicines 77.2 → 98.3%; AUROC 0.78 → 0.981; correct at 95% rejection 17 → 90.7%; dims ≈3 → 12–15", "yes"],
      ["Run 2 → 3: + studio reference photos", "validation 98.1%; *test* ePillID MAP@1 13.7 → 37.1%", "yes"],
      ["Run 3 → 4: + photo and crop-jitter augmentation", "*test* MEDISEG 41.8 → 48.2%; C3PI 84.4 → 83.2%", "yes"],
      ["Crop window 1.3–2.0×", "named correctly 86.1–87.2%", "1.55"],
      ["Margin m 0–0.3", "named correctly 87 → 54%; retakes 0 → 34%", "0.115"],
      ["Imprint re-ranking α, K", "top-1 72.1% (α = 0) → 81.6%; *test* 71.6 → 80.2%", "0.05, 100"],
      ["Gallery check: distinct tokens only", "named wrongly 35 → 30; other pills accepted 6,158 → 6,150", "yes"],
      ["Official ePillID protocol (runs 5–7)", "ePillID MAP 0.57–0.60; MEDISEG 52–54%", "benchmark"]], [3.4, 4.6, 1.2]),
  figure("fig05_training_curves", "fig:training", "Training. (a) Validation top-1 within a gallery of 10 medicines never trained on. (b) Effective embedding dimensions: with the new layers at 10⁻³ the embedding collapsed to about 3."),
  figure("fig12_identify_topk", "fig:identify", "Catalogue identification. (a) Top-k on held-out medicines: the imprint helps most when the pill's own imprint is read (92.9% first vs. 68.6% when it is not). (b) Validation grid for α and K; the box marks the served setting."),
  H2("Robustness", "sec:robustness"),
  P("*Other phones* are the main weakness (Table @tab:main, Figure @fig:openset). On MEDISEG, top-1 drops to 48% and the C3PI-calibrated threshold calls 70% of own pills unknown, though wrong names stay below 1%: the system fails safe but is not yet usable there. Small pills are the main factor (Figure @fig:robust(a): 36–37% top-1 under 100 px, 62% above), and more enrolment photos help only partly (Figure @fig:robust(b)). *Pills outside the catalogue*: 69.8% of MEDISEG pills get *no confident match* and 7.0% *possible match*, while 75.8% of catalogue pills get *possible match*, and the stated confidence tracks observed accuracy (Figure @fig:confidence). *CPU deployment*: with the smaller reader, identification loses one point (79.2%) and wrong names do not rise (Table @tab:main)."),
  figure("fig09_robustness", "fig:robust", "Robustness on MEDISEG (32 Hong Kong products, other phones, none in training). (a) Top-1 by pill size in the photo. (b) MEDISEG's deploy split (40 query pills): more enrolment photos help only partly."),
  figure("fig13_identify_confidence", "fig:confidence", "Calibrated confidence for identification. (a) Stated confidence against observed accuracy on held-out medicines. (b) Verdicts for catalogue pills and for pills not in the catalogue (MEDISEG)."),
  H2("Comparison with the ePillID benchmark", "sec:epillid"),
  P("For a like-for-like comparison, runs 5–7 follow ePillID's protocol and train on the references of all types (Table @tab:epillid, Figure @fig:epillid). Our embedding alone (MAP 0.57–0.60) is in the range of the paper's plain classifiers, not its best model. Re-ranking by imprint, which the paper's model did not use, raises MAP to 0.77–0.78 and GAP@1 to 0.921, above the paper's 0.912, while MAP@1 stays lower (69.0 vs. 92.0%). The paper's ablations attribute its advantage to its training recipe (contrastive and hard-mined triplet losses with softmax and ArcFace over side-specific classes, a 2,048-d embedding) rather than its backbone. We ruled out our crop style, the 64-d pooling, and local versus global pooling as causes."),
  table("tab:epillid", "ePillID, official protocol (fold 4, two-sided). The best published model is a ResNet-152 with compact bilinear pooling and multi-head metric learning. \"+ imprint\" uses OCR information the published models did not use.",
    ["", "MAP", "GAP", "MAP@1", "GAP@1", "top-5"], [
      ["[cite:usuyama2020epillid], best", "**0.958**", "**0.812**", "**92.0**", "0.912", "–"],
      ["[cite:usuyama2020epillid], plain classifiers", ".547–.727", ".241–.448", "35.7–56.8", ".331–.551", "–"],
      ["Ours, run 5", "0.597", "0.207", "45.7", "0.526", "77.5"],
      ["Ours, run 5 + imprint", "0.780", "0.598", "69.0", "**0.921**", "89.6"],
      ["Ours, run 7 + imprint", "0.774", "0.614", "67.6", "0.916", "89.5"]], [3.4, 1, 1, 1, 1, 0.9]),
  figure("fig10_epillid_benchmark", "fig:epillid", "ePillID benchmark (official protocol): MAP@1 and GAP@1 of our runs with and without imprint re-ranking against the best published model."),
  H2("Error analysis and interpretation", "sec:errors"),
  PARA("Imprints.", "The reader finds a token of the pill's own imprint on about 47% of validation photos. Figure @fig:imprint shows where it fails: debossed, small and round pills are hardest, and C3PI's own raters marked many imprints unreadable."),
  figure("fig11_imprint_reading", "fig:imprint", "Share of validation photos on which a token of the pill's own imprint is read (Florence-2-large, zero-shot), by imprint type, C3PI rater's judgement of visibility, pill size and shape."),
  PARA("Look-alikes.", "The hardest errors are medicines that look the same (Figure @fig:lookalike). With two strengths of one ingredient in a gallery, one is named as the other 0.11% of the time, or 0.58% when shape and colour also match. The imprint check settles retakes but cannot confirm strength, so strength is taken from the pharmacy label. Same-manufacturer pairs that share an imprint prefix are confused 0.01% of the time overall, but the worst pairs share Teva's 'TV', e.g. tramadol 50 mg (TV;58) and carvedilol 12.5 mg (TV;7295) at 5.8%."),
  figure("fig16_lookalike_error_analysis", "fig:lookalike", "Look-alikes, with both medicines enrolled in one gallery. (a) Wrong-name rates by kind of pair, look only and with the imprint check. (b) The worst same-manufacturer pairs."),
  PARA("What the model learned.", "Photos of a medicine never seen in training form tight clusters (Figure @fig:tsne), and the calibrated threshold separates a photo's similarity to its own medicine from its similarity to the closest of nine others (Figure @fig:sim). Tracing the served model on single photos shows two things. Capsule routing moves from equal weights to about 9% CNN and 91% ViT, and the refinement block's token attention is nearly uniform, consistent with that block not being a bottleneck. Overall, the hybrid learns an embedding that transfers to unseen medicines when it is trained with the right learning rates. Calibration gives predictable behaviour on new medicines in familiar conditions but not across phones, and the imprint is evidence that complements the look."),
  figure("fig14_similarity_distributions", "fig:sim", "Held-out photos: similarity to their own medicine's 3-photo prototype and to the closest of 9 other medicines, with the threshold τ. (In LaTeX: half width, side by side with the next figure.)", 3.3),
  figure("fig15_embedding_tsne", "fig:tsne", "t-SNE [cite:vandermaaten2008tsne] of the 4,005 held-out photos' embeddings; three medicines highlighted (photo counts in brackets).", 3.3),
  PARA("Negative results.", "Table @tab:negative lists what did not work. Two LoRA [cite:hu2022lora] fine-tunes of the imprint reader read more imprints on validation (44.0 → 53.5% and 45.0 → 50.7%). The first wrote memorised imprints onto new pills; the second doubled wrong names in galleries (30 → 61). Both were rejected, as were look-alike-character matching and per-candidate imprint scoring."),
  table("tab:negative", "What did not work (validation).",
    ["Tried", "Result", "Decision"], [
      ["LoRA fine-tune of Florence-2-large, v1 (19,312 photos of training medicines)", "imprints read 44.0 → 53.5%, but memorised imprints written onto new pills; identification and gallery check worse", "rejected"],
      ["LoRA v2 (also learns to output nothing on unreadable photos)", "own imprint read 45.0 → 50.7%; identification 82.0 → 83.0%; gallery wrong names 30 → 61", "rejected"],
      ["Look-alike characters (O/0, B/8) treated as equal", "measured and dropped", "rejected"],
      ["Scoring each candidate's imprint separately", "measured and dropped", "rejected"],
      ["Runs 6–7: stronger fine-tuning; benchmark-aligned training", "better validation and MEDISEG, not ePillID", "not served"]], [3.4, 4.6, 1.2]),
  H2("Software 3.0 in coding and evaluation", "sec:sw3-experiments"),
  P("The evaluation harness was written with the coding agent from natural-language specifications: the patient simulation, the look-alike and same-manufacturer stress tests, the top-k and confidence evaluations, the figures, and the API's 92 tests. Tests and review caught real defects, among them a safety bug in the label reader that linked an extended-release sticker to the immediate-release product; we fixed it by enforcing the release form. Our safeguards against over-trusting generated code were procedural: settings were chosen on validation, each test set was scored once, and every number in this paper is read from a file written by a script."),
  NOTE("[[TO DO]] The team decides how to evidence tool use (`PROGRESS_LOG.md`, git history); do not create mock logs."));

const doc = new Document({
  creator: "PharmaScan team", title: "PharmaScan: Approach and Experiments",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 40 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 20 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ children: [PageNumber.CURRENT], size: 18, color: INK2 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(buf => {
  const out = path.join(PAPER, "PharmaScan_approach_experiments.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1e6).toFixed(1), "MB");
});
