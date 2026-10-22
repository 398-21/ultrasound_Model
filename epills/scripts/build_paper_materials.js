// Builds docs/paper/PharmaScan_paper_materials.docx from docs/paper/figures (needs: npm install docx@9.5.1). Run: node scripts/build_paper_materials.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, BorderStyle, TableOfContents, PageBreak, Footer, PageNumber, ExternalHyperlink,
} = require("docx");

const PAPER = "C:/Users/sheng/Documents/GitHub/ultrasound_Model/epills/docs/paper";
const FIG = path.join(PAPER, "figures");
const REG = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(FIG, "figures.json"), "utf8")).map(r => [r.name, r]));
const W = 9026; // A4 text width at 1-inch margins, DXA
const FONT = "Calibri", MONO = "Consolas";
const BLUE = "2A78D6", INK2 = "52514E";

// ---- inline markup: **bold**, `code`, [[TO DO ...]] highlighted ----
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`|\[\[[^\]]+\]\])/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("`")) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: MONO, size: 18 }));
    else if (part.startsWith("[[")) out.push(new TextRun({ ...base, text: "[" + part.slice(2, -2) + "]", bold: true,
      color: "9A3412", shading: { type: ShadingType.CLEAR, fill: "FDE9DF", color: "auto" } }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120 }, ...o.para });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t),
  spacing: { after: 60 } });
const N = (t, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
  spacing: { after: 60 } });
const CODE = lines => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l || " ", font: MONO, size: 17 })],
  shading: { type: ShadingType.CLEAR, fill: "F4F3F0", color: "auto" },
  spacing: { after: i === lines.length - 1 ? 160 : 0 }, indent: { left: 120, right: 120 },
}));
const NOTE = (t, fill = "EEF4FC", edge = BLUE) => new Paragraph({
  children: runs(t), spacing: { before: 80, after: 160 }, indent: { left: 160, right: 160 },
  shading: { type: ShadingType.CLEAR, fill, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: edge, space: 6 } },
});
const TODO = t => NOTE("**For the team:** " + t, "FDF3E9", "EB6834");

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "C9C8C3" };
function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  widths = widths.map(w => Math.round(w * W / total));
  widths[widths.length - 1] = W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "E3EEFB", color: "auto" } : undefined,
    borders: { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: String(t).split("\n").map(line => new Paragraph({ children: runs(line, { size: 18, bold: head || undefined }) })),
  });
  const t = new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
  const out = [];
  if (opts.title) out.push(new Paragraph({ children: runs(opts.title, { bold: true, size: 19 }), spacing: { before: 120, after: 60 },
    keepNext: true }));
  out.push(t);
  out.push(new Paragraph({ children: runs(opts.note || "", { size: 17, color: INK2 }), spacing: { before: 60, after: 200 } }));
  return out;
}

function figure(name, num, widthIn = 6.2) {
  const r = REG[name];
  const buf = fs.readFileSync(path.join(FIG, r.png));
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20);
  const w = Math.min(widthIn, 6.27) * 96, h = w * ph / pw;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h },
        altText: { title: name, description: r.caption, name } })] }),
    new Paragraph({ spacing: { before: 60, after: 80 }, children: [
      new TextRun({ text: `Figure ${num}. `, bold: true, size: 18 }), new TextRun({ text: r.caption, size: 18 })] }),
    new Paragraph({ spacing: { after: 240 }, children: runs(
      `File: \`figures/${name}.pdf\` (LaTeX) / \`${name}.png\` (Word). Source: ${r.sources.slice(0, 2).map(s => "`" + s + "`").join(", ")}${r.sources.length > 2 ? " …" : ""}`,
      { size: 16, color: INK2 }) }),
  ];
}

// ======================================================================================================================
const body = [];
const add = (...xs) => xs.forEach(x => Array.isArray(x) ? body.push(...x) : body.push(x));

// ---- cover ----
add(new Paragraph({ spacing: { before: 2400, after: 120 }, children: [new TextRun({ text: "PharmaScan", size: 56, bold: true, color: BLUE })] }),
  new Paragraph({ spacing: { after: 360 }, children: [new TextRun({ text: "Materials for the Deep Learning paper (NeurIPS format)", size: 32 })] }),
  P("Content, numbers, figures, tables and references for writing the report in LaTeX, organised by the rubric in `epills/rubric.md`."),
  P("**Report due:** 25 October 2026.   **Prepared:** 19 October 2026.   **Model described:** run 4, `stage2_run4@2026-10-09` (the served model; final, no further training)."),
  P("**Figures:** `epills/docs/paper/figures/` (17 figures, PDF for LaTeX and 300-dpi PNG; captions in `figures.json`).   **References:** `epills/docs/paper/references.bib`.   **Regenerate figures:** `.venv\\Scripts\\python scripts\\make_paper_figures.py` from the `epills` folder."),
  NOTE("**How to read this document.** Every number comes from a result file in the repository (named next to each table or figure), so you can quote it in the paper. Text in normal paragraphs is draft wording you may reuse or rephrase. Orange boxes and [[TO DO]] marks are things only the team can supply (names, student IDs, lecture timestamps, final wording). Citations marked VERIFY in the .bib file must be checked against the publisher's page before submission. Figure numbers match the file names (fig01 … fig17), so they appear slightly out of order here; renumber them in the paper."),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  P("(In Word: right-click the contents and choose Update Field to fill in page numbers.)", { run: { color: INK2, size: 18 } }));

// ---- 0. rubric map ----
add(H1("0. Rubric map"),
  P("Where each rubric item is covered in this document, and which figures and tables support it."),
  table(["Paper section", "Weight", "Rubric asks for", "Here", "Figures / tables"], [
    ["Title, authors, abstract", "–", "Title, names, student IDs, abstract", "§1", "–"],
    ["Problem formulation", "15%", "The problem, why it matters, formal task; how Software 3.0 / Gen-AI helped", "§2", "Fig. 1; Table 1"],
    ["Related work", "15%", "≥ 10 citations; link to the lecture recordings (video, slide or time)", "§3, Appendix B", "Table 2"],
    ["Data", "15%", "Source, statistics, preprocessing, qualitative examples", "§4", "Figs. 3, 4; Tables 3–4"],
    ["Approach", "25%", "Reproducible: architecture, training, loss, optimisation, hyperparameters; Software 3.0 contribution", "§5", "Figs. 2, 6; Tables 5–6"],
    ["Experiments", "25%", "Baselines, ablations, hyperparameters, robustness, error analysis, interpretation; healthcare metrics; graphs and tables; Software 3.0 in coding and evaluation", "§6", "Figs. 5, 7–17; Tables 7–11"],
    ["Conclusion", "5%", "Summary, limitations, future work", "§7", "–"],
    ["Presentation and supplementary", "–", "10-minute Lectrace with 3–5 execution states; supplementary ZIP; cite any base code", "§8", "–"],
  ], [1.6, 0.6, 3.2, 0.9, 1.5]),
  P("**Suggested length split for 8–9 NeurIPS pages:** problem 1 page, related work 1, data 1, approach 2, experiments 2.5–3, conclusion 0.5. Use the full-width figures (`figure*`) for Figs. 1, 2, 7 and 10, and `\\includegraphics[width=\\linewidth]` with the PDF files. Figures 14, 15 and 17 are half-width and pair well side by side."));

// ---- 1. title & abstract ----
add(H1("1. Title, authors and abstract"),
  H2("Candidate titles"),
  B("PharmaScan: Calibrated Open-Set Pill Verification with a Hybrid CNN–ViT Embedding and Imprint Reading"),
  B("Is This My Pill? Few-Shot, Open-Set Medication Verification from Phone Photos"),
  B("From Faces to Pills: Repurposing a Hybrid CNN–ViT Expression Model for Medication Verification"),
  H2("Authors"),
  TODO("Names, student IDs and emails of every team member, in the order agreed by the team."),
  H2("Draft abstract (≈ 190 words)"),
  NOTE("Patients who take several medicines can confuse their pills, and taking the wrong one is a medication error. We frame pill verification from a phone photo as open-set retrieval: each pill must be matched to one of the patient's own enrolled medicines, rejected as not theirs, or sent back for another photo. We repurpose our earlier facial-expression model, a hybrid of VGG16 with CBAM attention and ViT-B/16 fused by a capsule block and refined by a transformer, as a 256-dimensional embedding trained with ArcFace on 29,471 NLM C3PI photos of 1,710 appearance groups, and we calibrate its decision thresholds on validation medicines only. On 200 held-out medicines never seen in training, a 10-medicine gallery enrolled from three photos names 76.0% of pills correctly and only 0.04% wrongly, while rejecting 94.8% of pills that are not the patient's. Reading the imprint with Florence-2 OCR raises identification among 4,936 catalogue products from 71.6% to 80.2% top-1, and a logistic confidence model makes the stated confidence match observed accuracy. Photos from other phones remain the main weakness: on MEDISEG the system fails safe, calling 70% of own pills unknown. The system is deployed as a public API.", "F4F3F0", "8A8984"),
  P("Numbers in the abstract: Table 7 (§6.3) and Fig. 12. If the team prefers the figure with the imprint check, use “78.3% named correctly, 0.06% wrongly” (Table 8)."));

// ---- 2. problem formulation ----
add(H1("2. Problem formulation (15%)"),
  H2("2.1 Why it matters"),
  B("Medication errors are a leading cause of avoidable harm in health care; the WHO's third Global Patient Safety Challenge, Medication Without Harm (2017), targets them [who2017medication]. Older patients on several medicines (polypharmacy) and their caregivers are the most exposed."),
  B("In Singapore, polyclinics and pharmacies often dispense tablets loose in a bag with a printed sticker (name, strength, dose, instructions; format explained in MOH's “Know about medication label” [moh_medication_label]). Once pills leave the strip or bag, the patient must tell them apart by look."),
  B("PharmaScan is the vision part of the team's Medication Care Agent (team blueprint `epills/PharmaScan_Care_Agent_Workflow.html`): the patient photographs a dose and the app checks it against their own medicines, reads pharmacy stickers and screens interactions. The patient confirms; the app never decides that a pill is safe to take."),
  ...figure("fig01_system_overview", 1),
  H2("2.2 Formal task"),
  P("**Main task: open-set verification against a personal gallery.** Let `f_θ` map a pill crop `x` to a unit vector `z = f_θ(x) ∈ R^256`. A patient enrols `G` medicines (≈ 10), each from `k = 3` photos; the prototype of medicine `j` is the normalised mean of its enrolment embeddings. For a query pill with similarities `s_j = z · p_j`, sorted so that `s_(1) ≥ s_(2)`:"),
  B("`s_(1) < τ` → “not one of your medicines” (unknown);"),
  B("`s_(1) ≥ τ` and `s_(1) − s_(2) < m` → “photograph it again, other side up” (retake / flip);"),
  B("otherwise → “this looks like your medicine j*”."),
  P("`τ = 0.8949` and `m = 0.115` are chosen on validation medicines only (τ rejects 95% of other medicines; m is the largest margin with at most 10% retakes; §5.5). LaTeX:"),
  ...CODE([
    "\\mathbf{p}_j = \\frac{\\sum_{i=1}^{k} f_\\theta(x_{j,i})}{\\lVert \\sum_{i=1}^{k} f_\\theta(x_{j,i}) \\rVert},\\qquad",
    "s_j = f_\\theta(x)^\\top \\mathbf{p}_j,\\qquad",
    "\\hat y = \\begin{cases} \\textsc{unknown} & s_{(1)} < \\tau \\\\ \\textsc{retake} & s_{(1)} \\ge \\tau,\\ s_{(1)} - s_{(2)} < m \\\\ \\arg\\max_j s_j & \\text{otherwise} \\end{cases}",
  ]),
  P("**Secondary task: catalogue identification.** For a pill that is not in the gallery, rank 4,936 US products by similarity to their reference embeddings, re-rank with the OCR-read imprint, and attach a calibrated confidence and a verdict (`possible_match`, `uncertain`, `no_confident_match`)."),
  P("**Why this framing (for the paper).** (1) The set of medicines is open: new medicines appear every day, so the model must work on medicines it never trained on, from a few enrolment photos, without retraining. (2) Errors are asymmetric: naming the wrong medicine is a safety risk, while “unknown” or “photograph again” only costs time. So the main metric is the wrong-name rate at a calibrated operating point, not plain accuracy. (3) The look of a pill cannot confirm its strength (same-ingredient strengths often look alike), so strength comes from the label, not from vision."),
  H2("2.3 Targets set before the experiments"),
  table(["Validation measure (10-medicine gallery, 3 enrolment photos)", "Target", "Run 4 (served)"], [
    ["Right pill first (top-1 within the gallery)", "≥ 95%", "97.9%"],
    ["AUROC, own vs. foreign pills", "≥ 0.90", "0.978"],
    ["Own pills named correctly when 95% of foreign pills are rejected", "≥ 80%", "87.9%"],
    ["Foreign pills wrongly accepted at that threshold", "≤ 2%", "1.2%"],
  ], [5, 1.2, 1.6], { title: "Table 1. Success criteria (validation, from the project plan) and the served model.", note: "Source: `docs/stage2_results.md` §2; `outputs/runs/stage2_run4/summary.json`." }),
  H2("2.4 Scope"),
  B("Visual evidence only (“looks like”), with the patient confirming. Not a medical device and not a claim of identity, strength or safety."),
  B("The API is stateless; the app's backend owns patient data and dose decisions. No real patient data was used (the demo patient is synthetic)."),
  H2("2.5 Software 3.0 / Gen-AI in the problem formulation"),
  P("The rubric asks how Software 3.0 (programming in natural language with large language models) helped frame the problem. What actually happened, with evidence in the repository:"),
  B("The team's blueprint and test plan (`docs/api_review_and_test_plan.md`) were reviewed with Claude Code (Anthropic's agentic coding assistant) into a numbered job list and trust boundaries (`docs/blueprint_review.md`, jobs B1–B27; decisions D1–D6 made by the team)."),
  B("Turning the safety asymmetry into measurable quantities (wrong-name rate, foreign-pill rejection, retake rate at a calibrated threshold; Table 1) and into test cases (`docs/evidence.md`, cases D01–C01) was done in conversation with the assistant, then checked by the team."),
  B("Human decisions stayed with the team: the target users and the Singapore sticker format (the MOH leaflet was chosen as the standard by a team member), the decision to stop retraining after 15 October, and which proposals to reject."),
  TODO("Agree on the wording of this paragraph. Say what each member did and where the AI assistant was used; the development log `epills/PROGRESS_LOG.md` records each session. Do not claim more than the log shows."));

// ---- 3. related work ----
add(H1("3. Related work (15%)"),
  P("Draft paragraphs, one per theme; bracketed keys are the BibTeX keys in `references.bib` (32 entries, more than the 10 required)."),
  H3("Pill image recognition"),
  P("The NLM Pill Image Recognition Challenge released the C3PI images and framed the task as matching consumer photos to reference photos [yaniv2016nlm]. Deep CNNs were soon applied to fine-grained pill identification [wong2017pill]. ePillID [usuyama2020epillid] made the task a low-shot benchmark (4,902 types, most with only reference photos) and reported its best results with multi-head metric learning and bilinear pooling; few-shot pill recognition was studied in [ling2020fewshot]. MEDISEG [mediseg2026] provides Hong Kong products photographed with other phones. **Our difference:** a per-patient open-set gallery with calibrated rejection and a retake option, tested across phones, plus imprint OCR as a second, independent cue."),
  H3("Backbones, attention and fusion"),
  P("Our model combines a convolutional backbone [simonyan2015vgg] with channel and spatial attention [woo2018cbam] and a vision transformer [dosovitskiy2021vit, vaswani2017attention], fused with capsule-style dynamic routing [sabour2017capsules]. The architecture is the team's earlier facial-expression model (§5.9), reused as a feature extractor."),
  H3("Metric learning"),
  P("Embeddings trained with triplet losses [schroff2015facenet], with P×K batches and hard mining [hermans2017triplet], or with margin-based softmax losses [wang2018cosface, deng2019arcface] allow recognising classes never seen in training; [musgrave2020reality] shows that careful, fair evaluation matters more than the choice of loss. We use ArcFace with a margin warm-up and P×K batches."),
  H3("Open-set recognition and calibration"),
  P("Open-set recognition must reject inputs from unseen classes [scheirer2013openset]; maximum-score thresholds are a strong baseline [hendrycks2017baseline]. Modern networks are over-confident [guo2017calibration]; we fit a logistic (Platt-style [platt1999probabilistic]) model on validation to turn similarities into a calibrated confidence."),
  H3("Reading text on pills, and adaptation"),
  P("We read imprints zero-shot with Florence-2 [xiao2024florence2]; two LoRA fine-tunes [hu2022lora] of the reader were tried and rejected (§6.9). Embeddings are visualised with t-SNE [vandermaaten2008tsne]."),
  H3("Medication safety data and Software 3.0"),
  P("Interaction screening uses DDInter [xiong2022ddinter] and drug names are normalised with RxNorm [nelson2011rxnorm]. Software 3.0 [karpathy2017software2, karpathy2025software3] describes programming with natural-language prompts to LLMs; we used an AI coding assistant [anthropic_claude_code] throughout (§5.10, §6.11)."),
  H2("3.1 Links to the lecture recordings"),
  P("The rubric asks to connect the related work to the course lectures (playlist: https://www.youtube.com/playlist?list=PLORG7zBQosxw). We could not watch the recordings, so the right-hand column is for the team to fill."),
  table(["Topic in our paper", "Where it appears", "Lecture video, slide or time"], [
    ["CNNs, VGG, transfer learning from ImageNet", "§5.2 backbone; phase-1 freezing", "[[TO DO]]"],
    ["Attention, transformers, ViT", "§5.2 ViT branch, refinement block", "[[TO DO]]"],
    ["Embeddings, metric learning, contrastive / margin losses", "§5.3 ArcFace; open-set gallery", "[[TO DO]]"],
    ["Regularisation and data augmentation", "§4.3 augmentation; dropout 0.4; weight decay", "[[TO DO]]"],
    ["Optimisation: AdamW, learning-rate schedules, early stopping", "§5.4; run 1 vs run 2", "[[TO DO]]"],
    ["Evaluation: ROC/AUROC, precision at k, calibration", "§6.1 metrics; Fig. 13", "[[TO DO]]"],
    ["Generative AI / LLMs / Software 3.0", "§2.5, §5.10, §6.11", "[[TO DO]]"],
    ["Deploying models (APIs, CPU vs GPU)", "§5.8; §6.6", "[[TO DO]]"],
  ], [3, 2.5, 2.5], { title: "Table 2. Lecture links (for the team to complete)." }));

// ---- 4. data ----
add(H1("4. Data (15%)"),
  H2("4.1 Sources"),
  table(["Dataset", "Role", "Size used", "Licence"], [
    ["NLM C3PI consumer photos (`C3PI_Test` JPG)", "Train, validation, test", "40,587 photos → 40,550 pill crops; 2,400 medicines (NDC9 + part); 15 photos per medicine (median; min 3, max 101)", "Research and development use only"],
    ["NLM C3PI studio references (`MC_C3PI_REFERENCE_SEG_V1.6`)", "Extra training images (runs 3–4)", "6,198 crops of training groups", "As above"],
    ["ePillID [usuyama2020epillid]", "Published benchmark (fold 4); runs 5–7 train on its folds", "4,902 types; 960 with consumer photos; fold-4 test: 192 types", "MIT"],
    ["MEDISEG [mediseg2026]", "Cross-phone test only", "32 Hong Kong products; 8,261 images (640×640), 13,973 pill queries; deploy split 40 queries", "CC BY 4.0"],
    ["Product catalogue (from C3PI metadata and reference images)", "Identification gallery", "4,936 US products with imprint, shape, colour", "As C3PI"],
    ["DDInter 2.0, RxNorm", "Safety rules and name normalisation (system, not model)", "–", "CC BY-NC-SA 4.0; NLM terms"],
  ], [2.2, 1.7, 3.2, 1.4], { title: "Table 3. Data sources.", note: "Source: `docs/datasets.md`; `data/processed/c3pi_splits/manifest.csv`; `outputs/runs/stage2_run4/eval/mediseg.json`." }),
  H2("4.2 Statistics and splits"),
  P("Splits are made by **appearance group**: pill types with the same imprint, shape, colour and size (the same physical pill sold under several product codes) form one group (union-find). Every group containing an ePillID fold-4 holdout type is removed from training and becomes the test set; the rest is split at random (seed fixed) 80 / 10 / 10. Leakage checks: shared groups and NDC9s across splits, identical files (SHA-1) and near-identical crops (`scripts/make_splits.py`, `leakage_report.md`). Training uses the group as the class label (1,710 classes)."),
  table(["Split", "Groups", "Photos", "Use"], [
    ["Train", "1,710", "29,471 (+ 6,198 references from run 3)", "Training"],
    ["Validation, enrolled", "214", "3,556", "Galleries for calibration and model selection"],
    ["Validation, unknown", "214", "3,555", "Foreign pills for calibration"],
    ["Test (held-out)", "200", "4,005", "Scored once; ePillID fold-4 types and their look-alikes"],
  ], [2, 1, 3, 3], { title: "Table 4. C3PI splits.", note: "Source: `data/processed/c3pi_splits/manifest.csv`." }),
  ...figure("fig03_data_statistics", 3),
  P("**MEDISEG pill sizes** (long side of the pill box, pixels): 5% 27, median 81, 95% 194 — much smaller than C3PI crops, which explains part of the cross-phone gap (Fig. 9a)."),
  H2("4.3 Preprocessing and augmentation"),
  B("**Cropping.** Each C3PI photo has one pill; a cropper finds its box, and the crop is a square window of 1.55× the box's long side (chosen on validation, Fig. 6b), resized to 224×224."),
  B("**Augmentation (run 4)** — colour jitter kept small (saturation 0.05, hue 0.0125) because colour identifies pills; crop jitter (window ×0.75–1.35, centre shift ≤ 10%) to mimic detector error; photo conditions for other phones: brightness and contrast ±0.3, white balance ±8%, downscale to 40–160 px (p 0.3), JPEG quality 30–90 (p 0.3), Gaussian noise σ 0.03 (p 0.2); a neighbouring pill pasted partly into the frame (p 0.2)."),
  B("**Ethics.** Public datasets only; no patient data. C3PI's licence limits use to research and development."),
  H2("4.4 Qualitative examples"),
  ...figure("fig04_qualitative_examples", 4, 5.6));

// ---- 5. approach ----
add(H1("5. Approach (25%)"),
  H2("5.1 Overview"),
  P("A detected pill is cropped, embedded by the hybrid network and compared by cosine similarity with the patient's enrolled medicines (or the catalogue). Decisions use thresholds calibrated on validation. An OCR imprint reader adds a second, independent check (Fig. 1)."),
  H2("5.2 Architecture"),
  ...figure("fig02_architecture", 2),
  table(["Block", "Details", "Output"], [
    ["Input", "RGB crop, 224×224", "3×224×224"],
    ["CNN branch", "VGG16 conv blocks (ImageNet weights) → CBAM (channel + spatial attention) → global average pooling → Dense 256, ReLU", "256"],
    ["ViT branch", "ViT-B/16 (`google/vit-base-patch16-224-in21k`), pooler output", "768"],
    ["Capsule fusion", "Per-branch MLP (Dense 256, ReLU, dropout 0.4, Dense 256) + batch norm; dynamic routing, 2 iterations, residual update", "256"],
    ["Transformer refinement", "Project to 4 tokens × 64 + learned positions; 2 encoder layers (4 heads, feed-forward 128, dropout 0.4); mean pool", "64"],
    ["Embedding head", "Linear → L2 normalisation", "256 (unit length)"],
    ["Auxiliary heads", "Shape (softmax over shapes); colour (multi-label sigmoid)", "–"],
  ], [1.6, 5, 1.2], { title: "Table 5. HybridPillNet.", note: "Source: `src/epills/core/models/hybrid.py`. Keras-style initialisation, matching the original TensorFlow model." }),
  H2("5.3 Loss"),
  P("Total loss = ArcFace on the embedding (over the 1,710 training groups) + 0.2 × shape cross-entropy + 0.2 × colour binary cross-entropy (unknown labels ignored). ArcFace margin `m = 0.3`, scale `s = 32`; the margin warms up 0 → 0.1 → 0.2 over the first three epochs, then 0.3."),
  ...CODE([
    "\\mathcal{L} = \\mathcal{L}_{\\text{ArcFace}} + \\lambda_{s}\\,\\mathcal{L}_{\\text{shape}} + \\lambda_{c}\\,\\mathcal{L}_{\\text{colour}},\\quad \\lambda_s=\\lambda_c=0.2",
    "\\mathcal{L}_{\\text{ArcFace}} = -\\log \\frac{e^{s\\cos(\\theta_{y}+m)}}{e^{s\\cos(\\theta_{y}+m)} + \\sum_{j\\neq y} e^{s\\cos\\theta_j}},\\quad s=32,\\ m=0.3",
  ]),
  H2("5.4 Optimisation and hyperparameters"),
  table(["Setting", "Phase 1", "Phase 2"], [
    ["Trainable", "All but ViT and VGG blocks 1–3", "Everything"],
    ["Epochs (max) / batch (P × K)", "25 / 32 groups × 4 photos", "35 / 16 × 4"],
    ["Learning rate: backbones / new layers / ArcFace table", "3e-5 / 3e-5 / 1e-3", "1e-5 / 1e-5 / 1e-4"],
    ["Optimiser", "AdamW, weight decay 1e-3", "same"],
    ["Schedule", "ReduceLROnPlateau (×0.5, patience 3, min 1e-6)", "same"],
    ["Early stopping", "Patience 6 per phase; best weights restored", "same"],
    ["Model selection", "Top-1 on validation-enrolled", "same; best: phase 2, epoch 35"],
    ["Seed", "2026", ""],
  ], [3, 2.6, 2.4], { title: "Table 6. Training settings of the served model (run 4).", note: "Source: `configs/stage2_run4.yaml`; `outputs/runs/stage2_run4/summary.json`." }),
  P("**Why these rates.** The first run trained the new layers at 1e-3 and the embedding collapsed to about 3 effective dimensions (enough for colour and shape, not imprints; Fig. 5b). Using the original model's rates (3e-5 then 1e-5), a separate, faster rate for the randomly initialised ArcFace table, and the margin warm-up fixed it (12–15 effective dimensions)."),
  H2("5.5 Calibration (validation only)"),
  B("**Simulation.** 1,000 patients, each with 10 medicines drawn from validation-enrolled groups and 3 enrolment photos per medicine; queries are their other photos (own) and validation-unknown photos (foreign)."),
  B("**Threshold** `τ = 0.8949`: 95% of foreign pills fall below it. **Margin** `m = 0.115`: the largest with ≤ 10% retakes. **Crop window** 1.55× (sweep 1.3–2.0)."),
  ...figure("fig06_calibration", 6),
  H2("5.6 Imprint reading and re-ranking"),
  B("**Reader.** Florence-2-large, zero-shot OCR on the crop at four rotations (GPU); Florence-2-base on the CPU path. The settings below were re-chosen for each reader on validation."),
  B("**Catalogue re-ranking.** The top `K = 100` products by look get `+ α × imprint score`, with `α = 0.05` (grid on validation, Fig. 12b). The imprint score rewards read tokens that are informative in the catalogue."),
  B("**Gallery check.** In the patient's gallery, only tokens that distinguish one enrolled imprint from the others count (for `TV;58` next to `TV;7295`, reading “TV” says nothing), with a minimum token weight of 0.5."),
  H2("5.7 Calibrated confidence for identification"),
  P("A logistic model fitted on validation: `confidence = σ(−12.50 + 12.50·s1 + 72.13·(s1 − s2) + 3.92·imprint_read)`, where s1, s2 are the top two similarities. Verdict: `no_confident_match` if s1 < 0.9462 or confidence < 0.2; `possible_match` if s1 ≥ 0.9658 and confidence ≥ 0.5; otherwise `uncertain` (`src/epills/core/catalogue/confidence.py`)."),
  H2("5.8 Implementation and reproducibility"),
  B("Python 3.13.15, PyTorch 2.11.0 (CUDA 12.8), Hugging Face Transformers 5.19.0; one NVIDIA RTX 4080 SUPER (16 GB)."),
  B("Served as a FastAPI service on a Hugging Face ZeroGPU Space (API 0.10.0: https://davidpansx-pharmascan-api.hf.space, `/docs`); 92 automated tests (`outputs/evidence/pytest.txt`)."),
  TODO("Add training time per run (from the training logs or file timestamps) if the paper reports compute."),
  ...CODE([
    "cd epills",
    ".venv\\Scripts\\python scripts\\make_splits.py",
    ".venv\\Scripts\\python scripts\\train_stage2.py --config configs\\stage2_run4.yaml",
    ".venv\\Scripts\\python scripts\\calibrate_stage2.py --run outputs\\runs\\stage2_run4 --windows 1.3 1.4 1.55 1.7 1.85 2.0",
    ".venv\\Scripts\\python scripts\\evaluate_stage2.py --run outputs\\runs\\stage2_run4",
    ".venv\\Scripts\\python scripts\\tune_identify_rerank.py --run outputs\\runs\\stage2_run4 --ocr large_c92",
    ".venv\\Scripts\\python scripts\\evaluate_identify_topk.py --run outputs\\runs\\stage2_run4 --ocr large_c92",
    ".venv\\Scripts\\python scripts\\calibrate_identify.py --run outputs\\runs\\stage2_run4 --ocr large_c92",
    ".venv\\Scripts\\python scripts\\make_paper_figures.py",
  ]),
  H2("5.9 Base code that must be cited"),
  B("**The team's facial-expression project** (`FER/exp-tuning.ipynb`, Kaggle; FER2013 and RAF-DB): the hybrid architecture, its initialisation and its training schedule were ported from it to PyTorch."),
  B("Pretrained weights: VGG16 (ImageNet, torchvision) and ViT-B/16 (ImageNet-21k, `google/vit-base-patch16-224-in21k`); Florence-2 (Microsoft) for OCR."),
  TODO("Confirm the origin of the FER notebook's architecture: if it was adapted from a published paper or a public notebook, cite that source too (the rubric requires citing any base code)."),
  H2("5.10 Software 3.0 contribution to the approach"),
  B("The PyTorch port of the TensorFlow/Keras FER model (`hybrid.py`, including Keras-style initialisation), the training and calibration scripts and the serving code were written with Claude Code from natural-language instructions, then reviewed and run by the team."),
  B("The collapse of run 1 was diagnosed with an effective-dimension measure added to the training log at that time (`effective_dims` in `src/epills/training/eval/retrieval.py`); the fix (original learning rates, separate ArcFace rate, margin warm-up) was proposed, run and accepted on the validation numbers."),
  B("Proposals were accepted or rejected on measured validation results, not on the assistant's word: two reader fine-tunes it set up were rejected (Table 11)."));

// ---- 6. experiments ----
add(H1("6. Experiments (25%)"),
  H2("6.1 Protocol and metrics"),
  B("**Test sets, each scored once** with every setting fixed on validation: C3PI held-out (200 new medicines, C3PI conditions); MEDISEG (32 products, other phones); ePillID fold 4 (published benchmark)."),
  B("**Healthcare-relevant metrics.** For the patient's own pills: named correctly / named wrongly (the safety-critical error) / photograph again / called unknown. For foreign pills: rejected / accepted as theirs. Plus top-1 and top-k, AUROC, MAP and GAP (ePillID's metrics), and a reliability diagram for the stated confidence. 95% intervals by resampling medicines."),
  H2("6.2 Baselines"),
  B("**Run 1:** the same architecture trained with a common recipe (new layers at 1e-3, no warm-up)."),
  B("**Published ePillID results** [usuyama2020epillid]: the best multi-head model and the plain classifiers."),
  B("**Look only vs. look + imprint** for identification and gallery matching."),
  B("**Pill finder:** an earlier box finder found exactly one box on only 28% of single-pill photos; the stand-in finder counts 87.8% right (Fig. 17)."),
  ...figure("fig17_pill_finder", 17, 4.2),
  H2("6.3 Main results"),
  ...figure("fig07_main_results", 7, 5.8),
  table(["Test measure", "Run 2", "Run 3", "Run 4 (served)"], [
    ["C3PI held-out: top-1 among all 200 medicines", "82.0%", "84.4%", "83.2%"],
    ["C3PI held-out, gallery of 10: named correctly / wrongly", "78.6% / 0.1%", "80.4% / 0.0%", "76.0% / 0.04%"],
    ["C3PI held-out: foreign pills rejected", "94.2%", "94.1%", "94.8%"],
    ["MEDISEG: top-1 among 32 products (pills > 100 px)", "38.9% (51.0%)", "41.8% (56.8%)", "48.2% (62.2%)"],
    ["MEDISEG, gallery of 10: named correctly / wrongly / own called unknown", "17.6 / 1.1 / 51.5%", "18.7 / 1.0 / 64.2%", "18.7 / 0.7 / 70.4%"],
    ["MEDISEG: foreign pills rejected", "78.7%", "89.3%", "93.1%"],
    ["ePillID strict (no test type seen): MAP / MAP@1", "0.255 / 13.7%", "0.506 / 37.1%", "0.555 / 42.3%"],
  ], [3.6, 1.4, 1.4, 1.6], { title: "Table 7. Test results (medicines never seen in training; 10-medicine galleries, 3 enrolment photos).", note: "Run 4, C3PI held-out, 95% intervals: named correctly 73.4–78.6%, named wrongly 0.02–0.07%, foreign rejected 93.7–95.9%. Sources: `outputs/runs/stage2_run{2,3,4}/eval/{c3pi_holdout,mediseg,epillid}.json`; `docs/stage2_results.md`." }),
  ...figure("fig08_open_set_outcomes", 8),
  table(["Run 4 + imprint check, C3PI held-out", "GPU reader (Florence-2-large)", "CPU reader (Florence-2-base)"], [
    ["Gallery of 10: named correctly / wrongly", "78.3% / 0.06%", "77.3% / 0.06%"],
    ["Gallery: photograph again / own called unknown", "7.2% / 14.5%", "–"],
    ["Foreign pills: called unknown / accepted", "94.8% / 3.5%", "–"],
    ["Catalogue (4,936 products): right product first / in top 5", "80.2% / 94.4% (look only 71.6% / 91.7%)", "79.2% / 93.9%"],
  ], [3.4, 2.6, 2.2], { title: "Table 8. The deployed pipeline with the imprint check.", note: "Sources: `outputs/runs/stage2_run4/eval/imprint_gallery_large_c92_distinct.json`, `identify_topk_large_c92.json`, `identify_rerank_large_c92.json`; `docs/evidence.md`." }),
  H2("6.4 Ablations"),
  table(["Change", "Effect (validation unless stated)", "Kept?"], [
    ["Run 1 → 2: original learning rates, separate ArcFace rate, margin warm-up", "Gallery top-1 77.2% → 98.3%; AUROC 0.78 → 0.981; correct at 95% rejection 17% → 90.7%; effective dims ≈3 → 12–15", "Yes"],
    ["Run 2 → 3: + C3PI studio reference photos", "Validation unchanged (98.1%); test ePillID MAP@1 13.7% → 37.1%", "Yes"],
    ["Run 3 → 4: + photo-condition and crop-jitter augmentation", "Test MEDISEG top-1 41.8% → 48.2%; C3PI held-out 84.4% → 83.2%", "Yes (served)"],
    ["Refinement block's 64-d pooling (ePillID gap analysis)", "Features taken before the block score the same on ePillID: the pooling is not the bottleneck", "Mean pool"],
    ["Crop window 1.3–2.0× pill box", "Named correctly 86.1–87.2%; best 1.55×", "1.55×"],
    ["Flip margin 0–0.3", "Named correctly 87% → 54%, retakes 0% → 34%; wrong names ≈1% → ≈0%", "0.115"],
    ["Imprint re-ranking weight α, depth K (re-chosen on the catalogue instead of on ePillID)", "Validation top-1 72.1% (α = 0) → 81.6% (α 0.05, K 100); larger α hurts. Test: right product first 74.8% → 80.2%", "α 0.05, K 100"],
    ["Gallery imprint check: any shared token vs distinct tokens only", "Validation: named wrongly 35 → 30, foreign pills accepted 6,158 → 6,150, correct −0.03 points", "Distinct"],
    ["Runs 5–7: ePillID's official protocol (+ stronger fine-tuning; + domain-paired batches)", "ePillID MAP 0.57–0.60; MEDISEG 52–54% (best on other phones); no ePillID gain from 6 or 7", "Benchmark only"],
  ], [3, 4, 1.2], { title: "Table 9. Ablations.", note: "Sources: `docs/stage2_results.md`; `outputs/runs/*/history.csv`, `calibration.json`, `eval/identify_rerank_large_c92.json`; `PROGRESS_LOG.md`." }),
  ...figure("fig05_training_curves", 5),
  H2("6.5 Hyperparameter sensitivity"),
  P("Figures 6 (margin, crop window) and 12b (re-ranking weight and depth). The crop window matters little (±0.5 points); the margin trades correct names for retakes almost one for one while wrong names are already near zero at 0.05; the re-ranking weight has a clear optimum because a large weight lets false OCR reads promote wrong products."),
  ...figure("fig12_identify_topk", 12),
  H2("6.6 Robustness"),
  ...figure("fig09_robustness", 9),
  B("**Other phones (MEDISEG):** recognition drops (top-1 48% vs 83%) and the threshold set on C3PI photos rejects most own pills (70%), while wrong names stay below 1%: the system fails safe. Small pills are the main factor; more enrolment photos help only partly."),
  B("**CPU vs GPU reader:** the served settings fit both; on the CPU path identification is 79.2% vs 80.2% (Table 8)."),
  B("**Pills not in the catalogue:** 70% of MEDISEG pills get `no_confident_match` and 7% `possible_match` (Fig. 13b)."),
  ...figure("fig13_identify_confidence", 13),
  H2("6.7 Error analysis"),
  ...figure("fig14_similarity_distributions", 14, 4.2),
  ...figure("fig15_embedding_tsne", 15, 4.2),
  ...figure("fig11_imprint_reading", 11),
  B("Imprints are read on about 47% of photos (GPU reader); debossed, small and round pills are hardest, and C3PI's own raters marked many imprints unreadable (Fig. 11)."),
  ...figure("fig16_lookalike_error_analysis", 16),
  B("**Look-alike strengths** (same ingredient, other strength, both in one gallery): named as the other strength 0.11% overall, 0.58% when shape and colour are the same. Every catalogue product lists its look-alike strengths, and strength is read from the label instead."),
  B("**Same-manufacturer pairs** share imprint prefixes; the worst are Teva pairs such as tramadol 50 mg (TV;58) vs carvedilol 12.5 mg (TV;7295), named as the other 5.8% of the time even with the imprint check."),
  H2("6.8 The ePillID benchmark"),
  ...figure("fig10_epillid_benchmark", 10),
  table(["Two-sided, fold 4", "MAP", "GAP", "MAP@1", "GAP@1", "top-5"], [
    ["Paper best (ResNet-152 + CBP, multi-head)", "0.958", "0.812", "92.0%", "0.912", "–"],
    ["Paper, plain classifiers", "0.547–0.727", "0.241–0.448", "35.7–56.8%", "0.331–0.551", "–"],
    ["Run 5 (official protocol)", "0.597", "0.207", "45.7%", "0.526", "77.5%"],
    ["Run 5 + imprint re-ranking", "0.780", "0.598", "69.0%", "0.921", "89.6%"],
    ["Run 7 + imprint re-ranking", "0.774", "0.614", "67.6%", "0.916", "89.5%"],
  ], [3.2, 1, 1, 1, 1, 0.9], { title: "Table 10. ePillID, official protocol.", note: "The imprint variant uses information the paper's model did not use, so report it separately. Source: `docs/stage2_results.md` §4; `outputs/runs/stage2_run{5,6,7}_official/eval/`." }),
  P("**Interpretation for the paper.** Our embedding alone matches the paper's plain classifiers, not its best model; with imprint re-ranking GAP@1 beats the paper (0.921 vs 0.912) but MAP does not. The paper's own ablations attribute its gain to its training recipe (contrastive + hard-mined triplet + softmax + ArcFace over side-specific classes, 2,048-d embedding, per-pill batches), not the backbone; we ruled out our crop style, the 64-d pooling and local vs global pooling."),
  H2("6.9 Negative results"),
  table(["What was tried", "Result", "Decision"], [
    ["LoRA fine-tune of Florence-2-large, v1 (19,312 photos of training medicines)", "Validation imprints read 44.0% → 53.5%, but it wrote imprints memorised from training onto new pills, so identification and the gallery check got worse", "Rejected"],
    ["LoRA v2 (also learns to output nothing on unreadable photos and wrong rotations), epoch 1", "Validation own-imprint read 45.0% → 50.7%, identification 82.0% → 83.0%, but gallery wrong names 30 → 61", "Rejected (safety)"],
    ["Look-alike characters (O/0, B/8) treated as equal in imprint matching", "Measured on validation and dropped", "Rejected"],
    ["Scoring each candidate's imprint separately", "Measured on validation and dropped", "Rejected"],
    ["Runs 6–7: stronger fine-tuning, benchmark-aligned training", "Better validation and MEDISEG, not ePillID", "Not served"],
  ], [3, 4, 1.4], { title: "Table 11. What did not work.", note: "Source: `PROGRESS_LOG.md` (13–14 Oct 2026); `docs/blueprint_review.md`." }),
  H2("6.10 Interpretation"),
  B("A face-expression architecture transfers to pills when trained as an embedding with a careful recipe; the learning rate of new layers decided between collapse and success."),
  B("Calibration on validation medicines gives predictable behaviour on new medicines in the same conditions (wrong names 0.04%), but thresholds do not transfer across phones: deployment needs per-phone or per-patient calibration."),
  B("Text on the pill is complementary to its look: imprint reading adds 8.6 points to identification (71.6% → 80.2%); the right product comes first for 92.9% of pills whose own imprint is read, against 68.6% when it is not."),
  H2("6.11 Software 3.0 in coding and evaluation"),
  B("**Code.** Evaluation scripts (gallery simulation, look-alike and same-maker stress tests, top-k, confidence calibration, figure generation), the API and its 92 tests were written with Claude Code from natural-language instructions; the team ran them and checked results."),
  B("**Bugs caught by tests and review**, e.g. a release-form safety bug in the label reader (an extended-release “LA” sticker was linked to the immediate-release product; fixed by enforcing the release form), a unit-parsing bug (“capsules” → “capsul”), grapefruit juice in a caution line read as a medicine, and a model-version mismatch that hid the error rates in the live API."),
  B("**Guarding against AI over-reach.** Test sets were scored once; settings were chosen on validation; suggestions were adopted only on measured results (Table 11 lists rejected ones)."),
  TODO("Decide how to present tool-use evidence. Use the real record (`PROGRESS_LOG.md`, git history, the conversation transcripts if the course accepts them). Do not create mock logs."));

// ---- 7. conclusion ----
add(H1("7. Conclusion (5%)"),
  H2("Draft conclusion"),
  P("We built and deployed an open-set pill verifier for patients on several medicines. A hybrid CNN–ViT, originally built for facial expressions, learns a 256-d embedding that recognises medicines it never saw from three enrolment photos; calibrated thresholds keep wrong names at 0.04% while rejecting 95% of foreign pills, and imprint OCR raises catalogue identification to 80.2% top-1 with an honest confidence. The main limitation is the photo domain: on other phones the system mostly says “unknown”."),
  H2("Limitations"),
  B("One training source (C3PI, US pills photographed 2011–2016); thresholds do not transfer to new phones."),
  B("Small pills and debossed imprints; the imprint is read on about half of the photos."),
  B("No test on the team's own phone photos of real Singapore medicines; the catalogue is US-only."),
  B("ePillID MAP below the published best."),
  H2("Future work"),
  N("Calibrate per phone or per patient at enrolment."),
  N("Train with ePillID's multi-head recipe on our architecture."),
  N("Photograph real Singapore medicines and stickers as the final test."),
  N("A larger or better-adapted imprint reader that keeps wrong names low."));

// ---- 8. presentation & supplementary ----
add(H1("8. Presentation and supplementary material"),
  H2("8.1 Lectrace, 10 minutes, 3–5 execution states"),
  P("A suggested sequence using the live API (`/ui` demo page or `/docs`) and the first-slice script (`scripts/demo_first_slice.py`, trace in `outputs/evidence/first_slice_trace*.json`):"),
  N("**Detect.** A dose photo → `/v1/detect`: pill boxes and count.", "steps"),
  N("**Match.** The crop's embedding against a 4-medicine synthetic gallery → “looks like your metformin” (similarity 0.96).", "steps"),
  N("**Reject.** A warfarin pill that is not theirs → unknown, “set it aside”; `/v1/identify` lists warfarin 1 mg first with `possible_match` and its confidence.", "steps"),
  N("**Imprint.** Show the OCR read moving the right product up the list (Fig. 12a).", "steps"),
  N("**Label.** A pharmacy sticker → `/v1/read-label` structured MOH fields, then `/v1/safety/medications` for interactions.", "steps"),
  TODO("Record the walkthrough on the live site; the free GPU quota is about 2 minutes per day without a Hugging Face login, so rehearse on the CPU path or log in."),
  H2("8.2 Supplementary ZIP"),
  B("Code: `epills/src`, `scripts`, `configs`, `tests`, `serving`; `README` with the commands of §5.8."),
  B("Results: `outputs/runs/stage2_run4/{summary.json, calibration.json, eval/*.json}` and the other runs' summaries; `docs/paper/figures/`."),
  B("Not included: raw datasets (licences; give download instructions from `docs/datasets.md`) and model weights (private Hugging Face repository; give access on request)."),
  B("Cite the FER base code and all pretrained weights (§5.9)."));

// ---- appendices ----
add(H1("Appendix A. Figure files"),
  table(["#", "File (in docs/paper/figures/)", "Rubric section"], [
    ["1", "fig01_system_overview", "Problem formulation"], ["2", "fig02_architecture", "Approach"],
    ["3", "fig03_data_statistics", "Data"], ["4", "fig04_qualitative_examples", "Data"],
    ["5", "fig05_training_curves", "Experiments (ablation)"], ["6", "fig06_calibration", "Approach / hyperparameters"],
    ["7", "fig07_main_results", "Experiments"], ["8", "fig08_open_set_outcomes", "Experiments (healthcare metrics)"],
    ["9", "fig09_robustness", "Experiments (robustness)"], ["10", "fig10_epillid_benchmark", "Experiments (baselines)"],
    ["11", "fig11_imprint_reading", "Experiments (error analysis)"], ["12", "fig12_identify_topk", "Experiments (hyperparameters)"],
    ["13", "fig13_identify_confidence", "Experiments (calibration)"], ["14", "fig14_similarity_distributions", "Experiments (error analysis)"],
    ["15", "fig15_embedding_tsne", "Experiments (interpretation)"], ["16", "fig16_lookalike_error_analysis", "Experiments (error analysis)"],
    ["17", "fig17_pill_finder", "Experiments (system)"],
  ], [0.5, 4, 3.5], { note: "Each figure exists as .pdf (vector, for LaTeX) and .png (300 dpi). Captions and source files: `figures.json`." }),
  P("LaTeX example:"),
  ...CODE([
    "\\begin{figure}[t]",
    "  \\centering",
    "  \\includegraphics[width=\\linewidth]{figures/fig07_main_results.pdf}",
    "  \\caption{Test results, each scored once. ...}",
    "  \\label{fig:main}",
    "\\end{figure}",
  ]),
  H1("Appendix B. References"),
  P("All entries are in `docs/paper/references.bib`. Before submission, open each one's publisher page and confirm the authors, title, venue and year; entries with a VERIFY note need it most (author lists shortened to “and others” were not checked in full)."),
  table(["Key", "Reference", "Used for"], [
    ["usuyama2020epillid", "Usuyama et al., ePillID dataset, CVPR Workshops 2020", "Benchmark, related work"],
    ["yaniv2016nlm", "Yaniv et al., NLM Pill Image Recognition Challenge, AIPR 2016", "C3PI origin"],
    ["wong2017pill", "Wong et al., fine-grained pill identification with deep CNNs, J. Biomed. Inform. 2017", "Related work"],
    ["ling2020fewshot", "Ling et al., Few-shot pill recognition, CVPR 2020", "Related work"],
    ["mediseg2026", "MEDISEG dataset, City figshare 2026", "Cross-phone test"],
    ["simonyan2015vgg", "Simonyan & Zisserman, VGG, ICLR 2015", "Backbone"],
    ["woo2018cbam", "Woo et al., CBAM, ECCV 2018", "Attention"],
    ["dosovitskiy2021vit", "Dosovitskiy et al., ViT, ICLR 2021", "Backbone"],
    ["sabour2017capsules", "Sabour et al., dynamic routing between capsules, NeurIPS 2017", "Fusion"],
    ["vaswani2017attention", "Vaswani et al., Transformer, NeurIPS 2017", "Refinement block"],
    ["deng2019arcface", "Deng et al., ArcFace, CVPR 2019", "Loss"],
    ["wang2018cosface", "Wang et al., CosFace, CVPR 2018", "Related losses"],
    ["schroff2015facenet", "Schroff et al., FaceNet, CVPR 2015", "Metric learning"],
    ["hermans2017triplet", "Hermans et al., In defense of the triplet loss, arXiv 2017", "P×K sampling"],
    ["musgrave2020reality", "Musgrave et al., A metric learning reality check, ECCV 2020", "Evaluation"],
    ["scheirer2013openset", "Scheirer et al., Toward open set recognition, TPAMI 2013", "Open set"],
    ["hendrycks2017baseline", "Hendrycks & Gimpel, OOD baseline, ICLR 2017", "Rejection"],
    ["guo2017calibration", "Guo et al., On calibration of modern neural networks, ICML 2017", "Confidence"],
    ["platt1999probabilistic", "Platt, probabilistic outputs, 1999", "Logistic calibration"],
    ["xiao2024florence2", "Xiao et al., Florence-2, CVPR 2024", "Imprint OCR"],
    ["hu2022lora", "Hu et al., LoRA, ICLR 2022", "Reader fine-tune (negative result)"],
    ["vandermaaten2008tsne", "van der Maaten & Hinton, t-SNE, JMLR 2008", "Fig. 15"],
    ["loshchilov2019adamw", "Loshchilov & Hutter, AdamW, ICLR 2019", "Optimiser"],
    ["paszke2019pytorch", "Paszke et al., PyTorch, NeurIPS 2019", "Implementation"],
    ["xiong2022ddinter", "Xiong et al., DDInter, Nucleic Acids Res. 2022", "Interactions"],
    ["nelson2011rxnorm", "Nelson et al., RxNorm at 6 years, JAMIA 2011", "Drug names"],
    ["who2017medication", "WHO, Medication Without Harm, 2017", "Motivation"],
    ["moh_medication_label", "MOH Singapore, Know about medication label", "Sticker standard"],
    ["karpathy2017software2 / karpathy2025software3", "Karpathy, Software 2.0 (2017); Software is changing (again) (2025)", "Software 3.0"],
    ["anthropic_claude_code", "Anthropic, Claude Code", "AI assistant used"],
  ], [2.4, 4.2, 2]));

// ======================================================================================================================
const doc = new Document({
  creator: "PharmaScan team", title: "PharmaScan: materials for the Deep Learning paper",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: BLUE }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: "52514E" }, paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    ...["numbers", "steps"].map(reference => ({ reference, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      new TextRun({ text: "PharmaScan paper materials · ", size: 16, color: INK2 }),
      new TextRun({ children: [PageNumber.CURRENT], size: 16, color: INK2 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(buf => {
  const out = path.join(PAPER, "PharmaScan_paper_materials.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1e6).toFixed(1), "MB");
});
