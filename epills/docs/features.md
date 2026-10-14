# PharmaScan: feature backlog (Stages 2–3 and the API)

New capabilities agreed on 2026-10-09 (F7–F10 added 2026-10-12), so they don't get lost. The front end is a
website. The plan and decisions are in
[ROADMAP.md](../ROADMAP.md), progress in [PROGRESS_LOG.md](../PROGRESS_LOG.md), the API contract in
[serving/README.md](../serving/README.md).

| ID | Feature | Status | Owner |
|---|---|---|---|
| F1a | **Which of my medicines is this?** Photo → the patient's medication name (`/v1/embed`, `/v1/match`) | ✅ built and tested locally (2026-10-09) | Claude |
| F1b | **What is this pill?** Photo (+ imprint text) → candidate names from a catalogue of all known pills (`/v1/identify`) | ✅ built and tested locally | Claude |
| F2 | **What does this medicine look like?** Name → pictures of every product with that name (`/v1/pills/search`) | ✅ built and tested locally | Claude |
| F3 | **One-call scan:** full photo → every pill's box, name and decision (`/v1/scan`) | ✅ built; works with detector boxes; the stand-in finder only on plain backgrounds | Claude |
| F4 | Merge pill detection (Stage 1) into the recognition model | ⏸ paused: to discuss with the Stage 1 teammate | You + teammate |
| F5 | **Continual learning after deployment**, in line with Singapore law | 💭 only being considered (2026-10-10). A prototype exists but is not in use: `/v1/feedback` (off by default), `scripts/monthly_retrain.py` | You decide whether to adopt it |
| F6 | Run 4: training for other photo conditions + jittered crop windows | ✅ done 2026-10-09: MEDISEG 42% → 48%, ePillID MAP@1 37% → 42%; C3PI held-out 84% → 83% | Claude |
| F7 | **Imprint reading in the API:** the photo's imprint read automatically, in `/v1/identify` and as a check in `/v1/match` / `/v1/scan` | ✅ built 2026-10-12 (API 0.5.0); re-tuned for the GPU reader 2026-10-13 (0.9.0): test, new medicines: named correctly 76.0% → 78.3%, other pills accepted 3.8% → 3.5%; identify first choice 74.8% → 80.2% (re-ranking chosen on the catalogue) | Claude |
| F8 | **Box and label reading:** photo of the medicine's box or the pharmacy label → name, strength, RxNorm ID | ✅ built 2026-10-12 (API 0.5.0); prescriptions, PDFs, printed directions and list check added 2026-10-13 (0.7.0); real photos still to test | Claude |
| F9 | **Embedding contract + `/v1/verify`:** a written spec for the web team; a yes/no "is this photo medicine X?" call | 🟡 contract done 2026-10-12 (API 0.6.0: `embedding_space_id`, required versions, `/v1/model-info`, docs/API.md); `/v1/verify` not built | Claude |
| F10 | **Embeddings in the browser:** the website runs the model itself, so photos never leave the phone | 📝 planned (2026-10-12), not built; large | You decide |

---

## F1a. Which of my medicines is this? (pill → name, within the patient's list)
- The app's dose check. The patient enrols each medication ("Metformin 500 mg") with 3–5 photos; the app keeps
  the name and the embeddings. The API returns the name with the decision: `match`, `unknown` (set the pill
  aside) or `ambiguous` (flip the pill and retake the photo).
- `POST /v1/embed`: pill photo (or photo + box) → 256-number embedding. Used at enrolment and at every check.
- `POST /v1/match`: query embeddings + the patient's gallery (id, name, embeddings) → decision, name, similarity.
- Expected quality (run 3, 10 medications, 3 enrolment photos, fixed threshold): on new medicines in C3PI-style
  photos, 79% named correctly, 0.0% wrongly named, 95% of other pills rejected. Lower on other phones and
  conditions (MEDISEG), where it mostly rejects instead of naming wrongly.

## F1b. What is this pill? (pill → name, without enrolment)
- Compares the photo with a **catalogue of every pill we have pictures of**: 4,823 US products (NDC9), 1,239
  generic names, from C3PI's studio reference photos and everyday photos.
- `POST /v1/identify`: photo (+ optional typed imprint, e.g. `M 547`) → top 5 candidates with name, strength,
  manufacturer, imprint, shape, colour, picture and similarity.
- **Accuracy, honestly** (run 3; new everyday photos, C3PI conditions):
  - products with everyday photos in the catalogue (2,362 of 4,936): right product first 70% (never-trained
    types) to 89% (trained types); in the top 5: 93–98%;
  - products known only from studio photos: right first about a third of the time, top 5 about two-thirds
    (ePillID result);
  - other phones and conditions are lower (MEDISEG).
  So the answer is always **"possible matches, check with your pharmacist"**, never a single definite name.
- The typed imprint narrows the candidates to products whose imprint matches; when the imprint is readable this
  is close to exact (every catalogue product has its imprint recorded).
- Limits: US products only (C3PI); a medicine outside the catalogue can only be "not found".

## F2. What does this medicine look like? (name → pictures)
- `GET /v1/pills/search?name=metformin` → every catalogue product containing that medicine: name, strength,
  manufacturer, imprint, shape, colour, size, and picture links. `GET /v1/pills/{id}` → one product;
  `GET /v1/pills/{id}/image/{n}` → the picture.
- Names are matched through RxNorm (brand ↔ generic: "Glucophage" finds metformin) and directly in C3PI's names.
- **One name → many different-looking pills** (each manufacturer and strength); the app should show all, with a
  strength filter.
- Pictures: RxImage studio photos (both sides, with size ruler) and C3PI challenge photos, served by the API.
- **Licence:** C3PI images are "for research and development purposes only": fine for our non-commercial
  prototype; a public product would need another image source (e.g. DailyMed label images).

## F3. One-call scan
- `POST /v1/scan`: full photo (+ optionally the gallery, boxes from a detector, the dose schedule) → for each pill:
  its box, decision and name (or catalogue candidates); with a schedule, also the dose-check alerts.
- Inside: find pills → crop each from the full-resolution photo with the crop contract (a square window around
  the tight pill box, its size chosen on validation for each model; run 3: 1.4× the long side) → embed → match →
  (dose rules).
- Measured on CPU: about 0.15 s per pill after a one-off model load.
- Until the Stage 1 detector exists, a **stand-in pill finder** works on plain backgrounds (pills on a mat or
  table, not touching). On MEDISEG's cluttered photos it found 1 of 40 pills, so the real detector is needed.
  The app can also send the detector's boxes; then the finder is skipped.

## F4. Merge detection into the recognition model (paused)
Paused on 2026-10-09 until you talk with the Stage 1 teammate. Claude's recommendation was **not** to merge the
networks:
- **Resolution:** one network sees the whole photo shrunk to ~640 px, so a pill is 50–100 px. MEDISEG measured
  the effect: pills under 100 px are named right 30% of the time, over 100 px 57%. Two stages crop each pill
  from the full-resolution photo.
- It would replace the FER architecture (VGG16+CBAM+ViT is a recognition network, not a detector).
- Joint training needs photos with boxes *and* identities; at scale only MEDISEG's 32 products have both.
- The goal behind it (one step for the app, robustness to the detector's boxes) is met by F3 (one call) and F6
  (training with jittered crop windows).

**To agree with the teammate:**
- Box format to send (pixel coordinates in the full-resolution photo) and who crops (the API, with F3's framing).
- Detector type: class-agnostic "pill" boxes are enough; identity comes from Stage 2.
- Where it runs (phone or server) and its licence (Ultralytics YOLO is AGPL-3.0; torchvision detectors are BSD).
- Training data for a pill detector: MEDISEG boxes (14k), the Kaggle VAIPE copy, and ~40k C3PI photos with the
  cropper's boxes.

## F5. Continual learning after deployment, in line with Singapore law
**Goal:** keep adding new medicines and pictures, and keep improving recognition, after deployment.

### Design: two levels
**Level 1: new medicines without retraining (immediate, automatic).** The model compares embeddings, so a new
medicine needs no new training: adding it means adding photos.
- Patient level: enrolment photos on the phone (as now).
- Catalogue level (F1b, F2): new products added from verified sources (pharmacist-taken photos, official
  images), each addition logged, the catalogue versioned and reversible.

**Level 2: better recognition by periodic retraining (scheduled, human-approved, never silent).**
1. **Collect:** only photos the user agreed to share for improvement (a separate opt-in), as pill crops; each
   with how its label was confirmed (pharmacist verified, user confirmed, or model only).
2. **Curate:** only verified labels enter training; flagged errors (overrides, "wrong pill" reports) are reviewed
   first; near-duplicates and leakage checked as for C3PI.
3. **Retrain** on a schedule (e.g. quarterly) with the same pipeline (train → calibrate → test).
4. **Gate:** the new model must not be worse than the current one on fixed test sets, especially the safety
   numbers (wrong-pill acceptance, unknown rejection); results go in a model card and change log.
5. **Approve and release:** a named person signs off; versioned release, staged rollout, one-step rollback.
6. **Monitor:** flip, unknown and override rates, and drift per app version; incidents reported.

**Not done: fully automatic online learning** (the model updating itself from every user answer). Users'
answers can be wrong or malicious, performance could drift silently, and Singapore's regulators expect
validated, documented changes (below).

### Singapore requirements to design for
- **PDPA (Personal Data Protection Act):** consent before using photos for model improvement (separate from
  using the app); use only for the purpose consented; tell users what is collected; let them access and correct
  it and withdraw consent (their photos then leave future training sets); keep data accurate; protect it
  (encryption, access control); keep it only as long as needed; **transfer limitation**: data leaving Singapore
  (e.g. Hugging Face servers) needs comparable protection; accountability; data-breach notification.
  Minimise: store pill crops only, strip location (EXIF), no faces or labels with names.
- **HSA, GL-04 Revision 4** (Software Medical Devices incl. machine-learning devices, a life-cycle approach,
  December 2025): ML updates go through structured change control; a **Change Management Program** allows
  pre-approved (pre-specified) updates to ship faster, with records and annual declarations; post-market
  monitoring and documentation of model behaviour and updates. Applies if PharmaScan counts as a medical device.
- **AIHGle 2.0** (MOH and HSA AI in Healthcare Guidelines, March 2026): total product life-cycle management,
  post-deployment monitoring, incident reporting and feedback channels, and documented roles for the developer,
  the deployer (e.g. a clinic or pharmacy) and the users; continuous-learning AI is covered as an emerging area.

### Open questions (for you and your supervisor)
- Does PharmaScan count as a medical device under HSA? Verifying pills and giving safety alerts may make it one.
- Where will data live? Hosting in Singapore avoids most transfer questions.
- Who verifies labels (pharmacists?), and the exact consent wording.

### What the prototype can include later
Level 1 (catalogue additions with a log), the consent flag and label-source field on stored crops, model and
catalogue version numbers in every API answer (already planned), and the regression gate as a script.

## F6. Run 4: other photo conditions + jittered crop windows
Run 3 plus training augmentation for what the test sets showed is missing:
- **Jittered crop windows:** the crop shifted and resized the way a detector's box errs (tighter, looser,
  off-centre), so recognition no longer needs perfect boxes.
- **Photo conditions:** stronger brightness and contrast, mild white-balance shifts, downscale-then-upscale (small
  pills), JPEG compression, sensor noise, and another pill partly in the frame.
- Same calibration and test sets afterwards; compared with run 3.

## F7. Imprint reading in the API (built 2026-10-12, API 0.5.0)
The imprint is the strongest clue between look-alikes (white round tablets). The text reader (Florence-2-base,
`src/epills/core/imprint/ocr.py`) reads the pill region of each crop in 4 rotations; on the Space it runs on the lent
GPU, with the CPU as fallback.
- **`/v1/identify`** (`read_imprint`, default on; skipped when the user types the imprint): the closest 500 catalogue
  products are re-ranked by how well their recorded imprint matches the read (`imprint.match.rerank`, alpha 0.3;
  re-chosen on validation for run 4, same as run 5). ePillID test, run 4: right product first 42% → **66%** (front +
  back), 33% → **51%** (one photo); MAP 0.555 → 0.759.
- **`/v1/match` and `/v1/scan`:** a gallery medication may carry its `imprint` (from `/v1/pills/search`). The read
  checks the look's decision (`inference.matching.check_imprint`): `confirmed`, `resolved` (a tie between two of the
  patient's medicines settled by the imprint), `conflict` (another medicine's imprint read: `match` becomes
  `ambiguous`, photograph again). An imprint never accepts a pill the look rejected. A read counts when its token is
  rare enough across the catalogue (`GALLERY_MIN_WEIGHT` 0.5, chosen on validation).
- **Measured** (`scripts/evaluate_imprint_gallery.py`, run 4, galleries of 10, 3 enrolment photos):

  | | Named correctly | Named wrongly | Turn-over requests | Other pills accepted |
  |---|---|---|---|---|
  | Validation, look only | 78.3% | 0.05% | 9.9% | 3.29% |
  | Validation, + imprint | **80.4%** | 0.04% | 7.9% | **3.07%** |
  | Test (new medicines), look only | 76.0% | 0.04% | 9.5% | 3.77% |
  | Test, + imprint | **77.0%** | 0.06% | 8.4% | **3.54%** |

  On the test, the imprint confirmed 26% of the patient's pills, settled 2.7% of ties and raised 1.6% of conflicts.
- **Honest limits:** the pill's own imprint is readable on only **38%** of photos (printed far better than engraved),
  so the imprint helps but can't make recognition near-perfect on its own. Small pills, blur and engraving are the
  hard cases. Measured on C3PI-style photos; not yet on other phones.
- **Cost:** about 0.3 s per pill on the GPU, seconds per pill on CPU; a second GPU call per request.
- **Fine-tuning the reader on C3PI's imprint text:** tried on 13 Oct (below).
- **Re-tuned 2026-10-13 (API 0.9.0)** for the reader on the GPU since 0.8.0 (Florence-2-large on the central 92% of
  the crop as the API frames it; `scripts/read_imprints_cache.py --preset large_c92`). The settings chosen on
  validation stayed the same (alpha 0.3, top 500, `GALLERY_MIN_WEIGHT` 0.5); the identify confidence was refitted.
  The pill's own imprint is now read on **47%** of test photos (was 38%).
  - ePillID test: right product first 42% → **73%** (front + back), 33% → **54%** (one photo).
  - Galleries of 10, test: named correctly 76.0% → **78.3%**, turn-over requests 9.5% → 7.2%, other pills accepted
    3.77% → 3.51%, named wrongly 0.04% → 0.06%.
  - `/v1/identify` (C3PI held-out medicines): first candidate right 71.4% → 74.8% with the old weights; then the
    re-ranking was chosen again on the catalogue itself (`scripts/tune_identify_rerank.py`; it had been chosen on the
    ePillID benchmark, where false reads hurt less): alpha 0.05, top 100. Test: first right **80.2%**, top 5
    **94.4%** (was 74.8% / 91.7%); photos whose imprint isn't read 58.5% → 68.6%. Identify confidence refitted:
    `possible_match` right 84% → 87%, non-catalogue pills called `possible_match` 9% → 7%.
  - **Fine-tuning the reader (job "imprint first", 13 Oct).** A LoRA adapter on Florence-2-large, trained on 19,312
    photos of training medicines (`scripts/train_imprint_reader.py`, `core/imprint/lora.py`; switched on for imprint
    reads only). v1 read more imprints (validation 44.0% → 53.5%) but *memorised* the training medicines' imprints
    and wrote them on new pills (another pill's imprint read 0.96% → 2.43%; a training-set token on 76% of held-out
    photos), so identify got worse (validation 79.2% → 76.0% at the best weight) and so did the gallery check
    (named wrongly 30 → 71): rejected. Two cheaper ideas also failed: treating look-alike characters as equal, and
    scoring each candidate's imprint instead of reading (+1 to 2 points, not worth a second GPU call). v2 also learnt
    to write nothing on unreadable photos and wrong rotations (one epoch, 14 Oct): fewer false reads than the served
    reader on 400 validation photos (1.0% vs 1.3%) and identify a little better (test: first right 80.2% → 81.2%,
    top 5 94.4% → 95.1%), but the gallery check got worse on validation (named wrongly 30 → 61, other pills
    accepted 6,150 → 6,361): **not adopted**. It would only run on the GPU path anyway, which the free per-caller
    quota limits to the first minutes of each day. The served reader stays (`imprint_gallery_large_lora_v2_distinct.json`,
    `identify_rerank_large_lora_v2.json`).
  - **Same-maker imprints (blueprint B27).** 3,149 pairs of different medicines from one maker sharing an imprint
    part, both in one gallery: named as the other 0.01%; Teva "93" pairs 0.00%. The worst pair is tramadol 50 mg
    (TV 58) next to carvedilol 12.5 mg (TV 7295): 5.8% of its photos. Before, reading only the shared part ("TV")
    *confirmed* a wrong look-match (36% of the wrong names). Now a match is confirmed or contradicted only by a part
    that tells the patient's medicines apart (`GALLERY_CONFIRM_DISTINCT`; status `unresolved` when only shared parts
    were read). On validation: named wrongly 35 → 30, other pills accepted 6,158 → 6,150, named correctly 81.49% →
    81.46%. Counting only distinct parts everywhere (also to settle ties) was measured and rejected: it accepted more
    pills that aren't the patient's (`imprint_gallery_large_c92_distinct_all.json`).

## F8. Box and label reading (built 2026-10-12, API 0.5.0)
For medicines the pill catalogue doesn't have (e.g. Singapore-only brands), and to add a medication without typing.
- `POST /v1/read-label` (photo of a manufacturer's box *or* a pharmacy dispensing label): the same Florence-2 model
  reads the lines of text (OCR with regions), then `src/epills/core/label.py` finds the medicines in them:
  1. **names:** RxNorm's 23,500 ingredient and brand names (fetched once), plus international names RxNorm lacks
     (paracetamol → acetaminophen, salbutamol, co-amoxiclav, …); phrases of 1–4 words, longest first; a misread
     word of 6+ letters still matches (one OCR error; two from 9 letters); everyday words that are also RxNorm names
     ("water", "relief") are skipped. RxNorm's own match scores are *not* used to find names: "Take 2 tablets 4 times
     a day" scores higher there than "PARACETAMOL 500MG TABLET";
  2. **strength** printed after the name, or on the next line;
  3. **product:** RxNorm's concept for name + strength + the printed dose form (tablet, capsule, …), kept only if its
     ingredients and strength agree;
  4. a brand and its ingredients merge into one candidate (Panadol Extra = paracetamol + caffeine).
- Returns `{name, brand, strength, ingredients, rxcui, rxnorm_name, found_in}` per candidate, best first; the app
  sends `{id, name, rxcui}` to `/v1/safety/medications` after the user confirms.
- **Checked** on printed examples (a Singapore-style dispensing label, Lipitor, metformin, Augmentin and Panadol Extra
  boxes, a misread "Atorvastatln", co-amoxiclav, a cream at 1%, lines with no medicine): all named correctly, with
  the tablet product where the form is printed. **Not yet tested on real phone photos** of boxes and labels.
- **Limits and safeguards:**
  - RxNorm is American: Singapore-only brands may not be found, so the ingredient names printed on the box carry it.
  - Glossy or curved boxes, small print and photos from far away cause misreadings: the user always confirms.
  - A dispensing label shows the patient's name: the API stores nothing and returns only the lines that name a
    medicine.
  - A few seconds per photo: the reader, then RxNorm lookups (cached).

## F9. Embedding contract and `/v1/verify`
- **Contract (a section in docs/API.md):**
  - an embedding is 256 float32 numbers of unit length, made only by the API (`/v1/embed`);
  - two embeddings are compared by their dot product (= cosine similarity), and only within one `model_version`;
  - the pill is framed by the crop contract (a square window of 1.55 × the pill's long side for run 4; the server
    frames it when sent a box);
  - the decision rule: `match` when the best similarity ≥ threshold *and* it beats the runner-up by ≥ margin (both
    returned by `/v1/match`);
  - when `model_version` changes (or a `409`), re-embed the stored enrolment photos.
- **`POST /v1/verify`:** photo (+ box) + one medication's embeddings → `same`, `not_same` or `unsure`, with the
  similarity and the threshold. It is a one-medication `/v1/match`. Without a runner-up there is no margin, so it
  needs its own calibration on validation (accept rate for wrong pills ≤ the gallery rate).

## F10. Embeddings in the browser (the website runs the model)
- Export the model to ONNX and run it with ONNX Runtime Web: WebGPU on recent Chrome, Edge and Safari, or WebAssembly
  (slow) elsewhere. Matching against the gallery is then just dot products in JavaScript.
- **Gains:** pill photos never leave the phone (less personal data processed on our server, which helps with the
  PDPA); no use of the server's GPU allowance; matching works offline.
- **Costs:**
  - a ~200 MB download (fp16; ~100 MB if quantised to int8), cached after the first visit and repeated for every
    new model;
  - several seconds per pill on older phones;
  - the browser's embeddings must equal the server's (test: cosine ≥ 0.999), or stored galleries stop matching.
    Quantisation can shift them, which means recalibrating;
  - CBAM and capsule layers need checking for ONNX export.
- **Recommendation:** later, and only if privacy is a selling point; keep the server path as the reference.

---

Sources (checked 2026-10-09):
[HSA GL-04 Revision 4 overview](https://www.pureglobal.com/news/revision-4-of-gl-04-singapore-enhances-oversight-of-machine-learning-enabled-software-medical-devices),
[HSA digital health](https://www.hsa.gov.sg/medical-devices/digital-health/),
[MOH AI in healthcare / AIHGle 2.0](https://moh.gov.sg/licensing-and-regulation/artificial-intelligence-in-healthcare),
[AIHGle 2.0 summary (Baker McKenzie)](https://www.bakermckenzie.com/en/insight/publications/2026/03/singapore-moh-and-hsa-launch-refreshed-ai-in-healthcare-guidelines),
[PDPC advisory guidelines on key concepts](https://www.pdpc.gov.sg/-/media/Files/PDPC/PDF-Files/Advisory-Guidelines/overview-of-dp-provisions---ch-10-(270717).pdf),
[PDPA obligations overview](https://www.aseanbriefing.com/news/understanding-personal-data-protection-in-singapore/).
