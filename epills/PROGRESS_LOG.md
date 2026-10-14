# ePills Progress Log

Newest entry first. The plan, gates and decisions are in [ROADMAP.md](ROADMAP.md); dataset details are in
[docs/datasets.md](docs/datasets.md). Update the status board below whenever something changes.

---

## Where we are (2026-10-13)

**Run 4 is the served model** (runs 5 and 6 showed no clear gain). The API is live on the free ZeroGPU Space
(v0.8.0). The team's agent blueprint is the plan of record (docs/blueprint_review.md): Stages 1 and 2 shipped as
v0.6.0 and v0.7.0, and the before-submission jobs B21–B27 are done. B24–B27 and the imprint re-tune for the GPU
reader are built and tested as **v0.9.0, not yet deployed**. **Submission: 16 October.**

Older summary (2026-10-09): run 3 was then the best model (84% top-1 over 200 held-out medicines; MEDISEG 42%:
weak on other phones and conditions).

## Plan at a glance

| Phase | What | Status | Next action | Owner |
|---|---|---|---|---|
| P0 Setup | Decisions, environment, project layout | ✅ done | — | — |
| P1 Data acquisition | Download and check the datasets | 🟡 C3PI, ePillID, DDInter done | MEDISEG download (small); own pill photos; DDInter email | Claude; You |
| P2 Data preparation | Metadata, crops, dedup, splits, labels | ✅ done for C3PI | MEDISEG and own-pill crops when those arrive | Claude |
| P3 Model | FER hybrid in PyTorch, losses, sampler, training script | ✅ done; 12 tests pass | — | — |
| P4 Training | Baseline run, then improvements and ablations | ✅ run 3 (+ references) is the best model (2026-10-08; `outputs/runs/stage2_run3/best.pt`); run 2 meets all validation targets; run 1 collapsed (fixed) | Run 4 (domain augmentation), when you decide | Claude runs it when you say; you read the results |
| P5 Open-set calibration | Rejection threshold and flip margin on validation | ✅ 2026-10-08: threshold 0.793, margin 0.170 (`calibration.json`) | Re-run for every new model | Claude |
| P6 Evaluation | ePillID official (fold 4), MEDISEG, own pills | 🟡 runs 2 and 3 scored on C3PI holdout, ePillID, MEDISEG (2026-10-08) | Own pills (G5) when photos exist; score every new model | Claude; you for own pills |
| P7 Stage 3 safety engine | DDInter + RxNorm rules, dose checks, openFDA | 🟡 built and tested 2026-10-08 | Pharmacist or supervisor review of the wording | You |
| P8 Imprint OCR | TrOCR tie-breaker (stretch goal) | ⬜ optional | — | — |
| P9 Write-up | Report, tables, figures | ⬜ last | — | You; Claude drafts |
| P10 API and hosting | FastAPI on Hugging Face Spaces (free) | ✅ live: https://davidpansx-pharmascan-api.hf.space, API 0.9.0 since 13 Oct (free ZeroGPU Space; `/docs`, demo `/ui`); the Space's token is read-only | Share docs/API.md and docs/evidence.md with the team | You |

**Up next, in order:**
1. Choose the model to serve: run 3 (best in C3PI conditions) or run 4 (better on other phones, safer there).
   Next modelling step: make the threshold transfer to new phones (it now rejects most own pills there).
2. ✅ API deployed (2026-10-10). Replace the Space's `HF_TOKEN` with a read-only token, and revoke the write token
   that was pasted in chat (docs/DEPLOY.md §2).
3. F4 (merging detection into the model): your talk with the Stage 1 teammate; also agree the box format.
4. Own pill photos (G5): skipped for now; still the deciding test when photos exist.

New features and their status: [docs/features.md](docs/features.md).

**Is the model usable? Targets on validation** (proposed 2026-10-08 for the prototype; agree them with your team):

| Metric (printed each epoch) | Meaning | Target |
|---|---|---|
| `gallery-10: top1` | right medicine picked from a patient-sized list of 10 | ≥ 95% |
| `unknown AUROC` | tells "one of yours" from "not yours" in that list (0.5 = chance) | ≥ 0.9 |
| `correct@95%-reject` | real pills accepted and named correctly when 95% of unknown pills are rejected | ≥ 80% |
| `wrong_accept_95_gallery10` (in `history.csv`) | real pills accepted with the wrong name: the dangerous error | ≤ 2% |
| `dims` | effective embedding dimensions; a drop toward 1–5 means collapse (run 1: ~3) | stays well above 5 |

All open-set numbers are measured per patient-sized gallery of 10 (from run 2 on). `history.csv` also keeps the
harsher versions against all 214 validation medicines (`auroc_unknown`, `known_correct_95`, `wrong_accept_95`).

## Status board

| Gate | What | Status | Next step | Owner |
|---|---|---|---|---|
| G1 | ePillID benchmark | ✅ passed 2026-10-07: downloaded and checked; holdout = fold 4; published metrics are two-sided | — | — |
| G2 | C3PI training photos | ✅ passed 2026-10-07 (your sign-off): 40,587 crops (199/200 clean); split by appearance group; leakage report clean; 1,710 training groups, median 15 crops | — | — |
| G3 | Attribute labels (shape, colour, size) | ✅ passed 2026-10-07: shape, colour and size for 100% of the 2,362 C3PI medicines | ePillID reference-only types: only if we use ePillID references in training | Claude |
| G4 | Cross-domain test | ✅ MEDISEG downloaded and checked 2026-10-08 (MD5s match; 32 products, 14,069 pills; Deploy set); scored with run 2. VAIPE dropped | If MEDISEG is also used for training: split it by product first | Claude |
| G5 | Own pills | ⏸ skipped for now (no photos yet, 2026-10-08) | Choose the pills; get the mat; photo sessions | You |
| G6 | Safety data | 🟡 DDInter severity in hand; advice text and duplications missing | Email the DDInter authors; RxNorm mapping | You; Claude |

| Decision | Status |
|---|---|
| D1 framework | ✅ PyTorch (2026-10-07): a faithful port of the FER model |
| D2 CNN branch | ✅ VGG16 + CBAM from the FER model (2026-10-07) |
| D3 earlier-hybrid parts as extra arms | ✅ superseded by D2: the main model *is* the earlier hybrid (CBAM, capsule fusion, refinement) |
| D4 measured inputs (size, colour) on public data | ⬜ open; the baseline runs without them (recommendation: metadata size + noise as an ablation) |
| D5 cross-domain set | ✅ MEDISEG (2026-10-07); VAIPE copies (Roboflow, Kaggle) only as Stage 1 extra training data |
| D6 DDInter "Unknown" severity | ✅ on screen only (2026-10-08): its own level `ungraded`, grey, not spoken |
| D7 C3PI studio reference photos in training | ✅ yes, as run 3 (2026-10-08): 6,198 crops of training groups only. ePillID MAP@1 12% → 37%; kept |
| D8 hosting | ✅ Hugging Face Spaces; crops sent to the API are accepted for the non-commercial prototype (2026-10-07). Free CPU Spaces now need PRO, so a free ZeroGPU Gradio Space serves the same API (2026-10-10) |

## Who does what

| Area | Claude | You |
|---|---|---|
| **Decisions** | Lays out the options and recommends one (ROADMAP §6) | Decides |
| **Data** | Download scripts, cropping, dedup, splits, leakage reports | Reviews samples and reports; collects and photographs own pills |
| **Access requests** | Drafts the emails | Sends them (DDInter authors); creates accounts if needed |
| **Model and training** | Model, losses, training code, configs; short tests only | Says when a full run starts; reads the results with Claude |
| **Evaluation** | Evaluation scripts, results tables, failure analysis | Agrees the targets; checks the results |
| **Stage 3** | Rule engine, DDInter ↔ RxNorm mapping, openFDA lookup, tests | Gets a pharmacist or supervisor to review alert wording |
| **API** | FastAPI app, Docker, Hugging Face Space setup | Creates the Hugging Face account and model repo; shares the API with the app team |
| **Write-up** | Drafts sections, tables and figures | Owns the report; checks course requirements |
| **Other stages (1, 4)** | Advice and briefs for teammates | Guides the teammates |

Everything runs on the desktop (RTX 4080 SUPER, Windows). The ePills Python environment is `epills/.venv`
(Python 3.13, PyTorch with CUDA); no WSL needed. Full training runs only start when you say so.

Note on older entries: before 2026-10-07 (night) the cropper lived in `src/epills/datasets/`; it is now
`src/epills/training/datasets/`. Their "Next" lists are history; the current plan is the table above.

---

## 2026-10-14: reader v2 decided; the CPU path checked

- **Reader v2 (epoch 1) not adopted.** Full re-read of every validation, held-out and ePillID photo, then the same
  choices on validation. Identify a little better (test: first right 80.2% → 81.2%, top 5 94.4% → 95.1%), but the
  gallery check worse on validation (named wrongly 30 → 61, other pills accepted 6,150 → 6,361). It would only run
  on the GPU path anyway. The served reader stays; no more fine-tuning before the cut-off.
- **The CPU path checked** (most live calls run there: the free GPU quota is per caller). New reads cache
  `base_c92` (Florence-2-base, framed as served). Validation chose the *same* settings as for the GPU reader
  (identify alpha 0.05, top 100; gallery 0.5), and the GPU-fitted confidence rule holds (stated 0.96 → 94% right,
  0.70 → 75%, 0.51 → 48%). Test: identify first right 79.2% (top 5 93.9%; it would be about 76% with the old
  weighting); galleries named correctly 76.0% → 77.3%, named wrongly 0.06%, other pills accepted 3.77% → 3.51%.
  So no code change; `/v1/model-info` gains `personal_gallery_with_imprint_cpu` and `catalogue_identify_cpu`.
- New: `scripts/calibrate_identify.py --served` (scores the served rule on any reads),
  `scripts/evaluate_reader_candidate.sh`.

## 2026-10-13 (evening): the rest of the blueprint's API jobs; identify re-tuned; reader fine-tune tried

- **Your decisions:** retraining is fine until 15 Oct, never after the submission; imprint reading first; finish every
  API item from the blueprint and the team's test plan before pushing.
- **API items finished (v0.9.0, not deployed; 87 tests pass):** B15 (withdrawal squashes the dataset repo's history
  and checks before saying `deleted`; audit log; user labels need review: `monthly_retrain.py review`), B17
  (`POST /v1/search-by-embedding`), B19 (`POST /v1/check-input`, tool `route_medication_image`), test case M02
  (`look_alike_strengths` on every product, a strength note in identify), co-amoxiclav and other international names
  in `/v1/pills/search`, the first slice end to end (`scripts/demo_first_slice.py`, 30 calls in 26 s), I01 top-k
  recorded (`scripts/evaluate_identify_topk.py`), and **docs/evidence.md** for the judges (every test case → evidence).
- **Identify re-tuned (no training):** the imprint re-ranking had been chosen on the ePillID benchmark; chosen again
  on the catalogue itself (`scripts/tune_identify_rerank.py`): alpha 0.05, top 100. Held-out medicines: right product
  first 74.8% → **80.2%**, top 5 91.7% → **94.4%**. Confidence refitted: `possible_match` right 87%, non-catalogue
  pills called `possible_match` 7%.
- **Reader fine-tune:** a LoRA adapter on Florence-2-large (`scripts/train_imprint_reader.py`, 19,312 photos of
  training medicines). v1 read more imprints (validation 44.0% → 53.5%) but wrote imprints it had memorised on new
  pills, so identify and the gallery check got worse: **rejected**. Also measured and dropped: look-alike characters
  (O/0, B/8) and scoring each candidate's imprint. v2 (also learns to write nothing on unreadable photos and wrong
  rotations) is training: epoch checks about 20:50 and 23:45; adopted only if catalogue identify on validation
  improves, then the full re-read and re-tune.
- **Deployed 0.9.0 (20:07, after your `hf auth login`; the Space kept its read-only token).** The live check found
  one bug of mine: `/v1/model-info` served no error rates (the model's version is `stage2_run4@2026-10-09`, not
  `stage2_run4`); fixed, tested against the catalogue's version, redeployed at 20:09. The first slice on the live
  API: the same answers, 30 calls in 23 s. Speed: every call under 3.5 s on the CPU path; the GPU was granted once
  (identify 1.8 s), then refused: Hugging Face's ZeroGPU quota belongs to the caller (2 minutes a day without a login),
  so most calls run on the CPU. Documented in docs/API.md ("Speed").
- **Reader v2, epoch 1 (400 validation photos):** own imprint read 45.0% → 50.7%, another pill's read 1.3% → 1.0%
  (learning to write nothing worked), identify first right 82.0% → 83.0% (small; could be noise). Paused at 20:35 on
  your request (no overnight training); epoch 2 dropped, epoch 1 kept (`outputs/imprint_reader/large_lora_v2/
  adapter_e1.pt`).
- **Next morning:** full check of v2 epoch 1 (~1.5 h): copy `adapter_e1.pt` to `adapter_best.pt`, then
  `read_imprints_cache.py --preset large_lora_v2`, `tune_identify_rerank.py --ocr large_lora_v2`,
  `evaluate_imprint_gallery.py --ocr large_lora_v2`; adopt only if identify (test) beats 80.2% and the gallery check
  doesn't name more pills wrongly. Then decide whether a full 2-epoch run (~6 h, daytime) is worth it before the
  15 Oct cut-off.
- **Still open:** the team's TEVA photo; real label photos (B8).

## 2026-10-13 (later): B24–B27, imprint re-tune for the GPU reader; API v0.9.0 built, not deployed

- **v0.8.0 (deployed last night):** B21 (`text_reading` status, no silent fallbacks), B22 (embeddings checked;
  `500 model_error`), B23 (input check: blisters, bottles, labels abstain). Every call under 3.5 s on the CPU path.
  Its docs were missing from docs/API.md; added today.
- **Overnight reads finished this morning** (the job had stopped at 23:53; resumed, nothing lost):
  Florence-2-large reads of all 7,111 validation, 4,005 held-out and 745 ePillID crops, framed as served
  (`data/interim/ocr/large_c92/`).
- **Re-tune for the GPU reader:** own imprint read on 47% of test photos (was 38%). The settings chosen on validation
  stayed the same (gallery 0.5; identify alpha 0.3, top 500); identify confidence refitted (`confidence.py`).
  Galleries of 10, test: named correctly 76.0% → 78.3%; ePillID right product first 42% → 73% (front + back);
  identify first candidate right 71.4% → 74.8%. Greedy decoding (78 vs 213 ms per pill) reads fewer imprints
  (44.8% vs 50.0%): the large reader stays.
- **B24:** `double_dose` carries `details.confirm_count` and `notify_caregiver: false`; docs/dose_rules.md item 8.
- **B25:** `/v1/model-info` has `similarity` and `error_rates` (8 sets, each with its source file; a test checks every
  rate against its file; empty for any other model).
- **B26:** NRIC/FIN-shaped ids or names refused (`personal_id`, never echoed back); names optional;
  `reference_set_id` passed through.
- **B27:** `scripts/evaluate_same_maker_imprints.py`: 3,149 same-maker pairs sharing an imprint part: named as the
  other 0.01%, Teva "93" 0.00%; worst pair tramadol 50 mg (TV 58) vs carvedilol 12.5 mg (TV 7295), 5.8%. Found:
  reading only a shared part ("TV") *confirmed* wrong look-matches. Fix (`GALLERY_CONFIRM_DISTINCT`): only parts that
  tell the patient's medicines apart confirm or contradict a match. Validation: named wrongly 35 → 30, other pills
  accepted 6,158 → 6,150, correct −0.03 points. Using distinct parts for ties too was rejected (accepted more pills
  that aren't the patient's).
- `/health` gives `text_reading.gpu_model`; API version 0.9.0. docs/API.md, docs/features.md (F7),
  docs/blueprint_review.md ("Before-submission results") updated. 81 tests pass.
- **Not done:** deploy (no Hugging Face login in this session) and the GPU speed check (the Space's free GPU
  allowance hasn't come back since last night; it still answers on the CPU).

## 2026-10-13: blueprint Stage 2 done, API v0.7.0

- B7, B9, B10, B12, B13, B14, B16 built, tested and deployed; details in docs/blueprint_review.md ("Stage 2
  results"). New docs: docs/agent_tools.md (agent tools and traces), docs/licences.md (licences, PDPA, HSA).
- Headlines: `/v1/read-label` now reads prescriptions and PDFs with directions, quantity and dates (source lines,
  never inferred) and compares with the patient's list; products carry strength, form and release as fields;
  safety answers say what could and couldn't be screened; `GET /v1/tools` for the agent; look-alike strengths
  measured (mostly "ambiguous", rarely wrong: 0.1% overall, 0.5% when the strengths look alike).
- Left for after the submission: B8 (needs your real label photos), B15, B17, B19; D1.

---

## 2026-10-12 (night): blueprint Stage 1 done, API v0.6.0 (submission 16 Oct)

- Your decisions: D1 skipped for now; D2–D6 as the blueprint proposes; submission 16 October. Staged plan in
  docs/blueprint_review.md (Stage 1 contract and safety, Stage 2 agent tools, Stage 3 freeze).
- Stage 1 (B1–B6, B11) built, tested and deployed; results in docs/blueprint_review.md ("Stage 1 results"). In short:
  pill count right 86–88% on 1–3-pill photos (was 28%), "looks like" wording and visual-evidence fields, required
  embedding versions with `/v1/model-info`, calibrated `confidence` / `verdict` for `/v1/identify`, response schemas
  and one error shape, `/v1/detect`, `/v1/scan` dose check deprecated, docs/dose_rules.md for the backend.
- docs/API.md updated for 0.6.0. 69 tests pass.

---

## 2026-10-12 (later): the team's agent blueprint reviewed; job list B1-B20

- Reviewed `PharmaScan_Care_Agent_Workflow.html` (hackathon Challenge 2, written against API v0.4.0) against v0.5.0:
  **docs/blueprint_review.md** has the reflection, six decisions for you (D1-D6) and 20 jobs (B1-B20) with owners,
  priorities and acceptance tests, to tackle one by one.
- Checked the blueprint's two failure reports: the stand-in pill finder gives 2+ boxes on **72%** of 300 single-pill
  C3PI photos (mostly small fragments beside the pill), and pills that aren't in the catalogue at all still get a
  median top-1 similarity of **0.929** (7% at 0.97 or more). Both are P0 jobs (B1, B4).

---

## 2026-10-12: imprint reading (F7) and box/label reading (F8) in the API (v0.5.0)

- Your decision: build F7 and F8 first (of F7–F10, added to docs/features.md today with the doctor-to-patient flow).
- **F7, imprint reading.** Florence-2 reads each pill's imprint (4 rotations), on the Space's GPU.
  - `/v1/identify` re-ranks the catalogue by the read imprint (alpha 0.3, top 500: re-chosen on validation for run 4,
    same as run 5). ePillID test, run 4: right product first 42% → 66% (front + back), 33% → 51% (one photo).
  - `/v1/match` / `/v1/scan`: gallery medications may carry their `imprint`. The read confirms, settles a tie, or
    turns a match into "photograph again" when another of the patient's medicines' imprint is read; it never
    accepts a pill the look rejected (`core.inference.matching.check_imprint`).
  - **Measured** (`scripts/evaluate_imprint_gallery.py`; galleries of 10; OCR of all 7,111 validation and 4,005
    held-out crops, ~10 min on the desktop GPU). Validation chose `GALLERY_MIN_WEIGHT` 0.5. Test on new medicines,
    once: named correctly 76.0% → 77.0%, other pills accepted 3.77% → 3.54%, named wrongly 0.04% → 0.06% (within
    the +0.1-point limit), turn-over requests 9.5% → 8.4%. The look-only baseline reproduced the calibration exactly.
  - **Honest limit:** the own imprint is readable on only 38% of photos, so the gain is modest; it can't reach
    near-100% on its own. Next step if wanted: fine-tune the reader on C3PI's imprint text.
- **F8, box/label reading.** `POST /v1/read-label`: Florence-2 reads the lines, `core/label.py` finds RxNorm
  ingredient and brand names (23,500, plus international names such as paracetamol), the strength and the dose form,
  and resolves the exact RxNorm product. RxNorm's own fuzzy scores proved useless for finding names ("Take 2 tablets
  4 times a day" outscores "PARACETAMOL 500MG TABLET"), hence the name list. Correct on 9 printed examples (dispensing
  label, Lipitor, metformin, Augmentin, Panadol Extra, a misread name, co-amoxiclav, a 1% cream, no medicine); **not
  yet tested on real phone photos**. Only lines naming a medicine are returned (labels show the patient's name).
- The Space also runs the text reader on the lent GPU (CPU fallback); demo page `/ui` gained "Read a box or label".
  `deploy_hf.py` no longer overwrites an existing `HF_TOKEN` Space secret (so your read-only token stays).
- docs/API.md (new fields, imprint-check statuses, `/v1/read-label`) and docs/features.md updated. 62 tests pass.

---

## 2026-10-10 (evening): API live on Hugging Face, free (ZeroGPU Space)

- **Live:** https://davidpansx-pharmascan-api.hf.space serves run 4 (`stage2_run4@2026-10-09`), the 4,936-pill
  catalogue and the safety database from the private `DavidPanSX/pharmascan-assets`. `/docs` is the API page, `/ui`
  a demo page (identify one pill; scan a photo).
- **Why ZeroGPU:** Hugging Face refused the Docker Space (402: Docker and Gradio Spaces on free CPU now need PRO).
  ZeroGPU hardware is free for personal accounts over 30 days old with a verified email, but only for Gradio Spaces.
  So **serving/space_app.py** wraps the unchanged REST API: Gradio demo mounted at `/ui`, model weights handed to
  ZeroGPU at start-up, each model pass on a lent GPU (`@spaces.GPU`), and the CPU when the GPU is refused (allowance
  used up), for 10 minutes (`serving/app/fallback.py`, tested). `/health` → `model.device` says which ran.
- **Checked on the Space:** 20 of 20 `/v1/embed` calls on the GPU, ~1.0–1.25 s each from the desktop (first 2.8 s);
  `/v1/scan` 1.9 s; search, pictures, `/ui` work. Its embedding of a test crop matches the local CPU one (cosine
  0.99994). Not yet seen on the Space: the CPU fallback (only happens when the allowance runs out) and its speed.
- **Fixes on the way:** Gradio's SSR Node server took port 7860 (`ssr_mode=False`); the token secret only reached
  the app on the second start. The API now runs model passes in a worker thread so a GPU wait doesn't block other
  requests.
- `scripts/deploy_hf.py space` builds the ZeroGPU Space by default (`--docker` for the PRO route);
  `serving/requirements-space.txt` pins torch 2.11.0 (ZeroGPU supports 2.8–2.13). docs/DEPLOY.md, docs/API.md
  (real base URL) and serving/README.md updated. 53 tests pass.

---

## 2026-10-10 (afternoon): API guide for the web app, Hugging Face deployment, monthly retraining

- **docs/API.md**: every endpoint with request/response examples, the main flows (enrol, dose check, identify,
  pictures, safety, opt-in photos), errors, and browser (fetch) examples, for the web teammate.
- **API v0.4.0**: CORS for browser calls (`EPILLS_CORS_ORIGINS`), optional key (`EPILLS_API_KEY`, header
  `X-API-Key`), the catalogue as one `catalogue.tar` download, version pinning for rollback
  (`EPILLS_MODEL_REVISION`), and opt-in photo collection: `POST /v1/feedback` (consent required; pill crop only, no
  EXIF; pushed to a private dataset repo every 10 min) and `DELETE /v1/feedback/{consent_id}` (withdrawal).
- **scripts/deploy_hf.py** (`assets`, `space`, `feedback`, `restart`) and **docs/DEPLOY.md** (step by step): private
  model repo for the assets (tagged per model version), a Docker Space with only the hosted code (staging checked:
  24 files, all 12 `/v1` routes load), settings and secrets set by the script.
- **Monthly retraining: a prototype only; you're just considering it (2026-10-10), so it is not in use** (collection off by default, no schedule).
  **scripts/monthly_retrain.py** (`pull`, `build`, `train`, `check`, `publish`): fine-tunes the live model on the
  month's verified photos, then a release gate against the live model and a report; publishing only after a
  person approves. `outputs/serving/served_run.txt` records the live run (written by `export_model.py`).
- New-feature status: F1–F3 are API features, independent of the training run; the catalogue must be rebuilt for
  the served model. The OCR re-ranking is not yet in `/v1/identify`.
- 52 tests pass.

---

## 2026-10-10: imprint re-ranking (OCR) beats the paper's GAP@1; run 6 (stronger fine-tuning) started 11:29

- Your decision: imprint OCR re-ranking, then run 6 at the higher learning rate.
- **OCR** (`src/epills/core/imprint/ocr.py`): Florence-2 (MIT licence, via transformers 5.19, no new packages),
  the pill region read in 4 rotations. On 150 validation crops a token of the pill's own imprint is read on **43%**
  (base; printed 63%, debossed 36%; C3PI "Clear" 74%, "Can not Identify" 0%), a random other pill's on 0.7%; 0.28 s
  per pill on the GPU. Florence-2-large reads 47% at twice the time; base kept (fast enough for the API on CPU).
- **Re-ranking** (`src/epills/core/imprint/match.py`): a candidate's recorded imprint tokens (C3PI metadata, by
  NDC9; known for 2,396 of the 4,902 ePillID types) are looked for in the reads; rare tokens weigh more (IDF); the
  top_k candidates get + alpha x score. **Chosen on validation only** (`scripts/evaluate_imprint.py`; 2,500
  validation photos of 250 types against all 4,902 types' references): alpha 0.3, top_k 500 (validation MAP
  0.445 → 0.599).
- **ePillID test, run 5 + imprint** (a separate variant: it uses imprint text the paper's model didn't):

  | two-sided | Run 5 plain | Run 5 + imprint | Paper |
  |---|---|---|---|
  | MAP / MAP@1 | 0.597 / 45.7% | **0.780 / 69.0%** | 0.958 / 92.0% |
  | GAP / GAP@1 | 0.207 / 0.526 | 0.598 / **0.921** | 0.812 / 0.912 |
  | top-5 | 77.5% | 89.6% | – |

  Single photo: MAP 0.542 → 0.658. 43% of the test photos have a token of their own imprint read.
- **Run 6** restarted 11:29 (`configs/stage2_run6_official.yaml`, unchanged). Epoch 1 already above run 5's best on
  validation (gallery-10 AUROC 0.980 vs 0.975, correct@95% 88.2% vs 85.7%). After
  training: calibration, ePillID and MEDISEG, and the imprint re-ranking re-tuned on validation, all automatic.
- **Run 6 result** (11:29–13:00, early stop after 26 epochs, best epoch 18; window 1.4, threshold 0.840, margin
  0.150; imprint re-ranking re-tuned on validation: alpha 0.3, top_k 500 again):

  | | Run 5 | Run 6 |
  |---|---|---|
  | Validation top-1 over 214 / gallery-10 AUROC / correct@95% | 83.1% / 0.975 / 85.7% | **86.9% / 0.982 / 89.9%** |
  | MEDISEG top-1 over 32 (pills over 100 px) | 52.3% (71.2%) | **54.1% (74.3%)** |
  | ePillID two-sided MAP / MAP@1, plain | **0.597 / 45.7%** | 0.570 / 40.4% |
  | ePillID two-sided + imprint: MAP / MAP@1 / GAP@1 | **0.780 / 69.0% / 0.921** | 0.774 / 68.1% / 0.886 |

  Stronger fine-tuning helped everyday-to-everyday matching (validation, MEDISEG) but not everyday-to-studio
  (ePillID). The epoch is picked on validation top-1 (everyday photos only), which doesn't reward the benchmark's
  task; and few batches pair a class's everyday photos with its references. So the remaining ePillID gap is the
  everyday-vs-studio domain gap, not underfitting.
- **Run 7, benchmark-aligned training** (your choice; `configs/stage2_run7_official.yaml`), started 13:31 from
  run 6 (with its saved ArcFace table), same data, architecture and learning rate:
  - **Epoch picked on a validation benchmark** (`select_metric: ref_map`): 4,280 validation everyday photos against
    the 9,804 references of all 4,902 ePillID types (validation types' references never trained on); MAP.
  - **Domain-paired batches**: a pill with both kinds of photo contributes 2 references + 2 everyday photos
    (`sampler.pair_domains`), plus a batch-hard triplet term (weight 0.5) next to ArcFace, whose hardest positives
    are then mostly everyday vs reference.
  - Smoke test: starts at validation ref MAP 0.481 (top-1 34.3%).
  - Afterwards automatically: calibration, ePillID, MEDISEG, imprint re-ranking re-tuned on validation.
- **Run 7 result** (13:31–14:40, early stop after 18 epochs, best epoch 12): validation ref MAP 0.481 → **0.514**
  only, though the training loss kept falling (ArcFace 5.4 → 3.4; the triplet term stayed tiny, 0.04 → 0.02:
  few hard cases in the batches). ePillID two-sided: plain MAP 0.578 / MAP@1 40.7%; + imprint 0.774 / 67.6% /
  GAP@1 0.916. MEDISEG 53.3%. Validation: correct@95% 91.2%. Window 1.7, threshold 0.782, margin 0.190.
- **Three training changes (runs 5–7) all land at ePillID MAP ~0.57–0.60 plain, ~0.78 with imprint.**
- **What the paper's own tables show** (arXiv 2005.14288v2, Table 1–2; code: `src/pillid_datasets.py`, `losses.py`,
  `arguments.py`):
  - Plain softmax classifiers score MAP 0.55–0.73 (ResNet-18 GAP 0.547 … ResNet-152 B-CNN 0.727): our 0.60 is in
    this range.
  - Their multi-head metric-learning recipe lifts even ResNet-18 (plain pooling) to 0.912, ResNet-152 + CBP to
    0.958. The gain comes from the recipe, not the backbone or the pooling.
  - Loss ablation (ResNet-50, validation GAP): softmax alone 61.8 → + contrastive + triplet 75.5 → + ArcFace 0.1
    77.2.
  - Recipe: Adam 1e-4 for the whole network (halved on plateau, patience 2); 2,048-d embedding; contrastive (margin
    1, Euclidean) + triplet with online hard mining; softmax and ArcFace over side classes (front and back
    separate); batches of 48 built per pill: its 2 references + its consumer photos + a random pill's references;
    trained on ePillID's own consumer photos (folds 0–3) + all references.
- **Decision (you, 2026-10-10): stop the benchmark work and report.** Results written up in
  [docs/stage2_results.md](docs/stage2_results.md) (draft for the report, P9): validation targets, strict test
  results of runs 2–4, the ePillID comparison (runs 5–7, ± imprint; GAP@1 beats the paper, MAP doesn't), the
  diagnosis, limitations and future work (first: the paper's multi-head recipe on our architecture).

## 2026-10-09 (night): run 5, a benchmark model under ePillID's official protocol (started 21:47)

- Goal (yours): beat the ePillID paper (two-sided MAP 95.76, MAP@1 92.01). Not like for like so far: the paper
  trains on the reference photos of **all** 4,902 types, test types included, and those photos are the test
  gallery; our runs 3–4 never see a test type (run 4: MAP 0.555, MAP@1 42.3%).
- **Run 5 follows the official protocol** (`scripts/add_epillid_refs.py`, `configs/stage2_run5_official.yaml`):
  run 4 + ePillID's own 9,240 reference photos, **including all 384 of the 192 test types**; validation types'
  references left out (498, keeps validation clean; slightly stricter than the paper); 3,500 reference-only types
  as new classes (24 left out as look-alikes of validation pills; 42 ambiguous pack parts outside the test set
  left out). No test-type everyday photo is trained on (checked: 0). Training: 44,909 photos, 5,387 classes;
  classes drawn by sqrt(photos) so the reference-only classes don't crowd out everyday photos (batches: 45%
  everyday, 44% ePillID references, 10% C3PI references). Colour loss now skips photos without a colour label.
- **A benchmark model only:** it has seen the test types' references, so the C3PI held-out and unknown-pill tests
  don't apply to it; validation, ePillID official and MEDISEG do. The app keeps the strict model (run 3 or 4).
- Fixed before results: the test protocol and metrics as before (two-sided pairs, gallery of all 4,902 types'
  references, window chosen on validation); main result without test-time augmentation.
- Epoch 1: validation gallery-10 top-1 74.4% (run 4: 70.7%).
- **Result (153 min, best phase 2 epoch 34; window 1.4, threshold 0.906, margin 0.100):**

  | ePillID official, two-sided | Run 4 (strict) | Run 5 (official protocol) | Paper |
  |---|---|---|---|
  | MAP / MAP@1 | 0.555 / 42.3% | **0.598 / 45.7%** | 0.958 / 92.0% |
  | GAP / GAP@1 | – | 0.207 / 0.526 | 0.812 / 0.912 |
  | top-5 | 72.0% | 77.5% | – |

  **Not beaten**: training on the test types' references added only ~4 points. MEDISEG improved to its best yet
  (top-1 52.3%; pills over 100 px 71%). Validation: gallery-10 top-1 97.7%, correct@95% 85.7% (margin 0).
- **Diagnosis** (no training; run 5's model):
  - Everyday photo → its type among all 4,902 types' references: test types 40% top-1 / 71% top-5; **types
    trained with everyday photos and references: only 54% / 83%**. So the model gets colour and shape right but
    can't separate look-alikes, even for pills it trained on.
  - Not the photo style: our own crops of the test photos do worse (29%) than ePillID's (40%).
  - **Not the 64-number mean-pool**: single-photo top-1 by network stage: ViT CLS (768) 35.6%, VGG branch (256)
    37.6%, both concatenated 45.4%, capsule fusion 40.0%, refinement (64) 40.8%, embedding 40.4%.
  - **The limit is the global summaries the backbones hand over**: the ViT's CLS token and VGG's average pool
    each reduce the whole image to one vector before fusion. Fine for facial expressions (global configuration),
    weak for pills, whose identity is in small imprint strokes. The paper's model uses bilinear pooling of local
    features, a design made for fine-grained recognition.
- Options (architecture changes, so your decision): local-detail pooling into the fusion; higher input resolution;
  imprint reading (OCR) to re-rank; training-only tweaks (small gains).
- You chose local-detail pooling and imprint OCR (2026-10-10). Then two findings changed the order:
  - Local pooling of run 5's own backbone features (no training) is no better than the global summaries
    (36–39% each; ~44% combined), so the detail isn't in the features yet, whichever way they're pooled.
  - **Runs 3–5 underfit**: best epoch = last epoch, and the loss still falling (run 5 phase 2: 8.4 → 6.2) when the
    learning rate had decayed to 5e-6.
- **Run 6** (`configs/stage2_run6_official.yaml`): continue run 5's weights in phase 2 at 3e-5 (ArcFace table 3e-4,
  rebuilt from class-mean embeddings since run 5 didn't save it; `best.pt` now saves the table) for up to 30
  epochs. Started 00:41, **stopped at your request before its first epoch**; ready to start again. Local pooling
  parked until the backbones are better trained; OCR still to do.

## 2026-10-09 (evening): run 4 (stage2_run4, other photo conditions + jittered crop windows). More robust, slightly less accurate in C3PI conditions

- 19:03–21:21 (133 min), all 25 + 35 epochs, best at the last epoch (still creeping up). `dims` ~12.
- Calibrated with the window chosen on validation: **1.55×** (1.3–2.0 all within 1.1 points), threshold **0.895**,
  margin **0.115**. Each model below is scored with its own validation-chosen window (run 2: 1.7×, default).

  | | Run 2 | Run 3 (+ refs) | Run 4 (+ refs + robustness) |
  |---|---|---|---|
  | **Validation** correct@95%, 10 medications, margin 0 | 90.7% | 89.7% | 87.2% |
  | **C3PI held-out** top-1 over 200 | 82.0% | **84.4%** | 83.2% |
  | C3PI held-out app: correct / wrong / own rejected / unknown rejected | 78.6 / 0.1 / – / 94.2% | **80.4** / 0.0 / 9.9 / 94.1% [correct 78.0–82.7] | 76.0 / 0.0 / 14.5 / 94.8% [73.4–78.6] |
  | **ePillID** two-sided MAP@1 / top-5 / MAP | 13.7% / 38.2% / 0.255 | 37.1% / 67.3% / 0.506 | **42.3% / 72.0% / 0.555** |
  | **MEDISEG** top-1 over 32 | 38.9% | 41.8% | **48.2%** |
  | MEDISEG by pill size: under 50 / 50–100 / over 100 px | – | 29.6 / 29.9 / 56.8% | **37.3 / 36.4 / 62.2%** |
  | MEDISEG app: correct / wrong / own rejected / unknown rejected | 17.6 / 1.1 / 51.5 / 78.7% | 18.7 / 1.0 / 64.2 / 89.3% | 18.7 / **0.7** / 70.4 / **93.1%** |
  | MEDISEG-Deploy 10-shot (40 pills): top-1; correct / wrong / flip / rejected | 35%; 6 / 1 / 10 / 23 | 52.5%; 6 / 0 / 4 / 30 | 45%; 3 / 0 / 1 / 36 |

- **Reading:**
  - Run 4 recognises pills in other conditions better: MEDISEG +6.4 points (small pills +7), ePillID +5 points,
    and is safer there (fewer wrong matches, more unknown pills rejected).
  - It is slightly worse in C3PI conditions: held-out top-1 −1.2 points, and in the app scenario it names 76.0%
    instead of 80.4% (it rejects more of the patient's own pills: 14.5% vs 9.9%). Wrong matches stay at 0.0%.
  - In MEDISEG's app scenario, correct matches don't rise (18.7% for both): the better recognition is cancelled
    by the threshold, which is set on C3PI photos and is too strict for other phones (70% of own pills rejected).
    **The threshold's transfer to new phones is now the main barrier**, more than recognition itself.
  - MEDISEG-Deploy has only 40 pills; differences of a few pills are noise.
- Which to serve is your call (below). Both exports and catalogues can be swapped in minutes.

---

## 2026-10-09: New features (pill → name, name → picture, one-call scan), the pill catalogue, run 4 started

- **Feature backlog** written: [docs/features.md](docs/features.md) (F1–F6), with the continual-learning design
  (F5) checked against Singapore's PDPA, HSA GL-04 Revision 4 (Dec 2025) and MOH/HSA AIHGle 2.0 (Mar 2026).
  F4 (merging detection into the recognition model) paused until you talk with the Stage 1 teammate.
- **Model for serving:** `scripts/export_model.py` → `outputs/serving/stage2_model.pt` (run 3, 204 MB in half
  precision, same embeddings: min cosine 1.00000), with its calibration and a version (`stage2_run3@2026-10-08`).
- **Catalogue** (`scripts/build_catalogue.py` → `data/processed/catalogue/`, 201 MB): 4,936 C3PI pill types,
  32,811 embeddings (13,730 reference crops of all pill types + up to 8 everyday crops each), 8,939 display
  pictures; RxNorm ingredients for 4,710 (95%; retired NDCs resolved through RxNav's NDC status).
  Self-check, new everyday photos against the catalogue: right product first 70% (never-trained types) / 77%
  (validation types) / 89% (trained types); top 5: 93% / 95% / 98%.
- **API** (v0.3.0, `serving/app/main.py`): `/v1/embed`, `/v1/match`, `/v1/scan`, `/v1/identify`,
  `/v1/pills/search`, `/v1/pills/{id}`, `/v1/pills/{id}/image/{n}`, besides the Stage 3 endpoints. Model and
  catalogue load once (or at start-up with `EPILLS_PRELOAD=1`); the catalogue must match the served model version.
  Hosted code still never imports training code. 46 tests pass.
- **Live test on CPU** (real model and catalogue, held-out C3PI photos):
  - `/v1/match`, 3 enrolled medicines: 5 of 6 own pills named right, 1 rejected as unknown (similarity 0.830 vs
    threshold 0.831), the never-enrolled pill rejected. ~0.15 s per pill after a one-off 14 s model load.
  - `/v1/identify`: right product first (famotidine, TEVA 5729); with the typed imprint, the only candidate.
  - `/v1/pills/search`: "metformin" 74 products, "Glucophage" → metformin (71), "lipitor 10" → atorvastatin.
  - `/v1/scan` on MEDISEG-Deploy's mixed photos with the dataset's boxes: all 40 pills "unknown" (as offline: the
    model is weak on these phones and small pills); the stand-in finder found 1 of 40 pills on their cluttered
    backgrounds, so `/v1/scan` needs the real detector there.
- **Crop window finding:** in the training crops the pill alone fills a median 59% of the frame, so a *tight*
  box (detector, MEDISEG, ePillID) needs a window of ~1.7×, not the 1.4× used so far (the C3PI cropper's 1.4
  applies to its own boxes, which include the shadow). Re-scored runs 2 and 3 with 1.7 (old results kept in
  `eval_window1.4/`): no real change (run 3: ePillID MAP@1 36.6% vs 37.1%, MEDISEG 39.7% vs 41.8%, MEDISEG-Deploy
  10-shot 40% vs 52.5% on 40 pills). Since the window may not be chosen on test sets, it is now chosen on
  validation in calibration (`--windows`) and travels with the exported model.
  Run 3's validation sweep (correct matches at 95% unknown rejection, 10 medications): 1.3× 89.2%, 1.4× 89.7%,
  1.55× 89.7%, 1.7× 89.1%, 1.85× 88.0%, 2.0× 86.8% → **1.4×** (threshold 0.819, margin 0.155; the old calibration on
  cropper crops: 0.831 / 0.150). Test sets with it: C3PI held-out correct 80.4%, unknown rejected 94.1%; ePillID and
  MEDISEG recognition unchanged (37.1%, 41.8%). Re-exported; the served model now carries window 1.4.
- **Run 4** (`configs/stage2_run4.yaml`) started 19:03: run 3 + jittered crop windows (×0.75–1.35, shift ±10%) and
  photo conditions (brightness/contrast ±30%, white balance ±8%, downscale to 40–160 px, JPEG 30–90, noise,
  another pill pasted at an edge); previewed before starting.

## 2026-10-08 (night): run 3 (stage2_run3, D7 reference photos). Big gain on studio references; other conditions barely move

- 21:15–23:21 (126 min): all 25 + 35 epochs; best at the very last epoch (phase 2, epoch 35), so it was still
  creeping up (top-1 over 214 within 85.5–86.1% over the last 5 epochs). `dims` ~14: no collapse.
- Calibrated the same way (validation): threshold **0.831**, margin **0.150**
  (`outputs/runs/stage2_run3/calibration.json`). Scored on the same test sets with the same fixed rules.

  | | Run 2 | Run 3 (+ references) |
  |---|---|---|
  | **Validation** gallery-10 top-1 / AUROC / correct@95% / wrong@95% | 98.3% / 0.981 / 90.7% / 1.0% | 98.1% / 0.979 / 90.2% / 1.1% |
  | Validation top-1 over 214 | 85.2% | 86.1% |
  | **C3PI held-out** top-1 over 200 | 82.0% | **84.4%** |
  | C3PI held-out app scenario: correct / wrong / unknown rejected | 78.6% / 0.1% / 94.2% | 79.3% / 0.0% / **95.1%** [94.1–96.0] |
  | **ePillID** two-sided MAP / MAP@1 / GAP / GAP@1 | 0.226 / 12.1% / 0.015 / 0.093 | **0.506 / 37.1% / 0.105 / 0.428** |
  | ePillID two-sided top-5 / single-photo MAP@1 | 32.9% / 8.3% | 67.3% / 30.7% |
  | **MEDISEG** top-1 over 32 (pills over 100 px) | 40.1% (53.2%) | 41.8% (56.8%) |
  | MEDISEG app scenario: correct / wrong / own rejected / unknown rejected | 18.5% / 1.0% / 51.1% / 79.5% | 18.0% / 0.9% / 67.6% / **91.4%** |
  | MEDISEG-Deploy 10-shot: top-1; correct / wrong / flip / rejected (of 40) | 42.5%; 4 / 1 / 12 / 23 | **52.5%; 6 / 0 / 4 / 30** |

- **Reading:**
  - The reference photos did what they were for: ePillID (everyday photo vs studio reference) tripled, MAP@1
    12% → 37%, top-5 33% → 67%. Still far from the paper's 92%, which trains on the references of all 4,902
    types *including the test types*; ours never sees a test type.
  - Small gain on new medicines in C3PI photos (84.4% vs 82.0%), and unknown rejection now meets 95% there.
  - Other phones and conditions (MEDISEG) barely improved in recognition (+1.7 points). The model got more
    cautious there: fewer wrong matches and far fewer unknown pills accepted (3.1% vs 5.4%), but more of the
    patient's own pills rejected (68% vs 51%).
  - So the remaining gap is camera, lighting, background and resolution, which run 4 (domain augmentation)
    targets.
- Run 3 is the better model on every test set and safer under shift: it replaces run 2 as the model to serve.

## 2026-10-08 (late night): D7 reference photos cropped; run 3 started

- Decision D7 (you, 2026-10-08): run 3 = run 2 + C3PI studio reference photos; run 4 (domain augmentation) later,
  if needed. Own pills (G5) skipped for now.
- **Reference crops** (`src/epills/training/datasets/c3pi_ref_crop.py`, `scripts/crop_c3pi_refs.py`):
  - MC_CHALLENGE (one pill side on plain grey) → 1 crop; RxImage (both sides, rulers, information bar) → 2 crops,
    with the other side, the edge-on profile and the ruler marks painted over in the background grey.
  - Same framing as the everyday crops (window 1.4× the long side, 384 px).
  - **Only training groups' references** (4,688 photos): references of validation and test pills would leak.
    Reference-only pill types (2,574 RxImage photos) left out: no group, no look-alike check.
  - 13 flagged crops (several blobs, touching sides) dropped after review; 40 random crops checked
    (`data/interim/c3pi/crops/references_review.jpg`).
  - Result: **6,198 reference crops** for 1,683 of the 1,710 training groups (3,164 MC_CHALLENGE, ~3,040
    RxImage) → `data/processed/c3pi_splits/manifest_with_refs.csv` (column `source`: everyday / reference).
    Validation and test splits unchanged.
- **Run 3** (`configs/stage2_run3.yaml`): run 2's settings, training manifest with references (35,669 photos).
  Started 21:15. Epoch 1: top-1 (214) 28.7%, gallery-10 72.9% (run 2: 29.4%, 72.8%).

## 2026-10-08 (late night): P5 calibration and P6 test sets for run 2. Generalises to new medicines, not to new photo conditions

- **MEDISEG downloaded** (gate G4): both figshare files, MD5s match (`MEDISEG.tar.gz` 0.44 GB,
  `MEDISEG-Deploy.tar.gz` 8 MB), unpacked to `data/raw/mediseg/` (518 MB). Contents in `docs/datasets.md`.
- **P5, calibration** (`scripts/calibrate_stage2.py`, validation only; 1,000 simulated patients, 10 medications,
  3 enrolment photos each) → `outputs/runs/stage2_run2/calibration.json`:
  - threshold **0.793**: 95% of unknown pills rejected; margin **0.170**: the largest with ≤ 10% flip requests.
  - Own pills: 82.2% matched correctly, 0.0% wrong (1.0% without the margin), 9.6% flip, 8.2% rejected.
    Unknown pills: 95.0% rejected, 1.5% flip, 3.5% accepted as one of the patient's.
- **P6, test sets** (`scripts/evaluate_stage2.py`; results in `outputs/runs/stage2_run2/eval/`). Fixed before
  scoring: model, framing (window 1.4× the pill's long side, as the cropper), threshold and margin.

  | Test set | Closed set top-1 | App scenario (10 medications, fixed threshold and margin) |
  |---|---|---|
  | C3PI photos of held-out types (200 medicines, 4,005 photos) | 82.0% over all 200 | correct 78.6% [76.3–81.0], wrong 0.1%, flip 10.0%, own rejected 11.3%; unknown rejected 94.2% [93.2–95.0], unknown accepted 4.2% |
  | MEDISEG 32 products (Hong Kong, other phones, clutter) | 40.1% over 32 | correct 18.5%, wrong 1.0%, flip 29.4%, own rejected 51.1%; unknown rejected 79.5%, unknown accepted 5.4% |
  | MEDISEG-Deploy (8 products, 40 pills in mixed photos) | 27.5% / 32.5% / 42.5% with 1 / 5 / 10 enrolment photos | 10-shot: 4 correct, 1 wrong, 12 flip, 23 rejected |

  [95% intervals from resampling medicines.] Without the margin, wrong matches would be 1.1% (C3PI) and 11.0%
  (MEDISEG): the margin matters most under domain shift.
- **ePillID official** (fold 4, simulated front + back pairs, gallery = reference photos of all 4,902 types):
  MAP 0.226, GAP 0.015, MAP@1 0.121, GAP@1 0.093 (paper's best: 0.958 / 0.812 / 0.920 / 0.912). Single photos:
  MAP@1 0.083. Not like for like: the paper trains on the reference photos of all 4,902 types (holdout types
  included); our model has never seen a reference photo. Checks: everyday-to-everyday within fold 4 gives 70.7%
  (1 enrolment photo) / 81.5% (2); everyday-to-reference restricted to the 192 holdout types gives 42.7%. So the
  pipeline works and the gap is everyday photos vs studio references (decision D7).
- **MEDISEG by pill size** (closed set): under 50 px 29.5%, 50–100 px 29.8%, over 100 px 53.2%. MEDISEG's images
  are 640 px, so half its pills are under 81 px long; the app's crops come from full-resolution photos.
- **Reading:** the model generalises to *new medicines* photographed like C3PI (82% vs 85% on validation), but not
  to *new photo conditions*: studio references, other phones, lighting, backgrounds, small pills. All training
  data is one source (C3PI). Under shift it mostly fails safe (rejects or asks to flip), but wrong matches rise to
  ~1% and unknown pills are accepted ~5% of the time: the C3PI threshold doesn't transfer.
- Options (decisions for the next run): D7 reference photos; domain augmentation (lighting, white balance,
  downscale/blur, JPEG, backgrounds, neighbouring pills); MEDISEG split by product into train/test halves; own
  pills (G5) as the deciding test.

## 2026-10-08 (night): run 2 (stage2_run2, config `configs/stage2_run2.yaml`). All validation targets met

- Same machine and data as run 1. 18:59–20:37 (98 min).
- Changes from run 1 (one fix): FER learning rates for the network (3e-5, then 1e-5), `metric_lr` 1e-3 / 1e-4
  for the ArcFace table only, margin warm-up 0 → 0.3 over 3 epochs.
- Phase 1: all 25 epochs, still improving at the end (~51 s + 10 s per epoch). Phase 2: early stop after epoch 32;
  best at epoch 26 (~122 s + 10 s per epoch). `dims` 12 → 15 throughout: no collapse.
- Best validation (phase 2, epoch 26), `outputs/runs/stage2_run2/best.pt`:

  | Metric | Patient-sized gallery of 10 | Target | All 214 val medicines | Run 1 (gallery 10) |
  |---|---|---|---|---|
  | top-1 | **98.3%** | ≥ 95% ✅ | 85.2% | 77.2% |
  | unknown AUROC | **0.981** | ≥ 0.9 ✅ | 0.887 | 0.78 |
  | correct @ 95% unknown rejected | **90.7%** | ≥ 80% ✅ | 48.8% | 17.3% |
  | wrong accept @ 95% | **1.0%** | ≤ 2% ✅ | 3.5% | 3.7% |

- **Against the targets: all met on validation.** Caveats:
  - The validation medicines are unseen, but the photos are the same C3PI conditions as training. The cross-domain
    tests (ePillID fold 4, MEDISEG, own pills) are the real measure, and will be lower.
  - The epoch was picked on this validation set (best of 57), so these numbers are slightly optimistic.
  - The rejection threshold here is set per gallery (mean 0.79). The app needs one fixed threshold and a flip
    margin, chosen in P5.
- Next: P5 calibration, the P6 evaluation script, then `/v1/embed` and `/v1/match` in the API. D7 (reference
  photos) waits for the cross-domain results.

## 2026-10-08 (evening): run 1 (stage2_baseline, config `configs/stage2_baseline.yaml`). Not usable; cause found and fixed

- Desktop, RTX 4080 SUPER, PyTorch 2.11 + CUDA 12.8, bf16 autocast. 16:35–18:21 (1 h 46 min).
- Data: 1,710 training groups, 29,471 photos; P × K = 32 × 4 (phase 1), 16 × 4 (phase 2).
- Phase 1: all 25 epochs, best at epoch 20, ~51 s + 10 s eval per epoch. Phase 2: all 35 epochs (early stopping
  never triggered: `top1_all` kept edging up within noise), best at epoch 29, ~122 s + 10 s per epoch.
- Best validation (phase 2, epoch 29):

  | Metric | All 214 val medicines | Patient-sized gallery of 10 | Target (gallery 10) |
  |---|---|---|---|
  | top-1 | 24.0% | 77.2% | ≥ 95% |
  | unknown AUROC | 0.53 | 0.78 | ≥ 0.9 |
  | correct @ 95% unknown rejected | 1.5% | 17.3% | ≥ 80% |
  | wrong accept @ 95% | 7.2% | 3.7% | ≤ 2% |

  The gallery-10 open-set columns were computed afterwards (`retrieval.evaluate` now reports them every epoch).
  The 214-medicine AUROC was too harsh a reading: there an unknown pill almost always has a look-alike.
- **Against the targets: not met on any metric.**
- **Why: the embeddings collapsed.** The 256-d embeddings of different pills use about 2–3 effective dimensions
  (VGG branch ~5, capsule fusion ~8, the 4 refinement tokens identical). That's enough for colour, shape and
  size, not for imprints. The training loss stayed near chance level for 1,710 classes (8.1 vs ln 1710 = 7.4).
  Every val photo sits within ~0.5° of some enrolled medicine, so `threshold_95` was 0.999998.
- Ruled out with short checks: label or sampler bugs; stale batch-norm statistics (re-estimated: no change); the
  ArcFace margin (collapses even without it); crop framing (the pill's long side is ~160 of the 224 input px).
- **Cause found: the learning rate of the new layers.** The port gave CBAM, fusion, refinement and heads
  `head_lr` 1e-3 (phase 2: 1e-4). The FER notebook trains everything at 3e-5 (phase 2: 1e-5). Probe with the FER
  rates, 3 phase-1 epochs: epoch 1 already reaches top-1 24.4% (all 214) / 64% (gallery 10), gallery-10 AUROC
  0.70. That's what the whole baseline reached in 1 h 46 min. But the embeddings still shrank (13 → 6 → 3 dims),
  and the ArcFace table barely learned (loss 18.4 → 17.1).
- **Fix: FER rates for the network, 1e-3 for the ArcFace table only, margin warm-up 0 → 0.3 over 3 epochs.** Probe,
  4 phase-1 epochs (~5 min): dims stay at 12–14. Epoch 4: gallery-10 top-1 90.2%, unknown AUROC 0.904,
  correct@95% 55%; top-1 over all 214: 56%. The loss rises over these epochs by design (the margin grows).
- Run 2 started 18:59 with these settings: `configs/stage2_run2.yaml`.
- Data limit to keep in mind: C3PI rates the imprint "Can not Identify" in 22% of these photos and "Partial" in
  40%.
- Code changes: `--phases` (run only some phases), `arcface_margin_warmup_epochs`, `metric_lr` (own learning
  rate for the ArcFace table), and new per-epoch metrics: gallery-10 AUROC / correct / wrong accept, and
  `embedding_dims` (collapse alarm).

## 2026-10-08: Stage 3 safety engine and its API endpoints. Baseline training running

- **Interaction database** (`scripts/build_safety_db.py` -> `data/processed/safety/safety.sqlite`, 17.8 MB, not in
  git; it goes to the Hugging Face repo with the weights):
  - DDInter 2.0: 1,971 drugs, 234,981 pairs (Major 39,082; Moderate 143,748; Minor 9,736; Unknown 42,415); no pair
    has conflicting levels across the 14 files.
  - Every DDInter drug mapped to RxNorm ingredient (IN) codes through the RxNav API: **1,946 of 1,971 (98.7%)**.
    The 25 unmapped are vaccines, contrast agents and radiotracers, never pills. Report:
    `ddinter_rxnorm_map.csv`, `build_report.json`; responses cached in `data/safety/rxnorm_cache/`.
  - Plain DDInter entries are preferred over qualified ones ("Morphine" over "Morphine (liposomal)"), since the app
    checks pills. 12 ingredients have two plain entries (salt or prodrug forms, e.g. fenofibrate / fenofibric acid);
    both are used and the more severe level wins.
- **Rules** (`src/epills/core/safety/`, hosted): `rxnorm.py` (identify medications), `db.py` (schema and lookups),
  `rules.py` (interactions, duplications, dose checks), `openfda.py` (label warnings).
  - Medications are identified by listed ingredients (for non-US medicines), NDC from the bottle, RxCUI, name, or a
    fuzzy name match (reported so the app can confirm "did you mean ...?"). Brand names work: Lipitor ->
    atorvastatin, Zoloft -> sertraline, Glucophage -> metformin; typos like "Metforminn 500" too.
  - A medication that can't be identified, or an RxNorm outage, is itself an alert: no alerts always means every
    medication was checked.
  - The plan's severity mapping (major red + spoken + caregiver; moderate amber + spoken; minor screen only); every
    message ends with "Check with your pharmacist." and never says take or stop (a test enforces both).
  - Duplications (same ingredient twice: brand + generic, Panadol + Tylenol) are amber, since two strengths are
    sometimes prescribed together on purpose.
  - Dose checks: unknown pill, double dose, a pill not part of this dose (red); missed, unsure pill (amber).
- **API** (`serving/app/main.py`, v0.2.0): `POST /v1/safety/medications`, `POST /v1/safety/dose`,
  `GET /v1/safety/label`; contract and alert format in `serving/README.md`. The database loads from
  `EPILLS_SAFETY_DB` or the Hugging Face repo (`EPILLS_MODEL_REPO`).
- **Tests:** 26 pass (14 new: rules on a made-up database with a fake RxNorm, wording rules, API endpoints). Live
  check against RxNorm and openFDA: Lipitor + clarithromycin, warfarin + Bayer aspirin + Advil, sertraline + tramadol
  -> serious (red); Panadol (ingredients) + Tylenol -> duplication; atorvastatin label warnings returned.
- **Finding, for decision D6:** a common list (atorvastatin, metformin, levothyroxine, omeprazole, with a brand and a
  generic of two of them) gave 13 spoken amber alerts; most are DDInter "Unknown" (listed, not graded). Too many
  alerts and users stop listening. **Decided (D6): ungraded pairs get their own level, `ungraded`, on screen only,
  not spoken.** Graded Major and Moderate alert as planned.
- **Not done:** deployment to Hugging Face (needs your account); DDInter advice text and duplication classes (DDInter
  email, G6).

---

## 2026-10-07 (late night): Stage 2 model ported from FER to PyTorch; training pipeline tested

- **Decisions:** prototype hosting with crops sent to the API is accepted (non-commercial demo). D1: PyTorch. The
  model is your FER network (`FER/exp-tuning.ipynb`) without the geometry branch, so D2 = VGG16 + CBAM.
- **Port** (`src/epills/core/models/hybrid.py`, hosted code):
  - VGG16 (ImageNet, blocks 4-5 trainable in phase 1) + your CBAM; ViT-B/16 in21k (same checkpoint, pooler output);
    your capsule fusion (2 routing iterations, residual update) and transformer refinement block (4 x 64 tokens,
    4 heads with Keras-style key_dim 64, 2 layers, post-norm); Keras initialisers and norm settings.
  - Stage 2 heads replace the 7-class softmax: 256-d L2-normalised embedding, shape (13) and colour (12, multi-label).
  - **Checked against your TensorFlow code**: parameter counts identical for every block (VGG16 14,714,688; CBAM
    99,043; fusion 396,288; refinement 232,448; ViT 86,389,248). With the TF weights loaded into the port, outputs
    match to 4e-7 (TF run on CPU; on GPU TF32 rounding gives 4e-4). Kept as `tests/test_hybrid_parity.py`.
  - Deliberate differences: torchvision's VGG16 weights with their own normalisation; the new heads.
- **Training pipeline:** P x K sampler and augmentations (`training/datasets/pills.py`), ArcFace / batch-hard
  triplet / auxiliary losses (`training/losses/metric.py`), validation metrics (`training/eval/retrieval.py`),
  matching with unknown rejection (`core/inference/matching.py`, shared with the API), `scripts/train_stage2.py`,
  `configs/stage2_baseline.yaml`, `requirements-train.txt`. 12 tests pass.
- **Speed (RTX 4080 SUPER, bf16):** phase 1 633 img/s (~47 s per epoch of 29,471 photos); phase 2 123 img/s at
  batch 64 (batch 128 needs 17 GB). CPU with 2 threads (free HF tier): ~0.3 s per pill.
- **Smoke test:** overfitting 10 groups reaches 96% top-1 in 16 epochs.
- **No full training yet** (your call: tests only for now). A full run was started by mistake and stopped after
  18 of 25 phase-1 epochs; kept for reference in `outputs/runs/stage2_baseline_stopped_epoch_partial/`. It was
  still improving: gallery-of-10 top-1 0.37 -> 0.72, top-1 against all 214 validation groups 0.04 -> 0.15, unknown
  AUROC ~0.55. These are not results; the run is incomplete.

---

## 2026-10-07 (night): Gate G2 passed; code split into hosted and not hosted

- **G2 passed** with your sign-off on the crop sample (`review_2027_*.jpg`) and the leakage report.
- **Goal:** your part (Stages 2 and 3) ends as an API the app calls, hosted for free (Hugging Face Spaces, Docker).
- **New layout** (README "Folder structure"; `serving/README.md`):
  - Hosted: `serving/` (FastAPI app, Dockerfile, `requirements.txt`) and `src/epills/core/` (models, inference,
    safety). Weights come from a Hugging Face model repo at start-up, never from git.
  - Not hosted: `src/epills/training/` (the C3PI cropper moved here from `src/epills/datasets/`), scripts, configs,
    notebooks, tests, docs, data, outputs.
  - `tests/test_layout.py` fails if hosted code imports training code.
- **API skeleton:** `GET /health` works locally (tested with uvicorn and `tests/test_serving.py`). The draft
  contract for `/v1/embed`, `/v1/match`, `/v1/safety/medications` and `/v1/safety/dose` is in `serving/README.md`.
  The server is stateless: galleries and schedules stay on the phone.
- **Not tested:** the Docker image (Docker isn't installed here); it's built by Hugging Face when the Space is set up.
- **Open for you:** privacy (crops leave the phone with a hosted model) and licences (DDInter NC-SA, C3PI research
  use) before the Space goes public; see `serving/README.md`.

---

## 2026-10-07 (evening): Cropper fixes, re-crop, splits and leakage report. Gate G2 ready for sign-off

- **Cropper fixes** (`src/epills/datasets/c3pi_crop.py`), for the review failures:
  - White pill on white paper: the shadow out-contrasted the pill. A shadow-like winner (darker, colourless) is now
    merged with the pill beside it; white and grey pills get a wider window.
  - Granite with a dark gap behind the counter edge: a second, local background model handles the step.
  - Fixed 12 of 13 collected failures; on 150 fresh photos, side by side, nothing got worse. Crops are framed as
    tightly as before (median pill length 71% of the window).
- **Re-crop:** all 40,587 photos cropped (first version: 40,550). Review sample (seed 2027): 199 of 200 clean; #146
  is a capsule half out of frame on sheet metal. The first run is kept as `crops_v1_superseded/`.
- **New `scripts/make_splits.py`** -> `data/processed/c3pi_splits/` (manifest, groups, leakage report, summary):
  - **Pill type = NDC9 + part.** 26 NDC9s are multi-pill packs (2-5 different pills under one code).
  - **Appearance groups:** pill types with identical imprint, shape, colour and size are one physical pill under
    several codes (repackagers, brand vs generic from one maker): 111 pill types in 49 shared groups; 2,400 pill types
    -> 2,338 groups. Splits are by group, and the training label is the group.
  - **ePillID holdout (fold 4)** removed by group: 204 pill types, plus 14 look-alikes of holdout pills in other
    products (e.g. Junel / Loestrin, Nortrel / Balziva).
  - **Splits (seed 2026):** train 29,471 photos / 1,710 groups; val_enrolled 3,556 / 214; val_unknown 3,555 / 214;
    epillid_holdout 4,005 / 200. Median 15 photos per group in every split.
- **Leakage checks:** 0 groups and 0 identical files across splits. Copies of the same photo across splits: 0.
  6,087 look-alike pairs were checked; a copy must match on centre texture and pixels (calibrated on JPEG-70
  re-encoded, shifted copies). Visual checks of the closest pairs: all different pills that share camera, paper and
  position. 9 NDC9s span two splits; all are multi-pill packs whose parts look different, so no leak.

---

## 2026-10-07 (later): C3PI metadata table and pill crops. Gate G3 passed; G2 waiting on review and dedup

### Metadata table (`scripts/parse_c3pi_metadata.py` → `data/interim/c3pi/metadata.csv`)
- One row per image from all 110 `ALLXML` files: 133,458 images.
  - The XML lists 316 consumer files twice, with identical entries; the repeats are dropped.
  - The table matches the download: 49,703 downloaded, including the 40,587 consumer JPGs.
- Columns: identity (NDC9/NDC11, part, names), attributes (shape, colour, size in mm, imprint, imprint type and
  colour, score, symbol), capture (camera, illumination, background, polarizer, clarity ratings), file size,
  SHA-1 and a `downloaded` flag. Two-tone colours are kept as `A;B`.
- **Gate G3:** for the 2,362 consumer pill types (NDC9):
  - Shape, colour, size, imprint type and score are present for 100%; imprint for 99.7%.
  - Photos of the same type disagree for a few types: shape 4, colour 32, size 17, imprint 66. Use the
    majority value.
  - 227 types are two-tone, so the colour head should be multi-label (ROADMAP §1.4).
- The consumer photos cover 27 backgrounds, 9 cameras and 11 lighting setups.
  - The grey cards and white papers make up about 70%.
  - Hard surfaces are rare (3.8% in total): granite, a human hand, dark laminate, denim, the counting tray,
    sheet metal and an amber pill bottle.

### Crops (`src/epills/datasets/c3pi_crop.py`, `scripts/crop_c3pi.py` → `data/interim/c3pi/crops/`)
- **Method:** classical, no learned model.
  - A robust quadratic background surface in Lab.
  - Colour-difference blobs over a threshold sweep, scored by contrast-to-noise against a ring around the blob,
    convexity and closeness to the centre.
  - A size prior: the expected pill length is the camera's px/mm times the metadata size in mm.
  - GrabCut refinement, so the full outline of a pale pill is found.
  - A square window 1.4× the pill's long side. It is widened to 2.2× the expected length when only part of the
    pill (or its shadow) was found, judged by length and by area against the shape.
- **Scale calibration** (4,000 random photos, `scale_table.csv`): px/mm is consistent within each camera and
  image width, with relative IQR 0.16–0.31. Examples: Canon G10 75.2, Kodak M590 50.3, Nikon S3000 38.3,
  Vivitar 27.3, webcam 21.7.
- **Run:** 40,587 photos in 21 min (14 processes, 32 photos/s).
  - 40,550 cropped (384×384 JPEG q95, 1.5 GB) and 37 with no pill found.
  - 1,645 windows widened to the expected size.
  - 4,470 photos show a second pill of the same kind; only one is cropped, and the photo is flagged
    `second_pill`.
- **Pill types:** 2,362 NDC9 with crops (min 3, median 15, max 101); 2,357 have ≥ 4.
- **Quality, checked by eye:**
  - Prototype iterations (150 random photos): v1 failed often; v2 about 90%; v3 about 93%.
  - The final method on a fresh random 200: 8 failures (96%).
  - The run's own review sample (`review_2026_*.jpg`, 200 random crops): **5 failures (97.5%)**. Four cut part
    of a pill at the edge (numbers 36, 76, 105, 133); one granite photo missed the pill (150).
  - Overall 13 failures in 400 (96.8%). The gate needs ≥ 90%.
  - Remaining failure modes: pale pills where only the shadow was found and the window is off-centre; granite;
    some hand photos.
- **Tests:** `tests/test_c3pi_crop.py` (synthetic pill, size-prior widening, scale table), 3 passed.
- **Environment:** `epills/.venv` (Python 3.13) from `requirements-data.txt`: numpy, opencv-python-headless,
  pandas, pillow, pytest. Model packages wait for D1.

### Next
- **You:** look through `data/interim/c3pi/crops/review_2026_0.jpg` and `_1.jpg` (what each number is:
  `review_2026.csv`).
- **Claude:** dedup and splits (ROADMAP §1.3) with a leakage report:
  - Map ePillID's consumer images to C3PI files by pixels.
  - Remove the ePillID holdout NDC9s.
  - Split the rest by type into train / validation-enrolled / validation-unknown.
  - Check near-duplicates across splits.
- **Optional:** crop the second pill too (+4,470 crops, often the other side), and crop the reference images.

---

## 2026-10-07: ePillID and DDInter on the desktop; official holdout fold confirmed. Gate G1 passed

- **Why:** `data/` isn't in git, so the 2026-10-06 downloads stayed on the laptop. Training runs on the desktop.
- **New `scripts/download_small_sources.py`:** downloads and checks ePillID and DDInter, reusing `fetch` from
  `download_c3pi.py`. Rerunnable.
  - **ePillID:** the SHA-256 matches `docs/datasets.md`. 13,532 labelled images (3,728 consumer, 9,804
    reference), 4,902 pill types, 960 with consumer images, 5 fold files; unzipped to `data/raw/epillid/`
    (389 MB).
  - **DDInter:** 14 files, 234,981 unique pairs over 1,971 drugs, the same as on the laptop. In
    `data/safety/ddinter/`.
- **Official holdout fold,** from the ePillID source (master branch):
  - **Fold 4:** 745 consumer images, 192 pill types (189 NDC9), none shared with folds 0–3. `train_cv.py` holds
    out the last sorted fold file; `arguments.py` and `configs/params.json` default to fold 4 as the test set.
  - The gallery holds the reference images of all 4,902 types, including the 384 of the holdout types.
- **Published metrics are two-sided:**
  - Queries are simulated front + back pairs of consumer photos of one pill type.
  - A type's score is the mean of the two sides' best-reference similarity (`sidepairs_agg=post_mean`).
  - The Papers-with-Code leaderboard is "Pill classification (both sides)".
  - Details in ROADMAP §1.3.

---

## 2026-10-06 (later): C3PI download script; full download started on the desktop. Gate G2

- **Where:** Claude ran it on the desktop (RTX 4080 SUPER PC, Windows Python 3.13, 1 TB free), from a desktop
  Claude session.
- **Script:** `scripts/download_c3pi.py` (standard library only).
  - Downloads metadata first, then the selected images: references first, then consumer photos one per pill
    type per round.
  - Checks every image against its SHA-1 in `ALLXML`.
  - Resumable: a file only gets its final name after its checks; reruns skip finished files; Ctrl+C stops
    cleanly.
- **Metadata:** all of `meta/` and the 110 `ALLXML` files (125 MB) are now in `data/raw/c3pi/` on the desktop.
  The directory's SHA-256 matches the laptop's copy.
- **Selection:**
  - 49,703 images, 88.3 GB: MC_CHALLENGE_V1.0 4,408, MC_API_RXNAV_V1.3 4,708, C3PI_Test 40,587 JPGs.
  - Correction: 295 of the 40,882 `C3PI_Test` JPG lines repeat an earlier line exactly, so there are 40,587
    distinct consumer photos (83.4 GB).
  - Every selected image has a SHA-1 in `ALLXML`.
  - No file names that are unsafe on Windows, and no case-only name collisions.
- **Test:** 700 images.
  - No failures and no checksum mismatches.
  - A rerun skipped all 700, and `--verify` re-hashed 500 with no mismatches.
  - The first 100 consumer photos covered 100 different pill types.
  - Throughput: about 7.5 MB/s with 8 parallel downloads and about 15 MB/s with 16 (now the default), so
    about 1.5–2 h for everything.
- **Full run:** 2026-10-06 22:28 → 2026-10-07 00:10. The log is `data/interim/c3pi/download_log.txt`.
  - Paused at 39.7 GB on your request, to check the size: the archive's 2.6 TB is the full archive; our
    selection is 88.3 GB.
  - Resumed at 23:23. The resume skipped the 26,489 finished files and replaced the 9 partial ones.
  - **Result:** all 49,703 images present, every one checked against its SHA-1, 0 failures, 0 leftover
    `.part` files.

    | Image type | Images | GB | Pill types (NDC11 + part) |
    |---|---|---|---|
    | C3PI_Test (consumer) | 40,587 | 83.37 | 2,432 |
    | MC_CHALLENGE_V1.0 | 4,408 | 2.20 | 2,148 |
    | MC_API_RXNAV_V1.3 | 4,708 | 2.69 | 4,676 |

  - On disk: 83 GiB in `data/raw/c3pi/images/`; 926 GB free afterwards.
  - Consumer photos per medicine:
    - By NDC9: 2,362 medicines, minimum 3, median 15, 10th percentile 11, 90th percentile 30, maximum 101.
      1,950 medicines have 10–19 photos.
    - By NDC11 + part: 2,413 of 2,432 types have ≥ 4.
  - Speed: about 5 MB/s on the small reference images, 14–21 MB/s on the consumer photos.
- **`rximage.zip`: not needed for images.**
  - Its originals are the same pictures as `MC_API_RXNAV_V1.3` / `MC_API_NLMIMAGE_V1.3`: listed via HTTP range
    requests, the file sizes differ by exactly 10 bytes. The disc folders cover 295 more NDC11s than the zip.
  - Its 11 MB of metadata (`table.csv`, `MONGOexport/rximagesAll.json`) is useful: RxCUI per NDC (Stage 3),
    plus shape, colour, size and imprint. It can be extracted from the zip without downloading it (offered,
    not done).
- **Not downloaded:** `rximage.zip` (7.3 GB; `--with-rximage` fetches it, with resume and MD5 check).
- **Note:** this is the first `epills` work on the desktop. The laptop's `data/` (ePillID zip, DDInter CSVs) is
  not here; copy or re-download it when training moves to the desktop.

---

## 2026-10-06: Project set up; data sources checked. Status: P0 / P1

### Done (Claude)
- Created `epills/` with the folder structure (see [README.md](README.md)), the roadmap, this log, the dataset
  notes and the own-pills photo protocol.
- **ePillID:** downloaded the release (160.7 MB) to `data/raw/epillid/` and checked the labels and fold files.
  - 13,532 images (3,728 consumer, 9,804 reference); 4,902 pill types, 960 with consumer images.
  - Pre-cropped at 224 px.
  - 5 fold files of about 192 pill types each.
- **C3PI:** analysed the archive without downloading it.
  - Read the image directory (133,774 rows) and all 110 disc listings, and joined them.
  - The full archive is 2.72 TB. The useful consumer photos are the **40,882 `C3PI_Test` JPGs: 2,394 pill types,
    84 GB**, median 15 photos per type.
  - References: `MC_CHALLENGE_V1.0` (2.2 GB) and RxImage (2.7 GB).
  - One metadata XML checked: it has shape, colour, size in mm, imprint, imprint type, capture conditions and
    SHA-1 per image.
  - Two sample photos viewed: each is one pill on a plain background, small in the frame, so it needs a crop.
- **ePillID vs C3PI overlap** (NDC9):
  - All 960 ePillID consumer pill types are in C3PI.
  - Each ePillID fold is about 190 pill types and about 3,900 C3PI photos.
  - About 2,170 types and 37k photos remain for training after removing one fold.
- **DDInter 2.0:** downloaded all 14 ATC-letter CSVs to `data/safety/ddinter/`. The page links only 8, but the
  other 6 are served at the same URL pattern.
  - 234,981 unique pairs over 1,971 drugs, with severity only. 18% of pairs are "Unknown".
  - No management text or duplications in the bulk files.
- **APIs:** the RxNorm API is live; the RxNav interaction API returns 404; the openFDA label API is live.
- **Cross-domain sets:**
  - VAIPE: vaipe.org doesn't respond, and the VAIPE-PCIL Google Drive link returns 404.
  - **MEDISEG** found as the replacement: 8,262 photos, 32 pill types, CC BY 4.0, 0.45 GB on figshare, files
    checked.

### Corrections to the original plan
- **C3PI** has about 41k consumer photos, not 133k. The 133k counts every file, including RAW files, video and
  calibrated PNGs.
- The **official ePillID protocol** trains on reference images of every pill type. A strict protocol is added
  to support the "works for unseen medications" claim.
- **DDInter's** bulk download lacks advice text and duplications, and its counts differ from the paper's.

### Next: Claude
1. `scripts/download_c3pi.py`:
   - downloads `ALLXML` first, then the selected image types
   - resumable, checked against SHA-1, with priority by photos per pill type
   - tested on about 200 files here
2. Parse all `ALLXML` into a metadata table and report attribute coverage (gate G3).
3. Confirm the ePillID holdout fold from the ePillID code.
4. Download and check MEDISEG.
5. Draft the two emails (VAIPE authors, DDInter authors).
6. Prototype the C3PI crop step on the sample photos.

### Next: You
1. Decide D1–D6 (ROADMAP §6). D1 matters first, because it decides the environment.
2. Free about **110 GB** on the desktop for C3PI.
3. Start choosing the over-the-counter pills (protocol §1), and check whether the Stage 1 mat exists or should
   be designed.
4. When the email drafts arrive: send them.

---

## Template: data step

```
## YYYY-MM-DD: <dataset> <step>. Gate <Gn>

- Who ran it, where (laptop / desktop), script and version
- Input: files, size; output: files, size, location
- Counts: pill types, images, per-type min / median / max
- Failures: checksum mismatches, crop failures (rate), dropped items and why
- Leakage check (if a split was made): NDC9 overlap, md5 duplicates, perceptual-hash near-duplicates
- Gate status after this step:
```

## Template: training run

```
## YYYY-MM-DD: run <n> (<name>, config <file>)

- Machine, PyTorch / CUDA versions, bf16 on/off
- Changes from the previous run (one factor at a time): loss (ArcFace / triplet), refine pool, aux heads,
  colour jitter, reference photos, measured inputs
- Data: training groups and photos; P x K per phase
- Phase 1 (ViT and VGG blocks 1-3 frozen): epochs run, best epoch, time per epoch
- Phase 2 (all trainable): epochs run, best epoch, time per epoch, peak VRAM
- Best validation: gallery-10, top-1 all, unknown AUROC, correct@95%-reject, wrong accept@95%
- Against the targets: met / not met, and what to change next
- (Final model only) ePillID official: MAP, GAP, MAP@1, GAP@1 (paper's best 95.76 / 81.20 / 92.01 / 91.19);
  MEDISEG; own pills
- Notes / anomalies:
```
