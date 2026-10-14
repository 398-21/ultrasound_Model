# Blueprint review: what PharmaScan still lacks

Reviewed 2026-10-12. Source: the team's blueprint `epills/PharmaScan_Care_Agent_Workflow.html` ("PharmaScan Care —
Agent Workflow & API Blueprint", Hackathon Challenge 2), written against API v0.4.0. Compared with API v0.5.0 (live
since today) and the code in this repository. Work through the jobs in section 5 one at a time; tick them off here.

## 1. The blueprint in brief

- **The product is an agent, not a classifier.** A Medication Care Agent plans, calls typed tools, checks the evidence,
  asks again when unsure, and remembers across sessions. Our API is one of its tools ("PharmaScan Vision API"), next
  to Document/Package OCR, Medication Knowledge, and a **deterministic safety gate** in the main backend.
- **Main actor is the patient.** A linked caregiver gets limited notifications with consent. **No doctor portal and
  no NEHR sync**: the patient adds medicines from a prescription PDF, pharmacy label, box or by hand, and confirms them.
- **The main app (FastAPI + PostgreSQL) owns all state:** regimens (versioned), personal galleries (vector(256)),
  dose events and history. Our API stays stateless.
- **Six flows:** architecture; enrolment; daily dose; unknown item; long-term care (adherence, health tracking,
  preventive tasks, review summary); agent loop (observe → plan → tool → evidence gate → clarify or act → memory).
- **Six trust boundaries:** prescriptions and labels are the authority, model output is only a visual candidate;
  galleries live in the main app; embedding spaces must match; *recognised ≠ should take ≠ swallowed* (separate
  `vision_status` and `intake_status`); a deterministic policy decides actions; demo and clinical use are separate.
- **An API decision table** (20 endpoints, P0–P2, each with an acceptance test), **5 demo scripts** (A–E), and a plan
  whose first slice is: manual enrolment → single loose pill → real `/v1/embed` + `/v1/match` → safety gate → the
  patient confirms intake → dose event. "Don't retrain the model before submission."
- **Reported failure:** a single TEVA pill photo through `/v1/scan` gave **two overlapping boxes**, then catalogue
  candidates with similarity **0.97–0.99** that weren't the pill.

## 2. Checked today: the two reported failures are real and general

- **One pill → several boxes.** The stand-in finder (`core/inference/pill_finder.py`) on 300 single-pill C3PI
  consumer photos: **2 or more boxes on 215 (72%)**, one box on 85. The extra boxes are mostly small fragments beside
  the pill (shadows, glare, pieces): median 10% of the pill's area; of 151 extra boxes in 120 photos, 106 touch or
  nearly touch the pill, 13 overlap it, 32 are elsewhere. So `/v1/scan` without detector boxes can't be trusted to
  count pills. The fix is mechanical (merge or drop fragments, flag the rest); see B1.
- **High similarity means little.** For MEDISEG pills, which are **not in the US catalogue at all**, the best catalogue
  match still scores a median **0.929**; 25% score ≥ 0.95 and 7% ≥ 0.97, with the runner-up a median 0.009 behind.
  A raw cosine of 0.97 is not "97% sure", and `/v1/identify` never says "no confident match". See B4.

## 3. Already covered by v0.5.0 (the blueprint predates it)

| Blueprint item | Now |
|---|---|
| Package OCR ("有字包装", en_ocr / ds_ocr / un_print: "待开发") | `POST /v1/read-label`: boxes and pharmacy labels → name, strength, ingredients, RxNorm code; correct on 9 printed examples. **Missing:** directions/sig, quantity, dates, PDF input, real-photo testing (B7, B8) |
| Imprint for loose pills (un_loose) | `/v1/identify` reads the imprint and re-ranks (ePillID first choice 42% → 66%); gallery imprint check in `/v1/match`, `/v1/scan` |
| Gallery is a request payload, not stored on HF | Yes, by design: the API stores nothing |
| `/v1/match` threshold, margin, unknown/ambiguous | Returned with every answer |
| Reject cross-version comparison | `409` when `model_version` is sent; **but it's optional** (B3) |
| Safety coverage ("checked/unchecked") | Per medication `not_in_interaction_database` and `unresolved` alerts; **no summary, no "no alert ≠ safe"** (B10) |
| Feedback off by default, opt-in, withdrawable | Yes; **deletion leaves the photos in the dataset repo's git history** (B15) |
| Cold start / model status | `/health` shows loaded parts and device; **no `ready` flag, no `Retry-After`** (B5) |

## 4. Where our work and the blueprint disagree: decisions for you

- **D1. Where prescriptions come from.** The blueprint has no doctor portal and no NEHR sync: patients upload or type
  their medicines and confirm them. Our walkthrough pages (English and Chinese) assume prescriptions arrive from the
  doctors' systems (simulated). *Recommendation:* follow the blueprint for the hackathon (patient enrolment through
  `/v1/read-label` + confirmation) and keep the doctor feed as "later"; then I update both walkthrough pages.
- **D2. What a gallery is built from.** The blueprint: the patient's own loose-pill photos, taken when they actually
  take the pill; packaging photos are kept but never embedded. Our earlier idea: galleries built from catalogue
  pictures (not built). Catalogue pictures may not be the product the patient holds. *Recommendation:* the patient's
  own photos are the verified gallery; catalogue pictures only as a lower-trust starter, marked as such, and only after
  measuring their accuracy (B12).
- **D3. Where dose rules live.** The blueprint moves them to the main backend (only it knows actual times, PRN,
  stopped medicines, unconfirmed doses). Ours run inside `/v1/scan` and `/v1/safety/dose`. *Recommendation:* the
  backend owns the decision; we keep ours as a stateless helper and hand over the rules as a spec with test cases
  (B11).
- **D4. Who finds the pills.** Options: a new `/v1/detect` in our API, the Stage 1 teammate's YOLO, or boxes drawn in
  the app. *Recommendation:* we ship `/v1/detect` now (fixed finder + quality flags) in the agreed box format; the
  Stage 1 model replaces the finder behind it later (B6).
- **D5. Wording of a match.** Our API says "This is your Metformin 500 mg." The blueprint's rule: "appearance matches
  the record" ≠ "safe to take". *Recommendation:* "This pill looks like your Metformin 500 mg." (B2).
- **D6. Language of messages.** The API answers in English; the blueprint is in Chinese. Should messages also come in
  Chinese (or other languages)? (B18)

**Decided 2026-10-12:** D1 skipped for now. D2–D6 as the blueprint proposes:
- **D2:** galleries come only from the patient's own loose-pill photos; packaging photos are kept by the app, never
  embedded; catalogue pictures stay a visual reference (no catalogue-built galleries).
- **D3:** dose rules move to the main backend; `/v1/scan` becomes candidates and diagnostics only, its dose check
  deprecated; `/v1/safety/dose` stays as a stateless reference implementation.
- **D4:** add `/v1/detect` (boxes, pill count, quality flags); the app may also send its own boxes.
- **D5:** "looks like your …"; visual evidence only.
- **D6:** the blueprint has the agent explain results to the patient, so the API returns structured statuses (with
  English messages as defaults) and the agent words them in the patient's language. B18 dropped.

**Submission: 16 October 2026.** Staged plan:

| Stage | Jobs | Target |
|---|---|---|
| 1. Contract and safety (P0) | B2 wording · B3 embedding contract + `/v1/model-info` · B5 response schemas, errors, readiness · B1 + B6 pill finder fix and `/v1/detect` · B11 dose rules out of `/v1/scan` · B4 calibrated confidence | 13 Oct |
| 2. Tools the agent needs (P1) | B13 product identity fields · B10 safety coverage and citations · B9 package check at dose time · B7 document extraction · B14 tool manifest · B12 look-alike strengths · B16 licence note | 15 Oct |
| 3. Before submission | B21–B27 (from the team's api_review_and_test_plan.md, 13 Oct) · deploy · end-to-end run of the blueprint's first slice · evidence for judging · docs frozen | 15–16 Oct |
| After | B8 (needs your photos) · B15 · B17 · B19; D1 | — |

## 5. Jobs

Owner: **API** = us (Stage 2/3 service); **App** = the web/backend team; **S1** = the Stage 1 teammate.
Priority as in the blueprint's table where it has one. Tick the box when done.

| ID | Job | Owner | Priority | Done |
|---|---|---|---|---|
| B1 | Fix one pill → several boxes; regression tests; quality flags in `/v1/scan` | API | P0 | ✅ 12 Oct: count right on 86–88% of C3PI photos with 1–3 pills (was 28% on single pills) |
| B2 | Honest wording: "looks like", vision vs intake, scores aren't probabilities | API | P0 | ✅ 12 Oct |
| B3 | Embedding contract: `embedding_space_id`, `preprocess_version`, `/v1/model-info`, required version check | API | P0 | ✅ 12 Oct |
| B4 | Calibrated confidence and "no confident match" for `/v1/identify` | API | P0 | ✅ 12 Oct: calibrated on held-out medicines |
| B5 | Response schemas in OpenAPI, error contract, cold start and `Retry-After`, determinism test | API | P0 | ✅ 12 Oct |
| B6 | `/v1/detect`: boxes, pill count, overlaps, quality flags (Stage 1 model plugs in later) | API + S1 | P1 | ✅ 12 Oct (stand-in; Stage 1 model plugs in later) |
| B7 | Document extraction: directions, quantity, dates, field provenance, PDF input | API | P1 | ✅ 13 Oct: printed fields with source lines; PDF (text and scanned) |
| B8 | Test `/v1/read-label` on real phone photos of boxes and labels | API + you | P1 | ☐ |
| B9 | Package check at dose time: "is this box one of my medicines, same strength?" | API | P1 | ✅ 13 Oct: `list_check` in `/v1/read-label` |
| B10 | Safety coverage summary; "no alert ≠ safe"; label citations | API | P1 | ✅ 13 Oct |
| B11 | Dose rules as a spec + test cases for the backend; `/v1/scan` dose part becomes a helper | API → App | P0 | ✅ 12 Oct: docs/dose_rules.md |
| B12 | Measure look-alike strengths (500 vs 850 mg, IR vs XR) | API | P1 | ✅ 13 Oct: 714 sibling pairs measured |
| B13 | Product identity fields in `/v1/pills/*`: strength, dose form, release, brand/generic, market | API | P1 | ✅ 13 Oct |
| B14 | Tool manifest for the agent (typed tool definitions from OpenAPI) + example traces | API | P1 | ✅ 13 Oct: `GET /v1/tools`, docs/agent_tools.md |
| B15 | Feedback: really delete on withdrawal (git history), audit log, label review queue | API | P2 | ✅ 13 Oct: history squashed, deletion checked; audit.jsonl; `monthly_retrain.py review` |
| B16 | Licences and regulatory note for the submission | API + you | P1 | ✅ 13 Oct: docs/licences.md (items to confirm marked) |
| B17 | `/v1/search-by-embedding` (identify without re-running the model) | API | P2 | ✅ 13 Oct: same candidates as `/v1/identify` (tested) |
| B18 | ~~Messages in other languages~~: dropped (D6: the agent words the answers) | — | — | — |
| B19 | Input type router (loose pill / printed package / blank blister / document), or leave it to the user and agent | API or App | P2 | ✅ 13 Oct: `POST /v1/check-input` (tool `route_medication_image`) |
| B20 | App side: regimen + provenance DB, enrolment, dose-check orchestrator, agent session, care summary, caregiver consent | App | P0–P1 | ☐ |
| B21 | No silent fallbacks: say when text reading (imprints, labels) is unavailable | API | P0 | ✅ 13 Oct (v0.8.0): `text_reading` status on every answer |
| B22 | Output validity: refuse non-finite vectors; contract tests (same image → same vector, dim, version) | API | P0 | ✅ 13 Oct (v0.8.0) |
| B23 | Input guard: loose pills vs printed package vs blister vs document; vision endpoints abstain on blisters and foil (test cases P02, P03) | API | P0 | ✅ 13 Oct (v0.8.0): `input_check`; a net, not a guarantee |
| B24 | D01 residual (2% unflagged double counts): documented; the backend confirms before recording an extra pill | API → App | P1 | ✅ 13 Oct: measured 1.4%; `confirm_count`, dose_rules.md item 8 |
| B25 | `/v1/model-info`: how similarity is computed and the measured error rates with their sources | API | P1 | ✅ 13 Oct: `similarity`, `error_rates` (checked against the files by a test) |
| B26 | Privacy in `/v1/match`: pseudonymous ids, names optional, `reference_set_id` passed through | API | P1 | ✅ 13 Oct: NRIC/FIN-shaped ids refused, never echoed |
| B27 | Same-manufacturer imprints (e.g. Teva "93"): confusion between different medicines measured | API | P2 | ✅ 13 Oct: 0.01% over 3,149 pairs; shared parts no longer confirm a match |

### Details

**B1. One pill → several boxes (P0).**
- *Why:* blueprint demo C; measured in section 2 (72% of single-pill photos).
- *Do:* in `find_pills`, merge boxes that overlap or nearly touch, and drop fragments much smaller than the largest
  pill unless clearly separate. Return quality flags (overlap, tiny box, touching the edge, low contrast) and expose
  them in `/v1/scan` per pill and per photo. Never count pills from boxes that carry a flag.
- *Done when:* the blueprint's tests pass: 1 pill → 1 box, overlapping boxes merged, 2 pills → 2 boxes, minimum box
  size. Re-run the 300-photo check (target ≥ 95% one box) and add a synthetic regression test. The team's TEVA photo
  should be the first test case: **please put it in the repo** (e.g. `tests/data/teva_single.jpg`).

**B2. Honest wording (P0).**
- *Why:* trust boundaries 01 and 04; `ds_explain`. "This is your X" claims identity from appearance.
- *Do:* messages say "This pill looks like your Metformin 500 mg." Every Stage 2 answer carries
  `"evidence": "visual"` and a short `limits` note. `dose.summary.all_ok` gets a sibling such as `all_visually_matched`,
  with a deprecation note for `all_ok`. State in API.md that similarities are not probabilities.
- *Done when:* no Stage 2 message says "is", "safe" or "take"; tests check the wording.

**B3. Embedding contract (P0).**
- *Why:* blueprint rows `/v1/embed`, `/v1/match`, `/v1/model-info`; trust boundary 03.
- *Do:* `/v1/embed` returns `embedding_space_id` (model version + crop contract + modality, e.g.
  `pill-crop:stage2_run4@2026-10-09:w1.55`), `preprocess_version`, `normalized: true`. Add `GET /v1/model-info`: service
  version vs model version, embedding space, threshold and margin, crop window, training coverage (C3PI US pills),
  known limits (other phones: MEDISEG 48%). `/v1/match` and `/v1/scan` refuse a gallery without a version, or with a
  different one. Write the contract section in API.md (this is F9 in features.md).
- *Done when:* same image twice gives the same vector (cosine ≥ 0.9999; GPU and CPU differ only by rounding); a
  mismatched gallery gets `409`; dimension and NaN checks.

**B4. Calibrated confidence for `/v1/identify` (P0).**
- *Why:* blueprint row `/v1/identify` ("non-catalogue rejection", "scores aren't probabilities"); section 2.
- *Do:* on validation, map (top-1 similarity, gap to the runner-up, imprint read) to the chance that the top-1 is
  right (isotonic or logistic fit). Return it as `confidence` with a band (low/medium/high). Add a `no_confident_match`
  verdict when a pill most likely isn't in the catalogue, chosen on validation with MEDISEG-like non-catalogue pills
  as negatives.
- *Done when:* on the test sets the stated confidence matches the observed accuracy (calibration curve); report how
  many non-catalogue pills are correctly told "no confident match" and how many catalogue pills are wrongly refused.

**B5. API contract and failure handling (P0).**
- *Why:* blueprint "NOW · 阻塞": "真实 Response Schema", "外部 API 失败处理"; `/health` row.
- *Do:* Pydantic response models for every endpoint, so `/openapi.json` documents the answers; one error shape
  (`code`, `detail`, `retry_after`). `/health` gains `ready` (all parts loaded) and the start time; `503` carries
  `Retry-After`. Document the sleep / cold-start behaviour (HF answers with a page, not JSON, while waking) and the
  GPU-allowance fallback for the app's retry logic.
- *Done when:* every endpoint's response is in the OpenAPI file; tests cover cold start and a missing model.

**B6. `/v1/detect` (P1; decision D4).**
- *Do:* photo → `boxes`, `pill_count`, per-box `quality_flags`, `found_by`. Same box format as Stage 1
  (`[x0, y0, x1, y1]` in photo pixels). The Stage 1 detector replaces the finder behind it when ready.
- *Done when:* the B1 tests pass through this endpoint; `/v1/scan` uses it.

**B7. Document extraction (P1).**
- *Why:* enrolment flow (en_ocr: name, strength, dose form, **directions**), demo A (prescription PDF).
- *Do:* extend `/v1/read-label` (or add `/v1/read-document`): directions as printed ("Take 2 tablets 4 times a day"),
  quantity, dispensing date, dose form; each field with the line it came from; `null` when not printed; never
  inferred. Accept PDF (text layer first, else render pages for OCR). Never return the patient's name.
- *Done when:* a synthetic prescription PDF and dispensing labels give the right fields; missing fields stay `null`.

**B8. Real photos for `/v1/read-label` (P1).**
- *Do:* 20–30 phone photos of real boxes and Singapore pharmacy labels (with the patient's name covered); measure name,
  strength and RxNorm code accuracy; fix what fails.
- *Needs from you:* the photos.

**B9. Package check at dose time (P1).**
- *Why:* daily dose flow ("有字包装：OCR → 匹配产品、规格、剂型").
- *Do:* `/v1/read-label` optionally takes the patient's medication list and answers per medicine: same product, same
  ingredient but **different strength or form**, or not on the list. A deterministic comparison, no model score.

**B10. Safety coverage and citations (P1).**
- *Do:* `/v1/safety/medications` adds a `coverage` summary (medications checked, ingredients not in DDInter, pairs
  not checkable) and the sentence "No alert doesn't mean the combination is safe." `/v1/safety/label` returns the
  source URL, label set id and effective date for citation (the blueprint's Medication Knowledge tool).

**B11. Dose rules handed to the backend (P0; decision D3).**
- *Do:* write the rules as a spec with test cases (missing pill, extra pill, double dose, unknown pill, unsure pill,
  not scheduled) and the cases the backend must add: actual time taken, PRN, stopped medicines, unconfirmed events,
  bad boxes don't count. Keep `/v1/safety/dose` as a stateless reference implementation.

**B12. Look-alikes and catalogue galleries (P1).**
- *Why:* blueprint acceptance for `/v1/match` and `/v1/pills/search` ("相似规格", 500 vs 850 mg, IR vs XR); D2.
- *Do:* from C3PI, pairs of the same ingredient in different strengths or release forms: how often the model tells
  them apart, with and without the imprint check. Also galleries built from catalogue pictures (reference photos) vs
  from the patient's photos: named correctly / wrongly.

**B13. Product identity fields (P1).**
- *Do:* `/v1/pills/search` and `/v1/pills/{id}` return `strength`, `dose_form`, `release` (IR/ER), `brand_or_generic`,
  `market: "US"`, the NDC, and the image licence ("C3PI: research and development use"); products are never merged
  across strength or form.

**B14. Tool manifest for the agent (P1).**
- *Do:* typed tool definitions (name, description, input schema, output schema) for embed, match, identify,
  read-label, safety checks, generated from the OpenAPI file, plus 2–3 example tool traces from the demo scripts.

**B15. Feedback deletion (P2).**
- *Found:* `DELETE /v1/feedback/{consent_id}` deletes the folder in the dataset repo, but the photos stay in the
  repo's git history, so "deleted" isn't true yet.
- *Do:* squash history after a withdrawal (or keep photos out of git), keep an audit log, and add a review queue
  before `user_confirmed` labels could ever be used. Collection stays off for the hackathon.

**B16. Licences and regulatory note (P1).**
- *Do:* one page for the submission: DDInter 2.0 (CC BY-NC-SA: non-commercial), C3PI images (research and development
  use), RxNorm/RxNav terms, openFDA, Florence-2 (MIT), the model's own data; HSA software-as-a-medical-device risk
  framing (the blueprint says not to assume exemption). Your call on what the hackathon counts as.

**B17–B19 (P2).** Embedding-based identify; messages in other languages (D6); an input type router (the blueprint
prefers the user's stated intent, with the agent asking when unclear, so this can stay in the app).

**B20. App side (not ours, listed so nothing falls between teams).** Patient regimen with provenance and versions;
enrolment with draft/pending states; dose-check orchestrator that stores `vision_status` and `intake_status`
separately; agent session with tool traces; care summary; scheduler; caregiver consent and notifications. Our jobs
B2, B3, B5 and B11 are what the app needs from us first.

## Stage 1 results (12 Oct, API v0.6.0)

- **B1/B6 pill finder:** the extra boxes were mostly *background*: vignetting and uneven light made corners
  "pills", and faint pills were found only in part. Now each pixel is compared with a robust smooth model of the mat
  (degree-3 surface per colour channel), faint parts join the clear part they touch (hysteresis), only pill-shaped
  blobs are kept, pieces merge and crumbs drop, and every box carries quality flags. `scripts/evaluate_pill_finder.py`
  (C3PI photos with known boxes; 2- and 3-pill photos made by copying the pill): count right 88% / 86% / 86% for 1 /
  2 / 3 pills (before: one box on 28% of single-pill photos); fully right (count and IoU ≥ 0.5) 76% / 71% / 71%;
  wrong and unflagged 13%. Still hard: white pills on white, speckled worktops. `POST /v1/detect` returns boxes,
  flags, `pill_count` and `count_reliable`; `/v1/scan` doesn't count flagged boxes.
- **B2 wording:** "This pill looks like your …"; every Stage 2 answer says `"evidence": "visual"` with a `limits`
  note; dose alerts say "not seen" instead of "missed"; `all_visually_matched` replaces `all_ok` (kept, deprecated).
- **B3 contract:** `embedding_space_id` and `preprocess_version` on every embedding; a gallery without its version
  is refused (`422`), a different one `409`; `GET /v1/model-info`.
- **B4 confidence:** `/v1/identify` returns `confidence`, `catalogue_fit`, `verdict` (`scripts/calibrate_identify.py`;
  fitted on validation, tested once): stated 0.93 → 95% right, 0.71 → 66%, 0.31 → 33%; 71% of pills that aren't in
  the catalogue get `no_confident_match`, 9% wrongly get `possible_match`.
- **B5 contract:** response models for every endpoint in `/openapi.json`; one error shape (`code`, `detail`,
  `retry_after`); `Retry-After` on `503`; `/health` has `ready` and `started_at`.
- **B11:** `/v1/scan` is diagnostics, its dose check deprecated and only run on a reliable count;
  `docs/dose_rules.md` gives the backend the rules, test cases and the seven checks only it can do.
- 69 tests pass (new: pill finder, contract, confidence, not-ready, unsure count).

## Stage 2 results (13 Oct, API v0.7.0)

- **B13 product identity:** every catalogue product (search, identify, scan) carries `strength`, `strengths`,
  `dose_form`, `release` (IR/ER/DR), `brand`, `brand_or_generic`, NDCs, `rxcui`, `market`, `image_licence`, parsed
  from its RxNorm name (4,910 of 4,932 names; packs list each kind of pill).
- **B10 safety coverage:** `/v1/safety/medications` returns `coverage` (medications screened, ingredients not in
  DDInter, pairs checked vs not checkable) and the note "No alert doesn't mean these medicines are safe together";
  `/v1/safety/label` returns `set_id`, `label_id`, `source` and a DailyMed `url` for citation.
- **B7 documents:** `/v1/read-label` reads boxes, pharmacy labels and prescriptions (photos, text PDFs, scanned PDFs;
  tall pages in overlapping bands) and returns per medicine `fields`: directions, quantity, dose form, date, expiry,
  each with its source line, `null` when not printed. Lines with names, NRIC, addresses or phones are never used.
  Checked on printed examples (label, box, two-medicine prescription PDF, scanned PDF); real photos still to test (B8).
- **B9 package check:** with `medications`, each medicine read gets `list_check`: `same_medicine`,
  `different_strength` ("This says 10 mg; your list has Lipitor 20 mg."), `contains_listed` or `not_on_list`.
- **B14 agent tools:** `GET /v1/tools` builds 11 typed tools from the OpenAPI document (blueprint names where it has
  them); docs/agent_tools.md has usage rules and traces for demos A, B and C.
- **B12 look-alike strengths** (`scripts/evaluate_lookalike_strengths.py`, 714 sibling pairs, both in the gallery):
  named correctly 80.1% → 81.9% with the imprint check, named as the sibling 0.07% → 0.10%, ambiguous 7.5% → 5.6%.
  Pairs that also look alike (131): correct 53% → 63%, ambiguous 35% → 26%, sibling 0.35% → 0.50%: the model
  mostly asks for another photo rather than guessing, and the imprint settles about a quarter of those.
- **B16:** docs/licences.md: terms of every data source and model, PDPA (incl. transfer outside Singapore: synthetic
  data only for the demo), HSA framing for the submission; items to confirm are marked.
- 73 tests pass.

## Before-submission results (12–13 Oct, API v0.8.0 and v0.9.0)

- **B21 no silent fallbacks (v0.8.0):** answers that may read text carry `text_reading` (`used`, `not_requested`,
  `unavailable`, `skipped`) and a note when nothing imprint-based was applied.
- **B22 output validity (v0.8.0):** embeddings are checked (finite, unit length) before use; otherwise `500
  model_error`. Contract tests: same image → same vector, dimension, versions.
- **B23 input guard (v0.8.0):** `input_type` (the user's word comes first) and an automatic check of the photo's
  description; blister packs, bottles, boxes and printed labels abstain. Small labelled set: no loose-pill photo of 14
  refused; 7 of 8 blisters, 6 of 7 bottles or boxes, 5 of 5 labels caught. A net, not a guarantee.
- **v0.8.0 speed:** Florence-2-large reads on the GPU, the base model on the CPU; every call under 3.5 s on the CPU
  path (`outputs/latency/`).
- **B24 extra pill (v0.9.0):** re-measured on 1,000 one-pill photos: count right 87.8%; an extra pill while the count
  is called reliable 1.4%. `double_dose` alerts carry `details.confirm_count` and `notify_caregiver: false`;
  docs/dose_rules.md item 8 says the backend asks the patient before recording an extra pill.
- **B25 model info (v0.9.0):** `/v1/model-info` gives `similarity` (how it is computed, why it isn't a probability)
  and `error_rates`: 8 measured sets, each with its source file and keys; a test checks every rate against its file.
- **B26 privacy (v0.9.0):** ids and names that look like an NRIC or FIN are refused (`personal_id`) and never echoed
  back, in `/v1/match`, `/v1/scan` and `/v1/safety/*`; names are optional; `reference_set_id` is passed through to the
  result.
- **Imprint re-tune for the GPU reader (v0.9.0):** with Florence-2-large reads (own imprint read on 47% of photos, was
  38%) the settings chosen on validation stayed the same; identify confidence refitted. Galleries of 10: named
  correctly 76.0% → 78.3%; ePillID right product first 42% → 73%; identify first candidate right 71.4% → 74.8%,
  then 80.2% (top 5 94.4%) once the re-ranking was chosen on the catalogue itself (alpha 0.05, top 100;
  `scripts/tune_identify_rerank.py`); `possible_match` right 87%, non-catalogue pills called `possible_match` 7%.
- **Reader fine-tune (13 Oct):** a LoRA adapter read more imprints (validation 44% → 54%) but wrote imprints it had
  memorised from the training medicines, so identify and the gallery check got worse: v1 rejected. v2 (also learnt
  to write nothing) helped identify a little (80.2% → 81.2%) but named more of the patient's pills wrongly on
  validation (30 → 61): not adopted. The served reader stays.
- **B27 same-maker imprints (v0.9.0):** 3,149 pairs of different medicines from one maker sharing an imprint part
  (`scripts/evaluate_same_maker_imprints.py`): named as the other 0.01%; Teva "93" pairs 0.00%; worst pair tramadol
  50 mg (TV 58) vs carvedilol 12.5 mg (TV 7295), 5.8% of its photos. Reading only a shared part used to *confirm* a
  wrong look-match (36% of the wrong names); now only parts that tell the patient's medicines apart confirm or
  contradict a match (validation: named wrongly 35 → 30, other pills accepted 6,158 → 6,150, correct −0.03 points).
- Tests: 81 pass.

## 6. Suggested order

Follow the blueprint's first slice (manual enrolment → one loose pill → embed + match → gate → intake → event):

1. **B2, B3, B5** together: the `/v1/embed` + `/v1/match` contract the app builds on.
2. **B1** (then **B6**): stop wrong pill counts.
3. **B11**: hand the dose rules to the backend.
4. **B4, B12**: honest confidence and the look-alike numbers.
5. **B7, B9, B8**: documents, package check, real photos (needs your photos).
6. **B10, B13, B14, B16**, then P2.

## 7. What I need from you

- Decisions **D1–D6** (section 4).
- The **TEVA single-pill photo** that gave two boxes (B1).
- Real **box and label photos**, names covered (B8).
- The **submission date**, so the order above fits.
