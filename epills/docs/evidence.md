# PharmaScan API: evidence for judging

What the PharmaScan API (the Medication Care Agent's vision and safety tools) was tested against, how, and with what
result. It answers the team's test plan (docs/api_review_and_test_plan.md) and the blueprint
(`PharmaScan_Care_Agent_Workflow.html`) case by case. Every number comes from a file in this repository; the API
itself reports the main ones at `GET /v1/model-info` → `error_rates`, each with its source file.

- **Live API:** https://davidpansx-pharmascan-api.hf.space (`/docs` to try every call, `/ui` demo page).
- **Version:** API 0.10.0, model `stage2_run4@2026-10-09` (run 4).
- **Scope:** the API is stateless and gives *visual evidence* ("looks like"). The main app's backend owns patient
  data, regimens and the dose decision (docs/dose_rules.md). No real patient data was used; the demo patient is
  synthetic.

## Check it yourself

From the `epills` folder (Windows, Python 3.13 venv):

| What | Command | Result |
|---|---|---|
| Unit and contract tests | `.venv\Scripts\python -m pytest tests` | 92 pass (`outputs/evidence/pytest.txt`) |
| Singapore dispensing stickers (P01) | `.venv\Scripts\python scripts\check_dispensing_labels.py` | `outputs/evidence/dispensing_labels/` (rendered stickers and what was read) |
| The blueprint's first slice, end to end | `.venv\Scripts\python scripts\demo_first_slice.py [--base <URL>]` | `outputs/evidence/first_slice_trace.json` |
| Pill counting (D01, D02) | `scripts\evaluate_pill_finder.py` | `outputs/eval/pill_finder.json` |
| Gallery matching (M01, M03) | `scripts\evaluate_imprint_gallery.py --run outputs\runs\stage2_run4 --ocr large_c92` | `outputs/runs/stage2_run4/eval/imprint_gallery_large_c92_distinct.json` |
| Look-alike strengths (M02) | `scripts\evaluate_lookalike_strengths.py --run outputs\runs\stage2_run4 --ocr large_c92` | `…/eval/lookalike_strengths_large_c92_distinct.json` |
| Same-maker imprints | `scripts\evaluate_same_maker_imprints.py --run outputs\runs\stage2_run4 --ocr large_c92` | `…/eval/same_maker_imprints_large_c92_distinct.json` |
| Identify top-k (I01) | `scripts\evaluate_identify_topk.py --run outputs\runs\stage2_run4 --ocr large_c92` | `…/eval/identify_topk_large_c92.json` |
| Identify confidence, non-catalogue pills (I02) | `scripts\calibrate_identify.py --run outputs\runs\stage2_run4 --ocr large_c92` | `…/eval/identify_confidence_large_c92.json` |
| Speed of the hosted API | `scripts\benchmark_latency.py` | `outputs/latency/*.json` |

All model measurements use photos of medicines the model never trained on (C3PI validation and held-out
medicines; MEDISEG for other phones). Settings were chosen on validation; the test sets were scored once.

## The test plan, case by case

| Case | Expected | Status | Evidence |
|---|---|---|---|
| D01 | 1 pill, no boxes: at most one detection | ✅ measured; residual handled | 1,000 one-pill photos: count right 87.8%; an extra pill while the count is called reliable 1.4%. Such `double_dose` alerts carry `confirm_count` and hold the caregiver notice until the patient confirms (dose_rules.md item 8). `tests/test_pill_finder.py`, `test_scan_never_counts_pills_when_the_finder_is_unsure`. The team's TEVA photo itself was never shared |
| D02 | Two pills: two detections | ✅ measured | Count right 86.0% (2 pills), 85.1% (3 pills); flagged boxes are never counted (`count_reliable`) |
| D03 | Crop → finite vector with dimension and version | ✅ tested | `test_embeddings_are_repeatable_and_bad_numbers_are_refused`: same image → same vector (cosine ≥ 0.9999 GPU vs CPU), 256 numbers, `embedding_space_id`; non-finite output → `500 model_error`, never passed on |
| M01 | Same product, new photo: match or calibrated ambiguous, error rates measured | ✅ measured | Galleries of 10 medicines, held-out medicines: named correctly 78.3%, named wrongly 0.06%, asked to photograph again 7.2%, own pill called unknown 14.5% |
| M02 | Same-looking other strength: no automatic strength | ✅ measured + flagged | 714 same-ingredient pairs both in one gallery: named as the other strength 0.11% (0.58% when shape and colour are the same). Every catalogue product lists `look_alike_strengths`, and identify notes that the strength needs the label; `/v1/read-label` compares the printed strength with the list (`different_strength`) |
| M03 | Pill not in the gallery: unknown or ambiguous | ✅ measured | Pills that aren't the patient's: called unknown 94.8%, accepted as one of theirs 3.5% |
| I01 | C3PI ground truth with imprint: top-k measured | ✅ measured | Held-out medicines (2,285 photos, 214 products): right product first 80.2%, top 5 94.4%, top 20 97.8% (look alone: 71.6%, 91.7%). With the pill's own imprint read: first 92.9%; without: 68.6% |
| I02 | Unknown US or non-US pill: candidates never claimed as verified | ✅ measured | 800 MEDISEG pills (Hong Kong, not in the catalogue): `no_confident_match` 70%, `possible_match` 7%. Every answer is "possible matches" with a note; `confidence` is calibrated, the raw similarity is labelled "not a probability" |
| P01 | Printed box or label: text with field provenance and user review | ✅ on MOH's example label and rendered stickers | `/v1/read-label` follows MOH's "Know about medication label": what (name, strength, release), how much / how often / when (instructions in parts), storage and side effects (caution kinds), why (indication), quantity, dates, dispensing pharmacy; each with its source line, `null` when not printed; patient name and NRIC never returned. MOH's own example label and four rendered stickers read correctly (`scripts/check_dispensing_labels.py`, `tests/test_label.py`). **Real phone photos of real stickers not yet tested** (job B8) |
| P02 | Clear unprinted blister: no pill match | ✅ abstains | Input check: 7 of 8 blister photos refused (small set; a net, not a guarantee); the user's stated `input_type` comes first |
| P03 | Opaque foil: ask for the source, abstain | ⚠ partly | Declared foil or blister abstains; the automatic check missed the one foil-back photo. The agent asks the user what the photo shows first (`route_medication_image`) |
| R01–R04 | Stopped regimens, duplicate reports, no photo, "unsure" | Backend (app) | Spec and test cases for the backend: docs/dose_rules.md items 2, 4, 5, 6; the API keeps vision and intake apart (`vision_status`, `intake_status` in the first-slice trace) |
| A01 | Vision API timeout: agent keeps the task | ✅ API side | One error shape with `code`; `503` with `Retry-After`; reading that couldn't run says `unavailable` / `skipped` instead of failing silently |
| A02 | Ambiguous image: agent asks, then picks the tool | ✅ API side | `POST /v1/check-input` → type and `next_tools`; `unchecked` → ask the user |
| A03 | Enrolment source without frequency: draft only | ✅ API side | `/v1/read-label` leaves unprinted fields `null` (tested); the app keeps the draft |
| S01 | Interaction check with an unknown ingredient: no "no interaction" claim | ✅ tested | `coverage` lists unscreened ingredients and pairs; every answer says "No alert doesn't mean these medicines are safe together" |
| C01 | Caregiver not authorised: no notification | App | API side: alerts say `notify_caregiver`; `double_dose` holds it until the patient confirms |

## The blueprint's API decisions

| Endpoint | Blueprint asked for | Done |
|---|---|---|
| `GET /health` | Accurate status, cold start, versions, schema | `ready`, `started_at`, device, readers; HTML while waking documented |
| `POST /v1/embed` | Crop contract; 256-d with space id, versions, `normalized` | ✅ |
| `POST /v1/match` | Gallery in the request; threshold, margin, unknown/ambiguous; refuse other spaces | ✅ `409` on another space, `422` without one; pseudonymous ids only (NRIC/FIN-shaped refused) |
| `POST /v1/identify` | US top-k, imprint/shape/colour, scores not probabilities, non-catalogue rejection | ✅ calibrated `confidence` and `verdict` |
| `POST /v1/scan` | Candidates and diagnostics only | ✅ dose check deprecated, only on a reliable count |
| `GET /v1/pills/*` | Product identity: strength, form, release, market; licence; 404 | ✅ |
| `POST /v1/detect` | Boxes, count, quality flags | ✅ stand-in finder until the Stage 1 detector plugs in |
| `GET /v1/model-info` | Service vs model version, space, preprocessing, thresholds, coverage | ✅ plus how similarity works and measured error rates with sources |
| `POST /v1/search-by-embedding` | Identify without re-running the model | ✅ same candidates as `/v1/identify` (tested) |
| `POST /v1/feedback`, `DELETE` | Opt-in, withdrawable, real deletion, audit, review queue | ✅ off for the hackathon; deletion squashes the repo history and is checked; audit log; user labels need review |
| `POST /v1/safety/medications` | Coverage, "no alert ≠ safe", DDInter non-commercial | ✅ |
| `POST /v1/safety/dose` | Core rules move to the backend | ✅ stateless reference + spec (dose_rules.md) |
| `GET /v1/safety/label` | Source, date, citation | ✅ DailyMed link, label date, set id |
| App endpoints (`/api/...`) | Enrolment, dose check, agent session, care summary | The app team's (job B20) |

## The first slice, run end to end

`scripts/demo_first_slice.py` (in-process, API 0.9.0, CPU): a synthetic patient with four C3PI validation medicines
(metformin, lisinopril, amlodipine, simvastatin; 3 enrolment photos each), 30 calls in 26 s.
- **Dose 1**, an unseen photo of their metformin: route → loose pills; one reliable box; "This pill looks like your
  metformin …" (similarity 0.96); intake recorded from the patient's answer, separately.
- **Dose 2**, a warfarin pill that isn't theirs: `unknown` → "Set it aside"; the same embedding gives catalogue
  candidates (warfarin 1 mg first, `possible_match`); dose rules: `unknown_pill` (major) and `not_seen` (never
  "missed").
- **Live:** the same run against the deployed API (0.9.0, 13 Oct) gave the same answers: 30 calls in 23 s
  (`outputs/evidence/first_slice_trace_davidpansx-pharmascan-api.hf.space.json`). Speed of every call:
  `outputs/latency/20261013_2011.json` (all under 3.5 s on the CPU path; see docs/API.md, "Speed").

## Limitations (as measured)

- **Other phones and lighting:** on MEDISEG (Hong Kong products, other phones), 70% of a patient's own pills are
  called unknown. Safe, but unhelpful; enrolment with the patient's own phone and background is the remedy.
- **US catalogue only:** 4,936 US products. Singapore-only products (e.g. gliclazide) can't be identified; the label
  reader and the patient's own gallery cover them.
- **Imprints:** the pill's own imprint is read on about 47% of photos; about a quarter of photos are unreadable even
  for C3PI's raters.
- **Pill finder:** a stand-in for the Stage 1 detector; plain backgrounds only.
- **GPU on the free tier:** Hugging Face lends the GPU within a daily quota per caller (2 minutes without a login), so
  most calls run on the CPU path: slower (identify 3.2 s against 1.8 s), with the smaller imprint reader. Measured
  on that path too (`*_cpu` in `/v1/model-info`): identify first right 79.2% (GPU 80.2%); galleries named correctly
  77.3% (GPU 78.3%), named wrongly 0.06% on both. The same settings were chosen on validation for both readers.
- **Label reading:** checked on printed examples and PDFs, not yet on real phone photos.
- **Input check:** a small labelled set; the user's stated type comes first.
- **Not a medical device:** visual evidence only; no claim of identity, strength or safety to take
  (docs/licences.md for the regulatory framing).

## Development record

- docs/blueprint_review.md: the blueprint review, jobs B1–B27 and their results by stage.
- PROGRESS_LOG.md: every working session, decisions and measurements, newest first.
- tests/: 92 tests, including contract tests for every endpoint.
