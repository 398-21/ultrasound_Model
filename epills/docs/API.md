# PharmaScan API: guide for the web app

Everything the web page can call, with examples. The live, clickable version of this list is the API's own page:
**`<BASE_URL>/docs`** (try any request there).

- **Base URL:** `https://davidpansx-pharmascan-api.hf.space` (locally: `http://localhost:7860`). A demo page for
  trying photos by hand: `<BASE_URL>/ui`.
- **Browser calls are allowed** (CORS is open; it can be limited to our site's address later).
- **API key:** none for now. If one is switched on, send it in a header on every `/v1` request:
  `X-API-Key: <key>`.
- **The server stores nothing** (except opt-in feedback photos, below). The web app keeps, per patient: the
  medication list, each medication's enrolment photos *and* their embeddings, and the dose schedule.
- **Visual evidence only.** Stage 2 answers say what a pill *looks like* (`"evidence": "visual"`, with a `limits`
  note). They never confirm the medicine, its strength, or that it is safe to take; the app keeps "looks like"
  (vision) and "taken" (intake) as separate records.
- **Embedding versions (required).** Every embedding comes with a `model_version` and an `embedding_space_id`. Store
  them with the embeddings and send one back with every gallery (`/v1/match`, `/v1/scan`): without it the API answers
  `422`, with another one `409`. Then re-embed the stored enrolment photos with `/v1/embed` (that's why the app
  keeps the photos). `GET /v1/model-info` describes the served model.
- **Wording.** Show `message` texts as they are, or let the agent reword them: they never say "take" or "stop", and
  alerts end with "Check with your pharmacist."
- **Loose pills only.** `/v1/identify`, `/v1/detect` and `/v1/scan` first check that the photo shows loose pills, not
  a blister pack, bottle, box or printed label (`input_check`, below). Otherwise they compare nothing and say why.
- **Nothing is skipped silently.** Answers that may read text say whether they did (`text_reading`, below); when
  reading was unavailable or skipped, nothing imprint-based was applied and a note says so.
- **First call after a quiet period** can take a few minutes (the free server sleeps after about two days without
  calls). While it wakes, Hugging Face answers with an HTML page, not JSON: call `/health` when the page opens and
  wait for `"ready": true`. A `503` comes with `Retry-After`.
- **Response shapes** for every endpoint are in `<BASE_URL>/openapi.json` (and `/docs`). Tool definitions for the
  agent: `GET /v1/tools` (docs/agent_tools.md).

## The main flows

| User does | Call |
|---|---|
| Adds a medication (name, photos) | `POST /v1/safety/medications` (whole list), then `POST /v1/embed` per enrolment photo (3–5 photos, both sides). Keep the pill's `imprint` (from `/v1/pills/search`) with the gallery |
| Adds a medicine from its pharmacy sticker (or prescription, or box; photo or PDF) | `POST /v1/read-label` (with the current list as `medications`): medicine, strength, quantity, instructions in parts (dose, times a day, food), what it is for, cautions. The user confirms them, then `POST /v1/safety/medications` |
| Checks a box or label at dose time | `POST /v1/read-label` with `medications`: `list_check` says same medicine / different strength / not on the list |
| Checks a dose (one photo of the pills) | `POST /v1/detect` (boxes; check `count_reliable`), `POST /v1/embed` per pill (with its box), `POST /v1/match` with the gallery; the **backend** applies the dose rules (docs/dose_rules.md). `POST /v1/scan` does the visual part in one call |
| Sends a photo and it isn't clear what it shows | `POST /v1/check-input`: loose pills, blister, box, label or document, and the tools that fit |
| Finds an unknown pill | `POST /v1/identify` (photo, optional imprint text); after `/v1/match` said `unknown`, `POST /v1/search-by-embedding` with the same embedding |
| Wants to see what a medicine looks like | `GET /v1/pills/search?name=…` then the pictures |
| Asks for the official warnings | `GET /v1/safety/label?rxcui=…` |
| Shares a checked photo / withdraws consent | `POST /v1/feedback` / `DELETE /v1/feedback/{consent_id}` |

---

## Health

### `GET /health`
```json
{"status": "ok", "ready": true, "api_version": "0.10.0", "started_at": "2026-10-13T06:10:00+00:00",
 "model": {"loaded": true, "version": "stage2_run4@2026-10-09",
           "embedding_space_id": "pill-crop:stage2_run4@2026-10-09:w1.55", "device": "zerogpu", "error": null},
 "catalogue": {"loaded": true, "products": 4936, "model_version": "stage2_run4@2026-10-09", "error": null},
 "safety_db": {"loaded": true, "built_at": "2026-10-08T17:20:09", "pairs": "234981"},
 "text_reading": {"loaded": true, "model": "florence-community/Florence-2-base",
                  "gpu_model": "florence-community/Florence-2-large", "error": null}}
```
`text_reading.model` reads on the CPU; `gpu_model` on the lent GPU (the larger reader, which reads more imprints).
Answers that read text say which one ran.
`model.device` says where photos run now: `zerogpu` (a GPU Hugging Face lends per call) or `cpu` (with the reason,
e.g. "ZeroGPU refused a recent call"). Same model either way: the embeddings differ only by rounding.

**Speed.** The GPU is free within a daily quota that belongs to the *caller* (Hugging Face: 2 minutes a day without
a Hugging Face login, 5 with a free account; it resets 24 hours after first use), so a busy app uses it up early and
the rest of the day runs on the CPU. Measured on the CPU path from our desktop (API 0.9.0, 13 Oct,
`outputs/latency/`): every call under 3.5 s; `/v1/match` 0.3 s, `/v1/embed` 0.3 s (2.9 s when it also reads the
imprint), `/v1/identify` 3.2 s, `/v1/detect` 0.8 s, `/v1/scan` with 3 pills 1.1 s, `/v1/read-label` 1.1 s (photo),
0.3 s (text PDF). On the GPU, identify took 1.8 s. The first call after a quiet period can take a few minutes (the
Space wakes up).

### `GET /v1/model-info`
The served model: `api_version` vs `model_version`, `embedding_space_id`, `preprocess_version`, `embedding_dim`
(256), `normalized`, `crop_window`, the decision `threshold` and `margin` with how they were calibrated, the imprint
settings, `training_coverage` and `known_limits` (e.g. other phones: 48% top-1 on MEDISEG).
`similarity` says how similarities are computed (cosine of normalised embeddings, per endpoint) and why they aren't
probabilities. `error_rates` lists the served model's measured error rates, each with `what` was tested, the `rates`
(fractions: 0.012 = 1.2%), and the evaluation file (`source`, `keys`) they come from; it is empty for a model they
weren't measured on.
```json
{"id": "pill_count", "what": "Stand-in pill finder ... on 1,000 one-pill C3PI photos (test case D01) ...",
 "rates": {"exact_count_1": 0.878, "over_count_while_reliable_1": 0.014, ...},
 "source": "outputs/eval/pill_finder.json", "keys": {"exact_count_1": "by_pills/1/exact_count", ...}}
```
Ids: `personal_gallery` (`/v1/match`, galleries of 10), `personal_gallery_with_imprint` (GPU reader) and
`personal_gallery_with_imprint_cpu` (CPU reader: most live calls), `personal_gallery_other_phones` (MEDISEG),
`same_ingredient_siblings`, `same_maker_imprints`, `catalogue_identify` and `catalogue_identify_cpu`, `pill_count`,
`input_check`.

## Pill verification and identification (Stage 2)

### `POST /v1/embed`: a photo → its embedding
Multipart form.

| Field | Required | |
|---|---|---|
| `image` | yes | the photo (JPEG/PNG; phone photos fine, EXIF rotation is handled) |
| `box` | no | `[x0, y0, x1, y1]` of the pill in the photo, in pixels. Without it the image must already be a crop with the pill in the middle filling ~60–70% |
| `read_imprint` | no | `true`: also read the pill's imprint, for `/v1/match`'s `query_reads` (default `false`) |

```json
{"embedding": [0.0213, -0.0871, ...256 numbers...], "dim": 256,
 "model_version": "stage2_run4@2026-10-09", "embedding_space_id": "pill-crop:stage2_run4@2026-10-09:w1.55",
 "preprocess_version": "crop-contract-1:window-1.55:crop-384:input-224", "normalized": true,
 "framed_by": "box", "imprint_reads": ["93 7267", "", "", ""], "evidence": "visual"}
```
`imprint_reads`: what the text reader saw in four rotations (`null` unless asked). Pass it on as is.
Use at **enrolment**: store each embedding with the medication, the `model_version` and the `embedding_space_id`.
Only loose-pill photos belong in a gallery (not boxes or blister packs). The same photo gives the same vector
(GPU and CPU differ only by rounding: cosine ≥ 0.9999); compare embeddings by dot product, only within one space.

### `POST /v1/match`: which of my medicines is this?
JSON.
```json
{"queries": [[...256 numbers...]],
 "gallery": [{"medication_id": "m1", "name": "Metformin 500 mg", "embeddings": [[...], [...], [...]],
              "imprint": "93;7267"},
             {"medication_id": "m2", "name": "Amlodipine 5 mg", "embeddings": [[...], [...]]}],
 "model_version": "stage2_run4@2026-10-09",
 "query_reads": [["93 7267", "", "", ""]]}
```
`model_version` (or `embedding_space_id`): **required**, the one stored with the gallery's embeddings.
`imprint` (optional, per medication): the pill's imprint as `/v1/pills/search` gives it. `query_reads` (optional): per
query, its `imprint_reads` from `/v1/embed`. With both, the imprint read on the pill checks the answer (below).

**Privacy.** Send only what the match needs: this dose's queries and the patient's gallery, never their history.
- `medication_id`: the app's **pseudonymous** id for the patient's medication (e.g. its `patient_medication_id`).
- `name` is optional; without it answers use `medication_id`.
- `reference_set_id` (optional, per medication): the app's id for the set of reference photos the embeddings come
  from. It comes back on the result (`reference_set_id`, and in `closest` / `runner_up`), so the backend can record
  which set a decision was made against.
- Anything shaped like a Singapore NRIC or FIN (e.g. `S1234567D`) in an id or name is refused: `422`,
  `detail[].type` = `personal_id`. The refused value is never echoed back. The same applies to `/v1/scan`'s
  `gallery` and to the ids sent to `/v1/safety/*`.
Answer, one result per query:
```json
{"results": [{"decision": "match", "medication_id": "m1", "name": "Metformin 500 mg", "similarity": 0.9312,
              "closest": {"medication_id": "m1", "name": "Metformin 500 mg", "similarity": 0.9312},
              "runner_up": {"medication_id": "m2", "name": "Amlodipine 5 mg", "similarity": 0.512},
              "message": "This pill looks like your Metformin 500 mg.",
              "imprint_check": "confirmed", "imprint_read_for": ["Metformin 500 mg"]}],
 "model_version": "stage2_run4@2026-10-09", "embedding_space_id": "pill-crop:stage2_run4@2026-10-09:w1.55",
 "threshold": 0.8949, "margin": 0.115, "evidence": "visual",
 "limits": "A visual comparison with reference photos only: it can't confirm what the medicine is, ..."}
```

| `decision` | Show |
|---|---|
| `match` | "This pill looks like your Metformin 500 mg." |
| `unknown` | "This pill doesn't look like any of your medicines. Set it aside. Check with your pharmacist." (`medication_id` is null; `closest` is only for display) |
| `ambiguous` | "…Turn it over and photograph it again." (ask for a new photo) |

`422` = no `model_version` / `embedding_space_id` sent; `409` = the gallery was embedded in another space: re-embed
the enrolment photos.

**The imprint check** (`imprint_check`; `null` when no reads were sent). The look decides; the imprint can confirm it,
settle a tie, or raise a doubt. It never accepts a pill the look rejected.

| `imprint_check` | Meaning | Effect on `decision` |
|---|---|---|
| `confirmed` | a part of the matched medication's imprint that none of the patient's other medications carry was read | none (can show "imprint matches") |
| `resolved` | a tie between two of the patient's medications; the imprint of exactly one was read | `ambiguous` → `match` |
| `conflict` | another of the patient's medications' imprint was read, and not this one's own part | `match` → `ambiguous` (photograph again) |
| `not_read` | none of the gallery's imprints could be read (most photos: engraved or small imprints) | none |
| `not_used` | the pill is `unknown`: an imprint never makes it a match | none |
| `unresolved` | the read doesn't tell them apart: both or neither of a tie, or only a part several of the patient's medications share (e.g. Teva's `TV` on `TV 58` and `TV 7295`) | none |
| `not_checked` | no gallery medication has an `imprint` | none |

`imprint_read_for`: names of the patient's medications whose imprint was read on this pill (shared parts included).
Measured on new medicines (galleries of 10, the GPU reader): named correctly 76.0% → 78.3%, asked to photograph again
9.5% → 7.2%, other pills accepted 3.8% → 3.5%, named wrongly 0.04% → 0.06%. The pill's own imprint is read on about
47% of photos, so it helps but can't make recognition near-perfect. Two medicines from one maker with a shared imprint
part are rarely confused (0.01% in 3,149 such pairs; `/v1/model-info` → `same_maker_imprints`).

### Input check and text reading (in `/v1/identify`, `/v1/detect`, `/v1/scan`)
Two optional form fields on each:

| Field | |
|---|---|
| `input_type` | what the user says the photo shows: `loose_pills`, `blister`, `package` or `document`. Anything but `loose_pills` abstains at once. Ask the user first (the blueprint's order); the automatic check is a safety net |
| `check_input` | default `true`: check the photo's description for blister packs, bottles, boxes and printed text. `false` skips it |

```json
"input_check": {"type": "blister", "caption": "a blister pack of white pills", "checked": true, "abstained": true,
                "message": "This looks like a blister pack. Pill recognition works on loose pills only: ..."}
```
`type`: `loose_pills`, `blister`, `package`, `printed`, `document` or `unchecked` (the check couldn't run). When
`abstained` is true the answer has no pills or candidates: show `message` (photograph the label, or ask the
pharmacist). Measured on a small labelled set: no loose-pill photo of 14 was refused; 7 of 8 blisters, 6 of 7 bottles
or boxes and 5 of 5 printed labels were caught. A safety net, not a guarantee.

`text_reading` on answers that may read the imprint (`/v1/embed` with `read_imprint`, `/v1/identify`, `/v1/scan`):

| `text_reading` | Meaning |
|---|---|
| `used` | the imprint was read and used |
| `not_requested` | reading wasn't asked for (or a typed `imprint` replaced it) |
| `unavailable` | the text reader isn't working: the answer is by appearance only, and a note says so |
| `skipped` | the GPU was busy and reading on the CPU would take too long: by appearance only, with a note |

### `POST /v1/detect`: where are the pills?
Multipart form: `image` (a photo of the pills, apart, on a plain surface), optional `input_type`, `check_input`.
```json
{"boxes": [[1820.2, 1356.8, 2272.4, 1786.7]], "box_flags": [[]], "pill_count": 1, "count_reliable": true,
 "flags": [], "found_by": "stand-in finder (plain backgrounds only)", "message": null}
```
- `box_flags` per box: `touches_edge` (cut by the photo's edge), `merged` (pieces joined), `irregular` (odd shape),
  `small` (much smaller than the others). `flags` per photo: `no_pills`, `count_uncertain`, `too_many`.
- **Count pills only when `count_reliable` is true**; otherwise ask for a new photo.
- A stand-in until the Stage 1 detector replaces it (same box format). On C3PI photos with 1–3 pills it gets the
  count right 86–88% of the time (the earlier finder: 28% on single pills); textured surfaces and white pills on white
  are the hard cases. Apps with their own detector send `boxes` to `/v1/scan` / `box` to `/v1/embed` instead.

### `POST /v1/scan`: every pill in a photo, in one call (diagnostics)
The visual part of a dose check in one call: detection, then each pill's decision (with a gallery) or catalogue
candidates (without one). **Not a dose verification:** the backend decides with the regimen (docs/dose_rules.md).
Multipart form.

| Field | Required | |
|---|---|---|
| `image` | yes | one photo of the dose (pills apart, on a plain mat or table) |
| `gallery` | no | the gallery as a **JSON string** (same shape as in `/v1/match`). Without it, each pill is identified against the catalogue instead |
| `schedule` | no | **Deprecated.** JSON string `[{"medication_id": "m1", "count": 1}, ...]`: adds an advisory dose check (`dose.deprecated: true`), run only when the count is reliable and only on counted pills |
| `boxes` | no | JSON string `[[x0, y0, x1, y1], ...]` from the pill detector (Stage 1). Without it, the stand-in finder of `/v1/detect` is used |
| `model_version` / `embedding_space_id` | with a gallery | as in `/v1/match` (required) |
| `read_imprint` | no | default `true`: reads each pill's imprint. With a gallery it runs the imprint check (only if a medication has an `imprint`); without one it re-ranks the catalogue candidates |
| `input_type` / `check_input` | no | the input check (above) |

```json
{"found_by": "stand-in finder (plain backgrounds only)", "model_version": "stage2_run4@2026-10-09",
 "detection": {"boxes": [...], "box_flags": [[]], "pill_count": 1, "count_reliable": true, "flags": [], ...},
 "pills": [{"box": [162.5, 143.8, 381.2, 387.5], "quality_flags": [], "counted": true,
            "decision": "match", "medication_id": "m1", "name": "Metformin 500 mg", "similarity": 0.93,
            "closest": {...}, "runner_up": {...}, "message": "This pill looks like your Metformin 500 mg.",
            "imprint_check": "confirmed", "imprint_read_for": ["Metformin 500 mg"],
            "imprint_reads": ["93 7267", "", "", ""]}],
 "dose": {"alerts": [ALERT, ...], "deprecated": true, "note": "Advisory and deprecated: ...",
          "summary": {"pills_seen": 3, "matched": 2, "unknown": 1, "unsure": 0, "all_visually_matched": false,
                      "all_ok": false, "count_reliable": true, "text": "I see three pills."}},
 "evidence": "visual", "limits": "..."}
```
No pill found: `"pills": []` and a `message` asking for a new photo. Draw the `box`es on the photo; a box with
`quality_flags` is shown but not counted (`counted: false`). When the count isn't reliable, the advisory dose check
returns only a `count_uncertain` alert.

### `POST /v1/identify`: what is this pill?
Multipart form: `images` (1 or 2 files: front and back of the **same** pill), optional `boxes` (JSON, one per
image), `imprint` (letters/numbers on the pill, e.g. `M 547`: narrows the list a lot), `shape`
(e.g. `ROUND`), `colour` (e.g. `WHITE`), `top_k` (default 5), `read_imprint` (default `true`: reads the imprint from
the photos and moves the products whose recorded imprint matches up the list; skipped when `imprint` is typed),
`input_type` and `check_input` (the input check, above).
On photos of medicines the model never trained on (one photo each), the right product comes first for 80% of pills
and is in the first five for 94% (by look alone: 72% and 92%; with the CPU reader that serves most calls: 79%
and 94%). When the pill's own imprint is read (about half the
photos) it comes first for 93%; when it isn't, 69%. Front and back photos together, or the typed imprint, do better.
```json
{"candidates": [{"pill_id": "001725729_1", "name": "FAMOTIDINE Tablets USP",
                 "rxnorm_name": "famotidine 20 MG Oral Tablet", "generic_name": "FAMOTIDINE",
                 "manufacturer": "TEVA", "ingredients": ["famotidine"], "shape": "ROUND", "colour": "BEIGE",
                 "size_mm": 6.0, "imprint": "TEVA;5729",
                 "images": [{"url": "/v1/pills/001725729_1/image/0", "kind": "rximage"}],
                 "similarity": 0.959, "imprint_match": true, "imprint_read": 0.71}],
 "notes": [], "note": "Possible matches from a catalogue of US pills, ... Check with your pharmacist.",
 "confidence": 0.93, "catalogue_fit": "good", "verdict": "possible_match",
 "verdict_message": "The first candidate looks most like this pill. A photo can't confirm it. ...",
 "imprint_reads": ["TEVA 5729", "", "", ""], "model_version": "stage2_run4@2026-10-09",
 "evidence": "visual", "limits": "..."}
```
`imprint_match`: the typed imprint fits this product. `imprint_read`: how strongly the read imprint fits it (0–1).
`similarity` is **not a probability** (pills that aren't in the catalogue still score about 0.93). Use these instead:

| Field | Meaning |
|---|---|
| `confidence` | chance the first candidate is right, *if the pill is a US catalogue product*. Checked on held-out medicines: stated 0.96 → 93% right, 0.70 → 75%, 0.51 → 49%, 0.36 → 39% |
| `catalogue_fit` | `good` / `weak` / `poor`: how well the best match fits at all |
| `verdict` | `possible_match` (76% of catalogue pills; 87% of those right), `uncertain`, `no_confident_match` (given to 70% of pills that aren't in the catalogue; 7% of them still get `possible_match`). Show `verdict_message` |

All four are `null` when a typed `imprint`, `shape` or `colour` filters the search (not calibrated for that).
**Always show these as possible matches with the `note`, never as "this is X".** US products only.

**Strength (test case M02).** Every product carries `look_alike_strengths`: catalogue products with the same
ingredients in another strength or release that have the same recorded shape and colour (about half the catalogue
has some, mostly white round generics). When the first candidate has any, `notes` says a photo can't tell the
strength: get it from the box or label.

### `POST /v1/search-by-embedding`: what is this pill, from an embedding you already have
JSON. The same answer as `/v1/identify` for the same photos, without running the model again, e.g. after
`/v1/match` said `unknown`:
```json
{"embeddings": [[...256 numbers...]], "model_version": "stage2_run4@2026-10-09",
 "imprint_reads": ["93 7267", "", "", ""], "top_k": 5}
```
`embeddings`: 1 or 2 (front and back of one pill) from `/v1/embed`; `model_version` (or `embedding_space_id`) is
required (`409` for another one). `imprint_reads`: what `/v1/embed` returned with `read_imprint=true`, both photos
together. Optional `imprint` (typed), `shape`, `colour`. The photo isn't checked again: `/v1/embed` takes loose-pill
photos only.

### `POST /v1/check-input`: what does this photo show?
Multipart form: `image`, optional `input_type` (what the user says it shows, taken as given). For the agent's
`route_medication_image` tool, before choosing another photo tool. PDFs are documents: send them to `/v1/read-label`.
```json
{"input_check": {"type": "blister", "caption": "a blister pack of white pills", "checked": true, "abstained": true,
                 "message": "This looks like a blister pack. ..."},
 "next_tools": [], "message": "A blister pack: pills in it can't be recognised. Photograph the box or pharmacy
 label, or ask the pharmacist."}
```
`next_tools`: `loose_pills` → `detect_pills`, `embed_pill`, `match_gallery`, `identify_pill`; box, bottle, printed
label or document → `extract_medication_document`; blister or `unchecked` → none: ask the user.

### `GET /v1/pills/search?name=metformin`: what does it look like?
Query: `name` (brand or generic, typos tolerated), optional `strength` (e.g. `500 MG`), `limit` (default 50).
```json
{"resolution": {"query": "Glucophage", "ingredients": [{"rxcui": "6809", "name": "metformin"}], "rxnorm": "ok"},
 "count": 71,
 "products": [{"pill_id": "000937267_1", "name": "...", "rxnorm_name": "24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet",
               "manufacturer": "TEVA", "ingredients": ["metformin"], "shape": "OVAL", "colour": "WHITE",
               "size_mm": 19.0, "imprint": "93;7267", "images": [{"url": "/v1/pills/000937267_1/image/0", "kind": "rximage"}],
               "match": "same"}],
 "source": "NLM C3PI (research and development use only); names: RxNorm (NLM)"}
```
`match`: `same` (exactly this medicine), `contains` (a combination with it), `name` (name match only). One
medicine has many different-looking pills (makers, strengths): show them all, with a strength filter.

Every product also carries its identity as fields, so products of another strength, dose form or release are never
merged: `strength` (e.g. `"500 MG"`, `"40 MG / 1100 MG"`), `strengths` (per ingredient), `dose_form` (e.g.
`"Extended Release Oral Tablet"`, `"Pack"` with `pack` listing each kind of pill), `release` (`IR` / `ER` / `DR`),
`brand`, `brand_or_generic`, `ndc9`, `ndc11`, `rxcui`, `rxnorm_status`, `market` (`"US"`), and `image_licence`
(C3PI pictures are for research and development use; a picture may not show the product the patient holds).
The same fields appear on `/v1/identify` and `/v1/scan` candidates.

**Look-alike strengths** (`scripts/evaluate_lookalike_strengths.py`, 714 pairs of one medicine's strengths or release
forms, both in the patient's gallery): named correctly 80% → 82% with the imprint check, named as the sibling
0.07% → 0.10%, the rest ambiguous or unknown. When the two strengths also look alike (same shape and colour, 131
pairs): named correctly 53% → 63%, ambiguous 35% → 26%, named as the sibling 0.35% → 0.50%. Mostly the model asks
for another photo rather than guessing; enrol photos of both sides.

### `GET /v1/pills/{pill_id}` and `GET /v1/pills/{pill_id}/image/{n}`
One product (same fields as above), and its pictures as JPEG: `rximage` (both sides with a mm ruler),
`reference` (studio), `photo` (everyday). Use the `url` values directly in `<img src="BASE_URL + url">`.

### `POST /v1/read-label`: what does the pharmacy sticker say?
Made for **Singapore dispensing labels**: the sticker the pharmacy or polyclinic puts on the medicine bag or box.
It also reads prescriptions (photo or PDF) and manufacturer boxes. Multipart form: `image` (a photo of the sticker,
close up and straight on, **or a PDF**), optional `medications` (JSON string: the patient's list,
`[{"id", "name", "rxcui"?, "strength"?}]`).

We follow MOH's guide **"Know about medication label"** (`docs/know-about-medication-label_eng.pdf`) as the
standard. Its example label

```
KEEP AWAY FROM CHILDREN
NIFEDIPINE LA 60MG TAB                    (30 TAB)
TAKE 1 TABLET ONCE EVERY MORNING
May be taken with or without food.
Swallow the tablet whole. Do not crush or chew.
                     for blood pressure          (handwritten by the pharmacist)
TAN AH SENG
SXXXX123A                              10/08/2021
KNOW YOUR MEDICINE GENERAL HOSPITAL PHARMACY
MEDICINE ROAD S(123456) TEL:12345678
```

comes back as (shortened; each `line` is where the value was read)

```json
{"label_kind": "dispensing",
 "dispensed_by": {"value": "KNOW YOUR MEDICINE GENERAL HOSPITAL PHARMACY", "line": "..."},
 "candidates": [{"name": "Nifedipine LA 60 mg", "strength": "60 mg", "release": "ER", "release_printed": "LA",
                 "ingredients": [{"rxcui": "7417", "name": "nifedipine"}],
                 "rxcui": "198035", "rxnorm_name": "24 HR nifedipine 60 MG Extended Release Oral Tablet",
                 "found_in": ["NIFEDIPINE LA 60MG TAB"],
                 "fields": {
                   "quantity": {"value": "30 tab", "amount": 30, "unit": "tablet", "line": "(30 TAB)"},
                   "instructions": {"text": "TAKE 1 TABLET ONCE EVERY MORNING May be taken with or without food.",
                                    "line": "...", "dose": {"amount": 1, "amount_max": null, "unit": "tablet"},
                                    "times_per_day": 1, "every_hours": null, "as_needed": false,
                                    "times_of_day": ["morning"], "food": "with_or_without_food",
                                    "duration": null, "max_per_day": null},
                   "indication": {"value": "blood pressure", "line": "for blood pressure"},
                   "cautions": [{"value": "KEEP AWAY FROM CHILDREN", "kind": "storage", "line": "..."},
                                {"value": "Swallow the tablet whole. Do not crush or chew.", "kind": "how_to_take",
                                 "line": "..."}],
                   "dose_form": {"value": "tablet", "line": "..."},
                   "date": {"value": "10/08/2021", "line": "10/08/2021"}, "expiry_date": null,
                   "directions": {"value": "TAKE 1 TABLET ONCE EVERY MORNING May be taken ...", "line": "..."}},
                 "list_check": null}],
 "note": "Medicines read from the photo. Check the name and strength ...", "source": "photo",
 "ocr_model": "florence-community/Florence-2-large"}
```

The patient's name, the NRIC and the pharmacy's address and phone never come back; the date printed beside the NRIC
comes back on its own.

**MOH's sections, field by field** (per medicine, `candidates[]`):

| MOH's guide | Field | Example |
|---|---|---|
| **What** name and strength | `name`, `strength` (per tablet or capsule), `release` (`ER` / `DR` when LA, SR, XR, XL, CR, MR, EC… is printed), ingredients, RxNorm code | NIFEDIPINE LA 60MG → `ER`, the extended-release product |
| **How much** | `instructions.dose`: `amount` (and `amount_max` for a range), `unit` | "Take 1-2 tablets" → 1, 2, `tablet` |
| **How often** | `instructions.times_per_day` (only when printed: "twice a day", "daily", "once every morning", BD/TDS/QDS/OD, OM/ON), `every_hours` ("every 6 hours", never turned into times a day), `as_needed` ("when necessary", PRN), `duration`, `max_per_day` | 1 |
| **When** | `instructions.times_of_day` (`morning`, `afternoon`, `evening`, `night`, `bedtime`), `instructions.food` (`before_food`, `after_food`, `with_food`, `with_or_without_food`, `empty_stomach`) | `["morning"]`, `with_or_without_food` |
| Store your medication | `cautions` with `kind: storage` | "KEEP AWAY FROM CHILDREN", "Keep in the fridge" |
| Manage common side-effects | `cautions` with `kind: side_effect` | "May cause drowsiness" |
| (how to take it) | `cautions` with `kind: how_to_take`; other notes `kind: warning` | "Swallow the tablet whole", "Avoid alcohol" |
| **Why** you are given it | `indication`, as printed (often handwritten) | "blood pressure" |
| Also on the label | `quantity` (`amount`, `unit`), `date`, `expiry_date`, `dose_form`, `dispensed_by` (response) | "(30 TAB)" → 30 `tablet` |
| | `directions`: the instruction text as one string (kept for older clients) | |

- **Printed, never inferred.** Every value comes with the line(s) it was read from; a part that isn't on the sticker
  is `null` (or `[]` / `false`). "Every 8 hours" is not turned into "3 times a day"; no indication is guessed.
- **Personal details:** lines with the patient's name, NRIC/FIN (also masked, like `SXXXX123A`), address or phone
  are never used or returned; a date on such a line comes back on its own, without the rest of the line.
- `label_kind`: `dispensing` (a sticker or prescription with instructions for the patient), `package` (a
  manufacturer's box: name and strength, no instructions) or `none` (no medicine found).
- Names in caution lines ("Avoid grapefruit juice", "Do not take with aspirin") are not taken as dispensed medicines.
- `list_check` (with `medications`): `same_medicine`, `different_strength` ("This says 10 mg; your list has Lipitor
  20 mg."), `contains_listed` (a combination including a listed ingredient) or `not_on_list`. A comparison of
  ingredients and strength, not a model score.
- `source`: `photo`, `pdf_text` (the PDF's own text) or `pdf_scan` (pages rendered and read, up to 3).
- Finds RxNorm's ingredient and brand names, plus international names RxNorm lacks (paracetamol, salbutamol,
  co-amoxiclav…), with the strength printed next to them. `rxcui` is the exact product when name, strength and dose
  form agree.
- **Always let the patient confirm** the medicine and the instructions against the sticker, then add it with
  `POST /v1/safety/medications` as `{"id", "name", "rxcui"}`. The backend may build the reminder schedule from
  `instructions`, but only after that confirmation (the blueprint's enrolment flow).
- No medicine found: `"candidates": []` and a `message` asking for a closer photo. `502` = RxNorm didn't answer.
- Tested on rendered Singapore-style stickers (clean and photo-like: tilted, blurred, on a table) and printed
  examples; **not yet on real phone photos of real stickers** (job B8). English lines only. Takes a few seconds.

## Safety checks (Stage 3)

### `POST /v1/safety/medications`: when the medication list changes
```json
{"medications": [{"id": "m1", "name": "Lipitor 10 mg"},
                 {"id": "m2", "name": "clarithromycin"},
                 {"id": "m3", "name": "Panadol (AU)", "ingredients": ["paracetamol"]}]}
```
Per medication, any of: `name`, `ndc` (US bottle code), `rxcui`, `ingredients` (for medicines without US codes).
Answer: `alerts` (interactions, duplications, unidentified medications), `medications` (how each was identified),
`all_identified`, `sources`, and:
```json
"coverage": {"medications": 4, "screened": 3, "not_screened": ["Herbal tea"], "ingredients_not_in_interaction_database": ["..."],
             "pairs": 6, "pairs_checked": 3, "pairs_not_checkable": 3},
"note": "No alert doesn't mean these medicines are safe together: only interactions listed in DDInter 2.0 ..."
```
Show the `note` with every result, and say which medicines couldn't be screened.

### `POST /v1/safety/dose`: the dose rules (reference implementation)
The main backend owns the dose decision (it knows the regimen, history, PRN and stopped medicines); this endpoint is
the stateless reference for it. Rules, test cases and what the backend adds: **docs/dose_rules.md**.
```json
{"schedule": [{"medication_id": "m1", "count": 1}],
 "pills": [{"medication_id": "m1", "decision": "match"}, {"decision": "unknown"}],
 "names": {"m1": "Metformin 500 mg"}}
```

### `GET /v1/safety/label?rxcui=…` or `?ingredient=…`
The warning sections of the official US label (openFDA), quoted as written; only when the user asks. For
citation: `label_date` (effective date), `set_id`, `label_id`, `source`, and `url` (the label on DailyMed).

### ALERT (the same shape everywhere)
```json
{"type": "interaction", "severity": "major", "colour": "red", "speak": true, "notify_caregiver": true,
 "message": "Lipitor 10 mg and clarithromycin have a serious interaction. Check with your pharmacist.",
 "medication_ids": ["m1", "m2"], "source": "DDInter 2.0 ...", "severity_graded": true, "details": {...}}
```

| severity | colour | read aloud | tell the caregiver |
|---|---|---|---|
| `major` | red | yes | yes |
| `moderate` | amber | yes | no |
| `ungraded` | grey | no | no |
| `minor` | grey | no | no |

`type`: `interaction`, `duplication`, `unresolved`, `unknown_pill`, `not_scheduled`, `double_dose`, `not_seen`
(a scheduled medicine not seen in the photo: never "missed"), `unsure_pill`, `count_uncertain`.

`double_dose` is the exception to the table: one pill boxed twice can cause it (1.4% of one-pill photos with the
stand-in finder), so it carries `details.confirm_count: true` and `notify_caregiver: false`. Ask the patient to count
before recording an extra pill; tell the caregiver only once the patient confirms (docs/dose_rules.md, item 8).

## Opt-in photos for the monthly improvement

Only when the user has **agreed** (a separate consent from using the app), and only with a checked label.

### `POST /v1/feedback`
Multipart form: `image`, optional `box`, `consent` (`true`), `consent_id` (the app's id of this user's consent;
letters, digits, `-`, `_`), `medication_name`, `label_source` (`pharmacist` or `user_confirmed`), optional `ndc`,
`rxcui`, `app_decision` (what the app had said: `match` / `unknown` / `ambiguous`), `app_medication_id`.
```json
{"stored": true, "item": "4f1c...", "note": "Thank you. The photo is used only to improve pill recognition."}
```
Only the pill crop is kept (no location data). `503` = collection is switched off on the server. Labels the user
confirmed wait for a person's review before retraining may use them; pharmacist-checked labels don't.

### `DELETE /v1/feedback/{consent_id}`
When the user withdraws consent: deletes all their shared photos, here and in the dataset repo, then rewrites the
repo's history so older versions don't keep them, and checks that nothing is left. Safe to repeat.
```json
{"deleted": true, "verified": true, "local_photos_removed": 3, "removed_from_repo": true, "history_squashed": true}
```
`deleted` is true only once that check passed. An audit log records every save and withdrawal with a hash of the
consent id (never photos or names).

## Errors

Every error has the same shape: `{"code": "...", "detail": "...", "retry_after": 20}` (`retry_after` only with `503`,
which also sends a `Retry-After` header).

| HTTP | `code` | Meaning | What the page does |
|---|---|---|---|
| 401 | `unauthorised` | missing or wrong `X-API-Key` | check the key |
| 404 | `not_found` | no such pill or picture | |
| 409 | `version_mismatch` | gallery embedded in another space | re-embed the enrolment photos |
| 422 | `invalid_input` | a field is missing or malformed (`detail` says which); `type` `personal_id`: an id or name looks like an NRIC or FIN | fix the request; send pseudonymous ids |
| 500 | `model_error` | the model returned invalid numbers for this photo (never passed on) | try another photo |
| 502 | `upstream_failed` | RxNorm or openFDA didn't answer | try later |
| 503 | `not_ready` | the model, catalogue or database is still loading (or feedback is off) | wait `retry_after` seconds; `/health` shows what's loading |
| (HTML page) | | the Space is waking up | wait, then call `/health` again |

## Example in the browser (JavaScript)

```js
const BASE = "https://davidpansx-pharmascan-api.hf.space";

// enrolment: one photo -> embedding
async function embed(file) {
  const form = new FormData();
  form.append("image", file);
  const r = await fetch(`${BASE}/v1/embed`, { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json()).detail);
  return r.json();                    // { embedding, model_version, ... }
}

// dose check: one photo, the patient's gallery and today's schedule
async function scan(file, gallery, modelVersion) {
  const form = new FormData();
  form.append("image", file);
  form.append("gallery", JSON.stringify(gallery));      // [{medication_id, name, embeddings, imprint}]
  form.append("model_version", modelVersion);           // required with a gallery
  const r = await fetch(`${BASE}/v1/scan`, { method: "POST", body: form });
  if (r.status === 409) { /* re-embed the enrolment photos, then retry */ }
  const body = await r.json();        // { detection, pills: [...] }
  if (!body.detection.count_reliable) { /* ask for a new photo: pills apart, plain surface */ }
  return body;                        // the backend applies the dose rules to body.pills
}

// pictures of a medicine
const res = await (await fetch(`${BASE}/v1/pills/search?name=${encodeURIComponent("metformin")}`)).json();
const firstPicture = BASE + res.products[0].images[0].url;   // use as <img src>
```

With `curl`:
```
curl -F image=@pill.jpg https://davidpansx-pharmascan-api.hf.space/v1/embed
curl "https://davidpansx-pharmascan-api.hf.space/v1/pills/search?name=metformin"
```
