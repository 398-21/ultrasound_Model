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
- **Model versions.** Every Stage 2 answer has a `model_version`. Store it with the embeddings. When the API
  answers with a new version (or a `409`), re-embed the stored enrolment photos with `/v1/embed`. That's why the
  app keeps the photos.
- **Wording.** Show `message` texts as they are: they never say "take" or "stop", and alerts end with "Check with
  your pharmacist."
- **First call after a quiet period** can take a few minutes (the free server sleeps after about two days without
  calls, then reloads the model); call `/health` when the page opens.

## The main flows

| User does | Call |
|---|---|
| Adds a medication (name, photos) | `POST /v1/safety/medications` (whole list), then `POST /v1/embed` per enrolment photo (3–5 photos, both sides) |
| Checks a dose (one photo of the pills) | `POST /v1/scan` with the gallery and the schedule: one call does everything |
| Finds an unknown pill | `POST /v1/identify` (photo, optional imprint text) |
| Wants to see what a medicine looks like | `GET /v1/pills/search?name=…` then the pictures |
| Asks for the official warnings | `GET /v1/safety/label?rxcui=…` |
| Shares a checked photo / withdraws consent | `POST /v1/feedback` / `DELETE /v1/feedback/{consent_id}` |

---

## Health

### `GET /health`
```json
{"status": "ok", "api_version": "0.4.0",
 "model": {"loaded": true, "version": "stage2_run4@2026-10-09", "device": "zerogpu", "error": null},
 "catalogue": {"loaded": true, "products": 4936, "model_version": "stage2_run4@2026-10-09", "error": null},
 "safety_db": {"loaded": true, "built_at": "2026-10-08T17:20:09", "pairs": "234981"}}
```
`model.device` says where photos run now: `zerogpu` (a GPU Hugging Face lends per call, free within a daily
allowance) or `cpu` (slower; used when no GPU was granted). Same model either way: the embeddings differ only by
rounding. Measured from our desktop: about 1 s per photo (3 s for the first after a pause), 2 s per scan.

## Pill verification and identification (Stage 2)

### `POST /v1/embed`: a photo → its embedding
Multipart form.

| Field | Required | |
|---|---|---|
| `image` | yes | the photo (JPEG/PNG; phone photos fine, EXIF rotation is handled) |
| `box` | no | `[x0, y0, x1, y1]` of the pill in the photo, in pixels. Without it the image must already be a crop with the pill in the middle filling ~60–70% |

```json
{"embedding": [0.0213, -0.0871, ...256 numbers...], "dim": 256,
 "model_version": "stage2_run4@2026-10-09", "framed_by": "box"}
```
Use at **enrolment**: store each embedding with the medication and the `model_version`.

### `POST /v1/match`: which of my medicines is this?
JSON.
```json
{"queries": [[...256 numbers...]],
 "gallery": [{"medication_id": "m1", "name": "Metformin 500 mg", "embeddings": [[...], [...], [...]]},
             {"medication_id": "m2", "name": "Amlodipine 5 mg", "embeddings": [[...], [...]]}],
 "model_version": "stage2_run4@2026-10-09"}
```
Answer, one result per query:
```json
{"results": [{"decision": "match", "medication_id": "m1", "name": "Metformin 500 mg", "similarity": 0.9312,
              "closest": {"medication_id": "m1", "name": "Metformin 500 mg", "similarity": 0.9312},
              "runner_up": {"medication_id": "m2", "name": "Amlodipine 5 mg", "similarity": 0.512},
              "message": "This is your Metformin 500 mg."}],
 "model_version": "stage2_run4@2026-10-09", "threshold": 0.8949, "margin": 0.115}
```

| `decision` | Show |
|---|---|
| `match` | the medication's name |
| `unknown` | "This pill isn't one of your medicines. Set it aside. Check with your pharmacist." (`medication_id` is null; `closest` is only for display) |
| `ambiguous` | "…Turn it over and photograph it again." (ask for a new photo) |

`409` = the gallery was embedded with another model version: re-embed the enrolment photos.

### `POST /v1/scan`: a whole dose in one call
Multipart form.

| Field | Required | |
|---|---|---|
| `image` | yes | one photo of the dose (pills apart, on a plain mat or table) |
| `gallery` | no | the gallery as a **JSON string** (same shape as in `/v1/match`). Without it, each pill is identified against the catalogue instead |
| `schedule` | no | JSON string: `[{"medication_id": "m1", "count": 1}, ...]`: adds the dose check |
| `boxes` | no | JSON string `[[x0, y0, x1, y1], ...]` from the pill detector (Stage 1). Without it, a simple finder is used that only works on plain backgrounds |
| `model_version` | no | as in `/v1/match` |

```json
{"found_by": "stand-in finder (plain backgrounds only)", "model_version": "stage2_run4@2026-10-09",
 "pills": [{"box": [162.5, 143.8, 381.2, 387.5], "decision": "match", "medication_id": "m1",
            "name": "Metformin 500 mg", "similarity": 0.93, "closest": {...}, "runner_up": {...},
            "message": "This is your Metformin 500 mg."}],
 "dose": {"alerts": [ALERT, ...],
          "summary": {"pills_seen": 3, "matched": 2, "unknown": 1, "unsure": 0, "all_ok": false,
                      "text": "I see three pills. ..."}}}
```
No pill found: `"pills": []` and a `message` asking for a new photo. Draw the `box`es on the photo.

### `POST /v1/identify`: what is this pill?
Multipart form: `images` (1 or 2 files: front and back of the **same** pill), optional `boxes` (JSON, one per
image), `imprint` (letters/numbers on the pill, e.g. `M 547`: narrows the list a lot), `shape`
(e.g. `ROUND`), `colour` (e.g. `WHITE`), `top_k` (default 5).
```json
{"candidates": [{"pill_id": "001725729_1", "name": "FAMOTIDINE Tablets USP",
                 "rxnorm_name": "famotidine 20 MG Oral Tablet", "generic_name": "FAMOTIDINE",
                 "manufacturer": "TEVA", "ingredients": ["famotidine"], "shape": "ROUND", "colour": "BEIGE",
                 "size_mm": 6.0, "imprint": "TEVA;5729",
                 "images": [{"url": "/v1/pills/001725729_1/image/0", "kind": "rximage"}],
                 "similarity": 0.959, "imprint_match": true}],
 "notes": [], "note": "Possible matches from a catalogue of US pills, ... Check with your pharmacist.",
 "model_version": "stage2_run4@2026-10-09"}
```
**Always show these as possible matches with the `note`, never as "this is X".** US products only.

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

### `GET /v1/pills/{pill_id}` and `GET /v1/pills/{pill_id}/image/{n}`
One product (same fields as above), and its pictures as JPEG: `rximage` (both sides with a mm ruler),
`reference` (studio), `photo` (everyday). Use the `url` values directly in `<img src="BASE_URL + url">`.

## Safety checks (Stage 3)

### `POST /v1/safety/medications`: when the medication list changes
```json
{"medications": [{"id": "m1", "name": "Lipitor 10 mg"},
                 {"id": "m2", "name": "clarithromycin"},
                 {"id": "m3", "name": "Panadol (AU)", "ingredients": ["paracetamol"]}]}
```
Per medication, any of: `name`, `ndc` (US bottle code), `rxcui`, `ingredients` (for medicines without US codes).
Answer: `alerts` (interactions, duplications, unidentified medications), `medications` (how each was identified),
`all_identified`, `sources`.

### `POST /v1/safety/dose`: the dose rules alone
(`/v1/scan` with a `schedule` already does this.)
```json
{"schedule": [{"medication_id": "m1", "count": 1}],
 "pills": [{"medication_id": "m1", "decision": "match"}, {"decision": "unknown"}],
 "names": {"m1": "Metformin 500 mg"}}
```

### `GET /v1/safety/label?rxcui=…` or `?ingredient=…`
The warning sections of the official US label (openFDA), quoted as written; only when the user asks.

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

`type`: `interaction`, `duplication`, `unresolved`, `unknown_pill`, `not_scheduled`, `double_dose`, `missed`,
`unsure_pill`.

## Opt-in photos for the monthly improvement

Only when the user has **agreed** (a separate consent from using the app), and only with a checked label.

### `POST /v1/feedback`
Multipart form: `image`, optional `box`, `consent` (`true`), `consent_id` (the app's id of this user's consent;
letters, digits, `-`, `_`), `medication_name`, `label_source` (`pharmacist` or `user_confirmed`), optional `ndc`,
`rxcui`, `app_decision` (what the app had said: `match` / `unknown` / `ambiguous`), `app_medication_id`.
```json
{"stored": true, "item": "4f1c...", "note": "Thank you. The photo is used only to improve pill recognition."}
```
Only the pill crop is kept (no location data). `503` = collection is switched off on the server.

### `DELETE /v1/feedback/{consent_id}`
When the user withdraws consent: deletes all their shared photos.
```json
{"deleted": true, "local_photos_removed": 3, "removed_from_repo": true}
```

## Errors

| Code | Meaning | What the page does |
|---|---|---|
| 401 | missing or wrong `X-API-Key` | check the key |
| 404 | no such pill or picture | |
| 409 | gallery embedded with another model version | re-embed the enrolment photos |
| 422 | a field is missing or malformed (the reply says which) | fix the request |
| 502 | openFDA didn't answer | try later |
| 503 | the model, catalogue or database isn't loaded yet (or feedback is off) | retry after a few seconds; `/health` shows what's loading |

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
async function scan(file, gallery, schedule, modelVersion) {
  const form = new FormData();
  form.append("image", file);
  form.append("gallery", JSON.stringify(gallery));      // [{medication_id, name, embeddings}]
  form.append("schedule", JSON.stringify(schedule));    // [{medication_id, count}]
  form.append("model_version", modelVersion);
  const r = await fetch(`${BASE}/v1/scan`, { method: "POST", body: form });
  if (r.status === 409) { /* re-embed the enrolment photos, then retry */ }
  return r.json();                    // { pills: [...], dose: { alerts, summary } }
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
