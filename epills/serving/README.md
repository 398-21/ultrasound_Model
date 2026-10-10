# PharmaScan API (hosted)

**For the web app:** [docs/API.md](../docs/API.md) (every endpoint with examples). **To deploy:** [docs/DEPLOY.md](../docs/DEPLOY.md).

Everything in this folder, plus `src/epills/core/`, is what gets deployed. Nothing else in the repository is
hosted: data, crops, training code, scripts and notebooks stay on the development machines.

## What is hosted, and what isn't

| Hosted | Not hosted |
|---|---|
| `serving/` (this API, its Dockerfile and requirements) | `src/epills/training/` (datasets, cropper, losses, evaluation, OCR training) |
| `src/epills/core/` (network definition, inference, Stage 3 rules) | `scripts/`, `configs/`, `notebooks/`, `tests/`, `docs/` |
| Trained weights, downloaded at start-up from a Hugging Face model repo | `data/` (C3PI, ePillID, MEDISEG photos and crops), `outputs/` (runs, checkpoints) |
| `safety.sqlite` (DDInter pairs + RxNorm mapping), downloaded at start-up like the weights | Raw DDInter CSVs, the RxNorm response cache |

`core` never imports `training` (checked by `tests/test_layout.py`), so the hosted code runs without any of it.

## Where it runs (free)

**Hugging Face Spaces, Gradio SDK on ZeroGPU** (free for personal accounts; plain-CPU and Docker Spaces now need PRO).
- The Space runs `space_app.py` (uploaded as its `app.py`): this API unchanged, a demo page at `/ui`, and each model
  pass on a GPU lent per call, or on the CPU when none is granted. `requirements-space.txt` pins a torch that
  ZeroGPU supports. With PRO, `serving/Dockerfile` builds the plain Docker version instead.
- Weights live in a separate Hugging Face **model** repo and are downloaded when the server starts
  (`EPILLS_MODEL_REPO`). They are never committed to git or baked into the image.
- A free Space sleeps after about 48 hours without requests; the first request then waits for it to start.
  Fine for the demo; send a request before presenting.
- Optional: an API key in the Space's secrets, checked on every request, so only our app can call it.

Local run, from the `epills` folder:
```
pip install -r serving/requirements.txt
set PYTHONPATH=src;.
uvicorn serving.app.main:app --reload --port 7860
```
Then open http://localhost:7860/docs for the interactive API page.

## API contract

The server is **stateless**: it stores no photos, galleries or schedules. The app keeps each patient's gallery
(embeddings) and schedule on the phone and sends what a request needs. The interactive page at `/docs` shows every
field and lets you try requests.

| Endpoint | Status | Input | Output | Used when |
|---|---|---|---|---|
| `GET /health` | **works** | none | status, API version; model (version), catalogue and safety database loaded | Start-up, monitoring |
| `POST /v1/safety/medications` | **works** | The medication list: per medication an `id` and any of `name` (brand or generic), `ndc`, `rxcui`, `ingredients` (for medicines without US codes) | Alerts (interactions, duplications, unidentified medications) and how each medication was identified | **A medication is added or changed** |
| `POST /v1/safety/dose` | **works** | The schedule for this dose (`medication_id`, `count`) and the checked pills from `/v1/match` (`medication_id`, `decision`) | Alerts (unknown pill, double dose, not part of this dose, missed, unsure) and a short summary | **Every dose check**, after `/v1/match` |
| `GET /v1/safety/label?rxcui=…` or `?ingredient=…` | **works** | RxCUI or ingredient name | Warning sections of the US label (openFDA), quoted as written | **When the user asks** |
| `POST /v1/embed` | **works** | A pill photo (a crop, or a photo + `box`) | Its 256-number embedding and the model version | **Enrolment** (the app stores the embeddings as the patient's gallery) and before `/v1/match` |
| `POST /v1/match` | **works** | Pill embeddings + the patient's gallery (`medication_id`, `name`, `embeddings`) | Per pill: `match` with the medication's name, `unknown` (set it aside) or `ambiguous` (turn it over, photograph again), with similarities and a message | **Every dose check** (feature F1a) |
| `POST /v1/scan` | **works** | A full photo; optional detector `boxes`, the `gallery`, the dose `schedule` | Every pill's box and decision (with a gallery) or catalogue candidates (without); with a schedule also the dose alerts | **Every dose check in one call** (F3) |
| `POST /v1/identify` | **works** | 1–2 photos of one pill (front, back); optional typed `imprint`, `shape`, `colour` | The most alike catalogue pills: names, ingredients, imprint, pictures, similarity | **"What is this pill?"** (F1b): possible matches, never a confirmed identity |
| `GET /v1/pills/search?name=…` | **works** | A brand or generic name; optional `strength` | Every catalogue product with that medicine, with pictures (`same`, then `contains`, then `name` matches) | **"What does it look like?"** (F2) |
| `GET /v1/pills/{pill_id}` and `/v1/pills/{pill_id}/image/{n}` | **works** | A catalogue id | Product details; a picture (JPEG) | Showing a candidate or product |

**Crop contract** (`src/epills/core/inference/preprocess.py`): a pill is cropped as a square window, a fixed
multiple of the long side of its **tight** box (the pill only), centred on it, from the **full-resolution** photo.
The multiple is chosen on validation for each model and shipped with it (run 3: 1.4×; validation scored 1.3–1.7×
the same). A detector sends tight boxes in the photo's pixels and the API crops. `/v1/scan` without boxes uses a stand-in finder that only
works on a plain background with pills apart; the Stage 1 detector replaces it.

**Model versions.** Every Stage 2 answer carries `model_version`. Embeddings from different versions can't be
compared: when the version changes, the app re-embeds its enrolment photos (so it keeps them), and `/v1/match`
refuses a gallery sent with another `model_version` (409). The catalogue is rebuilt for every model; the API
refuses to identify with a mismatched one (503).

**Identification accuracy** (feature F1b, run 3): for pills the model never trained on, the right product is first
about a third of the time and in the top 5 about two-thirds; a typed imprint narrows it a lot. Answers are worded as
possible matches with "Check with your pharmacist".

**Alerts** (every safety endpoint) have the same shape:
`type`, `severity` (major / moderate / ungraded / minor), `colour` (red / amber / grey), `speak`, `notify_caregiver`,
`message`, `medication_ids`, `source`, `severity_graded`, `details`.
- The plan's mapping: major = red, spoken, caregiver notified; moderate = amber, spoken; minor = on screen only.
- Every `message` ends with "Check with your pharmacist." and never says take or stop.
- DDInter's ungraded pairs ("Unknown", 18% of pairs) come back as `ungraded`: grey, on screen only, not spoken
  (decision D6), so common medication lists aren't drowned in alerts.
- A medication that can't be identified is itself an alert, so **no alerts means every medication was checked**,
  not that the check was skipped. `all_identified` says the same.
- Duplications are amber, not red: two strengths of one drug are sometimes prescribed together on purpose.

Example, a medication added:
```
POST /v1/safety/medications
{"medications": [{"id": "m1", "name": "Lipitor 10 mg"}, {"id": "m2", "name": "clarithromycin"}]}
-> alerts: [{"type": "interaction", "severity": "major", "colour": "red", "speak": true,
             "notify_caregiver": true, "message": "Lipitor 10 mg and clarithromycin have a serious interaction.
             Check with your pharmacist.", ...}]
```

When the model version changes, the app must re-embed its gallery (the embed response carries the version).

## Assets the server loads (not in git)

| Asset | Built by | Size | Where the Space gets it |
|---|---|---|---|
| `safety.sqlite`: DDInter pairs keyed to RxNorm ingredients | `scripts/build_safety_db.py` | 18 MB | The Hugging Face repo in `EPILLS_MODEL_REPO` (or a local path in `EPILLS_SAFETY_DB`) |
| `stage2_model.pt`: the trained Stage 2 model (half precision) with its calibration | `scripts/export_model.py` | 204 MB | The same repo (or `EPILLS_MODEL`) |
| `catalogue/`: products, RxNorm ingredients, embeddings, pictures | `scripts/build_catalogue.py` (rebuild for every model) | see below | The same repo, folder `catalogue/` (or `EPILLS_CATALOGUE`) |

With `EPILLS_PRELOAD=1` (set in the Dockerfile) all three load at start-up; `/health` shows what is loaded. On the
free CPU Space, one pill takes roughly a second to embed.

The server also calls two free public APIs at run time: RxNav (RxNorm, to identify medications) and openFDA
(labels). Spaces allow outgoing internet. Set `OPENFDA_API_KEY` in the Space's secrets for higher limits
(optional).

## Before going public: two decisions

1. **Privacy.** The project plan says photos never leave the phone. With a hosted model, pill crops are sent to
   the server. Options: (a) crops are processed in memory and never stored or logged, over HTTPS, and the privacy
   section of the plan says so; (b) the API is for the hackathon demo only, and the final app runs the exported
   model on the phone (TFLite / Core ML, as the plan intends).
2. **Licences.**
   - DDInter is CC BY-NC-SA 4.0: free, non-commercial use with attribution, and derived data under the same
     licence. Fine for a public, non-commercial demo if the API names DDInter as the source.
   - C3PI photos are for research and development only. The catalogue (features F1b, F2) hosts C3PI pictures:
     about 4,800 studio photos, downscaled. Keep the Space non-commercial and say so in its description; consider
     a private or gated Space. A public product would need another picture source (e.g. DailyMed label images).
