# PharmaScan API (hosted)

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

**Hugging Face Spaces, Docker SDK, free CPU tier** (2 vCPU, 16 GB RAM).
- The Space is built from this repository's `serving/Dockerfile` (paths relative to the `epills` folder).
- Weights live in a separate Hugging Face **model** repo and are downloaded when the server starts
  (`EPILLS_MODEL_REPO`). They are never committed to git or baked into the image.
- A free Space sleeps after about 48 hours without requests; the first request then takes about a minute.
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
| `GET /health` | **works** | none | status, API version, model and safety database loaded | Start-up, monitoring |
| `POST /v1/safety/medications` | **works** | The medication list: per medication an `id` and any of `name` (brand or generic), `ndc`, `rxcui`, `ingredients` (for medicines without US codes) | Alerts (interactions, duplications, unidentified medications) and how each medication was identified | **A medication is added or changed** |
| `POST /v1/safety/dose` | **works** | The schedule for this dose (`medication_id`, `count`) and the checked pills from `/v1/match` (`medication_id`, `decision`) | Alerts (unknown pill, double dose, not part of this dose, missed, unsure) and a short summary | **Every dose check**, after `/v1/match` |
| `GET /v1/safety/label?rxcui=…` or `?ingredient=…` | **works** | RxCUI or ingredient name | Warning sections of the US label (openFDA), quoted as written | **When the user asks** |
| `POST /v1/embed` | after training | Pill crops (from Stage 1), with size in mm and mean Lab colour when the mat gives them | One 256-d embedding per crop, plus the model version | **Enrolment**: the app stores them as the gallery |
| `POST /v1/match` | after training | Pill crops + the patient's gallery (medication id → embeddings) | Per crop: best medication, similarity, decision `match` / `unknown` / `ambiguous` | **Every dose check** |

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
| `safety.sqlite`: DDInter pairs keyed to RxNorm ingredients | `scripts/build_safety_db.py` | a few MB | The Hugging Face repo in `EPILLS_MODEL_REPO` (or a local path in `EPILLS_SAFETY_DB`) |
| `best.pt`: the trained Stage 2 model | `scripts/train_stage2.py` | about 400 MB | The same repo |

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
   - C3PI photos are for research and development only. They are never hosted; only weights trained on them are.
     Keep the Space non-commercial and say so in its description.
