# PharmaScan API (hosted)

Everything in this folder, plus `src/epills/core/`, is what gets deployed. Nothing else in the repository is
hosted: data, crops, training code, scripts and notebooks stay on the development machines.

## What is hosted, and what isn't

| Hosted | Not hosted |
|---|---|
| `serving/` (this API, its Dockerfile and requirements) | `src/epills/training/` (datasets, cropper, losses, evaluation, OCR training) |
| `src/epills/core/` (network definition, inference, Stage 3 rules) | `scripts/`, `configs/`, `notebooks/`, `tests/`, `docs/` |
| Trained weights, downloaded at start-up from a Hugging Face model repo | `data/` (C3PI, ePillID, MEDISEG photos and crops), `outputs/` (runs, checkpoints) |
| A small SQLite file of the DDInter pairs (Stage 3) | Raw DDInter CSVs |

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

## API contract (draft)

The server is **stateless**: it stores no photos, galleries or schedules. The app keeps each patient's gallery
(embeddings) and schedule on the phone and sends what a request needs.

| Endpoint | Input | Output | Used when |
|---|---|---|---|
| `GET /health` | none | status, API version, model loaded | Start-up, monitoring (**exists now**) |
| `POST /v1/embed` | One or more pill crops (from Stage 1), with length/width in mm and mean Lab colour when the mat gives them | One 256-d embedding per crop, plus the model version | **Enrolment**: the app stores the embeddings as the patient's gallery |
| `POST /v1/match` | Pill crops + the patient's gallery (medication id → embeddings) | Per crop: best medication, similarity, decision `match` / `unknown` / `ambiguous` (flip the pill and retake) | **Every dose check** |
| `POST /v1/safety/medications` | The medication list (names or RxCUIs, active ingredients) | Interacting pairs with severity, therapeutic duplications | **A medication is added or changed** |
| `POST /v1/safety/dose` | The schedule for this dose + the matched pills | Double doses, missed medications, unknown pills | **Every dose check**, after `/v1/match` |

Every safety result carries its source and ends with "check with your pharmacist"; the API never says take or stop.
When the model version changes, the app must re-embed its gallery (the response carries the version for this).

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
