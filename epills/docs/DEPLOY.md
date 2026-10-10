# Deploying the PharmaScan API to Hugging Face, and the monthly retraining

Step by step, on the desktop (Windows, PowerShell, in the `epills` folder). Commands use the project's Python:
`.venv\Scripts\python`. What gets hosted and why: [serving/README.md](../serving/README.md); the API itself:
[docs/API.md](API.md).

## What ends up where

| Where | What | Visibility |
|---|---|---|
| Space `<you>/pharmascan-api` | only the hosted code: `serving/`, `src/epills/core/`, the Dockerfile | public code (the API's address is public; optional API key) |
| Model repo `<you>/pharmascan-assets` | `stage2_model.pt` (204 MB), `catalogue.tar` (~200 MB), `safety.sqlite` (18 MB) | **private**; each upload tagged with the model version |
| Dataset repo `<you>/pharmascan-feedback` | opt-in pill photos for retraining (only if switched on) | **private** |
| Your desktop | everything else: data, training, the monthly retraining | not hosted |

Free tier: "CPU basic" Space (2 vCPU, 16 GB RAM). It sleeps after about 48 hours without requests; the first call
afterwards takes about a minute. One pill takes about a second on its CPU.

## 1. Choose and prepare the model (once per model)

```powershell
.venv\Scripts\python scripts\export_model.py --run outputs\runs\stage2_run4     # the model to serve
.venv\Scripts\python scripts\build_catalogue.py                                   # catalogue for that model (~15 min)
```
The catalogue must be rebuilt for every new model (the API refuses a mismatched one). The model's calibration
(threshold, margin, crop window) travels inside `stage2_model.pt`.

## 2. Hugging Face account and token (once)

1. Sign up at https://huggingface.co (free). Your user name is `<you>` below.
2. Settings → Access Tokens → **Create new token**, type **Write**, name it `pharmascan`. Copy it (starts with
   `hf_`). Keep it secret: never put it in git or in a message.

## 3. Upload the assets and create the Space

In PowerShell, for this window only:
```powershell
$env:HF_TOKEN = "hf_...your token..."
.venv\Scripts\python scripts\deploy_hf.py assets --user <you>    # uploads ~420 MB, tags the version
.venv\Scripts\python scripts\deploy_hf.py space  --user <you>    # uploads the code, sets the Space's settings
```
The second command creates the Space, sets `EPILLS_MODEL_REPO`, and stores the token as the Space secret `HF_TOKEN`
(so it can read the private assets). The Space then builds by itself: open
`https://huggingface.co/spaces/<you>/pharmascan-api` and watch the **Logs** tab (5–10 minutes the first time).

When it says "Running":
- `https://<you>-pharmascan-api.hf.space/health`: `model`, `catalogue` and `safety_db` should say `loaded: true`
  (about a minute after start).
- `https://<you>-pharmascan-api.hf.space/docs`: the clickable API page. Try `GET /v1/pills/search` with `metformin`.

Give your teammate this address and [docs/API.md](API.md).

## 4. Optional settings

Space → Settings → Variables and secrets (then restart the Space):

| Name | Kind | What it does |
|---|---|---|
| `EPILLS_CORS_ORIGINS` | variable | only these web addresses may call the API from a browser, e.g. `https://our-app.vercel.app` (default: any) |
| `EPILLS_API_KEY` | secret | every `/v1` call then needs the header `X-API-Key` (a light guard; a key in a web page isn't truly secret) |
| `EPILLS_MODEL_REVISION` | variable | pins a released version, e.g. `model-stage2_run4-2026-10-09` (the tag printed by `assets`): **rollback** |
| `OPENFDA_API_KEY` | secret | higher openFDA limits for `/v1/safety/label` (free key from open.fda.gov) |

Updating later: code changes → `deploy_hf.py space`; a new model → steps 1 and 3 (`assets`), then
`deploy_hf.py restart`.

## 5. Monthly retraining on users' photos (optional, not in use)

**Status (2026-10-10): only being considered; nothing below is switched on.** Photo collection is off by default and no schedule exists. Skip this section for the deployment; it describes the prototype in case you adopt it.

**What it does.** Patients can always enrol a new medicine at once: the model compares photos, so no retraining is
needed for that. The monthly retraining makes recognition *better*, using real photos from real phones, our
weakest point (docs/stage2_results.md): photos users agreed to share, labelled by a pharmacist (or confirmed by the
user, if you allow it).

**Safeguards** (Singapore's PDPA, HSA and MOH AI guidance; docs/features.md F5):
- Collection is **off** until you switch it on, and the app sends a photo only when the user has agreed (a separate
  consent) and the label was checked.
- Only the pill crop is stored, in a private repo, grouped by consent id; withdrawal deletes all of a user's photos.
- A new model is **never published automatically**: it must pass the release gate *and* a person approves it.
- Every version is tagged, so a bad update is rolled back by one setting (`EPILLS_MODEL_REVISION`).
- Open point for your supervisor: the photos are stored on Hugging Face's servers, outside Singapore (PDPA's
  transfer limitation). Fine for a prototype with test users; with real patients, check it first.

**Switch collection on (once):**
```powershell
.venv\Scripts\python scripts\deploy_hf.py feedback --user <you>
.venv\Scripts\python scripts\deploy_hf.py restart --user <you>
```
The web app then calls `POST /v1/feedback` (only with consent) and `DELETE /v1/feedback/{consent_id}` on withdrawal.

**Every month (about 1–2 hours on the GPU):**
```powershell
$env:HF_TOKEN = "hf_..."; $env:EPILLS_FEEDBACK_REPO = "<you>/pharmascan-feedback"
.venv\Scripts\python scripts\monthly_retrain.py pull build train check
```
1. `pull` downloads the month's photos; `build` keeps the checked ones, adds them to training (a known medicine joins
   its class, a new one becomes a new class) and holds back the newest 20% as this month's test.
2. `train` fine-tunes the live model on them (gently, up to 10 epochs).
3. `check` calibrates and tests it like every model, and writes
   `outputs\runs\stage2_monthly_<YYYYMM>\release_report.md`: the new model against the live one, with the gate
   (correct matches may drop at most 2 points, wrong matches may rise at most 0.2 points, unknown pills rejected
   stays at least 94%, or drops at most 1 point where the live model is below that) and accuracy on this month's
   held-back photos.
4. **Read the report.** If it says PASS and you agree:
   ```powershell
   .venv\Scripts\python scripts\monthly_retrain.py publish --from-run outputs\runs\stage2_monthly_<YYYYMM> --user <you>
   ```
   This exports the model, rebuilds the catalogue, uploads a new tagged version and restarts the Space. The app
   notices the new `model_version` and re-embeds its enrolment photos.

The live run is remembered in `outputs\serving\served_run.txt`, so next month starts from it automatically.

**Optional: start it automatically on the 1st of each month** (Windows Task Scheduler; publishing stays manual).
Store the two settings for your user once (`setx HF_TOKEN "hf_..."` and
`setx EPILLS_FEEDBACK_REPO "<you>/pharmascan-feedback"`), then:
```powershell
schtasks /Create /SC MONTHLY /D 1 /ST 02:00 /TN "PharmaScan monthly retraining" /TR "cmd /c cd /d C:\Users\sheng\Documents\GitHub\ultrasound_Model\epills && .venv\Scripts\python scripts\monthly_retrain.py pull build train check > outputs\monthly_retrain.log 2>&1"
```
The PC must be on at that time. Afterwards, read the report and publish by hand.

## Licences and privacy, before going public

- Non-commercial only: DDInter is CC BY-NC-SA 4.0; C3PI pictures (in the catalogue) are for research and
  development use. Say so in the Space description (the deploy script does).
- Photos sent to the API are processed in memory and not stored, except opt-in feedback photos.
- The prototype is not a medical device; every alert defers to a pharmacist.
