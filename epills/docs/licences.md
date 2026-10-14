# Licences and regulatory note (blueprint job B16)

For the hackathon submission (16 October 2026). PharmaScan is a non-commercial prototype demonstrated on synthetic
data; nothing here is legal advice. Items marked **to confirm** need a decision from the team (or the organisers)
before any use beyond the demo.

## What the hosted API uses

| Component | Used for | Terms (as recorded in docs/datasets.md and the sources) | What it means for us |
|---|---|---|---|
| NLM C3PI pill images and metadata | training the pill model; catalogue pictures and imprints | research and development purposes only | Fine for the prototype. Catalogue pictures carry an `image_licence` note. A public or commercial product needs another picture source (e.g. DailyMed label images). **To confirm:** whether the hackathon's public demo counts as research and development. |
| DDInter 2.0 | interaction severities (`/v1/safety/medications`) | CC BY-NC-SA 4.0: attribution, **non-commercial**, share-alike | Attribution is in the API description and every alert's `source`. Our derived `safety.sqlite` may only be shared under the same licence; no commercial use. **To confirm** for anything after the hackathon. |
| RxNorm via NLM's RxNav APIs | names, ingredients, codes (`/v1/safety/*`, `/v1/read-label`, `/v1/pills/search`) | free public APIs under NLM's terms of service and rate limits | We call the APIs and cache answers; we don't redistribute RxNorm files. Some source vocabularies inside RxNorm carry their own terms (UMLS licence) for bulk use. We use only RxNorm's own names and codes. |
| openFDA drug labels | `/v1/safety/label` | public US government data via openFDA, no endorsement implied | Quoted as written, with label date and DailyMed link; general information, not the patient's instructions. |
| Florence-2-base (Microsoft) | reading imprints, boxes, labels | MIT | Fine. |
| Pre-trained backbones: VGG16 (torchvision, ImageNet), ViT-B/16 (Google) | starting weights of the pill model | code and weights under their projects' licences; ImageNet itself is for non-commercial research | Fine for research. **To confirm** before commercial use. |
| PyTorch, transformers, FastAPI, Gradio, OpenCV, pypdf, pypdfium2 | software | BSD / Apache-2.0 / MIT style | Fine. |

Used for **evaluation only** (not in the served model or catalogue): ePillID (MIT), MEDISEG (CC BY 4.0).

## Personal data (PDPA)

- The API **stores nothing**: no photos, galleries, schedules or results (opt-in feedback is switched off). The app's
  backend holds all patient data; NRIC numbers stay out of every table but the login (PDPC NRIC guidelines: log in
  with Singpass).
- **Transfer limitation:** photos are processed on Hugging Face servers outside Singapore. For real patients that
  needs comparable protection (contract, consent) or hosting in Singapore. **Use synthetic data only for the demo.**
- `/v1/read-label` returns only lines that name or describe a medicine; lines with names, NRIC, addresses or phone
  numbers are never used.
- Opt-in photo collection (F5, off): separate consent, withdrawal deletes the photos, but the dataset repo's git
  history keeps them until squashed (job B15). Keep it off until that's fixed.

## Medical-device regulation (HSA)

- The blueprint's stance: design PharmaScan as a **potentially regulated software medical device** and manage risk
  accordingly. Saying "not a medical device" doesn't exempt it; the intended purpose decides.
- Verifying a patient's pills and raising safety alerts may make it a medical device under HSA's GL-04 (Revision 4,
  December 2025: software medical devices, including machine learning; change control for model updates). MOH and
  HSA's AIHGle 2.0 (March 2026) adds life-cycle management, monitoring and incident reporting.
- What the prototype already does toward that: every Stage 2 answer is visual evidence with stated limits; alerts
  defer to a pharmacist; versions are pinned and checked (`/v1/model-info`, `embedding_space_id`); calibration,
  test results and known limits are documented (docs/stage2_results.md, docs/blueprint_review.md); the backend owns
  the dose decision (docs/dose_rules.md).
- **For the submission:** state that it is a research prototype, not validated for clinical use, demonstrated on
  synthetic data, and that classification and clinical validation with HSA would come before any real use.

Sources: [HSA digital health](https://www.hsa.gov.sg/medical-devices/digital-health/),
[MOH AI in healthcare / AIHGle 2.0](https://moh.gov.sg/licensing-and-regulation/artificial-intelligence-in-healthcare),
[DDInter 2.0](https://ddinter2.scbdd.com/), [NLM C3PI](https://data.lhncbc.nlm.nih.gov/public/Pills/index.html),
[RxNav APIs](https://lhncbc.nlm.nih.gov/RxNav/), [openFDA](https://open.fda.gov/), and docs/datasets.md.
