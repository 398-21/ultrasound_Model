# PharmaScan: feature backlog (Stages 2–3 and the API)

New capabilities agreed on 2026-10-09, so they don't get lost. The plan and decisions are in
[ROADMAP.md](../ROADMAP.md), progress in [PROGRESS_LOG.md](../PROGRESS_LOG.md), the API contract in
[serving/README.md](../serving/README.md).

| ID | Feature | Status | Owner |
|---|---|---|---|
| F1a | **Which of my medicines is this?** Photo → the patient's medication name (`/v1/embed`, `/v1/match`) | ✅ built and tested locally (2026-10-09) | Claude |
| F1b | **What is this pill?** Photo (+ imprint text) → candidate names from a catalogue of all known pills (`/v1/identify`) | ✅ built and tested locally | Claude |
| F2 | **What does this medicine look like?** Name → pictures of every product with that name (`/v1/pills/search`) | ✅ built and tested locally | Claude |
| F3 | **One-call scan:** full photo → every pill's box, name and decision (`/v1/scan`) | ✅ built; works with detector boxes; the stand-in finder only on plain backgrounds | Claude |
| F4 | Merge pill detection (Stage 1) into the recognition model | ⏸ paused: to discuss with the Stage 1 teammate | You + teammate |
| F5 | **Continual learning after deployment**, in line with Singapore law | 💭 only being considered (2026-10-10). A prototype exists but is not in use: `/v1/feedback` (off by default), `scripts/monthly_retrain.py` | You decide whether to adopt it |
| F6 | Run 4: training for other photo conditions + jittered crop windows | ✅ done 2026-10-09: MEDISEG 42% → 48%, ePillID MAP@1 37% → 42%; C3PI held-out 84% → 83% | Claude |

---

## F1a. Which of my medicines is this? (pill → name, within the patient's list)
- The app's dose check. The patient enrols each medication ("Metformin 500 mg") with 3–5 photos; the app keeps
  the name and the embeddings. The API returns the name with the decision: `match`, `unknown` (set the pill
  aside) or `ambiguous` (flip the pill and retake the photo).
- `POST /v1/embed`: pill photo (or photo + box) → 256-number embedding. Used at enrolment and at every check.
- `POST /v1/match`: query embeddings + the patient's gallery (id, name, embeddings) → decision, name, similarity.
- Expected quality (run 3, 10 medications, 3 enrolment photos, fixed threshold): on new medicines in C3PI-style
  photos, 79% named correctly, 0.0% wrongly named, 95% of other pills rejected. Lower on other phones and
  conditions (MEDISEG), where it mostly rejects instead of naming wrongly.

## F1b. What is this pill? (pill → name, without enrolment)
- Compares the photo with a **catalogue of every pill we have pictures of**: 4,823 US products (NDC9), 1,239
  generic names, from C3PI's studio reference photos and everyday photos.
- `POST /v1/identify`: photo (+ optional typed imprint, e.g. `M 547`) → top 5 candidates with name, strength,
  manufacturer, imprint, shape, colour, picture and similarity.
- **Accuracy, honestly** (run 3; new everyday photos, C3PI conditions):
  - products with everyday photos in the catalogue (2,362 of 4,936): right product first 70% (never-trained
    types) to 89% (trained types); in the top 5: 93–98%;
  - products known only from studio photos: right first about a third of the time, top 5 about two-thirds
    (ePillID result);
  - other phones and conditions are lower (MEDISEG).
  So the answer is always **"possible matches, check with your pharmacist"**, never a single definite name.
- The typed imprint narrows the candidates to products whose imprint matches; when the imprint is readable this
  is close to exact (every catalogue product has its imprint recorded).
- Limits: US products only (C3PI); a medicine outside the catalogue can only be "not found".

## F2. What does this medicine look like? (name → pictures)
- `GET /v1/pills/search?name=metformin` → every catalogue product containing that medicine: name, strength,
  manufacturer, imprint, shape, colour, size, and picture links. `GET /v1/pills/{id}` → one product;
  `GET /v1/pills/{id}/image/{n}` → the picture.
- Names are matched through RxNorm (brand ↔ generic: "Glucophage" finds metformin) and directly in C3PI's names.
- **One name → many different-looking pills** (each manufacturer and strength); the app should show all, with a
  strength filter.
- Pictures: RxImage studio photos (both sides, with size ruler) and C3PI challenge photos, served by the API.
- **Licence:** C3PI images are "for research and development purposes only": fine for our non-commercial
  prototype; a public product would need another image source (e.g. DailyMed label images).

## F3. One-call scan
- `POST /v1/scan`: full photo (+ optionally the gallery, boxes from a detector, the dose schedule) → for each pill:
  its box, decision and name (or catalogue candidates); with a schedule, also the dose-check alerts.
- Inside: find pills → crop each from the full-resolution photo with the crop contract (a square window around
  the tight pill box, its size chosen on validation for each model; run 3: 1.4× the long side) → embed → match →
  (dose rules).
- Measured on CPU: about 0.15 s per pill after a one-off model load.
- Until the Stage 1 detector exists, a **stand-in pill finder** works on plain backgrounds (pills on a mat or
  table, not touching). On MEDISEG's cluttered photos it found 1 of 40 pills, so the real detector is needed.
  The app can also send the detector's boxes; then the finder is skipped.

## F4. Merge detection into the recognition model (paused)
Paused on 2026-10-09 until you talk with the Stage 1 teammate. Claude's recommendation was **not** to merge the
networks:
- **Resolution:** one network sees the whole photo shrunk to ~640 px, so a pill is 50–100 px. MEDISEG measured
  the effect: pills under 100 px are named right 30% of the time, over 100 px 57%. Two stages crop each pill
  from the full-resolution photo.
- It would replace the FER architecture (VGG16+CBAM+ViT is a recognition network, not a detector).
- Joint training needs photos with boxes *and* identities; at scale only MEDISEG's 32 products have both.
- The goal behind it (one step for the app, robustness to the detector's boxes) is met by F3 (one call) and F6
  (training with jittered crop windows).

**To agree with the teammate:**
- Box format to send (pixel coordinates in the full-resolution photo) and who crops (the API, with F3's framing).
- Detector type: class-agnostic "pill" boxes are enough; identity comes from Stage 2.
- Where it runs (phone or server) and its licence (Ultralytics YOLO is AGPL-3.0; torchvision detectors are BSD).
- Training data for a pill detector: MEDISEG boxes (14k), the Kaggle VAIPE copy, and ~40k C3PI photos with the
  cropper's boxes.

## F5. Continual learning after deployment, in line with Singapore law
**Goal:** keep adding new medicines and pictures, and keep improving recognition, after deployment.

### Design: two levels
**Level 1: new medicines without retraining (immediate, automatic).** The model compares embeddings, so a new
medicine needs no new training: adding it means adding photos.
- Patient level: enrolment photos on the phone (as now).
- Catalogue level (F1b, F2): new products added from verified sources (pharmacist-taken photos, official
  images), each addition logged, the catalogue versioned and reversible.

**Level 2: better recognition by periodic retraining (scheduled, human-approved, never silent).**
1. **Collect:** only photos the user agreed to share for improvement (a separate opt-in), as pill crops; each
   with how its label was confirmed (pharmacist verified, user confirmed, or model only).
2. **Curate:** only verified labels enter training; flagged errors (overrides, "wrong pill" reports) are reviewed
   first; near-duplicates and leakage checked as for C3PI.
3. **Retrain** on a schedule (e.g. quarterly) with the same pipeline (train → calibrate → test).
4. **Gate:** the new model must not be worse than the current one on fixed test sets, especially the safety
   numbers (wrong-pill acceptance, unknown rejection); results go in a model card and change log.
5. **Approve and release:** a named person signs off; versioned release, staged rollout, one-step rollback.
6. **Monitor:** flip, unknown and override rates, and drift per app version; incidents reported.

**Not done: fully automatic online learning** (the model updating itself from every user answer). Users'
answers can be wrong or malicious, performance could drift silently, and Singapore's regulators expect
validated, documented changes (below).

### Singapore requirements to design for
- **PDPA (Personal Data Protection Act):** consent before using photos for model improvement (separate from
  using the app); use only for the purpose consented; tell users what is collected; let them access and correct
  it and withdraw consent (their photos then leave future training sets); keep data accurate; protect it
  (encryption, access control); keep it only as long as needed; **transfer limitation**: data leaving Singapore
  (e.g. Hugging Face servers) needs comparable protection; accountability; data-breach notification.
  Minimise: store pill crops only, strip location (EXIF), no faces or labels with names.
- **HSA, GL-04 Revision 4** (Software Medical Devices incl. machine-learning devices, a life-cycle approach,
  December 2025): ML updates go through structured change control; a **Change Management Program** allows
  pre-approved (pre-specified) updates to ship faster, with records and annual declarations; post-market
  monitoring and documentation of model behaviour and updates. Applies if PharmaScan counts as a medical device.
- **AIHGle 2.0** (MOH and HSA AI in Healthcare Guidelines, March 2026): total product life-cycle management,
  post-deployment monitoring, incident reporting and feedback channels, and documented roles for the developer,
  the deployer (e.g. a clinic or pharmacy) and the users; continuous-learning AI is covered as an emerging area.

### Open questions (for you and your supervisor)
- Does PharmaScan count as a medical device under HSA? Verifying pills and giving safety alerts may make it one.
- Where will data live? Hosting in Singapore avoids most transfer questions.
- Who verifies labels (pharmacists?), and the exact consent wording.

### What the prototype can include later
Level 1 (catalogue additions with a log), the consent flag and label-source field on stored crops, model and
catalogue version numbers in every API answer (already planned), and the regression gate as a script.

## F6. Run 4: other photo conditions + jittered crop windows
Run 3 plus training augmentation for what the test sets showed is missing:
- **Jittered crop windows:** the crop shifted and resized the way a detector's box errs (tighter, looser,
  off-centre), so recognition no longer needs perfect boxes.
- **Photo conditions:** stronger brightness and contrast, mild white-balance shifts, downscale-then-upscale (small
  pills), JPEG compression, sensor noise, and another pill partly in the frame.
- Same calibration and test sets afterwards; compared with run 3.

---

Sources (checked 2026-10-09):
[HSA GL-04 Revision 4 overview](https://www.pureglobal.com/news/revision-4-of-gl-04-singapore-enhances-oversight-of-machine-learning-enabled-software-medical-devices),
[HSA digital health](https://www.hsa.gov.sg/medical-devices/digital-health/),
[MOH AI in healthcare / AIHGle 2.0](https://moh.gov.sg/licensing-and-regulation/artificial-intelligence-in-healthcare),
[AIHGle 2.0 summary (Baker McKenzie)](https://www.bakermckenzie.com/en/insight/publications/2026/03/singapore-moh-and-hsa-launch-refreshed-ai-in-healthcare-guidelines),
[PDPC advisory guidelines on key concepts](https://www.pdpc.gov.sg/-/media/Files/PDPC/PDF-Files/Advisory-Guidelines/overview-of-dp-provisions---ch-10-(270717).pdf),
[PDPA obligations overview](https://www.aseanbriefing.com/news/understanding-personal-data-protection-in-singapore/).
