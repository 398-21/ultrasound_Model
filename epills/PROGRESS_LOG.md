# ePills Progress Log

Newest entry first. The plan, gates and decisions are in [ROADMAP.md](ROADMAP.md); dataset details are in
[docs/datasets.md](docs/datasets.md). Update the status board below whenever something changes.

---

## Who does what

| Area | Claude | You |
|---|---|---|
| **Decisions** | Lays out the options and recommends one (ROADMAP §6) | Decide D1–D6 |
| **Public data** | Finds and checks sources; writes download scripts (selective, resumable, checked against checksums); downloads small sets here (ePillID, metadata, MEDISEG, DDInter) | Runs the big C3PI download (~89 GB) on the desktop and sends back its summary |
| **Access requests** | Drafts the emails and says exactly what to ask for | Sends the emails (VAIPE authors, DDInter authors); creates accounts (Roboflow, openFDA key; DrugBank if wanted) |
| **Own pills** | Writes the protocol; designs a printable mat if needed; checks the photos and labels once delivered | Buys or collects the pills, takes the photos, fills in `labels.csv` |
| **Data preparation** | Metadata parsing, cropping, dedup, splits, attribute labels, leakage report | Reviews the crop sample and leakage report; approves gate G2 |
| **Model and training** | All model, loss, sampler and training code; configs; smoke tests on CPU or the laptop | Installs the environment and runs training and ablations on the RTX 4080 Super; sends back logs |
| **Evaluation** | Evaluation scripts (MAP/GAP, open-set metrics, bootstrap CIs), results tables, failure analysis | Runs them on the desktop where the data and weights are |
| **Stage 3** | Rule engine, DDInter ↔ RxNorm mapping, openFDA lookup, tests | Gets a pharmacist or supervisor to review alert wording and severity mapping |
| **Write-up** | Drafts sections, tables and figures | Owns the report; checks course requirements |

Claude works on this laptop (RTX 3050 Ti, 4 GB), which is enough for scripts, small downloads and smoke tests.
Anything big (the C3PI download, training) runs on your desktop.

---

## Status board

| Gate | What | Status | Next step | Owner |
|---|---|---|---|---|
| G1 | ePillID benchmark | ✅ passed 2026-10-07: downloaded and checked on both machines; holdout = fold 4; published metrics are two-sided | — | Claude |
| G2 | C3PI training photos | ✅ passed 2026-10-07 (your sign-off): 40,587 crops (199/200 clean); split by appearance group; leakage report clean; 1,710 training groups with median 15 crops | — | — |
| G3 | Attribute labels (shape, colour, size) | ✅ passed 2026-10-07 for the C3PI training types: shape, colour and size for 100% of 2,362 NDC9 (metadata table) | Majority label per type where photos disagree; ePillID reference-only types later (rximage metadata) | Claude |
| G4 | Cross-domain test | 🟡 MEDISEG available; VAIPE at risk | Download MEDISEG; email the VAIPE authors | Claude; You |
| G5 | Own pills | ⬜ not started | Choose the pills; get the mat; photo sessions | You |
| G6 | Safety data | 🟡 DDInter severity in hand; advice text and duplications missing | Email the DDInter authors; RxNorm mapping | You; Claude |

| Decision | Status |
|---|---|
| D1 framework | ✅ PyTorch (2026-10-07): a faithful port of the FER model |
| D2 CNN branch | ✅ VGG16 + CBAM from the FER model (2026-10-07); ConvNeXt-T only as a possible ablation |
| D3 earlier-hybrid parts as extra arms | ⬜ open (recommendation: no) |
| D4 measured inputs on public data | ⬜ open (recommendation: metadata size + noise for training) |
| D5 cross-domain set | ⬜ open (recommendation: MEDISEG primary) |
| D6 DDInter "Unknown" severity | ⬜ open (recommendation: show as Moderate, "not graded") |

---

## 2026-10-07 (late night): Stage 2 model ported from FER to PyTorch; training pipeline tested

- **Decisions:** prototype hosting with crops sent to the API is accepted (non-commercial demo). D1: PyTorch. The
  model is your FER network (`FER/exp-tuning.ipynb`) without the geometry branch, so D2 = VGG16 + CBAM.
- **Port** (`src/epills/core/models/hybrid.py`, hosted code):
  - VGG16 (ImageNet, blocks 4-5 trainable in phase 1) + your CBAM; ViT-B/16 in21k (same checkpoint, pooler output);
    your capsule fusion (2 routing iterations, residual update) and transformer refinement block (4 x 64 tokens,
    4 heads with Keras-style key_dim 64, 2 layers, post-norm); Keras initialisers and norm settings.
  - Stage 2 heads replace the 7-class softmax: 256-d L2-normalised embedding, shape (13) and colour (12, multi-label).
  - **Checked against your TensorFlow code**: parameter counts identical for every block (VGG16 14,714,688; CBAM
    99,043; fusion 396,288; refinement 232,448; ViT 86,389,248). With the TF weights loaded into the port, outputs
    match to 4e-7 (TF run on CPU; on GPU TF32 rounding gives 4e-4). Kept as `tests/test_hybrid_parity.py`.
  - Deliberate differences: torchvision's VGG16 weights with their own normalisation; the new heads.
- **Training pipeline:** P x K sampler and augmentations (`training/datasets/pills.py`), ArcFace / batch-hard
  triplet / auxiliary losses (`training/losses/metric.py`), validation metrics (`training/eval/retrieval.py`),
  matching with unknown rejection (`core/inference/matching.py`, shared with the API), `scripts/train_stage2.py`,
  `configs/stage2_baseline.yaml`, `requirements-train.txt`. 12 tests pass.
- **Speed (RTX 4080 SUPER, bf16):** phase 1 633 img/s (~47 s per epoch of 29,471 photos); phase 2 123 img/s at
  batch 64 (batch 128 needs 17 GB). CPU with 2 threads (free HF tier): ~0.3 s per pill.
- **Smoke test:** overfitting 10 groups reaches 96% top-1 in 16 epochs.
- **No full training yet** (your call: tests only for now). A full run was started by mistake and stopped after
  18 of 25 phase-1 epochs; kept for reference in `outputs/runs/stage2_baseline_stopped_epoch_partial/`. It was
  still improving: gallery-of-10 top-1 0.37 -> 0.72, top-1 against all 214 validation groups 0.04 -> 0.15, unknown
  AUROC ~0.55. These are not results; the run is incomplete.

---

## 2026-10-07 (night): Gate G2 passed; code split into hosted and not hosted

- **G2 passed** with your sign-off on the crop sample (`review_2027_*.jpg`) and the leakage report.
- **Goal:** your part (Stages 2 and 3) ends as an API the app calls, hosted for free (Hugging Face Spaces, Docker).
- **New layout** (README "Folder structure"; `serving/README.md`):
  - Hosted: `serving/` (FastAPI app, Dockerfile, `requirements.txt`) and `src/epills/core/` (models, inference,
    safety). Weights come from a Hugging Face model repo at start-up, never from git.
  - Not hosted: `src/epills/training/` (the C3PI cropper moved here from `src/epills/datasets/`), scripts, configs,
    notebooks, tests, docs, data, outputs.
  - `tests/test_layout.py` fails if hosted code imports training code.
- **API skeleton:** `GET /health` works locally (tested with uvicorn and `tests/test_serving.py`). The draft
  contract for `/v1/embed`, `/v1/match`, `/v1/safety/medications` and `/v1/safety/dose` is in `serving/README.md`.
  The server is stateless: galleries and schedules stay on the phone.
- **Not tested:** the Docker image (Docker isn't installed here); it's built by Hugging Face when the Space is set up.
- **Open for you:** privacy (crops leave the phone with a hosted model) and licences (DDInter NC-SA, C3PI research
  use) before the Space goes public; see `serving/README.md`.

---

## 2026-10-07 (evening): Cropper fixes, re-crop, splits and leakage report. Gate G2 ready for sign-off

- **Cropper fixes** (`src/epills/datasets/c3pi_crop.py`), for the review failures:
  - White pill on white paper: the shadow out-contrasted the pill. A shadow-like winner (darker, colourless) is now
    merged with the pill beside it; white and grey pills get a wider window.
  - Granite with a dark gap behind the counter edge: a second, local background model handles the step.
  - Fixed 12 of 13 collected failures; on 150 fresh photos, side by side, nothing got worse. Crops are framed as
    tightly as before (median pill length 71% of the window).
- **Re-crop:** all 40,587 photos cropped (first version: 40,550). Review sample (seed 2027): 199 of 200 clean; #146
  is a capsule half out of frame on sheet metal. The first run is kept as `crops_v1_superseded/`.
- **New `scripts/make_splits.py`** -> `data/processed/c3pi_splits/` (manifest, groups, leakage report, summary):
  - **Pill type = NDC9 + part.** 26 NDC9s are multi-pill packs (2-5 different pills under one code).
  - **Appearance groups:** pill types with identical imprint, shape, colour and size are one physical pill under
    several codes (repackagers, brand vs generic from one maker): 111 pill types in 49 shared groups; 2,400 pill types
    -> 2,338 groups. Splits are by group, and the training label is the group.
  - **ePillID holdout (fold 4)** removed by group: 204 pill types, plus 14 look-alikes of holdout pills in other
    products (e.g. Junel / Loestrin, Nortrel / Balziva).
  - **Splits (seed 2026):** train 29,471 photos / 1,710 groups; val_enrolled 3,556 / 214; val_unknown 3,555 / 214;
    epillid_holdout 4,005 / 200. Median 15 photos per group in every split.
- **Leakage checks:** 0 groups and 0 identical files across splits. Copies of the same photo across splits: 0.
  6,087 look-alike pairs were checked; a copy must match on centre texture and pixels (calibrated on JPEG-70
  re-encoded, shifted copies). Visual checks of the closest pairs: all different pills that share camera, paper and
  position. 9 NDC9s span two splits; all are multi-pill packs whose parts look different, so no leak.

---

## 2026-10-07 (later): C3PI metadata table and pill crops. Gate G3 passed; G2 waiting on review and dedup

### Metadata table (`scripts/parse_c3pi_metadata.py` → `data/interim/c3pi/metadata.csv`)
- One row per image from all 110 `ALLXML` files: 133,458 images.
  - The XML lists 316 consumer files twice, with identical entries; the repeats are dropped.
  - The table matches the download: 49,703 downloaded, including the 40,587 consumer JPGs.
- Columns: identity (NDC9/NDC11, part, names), attributes (shape, colour, size in mm, imprint, imprint type and
  colour, score, symbol), capture (camera, illumination, background, polarizer, clarity ratings), file size,
  SHA-1 and a `downloaded` flag. Two-tone colours are kept as `A;B`.
- **Gate G3:** for the 2,362 consumer pill types (NDC9):
  - Shape, colour, size, imprint type and score are present for 100%; imprint for 99.7%.
  - Photos of the same type disagree for a few types: shape 4, colour 32, size 17, imprint 66. Use the
    majority value.
  - 227 types are two-tone, so the colour head should be multi-label (ROADMAP §1.4).
- The consumer photos cover 27 backgrounds, 9 cameras and 11 lighting setups.
  - The grey cards and white papers make up about 70%.
  - Hard surfaces are rare (3.8% in total): granite, a human hand, dark laminate, denim, the counting tray,
    sheet metal and an amber pill bottle.

### Crops (`src/epills/datasets/c3pi_crop.py`, `scripts/crop_c3pi.py` → `data/interim/c3pi/crops/`)
- **Method:** classical, no learned model.
  - A robust quadratic background surface in Lab.
  - Colour-difference blobs over a threshold sweep, scored by contrast-to-noise against a ring around the blob,
    convexity and closeness to the centre.
  - A size prior: the expected pill length is the camera's px/mm times the metadata size in mm.
  - GrabCut refinement, so the full outline of a pale pill is found.
  - A square window 1.4× the pill's long side. It is widened to 2.2× the expected length when only part of the
    pill (or its shadow) was found, judged by length and by area against the shape.
- **Scale calibration** (4,000 random photos, `scale_table.csv`): px/mm is consistent within each camera and
  image width, with relative IQR 0.16–0.31. Examples: Canon G10 75.2, Kodak M590 50.3, Nikon S3000 38.3,
  Vivitar 27.3, webcam 21.7.
- **Run:** 40,587 photos in 21 min (14 processes, 32 photos/s).
  - 40,550 cropped (384×384 JPEG q95, 1.5 GB) and 37 with no pill found.
  - 1,645 windows widened to the expected size.
  - 4,470 photos show a second pill of the same kind; only one is cropped, and the photo is flagged
    `second_pill`.
- **Pill types:** 2,362 NDC9 with crops (min 3, median 15, max 101); 2,357 have ≥ 4.
- **Quality, checked by eye:**
  - Prototype iterations (150 random photos): v1 failed often; v2 about 90%; v3 about 93%.
  - The final method on a fresh random 200: 8 failures (96%).
  - The run's own review sample (`review_2026_*.jpg`, 200 random crops): **5 failures (97.5%)**. Four cut part
    of a pill at the edge (numbers 36, 76, 105, 133); one granite photo missed the pill (150).
  - Overall 13 failures in 400 (96.8%). The gate needs ≥ 90%.
  - Remaining failure modes: pale pills where only the shadow was found and the window is off-centre; granite;
    some hand photos.
- **Tests:** `tests/test_c3pi_crop.py` (synthetic pill, size-prior widening, scale table), 3 passed.
- **Environment:** `epills/.venv` (Python 3.13) from `requirements-data.txt`: numpy, opencv-python-headless,
  pandas, pillow, pytest. Model packages wait for D1.

### Next
- **You:** look through `data/interim/c3pi/crops/review_2026_0.jpg` and `_1.jpg` (what each number is:
  `review_2026.csv`).
- **Claude:** dedup and splits (ROADMAP §1.3) with a leakage report:
  - Map ePillID's consumer images to C3PI files by pixels.
  - Remove the ePillID holdout NDC9s.
  - Split the rest by type into train / validation-enrolled / validation-unknown.
  - Check near-duplicates across splits.
- **Optional:** crop the second pill too (+4,470 crops, often the other side), and crop the reference images.

---

## 2026-10-07: ePillID and DDInter on the desktop; official holdout fold confirmed. Gate G1 passed

- **Why:** `data/` isn't in git, so the 2026-10-06 downloads stayed on the laptop. Training runs on the desktop.
- **New `scripts/download_small_sources.py`:** downloads and checks ePillID and DDInter, reusing `fetch` from
  `download_c3pi.py`. Rerunnable.
  - **ePillID:** the SHA-256 matches `docs/datasets.md`. 13,532 labelled images (3,728 consumer, 9,804
    reference), 4,902 pill types, 960 with consumer images, 5 fold files; unzipped to `data/raw/epillid/`
    (389 MB).
  - **DDInter:** 14 files, 234,981 unique pairs over 1,971 drugs, the same as on the laptop. In
    `data/safety/ddinter/`.
- **Official holdout fold,** from the ePillID source (master branch):
  - **Fold 4:** 745 consumer images, 192 pill types (189 NDC9), none shared with folds 0–3. `train_cv.py` holds
    out the last sorted fold file; `arguments.py` and `configs/params.json` default to fold 4 as the test set.
  - The gallery holds the reference images of all 4,902 types, including the 384 of the holdout types.
- **Published metrics are two-sided:**
  - Queries are simulated front + back pairs of consumer photos of one pill type.
  - A type's score is the mean of the two sides' best-reference similarity (`sidepairs_agg=post_mean`).
  - The Papers-with-Code leaderboard is "Pill classification (both sides)".
  - Details in ROADMAP §1.3.

---

## 2026-10-06 (later): C3PI download script; full download started on the desktop. Gate G2

- **Where:** Claude ran it on the desktop (RTX 4080 SUPER PC, Windows Python 3.13, 1 TB free), from a desktop
  Claude session.
- **Script:** `scripts/download_c3pi.py` (standard library only).
  - Downloads metadata first, then the selected images: references first, then consumer photos one per pill
    type per round.
  - Checks every image against its SHA-1 in `ALLXML`.
  - Resumable: a file only gets its final name after its checks; reruns skip finished files; Ctrl+C stops
    cleanly.
- **Metadata:** all of `meta/` and the 110 `ALLXML` files (125 MB) are now in `data/raw/c3pi/` on the desktop.
  The directory's SHA-256 matches the laptop's copy.
- **Selection:**
  - 49,703 images, 88.3 GB: MC_CHALLENGE_V1.0 4,408, MC_API_RXNAV_V1.3 4,708, C3PI_Test 40,587 JPGs.
  - Correction: 295 of the 40,882 `C3PI_Test` JPG lines repeat an earlier line exactly, so there are 40,587
    distinct consumer photos (83.4 GB).
  - Every selected image has a SHA-1 in `ALLXML`.
  - No file names that are unsafe on Windows, and no case-only name collisions.
- **Test:** 700 images.
  - No failures and no checksum mismatches.
  - A rerun skipped all 700, and `--verify` re-hashed 500 with no mismatches.
  - The first 100 consumer photos covered 100 different pill types.
  - Throughput: about 7.5 MB/s with 8 parallel downloads and about 15 MB/s with 16 (now the default), so
    about 1.5–2 h for everything.
- **Full run:** 2026-10-06 22:28 → 2026-10-07 00:10. The log is `data/interim/c3pi/download_log.txt`.
  - Paused at 39.7 GB on your request, to check the size: the archive's 2.6 TB is the full archive; our
    selection is 88.3 GB.
  - Resumed at 23:23. The resume skipped the 26,489 finished files and replaced the 9 partial ones.
  - **Result:** all 49,703 images present, every one checked against its SHA-1, 0 failures, 0 leftover
    `.part` files.

    | Image type | Images | GB | Pill types (NDC11 + part) |
    |---|---|---|---|
    | C3PI_Test (consumer) | 40,587 | 83.37 | 2,432 |
    | MC_CHALLENGE_V1.0 | 4,408 | 2.20 | 2,148 |
    | MC_API_RXNAV_V1.3 | 4,708 | 2.69 | 4,676 |

  - On disk: 83 GiB in `data/raw/c3pi/images/`; 926 GB free afterwards.
  - Consumer photos per medicine:
    - By NDC9: 2,362 medicines, minimum 3, median 15, 10th percentile 11, 90th percentile 30, maximum 101.
      1,950 medicines have 10–19 photos.
    - By NDC11 + part: 2,413 of 2,432 types have ≥ 4.
  - Speed: about 5 MB/s on the small reference images, 14–21 MB/s on the consumer photos.
- **`rximage.zip`: not needed for images.**
  - Its originals are the same pictures as `MC_API_RXNAV_V1.3` / `MC_API_NLMIMAGE_V1.3`: listed via HTTP range
    requests, the file sizes differ by exactly 10 bytes. The disc folders cover 295 more NDC11s than the zip.
  - Its 11 MB of metadata (`table.csv`, `MONGOexport/rximagesAll.json`) is useful: RxCUI per NDC (Stage 3),
    plus shape, colour, size and imprint. It can be extracted from the zip without downloading it (offered,
    not done).
- **Not downloaded:** `rximage.zip` (7.3 GB; `--with-rximage` fetches it, with resume and MD5 check).
- **Note:** this is the first `epills` work on the desktop. The laptop's `data/` (ePillID zip, DDInter CSVs) is
  not here; copy or re-download it when training moves to the desktop.

---

## 2026-10-06: Project set up; data sources checked. Status: P0 / P1

### Done (Claude)
- Created `epills/` with the folder structure (see [README.md](README.md)), the roadmap, this log, the dataset
  notes and the own-pills photo protocol.
- **ePillID:** downloaded the release (160.7 MB) to `data/raw/epillid/` and checked the labels and fold files.
  - 13,532 images (3,728 consumer, 9,804 reference); 4,902 pill types, 960 with consumer images.
  - Pre-cropped at 224 px.
  - 5 fold files of about 192 pill types each.
- **C3PI:** analysed the archive without downloading it.
  - Read the image directory (133,774 rows) and all 110 disc listings, and joined them.
  - The full archive is 2.72 TB. The useful consumer photos are the **40,882 `C3PI_Test` JPGs: 2,394 pill types,
    84 GB**, median 15 photos per type.
  - References: `MC_CHALLENGE_V1.0` (2.2 GB) and RxImage (2.7 GB).
  - One metadata XML checked: it has shape, colour, size in mm, imprint, imprint type, capture conditions and
    SHA-1 per image.
  - Two sample photos viewed: each is one pill on a plain background, small in the frame, so it needs a crop.
- **ePillID vs C3PI overlap** (NDC9):
  - All 960 ePillID consumer pill types are in C3PI.
  - Each ePillID fold is about 190 pill types and about 3,900 C3PI photos.
  - About 2,170 types and 37k photos remain for training after removing one fold.
- **DDInter 2.0:** downloaded all 14 ATC-letter CSVs to `data/safety/ddinter/`. The page links only 8, but the
  other 6 are served at the same URL pattern.
  - 234,981 unique pairs over 1,971 drugs, with severity only. 18% of pairs are "Unknown".
  - No management text or duplications in the bulk files.
- **APIs:** the RxNorm API is live; the RxNav interaction API returns 404; the openFDA label API is live.
- **Cross-domain sets:**
  - VAIPE: vaipe.org doesn't respond, and the VAIPE-PCIL Google Drive link returns 404.
  - **MEDISEG** found as the replacement: 8,262 photos, 32 pill types, CC BY 4.0, 0.45 GB on figshare, files
    checked.

### Corrections to the original plan
- **C3PI** has about 41k consumer photos, not 133k. The 133k counts every file, including RAW files, video and
  calibrated PNGs.
- The **official ePillID protocol** trains on reference images of every pill type. A strict protocol is added
  to support the "works for unseen medications" claim.
- **DDInter's** bulk download lacks advice text and duplications, and its counts differ from the paper's.

### Next: Claude
1. `scripts/download_c3pi.py`:
   - downloads `ALLXML` first, then the selected image types
   - resumable, checked against SHA-1, with priority by photos per pill type
   - tested on about 200 files here
2. Parse all `ALLXML` into a metadata table and report attribute coverage (gate G3).
3. Confirm the ePillID holdout fold from the ePillID code.
4. Download and check MEDISEG.
5. Draft the two emails (VAIPE authors, DDInter authors).
6. Prototype the C3PI crop step on the sample photos.

### Next: You
1. Decide D1–D6 (ROADMAP §6). D1 matters first, because it decides the environment.
2. Free about **110 GB** on the desktop for C3PI.
3. Start choosing the over-the-counter pills (protocol §1), and check whether the Stage 1 mat exists or should
   be designed.
4. When the email drafts arrive: send them.

---

## Template: data step

```
## YYYY-MM-DD: <dataset> <step>. Gate <Gn>

- Who ran it, where (laptop / desktop), script and version
- Input: files, size; output: files, size, location
- Counts: pill types, images, per-type min / median / max
- Failures: checksum mismatches, crop failures (rate), dropped items and why
- Leakage check (if a split was made): NDC9 overlap, md5 duplicates, perceptual-hash near-duplicates
- Gate status after this step:
```

## Template: training run

```
## YYYY-MM-DD: run <n> (<arm>)

- Machine, framework versions, mixed precision on/off
- Arm: CNN / ViT / fusion (cross-attention, concat) / L_metric (triplet, ArcFace) / measured inputs on/off /
  λ₁, λ₂ / colour jitter / flips
- Data: training pill types and images (after dedup); P×K
- Phase 1 (heads only, 5 epochs): best validation metric, time per epoch
- Phase 2 (all unfrozen): LR, layer decay, epochs, best validation metric, time per epoch, peak VRAM
- ePillID official: MAP, GAP, MAP@1, GAP@1 (paper's best: 95.76 / 81.20 / 92.01 / 91.19)
- ePillID strict: top-1, top-5, MAP
- Open set at the validation threshold (τ = , δ = ): wrong-pill acceptance, unknown rejection,
  correct acceptance, AUROC, flip-request rate [95% CIs]
- Cross-domain (MEDISEG / VAIPE): top-1, wrong-pill acceptance, unknown rejection
- Own pills: top-1, wrong-pill acceptance, unknown rejection, flip-request rate
- Runtime:
- Notes / anomalies:
```
