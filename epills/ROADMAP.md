# ePills Roadmap

The capstone app photographs a patient's pills on a mat and checks them against the medications enrolled
for that patient.
- **Stage 2, pill identification**, is the core deep-learning contribution: the model we design, train, tune
  and analyse.
- **Stage 3, the safety check engine**, applies fixed rules to curated data, with no trained model.
- Stage 1 (finding the pills on the mat and correcting scale and colour) is outside this roadmap, except
  where our own photos depend on it.

**Data comes first.** Without enough clean, leak-free data this project fails, so the data milestones come before
any model work. Each data milestone has a gate and a fallback (see Data plan). The dataset details behind
every number here are in [docs/datasets.md](docs/datasets.md).

Owners: **[Claude]** = Claude can do it from this repo; **[You]** = needs you (your machine, accounts,
emails, physical pills, decisions); **[Both]** = Claude prepares it and you run or approve it.

```
photo of today's pills on the mat
   │
   ▼  Stage 1 (outside this roadmap): find each pill, correct scale and colour on the mat
pill crops + size in mm + mean L*a*b* colour
   │
   ▼  Stage 2: hybrid ViT + CNN → 256-d embedding → nearest enrolled medication, or "unknown"
pill identities (or unknown / flip and retake)
   │
   ▼  Stage 3: fixed rules on curated data (DDInter 2.0, RxNorm, openFDA, the patient's schedule)
result shown to the patient; every alert ends with "check with your pharmacist"
```

---

## 1. Data plan

### 1.1 Inventory (checked 2026-10-06)
| Dataset | What we use | Download | Licence | Status | Role |
|---|---|---|---|---|---|
| **ePillID** | 3,728 consumer + 9,804 reference images, 4,902 pill types (960 with consumer images); pre-cropped at 224 px; 5 fold files | 161 MB (one zip) | MIT | **Secured**: downloaded and checked | Official benchmark; comparison with published baselines |
| **NLM C3PI, consumer photos** | `C3PI_Test` JPGs: **40,882 photos of 2,394 pill types (NDC11)**; median 15 photos per type; 2,389 types have ≥ 4 | **84 GB** (selective) | Research and development use only | **Available**: index and metadata checked; not downloaded | Main training set, after dedup |
| NLM C3PI, reference photos | `MC_CHALLENGE_V1.0` (4,408 JPGs, about 2 per type, front and back); RxImage references (4,708 JPGs, or `rximage.zip` 7.3 GB with metadata) | 2.2 GB + 2.7–7.3 GB | same | Available | Extra enrolment-style images; attribute metadata |
| NLM C3PI, metadata | `ALLXML/` (one XML per disc folder): shape, colour, **size in mm**, imprint text and type, score, camera, background, SHA-1 | ~130 MB | same | Available; one file checked | Labels for the shape and colour heads, the size input, imprint OCR, download checks |
| **MEDISEG** | 8,262 photos of 32 pill types (UK), from single pills to cluttered pill organisers; instance masks | 0.45 GB | CC BY 4.0 | **Available**: figshare files checked | Cross-domain test (new in this plan) |
| VAIPE | 9,426 phone photos, 96 pill labels, 5–10 pills per photo (Vietnam) | ? | not stated | **At risk**: vaipe.org doesn't respond; the VAIPE-PCIL crop link returns 404 | Cross-domain test, if we can get it |
| **Our enrolled pills** | 15–30 over-the-counter meds and vitamins; 5 photos per side to enrol, plus separate query photos and never-enrolled pills | ~1–3 GB | ours | **To collect** ([protocol](docs/own_pills_protocol.md)) | The real enroll-then-match test |
| DDInter 2.0 | Severity for 234,981 drug pairs (1,971 drugs) in 14 CSVs | 29 MB | CC BY-NC-SA 4.0 | **Secured**: downloaded | Stage 3 interaction severity |
| DDInter 2.0, advice text and duplications | Mechanism and management text; therapeutic duplications | website only | same | **Gap**: not in the bulk files | Stage 3 advice and duplication check |
| RxNorm API | Drug name → active ingredients | API | free | **Live** (checked) | Stage 3 ingredient mapping |
| openFDA drug label API | Label warnings | API | free | **Live** (checked) | Stage 3 label text |

**Corrections to the original plan from these checks:**
- **C3PI's "~133,000 consumer-grade photos" counts every file in the archive (2.7 TB).** That includes RAW
  camera files, videos, 2.2 TB of calibrated PNGs and reference shots. The usable consumer phone photos are the
  40,882 `C3PI_Test` JPGs (84 GB), so we download only those.
- **Every ePillID consumer pill type is in C3PI.** All 960 match on NDC9 (labeler + product), so dedup is
  essential, not optional. Each ePillID fold covers about 190 pill types and about 3,900 C3PI photos.
- **The official ePillID protocol trains on reference images of all 4,902 pill types**, so its test
  medications aren't fully unseen. We report it for comparison and add a strict protocol (section 3).
- **The DDInter bulk files hold only pair → severity.** 18% of pairs are "Unknown". Advice text and
  duplications are website-only, and the paper's 302,516 records / 2,310 drugs don't match the files (234,981
  pairs / 1,971 drugs).

### 1.2 Data gates
Each gate must pass before work that depends on it. If a gate fails, use its fallback and log the change.

| Gate | Passes when | Blocks | Fallback |
|---|---|---|---|
| **G1 Benchmark** | ePillID loads; the official holdout fold is identified | Official evaluation | — (already downloaded) |
| **G2 Training data** | C3PI consumer photos downloaded and checked against SHA-1; ≥ 90% cropped cleanly; after dedup ≥ 1,500 pill types with ≥ 4 crops | Main training | (a) Download fewer pill types (e.g. the ~2,000 with the most photos). (b) Train on ePillID + C3PI reference images with heavy augmentation; this is weaker, so report it as a limitation |
| **G3 Attribute labels** | Shape and colour for ≥ 80% of training pill types | Auxiliary heads | Mask the auxiliary losses for unlabelled types; take more labels from `rximage.zip` metadata or DailyMed |
| **G4 Cross-domain test** | At least one of MEDISEG or VAIPE is in hand, with crops | Cross-domain claim | MEDISEG is the primary option; VAIPE only if the authors or a mirror provide it |
| **G5 Real workflow** | ≥ 15 enrolled pills with enrolment and query photos, plus ≥ 5 never-enrolled pills | Real enroll-then-match claim | Fewer pills (minimum 10); report the smaller scale |
| **G6 Safety data** | DDInter severity mapped to RxNorm ingredients for all enrolled meds; advice text available for their pairs | Stage 3 alerts | Advice text only for our enrolled pairs, looked up on the DDInter site; same-ingredient duplication computed from RxNorm |

### 1.3 Splits and leakage rules
- **Pill identity = NDC9** (labeler + product). It's used for every split and dedup. It is conservative: 147
  ePillID pill types share an NDC9 with another type.
- **ePillID official** (confirmed from the ePillID code, 2026-10-07):
  - **Holdout = fold 4** (`..._5folds_4.csv`): 745 consumer images, 192 pill types. `train_cv.py` sorts the 5
    fold files and holds out the last one; the defaults in `arguments.py` and `configs/params.json` also name
    fold 4 as the test set.
  - Folds 0–3 are the 4-fold CV validation sets; together, folds 0–4 are the paper's 80/20 split.
  - The gallery is the reference images of all 4,902 pill types, including the holdout types' 384.
  - **The published numbers are two-sided.** The evaluation pairs a front and a back consumer photo of the same
    pill type, and a type's score is the mean of the two sides' similarity to that type's closest reference
    image (`metric_simul_sidepairs_eval=1`, `sidepairs_agg=post_mean`). The Papers-with-Code leaderboard is
    titled "Pill classification (both sides)". Single-side metrics (`f_`/`b_`/`s_`) are logged separately.
  - So compare like with like: report the two-sided protocol against the paper, and single-photo results
    separately.
- **C3PI training:**
  - Remove every NDC9 in the ePillID holdout fold.
  - Then split the remaining pill types into train / validation-enrolled / validation-unknown (e.g. 80 / 10 /
    10 by type).
  - After cropping, check for exact and near-duplicate images across splits (md5 and perceptual hash).
- **Strict novel-medication test:** the ePillID holdout pill types, with their reference images *also* removed
  from training. Enrol them from their reference images, then query with their consumer images.
- **Validation unknowns:** pill types never enrolled in the validation gallery. These set the rejection
  threshold.
- **Own pills:** enrolment photos and query photos come from different sessions (another day, other light,
  another phone where possible). Never-enrolled pills are only queries.
- **No test set is used for tuning.** Thresholds, δ and hyperparameters are chosen on validation only.

### 1.4 Labels for the auxiliary heads and measured inputs
- **Shape and colour:** from the C3PI XML (`Shape`, `Color`; C3PI also has `Imprint`, `ImprintType`, `Score`,
  `Size`).
  - All 960 ePillID consumer pill types match C3PI metadata, but only 2,483 of the 4,902 ePillID types do.
  - For the rest, use `rximage.zip` metadata or DailyMed product labels; otherwise mask the loss.
  - `Color` can hold two values (two-tone capsules), so check whether the colour head should be multi-label.
- **Measured size (mm):**
  - Public photos weren't taken on the mat, so size can't be measured from them.
  - Option A: in training, use the metadata `Size` (mm, longest dimension) with simulated measurement noise.
    Width isn't in the metadata, so estimate it from the crop's aspect ratio.
  - Option B: test the measured inputs only on our own mat photos.
  - Decision D4.
- **Mean L\*a\*b\* colour:** from the crop. Public crops aren't colour-corrected, so expect a shift between
  public and own photos.

---

## 2. Stage 2: pill identification

Our custom hybrid ViT + CNN maps each pill crop to a 256-dimensional embedding. Each pill is assigned to the
nearest enrolled medication, or rejected as unknown if no match is close enough.

### 2.1 Model design
- **CNN branch** (e.g. the first 3 stages of ConvNeXt-Tiny, or ResNet-50): local detail such as imprint strokes,
  score lines and edge bevels. ConvNeXt-T stages 1–3 give a 14×14 map at 224 px, which lines up with the ViT's
  patch grid.
- **ViT branch** (e.g. ViT-S/16): global shape and how colour is laid out across the pill.
- **Fusion, a key design choice to test:**
  - cross-attention from the ViT tokens (queries) to the CNN feature map (keys and values)
  - plain concatenation plus an MLP
- **Measured inputs:** pill length and width in mm, and mean colour in L\*a\*b\* from the mat-corrected crop,
  joined before the projection head. Pills that look alike in a photo often differ in size.
- **Heads:**
  - (a) a 256-d L2-normalised embedding, trained with a metric loss
  - (b) an auxiliary shape classifier
  - (c) an auxiliary colour classifier
  - Shape and colour labels come from the NLM reference metadata, where available (see 1.4).
- **Loss:** L = L_metric + λ₁·L_shape + λ₂·L_colour. Compare batch-hard triplet loss with ArcFace as L_metric.

### 2.2 Training protocol
- Initialise both branches from ImageNet weights.
- **P×K batches**, e.g. 32 pill types × 4 images each, so every batch has valid triplets. 2,389 C3PI pill types
  have ≥ 4 consumer photos. ePillID types with fewer images are sampled with replacement plus augmentation.
- Train only the heads for the first 5 epochs. Then unfreeze everything with a lower learning rate and
  layer-wise decay on the ViT.
- **Split by pill type, not by image.** Test medications are never seen in training, which shows that enrolment
  works for new medications (section 1.3).
- **Deduplicate across datasets:** remove the ePillID test pills from the C3PI training data, by NDC9 and then by
  image hash.
- **Augmentations:**
  - full 0–360° rotation, flips, blur and perspective
  - small colour jitter, because colour helps identify the pill; its size is one of the ablations
  - flips mirror the imprint text, which never happens in a real photo, so flips are in the augmentation
    ablation too
- **Compute:** the desktop RTX 4080 Super (16 GB). Batch 128 at 224 px with ConvNeXt-T + ViT-S/16 fits with mixed
  precision. Smoke tests run on CPU or the laptop.

### 2.3 Open-set rejection
- Score each pill by its cosine similarity to the nearest gallery entry. Each entry averages the enrolled
  images, front and back.
- Choose the rejection threshold on validation data to meet a safety target, such as rejecting at least 95% of
  unknown pills. Accepting a wrong pill is worse than rejecting a right one.
- If the top two matches are too close (margin below δ, also tuned on validation), the app asks the user to
  flip the pill and retake the photo.

### 2.4 Imprint reading (stretch goal)
- Fine-tune TrOCR-small on imprint text from the reference metadata (the C3PI `Imprint` field, e.g. `TEVA;5770`).
- Use it only to break ties between close matches.
- Many imprints are pressed into the pill rather than printed (`ImprintType` = DEBOSSED), and many appear on
  one side only, so OCR can't be the primary signal.

### 2.5 Ablations (one factor at a time, same split and seed)
| Factor | Arms |
|---|---|
| Fusion | cross-attention / concat + MLP |
| L_metric | batch-hard triplet / ArcFace |
| Colour jitter | none / small / large |
| Flips | on / off |
| Measured inputs | on / off |
| Auxiliary heads | on / off (λ₁, λ₂) |
| CNN branch | ConvNeXt-T stages 1–3 / ResNet-50 |

Differences within about 2 points are confirmed with 3 seeds.

---

## 3. Evaluation
| Test | Gallery | Queries | Main metrics |
|---|---|---|---|
| ePillID official | Reference images of all pill types | Holdout-fold consumer images | MAP, GAP, MAP@1, GAP@1, compared with the paper's best (ResNet-152 + compact bilinear pooling, multi-head metric learning: MAP 95.76, GAP 81.20, MAP@1 92.01, GAP@1 91.19) |
| ePillID strict (ours) | Same, but the holdout types' references were never trained on | Same | Top-1, top-5, MAP; the gap to the official protocol |
| Open set (validation → every test set) | Enrolled types | Enrolled + never-enrolled types | Wrong-pill acceptance rate (safety), unknown rejection rate, correct acceptance rate, AUROC, flip-request rate; all at the fixed validation threshold and δ, with bootstrap 95% CIs |
| Cross-domain | MEDISEG (and VAIPE if obtained): a few crops per class | The rest | Top-1, wrong-pill acceptance, unknown rejection |
| Real workflow | Our enrolled pills (5 photos per side) | Separate-session photos, including multi-pill mat photos and never-enrolled pills | Same as open set, per pill |

---

## 4. Stage 3: safety check engine
Stage 3 has no trained model. It applies fixed rules to curated data, because any alert a patient acts on must
come from a source a pharmacist can verify.

| Check | When it runs | Logic | Data source |
|---|---|---|---|
| Drug–drug interaction | When a medication is added or changed | Look up every pair of active ingredients; show severity and management advice | DDInter 2.0 (free); DrugBank if a licence is obtained |
| Therapeutic duplication | When a medication is added or changed | Two medications share an active ingredient, e.g. a brand and its generic | DDInter 2.0 duplication data; RxNorm ingredient mapping |
| Double dose | Every dose check | More pills of medication X than the schedule allows | Patient's schedule |
| Missed medication | Every dose check | A scheduled medication is missing from the photo | Patient's schedule |
| Unknown pill | Every dose check | Stage 2 rejected the pill | Stage 2 output |
| Official label text | When the user asks | Show the label's warnings section | openFDA drug label API |

**Data notes**
- **DDInter 2.0:**
  - The paper reports 2,310 drugs, 302,516 interaction records and 6,033 therapeutic duplications, with
    major, moderate and minor severity. It's free and needs no registration.
  - Licence: CC BY-NC-SA 4.0 (non-commercial).
  - The bulk CSVs (one per ATC letter) hold only drug IDs, names and the level. The download page links 8 of
    the 14 letters, but all 14 are served at the same URL pattern.
  - Advice text and duplications must come from the authors or the website (gate G6).
- **RxNorm API:** maps drug names to their active ingredients and is still live. Only RxNav's interaction API
  was discontinued, in January 2024; its endpoint now returns 404.
- **Same-ingredient duplication** (brand vs generic) can be computed from RxNorm ingredients alone. DDInter's
  duplication data adds class-level duplications (e.g. two NSAIDs).
- For medications without US identifiers, the enrolment form asks for the active ingredient and strength
  printed on the label.

**Severity mapping**
- **Major:** red result, spoken alert, caregiver notified.
- **Moderate:** amber result, spoken note.
- **Minor:** shown on screen only.
- **"Unknown" level** in DDInter (18% of pairs): show it as "interaction listed, severity not graded", treated
  like Moderate. (Proposed; decision D6.)
- Every alert ends with "check with your pharmacist". The app never tells anyone to stop taking a medication.

**Removed from the earlier plan:** the GNN and the DDI Corpus 2013. The corpus is a text-mining dataset that
labels sentences with interaction types (mechanism, effect, advice, int) but has no severity labels.
Interactions predicted by a model should never trigger a patient alert.

---

## 5. Milestones

### P0: Decisions and setup
- [ ] **[You]** Answer the decisions D1–D6 (section 6). D1 (framework) blocks P3; the others block only
      their own parts.
- [ ] **[You]** Free about **110 GB** on the desktop for C3PI: about 89 GB to download (consumer photos,
      references, metadata), plus crops and headroom.
- [ ] **[Both]** Desktop environment, once D1 is decided. Claude writes `requirements.txt` and the setup steps;
      you install them and run the GPU check.
- [x] **[Claude]** Project structure (`epills/`), roadmap, progress log, dataset notes (2026-10-06).
- [ ] **[Claude]** From the ePillID code, confirm which fold file is the official holdout.

### P1: Data acquisition (gates G1–G6)
- [x] **[Claude]** ePillID downloaded and checked (2026-10-06).
- [x] **[Claude]** C3PI index and disc listings analysed: sizes per image type, photos per pill type,
      overlap with ePillID (2026-10-06).
- [x] **[Claude]** DDInter 2.0 bulk CSVs downloaded: all 14 ATC letters (2026-10-06).
- [ ] **[Claude]** Write `scripts/download_c3pi.py`:
  - selective (only `C3PI_Test` JPGs, `MC_CHALLENGE_V1.0`, RxImage references and `ALLXML`)
  - resumable, checks each file's SHA-1 from the XML, logs failures
  - can be limited to a pill-type list
  - tested here on a ~200-file sample
- [ ] **[You]** Run the full C3PI download on the desktop (84 GB; probably hours, depending on your
      connection). Send back the summary it prints.
- [ ] **[Claude]** Download MEDISEG (0.45 GB) and check its classes, masks and licence.
- [ ] **[You]** Email the VAIPE authors for VAIPE / VAIPE-PCIL access (the site and the Drive link are dead).
      Claude drafts the email. Optional: try the Roboflow mirror with a free account.
- [ ] **[You]** Email the DDInter authors for a full export with management text and therapeutic
      duplications. Claude drafts the email.
- [ ] **[You]** Buy or collect the 15–30 over-the-counter pills and 5–10 never-enrolled pills, and photograph
      them following [docs/own_pills_protocol.md](docs/own_pills_protocol.md). This needs the mat (Stage 1);
      if there's no mat yet, Claude can design a printable one.
- [ ] **[Optional, You]** Get an openFDA API key (free; raises rate limits) and ask about a DrugBank academic
      licence.

### P2: Data preparation
- [x] **[Claude]** Parse `ALLXML` into one metadata table: NDC9/NDC11, file, class, camera, background,
      shape, colour, size, imprint, imprint type, score, SHA-1 (2026-10-07).
- [x] **[Claude]** Crop C3PI consumer photos (2026-10-07; second version: all 40,587 cropped, 199 of 200 clean in review). Each has one pill on a plain background, small in a 2–12 MP
      frame. Use background subtraction and contours, with a learned segmenter as a fallback. Save crops at
      a fixed size, with a crop-quality check and a random visual sample for you to review.
- [ ] **[Claude]** Crops for MEDISEG (from masks) and for our own photos (from Stage 1 output, or manual boxes).
- [x] **[Claude]** Dedup and splits (section 1.3), producing manifests in `data/processed/` and a leakage
      report.
- [ ] **[Claude]** Attribute labels (section 1.4) and coverage statistics (gate G3).
- [x] **[You]** Review the crop sample and the leakage report; approve gate G2 (2026-10-07).

### P3: Model (needs D1, D2)
- [ ] **[Claude]** Backbones, the two fusion arms, the measured-input MLP, the three heads, triplet and ArcFace
      losses, P×K sampler, augmentations, layer-wise LR decay.
- [ ] **[Claude]** Smoke tests on CPU or the laptop: shapes, one training step, freezing and unfreezing,
      save and load, a tiny overfit test on 10 pill types.

### P4: Training and ablations
- [ ] **[You]** Run the main training on the desktop with the config Claude provides; send back the run log.
- [ ] **[Claude]** Read the logs and pick the next run; keep the ablation table (2.5) up to date.
- [ ] **[You]** Run the ablation arms (each roughly 1–3 h; the first epoch's time is the real estimate).

### P5: Open-set calibration
- [ ] **[Claude]** Threshold and δ on validation for the safety target; the reports.

### P6: Evaluation
- [ ] **[Both]** ePillID official and strict; MEDISEG (and VAIPE); our own pills. Claude writes the evaluation
      scripts, you run them on the desktop.

### P7: Stage 3 safety engine
- [ ] **[Claude]**
  - map DDInter drug names to RxNorm ingredients
  - pair lookup, same-ingredient duplication, and the double-dose / missed / unknown rules
  - openFDA warnings lookup
  - unit tests on known pairs
- [ ] **[You]** Have a pharmacist or your supervisor review the alert wording and severity mapping.

### P8: Imprint OCR (stretch)
- [ ] **[Claude]** Imprint crops and labels from the C3PI metadata; TrOCR-small fine-tune script; tie-break
      logic.
- [ ] **[You]** Run the fine-tune on the desktop.

### P9: Write-up
- [ ] **[Both]** Results tables with CIs, ablations, the cross-domain gap, a failure gallery and limitations.

---

## 6. Decisions needed
| # | Question | Options | Recommendation | Blocks |
|---|---|---|---|---|
| D1 | Framework | PyTorch / TensorFlow | **Decided 2026-10-07: PyTorch** (a port of the FER model). Reasons: `timm` has ConvNeXt-T and ViT-S/16 with ImageNet weights, `pytorch-metric-learning` has batch-hard triplet and ArcFace, TrOCR is PyTorch-first, and it runs on the GPU on native Windows (no WSL2) | P3 |
| D2 | CNN branch for the main model | ConvNeXt-T stages 1–3 / ResNet-50 | **Decided 2026-10-07: VGG16 + CBAM, as in the FER model**; ConvNeXt-T only as an ablation. Earlier recommendation, ConvNeXt-T: its 14×14 map matches the ViT grid for cross-attention; ResNet-50 as an ablation | P3 |
| D3 | Reuse parts of the earlier hybrid (CBAM, capsule fusion) as extra arms? | yes / no | No by default; keeps the ablation focused on the two planned fusion arms | P3 |
| D4 | Measured inputs on public data | A: metadata size + noise / B: own photos only | A for training, and report B as the real test | P2, P3 |
| D5 | Cross-domain set if VAIPE stays unavailable | MEDISEG only / MEDISEG + Roboflow VAIPE mirror | MEDISEG as primary, VAIPE as a bonus | P6 |
| D6 | How to show DDInter's "Unknown" severity | as Moderate / screen only / hide | As Moderate with "severity not graded" | P7 |

## 7. Decision log
| Date | Decision | Reason |
|---|---|---|
| 2026-10-06 | Project lives in `epills/`; the FER and ultrasound work is out of scope | Change of direction; FER was finished on another machine |
| 2026-10-06 | Data milestones and gates come before model work | Without data the project fails |
| 2026-10-06 | From C3PI, download only `C3PI_Test` JPGs + references + metadata (~89 GB), not the 2.7 TB archive | The rest is RAW files, video and calibrated PNGs we don't use |
| 2026-10-06 | Pill identity = NDC9 for splits and dedup | ePillID's consumer types are all in C3PI; NDC9 is the shared key, and it errs on the side of removing more |
| 2026-10-06 | Report the ePillID official protocol and a strict novel-medication protocol | The official one trains on all pill types' reference images |
| 2026-10-06 | MEDISEG added as a cross-domain test | VAIPE's download links are dead; MEDISEG is open (CC BY 4.0) and checked |
| 2026-10-06 | Stage 3 uses fixed rules on curated data; GNN and DDI Corpus 2013 removed | Alerts must come from sources a pharmacist can verify |

## 8. Risks
| Risk | Mitigation |
|---|---|
| C3PI server goes away or throttles during the 84 GB download | Start early; resumable script; download the metadata first, then pill types in priority order (most photos first) |
| Crop failures on C3PI photos (shadows, reflections, textured backgrounds) | Quality check per crop; visual sample review; drop failed crops; learned segmenter fallback |
| Leakage between ePillID and C3PI | NDC9 dedup, then md5 and perceptual-hash checks; leakage report before training |
| Too few images per pill type for P×K | 2,389 C3PI types have ≥ 4; sample with replacement for the rest |
| Missing attributes for some pill types | Masked auxiliary losses (gate G3) |
| No mm scale or colour correction in public photos | Decision D4; report the public-to-own-photo shift |
| VAIPE unavailable | MEDISEG as the cross-domain set (D5) |
| Own photos depend on Stage 1 (the mat) | Printable mat from Claude, or manual crops and a ruler as fallback |
| Wrong pill accepted (the safety failure) | Threshold for ≥ 95% unknown rejection; report wrong-pill acceptance everywhere; flip and retake when the top two are close |
| A threshold set on validation may not hold for new phones or countries | Report open-set metrics at the fixed threshold per test set; re-check on our own pills |
| DDInter advice text and duplications missing from the bulk download | Ask the authors; fallback: look up our enrolled pairs on the site; RxNorm for same-ingredient duplication |
| Licences: C3PI is for research and development only; DDInter is non-commercial; DrugBank needs a licence | Research and coursework use only; don't redistribute raw data; cite every source |
| The app is not a medical device | Alerts only from curated sources, always deferring to a pharmacist; never "stop taking" |
