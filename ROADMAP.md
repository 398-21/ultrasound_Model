# Roadmap

## Goal
Reuse the multi-branch classifier from the FER study (`exp-tuning.ipynb`) for **abdominal ultrasound**.
The architecture is VGG16 + CBAM and ViT-B/16 (in21k), fused by a Capsule Fusion Block and refined by a
Transformer Refinement Block. The training recipe is CLAHE, oversampling, focal loss, MixUp, AdamW,
two-phase unfreezing and TTA.

Before porting, fix the FER pipeline so its numbers are trustworthy. The ultrasound work then runs as
two tracks:
- **Track A (active), trauma-oriented:** organ recognition as a FAST anatomy baseline, then free-fluid
  detection.
- **Track B (deferred):** pathology classification.

Ultrasound details are logged in `US_PROGRESS_LOG.md`.

## Milestones

### M1: Fix the FER pipeline (code done 2026-09-29; confirmed by the Kaggle run in M2)
- [x] Stop evaluating on an oversampled/augmented test set; add a separate validation split
- [x] Fix oversampling filename collisions (augmented images overwrote each other)
- [x] Make the ViT actually trainable (run on tf-keras so its weights are tracked)
- [x] Make MixUp reshuffle every epoch
- [x] Feed VGG16 the ImageNet (caffe) preprocessing it was trained with
- [x] Clean up: explicit layer names, weight-only checkpoints, single predict pass, pinned `transformers<5`

### M2: Rerun FER on Kaggle and record the honest baseline
- [x] Make training resumable across Kaggle's 12-hour sessions (2026-09-30; see `PROGRESS_LOG.md`)
- [ ] Run the fixed `exp-tuning.ipynb` on a Kaggle GPU, resuming session by session until training finishes
- [ ] Record the results below and in `PROGRESS_LOG.md`

| Run | Test acc | Macro-F1 | Macro AUC | TTA acc | Cross-dataset acc | Runtime |
|---|---|---|---|---|---|---|
| Original (`exp-tuning_original.ipynb`, **inflated**, see log) | 0.748 | 0.73 | 0.947 | 0.749 | 0.688 | ~6.5 h |
| Fixed | | | | | | |

## Ultrasound track A: trauma-oriented (active)
Data: the Vitale et al. abdominal US dataset (Kaggle `omoikane2002/ra-us-data`). It has 926 simulated
scans (AUS) with organ masks and 617 real scans (RUS), of which about 61 have masks. There are no trauma
cases, so this track builds the anatomy baseline first and treats trauma detection as a later stage.

### T1: Organ recognition (`us-organ-recognition.ipynb`)
- [x] Notebook written: multi-label organ labels from masks, the FER architecture with a sigmoid output,
      flip-free augmentation, rare-organ repeat sampling, binary focal loss
- [ ] First Kaggle run: confirm the mask palette, file pairing and CT-volume IDs (audit cell)
- [ ] Full run: record AUS test and RUS test macro AUC with 95% CIs in `US_PROGRESS_LOG.md`

### T2: Synthetic free-fluid proof-of-concept
- [ ] Paint anechoic fluid into the liver–kidney / spleen–kidney interfaces using the masks; review a
      gallery of painted scans
- [ ] Fine-tune from the T1 weights; report as a pipeline demonstration, **not** trauma accuracy

### T3: Real free-fluid (trauma) detection
- [ ] Find labeled FAST free-fluid data. Public data is scarce; most studies use private hospital data,
      and a small Morison's-pouch set may exist. Check access and license.
- [ ] Fine-tune from the T1/T2 backbone and validate on real positives

## Ultrasound track B: pathology (deferred)

### M3: Choose the ultrasound pathology dataset
The Vitale et al. dataset was ruled out for pathology: every real subject is healthy and it only has
organ masks. It is used in track A instead.

Candidates (all figures come from the dataset papers; **verify on download**):

| Dataset | Classes | Size | Patient IDs | Split | Fit with the FER pipeline |
|---|---|---|---|---|---|
| UIdataGB (gallbladder diseases) | 9 disease classes | ~10.7k images | **to verify** | none given | Closest: folder-per-class and multi-class with imbalance, like FER |
| GBCU (gallbladder cancer) | normal / benign / malignant | ~1.25k images, ~218 patients | yes | official train/test | Small; possible external test set for gallbladder via class mapping |
| Byra fatty liver | fatty / normal (biopsy-proven) | 550 images, 55 patients | yes | none given | Binary and small; patient-wise split is easy |

Selection criteria: real pathology labels, patient IDs (so splits can be patient-wise), size, and license.

- [ ] Download the preferred candidate and confirm class counts, patient IDs and license
- [ ] Record the choice in the decisions table

### M4: Build the ultrasound notebook (from the fixed FER notebook)
- [ ] Data loader for the chosen dataset, with a **patient-wise stratified split** into train/val/test
      (`StratifiedGroupKFold` / `GroupShuffleSplit`). No patient may appear in two splits.
- [ ] Keep CLAHE; check for burned-in text, calipers and UI overlays, then crop or mask them
- [ ] Remove **horizontal flips** from oversampling, training augmentation and TTA (flips change anatomical laterality)
- [ ] `num_classes` read from the data; architecture otherwise unchanged
- [ ] Metrics: per-class sensitivity/specificity, macro AUC, confusion matrix, bootstrap 95% CIs;
      patient-level aggregation where IDs exist
- [ ] Optional: Grad-CAM on the VGG branch for sanity checks

### M5: Train and evaluate
- [ ] Full two-phase training run on Kaggle
- [ ] Optional ablations: VGG-only, ViT-only, no capsule fusion, no TRB

### M6: External validation and write-up
- [ ] External test on a second dataset if a clinically sensible class mapping exists (e.g. UIdataGB → GBCU)
- [ ] Write up methods, results and limitations

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-09-29 | Fix all issues in the FER pipeline before porting | Leakage and bugs inflate the FER results; the port would inherit them |
| 2026-09-29 | Ultrasound task = pathology classification on a labeled dataset (option B) | Keeps the clinical goal; the pipeline carries over nearly 1:1 |
| 2026-09-29 | Vitale US simulation & segmentation dataset not used for pathology | Healthy subjects only; labels are organ masks, not diagnoses |
| 2026-09-29 | FER keeps horizontal flips; ultrasound drops them | Faces are roughly symmetric; abdominal anatomy is not |
| 2026-09-29 | Start with track A (trauma-oriented) on the Vitale dataset; pathology (track B) deferred | The data is available now and maps onto FAST anatomy |
| 2026-09-29 | T1 = multi-label organ recognition, T2 = synthetic-fluid proof-of-concept | The dataset has no trauma cases; synthetic fluid only demonstrates the pipeline |
| 2026-09-29 | Separate `US_PROGRESS_LOG.md` for the ultrasound study | Keeps the FER log clean |

## Risks

| Risk | Mitigation |
|---|---|
| Kaggle 12 h session limit (fine-tuning the ViT is slower) | Full-state checkpoint every epoch; each session stops after 11 h and the next one resumes (tested locally with the real model code); parallel data loading |
| `transformers` v5 dropped TensorFlow (`TFViTModel`) | Pin `transformers<5`; run on tf-keras |
| Candidate dataset lacks patient IDs → frames of one patient leak across splits | Verify in M3; prefer datasets with IDs, otherwise flag results as image-level |
| Small medical datasets vs a ~100M-parameter model → overfitting | Two-phase unfreezing, strong regularisation, patient-wise CV, report CIs |
| Class imbalance in pathology classes | Oversampling on train only + focal loss; report per-class sensitivity |
| Sim-to-real gap: the model trains on simulated scans and is tested on real ones | Report AUS and RUS separately; treat the gap as a result |
| Only ~61 annotated real scans → noisy estimates | Bootstrap 95% CIs; score RUS only on organs its masks contain |
| Synthetic fluid ≠ clinical trauma | Label T2 results as a pipeline demonstration; real validation waits for T3 data |
| Mask RGB palette unknown until the data is opened | Audit cell asserts every mask color maps to a legend entry before training |
| AUS slices from the same CT volume leak between train and val | Volume-wise split when filenames reveal the volume; otherwise a printed warning |
