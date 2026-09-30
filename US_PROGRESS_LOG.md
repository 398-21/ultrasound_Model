# Ultrasound Study: Progress Log

Newest entry first. Each training run gets its own entry (template below). The FER work is logged
separately in `PROGRESS_LOG.md`.

On 2026-09-30 the study switched from abdominal to **cardiac** ultrasound. The abdominal entries are kept
in the archive at the end. Training will run on a **desktop RTX 4080 Super later**, not now.

## Study summary

**Goal:** cardiac point-of-care ultrasound for **emergencies at sea / in open water**, built on the FER
architecture (VGG16 + CBAM and ViT-B/16, capsule fusion, transformer refinement).

**Why cardiac:** sudden cardiovascular events are reported as the leading cause of death at sea. Ships rarely
carry a doctor, and telemedicine links are slow and unreliable. A crew member with a handheld probe needs
help with two things:
1. **Getting a usable view** of the heart.
2. **Recognising emergency findings.**

**Plan:**

| Stage | What | Input | Data | Status |
|---|---|---|---|---|
| C1 | **View + quality guidance**: which of 6 views is on screen, and is the image good enough | still images | CACTUS (phantom); TMED-2 for a real-image check | planned; first |
| C2 | **Heart-attack (MI) detection** | **video** (16–32 frames, one heartbeat) | HMC-QU | planned, after C1 |
| C3 | **Pumping function** (ejection fraction, reduced EF) | **video** | EchoNet-Dynamic, then CAMUS | later |
| C4 | Arrest findings: cardiac standstill, pericardial effusion, RV dilation | video | no public labeled data found; MIMIC-IV-ECHO reports or a partner | needs data |

The results are research only. This is not a medical device.

### Datasets
Figures come from the dataset papers and pages; verify on download. Access is for research /
non-commercial use unless stated.

**Still images**
- **CACTUS**, used in C1 ([data on FRDR](https://www.frdr-dfdr.ca/repo/dataset/86beb91e-c0ba-496b-ad38-732b67e3d5f6),
  [paper](https://arxiv.org/abs/2503.05604)):
  - 37,736 images from a CAE Blue Phantom, so no real patients.
  - 6 view classes: apical 4-chamber (A4C), parasternal long axis, parasternal short axis at the aortic
    valve, parasternal short axis at the mitral valve, subcostal 4-chamber, and random.
  - An expert quality grade from 1 to 10 for each image. Open.
- **TMED-2**, the C1 real-image test ([dataset](https://tmed.cs.tufts.edu/tmed_v2.html),
  [access](https://tmed.cs.tufts.edu/data_access.html), [license](https://tmed.cs.tufts.edu/data_license.html)):
  - 599 studies from 577 patients; 5,261 labeled images.
  - View labels (PLAX/PSAX/A2C/A4C/other) and an aortic stenosis grade.
  - Access by request to the TMED team.

**Video**
- **HMC-QU**, used in C2 ([Kaggle](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset),
  [code](https://github.com/degerliaysen/MultiEchoAI)):
  - 162 patients (93 MI, 69 non-MI), 25 frames/s, Philips and GE machines.
  - 162 A4C clips and 160 A2C clips; LV masks for 109 A4C clips.
- **EchoNet-Dynamic**, used in C3 ([Stanford AIMI](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound)):
  10,030 A4C videos (unique patients) at 112×112, with ejection fraction and LV tracings. Free Stanford
  research agreement.
- **CAMUS**, used in C3 ([project](https://www.creatis.insa-lyon.fr/Challenge/camus/),
  [database](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8),
  [Kaggle mirror, unofficial](https://www.kaggle.com/datasets/toygarr/camus-dataset)):
  - 500 patients, A2C + A4C, one heartbeat each.
  - Ejection fraction, LV/myocardium/LA masks and image quality. Free registration.
  - Cite Leclerc et al., IEEE TMI 2019.
- **CardiacNet**, optional for right-heart strain ([code + data](https://github.com/xmed-lab/CardiacNet),
  [paper](https://arxiv.org/abs/2410.20769)): 496 pulmonary arterial hypertension cases
  (catheter-confirmed) and 231 atrial septal defect cases. Check the terms.
- **MIMIC-IV-ECHO**, a C4 candidate ([dataset](https://physionet.org/content/mimic-iv-echo/1.0/)):
  - About 525k DICOM video files, 7,243 studies, 4,579 patients.
  - Labels only via linked reports and measurements.
  - PhysioNet credentialed access: a training course plus an agreement.
  - Related sets: an [A4C PoCUS-style LV subset](https://physionet.org/content/mimic-iv-echo-ext-lvvol-a4c/1.0.0/)
    (1,064 videos, 806 patients) and [ECHOVIEW view labels](https://physionet.org/content/echoview/0.1/).
- **Not planned:**
  - [EchoNet-Pediatric](https://aimi.stanford.edu/datasets/echonet-pediatric): 7,643 A4C + PSAX videos of
    children.
  - [EchoNet-LVH](https://echonet.github.io/lvh/): 12,000 PLAX videos, wall thickness.
- [CardioBench](https://github.com/BioMedIA-MBZUAI/CardioBench) collects download instructions for most of
  these.

---

## 2026-09-30 (later): Video for motion-based stages; desktop training planned. Status: planning

### Decisions
- **Video for C2 and C3, still images for C1.**
  - A heart attack shows as abnormal wall motion, and pumping function is how much the ventricle squeezes.
    Both are judged from motion, so video is the clinically accurate input.
  - Which view is on screen, and its quality, are judged from a single frame, so C1 stays image-based.
    CACTUS is images only anyway.
- **The three-frames-as-colour-channels idea for C2 is dropped** in favour of real video.
- **The architecture is reused, with one change:**
  ```
  each frame ─► VGG16+CBAM ─┐
            └─► ViT ────────┴─► Capsule Fusion ─► one 256-d vector per frame
  T frame vectors + time positions ─► Transformer Refinement Block (attends across frames) ─► output
  ```
  The refinement block already is a transformer with positional embeddings. Today it attends over 4 tokens
  cut from one image; for video it attends over T frames. Every custom component stays.
- **Training later on the desktop RTX 4080 Super** (16 GB) under WSL2. The environment steps are in
  `ROADMAP.md` → E1.

### Video design (C2 first, reused in C3)
- **Clips:** 16–32 frames covering one heartbeat. HMC-QU at 224 px (or 112); EchoNet-Dynamic at its native
  112 px, since upscaling adds ~4× compute and no information. At 112 px the ViT needs its position
  embeddings interpolated, which Hugging Face supports.
- **Two phases, mirroring FER:**
  1. Frozen frame encoder: compute per-frame features once, then train only the temporal part. It takes
     minutes and is less prone to overfitting on small datasets such as HMC-QU (162 patients).
  2. End-to-end fine-tuning with batch 1–2 clips and mixed precision (loss and capsule "squash" kept in
     float32).
- **Data pipeline:** decode each video once into arrays (e.g. `.npy`) so the CPU doesn't decode video
  during training.
- **C2 validation:** patient-wise 5-fold CV, A4C and A2C separately then combined per patient;
  patient-level sensitivity, specificity and AUC with 95% CIs.
- **C3 validation:** EchoNet-Dynamic's official split; ejection-fraction error plus reduced-EF (< 40%)
  classification; CAMUS as a second, independent test.
- **Baselines:** starting from C1 weights vs ImageNet, and a lighter video network (R(2+1)D, as used by
  EchoNet).

### Estimated training times on the RTX 4080 Super
Compute-based estimates; the first epoch's time is the real number.

| Job | Setup | Estimate |
|---|---|---|
| C1 CACTUS | 37.7k images, 224 px | ~1.5–4 h |
| C2 HMC-QU, phase 1 (frozen, cached features) | all 5 folds | a few minutes |
| C2 HMC-QU, full fine-tune | 16 frames at 224 px, 5 folds | ~30–90 min |
| C3 EchoNet-Dynamic | 32 frames at 112 px, 7,460 training clips | ~2.5–5 h |
| C3 CAMUS | 500 patients × 2 views | ~15–45 min |

### Access to request now (lead times)
- [ ] **CACTUS:** open download from FRDR, no request.
- [ ] **HMC-QU:** Kaggle, no request.
- [ ] **TMED-2:** email the TMED team via the access page; approval can take time.
- [ ] **EchoNet-Dynamic:** register at Stanford AIMI and accept the research use agreement.
- [ ] **CAMUS:** register on the Human Heart Project database.
- [ ] **MIMIC-IV-ECHO** (only if C4 goes ahead): PhysioNet credentialing needs a training course and review,
      which takes the longest.

### Next (before any training)
- Download CACTUS and run the checks listed in the entry below.
- Prototype the video version of the architecture and smoke-test it locally on tiny random clips: it
  builds, trains a step, and freezes/unfreezes correctly.
- Set up WSL2 on the desktop when training is about to start (`ROADMAP.md` → E1).

---

## 2026-09-30: Switched to cardiac ultrasound. Status: planning C1

### Decisions
- **Cardiac instead of abdominal:**
  - Cardiovascular events are reported as the leading cause of death at sea
    ([maritime health review](https://pubmed.ncbi.nlm.nih.gov/21154295/);
    [industry claims data](https://splash247.com/the-disease-of-seafaring/)).
  - Echo is the key point-of-care heart exam.
- **Order: view and quality guidance first (C1), then MI detection (C2).**
  - Onboard operators aren't sonographers, so a usable view is the prerequisite for any finding.
  - CACTUS still images are a 1:1 port of the architecture.
- The **abdominal work is parked, not deleted**. `us-organ-recognition.ipynb` stays in the repo; see the
  archive below.
- **No horizontal flips** for cardiac. Flipping mirrors the left and right chambers and breaks the standard
  view orientation.

### C1 design (`cardiac-view-quality.ipynb`, to build from the fixed `exp-tuning.ipynb`)
- **Outputs:**
  - The view, as 6-class softmax, the same kind of head as FER's 7 emotions.
  - Image quality, as a grade 1–10 or usable / not usable. Pick after seeing the grade distribution.
- **Carried over from FER:**
  - CLAHE, oversampling on train only, focal loss, MixUp, AdamW, two-phase ViT unfreezing, TTA without
    flips.
  - Multi-session resume: about 38k images is roughly FER-sized, so it may not fit in one 12-hour Kaggle
    session. (Not needed on the desktop; see the later entry.)
- **Splits:** if CACTUS images come from continuous sweeps, neighbouring frames are near-duplicates. Split by
  sweep or session, not by single image, otherwise validation and test scores are inflated.
- **Real-image check:** predict on TMED-2 real images for the views both datasets share (A4C, PLAX, PSAX)
  and report the phantom-to-real gap.

### C2 design (superseded by the video design in the entry above)
- ~~Input: three frames from one heartbeat stacked as the image's 3 color channels.~~
- **Validation:** patient-wise 5-fold cross-validation (only 162 patients). Report patient-level
  sensitivity, specificity and AUC with 95% CIs, and compare with published HMC-QU results.
- **Comparison:** start from the C1 weights (an echo-trained backbone) versus from ImageNet.

### To check when downloading CACTUS
- [ ] Folder structure and class counts per view
- [ ] How quality grades are stored (file names, CSV) and their distribution
- [ ] Whether images come from continuous sweeps, and whether a sweep/session ID is available for splitting
- [ ] Image size, format, and any on-screen text or UI overlays
- [ ] License terms
- [ ] Request TMED-2 access in parallel (approval can take time)

---

## Template: training run (cardiac)

```
## YYYY-MM-DD: <notebook> run <n>

- Machine (desktop RTX 4080 Super / Kaggle), TF / tf-keras / transformers, mixed precision on/off
- Data: train / val / test counts per class; how the split was grouped (sweep / session / patient)
- Video only: frames per clip, frame size, how the heartbeat was selected; phase 1 on cached features?
- Phase 1: epochs run, best val_loss, val metric, time per epoch
- Phase 2: epochs run, best val_loss, val metric, time per epoch, peak VRAM
- C1 views: accuracy, macro-F1 [95% CI], per-view recall, most-confused pair
- C1 quality: MAE vs expert grade, or AUC for usable / not usable
- Real-image check (TMED-2, shared views): accuracy, per-view recall
- C2 MI (patient level, 5-fold): sensitivity, specificity, AUC [95% CI]
- C3 EF: MAE, reduced-EF (<40%) AUC [95% CI]; CAMUS external test
- Baseline comparison (R(2+1)D / ImageNet vs C1 weights):
- TTA:
- md5 / near-duplicate overlaps between splits:
- Runtime:
- Notes / anomalies:
```

---
