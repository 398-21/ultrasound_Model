# Roadmap

## Goal
Reuse the multi-branch classifier from the FER study (`exp-tuning.ipynb`) for **cardiac point-of-care
ultrasound in emergencies at sea / in open water**. The architecture is VGG16 + CBAM and ViT-B/16 (in21k),
fused by a Capsule Fusion Block and refined by a Transformer Refinement Block. The training recipe is
CLAHE, oversampling, focal loss, MixUp, AdamW, two-phase unfreezing and TTA.

**Why cardiac:** sudden cardiovascular events are the leading cause of death at sea. Onboard there is rarely
a doctor, and telemedicine links are slow, so a model has to help twice:
1. Help an untrained crew member get a usable heart view.
2. Flag emergency findings, starting with a heart attack (MI).

**Still images or video:**
- Recognising which view is on screen, and how good it is, works from **still images** (C1).
- Findings judged by motion — wall motion in a heart attack, pumping function, cardiac standstill — use
  **video** (C2, C3). Video is the clinically accurate input for those.

Training will run on a **desktop RTX 4080 Super later**, not now (see Compute).

Before porting, fix the FER pipeline so its numbers are trustworthy. The earlier abdominal ultrasound work
is **parked** (see the end of this file). Ultrasound details are logged in `US_PROGRESS_LOG.md`.

## Milestones

### M1: Fix the FER pipeline (code done 2026-09-29; confirmed by the Kaggle run in M2)
- [x] Stop evaluating on an oversampled/augmented test set; add a separate validation split
- [x] Fix oversampling filename collisions (augmented images overwrote each other)
- [x] Make the ViT actually trainable (run on tf-keras so its weights are tracked)
- [x] Make MixUp reshuffle every epoch
- [x] Feed VGG16 the ImageNet (caffe) preprocessing it was trained with
- [x] Clean up: explicit layer names, weight-only checkpoints, single predict pass, pinned `transformers<5`

### M2: Rerun FER and record the honest baseline
- [x] Make training resumable across Kaggle's 12-hour sessions (2026-09-30; removed 2026-10-02 because training
      moved to the desktop)
- [x] Run the fixed `exp-tuning.ipynb` to completion (results reported 2026-10-01)
- [x] Record the test results below and in `PROGRESS_LOG.md`. TTA, cross-dataset accuracy and runtime were
      not reported yet.

| Run | Test acc | Macro-F1 | Macro AUC | TTA acc | Cross-dataset acc | Runtime |
|---|---|---|---|---|---|---|
| Original (`exp-tuning_original.ipynb`, **inflated**, see log) | 0.748 | 0.73 | 0.947 | 0.749 | 0.688 | ~6.5 h |
| Fixed (corrected baseline, reported 2026-10-01) | 0.72 | 0.67 | 0.933 | not reported | not reported | not reported |

### M2b: Face-geometry ablation (FER)
Fear (F1 0.46), disgust (0.56) and sad (precision 0.56) are the weak classes. This milestone tests whether
facial-landmark geometry, added as a third capsule, helps them. The write-up is a paper-style draft:
[Geometry-Aware Capsule Fusion for Facial Expression Recognition](https://claude.ai/code/artifact/f3627e41-a21a-4b65-9685-6fd2c76d47ae).
- [x] Geometry branch, two-arm switch and ablation cell implemented and tested locally (2026-10-02; see
      `PROGRESS_LOG.md`)
- [x] Notebook simplified for the desktop: requirements and install cell at the top; checkpoints and the Kaggle
      session limit removed (2026-10-02)
- [ ] Baseline arm: reuse the corrected baseline's final weights with `PRETRAINED_WEIGHTS` (evaluation only, about
      30 min), or retrain
- [ ] Geometry arm: train (estimated 2.5–5.5 h on the RTX 4080 Super)
- [ ] Run the ablation cell and fill the results table in the paper draft
- [ ] If a difference is within about 2 points, confirm with 3 seeds per arm

## Compute: desktop RTX 4080 Super (planned, later)

### E1: Desktop setup (when training starts)
- [ ] **WSL2.** TensorFlow's last GPU release on native Windows was 2.10, and these notebooks need newer.
  1. `wsl --install -d Ubuntu-22.04` (PowerShell as administrator), then reboot. Keep the normal Windows
     NVIDIA driver; don't install a Linux driver inside WSL.
  2. In Ubuntu, create a venv and install
     `"tensorflow[and-cuda]==2.19.*" "tf-keras==2.19.*" "transformers>=4.40,<5" opencv-python-headless pandas scikit-learn matplotlib seaborn kagglehub`.
  3. Check the GPU:
     `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
  4. Open the project in VS Code through the WSL extension.
- [ ] **Local mode** for the notebooks:
  - Detect Kaggle vs desktop and point data paths at a local data folder (datasets downloaded with
    `kagglehub` or from their sites).
  - Resume stays available but isn't needed: there's no 12-hour limit.
  - Done for `exp-tuning.ipynb` (2026-10-02). It detects Kaggle vs desktop. On the desktop it uses `data/` and
    `outputs/` next to the notebook and `~/fer_scratch` for rebuilt images (on WSL keep that on the Linux side,
    not `/mnt/c`), with no time budget. All of these are editable in the settings cell's desktop block. The
    other notebooks are still Kaggle-only.
- [ ] **Optional mixed precision** (FP16 on the tensor cores): about 2× faster and half the memory. Keep
      the loss and the capsule "squash" in float32.

### Estimated training times on the RTX 4080 Super
These are compute-based estimates; the first epoch's time is the real number.

| Job | Setup | Estimate |
|---|---|---|
| FER (M2) | ~58k images, 224 px | ~3–6 min/epoch → ~2–5 h |
| C1 CACTUS | 37.7k images, 224 px | ~2–4 min/epoch → ~1.5–4 h |
| C2 HMC-QU, phase 1 | frozen frame encoder, features cached once | a few minutes for all 5 folds |
| C2 HMC-QU, full fine-tune | 16 frames at 224 px, 5-fold CV | ~10–20 s/epoch → ~30–90 min |
| C3 EchoNet-Dynamic | 32 frames at 112 px (native size), 7,460 training clips | ~4–7 min/epoch → ~2.5–5 h |
| C3 CAMUS | 500 patients × 2 views, one heartbeat | ~15–45 min |

**VRAM (16 GB):**
- Full fine-tuning at 224 px with 16-frame clips is tight: use batch 1–2 clips with mixed precision, or
  112 px.
- 112 px fits comfortably.

## Cardiac ultrasound track (active)

### Datasets
Figures come from the dataset papers and pages; verify on download. Access terms are research /
non-commercial use unless stated.

**Still images**

| Dataset | Labels | Size | Access | Links | Used in |
|---|---|---|---|---|---|
| CACTUS (2025) | 6 views (A4C, PLAX, PSAX-aortic, PSAX-mitral, subcostal, random) + quality grade 1–10 | 37,736 images from a **phantom** (CAE Blue Phantom) | Open | [data (FRDR)](https://www.frdr-dfdr.ca/repo/dataset/86beb91e-c0ba-496b-ad38-732b67e3d5f6) · [paper](https://arxiv.org/abs/2503.05604) | C1 |
| TMED-2 (Tufts) | View (PLAX/PSAX/A2C/A4C/other) + aortic stenosis | 599 studies, 577 patients, 5,261 labeled images | Request (email the TMED team) | [dataset](https://tmed.cs.tufts.edu/tmed_v2.html) · [access](https://tmed.cs.tufts.edu/data_access.html) · [license](https://tmed.cs.tufts.edu/data_license.html) | C1 real-image test |

**Video**

| Dataset | Views | Labels | Size | Access | Links | Used in |
|---|---|---|---|---|---|---|
| HMC-QU | A4C, A2C (25 fps) | Heart attack (MI) yes/no; wall-segment motion; LV masks for 109 A4C clips | 162 patients (93 MI): 162 A4C + 160 A2C videos | Kaggle | [Kaggle](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset) · [code](https://github.com/degerliaysen/MultiEchoAI) | C2 |
| EchoNet-Dynamic | A4C, 112×112 | Ejection fraction; LV tracings | 10,030 videos (unique patients) | Stanford AIMI, free research agreement | [Stanford AIMI](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound) | C3 |
| CAMUS | A2C + A4C, one heartbeat | Ejection fraction; LV/myocardium/LA masks; image quality | 500 patients | Free registration | [project](https://www.creatis.insa-lyon.fr/Challenge/camus/) · [database](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8) · [Kaggle mirror (unofficial)](https://www.kaggle.com/datasets/toygarr/camus-dataset) | C3 |
| CardiacNet (ECCV 2024) | echo videos | Pulmonary arterial hypertension (catheter-confirmed); atrial septal defect | 496 PAH + 231 ASD cases | GitHub (check terms) | [code + data](https://github.com/xmed-lab/CardiacNet) · [paper](https://arxiv.org/abs/2410.20769) | optional: right-heart strain |
| MIMIC-IV-ECHO | All views (DICOM videos) | Only via linked reports and measurements | ~525k files, 7,243 studies, 4,579 patients | PhysioNet credentialed (training course + agreement) | [dataset](https://physionet.org/content/mimic-iv-echo/1.0/) · [A4C PoCUS-style LV subset (1,064 videos)](https://physionet.org/content/mimic-iv-echo-ext-lvvol-a4c/1.0.0/) · [ECHOVIEW view labels](https://physionet.org/content/echoview/0.1/) | C4 candidate |
| EchoNet-Pediatric | A4C + PSAX | Ejection fraction; tracings | 7,643 videos | Stanford AIMI | [Stanford AIMI](https://aimi.stanford.edu/datasets/echonet-pediatric) · [project](https://echonet.github.io/pediatric/) | not planned (children) |
| EchoNet-LVH | PLAX | Wall thickness, LV dimensions | 12,000 videos | Stanford AIMI | [project](https://echonet.github.io/lvh/) | not planned (not an emergency) |

[CardioBench](https://github.com/BioMedIA-MBZUAI/CardioBench) collects download instructions for most of these.

**Not found as public labeled data:** cardiac standstill (arrest) and pericardial effusion.

### C1: View and quality guidance (CACTUS, still images)
- [ ] Download CACTUS and confirm:
  - folder structure, class counts, how quality grades are stored, and the license
  - whether images come from continuous sweeps. If they do, neighbouring frames are near-duplicates, so
    split by sweep/session, not by image.
- [ ] Notebook `cardiac-view-quality.ipynb` built from the fixed FER notebook:
  - 6-class view output (softmax, like FER)
  - a quality output: grade 1–10, or usable / not usable (decide after seeing the grade distribution)
  - CLAHE, **no horizontal flips**, oversampling on train only, focal loss
- [ ] Metrics:
  - views: per-view precision/recall, confusion matrix, macro-F1 with bootstrap 95% CIs
  - quality: error vs expert grade, or AUC for usable / not usable
- [ ] Real-image check on TMED-2, mapping A4C → A4C, PLAX → PLAX, PSAX → both PSAX classes. Report the
      phantom-to-real gap.

### C2: Heart-attack (MI) detection from video (HMC-QU)
- [ ] **Video version of the architecture.** Prototype it and smoke-test locally on tiny clips before any
      real training.
  - Each frame goes through the existing frame encoder (VGG16 + CBAM, ViT, capsule fusion), giving one
    256-d vector per frame.
  - The Transformer Refinement Block then attends across T frames with time positions, instead of 4 tokens
    from one image. All custom parts are kept.
- [ ] **Clips:** 16–32 frames covering one heartbeat, at 224 px (or 112 px). Pick frames using the LV masks
      where available, otherwise by fixed sampling.
- [ ] **Training in two phases:**
  1. Frozen frame encoder: compute per-frame features once and train only the temporal part. This is
     fast, and less prone to overfitting on 162 patients.
  2. End-to-end fine-tuning with small batches and mixed precision.
- [ ] **Validation:** patient-wise 5-fold CV. Use A4C and A2C separately, then combined per patient.
- [ ] **Metrics:** patient-level sensitivity, specificity and AUC with 95% CIs; compare with published
      HMC-QU results.
- [ ] **Comparisons:**
  - starting from C1 weights vs ImageNet
  - a lighter purpose-built video network (e.g. R(2+1)D, as used by EchoNet), to show the architecture earns
    its cost

### C3: Pumping function from video (EchoNet-Dynamic, CAMUS)
- [ ] Ejection-fraction regression plus reduced-EF classification (EF < 40%), using the same video
      architecture as C2.
- [ ] EchoNet-Dynamic at its native 112 px with the official train/val/test split. Then CAMUS as a second,
      independent test set.

### C4: Arrest findings (needs data)
- [ ] Cardiac standstill, pericardial effusion and RV dilation have no public labeled datasets that we found.
- [ ] Options:
  - Mine effusion and function labels from MIMIC-IV-ECHO reports (credentialed access).
  - Find a clinical or maritime-telemedicine partner.

## Parked: abdominal ultrasound (2026-09-30)
Kept in case the trauma track resumes. Details are in the archive section of `US_PROGRESS_LOG.md`.
- **Trauma track:** organ recognition on the Vitale AUS/RUS dataset. `us-organ-recognition.ipynb` is
  written and smoke-tested but has not run on Kaggle. It was followed by a synthetic free-fluid
  proof-of-concept and real FAST data (not started).
- **Pathology track:** gallbladder / fatty-liver candidates (UIdataGB, GBCU, Byra). No work started.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-09-29 | Fix all issues in the FER pipeline before porting | Leakage and bugs inflate the FER results; the port would inherit them |
| 2026-09-29 | Ultrasound task = pathology classification on a labeled dataset (option B) | Keeps the clinical goal; the pipeline carries over nearly 1:1 (superseded 2026-09-30) |
| 2026-09-29 | Vitale US simulation & segmentation dataset not used for pathology | Healthy subjects only; labels are organ masks, not diagnoses |
| 2026-09-29 | FER keeps horizontal flips; ultrasound drops them | Faces are roughly symmetric; anatomy is not |
| 2026-09-29 | Abdominal trauma track: organ recognition, then synthetic fluid | The dataset has no trauma cases (parked 2026-09-30) |
| 2026-09-29 | Separate `US_PROGRESS_LOG.md` for the ultrasound study | Keeps the FER log clean |
| 2026-09-30 | **Switch the ultrasound study to cardiac** | Cardiovascular events are the leading cause of death at sea; echo is the key point-of-care heart exam |
| 2026-09-30 | Order: view + quality guidance (CACTUS) first, then MI detection (HMC-QU) | Onboard operators aren't sonographers, so a usable view comes first; CACTUS is a 1:1 port of the architecture |
| 2026-09-30 | Abdominal work parked, not deleted | Notebook and notes kept in case the trauma track resumes |
| 2026-09-30 | MI input = 3 heartbeat frames as the 3 image channels | Gives the 2D architecture motion information without changing it (superseded the same day by video) |
| 2026-09-30 | **Video for motion-based stages (C2, C3); still images for C1** | Wall motion and pumping function are judged from motion; view and quality are judged from a frame |
| 2026-09-30 | Video architecture = existing frame encoder + Transformer Refinement Block across frames | Keeps every custom component; the refinement block is already a transformer with positional embeddings |
| 2026-09-30 | Train on the desktop RTX 4080 Super (WSL2), later | No 12-hour limit; 16 GB is enough at 112 px or with mixed precision |
| 2026-09-30 | EchoNet-Dynamic at its native 112 px | Upscaling to 224 px adds ~4× compute and no information |
| 2026-10-02 | FER: add facial-landmark geometry as a third capsule; skip the cheaper per-class bias adjustment | User's choice: test the geometry idea directly |
| 2026-10-02 | Landmarks from MediaPipe Face Landmarker 1.0.1, run in its own process and package folder | It needs numpy 2.4; TensorFlow 2.19 needs numpy below 2.2 |
| 2026-10-02 | Measure each face on the image and its mirror image, and average | Detector jitter on 48×48 faces made asymmetry measurements unreliable (correlation 0.14–0.72) |
| 2026-10-02 | Each ablation arm keeps its own checkpoint folder; the baseline may reuse the finished run | The arms must never resume from each other; old checkpoints load exactly into the new code (superseded the same day: checkpoints removed) |
| 2026-10-02 | Remove checkpoints and Kaggle session-limit handling from `exp-tuning.ipynb`; add a requirements section | Training moves to the desktop with no time limit; a finished run's final weights are reused with `PRETRAINED_WEIGHTS` instead |

## Risks

| Risk | Mitigation |
|---|---|
| Kaggle 12 h session limit | Train on the desktop, which has no limit. Resume support was removed on 2026-10-02, so a Kaggle run must now finish within 12 h |
| `transformers` v5 dropped TensorFlow (`TFViTModel`) | Pin `transformers<5`; run on tf-keras |
| TensorFlow has no GPU support on native Windows after 2.10 | Train under WSL2 (E1) |
| Small medical datasets vs a ~100M-parameter model → overfitting | Two-phase unfreezing, frozen-feature phase for video, strong regularisation, patient-wise CV, report CIs; watch phase-2 validation loss |
| Video cost: every frame runs VGG16 + ViT; 16 GB VRAM | 16–32 frames, 112 px where possible, mixed precision, cached frame features in phase 1, batch 1–2 clips |
| Video decoding on the CPU slows training | Decode clips once into arrays (e.g. `.npy`) before training |
| The heavy architecture may not beat lighter video networks | Compare with R(2+1)D in C2/C3 and report the cost-benefit honestly |
| Class imbalance | Oversampling on train only + focal loss; report per-class sensitivity |
| Phantom-to-real gap: CACTUS images come from a phantom | Test on real TMED-2 images; report the gap as a result |
| Near-duplicate frames from continuous sweeps leak across splits | Split by sweep/session/patient; md5 and visual checks |
| HMC-QU is small (162 patients) | Patient-wise 5-fold CV, bootstrap CIs, no heavy tuning on test folds |
| Access lead times: TMED-2 by request, MIMIC credentialing | Request early (see `US_PROGRESS_LOG.md`) |
| Research-only dataset licences; the model is not a medical device | Research results only; clinical use would need prospective validation and regulatory approval. Don't redistribute credentialed data. |
| At sea: handheld probes, poorer images, low-bandwidth telemedicine | Quality grading in C1; later, test on handheld-device images |
