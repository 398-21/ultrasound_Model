# Progress Log

Newest entry first. Each Kaggle run gets its own entry (template at the bottom).

---

## 2026-10-03 (night): `TRAIN_ON` setting for training on one dataset. Status: RAF-DB-only run ready

### Why
To test whether a model trained on RAF-DB (cleaner labels, 100 px faces) carries over to FER2013 (48 px, noisier
labels), and the other way round.

### What changed in `exp-tuning.ipynb`
- **Settings:**
  - `TRAIN_ON = "both"` (default), `"RAF-DB"` or `"FER2013"`, for the baseline arm only (asserted).
  - `image_source()` and `in_training_data()` moved here.
  - `VARIANT` gets `_train-<dataset>`, so the weights and predictions of a single-dataset run don't overwrite
    the main runs.
- **Preprocessing:**
  - Only the chosen dataset's training images.
  - `preprocess_dataset` now empties its output folder first. Before, images from an earlier run stayed in
    `~/fer_scratch/preprocessed`, so a narrower run would silently have trained on the old images too.
- **Near-duplicates:**
  - Checked against the training images in use. A single-dataset run gets its own list,
    `test_near_duplicates_train-<dataset>.csv`.
  - The dataset not trained on is labelled "(not trained on)" in the per-subset reports.

- **Geometry arm allowed too** (added the same night):
  - The training landmark table is named after `TRAIN_ON` (`landmarks_train-RAF-DB.csv.gz`); the test table is
    shared.
  - The ablation compares `baseline<suffix>` with `geometry<suffix>` for the current `TRAIN_ON`.
  - Why it's worth running: landmarks barely degrade at 48 px while the image model does, so geometry might help
    when training on sharp RAF-DB and testing on FER2013.

### Checks
- **Geometry dry run** (`TRAIN_ON = "RAF-DB"`, `USE_GEOMETRY = True`; every cell, training skipped):
  - Landmark table built: 12,271 rows, face found in 95.8%.
  - GeometrySequence has a vector for every training, validation and test image (0 missing).
  - The ablation waits for `predictions_{baseline,geometry}_train-RAF-DB_test.npz`.
- **Dry run with `TRAIN_ON = "RAF-DB"`** (environment through oversampling, no training):
  - 12,271 training images, all RAF-DB.
  - Split 11,045 / 1,226.
  - Oversampled to 25,346, so an epoch is about 43% of the main runs'.
  - Test set unchanged (7,177 FER2013 + 3,068 RAF-DB).
  - Only 3 test images (RAF-DB) have a copy in RAF-DB's training set, so all 7,177 FER2013 test images count
    as cross-dataset.
- **Default (`"both"`):** the settings, near-duplicate and ablation cells reproduce the evening entry's tables
  exactly.

---

## 2026-10-03 (evening): Near-duplicates left out of the test results; fake cross-dataset test removed

### Why
- **The folders are split by file, not by picture.** FER2013 was collected by web search and contains some
  pictures more than once, so copies ended up in both its training and test folders.
- **emotion-detection-fer is FER2013 again.** It's a Kaggle re-upload of FER2013 (PNG instead of JPEG, renamed
  class folders), so its test folder is FER2013's test split, which `combined/test` already contains.

### Near-duplicates in `combined/test`
- **Method:** pixel RMSE on 48×48 grayscale against every training image and its mirror image. The threshold is
  6 grey levels; re-encoded copies differ by about 2, different pictures by 20 or more.
- **604 of 10,245 test images (5.9%) have a near-identical training image:**
  - FER2013: 601 of 7,177 (8.4%)
  - RAF-DB: 3 of 3,068
  - none mirrored; 31 have a different label from their training copy
- The list is in `outputs/test_near_duplicates.csv`.

### Corrected results (the two saved runs, recomputed; no retraining)
| Test images | Baseline acc / macro-F1 | Geometry acc / macro-F1 | Difference acc / macro-F1 [95% CI] |
|---|---|---|---|
| **Without near-duplicates (9,641), headline** | 0.717 / 0.651 | 0.723 / 0.656 | +0.006 [−0.000, +0.013] / +0.005 [−0.006, +0.016] |
| All (10,245), as reported before | 0.723 / 0.666 | 0.728 / 0.672 | +0.006 [−0.001, +0.013] / +0.006 [−0.005, +0.017] |
| FER2013, without near-duplicates (6,576) | 0.671 / 0.634 | 0.676 / 0.630 | +0.005 [−0.003, +0.013] / −0.005 [−0.020, +0.010] |
| RAF-DB, without near-duplicates (3,065) | 0.815 / 0.721 | 0.824 / 0.744 | +0.009 [−0.004, +0.022] / **+0.023 [+0.002, +0.046]** |

- **The duplicates inflated the old numbers slightly:** the baseline loses 0.6 points of accuracy and 1.5 of
  macro-F1 once they're removed.
- **The conclusion holds:** no clear effect of geometry overall.
- **RAF-DB macro-F1 +2.3** is the only interval above zero. It comes from one run, and it's one of several
  subsets, so treat it as a lead, not a finding. It's plausible: RAF-DB's faces are larger and cleaner, so their
  landmarks are more reliable than on 48×48 FER2013.
- **The old "cross-dataset" numbers (0.683 / 0.686) are withdrawn.** They were the FER2013 test images again.

### Why the geometry adds little: what the measurements tell on their own
Method:
- Classifiers on the notebook's own measurements (`raw_geometry` on run 2's landmark tables).
- Trained on the training split, scored on the 9,641 test images without near-duplicates.

| Features | Logistic regression acc / macro-F1 | Gradient boosting acc / macro-F1 |
|---|---|---|
| All 34 measurements | 0.515 / 0.439 | 0.571 / 0.501 (small neural network: 0.596 / 0.503) |
| Mouth only (15) | 0.445 / 0.355 | 0.484 / 0.405 |
| Eyes, brows, nose, pose (19) | 0.443 / 0.380 | 0.500 / 0.435 |
| Mouth point positions: all 40 lip points MediaPipe gives, aligned | 0.467 / 0.381 | 0.463 / 0.377 (network: 0.522 / 0.380) |
| All 61 point positions, aligned | 0.515 / 0.441 | 0.553 / 0.476 (network: 0.584 / 0.500) |
| All 61 points + the 34 measurements | 0.529 / 0.455 | 0.570 / 0.499 (network: 0.584 / 0.483) |
| *Image model (baseline), for reference* | | *0.717 / 0.651* |

- **Point positions don't beat the 34 measurements.** The raw positions (eyes level, centred, scaled by eye
  distance, from the image's own detection) tell no more than the summary measurements, and slightly less for
  the mouth.
- **No landmark representation tried here gets past about 0.60 accuracy.**

- **Recall with mouth measurements alone** (gradient boosting): happy 0.73, surprise 0.54, neutral 0.48,
  disgust 0.38, angry 0.36, fear 0.25, sad 0.25.
- **Context:** chance is 0.143; always answering "happy" scores 0.301.
- **The branch isn't broken:**
  - The measurements carry real signal (four times chance).
  - They raised val_acc by 3 points while the ViT was frozen.
- **But they hold much less than the image model already sees.** The mouth separates happy and surprise.
  Fear/surprise, sad/neutral/angry and disgust/angry share mouth shapes and differ mainly in the eyes and brows.
- **Some measurements are noisy.** The two detections of the same training face (image and mirror image)
  correlate only 0.24–0.72 for 8 measurements, mostly the asymmetries (eye_asymmetry 0.24, brow_asymmetry 0.39,
  corner_asymmetry 0.43, mouth_shift 0.58).
- This isn't mainly a resolution effect: shrinking RAF-DB to 48 px doesn't make it worse (see the next section).

### Does resolution matter? RAF-DB test images shrunk to FER2013's size
Method:
- The trained baseline on the 3,065 RAF-DB test images (without near-duplicates), each shrunk from 100 px
  (INTER_AREA).
- Then the notebook's preprocessing: grayscale, resize to 224, CLAHE.
- The 100 px row reproduces the earlier RAF-DB result exactly.

| Size | Image model acc / macro-F1 | Disgust recall | Landmarks: face found | Measurements alone (network) acc / macro-F1 |
|---|---|---|---|---|
| 100 px (original) | 0.815 / 0.721 | 0.44 | 96.5% | 0.663 / 0.543 |
| 64 px | 0.768 / 0.656 | 0.25 | | |
| 48 px (FER2013's size) | 0.723 / 0.590 | 0.08 | 97.8% | 0.660 / 0.517 |
| 32 px | 0.675 / 0.535 | 0.03 | | |

- **The image model needs the resolution.** At FER2013's 48 px, RAF-DB loses 9 points of accuracy and 13 of
  macro-F1, and disgust is almost never recognised.
- **The gap between the two datasets:** FER2013 0.671 vs RAF-DB 0.815. Resolution explains about 9 of those
  14 points; the rest is labels and content.
- **The landmarks don't need it.** At 48 px the detector finds faces as often, its two detections agree as well
  (median correlation 0.90 vs 0.89), and the measurements classify almost as well.
- So the geometry's weakness is what it represents (coarse shape), not the pixel count. This corrects the
  earlier guess that detector noise on small faces was a main cause.
- **Caveat:** the model was trained on 100 px RAF-DB images, so part of the drop may come from the change in
  image style, not only the lost detail.
- **Enlarging images adds no detail.** The pipeline already upscales everything to 224 px; FER2013's detail
  was never recorded.

### What changed in `exp-tuning.ipynb` (25 → 26 cells, outputs cleared)
- **New cell after the train/val split:** finds the near-duplicates (about 5 min the first time, then read from
  the CSV). It also provides `test_subsets` and `report_test_subsets`.
- **Main run:** after the standard evaluation, it prints accuracy and macro-F1 for all test images, without
  near-duplicates, and for FER2013 and RAF-DB separately.
- **Ablation cell:**
  - The headline table is on the near-duplicate-free test set, with per-class rows and a plot.
  - Short tables follow for all images, FER2013 and RAF-DB.
  - The old `cross` comparison is gone.
- **Cross-dataset test is now optional:** `CROSS_DATASET = None` in the settings cell.
  - Point it at a folder under `data/` with `test/<7 classes>` from a different dataset, e.g. KDEF or ExpW.
  - `check_independent` stops the run if more than 1% of that folder's images appear in the combined data. On
    emotion-detection-fer it refuses, finding 7,177 of 7,178 images shared.
  - Predictions and landmarks are named `cross-<name>`, so the old emotion-detection-fer files in `outputs/`
    aren't picked up.
- **Executed copy:** run 2's notebook, with all its outputs, is kept as
  `outputs/exp-tuning_2026-10-03_geometry-run.ipynb`.
- **Checked:**
  - The settings, near-duplicate, cross-dataset (skip path) and ablation cells ran headless on the saved runs.
  - The per-run report was checked on run 1's predictions.
  - The training-cell change (one call) was not run end to end.

---

## 2026-10-03 (later): Face-geometry ablation, both arms on the desktop. Result: no clear improvement

### Run 1: baseline (the Kaggle run's final weights, evaluation only, about 20 min)
- **Reproduced the Kaggle report exactly:** test accuracy 0.7227, macro-F1 0.666, macro ROC AUC 0.9328, and the
  same per-class precision and recall.
- **New numbers:**
  - TTA accuracy 0.7077, so TTA *lowers* accuracy by 1.5 points.
  - Cross-dataset (emotion-detection-fer test, 7,178 images): accuracy 0.6831, macro-F1 0.66.
- Byte-identical images (informational): train–val 195, train–test 481, combined train–cross 0.

### Run 2: geometry, trained on the RTX 4080 SUPER (about 8.6 h in total)
- **Landmarks:** face found in 97.1% of training images, 97.0% of test images and 98.8% of cross-dataset images.
  Detection took a few minutes.
- **Phase 1:** all 25 epochs, about 640 s each (175 ms/step). val_loss went from 0.1067 to its best of 0.0791
  (epoch 23); val_acc 0.650 at the end.
- **Phase 2:** early-stopped after 12 epochs, about 1,155 s each (316 ms/step). Best val_loss 0.0643 at epoch 6
  (val_acc 0.722).
- **Training time:** 8.3 h (13:07–21:25).
- **Against the Kaggle baseline's training:**
  - End of phase 1: val_acc 0.650 vs 0.618, val_loss 0.080 vs 0.090. The geometry clearly helps while the ViT
    is frozen.
  - Best phase-2 val_loss: 0.0643 vs 0.0640. After end-to-end fine-tuning, the advantage is gone.

### Ablation (paired bootstrap, 2,000 resamples)
| Set | Metric | Baseline | Geometry | Difference [95% CI] |
|---|---|---|---|---|
| test (10,245) | accuracy | 0.723 | 0.728 | +0.006 [−0.001, +0.013] |
| test | macro-F1 | 0.666 | 0.672 | +0.006 [−0.005, +0.017] |
| cross (7,178) | accuracy | 0.683 | 0.686 | +0.003 [−0.004, +0.011] |
| cross | macro-F1 | 0.662 | 0.660 | −0.002 [−0.014, +0.010] |

- **Per class:** no clear change on the test set. On the cross-dataset test, neutral F1 is +0.016 [0.000, 0.031],
  borderline, and it's one of 18 comparisons.
- **The weak classes were not helped:**
  - Fear recall fell from 0.368 to 0.352 (test) and from 0.390 to 0.366 (cross).
  - Disgust recall went from 0.480 to 0.472 (test).
- **Run-to-run noise is larger than the effect:**
  - The arms disagree on 15.0% of test images: 553 are right only for the baseline, 612 only for geometry
    (McNemar p = 0.089).
  - Averaging the two arms' probabilities gives 0.736 accuracy and 0.680 macro-F1, more than geometry adds.
- **Caveats:** one run per arm, and the two arms ran on different machines (Kaggle and the desktop).

### What the test set contains, and accuracy per source
`combined` is FER2013 plus RAF-DB, already merged in the Kaggle dataset (RAF-DB was never added separately):
- **RAF-DB part:** the basic set, files named `train_#####_aligned` / `test_####_aligned`.
  - All 14 per-class counts match RAF-DB's official split exactly (e.g. 4,772 happy training images, 74 fear
    test images).
  - The images are RAF-DB's 100×100 colour aligned faces. The pipeline reads every image as grayscale, so their
    colour is dropped.
- **FER2013 part:** 48×48 grayscale.
- **For the paper:**
  - Describe the training data as FER2013 + RAF-DB, and cite both (RAF-DB: Li, Deng & Du, CVPR 2017).
  - RAF-DB is normally obtained from its authors under an academic-use agreement, so check that the Kaggle copy
    is covered.

| Source | Train | Test | Baseline acc | Geometry acc | Both averaged |
|---|---|---|---|---|---|
| FER2013 PrivateTest | | 3,589 | 0.683 | 0.693 | 0.696 |
| FER2013 PublicTest | | 3,588 | 0.683 | 0.682 | 0.689 |
| FER2013 (Training) | 28,698 | | | | |
| RAF-DB | 12,271 | 3,068 | 0.816 | 0.824 | 0.838 |

- **Published results, for context:**
  - FER2013: about 73% for a single model and about 75–77% for ensembles. People agree with its labels only
    about 65% of the time (Goodfellow et al., 2013).
  - RAF-DB: about 92% (e.g. POSTER++), with face-pretrained backbones on colour images.
- **Ceiling:** even matching the best published results on both parts would give about 0.7 × 0.76 + 0.3 × 0.92 ≈
  0.81 on this test set.
- **The emotion-detection-fer "cross-dataset" test is not independent.**
  - Every one of its 7,178 images matches an image in the FER2013 part of `combined/test`. The closest matches
    differ by 1.5–1.9 grey levels RMSE, which is JPEG re-encoding noise.
  - Its accuracy (0.683) is therefore the FER2013-part accuracy again, not a generalisation test. Don't report
    it as cross-dataset.
- **Train/test near-duplicates:** about 8% of FER2013 test images have a near-identical copy (RMSE < 6) in the
  training set, a known FER2013 issue. This slightly inflates test accuracy; report it or remove those images.

### Interpretation
The 34 measurements help while the image backbone is partly frozen. End-to-end fine-tuning of VGG16 and the ViT
recovers the same information, so after phase 2 the geometry is redundant.

### Next (decision pending)
- The roadmap rule applies (difference under 2 points → 3 seeds per arm). That means about 5 more runs of
  8.5 h on the desktop.
- The alternative is to report this as a single-run null result.

---

## 2026-10-03: Desktop environment (WSL2) set up; three blockers fixed. Status: ready to run both arms

How to use the desktop: `DESKTOP_TRAINING.md`.

### Environment
| | |
|---|---|
| Machine | RTX 4080 SUPER 16 GB (Windows driver 610.62), 16 CPU threads, 31 GB RAM (WSL gets about 15 GB) |
| Linux | WSL 3.0.1, Ubuntu 24.04.5, Python 3.12.3 |
| venv `~/venvs/fer` (kernel "Python (fer)") | TensorFlow 2.19.1 (CUDA 12.5.1, cuDNN 9.3), tf-keras 2.19.0, transformers 4.54.1, safetensors 0.8.0, numpy 2.1.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, OpenCV 5.0.0; `pip check` clean |

### Checks
- **GPU:**
  - TensorFlow created `/GPU:0` on the RTX 4080 SUPER (compute capability 8.9) with 13,495 MB and loaded
    cuDNN 9.3.
  - A small conv net trained on the GPU.
- **Notebook cells,** run headless with the venv from the project folder:
  - install cell: OK
  - environment cell: no GPU warning
  - settings cell: all three datasets found, with 7 classes each (40,969 / 10,245 / 7,178 images)
- **Pretrained weights:**
  - VGG16 ImageNet: 14.7M parameters.
  - ViT-B/16 in21k: 86.4M parameters, "All PyTorch model weights were used". The patch-embedding and layer-0
    query tensors are identical to `model.safetensors`.
- **MediaPipe 1.0.1:** installed with the notebook's own command into `~/fer_scratch/landmark_pkgs`. The
  notebook's `extract_landmarks.py` found faces in 40/40 CLAHE-preprocessed test images and in their mirror
  images.
- **Git:** `git status` matches between Windows and WSL (`core.autocrlf=true` in the repo's local config).
- **Kaggle baseline output** (`data/baseline_run/`):
  - It's the finished corrected-baseline run: 25 + 11 epochs, best val_loss 0.0640, final weights
    `vgg16_cbam_vit_incremental_unfreeze_improved`.
  - Its train/val split is the same as the desktop's (sorted file names, seeded shuffle), and the test set is
    identical.

### Fixed (found by these checks)
1. **transformers 4.55+ can't load the ViT into TensorFlow.**
   - The cause: the TF loader iterates over a `safe_open` handle (`for key in pt_state_dict`). No safetensors
     version supports that (checked 0.4.5, 0.5.3, 0.6.2, 0.7.0 and 0.8.0). 4.54.1 still loops over `.keys()`.
   - pip had picked 4.57.6, so the first run would have crashed while building the model. The earlier local
     tests used a tiny random ViT and never loaded the pretrained one.
   - Fix: `transformers>=4.40,<4.55` in the requirements and in the install cell (desktop and Kaggle lists). The
     environment cell now asserts it, with a clear message.
2. **MediaPipe needs `libGLESv2.so.2`** (Ubuntu package `libgles2`), which WSL's Ubuntu lacks. Without it, the
   face landmarker can't start. It's installed now, and the notebook's setup commands include
   `libgl1 libegl1 libgles2` and name Ubuntu 24.04.
3. **The landmark script looped forever when its detector couldn't start.**
   - A failing `Pool` initializer makes `multiprocessing` restart its workers endlessly. In the test that meant
     about 470,000 processes and 750 MB of tracebacks in 10 minutes.
   - Fix: the worker records the error and `detect()` raises it. The script now stops at the first image with
     `MediaPipe's face landmarker could not start: …`; with a broken model file it exits in under a second.

### Timing
- On Kaggle, the corrected baseline's last phase-2 epoch took 3,700 s. (`epoch_seconds` in `latest.json` is
  the last epoch's duration.)
- Scaled to the RTX 4080 SUPER, that suggests about 4–10 min per phase-1 epoch and 10–20 min per phase-2 epoch,
  so 4–8 h per training run. The compute-based 2.5–5.5 h looks optimistic. The first epochs will give the real
  numbers.

### Next
- Run 1 (baseline, evaluation only from the Kaggle weights), then run 2 (geometry). The steps are in
  `DESKTOP_TRAINING.md`.

---

## 2026-10-02 (later): Desktop-only training: checkpoints and session limit removed; requirements at the top

### Why
Training moves to the desktop (RTX 4080 Super), which has no 12-hour limit, so the resume machinery was dead weight.

### What changed in `exp-tuning.ipynb` (26 → 25 cells)
- **New first cell (markdown):** requirements.
  - OS: Linux, or Windows 11 with WSL2. Python 3.10–3.12. NVIDIA GPU. About 15 GB disk. Internet on the first run.
  - A pip list and the terminal setup commands.
- **Install cell** (replaces the old `!pip install transformers` cell):
  - Checks the Python version and installs any missing requirements.
  - On Kaggle it only pins `transformers<5`.
- **Removed:**
  - The session clock (`SESSION_START`), the 11-hour budget, the "Training across Kaggle sessions" notes and
    the whole checkpointing cell (`SessionCheckpoint`, resumable callbacks, `latest.json`, per-arm checkpoint
    folders).
  - The "training not finished" branches in the main run and the cross-dataset test.
- **Training cell:**
  - The plain two-phase schedule, with standard `EarlyStopping` and `ReduceLROnPlateau` (same settings as before).
  - It saves the final weights once, as `outputs/vgg16_cbam_vit_<arm>`.
- **New setting `PRETRAINED_WEIGHTS`:**
  - Points at a finished run's final weights (path without `.index`); training is then skipped and the model
    is only evaluated.
  - This is how the baseline arm reuses the Kaggle run: put `vgg16_cbam_vit_incremental_unfreeze_improved.index`
    and `.data-00000-of-00001` in `data/baseline_run/`.
  - Training curves are skipped in that case.
- **Environment cell:** warns when no GPU is found.

### Checks
- **Requirement list:** resolves on Linux x86-64 for Python 3.10 and 3.12, checked with `uv` for that platform.
  Resolved versions: TensorFlow 2.19.1, tf-keras 2.19.0, transformers 4.57.6, numpy 2.1.3, CUDA 12.5 / cuDNN 9.3.
  - A plain `pip --platform` dry run from Windows reports a false conflict: it evaluates TensorFlow's
    Windows-only dependency marker against the host.
- **Weight compatibility:** final weights saved the way the 2026-09-30 notebook saved them (after phase-2
  training, optimizer state included) load through `PRETRAINED_WEIGHTS` with all 113 weight tensors identical.
  Predict and evaluate work on the loaded model.
- **End to end:** the whole notebook ran three times on the small real-face test set, with a tiny ViT and VGG16
  without ImageNet weights, on CPU.
  1. The baseline trained both phases and saved its final weights.
  2. The geometry arm trained with 2 inputs and 3 capsules and produced both ablation tables.
  3. The baseline from `PRETRAINED_WEIGHTS` skipped training, evaluated and recomputed the ablation.
  - No checkpoint folders are created any more.

---

## 2026-10-02: Face-geometry branch (third capsule) and ablation added. Status: ready to train

Paper-style write-up (method, setup, checks, time estimates; results table to fill):
[Geometry-Aware Capsule Fusion for Facial Expression Recognition](https://claude.ai/code/artifact/f3627e41-a21a-4b65-9685-6fd2c76d47ae)

### What changed in `exp-tuning.ipynb` (19 → 26 cells)
| Part | Change |
|---|---|
| Settings cell (new) | `USE_GEOMETRY` picks the arm (`baseline` / `geometry`); Kaggle vs desktop detected automatically; all paths in one place. Desktop defaults: `data/` and `outputs/` next to the notebook, scratch in `~/fer_scratch`, editable in one block; prints the classes and file counts found |
| Landmark script (new, `%%writefile extract_landmarks.py`) | MediaPipe Face Landmarker on every preprocessed image and on its mirror image; black border 50% → 25% → none; one CSV row per image with found flags and 61 landmark positions |
| Face geometry cell (new) | Installs MediaPipe 1.0.1 into its own folder (it needs numpy 2.4, TensorFlow 2.19 needs < 2.2) and runs the script as a separate process; 34 scale-free, mirror-symmetric measurements, the two detections averaged; standardised on the training split, clipped to ±5, plus a found flag; prints coverage per class, detection agreement and class means |
| Capsule fusion block | Takes any number of branches (`num_inputs`); with 2 it is the original block with the same layer names |
| MixUp | Mixes every model input with the same λ and pairs |
| Model | Optional `geometry_input` → Dense 128 → Dropout 0.2 → Dense 256 → third capsule |
| Data | `GeometrySequence` wraps the directory iterators so each batch carries images + geometry; augmented copies use their source image's vector |
| Checkpoints | One folder per arm (`checkpoints/`, `checkpoints_geometry/`); `latest.json` found anywhere under the inputs, the other arm's skipped; no time budget on the desktop |
| Evaluation | Per-image test and cross-dataset predictions saved as `predictions_<arm>_{test,cross}.npz` |
| Ablation cell (new) | Per-class F1, macro-F1 and accuracy for both arms, the difference with a 95% paired bootstrap CI (2,000 resamples), macro-F1 split by face found / not found, and a plot |

### Checks (local: TensorFlow 2.19.1 + tf-keras 2.19.0 + transformers 4.57.6, CPU)
- **Landmark script and measurements on 400 Olivetti faces at FER resolution** (48×48, then CLAHE at 224):
  - 99.4% of images detected: every face, none of 5 noise images.
  - Averaged measurements exactly mirror-symmetric, and unchanged by rotation, scaling and shifting.
  - A synthetic smile and frown moved `corner_lift` by +0.10 and −0.10 eye distances.
- **Detection agreement (image vs mirror image):** 29 of 34 measurements agreed closely (correlation 0.86–0.98).
  `eye_asymmetry` 0.14, `brow_asymmetry` 0.37, `corner_asymmetry` 0.51, `mouth_shift` 0.66 and `brow_gap` 0.72
  did not, which is why the two detections are averaged.
- **Checkpoint compatibility:** a checkpoint saved by the previous notebook code loads into the new two-branch
  model with all 113 weight tensors identical. The finished corrected-baseline run can therefore serve as the
  baseline arm.
- **End to end:** the whole notebook ran three times on a small FER-shaped dataset of real faces plus noise
  images, with a tiny ViT and VGG16 without ImageNet weights.
  1. The baseline trained both phases and saved its predictions.
  2. The geometry arm found landmarks for 95–100% of each set, built 2 inputs and 3 capsules, trained,
     skipped the baseline's checkpoint and produced both ablation tables.
  3. A baseline rerun loaded its finished checkpoint and only evaluated.
- **Not verified:** full-size GPU training, landmark coverage on the real FER images, the result itself.

### How to run (desktop, RTX 4080 Super)
1. **Baseline arm:** `USE_GEOMETRY = False`. To skip retraining, put the finished corrected-baseline
   `checkpoints/` folder anywhere under `./data`; the run then only evaluates (about 30 min). Otherwise
   retraining takes 2.5–5.5 h.
2. **Geometry arm:** `USE_GEOMETRY = True`. The first run needs internet: MediaPipe (about 330 MB) and the
   3.7 MB model. Landmark detection takes 6–15 min once; training 2–5 h; total 2.5–5.5 h.
3. **Compare:** the last cell compares the arms. Copy its table into the paper draft's results section.

---

## 2026-10-01: Corrected baseline results (reported)

The fixed `exp-tuning.ipynb` (M1 corrections, resumable training) gave on the untouched test set
(10,245 images): **accuracy 0.72, macro-F1 0.67, macro ROC AUC 0.933**.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.66 | 0.58 | 0.62 | 1,119 |
| disgust | 0.67 | 0.48 | 0.56 | 271 |
| fear | 0.61 | 0.37 | 0.46 | 1,098 |
| happy | 0.93 | 0.88 | 0.90 | 2,959 |
| neutral | 0.68 | 0.73 | 0.70 | 1,913 |
| sad | 0.56 | 0.71 | 0.63 | 1,725 |
| surprise | 0.74 | 0.86 | 0.79 | 1,160 |

- The supports match the natural test set, so the leakage fix worked. Disgust F1 fell from the inflated
  0.85 to 0.56.
- Weakest: fear is mostly missed (recall 0.37), disgust half missed (0.48), and sad collects other classes'
  faces (precision 0.56).
- Not reported yet: TTA accuracy, cross-dataset accuracy, epochs/sessions used, runtime, confusion matrix.

---

## 2026-09-30: Training resumes across Kaggle sessions (12-hour limit); environment check fixed

### Why
A full run is too long for one 12-hour Kaggle session. The original run took about 6.5 h with a frozen ViT;
now phase 2 also trains the ViT, so it will be slower.

### How to run it
1. **First session:** Save Version → Save & Run All.
   - After every epoch the notebook saves the full training state to `/kaggle/working/checkpoints`.
   - Once 11 hours have passed (`TIME_BUDGET_HOURS`), it doesn't start another epoch. It ends cleanly so
     Kaggle keeps the output.
2. **Next session:** open the notebook, choose **Add Input**, and add the finished version's output (or a
   Dataset made from it). Then Save & Run All again. The training cell prints `Resuming from ...` and continues
   at the next epoch.
3. **Repeat until done.** The session that finishes training also runs the evaluation, TTA and the
   cross-dataset test. The others print "Training is not finished" and skip them.

What is saved in `checkpoints/`:

| File | Content |
|---|---|
| `ckpt_phase<P>_epoch<E>.*` | weights + optimizer state; the newest 2 are kept |
| `phase<P>_best.*` | best weights of the phase (for early stopping) |
| `latest.json` | phase, epoch, early-stopping and LR-schedule counters, learning rate, history, epoch time |

**Expected output size:** about 3–3.5 GB in phase 2. Each phase-2 checkpoint is about 1.2 GB (ViT-B + VGG16
weights plus Adam's two moment estimates), and there are 2 of them plus the best-weights files.

### Changes to `exp-tuning.ipynb`
- **Environment cell:**
  - Records `SESSION_START`.
  - **Bug fix:** the check `tf.keras.__version__.startswith("2.")` crashes under tf-keras, because `tf.keras`
    has no `__version__` there. It is replaced by `tf.keras.Model is tf_keras.Model`. The same fix is applied
    to `us-organ-recognition.ipynb`.
- **New cells: "Training across Kaggle sessions" (notes) and "Checkpointing across Kaggle sessions" (code):**
  - Checkpoints match weights by layer structure, not by weight order. The order changes when layers are
    unfrozen, so an order-based save/load can put weights into the wrong layers.
  - Early stopping and LR-on-plateau continue from their saved counters.
  - `SessionCheckpoint` saves every epoch and enforces the time budget.
- **Training cell:**
  - The two phases are described in a `PHASES` table.
  - `train_model()` resumes mid-phase: weights, optimizer, epoch counter and callback states.
  - At a phase switch it starts the next phase with a fresh optimizer, as in the original schedule.
  - If training already finished, it loads the final weights.
- The phase-best `ModelCheckpoint` is replaced by `SessionCheckpoint`.
- **Paths:** intermediate images (preprocessed / split / oversampled / cross) moved from `/kaggle/working` to
  `/kaggle/temp`, so each version's output holds only the checkpoints. They are rebuilt each session
  (about 15–25 min): the split is seeded, so it's identical every time; the augmented copies differ, which
  is fine.
- The cross-dataset cell only runs once training is finished.

### Verification (local, TensorFlow 2.19.1 + tf-keras 2.19.0 + transformers 4.57.6 in a scratch venv)
- **Stand-in model:** a partly frozen nested "VGG", a freezable `vit_wrapper`, and BatchNorm statistics,
  taken through 5 simulated sessions:
  1. Pause mid-phase 1.
  2. Resume, finish phase 1, pause at the phase switch.
  3. Start phase 2 with a fresh optimizer, pause mid-phase 2.
  4. Resume and finish.
  5. Training already complete: load only.
- **Checks on the stand-in run:**
  - Every resume restored the weights and optimizer state exactly.
  - Early-stopping and LR counters were restored.
  - The history ran continuously (6 epochs).
  - Only the newest 2 checkpoints were kept.
- **Real model code from the notebook:** environment cell, `ViTWrapper` around an HF `TFViTModel`, VGG16 + CBAM,
  capsule fusion, transformer refinement, focal loss and MixUp. To keep the test small it used a tiny random
  ViT config, VGG16 without ImageNet weights, and random 224×224 data.
  - Mid-phase resumes were exact: 113 weight tensors; 143 optimizer variables in phase 1, 219 in phase 2.
  - **The ViT really trains now:** all ViT tensors were frozen in phase 1 and updated in phase 2, together
    with VGG block 1.
- **Not tested locally:** real Kaggle mounts, GPU, and the full-size checkpoints' disk and time cost.

---

## 2026-09-29: FER notebook fixed (M1). Status: awaiting Kaggle rerun

### Files
- `exp-tuning.ipynb`: the fixed notebook, with outputs cleared. Upload this one to Kaggle and attach the
  same datasets as before (combined, emotion-detection-fer).
- `exp-tuning_original.ipynb`: the original notebook with all its outputs, kept for reference.
  - This file was **not** in git; the only commit (b17f69b) contains `test_V1`.
  - The original was overwritten in place during this change and restored from the Kaggle download
    `Downloads/exp-tuning-0d8707.ipynb`: same size, all 16 cell sources identical, outputs intact.
  - Commit both notebooks so the original is safe.

### What changed (cell by cell)
| Cell | Change |
|---|---|
| 0 pip | Only `transformers>=4.40,<5`; dropped mtcnn, lz4, tensorflow-addons |
| 1 **new** env | `TF_USE_LEGACY_KERAS=1` before TF import; asserts tf.keras 2.x and transformers <5; `SEED=42` |
| 2 CLAHE | unchanged |
| 3 **new** split | Stratified 90/10 train/val split of `preprocessed/train` → `split/{train,val}`; md5 overlap report train↔val, train↔test |
| 4 oversampling | Oversamples `split/train` only; unique file names (`aug_<k>_<name>.png`); asserts final counts == expected |
| old 3 | **Removed** (it oversampled the test set) |
| 5 imports | + `ModelCheckpoint` |
| 6 ViTWrapper | `freeze()` / `unfreeze_layers()` toggle the whole ViT; `get_config` |
| 10 MixupSequence | `super().__init__()`; forwards `on_epoch_end` so batches reshuffle every epoch |
| 12 main | train / **val** / **clean test** generators; `VGGPreprocess` layer (caffe) before VGG16; ViT named `vit_wrapper`, frozen in phase 1 and unfrozen in phase 2 (asserted); callbacks monitor val; best-weights checkpoints in `/kaggle/working/checkpoints`; `workers=4`; predict once in `evaluate_model`; `save_weights` + rebuild-and-load in TTA |
| 15 cross-dataset | Asserts class folders align by position; md5 overlap report combined train ↔ FER2013 test |

Unchanged: CLAHE, oversampling ratio rule, class weights → focal alpha, MixUp α=0.1, architecture,
AdamW LRs / weight decay, batch 16, epochs 25 + 35, early-stopping / LR-schedule settings, flips, TTA.

### Local verification (no TensorFlow / OpenCV / GPU on this machine)
- The notebook JSON loads, all 16 cell ids are unique, every code cell parses, and all outputs are cleared.
- Smoke test of the new split and oversampling functions, run from the notebook's own code on a synthetic
  3-class folder (stand-ins for `cv2` and `ImageDataGenerator`):
  - Counts match the expected values exactly (27 / 24 / 27).
  - Train and val file names are disjoint.
  - Rerunning the cells gives the same result.
  - The md5 report finds a planted duplicate.
- **Not verified locally:** anything that needs TensorFlow (the tf-keras switch, ViT weight tracking,
  `VGGPreprocess`, `fit` with workers, checkpoint save/load). The notebook asserts these on Kaggle.

### What to check on the Kaggle rerun
- [ ] Cell 1 prints tf-keras 2.x and transformers 4.x (no assertion error)
- [ ] Oversampling assertion passes (disgust ≈ 1038 × 10 after the 90/10 split)
- [ ] "Trainable weight tensors: ViT 0" in phase 1 and "ViT >0" in phase 2
- [ ] Test report support is the natural test set: angry 1119, disgust 271, fear 1098, happy 2959,
      neutral 1913, sad 1725, surprise 1160 (total 10,245)
- [ ] Note the md5 overlap counts (train↔test, combined train↔FER2013 test)
- [ ] Phase 2 epochs will be slower now that the ViT really trains. Training spans several sessions; see the
      2026-09-30 entry for how to resume.

---

## 2026-09-29: Review of the FER notebook, plan for the ultrasound study

### Done
- Reviewed `exp-tuning.ipynb`, including the saved training logs. The original notebook with its outputs is
  kept as `exp-tuning_original.ipynb`, a byte-identical copy of the Kaggle download `exp-tuning-0d8707.ipynb`.
- Chose the direction for the ultrasound study (see `ROADMAP.md` → Decisions).

### Baseline from the original run (`exp-tuning_original.ipynb`), inflated, do not cite
| Metric | Value |
|---|---|
| Test accuracy | 0.748 (TTA 0.749) |
| Macro-F1 | 0.73 |
| Macro ROC AUC | 0.947 |
| Cross-dataset (emotion-detection-fer test, 7,178 images) | 0.688 |
| Epochs run | phase 1: 17 / 25, phase 2: 10 / 35 (early stopping) |
| Time per epoch | ~840 s (3,646 steps × ~230 ms) |

### Findings
1. **Test leakage.** The test set was oversampled with augmentation and then used both as the validation
   set (early stopping, LR schedule) and as the final test. In the classification report, disgust has
   support 2,435, but the real test set has only 271 disgust images. Most of those "test" images are
   augmented copies of the same pictures.
2. **Oversampling lost images to filename collisions.** Keras names each saved augmented image
   `aug_<class>_0_<random 0–9999>.jpeg`, so a later image with the same number overwrites an earlier one.
   - Disgust: 1,153 originals + 10,377 augmented were written, which should give 11,530. Only 7,626
     survived (6,473 augmented). Writing 10,377 files into 10,000 possible names is expected to leave
     ~6,460 unique ones, so this matches.
   - Fear: 8,756 expected, 7,886 found.
3. **The ViT was most likely never fine-tuned.** `TFViTModel` is a tf-keras (Keras 2) model sitting inside
   a Keras 3 layer. Keras 3 only tracks its own variables, so the ViT weights were probably left out of
   the optimizer's variable list. Also, `unfreeze_layers()` set flags on sub-layers only.
4. **MixUp never reshuffled.** `MixupSequence` doesn't forward `on_epoch_end`, so the training order was
   shuffled once and then repeated every epoch.
5. **VGG16 got the wrong input scale.** It received 0–1 images instead of the ImageNet caffe preprocessing
   it was trained with (0–255, BGR, mean-subtracted).
6. Smaller issues:
   - `incremental_unfreeze` looked up `"vit_wrapper"`, but the layer was actually named `"vi_t_wrapper"`.
   - The full-model `.h5` save may not reload.
   - `evaluate_model` ran `predict` twice.
   - Unused installs: tensorflow-addons (it downgrades typeguard), mtcnn, lz4.
   - `transformers` v5 removed TensorFlow support, so it has to be pinned.

### Decisions
- Fix all of the above in `exp-tuning.ipynb` before porting. Horizontal flips stay for FER.
- Ultrasound: pathology classification on a labeled dataset (option B). The Vitale US simulation &
  segmentation dataset is ruled out because it only has healthy subjects and organ masks.

### Next
- Apply the fixes to `exp-tuning.ipynb` (M1), then rerun on Kaggle and fill in the M2 results.
- Verify the candidate ultrasound datasets (M3).

---

## Template: Kaggle run

```
## YYYY-MM-DD: <notebook> run <n>

- Commit / notebook version:
- Kaggle accelerator:            TF:            tf_keras:            transformers:
- Data: train (after oversampling) / val / test counts:
- Phase 1: epochs run, best val_loss, val_acc
- Phase 2: epochs run, best val_loss, val_acc
- Test: accuracy, macro-F1, macro AUC, per-class recall (weakest class)
- TTA accuracy:
- Cross-dataset / external accuracy:
- md5 overlap checks (train↔val, train↔test, train↔external):
- Runtime:
- Notes / anomalies:
```
