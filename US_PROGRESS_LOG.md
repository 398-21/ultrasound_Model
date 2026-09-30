# Ultrasound Study: Progress Log

Newest entry first. Each Kaggle run gets its own entry (template at the bottom). The FER work is logged
separately in `PROGRESS_LOG.md`.

## Study summary

**Goal:** a trauma-oriented abdominal ultrasound model built on the FER architecture (VGG16 + CBAM and
ViT-B/16, capsule fusion, transformer refinement).

**What this dataset can and cannot do:**
- It has organ masks but **no trauma cases**. The real subjects are healthy and the simulated scans come
  from normal CT.
- So it can teach normal anatomy, but it cannot train or validate a trauma detector on its own.

**Plan in three stages:**

| Stage | What | Data | Status |
|---|---|---|---|
| 1 | Multi-label **organ recognition**, the anatomy / FAST-window baseline (liver + kidney = RUQ, Morison's pouch; spleen + kidney = LUQ, splenorenal recess) | this dataset | code written, awaiting first Kaggle run |
| 2 | **Synthetic free-fluid proof-of-concept**: paint fluid into the liver–kidney / spleen–kidney gaps using the masks, then train fluid vs no fluid | this dataset + stage-1 weights | designed, coded after stage 1 runs |
| 3 | **Real free-fluid (trauma) detection** | labeled FAST data, still to be found | not started |

Stage 2 results show the pipeline works. They are **not** evidence that the model detects real trauma.

### Dataset
- **Source:** Vitale et al., abdominal US simulation & segmentation dataset (the CycleGAN ultrasound
  realism paper). Kaggle copy: `omoikane2002/ra-us-data`.
- **Real (RUS):** 617 scans from 11 healthy subjects on a SonoSite M-Turbo. Views include intercostal,
  subcostal, longitudinal, transverse, FAST-RUQ, FAST-LUQ and FAST-heart. Masks exist for a subset of
  about 61 scans (liver, kidneys, gallbladder, spleen, vessels).
- **Simulated (AUS):** 926 ray-cast scans from 13 VISCERAL CT volumes, each with a silver-standard mask.

Kaggle paths (root `/kaggle/input/datasets/omoikane2002/ra-us-data/abdominal_US`):

| Folder | Images | Masks | Used as |
|---|---|---|---|
| `AUS/images/train`, `AUS/annotations/train` | yes | yes | train + validation |
| `AUS/images/test`, `AUS/annotations/test` | yes | yes | in-domain test (simulated) |
| `RUS/images/test`, `RUS/annotations/test` | yes | only the annotated subset | real-world test (sim-to-real) |
| `RUS/images/train` | yes | none | qualitative predictions only |

**Mask colors (from the dataset description):**

| Organ | Color |
|---|---|
| liver | violet |
| kidney | yellow |
| pancreas | blue |
| vessels | red |
| adrenals | light blue |
| gallbladder | green |
| bones | white |
| spleen | pink |

The exact RGB values are confirmed by the notebook's audit cell.

---

## 2026-09-30: Environment cell fixed

- The check copied from the FER notebook, `tf.keras.__version__.startswith("2.")`, would crash under tf-keras,
  because `tf.keras` has no `__version__` there.
- It now checks `tf.keras.Model is tf_keras.Model`. This was tested with TensorFlow 2.19.1 + tf-keras 2.19.0.
- Multi-session resume (added to the FER notebook) isn't needed here: about 800 training images should train
  well within one 12-hour session.

---

## 2026-09-29: Direction chosen; stage 1 notebook written. Status: awaiting first Kaggle run

### Decisions
- Use this dataset for **organ recognition (stage 1)** plus a **synthetic-fluid proof-of-concept
  (stage 2)**. It is an anatomy baseline, not a trauma or pathology dataset.
- The pathology plan (option B) is **deferred**, not dropped (see `ROADMAP.md`, track B).
- The ultrasound study gets its own log (this file).

### Stage 1 design (`us-organ-recognition.ipynb`)
- **Labels:** each mask becomes 8 yes/no organ labels. An organ counts as present if it covers at least
  0.5% of the mask. Pixels are matched to the nearest legend color within a tolerance, so JPEG noise in
  the masks is handled.
- **Splits:**
  - Train and validation come from AUS train, split by CT volume when the filenames reveal it,
    otherwise at random (with a warning).
  - AUS test is the in-domain test set.
  - The annotated RUS test images are the real-world test. They are scored only on the organs the RUS
    masks actually contain, because an organ missing from a partial annotation doesn't mean it's missing
    from the image.
- **Carried over from the FER pipeline:**
  - Architecture (only the output layer changes: 8 sigmoid outputs instead of softmax).
  - CLAHE, MixUp, AdamW, the two-phase ViT freeze/unfreeze, checkpoints, TTA.
- **Ultrasound-specific changes:**
  - No horizontal flips, and the empty fan area is filled with black instead of mirrored anatomy.
  - Rare organs are repeated in the training list (up to 5×) instead of oversampling class folders.
  - Binary focal loss with per-organ weights.
- **Reporting:**
  - Per-organ ROC AUC, average precision, sensitivity/specificity/F1 (thresholds tuned on validation).
  - Macro AUC with bootstrap 95% CI, exact-match ratio and Hamming loss.
  - AUS vs RUS side by side, which shows the sim-to-real gap.
- **FAST readout:** a qualitative gallery of real scans with an RUQ/LUQ window guess (not scored; there
  are no view labels).

### Local verification (no TensorFlow / OpenCV / GPU on this machine)
- All 28 cells of `us-organ-recognition.ipynb` parse. The cells reused from `exp-tuning.ipynb` are copied
  verbatim.
- A smoke test ran the notebook's own data cells on a fake dataset with the same folder layout, using a
  small PIL stand-in for OpenCV:
  - 60 AUS train, 25 AUS test, 14 RUS test (only 10 annotated, with 5 organ types) and 6 RUS train images.
  - Noisy mask colors, tiny organ specks, and one mask named with a `_mask` suffix.
- Results:
  - Every image's organ labels matched the generated truth: noise tolerated, specks under 0.5% ignored.
  - Pairing handled the `_mask` suffix. Unannotated RUS images were excluded from scoring, and RUS was
    scored only on its 5 annotated organs.
  - Train and val are disjoint. Repeat sampling matched a hand computation (51 → 138 rows).
  - The evaluation code gave a per-organ table and a macro AUC inside its bootstrap CI.
  - A wrong liver color in `PALETTE` stops the run and prints the real color.
- One fix came out of this: the color audit now groups near-identical colors. Before, JPEG-style noise split
  one color into thousands of tiny variants, which could let a wrong palette slip past the check.
- **Not verified locally:** anything that needs TensorFlow (model build, training, prediction, TTA, saving).
  The notebook asserts the ViT freeze/unfreeze on Kaggle.

### Stage 2 design (to code after stage 1 runs)
1. For scans where the liver and kidney masks (or spleen and kidney) meet, find the band between them:
   dilate both masks and intersect.
2. Paint a thin anechoic (dark) crescent into that band, with soft edges, speckle matched to the
   surroundings, and random width and length.
3. Painted scans are fluid-positive; unmodified scans are fluid-negative. Keep the stage 1 splits.
4. Review a gallery of painted scans before training.
5. Fine-tune from the stage-1 weights and report AUC on synthetic positives, labeled as a pipeline
   demonstration only.

### To confirm on the first Kaggle run (the audit cell prints these)
- [ ] Exact RGB value for each organ color (if the audit assertion fails, paste the printed color table)
- [ ] Images and masks pair up by file name
- [ ] Whether file names contain a CT-volume or subject ID (set `GROUP_REGEX` for a volume-wise split)
- [ ] Image sizes and formats
- [ ] Which organs the RUS test masks contain
- [ ] Organ prevalence per split (any organ that never appears can't be scored)

### Next
- Run `us-organ-recognition.ipynb` on Kaggle (GPU and Internet on; attach `ra-us-data`).
- Fill in a run entry below, then build stage 2.

---

## Template: Kaggle run

```
## YYYY-MM-DD: <notebook> run <n>

- Notebook version / Kaggle accelerator / TF / tf_keras / transformers:
- Audit: palette confirmed? unmatched files? GROUP_REGEX used?
- Data: AUS train / val / AUS test / RUS test counts; train rows after repeat sampling
- Organ prevalence (train / AUS test / RUS test):
- Phase 1: epochs run, best val_loss, val AUC
- Phase 2: epochs run, best val_loss, val AUC
- AUS test: macro AUC [95% CI], per-organ AUC, exact match, Hamming loss
- RUS test (organs scored: ...): macro AUC [95% CI], per-organ AUC
- TTA (AUS / RUS macro AUC):
- md5 overlaps (train↔val, AUS train↔AUS test, AUS↔RUS):
- Runtime:
- Notes / anomalies:
```
