# Progress Log

Newest entry first. Each Kaggle run gets its own entry (template at the bottom).

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
