# Stage 2 results: pill verification with the FER hybrid

Draft results section for the PharmaScan report (P9), as of 2026-10-10. Every number below comes from the run
folders in `outputs/runs/` (`summary.json`, `calibration.json`, `eval/*.json`); the day-by-day record is in
[PROGRESS_LOG.md](../PROGRESS_LOG.md).

## 1. Setup in brief

- **Task.** The app photographs a dose; each pill must be matched to one of the patient's enrolled medications, or
  rejected as "not one of yours", or the user is asked to flip it and photograph it again.
- **Model.** Our facial-expression model (FER), ported to PyTorch: VGG16 with CBAM and ViT-B/16 in parallel, a
  capsule fusion block and a transformer refinement block, then a 256-number embedding (unit length) with shape and
  colour heads. Medications are compared by cosine similarity of embeddings, so new medicines need no retraining.
- **Training data.** NLM C3PI everyday photos (29,471 crops of 1,710 appearance groups), plus C3PI studio reference
  photos of the training groups from run 3 on (6,198 crops).
- **Splits.** By appearance group (look-alike pills share a group), so no medicine and no look-alike appears in
  two splits: train / validation-enrolled (214 groups) / validation-unknown (214) / test (the 192 pill types of
  ePillID's official holdout fold and their look-alikes; 4,005 C3PI photos of 200 groups).
- **Calibration (on validation only).** A patient is simulated with 10 medications enrolled from 3 photos each. The
  rejection threshold rejects 95% of pills that aren't the patient's. A "flip" margin turns close calls into a
  retake request (largest margin with at most 10% retakes). The crop framing is also chosen on validation.
- **Test sets, scored once with everything fixed beforehand.**
  - *C3PI held-out*: new medicines photographed like the training photos.
  - *MEDISEG*: 32 Hong Kong products, other phones, clutter, small pills (CC BY 4.0).
  - *ePillID*: the published benchmark (Usuyama et al., 2020). Everyday photos of the 192 holdout types, as
    front + back pairs, matched against the studio references of all 4,902 types (MAP, GAP, MAP@1, GAP@1).

## 2. Validation targets (patient-sized list of 10 medications)

| Run | What changed | Right pill | Unknown AUROC | Correct at 95% rejection | Wrong accepted |
|---|---|---|---|---|---|
| Targets | | ≥ 95% | ≥ 0.9 | ≥ 80% | ≤ 2% |
| 1 | Baseline (new layers at learning rate 1e-3) | 77.2% | 0.78 | 17% | 3.7% |
| 2 | FER learning rates, ArcFace margin warm-up | **98.3%** | **0.981** | **90.7%** | **1.0%** |
| 3 | + studio reference photos (D7) | 98.1% | 0.979 | 90.2% | 1.1% |
| 4 | + robustness augmentation (photo conditions, jittered crop windows) | 97.9% | 0.978 | 87.9% | 1.2% |

Run 1 collapsed: the 256-number embeddings used about 3 effective dimensions, enough for colour and shape but not
for imprints. Training the new layers at the FER notebook's rates (3e-5, then 1e-5) instead of 1e-3, with a
separate rate for the ArcFace class table and a margin warm-up, fixed it (12–15 effective dimensions).

## 3. Test results of the app model (strict: test pills never seen in any form)

At the calibrated threshold and margin; 10 medications, 3 enrolment photos; 95% intervals from resampling
medicines are in `eval/c3pi_holdout.json`.

| | Run 2 | Run 3 | Run 4 |
|---|---|---|---|
| **C3PI held-out**: top-1 over all 200 medicines | 82.0% | **84.4%** | 83.2% |
| C3PI held-out, app: named correctly / wrongly / unknown pills rejected | 78.6 / 0.1 / 94.2% | **80.4 / 0.0 / 94.1%** | 76.0 / 0.0 / 94.8% |
| **MEDISEG**: top-1 over 32 products (pills over 100 px) | 38.9% (51.0%) | 41.8% (56.8%) | **48.2% (62.2%)** |
| MEDISEG, app: named correctly / wrongly / own pills rejected / unknown rejected | 17.6 / 1.1 / 51.5 / 78.7% | 18.7 / 1.0 / 64.2 / 89.3% | 18.7 / **0.7** / 70.4 / **93.1%** |
| **ePillID** (strict) two-sided MAP / MAP@1 | 0.255 / 13.7% | 0.506 / 37.1% | **0.555 / 42.3%** |

- **New medicines photographed like the training photos work:** 80% named correctly with no wrong names, and 94–95%
  of other pills rejected.
- **Other phones and conditions are the weakness.** Recognition drops (MEDISEG top-1 42–48%; pills under 100 px:
  ~30–37%), and the threshold set on C3PI photos is too strict there: most of the patient's own pills are rejected
  (64–70%), although wrong names stay at ≤ 1%. The model fails safe, but it isn't yet usable on other phones.
- Studio reference photos (run 3) tripled ePillID MAP@1 (13.7% → 37.1%); robustness augmentation (run 4) helped
  MEDISEG (+6.4 points) at a small cost in C3PI conditions.

## 4. The ePillID benchmark

The paper's protocol trains on the reference photos of **all** 4,902 types, test types included, and those
photos are the test gallery. Runs 2–4 never see a test type, so for a like-for-like comparison runs 5–7 follow the
official protocol (all ePillID references in training, except validation types'; no test-type everyday photo).
They are benchmark models only. "+ imprint" re-ranks the candidates with the imprint read by OCR (Florence-2, four
rotations), using each type's imprint text from C3PI's metadata; it uses information the paper's model did not, so
it is reported as a separate variant.

| Two-sided, fold 4 | MAP | GAP | MAP@1 | GAP@1 | top-5 |
|---|---|---|---|---|---|
| Paper, best (ResNet-152 + CBP, multi-head metric learning) | **0.958** | **0.812** | **92.0%** | 0.912 | – |
| Paper, plain classifiers (ResNet-18 … ResNet-152 B-CNN) | 0.547–0.727 | 0.241–0.448 | 35.7–56.8% | 0.331–0.551 | – |
| Run 5: official protocol | 0.597 | 0.207 | 45.7% | 0.526 | 77.5% |
| Run 5 + imprint | 0.780 | 0.598 | 69.0% | **0.921** | 89.6% |
| Run 6: + stronger fine-tuning | 0.570 | 0.218 | 40.4% | 0.444 | 77.0% |
| Run 6 + imprint | 0.774 | 0.595 | 68.1% | 0.886 | 88.7% |
| Run 7: + benchmark-aligned selection and domain-paired batches | 0.578 | 0.268 | 40.7% | 0.493 | 78.8% |
| Run 7 + imprint | 0.774 | 0.614 | 67.6% | 0.916 | 89.5% |

- **Not beaten on MAP.** Our embedding alone scores like the paper's plain classifiers (0.57–0.60); with imprint
  re-ranking, 0.77–0.78. **GAP@1 beats the paper** with imprint re-ranking (0.921 vs 0.912).
- **Why the gap.** The paper's own tables show it comes from its training recipe, not its backbone: the same
  recipe takes a plain ResNet-18 to MAP 0.912. Its loss ablation: softmax alone 61.8 validation GAP; adding
  contrastive and hard-mined triplet losses 75.5; plus a small ArcFace term 77.2. The recipe: Adam at 1e-4 for the
  whole network, a 2,048-number embedding, contrastive + triplet with online hard mining, softmax and ArcFace over
  side-specific classes, and batches built per pill from its consumer photos and its two reference photos.
- **What we ruled out.** Not our crop style (our own crops of the test photos do worse), not the 64-number pooling in
  the refinement block (features before it score the same), and not local versus global pooling of the backbone
  features. Stronger fine-tuning (run 6) and benchmark-aligned training (run 7) improved validation and MEDISEG but
  not ePillID.
- **Side results.** The official-protocol models are our best on other phones: MEDISEG top-1 52–54% (pills over
  100 px: 71–74%).

## 5. Limitations

- One training source (C3PI, US pills photographed in 2011–2016) limits robustness to other phones and countries;
  the rejection threshold does not transfer to new phones.
- Pills under 100 px and debossed imprints are hard; OCR reads 43% of imprints (printed 63%, debossed 36%).
- No test on our own pill photos yet (G5): the decisive test of the app's real conditions.
- ePillID MAP below the published best; the paper's multi-head recipe has not been tried on our architecture.

## 6. Future work

1. Train the FER hybrid with the ePillID paper's multi-head recipe (contrastive + hard-mined triplet + softmax over
   side classes + ArcFace 0.1, Adam 1e-4, per-pill batches, ePillID's consumer photos), possibly with a larger
   embedding; the paper's evidence suggests this is the main lever for the benchmark.
2. Calibrate the threshold on photos from the target phones (or per patient at enrolment), so new phones don't
   reject most own pills.
3. A fine-tuned imprint reader (C3PI provides imprint text for every photo) or Florence-2-large (47% vs 43%).
4. Own pill photos (G5) as the final test.

## 7. Reproducing

From the `epills` folder, with `.venv`:
```
python scripts/train_stage2.py --config configs/stage2_run3.yaml      # app model (or run4)
python scripts/calibrate_stage2.py --run outputs/runs/stage2_run3 --windows 1.3 1.4 1.55 1.7 1.85 2.0
python scripts/evaluate_stage2.py --run outputs/runs/stage2_run3
python scripts/add_epillid_refs.py                                    # official-protocol training data
python scripts/train_stage2.py --config configs/stage2_run5_official.yaml
python scripts/evaluate_imprint.py --run outputs/runs/stage2_run5_official
```
