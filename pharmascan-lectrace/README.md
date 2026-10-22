# PharmaScan — Lectrace presentation

The 10-minute execution-explanation presentation for the Deep Learning mini-project, written as a
[lectrace](https://pypi.org/project/lectrace/) script (a step-through viewer for Python, inspired by Percy Liang's
edtrace). This folder is standalone: it needs only `lectrace` and `numpy`, not the training code, the datasets or the
model weights.

## Run it

```powershell
cd pharmascan-lectrace
python -m venv .venv                       # once
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\lectrace serve 01_pharmascan.py
```

Open http://localhost:7000. Keys: `→` next step, `←` back, `Shift+→` step over (stay in `main()`),
`u` step out of a function, `F` full screen, `E` hide the variable panel.

`lectrace build --output _site` makes a static site (e.g. for GitHub Pages). After changing `_helpers.py`, delete
`_site/.lectrace_cache.json` first: lectrace only notices changes to the lecture file itself.

## What runs live, and what is precomputed

| Part | Where |
|---|---|
| VGG16 + CBAM and ViT-B/16 backbones (≈100M parameters) | precomputed by `_precompute.py` → `data/backbones.npz` |
| Capsule fusion with dynamic routing | **live**, `capsule_fusion()` in `01_pharmascan.py` |
| Transformer refinement and embedding head | **live**, `refine_and_embed()` (layers in `_helpers.py`) |
| Gallery prototypes, threshold and margin decision | **live**, `enrol()`, `decide()` |
| Catalogue look scores, imprint re-ranking, logistic confidence | **live**, `rerank()`, `confidence()` |

The live parts use the served model's own weights (run 4, `stage2_run4@2026-10-09`, exported to
`data/weights.npz`, 645,382 numbers). The lecture prints `same_as_pytorch`, the cosine between its numpy forward pass
and the served PyTorch model on the same photo: 1.0000000. The identification result also matches the served API's
code exactly (escitalopram 5 mg first, confidence 0.925, `possible_match`).

`_precompute.py` needs the main project (`..\epills`, its `.venv`, the C3PI photos and the served model):

```powershell
..\epills\.venv\Scripts\python _precompute.py
```

The patient is the one in `epills/scripts/demo_first_slice.py`: four C3PI validation medicines (metformin ER 750 mg,
lisinopril 5 mg, amlodipine 10 mg, simvastatin) enrolled from 3 photos each, none seen in training. The foreign pill
is escitalopram 5 mg (imprint M;EC5). The first-slice demo uses a warfarin photo instead, but its imprint can't be
read ('O' only), so it can't show what re-ranking does.

## Speaking plan (10 minutes)

Step with `Shift+→` in `main()`; press `→` to step *into* a function only where marked.

| Time | Section | What to say |
|---|---|---|
| 0:00–2:30 | Intro | Medication errors and polypharmacy; Singapore pills dispensed loose. Why a classifier doesn't fit (every patient's list differs, new medicines): an embedding with open-set rejection. Accepted errors and abstention have different consequences and are reported separately. System and architecture figures. |
| 2:30–4:00 | **State 1**: photo → crop → two views | The pill finder's box; the crop framing (1.55×); the two backbone vectors (256 and 768). The attention images: be honest that CBAM's gate is almost all 1 and the ViT also looks at background patches. |
| 4:00–5:30 | **State 2**: capsule routing | Step **into** `capsule_fusion` and walk the loop. `c` starts at [0.5, 0.5], the agreement `u @ v` is larger for the ViT branch, so `c` becomes [0.09, 0.91]. Then step over `refine_and_embed`: unit length 1, `same_as_pytorch` = 1.0. The refinement's attention is almost uniform for this example. Its contribution requires a controlled ablation. |
| 5:30–7:00 | **State 3**: is it one of yours? | Prototypes from 3 photos. Step into `decide` for the metformin: 0.960 against τ 0.895, with a large gap to the runner-up, so it matches. The foreign pill's closest is amlodipine at 0.713 < τ, so "unknown": a classifier would have had to name one of the four. Show the chart. |
| 7:00–8:30 | **State 4**: identify + imprint | Look alone: escitalopram 5 mg is 4th among small white tablets. OCR reads "EC5"; rare tokens weigh more. Step into `rerank` (`nearest_five` shows the imprint scores) and `confidence` (weights, fit "good", p = 0.92). Still worded as a *possible* match. |
| 8:30–10:00 | Key result | Strict C3PI test: 78.3% correct and 0.06% wrong among enrolled queries. Among foreign queries, 94.8% are rejected and 3.5% incorrectly accepted (different denominators). Identification: 80.2% top-1 (71.6% by look). MEDISEG enrolled queries: 70.4% unknown and 0.7% wrong. Local enrolment and calibration require evaluation. Clinical safety has not been established. Visual evidence only: the label gives the strength and the patient confirms. |

All numbers come from `epills/outputs/runs/stage2_run4/eval/` (see the paper materials document,
`epills/docs/paper/PharmaScan_paper_materials.docx`).

## Files

- `01_pharmascan.py`: the lecture (traced). Helpers that matter are defined below `main()` so the viewer steps into them.
- `_helpers.py`: loading, textbook layers and chart specs (never traced or shown).
- `_precompute.py`: the one-off export from the served model.
- `data/`, `images/`: what the export wrote (about 6 MB), plus two figures from the paper.

## AI assistance

This script, the export and the checks were written with Claude Code (an AI coding agent), as the assignment
encourages. The team reviewed it, and the numbers were checked against the served model and the result files.
