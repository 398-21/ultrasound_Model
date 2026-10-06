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
| G1 | ePillID benchmark | ✅ downloaded and checked | Identify the official holdout fold | Claude |
| G2 | C3PI training photos | 🟡 checked, not downloaded | Download script → you run it on the desktop | Claude → You |
| G3 | Attribute labels (shape, colour, size) | 🟡 metadata format checked (one disc) | Parse all `ALLXML`; coverage stats | Claude |
| G4 | Cross-domain test | 🟡 MEDISEG available; VAIPE at risk | Download MEDISEG; email the VAIPE authors | Claude; You |
| G5 | Own pills | ⬜ not started | Choose the pills; get the mat; photo sessions | You |
| G6 | Safety data | 🟡 DDInter severity in hand; advice text and duplications missing | Email the DDInter authors; RxNorm mapping | You; Claude |

| Decision | Status |
|---|---|
| D1 framework (blocks model code) | ⬜ open (recommendation: PyTorch) |
| D2 CNN branch | ⬜ open (recommendation: ConvNeXt-T stages 1–3) |
| D3 earlier-hybrid parts as extra arms | ⬜ open (recommendation: no) |
| D4 measured inputs on public data | ⬜ open (recommendation: metadata size + noise for training) |
| D5 cross-domain set | ⬜ open (recommendation: MEDISEG primary) |
| D6 DDInter "Unknown" severity | ⬜ open (recommendation: show as Moderate, "not graded") |

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
