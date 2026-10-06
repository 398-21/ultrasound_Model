# Datasets

One section per data source: where it comes from, what's in it, its licence, what was checked and when, and
what's still open. Update a section whenever something is downloaded or checked. Every local copy lives under
`epills/data/`, which is not in git.

Last checked: 2026-10-06.

---

## ePillID (official benchmark)
- **Source:** <https://github.com/usuyama/ePillID-benchmark>, release `ePillID_data_v1.0` →
  `ePillID_data.zip` (160.7 MB).
- **Paper:** Usuyama et al., CVPR Workshops 2020, <https://arxiv.org/abs/2005.14288>.
- **Licence:** MIT.
- **Local copy:** `data/raw/epillid/ePillID_data.zip`.
  SHA-256 `b960b3b1f0ee145ae2be7cfe188de1feaba4286b5d2fb7388004d151ad7311ac`.
- **Contents (checked):**
  - `all_labels.csv`: 13,532 rows with columns `images, pilltype_id, label_code_id, prod_code_id, is_ref,
    is_front, is_new, image_path, label`.
  - 3,728 consumer images (`is_ref=False`) and 9,804 reference images, covering 4,902 pill types; 960 pill types
    have consumer images.
  - Images are pre-cropped at 224 px: `classification_data/fcn_mix_weight/dc_224` (consumer),
    `.../dr_224`, and `segmented_nih_pills_224` (reference).
  - `folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/..._{0..4}.csv`: 5 files of 745–746 consumer
    images and 192 pill types each, plus `_all.csv`.
- **Protocol in the paper:**
  - Consumer images are split 80/20 by pill type (holdout), with 4-fold CV on the 80%.
  - Models may use reference images of all 4,902 pill types.
  - Metrics: MAP, GAP, MAP@1, GAP@1.
  - Best baseline: ResNet-152 with compact bilinear pooling and multi-head metric learning, at MAP 95.76,
    GAP 81.20, MAP@1 92.01, GAP@1 91.19.
- **Pill identity:** `label_code_id` (labeler) + `prod_code_id` (product) → NDC9. 147 pill types share an NDC9
  with another type.
- **Open:**
  - [ ] Which of the 5 fold files is the official holdout (check the repo's training code).
  - [ ] Whether the published numbers use one side or both sides per query.

## NLM C3PI (main training set)
- **Source:** <https://data.lhncbc.nlm.nih.gov/public/Pills/index.html>. The files are dated 2024-08-28; the
  project ended in 2018 and won't be updated.
- **Licence:** images and metadata are for research and development purposes only.
- **Layout:**
  - `PillProjectDisc1` … `PillProjectDisc110`, each with an `images/` folder.
  - `ALLXML/PillProjectDisc<N>.xml`, one per disc, with metadata for every image in it.
  - `directory_consumer_grade_images.txt`: `|`-delimited, with fields NDC11, part, path, image type, name.
  - `rximage.zip` (7.3 GB): reference images in 6 resolutions, plus RxImage API metadata.
  - `sampleData.zip` (0.5 GB).
- **Local copies:**
  - `data/raw/c3pi/meta/directory_consumer_grade_images.txt` (133,774 rows). SHA-256
    `69ac367991b2cfa204219e5bd3fd7016f2be280771a5517d4fbc3cfe56b84302`.
  - `data/raw/c3pi/ALLXML/PillProjectDisc10.xml`, one example.

**Size by image type**, from the directory joined with the 110 disc listings. The whole archive is 2.72 TB.

| Image type | Ext | Files | GB | Pill types (NDC11) | Use |
|---|---|---|---|---|---|
| `C3PI_Test` | JPG | **40,882** | **84.0** | **2,394** | **Download.** Consumer-grade photos: the training set |
| `MC_CHALLENGE_V1.0` | JPG | 4,408 | 2.2 | 2,124 | **Download.** About 2 per type (front and back), as used in the 2016 NLM challenge |
| `MC_API_RXNAV_V1.3` | JPG | 4,708 | 2.7 | 4,531 | **Download** (or `rximage.zip` for its metadata). RxImage reference images |
| `MC_API_NLMIMAGE_V1.3` / `MC_SPL_SPLIMAGE_V3.0` | JPG | 4,708 each | 2.0 / 0.7 | 4,531 | Optional: other renderings of the same references |
| `MC_C3PI_REFERENCE_SEG_V1.6` | PNG | 41,330 | 139.3 | 4,861 | Optional, later: segmented studio shots, about 8.5 per type |
| `C3PI_Reference` | CR2 (RAW) | 10,334 | 235.5 | 4,861 | Skip |
| `MC_COOKED_CALIBRATED_V1.2` | PNG | 20,034 | 2,218 | 4,861 | Skip |
| `C3PI_Test` | WMV (video) | 2,503 | 30.8 | — | Skip |
| `C3PI_Test` | PNG | 159 | 2.7 | — | Skip |

- **Photos per pill type** (`C3PI_Test` JPG): minimum 3, median 15, 90th percentile 30, maximum 101.
  2,389 of 2,394 types have ≥ 4, so P×K with K = 4 works.
- **What a consumer photo looks like:** one pill, centred on a plain background (paper, laminate, …) and small
  in the frame; e.g. a capsule about 20% of the width of a 2560×1920 photo. Each photo needs a crop.
- **Metadata per image** (`ALLXML`, checked on disc 10):
  - IDs and names: `NDC9`, `NDC11`, `Part`, `Parts`, `LabeledBy`, `GenericName`, `ProprietaryName`
  - the file: `File/Name`, `Type`, `Size` (bytes), `Sha1`
  - the image type: `Class` (consumer) or `Layout` (reference)
  - capture: `Camera`, `Illumination`, `Background`
  - clarity ratings: `RatingImprint`, `RatingShape`, `RatingColor`, `RatingShadow`, `RatingBackground`
  - the pill: `Imprint` (e.g. `TEVA;5770`), `ImprintType` (e.g. `DEBOSSED`), `Color`, `Shape`, `Score`,
    `Symbol`, `Size` (mm)
  - `AcquisitionDate`
- **Overlap with ePillID** (on NDC9):
  - All 960 ePillID consumer pill types appear among the C3PI consumer photos.
  - 884 of C3PI's 2,362 consumer NDC9s are ePillID consumer types.
  - Each ePillID fold covers 188–191 NDC9s, which have 3,802–4,021 C3PI consumer photos.
  - Removing one fold leaves about 2,170 pill types and about 37,000 photos for training.
- **Attribute coverage for ePillID:** 2,483 of the 4,902 ePillID pill types have an NDC9 somewhere in C3PI, and
  so get C3PI metadata. That includes all 960 consumer types.
- **Open:**
  - [ ] Download script: selective, resumable, checked against SHA-1. Download `ALLXML` first.
  - [ ] Parse all of `ALLXML` into one table; check how complete the attribute fields are.
  - [ ] Crop pipeline and crop-quality rate (gate G2).
  - [ ] The NLM Pillbox archive isn't hosted at `data.lhncbc.nlm.nih.gov/public/` anymore. For attributes of
        the ePillID reference-only types, use `rximage.zip` metadata or DailyMed product labels.

## MEDISEG (cross-domain test)
- **Source:** <https://city.figshare.com/articles/dataset/MEDISEG/28574786>, published 2026-05-12.
  - `MEDISEG.tar.gz` (0.44 GB): <https://ndownloader.figshare.com/files/52926545>
  - `MEDISEG-Deploy.tar.gz` (0.01 GB): <https://ndownloader.figshare.com/files/64481325>
- **Paper:** <https://arxiv.org/abs/2603.10825>.
- **Licence:** CC BY 4.0.
- **Contents (from the paper):**
  - 8,262 images of 32 pill types, from single pills to cluttered pill organisers (dosette boxes), with overlap
    and occlusion.
  - Instance segmentation masks and class labels.
  - Test subsets: 3-Pills and 32-Pills.
- **Open:**
  - [ ] Download; check class names (UK products, so no NDC), masks, and how many crops per class.

## VAIPE (cross-domain test, at risk)
- **Described in:** <https://arxiv.org/abs/2303.09782> and the AI4VN 2022 VAIPE challenge.
  - 9,426 smartphone photos, 96 pill labels, 5–10 pills per photo.
  - Labelled from prescriptions (Vietnam).
- **Access (checked 2026-10-06):**
  - `https://vaipe.org/` doesn't respond.
  - The VAIPE-PCIL crop download linked from <https://github.com/vinuni-vishc/CG-IMIF> (Google Drive) returns
    404.
  - Not checked yet: a Roboflow Universe mirror (`universe.roboflow.com/kiet/vaipe-coopj`, free account).
- **Open:**
  - [ ] Email the authors (contact in the CG-IMIF repo); try the Roboflow mirror; confirm the terms.

## Our enrolled pills (real workflow test)
- **Collected by you,** following [own_pills_protocol.md](own_pills_protocol.md).
- **Stored in** `data/raw/own_pills/`, with a `labels.csv`.

---

## Stage 3 sources

### DDInter 2.0
- **Site:** <https://ddinter2.scbdd.com/>.
- **Paper:** Nucleic Acids Research 2025, <https://academic.oup.com/nar/article/53/D1/D1356/7740584>.
- **Licence:** CC BY-NC-SA 4.0. Personal, non-commercial, informational or scholarly use.
- **Bulk files:** `https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_<X>.csv`.
  - The download page links A, B, D, H, L, P, R, V. C, G, J, M, N and S are served at the same pattern.
  - All 14 are downloaded to `data/safety/ddinter/`.
- **Contents (checked):**
  - Columns: `DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level`.
  - 234,981 unique pairs over 1,971 drugs.
  - Levels: Moderate 143,748; Major 39,082; Minor 9,736; Unknown 42,415.
- **Not in the bulk files:**
  - mechanism and management text
  - therapeutic duplications
  - drug–food and drug–disease interactions
  - These are on the website (`/server/interaction/`, `/server/other_interaction/`) only.
- **Mismatch:** the paper reports 302,516 records over 2,310 drugs. Ask the authors whether the bulk files are
  current.
- **Open:**
  - [ ] Email the authors for a full export including management text and duplications.

### RxNorm API
- `https://rxnav.nlm.nih.gov/REST/…` is live (`rxcui.json?name=ibuprofen` → `5640`). Use it to map names to
  ingredients.
- The interaction endpoint (`/REST/interaction/…`) returns 404; it was discontinued in January 2024.

### openFDA drug label API
- `https://api.fda.gov/drug/label.json` is live. It works without a key at low volume; a free key raises the
  limits.
- Its disclaimer says results are unvalidated, so show them as label text only, never as advice.

### DrugBank (optional)
- Needs a licence (academic licences exist). Only used if one is obtained.
