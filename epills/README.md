# ePills

Pill identification and safety checks for a medication-adherence app (capstone).
- **Stage 2:** a hybrid ViT + CNN that maps each pill crop to a 256-d embedding, matches it to the patient's
  enrolled medications, or rejects it as unknown.
- **Stage 3:** a rule-based safety engine on curated drug-interaction data.

- Plan, data gates and decisions: [ROADMAP.md](ROADMAP.md)
- Status, who does what, and history: [PROGRESS_LOG.md](PROGRESS_LOG.md)
- Data sources, sizes, licences and checks: [docs/datasets.md](docs/datasets.md)
- How to photograph our own pills: [docs/own_pills_protocol.md](docs/own_pills_protocol.md)

## Folder structure
```
epills/
├── README.md
├── ROADMAP.md
├── PROGRESS_LOG.md
├── docs/
│   ├── datasets.md              one section per data source: source, licence, size, checks, open items
│   └── own_pills_protocol.md    collection and photo protocol for our enrolled pills
├── serving/                     HOSTED: the API (FastAPI), Dockerfile, run-time requirements; see serving/README.md
├── src/epills/                  the Python package
│   ├── core/                    HOSTED: everything the API needs at run time; never imports training
│   │   ├── models/              the hybrid ViT + CNN network (shared by training and serving)
│   │   ├── inference/           crop preprocessing, embeddings, gallery matching, unknown rejection
│   │   └── safety/              Stage 3 rules; DDInter, RxNorm and openFDA lookups
│   └── training/                not hosted: data preparation, training, evaluation
│       ├── datasets/            readers per dataset, C3PI cropper, P×K sampler, augmentations
│       ├── losses/              batch-hard triplet, ArcFace, auxiliary shape and colour losses
│       ├── eval/                retrieval (MAP/GAP), open-set metrics, thresholds, bootstrap CIs
│       └── ocr/                 imprint reader training (stretch goal)
├── configs/                     one config per training run / ablation arm
├── scripts/                     command-line steps: download, crop, split, train, evaluate
├── notebooks/                   exploration and result figures
├── tests/                       unit and smoke tests
├── data/                        NOT in git (licences, size)
│   ├── raw/                     downloads exactly as received, never edited
│   │   ├── epillid/             ePillID_data.zip
│   │   ├── c3pi/                meta/, ALLXML/, images/ (selected consumer and reference photos)
│   │   ├── mediseg/
│   │   ├── vaipe/
│   │   └── own_pills/           our photos + labels.csv (see the protocol)
│   ├── interim/                 parsed metadata tables, fixed-size crops
│   ├── processed/               manifests and splits used by training and evaluation
│   └── safety/                  ddinter/ CSVs, rxnorm_cache/, openfda_cache/
└── outputs/                     NOT in git: runs, weights, figures, reports
```

**Rules:**
- `data/raw/` is read-only once downloaded; every derived file goes to `interim/` or `processed/`, produced by
  a script.
- Every download and processing step gets an entry in the progress log.
- **Hosted vs not hosted:** only `serving/` and `src/epills/core/` are deployed (plus weights from a Hugging Face
  model repo). `core` must never import `training`; `tests/test_layout.py` checks this.
- Code under `src/epills/training/datasets/` is tracked. The repository's root `.gitignore` ignores every folder
  named `data/`, which is why the package folder is called `datasets`.
