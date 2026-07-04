# BioEpi-SAFE-LSTM

Code and processed reproducibility files for the manuscript:

**Surveillance-anchored digital fusion for probabilistic influenza forecasting in China, Japan and the United States**

BioEpi-SAFE-LSTM is a surveillance-anchored adaptive fusion framework for 1–4-week probabilistic influenza forecasting. It combines a surveillance-only quantile LSTM backbone with country-specific digital experts and activates auxiliary experts only when resolved probabilistic loss improves on the surveillance backbone.

## Repository contents

```text
BioEpi-SAFE-LSTM/
├── scripts/                  # Analysis, SAFE fusion, evaluation and figure scripts
├── configs/                  # Analysis configuration files
├── data/processed/           # Processed model-input prediction objects; no raw posts
├── data/folds/               # Rolling-origin fold calendar
├── results/tables/           # Manuscript and supplementary result tables
├── results/figures_v8/       # Season-level main figure outputs
├── results/figures_npj/      # Additional display figure outputs
├── results/metadata/         # Run manifests and QC summaries
├── docs/                     # Data dictionary, upload guide and release checklist
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── LICENSE
```

## Quick start

Create an environment:

```bash
conda env create -f environment.yml
conda activate bioepi-safe-lstm
```

or use pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify files:

```bash
python scripts/verify_checksums.py --root .
```

Run the full reproducibility workflow:

```bash
python scripts/run_all_reproducibility.py --root . --bootstrap-reps 2000
```

For a fast smoke test:

```bash
python scripts/run_all_reproducibility.py --root . --bootstrap-reps 50
```

If result tables are already present and only figures need to be regenerated:

```bash
python scripts/run_all_reproducibility.py --root . --skip-safe-fusion --skip-season-eval
```

## Data notes

This repository contains processed aggregate prediction files and result tables. Raw social-media posts, user identifiers, post text and profile metadata are not redistributed. The full reviewer data folder is listed in `data/reviewer_access/REVIEWER_DATA_LINK.txt` and should be replaced or supplemented by a persistent repository DOI before publication.

## Main outputs

Key outputs are written to:

```text
results/tables/v8_season_level/
results/tables/npj_display/
results/figures_v8/
results/figures_npj/
```

## Citation

Please cite the manuscript and this repository release. Repository citation metadata are provided in `CITATION.cff`.
