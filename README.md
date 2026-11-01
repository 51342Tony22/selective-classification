# Selective Classification under Class Imbalance

Research code for **Dual-Threshold Selective Classification Under Class Imbalance: Matched-Coverage Evaluation and Boundary Error Enrichment**, accepted at AIAM 2026.

The study compares optimized single-threshold, symmetric dual-threshold, and asymmetric dual-threshold policies at matched coverage. Separate data partitions are used for training, calibration, threshold selection, and testing. Experiments include Logistic Regression, LightGBM, and MLP on BAF, SECOM, and Online Shoppers.

## Contents

- `src/imbalance_selective/`: splitting, calibration, model fitting, selective metrics, policy search, and paired bootstrap analysis.
- `configs/`: dataset and experiment configurations.
- `scripts/`: data preparation, experiment runners, audits, and figure generation.
- `tests/`: deterministic numerical and pipeline checks.
- `results_imbalance_selective/final_phase2/`: saved main results and paper artifacts.
- `results_imbalance_selective/final_phase2_extension_baselines_when/`: a separately retained baseline extension.
- [Detailed research notes](README.research.md).
- [Main experiment report](results_imbalance_selective/final_phase2/FINAL_PHASE2_REPORT.md).

## Setup and tests

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

BAF is obtained from its original distribution; access may require Kaggle credentials. SECOM and Online Shoppers are obtained from UCI. Set each configuration's dataset path to your own local copy. Raw datasets and trained model artifacts are excluded from this Git repository.

```bash
python scripts/download_baf.py --help
python scripts/run_main_experiments.py --help
```

Review the configurations and dataset prerequisites before launching a full experiment. Saved result tables are provided for inspection without rerunning training.

## Findings and limits

Allowing abstention can reduce balanced selective risk. Asymmetric dual thresholds do not consistently outperform symmetric thresholds when coverage is matched. The abstention interval represents uncertainty/rejection, not an implemented human-review service. Interpret results by dataset, class distribution, and abstention budget.

`SOURCE_SNAPSHOT.json` records the selected local files as of 2026-11-01. Earlier drafts, raw data, local environments, and unrelated human-deferral studies are excluded.
