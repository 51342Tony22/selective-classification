# Single- and Dual-Threshold Selective Classification under Class Imbalance

This project is a separate research direction from the existing human-deferral / cost-sensitive BAF paper.

The abstention interval is treated as a reject / uncertain region for selective classification, not as a human review region.

## Current First Step

See `IMBALANCE_RESEARCH_RESTRUCTURE_PLAN.md` for the initial audit and implementation plan.

## Implemented So Far

- Generic dataset and prediction-artifact audit utilities.
- Leakage-aware data split utilities:
  - BAF temporal split
  - stratified train / calibration / threshold-selection / test split
  - repeated stratified cross-validation
- Selective-classification metrics for single- and dual-threshold policies.
- Threshold search for optimized single thresholds and budget-constrained dual thresholds.
- Probability calibration utilities for none, Platt, isotonic, and temperature scaling.
- Paired bootstrap confidence intervals for policy comparisons on the same test examples.
- Majority-class downsampling utilities for controlled training-imbalance experiments.
- Local dataset loaders for BAF Base, SECOM, and Online Shoppers.
- A minimal Logistic Regression smoke-test runner that selects thresholds on the threshold-selection split and evaluates on the test split.
- Local UCI data connection for Online Shoppers and SECOM.
- A fixed 50k BAF sample and first main logistic-regression experiment report:
  `results_imbalance_selective/main_50k_lr/REPORT.md`
- A three-model 50k main experiment report:
  `results_imbalance_selective/main_50k_three_models/REPORT.md`
- Phase 2 method-lock smoke test and BAF scale plan:
  `PHASE2_METHOD_LOCK_AND_BAF_SCALE_PLAN.md`
- Final Phase 2 result bundle:
  `results_imbalance_selective/final_phase2/FINAL_PHASE2_REPORT.md`
- Paper-ready Phase 2 diagnostics:
  `results_imbalance_selective/final_phase2/paper_artifacts/`

## Test Command

This project now has its own virtual environment:

```bash
cd /Users/starsfalling/Documents/AT/Project/human-deferral-worthwhile-new-direction
.venv/bin/python -m pytest
```

Current status:

```text
36 passed
```

## Data Setup

BAF Base is installed locally inside this project:

```text
data/raw/baf/Base.parquet
data/interim/baf_split_assignments.parquet
```

The Kaggle download script is:

```bash
.venv/bin/python scripts/download_baf.py --config configs/imbalance_selective/baf.yaml
```

If Kaggle access is unavailable but a local `Base.csv` or `Base.parquet` already exists:

```bash
.venv/bin/python scripts/download_baf.py \
  --config configs/imbalance_selective/baf.yaml \
  --source-file /path/to/Base.parquet
```

## Smoke Test Command

Set `dataset.local_path` in one of the config files first. The runner does not download data automatically.

```bash
cd /Users/starsfalling/Documents/AT/Project/human-deferral-worthwhile-new-direction
PYTHONPATH=src ../../baf-rcast-study/.venv/bin/python -m imbalance_selective.smoke_test \
  --config configs/imbalance_selective/online_shoppers.yaml \
  --output results_imbalance_selective/smoke_test/online_shoppers_lr_seed7.csv
```

For a controlled training-imbalance smoke test, add:

```bash
--train-ratio 1:10
```

Current smoke-test report:

`results_imbalance_selective/reports/smoke_test_summary.md`

## Guardrails

- Do not delete, overwrite, or rewrite the existing human-deferral paper.
- Do not write new selective-classification results into old result folders.
- Reuse BAF splits, predictions, calibration outputs, and threshold-search logic only after file-level alignment is verified.
- Keep economic-cost and human-review variables in the old project only.
