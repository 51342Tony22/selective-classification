# Smoke Test Summary

Generated after connecting the UCI datasets locally.

## Data Connected

- Online Shoppers: `data/raw/online_shoppers/online_shoppers_intention.csv`
  - Rows: 12,330
  - Positive rate: 0.154745
- SECOM: `data/raw/secom/`
  - Rows: 1,567
  - Positive rate: 0.066369

BAF Base is installed inside this project at `data/raw/baf/Base.parquet`. The BAF split assignments are stored at `data/interim/baf_split_assignments.parquet`.

## Natural Imbalance Smoke Tests

Logistic Regression, seed 7, no calibration.

| Dataset | Policy | Coverage | Abstention | Macro-F1 | Balanced Acc. | MCC | Minority Recall | Minority Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Online Shoppers | single default 0.5 | 1.000000 | 0.000000 | 0.683808 | 0.647222 | 0.416025 | 0.319372 | 0.701149 |
| Online Shoppers | single opt Macro-F1 | 1.000000 | 0.000000 | 0.730244 | 0.736701 | 0.460952 | 0.565445 | 0.529412 |
| Online Shoppers | dual 5% budget | 0.949757 | 0.050243 | 0.750588 | 0.740961 | 0.502187 | 0.543860 | 0.600000 |
| SECOM | single default 0.5 | 1.000000 | 0.000000 | 0.595546 | 0.593450 | 0.191190 | 0.238095 | 0.250000 |
| SECOM | single opt Macro-F1 | 1.000000 | 0.000000 | 0.599490 | 0.595157 | 0.199400 | 0.238095 | 0.263158 |
| SECOM | dual 5% budget | 1.000000 | 0.000000 | 0.599490 | 0.595157 | 0.199400 | 0.238095 | 0.263158 |

## Controlled Training Imbalance Smoke Tests

Requested training ratio: minority:majority = 1:10.

| Dataset | Realized Train Majority/Minority | Train Rows | Policy | Coverage | Abstention | Macro-F1 | Balanced Acc. | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Online Shoppers | 5.461135 | 7,398 | single opt Macro-F1 | 1.000000 | 0.000000 | 0.730244 | 0.736701 | 0.460952 |
| Online Shoppers | 5.461135 | 7,398 | dual 5% budget | 0.949757 | 0.050243 | 0.750588 | 0.740961 | 0.502187 |
| SECOM | 10.000000 | 550 | single opt Macro-F1 | 1.000000 | 0.000000 | 0.495807 | 0.496506 | -0.007715 |
| SECOM | 10.000000 | 550 | dual 5% budget | 1.000000 | 0.000000 | 0.495807 | 0.496506 | -0.007715 |

Note: Online Shoppers is naturally about 5.46:1 in the smoke-test training split, so requesting 1:10 does not downsample. Use `--train-ratio 1:5` for an actual controlled downsampling smoke test on that dataset.

## Output Files

- `results_imbalance_selective/smoke_test/online_shoppers_lr_seed7.csv`
- `results_imbalance_selective/smoke_test/secom_lr_seed7.csv`
- `results_imbalance_selective/smoke_test/online_shoppers_lr_seed7_train_1_10.csv`
- `results_imbalance_selective/smoke_test/secom_lr_seed7_train_1_10.csv`
