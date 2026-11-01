# Full BAF Prediction Reuse Audit

Overall status: PASS

| model | seed | split | rows | positives | months | status |
| --- | ---: | --- | ---: | ---: | --- | --- |
| logistic_regression | 7 | validation | 108168 | 1450 | 6 | ok |
| logistic_regression | 7 | test | 96843 | 1428 | 7 | ok |
| logistic_regression | 17 | validation | 108168 | 1450 | 6 | ok |
| logistic_regression | 17 | test | 96843 | 1428 | 7 | ok |
| logistic_regression | 177 | validation | 108168 | 1450 | 6 | ok |
| logistic_regression | 177 | test | 96843 | 1428 | 7 | ok |
| lightgbm | 7 | validation | 108168 | 1450 | 6 | ok |
| lightgbm | 7 | test | 96843 | 1428 | 7 | ok |
| lightgbm | 17 | validation | 108168 | 1450 | 6 | ok |
| lightgbm | 17 | test | 96843 | 1428 | 7 | ok |
| lightgbm | 177 | validation | 108168 | 1450 | 6 | ok |
| lightgbm | 177 | test | 96843 | 1428 | 7 | ok |
| mlp | 7 | validation | 108168 | 1450 | 6 | ok |
| mlp | 7 | test | 96843 | 1428 | 7 | ok |
| mlp | 17 | validation | 108168 | 1450 | 6 | ok |
| mlp | 17 | test | 96843 | 1428 | 7 | ok |
| mlp | 177 | validation | 108168 | 1450 | 6 | ok |
| mlp | 177 | test | 96843 | 1428 | 7 | ok |

The audit checks file existence, required probability columns, unique row IDs, expected month, and label alignment against the current BAF raw parquet.