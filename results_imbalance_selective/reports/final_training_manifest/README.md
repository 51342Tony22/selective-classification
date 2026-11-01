# Final Training Manifest

This manifest is a dry run. It does not train models and does not run the full experimental matrix.

## Stop Point

Before launching the full training jobs below, pause and confirm with the user.

## Dataset Readiness

| dataset | config | local_path | local_path_exists | split_assignments_path | split_assignments_exists | split_type |
| --- | --- | --- | --- | --- | --- | --- |
| baf | configs/imbalance_selective/baf.yaml | /Users/starsfalling/Documents/AT/Project/human-deferral-worthwhile-new-direction/data/raw/baf/Base.parquet | True | /Users/starsfalling/Documents/AT/Project/human-deferral-worthwhile-new-direction/data/interim/baf_split_assignments.parquet | True | temporal |
| secom | configs/imbalance_selective/secom.yaml | /Users/starsfalling/Documents/AT/Project/human-deferral-worthwhile-new-direction/data/raw/secom | True |  |  | repeated_stratified_cv |
| online_shoppers | configs/imbalance_selective/online_shoppers.yaml | /Users/starsfalling/Documents/AT/Project/human-deferral-worthwhile-new-direction/data/raw/online_shoppers/online_shoppers_intention.csv | True |  |  | stratified_train_calibration_validation_test |

## Matrix Summary

| dataset | experiment | training_jobs | evaluation_jobs |
| --- | --- | --- | --- |
| baf | controlled_imbalance | 36 | 1152 |
| baf | natural_imbalance | 9 | 288 |
| online_shoppers | controlled_imbalance | 36 | 1152 |
| online_shoppers | natural_imbalance | 9 | 288 |
| secom | controlled_imbalance | 540 | 17280 |
| secom | natural_imbalance | 135 | 4320 |

## Totals

- Training jobs: 765
- Evaluation jobs: 24480

## Files

- `training_jobs.csv`
- `evaluation_jobs.csv`
- `dataset_readiness.csv`
- `matrix_summary.csv`
