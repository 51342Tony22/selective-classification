# Phase 2 Method Lock and BAF Scale Plan

## 1. Phase 1 Freeze

Phase 1 has been frozen at:

`results_imbalance_selective/phase1_baseline_50k/`

The freeze includes the Phase 1 CSV files, figures, report, configs, and BAF 50k split assignments. The previous live result directory remains unchanged:

`results_imbalance_selective/main_50k_three_models/`

Important caveat: Phase 1 did not preserve original 1M BAF source row IDs for the 50k sample. It preserved the sampled 50k row indices and temporal split assignment. Future BAF samples should persist original source row IDs.

## 2. Is the Current Free Dual Threshold Asymmetric?

Yes, with a qualification. The Phase 1 dual-threshold implementation searches `low` and `high` freely under an abstention budget, so it is asymmetric in the sense that it does not force equal rejection mass below and above the optimized single-threshold boundary.

However, Phase 1 optimized macro-F1. Phase 2 changes the primary objective to balanced selective risk, so Phase 1 dual-threshold rows should be interpreted only as asymmetric-dual pilot results.

## 3. Symmetric Dual Definition

The symmetric dual threshold is anchored at the optimized single threshold `t*`.

For total abstention budget `b`:

- reject approximately `b / 2` of validation examples immediately below `t*`;
- reject approximately `b / 2` of validation examples immediately above `t*`;
- use the resulting lower and upper thresholds on the test set without re-optimization.

This gives:

`score < lower_threshold -> negative`

`lower_threshold <= score < upper_threshold -> abstain`

`score >= upper_threshold -> positive`

## 4. Four Locked Policy Families

1. `default_single_0_5`: fixed threshold 0.5, 100% coverage.
2. `optimized_single`: threshold selected on validation by balanced selective risk, 100% coverage.
3. `symmetric_dual_{budget}`: equal validation-side budget around `t*`.
4. `asymmetric_dual_{budget}`: free low/high search under the same total budget.

All four policies use the same validation split, test split, model, seed, training distribution, and optimization objective.

## 5. Primary Objective

Primary objective:

`Balanced Selective Risk = 1 - Balanced Accuracy on accepted samples`

The implementation selects thresholds by maximizing validation balanced accuracy on accepted samples.

## 6. Added Metrics

The Phase 2 smoke table records:

- balanced selective risk
- macro-F1
- MCC
- minority precision
- minority recall on accepted samples
- specificity
- total coverage
- positive coverage
- negative coverage
- positive abstention rate
- negative abstention rate
- minority share in abstention
- effective minority recall
- validation coverage
- test coverage
- coverage drift
- optimized single threshold
- lower threshold
- upper threshold
- abstention width
- lower-side abstention
- upper-side abstention
- allocation asymmetry

Effective minority recall is:

`TP on accepted test samples / total positive test samples`

This keeps rejected positive examples in the denominator.

## 7. Smoke Test Result

Smoke test file:

`results_imbalance_selective/phase2_method_smoke/online_lightgbm_seed7.csv`

Scope:

- Dataset: Online Shoppers
- Model: LightGBM
- Seed: 7
- Training distributions: natural, 1:3, 1:5, 1:10
- Policies: default single, optimized single, symmetric dual 1/5/10%, asymmetric dual 1/5/10%

Best balanced selective risk by training distribution:

| Training distribution | Best policy | Balanced selective risk | Macro-F1 | Effective minority recall | Test coverage | Allocation asymmetry |
|---|---|---:|---:|---:|---:|---:|
| natural | symmetric_dual_0.1 | 0.146 | 0.813 | 0.660 | 0.914 | 0.019 |
| 1:3 | symmetric_dual_0.1 | 0.129 | 0.805 | 0.749 | 0.914 | 0.170 |
| 1:5 | symmetric_dual_0.1 | 0.127 | 0.805 | 0.764 | 0.908 | 0.193 |
| 1:10 | symmetric_dual_0.1 | 0.139 | 0.773 | 0.770 | 0.898 | 0.079 |

Initial reading: the symmetric 10% dual policy is the best balanced-risk policy in this smoke test. Asymmetric dual sometimes improves macro-F1 or MCC, but it is not yet the balanced-risk winner in this single-dataset/single-seed check. This is exactly why the symmetric-versus-asymmetric mechanism comparison needs paired evaluation rather than cherry-picking the best row.

## 8. BAF 50k Split Positive Counts

BAF 50k split counts:

| Split | Rows | Positives | Negatives |
|---|---:|---:|---:|
| train, months 0-4 | 33,784 | 338 | 33,446 |
| calibration, month 5 | 5,966 | 70 | 5,896 |
| threshold selection, month 6 | 5,408 | 72 | 5,336 |
| test, month 7 | 4,842 | 71 | 4,771 |

## 9. What 1% Abstention Means

For BAF 50k:

- threshold-selection split: about 54 examples at 1%
- test split: about 48 examples at 1%

For Online Shoppers smoke test:

- threshold-selection split: about 18 examples at 1%
- test split: about 12 examples at 1%

For full BAF natural evaluation:

- month 6 validation: about 1,082 examples at 1%
- month 7 test: about 968 examples at 1%

This matters because 1% budgets on small validation sets can be only a handful of examples, especially for positive-class abstentions.

## 10. Can Full BAF Predictions Be Reused?

Likely yes, pending a final row-level audit.

The old project contains prediction files at:

`/Users/starsfalling/Desktop/Project/baf-rcast-study/outputs/predictions/`

Observed coverage:

- Models: logistic regression, LightGBM, MLP
- Seeds: 7, 17, 177
- Splits: train, calibration, validation, test
- Validation rows: 108,168, month 6
- Test rows: 96,843, month 7
- Columns: `row_id`, `month`, `y_true`, `raw_probability`, `prob_none`, `prob_platt`, `prob_isotonic`, `prob_temperature`

Before reuse, the next audit must verify:

- every model/seed has month 6 and month 7 files;
- row IDs and labels match the current BAF raw file;
- no cost/human-review variables are needed;
- probability columns are calibration-only mappings, not test-fitted transformations.

## 11. Full BAF Natural Experiment Runtime

If reusing old predictions, full BAF natural Phase 2 should not require retraining. Runtime will be dominated by threshold search and metric evaluation.

Expected runtime:

- With current Python-loop dual search and 101 candidates: roughly tens of minutes for all model/seed/calibration combinations.
- With a vectorized or reduced-grid threshold search: likely under 10 minutes.

Storage need is modest because predictions already exist. New Phase 2 threshold/result tables should be well under 100 MB unless bootstrap draws are stored in full.

## 12. 50k Controlled Sampling Plan

For BAF 50k controlled ratios:

- target ratios: 1:3, 1:5, 1:10, 1:20, 1:50
- sampling seeds: at least 3 independent sampling seeds
- validation and test splits fixed
- preserve source row IDs in future sample manifests
- do not duplicate minority examples

The feasibility audit is:

`CONTROLLED_IMBALANCE_FEASIBILITY.md`

The CSV audit is:

`results_imbalance_selective/reports/controlled_imbalance_feasibility.csv`

## 13. Leakage Check

Current Phase 2 smoke test avoids test-driven threshold selection:

- preprocessing is fit only on training rows;
- optimized single and dual thresholds are selected only on the threshold-selection split;
- test data is used only for evaluation;
- symmetric dual thresholds are anchored using validation scores only.

Remaining risk: the old full BAF prediction reuse path must be audited before it is used in the final paper.

## 14. Next Commands

Re-run the Phase 2 smoke test:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase2_smoke_test.py
```

Re-run the controlled-imbalance feasibility audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_controlled_imbalance.py
```

Run tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

Next implementation step:

add a full BAF prediction-reuse audit script that checks all old model/seed/split prediction files against current BAF row IDs, labels, months, and calibration columns.
