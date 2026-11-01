# Final Phase 2 Results

Run date: 2026-07-21

This bundle contains the final tie-safe 201-candidate analysis for the new-B paper direction:

- default 0.5 versus an optimized full-coverage threshold;
- optimized single versus symmetric and asymmetric dual thresholds;
- matched-validation-coverage comparison of reject allocation;
- controlled training-ratio trends;
- class-conditional coverage and effective minority recall (EMR);
- 41/101/201 threshold-grid sensitivity;
- full-BAF calibration ranking, tie, and selected-set audits.

## Result Files

- `all_phase2_results.csv`: 1,782 policy rows.
- `summary_by_policy.csv`: 594 grouped summary rows.
- `paired_differences.csv`: 1,944 paired comparison rows.
- `key_tables/`: descriptive best rows and paired-comparison means.
- `baf_full_prediction_audit.md`: audit of reused full-BAF predictions.
- `paper_artifacts/paired_unit_bootstrap_ci.csv`: 5,000-resample paired-unit intervals.
- `paper_artifacts/threshold_grid_sensitivity_summary.csv`: 41/101/201 comparison.
- `paper_artifacts/revision_audit/`: controlled-ratio, single-threshold, coverage, tie-budget, and calibration audits.

## Budget and Tie Rule

All primary dual policies treat 1%, 5%, and 10% as maximum validation rejection rates. Symmetric rejection allocates `floor(n * budget)` cases approximately equally below and above the optimized boundary. Equal score values are indivisible: a complete tie block is included only if it fits in the remaining allocation; otherwise the policy underuses its budget. Asymmetric search also enforces the maximum directly.

After this correction, the maximum validation-budget overshoot among primary symmetric and asymmetric policies is negative (`-0.000004` after floating-point rounding). The supplementary matched policy may differ from the symmetric target by at most 0.002, as specified.

If an accepted subset lacks a class, that class recall is explicitly set to zero. No final accepted subset triggers this convention: minimum observed positive and negative coverage are 0.762 and 0.836, respectively.

## Descriptive Best Primary Configurations

Matched policies are excluded. These are descriptive selected rows, not inferential evidence.

| Dataset | Model | Train/calibration | Policy | BSR | Macro-F1 | EMR | Coverage |
|---|---|---|---|---:|---:|---:|---:|
| BAF full | LightGBM | natural full/none | asymmetric 10% | 0.161 | 0.535 | 0.737 | 0.909 |
| BAF 50k | LightGBM | natural/none | asymmetric 10% | 0.173 | 0.499 | 0.765 | 0.904 |
| Online Shoppers | LightGBM | 1:3/none | symmetric 10% | 0.118 | 0.808 | 0.778 | 0.908 |
| SECOM | LightGBM | natural/none | symmetric 5% | 0.325 | 0.521 | 0.587 | 0.964 |

## Default Versus Optimized Single

Values average all experimental units within each dataset. Delta is optimized minus default.

| Dataset | Default BSR | Optimized BSR | Delta BSR | Mean optimized threshold |
|---|---:|---:|---:|---:|
| BAF full | 0.491 | 0.195 | -0.296 | 0.011 |
| BAF 50k | 0.427 | 0.270 | -0.156 | 0.101 |
| Online Shoppers | 0.278 | 0.185 | -0.093 | 0.163 |
| SECOM | 0.480 | 0.439 | -0.040 | 0.172 |

Optimizing balanced accuracy raises mean minority recall substantially but can lower macro-F1 because F1 is not the selection objective. The optimized threshold is therefore the required baseline for assessing any additional benefit from rejection.

## Dual Versus Optimized Single at 10%

Negative delta BSR favors the dual policy. Intervals summarize stability across paired configuration units sharing test observations; they are not population-level significance tests.

| Dataset | Symmetric delta BSR (95% CI) | Asymmetric delta BSR (95% CI) |
|---|---:|---:|
| BAF full | -0.0238 [-0.0256, -0.0217] | -0.0270 [-0.0281, -0.0258] |
| BAF 50k | -0.0177 [-0.0210, -0.0143] | -0.0187 [-0.0232, -0.0143] |
| Online Shoppers | -0.0295 [-0.0321, -0.0271] | -0.0292 [-0.0321, -0.0264] |
| SECOM | -0.0040 [-0.0113, 0.0027] | -0.0020 [-0.0047, 0.0005] |

The dual-threshold result remains stable on full BAF, BAF 50k, and Online Shoppers. SECOM remains inconclusive.

## Symmetric Versus Asymmetric

### Maximum-Budget Comparison

| Dataset | Asymmetric minus symmetric BSR | Bootstrap 95% CI | Mean test-coverage delta |
|---|---:|---:|---:|
| BAF full | -0.0032 | [-0.0048, -0.0017] | -0.0070 |
| BAF 50k | -0.0010 | [-0.0056, 0.0030] | 0.0091 |
| Online Shoppers | 0.0004 | [-0.0029, 0.0035] | -0.0000 |
| SECOM | 0.0020 | [-0.0053, 0.0095] | 0.0431 |

Full BAF favors asymmetry under the maximum-budget comparison, but asymmetry also rejects about 0.7 percentage points more test cases. The other intervals include zero.

### Matched-Coverage Comparison

| Dataset | Mean delta BSR | Bootstrap 95% CI | Validation coverage delta | Test coverage delta |
|---|---:|---:|---:|---:|
| BAF full | -0.0006 | [-0.0016, 0.0004] | -0.0000 | -0.0001 |
| BAF 50k | -0.0009 | [-0.0054, 0.0030] | -0.0001 | -0.0015 |
| Online Shoppers | -0.0001 | [-0.0034, 0.0032] | -0.0004 | -0.0018 |
| SECOM | -0.0005 | [-0.0081, 0.0074] | 0.0000 | 0.0006 |

All matched risk intervals include zero. Mean test-coverage gaps remain below 0.002 even though validation matching does not mathematically guarantee test matching. The evidence supports a reject-region benefit, not a stable benefit from asymmetric allocation.

## Controlled Training Imbalance

Each row averages nine model-seed units at a fixed training total. No monotonic pattern appears.

| Dataset | Ratio | Symmetric-single | Asymmetric-single | Asymmetric-symmetric |
|---|---:|---:|---:|---:|
| BAF 50k | 1:3 | -0.0210 | -0.0247 | -0.0038 |
| BAF 50k | 1:5 | -0.0186 | -0.0165 | 0.0021 |
| BAF 50k | 1:10 | -0.0242 | -0.0173 | 0.0069 |
| BAF 50k | 1:20 | -0.0102 | -0.0103 | -0.0001 |
| BAF 50k | 1:50 | -0.0122 | -0.0192 | -0.0070 |
| Online Shoppers | 1:3 | -0.0287 | -0.0264 | 0.0023 |
| Online Shoppers | 1:5 | -0.0278 | -0.0342 | -0.0064 |
| Online Shoppers | 1:10 | -0.0320 | -0.0292 | 0.0028 |
| SECOM | 1:3 | -0.0018 | -0.0042 | -0.0025 |
| SECOM | 1:5 | -0.0002 | -0.0017 | -0.0015 |
| SECOM | 1:10 | -0.0159 | 0.0040 | 0.0199 |

Dual-threshold gains do not increase monotonically with imbalance severity, and the asymmetric-minus-symmetric sign alternates. Dataset, model, and score geometry matter more than the training ratio alone.

## Class-Conditional Coverage

At 10%, positive coverage is generally below or close to negative coverage. For full BAF, asymmetric rejection has total coverage 0.913, positive coverage 0.895, negative coverage 0.913, and EMR 0.714 versus optimized-single EMR 0.783. Similar EMR reductions occur on every dataset. The full table is `paper_artifacts/revision_audit/class_conditional_coverage.csv`.

## Calibration Audit

Platt and temperature scaling have rank correlation 1.000 with uncalibrated scores and produce identical accepted/rejected row-ID sets for every audited 10% policy, model, seed, and split. They change numerical thresholds but not selection quality.

Isotonic calibration has minimum rank correlation 0.995 but only 326 unique scores on average, with a largest tie block of 16,787 rows. Minimum accepted- and rejected-set Jaccard similarities versus no calibration are 0.813 and 0.032. Under tie-safe budgets, the best primary LightGBM mean BSR is 0.165 for isotonic versus 0.161 without calibration. The earlier apparent isotonic improvement was a budget-overrun artifact caused by splitting a rank target through a large tie block.

Audit CSVs:

- `paper_artifacts/revision_audit/calibration_score_audit.csv`
- `paper_artifacts/revision_audit/calibration_policy_set_audit.csv`
- `paper_artifacts/revision_audit/tie_budget_audit.csv`

## Reproduction Commands

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_baf_full_predictions.py
PYTHONPATH=src .venv/bin/python scripts/run_phase2_baf_full_natural.py \
  --max-candidates 201 --include-matched-coverage
PYTHONPATH=src .venv/bin/python scripts/run_phase2_final_local.py \
  --max-candidates 201 --include-matched-coverage
PYTHONPATH=src .venv/bin/python scripts/summarize_phase2_final.py \
  --inputs results_imbalance_selective/final_phase2/local_50k_controlled_results.csv \
           results_imbalance_selective/final_phase2/baf_full_natural_results.csv \
  --output-dir results_imbalance_selective/final_phase2
PYTHONPATH=src .venv/bin/python scripts/create_phase2_paper_artifacts.py --iterations 5000
PYTHONPATH=src .venv/bin/python scripts/run_threshold_grid_sensitivity.py
PYTHONPATH=src .venv/bin/python scripts/create_revision_audit_artifacts.py
PYTHONPATH=src .venv/bin/python -m pytest -q
```
