from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


UNIT_KEYS = ["dataset", "model", "seed", "training_distribution", "calibration"]
METRICS = [
    "balanced_selective_risk",
    "macro_f1",
    "effective_minority_recall",
    "positive_coverage",
    "negative_coverage",
    "coverage",
]


def build_bee_gate_units(
    results: pd.DataFrame,
    *,
    gate_threshold: float = 1.0,
    budgets: Iterable[float] = (0.05, 0.10),
) -> pd.DataFrame:
    """Evaluate a validation-only BEE gate against untouched test outcomes."""
    required = set(UNIT_KEYS + METRICS)
    required.update(
        {
            "policy_label",
            "policy_family",
            "abstention_budget",
            "validation_boundary_error_enrichment",
            "validation_actual_abstention_rate",
        }
    )
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results missing required columns: {missing}")
    if gate_threshold < 1.0:
        raise ValueError("gate_threshold must be at least 1.0")

    budget_values = np.asarray(list(budgets), dtype=float)
    symmetric = results[
        (results["policy_family"] == "symmetric_dual")
        & results["abstention_budget"].astype(float).isin(budget_values)
    ].copy()
    optimized = results[results["policy_label"] == "optimized_single"].copy()
    if symmetric.empty or optimized.empty:
        raise ValueError("results must contain symmetric dual and optimized-single rows")

    optimized_columns = UNIT_KEYS + METRICS
    merged = symmetric.merge(
        optimized[optimized_columns],
        on=UNIT_KEYS,
        how="left",
        validate="many_to_one",
        suffixes=("_symmetric", "_single"),
    )
    if merged[[f"{metric}_single" for metric in METRICS]].isna().all(axis=1).any():
        raise ValueError("at least one symmetric row has no matching optimized-single row")

    bee = merged["validation_boundary_error_enrichment"].astype(float)
    realized_rejection = merged["validation_actual_abstention_rate"].astype(float)
    gate_active = bee.notna() & np.isfinite(bee) & (bee > gate_threshold) & (realized_rejection > 0.0)

    out = merged[UNIT_KEYS + ["abstention_budget"]].copy()
    out["gate_threshold"] = float(gate_threshold)
    out["validation_bee"] = bee
    out["validation_rejection_rate"] = realized_rejection
    out["gate_active"] = gate_active
    out["selected_policy"] = np.where(gate_active, "symmetric_dual", "optimized_single")
    for metric in METRICS:
        symmetric_values = merged[f"{metric}_symmetric"].astype(float)
        single_values = merged[f"{metric}_single"].astype(float)
        out[f"symmetric_{metric}"] = symmetric_values
        out[f"single_{metric}"] = single_values
        out[f"gated_{metric}"] = np.where(gate_active, symmetric_values, single_values)

    symmetric_delta = out["symmetric_balanced_selective_risk"] - out["single_balanced_selective_risk"]
    gated_delta = out["gated_balanced_selective_risk"] - out["single_balanced_selective_risk"]
    oracle_risk = np.minimum(
        out["symmetric_balanced_selective_risk"],
        out["single_balanced_selective_risk"],
    )
    out["symmetric_delta_bsr"] = symmetric_delta
    out["gated_delta_bsr"] = gated_delta
    out["gate_regret_vs_oracle"] = out["gated_balanced_selective_risk"] - oracle_risk
    out["test_symmetric_benefits"] = symmetric_delta < 0.0
    out["gate_decision_correct"] = gate_active == out["test_symmetric_benefits"]
    return out


def summarize_bee_gate(units: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "abstention_budget",
        "validation_bee",
        "gate_active",
        "gate_decision_correct",
        "symmetric_delta_bsr",
        "gated_delta_bsr",
        "gate_regret_vs_oracle",
        "gated_effective_minority_recall",
        "single_effective_minority_recall",
    }
    missing = sorted(required - set(units.columns))
    if missing:
        raise ValueError(f"gate units missing required columns: {missing}")
    summary = (
        units.groupby(["dataset", "abstention_budget"], dropna=False)
        .agg(
            n_units=("gate_active", "size"),
            mean_validation_bee=("validation_bee", "mean"),
            activation_rate=("gate_active", "mean"),
            decision_accuracy=("gate_decision_correct", "mean"),
            mean_symmetric_delta_bsr=("symmetric_delta_bsr", "mean"),
            mean_gated_delta_bsr=("gated_delta_bsr", "mean"),
            mean_gate_regret_vs_oracle=("gate_regret_vs_oracle", "mean"),
            mean_gated_emr=("gated_effective_minority_recall", "mean"),
            mean_single_emr=("single_effective_minority_recall", "mean"),
        )
        .reset_index()
    )
    summary["mean_gated_delta_emr"] = summary["mean_gated_emr"] - summary["mean_single_emr"]
    return summary
