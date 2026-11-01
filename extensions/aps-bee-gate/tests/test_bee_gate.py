from __future__ import annotations

import pandas as pd
import pytest

from imbalance_selective.bee_gate import build_bee_gate_units, summarize_bee_gate


def _row(policy: str, family: str, budget: float, bsr: float, bee: float) -> dict:
    return {
        "dataset": "toy",
        "model": "lr",
        "seed": 7,
        "training_distribution": "natural",
        "calibration": "none",
        "policy_label": policy,
        "policy_family": family,
        "abstention_budget": budget,
        "validation_boundary_error_enrichment": bee,
        "validation_actual_abstention_rate": budget,
        "balanced_selective_risk": bsr,
        "macro_f1": 0.7,
        "effective_minority_recall": 0.6,
        "positive_coverage": 0.9,
        "negative_coverage": 0.9,
        "coverage": 0.9,
    }


def test_bee_gate_uses_only_enriched_symmetric_policy():
    rows = [
        _row("optimized_single", "optimized_single", 0.0, 0.30, float("nan")),
        _row("symmetric_dual_0.05", "symmetric_dual", 0.05, 0.25, 1.5),
        _row("symmetric_dual_0.1", "symmetric_dual", 0.10, 0.35, 0.9),
    ]

    units = build_bee_gate_units(pd.DataFrame(rows), gate_threshold=1.0)

    active = units.set_index("abstention_budget")["gate_active"].to_dict()
    assert active == {0.05: True, 0.10: False}
    gated = units.set_index("abstention_budget")["gated_balanced_selective_risk"].to_dict()
    assert gated == {0.05: 0.25, 0.10: 0.30}
    assert units["gate_decision_correct"].all()


def test_bee_gate_summary_reports_test_independent_selection():
    rows = [
        _row("optimized_single", "optimized_single", 0.0, 0.30, float("nan")),
        _row("symmetric_dual_0.1", "symmetric_dual", 0.10, 0.25, 1.5),
    ]
    units = build_bee_gate_units(pd.DataFrame(rows), gate_threshold=1.0, budgets=(0.10,))
    summary = summarize_bee_gate(units)

    assert summary.loc[0, "activation_rate"] == 1.0
    assert summary.loc[0, "mean_gated_delta_bsr"] == pytest.approx(-0.05)
