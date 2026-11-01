import numpy as np

from common.bootstrap import paired_bootstrap_metric_difference, paired_bootstrap_table


def test_paired_bootstrap_metric_difference_reports_delta_b_minus_a():
    y = np.array([0, 0, 0, 1, 1, 1])
    decisions_a = np.array([0, 1, 0, 0, 1, 1])
    decisions_b = np.array([0, 0, 0, 1, 1, 1])

    out = paired_bootstrap_metric_difference(
        y,
        decisions_a,
        decisions_b,
        metric="macro_f1",
        iterations=50,
        seed=7,
    )

    assert out["observed_b"] > out["observed_a"]
    assert out["observed_delta_b_minus_a"] > 0
    assert out["ci_low"] <= out["ci_high"]


def test_paired_bootstrap_table_keeps_metric_order():
    y = np.array([0, 0, 1, 1])
    decisions_a = np.array([0, 1, 0, 1])
    decisions_b = np.array([0, 0, 1, 1])

    rows = paired_bootstrap_table(
        y,
        decisions_a,
        decisions_b,
        metrics=("macro_f1", "balanced_accuracy"),
        iterations=20,
        seed=7,
    )

    assert [row["metric"] for row in rows] == ["macro_f1", "balanced_accuracy"]

