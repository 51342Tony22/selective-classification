import pandas as pd
import pytest

from common.data_audit import audit_dataframe, prediction_artifact_audit, resolve_target_column


def test_resolve_target_column_rejects_ambiguous_targets():
    df = pd.DataFrame({"fraud_bool": [0, 1], "is_fraud": [0, 1]})

    with pytest.raises(ValueError, match="Multiple target columns"):
        resolve_target_column(df, ["fraud_bool", "is_fraud"])


def test_audit_dataframe_reports_class_balance_and_months():
    df = pd.DataFrame(
        {
            "month": [0, 0, 1, 1],
            "x": [1.0, 2.0, None, 4.0],
            "fraud_bool": [0, 0, 0, 1],
        }
    )

    audit = audit_dataframe(
        df,
        dataset_name="toy_baf",
        target_candidates=["fraud_bool"],
        positive_label=1,
        month_column="month",
    )

    assert audit["rows"] == 4
    assert audit["positives"] == 1
    assert audit["positive_rate"] == 0.25
    assert audit["unique_months"] == 2
    assert audit["mean_missing_rate"] > 0


def test_prediction_artifact_audit_checks_alignment_columns():
    df = pd.DataFrame(
        {
            "row_id": [10, 11, 12],
            "y_true": [0, 1, 0],
            "score": [0.1, 0.8, 0.2],
            "split": ["test", "test", "test"],
            "seed": [7, 7, 7],
            "model": ["lr", "lr", "lr"],
            "calibration": ["none", "none", "none"],
        }
    )

    audit = prediction_artifact_audit(df)

    assert audit["rows"] == 3
    assert audit["positive_rate"] == 1 / 3
    assert audit["score_min"] == 0.1
    assert audit["score_max"] == 0.8
    assert audit["models"] == "lr"

