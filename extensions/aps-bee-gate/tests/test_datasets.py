import pandas as pd
import pytest

from imbalance_selective.datasets import load_dataset_from_config


def test_load_online_shoppers_from_local_csv(tmp_path):
    path = tmp_path / "online.csv"
    pd.DataFrame(
        {
            "Administrative": [1, 2, 3, 4],
            "Month": ["Jan", "Feb", "Mar", "Apr"],
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Returning_Visitor", "Other"],
            "Revenue": [False, False, True, False],
        }
    ).to_csv(path, index=False)

    bundle = load_dataset_from_config(
        {
            "dataset": {
                "name": "online_shoppers_purchasing_intention",
                "local_path": str(path),
                "target": "Revenue",
                "positive_label": True,
            }
        }
    )

    assert bundle.name == "online_shoppers_purchasing_intention"
    assert bundle.y.tolist() == [0, 0, 1, 0]
    assert "Revenue" not in bundle.features.columns


def test_load_baf_drops_month_from_main_features(tmp_path):
    path = tmp_path / "Base.csv"
    pd.DataFrame(
        {
            "month": list(range(8)),
            "feature": range(8),
            "fraud_bool": [0, 0, 0, 0, 0, 0, 0, 1],
        }
    ).to_csv(path, index=False)

    bundle = load_dataset_from_config(
        {
            "dataset": {
                "name": "baf_base",
                "local_path": str(path),
                "target": "fraud_bool",
                "month_column": "month",
                "positive_label": 1,
            }
        }
    )

    assert bundle.metadata["month_column"] == "month"
    assert "month" not in bundle.features.columns
    assert "fraud_bool" not in bundle.features.columns


def test_missing_local_path_is_explicit():
    with pytest.raises(FileNotFoundError, match="no local_path configured"):
        load_dataset_from_config({"dataset": {"name": "secom", "local_path": None}})


def test_load_aps_combines_official_partitions_and_drops_marker(tmp_path):
    header = "class,aa_000,ab_000\n"
    preamble = "description\n" * 3
    training_path = tmp_path / "aps_training.csv"
    test_path = tmp_path / "aps_test.csv"
    training_path.write_text(preamble + header + "neg,1,na\npos,2,3\n", encoding="utf-8")
    test_path.write_text(preamble + header + "neg,4,5\npos,6,7\n", encoding="utf-8")

    bundle = load_dataset_from_config(
        {
            "dataset": {
                "name": "aps_failure",
                "training_path": str(training_path),
                "test_path": str(test_path),
                "target": "class",
                "positive_label": "pos",
            }
        }
    )

    assert bundle.y.tolist() == [0, 1, 0, 1]
    assert bundle.metadata["official_train_rows"] == 2
    assert bundle.metadata["official_test_rows"] == 2
    assert "__official_partition__" not in bundle.features.columns
    assert bundle.features["ab_000"].isna().sum() == 1
