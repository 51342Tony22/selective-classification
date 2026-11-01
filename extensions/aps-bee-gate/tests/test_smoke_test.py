import numpy as np
import pandas as pd

from common.data_split import SplitIndices
from imbalance_selective.datasets import DatasetBundle
from imbalance_selective.smoke_test import run_smoke_test_on_bundle


def test_run_smoke_test_on_bundle_outputs_expected_policy_rows():
    rng = np.random.default_rng(7)
    n = 220
    y = np.array([0] * 160 + [1] * 60)
    x_signal = y + rng.normal(0.0, 0.2, size=n)
    frame = pd.DataFrame(
        {
            "x_signal": x_signal,
            "group": ["a" if i % 2 else "b" for i in range(n)],
            "label": y,
        }
    )
    bundle = DatasetBundle(
        name="toy",
        frame=frame,
        features=frame.drop(columns=["label"]),
        y=pd.Series(y),
        target="label",
        metadata={},
    )
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    split = SplitIndices(
        train=np.sort(np.concatenate([neg[:90], pos[:30]])),
        calibration=np.sort(np.concatenate([neg[90:110], pos[30:40]])),
        threshold_selection=np.sort(np.concatenate([neg[110:130], pos[40:50]])),
        test=np.sort(np.concatenate([neg[130:160], pos[50:60]])),
    )

    results = run_smoke_test_on_bundle(bundle, split, seed=7, abstention_budget=0.05, max_candidates=None)

    assert set(results["policy_label"]) == {"single_default_0_5", "single_opt_macro_f1", "dual_budget_0.05"}
    assert set(results["model"]) == {"logistic_regression"}
    assert set(results["calibration"]) == {"none"}
    assert {
        "coverage",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "minority_recall",
        "train_positive_n",
        "train_negative_n",
        "realized_train_majority_per_minority",
    }.issubset(results.columns)
    dual = results.loc[results["policy_label"] == "dual_budget_0.05"].iloc[0]
    assert dual["abstention_rate"] <= 0.05 + 1e-12


def test_run_smoke_test_on_bundle_can_request_controlled_training_imbalance():
    rng = np.random.default_rng(8)
    n = 220
    y = np.array([0] * 160 + [1] * 60)
    frame = pd.DataFrame(
        {
            "x_signal": y + rng.normal(0.0, 0.2, size=n),
            "label": y,
        }
    )
    bundle = DatasetBundle(
        name="toy",
        frame=frame,
        features=frame.drop(columns=["label"]),
        y=pd.Series(y),
        target="label",
        metadata={},
    )
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y == 1)
    split = SplitIndices(
        train=np.sort(np.concatenate([neg[:100], pos[:20]])),
        calibration=np.sort(np.concatenate([neg[100:120], pos[20:30]])),
        threshold_selection=np.sort(np.concatenate([neg[120:140], pos[30:45]])),
        test=np.sort(np.concatenate([neg[140:160], pos[45:60]])),
    )

    results = run_smoke_test_on_bundle(bundle, split, seed=7, train_ratio="1:2", max_candidates=None)

    first = results.iloc[0]
    assert first["requested_train_ratio"] == "1:2"
    assert first["train_positive_n"] == 20
    assert first["train_negative_n"] == 40
    assert first["realized_train_majority_per_minority"] == 2.0
