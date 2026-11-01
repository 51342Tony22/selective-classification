from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imbalance_selective.preprocessing import build_model_pipeline


@pytest.mark.parametrize("model_name", ["logistic_regression", "lightgbm", "mlp"])
def test_model_pipeline_predicts_probabilities(model_name: str) -> None:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        {
            "numeric_a": rng.normal(size=80),
            "numeric_b": rng.normal(size=80),
            "category": np.where(np.arange(80) % 3 == 0, "x", "y"),
        }
    )
    y = ((features["numeric_a"] + (features["category"] == "x").astype(float)) > 0.4).astype(int)

    model = build_model_pipeline(model_name, features, seed=7)
    model.fit(features, y)
    probabilities = model.predict_proba(features.iloc[:5])

    assert probabilities.shape == (5, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_model_pipeline_rejects_unknown_model() -> None:
    features = pd.DataFrame({"x": [0, 1], "z": [1, 0]})
    with pytest.raises(ValueError, match="unsupported model"):
        build_model_pipeline("unknown", features)
