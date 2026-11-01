from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features: pd.DataFrame, *, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("no feature columns available after dropping target/leakage columns")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logistic_regression_pipeline(
    features: pd.DataFrame,
    *,
    seed: int = 7,
    class_weight: str | dict | None = None,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(
    features: pd.DataFrame,
    *,
    seed: int = 7,
    class_weight: str | dict | None = None,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features, scale_numeric=False)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=160,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight=class_weight,
                    random_state=seed,
                    n_jobs=4,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_mlp_pipeline(
    features: pd.DataFrame,
    *,
    seed: int = 7,
    class_weight: str | dict | None = None,
) -> Pipeline:
    if class_weight is not None:
        raise ValueError("MLPClassifier does not support class_weight in this pipeline")
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features, scale_numeric=True)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    early_stopping=True,
                    n_iter_no_change=8,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_model_pipeline(
    model_name: str,
    features: pd.DataFrame,
    *,
    seed: int = 7,
    class_weight: str | dict | None = None,
) -> Pipeline:
    if model_name == "logistic_regression":
        return build_logistic_regression_pipeline(features, seed=seed, class_weight=class_weight)
    if model_name == "lightgbm":
        return build_lightgbm_pipeline(features, seed=seed, class_weight=class_weight)
    if model_name == "mlp":
        return build_mlp_pipeline(features, seed=seed, class_weight=class_weight)
    raise ValueError(f"unsupported model: {model_name}")
