from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CONFIG_BY_DATASET = {
    "baf": "configs/imbalance_selective/baf.yaml",
    "secom": "configs/imbalance_selective/secom.yaml",
    "online_shoppers": "configs/imbalance_selective/online_shoppers.yaml",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def build_training_jobs(matrix_config: dict[str, Any], *, root: str | Path = ".") -> pd.DataFrame:
    root_path = Path(root)
    seeds = matrix_config.get("project", {}).get("seeds", [7, 17, 177])
    datasets = matrix_config.get("datasets", ["baf", "secom", "online_shoppers"])
    models = matrix_config.get("models", ["logistic_regression", "lightgbm", "mlp"])
    ratios = matrix_config.get("controlled_imbalance", {}).get("train_ratios", [])

    rows: list[dict[str, Any]] = []
    for dataset_key in datasets:
        dataset_config = load_yaml(root_path / CONFIG_BY_DATASET[dataset_key])
        split = dataset_config.get("dataset", {}).get("split", {})
        fold_ids = _fold_ids(split)
        for model in models:
            for seed in seeds:
                for fold_id in fold_ids:
                    rows.append(
                        _training_row(
                            dataset_key,
                            dataset_config,
                            model,
                            int(seed),
                            fold_id,
                            experiment="natural_imbalance",
                            train_ratio="",
                        )
                    )
                    for ratio in ratios:
                        rows.append(
                            _training_row(
                                dataset_key,
                                dataset_config,
                                model,
                                int(seed),
                                fold_id,
                                experiment="controlled_imbalance",
                                train_ratio=ratio,
                            )
                        )
    return pd.DataFrame(rows)


def build_evaluation_jobs(matrix_config: dict[str, Any], training_jobs: pd.DataFrame) -> pd.DataFrame:
    calibrations = matrix_config.get("calibration", {}).get("methods", ["none", "platt", "isotonic", "temperature"])
    single_baselines = matrix_config.get("thresholds", {}).get("single_baselines", [])
    budgets = matrix_config.get("thresholds", {}).get("abstention_budgets", [])

    rows: list[dict[str, Any]] = []
    for _, train in training_jobs.iterrows():
        base = train.to_dict()
        for calibration in calibrations:
            for baseline in single_baselines:
                rows.append(
                    {
                        **base,
                        "calibration": calibration,
                        "policy_family": "single",
                        "policy": baseline,
                        "abstention_budget": "",
                    }
                )
            for budget in budgets:
                rows.append(
                    {
                        **base,
                        "calibration": calibration,
                        "policy_family": "dual",
                        "policy": f"dual_budget_{budget:g}",
                        "abstention_budget": budget,
                    }
                )
    return pd.DataFrame(rows)


def dataset_readiness(root: str | Path = ".") -> pd.DataFrame:
    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    for dataset_key, config_rel in CONFIG_BY_DATASET.items():
        config = load_yaml(root_path / config_rel)
        dataset = config["dataset"]
        local_path = root_path / dataset["local_path"]
        split_path = dataset.get("split_assignments_path")
        split_abs = root_path / split_path if split_path else None
        rows.append(
            {
                "dataset": dataset_key,
                "config": config_rel,
                "local_path": str(local_path),
                "local_path_exists": local_path.exists(),
                "split_assignments_path": str(split_abs) if split_abs else "",
                "split_assignments_exists": bool(split_abs.exists()) if split_abs else "",
                "split_type": dataset.get("split", {}).get("type", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_matrix(training_jobs: pd.DataFrame, evaluation_jobs: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        training_jobs.groupby(["dataset", "experiment"], dropna=False)
        .size()
        .reset_index(name="training_jobs")
    )
    eval_grouped = (
        evaluation_jobs.groupby(["dataset", "experiment"], dropna=False)
        .size()
        .reset_index(name="evaluation_jobs")
    )
    return grouped.merge(eval_grouped, on=["dataset", "experiment"], how="left")


def _fold_ids(split_config: dict[str, Any]) -> list[str]:
    if split_config.get("type") != "repeated_stratified_cv":
        return [""]
    folds = int(split_config.get("folds", 5))
    repeats = int(split_config.get("repeats", 3))
    return [f"fold_{idx}" for idx in range(folds * repeats)]


def _training_row(
    dataset_key: str,
    dataset_config: dict[str, Any],
    model: str,
    seed: int,
    fold_id: str,
    *,
    experiment: str,
    train_ratio: str,
) -> dict[str, Any]:
    dataset = dataset_config["dataset"]
    return {
        "dataset": dataset_key,
        "dataset_name": dataset["name"],
        "domain": dataset.get("domain", ""),
        "split_type": dataset.get("split", {}).get("type", ""),
        "model": model,
        "seed": seed,
        "fold_id": fold_id,
        "experiment": experiment,
        "train_ratio": train_ratio,
        "will_fit_model": True,
        "final_training": False,
    }

