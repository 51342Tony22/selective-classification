from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from common.data_audit import normalize_binary_target, resolve_target_column


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    features: pd.DataFrame
    y: pd.Series
    target: str
    metadata: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def load_dataset_from_config(config: dict[str, Any]) -> DatasetBundle:
    dataset = config.get("dataset", {})
    name = dataset.get("name")
    if not name:
        raise ValueError("config missing dataset.name")

    if name == "baf_base":
        return load_baf_base(config)
    if name == "secom":
        return load_secom(config)
    if name == "online_shoppers_purchasing_intention":
        return load_online_shoppers(config)
    raise ValueError(f"unsupported dataset: {name}")


def load_baf_base(config: dict[str, Any]) -> DatasetBundle:
    dataset = config["dataset"]
    path = _required_local_path(dataset)
    frame = load_table(path)
    target = resolve_target_column(frame, [dataset.get("target", "fraud_bool"), "fraud_bool", "is_fraud"])
    y = normalize_binary_target(frame[target], positive_label=dataset.get("positive_label", 1))
    month_column = dataset.get("month_column", "month")
    if month_column not in frame.columns:
        raise ValueError(f"BAF month column missing: {month_column}")

    feature_frame = frame.drop(columns=[target])
    if not dataset.get("include_month_feature", False) and month_column in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=[month_column])

    return DatasetBundle(
        name="baf_base",
        frame=frame,
        features=feature_frame,
        y=y,
        target=target,
        metadata={"path": str(path), "month_column": month_column},
    )


def load_online_shoppers(config: dict[str, Any]) -> DatasetBundle:
    dataset = config["dataset"]
    path = _required_local_path(dataset)
    frame = load_table(path)
    target = resolve_target_column(frame, [dataset.get("target", "Revenue"), "Revenue"])
    positive_label = dataset.get("positive_label", True)
    y = normalize_binary_target(_coerce_bool_like(frame[target]), positive_label=positive_label)
    features = frame.drop(columns=[target])
    return DatasetBundle(
        name="online_shoppers_purchasing_intention",
        frame=frame,
        features=features,
        y=y,
        target=target,
        metadata={"path": str(path)},
    )


def load_secom(config: dict[str, Any]) -> DatasetBundle:
    dataset = config["dataset"]
    path = _required_local_path(dataset)

    if path.is_dir():
        frame = _load_secom_directory(path)
    else:
        frame = load_table(path)

    target = resolve_target_column(frame, [dataset.get("target", "label"), "label", "target", "class"])
    y = normalize_binary_target(frame[target], positive_label=dataset.get("positive_label", 1))
    features = frame.drop(columns=[target])
    return DatasetBundle(
        name="secom",
        frame=frame,
        features=features,
        y=y,
        target=target,
        metadata={"path": str(path)},
    )


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".tsv", ".txt", ".data"}:
        return pd.read_csv(p, sep=None, engine="python")
    raise ValueError(f"unsupported data file extension: {p.suffix}")


def _required_local_path(dataset_config: dict[str, Any]) -> Path:
    raw_path = dataset_config.get("local_path")
    if not raw_path:
        raise FileNotFoundError(
            f"dataset {dataset_config.get('name', '<unknown>')} has no local_path configured; "
            "provide a local file before running smoke tests"
        )
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"configured local_path does not exist: {path}")
    return path


def _load_secom_directory(path: Path) -> pd.DataFrame:
    data_candidates = [path / "secom.data", path / "secom.csv", path / "data.csv"]
    label_candidates = [path / "secom_labels.data", path / "secom_labels.csv", path / "labels.csv"]
    data_path = next((candidate for candidate in data_candidates if candidate.exists()), None)
    label_path = next((candidate for candidate in label_candidates if candidate.exists()), None)
    if data_path is None or label_path is None:
        raise FileNotFoundError(
            "SECOM directory must contain secom.data and secom_labels.data, "
            "or equivalent secom.csv/data.csv and labels files"
        )

    features = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN", "nan", "?"])
    labels = pd.read_csv(label_path, sep=r"\s+", header=None)
    if len(features) != len(labels):
        raise ValueError(f"SECOM feature rows ({len(features)}) do not match label rows ({len(labels)})")
    features = features.add_prefix("feature_")
    label_frame = pd.DataFrame({"label": labels.iloc[:, 0].replace({-1: 0, 1: 1}).astype(int)})
    return pd.concat([features, label_frame], axis=1)


def _coerce_bool_like(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    if set(lowered.dropna().unique()).issubset({"true", "false"}):
        return lowered.map({"true": True, "false": False})
    return series
