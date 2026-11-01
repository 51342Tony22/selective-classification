from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_target_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    names = list(dict.fromkeys(candidates))
    found = [name for name in names if name in df.columns]
    if not found:
        raise ValueError(f"No target column found. Expected one of {names}")
    if len(found) > 1:
        raise ValueError(f"Multiple target columns found: {found}")
    return found[0]


def normalize_binary_target(series: pd.Series, *, positive_label: Any = 1) -> pd.Series:
    values = set(series.dropna().unique().tolist())
    if positive_label not in values:
        raise ValueError(f"positive_label {positive_label!r} not found in target values {sorted(values)}")
    if len(values) != 2:
        raise ValueError(f"target must contain exactly two classes; found {sorted(values)}")
    return (series == positive_label).astype(int)


def column_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for column in df.columns:
        s = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(s.dtype),
                "missing_rate": float(s.isna().mean()) if n else 0.0,
                "unique_values": int(s.nunique(dropna=True)),
                "is_constant": bool(s.nunique(dropna=False) <= 1),
            }
        )
    return pd.DataFrame(rows)


def class_balance(y_binary: pd.Series) -> dict[str, float]:
    n = int(len(y_binary))
    positives = int((y_binary == 1).sum())
    negatives = int((y_binary == 0).sum())
    return {
        "n": float(n),
        "positives": float(positives),
        "negatives": float(negatives),
        "positive_rate": float(positives / n) if n else float("nan"),
        "negative_rate": float(negatives / n) if n else float("nan"),
    }


def audit_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    target_candidates: Iterable[str],
    positive_label: Any = 1,
    month_column: str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    target = resolve_target_column(df, target_candidates)
    y = normalize_binary_target(df[target], positive_label=positive_label)
    columns = column_audit(df)

    out: dict[str, Any] = {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": target,
        "positive_label": positive_label,
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": int(columns["is_constant"].sum()),
        "mean_missing_rate": float(columns["missing_rate"].mean()) if len(columns) else 0.0,
    }
    out.update(class_balance(y))

    if month_column is not None:
        if month_column not in df.columns:
            raise ValueError(f"month column missing: {month_column}")
        out["month_column"] = month_column
        out["unique_months"] = int(df[month_column].nunique(dropna=True))

    if source_path is not None:
        path = Path(source_path)
        out["source_path"] = str(path)
        if path.is_file():
            out["file_size_bytes"] = int(path.stat().st_size)
            out["sha256"] = sha256_file(path)
        elif path.is_dir():
            files = sorted(item.name for item in path.iterdir() if item.is_file())
            out["source_files"] = ",".join(files)
            out["file_size_bytes"] = int(sum((path / name).stat().st_size for name in files))

    return out


def prediction_artifact_audit(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] = ("row_id", "y_true", "score", "split", "seed", "model", "calibration"),
) -> dict[str, Any]:
    required = list(required_columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"prediction artifact missing required columns: {missing}")

    y = normalize_binary_target(df["y_true"], positive_label=1)
    score = pd.to_numeric(df["score"], errors="coerce")
    if score.isna().any():
        raise ValueError("score column must be numeric and non-missing")

    out: dict[str, Any] = {
        "rows": int(len(df)),
        "duplicate_row_ids": int(df["row_id"].duplicated().sum()),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "splits": ",".join(sorted(map(str, df["split"].dropna().unique().tolist()))),
        "seeds": ",".join(sorted(map(str, df["seed"].dropna().unique().tolist()))),
        "models": ",".join(sorted(map(str, df["model"].dropna().unique().tolist()))),
        "calibration_methods": ",".join(sorted(map(str, df["calibration"].dropna().unique().tolist()))),
    }
    out.update(class_balance(y))
    return out
