from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.data_split import assign_temporal_split
from imbalance_selective.datasets import load_config


def create_baf_sample(config_path: str | Path, *, overwrite: bool = False) -> pd.DataFrame:
    config = load_config(config_path)
    dataset = config["dataset"]
    sample_config = dataset.get("sample", {})
    source_path = Path(dataset.get("source_full_path", "data/raw/baf/Base.parquet"))
    output_path = Path(dataset["local_path"])
    split_path = Path(dataset["split_assignments_path"])
    target = dataset.get("target", "fraud_bool")
    month_column = dataset.get("month_column", "month")
    rows = int(sample_config.get("rows", 50000))
    seed = int(sample_config.get("seed", 7))

    if output_path.exists() and not overwrite:
        return pd.read_parquet(output_path)
    if not source_path.exists():
        raise FileNotFoundError(f"BAF source file not found: {source_path}")

    frame = pd.read_parquet(source_path)
    if rows > len(frame):
        raise ValueError(f"requested {rows} rows, but source has only {len(frame)}")
    for column in [target, month_column]:
        if column not in frame.columns:
            raise ValueError(f"required BAF column missing: {column}")

    sampled = stratified_exact_sample(frame, rows=rows, strata=[month_column, target], seed=seed)
    sampled = sampled.sort_values([month_column]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_path, index=False)
    pd.DataFrame(
        {
            "row_id": np.arange(len(sampled), dtype=int),
            month_column: sampled[month_column].to_numpy(),
            "split": assign_temporal_split(sampled[month_column]),
        }
    ).to_parquet(split_path, index=False)
    return sampled


def stratified_exact_sample(frame: pd.DataFrame, *, rows: int, strata: list[str], seed: int) -> pd.DataFrame:
    grouped = frame.groupby(strata, dropna=False, sort=True)
    counts = grouped.size().rename("count").reset_index()
    raw = counts["count"].to_numpy(dtype=float) * (rows / len(frame))
    base = np.floor(raw).astype(int)
    remainder = int(rows - base.sum())
    if remainder > 0:
        order = np.argsort(-(raw - base), kind="mergesort")
        base[order[:remainder]] += 1
    counts["take"] = base

    pieces = []
    rng = np.random.default_rng(seed)
    for _, row in counts.iterrows():
        mask = np.ones(len(frame), dtype=bool)
        for column in strata:
            mask &= frame[column].to_numpy() == row[column]
        group_indices = np.flatnonzero(mask)
        take = int(row["take"])
        if take == 0:
            continue
        chosen = rng.choice(group_indices, size=take, replace=False)
        pieces.append(frame.iloc[np.sort(chosen)])
    if not pieces:
        raise RuntimeError("stratified sample selected no rows")
    sampled = pd.concat(pieces, axis=0)
    if len(sampled) != rows:
        raise AssertionError(f"sample size mismatch: expected {rows}, got {len(sampled)}")
    return sampled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the fixed 50k BAF sample used by the main experiments.")
    parser.add_argument("--config", default="configs/imbalance_selective/baf_50k.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    sampled = create_baf_sample(args.config, overwrite=args.overwrite)
    target = load_config(args.config)["dataset"].get("target", "fraud_bool")
    print(
        f"created/read {len(sampled)} rows; "
        f"positive_rate={sampled[target].mean():.6f}; "
        f"positives={int(sampled[target].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
