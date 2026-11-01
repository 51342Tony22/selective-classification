from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def find_base_csv(download_dir: Path) -> Path:
    candidates = sorted(download_dir.rglob("Base.csv"))
    if not candidates:
        candidates = sorted(download_dir.rglob("*Base*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find Base.csv under Kaggle download dir: {download_dir}")
    return candidates[0]


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    df = pd.read_csv(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)


def install_from_local_source(source_path: Path, csv_out: Path, parquet_out: Path, keep_csv: bool) -> None:
    suffix = source_path.suffix.lower()
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        shutil.copy2(source_path, parquet_out)
        if not keep_csv:
            csv_out.unlink(missing_ok=True)
        return
    if suffix == ".csv":
        if source_path.resolve() != csv_out.resolve():
            shutil.copy2(source_path, csv_out)
        convert_csv_to_parquet(csv_out, parquet_out)
        if not keep_csv:
            csv_out.unlink(missing_ok=True)
        return
    raise ValueError(f"Unsupported local BAF source extension: {source_path.suffix}")


def write_split_assignments(parquet_path: Path, split_path: Path, target: str, month_column: str) -> None:
    df = pd.read_parquet(parquet_path, columns=[target, month_column])
    ordered_months = sorted(df[month_column].dropna().unique().tolist())
    if len(ordered_months) < 8:
        raise ValueError(f"BAF temporal split requires at least 8 months; found {len(ordered_months)}")
    selected = ordered_months[:8]
    month_to_split = {
        **{month: "train" for month in selected[0:5]},
        selected[5]: "calibration",
        selected[6]: "validation",
        selected[7]: "test",
    }
    out = pd.DataFrame(
        {
            "row_id": range(len(df)),
            "month": df[month_column].to_numpy(),
            "y_true": df[target].to_numpy(),
            "split": df[month_column].map(month_to_split).fillna("unused").to_numpy(),
        }
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(split_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BAF Base into this project.")
    parser.add_argument("--config", default="configs/imbalance_selective/baf.yaml")
    parser.add_argument("--source-file", default=None, help="Optional existing local Base.csv or Base.parquet to install.")
    parser.add_argument("--keep-csv", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(root / args.config)
    dataset = config["dataset"]
    slug = dataset["kaggle_slug"]
    csv_out = root / dataset["csv_path"]
    parquet_out = root / dataset["local_path"]
    split_out = root / dataset["split_assignments_path"]

    if args.source_file:
        source = Path(args.source_file).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Local BAF source file does not exist: {source}")
        install_from_local_source(source, csv_out, parquet_out, keep_csv=args.keep_csv)
        source_label = str(source)
    else:
        download_dir = Path(kagglehub.dataset_download(slug))
        source_csv = find_base_csv(download_dir)
        install_from_local_source(source_csv, csv_out, parquet_out, keep_csv=args.keep_csv)
        source_label = str(download_dir)

    write_split_assignments(
        parquet_out,
        split_out,
        target=dataset.get("target", "fraud_bool"),
        month_column=dataset.get("month_column", "month"),
    )

    print(f"Installed from: {source_label}")
    print(f"Parquet: {parquet_out}")
    print(f"Split assignments: {split_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
