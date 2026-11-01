from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = ("logistic_regression", "lightgbm", "mlp")
SEEDS = (7, 17, 177)
SPLITS = ("validation", "test")
REQUIRED_COLUMNS = {
    "row_id",
    "month",
    "y_true",
    "raw_probability",
    "prob_none",
    "prob_platt",
    "prob_isotonic",
    "prob_temperature",
}


def audit_predictions(
    *,
    prediction_root: str | Path,
    baf_path: str | Path,
) -> pd.DataFrame:
    root = Path(prediction_root)
    baf = pd.read_parquet(baf_path, columns=["fraud_bool", "month"])
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for split in SPLITS:
                path = root / model / f"seed_{seed}" / f"{split}.parquet"
                row = {
                    "model": model,
                    "seed": seed,
                    "split": split,
                    "path": str(path),
                    "exists": path.exists(),
                    "rows": 0,
                    "row_id_unique": False,
                    "required_columns_present": False,
                    "month_matches_expected": False,
                    "labels_match_baf": False,
                    "months": "",
                    "positive_n": 0,
                    "status": "missing",
                }
                if path.exists():
                    pred = pd.read_parquet(path)
                    row["rows"] = len(pred)
                    row["row_id_unique"] = pred["row_id"].is_unique if "row_id" in pred.columns else False
                    row["required_columns_present"] = REQUIRED_COLUMNS.issubset(pred.columns)
                    if row["required_columns_present"]:
                        expected_month = 6 if split == "validation" else 7
                        row_ids = pred["row_id"].to_numpy()
                        aligned = baf.iloc[row_ids]
                        row["months"] = ",".join(str(value) for value in sorted(pred["month"].unique().tolist()))
                        row["month_matches_expected"] = bool((pred["month"] == expected_month).all())
                        row["labels_match_baf"] = bool((aligned["fraud_bool"].to_numpy() == pred["y_true"].to_numpy()).all())
                        row["positive_n"] = int(pred["y_true"].sum())
                        row["status"] = "ok" if all(
                            [
                                row["row_id_unique"],
                                row["month_matches_expected"],
                                row["labels_match_baf"],
                            ]
                        ) else "failed"
                rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(audit: pd.DataFrame, output_path: str | Path) -> None:
    passed = bool((audit["status"] == "ok").all())
    lines = [
        "# Full BAF Prediction Reuse Audit",
        "",
        f"Overall status: {'PASS' if passed else 'FAIL'}",
        "",
        "| model | seed | split | rows | positives | months | status |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['split']} | {row['rows']} | "
            f"{row['positive_n']} | {row['months']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "The audit checks file existence, required probability columns, unique row IDs, expected month, and label alignment against the current BAF raw parquet.",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit old full-BAF prediction files for Phase 2 reuse.")
    parser.add_argument(
        "--prediction-root",
        default="/Users/starsfalling/Desktop/Project/baf-rcast-study/outputs/predictions",
    )
    parser.add_argument("--baf-path", default="data/raw/baf/Base.parquet")
    parser.add_argument("--output-csv", default="results_imbalance_selective/final_phase2/baf_full_prediction_audit.csv")
    parser.add_argument("--output-md", default="results_imbalance_selective/final_phase2/baf_full_prediction_audit.md")
    args = parser.parse_args(argv)

    audit = audit_predictions(prediction_root=args.prediction_root, baf_path=args.baf_path)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(csv_path, index=False)
    write_markdown(audit, args.output_md)
    print(audit.to_string(index=False))
    if not (audit["status"] == "ok").all():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
