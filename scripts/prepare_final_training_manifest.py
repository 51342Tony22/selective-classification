from __future__ import annotations

import argparse
from pathlib import Path

from imbalance_selective.experiment_matrix import (
    build_evaluation_jobs,
    build_training_jobs,
    dataset_readiness,
    load_yaml,
    summarize_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare final-training manifests without running training.")
    parser.add_argument("--matrix-config", default="configs/imbalance_selective/full_matrix.yaml")
    parser.add_argument("--out-dir", default="results_imbalance_selective/reports/final_training_manifest")
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(root / args.matrix_config)
    training_jobs = build_training_jobs(config, root=root)
    evaluation_jobs = build_evaluation_jobs(config, training_jobs)
    readiness = dataset_readiness(root=root)
    summary = summarize_matrix(training_jobs, evaluation_jobs)

    training_path = out_dir / "training_jobs.csv"
    evaluation_path = out_dir / "evaluation_jobs.csv"
    readiness_path = out_dir / "dataset_readiness.csv"
    summary_path = out_dir / "matrix_summary.csv"
    report_path = out_dir / "README.md"

    training_jobs.to_csv(training_path, index=False)
    evaluation_jobs.to_csv(evaluation_path, index=False)
    readiness.to_csv(readiness_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(_report(summary, readiness, training_jobs, evaluation_jobs), encoding="utf-8")

    print(f"Wrote {training_path}")
    print(f"Wrote {evaluation_path}")
    print(f"Wrote {readiness_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


def _report(summary, readiness, training_jobs, evaluation_jobs) -> str:
    total_training = len(training_jobs)
    total_evaluation = len(evaluation_jobs)
    return f"""# Final Training Manifest

This manifest is a dry run. It does not train models and does not run the full experimental matrix.

## Stop Point

Before launching the full training jobs below, pause and confirm with the user.

## Dataset Readiness

{_markdown_table(readiness)}

## Matrix Summary

{_markdown_table(summary)}

## Totals

- Training jobs: {total_training}
- Evaluation jobs: {total_evaluation}

## Files

- `training_jobs.csv`
- `evaluation_jobs.csv`
- `dataset_readiness.csv`
- `matrix_summary.csv`
"""


def _markdown_table(frame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
