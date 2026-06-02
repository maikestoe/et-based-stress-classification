"""Export VR Goalkeeper source-data tables from saved training summaries."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "paper" / "paper_experiment_matrix.csv"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "vr_goalkeeper" / "DL" / "2024-08-09"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "source_data" / "vr_model_comparison_metrics.csv"

METRIC_PATTERNS = {
    "macro_f1": r"Final Average Validation F1 Score:\s*([0-9.]+)\s*\nStandard Deviation:\s*([0-9.]+)",
    "weighted_f1": r"Final Average Weighted Validation F1 Score:\s*([0-9.]+)\s*\nStandard Deviation:\s*([0-9.]+)",
    "auc": r"Final Average Validation AUC:\s*([0-9.]+)\s*\nStandard Deviation:\s*([0-9.]+)",
    "precision": r"Final Average Validation Precision:\s*([0-9.]+)\s*\nStandard Deviation:\s*([0-9.]+)",
    "recall": r"Final Average Validation Recall:\s*([0-9.]+)\s*\nStandard Deviation:\s*([0-9.]+)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, type=Path)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    return parser.parse_args()


def read_matrix(matrix_path: Path) -> list[dict[str, str]]:
    with matrix_path.open(newline="") as file:
        return list(csv.DictReader(file))


def parse_final_summary(summary_path: Path) -> dict[str, float]:
    text = summary_path.read_text()
    metrics: dict[str, float] = {}
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not parse {metric_name} from {summary_path}")
        mean_value, std_value = (float(value) for value in match.groups())
        metrics[f"{metric_name}_mean"] = mean_value
        metrics[f"{metric_name}_std"] = std_value
        metrics[f"{metric_name}_mean_percent"] = mean_value * 100.0
        metrics[f"{metric_name}_std_percent"] = std_value * 100.0
    return metrics


def build_rows(matrix_rows: list[dict[str, str]], results_root: Path) -> list[dict[str, str | float]]:
    output_rows: list[dict[str, str | float]] = []
    for matrix_row in matrix_rows:
        if matrix_row["dataset"] != "vr_goalkeeper":
            continue

        experiment_number = matrix_row["experiment_id"].removeprefix("exp")
        summary_path = results_root / matrix_row["model"] / experiment_number / "final_summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        metrics = parse_final_summary(summary_path)
        output_rows.append(
            {
                "dataset": matrix_row["dataset"],
                "experiment_id": matrix_row["experiment_id"],
                "model": matrix_row["model"],
                "input_signal": matrix_row["input_signal"],
                "input_column": matrix_row["input_column"],
                **metrics,
                "source_log": str(summary_path.relative_to(REPO_ROOT)),
            }
        )
    return output_rows


def write_rows(output_path: Path, rows: list[dict[str, str | float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(read_matrix(args.matrix), args.results_root)
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
