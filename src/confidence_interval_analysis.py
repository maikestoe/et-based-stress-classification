import argparse
import csv
import glob
import json
import math
import os
from statistics import mean, stdev


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config_files")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_ROOT, "confidence_intervals")

MODEL_NAME_MAP = {
    "CNN": "CNN",
    "cnn": "CNN",
    "LSTM-1": "LSTM-1",
    "ConvLSTM-1": "ConvLSTM-1",
    "LSTM-3": "LSTM-3",
    "ConvLSTM-3": "ConvLSTM-3",
}
INPUT_NAME_MAP = {
    "meanDia_corrected": "PD",
    "velocity": "angular velocity",
    "acceleration": "angular acceleration",
    "position": "visual angle",
    "asymptotic_model": "asymptotic model",
    "fix_array": "fixations",
}
RF_SUBSET_LABELS = {
    "fix": "RF (fixation characteristics)",
    "mean": "RF (PD statistics)",
    "combined": "RF (combined)",
}
RF_METRIC_COLUMNS = {
    "accuracy": "acc",
    "f1": "f1",
    "recall": "rec",
    "precision": "prec",
    "roc_auc": "auc",
}
DL_METRIC_COLUMNS = {
    "macro_f1": "macro_f1",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "roc_auc": "roc_auc",
}

# Two-sided 95% t critical values for the sample sizes used here and nearby dfs.
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 95% confidence intervals from saved outer-LOSO fold metrics."
    )
    parser.add_argument(
        "--timestamp",
        default="2024-08-09",
        help="Experiment timestamp to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CI CSV tables are saved.",
    )
    return parser.parse_args()


def read_csv_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def t_critical_95(df):
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    return 1.96


def summarize_values(values):
    values = [float(value) for value in values if value not in ("", None)]
    n = len(values)
    if n == 0:
        return None
    value_mean = mean(values)
    value_sd = stdev(values) if n > 1 else 0.0
    value_sem = value_sd / math.sqrt(n) if n > 1 else 0.0
    margin = t_critical_95(n - 1) * value_sem if n > 1 else 0.0
    return {
        "n_folds": n,
        "mean": value_mean,
        "sd": value_sd,
        "sem": value_sem,
        "ci_lower": value_mean - margin,
        "ci_upper": value_mean + margin,
        "ci_half_width": margin,
    }


def format_summary_row(dataset, model_label, metric_name, source_path, values, extra=None):
    summary = summarize_values(values)
    if summary is None:
        return None
    row = {
        "dataset": dataset,
        "model": model_label,
        "metric": metric_name,
        "n_folds": summary["n_folds"],
        "mean": f"{summary['mean']:.6f}",
        "sd": f"{summary['sd']:.6f}",
        "sem": f"{summary['sem']:.6f}",
        "ci_lower": f"{summary['ci_lower']:.6f}",
        "ci_upper": f"{summary['ci_upper']:.6f}",
        "ci_half_width": f"{summary['ci_half_width']:.6f}",
        "mean_percent": f"{100.0 * summary['mean']:.2f}",
        "sd_percent": f"{100.0 * summary['sd']:.2f}",
        "ci_lower_percent": f"{100.0 * summary['ci_lower']:.2f}",
        "ci_upper_percent": f"{100.0 * summary['ci_upper']:.2f}",
        "ci_half_width_percent": f"{100.0 * summary['ci_half_width']:.2f}",
        "source_path": source_path,
    }
    if extra:
        row.update(extra)
    return row


def latest_file(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_json(path):
    with open(path) as file:
        return json.load(file)


def normalize_model_name(raw_name):
    return MODEL_NAME_MAP.get(str(raw_name), MODEL_NAME_MAP.get(str(raw_name).lower(), str(raw_name)))


def format_input_label(input_cols):
    if isinstance(input_cols, str):
        input_cols = [part.strip() for part in input_cols.split(",") if part.strip()]
    labels = [INPUT_NAME_MAP.get(col, col.replace("_", " ")) for col in input_cols]
    return ", ".join(labels) if labels else "default input"


def get_dataset_from_config(config):
    path_bits = " ".join(
        [
            str(config.get("df_prep_path", "")),
            str(config.get("DL", {}).get("path_results", "")),
        ]
    ).lower()
    return "ForDigitStress" if "fordigitstress" in path_bits else "VR goalkeeper"


def find_dl_metrics_path(config, timestamp):
    model = config["DL"]["model"]
    experiment_id = str(config["experiment_id"])
    result_root = str(config["DL"]["path_results"])
    result_root = result_root.replace("$TMPDIR", PROJECT_ROOT)
    result_root = result_root.replace("$TMPBASE", PROJECT_ROOT)
    if not os.path.isabs(result_root):
        result_root = os.path.abspath(os.path.join(SCRIPT_DIR, result_root))

    candidate_paths = [
        os.path.join(result_root, timestamp, model, experiment_id, "recovery", "outer_metrics_recovered.csv"),
        os.path.join(result_root, timestamp, model, experiment_id, "outer_metrics_recovered.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    # Some older ForDigitStress runs use uppercase CNN in the directory name.
    if model.lower() == "cnn":
        candidate_paths = [
            os.path.join(result_root, timestamp, "CNN", experiment_id, "recovery", "outer_metrics_recovered.csv"),
            os.path.join(result_root, timestamp, "CNN", experiment_id, "outer_metrics_recovered.csv"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return path
    return None


def compute_dl_ci_rows(timestamp):
    rows = []
    for config_path in sorted(glob.glob(os.path.join(CONFIG_DIR, "exp*.json"))):
        if any(suffix in os.path.basename(config_path) for suffix in ["_recovery", "_regenerate"]):
            continue
        config = load_json(config_path)
        if str(config.get("timestamp")) != str(timestamp):
            continue
        metrics_path = find_dl_metrics_path(config, timestamp)
        if metrics_path is None:
            continue

        metric_rows = read_csv_rows(metrics_path)
        dataset = get_dataset_from_config(config)
        input_label = format_input_label(config.get("DL", {}).get("input_cols", []))
        model_label = f"{normalize_model_name(config['DL']['model'])} ({input_label})"
        extra = {
            "experiment_id": str(config["experiment_id"]),
            "input_signal": input_label,
            "source_type": "DL",
        }
        for metric_name, column_name in DL_METRIC_COLUMNS.items():
            if column_name not in metric_rows[0]:
                continue
            row = format_summary_row(
                dataset=dataset,
                model_label=model_label,
                metric_name=metric_name,
                source_path=metrics_path,
                values=[row[column_name] for row in metric_rows],
                extra=extra,
            )
            if row:
                rows.append(row)
    return rows


def compute_rf_ci_rows():
    rows = []
    for subset_name, subset_label in RF_SUBSET_LABELS.items():
        pattern = os.path.join(
            RESULTS_ROOT,
            "vr_goalkeeper",
            "rf_baseline_vr_goalkeeper*",
            subset_name,
            "RF",
            "df_cv_results_RF_*.csv",
        )
        metrics_path = latest_file(pattern)
        if metrics_path is None:
            continue
        metric_rows = read_csv_rows(metrics_path)
        extra = {
            "experiment_id": "",
            "input_signal": subset_name,
            "source_type": "RF",
        }
        for metric_name, column_name in RF_METRIC_COLUMNS.items():
            if column_name not in metric_rows[0]:
                continue
            row = format_summary_row(
                dataset="VR goalkeeper",
                model_label=subset_label,
                metric_name=metric_name,
                source_path=metrics_path,
                values=[row[column_name] for row in metric_rows],
                extra=extra,
            )
            if row:
                rows.append(row)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rf_rows = compute_rf_ci_rows()
    dl_rows = compute_dl_ci_rows(args.timestamp)
    all_rows = rf_rows + dl_rows

    write_csv_rows(os.path.join(args.output_dir, "rf_confidence_intervals.csv"), rf_rows)
    write_csv_rows(os.path.join(args.output_dir, "dl_confidence_intervals.csv"), dl_rows)
    write_csv_rows(os.path.join(args.output_dir, "all_confidence_intervals.csv"), all_rows)

    print(f"Saved RF CI table with {len(rf_rows)} rows.")
    print(f"Saved DL CI table with {len(dl_rows)} rows.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
