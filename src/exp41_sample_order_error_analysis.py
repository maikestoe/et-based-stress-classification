"""
Experiment 41: evaluate whether classification errors depend on original sample order.

Priority of data sources
------------------------
1. Existing joined exp41 analysis CSV, if present
2. Saved original prediction arrays, if present
3. Saved recovered prediction arrays

This script never retrains a model. In the current workspace it is optimized to
reuse the already saved joined CSV from the previous exp41 analysis. If that
file is missing, it raises a clear error instead of retraining anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter


STYLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "plot_style2.txt"))
PRIMARY_COLOR = "#3E5F8A"
SECONDARY_COLOR = "#C44F5E"
NEUTRAL_COLOR = "#7D8A99"
FILL_PRIMARY = "#C7D2E0"
FILL_SECONDARY = "#E7C4CB"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "exp41_sample_order_heatmap",
    ["#F4F7FA", "#D9E1EB", "#9EB2CB", PRIMARY_COLOR],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the dependence between classification error and original sample order for exp41."
    )
    parser.add_argument(
        "--config",
        default="configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json",
        help="Experiment config used to locate result artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <experiment>/recovery/exp41_sample_order_error_analysis.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of equal-width bins across normalized sample order.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    if os.path.exists(STYLE_PATH):
        plt.style.use(STYLE_PATH)
    plt.rcParams["text.usetex"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def style_publication_axis(ax, categorical_x: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)
    ax.yaxis.labelpad = 10
    if not categorical_x:
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.get_offset_text().set_visible(False)


def load_json_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return json.load(file)


def resolve_path(base_dir: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.abspath(os.path.join(base_dir, maybe_relative))


def resolve_local_project_path(base_dir: str, path_with_placeholders: str) -> str:
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    normalized = path_with_placeholders
    replacements = {
        "$TMPDIR/data": os.path.join(project_root, "data"),
        "$TMPDIR/results": os.path.join(project_root, "results"),
        "$TMPBASE/data": os.path.join(project_root, "data"),
        "$TMPBASE/results": os.path.join(project_root, "results"),
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    expanded = os.path.expandvars(normalized)
    return resolve_path(base_dir, expanded)


def get_local_project_root(base_dir: str) -> str:
    return os.path.abspath(os.path.join(base_dir, ".."))


def get_experiment_root(config: dict, base_dir: str) -> str:
    project_root = get_local_project_root(base_dir)
    preferred_local_root = os.path.join(
        project_root,
        "results",
        "vr_goalkeeper",
        "DL",
        str(config["timestamp"]),
        str(config["DL"]["model"]),
        str(config["experiment_id"]),
    )
    if os.path.exists(preferred_local_root):
        return preferred_local_root

    return resolve_local_project_path(
        base_dir,
        os.path.join(
            config["DL"]["path_results"],
            config["timestamp"],
            config["DL"]["model"],
            str(config["experiment_id"]),
        ),
    )


def joined_csv_path_for_experiment(experiment_root: str) -> str:
    return os.path.join(
        experiment_root,
        "recovery",
        "exp41_cognitive_task_error_analysis",
        "exp41_prediction_signal_task_joined.csv",
    )


def load_joined_csv_rows(joined_csv_path: str) -> List[dict]:
    rows: List[dict] = []
    with open(joined_csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            test_id = int(row["test_id"]) if row.get("test_id") else int(row["ID"])
            if row.get("sample_idx") not in (None, ""):
                sample_idx = int(float(row["sample_idx"]))
            else:
                sample_idx = -1

            is_correct_raw = row.get("is_correct")
            if is_correct_raw in (None, ""):
                is_correct_raw = row.get("correct")
            is_correct_text = str(is_correct_raw).strip().lower()
            if is_correct_text in {"true", "1", "1.0"}:
                is_correct = 1
            elif is_correct_text in {"false", "0", "0.0"}:
                is_correct = 0
            else:
                is_correct = int(float(is_correct_raw))

            rows.append(
                {
                    "test_id": test_id,
                    "sample_idx": sample_idx,
                    "is_correct": is_correct,
                    "is_error": 1 - is_correct,
                    "shot": row.get("shot", ""),
                }
            )
    return rows


def ensure_sample_idx(rows: List[dict]) -> List[dict]:
    by_subject: Dict[int, List[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["test_id"], []).append(row)

    fixed_rows: List[dict] = []
    for test_id in sorted(by_subject):
        subject_rows = by_subject[test_id]
        has_sample_idx = all(row["sample_idx"] >= 0 for row in subject_rows)
        if not has_sample_idx:
            for idx, row in enumerate(subject_rows):
                row["sample_idx"] = idx
        subject_rows.sort(key=lambda item: item["sample_idx"])
        n_rows = len(subject_rows)
        for row in subject_rows:
            row["n_samples_subject"] = n_rows
            row["sample_order"] = int(row["sample_idx"]) + 1
            row["sample_order_rel"] = 0.0 if n_rows <= 1 else float(row["sample_idx"]) / float(n_rows - 1)
            row["sample_order_percent"] = 100.0 * row["sample_order_rel"]
            fixed_rows.append(row)
    return fixed_rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def median(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return 0.5 * float(values[mid - 1] + values[mid])


def rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    mean_a = mean(a)
    mean_b = mean(b)
    centered_a = [x - mean_a for x in a]
    centered_b = [y - mean_b for y in b]
    denom_a = math.sqrt(sum(x * x for x in centered_a))
    denom_b = math.sqrt(sum(y * y for y in centered_b))
    if denom_a == 0.0 or denom_b == 0.0:
        return float("nan")
    num = sum(x * y for x, y in zip(centered_a, centered_b))
    return num / (denom_a * denom_b)


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return pearson_corr(rankdata(x), rankdata(y))


def normal_approx_two_sided_pvalue_from_rho(rho: float, n: int) -> float:
    if n < 3 or math.isnan(rho):
        return float("nan")
    z = abs(rho) * math.sqrt(max(n - 1, 1))
    return math.erfc(z / math.sqrt(2.0))


def rows_by_subject(rows: List[dict]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for row in rows:
        out.setdefault(row["test_id"], []).append(row)
    for test_id in out:
        out[test_id].sort(key=lambda item: item["sample_idx"])
    return out


def compute_global_statistics(rows: List[dict]) -> List[dict]:
    sample_order = [float(row["sample_order"]) for row in rows]
    is_error = [float(row["is_error"]) for row in rows]
    results = []
    rho = spearman_rho(sample_order, is_error)
    p_value = normal_approx_two_sided_pvalue_from_rho(rho, len(rows))
    results.append(
        {
            "analysis_level": "global",
            "order_variable": "sample_order",
            "n_rows": len(rows),
            "spearman_rho": rho,
            "approx_p_value": p_value,
            "error_rate": mean(is_error),
        }
    )
    return results


def compute_subject_statistics(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    subject_map = rows_by_subject(rows)
    subject_rows: List[dict] = []
    rho_values: List[float] = []

    for test_id in sorted(subject_map):
        subject_rows_current = subject_map[test_id]
        sample_order = [float(row["sample_order"]) for row in subject_rows_current]
        is_error = [float(row["is_error"]) for row in subject_rows_current]

        if len(set(is_error)) < 2:
            rho = float("nan")
            p_value = float("nan")
        else:
            rho = spearman_rho(sample_order, is_error)
            p_value = normal_approx_two_sided_pvalue_from_rho(rho, len(subject_rows_current))
            if not math.isnan(rho):
                rho_values.append(rho)

        subject_rows.append(
            {
                "test_id": test_id,
                "n_rows": len(subject_rows_current),
                "error_rate": mean(is_error),
                "spearman_rho": rho,
                "approx_p_value": p_value,
            }
        )

    summary = [
        {
            "n_subjects": len(subject_rows),
            "subjects_with_defined_rho": len(rho_values),
            "mean_subject_rho": mean(rho_values),
            "median_subject_rho": median(rho_values),
            "positive_rho_fraction": mean([1.0 if value > 0 else 0.0 for value in rho_values]),
        }
    ]
    return subject_rows, summary


def compute_binned_error_rates(rows: List[dict], n_bins: int) -> List[dict]:
    binned: List[dict] = []
    max_order = max(int(row["sample_order"]) for row in rows)
    for bin_idx in range(n_bins):
        start = 1.0 + bin_idx * max_order / float(n_bins)
        end = 1.0 + (bin_idx + 1) * max_order / float(n_bins)
        start_idx = int(math.floor(start))
        end_idx = int(math.floor(end)) if bin_idx < n_bins - 1 else max_order
        if bin_idx == n_bins - 1:
            selected = [row for row in rows if start_idx <= row["sample_order"] <= max_order]
        else:
            selected = [row for row in rows if start_idx <= row["sample_order"] < end_idx]
        binned.append(
            {
                "order_bin": f"{start_idx}-{end_idx if bin_idx == n_bins - 1 else max(start_idx, end_idx - 1)}",
                "order_bin_start": start_idx,
                "order_bin_end": end_idx if bin_idx == n_bins - 1 else max(start_idx, end_idx - 1),
                "n_rows": len(selected),
                "n_subjects": len({row["test_id"] for row in selected}),
                "mean_order": mean([row["sample_order"] for row in selected]),
                "error_rate": mean([row["is_error"] for row in selected]),
            }
        )
    return binned


def compute_sample_order_error_rates(rows: List[dict]) -> List[dict]:
    max_order = max(int(row["sample_order"]) for row in rows)
    per_order: List[dict] = []
    for order in range(1, max_order + 1):
        selected = [row for row in rows if int(row["sample_order"]) == order]
        per_order.append(
            {
                "sample_order": order,
                "n_rows": len(selected),
                "n_subjects": len({row["test_id"] for row in selected}),
                "error_rate": mean([row["is_error"] for row in selected]),
            }
        )
    return per_order


def compute_subject_binned_error_rates(rows: List[dict], n_bins: int) -> Tuple[List[int], List[List[float]]]:
    subject_map = rows_by_subject(rows)
    subject_ids = sorted(subject_map)
    matrix: List[List[float]] = []
    max_order = max(int(row["sample_order"]) for row in rows)
    for test_id in subject_ids:
        subject_rows = subject_map[test_id]
        row_values: List[float] = []
        for bin_idx in range(n_bins):
            start = 1.0 + bin_idx * max_order / float(n_bins)
            end = 1.0 + (bin_idx + 1) * max_order / float(n_bins)
            start_idx = int(math.floor(start))
            end_idx = int(math.floor(end)) if bin_idx < n_bins - 1 else max_order
            if bin_idx == n_bins - 1:
                selected = [row for row in subject_rows if start_idx <= row["sample_order"] <= max_order]
            else:
                selected = [row for row in subject_rows if start_idx <= row["sample_order"] < end_idx]
            row_values.append(mean([row["is_error"] for row in selected]))
        matrix.append(row_values)
    return subject_ids, matrix


def save_sample_order_error_rate_plot(order_stats: List[dict], output_dir: str) -> None:
    x_values = [row["sample_order"] for row in order_stats]
    y_values = [row["error_rate"] for row in order_stats]

    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.axvspan(20.5, 39.5, color="#E6E8EC", alpha=0.6, zorder=0)
    ax.scatter(x_values, y_values, s=68, color=PRIMARY_COLOR, zorder=3)
    ax.set_xlabel("Original sample order within participant", fontsize=17)
    ax.set_ylabel("Error rate", fontsize=17)
    ax.set_xlim(0.5, 39.5)
    ax.set_ylim(0.0, min(1.0, max(y_values) + 0.05))
    ax.set_xticks(list(range(0, 41, 5)))
    fig.suptitle("ConvLSTM-3 (asymptotic model), VR goalkeeper dataset", fontsize=20, x=0.02, ha="left", y=1.04)
    ymax = ax.get_ylim()[1]
    ax.text(10.0, ymax * 0.98, "Non-stress", ha="center", va="top", fontsize=15, color="#4A5565")
    ax.text(30.0, ymax * 0.98, "Stress", ha="center", va="top", fontsize=15, color="#4A5565")
    style_publication_axis(ax)
    fig.savefig(os.path.join(output_dir, "exp41_error_rate_by_sample_order.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_error_rate_by_sample_order.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_subject_rho_plot(subject_stats: List[dict], output_dir: str) -> None:
    valid_rows = [row for row in subject_stats if not math.isnan(row["spearman_rho"])]
    if not valid_rows:
        return
    valid_rows.sort(key=lambda row: row["spearman_rho"])
    x_values = list(range(len(valid_rows)))
    y_values = [row["spearman_rho"] for row in valid_rows]
    subject_ids = [str(int(row["test_id"])) for row in valid_rows]
    colors = [SECONDARY_COLOR if value > 0 else PRIMARY_COLOR for value in y_values]

    fig, ax = plt.subplots(figsize=(10.8, 5.8), constrained_layout=True)
    ax.bar(x_values, y_values, color=colors, alpha=0.9)
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.1)
    ax.set_xticks(range(len(subject_ids)), subject_ids, rotation=45, ha="right", fontsize=11)
    ax.set_title("Experiment 41: subject-level order/error dependency", fontsize=20)
    ax.set_xlabel("Held-out subject ID, sorted by Spearman rho", fontsize=17)
    ax.set_ylabel("Spearman rho", fontsize=17)
    style_publication_axis(ax, categorical_x=True)
    fig.savefig(os.path.join(output_dir, "exp41_subjectwise_sample_order_rho.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_subjectwise_sample_order_rho.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_subject_bin_heatmap(subject_ids: List[int], heatmap_matrix: List[List[float]], bin_labels: List[str], output_dir: str) -> None:
    if not subject_ids or not heatmap_matrix:
        return
    fig, ax = plt.subplots(
        figsize=(10.5, max(6.0, 0.28 * len(subject_ids) + 2.0)),
        constrained_layout=True,
    )
    image = ax.imshow(heatmap_matrix, aspect="auto", cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
    ax.set_title("Experiment 41: subject-wise error rate across sample-order bins", fontsize=20)
    ax.set_xlabel("Original sample-order bin", fontsize=17)
    ax.set_ylabel("Held-out subject", fontsize=17)
    ax.set_xticks(list(range(len(heatmap_matrix[0]))))
    ax.set_xticklabels(bin_labels, fontsize=11, rotation=30, ha="right")
    ax.set_yticks(list(range(len(subject_ids))))
    ax.set_yticklabels([str(subject_id) for subject_id in subject_ids], fontsize=10)
    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label("Error rate", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    fig.savefig(os.path.join(output_dir, "exp41_subject_bin_error_heatmap.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_subject_bin_error_heatmap.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_sample_order_plot_bundle(
    order_stats: List[dict],
    binned_stats: List[dict],
    subject_stats: List[dict],
    subject_ids: List[int],
    heatmap_matrix: List[List[float]],
    output_dir: str,
) -> None:
    save_sample_order_error_rate_plot(order_stats, output_dir)
    save_subject_rho_plot(subject_stats, output_dir)
    save_subject_bin_heatmap(subject_ids, heatmap_matrix, [row["order_bin"] for row in binned_stats], output_dir)


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_text_summary(
    output_dir: str,
    global_stats: List[dict],
    subject_summary: List[dict],
    metadata: dict,
) -> None:
    global_rel = next(row for row in global_stats if row["order_variable"] == "sample_order")
    subject_row = subject_summary[0]

    lines = [
        "Experiment 41: sample-order dependency analysis",
        "",
        f"Data source priority result: {metadata['analysis_source']}",
        f"Rows analyzed: {global_rel['n_rows']}",
        f"Global Spearman rho (original sample order vs. error): {format_float(global_rel['spearman_rho'])}",
        f"Approx. two-sided p-value: {format_float(global_rel['approx_p_value'])}",
        "",
        f"Mean subject rho: {format_float(subject_row['mean_subject_rho'])}",
        f"Median subject rho: {format_float(subject_row['median_subject_rho'])}",
        f"Fraction of subjects with positive rho: {format_float(subject_row['positive_rho_fraction'])}",
    ]
    with open(os.path.join(output_dir, "exp41_sample_order_error_summary.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    configure_plot_style()
    config_path = os.path.abspath(args.config)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config = load_json_config(config_path)
    experiment_root = get_experiment_root(config, base_dir)
    output_dir = args.output_dir or os.path.join(experiment_root, "recovery", "exp41_sample_order_error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    joined_csv_path = joined_csv_path_for_experiment(experiment_root)
    if not os.path.exists(joined_csv_path):
        raise FileNotFoundError(
            "No joined exp41 analysis CSV found. This lightweight script intentionally avoids retraining and "
            "expects the existing file at: " + joined_csv_path
        )

    rows = ensure_sample_idx(load_joined_csv_rows(joined_csv_path))
    global_stats = compute_global_statistics(rows)
    subject_stats, subject_summary = compute_subject_statistics(rows)
    order_stats = compute_sample_order_error_rates(rows)
    binned_stats = compute_binned_error_rates(rows, args.n_bins)
    subject_ids, heatmap_matrix = compute_subject_binned_error_rates(rows, args.n_bins)

    write_csv(
        os.path.join(output_dir, "exp41_sample_order_joined.csv"),
        rows,
        ["test_id", "sample_idx", "sample_order", "n_samples_subject", "sample_order_rel", "sample_order_percent", "is_correct", "is_error", "shot"],
    )
    write_csv(
        os.path.join(output_dir, "exp41_sample_order_global_statistics.csv"),
        global_stats,
        ["analysis_level", "order_variable", "n_rows", "spearman_rho", "approx_p_value", "error_rate"],
    )
    write_csv(
        os.path.join(output_dir, "exp41_sample_order_subject_statistics.csv"),
        subject_stats,
        ["test_id", "n_rows", "error_rate", "spearman_rho", "approx_p_value"],
    )
    write_csv(
        os.path.join(output_dir, "exp41_sample_order_subject_summary.csv"),
        subject_summary,
        ["n_subjects", "subjects_with_defined_rho", "mean_subject_rho", "median_subject_rho", "positive_rho_fraction"],
    )
    write_csv(
        os.path.join(output_dir, "exp41_sample_order_error_rates.csv"),
        order_stats,
        ["sample_order", "n_rows", "n_subjects", "error_rate"],
    )
    write_csv(
        os.path.join(output_dir, "exp41_sample_order_binned_error_rates.csv"),
        binned_stats,
        ["order_bin", "order_bin_start", "order_bin_end", "n_rows", "n_subjects", "mean_order", "error_rate"],
    )

    metadata = {
        "analysis_source": "existing_joined_csv",
        "joined_csv_path": joined_csv_path,
        "config_path": config_path,
        "experiment_root": experiment_root,
        "output_dir": output_dir,
        "n_bins": int(args.n_bins),
        "n_rows": len(rows),
        "n_subjects": len({row["test_id"] for row in rows}),
        "note": "No retraining performed. Existing exp41 joined analysis CSV was reused.",
    }
    with open(os.path.join(output_dir, "exp41_sample_order_error_analysis_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    write_text_summary(output_dir, global_stats, subject_summary, metadata)
    save_sample_order_plot_bundle(
        order_stats,
        binned_stats,
        subject_stats,
        subject_ids,
        heatmap_matrix,
        output_dir,
    )
    print(f"Saved exp41 sample-order dependency analysis to {output_dir}")


if __name__ == "__main__":
    main()
