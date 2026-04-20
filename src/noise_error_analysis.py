"""
Noise-aware subject-level error analysis across saved experiments.

This script extends the existing subject-level error analysis with simple
noise / data-quality quantification derived from the preprocessing logic,
without modifying any preprocessing outputs or saved experiment artifacts.

Important safety note
---------------------
This script is strictly read-only with respect to the original preprocessing
pipeline. It reuses existing preprocessing functions in-memory to quantify
signal quality, but it never overwrites:
- raw data
- saved preprocessing dataframes
- Optuna studies
- trained weights
- experiment result files

What is quantified
------------------
VR goalkeeper dataset:
- blink fraction within the modeled shot windows
- fraction of samples where both pupil channels were invalid before mean-signal
  interpolation
- fraction of samples with invalid right-eye / left-eye signal
- fraction of single-eye-only samples

ForDigitStress dataset:
- mean eye-tracker confidence
- fraction of low-confidence samples
- fraction of invalid samples after the confidence-padding logic used before
  interpolation
- fraction of invalid candidate windows during extraction

These subject-level noise metrics are then joined with the outer-LOSO
subject-level macro-F1 scores already saved in the experiment result files.

Methodological scope
--------------------
Noise proxies are computed at subject level and are related to selected
representative model configurations. Scatter plots are restricted to the
manuscript-relevant best models: ConvLSTM-3 with the asymptotic-model input for
the VR goalkeeper dataset, and CNN/LSTM-1 with PD input for ForDigitStress.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr

import process_vr_goalkeeper
import process_fordigitstress
import segment_vr_goalkeeper
import segment_fordigitstress
from Classes import PdFilter
from pd_utils import compute_meanDia

from error_analysis_subjects import (
    CONFIG_DIR,
    STYLE_PATH,
    LOCAL_DATA_ROOT,
    COMPARISON_PALETTE,
    discover_matching_configs,
    aggregate_all_metrics,
    get_dataset_display_name,
    prettify_model_name,
    prettify_input_cols,
)


VR_RAW_ROOT = os.path.join(LOCAL_DATA_ROOT, "vr_goalkeeper")
FORDIGIT_RAW_ROOT = os.path.join(LOCAL_DATA_ROOT, "ForDigitStress")
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "noise_error_heatmap",
    [
        (0.00, "#E5C04A"),
        (0.25, "#C8C56A"),
        (0.50, "#8DBB6C"),
        (0.75, "#6C85A5"),
        (1.00, "#123B6D"),
    ],
)
PRIMARY_COLOR = COMPARISON_PALETTE[-1]
SECONDARY_COLOR = COMPARISON_PALETTE[-2]
INPUT_COL_ORDER_RAW = ["meanDia_corrected", "asymptotic_model"]
MODEL_ORDER_RAW = [
    "CNN",
    "LSTM-1",
    "ConvLSTM-1",
    "LSTM-3",
    "ConvLSTM-3",
]
FORDIGIT_NOISE_METRIC_ORDER = [
    "mean_confidence",
    "low_confidence_fraction",
    "extended_invalid_fraction",
    "invalid_window_fraction",
]
VR_NOISE_METRIC_ORDER = [
    "blink_fraction",
    "both_invalid_fraction",
    "single_eye_only_fraction",
]
SELECTED_NOISE_MODEL_KEYS = {
    "vr": {
        ("41", "ConvLSTM-3", "asymptotic_model"),
    },
    "fordigit": {
        ("201", "CNN", "meanDia_corrected"),
        ("202", "LSTM-1", "meanDia_corrected"),
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Quantify subject-level noise / data quality and relate it to "
            "outer-LOSO subject-level macro-F1 across saved experiments."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["vr", "fordigit", "both"],
        required=True,
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Experiment timestamp, for example 2024-08-09.",
    )
    parser.add_argument(
        "--config-dir",
        default=CONFIG_DIR,
        help="Directory containing experiment configs.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save publication-style summary plots.",
    )
    return parser.parse_args()


def configure_plot_style():
    """Apply the project plotting style when available."""
    if os.path.exists(STYLE_PATH):
        plt.style.use(STYLE_PATH)
    plt.rcParams["text.usetex"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def style_publication_axis(ax, categorical_x=False):
    """Apply the established publication-style axis formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=18)
    ax.yaxis.labelpad = 10
    if not categorical_x:
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.get_offset_text().set_visible(False)


def ensure_output_dir(configs, dataset):
    """Create the selected-model dataset-level noise-analysis output directory."""
    base_root = configs[0]["DL"]["path_results"]
    timestamp = configs[0]["timestamp"]
    output_dir = os.path.join(
        base_root,
        timestamp,
        f"{dataset}_noise_error_analysis",
        "selected_model_comparison",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def extended_invalid_mask(confidence_values, threshold=0.8, extra_invalid_samples=2):
    """Recreate the confidence-padding invalid-mask logic used before interpolation."""
    confidence_values = np.asarray(confidence_values, dtype=float)
    is_valid = confidence_values > threshold
    invalid_indices = np.where(~is_valid)[0]
    for idx in invalid_indices:
        start_idx = max(0, idx - extra_invalid_samples)
        end_idx = min(len(is_valid) - 1, idx + extra_invalid_samples)
        is_valid[start_idx : end_idx + 1] = False
    return ~is_valid


def compute_vr_subject_noise_metrics(subject_ids):
    """Compute VR subject-level noise metrics using the preprocessing pipeline in-memory."""
    rows = []
    eyes = ["left", "right"]

    for subject_id in sorted(set(int(value) for value in subject_ids)):
        shot_rows = []
        for task in ["noStress", "stress"]:
            raw_path = os.path.join(VR_RAW_ROOT, f"LogID_{subject_id}_{task}.csv")
            if not os.path.exists(raw_path):
                continue

            raw_df = pd.read_csv(
                raw_path,
                sep=";",
                skip_blank_lines=True,
                low_memory=False,
            )[
                [
                    "eye-x",
                    "eye-y",
                    "eye-z",
                    "time",
                    "left pupil size",
                    "right pupil size",
                    "blink left",
                    "blink right",
                    "animation",
                    "tmp stressor",
                    "Task",
                    "Task-Level",
                    "Task-Score",
                    "Task-Response time",
                    "head-x",
                    "head-y",
                    "head-z",
                    "normal time",
                    "focused Object",
                ]
            ]
            raw_df = raw_df.dropna(how="all").convert_dtypes()
            raw_df["ID"] = subject_id

            processed = raw_df[["ID", "normal time", "eye-x"]].copy()
            filt = PdFilter("default")
            for eye in eyes:
                processed = process_vr_goalkeeper.preprocess_pd(raw_df, filt, eye, processed)
            processed = compute_meanDia(raw_df, processed)

            start_shots, end_shots = segment_vr_goalkeeper.find_shots(raw_df)
            for shot_idx in range(len(start_shots)):
                end_time = float(raw_df["normal time"].iloc[end_shots[shot_idx]])
                start_time = end_time - 5.0
                start_idx = segment_vr_goalkeeper.find_nearest(processed["normal time"], start_time)
                end_idx = segment_vr_goalkeeper.find_nearest(processed["normal time"], end_time)
                if end_idx <= start_idx:
                    continue

                proc_window = processed.iloc[start_idx:end_idx].copy()
                raw_window = raw_df.iloc[start_idx:end_idx].copy()
                if proc_window.empty or raw_window.empty:
                    continue

                blink_any = (
                    raw_window["blink left"].fillna(True).astype(bool).to_numpy()
                    | raw_window["blink right"].fillna(True).astype(bool).to_numpy()
                )
                both_invalid = proc_window["bothwithout"].fillna(True).astype(bool).to_numpy()
                right_invalid = ~proc_window["isValidFinalright"].fillna(False).astype(bool).to_numpy()
                left_invalid = ~proc_window["isValidFinalleft"].fillna(False).astype(bool).to_numpy()
                single_eye_only = (
                    proc_window["lwithoutRR"].fillna(False).astype(bool).to_numpy()
                    | proc_window["rwithoutL"].fillna(False).astype(bool).to_numpy()
                )

                shot_rows.append(
                    {
                        "test_id": subject_id,
                        "task": task,
                        "shot": shot_idx,
                        "blink_fraction": float(np.mean(blink_any)),
                        "both_invalid_fraction": float(np.mean(both_invalid)),
                        "right_invalid_fraction": float(np.mean(right_invalid)),
                        "left_invalid_fraction": float(np.mean(left_invalid)),
                        "single_eye_only_fraction": float(np.mean(single_eye_only)),
                        "n_samples_window": int(proc_window.shape[0]),
                    }
                )

        if not shot_rows:
            continue

        shot_df = pd.DataFrame(shot_rows)
        rows.append(
            {
                "test_id": subject_id,
                "n_windows_noise": int(shot_df.shape[0]),
                "blink_fraction": float(shot_df["blink_fraction"].mean()),
                "both_invalid_fraction": float(shot_df["both_invalid_fraction"].mean()),
                "right_invalid_fraction": float(shot_df["right_invalid_fraction"].mean()),
                "left_invalid_fraction": float(shot_df["left_invalid_fraction"].mean()),
                "single_eye_only_fraction": float(shot_df["single_eye_only_fraction"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("test_id").reset_index(drop=True)


def evaluate_fordigit_windows(df, window_length):
    """Count valid and invalid candidate windows using the original extraction criteria."""
    start = 0
    counter_good = 0
    counter_bad = 0

    while start <= len(df) - window_length:
        segment = df.iloc[start : start + window_length].copy()
        idx_last_invalid = process_fordigitstress.get_conf_label(
            segment,
            process_fordigitstress.pdFilt_parameters["confidence"]["confidence_thresh"],
        )
        _, _, idx_last_change = process_fordigitstress.get_label(segment)

        if idx_last_invalid != -1:
            counter_bad += 1

        if idx_last_invalid == -1 and idx_last_change == -1:
            counter_good += 1
            start += int(window_length * 1)
        else:
            start += max(idx_last_invalid, idx_last_change) + 1

    return counter_good, counter_bad


def compute_fordigit_subject_noise_metrics(subject_ids):
    """Compute ForDigitStress subject-level noise metrics using existing preprocessing logic."""
    rows = []
    window_length = (
        process_fordigitstress.settings["windowLength_sec"]
        * process_fordigitstress.settings["samplingRate"]
    )
    conf_thresh = process_fordigitstress.pdFilt_parameters["confidence"]["confidence_thresh"]

    for subject_id in sorted(set(int(value) for value in subject_ids)):
        try:
            df = process_fordigitstress.load_combine_data(subject_id, FORDIGIT_RAW_ROOT + "/")
        except FileNotFoundError:
            continue
        if df is None or isinstance(df, float):
            continue

        confidence = df["confidence"].to_numpy(dtype=float)
        low_conf = confidence <= conf_thresh
        extended_invalid = extended_invalid_mask(confidence, threshold=conf_thresh, extra_invalid_samples=2)

        baseline_dat = df[df.situation == 1]
        base_snipped, _, end_idx = segment_fordigitstress.segment_baseline(
            baseline_dat["pupil_diameter"],
            baseline_dat["time"],
            baseline_dat["confidence"],
            conf_thresh,
        )
        baseline_success = base_snipped is not None
        base_correct_val = float(np.median(base_snipped)) if baseline_success else np.nan
        df_interview = df[df.situation == 0].copy()
        interview_good, interview_bad = evaluate_fordigit_windows(df_interview, window_length)

        post_df = df[df.situation == 2].copy()
        post_start_idx = int(end_idx) + 1 if baseline_success and end_idx is not None else 0
        nostress_df = post_df.iloc[post_start_idx:-1].copy()
        post_good, post_bad = evaluate_fordigit_windows(nostress_df, window_length) if not nostress_df.empty else (0, 0)

        total_checked_windows = interview_good + interview_bad + post_good + post_bad
        rows.append(
            {
                "test_id": subject_id,
                "n_samples_total": int(len(df)),
                "mean_confidence": float(np.mean(confidence)),
                "low_confidence_fraction": float(np.mean(low_conf)),
                "extended_invalid_fraction": float(np.mean(extended_invalid)),
                "baseline_segment_found": bool(baseline_success),
                "baseline_correction_value": base_correct_val,
                "valid_window_fraction": (
                    float((interview_good + post_good) / total_checked_windows)
                    if total_checked_windows > 0
                    else np.nan
                ),
                "invalid_window_fraction": (
                    float((interview_bad + post_bad) / total_checked_windows)
                    if total_checked_windows > 0
                    else np.nan
                ),
                "n_windows_checked": int(total_checked_windows),
            }
        )

    return pd.DataFrame(rows).sort_values("test_id").reset_index(drop=True)


def compute_noise_metrics(dataset, test_ids):
    """Dispatch dataset-specific noise quantification."""
    if dataset == "vr":
        return compute_vr_subject_noise_metrics(test_ids)
    return compute_fordigit_subject_noise_metrics(test_ids)


def get_dataset_slug(dataset):
    """Return a short filename-safe dataset slug."""
    return "vr_goalkeeper" if dataset == "vr" else "fordigitstress"


def filter_to_selected_noise_models(metrics_df, dataset):
    """Keep only the selected model/input combinations used for noise scatter plots."""
    selected_keys = SELECTED_NOISE_MODEL_KEYS[dataset]
    row_keys = list(
        zip(
            metrics_df["experiment_id"].astype(str),
            metrics_df["model"],
            metrics_df["input_cols"],
        )
    )
    filtered_df = metrics_df.loc[[key in selected_keys for key in row_keys]].copy()
    if filtered_df.empty:
        raise ValueError(
            f"No selected noise-analysis model configurations were found for dataset={dataset}."
        )
    return filtered_df.reset_index(drop=True)


def select_noise_columns(dataset, merged_df):
    """Return the core noise columns to analyze for the selected dataset."""
    candidates = get_noise_metric_order(dataset)
    return [column for column in candidates if column in merged_df.columns]


def get_noise_metric_order(dataset):
    """Return the intended display/analysis order for noise metrics."""
    return VR_NOISE_METRIC_ORDER if dataset == "vr" else FORDIGIT_NOISE_METRIC_ORDER


def build_noise_correlation_table(merged_df, noise_columns):
    """Compute experiment-wise Spearman correlations between macro-F1 and noise metrics."""
    rows = []
    grouped = merged_df.groupby(["experiment_id", "model", "input_cols"], dropna=False)
    for (experiment_id, model, input_cols), group in grouped:
        for metric_name in noise_columns:
            paired = group[["macro_f1", metric_name]].dropna()
            if paired.shape[0] < 3:
                continue
            corr, p_value = spearmanr(paired["macro_f1"], paired[metric_name])
            rows.append(
                {
                    "experiment_id": str(experiment_id),
                    "model": model,
                    "input_cols": input_cols,
                    "model_display": prettify_model_name(model),
                    "input_display": prettify_input_cols(input_cols),
                    "combination_display": f"{prettify_model_name(model)} ({prettify_input_cols(input_cols)})",
                    "noise_metric": metric_name,
                    "spearman_rho": float(corr),
                    "p_value": float(p_value),
                    "n_subjects": int(paired.shape[0]),
                }
            )
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df
    corr_df = add_combination_order_columns(corr_df)
    return corr_df.sort_values(["input_order", "model_order", "noise_metric"]).reset_index(drop=True)


def slugify(value):
    """Create a filename-safe slug."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def format_noise_metric_label(metric_name):
    """Convert noise-metric column names into readable axis labels."""
    label = metric_name.replace("_", " ")
    return label[:1].upper() + label[1:]


def select_scatter_plot_columns(noise_columns):
    """Keep the same noise proxies across scatter and correlation summary plots."""
    return noise_columns[:4]


def add_combination_order_columns(df):
    """Add stable ordering columns matching the manuscript result tables."""
    if df.empty:
        return df.copy()
    ordered = df.copy()
    input_order_map = {name: idx for idx, name in enumerate(INPUT_COL_ORDER_RAW)}
    model_order_map = {name: idx for idx, name in enumerate(MODEL_ORDER_RAW)}
    ordered["input_order"] = ordered["input_cols"].map(input_order_map).fillna(len(input_order_map)).astype(int)
    ordered["model_order"] = ordered["model"].map(model_order_map).fillna(len(model_order_map)).astype(int)
    return ordered


def save_per_combination_scatter_plots(merged_df, noise_columns, output_dir, dataset):
    """Save one scatter-plot figure per model/input combination."""
    if not noise_columns:
        return

    combo_dir = os.path.join(output_dir, "combination_plots")
    os.makedirs(combo_dir, exist_ok=True)
    dataset_slug = get_dataset_slug(dataset)

    ordered_df = add_combination_order_columns(merged_df)
    grouped = ordered_df.groupby(
        ["experiment_id", "model", "input_cols", "model_display", "input_display"],
        dropna=False,
        sort=False,
    )
    plot_columns = select_scatter_plot_columns(noise_columns)

    for (_, _, _, model_display, input_display), group in grouped:
        n_cols = min(2, len(plot_columns))
        n_rows = int(np.ceil(len(plot_columns) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12.0, max(4.8, n_rows * 4.2)),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, metric_name in zip(axes, plot_columns):
            plot_df = group[["macro_f1", metric_name]].dropna()
            ax.scatter(
                plot_df[metric_name],
                plot_df["macro_f1"],
                color=SECONDARY_COLOR,
                edgecolor=PRIMARY_COLOR,
                linewidth=0.6,
                s=34,
                alpha=0.85,
            )
            if plot_df.shape[0] >= 3:
                corr, _ = spearmanr(plot_df[metric_name], plot_df["macro_f1"])
                ax.text(
                    0.98,
                    0.02,
                    f"Spearman rho = {corr:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=14,
                    color=PRIMARY_COLOR,
                )
            ax.set_xlabel(format_noise_metric_label(metric_name), fontsize=17)
            ax.set_ylabel("Macro F1-score", fontsize=17)
            ax.set_ylim(0.0, 1.05)
            style_publication_axis(ax)

        for ax in axes[len(plot_columns) :]:
            ax.axis("off")

        combo_title = f"{model_display} ({input_display})"
        fig.suptitle(
            f"Subject-level performance vs noise\n{combo_title}",
            fontsize=23,
        )
        combo_slug = slugify(f"{model_display}_{input_display}")
        fig.savefig(
            os.path.join(combo_dir, f"{dataset_slug}_noise_vs_macro_f1_{combo_slug}.png"),
            dpi=250,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(combo_dir, f"{dataset_slug}_noise_vs_macro_f1_{combo_slug}.pdf"),
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_noise_correlation_heatmap(corr_df, output_dir, dataset):
    """Save a heatmap of noise/performance correlations by model/input combination."""
    if corr_df.empty:
        return

    matrix = corr_df.pivot_table(
        index="combination_display",
        columns="noise_metric",
        values="spearman_rho",
        aggfunc="first",
    )
    row_order_df = (
        corr_df.loc[:, ["combination_display", "input_order", "model_order"]]
        .drop_duplicates()
        .sort_values(["input_order", "model_order", "combination_display"])
    )
    matrix = matrix.reindex(row_order_df["combination_display"].tolist())
    metric_order = get_noise_metric_order(dataset)
    matrix = matrix.loc[:, [col for col in metric_order if col in matrix.columns]]

    fig, ax = plt.subplots(
        figsize=(max(9.0, matrix.shape[1] * 2.1), max(4.8, matrix.shape[0] * 0.9)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=HEATMAP_CMAP, vmin=-1.0, vmax=1.0)
    ax.set_title("Noise-performance correlation", fontsize=24)
    ax.set_xlabel("Noise proxy", fontsize=22)
    ax.set_ylabel("Model (input signal)", fontsize=22)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [format_noise_metric_label(name).replace(" ", "\n") for name in matrix.columns],
        fontsize=12,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([label.replace(" (", "\n(") for label in matrix.index], fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Spearman rho", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    dataset_slug = get_dataset_slug(dataset)
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_noise_performance_correlation_heatmap.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_noise_performance_correlation_heatmap.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_noise_correlation_matrix(corr_df, dataset):
    """Return the ordered heatmap matrix used for noise-performance correlations."""
    matrix = corr_df.pivot_table(
        index="combination_display",
        columns="noise_metric",
        values="spearman_rho",
        aggfunc="first",
    )
    row_order_df = (
        corr_df.loc[:, ["combination_display", "input_order", "model_order"]]
        .drop_duplicates()
        .sort_values(["input_order", "model_order", "combination_display"])
    )
    matrix = matrix.reindex(row_order_df["combination_display"].tolist())
    metric_order = get_noise_metric_order(dataset)
    return matrix.loc[:, [col for col in metric_order if col in matrix.columns]]


def draw_noise_correlation_heatmap_axis(ax, corr_df, dataset, title):
    """Draw a compact noise-performance heatmap on an existing axis."""
    matrix = build_noise_correlation_matrix(corr_df, dataset)
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=HEATMAP_CMAP, vmin=-1.0, vmax=1.0)
    ax.set_title(title, fontsize=18, linespacing=1.12)
    ax.set_xlabel("Noise proxy", fontsize=19, labelpad=8)
    ax.set_ylabel("Model (input signal)", fontsize=19, labelpad=8)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        [format_noise_metric_label(name).replace(" ", "\n") for name in matrix.columns],
        fontsize=13,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([label.replace(" (", "\n(") for label in matrix.index], fontsize=13)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            if np.isnan(value):
                continue
            text_color = "white" if value > 0.55 else "#10243E"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
            )

    return im


def get_combined_output_dir(timestamp):
    """Return a shared output directory for cross-dataset noise-analysis plots."""
    output_dir = os.path.join(os.path.dirname(LOCAL_DATA_ROOT), "results", "error_analysis", str(timestamp))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_combined_noise_correlation_heatmap(corr_by_dataset, output_dir):
    """Save VR and ForDigitStress noise-performance heatmaps side by side."""
    if corr_by_dataset.get("vr") is None or corr_by_dataset.get("fordigit") is None:
        return
    if corr_by_dataset["vr"].empty or corr_by_dataset["fordigit"].empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    last_im = draw_noise_correlation_heatmap_axis(
        axes[0],
        corr_by_dataset["vr"],
        "vr",
        "VR goalkeeper dataset",
    )
    last_im = draw_noise_correlation_heatmap_axis(
        axes[1],
        corr_by_dataset["fordigit"],
        "fordigit",
        "ForDigitStress dataset",
    )
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.86, location="right")
    cbar.set_label("Spearman rho", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    output_base = os.path.join(output_dir, "noise_performance_correlation_heatmap_combined")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(output_dir, noise_df, merged_df, corr_df, dataset, timestamp):
    """Save CSV outputs and metadata."""
    noise_df.to_csv(os.path.join(output_dir, "subject_noise_metrics.csv"), index=False)
    merged_df.to_csv(os.path.join(output_dir, "subject_noise_error_joined.csv"), index=False)
    corr_df.to_csv(os.path.join(output_dir, "noise_error_correlations_by_experiment.csv"), index=False)

    metadata = {
        "dataset": dataset,
        "dataset_display_name": get_dataset_display_name(dataset),
        "timestamp": timestamp,
        "selection": "selected_model_comparison",
        "analysis_type": "read_only_noise_error_analysis",
    }
    with open(os.path.join(output_dir, "noise_error_analysis_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def print_summary(noise_df, corr_df, dataset):
    """Print a short console summary."""
    print(f"\nNoise-aware error analysis summary for {get_dataset_display_name(dataset)}")
    print("------------------------------------------------------------")
    print(f"Subjects with computed noise metrics: {noise_df.shape[0]}")
    if corr_df.empty:
        print("No experiment-wise noise/performance correlations could be computed.")
        return

    row_order_df = (
        corr_df.loc[:, ["combination_display", "input_order", "model_order"]]
        .drop_duplicates()
        .sort_values(["input_order", "model_order", "combination_display"])
    )
    for combination_name in row_order_df["combination_display"].tolist():
        group = corr_df[corr_df["combination_display"] == combination_name]
        strongest = (
            group.assign(abs_rho=group["spearman_rho"].abs())
            .sort_values("abs_rho", ascending=False)
            .head(2)
        )
        print(combination_name)
        for _, row in strongest.iterrows():
            print(f"  {row['noise_metric']}: rho={row['spearman_rho']:.3f}, p={row['p_value']:.4g}")


def run_analysis(configs, dataset, timestamp, save_plots):
    """Run the selected-model noise-aware analysis and save outputs."""
    output_dir = ensure_output_dir(configs, dataset)

    metrics_df = aggregate_all_metrics(configs)
    metrics_df = filter_to_selected_noise_models(metrics_df, dataset)
    noise_df = compute_noise_metrics(dataset, metrics_df["test_id"].unique())
    merged_df = metrics_df.merge(noise_df, on="test_id", how="left", validate="many_to_one")
    noise_columns = select_noise_columns(dataset, merged_df)
    plot_noise_columns = select_scatter_plot_columns(noise_columns)
    corr_df = build_noise_correlation_table(merged_df, plot_noise_columns)

    save_outputs(
        output_dir,
        noise_df,
        merged_df,
        corr_df,
        dataset,
        timestamp,
    )

    if save_plots:
        save_noise_correlation_heatmap(corr_df, output_dir, dataset)
        save_per_combination_scatter_plots(merged_df, plot_noise_columns, output_dir, dataset)

    print(f"\nSaved selected-model comparison to {output_dir}")
    print_summary(noise_df, corr_df, dataset)
    return {
        "output_dir": output_dir,
        "noise_df": noise_df,
        "merged_df": merged_df,
        "corr_df": corr_df,
    }


def main():
    """Run the noise-aware subject-level error analysis."""
    args = parse_args()
    configure_plot_style()

    if args.dataset == "both":
        results_by_dataset = {}
        for dataset in ["vr", "fordigit"]:
            configs = discover_matching_configs(
                config_dir=args.config_dir,
                dataset=dataset,
                timestamp=args.timestamp,
            )
            results_by_dataset[dataset] = run_analysis(
                configs=configs,
                dataset=dataset,
                timestamp=args.timestamp,
                save_plots=args.save_plots,
            )
        if args.save_plots:
            combined_output_dir = get_combined_output_dir(args.timestamp)
            save_combined_noise_correlation_heatmap(
                corr_by_dataset={
                    dataset: results["corr_df"]
                    for dataset, results in results_by_dataset.items()
                },
                output_dir=combined_output_dir,
            )
            print(f"\nSaved combined noise-performance heatmap to {combined_output_dir}")
        return

    configs = discover_matching_configs(
        config_dir=args.config_dir,
        dataset=args.dataset,
        timestamp=args.timestamp,
    )
    run_analysis(
        configs=configs,
        dataset=args.dataset,
        timestamp=args.timestamp,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
