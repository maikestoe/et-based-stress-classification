"""
Subject-level error analysis across saved deep-learning experiments.

This script performs a practical error analysis based on the held-out
outer-LOSO test subjects. It does not recompute predictions. Instead, it
reuses per-test-ID outer-fold macro-F1 metrics already saved in the result
files and summarizes subject-specific difficulty across experiments.

Why this analysis is useful
---------------------------
- identifies subjects that are consistently difficult across models
- highlights whether errors are concentrated in a subset of individuals
- supports discussion of robustness, inter-subject variability, and possible
  label ambiguity or subject-specific noise

Datasets
--------
- VR goalkeeper: experiments exp1 ... exp42
- ForDigitStress: experiments exp201 ... exp205
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
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import transforms

try:
    from fau_colors import colors, colors_dark
except ImportError:
    colors = None
    colors_dark = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config_files")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOCAL_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
LOCAL_RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
STYLE_PATH = os.path.join(SCRIPT_DIR, "plot_style2.txt")
PRIMARY_COLOR = "#465368"
SECONDARY_COLOR = "#90AEC6"
ACCENT_COLOR = "#C44F5E"
COMPARISON_PALETTE = ["#F1F1F1", "#D9DEE7", "#BFC8D6", "#91A2BB", "#6C85A5", "#123B6D"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("error_analysis_heatmap", COMPARISON_PALETTE)
RANK_CORR_CMAP = LinearSegmentedColormap.from_list(
    "subject_difficulty_rank_corr",
    [
        (0.00, "#E5C04A"),
        (0.25, "#C8C56A"),
        (0.50, "#8DBB6C"),
        (0.75, "#6C85A5"),
        (1.00, "#123B6D"),
    ],
)
MODEL_NAME_MAP = {
    "CNN": "CNN",
    "LSTM-1": "LSTM-1",
    "LSTM-3": "LSTM-3",
    "ConvLSTM-1": "ConvLSTM-1",
    "ConvLSTM-3": "ConvLSTM-3",
}
INPUT_NAME_MAP = {
    "meanDia_corrected": "PD",
    "velocity": "Angular velocity",
    "acceleration": "Angular acceleration",
    "position": "Visual angle",
    "asymptotic_model": "Asymptotic model",
    "fix_array": "Fixations",
}
MODEL_ORDER = ["CNN", "LSTM-1", "ConvLSTM-1", "LSTM-3", "ConvLSTM-3"]
INPUT_ORDER = [
    "PD",
    "Angular velocity",
    "Angular acceleration",
    "Visual angle",
    "Asymptotic model",
    "Fixations",
]
VR_BEST_INPUT_PER_ARCHITECTURE_EXPERIMENTS = ["4", "15", "20", "32", "41"]
VR_BEST_ARCHITECTURE_PER_INPUT_EXPERIMENTS = ["1", "38", "21", "4", "41", "6"]

if colors_dark is not None and hasattr(colors_dark, "tech"):
    PRIMARY_COLOR = colors_dark.tech
if colors is not None and hasattr(colors, "med"):
    ACCENT_COLOR = colors.med


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run subject-level error analysis across saved experiments using "
            "outer-LOSO test-subject macro-F1 scores."
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
        "--bottom-k",
        type=int,
        default=5,
        help="How many hardest subjects per experiment should be highlighted.",
    )
    parser.add_argument(
        "--difficulty-threshold",
        type=float,
        default=0.8,
        help="Macro-F1 threshold below which a subject is counted as difficult.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save simple heatmap/barplot summaries.",
    )
    return parser.parse_args()


def prettify_model_name(model_name):
    """Map internal model names to natural manuscript-style labels."""
    return MODEL_NAME_MAP.get(model_name, model_name)


def prettify_input_cols(input_cols):
    """Map internal input names to natural display labels."""
    parts = [part.strip() for part in str(input_cols).split(",") if part.strip()]
    pretty = [INPUT_NAME_MAP.get(part, part) for part in parts]
    return ", ".join(pretty)


def get_dataset_display_name(dataset):
    """Return the full manuscript-style dataset name for titles."""
    if dataset == "vr":
        return "VR goalkeeper dataset"
    if dataset == "fordigit":
        return "ForDigitStress dataset"
    return str(dataset)


def get_dataset_slug(dataset):
    """Return a short filename-safe dataset slug."""
    if dataset == "vr":
        return "vr_goalkeeper"
    if dataset == "fordigit":
        return "fordigitstress"
    return str(dataset).strip().lower().replace(" ", "_")


def build_experiment_display_label(experiment_id, model, input_cols):
    """Create a short natural display label for one experiment."""
    return f"{prettify_model_name(model)} ({prettify_input_cols(input_cols)})"


def wrap_experiment_display_label(experiment_id, model, input_cols):
    """Wrap a display label across lines for crowded x-axes."""
    return f"{prettify_model_name(model)}\n({prettify_input_cols(input_cols)})"


def _sort_key_for_heatmap_label(meta):
    model = meta.get("model", "")
    input_signal = meta.get("input_signal", "")
    model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    input_rank = INPUT_ORDER.index(input_signal) if input_signal in INPUT_ORDER else len(INPUT_ORDER)
    return model_rank, input_rank, meta.get("label", "")


def normalize_project_path(path_value, kind):
    """Map old cluster-style paths onto the local project layout."""
    expanded = os.path.expandvars(path_value)
    normalized = expanded.replace("\\", "/")

    if "data/" in normalized:
        suffix = normalized.split("data/", 1)[1]
        return os.path.join(LOCAL_DATA_ROOT, suffix)
    if "results/" in normalized:
        suffix = normalized.split("results/", 1)[1]
        return os.path.join(LOCAL_RESULTS_ROOT, suffix)

    if kind == "data" and normalized.startswith("data/"):
        suffix = normalized.split("data/", 1)[1]
        return os.path.join(LOCAL_DATA_ROOT, suffix)
    if kind == "results" and normalized.startswith("results/"):
        suffix = normalized.split("results/", 1)[1]
        return os.path.join(LOCAL_RESULTS_ROOT, suffix)

    return expanded


def load_json_config(config_path):
    """Load and normalize an experiment config."""
    with open(config_path, "r") as file:
        config = json.load(file)
    config["DL"]["path_results"] = normalize_project_path(config["DL"]["path_results"], kind="results")
    config["df_prep_path"] = normalize_project_path(config["df_prep_path"], kind="data")
    config["_config_path"] = config_path
    return config


def get_base_folder(config):
    """Return the experiment root result folder."""
    return os.path.join(
        config["DL"]["path_results"],
        config["timestamp"],
        config["DL"]["model"],
        str(config["experiment_id"]),
    )


def result_folder_exists(config):
    """Check whether the experiment folder exists locally."""
    return os.path.isdir(get_base_folder(config))


def discover_matching_configs(config_dir, dataset, timestamp):
    """Discover the expected saved configs for the requested dataset/timestamp."""
    experiment_ids = list(range(1, 43)) if dataset == "vr" else list(range(201, 206))
    configs = []

    for experiment_id in experiment_ids:
        config_path = os.path.join(config_dir, f"exp{experiment_id}.json")
        if not os.path.exists(config_path):
            continue
        config = load_json_config(config_path)
        if str(config.get("timestamp")) != str(timestamp):
            continue
        if result_folder_exists(config):
            configs.append(config)

    if not configs:
        raise FileNotFoundError(
            f"No saved experiment configs found for dataset={dataset} and timestamp={timestamp}."
        )
    return configs


def ensure_output_dir(configs, dataset):
    """Create the dataset-level error-analysis output directory."""
    base_root = configs[0]["DL"]["path_results"]
    timestamp = configs[0]["timestamp"]
    output_dir = os.path.join(base_root, timestamp, f"{dataset}_error_analysis_subjects")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def parse_final_summary_performance(summary_path):
    """
    Parse per-test-ID macro-F1 lines from final_summary.txt.

    The original training pipeline writes lines like:
    - Performance for test id 29:
    -  Macro f1 score: 0.95
    """
    rows = []
    current_test_id = None
    pattern_test_id = re.compile(r"Performance for test id (\d+):")
    pattern_macro_f1 = re.compile(r"Macro f1 score:\s*([0-9eE+\-.]+)")

    with open(summary_path, "r") as file:
        for line in file:
            stripped = line.strip()
            match_test_id = pattern_test_id.search(stripped)
            if match_test_id:
                current_test_id = int(match_test_id.group(1))
                continue

            match_f1 = pattern_macro_f1.search(stripped)
            if match_f1 and current_test_id is not None:
                rows.append(
                    {
                        "test_id": current_test_id,
                        "macro_f1": float(match_f1.group(1)),
                    }
                )
                current_test_id = None

    if not rows:
        raise ValueError(f"No per-test-ID performance lines found in {summary_path}")
    return pd.DataFrame(rows)


def load_experiment_fold_metrics(config):
    """
    Load subject-level outer-fold metrics for one experiment.

    Preference order:
    1. outer_metrics_recovered.csv if present
    2. parse final_summary.txt
    """
    base_folder = get_base_folder(config)
    candidate_csv_paths = [
        os.path.join(base_folder, "outer_metrics_recovered.csv"),
        os.path.join(base_folder, "recovery", "outer_metrics_recovered.csv"),
    ]

    metrics_df = None
    for path in candidate_csv_paths:
        if os.path.exists(path):
            metrics_df = pd.read_csv(path)
            break

    if metrics_df is None:
        summary_path = os.path.join(base_folder, "final_summary.txt")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Neither outer_metrics_recovered.csv nor final_summary.txt found in {base_folder}."
            )
        metrics_df = parse_final_summary_performance(summary_path)

    if "test_id" not in metrics_df.columns:
        raise ValueError(f"Expected 'test_id' column in metrics for {base_folder}.")
    if "macro_f1" not in metrics_df.columns:
        raise ValueError(f"Expected 'macro_f1' column in metrics for {base_folder}.")

    keep_cols = [col for col in ["test_id", "macro_f1", "precision_macro", "recall_macro", "roc_auc"] if col in metrics_df.columns]
    metrics_df = metrics_df.loc[:, keep_cols].copy()
    metrics_df["test_id"] = metrics_df["test_id"].astype(int)
    metrics_df["macro_f1"] = metrics_df["macro_f1"].astype(float)
    metrics_df["experiment_id"] = str(config["experiment_id"])
    metrics_df["model"] = config["DL"]["model"]
    metrics_df["input_cols"] = ",".join(config["DL"]["input_cols"])
    metrics_df["model_display"] = prettify_model_name(config["DL"]["model"])
    metrics_df["input_display"] = prettify_input_cols(",".join(config["DL"]["input_cols"]))
    metrics_df["experiment_display"] = build_experiment_display_label(
        config["experiment_id"],
        config["DL"]["model"],
        ",".join(config["DL"]["input_cols"]),
    )
    metrics_df["config_path"] = config["_config_path"]
    return metrics_df.sort_values("test_id").reset_index(drop=True)


def aggregate_all_metrics(configs):
    """Collect per-subject outer-fold metrics across all discovered experiments."""
    frames = []
    for config in configs:
        frames.append(load_experiment_fold_metrics(config))
    return pd.concat(frames, ignore_index=True)


def build_subject_overall_summary(metrics_df, difficulty_threshold):
    """Summarize how difficult each subject is across all models."""
    summary = (
        metrics_df.groupby("test_id", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            median_macro_f1=("macro_f1", "median"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
            n_experiments=("macro_f1", "count"),
            num_below_threshold=("macro_f1", lambda x: int(np.sum(np.asarray(x) < difficulty_threshold))),
        )
        .sort_values(["mean_macro_f1", "median_macro_f1", "test_id"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return summary


def build_experiment_summary(metrics_df):
    """Summarize subject-level spread per experiment."""
    summary = (
        metrics_df.groupby(["experiment_id", "model", "input_cols"], as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            min_macro_f1=("macro_f1", "min"),
            median_macro_f1=("macro_f1", "median"),
            max_macro_f1=("macro_f1", "max"),
            n_subjects=("macro_f1", "count"),
        )
        .sort_values(["mean_macro_f1", "std_macro_f1"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["model_display"] = summary["model"].map(prettify_model_name)
    summary["input_display"] = summary["input_cols"].map(prettify_input_cols)
    summary["experiment_display"] = summary.apply(
        lambda row: build_experiment_display_label(row["experiment_id"], row["model"], row["input_cols"]),
        axis=1,
    )
    return summary


def build_hardest_subjects_table(metrics_df, bottom_k):
    """Return the bottom-k subjects per experiment by macro-F1."""
    rows = []
    grouped = metrics_df.groupby(["experiment_id", "model", "input_cols"], dropna=False)
    for (experiment_id, model, input_cols), group in grouped:
        hardest = group.sort_values(["macro_f1", "test_id"], ascending=[True, True]).head(bottom_k)
        for rank, (_, row) in enumerate(hardest.iterrows(), start=1):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model": model,
                    "input_cols": input_cols,
                    "experiment_display": build_experiment_display_label(experiment_id, model, input_cols),
                    "rank_within_experiment": rank,
                    "test_id": int(row["test_id"]),
                    "macro_f1": float(row["macro_f1"]),
                }
            )
    return pd.DataFrame(rows)


def build_subject_consistency_table(hardest_df):
    """Count how often each subject appears among the hardest cases."""
    if hardest_df.empty:
        return pd.DataFrame(columns=["test_id", "times_in_bottom_k"])

    summary = (
        hardest_df.groupby("test_id", as_index=False)
        .agg(times_in_bottom_k=("experiment_id", "count"))
        .sort_values(["times_in_bottom_k", "test_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def build_model_subject_matrix(metrics_df):
    """Build a subject-by-experiment matrix of macro-F1 scores."""
    matrix = metrics_df.pivot_table(
        index="test_id",
        columns=["experiment_id", "model", "input_cols"],
        values="macro_f1",
        aggfunc="first",
    )
    return matrix.sort_index()


def build_rank_correlation_table(metrics_df):
    """
    Compute subject-difficulty correlation across experiments.

    The correlation is based on subject-wise macro-F1 values. High positive
    correlation means the same subjects tend to be easy/hard across models.
    """
    matrix = build_model_subject_matrix(metrics_df)
    correlations = []
    columns = list(matrix.columns)
    for idx_a in range(len(columns)):
        for idx_b in range(idx_a + 1, len(columns)):
            col_a = columns[idx_a]
            col_b = columns[idx_b]
            paired = matrix.loc[:, [col_a, col_b]].dropna()
            if paired.shape[0] < 3:
                continue
            corr = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
            correlations.append(
                {
                    "experiment_a": col_a[0],
                    "model_a": col_a[1],
                    "input_cols_a": col_a[2],
                    "experiment_b": col_b[0],
                    "model_b": col_b[1],
                    "input_cols_b": col_b[2],
                    "n_subjects_overlap": int(paired.shape[0]),
                    "spearman_subject_macro_f1": float(corr),
                }
            )
    return pd.DataFrame(correlations).sort_values(
        "spearman_subject_macro_f1", ascending=False
    ).reset_index(drop=True)


def filter_metrics_by_experiment_ids(metrics_df, experiment_ids):
    """Keep rows for a fixed ordered set of experiment IDs."""
    experiment_ids = [str(experiment_id) for experiment_id in experiment_ids]
    filtered = metrics_df[metrics_df["experiment_id"].astype(str).isin(experiment_ids)].copy()
    filtered["_experiment_order"] = filtered["experiment_id"].astype(str).map(
        {experiment_id: idx for idx, experiment_id in enumerate(experiment_ids)}
    )
    return filtered.sort_values(["_experiment_order", "test_id"]).drop(columns=["_experiment_order"])


def _sort_key_for_reduced_label(meta, sort_mode):
    if sort_mode == "input":
        input_signal = meta.get("input_signal", "")
        input_rank = INPUT_ORDER.index(input_signal) if input_signal in INPUT_ORDER else len(INPUT_ORDER)
        model = meta.get("model", "")
        model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
        return input_rank, model_rank, meta.get("label", "")
    return _sort_key_for_heatmap_label(meta)


def build_rank_correlation_matrix(rank_corr_df, sort_mode="model"):
    """Build a compact symmetric rank-correlation matrix and label metadata."""
    if rank_corr_df.empty:
        return pd.DataFrame(), {}

    label_meta = {}
    for _, row in rank_corr_df.iterrows():
        label_a = build_experiment_display_label(row["experiment_a"], row["model_a"], row["input_cols_a"])
        label_b = build_experiment_display_label(row["experiment_b"], row["model_b"], row["input_cols_b"])
        label_meta[label_a] = {
            "label": label_a,
            "model": prettify_model_name(row["model_a"]),
            "input_signal": prettify_input_cols(row["input_cols_a"]),
        }
        label_meta[label_b] = {
            "label": label_b,
            "model": prettify_model_name(row["model_b"]),
            "input_signal": prettify_input_cols(row["input_cols_b"]),
        }

    labels = [item["label"] for item in sorted(label_meta.values(), key=lambda item: _sort_key_for_reduced_label(item, sort_mode))]
    matrix_array = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(matrix_array, 1.0)
    matrix = pd.DataFrame(matrix_array, index=labels, columns=labels)

    for _, row in rank_corr_df.iterrows():
        label_a = build_experiment_display_label(row["experiment_a"], row["model_a"], row["input_cols_a"])
        label_b = build_experiment_display_label(row["experiment_b"], row["model_b"], row["input_cols_b"])
        matrix.loc[label_a, label_b] = row["spearman_subject_macro_f1"]
        matrix.loc[label_b, label_a] = row["spearman_subject_macro_f1"]

    return matrix, label_meta


def make_compact_correlation_label(label, label_meta, label_mode):
    """Return short multi-line labels for compact reduced heatmaps."""
    meta = label_meta[label]
    if label_mode == "architecture":
        return f"{meta['model']}\n({meta['input_signal']})"
    if label_mode == "signal":
        return f"{meta['input_signal']}\n({meta['model']})"
    return label.replace(" (", "\n(")


def draw_compact_rank_correlation_heatmap(ax, matrix, label_meta, title, label_mode):
    """Draw a compact annotated rank-correlation heatmap."""
    matrix_values = matrix.to_numpy(dtype=float)
    matrix_values = np.ma.masked_where(np.triu(np.ones_like(matrix_values, dtype=bool), k=1), matrix_values)
    im = ax.imshow(matrix_values, aspect="equal", cmap=RANK_CORR_CMAP, vmin=-1.0, vmax=1.0)
    labels = matrix.index.tolist()
    compact_labels = [make_compact_correlation_label(label, label_meta, label_mode) for label in labels]
    ax.set_title(title, fontsize=20, linespacing=1.12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(compact_labels, rotation=55, ha="right", fontsize=13)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(compact_labels, fontsize=14)
    ax.tick_params(axis="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            if col_idx > row_idx:
                continue
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


def save_combined_reduced_rank_correlation_heatmaps(vr_metrics_df, fordigit_metrics_df, output_dir):
    """Save reduced subject-difficulty agreement heatmaps in one figure."""
    panels = [
        (
            "VR goalkeeper dataset\nBest input per architecture",
            filter_metrics_by_experiment_ids(vr_metrics_df, VR_BEST_INPUT_PER_ARCHITECTURE_EXPERIMENTS),
            "architecture",
        ),
        (
            "VR goalkeeper dataset\nBest architecture per input",
            filter_metrics_by_experiment_ids(vr_metrics_df, VR_BEST_ARCHITECTURE_PER_INPUT_EXPERIMENTS),
            "signal",
        ),
        (
            "ForDigitStress dataset\nPD architectures",
            fordigit_metrics_df,
            "architecture",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.8), constrained_layout=True)
    last_im = None
    for ax, (title, panel_metrics_df, label_mode) in zip(axes, panels):
        rank_corr_df = build_rank_correlation_table(panel_metrics_df)
        sort_mode = "input" if label_mode == "signal" else "model"
        matrix, label_meta = build_rank_correlation_matrix(rank_corr_df, sort_mode=sort_mode)
        last_im = draw_compact_rank_correlation_heatmap(ax, matrix, label_meta, title, label_mode)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.82, location="right")
    cbar.set_label("Spearman correlation", fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, "subject_difficulty_rank_correlation_reduced_combined")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def save_consistency_barplot(consistency_df, output_dir, dataset, top_n=5, total_experiments=None):
    """Save a barplot showing how often subjects appear among hardest cases."""
    if consistency_df.empty:
        return

    plot_df = consistency_df.head(top_n).copy()
    if total_experiments is None or total_experiments <= 0:
        total_experiments = max(int(consistency_df["times_in_bottom_k"].max()), 1)
    plot_df["percent_of_experiments"] = 100.0 * plot_df["times_in_bottom_k"] / float(total_experiments)

    fig, ax = plt.subplots(figsize=(11.8, 6.0), constrained_layout=True)
    draw_consistency_barplot_axis(ax, plot_df, dataset, top_n=top_n, show_ylabel=True)

    dataset_slug = get_dataset_slug(dataset)
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_consistency.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_consistency.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def draw_consistency_barplot_axis(ax, plot_df, dataset, top_n=5, show_ylabel=True):
    """Draw a participant bottom-k frequency barplot onto an existing axis."""
    ax.bar(
        plot_df["test_id"].astype(str),
        plot_df["percent_of_experiments"],
        color=SECONDARY_COLOR,
        edgecolor="none",
    )
    ax.set_title(get_dataset_display_name(dataset), fontsize=24)
    ax.set_xlabel("Test participant", fontsize=22)
    if show_ylabel:
        ax.set_ylabel(f"Frequency among bottom-{top_n}%", fontsize=21)
    ax.set_ylim(0, 100)
    style_publication_axis(ax, categorical_x=True)

    return ax


def save_combined_consistency_barplot(consistency_by_dataset, output_dir, bottom_k=5, total_experiments_by_dataset=None):
    """Save VR and ForDigitStress bottom-k frequency plots side by side."""
    total_experiments_by_dataset = total_experiments_by_dataset or {}
    datasets = ["vr", "fordigit"]
    plot_data = {}
    for dataset in datasets:
        consistency_df = consistency_by_dataset.get(dataset)
        if consistency_df is None or consistency_df.empty:
            return
        total_experiments = total_experiments_by_dataset.get(dataset)
        if total_experiments is None or total_experiments <= 0:
            total_experiments = max(int(consistency_df["times_in_bottom_k"].max()), 1)
        plot_df = consistency_df.head(bottom_k).copy()
        plot_df["percent_of_experiments"] = 100.0 * plot_df["times_in_bottom_k"] / float(total_experiments)
        plot_data[dataset] = plot_df

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharey=True, constrained_layout=True)
    draw_consistency_barplot_axis(axes[0], plot_data["vr"], "vr", top_n=bottom_k, show_ylabel=True)
    draw_consistency_barplot_axis(axes[1], plot_data["fordigit"], "fordigit", top_n=bottom_k, show_ylabel=False)
    axes[1].tick_params(axis="y", labelleft=False)

    output_base = os.path.join(output_dir, "subject_difficulty_consistency_combined")
    fig.savefig(f"{output_base}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_subject_difficulty_profile(subject_summary_df, output_dir, dataset, top_n=20):
    """Save a ranked subject-difficulty profile with variability bands."""
    dataset_name = get_dataset_display_name(dataset)
    if subject_summary_df.empty:
        return

    plot_df = subject_summary_df.sort_values("mean_macro_f1", ascending=True).head(top_n).copy()
    plot_df["subject_label"] = plot_df["test_id"].astype(str)
    x = np.arange(plot_df.shape[0])

    fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    ax.bar(
        x,
        plot_df["mean_macro_f1"],
        color=SECONDARY_COLOR,
        edgecolor="none",
        zorder=2,
    )
    yerr = plot_df["std_macro_f1"].fillna(0.0).to_numpy()
    ax.errorbar(
        x,
        plot_df["mean_macro_f1"],
        yerr=yerr,
        fmt="none",
        ecolor=PRIMARY_COLOR,
        elinewidth=1.1,
        capsize=3,
        zorder=3,
    )
    ax.scatter(
        x,
        plot_df["min_macro_f1"],
        color=ACCENT_COLOR,
        s=22,
        zorder=4,
        label="Minimum across experiments",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subject_label"], rotation=0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Hardest participants by average macro F1-score", fontsize=24)
    ax.set_xlabel("Test participant", fontsize=22)
    ax.set_ylabel("Macro F1-score", fontsize=22)
    ax.legend(
        frameon=False,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
    )
    style_publication_axis(ax, categorical_x=True)

    dataset_slug = get_dataset_slug(dataset)
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_profile.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_profile.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_rank_correlation_heatmap(rank_corr_df, output_dir, dataset):
    """Save a heatmap of subject-difficulty rank correlation across experiments."""
    if rank_corr_df.empty:
        return

    label_meta = {}
    for _, row in rank_corr_df.iterrows():
        label_a = build_experiment_display_label(row["experiment_a"], row["model_a"], row["input_cols_a"])
        label_b = build_experiment_display_label(row["experiment_b"], row["model_b"], row["input_cols_b"])
        label_meta[label_a] = {
            "label": label_a,
            "model": prettify_model_name(row["model_a"]),
            "input_signal": prettify_input_cols(row["input_cols_a"]),
        }
        label_meta[label_b] = {
            "label": label_b,
            "model": prettify_model_name(row["model_b"]),
            "input_signal": prettify_input_cols(row["input_cols_b"]),
        }

    labels = [item["label"] for item in sorted(label_meta.values(), key=_sort_key_for_heatmap_label)]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    matrix_array = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(matrix_array, 1.0)
    matrix = pd.DataFrame(matrix_array, index=labels, columns=labels)

    for _, row in rank_corr_df.iterrows():
        label_a = build_experiment_display_label(row["experiment_a"], row["model_a"], row["input_cols_a"])
        label_b = build_experiment_display_label(row["experiment_b"], row["model_b"], row["input_cols_b"])
        matrix.loc[label_a, label_b] = row["spearman_subject_macro_f1"]
        matrix.loc[label_b, label_a] = row["spearman_subject_macro_f1"]

    fig, ax = plt.subplots(
        figsize=(max(11.2, matrix.shape[1] * 0.76), max(9.1, matrix.shape[0] * 0.66)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=RANK_CORR_CMAP, vmin=-1.0, vmax=1.0)
    ax.set_title("Participant-difficulty agreement", fontsize=28)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([label_meta[label]["input_signal"] for label in matrix.columns.tolist()], rotation=90, fontsize=15)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([label_meta[label]["input_signal"] for label in matrix.index.tolist()], fontsize=15)

    group_ranges = []
    start_idx = 0
    while start_idx < len(labels):
        model_name = label_meta[labels[start_idx]]["model"]
        end_idx = start_idx
        while end_idx + 1 < len(labels) and label_meta[labels[end_idx + 1]]["model"] == model_name:
            end_idx += 1
        group_ranges.append((model_name, start_idx, end_idx))
        start_idx = end_idx + 1

    for group_idx, (_, start_idx, end_idx) in enumerate(group_ranges):
        if group_idx % 2 == 0:
            ax.axvspan(start_idx - 0.5, end_idx + 0.5, color="#FFFFFF", alpha=0.12, ec="none", zorder=2)
            ax.axhspan(start_idx - 0.5, end_idx + 0.5, color="#FFFFFF", alpha=0.12, ec="none", zorder=2)

    for _, _, end_idx in group_ranges[:-1]:
        ax.axvline(end_idx + 0.5, color="#FFFFFF", linewidth=3.0, alpha=1.0, zorder=3)
        ax.axhline(end_idx + 0.5, color="#FFFFFF", linewidth=3.0, alpha=1.0, zorder=3)
        ax.axvline(end_idx + 0.5, color="#AAB6C5", linewidth=1.0, alpha=0.85, zorder=4)
        ax.axhline(end_idx + 0.5, color="#AAB6C5", linewidth=1.0, alpha=0.85, zorder=4)

    x_text_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for model_name, start_idx, end_idx in group_ranges:
        center = (start_idx + end_idx) / 2
        x0 = start_idx - 0.45
        x1 = end_idx + 0.45
        y_top = -0.15
        y_hook = -0.11
        ax.plot([x0, x1], [y_top, y_top], transform=x_text_transform, color=PRIMARY_COLOR, linewidth=1.4, clip_on=False)
        ax.plot([x0, x0], [y_top, y_hook], transform=x_text_transform, color=PRIMARY_COLOR, linewidth=1.4, clip_on=False)
        ax.plot([x1, x1], [y_top, y_hook], transform=x_text_transform, color=PRIMARY_COLOR, linewidth=1.4, clip_on=False)
        ax.text(
            center,
            -0.16,
            model_name,
            transform=x_text_transform,
            rotation=0,
            ha="center",
            va="top",
            fontsize=17,
            fontweight="semibold",
            color=PRIMARY_COLOR,
            clip_on=False,
        )
        y0 = start_idx - 0.45
        y1 = end_idx + 0.45
        x_left = -0.15
        x_hook = -0.11
        ax.plot([x_left, x_left], [y0, y1], transform=y_text_transform, color=PRIMARY_COLOR, linewidth=1.4, clip_on=False)
        ax.plot([x_hook, x_left], [y0, y0], transform=y_text_transform, color=PRIMARY_COLOR, linewidth=1.4, clip_on=False)
        ax.plot([x_hook, x_left], [y1, y1], transform=y_text_transform, color=PRIMARY_COLOR, linewidth=1.4, clip_on=False)
        ax.text(
            -0.17,
            center,
            model_name,
            transform=y_text_transform,
            rotation=90,
            ha="right",
            va="center",
            fontsize=17,
            fontweight="semibold",
            color=PRIMARY_COLOR,
            clip_on=False,
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Spearman correlation", fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    dataset_slug = get_dataset_slug(dataset)
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_rank_correlation_heatmap.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_rank_correlation_heatmap.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_outputs(output_dir, metrics_df, experiment_summary_df, subject_summary_df, hardest_df, consistency_df, rank_corr_df, args, configs):
    """Save all tables and metadata."""
    dataset_slug = get_dataset_slug(args.dataset)
    metrics_df.to_csv(os.path.join(output_dir, f"{dataset_slug}_subject_error_metrics_all_experiments.csv"), index=False)
    experiment_summary_df.to_csv(os.path.join(output_dir, f"{dataset_slug}_subject_error_summary_by_experiment.csv"), index=False)
    subject_summary_df.to_csv(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_overall.csv"), index=False)
    hardest_df.to_csv(os.path.join(output_dir, f"{dataset_slug}_hardest_subjects_by_experiment.csv"), index=False)
    consistency_df.to_csv(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_consistency.csv"), index=False)
    rank_corr_df.to_csv(os.path.join(output_dir, f"{dataset_slug}_subject_difficulty_rank_correlation.csv"), index=False)

    metadata = {
        "dataset": args.dataset,
        "timestamp": args.timestamp,
        "bottom_k": args.bottom_k,
        "difficulty_threshold": args.difficulty_threshold,
        "num_experiments": len(configs),
        "experiments": [
            {
                "experiment_id": str(config["experiment_id"]),
                "model": config["DL"]["model"],
                "input_cols": config["DL"]["input_cols"],
                "config_path": config["_config_path"],
            }
            for config in configs
        ],
    }
    with open(os.path.join(output_dir, "subject_error_analysis_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def print_summary(output_dir, metrics_df, subject_summary_df, consistency_df):
    """Print a concise console summary."""
    print("\nSubject-level error analysis summary")
    print("-----------------------------------")
    print(f"Output directory: {output_dir}")
    print(f"Experiments analyzed: {metrics_df[['experiment_id', 'model', 'input_cols']].drop_duplicates().shape[0]}")
    print(f"Subjects analyzed: {metrics_df['test_id'].nunique()}")
    print("Hardest subjects overall (lowest mean macro-F1):")
    for _, row in subject_summary_df.head(5).iterrows():
        print(
            f"  subject {int(row['test_id'])}: "
            f"mean={row['mean_macro_f1']:.4f}, "
            f"min={row['min_macro_f1']:.4f}, "
            f"below_threshold={int(row['num_below_threshold'])}"
        )
    if not consistency_df.empty:
        print("Subjects most often among hardest cases:")
        for _, row in consistency_df.head(5).iterrows():
            print(f"  subject {int(row['test_id'])}: {int(row['times_in_bottom_k'])} times")


def run_subject_error_analysis_for_dataset(args, dataset):
    """Run the subject-level error analysis for one dataset."""
    configs = discover_matching_configs(args.config_dir, dataset, args.timestamp)
    output_dir = ensure_output_dir(configs, args.dataset)
    configure_plot_style()

    metrics_df = aggregate_all_metrics(configs)
    experiment_summary_df = build_experiment_summary(metrics_df)
    subject_summary_df = build_subject_overall_summary(metrics_df, args.difficulty_threshold)
    hardest_df = build_hardest_subjects_table(metrics_df, args.bottom_k)
    consistency_df = build_subject_consistency_table(hardest_df)
    rank_corr_df = build_rank_correlation_table(metrics_df)

    save_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        experiment_summary_df=experiment_summary_df,
        subject_summary_df=subject_summary_df,
        hardest_df=hardest_df,
        consistency_df=consistency_df,
        rank_corr_df=rank_corr_df,
        args=args,
        configs=configs,
    )

    if args.save_plots:
        save_consistency_barplot(
            consistency_df,
            output_dir,
            args.dataset,
            top_n=args.bottom_k,
            total_experiments=len(configs),
        )
        save_subject_difficulty_profile(subject_summary_df, output_dir, args.dataset)
        save_rank_correlation_heatmap(rank_corr_df, output_dir, args.dataset)

    print_summary(output_dir, metrics_df, subject_summary_df, consistency_df)
    return {
        "configs": configs,
        "output_dir": output_dir,
        "metrics_df": metrics_df,
        "subject_summary_df": subject_summary_df,
        "consistency_df": consistency_df,
    }


def get_combined_output_dir(timestamp):
    """Return a shared output directory for cross-dataset supplementary plots."""
    output_dir = os.path.join(LOCAL_RESULTS_ROOT, "error_analysis", str(timestamp))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    """Run the subject-level error analysis."""
    args = parse_args()
    configure_plot_style()

    if args.dataset == "both":
        results_by_dataset = {}
        for dataset in ["vr", "fordigit"]:
            dataset_args = argparse.Namespace(**vars(args))
            dataset_args.dataset = dataset
            results_by_dataset[dataset] = run_subject_error_analysis_for_dataset(dataset_args, dataset)

        if args.save_plots:
            combined_output_dir = get_combined_output_dir(args.timestamp)
            save_combined_consistency_barplot(
                consistency_by_dataset={
                    dataset: results["consistency_df"]
                    for dataset, results in results_by_dataset.items()
                },
                output_dir=combined_output_dir,
                bottom_k=args.bottom_k,
                total_experiments_by_dataset={
                    dataset: len(results["configs"])
                    for dataset, results in results_by_dataset.items()
                },
            )
            save_combined_reduced_rank_correlation_heatmaps(
                vr_metrics_df=results_by_dataset["vr"]["metrics_df"],
                fordigit_metrics_df=results_by_dataset["fordigit"]["metrics_df"],
                output_dir=combined_output_dir,
            )
            print(f"Saved combined consistency plot to {combined_output_dir}")
        return

    run_subject_error_analysis_for_dataset(args, args.dataset)


if __name__ == "__main__":
    main()
