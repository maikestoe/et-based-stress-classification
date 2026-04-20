"""
Statistical significance analysis for paired outer-LOSO macro-F1 scores.

This script performs within-dataset significance testing for the VR goalkeeper
dataset only. The statistical unit is the outer LOSO test fold / held-out test
subject. Window-level predictions are intentionally not used here.

Tested comparisons
------------------
1. RF combined vs. best overall DL model
2. RF PD statistics vs. best PD-based CNN model

Why these comparisons?
----------------------
They correspond to the most relevant within-dataset baseline-vs-DL questions
for the VR goalkeeper dataset discussed in the manuscript:
- whether the best overall deep-learning model outperforms the strongest
  feature-based random-forest baseline using the combined feature subset
- whether the best PD-based CNN outperforms the RF baseline that also uses
  pupil-diameter-derived statistics

Methodological note
-------------------
Wilcoxon signed-rank is used as the primary test because the comparisons are
paired across the same outer LOSO test subjects and the sample size is modest.
Shapiro-Wilk is reported on the paired fold-wise differences as a diagnostic,
but the Wilcoxon test remains the inferential test reported for the paired
comparisons.
"""

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(SCRIPT_DIR, "plot_style2.txt")
PRIMARY_BAR_COLOR = "#90AEC6"
NEGATIVE_BAR_COLOR = "#B1263E"
DIAGNOSTIC_AXIS_LABEL_FONTSIZE = 18
DIAGNOSTIC_TICK_LABEL_FONTSIZE = 16
DIAGNOSTIC_X_TICK_LABEL_FONTSIZE = 12
DIAGNOSTIC_TITLE_FONTSIZE = 18

DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "results",
    "vr_goalkeeper",
    "statistical_significance",
)

DEFAULT_VR_RESULTS_ROOT = os.path.join(
    SCRIPT_DIR,
    "..",
    "results",
    "vr_goalkeeper",
)


@dataclass(frozen=True)
class ModelSource:
    """Description of where a model's fold-wise macro-F1 scores are loaded from."""

    source_type: str
    label: str
    subset_name: str | None = None
    timestamp: str | None = None
    model_name: str | None = None
    experiment_id: str | None = None


@dataclass(frozen=True)
class ComparisonSpec:
    """Paired comparison specification."""

    comparison_id: str
    dataset: str
    rationale: str
    model_a: ModelSource
    model_b: ModelSource


VR_COMPARISONS = [
    ComparisonSpec(
        comparison_id="vr_rf_combined_vs_best_overall_dl",
        dataset="VR goalkeeper",
        rationale=(
            "Compare the strongest feature-based RF baseline using the combined "
            "feature subset against the best overall DL model identified for the "
            "VR goalkeeper dataset."
        ),
        model_a=ModelSource(
            source_type="rf",
            label="RF (combined)",
            subset_name="combined",
        ),
        model_b=ModelSource(
            source_type="dl",
            label="ConvLSTM-3 (Asymptotic model)",
            timestamp="2024-08-09",
            model_name="ConvLSTM-3",
            experiment_id="41",
        ),
    ),
    ComparisonSpec(
        comparison_id="vr_rf_pd_vs_best_pd_cnn",
        dataset="VR goalkeeper",
        rationale=(
            "Compare the RF baseline based on pupil-diameter-derived statistics "
            "against the best PD-based CNN model for the same dataset."
        ),
        model_a=ModelSource(
            source_type="rf",
            label="RF (PD statistics)",
            subset_name="mean",
        ),
        model_b=ModelSource(
            source_type="dl",
            label="CNN (PD)",
            timestamp="2024-08-09",
            model_name="CNN",
            experiment_id="1",
        ),
    ),
]


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run paired statistical significance analyses on outer-LOSO fold-wise "
            "macro-F1 scores for the VR goalkeeper dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV tables and optional plots will be written.",
    )
    parser.add_argument(
        "--vr-results-root",
        default=DEFAULT_VR_RESULTS_ROOT,
        help=(
            "Root directory containing vr_goalkeeper result folders. Adapt this if "
            "your project results were moved."
        ),
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save simple diagnostic plots for the paired fold-wise differences.",
    )
    return parser.parse_args()


def configure_plot_style():
    """Apply the project plotting style when available."""
    if os.path.exists(STYLE_PATH):
        plt.style.use(STYLE_PATH)
    plt.rcParams["text.usetex"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def ensure_output_dir(output_dir):
    """Create the output directory if needed."""
    os.makedirs(output_dir, exist_ok=True)


def find_latest_file(pattern):
    """
    Return the most recently modified file matching a glob pattern.

    This prefers the newest export, which is useful because multiple RF runs are
    present in the project results. Adapt this selection logic if you want to pin
    a specific RF run instead of using the latest matching export.
    """
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(matches, key=os.path.getmtime)


def load_rf_fold_scores(vr_results_root, subset_name, model_label):
    """
    Load fold-wise RF macro-F1 scores for a given VR feature subset.

    The `mean` subset corresponds to PD statistics in the manuscript tables,
    while `combined` corresponds to the combined feature baseline.
    """
    pattern = os.path.join(
        vr_results_root,
        "rf_baseline_vr_goalkeeper*",
        subset_name,
        "RF",
        "df_cv_results_RF_*.csv",
    )
    csv_path = find_latest_file(pattern)
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns or "f1" not in df.columns:
        raise ValueError(
            f"Unexpected RF CSV schema in {csv_path}. Expected columns 'ID' and 'f1'."
        )

    scores = (
        df.loc[:, ["ID", "f1"]]
        .rename(columns={"ID": "test_id", "f1": "macro_f1"})
        .copy()
    )
    scores["test_id"] = scores["test_id"].astype(int)
    scores["macro_f1"] = scores["macro_f1"].astype(float)
    scores["model_label"] = model_label
    scores["source_path"] = csv_path
    return scores.sort_values("test_id").reset_index(drop=True)


def load_dl_fold_scores(vr_results_root, timestamp, model_name, experiment_id, model_label):
    """Load fold-wise DL macro-F1 scores from recovered outer metrics."""
    base_dir = os.path.join(
        vr_results_root,
        "DL",
        timestamp,
        model_name,
        str(experiment_id),
    )
    candidate_paths = [
        os.path.join(base_dir, "outer_metrics_recovered.csv"),
        os.path.join(base_dir, "recovery", "outer_metrics_recovered.csv"),
    ]
    csv_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find recovered outer metrics for {model_label} in {base_dir}."
        )

    df = pd.read_csv(csv_path)
    if "test_id" not in df.columns or "macro_f1" not in df.columns:
        raise ValueError(
            f"Unexpected DL CSV schema in {csv_path}. Expected columns 'test_id' and 'macro_f1'."
        )

    scores = df.loc[:, ["test_id", "macro_f1"]].copy()
    scores["test_id"] = scores["test_id"].astype(int)
    scores["macro_f1"] = scores["macro_f1"].astype(float)
    scores["model_label"] = model_label
    scores["source_path"] = csv_path
    return scores.sort_values("test_id").reset_index(drop=True)


def load_model_scores(vr_results_root, model_source):
    """Dispatch model-score loading based on source type."""
    if model_source.source_type == "rf":
        return load_rf_fold_scores(
            vr_results_root=vr_results_root,
            subset_name=model_source.subset_name,
            model_label=model_source.label,
        )
    if model_source.source_type == "dl":
        return load_dl_fold_scores(
            vr_results_root=vr_results_root,
            timestamp=model_source.timestamp,
            model_name=model_source.model_name,
            experiment_id=model_source.experiment_id,
            model_label=model_source.label,
        )
    raise ValueError(f"Unsupported source_type: {model_source.source_type}")


def compute_descriptive_statistics(scores_df, comparison_id, dataset_name):
    """Compute descriptive statistics for one model."""
    model_label = scores_df["model_label"].iloc[0]
    values = scores_df["macro_f1"].to_numpy(dtype=float)
    source_path = scores_df["source_path"].iloc[0]
    return {
        "comparison_id": comparison_id,
        "dataset": dataset_name,
        "model_label": model_label,
        "n_folds": int(values.size),
        "mean_macro_f1": float(np.mean(values)),
        "std_macro_f1": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median_macro_f1": float(np.median(values)),
        "min_macro_f1": float(np.min(values)),
        "max_macro_f1": float(np.max(values)),
        "source_path": source_path,
    }


def build_paired_dataframe(comparison_spec, scores_a, scores_b):
    """Align two model score tables by outer test ID and compute paired differences."""
    paired = scores_a.loc[:, ["test_id", "macro_f1"]].rename(
        columns={"macro_f1": "macro_f1_a"}
    ).merge(
        scores_b.loc[:, ["test_id", "macro_f1"]].rename(columns={"macro_f1": "macro_f1_b"}),
        on="test_id",
        how="inner",
        validate="one_to_one",
    )
    paired["difference_b_minus_a"] = paired["macro_f1_b"] - paired["macro_f1_a"]
    paired["comparison_id"] = comparison_spec.comparison_id
    paired["dataset"] = comparison_spec.dataset
    paired["model_a_label"] = comparison_spec.model_a.label
    paired["model_b_label"] = comparison_spec.model_b.label
    return paired.sort_values("test_id").reset_index(drop=True)


def get_first_existing_column(df, candidates):
    """Return the first matching column name from a list of candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def run_shapiro_wilk(differences):
    """
    Run Shapiro-Wilk on paired differences via Pingouin.

    Normality is checked on the paired fold-wise differences, because that is
    the relevant assumption for a paired parametric test.
    """
    differences = pd.Series(np.asarray(differences, dtype=float), name="difference")
    unique_values = np.unique(np.round(differences.to_numpy(), decimals=12))
    if differences.size < 3 or unique_values.size < 3:
        return np.nan, np.nan, np.nan

    result = pg.normality(differences, method="shapiro")
    return (
        float(result["W"].iloc[0]),
        float(result["pval"].iloc[0]),
        bool(result["normal"].iloc[0]),
    )


def run_wilcoxon_test(x_values, y_values):
    """
    Run the paired Wilcoxon signed-rank test via Pingouin.

    Pingouin also returns matched-pairs rank-biserial correlation (RBC), which
    we use as the effect size.
    """
    result = pg.wilcoxon(x=x_values, y=y_values, alternative="two-sided")
    w_col = get_first_existing_column(result, ["W-val", "W_val", "T", "stat", "statistic"])
    p_col = get_first_existing_column(result, ["p-val", "p_val", "p", "pvalue"])
    rbc_col = get_first_existing_column(result, ["RBC", "rbc"])
    cles_col = get_first_existing_column(result, ["CLES", "cles"])
    if w_col is None or p_col is None:
        raise ValueError(
            "Could not identify Wilcoxon statistic/p-value columns in Pingouin output. "
            f"Available columns: {list(result.columns)}"
        )
    return {
        "wilcoxon_w": float(result[w_col].iloc[0]),
        "wilcoxon_p": float(result[p_col].iloc[0]),
        "effect_size_rank_biserial": float(result[rbc_col].iloc[0]) if rbc_col is not None else np.nan,
        "common_language_effect_size": float(result[cles_col].iloc[0]) if cles_col is not None else np.nan,
    }


def compute_inferential_statistics(comparison_spec, paired_df):
    """Compute inferential statistics for one paired model comparison."""
    differences = paired_df["difference_b_minus_a"].to_numpy(dtype=float)
    shapiro_w, shapiro_p, shapiro_normal = run_shapiro_wilk(differences)
    wilcoxon_result = run_wilcoxon_test(
        x_values=paired_df["macro_f1_a"].to_numpy(dtype=float),
        y_values=paired_df["macro_f1_b"].to_numpy(dtype=float),
    )

    mean_diff = float(np.mean(differences))
    median_diff = float(np.median(differences))
    std_diff = float(np.std(differences, ddof=1)) if differences.size > 1 else np.nan

    if mean_diff > 0:
        direction = f"{comparison_spec.model_b.label} > {comparison_spec.model_a.label}"
    elif mean_diff < 0:
        direction = f"{comparison_spec.model_a.label} > {comparison_spec.model_b.label}"
    else:
        direction = "no mean difference"

    return {
        "comparison_id": comparison_spec.comparison_id,
        "dataset": comparison_spec.dataset,
        "rationale": comparison_spec.rationale,
        "model_a_label": comparison_spec.model_a.label,
        "model_b_label": comparison_spec.model_b.label,
        "n_pairs": int(differences.size),
        "mean_difference_b_minus_a": mean_diff,
        "median_difference_b_minus_a": median_diff,
        "std_difference_b_minus_a": std_diff,
        "min_difference_b_minus_a": float(np.min(differences)),
        "max_difference_b_minus_a": float(np.max(differences)),
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "shapiro_normal": shapiro_normal,
        "wilcoxon_w": wilcoxon_result["wilcoxon_w"],
        "wilcoxon_p": wilcoxon_result["wilcoxon_p"],
        "effect_size_rank_biserial": wilcoxon_result["effect_size_rank_biserial"],
        "common_language_effect_size": wilcoxon_result["common_language_effect_size"],
        "direction": direction,
    }


def holm_correction(p_values, alpha=0.05):
    """Apply Holm correction via Pingouin."""
    p_values = np.asarray(p_values, dtype=float)
    reject, adjusted = pg.multicomp(p_values, alpha=alpha, method="holm")
    return np.asarray(adjusted, dtype=float), np.asarray(reject, dtype=bool)


def format_difference_axis_label(model_b_label, model_a_label):
    """Format a compact axis label for paired model differences."""
    return "Macro-F1 difference"


def save_diagnostic_plot(paired_df, output_dir):
    """Save a simple diagnostic figure for paired fold-wise differences."""
    comparison_id = paired_df["comparison_id"].iloc[0]
    model_a_label = paired_df["model_a_label"].iloc[0]
    model_b_label = paired_df["model_b_label"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), constrained_layout=True)

    differences = paired_df["difference_b_minus_a"].to_numpy(dtype=float)
    test_id_labels = paired_df["test_id"].astype(str).to_list()
    x_positions = np.arange(len(test_id_labels))
    staggered_test_id_labels = [
        label if idx % 2 == 0 else f"\n{label}"
        for idx, label in enumerate(test_id_labels)
    ]
    bar_colors = [
        PRIMARY_BAR_COLOR if value >= 0 else NEGATIVE_BAR_COLOR
        for value in differences
    ]

    axes[0].axhline(0.0, color="#7A8796", linewidth=1.0, linestyle="--")
    axes[0].bar(
        x_positions,
        differences,
        color=bar_colors,
        edgecolor="none",
        width=0.75,
    )
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(staggered_test_id_labels)
    axes[0].set_xlabel("Outer test ID", fontsize=DIAGNOSTIC_AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel(
        format_difference_axis_label(model_b_label, model_a_label),
        fontsize=DIAGNOSTIC_AXIS_LABEL_FONTSIZE,
    )
    axes[0].set_title("Paired fold-wise differences", fontsize=DIAGNOSTIC_TITLE_FONTSIZE)
    axes[0].tick_params(axis="x", rotation=0, labelsize=DIAGNOSTIC_X_TICK_LABEL_FONTSIZE)
    axes[0].tick_params(axis="y", labelsize=DIAGNOSTIC_TICK_LABEL_FONTSIZE)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].hist(
        paired_df["difference_b_minus_a"],
        bins=min(10, max(4, paired_df.shape[0] // 3)),
        color=PRIMARY_BAR_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].axvline(0.0, color="#7A8796", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel(
        format_difference_axis_label(model_b_label, model_a_label),
        fontsize=DIAGNOSTIC_AXIS_LABEL_FONTSIZE,
    )
    axes[1].set_ylabel("Count", fontsize=DIAGNOSTIC_AXIS_LABEL_FONTSIZE)
    axes[1].set_title("Difference distribution", fontsize=DIAGNOSTIC_TITLE_FONTSIZE)
    axes[1].tick_params(axis="both", labelsize=DIAGNOSTIC_TICK_LABEL_FONTSIZE)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    png_path = os.path.join(output_dir, f"{comparison_id}_paired_differences.png")
    pdf_path = os.path.join(output_dir, f"{comparison_id}_paired_differences.pdf")
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_csv_tables(output_dir, descriptive_rows, paired_rows, inferential_rows):
    """Save the descriptive, paired, and inferential result tables."""
    descriptive_df = pd.DataFrame(descriptive_rows).sort_values(
        ["comparison_id", "model_label"]
    )
    paired_df = pd.DataFrame(paired_rows).sort_values(["comparison_id", "test_id"])
    inferential_df = pd.DataFrame(inferential_rows).sort_values("comparison_id")

    descriptive_df.to_csv(
        os.path.join(output_dir, "descriptive_statistics.csv"), index=False
    )
    paired_df.to_csv(
        os.path.join(output_dir, "paired_fold_scores.csv"), index=False
    )
    inferential_df.to_csv(
        os.path.join(output_dir, "inferential_statistics.csv"), index=False
    )


def print_console_summary(inferential_rows):
    """Print a short console summary."""
    print("\nStatistical significance analysis summary")
    print("----------------------------------------")
    for row in inferential_rows:
        print(
            f"{row['comparison_id']}: "
            f"W={row['wilcoxon_w']:.4f}, "
            f"p={row['wilcoxon_p']:.4g}, "
            f"Holm p={row['holm_adjusted_p']:.4g}, "
            f"RBC={row['effect_size_rank_biserial']:.4f}, "
            f"direction={row['direction']}"
        )


def main():
    """Run the VR-only paired significance analysis."""
    args = parse_args()
    ensure_output_dir(args.output_dir)
    configure_plot_style()

    descriptive_rows = []
    paired_rows = []
    inferential_rows = []

    for comparison_spec in VR_COMPARISONS:
        scores_a = load_model_scores(args.vr_results_root, comparison_spec.model_a)
        scores_b = load_model_scores(args.vr_results_root, comparison_spec.model_b)

        descriptive_rows.append(
            compute_descriptive_statistics(
                scores_df=scores_a,
                comparison_id=comparison_spec.comparison_id,
                dataset_name=comparison_spec.dataset,
            )
        )
        descriptive_rows.append(
            compute_descriptive_statistics(
                scores_df=scores_b,
                comparison_id=comparison_spec.comparison_id,
                dataset_name=comparison_spec.dataset,
            )
        )

        paired_df = build_paired_dataframe(comparison_spec, scores_a, scores_b)
        if paired_df.empty:
            raise ValueError(
                f"No overlapping test IDs found for comparison {comparison_spec.comparison_id}."
            )

        paired_rows.extend(paired_df.to_dict(orient="records"))
        inferential_rows.append(
            compute_inferential_statistics(comparison_spec, paired_df)
        )

        if args.save_plots:
            save_diagnostic_plot(paired_df, args.output_dir)

    adjusted_p, reject = holm_correction(
        [row["wilcoxon_p"] for row in inferential_rows],
        alpha=0.05,
    )
    for idx, row in enumerate(inferential_rows):
        row["holm_adjusted_p"] = float(adjusted_p[idx]) if not np.isnan(adjusted_p[idx]) else np.nan
        row["reject_holm_alpha_0_05"] = bool(reject[idx])

    save_csv_tables(
        output_dir=args.output_dir,
        descriptive_rows=descriptive_rows,
        paired_rows=paired_rows,
        inferential_rows=inferential_rows,
    )
    print_console_summary(inferential_rows)


if __name__ == "__main__":
    main()
