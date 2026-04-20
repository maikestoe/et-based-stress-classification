import argparse
import csv
import os
import re
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STYLE_PATH = os.path.join(SCRIPT_DIR, "plot_style2.txt")
RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "vr_goalkeeper",
    "rf_baseline_vr_goalkeeper_RS15",
)
DEFAULT_INPUT_FILES = {
    "combined": os.path.join(RESULTS_ROOT, "combined", "RF", "df_cv_results_RF_20241216-130646.csv")
}
DEFAULT_TITLE = "Feature selection in the combined RF baseline"
DEFAULT_OUTPUT_NAME = "rf_combined_feature_selection_frequency"

FEATURE_ORDER = [
    "mean",
    "median",
    "variance",
    "std",
    "skew",
    "max_val",
    "min_val",
    "kurt",
    "range",
    "1st_quantile",
    "3rd_quantile",
    "harmonic_mean",
    "samples_till_max",
    "slope1",
    "slope2",
    "meanfixationDuration_asymptotic_model",
    "fixation_durations_asymptotic_model",
    "counter_asymptotic_model",
]

FEATURE_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "variance": "Variance",
    "std": "Standard deviation",
    "skew": "Skewness",
    "max_val": "Maximum value",
    "min_val": "Minimum value",
    "kurt": "Kurtosis",
    "range": "Range",
    "1st_quantile": "1st quantile",
    "3rd_quantile": "3rd quantile",
    "harmonic_mean": "Harmonic mean",
    "samples_till_max": "Samples until maximum",
    "slope1": "Slope (first half)",
    "slope2": "Slope (second half)",
    "meanfixationDuration_asymptotic_model": "Average fixation duration",
    "fixation_durations_asymptotic_model": "Fixation durations",
    "counter_asymptotic_model": "# fixations",
}

FEATURE_GROUPS = {
    "meanfixationDuration_asymptotic_model": "Fixation",
    "fixation_durations_asymptotic_model": "Fixation",
    "counter_asymptotic_model": "Fixation",
}

PD_COLOR = "#90AEC6"
FIXATION_COLOR = "#E5C04A"
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 17
PERCENT_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 23


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot RF feature-selection frequencies for the combined feature baseline."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to df_cv_results_RF_*.csv. Defaults to the saved combined RF results.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory where the figure and frequency CSV should be saved.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base filename for PNG/PDF/CSV outputs.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering for matplotlib.",
    )
    return parser.parse_args()


def configure_plot_style(use_latex=True):
    plt.style.use(STYLE_PATH)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.dpi"] = 1000
    if not use_latex:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "cmr10", "DejaVu Serif"]


def parse_feature_names(raw_value):
    return re.findall(r"'([^']+)'", raw_value)


def load_feature_selection_counts(input_path):
    counter = Counter()
    fold_count = 0
    selected_counts = []
    with open(input_path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features = parse_feature_names(row["feature_names"])
            counter.update(features)
            fold_count += 1
            selected_counts.append(len(features))

    rows = []
    selected_feature_names = set(counter.keys())
    ordered_features = [feature for feature in FEATURE_ORDER if feature in selected_feature_names]
    for feature_name in ordered_features:
        count = counter.get(feature_name, 0)
        rows.append(
            {
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ")),
                "group": FEATURE_GROUPS.get(feature_name, "PD statistics"),
                "selected_folds": count,
                "selection_frequency_percent": 100.0 * count / fold_count if fold_count else 0.0,
            }
        )
    return rows, fold_count, selected_counts


def save_frequency_csv(rows, output_path):
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feature",
                "label",
                "group",
                "selected_folds",
                "selection_frequency_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_feature_selection(rows, fold_count, output_base_path, title):
    sorted_rows = sorted(
        rows,
        key=lambda row: (row["selected_folds"], row["group"] == "Fixation", row["label"]),
        reverse=True,
    )
    labels = [row["label"] for row in sorted_rows]
    counts = [row["selected_folds"] for row in sorted_rows]
    colors = [FIXATION_COLOR if row["group"] == "Fixation" else PD_COLOR for row in sorted_rows]
    y_positions = list(range(len(sorted_rows)))

    fig_height = max(4.4, 0.38 * len(sorted_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_height), constrained_layout=True)
    ax.barh(y_positions, counts, color=colors, edgecolor="none", linewidth=0.0, zorder=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, fold_count + 1.2)
    ax.set_xlabel("Number of outer folds selected", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Feature", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(axis="x", alpha=0.18, zorder=1)
    style_axis(ax)

    for y_pos, count in zip(y_positions, counts):
        percent = 100.0 * count / fold_count if fold_count else 0.0
        ax.text(
            count + 0.25,
            y_pos,
            f"{percent:.0f}%",
            va="center",
            ha="left",
            fontsize=PERCENT_LABEL_FONTSIZE,
            color="#465368",
            clip_on=False,
        )

    present_groups = {row["group"] for row in sorted_rows}
    legend_handles = []
    if "PD statistics" in present_groups:
        legend_handles.append(Patch(facecolor=PD_COLOR, edgecolor="none", label="PD statistics"))
    if "Fixation" in present_groups:
        legend_handles.append(Patch(facecolor=FIXATION_COLOR, edgecolor="none", label="Fixation characteristics"))
    if len(legend_handles) > 1:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            loc="lower right",
        )

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    args = parse_args()
    configure_plot_style(use_latex=not args.no_latex)

    input_path = args.input or DEFAULT_INPUT_FILES["combined"]
    output_dir = args.output_dir or os.path.dirname(input_path)
    output_name = args.output_name or DEFAULT_OUTPUT_NAME
    os.makedirs(output_dir, exist_ok=True)

    rows, fold_count, selected_counts = load_feature_selection_counts(input_path)
    output_base_path = os.path.join(output_dir, output_name)
    save_frequency_csv(rows, f"{output_base_path}.csv")
    plot_feature_selection(rows, fold_count, output_base_path, DEFAULT_TITLE)

    mean_selected = sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
    print(f"Saved feature-selection plot to {output_base_path}.png/.pdf")
    print(f"Outer folds: {fold_count}; mean selected features: {mean_selected:.2f}")


if __name__ == "__main__":
    main()
