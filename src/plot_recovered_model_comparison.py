import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(SCRIPT_DIR, "plot_style2.txt")


VR_RESULTS = {
    "PD": {
        "CNN": (88.85, 8.70),
        "LSTM-1": (62.53, 11.84),
        "ConvLSTM-1": (84.51, 10.53),
        "LSTM-3": (65.39, 12.34),
        "ConvLSTM-3": (81.79, 12.36),
    },
    "Velocity": {
        "CNN": (94.99, 8.28),
        "LSTM-1": (59.61, 14.25),
        "ConvLSTM-1": (95.33, 4.84),
        "LSTM-3": (74.97, 19.09),
        "ConvLSTM-3": (95.96, 6.07),
    },
    "Acceleration": {
        "CNN": (94.36, 6.33),
        "LSTM-1": (52.28, 15.23),
        "ConvLSTM-1": (94.55, 6.77),
        "LSTM-3": (60.56, 18.14),
        "ConvLSTM-3": (94.12, 7.35),
    },
    "Visual angle": {
        "CNN": (95.81, 5.84),
        "LSTM-1": (54.72, 14.45),
        "ConvLSTM-1": (93.67, 6.80),
        "LSTM-3": (72.67, 23.16),
        "ConvLSTM-3": (95.33, 6.91),
    },
    "Asymptotic model": {
        "CNN": (95.26, 7.67),
        "LSTM-1": (61.99, 15.76),
        "ConvLSTM-1": (94.91, 7.28),
        "LSTM-3": (82.73, 13.33),
        "ConvLSTM-3": (95.98, 5.72),
    },
    "Fixations": {
        "CNN": (91.67, 6.04),
        "LSTM-1": (68.01, 13.45),
        "ConvLSTM-1": (87.84, 8.33),
        "LSTM-3": (69.02, 12.83),
        "ConvLSTM-3": (89.92, 8.83),
    },
}

VR_FEATURE_BASELINES = {
    "PD statistics": 76.47,
    "Fixation characteristics": 78.46,
    "Combined": 83.64,
}


FORDIGIT_RESULTS = {
    "F1_macro": {
        "CNN": (57.77, 6.72),
        "LSTM-1": (53.13, 5.06),
        "ConvLSTM-1": (53.54, 9.29),
        "LSTM-3": (50.60, 13.66),
        "ConvLSTM-3": (53.06, 7.44),
    },
    "F1_weighted": {
        "CNN": (79.57, 7.14),
        "LSTM-1": (80.89, 5.47),
        "ConvLSTM-1": (75.79, 14.16),
        "LSTM-3": (73.28, 18.29),
        "ConvLSTM-3": (80.44, 6.93),
    },
}


MODEL_ORDER = ["CNN", "LSTM-1", "ConvLSTM-1", "LSTM-3", "ConvLSTM-3"]
MODEL_COLORS = {
    "CNN": "#5D7FA6",
    "LSTM-1": "#D9DEE7",
    "ConvLSTM-1": "#91A2BB",
    "LSTM-3": "#EEF1F5",
    "ConvLSTM-3": "#BFC8D6",
}

COMBINED_TICK_LABEL_SIZE = 21
COMBINED_XTICK_LABEL_SIZE = 17

SIGNAL_LABELS = {
    "PD": "PD",
    "Velocity": "Angular\nvelocity",
    "Acceleration": "Angular\nacceleration",
    "Visual angle": "Visual\nangle",
    "Asymptotic model": "Asymptotic\nmodel",
    "Fixations": "Fixations",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create model comparison plots from hardcoded paper F1 values."
    )
    parser.add_argument(
        "--dataset",
        choices=["vr", "fordigit", "both"],
        required=True,
        help="Which paper table to visualize."
    )
    parser.add_argument(
        "--metric",
        choices=["F1_macro", "F1_weighted"],
        default="F1_macro",
        help="Metric to plot for the ForDigitStress dataset."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "..", "results", "paper_model_comparison"),
        help="Directory where the plot and CSV export should be written."
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering for matplotlib."
    )
    parser.add_argument(
        "--tex-path",
        default=None,
        help="Optional path to LaTeX binaries."
    )
    return parser.parse_args()


def configure_publication_plot_style(latex=True, tex_path=None):
    plt.style.use(STYLE_PATH)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.dpi"] = 1000

    if not latex:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "cmr10", "DejaVu Serif"]
        return

    if tex_path:
        os.environ["PATH"] = f"{tex_path}:{os.environ['PATH']}"

    pdflatex_path = subprocess.run(
        ["which", "pdflatex"], capture_output=True, text=True
    ).stdout.strip()

    if pdflatex_path:
        plt.rcParams["text.usetex"] = True
        print(f"Using LaTeX from: {pdflatex_path}")
    else:
        print("pdflatex not found. Falling back to serif font without LaTeX.")
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "cmr10", "DejaVu Serif"]


def style_publication_axis(ax, categorical_x=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=20)

    if not categorical_x:
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.get_offset_text().set_visible(False)

    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.get_offset_text().set_visible(False)


def get_percent_axis_label(base_label):
    if plt.rcParams.get("text.usetex", False):
        return rf"{base_label} (\%)"
    return f"{base_label} (%)"


def get_metric_display_label(metric_name):
    metric_labels = {
        "F1_macro": "Macro F1-score",
        "F1_weighted": "Weighted F1-score",
    }
    return metric_labels.get(metric_name, metric_name.replace("_", " "))


def get_model_legend_handles():
    return [
        Patch(facecolor=MODEL_COLORS[model_name], edgecolor="none", label=model_name)
        for model_name in MODEL_ORDER
    ]


def left_align_legend(legend):
    """Keep legend titles and entries visually left-aligned across Matplotlib versions."""
    if legend is None:
        return
    try:
        legend.set_alignment("left")
    except AttributeError:
        pass
    legend._legend_box.align = "left"
    title = legend.get_title()
    if title is not None:
        title.set_ha("left")
    for text in legend.get_texts():
        text.set_ha("left")


def build_vr_dataframe():
    rows = []
    for signal_name, signal_results in VR_RESULTS.items():
        for model_name in MODEL_ORDER:
            mean_value, std_value = signal_results[model_name]
            rows.append(
                {
                    "input_signal": signal_name,
                    "model": model_name,
                    "mean_f1": mean_value,
                    "std_f1": std_value,
                }
            )
    return pd.DataFrame(rows)


def build_fordigit_dataframe(metric_name):
    rows = []
    for model_name in MODEL_ORDER:
        mean_value, std_value = FORDIGIT_RESULTS[metric_name][model_name]
        rows.append(
            {
                "metric": metric_name,
                "model": model_name,
                "mean_f1": mean_value,
                "std_f1": std_value,
            }
        )
    return pd.DataFrame(rows)


def draw_vr_comparison_axis(ax, df, show_ylabel=True, title="VR goalkeeper"):
    signal_order = list(VR_RESULTS.keys())
    signal_tick_labels = [SIGNAL_LABELS.get(signal_name, signal_name) for signal_name in signal_order]
    group_spacing = 1.60
    x = np.arange(len(signal_order)) * group_spacing
    width = 0.11
    bar_step = width * 1.08

    for idx, model_name in enumerate(MODEL_ORDER):
        model_df = df[df["model"] == model_name].set_index("input_signal").loc[signal_order].reset_index()
        ax.bar(
            x + (idx - (len(MODEL_ORDER) - 1) / 2.0) * bar_step,
            model_df["mean_f1"],
            width=width,
            yerr=model_df["std_f1"],
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="none",
            linewidth=0.0,
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
            zorder=3
        )

    ax.set_xticks(x)
    ax.set_xticklabels(signal_tick_labels, rotation=0, ha="center")
    if show_ylabel:
        ax.set_ylabel(get_percent_axis_label("Macro F1-score"), fontsize=24, labelpad=8)
    ax.set_xlabel("Input signal", fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.set_ylim(0, 110)
    ax.set_xlim(x[0] - 0.72, x[-1] + 0.72)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(VR_FEATURE_BASELINES["PD statistics"], color="#AEB7C2", linestyle=":", linewidth=1.1, alpha=0.8, zorder=1)
    ax.axhline(VR_FEATURE_BASELINES["Fixation characteristics"], color="#BCC4CD", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
    ax.axhline(VR_FEATURE_BASELINES["Combined"], color="#95A2B1", linestyle="-.", linewidth=1.2, alpha=0.85, zorder=1)
    style_publication_axis(ax, categorical_x=True)
    ax.tick_params(axis="x", labelsize=15, pad=10)

    return ax


def get_vr_baseline_legend_handles():
    return [
        plt.Line2D([0], [0], color="#7E8A97", linestyle=":", linewidth=1.4, label="PD statistics"),
        plt.Line2D([0], [0], color="#A1AAB5", linestyle="--", linewidth=1.4, label="Fixation characteristics"),
        plt.Line2D([0], [0], color="#2C4668", linestyle="-.", linewidth=1.6, label="Combined"),
    ]


def add_vr_legends(ax):
    model_legend = ax.legend(
        handles=get_model_legend_handles(),
        ncol=5,
        frameon=False,
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16)
    )
    ax.add_artist(model_legend)

    ax.legend(
        handles=get_vr_baseline_legend_handles(),
        frameon=False,
        fontsize=13,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=3
    )


def plot_vr_comparison(df, output_base_path):
    fig, ax = plt.subplots(figsize=(15.8, 9.8), constrained_layout=True)
    draw_vr_comparison_axis(ax, df, show_ylabel=True, title="Macro F1-score across input signals")
    add_vr_legends(ax)

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_fordigit_comparison_axis(ax, df, metric_name, show_ylabel=True, title=None):
    metric_label = get_metric_display_label(metric_name)
    x_center = np.array([0.0])
    width = 0.10
    bar_step = 0.13
    model_df = df.set_index("model").loc[MODEL_ORDER]
    means = model_df["mean_f1"].to_numpy()
    stds = model_df["std_f1"].to_numpy()
    for idx, model_name in enumerate(MODEL_ORDER):
        ax.bar(
            x_center + (idx - 2) * bar_step,
            means[idx],
            width=width,
            yerr=stds[idx],
            color=MODEL_COLORS[model_name],
            edgecolor="none",
            linewidth=0.0,
            capsize=4,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xticks(x_center)
    ax.set_xticklabels(["PD"], rotation=0, ha="center")
    if show_ylabel:
        ax.set_ylabel(get_percent_axis_label(metric_label), fontsize=24, labelpad=10)
    ax.set_xlabel("Input signal", fontsize=24)
    ax.set_title(title or f"{metric_label} comparison", fontsize=24)
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.62, 0.62)
    ax.grid(axis="y", alpha=0.2)
    style_publication_axis(ax, categorical_x=True)
    ax.tick_params(axis="x", labelsize=16, pad=8)

    return ax


def plot_fordigit_comparison(df, metric_name, output_base_path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    draw_fordigit_comparison_axis(ax, df, metric_name, show_ylabel=True)
    ax.legend(
        handles=get_model_legend_handles(),
        ncol=3,
        frameon=False,
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18)
    )

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_combined_comparison(vr_df, fordigit_df, metric_name, output_base_path):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 7.4),
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"width_ratios": [2.45, 1.0], "wspace": 0.06},
    )
    fig.subplots_adjust(bottom=0.36, left=0.07, right=0.98, top=0.86, wspace=0.08)

    draw_vr_comparison_axis(
        axes[0],
        vr_df,
        show_ylabel=True,
        title="VR goalkeeper dataset",
    )
    draw_fordigit_comparison_axis(
        axes[1],
        fordigit_df,
        metric_name,
        show_ylabel=False,
        title="ForDigitStress dataset",
    )

    axes[1].tick_params(axis="y", labelleft=False)
    for ax in axes:
        ax.set_xlabel("")

    for ax in axes:
        axis_position = ax.get_position()
        axis_center = (axis_position.x0 + axis_position.x1) / 2.0
        fig.text(axis_center, 0.225, "Input signal", ha="center", va="center", fontsize=26)

    vr_axis_position = axes[0].get_position()
    legend_left = vr_axis_position.x0
    baseline_legend = fig.legend(
        handles=get_vr_baseline_legend_handles(),
        ncol=3,
        frameon=False,
        fontsize=13,
        title="VR goalkeeper dataset feature-based baselines",
        title_fontsize=13,
        loc="lower left",
        bbox_to_anchor=(legend_left, 0.105),
        bbox_transform=fig.transFigure,
    )
    left_align_legend(baseline_legend)

    model_legend = fig.legend(
        handles=get_model_legend_handles(),
        ncol=5,
        frameon=True,
        fontsize=16,
        title="Model architecture",
        title_fontsize=16,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        bbox_transform=fig.transFigure,
    )
    model_legend.get_frame().set_facecolor("white")
    model_legend.get_frame().set_edgecolor("#AEB7C2")
    model_legend.get_frame().set_linewidth(0.8)
    model_legend.get_frame().set_alpha(1.0)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=COMBINED_TICK_LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=COMBINED_XTICK_LABEL_SIZE)

    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    args = parse_args()
    configure_publication_plot_style(latex=not args.no_latex, tex_path=args.tex_path)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset == "vr":
        df = build_vr_dataframe()
        output_base_path = os.path.join(output_dir, "vr_macro_f1_model_comparison")
        df.to_csv(f"{output_base_path}.csv", index=False)
        plot_vr_comparison(df, output_base_path)
        print(f"Saved VR comparison plot to {output_dir}")
        return

    if args.dataset == "both":
        vr_df = build_vr_dataframe()
        fordigit_df = build_fordigit_dataframe(args.metric)
        output_base_path = os.path.join(output_dir, f"vr_fordigit_{args.metric.lower()}_model_comparison")
        pd.concat(
            [
                vr_df.assign(dataset="VR goalkeeper", metric="F1_macro"),
                fordigit_df.assign(dataset="ForDigitStress"),
            ],
            ignore_index=True,
            sort=False,
        ).to_csv(f"{output_base_path}.csv", index=False)
        plot_combined_comparison(vr_df, fordigit_df, args.metric, output_base_path)
        print(f"Saved combined model comparison plot to {output_dir}")
        return

    df = build_fordigit_dataframe(args.metric)
    output_base_path = os.path.join(output_dir, f"fordigit_{args.metric.lower()}_model_comparison")
    df.to_csv(f"{output_base_path}.csv", index=False)
    plot_fordigit_comparison(df, args.metric, output_base_path)
    print(f"Saved ForDigitStress comparison plot to {output_dir}")


if __name__ == "__main__":
    main()
