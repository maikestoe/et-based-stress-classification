import argparse
import json
import os
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(SCRIPT_DIR, "plot_style2.txt")
CONFUSION_TICK_LABEL_FONTSIZE = 20
CONFUSION_CELL_LABEL_FONTSIZE = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot saved confusion matrices without rerunning recovery."
    )
    parser.add_argument("--config", required=True, help="Path to recovery config JSON.")
    parser.add_argument(
        "--tex-path",
        default=None,
        help="Optional path to LaTeX binaries."
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering for matplotlib."
    )
    return parser.parse_args()


def load_json_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)

    if "DL" in config and "path_results" in config["DL"]:
        config["DL"]["path_results"] = os.path.expandvars(config["DL"]["path_results"])
    if "df_prep_path" in config:
        config["df_prep_path"] = os.path.expandvars(config["df_prep_path"])
    return config


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


def style_publication_axis(ax, categorical_x=False, categorical_y=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=20)

    if not categorical_x:
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.get_offset_text().set_visible(False)

    if not categorical_y:
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.get_offset_text().set_visible(False)


def interpolate_hex_color(start_hex, end_hex, weight):
    start_rgb = np.array([int(start_hex[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    end_rgb = np.array([int(end_hex[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    mixed_rgb = (1 - weight) * start_rgb + weight * end_rgb
    return tuple((mixed_rgb / 255.0).tolist())


def interpolate_text_color(rgb_color):
    r, g, b = rgb_color[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#10243E" if luminance > 0.56 else "white"


def format_percent_value(value):
    if plt.rcParams.get("text.usetex", False):
        return f"{value:.1f}\\%"
    return f"{value:.1f}%"


def save_confusion_matrix_heatmap(output_base_path, cm, title, include_row_percent=False):
    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=False)
    max_value = np.nanmax(cm) if np.nanmax(cm) > 0 else 1.0
    correct_light = "#F4F6F9"
    correct_dark = "#8DBB6C"
    incorrect_light = "#FBF1F1"
    incorrect_dark = "#C46A6A"

    cell_colors = np.zeros((cm.shape[0], cm.shape[1], 3), dtype=float)
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            weight = float(cm[row_idx, col_idx] / max_value)
            if row_idx == col_idx:
                cell_colors[row_idx, col_idx] = interpolate_hex_color(correct_light, correct_dark, weight)
            else:
                cell_colors[row_idx, col_idx] = interpolate_hex_color(incorrect_light, incorrect_dark, weight)

    ax.imshow(cell_colors, aspect="equal")
    class_labels = ["No stress", "Stress"]
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label", fontsize=22)
    ax.set_ylabel("True label", fontsize=22, labelpad=6)
    style_publication_axis(ax, categorical_x=True, categorical_y=True)
    ax.tick_params(axis="x", labelsize=CONFUSION_TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=CONFUSION_TICK_LABEL_FONTSIZE)
    fig.subplots_adjust(top=0.78)
    fig.suptitle(title, x=0.02, y=0.92, ha="left", va="top", fontsize=18)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            value = int(cm[row_idx, col_idx])
            if include_row_percent:
                row_percent = 100.0 * cm[row_idx, col_idx] / row_sums[row_idx, 0]
                annotation = f"{value}\n({format_percent_value(row_percent)})"
            else:
                annotation = f"{value}"
            ax.text(
                col_idx,
                row_idx,
                annotation,
                ha="center",
                va="center",
                fontsize=CONFUSION_CELL_LABEL_FONTSIZE,
                color=interpolate_text_color(cell_colors[row_idx, col_idx])
            )

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def get_base_folder(config):
    return os.path.join(
        config["DL"]["path_results"],
        config["timestamp"],
        config["DL"]["model"],
        config["experiment_id"],
    )


def is_fordigitstress_config(config):
    df_path = str(config.get("df_prep_path", "")).lower()
    results_path = str(config.get("DL", {}).get("path_results", "")).lower()
    return "fordigitstress" in df_path or "fordigitstress" in results_path


def get_dataset_label(config):
    return "fordigitstress" if is_fordigitstress_config(config) else "vr_goalkeeper"


def get_model_display_name(config):
    model_name = str(config.get("DL", {}).get("model", "")).lower()
    if model_name in {"cnn", "cnn_pooling"}:
        return "CNN"
    if model_name == "lstm-1":
        return "LSTM-1"
    if model_name == "lstm-3":
        return "LSTM-3"
    if model_name == "convlstm-1":
        return "ConvLSTM-1"
    if model_name == "convlstm-3":
        return "ConvLSTM-3"
    return str(config.get("DL", {}).get("model", "Model"))


def get_input_signal_display_name(config):
    input_cols = config.get("DL", {}).get("input_cols", [])
    if not input_cols:
        return "Signal"
    signal_name = input_cols[0]
    signal_labels = {
        "meanDia_corrected": "PD",
        "velocity": "angular velocity",
        "acceleration": "angular acceleration",
        "position": "visual angle",
        "asymptotic_model": "asymptotic model",
        "fixations": "fixations",
    }
    return signal_labels.get(signal_name, signal_name.replace("_", " "))


def get_dataset_display_name(config):
    return "ForDigitStress dataset" if is_fordigitstress_config(config) else "VR goalkeeper dataset"


def build_confusion_matrix_title(config):
    return f"{get_model_display_name(config)} ({get_input_signal_display_name(config)})\n{get_dataset_display_name(config)}"


def replot_subject_confusion_matrices(config):
    base_folder = get_base_folder(config)
    dataset_label = get_dataset_label(config)
    model_label = config["DL"]["model"].lower()
    experiment_label = f"exp{config['experiment_id']}"
    include_row_percent = True
    replotted = []

    for test_id in config.get("valid_IDs", []):
        recovery_dir = os.path.join(base_folder, f"test_ID_{test_id}", "recovery")
        csv_path = os.path.join(recovery_dir, "confusion_matrix_outer_recovered.csv")
        if not os.path.exists(csv_path):
            continue
        cm = np.loadtxt(csv_path, delimiter=",")
        output_base_path = os.path.join(
            recovery_dir,
            f"{dataset_label}_{experiment_label}_{model_label}_confusion_matrix_outer_recovered"
        )
        save_confusion_matrix_heatmap(
            output_base_path=output_base_path,
            cm=cm,
            title=build_confusion_matrix_title(config),
            include_row_percent=include_row_percent
        )
        replotted.append(output_base_path)

    return replotted


def replot_aggregated_confusion_matrix(config):
    recovery_base = os.path.join(get_base_folder(config), "recovery")
    csv_path = os.path.join(recovery_base, "confusion_matrix_outer_aggregated.csv")
    if not os.path.exists(csv_path):
        return None

    dataset_label = get_dataset_label(config)
    model_label = config["DL"]["model"].lower()
    experiment_label = f"exp{config['experiment_id']}"
    output_base_path = os.path.join(
        recovery_base,
        f"{dataset_label}_{experiment_label}_{model_label}_confusion_matrix_outer_aggregated"
    )
    save_confusion_matrix_heatmap(
        output_base_path=output_base_path,
        cm=np.loadtxt(csv_path, delimiter=","),
        title=build_confusion_matrix_title(config),
        include_row_percent=True
    )
    return output_base_path


def main():
    args = parse_args()
    configure_publication_plot_style(latex=not args.no_latex, tex_path=args.tex_path)
    config = load_json_config(args.config)

    replotted_subjects = replot_subject_confusion_matrices(config)
    aggregated_path = replot_aggregated_confusion_matrix(config)

    print(f"Replotted {len(replotted_subjects)} subject-level confusion matrices.")
    if aggregated_path:
        print(f"Replotted aggregated confusion matrix: {aggregated_path}")
    else:
        print("No aggregated confusion matrix CSV found.")


if __name__ == "__main__":
    main()
