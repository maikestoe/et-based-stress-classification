import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs", "examples")
STYLE_PATH = os.path.join(SCRIPT_DIR, "plot_style2.txt")
COMPARISON_PALETTE = [
    "#123B6D",
    "#5D7FA6",
    "#91A2BB",
    "#8DBB6C",
    "#E5C04A",
    (196 / 255.0, 79 / 255.0, 94 / 255.0, 0.78),
]
MULTI_CURVE_LEGEND_FONTSIZE = 13
SINGLE_CURVE_LEGEND_FONTSIZE = 14
MODEL_NAME_MAP = {
    "CNN": "CNN",
    "LSTM-1": "LSTM-1",
    "ConvLSTM-1": "ConvLSTM-1",
    "LSTM-3": "LSTM-3",
    "ConvLSTM-3": "ConvLSTM-3",
}
SIGNAL_LABEL_MAP = {
    "meanDia_corrected": "PD",
    "velocity": "angular velocity",
    "acceleration": "angular acceleration",
    "position": "visual angle",
    "asymptotic_model": "asymptotic model",
    "fix_array": "fixations",
}
MODEL_ORDER = ["CNN", "LSTM-1", "ConvLSTM-1", "LSTM-3", "ConvLSTM-3"]
SIGNAL_ORDER = ["PD", "Angular velocity", "Angular acceleration", "Visual angle", "Asymptotic model", "Fixations"]
VR_REPRESENTATIVE_CONFIGS = [
    "vr_goalkeeper_cnn_pd.json",
    "vr_goalkeeper_convlstm3_asymptotic_recovery.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ROC and PR comparison plots from recovered outer-fold predictions."
    )
    parser.add_argument(
        "--config-list",
        default=None,
        help="Optional JSON file with an explicit list of recovery config files to plot."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional reference recovery config. If omitted, matching configs are inferred from --dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["vr", "fordigit"],
        default=None,
        help="Use all matching recovery configs for this dataset, similar to the paper comparison plot workflow."
    )
    parser.add_argument(
        "--config-dir",
        default=CONFIG_DIR,
        help="Directory containing recovery config files."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <path_results>/<timestamp>/curve_comparison."
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


def normalize_path(path_value):
    expanded = os.path.expandvars(path_value)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(PROJECT_ROOT, expanded))


def load_json_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)

    if "base_config" in config:
        base_config_path = config["base_config"]
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), base_config_path)
        with open(base_config_path, "r") as file:
            base_config = json.load(file)
        config = deep_update_config(base_config, config)

    config["__config_path"] = os.path.abspath(config_path)
    config["__normalized_df_prep_path"] = normalize_path(config["df_prep_path"])
    config["__normalized_results_path"] = normalize_path(config["DL"]["path_results"])
    return config


def deep_update_config(base_config, override_config):
    merged = dict(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_list(config_list_path):
    with open(config_list_path, "r") as file:
        config_list = json.load(file)

    base_dir = os.path.dirname(os.path.abspath(config_list_path))
    config_paths = []
    for config_entry in config_list.get("configs", []):
        if os.path.isabs(config_entry):
            config_paths.append(config_entry)
        else:
            config_paths.append(os.path.abspath(os.path.join(base_dir, config_entry)))

    return {
        "configs": config_paths,
        "output_dir": config_list.get("output_dir"),
        "dataset_title": config_list.get("dataset_title"),
        "roc_title": config_list.get("roc_title"),
        "pr_title": config_list.get("pr_title"),
        "roc_legend_loc": config_list.get("roc_legend_loc"),
        "pr_legend_loc": config_list.get("pr_legend_loc"),
    }


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


def style_publication_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=18)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)


def get_base_folder(config):
    return os.path.join(
        config["__normalized_results_path"],
        config["timestamp"],
        config["DL"]["model"],
        str(config["experiment_id"])
    )


def get_model_display_name(config):
    model_name = MODEL_NAME_MAP.get(config["DL"]["model"], config["DL"]["model"])
    input_cols = config["DL"].get("input_cols", [])
    mapped_inputs = [SIGNAL_LABEL_MAP.get(input_col, input_col) for input_col in input_cols]
    input_label = ", ".join(mapped_inputs) if mapped_inputs else "Default input"
    return f"{model_name} ({input_label})"


def get_model_sort_key(record):
    model_name = MODEL_NAME_MAP.get(record["model"], record["model"])
    signal_name = SIGNAL_LABEL_MAP.get(record["input_cols"], record["input_cols"])
    model_idx = MODEL_ORDER.index(model_name) if model_name in MODEL_ORDER else len(MODEL_ORDER)
    signal_idx = SIGNAL_ORDER.index(signal_name) if signal_name in SIGNAL_ORDER else len(SIGNAL_ORDER)
    return model_idx, signal_idx, record["label"]


def get_matching_recovery_configs(reference_config, config_dir):
    matching_configs = []
    for filename in sorted(os.listdir(config_dir)):
        if not filename.endswith("_recovery.json"):
            continue

        candidate_config = load_json_config(os.path.join(config_dir, filename))
        if candidate_config["__normalized_df_prep_path"] != reference_config["__normalized_df_prep_path"]:
            continue
        if candidate_config["__normalized_results_path"] != reference_config["__normalized_results_path"]:
            continue
        if candidate_config.get("timestamp") != reference_config.get("timestamp"):
            continue
        matching_configs.append(candidate_config)
    return matching_configs


def get_dataset_matching_configs(dataset_name, config_dir):
    if dataset_name == "vr":
        matching_configs = []
        missing_configs = []
        for filename in VR_REPRESENTATIVE_CONFIGS:
            config_path = os.path.join(config_dir, filename)
            if os.path.exists(config_path):
                matching_configs.append(load_json_config(config_path))
            else:
                missing_configs.append(config_path)
        if missing_configs:
            print("Missing representative VR recovery configs:")
            for config_path in missing_configs:
                print(f"  - {config_path}")
        return matching_configs

    matching_configs = []
    dataset_pattern = "study_pk_vr" if dataset_name == "vr" else "fordigitstress"

    for filename in sorted(os.listdir(config_dir)):
        if not filename.endswith("_recovery.json"):
            continue

        candidate_config = load_json_config(os.path.join(config_dir, filename))
        df_path = candidate_config["__normalized_df_prep_path"].lower()
        results_path = candidate_config["__normalized_results_path"].lower()

        if dataset_pattern in df_path or dataset_pattern in results_path:
            matching_configs.append(candidate_config)

    return matching_configs


def get_recovery_array_paths(config):
    recovery_dir = os.path.join(get_base_folder(config), "recovery")
    y_true_path = os.path.join(recovery_dir, "y_true_outer_aggregated.npy")
    y_score_path = os.path.join(recovery_dir, "y_score_outer_aggregated.npy")
    if os.path.exists(y_true_path) and os.path.exists(y_score_path):
        return y_true_path, y_score_path
    return None, None


def collect_curve_data(reference_config, config_dir):
    records = []
    used_configs = []
    skipped_configs = []
    for candidate_config in get_matching_recovery_configs(reference_config, config_dir):
        y_true_path, y_score_path = get_recovery_array_paths(candidate_config)
        if y_true_path is None:
            print(f"Skipping {candidate_config['__config_path']}: aggregated recovery arrays not found.")
            skipped_configs.append(candidate_config["__config_path"])
            continue

        y_true = np.load(y_true_path)
        y_score = np.load(y_score_path)

        roc_auc = roc_auc_score(y_true, y_score)
        pr_ap = average_precision_score(y_true, y_score)

        records.append({
            "label": get_model_display_name(candidate_config),
            "model": candidate_config["DL"]["model"],
            "experiment_id": str(candidate_config["experiment_id"]),
            "input_cols": ", ".join(candidate_config["DL"].get("input_cols", [])),
            "y_true": y_true,
            "y_score": y_score,
            "roc_auc": roc_auc,
            "average_precision": pr_ap,
        })
        used_configs.append(candidate_config["__config_path"])

    return sorted(records, key=get_model_sort_key), used_configs, skipped_configs


def collect_curve_data_from_configs(configs):
    records = []
    used_configs = []
    skipped_configs = []
    for candidate_config in configs:
        y_true_path, y_score_path = get_recovery_array_paths(candidate_config)
        if y_true_path is None:
            print(f"Skipping {candidate_config['__config_path']}: aggregated recovery arrays not found.")
            skipped_configs.append(candidate_config["__config_path"])
            continue

        y_true = np.load(y_true_path)
        y_score = np.load(y_score_path)

        roc_auc = roc_auc_score(y_true, y_score)
        pr_ap = average_precision_score(y_true, y_score)

        records.append({
            "label": get_model_display_name(candidate_config),
            "model": candidate_config["DL"]["model"],
            "experiment_id": str(candidate_config["experiment_id"]),
            "input_cols": ", ".join(candidate_config["DL"].get("input_cols", [])),
            "y_true": y_true,
            "y_score": y_score,
            "roc_auc": roc_auc,
            "average_precision": pr_ap,
        })
        used_configs.append(candidate_config["__config_path"])

    return sorted(records, key=get_model_sort_key), used_configs, skipped_configs


def save_curve_summary(output_dir, records):
    summary_df = pd.DataFrame([
        {
            "label": record["label"],
            "model": record["model"],
            "experiment_id": record["experiment_id"],
            "input_cols": record["input_cols"],
            "roc_auc": record["roc_auc"],
            "average_precision": record["average_precision"],
        }
        for record in records
    ]).sort_values("average_precision", ascending=False)
    summary_df.to_csv(os.path.join(output_dir, "curve_comparison_summary.csv"), index=False)


def draw_roc_comparison_axis(ax, records, title, legend_loc="lower right"):
    for idx, record in enumerate(records):
        fpr, tpr, _ = roc_curve(record["y_true"], record["y_score"])
        ax.plot(
            fpr,
            tpr,
            color=COMPARISON_PALETTE[idx % len(COMPARISON_PALETTE)],
            linewidth=1.5,
            label=f"{record['label']} (AUC = {record['roc_auc']:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#7E8A97",
        linewidth=1.0,
        label="Random classifier"
    )
    ax.set_xlabel("False positive rate", fontsize=24)
    ax.set_ylabel("True positive rate", fontsize=24)
    ax.set_title(title, fontsize=24)
    legend_fontsize = MULTI_CURVE_LEGEND_FONTSIZE if len(records) > 1 else SINGLE_CURVE_LEGEND_FONTSIZE
    ax.legend(frameon=False, fontsize=legend_fontsize, loc=legend_loc, handlelength=1.8, labelspacing=0.35)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="both", alpha=0.18)
    style_publication_axis(ax)

    return ax


def save_roc_comparison_plot(output_base_path, records, title, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
    draw_roc_comparison_axis(ax, records, title, legend_loc=legend_loc)

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_pr_comparison_axis(ax, records, title, legend_loc="lower left"):
    positive_rates = [float(np.mean(record["y_true"])) for record in records]
    baseline_rate = positive_rates[0]

    for idx, record in enumerate(records):
        precision, recall, _ = precision_recall_curve(record["y_true"], record["y_score"])
        ax.plot(
            recall,
            precision,
            color=COMPARISON_PALETTE[idx % len(COMPARISON_PALETTE)],
            linewidth=1.5,
            label=f"{record['label']} (AP = {record['average_precision']:.3f})"
        )

    ax.axhline(
        baseline_rate,
        linestyle="--",
        color="#7E8A97",
        linewidth=1.0,
        label=f"Random classifier (baseline = {baseline_rate:.3f})"
    )

    ax.set_xlabel("Recall", fontsize=24)
    ax.set_ylabel("Precision", fontsize=24)
    ax.set_title(title, fontsize=24)
    legend_fontsize = MULTI_CURVE_LEGEND_FONTSIZE if len(records) > 1 else SINGLE_CURVE_LEGEND_FONTSIZE
    ax.legend(frameon=False, fontsize=legend_fontsize, loc=legend_loc, handlelength=1.8, labelspacing=0.35)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="both", alpha=0.18)
    style_publication_axis(ax)

    return ax


def save_pr_comparison_plot(output_base_path, records, title, legend_loc="lower left"):
    fig, ax = plt.subplots(figsize=(7.8, 6.0), constrained_layout=True)
    draw_pr_comparison_axis(ax, records, title, legend_loc=legend_loc)

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_combined_curve_comparison_plot(
    output_base_path,
    records,
    roc_title,
    pr_title,
    roc_legend_loc="lower right",
    pr_legend_loc="lower left",
):
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), constrained_layout=True)
    draw_roc_comparison_axis(axes[0], records, roc_title, legend_loc=roc_legend_loc)
    draw_pr_comparison_axis(axes[1], records, pr_title, legend_loc=pr_legend_loc)

    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def infer_dataset_title(reference_config):
    df_path = reference_config["__normalized_df_prep_path"].lower()
    if "fordigitstress" in df_path:
        return "ForDigitStress dataset"
    return "VR goalkeeper dataset"


def infer_dataset_title_from_name(dataset_name):
    return "ForDigitStress dataset" if dataset_name == "fordigit" else "VR goalkeeper dataset"


def infer_dataset_slug(dataset_title):
    return "fordigitstress" if "ForDigitStress" in dataset_title else "vr_goalkeeper"


def build_curve_titles(dataset_title, num_records):
    if num_records == 1:
        return (
            "ROC curve",
            "Precision-recall curve",
        )
    return (
        "ROC curve comparison across models",
        "Precision-recall curve comparison across models",
    )


def main():
    args = parse_args()
    configure_publication_plot_style(latex=not args.no_latex, tex_path=args.tex_path)

    provided_sources = [args.config_list is not None, args.config is not None, args.dataset is not None]
    if sum(provided_sources) != 1:
        raise ValueError("Pass exactly one of --config-list, --config, or --dataset.")

    if args.config_list is not None:
        config_list = load_config_list(args.config_list)
        explicit_configs = [load_json_config(config_path) for config_path in config_list["configs"]]
        if not explicit_configs:
            raise ValueError("The config list did not contain any recovery configs.")
        records, used_configs, skipped_configs = collect_curve_data_from_configs(explicit_configs)
        dataset_title = config_list["dataset_title"] or infer_dataset_title(explicit_configs[0])
        default_output_dir = (
            normalize_path(config_list["output_dir"])
            if config_list["output_dir"]
            else os.path.join(explicit_configs[0]["__normalized_results_path"], explicit_configs[0]["timestamp"], "curve_comparison")
        )
        custom_roc_title = config_list.get("roc_title")
        custom_pr_title = config_list.get("pr_title")
        roc_legend_loc = config_list.get("roc_legend_loc") or "lower right"
        pr_legend_loc = config_list.get("pr_legend_loc") or "lower left"
    elif args.config is not None:
        reference_config = load_json_config(args.config)
        records, used_configs, skipped_configs = collect_curve_data(reference_config, args.config_dir)
        dataset_title = infer_dataset_title(reference_config)
        default_output_dir = os.path.join(reference_config["__normalized_results_path"], reference_config["timestamp"], "curve_comparison")
        custom_roc_title = None
        custom_pr_title = None
        roc_legend_loc = "lower right"
        pr_legend_loc = "upper right" if "fordigitstress" in reference_config["__normalized_df_prep_path"].lower() else "lower right"
    else:
        matching_configs = get_dataset_matching_configs(args.dataset, args.config_dir)
        if not matching_configs:
            raise FileNotFoundError(f"No matching recovery configs found for dataset '{args.dataset}'.")
        records, used_configs, skipped_configs = collect_curve_data_from_configs(matching_configs)
        dataset_title = infer_dataset_title_from_name(args.dataset)
        first_config = matching_configs[0]
        default_output_dir = os.path.join(first_config["__normalized_results_path"], first_config["timestamp"], "curve_comparison")
        custom_roc_title = None
        custom_pr_title = None
        roc_legend_loc = "lower right"
        pr_legend_loc = "upper right" if args.dataset == "fordigit" else "lower right"

    if len(records) == 0:
        raise ValueError("No recovered experiments/models with aggregated scores were found.")

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else default_output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    dataset_slug = infer_dataset_slug(dataset_title)
    roc_title, pr_title = build_curve_titles(dataset_title, len(records))
    if custom_roc_title:
        roc_title = custom_roc_title
    if custom_pr_title:
        pr_title = custom_pr_title
    save_curve_summary(output_dir, records)
    save_roc_comparison_plot(
        output_base_path=os.path.join(output_dir, f"{dataset_slug}_roc_curve_comparison"),
        records=records,
        title=roc_title,
        legend_loc=roc_legend_loc,
    )
    save_pr_comparison_plot(
        output_base_path=os.path.join(output_dir, f"{dataset_slug}_pr_curve_comparison"),
        records=records,
        title=pr_title,
        legend_loc=pr_legend_loc,
    )
    save_combined_curve_comparison_plot(
        output_base_path=os.path.join(output_dir, f"{dataset_slug}_roc_pr_curve_comparison"),
        records=records,
        roc_title=roc_title,
        pr_title=pr_title,
        roc_legend_loc=roc_legend_loc,
        pr_legend_loc=pr_legend_loc,
    )
    print(f"Saved ROC/PR comparison plots to {output_dir}")
    print("Included configs:")
    for config_path in used_configs:
        print(f"  - {config_path}")
    if skipped_configs:
        print("Skipped configs (missing aggregated recovery arrays):")
        for config_path in skipped_configs:
            print(f"  - {config_path}")


if __name__ == "__main__":
    main()
