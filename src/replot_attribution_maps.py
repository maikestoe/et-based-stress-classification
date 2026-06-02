import argparse
import json
import os
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import recover_outer_cm as roc


BLUE_YELLOW_CMAP = LinearSegmentedColormap.from_list(
    "saliency_blue_yellow",
    ["#E5C04A", "#D8D786", "#8DBB6C", "#5D7FA6", "#123B6D"]
)
BLUE_YELLOW_INVERTED_CMAP = LinearSegmentedColormap.from_list(
    "saliency_blue_yellow_inverted",
    ["#123B6D", "#5D7FA6", "#8DBB6C", "#D8D786", "#E5C04A"]
)
BLUES_CMAP = LinearSegmentedColormap.from_list(
    "saliency_blues",
    ["#2C4668", "#5D789B", "#8EA0B8", "#C7CFDB", "#F4F6F9"]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replot saved saliency/occlusion figures from stored attribution payloads."
    )
    parser.add_argument("--config", required=True, help="Path to recovery config JSON.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["saliency", "occlusion"],
        default=["saliency", "occlusion"],
        help="Which attribution methods to replot."
    )
    parser.add_argument(
        "--colormap",
        choices=["blue_yellow", "blue_yellow_inverted", "blues"],
        default="blue_yellow_inverted",
        help="Colormap to use for the attribution heatmaps."
    )
    parser.add_argument(
        "--limits",
        choices=["local", "saved"],
        default="local",
        help=(
            "How to choose heatmap limits during replotting. "
            "'local' recomputes limits from each saved payload for better visual contrast; "
            "'saved' reuses the original stored limits."
        )
    )
    parser.add_argument(
        "--single-limits",
        choices=["local", "saved"],
        default=None,
        help=(
            "Optional override for single-example plots only. "
            "If omitted, --limits is used."
        )
    )
    parser.add_argument(
        "--mean-limits",
        choices=["local", "saved"],
        default=None,
        help=(
            "Optional override for mean-case plots only. "
            "If omitted, --limits is used."
        )
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
    parser.add_argument(
        "--payload-kind",
        choices=["all", "mean", "single"],
        default="all",
        help="Whether to replot all payloads, only aggregated mean payloads, or only single-example payloads."
    )
    return parser.parse_args()


def load_json_config(config_path):
    source_root = os.path.abspath(os.path.dirname(__file__))
    with open(config_path, "r") as file:
        config = json.load(file)
    if "DL" in config and "path_results" in config["DL"]:
        path_results = os.path.expandvars(config["DL"]["path_results"])
        if not os.path.isabs(path_results):
            path_results = os.path.abspath(os.path.join(source_root, path_results))
        config["DL"]["path_results"] = path_results
    if "df_prep_path" in config:
        df_prep_path = os.path.expandvars(config["df_prep_path"])
        if not os.path.isabs(df_prep_path):
            df_prep_path = os.path.abspath(os.path.join(source_root, df_prep_path))
        config["df_prep_path"] = df_prep_path
    return config


def configure_publication_plot_style(latex=True, tex_path=None):
    plt.style.use(roc.STYLE_PATH)
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


def set_colormap(colormap_name):
    if colormap_name == "blue_yellow":
        roc.SALIENCY_CMAP = BLUE_YELLOW_CMAP
        roc.ATTRIBUTION_STD_FILL_COLOR = "#E5C04A"
    elif colormap_name == "blue_yellow_inverted":
        roc.SALIENCY_CMAP = BLUE_YELLOW_INVERTED_CMAP
        roc.ATTRIBUTION_STD_FILL_COLOR = "#5D7FA6"
    else:
        roc.SALIENCY_CMAP = BLUES_CMAP
        roc.ATTRIBUTION_STD_FILL_COLOR = "#E5C04A"


def load_model_params(npz_file):
    model_params_json = str(npz_file["model_params"].item())
    try:
        return json.loads(model_params_json)
    except Exception:
        return {}


def format_replot_title(title):
    clean_title = str(title).split(", test ID", 1)[0]
    return roc.format_case_text(clean_title)


def get_single_output_base_path(npz_path):
    base_path = os.path.splitext(npz_path)[0]
    method_name = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
    suffix = f"_{method_name}"
    if method_name in {"saliency", "occlusion"} and not base_path.endswith(suffix):
        return f"{base_path}{suffix}"
    return base_path


def get_limits_from_single_payload(data, limit_mode):
    if limit_mode == "saved":
        return (
            None if np.isnan(float(data["heatmap_vmin"])) else float(data["heatmap_vmin"]),
            None if np.isnan(float(data["heatmap_vmax"])) else float(data["heatmap_vmax"]),
        )

    saliency = np.asarray(data["saliency"], dtype=float)
    signal = np.asarray(data["signal"], dtype=float)
    _, saliency_for_plot = roc.align_signal_and_saliency_for_plot(
        signal=signal,
        saliency=saliency,
        model_name=str(data["model_name"].item()),
        model_params=load_model_params(data),
    )
    saliency_smooth = roc.gaussian_filter1d(np.asarray(saliency_for_plot, dtype=float), sigma=3)
    finite_vals = saliency_smooth[np.isfinite(saliency_smooth)]
    positive_vals = finite_vals[finite_vals > np.min(finite_vals) + 1e-12] if finite_vals.size else finite_vals

    source_vals = positive_vals if positive_vals.size >= 10 else finite_vals
    vmin = float(np.percentile(source_vals, 2.0))
    vmax = float(np.percentile(source_vals, 95.0))
    if vmax <= vmin:
        vmin = float(np.min(saliency_smooth))
        vmax = float(np.max(saliency_smooth))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return vmin, vmax


def get_limits_from_mean_payload(data, limit_mode):
    if limit_mode == "saved":
        return (
            None if np.isnan(float(data["heatmap_vmin"])) else float(data["heatmap_vmin"]),
            None if np.isnan(float(data["heatmap_vmax"])) else float(data["heatmap_vmax"]),
        )

    saliency_matrix = np.asarray(data["saliency_matrix"], dtype=float)
    mean_saliency = saliency_matrix.mean(axis=0)
    vmin = float(np.percentile(mean_saliency, 1.0))
    vmax = float(np.percentile(mean_saliency, 99.0))
    if vmax <= vmin:
        vmin = float(np.min(mean_saliency))
        vmax = float(np.max(mean_saliency))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return vmin, vmax


def replot_single_payload(npz_path, limit_mode):
    data = np.load(npz_path, allow_pickle=True)
    output_base_path = get_single_output_base_path(npz_path)
    png_path = f"{output_base_path}.png"
    pdf_path = f"{output_base_path}.pdf"
    heatmap_vmin, heatmap_vmax = get_limits_from_single_payload(data, limit_mode)
    roc.plot_signal_with_saliency(
        signal=data["signal"],
        saliency=data["saliency"],
        title=format_replot_title(data["title"].item()),
        save_path=png_path,
        signal_label=str(data["signal_label"].item()),
        attribution_label=str(data["attribution_label"].item()),
        model_name=str(data["model_name"].item()),
        model_params=load_model_params(data),
        smoothing_sigma=3,
        top_k_regions=5,
        threshold_quantile=0.90,
        min_region_length=3,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax
    )
    # Also refresh PDF to keep both formats aligned.
    roc.plot_signal_with_saliency(
        signal=data["signal"],
        saliency=data["saliency"],
        title=format_replot_title(data["title"].item()),
        save_path=pdf_path,
        signal_label=str(data["signal_label"].item()),
        attribution_label=str(data["attribution_label"].item()),
        model_name=str(data["model_name"].item()),
        model_params=load_model_params(data),
        smoothing_sigma=3,
        top_k_regions=5,
        threshold_quantile=0.90,
        min_region_length=3,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax
    )
    missing = [path for path in (png_path, pdf_path) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Expected replot outputs were not created: " + ", ".join(missing)
        )


def replot_mean_payload(npz_path, limit_mode):
    data = np.load(npz_path, allow_pickle=True)
    output_base_path = os.path.splitext(npz_path)[0]
    heatmap_vmin, heatmap_vmax = get_limits_from_mean_payload(data, limit_mode)
    roc.save_mean_signal_and_saliency_case_plot(
        output_base_path=output_base_path,
        case_name=str(data["case_name"].item()),
        signal_series_list=[row for row in data["signal_matrix"]],
        saliency_series_list=[row for row in data["saliency_matrix"]],
        signal_label=str(data["signal_label"].item()),
        attribution_label=str(data["attribution_label"].item()),
        model_name=str(data["model_name"].item()),
        model_params=load_model_params(data),
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax
    )


def replot_payloads_for_method(method_dir, limit_mode):
    payload_paths = []
    for root, _, files in os.walk(method_dir):
        for file_name in files:
            if file_name.endswith(".npz"):
                payload_paths.append(os.path.join(root, file_name))

    payload_paths = sorted(payload_paths)
    single_count = 0
    mean_count = 0
    selected_count = 0
    failed_paths = []
    for payload_path in payload_paths:
        data = np.load(payload_path, allow_pickle=True)
        is_mean_payload = "signal_matrix" in data.files
        if payload_kind == "mean" and not is_mean_payload:
            continue
        if payload_kind == "single" and is_mean_payload:
            continue
        selected_count += 1
        try:
            if is_mean_payload:
                replot_mean_payload(payload_path, limit_mode)
                mean_count += 1
            else:
                replot_single_payload(payload_path, limit_mode)
                single_count += 1
        except Exception as exc:
            failed_paths.append((payload_path, str(exc)))
    return single_count, mean_count, selected_count, failed_paths


def replot_payloads_for_method_with_modes(method_dir, single_limit_mode, mean_limit_mode, payload_kind="all"):
    payload_paths = []
    for root, _, files in os.walk(method_dir):
        for file_name in files:
            if file_name.endswith(".npz"):
                payload_paths.append(os.path.join(root, file_name))

    payload_paths = sorted(payload_paths)
    single_count = 0
    mean_count = 0
    selected_count = 0
    failed_paths = []
    for payload_path in payload_paths:
        data = np.load(payload_path, allow_pickle=True)
        is_mean_payload = "signal_matrix" in data.files
        if payload_kind == "mean" and not is_mean_payload:
            continue
        if payload_kind == "single" and is_mean_payload:
            continue
        selected_count += 1
        try:
            if is_mean_payload:
                replot_mean_payload(payload_path, mean_limit_mode)
                mean_count += 1
            else:
                replot_single_payload(payload_path, single_limit_mode)
                single_count += 1
        except Exception as exc:
            failed_paths.append((payload_path, str(exc)))
    return single_count, mean_count, selected_count, failed_paths


def find_method_dirs(experiment_base, method_name):
    method_dirs = set()

    top_level = os.path.join(experiment_base, "recovery", method_name)
    if os.path.exists(top_level):
        method_dirs.add(top_level)

    for root, dirs, _ in os.walk(experiment_base):
        if os.path.basename(root) != "recovery":
            continue
        candidate = os.path.join(root, method_name)
        if os.path.exists(candidate):
            method_dirs.add(candidate)

    return sorted(method_dirs)


def main():
    args = parse_args()
    configure_publication_plot_style(latex=not args.no_latex, tex_path=args.tex_path)
    set_colormap(args.colormap)
    config = load_json_config(args.config)
    single_limit_mode = args.single_limits or args.limits
    mean_limit_mode = args.mean_limits or args.limits

    experiment_base = roc.get_base_folder(config)
    total_payloads = 0
    for method_name in args.methods:
        method_dirs = find_method_dirs(experiment_base, method_name)
        if not method_dirs:
            print(f"Skipping {method_name}: no directories found below {experiment_base}")
            continue
        method_single = 0
        method_mean = 0
        method_payloads = 0
        method_failures = []
        for method_dir in method_dirs:
            single_count, mean_count, payload_count, failed_paths = replot_payloads_for_method_with_modes(
                method_dir,
                single_limit_mode,
                mean_limit_mode,
                args.payload_kind,
            )
            method_single += single_count
            method_mean += mean_count
            method_payloads += payload_count
            method_failures.extend(failed_paths)
        total_payloads += method_payloads
        print(
            f"Replotted {method_payloads} payloads for {method_name} "
            f"({method_single} single-example, {method_mean} mean-case)"
        )
        if method_failures:
            print(f"Failed payloads for {method_name}:")
            for payload_path, error_text in method_failures:
                print(f"  {payload_path}")
                print(f"    {error_text}")

    if total_payloads == 0:
        print("No attribution payloads found. Run recover_outer_cm.py once to create .npz replay files.")


if __name__ == "__main__":
    main()
