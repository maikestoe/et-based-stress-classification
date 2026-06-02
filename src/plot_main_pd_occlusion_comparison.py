import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
STYLE_PATH = SRC_DIR / "plot_style2.txt"

DEFAULT_VR_PAYLOAD = (
    RESULTS_DIR
    / "vr_goalkeeper"
    / "DL"
    / "2024-08-09"
    / "CNN"
    / "1"
    / "recovery"
    / "occlusion"
    / "correct_stress_mean_occlusion.npz"
)
DEFAULT_FORDIGIT_PAYLOAD = (
    RESULTS_DIR
    / "fordigitstress"
    / "DL"
    / "2024-08-09"
    / "CNN"
    / "201"
    / "recovery"
    / "occlusion"
    / "correct_stress_mean_occlusion.npz"
)
DEFAULT_OUTPUT_DIR = SRC_DIR / "plots" / "main_pd_occlusion_comparison"

LINE_COLOR = "#3E5F8A"
FILL_COLOR = "#5D7FA6"
TEXT_COLOR = "#323034"
GRID_COLOR = "#B1AFB5"
CMAP = LinearSegmentedColormap.from_list(
    "occlusion_blue_yellow_inverted",
    ["#123B6D", "#5D7FA6", "#8DBB6C", "#D8D786", "#E5C04A"],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact two-panel occlusion-analysis figure for correctly "
            "classified stress samples from the PD CNN models."
        )
    )
    parser.add_argument(
        "--vr-payload",
        type=Path,
        default=DEFAULT_VR_PAYLOAD,
        help="Recovered VR Goalkeeper correct-stress mean occlusion .npz payload.",
    )
    parser.add_argument(
        "--fordigit-payload",
        type=Path,
        default=DEFAULT_FORDIGIT_PAYLOAD,
        help="Recovered ForDigitStress correct-stress mean occlusion .npz payload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated PNG and PDF files.",
    )
    parser.add_argument(
        "--output-name",
        default="main_pd_cnn_correct_stress_occlusion_comparison",
        help="Base filename without extension.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help="Trial window represented by the attribution vectors.",
    )
    parser.add_argument(
        "--separate-scales",
        action="store_true",
        help="Use separate heatmap color limits for the two datasets.",
    )
    parser.add_argument(
        "--saved-scales",
        action="store_true",
        help="Use the heatmap limits stored in the payloads instead of mean-plot limits.",
    )
    parser.add_argument(
        "--use-latex",
        action="store_true",
        help="Use LaTeX text rendering if a local LaTeX installation is available.",
    )
    return parser.parse_args()


def configure_style(use_latex):
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    plt.rcParams.update(
        {
            "text.usetex": bool(use_latex),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_payload(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing attribution payload: {path}")

    data = np.load(path, allow_pickle=True)
    required_keys = {
        "signal_matrix",
        "saliency_matrix",
        "signal_label",
        "attribution_label",
        "model_name",
        "model_params",
    }
    missing = sorted(required_keys.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    signal_matrix = np.asarray(data["signal_matrix"], dtype=float)
    attribution_matrix = np.asarray(data["saliency_matrix"], dtype=float)
    if signal_matrix.shape != attribution_matrix.shape:
        raise ValueError(
            f"Signal and attribution matrices must have the same shape in {path}; "
            f"got {signal_matrix.shape} and {attribution_matrix.shape}."
        )

    return {
        "path": path,
        "signal_matrix": signal_matrix,
        "attribution_matrix": attribution_matrix,
        "signal_label": str(data["signal_label"].item()),
        "attribution_label": str(data["attribution_label"].item()),
        "model_name": str(data["model_name"].item()),
        "model_params": load_model_params(data),
        "saved_vmin": scalar_or_none(data, "heatmap_vmin"),
        "saved_vmax": scalar_or_none(data, "heatmap_vmax"),
    }


def scalar_or_none(data, key):
    if key not in data.files:
        return None
    value = float(data[key])
    return None if np.isnan(value) else value


def load_model_params(data):
    try:
        return json.loads(str(data["model_params"].item()))
    except Exception:
        return {}


def summarize_matrix(matrix):
    return {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0),
        "n": matrix.shape[0],
        "length": matrix.shape[1],
    }


def time_axis(length, duration_seconds):
    return np.linspace(0.0, duration_seconds, length, endpoint=False)


def format_signal_label(label):
    if label.strip().upper() == "PD":
        return "PD"
    return label


def get_heatmap_limits(payloads, separate_scales, saved_scales):
    limits = []
    for payload in payloads:
        mean_attr = summarize_matrix(payload["attribution_matrix"])["mean"]
        vmin = 0.0
        vmax = payload["saved_vmax"] if saved_scales else None
        if vmax is None or vmax <= vmin:
            vmax = float(np.percentile(mean_attr, 99.0))
        if vmax <= vmin:
            vmax = float(np.max(mean_attr))
        if vmax <= vmin:
            vmax = vmin + 1e-8
        limits.append((vmin, vmax))

    if separate_scales:
        return limits

    shared_vmax = max(vmax for _, vmax in limits)
    return [(0.0, shared_vmax) for _ in limits]


def style_axis(ax, grid=True):
    if grid:
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)


def plot_dataset_column(
    axes,
    payload,
    title,
    duration_seconds,
    heatmap_vmin,
    heatmap_vmax,
    show_y_labels,
):
    signal = summarize_matrix(payload["signal_matrix"])
    attribution = summarize_matrix(payload["attribution_matrix"])
    time = time_axis(signal["length"], duration_seconds)

    ax_signal, ax_attr, ax_heatmap = axes

    ax_signal.plot(time, signal["mean"], color=LINE_COLOR, linewidth=1.2)
    ax_signal.fill_between(
        time,
        signal["mean"] - signal["std"],
        signal["mean"] + signal["std"],
        color=FILL_COLOR,
        alpha=0.10,
        linewidth=0,
    )
    ax_signal.set_title(f"{title}\n(n = {signal['n']})", loc="left", pad=3)
    if show_y_labels:
        ax_signal.set_ylabel(format_signal_label(payload["signal_label"]))
    style_axis(ax_signal)

    ax_attr.plot(time, attribution["mean"], color=LINE_COLOR, linewidth=1.2)
    ax_attr.fill_between(
        time,
        np.maximum(0.0, attribution["mean"] - attribution["std"]),
        attribution["mean"] + attribution["std"],
        color=FILL_COLOR,
        alpha=0.26,
        linewidth=0,
    )
    ax_attr.set_ylim(bottom=0.0)
    if show_y_labels:
        ax_attr.set_ylabel("Importance")
    style_axis(ax_attr)

    image = attribution["mean"][np.newaxis, :]
    im = ax_heatmap.imshow(
        image,
        aspect="auto",
        cmap=CMAP,
        extent=[0.0, duration_seconds, 0.0, 1.0],
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xlabel("Time within trial (s)")
    if show_y_labels:
        ax_heatmap.set_ylabel("")
    style_axis(ax_heatmap, grid=False)
    ax_heatmap.spines["left"].set_visible(False)

    for ax in axes:
        ax.set_xlim(0.0, duration_seconds)

    return im


def create_figure(vr_payload, fordigit_payload, args):
    payloads = [vr_payload, fordigit_payload]
    limits = get_heatmap_limits(payloads, args.separate_scales, args.saved_scales)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 3.85),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.1, 2.35, 0.42], "hspace": 0.04, "wspace": 0.10},
    )

    images = []
    images.append(
        plot_dataset_column(
            axes[:, 0],
            vr_payload,
            "VR Goalkeeper",
            args.duration_seconds,
            limits[0][0],
            limits[0][1],
            show_y_labels=True,
        )
    )
    images.append(
        plot_dataset_column(
            axes[:, 1],
            fordigit_payload,
            "ForDigitStress",
            args.duration_seconds,
            limits[1][0],
            limits[1][1],
            show_y_labels=False,
        )
    )

    if args.separate_scales:
        for image, ax in zip(images, axes[2, :]):
            cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.35, pad=0.48)
            cbar.ax.tick_params(labelsize=6, length=2)
    else:
        cbar = fig.colorbar(
            images[0],
            ax=axes.ravel().tolist(),
            orientation="vertical",
            fraction=0.025,
            pad=0.012,
        )
        cbar.set_label("Occlusion importance", fontsize=8)
        cbar.ax.tick_params(labelsize=6, length=2)

    return fig


def main():
    args = parse_args()
    configure_style(args.use_latex)

    vr_payload = load_payload(args.vr_payload)
    fordigit_payload = load_payload(args.fordigit_payload)

    fig = create_figure(vr_payload, fordigit_payload, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.output_name}.png"
    pdf_path = args.output_dir / f"{args.output_name}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
