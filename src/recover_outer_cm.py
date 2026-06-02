import gc
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from pickle_compat import install_pandas_pickle_compat

install_pandas_pickle_compat()

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter1d
from tensorflow.keras import backend as K

from DL_models import get_model
from DL_train_CV import transform_data_as_input
from DL_utils import get_training_data, load_config, process_data, set_DL_config

try:
    from fau_colors import colors_dark
except ImportError:
    colors_dark = None


STYLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "plot_style2.txt"))
PUBLICATION_LINE_COLOR = "#3E5F8A"
ATTRIBUTION_STD_FILL_COLOR = "#5D7FA6"
ATTRIBUTION_TITLE_FONTSIZE = 22
ATTRIBUTION_AXIS_LABEL_FONTSIZE = 22
ATTRIBUTION_TICK_FONTSIZE = 18
ATTRIBUTION_COLORBAR_LABEL_FONTSIZE = 20
ATTRIBUTION_COLORBAR_TICK_FONTSIZE = 16
ATTRIBUTION_REGION_LABEL_FONTSIZE = 15
CONFUSION_TICK_LABEL_FONTSIZE = 20
CONFUSION_CELL_LABEL_FONTSIZE = 20
YELLOW_CURVE_COLOR = "#E5C04A"
TRANSLUCENT_RED_CURVE_COLOR = (196 / 255.0, 79 / 255.0, 94 / 255.0, 0.78)
SALIENCY_CMAP = LinearSegmentedColormap.from_list(
    "saliency_blue_yellow_inverted",
    ["#123B6D", "#5D7FA6", "#8DBB6C", "#D8D786", "#E5C04A"]
)
SALIENCY_ANNOTATION_COLOR = "#1F3552"
INPUT_SIGNAL_LABELS = {
    "meanDia_corrected": "PD",
    "velocity": "angular velocity",
    "acceleration": "angular acceleration",
    "position": "visual angle",
    "asymptotic_model": "asymptotic model",
    "fixations": "fixations"
}
ATTRIBUTION_CASE_NAMES = (
    "correct_non_stress",
    "correct_stress",
    "wrong_non_stress",
    "wrong_stress",
)


def empty_attribution_case_map():
    """Return the standard attribution case map used for per-case aggregation."""
    return {case_name: [] for case_name in ATTRIBUTION_CASE_NAMES}


def get_segment_layout(model_name, model_params, signal_length):
    model_name = str(model_name).lower()
    if "convlstm" not in model_name:
        return None

    if not isinstance(model_params, dict):
        return None

    num_segments = model_params.get("num_segments")
    if num_segments is None and "shared" in model_params and isinstance(model_params["shared"], dict):
        num_segments = model_params["shared"].get("num_segments")

    if num_segments is None:
        return None

    num_segments = int(num_segments)
    if num_segments <= 1 or signal_length <= 1:
        return None

    segment_length = signal_length / num_segments
    boundaries = [i * segment_length for i in range(1, num_segments)]
    return {
        "num_segments": num_segments,
        "segment_length": segment_length,
        "boundaries": boundaries
    }


def draw_segment_guides(ax, segment_layout, add_label=False):
    if not segment_layout:
        return

    for boundary in segment_layout["boundaries"]:
        ax.axvline(boundary, color="#6B7C93", linestyle="--", linewidth=0.6, alpha=0.22, zorder=0)

    if add_label:
        ax.text(
            0.995,
            0.96,
            f"{segment_layout['num_segments']} segments (~{segment_layout['segment_length']:.0f} samples each)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#4A5A70",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75
            }
        )


def get_saliency_x_label(model_name):
    model_name = str(model_name).lower()
    if "convlstm" in model_name:
        return "Sample index (segment-projected)"
    if "lstm" in model_name:
        return "Sample index"
    if "cnn" in model_name:
        return "Sample index"
    return "Sample index"

def get_trial_parameters_from_study(config, trial):
    model_parameters = {}
    model_config = config['optuna'][config['DL']['model']]

    for parameter in model_config:
        model_parameters[parameter] = trial.params[parameter]

    return model_parameters


def configure_publication_plot_style(latex=True, tex_path=None):
    """
    Configure matplotlib with the publication plot style used in the analysis scripts.

    Falls back gracefully if LaTeX is unavailable.
    """
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


def load_best_trial_from_db(db_path, test_id, experiment_id):
    storage_url = f"sqlite:///{db_path}"
    study_name = f"{test_id}_optunaStudy_{experiment_id}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return study.best_trial


def predict_for_subject(config, test_id):
    base_folder = os.path.join(
        config['DL']['path_results'],
        config['timestamp'],
        config['DL']['model'],
        config['experiment_id']
    )

    subject_folder = os.path.join(base_folder, f"test_ID_{test_id}")
    weights_path = os.path.join(subject_folder, "best_weights_foldnan.weights.h5")

    db_path = os.path.join(base_folder, "optunaStudy.db")
    best_trial = load_best_trial_from_db(db_path, test_id, config['experiment_id'])

    batch_size = best_trial.params['batch_size']
    model_params = get_trial_parameters_from_study(config, best_trial)

    with open(config['df_prep_path'], "rb") as fh:
        df = pickle.load(fh)

    df['lab_num'] = df['lab_num'].astype(int)

    train_idx = df.index[df['ID'] != test_id].tolist()
    test_idx = df.index[df['ID'] == test_id].tolist()

    x_train, y_train, x_test, y_test, train_ids, test_ids = get_training_data(df, train_idx, test_idx, config)
    x_test_raw_for_plot = np.array(x_test, copy=True)
    del df, train_idx, test_idx

    x_train, y_train, x_test, y_test = process_data(x_train, y_train, x_test, y_test, train_ids, config)
    del train_ids, test_ids
    gc.collect()

    # IMPORTANT: keep original input dimensions before model-specific transformation
    data_dim = np.size(x_train, 2)
    timesteps = np.size(x_train, 1)
    num_classes = 2

    # input transform exactly as during training
    x_train_trans, x_test_trans = transform_data_as_input(x_train, x_test, model_params, config)
    del x_train, y_train
    gc.collect()

    model = get_model(config['DL']['model'], data_dim, timesteps, num_classes, model_params)
    model.load_weights(weights_path)

    # direct prediction on x_test
    y_prob = model.predict(x_test_trans, batch_size=batch_size, verbose=0)
    y_test = np.asarray(y_test)

    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.astype(int).flatten()

    if y_prob.shape[1] > 1:
        y_score = y_prob[:, 1]
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_score = y_prob.flatten()
        y_pred = (y_score > 0.5).astype(int)

    signal_label = get_input_signal_display_name(config)
    use_raw_signal_for_plot = should_use_raw_signal_for_saliency_plot(config)
    attribution_methods = get_attribution_methods(config)
    attribution_results = {}

    if supports_attribution(config, x_test_trans):
        case_indices = select_saliency_indices_by_case(y_true, y_pred)
        case_indices = filter_saliency_case_indices(config, case_indices)
        attribution_results = {
            method: {
                "cases": empty_attribution_case_map(),
                "signals": empty_attribution_case_map(),
                "plot_jobs": []
            }
            for method in attribution_methods
        }

        for attribution_method in attribution_methods:
            for case_name, indices in case_indices.items():
                for idx in indices:
                    sample = x_test_trans[idx]
                    if attribution_method == "occlusion":
                        attribution = compute_temporal_occlusion(model, sample, config)
                        attribution_label = "Occlusion importance"
                    else:
                        attribution = compute_saliency(model, sample)
                        attribution_label = "Saliency"
                    plot_signal_source = x_test_raw_for_plot[idx] if use_raw_signal_for_plot else x_test[idx]
                    signal_for_plot, saliency_for_plot = align_signal_and_saliency_for_plot(
                        signal=plot_signal_source,
                        saliency=attribution,
                        model_name=config["DL"]["model"],
                        model_params=model_params
                    )

                    attribution_results[attribution_method]["cases"][case_name].append(saliency_for_plot)
                    attribution_results[attribution_method]["signals"][case_name].append(signal_for_plot)
                    attribution_results[attribution_method]["plot_jobs"].append(
                        {
                            "case_name": case_name,
                            "sample_idx": idx,
                            "signal": signal_for_plot,
                            "saliency": saliency_for_plot,
                            "attribution_label": attribution_label,
                        }
                    )

        for attribution_method in attribution_methods:
            method_result = attribution_results[attribution_method]
            heatmap_vmin, heatmap_vmax = get_shared_heatmap_limits(method_result["cases"])
            attribution_dir = os.path.join(subject_folder, "recovery", attribution_method)
            os.makedirs(attribution_dir, exist_ok=True)

            for plot_job in method_result["plot_jobs"]:
                saliency_base = os.path.join(subject_folder, "recovery", attribution_method, plot_job["case_name"])
                os.makedirs(saliency_base, exist_ok=True)

                save_path = os.path.join(
                    saliency_base,
                    f"{plot_job['case_name']}_testID_{test_id}_sample_{plot_job['sample_idx']}_{attribution_method}.png"
                )
                output_base_path = os.path.splitext(save_path)[0]

                plot_signal_with_saliency(
                    signal=plot_job["signal"],
                    saliency=plot_job["saliency"],
                    title=format_single_attribution_title(
                        plot_job["case_name"], test_id, plot_job["sample_idx"]
                    ),
                    save_path=save_path,
                    signal_label=signal_label,
                    attribution_label=plot_job["attribution_label"],
                    model_name=config["DL"]["model"],
                    model_params=model_params,
                    smoothing_sigma=3,
                    top_k_regions=5,
                    threshold_quantile=0.90,
                    min_region_length=3,
                    heatmap_vmin=heatmap_vmin,
                    heatmap_vmax=heatmap_vmax
                )
                plot_signal_with_saliency(
                    signal=plot_job["signal"],
                    saliency=plot_job["saliency"],
                    title=format_single_attribution_title(
                        plot_job["case_name"], test_id, plot_job["sample_idx"]
                    ),
                    save_path=f"{output_base_path}.pdf",
                    signal_label=signal_label,
                    attribution_label=plot_job["attribution_label"],
                    model_name=config["DL"]["model"],
                    model_params=model_params,
                    smoothing_sigma=3,
                    top_k_regions=5,
                    threshold_quantile=0.90,
                    min_region_length=3,
                    heatmap_vmin=heatmap_vmin,
                    heatmap_vmax=heatmap_vmax
                )
                save_single_attribution_payload(
                    output_base_path=output_base_path,
                    signal=plot_job["signal"],
                    saliency=plot_job["saliency"],
                    title=format_single_attribution_title(
                        plot_job["case_name"], test_id, plot_job["sample_idx"]
                    ),
                    signal_label=signal_label,
                    attribution_label=plot_job["attribution_label"],
                    model_name=config["DL"]["model"],
                    model_params=model_params,
                    heatmap_vmin=heatmap_vmin,
                    heatmap_vmax=heatmap_vmax
                )

            del method_result["plot_jobs"]

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan

    recovery_dir = os.path.join(subject_folder, "recovery")
    os.makedirs(recovery_dir, exist_ok=True)
    dataset_label = get_dataset_label(config)
    model_label = config["DL"]["model"].lower()
    experiment_label = f"exp{config['experiment_id']}"
    np.savetxt(os.path.join(recovery_dir, "confusion_matrix_outer_recovered.csv"), cm, delimiter=",", fmt="%d")
    save_confusion_matrix_heatmap(
        output_base_path=os.path.join(
            recovery_dir,
            f"{dataset_label}_{experiment_label}_{model_label}_confusion_matrix_outer_recovered"
        ),
        cm=cm,
        title=build_confusion_matrix_title(config),
        include_row_percent=True
    )
    np.save(os.path.join(recovery_dir, "y_true_outer.npy"), y_true)
    np.save(os.path.join(recovery_dir, "y_pred_outer.npy"), y_pred)
    np.save(os.path.join(recovery_dir, "y_score_outer.npy"), y_score)

    with open(os.path.join(recovery_dir, "outer_metrics_recovered.txt"), "w") as f:
        f.write(f"test_id: {test_id}\n")
        f.write(f"macro_f1: {f1:.6f}\n")
        f.write(f"precision_macro: {precision:.6f}\n")
        f.write(f"recall_macro: {recall:.6f}\n")
        f.write(f"roc_auc: {auc:.6f}\n")

    del model, x_test_trans, y_test, y_prob
    K.clear_session()
    gc.collect()

    return {
        "test_id": test_id,
        "cm": cm,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "y_true": y_true,
        "y_score": y_score,
        "saliency_cases": attribution_results,
        "signal_label": signal_label,
        "model_params": model_params,
        "attribution_method": "both" if len(attribution_methods) > 1 else attribution_methods[0]
    }


def get_attribution_methods(config):
    recovery_cfg = config.get("recovery", {})
    method = str(recovery_cfg.get("attribution_method", "saliency")).strip().lower()
    if method == "both":
        return ["saliency", "occlusion"]
    if method not in {"saliency", "occlusion"}:
        raise ValueError(f"Unsupported attribution_method '{method}'. Use 'saliency', 'occlusion', or 'both'.")
    return [method]


def supports_attribution(config, x_test_trans):
    """
    Determine whether attribution computation is supported for the current model/input.

    :param config: Configuration dictionary.
    :type config: dict
    :param x_test_trans: Transformed test input.
    :type x_test_trans: np.ndarray | list
    :return: True if attribution should be computed, False otherwise.
    :rtype: bool
    """
    recovery_cfg = config.get("recovery", {})
    if not recovery_cfg.get("compute_saliency", False):
        return False

    model_name = config['DL']['model'].lower()

    # Skip multi-input models
    if isinstance(x_test_trans, list):
        return False

    # Only support single-input arrays here
    if not isinstance(x_test_trans, np.ndarray):
        return False

    return True


def compute_saliency(model, sample):
    """
    Compute a gradient-based saliency map for a single input sample.

    :param model: Trained Keras model.
    :type model: tf.keras.Model
    :param sample: Single input sample without batch dimension.
    :type sample: np.ndarray
    :return: Absolute saliency values with the same shape as the input sample.
    :rtype: np.ndarray
    """
    sample_tf = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(sample_tf)
        preds = model(sample_tf, training=False)

        if preds.shape[-1] > 1:
            class_idx = tf.argmax(preds[0])
            class_score = preds[:, class_idx]
        else:
            class_score = preds[:, 0]

    grads = tape.gradient(class_score, sample_tf)
    saliency = tf.abs(grads)[0].numpy()
    return saliency


def compute_temporal_occlusion(model, sample, config):
    """
    Compute a temporal occlusion importance profile for a single sample.
    """
    sample = np.asarray(sample, dtype=np.float32)
    flat_sample = sample.reshape(-1)
    if flat_sample.size == 0:
        return np.zeros_like(sample, dtype=float)

    recovery_cfg = config.get("recovery", {})
    window_size = int(recovery_cfg.get("occlusion_window_size", 15))
    stride = int(recovery_cfg.get("occlusion_stride", max(1, window_size // 3)))
    fill_mode = str(recovery_cfg.get("occlusion_fill", "mean")).lower()
    baseline_value = 0.0 if fill_mode == "zero" else float(np.mean(flat_sample))

    base_preds = model(sample[None, ...], training=False).numpy()[0]
    class_idx = int(np.argmax(base_preds)) if base_preds.shape[-1] > 1 else 0
    base_score = float(base_preds[class_idx])

    window_size = max(1, min(window_size, flat_sample.size))
    stride = max(1, stride)
    importance = np.zeros(flat_sample.size, dtype=float)
    counts = np.zeros(flat_sample.size, dtype=float)

    for start in range(0, flat_sample.size, stride):
        end = min(start + window_size, flat_sample.size)
        occluded = flat_sample.copy()
        occluded[start:end] = baseline_value
        occluded = occluded.reshape(sample.shape)

        occluded_preds = model(occluded[None, ...], training=False).numpy()[0]
        occluded_score = float(occluded_preds[class_idx])
        score_drop = max(0.0, base_score - occluded_score)
        importance[start:end] += score_drop
        counts[start:end] += 1.0

        if end >= flat_sample.size:
            break

    counts[counts == 0] = 1.0
    return (importance / counts).reshape(sample.shape)


def get_input_signal_display_name(config):
    input_cols = config.get("DL", {}).get("input_cols", [])
    if not input_cols:
        return "Signal"
    signal_name = input_cols[0]
    return INPUT_SIGNAL_LABELS.get(signal_name, signal_name.replace("_", " "))


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


def get_dataset_display_name(config):
    return "ForDigitStress dataset" if is_fordigitstress_config(config) else "VR goalkeeper dataset"


def build_confusion_matrix_title(config):
    return f"{get_model_display_name(config)} ({get_input_signal_display_name(config)})\n{get_dataset_display_name(config)}"


def should_use_raw_signal_for_saliency_plot(config):
    recovery_cfg = config.get("recovery", {})
    if "use_raw_signal_for_plot" in recovery_cfg:
        return bool(recovery_cfg["use_raw_signal_for_plot"])

    input_cols = config.get("DL", {}).get("input_cols", [])
    if len(input_cols) == 1 and input_cols[0] == "fix_array":
        return True
    return False


def resize_1d_series(values, target_length):
    values = np.asarray(values, dtype=float)
    if len(values) == target_length:
        return values
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(x_new, x_old, values)


def reduce_to_time_series(arr):
    """
    Reduce an input array of arbitrary shape to a 1D time series.

    Strategy:
    - squeeze singleton dimensions
    - if already 1D: return as is
    - otherwise assume the longest axis is time
    - average over all remaining axes

    :param arr: Input array
    :type arr: np.ndarray
    :return: 1D array
    :rtype: np.ndarray
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr

    time_axis = int(np.argmax(arr.shape))
    arr = np.moveaxis(arr, time_axis, 0)

    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1)

    return arr


def reduce_saliency_to_time_series(arr, model_name):
    """
    Reduce a saliency tensor to a 1D time series in a model-aware way.

    For ConvLSTM inputs, gradients are computed on segmented inputs with shape
    roughly `(num_segments, 1, segment_length, 1)`. Here we flatten the segment
    axis back into the original temporal order to obtain a temporal importance
    profile that is easier to interpret.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    model_name = str(model_name).lower()

    if "convlstm" in model_name:
        if arr.ndim == 0:
            return np.asarray([float(arr)])
        if arr.ndim == 1:
            return arr.astype(float)
        return arr.reshape(-1).astype(float)

    return reduce_to_time_series(arr).astype(float)


def align_signal_and_saliency_for_plot(signal, saliency, model_name, model_params=None):
    """
    Align the original signal and the saliency trace to the same time axis.

    For ConvLSTM models the saliency is back-projected from segmented input to a
    flattened temporal profile. If segmentation trimmed trailing samples, the
    original signal is trimmed accordingly.
    """
    signal_1d = reduce_to_time_series(signal).astype(float)
    saliency_1d = reduce_saliency_to_time_series(saliency, model_name)

    if len(signal_1d) == len(saliency_1d):
        return signal_1d, saliency_1d

    model_name = str(model_name).lower()
    if "convlstm" in model_name and len(signal_1d) >= len(saliency_1d):
        return signal_1d[:len(saliency_1d)], saliency_1d

    saliency_1d = resize_1d_series(saliency_1d, len(signal_1d))
    return signal_1d, saliency_1d


def get_top_salient_regions(saliency, top_k=5, threshold_quantile=0.90, min_region_length=3):
    """
    Extract top salient contiguous regions from a 1D saliency signal.

    :param saliency: 1D saliency array
    :type saliency: np.ndarray
    :param top_k: Number of top regions to return
    :type top_k: int
    :param threshold_quantile: Quantile used to define salient regions
    :type threshold_quantile: float
    :param min_region_length: Minimum number of consecutive points per region
    :type min_region_length: int
    :return: List of tuples (start_idx, end_idx, mean_saliency)
    :rtype: list[tuple]
    """
    saliency = np.asarray(saliency).astype(float)
    if saliency.ndim != 1:
        raise ValueError("Saliency must be 1D")

    threshold = np.quantile(saliency, threshold_quantile)
    mask = saliency >= threshold

    regions = []
    start = None

    for i, is_salient in enumerate(mask):
        if is_salient and start is None:
            start = i
        elif not is_salient and start is not None:
            end = i - 1
            if (end - start + 1) >= min_region_length:
                mean_val = float(np.mean(saliency[start:end + 1]))
                regions.append((start, end, mean_val))
            start = None

    if start is not None:
        end = len(mask) - 1
        if (end - start + 1) >= min_region_length:
            mean_val = float(np.mean(saliency[start:end + 1]))
            regions.append((start, end, mean_val))

    regions = sorted(regions, key=lambda x: x[2], reverse=True)
    return regions[:top_k]


def format_attribution_case_name(case_name):
    return str(case_name).replace("_", " ")


def format_case_text(text):
    normalized = str(text).strip().replace("_", " ").lower()
    normalized = normalized.replace("non stress", "non-stress")
    if not normalized:
        return normalized
    return normalized[0].upper() + normalized[1:]


def format_attribution_case_title(case_name):
    return format_case_text(format_attribution_case_name(case_name))


def format_single_attribution_title(case_name, test_id, sample_idx):
    return format_attribution_case_title(case_name)


def format_mean_signal_label(signal_label):
    signal_label = str(signal_label).strip()
    if signal_label.lower() == "pd":
        return "Mean PD"
    if signal_label.lower() == "asymptotic model":
        return "Mean asympt. model"
    return f"Mean {signal_label}"


def format_signal_axis_label(signal_label):
    signal_label = str(signal_label).strip()
    if signal_label.lower() == "asymptotic model":
        return "Asympt. model"
    return signal_label


def set_left_aligned_ylabel(ax, label, x_coord, y_coord=0.5):
    ax.set_ylabel(label, fontsize=ATTRIBUTION_AXIS_LABEL_FONTSIZE)
    ax.yaxis.set_label_coords(x_coord, y_coord)
    ax.yaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("center")


def format_mean_attribution_axis_label(attribution_label):
    label = str(attribution_label).strip().lower()
    if label == "occlusion importance":
        return "Mean occl. importance"
    if label == "saliency":
        return "Mean saliency"
    return f"Mean {label}"


def format_mean_attribution_colorbar_label(attribution_label):
    label = str(attribution_label).strip().lower()
    if label == "occlusion importance":
        return "Mean occl. importance"
    if label == "saliency":
        return "Mean saliency"
    return f"Mean {label}"


def plot_signal_with_saliency(
    signal,
    saliency,
    title="",
    save_path=None,
    signal_label="Signal",
    attribution_label="Saliency",
    model_name="",
    model_params=None,
    figsize=(11, 5),
    smoothing_sigma=3,
    top_k_regions=5,
    threshold_quantile=0.90,
    min_region_length=3,
    heatmap_vmin=None,
    heatmap_vmax=None
):
    """
    Plot time series signal and corresponding smoothed attribution in aligned panels.
    Top salient regions are highlighted in the signal plot.

    :param signal: Input signal array
    :type signal: np.ndarray
    :param saliency: Saliency array
    :type saliency: np.ndarray
    :param title: Plot title
    :type title: str
    :param save_path: Output path
    :type save_path: str | None
    :param figsize: Figure size
    :type figsize: tuple
    :param smoothing_sigma: Gaussian smoothing sigma for saliency
    :type smoothing_sigma: float
    :param top_k_regions: Number of salient regions to highlight
    :type top_k_regions: int
    :param threshold_quantile: Quantile threshold for salient region extraction
    :type threshold_quantile: float
    :param min_region_length: Minimum contiguous region length
    :type min_region_length: int
    :return: None
    :rtype: None
    """

    signal_1d = reduce_to_time_series(signal)
    saliency_1d = reduce_to_time_series(saliency)

    if len(signal_1d) != len(saliency_1d):
        raise ValueError(
            f"Signal and saliency must have same length after reduction, "
            f"got {len(signal_1d)} and {len(saliency_1d)}"
        )

    # Smooth saliency
    saliency_smooth = gaussian_filter1d(saliency_1d.astype(float), sigma=smoothing_sigma)

    # Normalize only for region extraction/overlay, but keep absolute values for the heatmap
    saliency_norm = (saliency_smooth - np.min(saliency_smooth)) / (
        np.max(saliency_smooth) - np.min(saliency_smooth) + 1e-8
    )

    if heatmap_vmin is None:
        heatmap_vmin = float(np.min(saliency_smooth))
    if heatmap_vmax is None:
        heatmap_vmax = float(np.max(saliency_smooth))
    if heatmap_vmax <= heatmap_vmin:
        heatmap_vmax = heatmap_vmin + 1e-8

    # Find top salient regions
    top_regions = get_top_salient_regions(
        saliency=saliency_norm,
        top_k=top_k_regions,
        threshold_quantile=threshold_quantile,
        min_region_length=min_region_length
    )

    time = np.arange(len(signal_1d))
    segment_layout = get_segment_layout(model_name, model_params, len(signal_1d))

    annotation_color = SALIENCY_ANNOTATION_COLOR
    signal_color = PUBLICATION_LINE_COLOR

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(figsize[0], max(figsize[1], 7.7)),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.7, 0.55], "hspace": 0.08}
    )

    # --- Top panel: signal ---
    ax1.plot(time, signal_1d, color=signal_color, linewidth=1.6, zorder=3)
    ax1.set_ylabel(format_signal_axis_label(signal_label), fontsize=ATTRIBUTION_AXIS_LABEL_FONTSIZE)
    ax1.set_title(title, fontsize=ATTRIBUTION_TITLE_FONTSIZE)
    ax1.grid(alpha=0.18, axis="y")
    ax1.margins(x=0.01, y=0.08)
    style_publication_axis(ax1)
    ax1.tick_params(axis="both", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    draw_segment_guides(ax1, segment_layout, add_label=True)

    y_min, y_max = ax1.get_ylim()
    y_text = y_max - 0.06 * (y_max - y_min)

    # Highlight top salient regions with a light background overlay.
    for rank, (start, end, mean_val) in enumerate(top_regions, start=1):
        region_color = SALIENCY_CMAP(np.clip(mean_val, 0.0, 1.0))
        ax1.axvspan(start, end, color=region_color, alpha=0.10, zorder=1, ec="none")
        ax1.text(
            x=(start + end) / 2,
            y=y_text,
            s=str(rank),
            ha="center",
            va="top",
            fontsize=ATTRIBUTION_REGION_LABEL_FONTSIZE,
            color=annotation_color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7
            }
        )

    # --- Middle panel: attribution time series ---
    ax2.plot(time, saliency_smooth, color=PUBLICATION_LINE_COLOR, linewidth=1.8, zorder=3)
    ax2.fill_between(
        time,
        0.0,
        saliency_smooth,
        color=ATTRIBUTION_STD_FILL_COLOR,
        alpha=0.20,
        zorder=2
    )
    ax2.set_ylabel(attribution_label, fontsize=ATTRIBUTION_AXIS_LABEL_FONTSIZE)
    ax2.grid(alpha=0.18, axis="y")
    ax2.set_ylim(bottom=0.0)
    ax2.margins(x=0.01, y=0.08)
    style_publication_axis(ax2)
    ax2.tick_params(axis="both", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    draw_segment_guides(ax2, segment_layout)

    # --- Bottom panel: saliency heatbar ---
    saliency_img = saliency_smooth[np.newaxis, :]
    im = ax3.imshow(
        saliency_img,
        aspect="auto",
        cmap=SALIENCY_CMAP,
        extent=[time.min(), time.max(), 0, 1],
        vmin=heatmap_vmin,
        vmax=heatmap_vmax
    )

    # Overlay region borders
    for start, end, mean_val in top_regions:
        region_color = SALIENCY_CMAP(np.clip(mean_val, 0.0, 1.0))
        ax3.axvspan(start, end, color=region_color, alpha=0.30)

    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.set_xlabel(get_saliency_x_label(model_name), fontsize=ATTRIBUTION_AXIS_LABEL_FONTSIZE)
    style_publication_axis(ax3)
    ax3.tick_params(axis="x", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    ax3.spines["left"].set_visible(False)
    draw_segment_guides(ax3, segment_layout)

    # Colorbar on the right
    cbar = fig.colorbar(im, ax=[ax1, ax2, ax3], orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(format_mean_attribution_colorbar_label(attribution_label), fontsize=ATTRIBUTION_COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=ATTRIBUTION_COLORBAR_TICK_FONTSIZE)

    ax1.yaxis.set_label_coords(-0.08, 0.5)
    ax2.yaxis.set_label_coords(-0.08, 0.5)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
        plt.close()
    else:
        plt.show()


def save_mean_saliency_case_plot(output_base_path, case_name, saliency_series_list, model_name="", model_params=None):
    if not saliency_series_list:
        return

    target_length = int(np.median([len(series) for series in saliency_series_list]))
    saliency_matrix = np.vstack([resize_1d_series(series, target_length) for series in saliency_series_list])
    mean_saliency = saliency_matrix.mean(axis=0)
    std_saliency = saliency_matrix.std(axis=0)
    time = np.arange(target_length)
    segment_layout = get_segment_layout(model_name, model_params, target_length)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(9, 5.2),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.8, 0.55], "hspace": 0.06}
    )

    ax1.plot(time, mean_saliency, color=PUBLICATION_LINE_COLOR, linewidth=1.6)
    ax1.fill_between(
        time,
        np.maximum(0.0, mean_saliency - std_saliency),
        mean_saliency + std_saliency,
        color=ATTRIBUTION_STD_FILL_COLOR,
        alpha=0.28
    )
    set_left_aligned_ylabel(ax1, "Mean saliency", -0.10)
    ax1.set_title(format_attribution_case_title(case_name), fontsize=ATTRIBUTION_TITLE_FONTSIZE)
    ax1.grid(axis="y", alpha=0.18)
    ax1.set_ylim(bottom=0.0)
    style_publication_axis(ax1)
    ax1.tick_params(axis="both", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    draw_segment_guides(ax1, segment_layout, add_label=True)

    saliency_img = mean_saliency[np.newaxis, :]
    im = ax2.imshow(
        saliency_img,
        aspect="auto",
        cmap=SALIENCY_CMAP,
        extent=[time.min(), time.max(), 0, 1]
    )
    ax2.set_yticks([])
    ax2.set_ylabel("")
    ax2.set_xlabel(get_saliency_x_label(model_name), fontsize=ATTRIBUTION_AXIS_LABEL_FONTSIZE)
    style_publication_axis(ax2)
    ax2.tick_params(axis="x", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    ax2.spines["left"].set_visible(False)
    draw_segment_guides(ax2, segment_layout)

    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(format_mean_attribution_colorbar_label("Saliency"), fontsize=ATTRIBUTION_COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=ATTRIBUTION_COLORBAR_TICK_FONTSIZE)

    fig.savefig(f"{output_base_path}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_mean_signal_and_saliency_case_plot(
    output_base_path,
    case_name,
    signal_series_list,
    saliency_series_list,
    signal_label="Signal",
    attribution_label="Saliency",
    model_name="",
    model_params=None,
    heatmap_vmin=None,
    heatmap_vmax=None
):
    if not signal_series_list or not saliency_series_list:
        return

    target_length = int(np.median([len(series) for series in saliency_series_list]))
    signal_matrix = np.vstack([resize_1d_series(series, target_length) for series in signal_series_list])
    saliency_matrix = np.vstack([resize_1d_series(series, target_length) for series in saliency_series_list])

    mean_signal = signal_matrix.mean(axis=0)
    std_signal = signal_matrix.std(axis=0)
    mean_saliency = saliency_matrix.mean(axis=0)
    std_saliency = saliency_matrix.std(axis=0)

    if heatmap_vmin is None:
        heatmap_vmin = float(np.min(mean_saliency))
    if heatmap_vmax is None:
        heatmap_vmax = float(np.max(mean_saliency))
    if heatmap_vmax <= heatmap_vmin:
        heatmap_vmax = heatmap_vmin + 1e-8

    time = np.arange(target_length)
    segment_layout = get_segment_layout(model_name, model_params, target_length)
    is_asymptotic_mean_occlusion = (
        str(signal_label).strip().lower() == "asymptotic model"
        and str(attribution_label).strip().lower() == "occlusion importance"
    )
    figure_height = 7.2 if is_asymptotic_mean_occlusion else 6.6
    save_padding = 0.24 if is_asymptotic_mean_occlusion else 0.08

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(9, figure_height),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 2.6, 0.5], "hspace": 0.06}
    )

    ax1.plot(time, mean_signal, color=PUBLICATION_LINE_COLOR, linewidth=1.6)
    ax1.fill_between(
        time,
        mean_signal - std_signal,
        mean_signal + std_signal,
        color=ATTRIBUTION_STD_FILL_COLOR,
        alpha=0.10
    )
    signal_label_y_coord = 0.58 if is_asymptotic_mean_occlusion else 0.5
    signal_label_x_coord = -0.13 if is_asymptotic_mean_occlusion else -0.12
    set_left_aligned_ylabel(
        ax1,
        format_mean_signal_label(signal_label),
        signal_label_x_coord,
        y_coord=signal_label_y_coord,
    )
    ax1.set_title(format_attribution_case_title(case_name), fontsize=ATTRIBUTION_TITLE_FONTSIZE)
    ax1.grid(axis="y", alpha=0.18)
    style_publication_axis(ax1)
    ax1.tick_params(axis="both", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    draw_segment_guides(ax1, segment_layout, add_label=True)

    ax2.plot(time, mean_saliency, color=PUBLICATION_LINE_COLOR, linewidth=1.6)
    ax2.fill_between(
        time,
        np.maximum(0.0, mean_saliency - std_saliency),
        mean_saliency + std_saliency,
        color=ATTRIBUTION_STD_FILL_COLOR,
        alpha=0.28
    )
    attribution_label_y_coord = 0.36 if is_asymptotic_mean_occlusion else 0.5
    attribution_label_x_coord = -0.13 if is_asymptotic_mean_occlusion else -0.12
    set_left_aligned_ylabel(
        ax2,
        format_mean_attribution_axis_label(attribution_label),
        attribution_label_x_coord,
        y_coord=attribution_label_y_coord,
    )
    ax2.grid(axis="y", alpha=0.18)
    ax2.set_ylim(bottom=0.0)
    style_publication_axis(ax2)
    ax2.tick_params(axis="both", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    draw_segment_guides(ax2, segment_layout)

    saliency_img = mean_saliency[np.newaxis, :]
    im = ax3.imshow(
        saliency_img,
        aspect="auto",
        cmap=SALIENCY_CMAP,
        extent=[time.min(), time.max(), 0, 1],
        vmin=heatmap_vmin,
        vmax=heatmap_vmax
    )
    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.set_xlabel(get_saliency_x_label(model_name), fontsize=ATTRIBUTION_AXIS_LABEL_FONTSIZE)
    style_publication_axis(ax3)
    ax3.tick_params(axis="x", labelsize=ATTRIBUTION_TICK_FONTSIZE)
    ax3.spines["left"].set_visible(False)
    draw_segment_guides(ax3, segment_layout)

    cbar = fig.colorbar(im, ax=[ax1, ax2, ax3], orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(format_mean_attribution_colorbar_label(attribution_label), fontsize=ATTRIBUTION_COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=ATTRIBUTION_COLORBAR_TICK_FONTSIZE)

    fig.savefig(f"{output_base_path}.png", dpi=300, bbox_inches="tight", pad_inches=save_padding)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=save_padding)
    plt.close(fig)


def select_saliency_indices_by_case(y_true, y_pred):
    """
    Select representative indices for different prediction cases.

    Cases:
    - correct_non_stress: true=0, pred=0
    - correct_stress: true=1, pred=1
    - wrong_non_stress: true=0, pred=1
    - wrong_stress: true=1, pred=0

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_pred: Predicted labels.
    :type y_pred: np.ndarray
    :return: Dictionary mapping case names to lists of indices.
    :rtype: dict
    """
    return {
        "correct_non_stress": np.where((y_true == 0) & (y_pred == 0))[0].tolist(),
        "correct_stress": np.where((y_true == 1) & (y_pred == 1))[0].tolist(),
        "wrong_non_stress": np.where((y_true == 0) & (y_pred == 1))[0].tolist(),
        "wrong_stress": np.where((y_true == 1) & (y_pred == 0))[0].tolist(),
    }


def filter_saliency_case_indices(config, case_indices):
    recovery_cfg = config.get("recovery", {})
    target_mode = recovery_cfg.get("saliency_target", "all")
    num_examples = int(recovery_cfg.get("saliency_num_examples_per_subject", 1))

    filtered_case_indices = dict(case_indices)
    if target_mode == "correct_only":
        filtered_case_indices["wrong_non_stress"] = []
        filtered_case_indices["wrong_stress"] = []
    elif target_mode == "wrong_only":
        filtered_case_indices["correct_non_stress"] = []
        filtered_case_indices["correct_stress"] = []

    for case_name, indices in filtered_case_indices.items():
        filtered_case_indices[case_name] = indices[:num_examples]

    return filtered_case_indices


def get_shared_heatmap_limits(case_series_map):
    """Return a common heatmap range across all case series for one attribution method."""
    values = []
    for series_list in case_series_map.values():
        for series in series_list:
            arr = np.asarray(series, dtype=float).reshape(-1)
            if arr.size:
                values.append(arr)
    if not values:
        return None, None
    concatenated = np.concatenate(values)
    vmin = float(np.min(concatenated))
    vmax = float(np.max(concatenated))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return vmin, vmax


def get_base_folder(config):
    return os.path.join(
        config['DL']['path_results'],
        config['timestamp'],
        config['DL']['model'],
        config['experiment_id']
    )


def is_fordigitstress_config(config):
    df_path = str(config.get("df_prep_path", "")).lower()
    results_path = str(config.get("DL", {}).get("path_results", "")).lower()
    return "fordigitstress" in df_path or "fordigitstress" in results_path


def get_dataset_label(config):
    return "fordigitstress" if is_fordigitstress_config(config) else "vr_goalkeeper"


def initialize_aggregation():
    return {
        "aggregated_cm": np.zeros((2, 2), dtype=int),
        "results": [],
        "all_y_true": [],
        "all_y_score": [],
        "aggregated_saliency_cases": {},
        "aggregated_signal_cases": {},
        "signal_label": None,
        "model_params": None,
        "attribution_method": None,
        "attribution_methods": []
    }


def accumulate_subject_result(aggregation, subject_result):
    aggregation["aggregated_cm"] += subject_result["cm"]
    aggregation["results"].append(subject_result)
    aggregation["all_y_true"].append(subject_result["y_true"])
    aggregation["all_y_score"].append(subject_result["y_score"])
    if aggregation["model_params"] is None:
        aggregation["model_params"] = subject_result.get("model_params")
    if aggregation["signal_label"] is None:
        aggregation["signal_label"] = subject_result.get("signal_label")
    if aggregation["attribution_method"] is None:
        aggregation["attribution_method"] = subject_result.get("attribution_method", "saliency")

    for method_name, method_result in subject_result["saliency_cases"].items():
        if method_name not in aggregation["aggregated_saliency_cases"]:
            aggregation["aggregated_saliency_cases"][method_name] = empty_attribution_case_map()
            aggregation["aggregated_signal_cases"][method_name] = empty_attribution_case_map()
        for case_name, saliency_list in method_result["cases"].items():
            aggregation["aggregated_saliency_cases"][method_name][case_name].extend(saliency_list)
        for case_name, signal_list in method_result["signals"].items():
            aggregation["aggregated_signal_cases"][method_name][case_name].extend(signal_list)


def build_results_dataframe(results):
    return pd.DataFrame([
        {
            "test_id": result["test_id"],
            "macro_f1": result["f1"],
            "precision_macro": result["precision"],
            "recall_macro": result["recall"],
            "roc_auc": result["auc"]
        }
        for result in results
    ])


def style_publication_axis(ax, categorical_x=False, categorical_y=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=22)

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


def get_text_color_for_background(rgb_color):
    r, g, b = rgb_color
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#F8FAFC" if luminance < 0.55 else "#1F2937"


def format_percent_value(value):
    if plt.rcParams.get("text.usetex", False):
        return f"{value:.1f}\\%"
    return f"{value:.1f}%"


def save_single_attribution_payload(
    output_base_path,
    signal,
    saliency,
    title,
    signal_label,
    attribution_label,
    model_name,
    model_params,
    heatmap_vmin,
    heatmap_vmax
):
    np.savez_compressed(
        f"{output_base_path}.npz",
        signal=np.asarray(signal, dtype=float),
        saliency=np.asarray(saliency, dtype=float),
        title=np.asarray(title),
        signal_label=np.asarray(signal_label),
        attribution_label=np.asarray(attribution_label),
        model_name=np.asarray(model_name),
        model_params=np.asarray(json.dumps(model_params if model_params is not None else {})),
        heatmap_vmin=np.asarray(heatmap_vmin if heatmap_vmin is not None else np.nan, dtype=float),
        heatmap_vmax=np.asarray(heatmap_vmax if heatmap_vmax is not None else np.nan, dtype=float),
    )


def save_mean_attribution_payload(
    output_base_path,
    case_name,
    signal_series_list,
    saliency_series_list,
    signal_label,
    attribution_label,
    model_name,
    model_params,
    heatmap_vmin,
    heatmap_vmax
):
    target_length = int(np.median([len(series) for series in saliency_series_list]))
    signal_matrix = np.vstack([resize_1d_series(series, target_length) for series in signal_series_list])
    saliency_matrix = np.vstack([resize_1d_series(series, target_length) for series in saliency_series_list])
    np.savez_compressed(
        f"{output_base_path}.npz",
        case_name=np.asarray(case_name),
        signal_matrix=signal_matrix.astype(float),
        saliency_matrix=saliency_matrix.astype(float),
        signal_label=np.asarray(signal_label),
        attribution_label=np.asarray(attribution_label),
        model_name=np.asarray(model_name),
        model_params=np.asarray(json.dumps(model_params if model_params is not None else {})),
        heatmap_vmin=np.asarray(heatmap_vmin if heatmap_vmin is not None else np.nan, dtype=float),
        heatmap_vmax=np.asarray(heatmap_vmax if heatmap_vmax is not None else np.nan, dtype=float),
    )


def save_curve_plot(
    output_base_path,
    x_values,
    y_values,
    label,
    x_label,
    y_label,
    title,
    legend_loc,
    chance_line=False,
    legend_bbox_to_anchor=None,
    curve_color=None
):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(
        x_values,
        y_values,
        label=label,
        color=curve_color or PUBLICATION_LINE_COLOR,
        linewidth=1.3
    )

    if chance_line:
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)

    ax.set_xlabel(x_label, fontsize=26)
    ax.set_ylabel(y_label, fontsize=26, labelpad=4)
    ax.set_title(title, fontsize=26)
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        frameon=False,
        fontsize=22
    )
    ax.margins(x=0.02, y=0.05)
    style_publication_axis(ax)

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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
            text_color = get_text_color_for_background(cell_colors[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                annotation,
                ha="center",
                va="center",
                fontsize=CONFUSION_CELL_LABEL_FONTSIZE,
                color=text_color
            )

    fig.savefig(f"{output_base_path}.png", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base_path}.pdf", dpi=1000, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_aggregated_outputs(recovery_base, aggregation, config):
    np.savetxt(
        os.path.join(recovery_base, "confusion_matrix_outer_aggregated.csv"),
        aggregation["aggregated_cm"],
        delimiter=",",
        fmt="%d"
    )
    dataset_label = get_dataset_label(config)
    model_label = config["DL"]["model"].lower()
    experiment_label = f"exp{config['experiment_id']}"
    save_confusion_matrix_heatmap(
        output_base_path=os.path.join(
            recovery_base,
            f"{dataset_label}_{experiment_label}_{model_label}_confusion_matrix_outer_aggregated"
        ),
        cm=aggregation["aggregated_cm"],
        title=build_confusion_matrix_title(config),
        include_row_percent=True
    )

    df_results = build_results_dataframe(aggregation["results"])
    all_y_true = np.concatenate(aggregation["all_y_true"])
    all_y_score = np.concatenate(aggregation["all_y_score"])

    np.save(os.path.join(recovery_base, "y_true_outer_aggregated.npy"), all_y_true)
    np.save(os.path.join(recovery_base, "y_score_outer_aggregated.npy"), all_y_score)

    fpr, tpr, roc_thresholds = roc_curve(all_y_true, all_y_score)
    roc_auc = roc_auc_score(all_y_true, all_y_score)
    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": roc_thresholds
    }).to_csv(os.path.join(recovery_base, "roc_curve_outer_aggregated.csv"), index=False)
    save_curve_plot(
        output_base_path=os.path.join(recovery_base, "roc_curve_outer_aggregated"),
        x_values=fpr,
        y_values=tpr,
        label=f"AUC = {roc_auc:.3f}",
        x_label="False positive rate",
        y_label="True positive rate",
        title="Aggregated outer-fold ROC curve",
        legend_loc="lower right",
        chance_line=True,
        curve_color=YELLOW_CURVE_COLOR
    )

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(all_y_true, all_y_score)
    ap = average_precision_score(all_y_true, all_y_score)
    pd.DataFrame({
        "precision": pr_precision[:-1],
        "recall": pr_recall[:-1],
        "threshold": pr_thresholds
    }).to_csv(os.path.join(recovery_base, "pr_curve_outer_aggregated.csv"), index=False)
    save_curve_plot(
        output_base_path=os.path.join(recovery_base, "pr_curve_outer_aggregated"),
        x_values=pr_recall,
        y_values=pr_precision,
        label=f"AP = {ap:.3f}",
        x_label="Recall",
        y_label="Precision",
        title="Aggregated outer-fold precision-recall curve",
        legend_loc="upper right",
        legend_bbox_to_anchor=(0.98, 0.98),
        curve_color=TRANSLUCENT_RED_CURVE_COLOR
    )

    with open(os.path.join(recovery_base, "aggregated_curve_metrics.txt"), "w") as f:
        f.write(f"roc_auc_outer_aggregated: {roc_auc:.6f}\n")
        f.write(f"average_precision_outer_aggregated: {ap:.6f}\n")

    results_path = os.path.join(recovery_base, "outer_metrics_recovered.csv")
    df_results.to_csv(results_path, index=False)
    print(f"Saved results {results_path}")

    return aggregation["aggregated_cm"]


def save_aggregated_saliency_outputs(recovery_base, aggregation, config):
    model_name = config.get("DL", {}).get("model", "")
    model_params = aggregation.get("model_params")
    signal_label = aggregation.get("signal_label") or get_input_signal_display_name(config)
    for attribution_method, case_map in aggregation["aggregated_saliency_cases"].items():
        attribution_label = "Occlusion importance" if attribution_method == "occlusion" else "Saliency"
        saliency_root = os.path.join(recovery_base, attribution_method)
        os.makedirs(saliency_root, exist_ok=True)
        heatmap_vmin, heatmap_vmax = get_shared_heatmap_limits(case_map)

        summary_rows = []
        for case_name, saliency_list in case_map.items():
            signal_list = aggregation["aggregated_signal_cases"].get(attribution_method, {}).get(case_name, [])
            saliency_series_list = [reduce_to_time_series(saliency) for saliency in saliency_list]
            signal_series_list = [reduce_to_time_series(signal) for signal in signal_list]
            summary_rows.append({
                "case_name": case_name,
                "n_examples": len(saliency_series_list),
                "attribution_method": attribution_method
            })

            if not saliency_series_list or not signal_series_list:
                continue

            save_mean_signal_and_saliency_case_plot(
                output_base_path=os.path.join(saliency_root, f"{case_name}_mean_{attribution_method}"),
                case_name=case_name,
                signal_series_list=signal_series_list,
                saliency_series_list=saliency_series_list,
                signal_label=signal_label,
                attribution_label=attribution_label,
                model_name=model_name,
                model_params=model_params,
                heatmap_vmin=heatmap_vmin,
                heatmap_vmax=heatmap_vmax
            )
            save_mean_attribution_payload(
                output_base_path=os.path.join(saliency_root, f"{case_name}_mean_{attribution_method}"),
                case_name=case_name,
                signal_series_list=signal_series_list,
                saliency_series_list=saliency_series_list,
                signal_label=signal_label,
                attribution_label=attribution_label,
                model_name=model_name,
                model_params=model_params,
                heatmap_vmin=heatmap_vmin,
                heatmap_vmax=heatmap_vmax
            )

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(saliency_root, f"{attribution_method}_case_summary.csv"),
            index=False
        )


def main():
    configure_publication_plot_style()
    config = load_config()
    set_DL_config(config)
    aggregation = initialize_aggregation()

    for test_id in config['valid_IDs']:
        print(f"Recovering outer predictions for test_ID {test_id}")
        subject_result = predict_for_subject(config, test_id)
        accumulate_subject_result(aggregation, subject_result)

    recovery_base = os.path.join(get_base_folder(config), "recovery")
    os.makedirs(recovery_base, exist_ok=True)

    aggregated_cm = save_aggregated_outputs(recovery_base, aggregation, config)
    save_aggregated_saliency_outputs(recovery_base, aggregation, config)

    print("Aggregated outer confusion matrix:")
    print(aggregated_cm)


if __name__ == "__main__":
    main()
