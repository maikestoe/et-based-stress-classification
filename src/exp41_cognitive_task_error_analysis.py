"""
Experiment 41: signal-shape and cognitive-task error analysis.

This script joins the saved outer-fold predictions of experiment 41 with:
- the prepared DL dataframe windows
- the original VR raw logs

It focuses on four questions:
1. Do misclassified samples differ from the class mean signal shape?
2. How far before/after the 5 s model window did the cognitive task start/end?
3. Are classification errors related to cognitive-task errors (e.g. red stressor)?
4. Are classification errors related to task difficulty level?
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, spearmanr

import segment_vr_goalkeeper

try:
    from statannotations.Annotator import Annotator
except ImportError:
    Annotator = None


# Compatibility fixes for older pandas pickles
sys.modules["pandas.core.indexes.numeric"] = pd.core.indexes.base
if not hasattr(pd, "Int64Index"):
    pd.Int64Index = pd.Index
if not hasattr(pd.core.indexes.base, "Int64Index"):
    pd.core.indexes.base.Int64Index = pd.Index


STYLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "plot_style2.txt"))


def configure_plot_style() -> None:
    plt.style.use(STYLE_PATH)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    pdflatex_path = shutil.which("pdflatex")
    kpsewhich_path = shutil.which("kpsewhich")

    latex_ready = bool(pdflatex_path)
    if latex_ready and kpsewhich_path:
        font_check = subprocess.run(
            [kpsewhich_path, "cmr12.tfm"],
            capture_output=True,
            text=True,
            check=False,
        )
        latex_ready = bool(font_check.stdout.strip())

    if latex_ready:
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["text.usetex"] = False
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "cmr10", "DejaVu Serif"]


configure_plot_style()

LINE_TRUE = "#3E5F8A"
LINE_ALT = "#C44F5E"
FILL_TRUE = "#C7D2E0"
FILL_ALT = "#E7C4CB"
BOX_BLUE = "#4C72B0"
BOX_BLUE_LIGHT = "#D7E3F4"

TASK_FILE_BY_LABEL = {
    "nostress": "noStress",
    "stress": "stress",
}

DISPLAY_LABEL_BY_CLASS = {
    0: "No stress",
    1: "Stress",
}

SIGNAL_COL = "asymptotic_model"


@dataclass
class TaskWindowMetrics:
    task_block_found: bool
    task_level_mode: float
    task_score_first: float
    task_score_last: float
    task_score_gain: float
    task_score_non_increasing: bool
    task_response_time_last: float
    task_started_before_window_s: float
    task_ended_before_window_s: float
    task_block_duration_s: float
    red_fraction: float
    green_fraction: float
    score_display_fraction: float
    none_fraction: float
    dominant_stressor: str
    task_wrong_event_count: int
    task_correct_event_count: int
    any_wrong_event: bool
    any_correct_event: bool
    cognitive_task_incorrect: bool
    cognitive_task_correct: bool
    cognitive_task_outcome: str
    shot_direction: str
    animation_end: str
    window_start_time: float
    window_end_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run signal-shape and cognitive-task error analysis for experiment 41."
    )
    parser.add_argument(
        "--config",
        default="configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json",
        help="Experiment config used to locate prepared data and results."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <experiment>/recovery/exp41_cognitive_task_error_analysis."
    )
    return parser.parse_args()


def load_json_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def resolve_path(base_dir: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.abspath(os.path.join(base_dir, maybe_relative))


def resolve_local_project_path(base_dir: str, path_with_placeholders: str) -> str:
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    normalized = path_with_placeholders

    replacements = {
        "$TMPDIR/data": os.path.join(project_root, "data"),
        "$TMPDIR/results": os.path.join(project_root, "results"),
        "$TMPBASE/data": os.path.join(project_root, "data"),
        "$TMPBASE/results": os.path.join(project_root, "results"),
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    expanded = os.path.expandvars(normalized)

    # Safety fallback for locally expanded temp paths such as /var/folders/.../T/data.
    temp_markers = [
        os.path.join("data", "vr_goalkeeper", "dataframes", "DL_out.pkl"),
        os.path.join("results", "vr_goalkeeper", "DL"),
    ]
    if not os.path.exists(expanded):
        for marker in temp_markers:
            if marker in expanded:
                suffix = expanded.split(marker, 1)[1]
                expanded = os.path.join(project_root, marker, suffix.lstrip(os.sep))
                break

    return resolve_path(base_dir, expanded)


def get_local_project_root(base_dir: str) -> str:
    return os.path.abspath(os.path.join(base_dir, ".."))


def get_experiment_root(config: dict, base_dir: str) -> str:
    project_root = get_local_project_root(base_dir)
    preferred_local_root = os.path.join(
        project_root,
        "results",
        "vr_goalkeeper",
        "DL",
        str(config["timestamp"]),
        str(config["DL"]["model"]),
        str(config["experiment_id"]),
    )
    if os.path.exists(preferred_local_root):
        return preferred_local_root

    return resolve_local_project_path(
        base_dir,
        os.path.join(
            config["DL"]["path_results"],
            config["timestamp"],
            config["DL"]["model"],
            str(config["experiment_id"]),
        ),
    )


def load_prepared_dataframe(config: dict, base_dir: str) -> pd.DataFrame:
    project_root = get_local_project_root(base_dir)
    preferred_local_df = os.path.join(
        project_root,
        "data",
        "vr_goalkeeper",
        "dataframes",
        "DL_out.pkl",
    )
    if os.path.exists(preferred_local_df):
        df_path = preferred_local_df
    else:
        df_path = resolve_local_project_path(base_dir, config["df_prep_path"])
    with open(df_path, "rb") as fh:
        df = pickle.load(fh)
    return df


def label_to_task_file(lab_str: str) -> str:
    key = str(lab_str).strip().lower()
    if key not in TASK_FILE_BY_LABEL:
        raise KeyError(f"Unsupported lab_str value: {lab_str}")
    return TASK_FILE_BY_LABEL[key]


def clean_string_series(series: pd.Series, fill_value: str = "null") -> pd.Series:
    return (
        series.astype("string")
        .fillna(fill_value)
        .str.strip()
        .replace({"": fill_value, "<NA>": fill_value})
    )


def style_axis(ax, integer_x: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def style_categorical_ticks(ax, rotation: float = 0.0, ha: str = "center") -> None:
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=rotation, ha=ha)
    ax.tick_params(axis="x", pad=6)


def level_label(level: float) -> str:
    return str(int(level)) if float(level).is_integer() else f"{level:g}"


def p_value_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def mann_whitney_p_value(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def annotate_p_value(ax, x1: float, x2: float, y: float, h: float, p_value: float) -> bool:
    if not np.isfinite(p_value) or p_value >= 0.05:
        return False
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#444444", linewidth=1.0)
    ax.text(
        (x1 + x2) / 2.0,
        y + h * 1.15,
        p_value_to_stars(p_value),
        ha="center",
        va="bottom",
        fontsize=12,
        color="#333333",
    )
    return True


def draw_blue_boxplot(ax, data: pd.DataFrame, x: str, y: str, order: List[str], width: float = 0.6) -> None:
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        order=order,
        ax=ax,
        width=width,
        fliersize=2.5,
        boxprops={"facecolor": BOX_BLUE_LIGHT, "edgecolor": BOX_BLUE, "linewidth": 1.15},
        whiskerprops={"color": BOX_BLUE, "linewidth": 1.0},
        capprops={"color": BOX_BLUE, "linewidth": 1.0},
        medianprops={"color": BOX_BLUE, "linewidth": 1.35},
    )


def draw_blue_stripplot(ax, data: pd.DataFrame, x: str, y: str, order: List[str], size: float = 2.6, jitter: float = 0.18) -> None:
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        order=order,
        ax=ax,
        color=BOX_BLUE,
        alpha=0.35,
        size=size,
        jitter=jitter,
    )


def apply_statannotations(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    pairs: List[Tuple[str, str]],
    order: List[str],
) -> bool:
    if Annotator is None or data.empty or not pairs:
        return False
    try:
        annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="outside",
            comparisons_correction=None,
            hide_non_significant=True,
            verbose=0,
        )
        annotator.apply_and_annotate()
        return True
    except Exception:
        return False


def reduce_signal(arr: object) -> np.ndarray:
    values = np.asarray(arr, dtype=float).squeeze()
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Signals must have same length.")
    if np.allclose(np.nanstd(a), 0.0) or np.allclose(np.nanstd(b), 0.0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.dot(a, b) / denom)


def normalized_rmse(a: np.ndarray, b: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((a - b) ** 2))
    scale = np.nanstd(b)
    if np.isclose(scale, 0.0):
        return np.nan
    return float(rmse / scale)


def contiguous_true_block(bool_values: np.ndarray, anchor_indices: np.ndarray) -> Tuple[int, int]:
    start_idx = int(anchor_indices[0])
    end_idx = int(anchor_indices[-1])
    while start_idx > 0 and bool_values[start_idx - 1]:
        start_idx -= 1
    while end_idx + 1 < len(bool_values) and bool_values[end_idx + 1]:
        end_idx += 1
    return start_idx, end_idx


def all_true_blocks(bool_values: np.ndarray) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    block_start = None
    for idx, value in enumerate(bool_values):
        if value and block_start is None:
            block_start = idx
        elif not value and block_start is not None:
            blocks.append((block_start, idx - 1))
            block_start = None
    if block_start is not None:
        blocks.append((block_start, len(bool_values) - 1))
    return blocks


def mode_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    mode_values = values.mode()
    if mode_values.empty:
        return np.nan
    return float(mode_values.iloc[0])


def last_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def first_numeric(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def compute_task_window_metrics(raw_df: pd.DataFrame, shot_idx: int, window_duration_s: float = 5.0) -> TaskWindowMetrics:
    start_shots, end_shots = segment_vr_goalkeeper.find_shots(raw_df)
    if shot_idx >= len(end_shots):
        raise IndexError(f"shot {shot_idx} out of bounds for subject log with {len(end_shots)} shots")

    end_idx = int(end_shots[shot_idx])
    end_time = float(raw_df["normal time"].iloc[end_idx])
    start_time = end_time - window_duration_s

    time_values = pd.to_numeric(raw_df["normal time"], errors="coerce")
    window_mask = (time_values >= start_time) & (time_values <= end_time)
    window_df = raw_df.loc[window_mask].copy()
    if window_df.empty:
        raise ValueError(f"Empty raw window for shot {shot_idx}")

    task_bool = raw_df["Task"].fillna(False).astype(bool).to_numpy()
    task_started_before_window_s = np.nan
    task_ended_before_window_s = np.nan
    task_block_duration_s = np.nan
    task_block_found = False

    task_blocks = all_true_blocks(task_bool)
    preceding_blocks = []
    for block_start_idx, block_end_idx in task_blocks:
        block_end_time = float(time_values.iloc[block_end_idx])
        if block_end_time <= start_time:
            preceding_blocks.append((block_start_idx, block_end_idx))

    block_df = window_df.iloc[0:0].copy()
    interval_df = window_df.copy()
    if preceding_blocks:
        block_start_idx, block_end_idx = preceding_blocks[-1]
        block_start_time = float(time_values.iloc[block_start_idx])
        block_end_time = float(time_values.iloc[block_end_idx])
        task_started_before_window_s = max(0.0, start_time - block_start_time)
        task_ended_before_window_s = max(0.0, start_time - block_end_time)
        task_block_duration_s = max(0.0, block_end_time - block_start_time)
        task_block_found = True
        block_df = raw_df.iloc[block_start_idx : block_end_idx + 1].copy()
        interval_df = raw_df.iloc[block_start_idx : end_idx + 1].copy()

    stressor_source_df = interval_df if task_block_found else window_df
    stressor_series = clean_string_series(stressor_source_df["tmp stressor"], fill_value="null")
    stressor_counts = stressor_series.value_counts(normalize=True)

    red_fraction = float(stressor_counts.get("Red", 0.0))
    green_fraction = float(stressor_counts.get("Green", 0.0))
    score_display_fraction = float(stressor_counts.get("Score display", 0.0))
    if np.isclose(score_display_fraction, 0.0):
        score_display_fraction = float(stressor_counts.get("score display", 0.0))
    none_fraction = float(stressor_counts.get("Non", 0.0))
    dominant_stressor = stressor_series.mode().iloc[0] if not stressor_series.mode().empty else "null"
    task_score_first = first_numeric(block_df["Task-Score"]) if task_block_found else np.nan
    task_score_last = last_numeric(block_df["Task-Score"]) if task_block_found else np.nan
    task_score_gain = task_score_last - task_score_first if task_block_found and np.isfinite(task_score_first) and np.isfinite(task_score_last) else np.nan
    task_score_non_increasing = bool(np.isfinite(task_score_gain) and task_score_gain <= 0.0)
    task_wrong_event_count = int((stressor_series == "Red").sum())
    task_correct_event_count = int((stressor_series == "Green").sum())
    any_wrong_event = bool(task_wrong_event_count > 0)
    any_correct_event = bool(task_correct_event_count > 0)
    cognitive_task_incorrect = any_wrong_event
    cognitive_task_correct = any_correct_event
    if cognitive_task_incorrect and cognitive_task_correct:
        cognitive_task_outcome = "mixed"
    elif cognitive_task_incorrect:
        cognitive_task_outcome = "incorrect"
    elif cognitive_task_correct:
        cognitive_task_outcome = "correct"
    else:
        cognitive_task_outcome = "unresolved"

    shot_direction_series = clean_string_series(window_df["shot direction"], fill_value="null")
    animation_series = clean_string_series(window_df["animation"], fill_value="null")

    return TaskWindowMetrics(
        task_block_found=task_block_found,
        task_level_mode=mode_numeric(block_df["Task-Level"]) if task_block_found else np.nan,
        task_score_first=task_score_first,
        task_score_last=task_score_last,
        task_score_gain=task_score_gain,
        task_score_non_increasing=task_score_non_increasing,
        task_response_time_last=last_numeric(block_df["Task-Response time"]) if task_block_found else np.nan,
        task_started_before_window_s=task_started_before_window_s,
        task_ended_before_window_s=task_ended_before_window_s,
        task_block_duration_s=task_block_duration_s,
        red_fraction=red_fraction,
        green_fraction=green_fraction,
        score_display_fraction=score_display_fraction,
        none_fraction=none_fraction,
        dominant_stressor=str(dominant_stressor),
        task_wrong_event_count=task_wrong_event_count,
        task_correct_event_count=task_correct_event_count,
        any_wrong_event=any_wrong_event,
        any_correct_event=any_correct_event,
        cognitive_task_incorrect=cognitive_task_incorrect,
        cognitive_task_correct=cognitive_task_correct,
        cognitive_task_outcome=cognitive_task_outcome,
        shot_direction=str(shot_direction_series.mode().iloc[0] if not shot_direction_series.mode().empty else "null"),
        animation_end=str(animation_series.iloc[-1]),
        window_start_time=float(start_time),
        window_end_time=float(end_time),
    )


def load_raw_phase_df(raw_root: str, subject_id: int, lab_str: str) -> pd.DataFrame:
    task_file = label_to_task_file(lab_str)
    raw_path = os.path.join(raw_root, f"LogID_{subject_id}_{task_file}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw log not found: {raw_path}")

    raw_df = pd.read_csv(raw_path, sep=";", skip_blank_lines=True, low_memory=False)
    raw_df = raw_df.dropna(how="all").convert_dtypes()
    return raw_df


def load_prediction_table(df_prep: pd.DataFrame, experiment_root: str, valid_ids: Iterable[int]) -> pd.DataFrame:
    all_subject_rows: List[pd.DataFrame] = []

    df_prep = df_prep.copy()
    df_prep["lab_num"] = df_prep["lab_num"].astype(int)

    for test_id in valid_ids:
        subject_df = df_prep[df_prep["ID"] == test_id].copy()
        subject_df = subject_df.reset_index().rename(columns={"index": "df_index"})
        subject_df["sample_idx"] = np.arange(len(subject_df))

        recovery_dir = os.path.join(experiment_root, f"test_ID_{test_id}", "recovery")
        y_true = np.load(os.path.join(recovery_dir, "y_true_outer.npy"))
        y_pred = np.load(os.path.join(recovery_dir, "y_pred_outer.npy"))
        y_score = np.load(os.path.join(recovery_dir, "y_score_outer.npy"))

        if len(subject_df) != len(y_true):
            raise ValueError(
                f"Length mismatch for test_ID {test_id}: df rows={len(subject_df)} vs predictions={len(y_true)}"
            )
        if not np.array_equal(subject_df["lab_num"].to_numpy(dtype=int), y_true.astype(int)):
            raise ValueError(f"Label mismatch for test_ID {test_id}; fold order no longer matches DL_out order.")

        subject_df["y_true"] = y_true.astype(int)
        subject_df["y_pred"] = y_pred.astype(int)
        subject_df["y_score"] = y_score.astype(float)
        subject_df["is_correct"] = subject_df["y_true"] == subject_df["y_pred"]
        subject_df["predicted_label"] = subject_df["y_pred"].map(DISPLAY_LABEL_BY_CLASS)
        subject_df["true_label"] = subject_df["y_true"].map(DISPLAY_LABEL_BY_CLASS)
        subject_df["error_type"] = np.where(
            subject_df["is_correct"],
            "correct",
            np.where(subject_df["y_true"] == 1, "stress_to_nostress", "nostress_to_stress"),
        )

        all_subject_rows.append(subject_df)

    return pd.concat(all_subject_rows, ignore_index=True)


def compute_class_mean_signals(df: pd.DataFrame, signal_col: str) -> Dict[int, np.ndarray]:
    means = {}
    for class_id in [0, 1]:
        class_rows = df[(df["y_true"] == class_id) & (df["is_correct"])]
        if class_rows.empty:
            raise ValueError(f"No correctly classified rows available for class {class_id}")
        signal_matrix = np.vstack([reduce_signal(values) for values in class_rows[signal_col]])
        means[class_id] = signal_matrix.mean(axis=0)
    return means


def add_signal_similarity_metrics(df: pd.DataFrame, class_means: Dict[int, np.ndarray], signal_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        signal = reduce_signal(row[signal_col])
        true_mean = class_means[int(row["y_true"])]
        pred_mean = class_means[int(row["y_pred"])]
        rows.append(
            {
                "corr_to_true_mean": pearson_safe(signal, true_mean),
                "corr_to_pred_mean": pearson_safe(signal, pred_mean),
                "cosine_to_true_mean": cosine_similarity(signal, true_mean),
                "cosine_to_pred_mean": cosine_similarity(signal, pred_mean),
                "nrmse_to_true_mean": normalized_rmse(signal, true_mean),
                "nrmse_to_pred_mean": normalized_rmse(signal, pred_mean),
                "corr_pred_minus_true": pearson_safe(signal, pred_mean) - pearson_safe(signal, true_mean),
                "nrmse_true_minus_pred": normalized_rmse(signal, true_mean) - normalized_rmse(signal, pred_mean),
                "early_peak_0_75": float(np.nanmax(signal[:75])),
                "early_peak_0_100": float(np.nanmax(signal[:100])),
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def enrich_with_raw_task_metadata(df: pd.DataFrame, raw_root: str) -> pd.DataFrame:
    enriched_rows = []
    raw_cache: Dict[Tuple[int, str], pd.DataFrame] = {}

    for _, row in df.iterrows():
        cache_key = (int(row["ID"]), str(row["lab_str"]))
        if cache_key not in raw_cache:
            raw_cache[cache_key] = load_raw_phase_df(raw_root, int(row["ID"]), str(row["lab_str"]))

        metrics = compute_task_window_metrics(raw_cache[cache_key], int(row["shot"]))
        payload = row.to_dict()
        payload.update(metrics.__dict__)
        enriched_rows.append(payload)

    return pd.DataFrame(enriched_rows)


def plot_class_means(df: pd.DataFrame, class_means: Dict[int, np.ndarray], output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True, sharey=True)
    for class_id, ax in zip([0, 1], axes):
        class_rows = df[(df["y_true"] == class_id) & (df["is_correct"])]
        signal_matrix = np.vstack([reduce_signal(values) for values in class_rows[SIGNAL_COL]])
        mean_signal = class_means[class_id]
        std_signal = signal_matrix.std(axis=0)
        x = np.arange(len(mean_signal))

        ax.plot(x, mean_signal, color=LINE_TRUE, linewidth=1.8)
        ax.fill_between(x, mean_signal - std_signal, mean_signal + std_signal, color=FILL_TRUE, alpha=0.45)
        ax.set_title(f"Correct {DISPLAY_LABEL_BY_CLASS[class_id]}", fontsize=18)
        ax.set_xlabel("Sample index", fontsize=16)
        style_axis(ax, integer_x=False)
        ax.grid(axis="y", alpha=0.18)

    axes[0].set_ylabel("Mean asymptotic model", fontsize=16)
    fig.savefig(os.path.join(output_dir, "exp41_correct_class_mean_signals.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_correct_class_mean_signals.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_distributions(df: pd.DataFrame, output_dir: str) -> None:
    plot_df = df.copy()
    plot_df["correctness_label"] = np.where(plot_df["is_correct"], "Correct", "Misclassified")
    plot_df["group_label"] = plot_df.apply(
        lambda row: f"{DISPLAY_LABEL_BY_CLASS[int(row['y_true'])]}\n{row['correctness_label']}",
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["corr_to_true_mean", "nrmse_to_true_mean"],
        ["Correlation to true-class mean", "Normalized RMSE to true-class mean"],
    ):
        metric_df = plot_df[["group_label", "y_true", "correctness_label", metric]].dropna().copy()
        order = [
            f"{DISPLAY_LABEL_BY_CLASS[class_id]}\n{correctness}"
            for class_id in [0, 1]
            for correctness in ["Correct", "Misclassified"]
        ]
        order = [label for label in order if label in metric_df["group_label"].unique()]
        if metric_df.empty or not order:
            continue

        draw_blue_boxplot(ax, metric_df, "group_label", metric, order, width=0.6)
        draw_blue_stripplot(ax, metric_df, "group_label", metric, order, size=2.6, jitter=0.18)

        pairs = []
        for class_id in [0, 1]:
            left = f"{DISPLAY_LABEL_BY_CLASS[class_id]}\nCorrect"
            right = f"{DISPLAY_LABEL_BY_CLASS[class_id]}\nMisclassified"
            if left in order and right in order:
                pairs.append((left, right))
        annotated = apply_statannotations(ax, metric_df, "group_label", metric, pairs, order)
        if not annotated and pairs:
            ymax = float(metric_df[metric].max())
            ymin = float(metric_df[metric].min())
            yrange = max(1e-6, ymax - ymin)
            top = ymax + 0.18 * yrange
            step = 0.10 * yrange
            n_annotations = 0
            for pair_idx, pair in enumerate(pairs):
                left_vals = metric_df.loc[metric_df["group_label"] == pair[0], metric].to_numpy()
                right_vals = metric_df.loc[metric_df["group_label"] == pair[1], metric].to_numpy()
                p_value = mann_whitney_p_value(left_vals, right_vals)
                x1 = order.index(pair[0])
                x2 = order.index(pair[1])
                added = annotate_p_value(ax, x1, x2, top + n_annotations * step, 0.03 * yrange, p_value)
                if added:
                    n_annotations += 1
            if n_annotations > 0:
                ax.set_ylim(ymin - 0.05 * yrange, top + n_annotations * step + 0.12 * yrange)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.18)
        style_axis(ax)
        style_categorical_ticks(ax, rotation=12, ha="right")

    fig.savefig(os.path.join(output_dir, "exp41_signal_similarity_error_distributions.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_signal_similarity_error_distributions.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_task_timing(df: pd.DataFrame, output_dir: str) -> None:
    stress_df = df[df["y_true"] == 1].copy()
    if stress_df.empty:
        return

    stress_df["correctness_label"] = np.where(stress_df["is_correct"], "Correct stress", "Misclassified stress")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["task_started_before_window_s", "task_ended_before_window_s"],
        ["Task started before 5 s window", "Task ended before 5 s window"],
    ):
        metric_df = stress_df[["correctness_label", metric]].dropna().copy()
        order = [label for label in ["Correct stress", "Misclassified stress"] if label in metric_df["correctness_label"].unique()]
        if metric_df.empty or not order:
            continue

        draw_blue_boxplot(ax, metric_df, "correctness_label", metric, order, width=0.58)
        draw_blue_stripplot(ax, metric_df, "correctness_label", metric, order, size=2.8, jitter=0.15)

        if len(order) == 2:
            pair = [(order[0], order[1])]
            annotated = apply_statannotations(ax, metric_df, "correctness_label", metric, pair, order)
            if not annotated:
                left_vals = metric_df.loc[metric_df["correctness_label"] == order[0], metric].to_numpy()
                right_vals = metric_df.loc[metric_df["correctness_label"] == order[1], metric].to_numpy()
                p_value = mann_whitney_p_value(left_vals, right_vals)
                ymax = float(metric_df[metric].max())
                ymin = float(metric_df[metric].min())
                yrange = max(1e-6, ymax - ymin)
                y = ymax + 0.12 * yrange
                added = annotate_p_value(ax, 0, 1, y, 0.04 * yrange, p_value)
                if added:
                    ax.set_ylim(ymin - 0.05 * yrange, y + 0.18 * yrange)

        ax.set_title(title, fontsize=16)
        ax.set_ylabel("Seconds", fontsize=15)
        ax.set_xlabel("")
        style_categorical_ticks(ax, rotation=0, ha="center")
        ax.grid(axis="y", alpha=0.18)
        style_axis(ax)
    fig.savefig(os.path.join(output_dir, "exp41_task_timing_vs_classification_errors.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_task_timing_vs_classification_errors.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_task_level_correctness(df: pd.DataFrame, output_dir: str) -> None:
    task_df = df[df["task_block_found"]].copy()
    task_df["task_level_mode"] = pd.to_numeric(task_df["task_level_mode"], errors="coerce")
    task_df = task_df.dropna(subset=["task_level_mode"])
    if task_df.empty:
        return

    level_summary = (
        task_df.groupby(["task_level_mode", "is_correct"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["task_level_mode", "is_correct"])
    )
    pivot_counts = (
        level_summary.pivot(index="task_level_mode", columns="is_correct", values="n_samples")
        .fillna(0.0)
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.4), constrained_layout=True)

    x = np.arange(len(pivot_counts.index))
    correct_counts = pivot_counts.get(True, pd.Series(0.0, index=pivot_counts.index)).to_numpy(dtype=float)
    wrong_counts = pivot_counts.get(False, pd.Series(0.0, index=pivot_counts.index)).to_numpy(dtype=float)
    axes[0].bar(x, correct_counts, color=LINE_TRUE, alpha=0.85, label="Correct")
    axes[0].bar(x, wrong_counts, bottom=correct_counts, color=LINE_ALT, alpha=0.85, label="Misclassified")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(int(v)) if float(v).is_integer() else f"{v:g}" for v in pivot_counts.index])
    axes[0].set_xlabel("Task level", fontsize=15)
    axes[0].set_ylabel("Number of samples", fontsize=15)
    axes[0].set_title("Task level by correctness", fontsize=16)
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].grid(axis="y", alpha=0.18)
    style_axis(axes[0], integer_x=False)

    boxplot_rows = []
    pair_candidates = []
    for level in sorted(task_df["task_level_mode"].dropna().unique()):
        level_df = task_df[task_df["task_level_mode"] == level]
        level_group_labels = []
        for is_correct, label in [(True, "Correct"), (False, "Misclassified")]:
            values = level_df[level_df["is_correct"] == is_correct]["corr_to_true_mean"].dropna().to_numpy()
            if values.size == 0:
                continue
            short_label = "Correct" if is_correct else "Misclass."
            group_label = f"Level {level_label(level)}\n{short_label}"
            level_group_labels.append(group_label)
            for value in values:
                boxplot_rows.append({"group_label": group_label, "corr_to_true_mean": value})
        if len(level_group_labels) == 2:
            pair_candidates.append((level_group_labels[0], level_group_labels[1]))

    if boxplot_rows:
        boxplot_df = pd.DataFrame(boxplot_rows)
        order = list(dict.fromkeys(boxplot_df["group_label"].tolist()))
        draw_blue_boxplot(axes[1], boxplot_df, "group_label", "corr_to_true_mean", order, width=0.6)
        draw_blue_stripplot(axes[1], boxplot_df, "group_label", "corr_to_true_mean", order, size=2.4, jitter=0.16)
        annotated = apply_statannotations(axes[1], boxplot_df, "group_label", "corr_to_true_mean", pair_candidates, order)
        if not annotated and pair_candidates:
            ymax = float(boxplot_df["corr_to_true_mean"].max())
            ymin = float(boxplot_df["corr_to_true_mean"].min())
            yrange = max(1e-6, ymax - ymin)
            top = ymax + 0.15 * yrange
            step = 0.10 * yrange
            n_annotations = 0
            for pair_idx, pair in enumerate(pair_candidates):
                left_vals = boxplot_df.loc[boxplot_df["group_label"] == pair[0], "corr_to_true_mean"].to_numpy()
                right_vals = boxplot_df.loc[boxplot_df["group_label"] == pair[1], "corr_to_true_mean"].to_numpy()
                p_value = mann_whitney_p_value(left_vals, right_vals)
                x1 = order.index(pair[0])
                x2 = order.index(pair[1])
                added = annotate_p_value(axes[1], x1, x2, top + n_annotations * step, 0.03 * yrange, p_value)
                if added:
                    n_annotations += 1
            if n_annotations > 0:
                axes[1].set_ylim(ymin - 0.05 * yrange, top + n_annotations * step + 0.12 * yrange)
        style_categorical_ticks(axes[1], rotation=25, ha="right")

    axes[1].set_xlabel("Task level and correctness", fontsize=15)
    axes[1].set_ylabel("Correlation to true-class mean", fontsize=15)
    axes[1].set_title("Signal similarity by task level and correctness", fontsize=16)
    axes[1].grid(axis="y", alpha=0.18)
    style_axis(axes[1], integer_x=False)

    fig.savefig(os.path.join(output_dir, "exp41_task_level_correct_wrong_analysis.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "exp41_task_level_correct_wrong_analysis.pdf"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_group_error_rate_plot(
    summary_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    output_base: str,
    rotate_labels: bool = False,
    tick_label_map: Optional[Dict[str, str]] = None,
) -> None:
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    x_values = summary_df[x_col].astype(str)
    if tick_label_map:
        x_values = x_values.map(lambda value: tick_label_map.get(value, value))
    ax.bar(x_values, summary_df["error_rate"], color=LINE_TRUE, alpha=0.9)
    ax.set_ylabel("Classification error rate", fontsize=15)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0.0, min(1.0, float(summary_df["error_rate"].max()) + 0.08))
    ax.grid(axis="y", alpha=0.18)
    if rotate_labels:
        ax.tick_params(axis="x", rotation=30)
    style_axis(ax)
    fig.savefig(f"{output_base}.png", dpi=250, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_error_rate_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary_df = (
        df.dropna(subset=[group_col])
        .groupby(group_col, dropna=False)
        .agg(
            n_samples=("is_correct", "size"),
            error_rate=("is_correct", lambda x: float(1.0 - np.mean(np.asarray(x, dtype=float))))
        )
        .reset_index()
        .sort_values(group_col)
    )
    return summary_df


def save_summary_tables(df: pd.DataFrame, output_dir: str) -> None:
    full_path = os.path.join(output_dir, "exp41_prediction_signal_task_joined.csv")
    df.to_csv(full_path, index=False)

    by_task_level = build_error_rate_summary(df[df["task_block_found"]], "task_level_mode")
    by_task_level.to_csv(os.path.join(output_dir, "exp41_error_rate_by_task_level.csv"), index=False)

    stress_only = df[df["y_true"] == 1].copy()
    by_task_outcome = build_error_rate_summary(stress_only, "cognitive_task_outcome")
    by_task_outcome.to_csv(os.path.join(output_dir, "exp41_error_rate_by_cognitive_task_outcome.csv"), index=False)

    by_score_gain = build_error_rate_summary(stress_only, "task_score_gain")
    by_score_gain.to_csv(os.path.join(output_dir, "exp41_error_rate_by_task_score_gain.csv"), index=False)

    similarity_summary = (
        df.groupby(["y_true", "is_correct"])
        .agg(
            n_samples=("ID", "size"),
            corr_to_true_mean=("corr_to_true_mean", "mean"),
            corr_to_pred_mean=("corr_to_pred_mean", "mean"),
            nrmse_to_true_mean=("nrmse_to_true_mean", "mean"),
            task_started_before_window_s=("task_started_before_window_s", "mean"),
            task_ended_before_window_s=("task_ended_before_window_s", "mean"),
            task_wrong_event_count=("task_wrong_event_count", "mean"),
            task_correct_event_count=("task_correct_event_count", "mean"),
            task_score_gain=("task_score_gain", "mean"),
        )
        .reset_index()
    )
    similarity_summary.to_csv(os.path.join(output_dir, "exp41_similarity_summary_by_classification_status.csv"), index=False)


def save_text_summary(df: pd.DataFrame, output_dir: str) -> None:
    stress_df = df[df["y_true"] == 1].copy()
    task_df = df[df["task_block_found"]].copy()

    lines = []
    lines.append("Experiment 41 cognitive-task error analysis")
    lines.append("")
    lines.append(f"n_samples_total: {len(df)}")
    lines.append(f"n_misclassified_total: {int((~df['is_correct']).sum())}")
    lines.append(f"overall_error_rate: {1.0 - float(df['is_correct'].mean()):.4f}")
    lines.append("")

    if not stress_df.empty:
        corr = spearmanr(
            pd.to_numeric(stress_df["task_level_mode"], errors="coerce"),
            (~stress_df["is_correct"]).astype(int),
            nan_policy="omit",
        )
        lines.append(f"stress_only_mean_task_started_before_window_s: {np.nanmean(stress_df['task_started_before_window_s']):.4f}")
        lines.append(f"stress_only_mean_task_ended_before_window_s: {np.nanmean(stress_df['task_ended_before_window_s']):.4f}")
        lines.append(f"stress_only_mean_task_wrong_event_count: {np.nanmean(stress_df['task_wrong_event_count']):.4f}")
        lines.append(f"stress_only_mean_task_correct_event_count: {np.nanmean(stress_df['task_correct_event_count']):.4f}")
        lines.append(f"stress_only_mean_task_score_gain: {np.nanmean(stress_df['task_score_gain']):.4f}")
        lines.append(f"stress_only_fraction_task_score_non_increasing: {np.nanmean(stress_df['task_score_non_increasing'].astype(float)):.4f}")
        lines.append(f"stress_only_fraction_cognitive_task_incorrect: {np.nanmean(stress_df['cognitive_task_incorrect'].astype(float)):.4f}")
        lines.append(f"stress_only_fraction_cognitive_task_correct: {np.nanmean(stress_df['cognitive_task_correct'].astype(float)):.4f}")
        lines.append(f"stress_only_error_rate_when_cognitive_task_incorrect: {1.0 - float(stress_df[stress_df['cognitive_task_incorrect']]['is_correct'].mean()) if (stress_df['cognitive_task_incorrect']).any() else np.nan:.4f}")
        lines.append(f"stress_only_error_rate_when_cognitive_task_not_incorrect: {1.0 - float(stress_df[~stress_df['cognitive_task_incorrect']]['is_correct'].mean()) if (~stress_df['cognitive_task_incorrect']).any() else np.nan:.4f}")
        lines.append(f"stress_only_spearman_task_level_vs_error: rho={corr.statistic:.4f}, p={corr.pvalue:.4g}")

    if not task_df.empty:
        corr_shape = spearmanr(
            pd.to_numeric(task_df["task_level_mode"], errors="coerce"),
            pd.to_numeric(task_df["corr_to_true_mean"], errors="coerce"),
            nan_policy="omit",
        )
        lines.append(f"preceding_task_spearman_task_level_vs_corr_to_true_mean: rho={corr_shape.statistic:.4f}, p={corr_shape.pvalue:.4g}")

    with open(os.path.join(output_dir, "exp41_cognitive_task_error_summary.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_json_config(resolve_path(script_dir, args.config))
    experiment_root = get_experiment_root(config, script_dir)
    raw_root = resolve_path(script_dir, "data/vr_goalkeeper")
    df_path = os.path.join(get_local_project_root(script_dir), "data", "vr_goalkeeper", "dataframes", "DL_out.pkl")
    output_dir = args.output_dir or os.path.join(
        experiment_root,
        "recovery",
        "exp41_cognitive_task_error_analysis",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Paths] script_dir={script_dir}")
    print(f"[Paths] experiment_root={experiment_root}")
    print(f"[Paths] raw_root={raw_root}")
    print(f"[Paths] preferred_df_prep={df_path}")
    print(f"[Paths] output_dir={output_dir}")

    df_prep = load_prepared_dataframe(config, script_dir)
    prediction_df = load_prediction_table(df_prep, experiment_root, config["valid_IDs"])
    class_means = compute_class_mean_signals(prediction_df, SIGNAL_COL)
    prediction_df = add_signal_similarity_metrics(prediction_df, class_means, SIGNAL_COL)
    analysis_df = enrich_with_raw_task_metadata(prediction_df, raw_root)

    plot_class_means(analysis_df, class_means, output_dir)
    plot_similarity_distributions(analysis_df, output_dir)
    plot_task_timing(analysis_df, output_dir)
    plot_task_level_correctness(analysis_df, output_dir)

    task_level_df = build_error_rate_summary(analysis_df[analysis_df["task_block_found"]], "task_level_mode")
    save_group_error_rate_plot(
        task_level_df,
        x_col="task_level_mode",
        x_label="Task level",
        title="Experiment 41: classification error rate by preceding task difficulty",
        output_base=os.path.join(output_dir, "exp41_error_rate_by_task_level"),
    )

    stress_df = analysis_df[analysis_df["y_true"] == 1].copy()
    task_outcome_df = build_error_rate_summary(stress_df, "cognitive_task_outcome")
    save_group_error_rate_plot(
        task_outcome_df,
        x_col="cognitive_task_outcome",
        x_label="Cognitive task outcome",
        title="Experiment 41: stress-trial error rate by cognitive task outcome",
        output_base=os.path.join(output_dir, "exp41_error_rate_by_cognitive_task_outcome"),
    )

    save_summary_tables(analysis_df, output_dir)
    save_text_summary(analysis_df, output_dir)

    metadata = {
        "analysis_type": "exp41_cognitive_task_error_analysis",
        "config_path": resolve_path(script_dir, args.config),
        "experiment_root": experiment_root,
        "raw_root": raw_root,
        "output_dir": output_dir,
        "signal_col": SIGNAL_COL,
    }
    with open(os.path.join(output_dir, "exp41_cognitive_task_error_analysis_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved exp41 cognitive-task error analysis to {output_dir}")


if __name__ == "__main__":
    main()
