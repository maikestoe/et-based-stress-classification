"""
Model complexity analysis for fold-specific deep-learning models.

This script evaluates model complexity for trained experiments using the
best Optuna trial of each outer LOSO fold. Because hyperparameters may differ
between outer folds, complexity is reported per fold and summarized across
folds and, when multiple experiments are selected, across models.

What is measured
----------------
- trainable parameters
- non-trainable parameters
- total parameters
- approximate parameter memory footprint
- saved weight-file size when available
- optional inference latency on the held-out outer-fold inputs

This script does not retrain models. It reuses saved Optuna studies and
rebuilds the best model architecture for each fold from the recorded best
trial parameters.
"""

import argparse
import gc
import json
import os
import pickle
import time

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from pickle_compat import install_pandas_pickle_compat

install_pandas_pickle_compat()

from DL_CV import get_trial_parameters
from DL_models import get_model
from DL_train_CV import transform_data_as_input
from DL_utils import get_training_data, process_data, set_DL_config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_SUBDIR = "complexity_analysis"
CONFIG_DIR = os.path.join(SCRIPT_DIR, "config_files")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOCAL_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
LOCAL_RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
MODEL_NAME_MAP = {
    "CNN": "CNN",
    "LSTM-1": "LSTM-1",
    "ConvLSTM-1": "ConvLSTM-1",
    "LSTM-3": "LSTM-3",
    "ConvLSTM-3": "ConvLSTM-3",
}
SIGNAL_LABEL_MAP = {
    "meanDia_corrected": "PD",
    "velocity": "velocity",
    "acceleration": "acceleration",
    "position": "visual angle",
    "asymptotic_model": "asymptotic",
    "fix_array": "fixations",
}
REPRESENTATIVE_EXPERIMENT_IDS = {
    "vr": ["5", "14", "23", "32", "41"],
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Assess fold-wise model complexity for one or more saved DL "
            "experiments using best-trial parameters from Optuna."
        )
    )
    parser.add_argument(
        "--config",
        help="Optional path to one experiment config JSON. If omitted, matching experiments are discovered automatically.",
    )
    parser.add_argument(
        "--dataset",
        choices=["vr", "fordigit"],
        help="Dataset used for automatic experiment discovery.",
    )
    parser.add_argument(
        "--timestamp",
        help="Timestamp used for automatic experiment discovery, for example 2024-08-09.",
    )
    parser.add_argument(
        "--config-dir",
        default=CONFIG_DIR,
        help="Directory scanned for experiment configs during automatic discovery.",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Output subdirectory name.",
    )
    parser.add_argument(
        "--representative-only",
        action="store_true",
        help=(
            "For supported datasets, restrict the analysis to a curated set of "
            "representative experiments used for the publication complexity table."
        ),
    )
    parser.add_argument(
        "--measure-inference",
        action="store_true",
        help="Also measure inference latency on the held-out fold inputs.",
    )
    parser.add_argument(
        "--timing-batch-size",
        type=int,
        default=1,
        help="Batch size used for inference timing. Default: 1.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup forward passes before timing. Default: 5.",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=20,
        help="Number of timed forward passes per fold. Default: 20.",
    )
    args = parser.parse_args()
    if not args.config and not (args.dataset and args.timestamp):
        parser.error("Provide either --config or both --dataset and --timestamp.")
    return args


def load_json_config(config_path):
    """Load and expand the experiment config."""
    with open(config_path, "r") as file:
        config = json.load(file)
    config["DL"]["path_results"] = normalize_project_path(config["DL"]["path_results"], kind="results")
    config["df_prep_path"] = normalize_project_path(config["df_prep_path"], kind="data")
    return config


def normalize_project_path(path_value, kind):
    """
    Map old cluster-style paths onto the local project layout.

    The original experiment configs often use `$TMPDIR/...`, while local runs
    use `../DL_project_*`. For complexity analysis we only need to resolve the
    corresponding local project roots.
    """
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


def get_base_folder(config):
    """Return the experiment root folder."""
    return os.path.join(
        config["DL"]["path_results"],
        config["timestamp"],
        config["DL"]["model"],
        str(config["experiment_id"]),
    )


def get_dataset_label_from_config(config):
    """Infer a short dataset label from the config/result paths."""
    path_blob = " ".join(
        [
            str(config.get("df_prep_path", "")),
            str(config.get("DL", {}).get("path_results", "")),
        ]
    ).lower()
    if "study_pk_vr" in path_blob:
        return "vr"
    if "fordigitstress" in path_blob or "fordigit" in path_blob:
        return "fordigit"
    return "unknown"


def get_dataset_result_root(config):
    """Return the dataset-level DL result root."""
    base_path = config["DL"]["path_results"]
    return os.path.join(base_path, config["timestamp"])


def ensure_output_dir(output_dir):
    """Create and return the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_multi_experiment_output_dir(configs, output_subdir, representative_only=False):
    """Return a shared output directory for multiple matching experiments."""
    first_config = configs[0]
    dataset_root = get_dataset_result_root(first_config)
    dataset_label = get_dataset_label_from_config(first_config)
    suffix = "representative_models" if representative_only else "all_models"
    return os.path.join(
        dataset_root,
        f"{dataset_label}_{output_subdir}_{suffix}",
    )


def load_best_trial_from_db(config, test_id):
    """Load the best Optuna trial for one outer fold."""
    db_path = os.path.join(get_base_folder(config), "optunaStudy.db")
    storage_url = f"sqlite:///{db_path}"
    study_name = f"{test_id}_optunaStudy_{config['experiment_id']}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return study.best_trial


def result_db_exists_for_config(config):
    """Check whether the experiment has a saved Optuna study."""
    db_path = os.path.join(get_base_folder(config), "optunaStudy.db")
    return os.path.exists(db_path)


def discover_matching_configs(config_dir, dataset, timestamp, representative_only=False):
    """
    Discover experiment configs for one dataset/timestamp combination.

    For this project, the training experiments are enumerated explicitly:
    - VR: exp1 ... exp42
    - ForDigitStress: exp201 ... exp205
    """
    if representative_only and dataset in REPRESENTATIVE_EXPERIMENT_IDS:
        experiment_ids = [int(experiment_id) for experiment_id in REPRESENTATIVE_EXPERIMENT_IDS[dataset]]
    else:
        experiment_ids = (
            list(range(1, 43)) if dataset == "vr" else list(range(201, 206))
        )
    matched = []

    for experiment_id in experiment_ids:
        config_path = os.path.join(config_dir, f"exp{experiment_id}.json")
        if not os.path.exists(config_path):
            continue

        config = load_json_config(config_path)
        config_timestamp = str(config.get("timestamp"))
        if config_timestamp != str(timestamp):
            continue
        if not result_db_exists_for_config(config):
            continue

        config["_config_path"] = config_path
        matched.append(config)

    if not matched:
        raise FileNotFoundError(
            f"No saved experiment configs found for dataset={dataset} and timestamp={timestamp} in {config_dir}."
        )
    return matched


def validate_config_timestamp_consistency(configs, expected_timestamp):
    """Ensure all selected configs belong to the same requested timestamp series."""
    seen_timestamps = sorted({str(config.get("timestamp")) for config in configs})
    if seen_timestamps != [str(expected_timestamp)]:
        raise ValueError(
            "Loaded configs do not share the expected timestamp. "
            f"Expected only {expected_timestamp}, found {seen_timestamps}."
        )


def build_outer_fold_inputs(config, test_id, model_params):
    """
    Build processed held-out inputs for one outer fold.

    This mirrors the original data preparation path used for training and
    inference. If your preprocessing pipeline changes, adapt this function.
    """
    with open(config["df_prep_path"], "rb") as fh:
        df = pickle.load(fh)

    df["lab_num"] = df["lab_num"].astype(int)
    train_idx = df.index[df["ID"] != test_id].tolist()
    test_idx = df.index[df["ID"] == test_id].tolist()

    x_train, y_train, x_test, y_test, train_ids, test_ids = get_training_data(
        df, train_idx, test_idx, config
    )
    del df, train_idx, test_idx

    x_train, y_train, x_test, y_test = process_data(
        x_train, y_train, x_test, y_test, train_ids, config
    )
    x_test_pre_transform = x_test
    del train_ids, test_ids, x_train, y_train, y_test
    gc.collect()

    # We only need the transformed held-out input for complexity/timing.
    _, x_test = transform_data_as_input(x_test, x_test, model_params, config)
    return x_test_pre_transform, x_test


def get_outer_weight_path(config, test_id):
    """Return the saved outer-model weight path if it exists."""
    candidate = os.path.join(
        get_base_folder(config),
        f"test_ID_{test_id}",
        "best_weights_foldnan.weights.h5",
    )
    return candidate if os.path.exists(candidate) else None


def maybe_load_weights(model, weight_path):
    """Load saved weights when available."""
    if weight_path is not None and os.path.exists(weight_path):
        model.load_weights(weight_path)


def parameter_memory_bytes(model):
    """Approximate parameter memory footprint in bytes."""
    total_bytes = 0
    for variable in model.weights:
        dtype_size = int(tf.as_dtype(variable.dtype).size)
        total_bytes += dtype_size * int(np.prod(variable.shape))
    return total_bytes


def prepare_batch(x_input, batch_size):
    """Create a small inference batch from held-out inputs."""
    if isinstance(x_input, list):
        return [np.asarray(branch[:batch_size]).astype(np.float32) for branch in x_input]
    return np.asarray(x_input[:batch_size]).astype(np.float32)


def run_forward_pass(model, batch):
    """Run a single forward pass."""
    if isinstance(batch, list):
        tensors = [tf.convert_to_tensor(branch) for branch in batch]
        _ = model(tensors, training=False)
    else:
        tensor = tf.convert_to_tensor(batch)
        _ = model(tensor, training=False)


def measure_inference_latency(model, x_input, batch_size, warmup_runs, timing_runs):
    """Measure inference latency in milliseconds."""
    batch = prepare_batch(x_input, batch_size=batch_size)

    for _ in range(warmup_runs):
        run_forward_pass(model, batch)

    timings_ms = []
    for _ in range(timing_runs):
        start = time.perf_counter()
        run_forward_pass(model, batch)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    sample_count = batch[0].shape[0] if isinstance(batch, list) else batch.shape[0]
    return {
        "timing_batch_size": int(sample_count),
        "inference_mean_ms_per_batch": float(np.mean(timings_ms)),
        "inference_std_ms_per_batch": float(np.std(timings_ms, ddof=1)) if len(timings_ms) > 1 else 0.0,
        "inference_mean_ms_per_sample": float(np.mean(timings_ms) / sample_count),
        "inference_std_ms_per_sample": float(
            (np.std(timings_ms, ddof=1) / sample_count) if len(timings_ms) > 1 else 0.0
        ),
    }


def analyze_fold_complexity(config, test_id, measure_inference, timing_batch_size, warmup_runs, timing_runs):
    """Build one fold-specific best model and collect complexity metrics."""
    best_trial = load_best_trial_from_db(config, test_id)
    model_params = get_trial_parameters(config, best_trial)

    # Rebuild a fold-specific model from the saved best-trial hyperparameters.
    x_input_pre_transform, x_input = build_outer_fold_inputs(config, test_id, model_params)
    arr = np.asarray(x_input_pre_transform)
    data_dim = int(arr.shape[2])
    timesteps = int(arr.shape[1])

    model = get_model(config["DL"]["model"], data_dim, timesteps, 2, model_params)
    weight_path = get_outer_weight_path(config, test_id)
    maybe_load_weights(model, weight_path)

    result = {
        "test_id": int(test_id),
        "best_trial_number": int(best_trial.number),
        "best_trial_value": float(best_trial.value),
        "experiment_id": str(config["experiment_id"]),
        "model": config["DL"]["model"],
        "input_cols": ",".join(config["DL"]["input_cols"]),
        "config_path": config.get("_config_path", ""),
        "trainable_params": int(np.sum([K.count_params(p) for p in model.trainable_weights])),
        "non_trainable_params": int(np.sum([K.count_params(p) for p in model.non_trainable_weights])),
        "total_params": int(model.count_params()),
        "parameter_memory_bytes": int(parameter_memory_bytes(model)),
        "weight_file_path": weight_path if weight_path is not None else "",
        "weight_file_size_bytes": int(os.path.getsize(weight_path)) if weight_path is not None else np.nan,
        "best_trial_params_json": json.dumps(best_trial.params, sort_keys=True),
    }

    if measure_inference:
        result.update(
            measure_inference_latency(
                model=model,
                x_input=x_input,
                batch_size=timing_batch_size,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
            )
        )

    del x_input_pre_transform, x_input, model, best_trial, model_params
    K.clear_session()
    gc.collect()
    return result


def build_summary_table(per_fold_df, group_columns):
    """Summarize complexity metrics across outer folds."""
    numeric_columns = [
        "best_trial_value",
        "trainable_params",
        "non_trainable_params",
        "total_params",
        "parameter_memory_bytes",
        "weight_file_size_bytes",
        "inference_mean_ms_per_batch",
        "inference_std_ms_per_batch",
        "inference_mean_ms_per_sample",
        "inference_std_ms_per_sample",
    ]
    available_columns = [column for column in numeric_columns if column in per_fold_df.columns]

    rows = []
    grouped = per_fold_df.groupby(group_columns, dropna=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_columns, group_key))
        for column in available_columns:
            values = pd.to_numeric(group_df[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **group_values,
                    "metric": column,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max()),
                    "n_folds": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def format_mean_std(mean_value, std_value, decimals=2):
    """Format summary statistics as mean ± std strings."""
    if pd.isna(mean_value):
        return ""
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def build_publication_table(summary_df):
    """Create a paper-friendly wide table from the experiment-level summary."""
    if summary_df.empty:
        return pd.DataFrame()

    summary_pivot = summary_df.pivot_table(
        index=["experiment_id", "model", "input_cols"],
        columns="metric",
        values=["mean", "std"],
        aggfunc="first",
    )

    rows = []
    for index_key, row in summary_pivot.iterrows():
        experiment_id, model, input_cols = index_key
        signal_label = SIGNAL_LABEL_MAP.get(str(input_cols), str(input_cols))
        model_label = MODEL_NAME_MAP.get(str(model), str(model))
        table_label = f"{model_label} ({signal_label})"

        total_params_mean = row.get(("mean", "total_params"), np.nan)
        total_params_std = row.get(("std", "total_params"), np.nan)
        memory_mean_mb = row.get(("mean", "parameter_memory_bytes"), np.nan)
        memory_std_mb = row.get(("std", "parameter_memory_bytes"), np.nan)
        weight_mean_mb = row.get(("mean", "weight_file_size_bytes"), np.nan)
        weight_std_mb = row.get(("std", "weight_file_size_bytes"), np.nan)
        latency_mean = row.get(("mean", "inference_mean_ms_per_sample"), np.nan)
        latency_std = row.get(("std", "inference_mean_ms_per_sample"), np.nan)

        if not pd.isna(memory_mean_mb):
            memory_mean_mb /= (1024 ** 2)
        if not pd.isna(memory_std_mb):
            memory_std_mb /= (1024 ** 2)
        if not pd.isna(weight_mean_mb):
            weight_mean_mb /= (1024 ** 2)
        if not pd.isna(weight_std_mb):
            weight_std_mb /= (1024 ** 2)

        rows.append(
            {
                "experiment_id": str(experiment_id),
                "model": str(model),
                "input_cols": str(input_cols),
                "model_label": table_label,
                "parameters_mean": total_params_mean,
                "parameters_std": total_params_std,
                "parameters_mean_pm_std": format_mean_std(total_params_mean, total_params_std, decimals=0),
                "memory_mb_mean": memory_mean_mb,
                "memory_mb_std": memory_std_mb,
                "memory_mb_mean_pm_std": format_mean_std(memory_mean_mb, memory_std_mb, decimals=2),
                "weight_size_mb_mean": weight_mean_mb,
                "weight_size_mb_std": weight_std_mb,
                "weight_size_mb_mean_pm_std": format_mean_std(weight_mean_mb, weight_std_mb, decimals=2),
                "latency_ms_per_sample_mean": latency_mean,
                "latency_ms_per_sample_std": latency_std,
                "latency_ms_per_sample_mean_pm_std": format_mean_std(latency_mean, latency_std, decimals=4),
            }
        )

    return pd.DataFrame(rows).sort_values(["model_label", "experiment_id"]).reset_index(drop=True)


def filter_publication_table(publication_df, dataset, representative_only):
    """Restrict the publication table to curated representative experiments when requested."""
    if publication_df.empty or not representative_only:
        return publication_df
    selected_ids = REPRESENTATIVE_EXPERIMENT_IDS.get(dataset)
    if not selected_ids:
        return publication_df
    return publication_df[publication_df["experiment_id"].astype(str).isin(selected_ids)].reset_index(drop=True)


def save_results(output_dir, per_fold_df, summary_df, overall_df, publication_df, args, configs):
    """Save CSV outputs and a small metadata JSON."""
    per_fold_df.to_csv(os.path.join(output_dir, "complexity_per_fold.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "complexity_summary_by_experiment.csv"), index=False)
    overall_df.to_csv(os.path.join(output_dir, "complexity_summary_by_model.csv"), index=False)
    publication_df.to_csv(os.path.join(output_dir, "complexity_publication_table.csv"), index=False)

    metadata = {
        "config_path": args.config,
        "dataset": args.dataset,
        "timestamp": args.timestamp,
        "measure_inference": args.measure_inference,
        "timing_batch_size": args.timing_batch_size,
        "warmup_runs": args.warmup_runs,
        "timing_runs": args.timing_runs,
        "num_experiments": len(configs),
        "experiments": [
            {
                "experiment_id": str(config["experiment_id"]),
                "model": config["DL"]["model"],
                "input_cols": config["DL"]["input_cols"],
                "config_path": config.get("_config_path", ""),
            }
            for config in configs
        ],
    }
    with open(os.path.join(output_dir, "complexity_analysis_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def print_summary(output_dir, per_fold_df, overall_df, measure_inference):
    """Print a short console summary."""
    print("\nModel complexity analysis summary")
    print("--------------------------------")
    print(f"Output directory: {output_dir}")
    print(f"Analyzed outer folds: {len(per_fold_df)}")
    print(f"Analyzed experiments: {per_fold_df[['experiment_id', 'model', 'input_cols']].drop_duplicates().shape[0]}")
    if not overall_df.empty:
        params_rows = overall_df[overall_df["metric"] == "total_params"]
        if not params_rows.empty:
            print("Total params by model/input:")
            for _, row in params_rows.iterrows():
                row_label = f"{row['model']} | {row['input_cols']}"
                if "experiment_id" in row.index:
                    row_label = f"exp {row['experiment_id']} | {row_label}"
                print(f"  {row_label}: mean={row['mean']:.1f}, std={row['std']:.1f}")
        if measure_inference:
            latency_rows = overall_df[overall_df["metric"] == "inference_mean_ms_per_sample"]
            if not latency_rows.empty:
                print("Inference latency per sample by model/input:")
                for _, row in latency_rows.iterrows():
                    row_label = f"{row['model']} | {row['input_cols']}"
                    if "experiment_id" in row.index:
                        row_label = f"exp {row['experiment_id']} | {row_label}"
                    print(f"  {row_label}: mean={row['mean']:.4f} ms, std={row['std']:.4f} ms")


def main():
    """Run the complexity analysis."""
    args = parse_args()
    if args.config:
        config = load_json_config(args.config)
        config["_config_path"] = args.config
        configs = [config]
        output_dir = ensure_output_dir(
            os.path.join(get_base_folder(config), args.output_subdir)
        )
    else:
        configs = discover_matching_configs(
            config_dir=args.config_dir,
            dataset=args.dataset,
            timestamp=args.timestamp,
            representative_only=args.representative_only,
        )
        validate_config_timestamp_consistency(configs, args.timestamp)
        output_dir = ensure_output_dir(
            get_multi_experiment_output_dir(configs, args.output_subdir, representative_only=args.representative_only)
        )

    set_DL_config()

    rows = []
    for config in configs:
        print(
            f"Analyzing experiment {config['experiment_id']} "
            f"({config['DL']['model']} | {','.join(config['DL']['input_cols'])})"
        )
        for test_id in config["valid_IDs"]:
            rows.append(
                analyze_fold_complexity(
                    config=config,
                    test_id=test_id,
                    measure_inference=args.measure_inference,
                    timing_batch_size=args.timing_batch_size,
                    warmup_runs=args.warmup_runs,
                    timing_runs=args.timing_runs,
                )
            )

    per_fold_df = pd.DataFrame(rows).sort_values("test_id")
    summary_df = build_summary_table(
        per_fold_df,
        group_columns=["experiment_id", "model", "input_cols"],
    )
    overall_df = build_summary_table(
        per_fold_df,
        group_columns=["model", "input_cols"],
    )
    publication_df = build_publication_table(summary_df)
    publication_df = filter_publication_table(publication_df, args.dataset, args.representative_only)
    save_results(output_dir, per_fold_df, summary_df, overall_df, publication_df, args, configs)
    print_summary(output_dir, per_fold_df, overall_df, args.measure_inference)


if __name__ == "__main__":
    main()
