"""Utilities for loading recovered per-sample prediction rows."""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from pickle_compat import install_pandas_pickle_compat

from error_analysis_subjects import (
    get_base_folder,
    prettify_input_cols,
    prettify_model_name,
)


install_pandas_pickle_compat()


def load_prepared_dataframe(config):
    """Load the prepared dataframe referenced by a config."""
    with open(config["df_prep_path"], "rb") as file:
        return pickle.load(file)


def find_recovered_subject_ids(config):
    """Return test IDs for which per-subject recovered predictions exist."""
    recovered_ids = []
    base_folder = get_base_folder(config)
    for test_id in config["valid_IDs"]:
        recovery_dir = os.path.join(base_folder, f"test_ID_{test_id}", "recovery")
        y_true_path = os.path.join(recovery_dir, "y_true_outer.npy")
        y_pred_path = os.path.join(recovery_dir, "y_pred_outer.npy")
        y_score_path = os.path.join(recovery_dir, "y_score_outer.npy")
        if os.path.exists(y_true_path) and os.path.exists(y_pred_path) and os.path.exists(y_score_path):
            recovered_ids.append(int(test_id))
    return recovered_ids


def load_recovered_prediction_rows(config):
    """Load per-sample recovered predictions and map them back to subject/shot rows."""
    recovered_ids = find_recovered_subject_ids(config)
    if not recovered_ids:
        return pd.DataFrame()

    df_prep = load_prepared_dataframe(config)
    rows = []
    base_folder = get_base_folder(config)
    input_display = prettify_input_cols(",".join(config["DL"]["input_cols"]))
    model_display = prettify_model_name(config["DL"]["model"])
    combination_display = f"{model_display} ({input_display})"

    for test_id in recovered_ids:
        subject_df = df_prep[df_prep["ID"] == test_id].copy().reset_index(drop=True)
        recovery_dir = os.path.join(base_folder, f"test_ID_{test_id}", "recovery")
        y_true = np.load(os.path.join(recovery_dir, "y_true_outer.npy"))
        y_pred = np.load(os.path.join(recovery_dir, "y_pred_outer.npy"))
        y_score = np.load(os.path.join(recovery_dir, "y_score_outer.npy"))

        n = min(len(subject_df), len(y_true), len(y_pred), len(y_score))
        if n == 0:
            continue
        subject_df = subject_df.iloc[:n].copy()

        subject_df["test_id"] = int(test_id)
        subject_df["y_true"] = y_true[:n].astype(int)
        subject_df["y_pred"] = y_pred[:n].astype(int)
        subject_df["y_score"] = y_score[:n].astype(float)
        subject_df["correct"] = (subject_df["y_true"] == subject_df["y_pred"]).astype(int)
        subject_df["fp"] = ((subject_df["y_true"] == 0) & (subject_df["y_pred"] == 1)).astype(int)
        subject_df["fn"] = ((subject_df["y_true"] == 1) & (subject_df["y_pred"] == 0)).astype(int)
        subject_df["confidence_margin"] = np.abs(subject_df["y_score"] - 0.5)
        subject_df["model_display"] = model_display
        subject_df["input_display"] = input_display
        subject_df["combination_display"] = combination_display
        rows.append(subject_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
