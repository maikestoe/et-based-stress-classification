"""
Preprocessing and Analysis for the ForDigitStress Dataset

This script provides functions to preprocess the ForDigitStress dataset. It loads and combines the public
ForDigitStress files, extracts baseline-corrected pupil-diameter windows, checks signal quality and label consistency,
and saves the resulting dataframe for model training.

Functions:
----------
- load_combine_data(id, filepath):
    Load and combine stress and pupil feature data for a given participant ID.

- get_conf_label(df, conf_thresh):
    Check if the eye tracker confidence is below a certain threshold and find the last index where this happens.

- get_label(df):
    Define the label if only one label is present within the whole window and label as invalid if multiple labels are
    present.

- main(data_dir, output_path, invalid_ids, settings):
    Process the ForDigitStress data for all participants and save results.

Dependencies:
-------------
- pandas
- numpy
- os
- segment_fordigitstress (custom module)

Author: Maike Laut
Date: 20.06.2024
"""

import pandas as pd
import os
import numpy as np
import segment_fordigitstress

from pd_utils import baseline_correct
from pd_utils import interpolate_confidence

settings = {
    'windowLength_sec': 5,  # Length of sample window in seconds (25 Hz = 125 samples)
    'samplingRate': 25  # Hz
}

pdFilt_parameters = {
     'confidence': {
     'confidence_thresh': 0.8
     }
}


# -------------------------------------------------------
# Section: Utility Functions
# -------------------------------------------------------


def load_combine_data(id, filepath):
    """
    Load and combine stress and pupil feature data for a given participant ID.

    :param id: Participant ID.
    :type id: int
    :param filepath: Path to the dataset.
    :type filepath: str
    :return: Combined dataframe of stress and pupil features.
    :rtype: pandas.DataFrame
    """

    stress_df = pd.read_csv(f'{filepath}VP{id}/stress.csv', sep=';')
    pupil_df = pd.read_csv(f'{filepath}VP{id}/pupil_features.csv', sep=';')

    combined_df = pd.concat([stress_df.reset_index(drop=True), pupil_df.reset_index(drop=True)], axis=1)

    # Add time column
    num_samples = combined_df.shape[0]
    combined_df['time'] = [i / settings['samplingRate'] for i in range(num_samples)]

    return combined_df



def get_conf_label(df, conf_thresh):
    """
    Check if the eye tracker confidence in df is below a certain confidence value and find the last index where this
    happens.

    :param df: Dataframe containing eye tracker confidence.
    :type df: pandas.DataFrame
    :param conf_thresh: Tracker confidence threshold.
    :type conf_thresh: float
    :return: Index of the last non-valid confidence value.
    :rtype: int
    """

    above_thresh = df['confidence'] > conf_thresh
    # if less than 80% of the samples are above the confidence threshold
    if len(np.where(above_thresh)[0])/len(above_thresh) < 0.8:
        idx_nonvalid_last = settings['samplingRate']  # Jump to next window with overlap
    else:
        idx_nonvalid_last = -1  # indicate as valid
    return idx_nonvalid_last


def find_last_change(arr):
    """
    Find the index of the last change in an array.

    This function iterates through an array from the end to the beginning and returns the index of the last element
    where a change occurs (i.e., the element is different from the next one).

    :param arr: Array to be checked for changes.
    :type arr: list or numpy.ndarray
    :return: Index of the last change in the array. If no change is found, returns -1.
    :rtype: int
    """

    for i in range(len(arr) - 2, -1, -1):
        if arr[i] != arr[i + 1]:
            return i
    return -1


def get_label(df):
    """
    Define the label if only one label is present within the whole window and label as invalid if multiple labels are present.

    :param df: DataFrame of the segmented window containing data and labels.
    :type df: pandas.DataFrame
    :return: A tuple containing the text label, numeric label, and index of the last label change.
    :rtype: tuple(str, int, int)
    """

    lab_orig = df['stress'].values
    if 0 in lab_orig and 1 in lab_orig:
        label_num = -1
        label = 'invalid'
    elif 0 in lab_orig:
        label_num = 1
        label = 'stress'
    elif 1 in lab_orig:
        label_num = 0
        label = 'nostress'
    else:
        label = np.nan
        label_num = np.nan
        print('Invalid label detected - CAUTION')

    # find the index of last change of label
    last_change_idx = find_last_change(lab_orig)  # if no change this returns -1 -> valid window for segmentation

    return label, label_num, last_change_idx


def preprocess_pd_pipeline(df, base_correct_val):
    """
    Apply preprocessing to pupil diameter data: interpolate low-confidence values and perform baseline correction.

    This function performs two key preprocessing steps on pupil diameter data:
    1. Interpolates samples with low confidence.
    2. Applies baseline correction using a precomputed baseline value.

    The corrected signal is stored in both 'pd_corrected' and 'pd_preprocessed' columns for further analysis.

    :param df: Input DataFrame containing pupil diameter and confidence values.
               Must include a 'pd_interp' column (interpolated or raw pupil data).
    :type df: pandas.DataFrame
    :param base_correct_val: Value used for baseline correction (e.g., median from baseline segment).
    :type base_correct_val: float

    :return: DataFrame with two new columns:
             - 'pd_corrected': Baseline-corrected pupil diameter.
             - 'pd_preprocessed': Copy of 'pd_corrected' used for modeling or analysis.
    :rtype: pandas.DataFrame
    """

    df = interpolate_confidence(df)
    pd_corrected = baseline_correct(df['pd_interp'], base_correct_val)

    df['pd_corrected'] = pd_corrected
    df['pd_preprocessed'] = df['pd_corrected'].copy(deep=True)
    return df


def extract_windows(df, start, ID, search_label, DL_out, window_length, counter_good, counter_bad):
    """
    Extract non-overlapping, valid time windows with consistent labeling and good data quality.

    This function scans through the input DataFrame and segments it into fixed-length time windows. It checks for:
    - sufficient eye-tracking confidence,
    - no label transitions within a window,
    - and whether the window matches the desired label ('stress' or 'nostress').

    Valid windows are added to the output DataFrame, along with counters tracking the number of good and bad segments.

    :param df: DataFrame containing time-series data (e.g. pupil diameter, confidence, labels).
    :type df: pandas.DataFrame
    :param start: Starting index for window extraction.
    :type start: int
    :param ID: Participant ID for tracking.
    :type ID: int
    :param search_label: Target label to extract ('stress' or 'nostress').
    :type search_label: str
    :param DL_out: DataFrame to which valid segments will be appended.
    :type DL_out: pandas.DataFrame
    :param window_length: Number of samples in each window.
    :type window_length: int
    :param counter_good: Counter for valid windows (quality and label checks passed).
    :type counter_good: int
    :param counter_bad: Counter for invalid windows (bad quality or label mix).
    :type counter_bad: int

    :return: Tuple containing:
        - Updated DL_out DataFrame,
        - Number of valid windows with the target label,
        - Updated bad window counter (int),
        - Updated good window counter (int).
    :rtype: tuple
    """

    counter_label = 0

    while start <= len(df) - window_length:
        segment = df.iloc[start:start + window_length].copy()

        idx_last_invalidConf = get_conf_label(segment, pdFilt_parameters['confidence']['confidence_thresh'])
        label, label_num, idx_last_change = get_label(segment)

        if idx_last_invalidConf != -1:  # invalid due to bad quality
            counter_bad += 1

        if idx_last_invalidConf == -1 and idx_last_change == -1:  # if window has good quality and non-overlapping labels
            start += int(window_length * 1)  # Next window can start without an overlap of 50%
            # save window
            if label == search_label:  # and counter_label < 20:
                new_row_DL = {
                    'ID': ID,
                    'lab_str': label,
                    'lab_num': label_num,
                    'shot': counter_label,  # sample
                    'meanDia': segment['pupil_diameter'],  # pd right
                    'meanDia_corrected': segment['pd_preprocessed'],
                    'time': segment['time']
                }
                DL_out = pd.concat([DL_out, pd.DataFrame([new_row_DL])], ignore_index=True)

                counter_label += 1
            counter_good += 1

        else:  # if window has bad quality or overlapping labels
            # Check where quality/ label problem appears and set the start of next window after the last quality/label problem
            start += max(idx_last_invalidConf, idx_last_change) + 1
    return DL_out, counter_label, counter_bad, counter_good


def main(data_dir, output_path, invalid_ids, settings):
    """
    Process the ForDigitStress data for all participants and save results.

    This function iterates through all participants, segments the data, extracts labels, and computes gaze-related
    features and time series signals for stress classification. The resulting dataframes are saved in a pickle and a
    csv file.

    :param data_dir: Path where the dataset is stored.
    :type data_dir: str
    :param output_path: Path where the extracted dataframes are stored.
    :type output_path: str
    :param invalid_ids: List of IDs with invalid data to be excluded from further analysis.
    :type invalid_ids: list
    :param settings: Dictionary containing settings for data processing.
    :type settings: dict
    :return: None
    :rtype: None

    """

    # Create output dataframe
    DL_out = pd.DataFrame(columns=['ID', 'lab_str', 'lab_num', 'shot', 'meanDia', 'meanDia_corrected', 'time'])
    counter_valid = 0
    counter_badQuality = 0
    counter_stress_total = 0
    counter_nostress_total = 0
    # Iterate over participants
    for ID in range(10, 51):
        if ID in invalid_ids:
            continue  # Skip excluded IDs

        df = load_combine_data(ID, data_dir)

        # Extract baseline segment from preparation phase and compute correction value
        baseline_dat = df[df.situation == 1]   # 1: preparation, 2: post-interview
        base_snipped, _, _ = segment_fordigitstress.segment_baseline(baseline_dat['pupil_diameter'], baseline_dat['time'], baseline_dat['confidence'], pdFilt_parameters['confidence']['confidence_thresh'])
        base_correct_val = np.median(base_snipped)

        df = preprocess_pd_pipeline(df, base_correct_val)

        df_interview = df[df.situation == 0]
        window_length = settings['windowLength_sec'] * settings['samplingRate']  # number of samples in 5 seconds window: 125

        # EXTRACT STRESS WINDOWS IN INTERVIEW PHASE
        search_label = 'stress'
        start = 0
        counter_good = 0
        counter_bad = 0
        DL_out, counter_label, counter_bad, counter_good = extract_windows(df_interview, start, ID, search_label, DL_out, window_length, counter_good, counter_bad)
        print('ID ' + str(ID))
        print('     Stress windows: ' + str(counter_label))
        counter_stress_total += counter_label  # counter for stress windows over IDs

        post_interview_dat = df[df.situation == 2]
        post_start_idx = 0
        nostress_dat = post_interview_dat.iloc[post_start_idx+1:-1].copy()
        nostress_dat = preprocess_pd_pipeline(nostress_dat, base_correct_val)

        DL_out, counter_label, counter_bad, counter_good = extract_windows(nostress_dat, post_start_idx+1, ID, 'nostress', DL_out, window_length, counter_good, counter_bad)

        print('     Nostress windows: ' + str(counter_label))
        print('     Valid windows: ' + str(counter_good))
        print('     Bad quality windows: ' + str(counter_bad))
        print('------')
        counter_nostress_total += counter_label
        counter_valid += counter_good
        counter_badQuality += counter_bad


    print('Valid windows overall: ' + str(counter_valid))
    print('Bad quality windows overall: ' + str(counter_badQuality))
    print('Total checked windows: ' + str(counter_badQuality + counter_valid))
    print('.....')
    print('Total stress windows: ' + str(counter_stress_total))
    print('Total nostress windows: ' + str(counter_nostress_total))

    # Save dataframes to pickle and csv
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    DL_out.to_pickle(output_path + "DL_out.pkl")
    DL_out.to_csv(output_path + "DL_out.csv")
    print('Saved DL_out.csv to ' + output_path + 'DL_out.csv')
    unique_values = DL_out['ID'].unique()
    for value in unique_values:
        print(value)


if __name__ == "__main__":
    data_dir = "data/fordigitstress/"
    invalid_ids = [12, 15, 16, 18, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48]
    output_path = 'data/fordigitstress/dataframes/'
    main(data_dir, output_path, invalid_ids, settings)
