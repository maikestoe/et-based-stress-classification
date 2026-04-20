"""
Preprocessing for the VR goalkeeper dataset (vr_goalkeeper)

This script provides functions to preprocess raw data, filter pupil diameter signals, extract study phases,
compute gaze-related features, and segment baseline periods for the VR goalkeeper dataset (vr_goalkeeper). The main
function iterates through all participants, processes their data, and saves the results in dataframes.

Functions:
----------
- getLabel(task):
    Define the label of each study phase.

- preprocess_pd(raw_df, filt, eye, out_df):
    Filter the pupil diameter data.

- get_amyl(saliva_sheet, ID, phase):
    Retrieve and normalize S-amylase levels from saliva samples.

- plot_interp_orig(processed, df_interp, start_time, end_time, start, end):
    Plot the original and interpolated pupil diameter signals.

- fixationPipeline(df, settings, focused_objects):
    Compute fixations and related features using implemented methods.

- process_participants(raw_data_path, output_path, excluded_IDs):
    Process data for all participants and save results.

Dependencies:
-------------
- numpy
- pandas
- seaborn
- matplotlib
- segment_vr_goalkeeper (custom module)
- IPA_utils (custom module)
- fix_utils (custom module)
- pd_utils (custom module)
- Classes (custom module)

Author: Maike Laut
Date: 20.06.2024
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os

from matplotlib import pyplot as plt

import segment_vr_goalkeeper
import IPA_utils
import fix_utils as fix
from pd_utils import compute_meanDia, interpolate_and_resample, get_stats, get_slope, baseline_correct, downsample, downsample_mask_nearest, compute_single_preprocessed
from Classes import PdFilter, ipaSettings, lhipaSettings, fixationfilterSettings

# pd.set_option('display.max_rows', None)


def getLabel(task):
    """
    Define the label of each study phase.

    :param task: Name of the study phase.
    :type task: str
    :return: Tuple containing the label as a string (stress or nostress) and the integer representation of the label (0: nostress, 1: stress).
    :rtype: tuple(str, int)

    """

    if task == 'noStress':
        label = 'nostress'
        label_num = 0
    elif task == 'stress':
        label = 'stress'
        label_num = 1
    else:
        label = np.nan
        label_num = np.nan

    return label, label_num


def preprocess_pd(raw_df, filt, eye, out_df):
    """
    Filter the pupil diameter data.

    :param raw_df: Dataframe containing raw study data.
    :type raw_df: pandas.DataFrame
    :param filt: PupilDiameter filter containing filter parameters.
    :type filt: PdFilter
    :param eye: Indicates if the data of the left or right eye should be processed.
    :type eye: str
    :param out_df: Output dataframe accumulating output over time.
    :type out_df: pandas.DataFrame
    :return: DataFrame with added information on every filter step.
    :rtype: pandas.DataFrame
    """

    blinks = raw_df['blink ' + eye].to_numpy(dtype=bool, na_value=True)
    t = raw_df['normal time'].to_numpy(dtype=float, na_value=np.nan) * 1000  # Transform time vector to ms
    d = raw_df[eye + ' pupil size'].to_numpy(dtype=float, na_value=np.nan)

    isValidNan, isValidRemoveOoB, isValidRemoveConf, isValidSpeedFilter, isValid_Running, speedFiltData, filtData, totalTime \
        = filt.filter(blinks, t, d, np.nan)

    out_df.insert(2, 'ValidChangesPerProcessingStep' + str(eye), list(
        zip(isValidNan, isValidRemoveOoB, isValidSpeedFilter, isValid_Running)))

    out_df.insert(2, 'isValidFinal' + str(eye), isValid_Running)
    out_df.insert(2, 'SpeedFiltData' + str(eye), speedFiltData[0])

    filtData_zipped = list(zip(*filtData[0]))

    out_df.insert(2, 'filt_Data_DevFilterPerPass' + str(eye), filtData_zipped)
    filtData_BL_zipped = list(zip(*filtData[3]))

    out_df.insert(2, 'smoothBLPerPass' + str(eye), filtData_BL_zipped)

    return out_df


def get_amyl(saliva_sheet, ID, phase):
    """
    Retrieve and normalize alpha-amylase levels from saliva samples.

    :param saliva_sheet: Dataframe containing saliva sample data.
    :type saliva_sheet: pandas.DataFrame
    :param ID: Participant ID.
    :type ID: int
    :param phase: Study phase ('nostress' or 'stress').
    :type phase: str
    :return: Normalized alpha-amylase level.
    :rtype: float
    """

    saliva_id = saliva_sheet[saliva_sheet['participant ID'] == ID]
    saliva_id_amyl = saliva_id[saliva_id['type'] == 'S-amylase']

    b1 = saliva_id_amyl[saliva_id_amyl['phase'] == 4].values[0][3]

    if pd.isna(b1):
        b1 = saliva_id_amyl[saliva_id_amyl['phase'] == 1].values[0][3]

    if isinstance(b1, str):
        base_amyl = float(b1.replace(',', '.'))
    else:
        base_amyl = b1

    if phase == 'nostress':  # nostress phase
        place = 2
    else:  # stress phase
        place = 3

    b2 = saliva_id_amyl[saliva_id_amyl['phase'] == place].values[0][3]
    if isinstance(b2, str):
        amyl = float(b2.replace(',', '.')) / base_amyl
    else:
        amyl = b2 / base_amyl

    return amyl

def extract_fixed_length_signal(df, time_col, signal_col, start_time, fs, duration=5.0):
    """
    Extract a fixed-length signal window starting at the sample nearest to start_time.

    :param df: DataFrame containing time and signal columns.
    :type df: pandas.DataFrame
    :param time_col: Name of time column.
    :type time_col: str
    :param signal_col: Name of signal column.
    :type signal_col: str
    :param start_time: Desired window start time in seconds.
    :type start_time: float
    :param fs: Sampling frequency in Hz.
    :type fs: int
    :param duration: Window duration in seconds.
    :type duration: float
    :return: Fixed-length NumPy array.
    :rtype: numpy.ndarray
    """
    target_len = int(duration * fs)
    start_idx = segment_vr_goalkeeper.find_nearest(df[time_col], start_time)

    arr = df[signal_col].iloc[start_idx:start_idx + target_len].to_numpy(dtype=float, na_value=np.nan)

    if len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)), mode='constant', constant_values=np.nan)
    elif len(arr) > target_len:
        arr = arr[:target_len]

    return arr


def plot_interp_orig(processed, df_interp, start_time, end_time, start, end, dia_col):
    """
    Plot the original mean pupil diameter signal after filtering and pChip interpolation.

    :param processed: Dataframe containing processed pupil diameter data.
    :type processed: pandas.DataFrame
    :param df_interp: Dataframe containing interpolated and resampled pupil diameter data.
    :type df_interp: pandas.DataFrame
    :param start_time: Time corresponding to the start of the shot.
    :type start_time: float
    :param end_time: Time corresponding to the end of the shot.
    :type end_time: float
    :param start: Closest index to the start time for the interpolated signal.
    :type start: int
    :param end: Closest index to the end time for the interpolated signal.
    :type end: int
    :param dia_col: Column containing the interpolated/downsampled diameter.
    :type dia_col: str
    :return: None
    """

    plt.figure()
    start_old = segment_vr_goalkeeper.find_nearest(processed['normal time'], start_time)
    end_old = segment_vr_goalkeeper.find_nearest(processed['normal time'], end_time)

    # processed (blau)
    x2 = processed['normal time'].iloc[start_old:end_old].to_numpy(dtype=float, na_value=np.nan)
    y2 = processed['meanDia'].iloc[start_old:end_old].to_numpy(dtype=float, na_value=np.nan)

    # df_interp (grün)
    x1 = df_interp['normal time'].iloc[start:end].to_numpy(dtype=float, na_value=np.nan)
    y1 = df_interp[dia_col].iloc[start:end].to_numpy(dtype=float, na_value=np.nan)

    plt.plot(x1, y1, linewidth=0.4, color='green')
    plt.plot(x2, y2, linewidth=0.4, color='blue')
    plt.show()


def fixationPipeline(df, settings, focused_objects):
    """
    Compute fixations and related features of the signal using all implemented methods.

    :param df: Gaze data.
    :type df: pandas.DataFrame
    :param settings: Fixation computation settings as used method.
    :type settings: dict
    :param focused_objects: Strings with focused objects defined by colliders in the environment.
    :type focused_objects: pandas.Series
    :return: Tuple containing fixation features and time series signals for the deep learning pipeline.
    :rtype: tuple(dict, dict)
    """

    # remove blinks and samples around blinks
    #eyedata = fix.preprocessing_pd(df.reset_index(),
    #                               settings)
    eyedata = df[['time', 'eye-x', 'eye-y', 'eye-z', 'head-x', 'head-y', 'head-z', 'animation', 'normal time', 'ID',
                  'blink left', 'blink right']]
    settings['length'] = len(eyedata) - 1
    # print(settings['length'])

    eye = np.zeros([9, len(eyedata)])
    head = np.zeros([3, len(eyedata)])
    timestamp = np.array(eyedata['normal time'])
    head[0:3] = eyedata[['head-x', 'head-y', 'head-z']].T
    eye[0:3] = eyedata[['eye-x', 'eye-y', 'eye-z']].T

    # convert the gaze vector from x-,y-,z- coordinates to the visual angle
    position = fix.getPosition(eye, head, settings, offset=[0, 0, 51.96])

    # get the asymptotic model based on the visual angle and search for saccades with a predefined threshold
    asymptotic_model = fix.asymptoticModel(position, settings)
    fix.detectSaccadesVelocity(asymptotic_model, eye, settings, 'asymptotic_model')  # results in eye[3]

    # get the angular velocity based on the visual angle and search for saccades with a predefined threshold
    velocity = fix.getVelocity(position, settings)
    #    detection_method_time_series = velocity
    fix.detectSaccadesVelocity(velocity, eye, settings, 'velocity')  # results in eye[4]

    # get the angular acceleration based on the angular velocity and search for saccades with an adaptive threshold
    acceleration = fix.getAcceleration(velocity, settings)
    fix.getSaccadesAcceleration(acceleration, eye, settings)  # results in eye[5]

    # get mean number of fixations and fixation duration
    fix_features, fix_array = fix.getFixations('asymptotic_model', eye, timestamp, settings, focused_objects)
    fix_features_acc, fix_array_acc = fix.getFixations('acceleration', eye, timestamp,  settings, focused_objects)
    fix_features.update(fix_features_acc)
    fix_features_vel, fix_array_vel = fix.getFixations('velocity', eye, timestamp,  settings, focused_objects)
    fix_features.update(fix_features_vel)

    # Padding of arrays for DL -> 450 samples corresponds to 5s with 90Hz sampling frequency
    DL_fix = {
        'velocity': np.pad(velocity, (0, 450 - len(velocity)), mode='constant'),
        'acceleration': np.pad(acceleration, (0, 450 - len(acceleration)), mode='constant'),
        'position': np.pad(position, (0, 450 - len(position)), mode='constant'),
        'asymptotic_model': np.pad(asymptotic_model, (0, 450 - len(asymptotic_model)), mode='constant'),
        'fix_array': np.pad(fix_array.astype(int), (0, 450 - len(fix_array)), mode='constant')
    }
    # saccades raw info for all methods could be added to output (eye[3:6])
    return fix_features, DL_fix


def process_participants(raw_data_path, output_path, excluded_IDs, target_fs=None):
    """
    Iterate through all participants, segment the data, extract labels and compute gaze-related features and time
    series signals for stress classification. The resulting dataframes are saved in a pickle and a csv file.

    :param raw_data_path: Path where the dataset is stored.
    :type raw_data_path: str
    :param output_path: Path where the extracted dataframes are stored.
    :type output_path: str
    :param excluded_IDs: IDs with invalid data to be excluded from further analysis.
    :type excluded_IDs: list
    :return: None
    :rtype: None
    """

    if target_fs is None:
        target_fs = [90]
    tasks = ['noStress', 'stress']  # tutorialeye
    eyes = ['left', 'right']

    # Define columns of output dataframe for DL
    DL_columns = ['ID', 'lab_str', 'lab_num', 'shot', 'meanDia']
    # Initialize an empty output DataFrame for DL
    DL_out = pd.DataFrame(columns=DL_columns)

    # Define columns of output dataframe for features
    features_columns = ['ID', 'lab_str', 'lab_num', 'shot']
    # Initialize an empty output DataFrame for DL
    features_out = pd.DataFrame(columns=features_columns)

    # Load saliva information. Older local exports used result.xlsx; the public
    # dataset uses meta.xlsx.
    meta_file = raw_data_path + 'result.xlsx'
    if not os.path.exists(meta_file):
        meta_file = raw_data_path + 'meta.xlsx'
    results_file = pd.ExcelFile(meta_file, engine='openpyxl')
    saliva_sheet = pd.read_excel(results_file, 'Saliva')

    # Decide which baseline to use
    base = 'postBaseline'  # (post)baseline: 'postBaseline' - (pre)baseline: 'baseline'

    # Iterate over participants
    for ID in range(0, 30):
        if ID in excluded_IDs:
            continue  # skip data with missing no stress or post-baseline phase data

        print('----- Processing ID ' + str(ID) + ' -------------------------')

        # Load baseline
        base_df = pd.read_csv(raw_data_path + 'LogID_' + str(ID) + '_' + base + '.csv', sep=';', skip_blank_lines=True, low_memory=False)
        # Convert object cols to correct types
        base_df = base_df.dropna(how="all")
        base_df = base_df.convert_dtypes()
        processed_base = base_df[['normal time', 'eye-x']].copy()
        processed_base['ID'] = ID
        processed_base.ID = (np.ones(len(base_df.time)) * ID).astype(int)
        processed_base_unfilt = base_df.dropna(how="all").convert_dtypes()
        processed_base_unfilt['ID'] = ID
        processed_base_unfilt.ID = (np.ones(len(base_df.time)) * ID).astype(int)
        # Filter the baseline to get a mean signal for both eyes
        # create filter with default settings
        filt_base = PdFilter('default')
        ##############################################

        # Iterate over both eyes
        for eye in eyes:
            print('--------------- ' + eye + ' Eye ----------------------')

            ##############################################
            # preprocess eyes individually
            processed_base = preprocess_pd(base_df, filt_base, eye, processed_base)
            ##############################################
            # compute mean pupil diameter

        processed_base = compute_meanDia(base_df, processed_base)

        # interpolate single eye signal
        processed_base_single = compute_single_preprocessed(base_df, processed_base)

        base_correct_vals = {}  # fs -> baseline median  # for single: .5

        # Standard (90 Hz, ohne Downsampling)
        base_snipped, _ = segment_vr_goalkeeper.segment_baseline(
            processed_base["meanDia"], processed_base["normal time"], processed_base["bothwithout"]
        )

        base_snipped_single, _ = segment_vr_goalkeeper.segment_baseline(
            processed_base_single["right_pupil"], processed_base_single["normal time"], ~processed_base["isValidFinalright"]
        )

        base_snipped_unfilt = segment_vr_goalkeeper.segment_baseline_unfilt(processed_base_unfilt["right pupil size"], processed_base_unfilt["normal time"])

        base_correct_vals[90] = float(np.nanmedian(np.asarray(base_snipped, dtype=float)))
        base_correct_vals[90.5] = float(np.nanmedian(np.asarray(base_snipped_single, dtype=float)))
        base_correct_vals[90.05] = float(np.nanmedian(np.asarray(base_snipped_unfilt, dtype=float)))

        # Weitere fs
        for fs in target_fs:
            fs = int(fs)
            if fs == 90:
                continue

            base_ds = downsample(processed_base, "normal time", "meanDia", target_fs=fs)
            base_ds_single = downsample(processed_base_single, "normal time", "right_pupil", target_fs=fs)
            base_ds_unfilt = downsample(processed_base_unfilt, "normal time", "right pupil size", target_fs=fs)

            # downsample mask
            bothwithout_ds = downsample_mask_nearest(
                t_src=processed_base["normal time"].to_numpy(dtype=float),
                mask_src=processed_base["bothwithout"],
                t_dst=base_ds["normal time"].to_numpy(dtype=float),
            )

            isValidFinalright_ds = downsample_mask_nearest(
                t_src=processed_base_single["normal time"].to_numpy(dtype=float),
                mask_src=processed_base_single["isValidFinalright"],
                t_dst=base_ds_single["normal time"].to_numpy(dtype=float),
            )

            base_snipped_ds, _ = segment_vr_goalkeeper.segment_baseline(
                base_ds[f"meanDia_downsampled_{fs}"],
                base_ds["normal time"],
                bothwithout_ds,
            )

            base_snipped_ds_single, _ = segment_vr_goalkeeper.segment_baseline(
                base_ds_single[f"right_pupil_downsampled_{fs}"],
                base_ds_single["normal time"],
                ~isValidFinalright_ds
            )

            base_correct_vals[fs] = float(np.nanmedian(np.asarray(base_snipped_ds, dtype=float)))
            base_correct_vals[fs+0.5] = float(np.nanmedian(np.asarray(base_snipped_ds_single, dtype=float)))
            base_correct_vals[fs + 0.05] = float(np.nanmedian(np.asarray(base_snipped_ds_single, dtype=float)))

        # Iterate over study phases
        for phase in range(0, 2):
            task = tasks[phase]
            label, label_num = getLabel(task)
            print('---------- Task ' + str(label) + ' ----------------------')

            #############################
            # Import data
            # Get amylase measurement
            amyl = get_amyl(saliva_sheet, ID, label)

            raw_df = pd.read_csv(raw_data_path + 'LogID_' + str(ID) + '_' + task + '.csv', sep=';', skip_blank_lines=True, low_memory=False)[['eye-x',
                                                                                                     'eye-y', 'eye-z',
                                                                                                     'time',
                                                                                                     'left pupil size',
                                                                                                     'right pupil size',
                                                                                                     'blink left',
                                                                                                     'blink right',
                                                                                                     'animation',
                                                                                                     'tmp stressor',
                                                                                                     'Task',
                                                                                                     'Task-Level',
                                                                                                     'Task-Score',
                                                                                                     'Task-Response time',
                                                                                                     'head-x', 'head-y',
                                                                                                     'head-z',
                                                                                                     'normal time',
                                                                                                     'tmp stressor',
                                                                                                     'focused Object'
                                                                                                     ]]

            raw_df = raw_df.dropna(how="all")
            raw_df = raw_df.convert_dtypes()
            raw_df['ID']=ID
            raw_df.ID = (np.ones(len(raw_df.time)) * ID).astype(int)

            ###########################
            # Normal time -> sampling frequency of 90Hz (refresh rate of the HMD)
            # Initiate output to save processed pd data
            processed = raw_df[['ID', 'normal time', 'eye-x']].copy()
            print('##################')

            ##############################################

            # create filter with default settings
            filt = PdFilter('default')
            ##############################################

            # Iterate over both eyes
            for eye in eyes:
                print('--------------- ' + eye + ' Eye ----------------------')

                ##############################################
                # preprocess eyes individually
                processed = preprocess_pd(raw_df, filt, eye, processed)

                ##############################################
                # compute mean pupil diameter
            processed = compute_meanDia(raw_df, processed)
            processed_single = compute_single_preprocessed(raw_df, processed)
            #####################################
            # Segment the single shots
            start_shots, end_shots = segment_vr_goalkeeper.find_shots(raw_df)
            # Interpolate and resample the meanDiameter signal and the time to regular sampling of 90Hz
            df_interp = interpolate_and_resample(processed, 'normal time', 'meanDia')
            df_interp_single = interpolate_and_resample(processed_single, 'normal time', 'right_pupil')
            df_unfilt = interpolate_and_resample(raw_df, 'normal time', 'right pupil size')

            df_ds_by_fs = {}
            df_single_ds_by_fs = {}
            df_unfilt_by_fs = {}
            for fs in target_fs:
                fs = int(fs)
                if fs == 90:
                    continue
                df_ds_by_fs[fs] = downsample(df_interp, "normal time", "meanDia_resampled", target_fs=fs)
                df_single_ds_by_fs[fs] = downsample(df_interp_single, "normal time", "right_pupil_resampled", target_fs=fs)
                df_unfilt_by_fs[fs] = downsample(df_unfilt, "normal time", "right pupil size_resampled", target_fs=fs)

            # Iterate over individual shots
            for s in range(0, len(start_shots)):

                print('Processing shot ' + str(s))

                # Correct the start time to assure same length before shot and only consider starting with the prep time
                start_time = raw_df['normal time'][
                                 end_shots[s]] - 5  # normal time represents time in seconds

                end_time = raw_df['normal time'][end_shots[s]]
                single_shot_90 = extract_fixed_length_signal(
                    df=df_interp_single,
                    time_col="normal time",
                    signal_col="right_pupil_resampled",
                    start_time=start_time,
                    fs=90,
                    duration=5.0
                )

                start_raw = segment_vr_goalkeeper.find_nearest(processed['normal time'], start_time)
                end_raw = segment_vr_goalkeeper.find_nearest(processed['normal time'], end_time)

                # COMPUTE PD FEATURES AND SIGNALS
                meanDia_shot_90 = extract_fixed_length_signal(
                    df=df_interp,
                    time_col="normal time",
                    signal_col="meanDia_resampled",
                    start_time=start_time,
                    fs=90,
                    duration=5.0
                )

                time_shot_90 = extract_fixed_length_signal(
                    df=df_interp_single,
                    time_col="normal time",
                    signal_col="normal time",
                    start_time=start_time,
                    fs=90,
                    duration=5.0
                )

                unfilt_shot_90 = extract_fixed_length_signal(
                    df=df_unfilt,
                    time_col="normal time",
                    signal_col="right pupil size_resampled",
                    start_time=start_time,
                    fs=90,
                    duration=5.0
                )

                if len(single_shot_90) != 450 or len(unfilt_shot_90) != 450:
                    if len(single_shot_90) != 450:
                        print("SINGLE PROBLEM")
                    elif len(unfilt_shot_90) != 450:
                        print("UNFILT PROBLEM")
                meanDia_corr_90 = baseline_correct(meanDia_shot_90, base_correct_vals[90])
                single_shot_corr_90 = baseline_correct(single_shot_90, base_correct_vals[90.5])
                unfilt_shot_corr_90 = baseline_correct(unfilt_shot_90, base_correct_vals[90.05])

                pd_stats = get_stats(meanDia_shot_90)
                pd_slope = get_slope(meanDia_shot_90, time_shot_90)
                # Plot the interpolated signal and original signal to see influence of interpolation
                #plot_interp_orig(processed, df_interp, start_time, end_time, start_interp, end_interp, 'meanDia_resampled')

                lhipa = IPA_utils.compute_lhipa(meanDia_shot_90, end_time - start_time, lhipaSettings)
                ipa = IPA_utils.compute_ipa(meanDia_shot_90, end_time - start_time, ipaSettings)

                # COMPUTE FIXATION FEATURES AND SIGNALS
                raw_df_shot = raw_df[start_raw:end_raw]
                lhipa_raw_l = IPA_utils.compute_lhipa(raw_df_shot['right pupil size'], end_time - start_time, lhipaSettings)
                ipa_raw_l = IPA_utils.compute_ipa(raw_df_shot['right pupil size'], end_time - start_time, ipaSettings)

                fix_features, DL_fix = fixationPipeline(raw_df_shot, fixationfilterSettings, raw_df_shot['focused Object'])

                # UPDATE DATAFRAMES
                # Create a new row with the extracted data for DL

                new_row_DL = {
                    'ID': ID,
                    'lab_str': label,
                    'lab_num': label_num,
                    'shot': s,
                    'meanDia_fs90': meanDia_shot_90,
                    'meanDia_corrected_fs90': meanDia_corr_90,
                    'single_corrected_fs90': single_shot_corr_90,
                    'unfilt_corrected_fs90': unfilt_shot_corr_90,
                    'time_fs90': time_shot_90
                }

                # Append downsampled PD signals
                for fs in target_fs:
                    fs = int(fs)
                    if fs == 90:
                        continue

                    df_ds = df_ds_by_fs[fs]
                    df_ds_single = df_single_ds_by_fs[fs]
                    df_ds_unfilt = df_unfilt_by_fs[fs]
                    single_shot_fs = extract_fixed_length_signal(
                        df=df_ds_single,
                        time_col="normal time",
                        signal_col=f"right_pupil_resampled_downsampled_{fs}",
                        start_time=start_time,
                        fs=fs,
                        duration=5.0
                    )

                    meanDia_shot_fs = extract_fixed_length_signal(
                        df=df_ds,
                        time_col="normal time",
                        signal_col=f"meanDia_resampled_downsampled_{fs}",
                        start_time=start_time,
                        fs=fs,
                        duration=5.0
                    )

                    unfilt_shot_fs = extract_fixed_length_signal(
                        df=df_ds_unfilt,
                        time_col="normal time",
                        signal_col=f"right pupil size_resampled_downsampled_{fs}",
                        start_time=start_time,
                        fs=fs,
                        duration=5.0
                    )
                    time_shot_fs = extract_fixed_length_signal(
                        df=df_ds,
                        time_col="normal time",
                        signal_col="normal time",
                        start_time=start_time,
                        fs=fs,
                        duration=5.0
                    )

                    meanDia_corr_fs = baseline_correct(meanDia_shot_fs, base_correct_vals[fs])
                    single_corr_fs = baseline_correct(single_shot_fs, base_correct_vals[fs+0.5])
                    unfilt_corr_fs = baseline_correct(unfilt_shot_fs, base_correct_vals[fs+0.05])

                    new_row_DL[f"meanDia_fs{fs}"] = meanDia_shot_fs
                    new_row_DL[f"meanDia_corrected_fs{fs}"] = meanDia_corr_fs
                    new_row_DL[f"single_corrected_fs{fs}"] = single_corr_fs
                    new_row_DL[f"unfilt_corrected_{fs}"] = unfilt_corr_fs
                    new_row_DL[f"single_fs{fs}"] = single_shot_fs
                    new_row_DL[f"unfilt_fs{fs}"] = unfilt_shot_fs
                    new_row_DL[f"time_fs{fs}"] = time_shot_fs

                # Append the new row to the DataFrame
                new_row_DL.update(DL_fix)
                DL_out = pd.concat([DL_out, pd.DataFrame([new_row_DL])], ignore_index=True)

                # Create a new row with the extracted data for feature-based methods
                new_row_features = {
                    'ID': ID,
                    'lab_str': label,
                    'lab_num': label_num,
                    'shot': s,
                    'amylase': amyl,
                    'lhipa': lhipa,
                    'ipa': ipa,
                    'lhipa_raw_l': lhipa_raw_l,
                    'ipa_raw_l': ipa_raw_l
                }
                new_row_features.update(pd_stats)
                new_row_features.update(pd_slope)
                new_row_features.update(fix_features)
                # Append the new row to the DataFrame
                features_out = pd.concat([features_out, pd.DataFrame([new_row_features])], ignore_index=True)

    # Save dataframes to pickle and csv
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    print(DL_out.columns)
    DL_out.to_pickle(output_path + "DL_out.pkl")
    DL_out.to_csv(output_path + "DL_out.csv")
    print('Saved DL_out.csv to ' + output_path + 'DL_out.csv')

    features_out.to_pickle(output_path + "features_out.pkl")
    features_out.to_csv(output_path + "features_out.csv")
    print('Saved features_out.csv to ' + output_path + 'features_out.csv')


if __name__ == "__main__":
    # Process information
    eyes = ['left', 'right']
    raw_data_path = 'data/vr_goalkeeper/'
    output_path = 'data/vr_goalkeeper/dataframes_downsampled/'
    excluded_IDs = [5, 25, 28]

    process_participants(raw_data_path, output_path, excluded_IDs, target_fs=[15,25,30,45,90])
