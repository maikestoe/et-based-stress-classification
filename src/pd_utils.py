################################################################################
# Adapted from Kret and Sjak-Shie, Behav Res, 2019 (https://github.com/ElioS-S/pupil-size)
#
# Filter implemented after Kret and Sjak-Shie, Behav Res, 2019
# (https://link.springer.com/article/10.3758/s13428-018-1075-y)
################################################################################

"""
Preprocessing and Filtering Pupil Diameter Data

This script provides functions to preprocess and filter pupil diameter data based on various criteria, including
dilation speed, deviation from a smooth baseline, and out-of-bounds values. It also includes functions for interpolating
and resampling data, as well as calculating statistics and baseline correction.

Functions:
----------
- madCalc(d, n):
    Calculates the rejection threshold using the mad method.

- expandGaps(t, isValid_in, filterSettings):
    Function for removing samples around gaps.

- madSpeedFilter(t, d, isValid_in, filterSettings):
    Filters a diameter timeseries based on the dilation speeds.

- madDeviationFilter(t_ms, dia, isValid_In, filterSettings):
    Filters a diameter timeseries based on the deviation from a smooth baseline.

- removeLoners(t_ms, validzIn, filtSettings):
    Function for removing isolated sections of data.

- deviationCalculator(t_ms, dia, isValid_In, tInterp, smoothFiltA, smoothFiltB):
    Function for calculating deviation metrics.

- removeOutOfBounds(t, d, isValid_in, filterSettings):
    Removes samples that are not within the acceptable range.

- removeLowConfidence(t, c, isValid_in, filterSettings):
    Remove samples where a low confidence value was measured.

- getTotalTime(t, isValid):
    Compute the total duration of a shot.

- interpolate_pd_single(processed_df, interp_type='pChip'):
    Interpolate pupil diameter data for a single eye.

- compute_meanDia(raw_df, processed_df, interp_type='pChip'):
    Compute the mean pupil diameter.

- interpolate_and_resample(dataframe, time_col, diameter_col):
    Interpolate missing values and resample the data to 90Hz.

- get_stats(meanDia):
    Calculates pupil diameter related statistics.

- get_slope(meanDia, time):
    Calculate the slope of the mean pupil diameter data.

- baseline_correct(meanDia_shot, base_correct_val):
    Divisive baseline correction of mean pupil diameter data.

Dependencies:
-------------
- numpy
- pandas
- scipy
- matplotlib

Author: Maike Laut
Date: 20.06.2024
"""

import numpy as np
import pandas as pd
import scipy.signal as scipy_signal
from scipy import interpolate, signal
import scipy.stats as st
from matplotlib import pyplot as plt
from statistics import harmonic_mean
from fractions import Fraction


def madCalc(d, n):
    """
    Calculates the rejection threshold using the mad method.

    :param d: Max dilation speeds (speedFilter) or residuals per pass (madDeviationFilter).
    :type d: numpy.ndarray
    :param n: MAD multiplier.
    :type n: int

    :returns: Median, Median Absolute Deviation (MAD), Calculated threshold.
    :rtype: tuple(float, float, float)
    """

    # calculates the rejection threshold using the mad method
    notnan = ~np.isnan(d)
    d = d[notnan]
    #calculate the median
    med_d = np.median(d)
    notnan = ~np.isnan(med_d)
    med_d = med_d[notnan]

    # Calculate the mad
    mad = np.median(abs(d-med_d))

    # Calculate the threshold
    thresh = med_d + (n * mad)

    return med_d, mad, thresh


def expandGaps(t, isValid_in, filterSettings):
    """
    Function for removing samples around gaps.

    :param t: Time in ms.
    :type t: numpy.ndarray
    :param isValid_in: Indices of valid samples.
    :type isValid_in: numpy.ndarray of np.bool
    :param filterSettings: Settings for different filter types.
    :type filterSettings: dict

    :returns: Indices of rejected samples.
    :rtype: numpy.ndarray
    """

    # Get settings
    minGap = filterSettings['gaps']['minGap']
    maxGap = filterSettings['gaps']['maxGap']
    backPadding = filterSettings['gaps']['backPadding']
    fwdPadding = filterSettings['gaps']['fwdPadding']

    valid_t = t[isValid_in]
    valid_idx = np.where(isValid_in)[0]

    # Calculate the duration of each gap and test if it exceeds the threshold
    if ~np.isnan(minGap) or ~np.isnan(maxGap):

        gaps = np.diff(valid_t)

        a = gaps > minGap
        b = gaps < maxGap

        isGapThatNeedsPadding = np.logical_and(a, b)

        # get the start and end times of each gap

        gapStartTimes = valid_t[np.append(isGapThatNeedsPadding, False)]
        gapEndTimes = valid_t[np.append(False, isGapThatNeedsPadding)]

        # Padd gaps that need padding
        if backPadding > 0 or fwdPadding > 0:
            # Detect samples around the gaps

            isNearGap = []

            for i in range(0, len(gapEndTimes)):
                a = np.where(valid_t > gapStartTimes[i] - backPadding)[0]
                b = np.where(valid_t < gapEndTimes[i] + fwdPadding)[0]
                isNearGap.append(np.intersect1d(a, b))

            if isNearGap != []:
                isNearGap = np.concatenate(isNearGap).ravel()

            # Reject samples too near a gap
            isValid_in[valid_idx[isNearGap]]==False

    return isValid_in


def madSpeedFilter(t, d, isValid_in, filterSettings):
    """
    Filters a diameter timeseries based on the dilation speeds for blink detection and different artifacts resulting
    in large gaps between adjacent samples.

    :param d: Pupil diameter data.
    :type d: numpy.ndarray
    :param t: Time in ms.
    :type t: numpy.ndarray
    :param isValid_in: Indices of valid samples.
    :type isValid_in: numpy.ndarray of np.bool
    :param filterSettings: Settings for different filter types.
    :type filterSettings: dict

    :returns: Indices of rejected samples, Dilation speed data, Number of rejected samples.
    :rtype: tuple(numpy.ndarray, list, int)
    """

    # filters a diameter timeseries based on the dilation speeds

    # get parameters and current data
    maxCalcDist = filterSettings['dilationSpeed']['max_gap']
    multiplier = filterSettings['dilationSpeed']['multiplier']
    curDiameters = d[isValid_in]
    cur_t_ms = t[isValid_in]
    maxDilationSpeeds = np.full_like(d, np.nan)

    # Calculate the dilation speeds:
    curDilationSpeeds = np.diff(curDiameters)/np.diff(cur_t_ms)

    # The maximum gap over which a change is considered:
    curDilationSpeeds[np.diff(cur_t_ms) > maxCalcDist] = np.nan

    # Generate a two column array with the back and forward dilation speeds:
    backFwdDilations = np.array([np.insert(curDilationSpeeds, 0, 0),
                                 np.insert(curDilationSpeeds, -1, 0)])

    # Calculate the deviation per sample:
    maxDilationSpeeds[isValid_in] = np.max(abs(backFwdDilations), axis=0)

    # Calculate the MAD stats:
    med_d, mad, thresh = madCalc(maxDilationSpeeds, multiplier)

    # Determine the outliers
    isValid_out = isValid_in & (maxDilationSpeeds <= thresh)

    # Remove remaining islands:
    isValid_out = removeLoners(t, isValid_out, filterSettings)

    # Blinks and other artifact with large inter-sample differences may exhibit distort the signal surrounding samples
    # that do not exceed the filter criteria. As such, remove samples surrounding gaps of a certain size

    isValid_out = expandGaps(t, isValid_out, filterSettings)

    # Set output
    speedFiltData = [maxDilationSpeeds, mad, thresh, med_d]

    # Feedback
    #print('Dilation Filter: ' + str(sum(~isValid_out & isValid_in)) + ' samples removed.\n')

    return isValid_out, speedFiltData, sum(~isValid_out & isValid_in)


def madDeviationFilter(t_ms, dia, isValid_In, filterSettings):
    """
    Filters a diameter timeseries based on the deviation from a smooth baseline.

    :param dia: Pupil diameter data.
    :type dia: numpy.ndarray
    :param t_ms: Time in ms.
    :type t_ms: numpy.ndarray
    :param isValid_In: Indices of valid samples.
    :type isValid_In: numpy.ndarray of np.bool
    :param filterSettings: Settings for different filter types.
    :type filterSettings: dict

    :returns: Indices of rejected samples, Filter data.
    :rtype: tuple(numpy.ndarray, list)
    """

    # filters a diameter timeseries based on the deviation from a smooth trendline

    # Get settings
    Npasses = filterSettings['deviation']['Npasses']
    interpFs = filterSettings['deviation']['interpFs']
    lowpassCF = filterSettings['deviation']['lowpassCF']
    madMultiplier = filterSettings['deviation']['multiplier']
    if np.isnan(t_ms[-1]):
        tInterp = np.arange(t_ms[0], t_ms[-2], int(1000/interpFs))
    else:
        tInterp = np.arange(t_ms[0], t_ms[-1], int(1000/interpFs))

    smoothFiltB, smoothFiltA = scipy_signal.butter(1, lowpassCF / interpFs / 2)

    # Create a local copy of the valid samples, return it if there is not
    # enough data for the filter calculations:
    isValid_Running = isValid_In

    if sum(isValid_In) < 3:
        filtData = []
    else:
        # Remove the previously rejected samples, these are no longer to be
        # considered:
        dia[~isValid_In] = np.nan

        # Preallocate
        isValid_Running = isValid_In
        isValidPerPass = np.zeros([Npasses, len(isValid_In)], dtype=bool)
        residualsPerPass = np.zeros([Npasses, len(isValid_In)])
        threshPerPass = np.zeros([Npasses, 1])
        smoothBaselinePerPass = np.zeros([Npasses, len(isValid_In)])

        # Break if the filter is not doing anything:
        isDone = False

        # Allow for multiple passes:
        for passIndx in range(0, Npasses):

            # If the last filter step did not have any effect, neither will this one
            if isDone:
                continue

            # Track the validity
            isValid_Start = isValid_Running

            # Calculate the smooth baseline and deviations therefrom

            residualsPerPass[passIndx], smoothBaselinePerPass[passIndx] = deviationCalculator(t_ms, dia,
                                                                                          isValid_Running & isValid_In,
                                                                                          tInterp, smoothFiltA,
                                                                                          smoothFiltB)

            # Calculate the MAD stats:
            med_d, mad, threshPerPass[passIndx] = madCalc(residualsPerPass[passIndx], madMultiplier)

            # Identify the outliers, and run the isolated sample rejection filter
            isValid_Running = (residualsPerPass[:][passIndx] <= threshPerPass[passIndx]) & isValid_In
            isValid_Running = removeLoners(t_ms, isValid_Running, filterSettings)

            # Log loop vars
            isValidPerPass[:][passIndx] = isValid_Running
            residualsPerPass[:][passIndx] = residualsPerPass[:][passIndx]

            # Determine if this filter step had an effect
            if passIndx > 1 & all(isValid_Start == isValid_Running): #& passIndx != (Npasses - 1):

                # Copy the current results to teh other columns
                isValidPerPass[:][passIndx + 1:] = np.tile(isValid_Running, ((Npasses - 1 - passIndx), 1))
                residualsPerPass[:][passIndx + 1:] = np.tile(residualsPerPass[passIndx], ((Npasses - 1 - passIndx), 1))

                smoothBaselinePerPass[:][passIndx + 1:] = np.tile(smoothBaselinePerPass[passIndx],
                                                               ((Npasses - 1 - passIndx), 1))

                threshPerPass[:][passIndx + 1:] = np.tile(threshPerPass[passIndx], ((Npasses - 1 - passIndx), 1))

                isDone = True

        # Set output
        filtData = [isValidPerPass, residualsPerPass, threshPerPass, smoothBaselinePerPass]

    return isValid_Running, filtData


def removeLoners(t_ms, validzIn, filtSettings):
    """
    Function for removing isolated sections of data.

    :param t_ms: Time in ms.
    :type t_ms: numpy.ndarray
    :param validzIn: Indices of valid samples.
    :type validzIn: numpy.ndarray of np.bool
    :param filtSettings: Contains filter settings.
    :type filtSettings: dict

    :returns: Indices of rejected samples.
    :rtype: numpy.ndarray of np.bool
    """

    t_copy = t_ms
    maxSep = filtSettings['isolatedSample']['islandSeperation_ms']
    minIslandWidth = filtSettings['isolatedSample']['minIslandWidth_ms']

    # Isolate the usable samples
    validIndx = np.where(validzIn)[0]
    tValidz = t_copy[validzIn]
    # Return if there are not enough samples
    if sum(validzIn) < 3:
        validzOut = validzIn
    else:
        theSea = np.diff(tValidz) > maxSep
        if theSea.size > 0:

            theSeaShoreLeft = np.append(True, theSea)
            # theSeaShoreRight = np.append(theSea,True)

            # Place samples into bins (islands), correct for edge exclusion:
            islandBinz = tValidz[theSeaShoreLeft]
            islandNum = np.digitize(tValidz, islandBinz)

            # Detect small islands, and their samples
            tinyIslands = np.argwhere((np.diff(islandBinz)) < minIslandWidth).ravel()
            tinyIslanders = np.isin(islandNum, tinyIslands)

            # Assign output (map the valid samples back to the original series)
            validzOut = validzIn

            validzOut[validIndx[tinyIslanders]] = False

        else:
            validzOut = validzIn
    return validzOut


def deviationCalculator(t_ms, dia, isValid_In, tInterp, smoothFiltA, smoothFiltB):
    """
    Function for calculating deviation metrics.

    :param dia: Pupil diameter data.
    :type dia: numpy.ndarray
    :param t_ms: Time.
    :type t_ms: numpy.ndarray
    :param isValid_In: Indices of valid samples.
    :type isValid_In: numpy.ndarray of np.bool
    :param tInterp: Interpolated time vector.
    :type tInterp: numpy.ndarray
    :param smoothFiltA: Numerator of Butterworth filter.
    :type smoothFiltA: numpy.ndarray
    :param smoothFiltB: Denominator of Butterworth filter.
    :type smoothFiltB: numpy.ndarray

    :returns: Deviation values and smooth baseline.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """

    # Extract currently valid data
    assert len(t_ms) == len(isValid_In), 'Vectors t_ms & isValid_In do not agree.'
    assert len(dia) == len(isValid_In), 'Vectors dia & isValid_In do not agree.'

    diaValid = dia[isValid_In & ~np.isnan(dia)]
    tValid = t_ms[isValid_In & ~np.isnan(dia)]

    # Generate smooth signal using linear interpolation, and nearest neighbour extrapolation
    # Use only the currently valid samples
    uniformBaseline = np.interp(tInterp, tValid, diaValid)

    # Low pass filter the uniform signal and map it back to the original timevector
    smoothUniformBaseline = scipy_signal.filtfilt(smoothFiltB, smoothFiltA, uniformBaseline)
    smoothBaseline = np.interp(t_ms, tInterp, smoothUniformBaseline)
    #smoothBaseline = scipy_signal.resample(smoothUniformBaseline, len(t_ms))

    # Calculate the deviation
    dev = abs(dia - smoothBaseline)

    return dev, smoothBaseline


def removeOutOfBounds(t, d, isValid_in, filterSettings):
    """Removes samples that are not within the acceptable range.
    Args:
        d (np.ndarray): pupil diameter data in cm.
        t (np.ndarray): time.
        isValid_in (np.ndarray of np.bool): Indices of valid samples.
        filterSettings (dict): filter parameters.
    Returns:
        isValid_out (np.ndarray of np.bool): Indices of rejected samples.
        num_rejected (int): Number of rejected samples in this filter step.
        """

    max_val = filterSettings['d_max']
    min_val = filterSettings['d_min']

    inval = filterSettings['invalid']
    assert(max_val > min_val)
    # Find out of range samples
    too_large = d > max_val
    too_small = (d < min_val) & (d != inval)
    invalid = d == inval

    isValid_out = ~too_large & ~too_small & ~np.isnan(d) & ~invalid & isValid_in
    assert not np.any(np.diff(t[isValid_out]) == 0)
    #print('Range Filter before remove Loners: ' + str(sum(~isValid_out & isValid_in)) + ' samples removed.\n')

    isValid_out = removeLoners(t, isValid_out, filterSettings)

    # Feedback
    #print('Range Filter after remove Loners: ' + str(sum(~isValid_out & isValid_in)) + ' samples removed.\n')

    num_rejected = sum(~isValid_out & isValid_in)

    return isValid_out, num_rejected


def removeLowConfidence(t, c, isValid_in, filterSettings):
    """
    Remove samples where a low confidence value (below the specified threshold) was measured.

    :param t: Time.
    :type t: numpy.ndarray
    :param c: Confidence.
    :type c: numpy.ndarray
    :param isValid_in: Indices of valid samples.
    :type isValid_in: numpy.ndarray of np.bool
    :param filterSettings: Filter parameters.
    :type filterSettings: dict

    :returns: Indices of rejected samples, Number of rejected samples.
    :rtype: tuple(numpy.ndarray of np.bool, int)
    """
    thresh = filterSettings['confidence']['confidence_thresh']
    # Find out of confidence range samples
    too_small = c < thresh
    isValid_out = ~too_small & isValid_in

    assert ~any(np.diff(t[isValid_out]) == 0)

    #isValid_out = removeLoners(t, isValid_out, filterSettings)

    # Feedback
    print('Low confidence filter: ' + str(sum(~isValid_out & isValid_in)) + ' samples removed.\n')
    num_rejected = sum(~isValid_out & isValid_in)
    return isValid_out, num_rejected


def getTotalTime(t, isValid):
    """
    Compute the total duration of a shot.

    :param t: Time in ms.
    :type t: numpy.ndarray
    :param isValid: Boolean array indicating if value is valid or filtered out in previous filter step.
    :type isValid: numpy.ndarray

    :returns: List containing zero and the length of the signal in ms.
    :rtype: list
    """

    #include edges
    isValid = np.append(True, np.append(isValid, False))
    t = np.append(t[0], np.append(t, t[-1]))
    first = True
    last = True
    time = 0.0
    for i in range(len(t)):

        if isValid[i] & first:

            time -= t[i]
            first = False
            last = True

        elif ~isValid[i] & last:

            time += t[i]
            last = False
            first = True

    return [0.0, time]


def interpolate_pd_single(processed_df, interp_type='pChip'):
    """
    Interpolate pupil diameter data for a single eye.

    :param processed_df: Dataframe containing pd data and filter results.
    :type processed_df: pandas.DataFrame
    :param interp_type: Interpolation method ('linear', 'pChip', 'akima').
    :type interp_type: str

    :returns: Dataframe containing raw data, filter information and interpolated data.
    :rtype: pandas.DataFrame
    """

    t_ms = np.array(processed_df['time'] * 1000)  # time in ms
    pd_right = np.array(processed_df['right pupil size'])
    isValid_right = np.array(processed_df['isValidFinalright'])
    pd_right[~isValid_right] = np.nan
    pd_right_fixed = pd_right

    valid_indices = np.where(~np.isnan(pd_right))[0]

    if interp_type in ('linear', 'pChip', 'akima'):
        interp_func = interpolate.PchipInterpolator if interp_type == 'pChip' else interpolate.Akima1DInterpolator
        interp_r = interp_func(t_ms[valid_indices], pd_right[valid_indices])
        pd_right_interpolated = interp_r(t_ms)
        pd_right_fixed[~isValid_right] = pd_right_interpolated[~isValid_right]
    else:
        print('Specify valid interpolation method!')

    processed_df['interp'] = pd_right_interpolated
    return processed_df


def interpolate_notValid(df, isValid, interp_type='pChip'):
    t_ms = np.array(df['time'] * 1000)  # time in ms
    pd = np.array(df['pupil_diameter'])

    pd[~isValid] = np.nan
    pd_fixed = pd

    valid_indices = np.where(~np.isnan(pd))[0]

    if interp_type in ('linear', 'pChip', 'akima'):
        interp_func = interpolate.PchipInterpolator if interp_type == 'pChip' else interpolate.Akima1DInterpolator
        interp_r = interp_func(t_ms[valid_indices], pd[valid_indices])
        pd_interpolated = interp_r(t_ms)
        pd_fixed[~isValid] = pd_interpolated[~isValid]
    else:
        print('Specify valid interpolation method!')

    df.loc[:, 'pd_interp'] = pd_interpolated

    return df


def interpolate_confidence(df, interp_type='pChip', confidence_threshold=0.8, extra_invalid_samples=2):
    confidence = np.array(df['confidence'])
    isValid = np.array(confidence) > confidence_threshold

    invalid_indices = np.where(~isValid)[0]

    # Extend invalid samples by extra_invalid_samples on both sides
    for idx in invalid_indices:
        start_idx = max(0, idx - extra_invalid_samples)
        end_idx = min(len(isValid) - 1, idx + extra_invalid_samples)
        isValid[start_idx:end_idx + 1] = False

    df = interpolate_notValid(df, isValid, interp_type=interp_type)
    return df


def compute_meanDia(raw_df, processed_df, interp_type='pChip'):
    """
    Compute the mean pupil diameter.

    :param raw_df: Raw data containing 'time', 'left pupil size', 'right pupil size'.
    :type raw_df: pandas.DataFrame
    :param processed_df: Processed data containing 'isValidFinalleft', 'isValidFinalright'.
    :type processed_df: pandas.DataFrame
    :param interp_type: Interpolation type ('linear', 'pChip', 'akima').
    :type interp_type: str

    :returns: Processed data with additional columns: 'meanDia', 'lwithoutR', 'rwithoutL', 'bothwithout'.
    :rtype: pandas.DataFrame
    """

    t_ms = raw_df['normal time'].to_numpy(dtype=float, na_value=np.nan) * 1000  # time in ms

    pd_left = raw_df['left pupil size'].to_numpy(dtype=float, na_value=np.nan)
    pd_right = raw_df['right pupil size'].to_numpy(dtype=float, na_value=np.nan)

    isValid_left = np.array(processed_df['isValidFinalleft'])
    isValid_right = np.array(processed_df['isValidFinalright'])

    pd_left[~isValid_left] = np.nan
    pd_right[~isValid_right] = np.nan

    # Identify all the single pupil data rows:
    lwithoutR = ~np.isnan(pd_left) & np.isnan(pd_right)
    rwithoutL = ~np.isnan(pd_right) & np.isnan(pd_left)

    bothwithout = np.isnan(pd_left) & np.isnan(pd_right)

    # Get the difference between left and right diameters:
    diamDiff = (pd_right - pd_left)
    diamDiffRows = ~np.isnan(diamDiff)

    ## Generate mean Diameter

    # Calculate the fixed left and right pupil diameters
    if sum(diamDiffRows) > 2:

        # Interpolate the differences to the full time vector:
        diamDiffCont = np.interp(t_ms, t_ms[diamDiffRows], diamDiff[diamDiffRows])

        # Synthesize data for the left eye when data for the right is available using the previously calculated difference
        l_fixed = pd_left
        l_fixed[rwithoutL] = pd_right[rwithoutL] - diamDiffCont[rwithoutL]

        # Same for other pupil
        r_fixed = pd_right
        r_fixed[lwithoutR] = pd_left[lwithoutR] + diamDiffCont[lwithoutR]

        # Deal with parts where both PD data is corrupted
        r_fixed[bothwithout] = np.nan
        l_fixed[bothwithout] = np.nan

        if interp_type in ('linear', 'pChip', 'akima'):
            interp_func = interpolate.PchipInterpolator if interp_type == 'pChip' else interpolate.Akima1DInterpolator
            interp_r = interp_func(t_ms[~bothwithout], r_fixed[~bothwithout])
            interp_l = interp_func(t_ms[~bothwithout], l_fixed[~bothwithout])
            r_fixed[bothwithout] = interp_r(t_ms[bothwithout])
            l_fixed[bothwithout] = interp_l(t_ms[bothwithout])
        else:
            r_fixed = r_fixed[~bothwithout]
            l_fixed = l_fixed[~bothwithout]

        # Calculate the mean
        meanDia = np.mean([l_fixed, r_fixed], axis=0)

    else:
        meanDia = []
        print('mean diameter could not be computed')

    processed_df.loc[:, 'meanDia'] = meanDia
    processed_df.loc[:, 'lwithoutRR'] = lwithoutR
    processed_df.loc[:, 'rwithoutL'] = rwithoutL
    processed_df.loc[:, 'bothwithout'] = bothwithout

    return processed_df


def compute_single_preprocessed(
    raw_df,
    processed_df,
    interp_type: str = "pChip",
    eye: str = "right",
):
    """
    Compute the preprocessed and interpolated pupil signal for a single eye.

    Invalid samples are set to NaN based on the corresponding validity flag and
    then interpolated using the selected interpolation method.

    :param raw_df: Raw data containing at least 'normal time' and
        '<eye> pupil size'.
    :type raw_df: pandas.DataFrame
    :param processed_df: Processed data containing the validity flag
        'isValidFinal<eye>'.
    :type processed_df: pandas.DataFrame
    :param interp_type: Interpolation type. Supported values are
        'linear', 'pChip', and 'akima'.
    :type interp_type: str
    :param eye: Eye to process, e.g. 'left' or 'right'.
    :type eye: str
    :return: Copy of ``processed_df`` with an additional interpolated pupil
        signal column.
    :rtype: pandas.DataFrame
    """
    processed_df = processed_df.copy()

    time_col = "normal time"
    pupil_col = f"{eye} pupil size"
    valid_col = f"isValidFinal{eye}"
    output_col = f"{eye}_pupil"

    # Convert time to milliseconds
    t_ms = raw_df[time_col].to_numpy(dtype=float, na_value=np.nan) * 1000.0

    # Load pupil signal and validity mask
    pd_single = raw_df[pupil_col].to_numpy(dtype=float, na_value=np.nan)
    is_valid_single = processed_df[valid_col].to_numpy(dtype=bool)

    # Mark invalid samples as missing
    pd_single[~is_valid_single] = np.nan

    # Interpolate missing values if possible
    valid_mask = ~np.isnan(pd_single)
    missing_mask = np.isnan(pd_single)

    if np.any(missing_mask):
        if np.sum(valid_mask) < 2:
            raise ValueError(
                f"Not enough valid samples available to interpolate the {eye} eye signal."
            )

        if interp_type == "linear":
            pd_single[missing_mask] = np.interp(
                t_ms[missing_mask],
                t_ms[valid_mask],
                pd_single[valid_mask],
            )
        elif interp_type == "pChip":
            interp_func = interpolate.PchipInterpolator(
                t_ms[valid_mask],
                pd_single[valid_mask],
                extrapolate=False,
            )
            pd_single[missing_mask] = interp_func(t_ms[missing_mask])
        elif interp_type == "akima":
            interp_func = interpolate.Akima1DInterpolator(
                t_ms[valid_mask],
                pd_single[valid_mask],
            )
            pd_single[missing_mask] = interp_func(t_ms[missing_mask])
        else:
            raise ValueError(
                "Unsupported interp_type. Choose from 'linear', 'pChip', or 'akima'."
            )

    processed_df[output_col] = pd_single

    return processed_df


import numpy as np
import pandas as pd
from scipy import interpolate


def interpolate_and_resample(
    df: pd.DataFrame,
    time_col: str,
    diameter_col: str,
    fs: int = 90,
):
    """
    Interpolate missing values and resample the data.

    PCHIP -> np.arange(min, max, 1/fs) -> interpolator(new_time)

    :param df: input data
    :param time_col: time column (seconds)
    :param diameter_col: pupil diameter column
    :param fs: target sampling rate
    :return: DataFrame with {time_col, f"{diameter_col}_resampled"}
    """

    t_num = pd.to_numeric(df[time_col], errors="coerce")
    y_num = pd.to_numeric(df[diameter_col], errors="coerce")

    y_num = y_num.replace(-1, np.nan)
    # optional:
    # y_num = y_num.where(y_num > 0, np.nan)

    valid = np.isfinite(t_num) & np.isfinite(y_num)

    x = t_num.loc[valid].to_numpy(dtype=float)
    y = y_num.loc[valid].to_numpy(dtype=float)

    # basic checks
    if len(x) < 2:
        raise ValueError(f"Not enough valid samples in column '{diameter_col}'.")

    # sort by time
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # remove duplicate timestamps
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]

    if len(x) < 2:
        raise ValueError(f"Not enough unique time points in column '{diameter_col}'.")

    interpolator = interpolate.PchipInterpolator(x, y)

    dt = 1.0 / fs
    t_fs = np.arange(x.min(), x.max(), dt)
    sig_fs = interpolator(t_fs)

    return pd.DataFrame({time_col: t_fs, f"{diameter_col}_resampled": sig_fs})

def downsample_mask_nearest(t_src, mask_src, t_dst):
    """
    Nearest-neighbor resampling of a boolean mask based on time axis.
    mask_src: True = bothwithout (bad), False = ok
    """
    t_src = np.asarray(t_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)

    # pandas nullable boolean -> numpy bool
    mask_src = pd.Series(mask_src).fillna(True).astype(bool).to_numpy()

    idx = np.searchsorted(t_src, t_dst, side="left")
    idx = np.clip(idx, 1, len(t_src) - 1)

    left = idx - 1
    right = idx

    choose_right = (t_dst - t_src[left]) > (t_src[right] - t_dst)
    idx_nearest = np.where(choose_right, right, left)

    return mask_src[idx_nearest]


def downsample(
    dataframe: pd.DataFrame,
    time_col: str,
    diameter_col: str,
    target_fs: int = 25,
    original_fs: int = 90,
    polyphase_window=("kaiser", 5.0),
):
    # --- 1) Extract time + signal (drop rows where either is missing) ---
    df = dataframe[[time_col, diameter_col]].dropna()
    t = df[time_col].to_numpy(dtype=float)
    y = df[diameter_col].to_numpy(dtype=float)

    if len(t) < 2:
        return pd.DataFrame({time_col: t, f"{diameter_col}_downsampled_{target_fs}": y})

    # --- 2) Polyphase resample y: original_fs -> target_fs ---
    ratio = Fraction(target_fs, original_fs).limit_denominator()
    up, down = ratio.numerator, ratio.denominator

    y_ds = signal.resample_poly(y, up, down, window=polyphase_window)

    # --- 3) New uniform time axis aligned to start time ---
    t0 = t[0]
    n_target = len(y_ds)
    t_ds = t0 + np.arange(n_target) / float(target_fs)

    return pd.DataFrame(
        {time_col: t_ds, f"{diameter_col}_downsampled_{target_fs}": y_ds}
    )


def get_stats(meanDia):
    """
    Calculates pupil diameter related statistics such as mean, standard deviation, etc.

    :param meanDia: Mean pupil diameter in mm.
    :type meanDia: pandas.Series

    :returns: Dictionary with pupil diameter statistics including mean, standard deviation, skewness, maximum, median, variance, minimum, kurtosis, and range.
    :rtype: dict
    """

    stats = {
        'mean': np.mean(meanDia),
        'std': np.std(meanDia),
        'skew': st.skew(meanDia),
        'max_val': np.max(meanDia),
        'median': np.median(meanDia),
        'variance': np.var(meanDia),
        'min_val': np.min(meanDia),
        'kurt': st.kurtosis(meanDia),
        'range': np.max(meanDia) - np.min(meanDia),
        '1st_quantile': np.percentile(meanDia, 25),
        '3rd_quantile': np.percentile(meanDia, 75),
        'harmonic_mean': harmonic_mean(meanDia),
        'samples_till_max': np.argmax(meanDia)
    }

    return stats


def get_slope(meanDia, time):
    """
    Calculate the slope of the first and second half of the mean pupil diameter data of both eyes.

    :param meanDia: Mean pupil diameter in mm.
    :type meanDia: pandas.Series
    :param time: Time in ms.
    :type time: numpy.ndarray

    :returns: Slope and intercept of the first and second half of the mean pupil diameter.
    :rtype: dict
    """

    slope1, intercept1 = np.polyfit(time[0:int(len(time)/2)], meanDia[0:int(len(time)/2)], 1)
    slope2, intercept2 = np.polyfit(time[int(len(time)/2):len(time)], meanDia[int(len(time)/2):len(time)], 1)

    slope_features = {
        'slope1': slope1,
        'intercept1': intercept1,
        'slope2': slope2,
        'intercept2': intercept2
    }
    return slope_features


def baseline_correct(meanDia_shot, base_correct_val):
    """
    Divisive baseline correction of mean pupil diameter data.

    :param meanDia_shot: Mean pupil diameter data.
    :type meanDia_shot: pandas.Series
    :param base_correct_val: Value to use for baseline correction.
    :type base_correct_val: float

    :returns: Baseline corrected mean pupil diameter data.
    :rtype: pandas.Series
    """

    meanDia_shot_corrected = (meanDia_shot - base_correct_val) / base_correct_val  # meanDia_shot/base_correct_val - 1#(meanDia_shot - base_correct_val) / base_correct_val
    return meanDia_shot_corrected
