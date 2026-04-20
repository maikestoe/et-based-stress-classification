"""
Segmentation Functions for the VR goalkeeper dataset (vr_goalkeeper)

This script provides functions to segment data from the VR goalkeeper dataset (vr_goalkeeper) into various phases, identify
individual shots, and segment baseline periods for further analysis. The functions are designed to handle raw data,
perform necessary preprocessing steps, and return relevant segments and indices for analysis.

Functions:
----------
- computePhases(animation, Task):
    Segments the data into individual phases of each shot.

- find_shots(df_raw):
    Segments individual shots and identifies start and end indices of each phase.

- segment_baseline(mean_pd, time, bothwithout):
    Segments a baseline period for baseline correction of pupil diameter (PD) data.

Example Usage:
--------------
1. Segment the data into phases using `computePhases`.
2. Identify the nearest value in an array using `find_nearest`.
3. Segment individual shots from raw data using `find_shots`.
4. Segment a baseline period for PD data using `segment_baseline`.

Dependencies:
-------------
- numpy
- pandas
- utils (find_nearest)

Author: Maike Laut
Date: 20.06.2024
"""

# Imports
import numpy as np
import pandas as pd
from utils import find_nearest


def computePhases(animation, Task):
    """
    Segments the data from the VR goalkeeper dataset (vr_goalkeeper) into the individual phases of each shot.

    This function processes the animation and Task data to identify and segment the phases of each shot.

    :param animation: Phase and shot information ('nicht gehalten', 'gehalten', 'Tooor!', 'Anlauf').
    :type animation: numpy.ndarray
    :param Task: Information if the cognitive task was running.
    :type Task: numpy.ndarray
    :return: A tuple containing boolean masks indicating valid samples for each phase, a DataFrame with phase information, and an array indicating success (0 for missed, 1 for saved).
    :rtype: tuple(list, pandas.DataFrame, numpy.ndarray)

    :Example:

    >>> animation = np.array(['gehalten', 'Anlauf', 'nicht gehalten', 'Tooor!', 'gehalten'])
    >>> Task = np.array([0, 0, 1, 0, 1])
    >>> phasesIsValid, phases_df, success = computePhases(animation, NBackTask)
    """

    animation = np.asarray(animation, dtype=object).ravel()
    filterN = np.asarray(Task, dtype=bool).ravel()

    # robust elementwise compare (pandas kann mit nan sauber umgehen)
    anim = pd.Series(animation, dtype="object")

    nichtgehalten = np.argwhere(anim.eq("nicht gehalten").to_numpy()).ravel()
    animation = animation.copy()
    animation[nichtgehalten] = np.nan

    filterP = pd.isnull(animation)

    gehalten = np.argwhere(pd.Series(animation, dtype="object").eq("gehalten").to_numpy()).ravel()
    tor = np.argwhere(pd.Series(animation, dtype="object").eq("Tooor!").to_numpy()).ravel()

    ###############
    # Compute a single success value per shot (20 shots in total), where 0 equals miss and 1 equals saved
    success = np.empty((len(animation)))
    success[:] = np.nan
    success[gehalten] = 1
    success[nichtgehalten] = 0
    success[tor] = 2
    success = success[np.logical_not(np.isnan(success))]
    for i in range(0, len(success)):
        if success[i] == 0 and i != 0:
            success[i-1] = np.nan
        if success[i] == 2:
            success[i] = 0

    success = success[np.logical_not(np.isnan(success))]

    #######################

    filterR = pd.Series(animation, dtype="object").eq("Anlauf").to_numpy()  # runup

    filterS = ~filterP & ~filterR  # shot
    allphases = filterP | filterR | filterS | filterN
    phasesIsValid = [allphases, filterP, filterR, filterS, filterN]
    phases_df = pd.DataFrame({'allphases': allphases, 'prep': filterP, 'runup': filterR, 'shot': filterS,
                              'task': filterN})

    return phasesIsValid, phases_df, success


def find_shots(df_raw):
    """
    Segment individual shots from the VR goalkeeper dataset (vr_goalkeeper) and identify start and end indices of each phase.

    :param df_raw: DataFrame containing the study raw data of one participant.
    :type df_raw: pandas.DataFrame
    :return: A tuple containing the list of all start indices and the list of all end indices.
    :rtype: tuple(pandas.Index, pandas.Index)

    :Example:

    >>> df_raw = pd.DataFrame({'animation': ['gehalten', 'Anlauf', 'nicht gehalten'], 'Task': [0, 0, 1]})
    >>> start_shots, end_shots = find_shots(df_raw)
    """
    # get phases of each shot (overall, nback, prep, runup, shot) and success for individual penalties
    anim_s = df_raw["animation"].astype("string").str.strip()
    anim_s = anim_s.replace({pd.NA: np.nan})
    animation = anim_s.astype(object).to_numpy().ravel()

    task = df_raw["Task"].fillna(False).astype(bool).to_numpy().ravel()

    phasesIsValid, phases_df, success = computePhases(animation, task)

    # phases_df.prep --> boolean values for all idxs if part of preparation or not
    prep_idxs_starts = phases_df.loc[phases_df.prep != phases_df.prep.shift()].prep

    # prep:       T T T T F F F F F T T T T F F F n
    # prep.shift: n T T T T F F F F F T T T T F F F
    # !=          1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1
    # runup:      0       1         0       1     0  ---> end
    # prep:       1       0         1       0     1  ---> start
    prep_idxs_ends = phases_df.loc[
        phases_df.prep != phases_df.prep.shift()].runup
    # prep_idxs: True: Start of preparation; False: End of preparation

    # get start of each preparation phase
    start_shots = prep_idxs_starts[prep_idxs_starts].index
    start_shots = start_shots[0:len(start_shots) - 1]

    # get end of each preparation phase
    end_shots = prep_idxs_ends[prep_idxs_ends].index

    return start_shots, end_shots


def segment_baseline(mean_pd, time, bothwithout, start_s=30.0, length_s=1.0):
    """
    Segment a baseline period for baseline correction of PD data.

    Searches for a snippet without artifacts after start_s seconds with a duration of length_s seconds.

    Returns: (base_snipped, base_snipped_time) as numpy arrays.
    If you need pandas Series, wrap them outside.
    """

    # --- normalize inputs to 1D numpy arrays ---
    if isinstance(mean_pd, pd.Series):
        y = mean_pd.to_numpy(dtype=float, na_value=np.nan)
    else:
        y = np.asarray(mean_pd, dtype=float)

    if isinstance(time, pd.Series):
        t = time.to_numpy(dtype=float, na_value=np.nan)
    else:
        t = np.asarray(time, dtype=float)

    # bothwithout: True means "both missing/artifact" (bad)
    if isinstance(bothwithout, pd.Series):
        bw = bothwithout.fillna(True).astype(bool).to_numpy()
    else:
        bw = pd.Series(bothwithout).fillna(True).astype(bool).to_numpy()

    y = y.ravel()
    t = t.ravel()
    bw = bw.ravel()

    # --- basic sanity ---
    n = min(len(y), len(t), len(bw))
    y, t, bw = y[:n], t[:n], bw[:n]
    if n < 2:
        return y, t

    # time relative to start
    t_base = t - t[0]

    # helper: nearest index
    def nearest_idx(arr, val):
        arr = np.asarray(arr)
        i = int(np.argmin(np.abs(arr - val)))
        return i

    # --- search for artifact-free snippet ---
    start = float(start_s)
    t_end = float(t_base[-1])

    snip_i0 = None
    snip_i1 = None

    while start < t_end:
        i0 = nearest_idx(t_base, start)
        i1 = nearest_idx(t_base, start + float(length_s))
        if i1 < i0:
            i0, i1 = i1, i0

        # include i1 in the slice (+1)
        bw_snip = bw[i0 : i1 + 1]

        # if any bad sample -> move forward
        if np.any(bw_snip):
            start += float(length_s)
        else:
            snip_i0, snip_i1 = i0, i1
            break

    # fallback: first length_s seconds
    if snip_i0 is None or snip_i1 is None:
        snip_i0 = nearest_idx(t_base, 0.0)
        snip_i1 = nearest_idx(t_base, float(length_s))
        if snip_i1 < snip_i0:
            snip_i0, snip_i1 = snip_i1, snip_i0

    base_snipped = y[snip_i0 : snip_i1 + 1]
    base_snipped_time = t_base[snip_i0 : snip_i1 + 1]
    return base_snipped, base_snipped_time


def segment_baseline_unfilt(signal, time, start_s=30.0, length_s=1.0):
    """
    Segment a baseline period from a monocular, unfiltered pupil signal.

    Extracts the first snippet of duration length_s after start_s seconds that
    does not contain NaN values (e.g., blinks or missing samples).

    Parameters
    ----------
    signal : array-like or pd.Series
        Monocular pupil signal.
    time : array-like or pd.Series
        Time vector corresponding to the signal.
    start_s : float
        Time after recording start where baseline search begins.
    length_s : float
        Duration of the baseline snippet.

    Returns
    -------
    base_snippet : np.ndarray
        Extracted baseline signal.
    base_snippet_time : np.ndarray
        Corresponding time values relative to recording start.
    """

    # --- normalize inputs to numpy arrays ---
    if isinstance(signal, pd.Series):
        y = signal.to_numpy(dtype=float, na_value=np.nan)
    else:
        y = np.asarray(signal, dtype=float)

    if isinstance(time, pd.Series):
        t = time.to_numpy(dtype=float, na_value=np.nan)
    else:
        t = np.asarray(time, dtype=float)

    y = y.ravel()
    t = t.ravel()

    # --- basic sanity ---
    n = min(len(y), len(t))
    y, t = y[:n], t[:n]

    if n < 2:
        return y, t

    # relative time
    t_base = t - t[0]

    # helper: nearest index
    def nearest_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    start = float(start_s)
    t_end = float(t_base[-1])

    snip_i0 = None
    snip_i1 = None

    # --- search for NaN-free snippet ---
    while start < t_end:

        i0 = nearest_idx(t_base, start)
        i1 = nearest_idx(t_base, start + float(length_s))

        if i1 < i0:
            i0, i1 = i1, i0

        y_snip = y[i0:i1 + 1]

        # valid snippet if no NaNs
        if np.isnan(y_snip).any():
            start += float(length_s)
        else:
            snip_i0, snip_i1 = i0, i1
            break

    # --- fallback: first length_s seconds ---
    if snip_i0 is None:
        snip_i0 = nearest_idx(t_base, 0.0)
        snip_i1 = nearest_idx(t_base, float(length_s))

        if snip_i1 < snip_i0:
            snip_i0, snip_i1 = snip_i1, snip_i0

    base_snippet = y[snip_i0:snip_i1 + 1]
    base_snippet_time = t_base[snip_i0:snip_i1 + 1]

    return base_snippet, base_snippet_time
