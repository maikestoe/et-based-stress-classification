################################################################################
# Adapted from:
# Duchowski, A. T., Krejtz, K., Krejtz, I., Biele, C.,
# Niedzielska, A., Kiefer, P., Raubal, M., & Giannopoulos, I. (2018).
# The Index of Pupillary Activity: Measuring Cognitive Load vis-a-vis
# Task Difficulty with Pupil Oscillation. Proceedings of CHI 2018.
# https://doi.org/10.1145/3173574.3173856
#
# Duchowski, A. T., Krejtz, K., Gehrer, N. A., Bafna, T.,
# & Baekgaard, P. (2020). The Low/High Index of Pupillary Activity.
# Proceedings of CHI 2020. https://doi.org/10.1145/3313831.3376394
#
# Compute index of pupillary activity (IPA) metric
################################################################################

"""
Preprocessing and Analysis of Pupil Diameter Data

This script provides functions to compute metrics related to pupillary activity, including the index of pupillary
activity (IPA) and the low/high index of pupillary activity (LHIPA). These metrics are calculated based on wavelet
transforms and thresholding techniques as described by Duchowski et al. (CHI 2018; IPA) and Duchowski et al.
(CHI 2020; LHIPA).

Functions:
----------
- compute_ipa(d, t, ipaSettings):
    Computes the index of pupillary activity (IPA) as described by Duchowski et al. (CHI 2018).

- modmax(d):
    Computes the modulus maxima.

- compute_lhipa(d, t, lhipaSettings):
    Computes the low/high index of pupillary activity (LHIPA) as introduced by Duchowski et al. (CHI 2020).

Dependencies:
-------------
- math
- pywt
- numpy
- pandas

Author: Maike Laut
Date: 20.06.2024
"""
import math
import pywt
import numpy as np
import pandas as pd

def compute_ipa(d, t, ipaSettings):
    """
    Computes the index of pupillary activity (IPA) as described by Duchowski et al. (CHI 2018).

    :param d: Pupil diameter data.
    :type d: numpy.ndarray
    :param t: Duration of d in seconds.
    :type t: float
    :param ipaSettings: Contains wavelet type and parameter to define threshold.
    :type ipaSettings: dict

    :returns: IPA value.
    :rtype: float
    """
    # obtain 2-level DWT of pupil diameter signal d
    try:
        coeffs = pywt.wavedec(d, ipaSettings['wavelet'], mode='periodization', level=2)
        if len(coeffs) < 3:
            return None
        cA2, cD2, cD1 = coeffs

    except ValueError:
        return None

    # Normalize by 1/2^j, j=2 for 2-level DWT
    cA2[:] = [x / math.sqrt(4.0) for x in cA2]
    cD1[:] = [x / math.sqrt(2.0) for x in cD1]
    cD2[:] = [x / math.sqrt(4.0) for x in cD2]

    cD = {
        'cA2': cA2,
        'cD1': cD1,
        'cD2': cD2
    }

    # Detect modulus maxima
    cD2m = modmax(cD[ipaSettings['cDn']])

    # Threshold
    lamb = np.std(cD2m) * math.sqrt(ipaSettings['param'] * np.log2(len(cD2m)))
    cD2t = pywt.threshold(cD2m, lamb, mode="hard")

    # Compute IPA
    ctr = 0
    for i in range(len(cD2t)):
        if math.fabs(cD2t[i]) > 0:
            ctr += 1
    ipa = float(ctr) / t

    return ipa


def modmax(d):
    """
    Computes the modulus maxima.

    :param d: Wavelet modulus.
    :type d: numpy.ndarray

    :returns: Wavelet modulus with applied modulus maximum detection.
    :rtype: numpy.ndarray
    """

    # compute signal modulus
    m = [0.0] * len(d)
    for i in range(len(d)):
        m[i] = math.fabs(d[i])

    # if value is larger than both neighbors, and strictly larger than either, then it is a local maximum
    t = [0.0] * len(d)
    for i in range(len(d)):
        ll = m[i - 1] if i >= 1 else m[i]
        oo = m[i]
        rr = m[i + 1] if i < len(d) - 2 else m[i]
        if (ll <= oo and oo >= rr) and (ll < oo or oo > rr):
            # compute magnitude
            t[i] = math.sqrt(d[i] ** 2)
        else:
            t[i] = 0.0
    return t


def compute_lhipa(d, t, lhipaSettings):
    """
    Computes the low/high index of pupillary activity (LHIPA) as introduced by Duchowski et al. (CHI 2020).

    :param d: Pupil diameter signal in mm. Can be pandas.Series (incl. Float64), numpy array, list.
    :type d: Union[pandas.Series, numpy.ndarray, list]
    :param t: Duration of signal d in seconds.
    :type t: float
    :param lhipaSettings: Contains the wavelet family used for the LHIPA computation.
    :type lhipaSettings: dict
    :returns: LHIPA value.
    :rtype: float
    """

    # --- 0) Basic sanity checks
    if t is None or t <= 0:
        return 0.0

    # --- 1) Convert input to a NumPy float array (PyWavelets expects numeric numpy dtype)
    if isinstance(d, pd.Series):
        x = d.to_numpy(dtype=float, na_value=np.nan).copy()
    else:
        # covers numpy arrays, lists, pandas arrays
        x = np.asarray(d, dtype=float).copy()

    # flatten just in case (e.g., shape (N,1))
    x = x.ravel()

    # too short -> no meaningful decomposition
    if x.size < 2:
        return 0.0

    # --- 2) Handle NaNs (choose a conservative strategy)
    # Option A (recommended here): linear interpolation for internal NaNs; edge NaNs forward/back fill.
    # If you prefer to simply drop NaNs or return 0, tell me.
    if np.isnan(x).any():
        s = pd.Series(x)
        # interpolate inside, then fill edges
        s = s.interpolate(method="linear", limit_direction="both")
        x = s.to_numpy(dtype=float)

        # if still NaNs (e.g., all-NaN input), bail out
        if np.isnan(x).any():
            return 0.0

    # --- 3) Compute max decomposition level
    w = pywt.Wavelet(lhipaSettings["wavelet"])
    maxlevel = pywt.dwt_max_level(len(x), filter_len=w.dec_len)
    if maxlevel <= 1:
        return 0.0

    hif, lof = 1, int(maxlevel / 2)

    # --- 4) Wavelet detail coefficients
    cD_H = pywt.downcoef("d", x, lhipaSettings["wavelet"], mode="periodization", level=hif)
    cD_L = pywt.downcoef("d", x, lhipaSettings["wavelet"], mode="periodization", level=lof)

    # normalize (note: original code uses hif for both; keeping as-is to preserve behavior)
    cD_H = cD_H / math.sqrt(2 ** hif)
    cD_L = cD_L / math.sqrt(2 ** hif)

    # --- 5) LH:HF ratio (guard against division by zero)
    # index mapping factor
    factor = (2 ** lof) / (2 ** hif)
    cD_LH = np.empty_like(cD_L, dtype=float)
    for i in range(len(cD_L)):
        j = int(factor * i)
        denom = cD_H[j] if j < len(cD_H) else np.nan
        cD_LH[i] = cD_L[i] / denom if denom not in (0.0, -0.0) else 0.0

    # if any NaNs from denom overflow, replace safely
    if np.isnan(cD_LH).any():
        cD_LH = np.nan_to_num(cD_LH, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 6) Modulus maxima + thresholding
    cD_LHm = modmax(cD_LH)

    lamb = np.std(cD_LHm) * math.sqrt(2.0 * np.log2(len(cD_LHm)))
    cD_LHt = pywt.threshold(cD_LHm, lamb, mode="soft")

    # --- 7) LHIPA count / time
    ctr = np.count_nonzero(np.abs(cD_LHt) > 0)
    return float(ctr) / float(t)
