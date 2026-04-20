"""
Segmentation Functions for the ForDigitStress Study

This script provides functions to segment baseline periods for baseline correction of ForDigitStress data.
The functions are designed to handle raw pupil diameter (PD) data, perform necessary preprocessing steps,
and return relevant baseline segments for analysis.

Functions:
----------
- segment_baseline(pd, time, confidence, conf_thresh):
    Segments a baseline period for baseline correction of PD data.

Dependencies:
-------------
- utils (find_nearest)

Author: Maike Laut
Date: 20.06.2024
"""

from utils import find_nearest


def segment_baseline(pd, time, confidence, conf_thresh):
    """
    Segment a baseline period for baseline correction of ForDigitStress data.

    This function searches for a one-second signal snippet without artifacts after the initial 30 seconds of the baseline measurement.

    :param pd: PD data to be segmented.
    :type pd: pandas.Series
    :param time: Time in seconds.
    :type time: pandas.Series
    :param confidence: Confidence values provided by the eye tracker.
    :type confidence: pandas.Series
    :param conf_thresh: Confidence threshold defining good quality data.
    :type conf_thresh: float
    :return: A tuple containing the mean pupil diameter segment for baseline correction and the corresponding time in seconds.
    :rtype: tuple(pandas.Series, pandas.Series)

    """

    start = time.iloc[0] + 30  # start to search for snipped in seconds
    length = 1  # length of snipped in seconds

    # Find a window with high confidence values
    while start < time.iloc[-1]-1:
        snipped_start = find_nearest(time, start)    # Select start at approximately 30 seconds
        snipped_end = find_nearest(time, start + length)

        confidence_snipped = confidence.iloc[snipped_start:snipped_end+1]
        if any(value < conf_thresh for value in confidence_snipped):
            start += length
        else:
            #break

            base_snipped = pd.iloc[snipped_start:snipped_end+1]
            base_snipped_time = time.iloc[snipped_start:snipped_end+1]

            return base_snipped, base_snipped_time, snipped_end

    print('No window found')
    return None, None
