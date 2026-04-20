"""
Utility Functions for the eye tracking datasets vr_goalkeeper and forDigitStress

This script provides utility functions used in the segmentation and analysis of data from the VR goalkeeper dataset
(vr_goalkeeper) and forDigitStress.

Functions:
----------
- find_nearest(array, value):
    Finds the array element whose value is closest to the given value.

Dependencies:
-------------
- numpy

Author: Maike Laut
Date: 20.06.2024
"""

import numpy as np


def find_nearest(array, value):
    """
    Find the array element whose value is closest to the given value.

    Args:
        array (numpy.ndarray): Array containing values.
        value (float): Value to find the closest element to.

    Returns:
        int: Index of the closest value.

    Example:
        >>> array = np.array([1, 3, 7, 8, 10])
        >>> value = 5
        >>> idx = find_nearest(array, value)
        >>> print(idx)
        1
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
