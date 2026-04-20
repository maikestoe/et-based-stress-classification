"""
This module contains settings and classes related to eye tracking data processing.

Settings dictionaries:
- `ipaSettings`: Settings for IPA analysis.
- `lhipaSettings`: Settings for LHIPA analysis.
- `fixationfilterSettings`: Settings for fixation filter parameters.
- `shallowMLSettings`: Settings for shallow machine learning models.

Classes:
- `PupilDiameter`: Class representing pupil diameter data with methods to retrieve and set attributes.
- `PdFilter`: Class implementing a pupil diameter filtering algorithm with configurable parameters.

Dependencies:
-------------
- numpy
- pandas
- pd_utils (custom)

Authors: Maike Laut, Wolfgang Mehringer, Oliver Korn
Date: 20.06.2024
"""

import numpy as np
import pandas as pd
import pd_utils

ipaSettings = {
    'wavelet': 'sym16',  # sym16 in ipa paper, db in ica paper, for 50Hz maybe db4
    'param': 2,
    'level': 2,
    'cDn': 'cD2'
}

lhipaSettings = {
    'wavelet': 'db6'
}

fixationfilterSettings = {
    'detection method': 'velocity',

    'length': '',
    'frameRate': 0.011,  # time difference between frames in seconds
    'shots': 20,
    'gap_blinks': 100,

    'velocity': {
        'low-pass filter': 5,
        'threshold': 130,

    }, 'acceleration': {
        'low-pass filter': 5,
        'high-pass filter': 7,
        'tmin': 1,
        'tmax': 19,
        'constant': 1000

    }, 'asymptotic_model': {
        'threshold': 130,
        'asymptote': 750

    }, 'FIR filter - low-pass': {
        '2-tap': [1, 1],
        '5-tap': [1, 2, 3, 2, 1],
    }, 'FIR filter - high-pass': {
        '2-tap': [-1, 1],
        '7-tap': [-3, -2, -1, 0, 1, 2, 3]
    }
}

shallowMLSettings = {
    'linearSVM': {
        'model': 'SVM',
        'kernel': 'linear',
        'c': [np.power(2, float(x)) for x in np.arange(-10, 11)], # for SVM  -10, 11
        'n_features': []
    },
    'nonlinearSVM': {
        'model': 'SVM',
        'kernel': 'rbf',
        'c': [np.power(2, float(x)) for x in np.arange(-10, 11)],
        'gamma': [np.power(2, float(x)) for x in np.arange(-10, 11)],
        'n_features': []
    },
    'kNN': {
        'model': 'kNN',
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'weights': ['uniform', 'distance'],
        'n_features': []
    },
    'LDA': {
        'model': 'LDA',
        'solver': ['svd', 'lsqr', 'eigen'],
        'n_features': []
    },
    'RF': {
        'model': 'RF',
        'n_estimators': [300],
        'max_features': ['auto'],
        'max_depth': [int(x) for x in np.linspace(10, 600, num=5)],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': ['True'],
        'n_features': []
    }
}


class PupilDiameter:
    """
    Class representing pupil diameter data.

    Attributes:
    - `time`: Time data for pupil diameter measurements.
    - `pd_l`: Left pupil diameter measurements.
    - `pd_r`: Right pupil diameter measurements.
    - `blink_l`: Left eye blink status.
    - `blink_r`: Right eye blink status.
    - `pd_l_filt_speed`: Filtered left pupil diameter based on speed.
    - `pd_r_filt_speed`: Filtered right pupil diameter based on speed.
    - `fft_l`: Fast Fourier Transform (FFT) of left pupil diameter.
    - `fft_r`: Fast Fourier Transform (FFT) of right pupil diameter.
    - `mean_smoothBL_l`: Mean smoothed baseline of left pupil diameter.

    Methods:
    - `get_df()`: Returns a pandas DataFrame containing pupil diameter data.
    - `get_pd_l_filt()`: Returns the filtered left pupil diameter.
    - `set_pd_l_filt(pd_l_filt)`: Sets the filtered left pupil diameter.
    - `get_pd_r_filt()`: Returns the filtered right pupil diameter.
    - `set_pd_r_filt(pd_r_filt)`: Sets the filtered right pupil diameter.
    - `set_blink_l(blink_l)`: Sets the left eye blink status.
    - `set_blink_r(blink_r)`: Sets the right eye blink status.
    - `set_time(time)`: Sets the time data for pupil diameter measurements.
    - `get_fft_l()`: Returns the FFT of left pupil diameter.
    - `get_fft_r()`: Returns the FFT of right pupil diameter.
    - `set_fft_l(fft_l)`: Sets the FFT of left pupil diameter.
    - `set_fft_r(fft_r)`: Sets the FFT of right pupil diameter.
    """

    def __init__(self):
        self.time = np.nan
        self.pd_l = np.nan
        self.pd_r = np.nan
        self.blink_l = np.nan
        self.blink_r = np.nan
        self.pd_l_filt_speed = np.nan
        self.pd_r_filt_speed = np.nan
        self.fft_l = np.nan
        self.fft_r = np.nan
        self.mean_smoothBL_l = np.nan

    def get_df(self):
        """
        Returns a pandas DataFrame containing pupil diameter data.

        :return: DataFrame with columns: pd_l, pd_r, time, blink_l, blink_r, pd_l_filt_speed, pd_l_filt, mean_smoothBL_l.
        :rtype: pandas.DataFrame
        """

        return pd.DataFrame({'pd_l': self.pd_l,
                             'pd_r': list(self.pd_r),
                             'time': list(self.time),
                             'blink_l': list(self.blink_l),
                             'blink_r': list(self.blink_r),
                             'pd_l_filt_speed': list(self.pd_l_filt_speed),
                             'pd_l_filt': list(self.pd_l_filt),
                             'mean_smoothBL_l': list(self.mean_smoothBL_l),
                             },
                            columns=['pd_l', 'pd_r', 'time',
                                     'blink_l', 'blink_r', 'NanValidity',
                                     'RemoveOfBValidity', 'SpeedFilterValidity',
                                     'ResAnalysisValidity', 'pd_l_filt_speed',
                                     'pd_l_filt', 'smoothBL_l', 'mean_smoothBL_l'])

    def get_pd_l_filt(self):
        """
        Returns the filtered left pupil diameter.

        :return: Filtered left pupil diameter.
        :rtype: float
        """

        return self.pd_l_filt

    def set_pd_l_filt(self, pd_l_filt):
        """
        Sets the filtered left pupil diameter.

        :param pd_l_filt: Filtered left pupil diameter value to set.
        :type pd_l_filt: float
        """

        self.pd_l_filt = pd_l_filt

    def get_pd_r_filt(self):
        """
        Returns the filtered right pupil diameter.

        :return: Filtered right pupil diameter.
        :rtype: float
        """

        return self.pd_r_filt

    def set_pd_r_filt(self, pd_r_filt):
        """
        Sets the filtered right pupil diameter.

        :param pd_r_filt: Filtered right pupil diameter value to set.
        :type pd_r_filt: float
        """

        self.pd_r_filt = pd_r_filt

    def set_blink_l(self, blink_l):
        """
        Sets the left eye blink status.

        :param blink_l: Left eye blink status to set.
        :type blink_l: float
        """

        self.pd_blink_l = blink_l

    def set_blink_r(self, blink_r):
        """
        Sets the right eye blink status.

        :param blink_r: Right eye blink status to set.
        :type blink_r: float
        """

        self.pd_blink_r = blink_r

    def set_time(self, time):
        """
        Sets the time data for pupil diameter measurements.

        :param time: Time data for pupil diameter measurements to set.
        :type time: float
        """

        self.time = time

    def get_fft_l(self):
        """
        Returns the FFT of left pupil diameter.

        :return: FFT of left pupil diameter.
        :rtype: float
        """

        return self.fft_l

    def get_fft_r(self):
        """
        Returns the FFT of right pupil diameter.

        :return: FFT of right pupil diameter.
        :rtype: float
        """

        return self.fft_r

    def set_fft_l(self, fft_l):
        """
        Sets the FFT of left pupil diameter.

        :param fft_l: FFT of left pupil diameter value to set.
        :type fft_l: float
        """

        self.fft_l = fft_l

    def set_fft_r(self, fft_r):
        """
        Sets the FFT of right pupil diameter.

        :param fft_r: FFT of right pupil diameter value to set.
        :type fft_r: float
        """

        self.fft_r = fft_r


class PdFilter:
    """
    Class implementing pupil diameter filtering algorithm.

    Attributes:
    - `parameters`: Dictionary containing filter parameters.

    Methods:
    - `__init__(parameters)`: Initializes the PdFilter instance with given parameters.
    - `filter(blinks, t, d, confidence)`: Applies pupil diameter filtering algorithm based on given inputs.

    Note: This class relies on external modules such as `pd_utils`.
    """

    def __init__(self, parameters):
        """
        Initializes the PdFilter instance with given parameters.

        :param parameters: Dictionary containing filter parameters.
        :type parameters: dict
        """

        if parameters == 'default':
            self.parameters = {
                'd_min': 1.5,  # minimum pupil diameter in cm
                'd_max': 9.0,  # maximum pupil diameter in cm
                'invalid': -1,  # samples indicated as invalid by system
                'dilationSpeed': {
                    'max_gap': 200,  # ms
                    'multiplier': 16,  # # of mads
                },
                # Edge removal filter criteria
                'gaps': {
                    'minGap': 75,  # in ms
                    'maxGap': 2000,  # in ms
                    'backPadding': 50,  # in ms
                    'fwdPadding': 50  # in ms
                },
                'deviation': {
                    'Npasses': 4,  # # of passes
                    'multiplier': 16,  # of MADs
                    'interpFs': 100,  # Hz
                    'lowpassCF': 16  # Hz
                },
                'isolatedSample': {
                    'islandSeperation_ms': 40,  # ms
                    'minIslandWidth_ms': 50  # ms
                },
                'confidence': {
                    'confidence_thresh': 0.7
                }
            }
        else:
            self.parameters = parameters

    def filter(self, blinks, t, d, confidence):
        """
        Applies pupil diameter filtering algorithm based on given inputs.

        :param blinks: Blink data.
        :type blinks: numpy.ndarray
        :param t: Time data.
        :type t: numpy.ndarray
        :param d: Pupil diameter data.
        :type d: numpy.ndarray
        :param confidence: Confidence data.
        :type confidence: numpy.ndarray

        :return: Tuple containing filtering results.
        :rtype: tuple
        """

        if len(blinks) > 1:
            if np.isnan(blinks[-1]):
                blinks[-1] = True
                blinks = blinks[0:len(blinks) - 1]
                blinks_append = [False]
                blinks = np.array(blinks, dtype=bool)
                blinks = np.append(blinks, blinks_append)

            isBlinking = ~np.isnan(d) & ~blinks
        else:
            isBlinking = ~np.isnan(d)


        # remove samples that are defined as smaller or larger than criteria or invalid
        isValidRemoveOoB, sum_removed = pd_utils.removeOutOfBounds(t, d, isBlinking, self.parameters)

        # remove samples with low confidence ratings
        #if isinstance(confidence, np.ndarray) :
        #    isValidRemoveConf, sum_removed_conf = pd_utils.removeLowConfidence(t, confidence, isValidRemoveOoB, self.parameters)
        #else:
        isValidRemoveConf = isValidRemoveOoB
        sum_removed_conf = 0

        # (1) Blink detection: remove blinks and other artifacts that exhibit large differences between samples
        isValidSpeedFilter, speedFiltData, sum_filtered = pd_utils.madSpeedFilter(t, d, isValidRemoveConf,
                                                                                  self.parameters)

        # (2) remove trend-line deviation outliers, and (3) temporally isolated samples
        isValid_Running, filtData = pd_utils.madDeviationFilter(t, d, isValidSpeedFilter, self.parameters)
        # print(filtData)
        # print('----------')
        totalTime = pd_utils.getTotalTime(t, isValid_Running)

        return isBlinking, isValidRemoveOoB, isValidRemoveConf, isValidSpeedFilter, isValid_Running, speedFiltData, filtData, totalTime
