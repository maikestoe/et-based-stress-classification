################################################################################
# Adapted from:
# Duchowski, A. T., Medlin, E., Cournia, N., Murphy, H.,
# Gramopadhye, A., Nair, S., Vorah, J., & Melloy, B. (2002).
# 3-D eye movement analysis. Behavior Research Methods, Instruments,
# & Computers, 34(4), 573-591. https://doi.org/10.3758/BF03195486
#
# Compute number of fixations and fixation duration with different saccade detection methods
################################################################################

"""
Preprocessing and Analysis of Gaze Data

This script provides functions to compute metrics related to eye movements, including the number of fixations and
fixation durations using different saccade detection methods. The methods are adapted from the 3-D eye movement
analysis approach by Duchowski et al. (2002; https://doi.org/10.3758/BF03195486).

Functions:
----------
- getPosition(eye, head, filterSettings, offset=None):
    Converts the gaze vector from x-, y-, z- coordinates to the visual angle.

- getVelocity(position, filterSettings):
    Computes the angular velocity.

- getAcceleration(velocity, filterSettings):
    Computes the angular acceleration.

- asymptoticModel(position, filterSettings):
    Computes the asymptotic model.

- detectSaccadesVelocity(velocity, eye, filterSettings, method):
    Detects saccades based on the velocity and saves the captured saccades in relation to the (x,y,z) gaze vector in the 2darray eye.

- getSaccadesAcceleration(acceleration, eye, filterSettings):
    Detects saccades based on the acceleration with an adaptive threshold and saves the captured saccades in relation to the (x,y,z) gaze vector in the 2darray eye.

- getAdaptiveThreshold(acceleration, i, filterSettings):
    Calculates the adaptive threshold for every sample i.

- getFixations(method, eye, timestamp, filterSettings, focused_object):
    Calculates the number of fixations and fixation duration.

- preprocessing_pd(df, filterSettings):
    Function for removing blinks and samples around blinks for vr_goalkeeper data.

Dependencies:
-------------
- numpy
- math
- pandas

Author: Maike Laut
Date: 20.06.2024
"""

import numpy as np
import math


def getPosition(eye, head, filterSettings, offset=None):
    """
    Convert the gaze vector from x-, y-, z- coordinates to the visual angle.

    :param eye: (x, y, z)-coordinates of the gaze direction.
    :type eye: numpy.ndarray
    :param head: (x, y, z)-coordinates of the head position.
    :type head: numpy.ndarray
    :param filterSettings: Settings for the fixation filters.
    :type filterSettings: dict
    :param offset: Offset depending on dataset. Defaults to None, otherwise list with one element per eye data dimension.
    :type offset: None or list

    :returns: Visual angle/position.
    :rtype: numpy.ndarray
    """

    #initialize variables
    if offset is None:
        offset = [0, 0, 0]

    length = filterSettings['length']
    counter = np.zeros([length])
    denominator = np.zeros([length])
        
    v = np.zeros([3, length + 1])
    v_i = np.zeros([length])
    v_i1 = np.zeros([length])
    average_head = np.zeros([3])
    position = np.zeros([length])
    
    #calculate eye direction vector: v = p - average(h)     
    for i in range(0, 3):
        average_head[i] = (sum(head[i]) / (length + 1))
        average_head[i] += offset[i]
        for j in range(length + 1):
            v[i][j] = eye[i][j] - average_head[i]
        v[i][j] = v[i][j]

    #calculate visual angle = cos-1(v(i)*v(i+1)/|v(i)|*|v(i+1)|)
    for j in range(length):  # counter: dot product of vi and vi+1
        for i in range(0, 3):
            counter[j] += (v[i][j] * v[i][j + 1])
            v_i[j] += v[i][j] ** 2
            v_i1[j] += v[i][j + 1] ** 2
        denominator[j] = math.sqrt(v_i[j]) * math.sqrt(v_i1[j])  # in radians
        arccos_value = counter[j] / denominator[j]

        if arccos_value > 1:
            arccos_value = 1
        elif arccos_value < -1:
            arccos_value = -1

        position[j] = math.degrees(np.arccos(arccos_value))
    return position


def getVelocity(position, filterSettings): 
    """
    Computes the angular velocity.

    :param position: The eye data converted into visual angle.
    :type position: numpy.ndarray
    :param filterSettings: Filter settings.
    :type filterSettings: dict

    :returns: Angular velocity.
    :rtype: numpy.ndarray
    """
        
    #initialize variables
    kv = filterSettings['velocity']['low-pass filter']
    h = filterSettings['FIR filter - low-pass'][str(kv) + '-tap']
    frame_rate = filterSettings['frameRate']
    length = filterSettings['length']
    
    velocity = np.zeros([length - kv])

    #calculating angular velocity = (1/Δt) * ∑ (position * h)
    for i in range(length - kv):
        for j in range(kv):
            velocity[i] = (velocity[i] + position[i + j] * h[j])
        velocity[i] = velocity[i] / (frame_rate * kv)
        
    return velocity    


def getAcceleration(velocity, filterSettings):
    """
    Computes the angular acceleration.

    :param velocity: Angular velocity.
    :type velocity: numpy.ndarray
    :param filterSettings: Filter settings.
    :type filterSettings: dict

    :returns: Angular acceleration.
    :rtype: numpy.ndarray
    """

    #initialize variables
    kv = filterSettings['acceleration']['low-pass filter']
    ka = filterSettings['acceleration']['high-pass filter']
    g = filterSettings['FIR filter - high-pass'][str(ka) + '-tap']    
    frameRate = filterSettings['frameRate']
    length = filterSettings['length']
    
    acceleration = np.zeros([length - kv - ka])
    
    #calculating angular acceleration(i) = (1/Δt) * ∑ (velocity(i+j) * h)
    for i in range(length - kv - ka):
        for j in range(ka):
            acceleration[i] = acceleration[i] + velocity[i + j] * g[j]
        acceleration[i] = acceleration[i] / (frameRate * ka)
    return acceleration


def asymptoticModel(position, filterSettings):
    """
    Computes the asymptotic model.

    :param position: The eye data converted into visual angle.
    :type position: numpy.ndarray
    :param filterSettings: Filter settings.
    :type filterSettings: dict

    :returns: Asymptotic model.
    :rtype: numpy.ndarray
    """

    #initialize variables
    asymptote = filterSettings['asymptotic_model']['asymptote']
    asymptoticModel = np.zeros([len(position)])
    #calculate asymptotic model
    for i in range(len(position) - 1):
        asymptoticModel[i] = asymptote * (1 - math.exp((-1) * (position[i] / 15)))
        
    return asymptoticModel


def detectSaccadesVelocity(velocity, eye, filterSettings, method):
    """
    Detects saccades based on the velocity and saves the captured saccades in relation to the (x,y,z) gaze vector in the 2darray eye.

    :param velocity: Angular velocity or asymptotic model.
    :type velocity: numpy.ndarray
    :param filterSettings: Filter settings.
    :type filterSettings: dict
    :param eye: Gaze vector.
    :type eye: numpy.ndarray
    :param method: Method.
    :type method: str

    :return: None
    """
       
    #initialize variables   
    length = filterSettings['length']
    kv = filterSettings['velocity']['low-pass filter']
    threshold = filterSettings[method]['threshold']

    index = 3
    if method == 'velocity':
        index = index + 1
    
    #detect saccades based on a self-chosen threshold
    for i in range(length - kv):
        if abs(velocity[i]) >= threshold:
            #saccade
            eye[index][i] = 1

        else:
            #fixation
            eye[index][i] = 0


def getSaccadesAcceleration(acceleration, eye, filterSettings):
    """
    Detects saccades based on the acceleration with an adaptive threshold and saves the captured saccades in relation to the (x,y,z) gaze vector in the 2darray eye.

    :param acceleration: Angular acceleration.
    :type acceleration: numpy.ndarray
    :param filterSettings: Filter settings.
    :type filterSettings: dict
    :param eye: Gaze vector to save the found saccades to the (x,y,z) coordinates.
    :type eye: numpy.ndarray

    :return: Gaze vector with the detected saccades in assignment to the (x,y,z) coordinates.
    :rtype: numpy.ndarray
    """
        
    #initialize variables      
    length = filterSettings['length']
    kv = filterSettings['acceleration']['low-pass filter']
    ka = filterSettings['acceleration']['high-pass filter']
    tmin = filterSettings['acceleration']['tmin']
    tmax = filterSettings['acceleration']['tmax']
    
    #detect saccades based on an adaptive threshold
    for i in range(length - kv - ka):
        
        if abs(acceleration[i]) >= getAdaptiveThreshold(acceleration, i, filterSettings) :
            #check whether saccade duration lays within a predefined saccade length
            for j in range(i + tmin, i + tmax, 1):
                if j < length - kv - ka:
                    #check if acceleration is higher than the adaptive threshold B and if there is a change of sign 
                    if abs(acceleration[j]) >= getAdaptiveThreshold(acceleration, j, filterSettings) and \
                            (np.sign(acceleration[j]) != np.sign(acceleration[i])):
                        for l in range(i, j):
                            #saccade
                            eye[5][l] = 1
                    else:
                        #fixation
                        eye[5][i] = 0
                else:
                    break
                
    return eye


def getAdaptiveThreshold(acceleration, i, filterSettings):
    """
    Calculates the adaptive threshold for every sample i.

    :param acceleration: Angular acceleration.
    :type acceleration: ndarray
    :param i: Single sample of acceleration.
    :type i: int
    :param filterSettings: Filter settings.
    :type filterSettings: dict

    :return: Adaptive threshold.
    :rtype: int
    """
    
    #initialize variables
    kv = filterSettings['acceleration']['low-pass filter']
    ka = filterSettings['acceleration']['high-pass filter']  
    constant = filterSettings['acceleration']['constant']
    length = filterSettings['length']
    
    threshold = 0
    #calculate adaptive threshold = constant + sqrt(1/k ∑ (acceleration(i+k))**2)
    for j in range(0, ka, 1):
        if j < length - kv - ka - i:
            threshold += acceleration[i+j] ** 2
        else: 
            break
    
    threshold = math.sqrt((1 / ka) * threshold)
    #add constand t
    threshold += constant
    
    return threshold


def getFixations(method, eye, timestamp, filterSettings, focused_object):
    """
        Calculates the number of fixations and fixation duration.

        :param method: The chosen saccade detection method.
        :type method: str
        :param eye: Gaze vector to save the found fixations and fixation duration to the (x, y, z) coordinates.
        :type eye: numpy.ndarray
        :param timestamp: Time to calculate fixation duration.
        :type timestamp: numpy.ndarray
        :param filterSettings: Filter settings.
        :type filterSettings: dict
        :param focused_object: Focused objects during fixation.
        :type focused_object: numpy.ndarray

        :return: Tuple containing:
            - fixation_features (dict): Fixation features including mean duration, count, and counts for different objects.
            - fixation_array (numpy.ndarray): Array indicating fixation samples.
        :rtype: tuple(dict, numpy.ndarray)
        """

    count_surrounding = 0
    count_ball_player = 0

    length = filterSettings['length']
    fixation = False
    fixation_list = []
    fixation_duration = 0.0
    fixation_start = counter = 0

    if method == 'asymptotic_model':
        kv = filterSettings['velocity']['low-pass filter']
        ka = 7
        index = 3
    elif method == 'velocity':
        kv = filterSettings['velocity']['low-pass filter']
        ka = 7
        index = 4
    else:
        kv = filterSettings['acceleration']['low-pass filter']
        ka = filterSettings['acceleration']['high-pass filter']
        index = 5

    for i in range(length - ka - kv):

        # ensure that the last sample is a 1 (saccade -> count a started fixation)
        if i == length - ka - kv - 1:
            eye[index][i] = 1

        #check whether current sample is a fixation
        if eye[index][i] == 0:
            #check if current sample is the start point of the fixation
            if not fixation:
                fixation_start = i
            fixation = True

        # if current sample isn´t a fixation
        else:
            #check if current sample is the end point of the fixation
            if fixation:
                #Calculation of the fixation duration by subtracting the end point from the start point of the fixation
                tmp_duration = float(timestamp[i]) - float(timestamp[fixation_start])
                # add fixation duration to the fixation of the chosen method
                eye[(index + 3)][fixation_start] = tmp_duration
                # add calculated fixation duration to the previous one to calculate the mean afterward
                fixation_duration += tmp_duration

                counter = counter + 1

                # AOI ANALYSIS
                tmp_objects = focused_object[fixation_start:i]
                tmp_objects = tmp_objects[tmp_objects != 'goal']
                tmp_objects = tmp_objects[tmp_objects != 'handball goal']

                count_surrounding += len(tmp_objects[tmp_objects == 'Nan'])
                count_ball_player += len(tmp_objects[tmp_objects != 'Nan'])

            fixation = False
        fixation_list.append(fixation)

    if counter == 0:
        meanfixation_duration = fixation_duration = float(timestamp[i]) - float(timestamp[fixation_start])
    else:
        meanfixation_duration = fixation_duration / counter  # compute mean fixation duration

    fixation_array = np.array(fixation_list)

    fixation_features = {
        'meanfixationDuration_' + str(method): meanfixation_duration,
        'fixation_durations_' + str(method): fixation_duration,
        'counter_' + str(method): counter / len(focused_object),  # normalize by number of samples
        'count_surroundings_' + str(method): count_surrounding / len(focused_object),
        'count_ball_player_' + str(method): count_ball_player / len(focused_object) # normalize by number of samples
    }

    return fixation_features, fixation_array


def preprocessing_pd(df, filterSettings):
    """
    Function for removing blinks and samples around blinks for vr_goalkeeper data.

    :param df: Gaze data.
    :type df: pandas.DataFrame
    :param filterSettings: Filter parameters.
    :type filterSettings: dict

    :returns: Gaze data cleansed of blinks.
    :rtype: pandas.DataFrame
    """

    #initialize variables
    exclude = ['gehalten', 'Tooor!', 'Flugphase + Kontakt', 'Schuss']
    eyedata = df[['time', 'eye-x', 'eye-y', 'eye-z', 'head-x', 'head-y', 'head-z', 'animation', 'normal time', 'ID',
                  'blink left', 'blink right']]
    blinks = [np.array(df['blink left']), np.array(df['blink right'])]
    isValid = ~blinks[0] & ~blinks[1] 
    t = df[['time']]
    gap = filterSettings['gap_blinks']

    #detect blinks in dataframe
    indx = np.where(~isValid)[0]
    for i in indx:
        j = i

        #edge detection
        if i <= (gap - 1):
            isValid[:i] = False 
            continue
        if i >= len(t) - (gap +1):
            isValid[i:] = False
            break
        
        #mark data 100ms before and after blink
        while t['time'][j] <= t['time'][i] + gap:
            
            isValid[j] = False
            j += 1 
            
        while t['time'][j] >= t['time'][i] - gap:
            isValid[j] = False 
            j -= 1

    #special treatment for flight phase
    for i in exclude:
        isValid[eyedata['animation'] == i] = True

    #reject invalid samples
    return eyedata[:][isValid]

