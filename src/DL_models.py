"""
Deep Learning Models
=============================

This module provides various functions and classes to create deep learning models using TensorFlow and Keras.
The models include convolutional neural networks (CNNs), long short-term memory
networks (LSTMs), and convolutional LSTM models for single-input time-series classification.

The following functionalities are provided:

Functions:
----------
- cnn_model(data_dim, timesteps, num_classes, params):
    Creates a 1D CNN model.

- lstm1_model(data_dim, timesteps, num_classes, params):
    Creates a single-layer LSTM model.

- lstm3_model(data_dim, timesteps, num_classes, params):
    Creates a three-layer LSTM model.

- convlstm1_model(data_dim, timesteps, num_classes, params):
    Creates a single-layer ConvLSTM model.

- convlstm3_model(data_dim, timesteps, num_classes, params):
    Creates a three-layer ConvLSTM model.

- get_model(model, data_dim, timesteps, num_classes, params):
    Returns the specified model.

Dependencies:
-------------
- tensorflow
- tensorflow.keras

Author: Maike Laut
Date: 20.06.2024
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dropout, Flatten, Dense, ConvLSTM2D, BatchNormalization,
                                     LSTM, Input)
from tensorflow.keras import regularizers

from tensorflow.keras.initializers import GlorotUniform


def cnn_model(data_dim, timesteps, num_classes, params):
    """
    Creates a 1D convolutional neural network (CNN) model.

    :param int data_dim: Dimension of input data.
    :param int timesteps: Number of samples.
    :param int num_classes: Number of classes the model should predict.
    :param dict params: Model parameters, such as the number of filters, kernel size, pool size, and dropout rates.
    :return: A Keras Sequential model ready to be compiled.
    :rtype: keras.Sequential
    """

    initializer = GlorotUniform(seed=66)

    # Define the input layer
    inputs = Input(shape=(timesteps, data_dim))

    # Define the first Conv1D layer
    x = Conv1D(filters=params['filters1'], kernel_size=params['k_size'], activation='relu', name='conv1',
               kernel_initializer=initializer)(inputs)
    x = MaxPooling1D(pool_size=params['p_size'], name='maxpool1')(x)

    # Define the second Conv1D layer
    x = Conv1D(filters=params['filters2'], kernel_size=params['k_size'], activation='relu', name='conv2',
               kernel_initializer=initializer)(x)
    x = Dropout(params['drop1'], name='drop1')(x)
    x = MaxPooling1D(pool_size=params['p_size'], name='maxpool2')(x)

    # Define the third Conv1D layer
    x = Conv1D(filters=params['filters3'], kernel_size=params['k_size'], activation='relu', name='conv3',
               kernel_initializer=initializer)(x)
    x = Dropout(params['drop2'], name='drop2')(x)
    x = MaxPooling1D(pool_size=params['p_size'], padding='same', name='maxpool3')(x)

    x = Flatten()(x)
    x = Dropout(params['drop3'], name='drop3')(x)
    x = Dense(params['dense'], activation='relu', name='dense')(x)
    outputs = Dense(num_classes, activation='softmax', name='softmax', kernel_initializer=initializer)(x)

    # Create the model
    cnn = Model(inputs=inputs, outputs=outputs)

    del outputs, x, inputs, initializer
    return cnn


def lstm1_model(data_dim, timesteps, num_classes, params):
    """
    Creates a single-layer LSTM model.

    :param int data_dim: Dimension of input data.
    :param int timesteps: Number of samples.
    :param int num_classes: Number of classes the model should predict.
    :param dict params: Model parameters, such as the number of filters, kernel size, pool size, and dropout rates.

    :return: A Keras Sequential model ready to be compiled.
    :rtype: keras.Sequential
    """

    lstm1 = Sequential()
    lstm1.add(LSTM(params['units'], input_shape=(timesteps, data_dim)))
    lstm1.add(BatchNormalization())
    lstm1.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=params['k_l1'], l2=params['k_l2'])))

    return lstm1


def lstm3_model(data_dim, timesteps, num_classes, params):
    """
    Creates a three-layer LSTM model.

    :param int data_dim: Dimension of input data.
    :param int timesteps: Number of samples.
    :param int num_classes: Number of classes the model should predict.
    :param dict params: Model parameters, such as the number of filters, kernel size, pool size, and dropout rates.

    :return: A Keras Sequential model ready to be compiled.
    :rtype: keras.Sequential
    """

    lstm3 = Sequential()
    lstm3.add(LSTM(params['units'], return_sequences=True, input_shape=(timesteps, data_dim)))
    lstm3.add(BatchNormalization())
    lstm3.add(LSTM(params['units2'], return_sequences=True))
    lstm3.add(BatchNormalization())
    lstm3.add(LSTM(params['units3']))
    lstm3.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=params['k_l1'], l2=params['k_l2'])))

    return lstm3


def convlstm1_model(data_dim, timesteps, num_classes, params):
    """
    Creates a single-layer ConvLSTM model.

    :param int data_dim: Dimension of input data.
    :param int timesteps: Number of samples.
    :param int num_classes: Number of classes the model should predict.
    :param dict params: Model parameters, such as the number of filters, kernel size, pool size, and dropout rates.

    :return: A Keras Sequential model ready to be compiled.
    :rtype: keras.Sequential
    """

    convlstm1 = Sequential()
    convlstm1.add(ConvLSTM2D(params['filter'], kernel_size=(1, params['k_size']), activation='relu', input_shape=(params['num_segments'], 1, int(timesteps/params['num_segments']), data_dim)))   # 64, 3
    convlstm1.add(Dropout(params['drop']))  # 0.5
    convlstm1.add(Flatten())
    convlstm1.add(Dense(num_classes, activation='softmax'))
    return convlstm1


def convlstm3_model(data_dim, timesteps, num_classes, params):
    """
    Creates a three-layer ConvLSTM model.

    :param int data_dim: Dimension of input data.
    :param int timesteps: Number of samples.
    :param int num_classes: Number of classes the model should predict.
    :param dict params: Model parameters, such as the number of filters, kernel size, pool size, and dropout rates.

    :return: A Keras Sequential model ready to be compiled.
    :rtype: keras.Sequential
    """

    convlstm3 = Sequential()
    convlstm3.add(ConvLSTM2D(params['filter1'], kernel_size=(1, params['k_size']), activation='relu', input_shape=(params['num_segments'], 1, int(timesteps/params['num_segments']), data_dim), return_sequences=True))   # 64, 3
    convlstm3.add(Dropout(params['drop1']))
    convlstm3.add(ConvLSTM2D(params['filter2'], kernel_size=(1, params['k_size']), activation='relu', return_sequences=True))
    convlstm3.add(Dropout(params['drop2']))
    convlstm3.add(ConvLSTM2D(params['filter3'], kernel_size=(1, params['k_size']), activation='relu', return_sequences=True))
    convlstm3.add(Dense(params['dense'], activation='relu'))
    convlstm3.add(Flatten())
    convlstm3.add(Dense(num_classes, activation='softmax'))
    return convlstm3


def get_model(model, data_dim, timesteps, num_classes, params):
    """
    Returns the specified model.

    :param str model: Name of the model.
    :param int data_dim: Dimension of input data.
    :param int timesteps: Number of samples.
    :param int num_classes: Number of classes the model should predict.
    :param dict params: Model parameters, e.g., number of filters. Depends on the model type.

    :return: A Keras model ready to be compiled.
    :rtype: keras.Sequential
    """

    if model == 'CNN':
        return cnn_model(data_dim, timesteps, num_classes, params)
    if model == 'LSTM-1':
        return lstm1_model(data_dim, timesteps, num_classes, params)
    if model == 'LSTM-3':
        return lstm3_model(data_dim, timesteps, num_classes, params)
    if model == 'ConvLSTM-1':
        return convlstm1_model(data_dim, timesteps, num_classes, params)
    if model == 'ConvLSTM-3':
        return convlstm3_model(data_dim, timesteps, num_classes, params)
    raise ValueError(f"Unsupported model type: {model}")
