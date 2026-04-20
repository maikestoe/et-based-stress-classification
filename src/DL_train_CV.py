"""
Training Utilities with Optuna for Deep Learning Models

This module provides utilities for training and evaluating deep learning models with integrated Optuna-based
hyperparameter optimization. It supports CNN, LSTM, and ConvLSTM models,
and includes utilities for data preparation, training execution, metric evaluation, and logging.

Author: Maike Laut
Date: 20.06.2024
"""


# Imports
import time
import os
import numpy as np
import gc
import tensorflow as tf

from DL_models import get_model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, BackupAndRestore
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import auc as auc_sklearn
from keras.metrics import F1Score, AUC, Precision, Recall
from memory_profiler import profile

from tensorflow.keras import mixed_precision

LEARNING_RATE = 0
WEIGHT_ALPHA = 0
WEIGHT_GAMMA = 0


def trim_and_split(array, num_segments, axis=0):
    """
    Split an array into specified segments.

    :param array: Input array to be split.
    :type array: np.ndarray
    :param num_segments: Number of segments to split the array into.
    :type num_segments: int
    :param axis: Axis along which to split the array.
    :type axis: int, optional
    :return: A list of split arrays.
    :rtype: list[np.ndarray]
    """

    segment_length = len(array) // num_segments
    total_length = segment_length * num_segments
    # Trim the array to make its length divisible by num_segments
    trimmed_array = array[:total_length]
    # Split the trimmed array
    del total_length
    return np.split(trimmed_array, num_segments, axis=axis)


def prepare_data_for_convlstm(x_data, num_segments):
    """
    Reshape and split data for the ConvLSTM model.

    :param x_data: Input data array.
    :type x_data: np.ndarray
    :param num_segments: Number of segments to split the data into.
    :type num_segments: int
    :return: Transformed data array.
    :rtype: np.ndarray
    """

    wLength_splitted = x_data.shape[1] // num_segments
    x_splitted = np.zeros((len(x_data), num_segments, wLength_splitted, 1))

    for idx, sample in enumerate(x_data):
        splits = np.array_split(sample[:wLength_splitted * num_segments], num_segments, axis=0)
        for i, split in enumerate(splits):
            x_splitted[idx, i, :, 0] = np.squeeze(split)
    del splits
    return np.expand_dims(x_splitted, axis=2)  # Adding an additional dimension for compatibility


#@profile
def create_tf_dataset(x_data, y_data, num_classes, batch_size, config):
    """
    Create a TensorFlow dataset.

    :param x_data: Input data array or list of arrays.
    :type x_data: np.ndarray or list[np.ndarray]
    :param y_data: Labels for the input data.
    :type y_data: np.ndarray
    :param num_classes: Number of classes.
    :type num_classes: int
    :param batch_size: Batch size for the dataset.
    :type batch_size: int
    :return: A TensorFlow dataset.
    :rtype: tf.data.Dataset
    """
    # Uncomment to prevent data shuffling
    # one-hot encoded y is needed for categorical cross-entropy loss
    y_data_categorical = tf.keras.utils.to_categorical(y_data, num_classes=num_classes)

    # Respect the experiment config instead of forcing shuffling for specific models.
    indices = np.arange(len(y_data))
    if config.get('shuffle', False):
        np.random.shuffle(indices)

    x_data_shuffled = x_data[indices]
    dataset = tf.data.Dataset.from_tensor_slices((x_data_shuffled, y_data_categorical[indices]))

    # Batch and prefetch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    del x_data_shuffled, indices, y_data_categorical
    gc.collect()
    return dataset


#@profile
def transform_data_as_input(x_train, x_val, params, config):
    """
    Transform data according to the model requirements.

    :param x_train: Training data.
    :type x_train: np.ndarray
    :param x_val: Validation data.
    :type x_val: np.ndarray
    :param params: Parameters for data transformation.
    :type params: dict
    :param config: Configuration dictionary containing model specifications.
    :type config: dict
    :return: Transformed training and validation data.
    :rtype: tuple(np.ndarray, np.ndarray)
    """

    if config['DL']['model'] in ['ConvLSTM-1', 'ConvLSTM-3']:
        print('Process data for ConvLSTM model')
        x_train = prepare_data_for_convlstm(x_train, params['num_segments'])
        print('x train shape: ' + str(x_train.shape))
        x_val = prepare_data_for_convlstm(x_val, params['num_segments'])
        # else don#t do any transformations
        print('x val shape: ' + str(x_val.shape))

    else:
        # Don’t transform inputs for CNN and LSTM models.
        print('x train shape: ' + str(x_train.shape))
        print('x val shape: ' + str(x_val.shape))

    del params, config
    return x_train, x_val



def run_single_validation_cycle(folder, x_train, x_val, y_train, y_val, data_dim, timesteps, config, num_classes, fold,
                                lr, weight_alpha, weight_gamma, batch_size, trial, params):
    """
    Runs a single validation cycle, training and testing a model with given training and validation data,
    and saves results to a log file.

    :param folder: Path to where the results will be saved.
    :type folder: str
    :param x_train: Training samples.
    :type x_train: np.ndarray
    :param x_val: Validation samples.
    :type x_val: np.ndarray
    :param y_train: Training labels.
    :type y_train: np.ndarray
    :param y_val: Validation labels.
    :type y_val: np.ndarray
    :param data_dim: Data dimensions.
    :type data_dim: int
    :param timesteps: Number of samples.
    :type timesteps: int
    :param config: Configuration dictionary containing model specifications and training parameters.
    :type config: dict
    :param num_classes: Number of classes for the classification problem.
    :type num_classes: int
    :param fold: Current training split.
    :type fold: int
    :param lr: Current learning rate.
    :type lr: float
    :param weight_alpha: Current alpha parameter for the focal loss.
    :type weight_alpha: float
    :param weight_gamma: Current gamma parameter for the focal loss.
    :type weight_gamma: float
    :param batch_size: Current batch size.
    :type batch_size: int
    :param trial: Current Optuna trial.
    :type trial: optuna.trial.Trial
    :param params: Model parameters selected for the current trial.
    :type params: dict
    :return: A tuple containing the training history, trained model, and the duration of the validation cycle.
    :rtype: tuple(keras.callbacks.History, keras.Model, str)
    """

    global LEARNING_RATE
    LEARNING_RATE = lr
    global WEIGHT_ALPHA
    global WEIGHT_GAMMA
    WEIGHT_ALPHA = weight_alpha
    WEIGHT_GAMMA = weight_gamma
    duration = time.time()

    # define the checkpoint path
    epochs = config['DL']['epochs']

    # (train and) validate the model
    x_train, x_val = transform_data_as_input(x_train, x_val, params, config)
    # from_logits=False is default for both loss functions
    loss = CategoricalFocalCrossentropy(from_logits=False, alpha=weight_alpha, gamma=weight_gamma)

    print("Training fold " + str(fold) + "...")
    model = get_model(config['DL']['model'], data_dim, timesteps, num_classes, params)
    train_dataset = create_tf_dataset(x_train, y_train, num_classes, batch_size, config)
    val_dataset = create_tf_dataset(x_val, y_val, num_classes, batch_size, config)
    del x_train, x_val
    gc.collect()

    checkpoint_path = os.path.join(folder, 'best_weights_fold' + str(fold) + '.weights.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,  # File path to save the model
        monitor='val_macro_f1_score',   #'val_macro_f1_score', #'val_auc',  #'val_auc',  # 'val_f1',  # Metric to monitor
        verbose=1,  # Verbosity mode
        save_best_only=True,  # Save only the best model
        mode='max',  # Mode (min, max, or auto)
        save_weights_only=True  # False  # Whether to save only the model weights
    )

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_macro_f1_score', patience=20, restore_best_weights=True, mode='max')

    ## Define learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_macro_f1_score', factor=0.5, patience=15, mode='max')

    # Define backup and restore callback
    backup_dir = os.path.join(folder, 'backup_fold' + str(fold))
    backup_restore = BackupAndRestore(backup_dir)

    optimizer_type = config['DL']['optimizer']
    if optimizer_type == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss=loss, metrics=[F1Score(average='macro', name='macro_f1_score'),
                                                           AUC(name='auc', multi_label=True, num_labels=2,
                                                               from_logits=False, curve='PR'),
                                                           Precision(name='precision'), Recall(name='recall'),
                                                           F1Score(average='weighted', name='weighted_f1_score')])

    K.clear_session()
    gc.collect()
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset,
                        callbacks=[checkpoint, early_stopping, lr_scheduler, backup_restore])

    del train_dataset, checkpoint, early_stopping, lr_scheduler
    K.clear_session()
    gc.collect()

    print('================EVALUATE================')
    # Make sure to use the weights from the best epoch
    model.load_weights(checkpoint_path)

    results = model.evaluate(val_dataset, verbose=1, batch_size=batch_size)   # Training and inference mode differ, e.g. in batch normalization, so this will result in different numbers than the validation metrics during training!!!
    print('========================================')

    print('+++++++++++++++++++VALIDATION RESULTS BY model.evaluate FOLD ' + str(fold) + '+++++++++++++++++++')
    headers = ['Loss', 'MacroF1', 'AUC', 'Precision', 'Recall', 'WeightedF1']
    print(tabulate([results], headers=headers, tablefmt='pretty'))
    duration = time.time() - duration
    print('finished in ', duration, ' seconds -> ', duration / 60, ' minutes')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    trial.set_user_attr("best_model_path", checkpoint_path)

    K.clear_session()
    gc.collect()

    # Compute confusion matrix and other metrics
    predictions, labels = get_predictions_and_labels(model, val_dataset)
    del val_dataset, model
    gc.collect()
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert one-hot encoded labels to binary labels if necessary
    if labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    # Convert predictions to binary labels
    if predictions.shape[1] > 1:
        pred_probs = predictions[:, 1]
        predictions = np.argmax(predictions, axis=1)
    else:
        pred_probs = predictions.flatten()
        predictions = (pred_probs > 0.5).astype(int)  # Apply threshold to get binary labels

    cm = confusion_matrix(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    print(f"Confusion matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score Macro: {f1:.4f}")
    print(f"F1-Score Weighted: {f1_weighted:.4f}")
    precision, recall, _ = precision_recall_curve(labels, predictions)
    auc_pr = auc_sklearn(recall, precision)
    print(f'Area Under Precision-Recall Curve: {auc_pr}')

    del auc_pr, precision, recall, f1_weighted, f1, predictions, pred_probs, labels, trial, headers, (
        results), optimizer_type, opt, backup_restore, backup_dir, checkpoint_path, loss, epochs
    K.clear_session()
    gc.collect()
    return history, str(duration), cm


#@profile
def get_predictions_and_labels(model, dataset):
    """
    Predict on the whole dataset and collect labels.
    Works for tf.data.Dataset that yields (x, y).
    """
    # 1) Predictions for all batches
    predictions = model.predict(dataset, verbose=0)

    # 2) Collect labels from dataset
    labels = np.concatenate([y.numpy() for _, y in dataset], axis=0)

    return predictions, labels


def get_avg_metrics(history_list):
    """
    Finds the best epoch from each cross-validation run and calculates the mean for all metrics for LOSOCV.

    :param history_list: A list containing the histories from each CV run.
    :type history_list: list[keras.callbacks.History]
    :return: A tuple containing a dictionary of metrics from the best epoch and a dictionary of average metrics from LOSOCV.
    :rtype: tuple(dict, dict)
    """

    counter = 0
    result_history = {}
    (loss, val_loss, precision, val_precision, recall, val_recall, macro_f1_score,
     val_macro_f1_score, auc, val_auc, weighted_f1_score, val_weighted_f1_score) = [], [], [], [], [], [], [], [], [], [], [], []

    for history in history_list:
        best_epoch = np.argmax(history.history['val_macro_f1_score'])
        loss.append(history.history['loss'][best_epoch])
        val_loss.append(history.history['val_loss'][best_epoch])
        precision.append(history.history['precision'][best_epoch])
        val_precision.append(history.history['val_precision'][best_epoch])
        recall.append(history.history['recall'][best_epoch])
        val_recall.append(history.history['val_recall'][best_epoch])
        macro_f1_score.append(history.history['macro_f1_score'][best_epoch])
        val_macro_f1_score.append(history.history['val_macro_f1_score'][best_epoch])
        auc.append(history.history['auc'][best_epoch])
        val_auc.append(history.history['val_auc'][best_epoch])
        weighted_f1_score.append(history.history['weighted_f1_score'][best_epoch])
        val_weighted_f1_score.append(history.history['val_weighted_f1_score'][best_epoch])

        counter = counter + 1

    result_history['loss'] = loss
    result_history['val_loss'] = val_loss
    result_history['precision'] = precision
    result_history['val_precision'] = val_precision
    result_history['recall'] = recall
    result_history['val_recall'] = val_recall
    result_history['macro_f1_score'] = macro_f1_score
    result_history['val_macro_f1_score'] = val_macro_f1_score
    result_history['auc'] = auc
    result_history['val_auc'] = val_auc
    result_history['weighted_f1_score'] = weighted_f1_score
    result_history['val_weighted_f1_score'] = val_weighted_f1_score

    avg_history = {'loss': np.mean(result_history['loss']), 'val_loss': np.mean(result_history['val_loss']),
                   'precision': np.mean(result_history['precision']),
                   'val_precision': np.mean(result_history['val_precision']),
                   'recall': np.mean(result_history['recall']),
                   'val_recall': np.mean(result_history['val_recall']),
                   'macro_f1_score': np.mean(result_history['macro_f1_score']),
                   'val_macro_f1_score': np.mean(result_history['val_macro_f1_score']),
                   'auc': np.mean(result_history['auc']),
                   'val_auc': np.mean(result_history['val_auc']),
                   'weighted_f1_score': np.mean(result_history['weighted_f1_score']),
                   'val_weighted_f1_score': np.mean(result_history['val_weighted_f1_score'])

                   }

    del val_weighted_f1_score, weighted_f1_score, val_auc, auc, val_macro_f1_score, macro_f1_score, val_recall, (
        val_precision), recall, precision, val_loss, loss, best_epoch, counter
    return result_history, avg_history
