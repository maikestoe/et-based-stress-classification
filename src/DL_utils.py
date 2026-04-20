"""
Deep Learning Utilities
===========================================

This module contains various utility functions and classes to facilitate deep learning tasks
such as setting global seeds, GPU configuration, data preprocessing, model checkpoint handling,
and custom metrics for training neural networks with TensorFlow and Keras.

The following functionalities are provided:

- **Global Configuration and GPU Setup**: Functions to set global seeds and configure GPU settings.
- **Data Loading and Processing**: Functions to load and preprocess data, including standardization and balancing.
- **Custom Metrics and Losses**: Custom TensorFlow metrics (precision, recall, F1 score) and a weighted binary cross-entropy loss function.
- **Callbacks**: Custom Keras callbacks for early stopping, model checkpointing, learning rate reduction, and Optuna pruning.
- **Model Handling**: Functions to save and load model checkpoints, and clean up old models.
- **Logging**: Functions to write configuration parameters to a log file.

Modules
-------

- `TrainingSummary`: Callback to log training summary.
- `set_global_seeds`: Set global seeds for reproducibility.
- `set_gpu`: Configure GPU settings.
- `set_DL_config`: Set global seeds and configure GPU.
- `load_config`: Load configuration settings from a JSON file.
- `weighted_binary_crossentropy`: Define a custom weighted binary cross-entropy loss.
- `MacroPrecision`: Custom Keras metric for macro precision.
- `MacroRecall`: Custom Keras metric for macro recall.
- `MacroF1Score`: Custom Keras metric for macro F1 score.
- `MetricsCallback`: Callback to compute and log additional metrics at the end of each epoch.
- `setup_checkpoint`: Setup model checkpoint callback.
- `setup_early_stopping`: Setup early stopping callback.
- `setup_lr_reducer`: Setup learning rate reduction callback.
- `setup_pruning`: Setup Optuna pruning callback.
- `get_callbacks`: Get all configured callbacks for model training.
- `load_model_checkpoint`: Load model weights from a checkpoint.
- `print_class_distribution`: Print the distribution of classes in a dataset.
- `standardize_input`: Standardize training and validation data.
- `process_data`: Apply data balancing and standardization to training and validation data.
- `get_x_y`: Extract features and labels from a DataFrame.
- `get_training_data`: Get training and testing data from a DataFrame.
- `cleanup_old_models`: Remove old model files, keeping only the best ones.
- `delete_old_models`: Delete old model files except the best model for each fold.
- `write_parameters_to_log_file`: Write configuration parameters to a log file.

Example Usage
-------------

```python
from DL_utils import set_global_seeds, load_config, get_callbacks, process_data

# Set global seeds for reproducibility
set_global_seeds(42)

# Load configuration
config = load_config('configs/examples/vr_goalkeeper_cnn_pd.json')

# Get callbacks for model training
callbacks = get_callbacks(folder='results/', trial=None, model_checkpoint_path='model_checkpoint.h5')

# Preprocess data
x_train, y_train, x_val, y_val = process_data(x_train, y_train, x_val, y_val, config, scaler_directory='scalers/')

Dependencies
------------

- numpy
- tensorflow
- random
- os
- json
- sklearn
- argparse
- joblib
- logging
- optuna
- imblearn
- sklearn

Author: Maike Laut
Date: 20.06.2024
"""


import numpy as np
import os
import tensorflow as tf
import random
import json
import sklearn
import argparse
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import shuffle
from collections import Counter
from keras.mixed_precision import set_global_policy
from sklearn.utils import resample
import gc

def set_global_seeds(seed_value=42):
    """
    Sets the global random seeds for reproducibility.

    :param seed_value: The seed value to set. Default is 42.
    :type seed_value: int
    """

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    #os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    # Optional: Configure TensorFlow to limit sources of non-determinism
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    #tf.config.threading.set_intra_op_parallelism_threads(1)


def set_gpu():
    """
    Configures TensorFlow to enable memory growth for all GPUs.
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for all GPUs")
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(f"Failed to set memory growth: {e}")


def set_DL_config(config=None):
    """
    Sets the global seeds and GPU configuration for deep learning.
    """
    set_global_seeds()
    set_gpu()
    mixed_precision_enabled = True
    if config is not None:
        mixed_precision_enabled = config.get('mixed_precision', config.get('DL', {}).get('mixed_precision', True))
    set_global_policy("mixed_float16" if mixed_precision_enabled else "float32")
    #memory_info = tf.config.experimental.get_memory_info('GPU:0')
    #print('Initial memory info:', memory_info)


def _deep_update_config(base_config, override_config):
    """Recursively merge a small override config into a full base config."""
    merged = dict(base_config)
    for key, value in override_config.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_update_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config():
    """
    Loads the configuration file for the deep learning study.

    :return: Configuration dictionary.
    :rtype: dict
    """

    parser = argparse.ArgumentParser(description='Deep learning for eye-tracking-based stress classification')
    parser.add_argument('--config', type=str, help='Path to configuration file', default='configs/examples/vr_goalkeeper_cnn_pd.json')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as file:
        config = json.load(file)
    if 'base_config' in config:
        base_config_path = config['base_config']
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(os.path.dirname(config_path), base_config_path)
        with open(base_config_path, 'r') as file:
            base_config = json.load(file)
        config = _deep_update_config(base_config, config)
    del parser, args
    if 'DL' in config and 'path_results' in config['DL']:
        config['DL']['path_results'] = os.path.expandvars(config['DL']['path_results'])
    if 'df_prep_path' in config:
        config['df_prep_path'] = os.path.expandvars(config['df_prep_path'])
    if 'optuna' in config and 'storage_url' in config['optuna']:
        config['optuna']['storage_url'] = os.path.expandvars(config['optuna']['storage_url'])
    if 'path_results' in config:
        config['path_results'] = os.path.expandvars(config['path_results'])

    return config


def print_class_distribution(labels, set_name=""):
    """
    Prints the distribution of classes in a given set.

    :param labels: Target variable array.
    :type labels: np.array
    :param set_name: Name of the set (e.g., 'Training', 'Validation').
    :type set_name: str
    """

    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    total_samples = sum(counts)
    print(f"{set_name} Set Class Distribution:")
    for class_label, count in class_distribution.items():
        print(f"    Class {class_label}: {count} samples, {count / total_samples * 100:.2f}%")
    print("\n")
    del unique, counts, total_samples


def standardize_input(x_train, x_val, directory, load_scaler=False):
    """
    Standardizes the input data using the StandardScaler.

    :param x_train: Training data to be standardized.
    :type x_train: np.ndarray
    :param x_val: Validation data to be standardized.
    :type x_val: np.ndarray
    :param directory: Directory where the scaler files are saved.
    :type directory: str
    :return: Standardized training and validation data.
    :rtype: tuple
    :raises FileNotFoundError: If scaler file is not found in the specified directory.

    This function standardizes the training and validation data using a StandardScaler.
    If `x_train` is None, only `x_val` is standardized using the saved scalers in the specified directory.
    Otherwise, both `x_train` and `x_val` are standardized, and the scalers are saved in the directory.
    """

    scaler = sklearn.preprocessing.StandardScaler()

    n_samples_train, n_timesteps, n_channels = x_train.shape
    x_train_scaled = np.zeros_like(x_train)

    if (x_train is None and x_val is not None) or load_scaler:

        # Standardize x_val using the loaded scaler
        x_val_scaled = np.zeros_like(x_val)
        n_samples_val, n_timesteps, n_channels = x_val.shape
        for i in range(n_channels):
            if os.path.exists(os.path.join(directory, str(i) + '_scaler.gz')):
                # Load the scaler
                scaler = joblib.load(os.path.join(directory, str(i) + '_scaler.gz'))
                print("Scaler loaded successfully from " + directory)
            else:
                raise FileNotFoundError("Scaler file not found under ." + directory)
            x_val_channel = x_val[:, :, i].reshape(-1, 1)
            x_val_scaled[:, :, i] = scaler.transform(x_val_channel).reshape(n_samples_val, n_timesteps)
            if load_scaler:
                x_train_channel = x_train[:, :, i].reshape(-1, 1)
                x_train_scaled[:, :, i] = scaler.transform(x_train_channel).reshape(n_samples_train, n_timesteps)
        if load_scaler:
            return x_train_scaled, x_val_scaled
        else:
            return None, x_val_scaled

    if x_val is not None:
        n_samples_val = x_val.shape[0]
        x_val_scaled = np.zeros_like(x_val)
    else:
        x_val_scaled = None

    # Fit one scaler per input channel.
    for i in range(n_channels):
        x_train_channel = x_train[:, :, i].reshape(-1, 1)
        scaler.fit(x_train_channel)

        # Scale train data and assign to container
        x_train_scaled[:, :, i] = scaler.transform(x_train_channel).reshape(n_samples_train, n_timesteps)

        if x_val is not None:
            x_val_channel = x_val[:, :, i].reshape(-1, 1)
            x_val_scaled[:, :, i] = scaler.transform(x_val_channel).reshape(n_samples_val, n_timesteps)

        if directory:
            os.makedirs(directory, exist_ok=True)
            joblib.dump(scaler, os.path.join(directory, str(i) + '_scaler.gz'))
            print("Scaler saved successfully to " + directory)

    return x_train_scaled, x_val_scaled


def undersample_to_balance_classes(X, y):
    min_class_size = min([sum(y == cls) for cls in np.unique(y)])
    X_resampled, y_resampled = [], []
    for cls in np.unique(y):
        X_cls, y_cls = X[y == cls], y[y == cls]
        X_cls, y_cls = resample(X_cls, y_cls, replace=False, n_samples=min_class_size, random_state=42)
        X_resampled.append(X_cls)
        y_resampled.append(y_cls)

        # Free memory resources
        del X_cls, y_cls
        gc.collect()

    return np.vstack(X_resampled), np.concatenate(y_resampled)


def limit_class_size(X, y, max_class_size):
    unique_classes = np.unique(y)
    X_resampled, y_resampled = [], []
    for cls in unique_classes:
        X_cls, y_cls = X[y == cls], y[y == cls]
        if len(y_cls) > max_class_size:
            X_cls, y_cls = resample(X_cls, y_cls, replace=False, n_samples=max_class_size, random_state=42)
        X_resampled.append(X_cls)
        y_resampled.append(y_cls)

        # Free memory resources
        del X_cls, y_cls
        gc.collect()

    return np.vstack(X_resampled), np.concatenate(y_resampled)


def ensure_consistent_shape(df):
    # Shuffle the dataframe and
    consistent_shape = df['data'].apply(lambda x: x.shape[0]).mode()[0]
    df = df[df['data'].apply(lambda x: x.shape[0] == consistent_shape)]
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def balance_data_individualID(df_train, config):
    balanced_dfs = []
    max_class_size = 250
    for name, group in df_train.groupby('id'):
        X, y = np.vstack(group['data'].values), group['label'].values
        print('Original training samples for ID ' + str(name) + ': ' + str(len(X)))

        if config['DL']['smote']:
            print('Applying SMOTE')
            X, y = SMOTE(sampling_strategy={cls: max_class_size for cls in np.unique(y) if sum(y == cls) < max_class_size}, random_state=42).fit_resample(X, y)
            print('Training samples after SMOTE for ID ' + str(name) + ': ' + str(len(X)))
        elif config['DL']['adasyn']:
            print('Applying ADASYN')
            X, y = ADASYN(sampling_strategy={cls: max_class_size for cls in np.unique(y) if sum(y == cls) < max_class_size}, random_state=42).fit_resample(X, y)
            print('Training samples after ADASYN for ID ' + str(name) + ': ' + str(len(X)))

        if config['DL']['rand_undersample']:
            # Use random undersampling in case less than max_class_size samples exist for majority class
            #X, y = limit_class_size(X, y, max_class_size=max_class_size)
            print('Training samples after limiting class size for ID ' + str(name) + ': ' + str(len(X)))
            X, y = undersample_to_balance_classes(X, y)

        balanced_group = pd.DataFrame({'data': list(X), 'label': y, 'id': name})
        balanced_dfs.append(balanced_group)

        # Free memory resources
        del X, y, balanced_group
        gc.collect()

    del max_class_size
    df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
    if config['shuffle']:
        return ensure_consistent_shape(df_balanced)
    else:
        return df_balanced


def balance_data_whole_dataset(df_train, config):
    X = np.vstack(df_train['data'].values)
    y = df_train['label'].values
    ids = df_train['id'].values
    max_class_size = 250 * len(set(ids))
    print('Max class size: ' + str(max_class_size))

    print('Original training samples in the whole dataset: ' + str(len(X)))

    # Apply SMOTE or ADASYN if specified
    if config['DL']['smote']:
        print('Applying SMOTE')
        X, y = SMOTE(sampling_strategy={cls: max_class_size for cls in np.unique(y) if sum(y == cls) < max_class_size},
                     random_state=42).fit_resample(X, y)
        print('Training samples after SMOTE: ' + str(len(X)))
    elif config['DL']['adasyn']:
        print('Applying ADASYN')
        X, y = ADASYN(sampling_strategy={cls: max_class_size for cls in np.unique(y) if sum(y == cls) < max_class_size},
                      random_state=42).fit_resample(X, y)
        print('Training samples after ADASYN: ' + str(len(X)))

    # If specified, apply random undersampling to balance classes to the size of the smallest class
    if config['DL']['rand_undersample']:
        print('Applying random undersampling')
        X, y = undersample_to_balance_classes(X, y)
        print('Training samples after undersampling: ' + str(len(X)))

    # Create a new DataFrame to hold the balanced dataset
    balanced_df = pd.DataFrame({'data': list(X), 'label': y})

    # Shuffle and ensure consistent shape
    if config['shuffle']:
        balanced_df = ensure_consistent_shape(balanced_df)

    # Free memory resources
    del X, y, ids
    gc.collect()

    return balanced_df



def process_data_noStandardization(x_train, y_train, x_val, y_val, train_ids, config):
    """
    Applies data balancing according to the configuration and standardizes the data.

    :param x_train: Training data to be processed.
    :type x_train: np.ndarray
    :param y_train: Training labels to be processed.
    :type y_train: np.ndarray
    :param x_val: Validation data to be processed.
    :type x_val: np.ndarray
    :param y_val: Validation labels to be processed.
    :type y_val: np.ndarray
    :param train_ids: Training IDs.
    :type train_ids: np.ndarray
    :param config: Configuration settings for data processing.
    :type config: dict
    :return: Processed training and validation data and labels.
    :rtype: tuple

    This function balances the training data using SMOTE, ADASYN, or/and random undersampling based on the configuration settings.
    """

    n_samples, n_timesteps, n_features = x_train.shape
    print('n_samples beginning of function: ' + str(n_samples))

    if config['DL']['smote'] or config['DL']['adasyn'] or config['DL']['rand_undersample']:
        print('Balancing data...')
        df_train = pd.DataFrame(
            {'id': train_ids, 'label': y_train, 'data': list(x_train.reshape(x_train.shape[0], -1))})
        #df_balanced = balance_data_individualID(df_train, config)
        df_balanced = balance_data_whole_dataset(df_train, config)

        x_train_reshaped = np.vstack(df_balanced['data'].values)
        y_train = df_balanced['label'].values
        x_train = x_train_reshaped.reshape(-1, n_timesteps, x_train.shape[2])
        del df_balanced, x_train_reshaped, df_train

    print_class_distribution(y_train, set_name="Training")

    # Free memory resources
    del n_samples, n_timesteps, n_features
    gc.collect()

    return x_train, y_train, x_val, y_val


def process_data(x_train, y_train, x_val, y_val, train_ids, config, scaler_directory=[], load_scaler=False):
    """
    Applies data balancing according to the configuration and standardizes the data.

    :param x_train: Training data to be processed.
    :type x_train: np.ndarray
    :param y_train: Training labels to be processed.
    :type y_train: np.ndarray
    :param x_val: Validation data to be processed.
    :type x_val: np.ndarray
    :param y_val: Validation labels to be processed.
    :type y_val: np.ndarray
    :param train_ids: Training IDs.
    :type train_ids: np.ndarray
    :param config: Configuration settings for data processing.
    :type config: dict
    :param scaler_directory: Directory where scalers are saved.
    :type scaler_directory: list
    :return: Processed training and validation data and labels.
    :rtype: tuple

    This function performs the following steps:
    1. Standardizes the training and validation data.
    2. Balances the training data using SMOTE, ADASYN, or/and random undersampling based on the configuration settings.
    """

    if x_train is not None:
        n_samples, n_timesteps, n_features = x_train.shape
        print('n_samples beginning of function: ' + str(n_samples))

        # Standardization
        print("Applying standardization...")
        x_train, x_val = standardize_input(x_train, x_val, scaler_directory, load_scaler)

        if config['DL']['smote'] or config['DL']['adasyn'] or config['DL']['rand_undersample']:
            print('Balancing data...')
            df_train = pd.DataFrame(
                {'id': train_ids, 'label': y_train, 'data': list(x_train.reshape(x_train.shape[0], -1))})
            #df_balanced = balance_data_individualID(df_train, config)
            df_balanced = balance_data_whole_dataset(df_train, config)

            x_train_reshaped = np.vstack(df_balanced['data'].values)
            y_train = df_balanced['label'].values
            x_train = x_train_reshaped.reshape(-1, n_timesteps, x_train.shape[2])
            del df_balanced, x_train_reshaped, df_train

        print_class_distribution(y_train, set_name="Training")

        # Free memory resources
        del n_samples, n_timesteps, n_features
        gc.collect()

    return x_train, y_train, x_val, y_val


def print_class_distribution(y, set_name="Set"):
    """
        Prints the distribution of classes in the given label set.

        :param y: Labels whose distribution is to be printed.
        :type y: np.ndarray or list
        :param set_name: Name of the set being analyzed (e.g., "Training", "Validation").
        :type set_name: str, optional
        :return: None

        This function performs the following steps:
        1. Counts the number of samples in each class.
        2. Calculates the total number of samples.
        3. Prints the distribution of classes, including the number of samples and their percentage of the total.
        """
    counter = Counter(y)
    total_samples = sum(counter.values())
    print(f"{set_name} Class Distribution:")
    for label, count in counter.items():
        percentage = (count / total_samples) * 100
        print(f"    Class {label}: {count} samples, {percentage:.2f}%")
    del counter, total_samples, percentage


def get_x_y(df, idx, config):
    """
    Extracts input features and labels from the dataframe based on provided indices.

    :param df: The dataframe containing the data.
    :type df: pd.DataFrame
    :param idx: Indices to select specific rows from the dataframe.
    :type idx: list or np.array
    :param config: Configuration settings containing the names of input columns.
    :type config: dict
    :return: Tuple of input features and labels.
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    # idx contains indices of the dataframe, not positional indexes!
    x = np.array(df[config['DL']['input_cols']][df.index.isin(idx)].values.tolist())
    x = np.transpose(x, (0, 2, 1))
    y = np.array(df['lab_num'][df.index.isin(idx)].values.tolist())
    ids = np.array(df['ID'][df.index.isin(idx)].values.tolist())
    return x, y, ids


def get_training_data(df, train_idx, test_idx, config):
    """
    Extracts training and testing data from the dataframe based on provided indices.

    :param df: The dataframe containing the data.
    :type df: pd.DataFrame
    :param train_idx: Indices to select training data from the dataframe.
    :type train_idx: list or np.array
    :param test_idx: Indices to select testing data from the dataframe.
    :type test_idx: list or np.array
    :param config: Configuration settings containing the names of input columns.
    :type config: dict
    :return: Tuple containing training features, training labels, testing features, and testing labels.
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    if train_idx is not None:
        x_train, y_train, ids_train = get_x_y(df, train_idx, config)
        if config['shuffle']:
            x_train, y_train, ids_train = shuffle(x_train, y_train, ids_train, random_state=42)
    else:
        x_train, y_train, ids_train = [], [], []
    if test_idx is not None:
        x_test, y_test, ids_test = get_x_y(df, test_idx, config)
    else:
        x_test, y_test, ids_test = [], [], []

    return x_train, y_train, x_test, y_test, ids_train, ids_test


def cleanup_old_models(folder, study, keep_top_pct=5):
    """
    Removes old model files, keeping only the top percentage of best trials.

    :param folder: Path to the folder containing the model files.
    :type folder: str
    :param study: Optuna study object containing trial results.
    :type study: optuna.study.Study
    :param keep_top_pct: Percentage of top trials to keep.
    :type keep_top_pct: int
    """

    # Calculate the number of trials to keep
    num_trials_to_keep = int(len(study.trials) * (keep_top_pct / 100))
    best_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)[:num_trials_to_keep]

    # Create a set of model checkpoint paths for the best trials
    best_model_paths = {f"{folder}/best_model_trial_{trial.number}.h5" for trial in best_trials}

    # Remove all other model files
    for model_file in os.listdir(folder):
        model_path = os.path.join(folder, model_file)
        if model_path.endswith(".h5") and model_path not in best_model_paths:
            os.remove(model_path)
            print(f"Removed {model_path}")

    del num_trials_to_keep, best_trials


def delete_old_models(best_trial_number, config, test_id):
    """
    Deletes old model files, keeping only the best model for each fold.

    :param best_trial_number: The number of the best trial.
    :type best_trial_number: int
    :param config: Configuration settings containing paths and other parameters.
    :type config: dict
    :param test_id: The ID of the test.
    :type test_id: int
    """

    path_results = config['DL']['path_results'] + config['timestamp'] + '/' + config['DL']['model'] + '/' + str(config['experiment_id']) + '/test_ID_' + str(test_id) + '/'
    for fold in range(0, config['DL']['n_splits_inner']):
        path_results_id_fold = path_results + 'fold_' + str(fold) + '/'
        keep_file = 'best_weights_' + str(best_trial_number) + '_' + str(fold) + '.keras'
        #print('Keeping file: ' + keep_file)
        files_in_folder = os.listdir(path_results_id_fold)
        for file_name in files_in_folder:
            file_path = os.path.join(path_results_id_fold, file_name)
            if os.path.isfile(file_path) and file_name != keep_file:
                os.remove(file_path)
                #print(f'Deleted {file_path}')

    del path_results, path_results_id_fold, keep_file, files_in_folder


def write_parameters_to_log_file(folder, config):
    """
    Writes defined parameters for the run to a file called [timestamp]_config.txt.

    :param folder: Path to where to save the results.
    :type folder: str
    :param config: Configuration settings containing parameters for the run.
    :type config: dict
    :return: None
    """

    config_path = os.path.join(folder, config['timestamp'] + '_config.txt')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)

    del config_path
