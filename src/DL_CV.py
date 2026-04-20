"""
This script performs deep learning model training and parameter optimization using Optuna,
based on a leave-one-subject-out cross-validation scheme. It saves results including metrics,
best parameters, and training logs.

Dependencies:
- Python 3.7
- Required Python packages: pickle5, os, optuna, gzip, logging, numpy, DL_train_CV, joblib, warnings,
  gc, tabulate, matplotlib.pyplot, tensorflow.keras.backend, sklearn.model_selection.GroupShuffleSplit,
  sklearn.utils.class_weight.compute_class_weight, DL_utils.set_DL_config, DL_utils.load_config,
  DL_utils.process_data, DL_utils.get_training_data, DL_utils.delete_old_models, DL_utils.write_parameters_to_log_file

Usage:
- Ensure all necessary dependencies are installed.
- Configure the DL_config file.
- Run the script to perform model training and optimization.

Author: Maike Laut
Date: 20.06.2024
"""

# imports
import os
import optuna
import gzip
import logging
import numpy as np
import DL_train_CV
import gc
import pandas as pd
import pickle
import subprocess

from tabulate import tabulate
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_param_importances
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn.model_selection import GroupShuffleSplit
from DL_utils import set_DL_config, load_config, process_data, get_training_data, write_parameters_to_log_file, set_gpu
from memory_profiler import profile
from pickle_compat import install_pandas_pickle_compat

import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
# Info, warning, error and critical messages are logged (time, log level, message) to console

global trial
optuna.logging.disable_default_handler()  # use custom logging configuration
install_pandas_pickle_compat()


def get_mean_std_from_hist_list(history_list, metric_name):
    """
    Calculate the mean and standard deviation of a specified metric from a list of training histories.

    :param history_list: List of training history objects.
    :type history_list: list
    :param metric_name: Name of the metric to calculate mean and std (e.g., 'val_accuracy').
    :type metric_name: str
    :return: A tuple containing the overall mean and standard deviation of the specified metric.
    :rtype: tuple
    """

    values = []  # To store the final precision values from each history

    # Iterate through each history object in the list
    for history in history_list:
        values.append(history[metric_name])

    overall_mean = np.mean(values)
    overall_std = np.std(values)
    return overall_mean, overall_std


def select_param_set(config, trial):
    """
    Choose a parameter set for the current trial.

    :param trial: The trial object from Optuna.
    :type trial: optuna.trial.Trial
    :param config: Configuration dictionary specifying model and optuna parameters.
    :type config: dict
    :return: A dictionary containing the selected parameter set, suitable for the model architecture.
    :rtype: dict
    """

    model = config['DL']['model']
    model_params = config['optuna'][model]
    params = {}

    for param_name, param_config in model_params.items():
        params[param_name] = suggest_param(trial, param_name, param_config)
    return params


def suggest_param(trial, name, config):
    """
    Suggest a parameter for the trial based on its configuration.

    :param trial: The trial object.
    :type trial: optuna.trial.Trial
    :param name: The parameter name.
    :type name: str
    :param config: The parameter configuration including its range and type.
    :type config: list
    :return: The suggested parameter value.
    :rtype: The type of the suggested parameter value.
    """
    if config[-1] == 'categ':
        return trial.suggest_categorical(name, config[:-1])
    elif config[-1] == 'float':
        return trial.suggest_float(name, *config[:-1], log=False)
    elif config[-1] == 'loguni:':
        return trial.suggest_float(name, *config[:-1], log=True)
    else:  # Assuming 'uni' for uniform distribution/ deprecated so using suggest_float
        return trial.suggest_float(name, *config[:-1])


#@profile
def evaluate(duration_list, history_list):
    """
    Print the average results and return the average training and validation metrics.

    :param duration_list: List of durations.
    :type duration_list: list
    :param history_list: List of training histories.
    :type history_list: list
    :return: The average validation accuracy.
    :rtype: float
    """


    # create logs
    result_history, avg_history = DL_train_CV.get_avg_metrics(history_list)
    total_duration = np.sum(duration_list)

    print('=== Average of results ===')
    header = ('loss', 'val_loss', 'macro_precision', 'val_macro_precision', 'macro_recall', 'val_macro_recall',
              'macro_f1_score', 'val_macro_f1_score', 'auc', 'val_auc', 'weighted_f1_score', 'val_weighted_f1_score')
    row = [avg_history['loss'], avg_history['val_loss'], avg_history['precision'],
           avg_history['val_precision'], avg_history['recall'], avg_history['val_recall'],
           avg_history['macro_f1_score'], avg_history['val_macro_f1_score'], avg_history['auc'], avg_history['val_auc'],
           avg_history['weighted_f1_score'], avg_history['val_weighted_f1_score']]
    print(tabulate([row], headers=header, tablefmt='pretty'))
    print('finished in ', total_duration, ' seconds -> ', total_duration / 60, ' minutes')
    print('=========================')
    del result_history, total_duration

    return avg_history


#@profile
def train_DL_model(lr, weight_alpha, weight_gamma, batch_size, cur_trial, params):
    """
    Train the specified deep learning model by running a single training and validation cycle using cross-validation.

    :param lr: Selected learning rate.
    :type lr: float
    :param weight: Positive weight for loss.
    :type weight: float
    :param batch_size: Selected batch size.
    :type batch_size: int
    :param cur_trial: Current Optuna optimization trial.
    :type cur_trial: optuna.trial.Trial
    :param params: Specified parameters for the DL model, including output folder path, etc.
    :type params: dict
    :return: A tuple containing the validation accuracy, trained DL model, and parameters.
    :rtype: tuple(float, Any, dict)
    """

    logger = logging.getLogger('train_DL_models')

    # Avoid adding multiple handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    try:
        logger.info(f"Starting training model with lr={lr}, weight_alpha={weight_alpha}, weight_gamma={weight_gamma}, "
                    f"batch_size={batch_size}, params={params}")

        duration_list, history_list = list(), list()

        # load data
        with open(config['df_prep_path'], "rb") as fh:
            df = pickle.load(fh)
            df['lab_num'] = df['lab_num'].astype(int)

        # Leave out the test ID
        df_sub = df[df.ID != test_id]
        df_sub = df_sub[df_sub.ID.isin(config['valid_IDs'])]

        del df
        gc.collect()
        K.clear_session()

        # n_splits-fold Cross Validation
        gss = GroupShuffleSplit(n_splits=config['DL']['n_splits_inner'], test_size=config['DL']['validation_size_inner'],
                                random_state=42)

        print('Start training ' + str(config['DL']['model']) + '  model...')
        fold = 0
        # Create folder for results
        folder = config['DL']['path_results'] + str(config['timestamp']) + '/' + config['DL']['model'] + '/'
        os.makedirs(folder, exist_ok=True)

        print('TRIAL: ' + str(cur_trial.number))

        confusion_matrices, loss_list, macro_f1_list, val_loss_list, val_macro_f1_list = [], [], [], [], []
        folder_out = os.path.join(folder, config['experiment_id'], 'test_ID_' + str(test_id))
        for train_pos, validate_pos in gss.split(df_sub, groups=df_sub.ID):

            train = df_sub.index[train_pos]
            validate = df_sub.index[validate_pos]

            x_train, y_train, x_val, y_val, train_ids, test_ids = get_training_data(df_sub, train, validate, config)

            # Apply data balancing if necessary and standardize each input column individually and apply balancing techniques
            x_train, y_train, x_val, y_val = process_data(x_train, y_train, x_val, y_val, train_ids, config)
            del train_ids, test_ids

            #memory_info = tf.config.experimental.get_memory_info('GPU:0')
            #print('Memory info prior to inner loop training function:', memory_info)
            (history, duration, cm) = DL_train_CV.run_single_validation_cycle(folder_out,
                                                                                 x_train, x_val, y_train, y_val,
                                                                                 np.size(x_train, 2),
                                                                                 np.size(x_train, 1),
                                                                                 config, 2, fold, lr,
                                                                                 weight_alpha, weight_gamma,
                                                                                 batch_size, cur_trial, params)
            del x_train, y_train, x_val, y_val
            print('Single run completed')

            confusion_matrices.append(cm)
            loss_list.append(history.history['loss'])
            macro_f1_list.append(history.history['macro_f1_score'])
            val_loss_list.append(history.history['val_loss'])
            val_macro_f1_list.append(history.history['val_macro_f1_score'])
            best_epoch = np.argmax(history.history['val_macro_f1_score'])
            best_val_f1 = history.history['val_macro_f1_score'][best_epoch]
            cur_trial.report(best_val_f1, fold)
            fold += 1

            duration_list.append(float(duration))
            history_list.append(history)

            if cur_trial.should_prune():
                print('Pruning trial')
                gc.collect()
                K.clear_session()
                raise optuna.TrialPruned()

            # Clear memory after each fold
            del history, duration, cm
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()

        print('Not pruned')
        avg_history = evaluate(duration_list, history_list)
        aggregated_cm = np.sum(confusion_matrices, axis=0)
        cur_trial.set_user_attr("confusion_matrix", aggregated_cm.tolist())
        cur_trial.set_user_attr("loss_list", loss_list)
        cur_trial.set_user_attr("val_loss_list", val_loss_list)
        cur_trial.set_user_attr("macro_f1_score_list", macro_f1_list)
        cur_trial.set_user_attr("val_macro_f1_score_list", val_macro_f1_list)

        logger.info(f"Training of trial completed successfully with average validation macro f1-score={avg_history['val_macro_f1_score']}")

        print('Returning: ' + str(avg_history['val_macro_f1_score']))
        return avg_history['val_macro_f1_score'], params

    except optuna.exceptions.TrialPruned:
        print(f"Trial {cur_trial.number} pruned")
        raise
    except Exception as e:
        print('Error in trial')
        logger.error(f"Error in trial {cur_trial.number}: {e}", exc_info=True)
        return None, None

    finally:
        # Clean up the logger after the trial
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        logger = None


#@profile
def create_objective(config):
    """
    Create the objective function for Optuna to optimize. This function defines the hyperparameters to be suggested and
    evaluates a cross-validation score.

    :param config: Configuration dictionary specifying model and Optuna parameters.
    :type config: dict
    :return: The objective function for Optuna to use in the optimization process.
    :rtype: function
    """

    def objective(trial):
        global best_model_path
        # define objective to optimize. Contains suggesting parameters and evaluating for a cross-validation score
        K.clear_session()
        lr = trial.suggest_float('learning_rate', config['optuna']['lr_low'], config['optuna']['lr_high'], log=True)
        weight_alpha = trial.suggest_float('weight_alpha', config['optuna']['weight_alpha'][0],
                                           config['optuna']['weight_alpha'][1], log=False)
        weight_gamma = trial.suggest_float('weight_gamma', config['optuna']['weight_gamma'][0],
                                           config['optuna']['weight_gamma'][1], log=False)
        batch_size = trial.suggest_categorical('batch_size', config['optuna']['batch_size'])

        print('Trial parameters:')
        print('Learning rate: ' + str(lr))
        print('Weight alpha: ' + str(weight_alpha))
        print('Weight gamma: ' + str(weight_gamma))
        print('Batch size: ' + str(batch_size))

        params = select_param_set(config, trial)
        cross_val_score, best_inner_params = train_DL_model(lr, weight_alpha, weight_gamma, batch_size, trial, params)
        best_model_path = path_results + 'test_ID_' + str(test_id)

        if cross_val_score is None:
            print('Cross val score is None!')
            return float('-inf')

        # Check if this trial has the best performance and save parameters
        if cross_val_score > trial.study.user_attrs.get("best_score", float('-inf')):
            trial.study.set_user_attr("best_score", cross_val_score)
            trial.study.set_user_attr("best_params", best_inner_params)

        return cross_val_score

    return objective


#@profile
def log_progress(study, trial):
    """
    Log progress after each trial to monitor and debug.

    :param study: The Optuna study object.
    :type study: optuna.study.Study
    :param trial: The completed Optuna trial.
    :type trial: optuna.trial.Trial
    :return: None
    :rtype: None
    """

    print(f"Trial {trial.number} completed: {trial.value}")
    print(f"Best so far: {study.best_trial.value}")


#@profile
def cleanup_checkpoints(study):
    best_trial = study.best_trial
    for trial in study.trials:
        if trial != best_trial:
            best_model_path = trial.user_attrs.get("best_model_path")
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                trial.user_attrs["best_model_path"] = None  # Clear reference to path
    gc.collect()


#@profile
def create_save_training_plots(path_figures, val_macro_f1, macro_f1, loss, val_loss, identity):
    """
    Save training plots for accuracy and loss.

    :param path_figures: Path to save the figures.
    :type path_figures: str
    :param history: Training history object.
    :type history: keras.callbacks.History
    :return: None
    :rtype: None
    """

    os.makedirs(path_figures, exist_ok=True)
    folder_label = os.path.basename(os.path.normpath(path_figures))
    filename_prefix = f"{folder_label}_" if folder_label.startswith("test_ID_") else ""
    # summarize history for f1 score
    plt.plot(val_macro_f1, label='val macro f1-score')
    plt.plot(macro_f1, label='train macro f1-score')
    plt.title('Training logs')
    plt.ylabel('macro averaged f1-score')
    plt.ylim(bottom=0, top=1)
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(path_figures, f"{filename_prefix}f1_fold{identity}.png"))
    plt.close()


    # summarize history for loss
    plt.figure()
    plt.plot(val_loss, label='val loss')
    plt.plot(loss, label='train loss')
    plt.title('Training logs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(path_figures, f"{filename_prefix}loss_fold{identity}.png"))
    plt.close()

    plt.figure().clear()


#@profile
def optimize_optuna():
    """
    Set up and run an Optuna study for hyperparameter optimization of a deep learning model. This function handles
    the configuration, resumption, and execution of the optimization process, and it saves and visualizes the results.

    :return: An Optuna study.
    :rtype: optuna.study.Study
    """

    # Save study results and trial dataframes
    base_path = os.path.join(config['DL']['path_results'], config['timestamp'], config['DL']['model'],
                             config['experiment_id'])

    os.makedirs(base_path, exist_ok=True)

    plot = False
    study_name = f"{test_id}_optunaStudy_{config['experiment_id']}"
    storage_url = 'sqlite:///' + base_path +'/optunaStudy.db'

    print('OPTUNA STORAGE: ' + str(storage_url))

    print(f"Starting new study or resuming existing: {study_name}")
    sampler = optuna.samplers.TPESampler(seed=666)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    # Determine the number of remaining trials
    completed_and_pruned_trials = [trial for trial in study.trials if
                                   trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]]
    remaining_trials = config['optuna']['n_trials'] - len(completed_and_pruned_trials)
    print(f"Remaining trials: {remaining_trials}")

    if remaining_trials > 0:
        study.optimize(create_objective(config), n_trials=remaining_trials, n_jobs=config['optuna']['n_jobs'], callbacks=[log_progress])
        plot = True

    # Summarize study results
    print(f"Study statistics after optimization: \n" 
          f"    Number of finished trials: {len(study.trials)}\n" 
          f"    Number of pruned trials: {sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}\n" 
          f"    Number of complete trials: {sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)}")

    # Display best trial info
    if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        best_trial = study.best_trial
        print(f"Best trial - Value: {best_trial.value}, Params: {best_trial.params}")
    else:
        print("No successful trials completed.")

    study_path = os.path.join(base_path, f"study_{config['DL']['model']}_{test_id}.gz")

    with gzip.open(study_path, 'wb') as f:
        pickle.dump(study, f)

    path_id = os.path.join(config['DL']['path_results'], config['timestamp'], config['DL']['model'],
                             config['experiment_id'], 'test_ID_' + str(test_id))

    # Save aggregated confusion matrix
    best_cm = np.array(best_trial.user_attrs["confusion_matrix"])
    csv_path = os.path.join(path_id, "best_trial_confusion_matrix.csv")
    np.savetxt(csv_path, best_cm, delimiter=",", fmt="%d")

    # Save training plots
    val_macro_f1 = best_trial.user_attrs["val_macro_f1_score_list"]
    macro_f1 = best_trial.user_attrs["macro_f1_score_list"]
    val_loss = best_trial.user_attrs["val_loss_list"]
    loss = best_trial.user_attrs["loss_list"]
    if plot:
        for split in range(0, len(loss)):
            print('Creating training plot split: ' + str(split))
            create_save_training_plots(path_id, val_macro_f1[split], macro_f1[split], loss[split], val_loss[split], str(split))

        folder_label = os.path.basename(os.path.normpath(path_id))
        filename_prefix = f"{folder_label}_" if folder_label.startswith("test_ID_") else ""

        try:
            fig1 = plot_optimization_history(study)
            fig1.write_image(os.path.join(path_id, f'{filename_prefix}optimization_history.png'), format='png', engine='kaleido')
            del fig1

            # Plot intermediate values
            fig2 = plot_intermediate_values(study)
            fig2.update_layout(
                title="Intermediate Values",
                xaxis_title="Number of Folds",
                yaxis_title="Objective Value",
                legend_title="Legend",
                showlegend=True
            )
            fig2.write_image(os.path.join(path_id, f'{filename_prefix}intermediate_values.png'), format='png', engine='kaleido')
            del fig2

            # Plot parameter importances
            fig3 = plot_param_importances(study)
            fig3.write_image(os.path.join(path_id, f'{filename_prefix}param_importance.png'), format='png', engine='kaleido')
            del fig3
        except ImportError as exc:
            print(f"Skipping Optuna visualization because an optional plotting dependency is missing: {exc}")
        except Exception as exc:
            print(f"Skipping Optuna visualization because plotting failed: {exc}")

    # Cleanup the checkpoints to save storage
    cleanup_checkpoints(study)
    del plot, study_name,
    return study



def write_final_summary(folder, val_f1_outer, avg_histories, best_params_list, ids):
    """
    Write a final summary of the model performance across all cross-validation folds to a text file. This summary includes
    mean and standard deviation of various metrics and the best hyperparameters for each fold.

    :param folder: The folder where the summary file will be saved.
    :type folder: str
    :param val_f1_outer: A list of validation F1 scores from the outer loop of cross-validation.
    :type val_f1_outer: list
    :param avg_histories: A list of average training histories, each containing metrics for a fold.
    :type avg_histories: list
    :param best_params_list: A list of dictionaries, each containing the best parameters for one fold.
    :type best_params_list: list
    :param ids: A list of IDs corresponding to each fold.
    :type ids: list
    :return: None
    :rtype: None
    """

    summary_path = os.path.join(folder, 'final_summary.txt')

    # Calculate the mean and standard deviation for validation accuracy across all folds
    mean_val_auc, std_val_auc = get_mean_std_from_hist_list(avg_histories, 'val_auc')
    mean_val_prec, std_val_prec = get_mean_std_from_hist_list(avg_histories, 'val_precision')
    mean_val_rec, std_val_rec = get_mean_std_from_hist_list(avg_histories, 'val_recall')
    mean_val_f1, std_val_f1 = get_mean_std_from_hist_list(avg_histories, 'val_macro_f1_score')
    mean_val_f1_weighted, std_val_f1_weighted = get_mean_std_from_hist_list(avg_histories, 'val_weighted_f1_score')

    print('FINAL RESULTS: ')
    print('mean_macro_f1: ', mean_val_f1)
    print('mean_std: ', std_val_f1)

    print('mean_weighted_f1: ', mean_val_f1_weighted)
    print('mean_std: ', std_val_f1_weighted)

    with open(summary_path, 'w') as file:
        file.write(f"Final Average Validation F1 Score: {mean_val_f1:.4f}\n")
        file.write(f"Standard Deviation: {std_val_f1:.4f}\n\n")

        file.write(f"Final Average Weighted Validation F1 Score: {mean_val_f1_weighted:.4f}\n")
        file.write(f"Standard Deviation: {std_val_f1_weighted:.4f}\n\n")

        file.write(f"Final Average Validation AUC: {mean_val_auc:.4f}\n")
        file.write(f"Standard Deviation: {std_val_auc:.4f}\n\n")

        file.write(f"Final Average Validation Precision: {mean_val_prec:.4f}\n")
        file.write(f"Standard Deviation: {std_val_prec:.4f}\n\n")

        file.write(f"Final Average Validation Recall: {mean_val_rec:.4f}\n")
        file.write(f"Standard Deviation: {std_val_rec:.4f}\n\n")

        # best_params_list is a list of dictionaries, each containing the best parameters for one fold
        for fold_index, best_params in enumerate(best_params_list):
            file.write(f"Best Parameters for test id {ids[fold_index]}:\n")
            for param, value in best_params.items():
                file.write(f"  {param}: {value}\n")
            file.write("\n")
        for fold_index, best_params in enumerate(best_params_list):
            file.write(f"Performance for test id {ids[fold_index]}:\n")
            file.write(f" Macro f1 score: {val_f1_outer[fold_index]}\n")
            file.write("\n")

    del summary_path, mean_val_auc, std_val_auc, mean_val_prec, std_val_prec, mean_val_rec, std_val_rec, mean_val_f1, std_val_f1, mean_val_f1_weighted, std_val_f1_weighted


#@profile
def summarize_trial(opt_study, folder_ID):
    """
    Summarize the results of an Optuna study, including statistics on completed and pruned trials,
    and details of the best trial. The summary is printed and also saved to a text file.

    :param opt_study: The Optuna study object containing all the trial results.
    :type opt_study: optuna.study.Study
    :param folder_ID: The folder where the summary file will be saved.
    :type folder_ID: str
    :return: None
    :rtype: None
    """

    pruned_trials = [t for t in opt_study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in opt_study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print('Study statistics: ')
    print('     Number of finished trials: ', len(opt_study.trials))
    print('     Number of pruned trials: ', len(pruned_trials))
    print('     Number of complete trials: ', len(complete_trials))

    print('Best trial')
    trial = opt_study.best_trial
    print(trial.value)

    print('         Value: {}'.format(trial.value))
    print('         Params: ')
    for key, value in trial.params.items():
        print('     {}: {}'.format(key, value))

    with open(folder_ID + '/best_trial_summary_' + config['DL']['model'] + '.txt', 'w') as file:

        file.write('Held out test ID: ' + str(test_id) + '\n')
        file.write('Best trial #:' + str(trial.number) + '\n')
        file.write('         Value: {}'.format(trial.value) + '\n')
        file.write('         Params: ' + '\n')
        for key, value in trial.params.items():
            file.write('            {}: {}'.format(key, value) + '\n')
        file.write('\n')
        file.write('Number of finished trials: ' + str(len(opt_study.trials)) + '\n')
        file.write('Number of pruned trials: ' + str(len(pruned_trials)) + '\n')
        file.write('Number of complete trials: ' + str(len(complete_trials)) + '\n')
        file.write('----------------------------------------------------\n')

    del pruned_trials, complete_trials, trial


#
def get_trial_parameters(config, trial):
    """
    Extract the optimal parameters from an Optuna trial based on the model configuration.

    :param config: Configuration dictionary specifying model and Optuna parameters.
    :type config: dict
    :param trial: The trial object containing parameter results.
    :type trial: optuna.trial.FrozenTrial
    :return: A dictionary with the optimal parameter set.
    :rtype: dict
    """

    model_parameters = {}
    model_config = config['optuna'][config['DL']['model']]

    for parameter in model_config:
        if parameter in trial.params:
            model_parameters[parameter] = trial.params[parameter]
        else:
            raise ValueError(f"Parameter {parameter} not found in best trial parameters.")

    del model_config
    return model_parameters


#@profile
def main(config):
    """
    Train the specified deep learning model in a leave-one-subject-out cross-validation and compute the mean
    classification metrics over all folds. Additionally, model parameters are optimized using Optuna.

    :param config: Specifies the parameters of the training, including the DL model, output folder path, and parameter
                   search space for the model optimization using Optuna.
    :type config: dict
    :return: A tuple containing the mean test accuracy and standard deviation of the test accuracy over all LOSO-CV folds.
    :rtype: tuple(float, float)
    """

    global test_id
    global path_results
    test_path = "$TMPDIR"
    print(os.walk(test_path))
    print('§§§§§§§§§§§§§§§§§§§§§§§§')

    val_f1_outer = []
    avg_histories = []
    path_results = config['DL']['path_results']

    folder = os.path.join(path_results, config['timestamp'], config['DL']['model'], config['experiment_id'])
    os.makedirs(folder, exist_ok=True)
    write_parameters_to_log_file(folder, config)

    best_params_list = []
    outer_confusion_matrices = []

    outer_test_ids = config.get("outer_test_IDs", config["valid_IDs"])
    for test_id in outer_test_ids:  # outer CV loop
        folder_ID = os.path.join(folder, 'test_ID_' + str(test_id))
        os.makedirs(folder_ID, exist_ok=True)
        print(f"     Test ID: {test_id}")
        #memory_info = tf.config.experimental.get_memory_info('GPU:0')
        #print('Memory info prior to optimization:', memory_info)
        opt_study = optimize_optuna()

        # Train model on whole train_val set based on best parameter subset
        lr = opt_study.best_trial.params['learning_rate']
        weight_alpha = opt_study.best_trial.params['weight_alpha']
        weight_gamma = opt_study.best_trial.params['weight_gamma']
        batch_size = opt_study.best_trial.params['batch_size']

        model_parameter = get_trial_parameters(config, opt_study.best_trial)

        # load data
        with open(config['df_prep_path'], "rb") as fh:
            df = pickle.load(fh)

        df['lab_num'] = df['lab_num'].astype(int)
        train_idx = df.index[df['ID'] != test_id].tolist()
        test_idx = df.index[df['ID'] == test_id].tolist()

        x_train, y_train, x_test, y_test, train_ids, test_ids = get_training_data(df, train_idx, test_idx, config)
        del df, train_idx, test_idx

        # Balance data if necessary and standardize each input column individually
        x_train, y_train, x_test, y_test = process_data(x_train, y_train, x_test, y_test, train_ids, config)

        del train_ids, test_ids
        print('********TRAINING OUTER LOOP FOR ID ' + str(test_id) + ' STARTED**********')
        (history, duration, cm) = DL_train_CV.run_single_validation_cycle(folder_ID, x_train,
                                                                             x_test, y_train, y_test,
                                                                             np.size(x_train, 2),
                                                                             np.size(x_train, 1),
                                                                             config, 2,
                                                                             np.nan, lr, weight_alpha, weight_gamma,
                                                                             batch_size,
                                                                             opt_study.best_trial,
                                                                             model_parameter)

        del x_train, y_train, x_test, y_test
        K.clear_session()
        gc.collect()

        # Save confusion matrix and training plots
        csv_path = os.path.join(folder_ID, "confusion_matrix_outer.csv")

        # Save the confusion matrix as a CSV file
        np.savetxt(csv_path, cm, delimiter=",", fmt="%d")
        outer_confusion_matrices.append(cm)

        create_save_training_plots(
            folder_ID,
            history.history['val_macro_f1_score'],
            history.history['macro_f1_score'],
            history.history['loss'],
            history.history['val_loss'],
            'outer'
        )

        summarize_trial(opt_study, folder_ID)
        gc.collect()
        best_params_list.append(opt_study.best_trial.params)

        print('Best trial number: ' + str(int(opt_study.best_trial.number)))
        # Analyse and save results
        avg_history = evaluate([float(duration)], [history])
        val_f1_outer.append(avg_history['val_macro_f1_score'])
        avg_histories.append(avg_history)

        del history, avg_history, opt_study, folder_ID, lr, weight_alpha, weight_gamma, batch_size, model_parameter, duration, cm, csv_path
        x_train = y_train = x_test = y_test = history = None
        K.clear_session()
        gc.collect()

    aggregated_outer_cm = np.sum(outer_confusion_matrices, axis=0)
    csv_path_outer = os.path.join(folder, "confusion_matrix_outer_aggregated.csv")
    np.savetxt(csv_path_outer, aggregated_outer_cm, delimiter=",", fmt="%d")
    write_final_summary(folder, val_f1_outer, avg_histories, best_params_list, config['valid_IDs'])


# Check CUDA version
def check_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        return output
    except Exception as e:
        return f"Error checking CUDA version: {e}"


# Check cuDNN version
def check_cudnn_version():
    try:
        cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
        return f"cuDNN version: {cudnn_version}"
    except Exception as e:
        return f"Error checking cuDNN version: {e}"


if __name__ == "__main__":
    set_DL_config()
    config = load_config()
    main(config)
