"""
Feature-Based Random Forest Model Training and Evaluation Script

This script provides functionalities to train and evaluate various machine learning models including
Support Vector Machines (SVM), k-Nearest Neighbors (kNN), Linear Discriminant Analysis (LDA), and
Random Forest (RF) using LOSO nested cross-validation. The script also includes feature selection and balancing
of training data to handle imbalanced datasets.

Modules and Functions:
----------------------
- balance(y_train, features_train):
    Balances the training data by undersampling the majority class.

- select_features(x_train, y_train, number):
    Selects the top `number` of features based on the chi-squared test.

- main(settings, feature_list, f_out):
    Main function to run the machine learning model with cross-validation.

Global Variables:
-----------------
- settings_linearSVM: Dictionary containing settings for linear SVM model.
- settings_nonlinearSVM: Dictionary containing settings for non-linear SVM model.
- settings_kNN: Dictionary containing settings for kNN model.
- settings_LDA: Dictionary containing settings for LDA model.
- settings_RF: Dictionary containing settings for RF model.

Example Usage:
--------------
1. Configure the settings for the desired model (e.g., settings_RF).
2. Prepare the list of features to be used for training.
3. Call the main function with the configured settings, feature list, and output folder.

#    >>> mean, std = main(settings_RF, feature_list=feature_subsets[i], f_out=output_folder + subset_names[i] + '/')

Note: The script expects a specific data structure and input files, which should be placed in the specified data path.

Dependencies:
-------------
- numpy
- pandas
- sklearn
- itertools
- time
- warnings
- os

Author: Maike Laut
Date: 20.06.2024
"""

import numpy as np
import pandas as pd
import itertools
import time
import warnings
import os
import random
import argparse

from sklearn.model_selection import GroupShuffleSplit, LeaveOneOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import feature_selection
from pickle_compat import install_pandas_pickle_compat

# Parameter optimization ranges and settings for classifiers
install_pandas_pickle_compat()
settings_linearSVM = {
    'model': 'SVM',
    'kernel': 'linear',
    'c': [np.power(2, float(x)) for x in np.arange(-10, 11)],
    'n_features': []
}

settings_nonlinearSVM = {
    'model': 'SVM',
    'kernel': 'rbf',
    'c': [np.power(2, float(x)) for x in np.arange(-10, 11)],
    'gamma': [np.power(2, float(x)) for x in np.arange(-10, 11)],
    'n_features': []
}

settings_kNN = {
    'model': 'kNN',
    'n_neighbors': list(range(3, 16)),
    'weights': ['uniform', 'distance'],
    'n_features': []
}

settings_LDA = {
    'model': 'LDA',
    'solver': ['svd', 'lsqr', 'eigen'],
    'n_features': []
}

settings_RF = {
    'model': 'RF',
    'n_estimators': [300],
    'max_features': ['sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 600, num=5)],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True],
    'n_features': []
}



def balance(y_train, features_train):
    """
    Balances the training data by undersampling the majority class.

    This function identifies the minority class in the training labels, calculates the number of samples to be removed from the majority class to balance the dataset, and returns the balanced training labels and features.

    :param y_train: The training labels.
    :type y_train: pandas.Series
    :param features_train: The training features.
    :type features_train: pandas.DataFrame
    :return: A tuple containing the balanced training labels and features.
    :rtype: tuple(pandas.Series, pandas.DataFrame)

    """
    class_counts = y_train.value_counts()
    min_class = class_counts.idxmin()
    n_samples = class_counts.max() - class_counts.min()

    # Sample from the majority class
    drop_indices = y_train[y_train != min_class].sample(n_samples, random_state=15).index
    y_train_balanced = y_train.drop(index=drop_indices)
    features_train_balanced = features_train.drop(index=drop_indices)
    return y_train_balanced, features_train_balanced


def select_features(x_train, y_train, number):
    """
        Selects the top `number` of features based on the chi-squared test.

        :param x_train: The training features.
        :type x_train: numpy.ndarray or pandas.DataFrame
        :param y_train: The training labels.
        :type y_train: numpy.ndarray or pandas.Series
        :param number: The number of top features to select.
        :type number: int
        :return: A tuple containing the transformed training features and the fitted feature selector.
        :rtype: tuple(numpy.ndarray, sklearn.feature_selection.SelectKBest)
        """

    fs = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=number)
    fs.fit(x_train, y_train)
    x_train_fs = fs.transform(x_train)
    return x_train_fs, fs


def main(settings, feature_list, f_out, data_path="data/vr_goalkeeper/dataframes/", test_ids=None):
    """
        Main function to run the machine learning model with LOSO-CV nested cross-validation and parameter optimization.

        :param settings: Settings for the model.
        :type settings: dict
        :param feature_list: List of features to use.
        :type feature_list: list
        :param f_out: Output folder path.
        :type f_out: str
        :return: Mean and standard deviation of the outer accuracy.
        :rtype: tuple(float, float)
        """

    settings['n_features'] = range(1, len(feature_list)+1)
    if len(feature_list) == 1:
        settings['n_features'] = [1]
    warnings.filterwarnings('ignore')

    df = pd.read_pickle(os.path.join(data_path, "features_out.pkl"))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=15)

    # Leave one subject out CV
    # ids = list(df_prep['ID'].drop_duplicates().array) doesn´t preserve order!
    ids = [10, 20, 4, 12, 0, 22, 6, 29, 8, 13, 2, 19, 3, 21, 16, 14, 7, 23, 26, 11, 15, 17, 27, 1, 18, 24, 9]
    if test_ids is not None:
        ids = test_ids

    test_ids, feature_names_out, number_features_out, parameters_out, parameter_names_out = [], [], [], [], []
    precision_out_list, recall_out_list, f1_out_list, roc_auc_out_list, prec_rec_out_list = [], [], [], [], []
    confusion_matrix_out_list, acc_inner_list, acc_std_inner_list, roc_curve_out_list, acc_out_list = [], [], [], [], []


    y_true_out_list, y_pred_out_list, shot_ids_out_list = [], [], []

    for k in range(0, len(ids)):  # groups=df['ID']):  # groups=df_prep['ID']  # df_prep['shot_ID']  # outer CV loopc

        number_features_selected = []
        fs_selected = []
        res_inner = []
        res_std_inner = []

        test_id = ids[k]
        test_ids.append(test_id)
        print('     Test ID: ' + str(test_id))

        df_test = df[df['ID'] == test_id]
        df_train_val = df[df['ID'] != test_id]

        features_train_val = df_train_val[feature_list]
        features_test = df_test[feature_list]
        #print(features_test)

        y_train_val = df_train_val['lab_str']
        y_test = df_test['lab_str']
        y_test_num = np.array(df_test['lab_num'].values, dtype=np.int64)
        #print('Test ID: ' + str(list(dict.fromkeys(df_test.ID.tolist()))))

        # balance trainval data if necessary
        y_train_val, features_train_val = balance(y_train_val, features_train_val)
        scale_train_val = MinMaxScaler().fit(features_train_val)

        features_train_val_scaled = scale_train_val.transform(features_train_val)
        features_test_scaled = scale_train_val.transform(features_test)

        fs_list = []
        number_features = []
        res_param = []
        res_std_param = []

        if settings['model'] == 'SVM' and settings['kernel'] == 'linear':
            c_list = []
            combinations = list(itertools.product(settings['n_features'], settings['c']))
        elif settings['model'] == 'SVM' and settings['kernel'] == 'rbf':
            c_list, gamma_list = [], []
            combinations = list(itertools.product(settings['n_features'], settings['c'], settings['gamma']))
        elif settings['model'] == 'kNN':
            n_neighbors_list, weights_list = [], []
            combinations = list(itertools.product(settings['n_features'], settings['n_neighbors'], settings['weights']))
        elif settings['model'] == 'LDA':
            solver_list = []
            combinations = list(itertools.product(settings['n_features'], settings['solver']))
        elif settings['model'] == 'RF':
            n_est_list, max_feat_list, max_depth_list, min_s_split_list, min_s_leaf_list, bootstrap_list = [], [], [], [], [], []
            combinations = list(itertools.product(settings['n_features'], settings['n_estimators'], settings['max_features'],
                                                  settings['max_depth'], settings['min_samples_split'], settings['min_samples_leaf'], settings['bootstrap']))

        for combination in combinations:
            res_cv_inner = []

            # print('Fitting linear SVM model with c=' + str(c))
            if settings['model'] == 'SVM' and settings['kernel'] == 'linear':
                model = SVC(kernel='linear', C=combination[1], random_state=15)
            elif settings['model'] == 'SVM' and settings['kernel'] == 'rbf':
                model = SVC(kernel='rbf', C=combination[1], gamma=combination[2], random_state=15)
            elif settings['model'] == 'kNN':
                model = KNeighborsClassifier(n_neighbors=combination[1], weights=combination[2])
            elif settings['model'] == 'LDA':
                model = LinearDiscriminantAnalysis(solver=combination[1])
            elif settings['model'] == 'RF':
                model = RandomForestClassifier(n_estimators=combination[1], max_features=combination[2],
                                               max_depth=combination[3],
                                               min_samples_split=combination[4], min_samples_leaf=combination[5],
                                               bootstrap=combination[6], random_state=15, n_jobs=-1)

            gss = GroupShuffleSplit(n_splits=settings.get("inner_splits", 5), train_size=0.75, random_state=15)
            gss.get_n_splits()
            g = gss.split(df_train_val['mean'], df_train_val['lab_num'], groups=df_train_val['ID'])

            for train_inner_idx, test_inner_idx in g:
                df_train = df_train_val.iloc[train_inner_idx]
                df_val = df_train_val.iloc[test_inner_idx]

                features_train = df_train[feature_list]
                features_val = df_val[feature_list]

                y_train = df_train['lab_str']
                y_val = df_val['lab_str']

                # balance training data if necessary
                y_train, features_train = balance(y_train, features_train)
                scale_inner = MinMaxScaler().fit(features_train)
                features_train_scaled = scale_inner.transform(features_train)
                features_val_scaled = scale_inner.transform(features_val)

                if len(feature_list) > 1:
                    features_train_scaled_sel, fs = select_features(features_train_scaled, y_train, combination[0])
                    features_val_scaled_sel = fs.transform(features_val_scaled)
                else:
                    features_train_scaled_sel = features_train_scaled
                    features_val_scaled_sel = features_val_scaled
                    fs = ''

                model.fit(features_train_scaled_sel, y_train)
                y_pred_param = model.predict(features_val_scaled_sel)
                acc_param = accuracy_score(y_val, y_pred_param)
                res_cv_inner.append(acc_param)

            res_param.append(np.mean(res_cv_inner))
            res_std_param.append(np.std(res_cv_inner))
            # Get best combination
            fs_list.append(fs)
            number_features.append(combination[0])
            if settings['model'] == 'SVM':
                c_list.append(combination[1])
                if settings['kernel'] == 'rbf':
                    gamma_list.append(combination[2])
            elif settings['model'] == 'kNN':
                n_neighbors_list.append(combination[1])
                weights_list.append(combination[2])
            elif settings['model'] == 'LDA':
                solver_list.append(combination[1])
            elif settings['model'] == 'RF':
                n_est_list.append(combination[1])
                max_feat_list.append(combination[2])
                max_depth_list.append(combination[3])
                min_s_split_list.append(combination[4])
                min_s_leaf_list.append(combination[5])
                bootstrap_list.append(combination[6])

        best_run_idx = np.argmax(res_param)
        res_inner.append(res_param[best_run_idx])
        res_std_inner.append(res_std_param[best_run_idx])
        fs_selected.append(fs_list[best_run_idx])
        number_features_selected.append(number_features[best_run_idx])

        # train outer model
        if settings['model'] == 'SVM' and settings['kernel'] == 'linear':
            model_opt = SVC(kernel='linear', C=c_list[best_run_idx], random_state=15)
            parameters_out.append(np.array([c_list[best_run_idx]]))
            parameter_names_out.append(['c'])
        elif settings['model'] == 'SVM' and settings['kernel'] == 'rbf':
            model_opt = SVC(kernel='rbf', C=c_list[best_run_idx],gamma=gamma_list[best_run_idx], random_state=15)
            parameters_out.append(np.array([c_list[best_run_idx], gamma_list[best_run_idx]]))
            parameter_names_out.append(['c', 'gamma'])
        elif settings['model'] == 'kNN':
            model_opt = KNeighborsClassifier(n_neighbors=n_neighbors_list[best_run_idx],weights=weights_list[best_run_idx])
            parameters_out.append(np.array([n_neighbors_list[best_run_idx], weights_list[best_run_idx]]))
            parameter_names_out.append(['n_neighbors', 'weights'])
        elif settings['model'] == 'LDA':
            model_opt = LinearDiscriminantAnalysis(solver=solver_list[best_run_idx])
            parameters_out.append(np.array([solver_list[best_run_idx]]))
            parameter_names_out.append(['solver_list'])
        elif settings['model'] == 'RF':
            model_opt = RandomForestClassifier(n_estimators=n_est_list[best_run_idx], max_features=max_feat_list[best_run_idx],
                                               max_depth=max_depth_list[best_run_idx], min_samples_split=min_s_split_list[best_run_idx],
                                               min_samples_leaf=min_s_leaf_list[best_run_idx],bootstrap=bootstrap_list[best_run_idx], random_state=15, n_jobs=-1)
            parameter_names_out.append(['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap'])
            parameters_out.append(np.array([n_est_list[best_run_idx], max_feat_list[best_run_idx], max_depth_list[best_run_idx],
                                            min_s_split_list[best_run_idx], min_s_leaf_list[best_run_idx], bootstrap_list[best_run_idx]]))

        fs = fs_list[best_run_idx]
        if len(feature_list) > 1:
            features_train_val_scaled_sel = fs.transform(features_train_val_scaled)
            features_test_scaled = fs.transform(features_test_scaled)
        else:
            features_train_val_scaled_sel = features_train_val_scaled

        model_opt.fit(features_train_val_scaled_sel, y_train_val)

        y_pred_test = model_opt.predict(features_test_scaled)
        # Evaluate results
        y_pred_test_bool = np.array([1 if x == 'stress' else 0 for x in y_pred_test])
        y_pred_test_num = y_pred_test_bool.astype(int)
        acc_out = accuracy_score(y_test_num, y_pred_test_num)
        f1_out = f1_score(y_test_num, y_pred_test_num)
        recall_out = recall_score(y_test_num, y_pred_test_num)
        precision_out = precision_score(y_test_num, y_pred_test_num)
        roc_auc_out = roc_auc_score(y_test_num, y_pred_test_num)
        precision_recall_out = precision_recall_curve(y_test_num, y_pred_test_num)
        roc_curve_out = roc_curve(y_test_num, y_pred_test_num)
        confusion_matrix_out = confusion_matrix(y_test_num, y_pred_test_num)

        # Get number and names of selected features
        if len(feature_list) > 1:
            indizes = np.array(fs.scores_.argsort())[-number_features[best_run_idx]:][::-1]
            feature_names = np.array(feature_list)[indizes]
        else:
            feature_names = feature_list[0]

        acc_out_list.append(acc_out)
        acc_inner_list.append(res_inner)
        acc_std_inner_list.append(res_std_inner)
        f1_out_list.append(f1_out)
        recall_out_list.append(recall_out)
        roc_auc_out_list.append(roc_auc_out)
        precision_out_list.append(precision_out)
        prec_rec_out_list.append(precision_recall_out)
        roc_curve_out_list.append(roc_curve_out)
        confusion_matrix_out_list.append(confusion_matrix_out)

        y_true_out_list.append(y_test)
        y_pred_out_list.append(y_pred_test)

        feature_names_out.append(feature_names)
        if len(feature_list) > 1:
            number_features_out.append(len(indizes))
        else:
            number_features_out.append(len(feature_list))

        shot_ids_out_list.append(df_test['shot'])

    print('     Total result: ' + str(np.mean(acc_out_list)) + ' +- ' + str(np.std(acc_out_list)))

    dict_cv_results = {'ID': test_ids, 'acc': acc_out_list, 'feature_names': feature_names_out, 'number_features': number_features_out,
                       'parameters': parameters_out, 'parameter_names': parameter_names_out, 'f1': f1_out_list, 'rec': recall_out_list,
                       'auc': roc_auc_out_list, 'prec': precision_out_list, 'acc_inner': acc_inner_list,'acc_std_inner': acc_std_inner_list,
                       'prec_rec': prec_rec_out_list, 'roc': roc_curve_out_list, 'y_true': y_true_out_list, 'y_pred': y_pred_out_list,
                       'conf_matrix': confusion_matrix_out_list, 'shot_id': shot_ids_out_list}

    dict_overall_results = {'mean_acc': np.mean(acc_out_list), 'std_acc': np.std(acc_out_list), 'settings': settings, 'mean_number_features': np.mean(number_features_out),
                            'std_number_features': np.std(number_features_out), 'mean_f1': np.mean(f1_out_list), 'std_f1': np.std(f1_out_list),
                            'mean_rec': np.mean(recall_out_list), 'std_recall': np.std(recall_out_list), 'mean_prec': np.mean(precision_out_list),
                            'std_prec': np.std(precision_out_list), 'mean_auc': np.mean(roc_auc_out_list), 'std_auc': np.std(roc_auc_out_list)}

    df_cv_results = pd.DataFrame(dict_cv_results)
    df_overall_results = pd.DataFrame(dict_overall_results)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f_out + settings['model'] + '/'):
        os.makedirs(f_out + settings['model'] + '/')
    df_cv_results.to_pickle(f_out + settings['model'] + '/df_cv_results_' + settings['model'] + '_' + timestr + '.pkl')
    df_cv_results.to_csv(f_out + settings['model'] + '/df_cv_results_' + settings['model'] + '_' + timestr + '.csv')

    df_overall_results.to_pickle(f_out + settings['model'] + '/df_overall_results_' + settings['model'] + '_' + timestr + '.pkl')
    df_overall_results.to_csv(f_out + settings['model'] + '/df_overall_results_' + settings['model'] + '_' + timestr + '.csv')

    np.save(f_out + settings['model'] + "/feature_list_" + settings['model'] + "_" + timestr + ".npy", np.array(feature_list))

    return np.mean(acc_out_list), np.std(acc_out_list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the feature-based random-forest baseline for the VR goalkeeper dataset."
    )
    parser.add_argument(
        "--data-path",
        default="data/vr_goalkeeper/dataframes/",
        help="Directory containing features_out.pkl."
    )
    parser.add_argument(
        "--output-dir",
        default="results/vr_goalkeeper/rf_baseline_vr_goalkeeper/",
        help="Directory where RF baseline outputs are written."
    )
    parser.add_argument(
        "--subset",
        choices=["fix", "pd", "combined", "all"],
        default="all",
        help="Feature subset to train. Use 'all' for the full baseline comparison."
    )
    parser.add_argument(
        "--test-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of held-out participant IDs for a short local test."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a very small hyperparameter grid and two held-out participants."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(15)
    random.seed(15)
    output_folder = args.output_dir

    feature_sets = {
        "fix": ['meanfixationDuration_asymptotic_model', 'fixation_durations_asymptotic_model',
                'counter_asymptotic_model'],
        "pd": ['mean', 'median', 'variance', 'std', 'skew', 'max_val', 'min_val', 'kurt', 'range', '1st_quantile',
               '3rd_quantile', 'harmonic_mean', 'samples_till_max', 'slope1', 'slope2'],
        "combined": ['mean', 'median', 'variance', 'std', 'skew', 'max_val', 'min_val', 'kurt', 'range', '1st_quantile',
                     '3rd_quantile', 'harmonic_mean', 'samples_till_max', 'slope1', 'slope2',
                     'meanfixationDuration_asymptotic_model', 'fixation_durations_asymptotic_model',
                     'counter_asymptotic_model'],
    }
    selected_subset_names = list(feature_sets) if args.subset == "all" else [args.subset]
    test_ids = args.test_ids
    settings = dict(settings_RF)

    if args.smoke_test:
        test_ids = test_ids or [0, 1]
        settings.update({
            "n_estimators": [20],
            "max_depth": [10],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "inner_splits": 2,
        })

    # If multiple experiments should be conducted, iterate over feature subsets
    for subset_name in selected_subset_names:
        mean, std = main(
            settings,
            feature_list=feature_sets[subset_name],
            f_out=os.path.join(output_folder, subset_name, ""),
            data_path=args.data_path,
            test_ids=test_ids,
        )
        print(f"{subset_name}: {mean:.4f} +- {std:.4f}")
