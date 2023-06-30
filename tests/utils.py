import os
import os.path as osp
from copy import deepcopy
from enum import Enum

import pandas as pd

from edo import Task, TASK_ERROR_MSG
from edo.utils import find_and_load, index_of_smiles
from edo.config import UTILS, CSV, parse_shap_config
from edo.data import unlog_stability, load_and_preprocess


# # # # # # # # # # # # # # # #
# P A R A M S   S E C T I O N #
# (1-e)y <= y' <= (1+e)y  or  (y >= enough and y' >= enough)
e = 0.2       # accepted difference in prediction
enough = 7.5  # if more stable than just stable
tanimoto_threshold = 0.3
# # # # # # # # # # # # # # # #


def load_ml_files(directory):
    x_train = find_and_load(directory, '-x.pickle', protocol='pickle')
    x_test = find_and_load(directory, '-test_x.pickle', protocol='pickle')

    smiles_train = find_and_load(directory, '-smiles.pickle', protocol='pickle')
    smiles_test = find_and_load(directory, '-test_smiles.pickle', protocol='pickle')

    return x_train, x_test, smiles_train, smiles_test


def load_shap_files(directory, task, check_unlogging=True):
    shap_cfg = parse_shap_config \
        ([osp.join(directory, f) for f in os.listdir(directory) if 'shap' in f and 'cfg' in f][0])
    if check_unlogging and task == Task.REGRESSION:
        assert shap_cfg[UTILS]["unlog"], f"{directory} contains SHAP values for an estimator that was not unlogged!"

    smiles_order = find_and_load(directory, 'canonised.npy', protocol='numpy')
    X_full = find_and_load(directory, 'X_full.npy', protocol='numpy')
    morgan_repr = find_and_load(directory, "morgans.npy", protocol='numpy')
    true_ys = find_and_load(directory, 'true_ys.npy', protocol='numpy')
    preds = find_and_load(directory, 'predictions', protocol='numpy')
    expected_values = find_and_load(directory, 'expected_values.npy', protocol='numpy')
    shap_values = find_and_load(directory, 'SHAP_values.npy', protocol='numpy')
    background_data = find_and_load(directory, 'background_data.pickle', protocol='pickle')

    if task == Task.CLASSIFICATION:
        classes_order = find_and_load(directory, 'classes_order.npy', protocol='numpy')
    elif task == Task.REGRESSION:
        classes_order = None
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return shap_cfg, smiles_order, X_full, morgan_repr, true_ys, preds, classes_order, expected_values, shap_values, background_data


def get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order):
    d = {}
    columns = ['true', ]

    if task == Task.CLASSIFICATION:
        columns.extend(Stability(i).name for i in classes_order)
        for i in range(len(smiles_order)):
            d[smiles_order[i]] = true_ys[i], *preds[i]
    elif task == Task.REGRESSION:
        columns.append('predicted')
        for i in range(len(smiles_order)):
            d[smiles_order[i]] = true_ys[i], preds[i]
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    smiles_true_predicted_df = pd.DataFrame.from_dict(d, orient='index', columns=columns)
    return smiles_true_predicted_df


def get_smiles_correct(smiles_true_predicted_df, task, task_cfg, data_cfg, classes_order):
    # 1 b) zostawienie wyłącznie poprawnych
    correct = deepcopy(smiles_true_predicted_df)

    if data_cfg[CSV]['scale'] is None:
            log_scale = False
    elif 'log' == data_cfg[CSV]['scale']:
        log_scale = True
    else:
        raise NotImplementedError(f"scale {data_cfg[CSV]['scale']} is not implemented.")

    if task == Task.CLASSIFICATION:
        class_cols = [Stability(i).name for i in classes_order]

        correct['true_class'] = correct.apply(lambda row: task_cfg[UTILS]['cutoffs'](float(row.true), log_scale=log_scale), axis=1)
        correct['predicted_class'] = correct.apply(lambda row: pd.to_numeric(row.loc[class_cols]).nlargest(1).index[0], axis=1)
        correct = correct[correct.predicted_class == correct.true_class.apply(lambda c: Stability(c).name)]

    elif task == Task.REGRESSION:
        if log_scale:
            correct['true_h'] = correct.apply(lambda row: unlog_stability(pd.to_numeric(row.true)), axis=1)
            correct['predicted_h'] = correct.apply(lambda row: unlog_stability(pd.to_numeric(row.predicted)), axis=1)
        else:
            correct['true_h'] = correct.true
            correct['predicted_h'] = correct.predicted

        correct['within_limits'] = correct.apply(lambda row: (1-e)*row.true_h <= row.predicted_h <= (1+e)*row.true_h, axis=1)
        correct['within_limits'] = correct.apply(lambda row: row.within_limits or (row.true_h >= enough and row.predicted_h >= enough), axis=1)
        correct = correct[correct.within_limits == True]
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return set(correct.index)


def get_smiles_stability_value(smiles_true_predicted_df, data_cfg, task_cfg):
    # returns three sets of smiles depending on the molecule's true class
    stability = deepcopy(smiles_true_predicted_df)

    if data_cfg[CSV]['scale'] is None:
        log_scale = False
    elif 'log' == data_cfg[CSV]['scale']:
        log_scale = True
    else:
        raise NotImplementedError(f"scale {data_cfg[CSV]['scale']} is not implemented.")

    stability['true_class'] = stability.apply(
        lambda row: task_cfg[UTILS]['cutoffs'](float(row.true), log_scale=log_scale), axis=1)

    low = stability[stability.true_class == 0]
    med = stability[stability.true_class == 1]
    high = stability[stability.true_class == 2]

    return set(low.index), set(med.index), set(high.index)


def filter_samples(mol_filter, feature_filter, X_full, shap_values, task, smiles_order):
    # ordering is important
    if isinstance(mol_filter, set):
        mol_filter = list(mol_filter)
    if isinstance(feature_filter, set):
        feature_filter = list(feature_filter)

    mol_indices = index_of_smiles(smiles_order, mol_filter)

    filtered_X = X_full[mol_indices][:, feature_filter]
    filtered_df = pd.DataFrame(filtered_X, columns=feature_filter, index=mol_filter)

    if task == Task.CLASSIFICATION:
        filtered_shaps = shap_values[:, mol_indices][:, :, feature_filter]
    elif task == Task.REGRESSION:
        filtered_shaps = shap_values[mol_indices][:, feature_filter]
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return filtered_X, filtered_df, filtered_shaps, mol_filter, mol_indices, feature_filter


class Stability(Enum):
    UNSTABLE = 0
    MEDIUM = 1
    STABLE = 2


class Relation(Enum):
    # TODO: should be renamed, we have relation in src.optimisation
    MOST = 'most'
    LEAST = 'least'
