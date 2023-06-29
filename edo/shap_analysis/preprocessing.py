import numpy as np
import pandas as pd

from copy import deepcopy

from .. import Task, TASK_ERROR_MSG
from ..config import UTILS, CSV
from ..data import unlog_stability
from ..utils import get_configs_and_model#, find_and_load

from . import Category
from .utils import load_shap_files, load_ml_files, index_of_smiles


# # # # # # # # # # # # # # # #
# P A R A M S   S E C T I O N #
# (1-e)y <= y' <= (1+e)y  or  (y >= enough and y' >= enough)
e = 0.2       # accepted difference in prediction
enough = 7.5  # if more stable than just stable
tanimoto_threshold = 0.3
# # # # # # # # # # # # # # # #




def get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order):
    d = {}
    columns = ['true', ]
    
    if task == Task.CLASSIFICATION:
        columns.extend(Category(i).name for i in  classes_order)
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
        class_cols = [Category(i).name for i in classes_order]
               
        correct['true_class'] = correct.apply(lambda row: task_cfg[UTILS]['cutoffs'](float(row.true), log_scale=log_scale), axis=1)
        correct['predicted_class'] = correct.apply(lambda row: pd.to_numeric(row.loc[class_cols]).nlargest(1).index[0], axis=1)
        correct = correct[correct.predicted_class == correct.true_class.apply(lambda c: Category(c).name)]
    
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


def get_present_features(x_train, threshold):
    """
    this returns indices of features that are present and absent
    in at least (treshold * n_samples) molecules in the training set
    """
    
    n_samples = x_train.shape[0]
    summed = np.sum(x_train != 0, axis=0)
    assert summed.shape[0] == x_train.shape[1]
    
    # threshold must be met from both ends
    sufficient = summed/n_samples >= threshold  # sufficient is array of bools
    not_too_many = summed/n_samples <= (1 - threshold)
    satisfied = np.logical_and(sufficient, not_too_many)
    
    # todo: może dopisać też po nazwach?
    
    return sorted(list(set(np.array(range(len(satisfied)))[satisfied])))


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
