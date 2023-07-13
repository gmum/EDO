import os
import os.path as osp
import numpy as np
import pandas as pd

from copy import deepcopy

from .. import make_origin, Task, TASK_ERROR_MSG
from .._check import _check_unlogging
from ..utils import find_and_load, get_all_subfolders, get_all_files, usv
from ..config import parse_shap_config, UTILS

from .shapcalculator import SHAPCalculator


# # # # # # # # # # # #
# L O A D   S T U F F #
# # # # # # # # # # # #


# ISSUE 131 TODO
def find_experiment(results, stage, origin):
    # What's the path for the experiments I'm looking for?
    origin = make_origin(origin)
    stage = stage.lower()

    path = osp.join(results, origin.split, stage)
    exp_dir = f'{origin.dataset}-{origin.representation}-{origin.task}-{origin.model}'

    candidates = [c for c in get_all_subfolders(path, extend=False) if exp_dir in c]
    exp_path = osp.join(path, usv(candidates))
    return exp_path


def load_train_test(mldir):
    # returns (x, y, smiles) for train and test
    x_train = find_and_load(mldir, '-x.pickle', protocol='pickle')
    x_test = find_and_load(mldir, '-test_x.pickle', protocol='pickle')

    y_train = find_and_load(mldir, '-y.pickle', protocol='pickle')
    y_test = find_and_load(mldir, '-test_y.pickle', protocol='pickle')

    smiles_train = find_and_load(mldir, '-smiles.pickle', protocol='pickle')
    smiles_test = find_and_load(mldir, '-test_smiles.pickle', protocol='pickle')

    return (x_train, y_train, smiles_train), (x_test, y_test, smiles_test),


def load_predictions(mldir, task):
    # zero, one, two w preds to indeksy klas a nie ich tożsamość (unstable, medium, stable)
    # ( w tym miejscu nie jesteśmy w stanie sprawdzić w jakiej kolejności model zwracał klasy)
    def _load_and_parse(files):
        preds = pd.concat([pd.read_csv(f, sep='\t') for f in files])
        preds = preds.drop_duplicates()
        preds = preds.set_index('smiles')

        if task == Task.CLASSIFICATION:
            mlcp = preds.class_probabilities.str.extract(
                r'(?P<zero>[e0-9.+-]*)\s+(?P<one>[e0-9.+-]*)\s+(?P<two>[e0-9.+-]*)')
            preds = preds.join(mlcp)
            preds = preds.astype({'zero': float, 'one': float, 'two': float})
        return preds

    all_preds = [f for f in get_all_files(mldir, extend=False) if 'predictions' in f]
    train_df = _load_and_parse(osp.join(mldir, f) for f in all_preds if 'train' in f)
    test_df = _load_and_parse(osp.join(mldir, f) for f in all_preds if 'test' in f)
    return train_df, test_df


# ISSUE 131 TODO
def load_shap_files(shapdir, task, check_unlogging):
    # smaller version of src.shap_analysis.utils load_shap_files with more checks
    shap_cfg = parse_shap_config(usv([osp.join(shapdir, f) for f in os.listdir(shapdir) if 'shap' in f and 'cfg' in f]))
    unlog = shap_cfg[UTILS]["unlog"]
    if check_unlogging:
        _check_unlogging(unlog, task)

    smiles_order = find_and_load(shapdir, 'smiles.npy', protocol='numpy')  # changed from canonised.npy
    X_full = find_and_load(shapdir, 'X_full.npy', protocol='numpy')
    true_ys = find_and_load(shapdir, 'true_ys.npy', protocol='numpy')
    shap_values = find_and_load(shapdir, 'SHAP_values.npy', protocol='numpy')

    if task == Task.CLASSIFICATION:
        classes_order = find_and_load(shapdir, 'classes_order.npy', protocol='numpy')
    elif task == Task.REGRESSION:
        classes_order = None
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return smiles_order, X_full, true_ys, classes_order, shap_values


def load_model(results_dir, origin, check_unlogging=True):
    mldir = find_experiment(results_dir, 'ml', origin)
    shapdir = find_experiment(results_dir, 'shap', origin)

    # load model
    model_fname = usv([pkl for pkl in os.listdir(mldir) if 'model.pickle' in pkl])
    evaluation_model = find_and_load(mldir, model_fname, protocol='pickle')  # evaluate optimisation procedure
    shap_model = SHAPCalculator(shapdir, mldir, origin, check_unlogging)     # calculate SHAP values

    return evaluation_model, shap_model


# # # # # # # # #
# P R E D I C T #
# # # # # # # # #


def _get_pred_single_sample(f_vals, model, task):
    # f_vals dla jednego konkretnego związku
    if task == Task.CLASSIFICATION:
        return model.predict_proba(f_vals.reshape(1, -1))
    elif task == Task.REGRESSION:
        return model.predict(f_vals.reshape(1, -1))
    else:
        raise ValueError(TASK_ERROR_MSG(task))


def get_predictions_before_after_slow(samples, model, task):
    before, after = [], []
    for sample in samples:
        before.append(_get_pred_single_sample(sample.original_f_vals, model, task))
        after.append(_get_pred_single_sample(sample.f_vals, model, task))
    return np.array(before), np.array(after)


def _get_pred(f_vals, model, task):
    # f_vals - array
    if task == Task.CLASSIFICATION:
        return model.predict_proba(f_vals)
    elif task == Task.REGRESSION:
        return model.predict(f_vals)
    else:
        raise ValueError(TASK_ERROR_MSG(task))


def get_predictions_before_after(samples, model, task):
    if len(samples)<=1:
        return get_predictions_before_after_slow(samples, model, task)

    # optimised get_predictions_before_after_slow
    before = np.array([sample.original_f_vals for sample in samples])
    after = np.array([sample.f_vals for sample in samples])

    return _get_pred(before, model, task), _get_pred(after, model, task)


# # # # # # # # # # # # # # #
# P R E P R O C E S S I N G #
# # # # # # # # # # # # # # #
# poniższe funkcje są zależne od tego jak sobie wczytaliśmy dane

def filter_correct_predictions_only(df, task):
    # return list of SMILES of correct predictions
    if task == Task.CLASSIFICATION:
        # classification SVMs might give different answers for `predict` and `predict_proba`
        df = deepcopy(df[df.true == df.predicted])
        # TODO uwaga! ten kawałek zakłada, że classes_order = [0, 1, 2] (CHYBA)
        df['pred_probs'] = df.apply(lambda row: np.argmax([row.zero, row.one, row.two]), axis=1)
        return sorted(df[df.true == df.pred_probs].index.tolist())
    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))


def group_samples(df, task):
    # return lists of SMILES of unstable, medium, stable

    if task == Task.CLASSIFICATION:
        unstable = sorted(df[df.true == 0].index.tolist())
        medium = sorted(df[df.true == 1].index.tolist())
        stable = sorted(df[df.true == 2].index.tolist())
    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))
    return unstable, medium, stable


def intersection_list(*argv):
    # intersection between lists
    sets = [set(arg) for arg in argv]
    intersection = sets[0].intersection(*sets)
    return sorted(list(intersection))


def difference_list(*argv):
    # list1 - list2 ... - listN
    sets = [set(arg) for arg in argv]
    difference = sets[0].difference(*sets[1:])
    return sorted(list(difference))


def get_present_features(x_train, threshold):
    """
    Return indices of features that are present and absent in at least (threshold * n_samples) samples in the training set
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
