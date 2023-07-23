import os.path as osp
import numpy as np
import pandas as pd

from copy import deepcopy

from .. import make_origin, Task, TASK_ERROR_MSG
from .._check import _check_unlogging
from ..utils import find_and_load, get_all_subfolders, get_all_files, usv
from ..config import parse_shap_config, UTILS

from .shapcalculator import SHAPCalculator

"""
- utils to load data
- utils for preprocessing
- utils to make predictions
"""


# # # # # # # # # # # #
# L O A D   U T I L S #
# # # # # # # # # # # #


def find_experiment(results_dir, stage, origin):
    """
    Determine path to the experiment of interest.
    :param results_dir: str: path to root directory with results
    :param stage: str: `ml` or `shap` - looking for results of model training or explaining the predictions?
    :param origin: Origin: identification of the experiment
    :return: str: path
    """
    origin = make_origin(origin)
    stage = stage.lower()

    path = osp.join(results_dir, origin.split, stage)
    exp_dir = f'{origin.dataset}-{origin.representation}-{origin.task}-{origin.model}'

    candidates = [c for c in get_all_subfolders(path, extend=False) if exp_dir in c]
    exp_path = osp.join(path, usv(candidates))
    return exp_path


def load_train_test(ml_dir):
    """
    Load train and test data from ml_dir.
    :param ml_dir: str: path to directory with experiment results
    :return: List[Tuple[np.array[samples x features], np.array[samples], np.array[samples]]]:
             (x, y, smiles) for train and test
    """
    x_train = find_and_load(ml_dir, '-x.pickle', protocol='pickle')
    x_test = find_and_load(ml_dir, '-test_x.pickle', protocol='pickle')

    y_train = find_and_load(ml_dir, '-y.pickle', protocol='pickle')
    y_test = find_and_load(ml_dir, '-test_y.pickle', protocol='pickle')

    smiles_train = find_and_load(ml_dir, '-smiles.pickle', protocol='pickle')
    smiles_test = find_and_load(ml_dir, '-test_smiles.pickle', protocol='pickle')

    return (x_train, y_train, smiles_train), (x_test, y_test, smiles_test),


def load_predictions(ml_dir, task):
    """
    Load train and test predictions from ml_dir.
    :param ml_dir: str: path to directory with experiment results
    :param task: Task: are these results for classification or regression?
    :return: (pandas.DataFrame, pandas.DataFrame): train data, test data
    """

    def _load_and_parse(files):
        preds = pd.concat([pd.read_csv(f, sep='\t') for f in files])
        preds = preds.drop_duplicates()
        preds = preds.set_index('smiles')

        if task == Task.CLASSIFICATION:
            mlcp = preds.class_probabilities.str.extract(
                r'(?P<zero>[e0-9.+-]*)\s+(?P<one>[e0-9.+-]*)\s+(?P<two>[e0-9.+-]*)')
            preds = preds.join(mlcp)
            # Note `zero`, `one`, and `two`are class indices and not their categories (unstable, medium, stable)
            preds = preds.astype({'zero': float, 'one': float, 'two': float})
        return preds

    all_preds = [f for f in get_all_files(ml_dir, extend=False) if 'predictions' in f]
    train_df = _load_and_parse(osp.join(ml_dir, f) for f in all_preds if 'train' in f)
    test_df = _load_and_parse(osp.join(ml_dir, f) for f in all_preds if 'test' in f)
    return train_df, test_df


def load_shap_files(shap_dir, task, check_unlogging):
    """
    Load data and SHAP values from shap_dir.
    :param shap_dir: path to directory with experiment results
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :param check_unlogging: boolean: if True will ensure that regressors are unlogged and classifiers are not
    :return: (np.array[samples], np.array[samples x features], np.array[samples], np.array[classes],
             np.array[(classes x) samples x features]): SMILES, concatenated train and test data, true labels,
             classes order from the model used to calculate SHAP values or None in the case of regressors, SHAP values
    """
    shap_cfg = parse_shap_config(
        usv([osp.join(shap_dir, f) for f in get_all_files(shap_dir, extend=False) if 'shap' in f and 'cfg' in f]))
    unlog = shap_cfg[UTILS]["unlog"]
    if check_unlogging:
        _check_unlogging(unlog, task)

    smiles_order = find_and_load(shap_dir, 'smiles.npy', protocol='numpy')
    X_full = find_and_load(shap_dir, 'X_full.npy', protocol='numpy')
    true_ys = find_and_load(shap_dir, 'true_ys.npy', protocol='numpy')
    shap_values = find_and_load(shap_dir, 'SHAP_values.npy', protocol='numpy')

    if task == Task.CLASSIFICATION:
        classes_order = find_and_load(shap_dir, 'classes_order.npy', protocol='numpy')
    elif task == Task.REGRESSION:
        classes_order = None
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return smiles_order, X_full, true_ys, classes_order, shap_values


def load_model(results_dir, origin, check_unlogging=True):
    """
    Load a model and return it unchanged (for evaluation of the optimisation procedure) and as a SHAPCalculator
    :param results_dir: str: path to root directory with results
    :param origin: Origin: identification of the experiment
    :param check_unlogging: boolean: if True will ensure that regressors are unlogged and classifiers are not;
                                     default: True
    :return: (sklearn model, SHAPCalculator): model for evaluation of the optimisation procedure,
                                              model for calculation of SHAP values
    """
    ml_dir = find_experiment(results_dir, 'ml', origin)
    shap_dir = find_experiment(results_dir, 'shap', origin)

    # load model for...
    model_fname = usv([pkl for pkl in get_all_files(ml_dir, extend=False) if 'model.pickle' in pkl])
    evaluation_model = find_and_load(ml_dir, model_fname, protocol='pickle')  # ... evaluation of optimisation procedure
    shap_model = SHAPCalculator(shap_dir, ml_dir, origin, check_unlogging)  # ... calculation of SHAP values

    return evaluation_model, shap_model


# # # # # # # # # # # # # # # # # # # # #
# P R E P R O C E S S I N G   U T I L S #
# # # # # # # # # # # # # # # # # # # # #


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


def get_correct_predictions(df, task):
    """
    Find samples for which prediction is correct and return their indices (row labels).

    NOTE: in the case of classifiers, we assume that there are three classes and that the predicted probabilities are
    given in columns named `zero`, `one` and `two`.

    NOTE: classification SVMs might give different answers for `predict` and `numpy.argmax(predict_proba)` because of
    how `predict_proba` is implemented in sklearn. In this implementation, a prediction is correct only if both
    `predict` and `numpy.argmax(predict_proba)` return the correct prediction.

    :param df: pandas.DataFrame: predictions in a DataFrame that contains columns `true` (ground-truth) and `predicted`
    (prediction, i.e. the result of calling function `predict` on a sklearn model)
    :param task: Task: is the model used to calculate predictions a classifier or a regressor
    :return: List[Hashable]: a sorted list with indices of samples for which the prediction is correct
    """
    df_correct = deepcopy(df[df.true == df.predicted])

    if task == Task.CLASSIFICATION:
        df_correct['pred_proba'] = df_correct.apply(lambda row: np.argmax([row.zero, row.one, row.two]), axis=1)
        df_correct = df_correct[df_correct.true == df_correct.pred_proba]
    elif task == Task.REGRESSION:
        raise NotImplementedError
        pass  # doing nothing should be OK but I've never tested this
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return sorted(df_correct.index.tolist())


def group_samples(df, task, n_groups=None):
    """
    Return indices of samples in each group (class).
    :param df: pandas.DataFrame: predictions in a DataFrame that contains a column `true` with ground-truth labels
    :param task: Task: is the model used to calculate predictions a classifier or a regressor
    :param n_groups: int or None: the number of groups, if None it will be determined; default: None
    :return: (List[Hashable], ...): a tuple with a sorted list of indices for each group
    """
    if task == Task.CLASSIFICATION:
        n_groups = 1 + df.true.max() if n_groups is None else n_groups
        groups = (sorted(df[df.true == i].index.tolist()) for i in range(n_groups))
    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    n_samples = sum([len(g) for g in groups])
    assert df.shape[0] == n_samples, f"{df.shape[0]} samples grouped into {n_samples} samples (n_groups={n_groups})."

    return groups


def get_present_features(x_train, threshold):
    """
    Return indices of features that are both present and absent in at least `threshold * n_samples` samples in `x_train`
    :param x_train: numpy.array[samples x features]: train set
    :param threshold: int: a minimal number of times a feature must be present and absent
    :return: List[int]: indices of features that are present and absent in at least (threshold * n_samples) samples
    """
    n_samples = x_train.shape[0]
    n_occurrences = np.sum(x_train != 0, axis=0)
    assert n_occurrences.shape[0] == x_train.shape[1]

    # threshold must be met from both ends
    sufficient = n_occurrences / n_samples >= threshold
    not_too_many = n_occurrences / n_samples <= (1 - threshold)
    satisfied = np.logical_and(sufficient, not_too_many)  # an array of bools
    return sorted(list(set(np.array(range(len(satisfied)))[satisfied])))


# # # # # # # # # # # # # # # # # #
# P R E D I C T I O N   U T I L S #
# # # # # # # # # # # # # # # # # #

def get_predictions_before_after(samples, model, task):
    """
    For each sample, calculate two predictions: using original feature values (before optimisation) and final feature
    values (after optimisation).
    :param samples: Iterable[Sample]: samples for which predictions should be calculated
    :param model: sklearn-like model to calculate predictions
    :param task: Task: call `model.predict` (regression) or `model.predict_proba` (classification)
    :return: (numpy.array[samples (x classes)]: before, numpy.array[samples (x classes)]: after): predictions before
    and after optimisation for each sample
    """
    before = np.array([sample.original_f_vals for sample in samples])
    after = np.array([sample.f_vals for sample in samples])

    return _get_pred(before, model, task), _get_pred(after, model, task)


def _get_pred(f_vals, model, task):
    """
    Use `model` to calculate predictions for `f_vals`.
    :param f_vals: np.array[samples x features]: feature values
    :param model: sklearn-like model to calculate predictions
    :param task: Task: call `model.predict` (regression) or `model.predict_proba` (classification)
    :return: numpy.array[samples (x classes)]: predictions for each sample
    """
    if task == Task.CLASSIFICATION:
        return model.predict_proba(f_vals)
    elif task == Task.REGRESSION:
        return model.predict(f_vals)
    else:
        raise ValueError(TASK_ERROR_MSG(task))
