import os
import os.path as osp
import types
import collections
import json
import pickle
import numpy as np
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.metrics import precision_score, recall_score, confusion_matrix, make_scorer
from .config import parse_data_config, parse_representation_config, parse_task_config, parse_model_config


def force_classification(model, cutoffs, **kwargs):
    """
    Augments a regressor model to perform classification.
    :param model: sklearn-like regressor
    :param cuttoffs: cuttoffs for changing regression to classification
    :param kwargs: params for cuttoffs function
    :return: model
    """

    model.old_predict = model.predict
    model.cutoffs = cutoffs
    model.cutoffs_kwargs = kwargs

    def new_predict(self, X):
        y = self.old_predict(X)
        y = self.cutoffs(y, self.cutoffs_kwargs)
        return y

    model.predict = types.MethodType(new_predict, model)
    return model


from sklearn.metrics._scorer import _BaseScorer
class NanSafeScorer(_BaseScorer):
    def __init__(self, scorer):
        assert isinstance(scorer, _BaseScorer)
        super().__init__(scorer._score_func, scorer._sign, scorer._kwargs)
        self._scorer = scorer
                                           
        if any([fname in self._score_func.__name__ for fname in ['roc_auc']]):
            self._worst = 0
        elif 'error' in self._score_func.__name__:
            self._worst = np.inf if self._sign > 0 else -np.inf
        else:
            raise NotImplementedError(f'Unimplemented _score_func {self._score_func.__name__}.')


    def __call__(self, estimator, X, y_true, sample_weight=None):
        try:
            return self._scorer(estimator, X, y_true, sample_weight)
        except ValueError:
            return self._worst


def get_scorer(scoring):
    # extension of sklearn.metrics.get_scorer
    # to use sklearn's precision and recall with average=None, and confusion_matrix
    # mind that we cheat a little so don't use these additional scorers in grid search or sth
    if 'precision_none' == scoring:
        return make_scorer(precision_score, greater_is_better=True, needs_proba=False, needs_threshold=False, average=None)
    elif 'recall_none' == scoring:
        return make_scorer(recall_score, greater_is_better=True, needs_proba=False, needs_threshold=False, average=None)
    elif 'confusion_matrix' == scoring:
        return make_scorer(confusion_matrix, greater_is_better=True, needs_proba=False, needs_threshold=False)
    else:
        return sklearn_get_scorer(scoring)


def debugger_decorator(func):
    def wrapper(*args, **kwargs):
        print(f'\nCalling {func} with params:')
        for a in args:
            print(a)
        for k in kwargs:
            print(f'{k}: {kwargs[k]}')
        returned_values = func(*args, **kwargs)
        print(f'\nReturned values are: {returned_values}\n')
        return returned_values

    return wrapper


def usv(it):
    """Unpack single value"""
    assert isinstance(it, collections.Iterable)
    if len(it) == 1:
        return it[0]
    elif len(it) == 0:
        return None
    else:
        raise ValueError(f'len(it)={len(it)}')


def get_all_subfolders(path, extend=False):
    subfolders = [folder for folder in os.listdir(path) if osp.isdir(osp.join(path, folder))]
    if extend:
        subfolders = [osp.join(path, f) for f in subfolders]
    return subfolders


def get_all_files(path, extend=False):
    files = [folder for folder in os.listdir(path) if osp.isfile(osp.join(path, folder))]
    if extend:
        files = [osp.join(path, f) for f in files]
    return files


def get_configs_and_model(folder_path):
    """Go through folder with results and retrieve configs and pickled model"""
    configs = [osp.join(folder_path, cfg) for cfg in os.listdir(folder_path) if cfg.endswith('.cfg')]
    data_cfg = parse_data_config(usv([dc for dc in configs if 'rat' in dc or 'human' in dc]))
    repr_cfg = parse_representation_config(usv([rc for rc in configs if rc.endswith(('maccs.cfg', 'padel.cfg', 'fp.cfg')) or 'morgan' in rc]))
    task_cfg = parse_task_config(usv([tc for tc in configs if 'regression' in tc or 'classification' in tc]))
    try:
        model_cfg = parse_model_config(usv([mc for mc in configs if mc.endswith(('nb.cfg', 'svm.cfg', 'trees.cfg'))]))
    except TypeError as err:
        # knn has no config file
        if 'knn' in folder_path:
            model_cfg = None
        else:
            raise err
        
    model_pickle = usv([osp.join(folder_path, pkl) for pkl in os.listdir(folder_path) if 'model.pickle' in pkl])

    return data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle


def find_and_load(directory, pattern, protocol='numpy'):
    """Scan the directory to find a filename matching the pattern and load it using numpy, pickle or json protocol."""
    fname = usv([osp.join(directory, f) for f in os.listdir(directory) if pattern in f])
    if protocol == 'numpy':
        arr = np.load(fname, allow_pickle=False)
    elif protocol == 'pickle':
        with open(fname, 'rb') as f:
            arr = pickle.load(f)
    elif protocol == 'json':
        with open(fname, 'r') as f:
            arr = json.load(f)
    else:
        raise NotImplementedError(f"Protocol must be `numpy`, `pickle` or `json`. Is {protocol}.")
    return arr
