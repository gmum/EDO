import os
import os.path as osp

import shap
import numpy as np
from functools import lru_cache

from .sample import Sample
from .._check import _check_unlogging
from .. import Task, TASK_ERROR_MSG, make_origin
from ..config import parse_shap_config, UTILS
from ..utils import find_and_load, get_configs_and_model, usv
from ..wrappers import Unloger


class SHAPCalculator(object):
    def __init__(self, shap_dir, ml_dir, origin, check_unlogging=True):
        """
        Construct a SHAP explainer using data loaded from directories [shap|ml]_dir/origin.
        :param shap_dir: str: path to root directory for explanation data
        :param ml_dir: str: path to root directory with trained models
        :param origin: Origin: identification of the experiment
        :param check_unlogging: boolean: if True will ensure that regressors are unlogged and classifiers are not;
                                         default: True
        """

        # shap_dir - load shap_cfg and background data
        shap_cfg = parse_shap_config(
            usv([osp.join(shap_dir, f) for f in os.listdir(shap_dir) if 'shap' in f and 'cfg' in f]))
        unlog = shap_cfg[UTILS]["unlog"]
        link = shap_cfg[UTILS]["link"]
        background_data = find_and_load(shap_dir, "background_data.pickle", protocol='pickle')

        # ml_dir - load model and get Task info
        _, _, task_cfg, _, model_path = get_configs_and_model(ml_dir)
        task = Task(task_cfg[UTILS]['task'])
        model = find_and_load(ml_dir, osp.basename(model_path), protocol='pickle')

        if check_unlogging:
            _check_unlogging(unlog, task)

        if unlog:
            model = Unloger(model)

        # load classes_order from shap_dir and make explainer
        if task == Task.CLASSIFICATION:
            classes_order = find_and_load(shap_dir, 'classes_order.npy', protocol='numpy')
            e = shap.KernelExplainer(model.predict_proba, background_data, link=link)
        elif task == Task.REGRESSION:
            classes_order = None
            e = shap.KernelExplainer(model.predict, background_data, link=link)
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        # put everything into object
        self._repr_params = {'shap_dir': shap_dir, 'ml_dir': ml_dir,
                             'origin': origin, 'check_unlogging': check_unlogging}
        self._origin = make_origin(origin)
        self._task = task

        self._model = model
        self._unlog = unlog
        self._classes_order = classes_order

        self._explainer = e
        self._link = link
        self._background_data = background_data

    def __str__(self):
        return f"{self._origin}, link={self._link}, unlog={self._unlog}"

    def __repr__(self):
        return f"SHAPCalculator(**{repr(self._repr_params)})"

    def shap_values(self, x):
        """
        Calculate SHAP values for x.
        :param x: numpy.array [features] or Sample: a single sample
        :return: np.array: SHAP values
        """
        assert isinstance(x, (np.ndarray, Sample)), f"{type(x)}"
        try:
            assert self._origin == x.origin, f"Origin mismatch: {self._origin} != {x.origin}"
            assert np.all(self._classes_order == x.classes_order), f"Classes order mismatch: {self._classes_order} != {x.classes_order}"
            f_vals = x.f_vals
        except AttributeError:
            # it's a numpy array
            f_vals = x

        # convert to tuple because np.array is not hashable
        return self._shap_values(tuple(f_vals))

    @lru_cache(maxsize=None)
    def _shap_values(self, x):
        """
        Calculate SHAP values for x.
        :param x: numpy.array [features] or Sample: a single sample
        :return: np.array: SHAP values
        """
        # NOTE: in the case of regression, calling np.array on x might not work
        return np.array(self._explainer.shap_values(np.array(x)))

    def predict(self, X):
        """
        Calculate prediction for X.
        :param X: np.array [samples x features]: samples
        :return: np.array [samples]: predictions
        """
        return self._model.predict(X)

    def predict_proba(self, X):
        """
        Calculate prediction for X and return as probabilities for each class (only for classifiers!).
        :param X: np.array [samples x features]: samples
        :return: np.array [samples x classes]: predictions
        """
        return self._model.predict_proba(X)
