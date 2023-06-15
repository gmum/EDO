import os
import os.path as osp

import shap

import numpy as np

from functools import lru_cache

from .sample import Sample
from .. import Task, TASK_ERROR_MSG, make_origin
from ..config import parse_shap_config, utils_section
from ..utils import find_and_load, get_configs_and_model, usv
from ..data import Unlogger
from ..shap_analysis._check import _check_unlogging


class SHAPCalculator(object):
    def __init__(self, shapdir, mldir, origin, check_unlogging=True):
        # shapdir - load shap_cfg and background data
        shap_cfg = parse_shap_config(
            usv([osp.join(shapdir, f) for f in os.listdir(shapdir) if 'shap' in f and 'cfg' in f]))
        unlog = shap_cfg[utils_section]["unlog"]
        link = shap_cfg[utils_section]["link"]
        background_data = find_and_load(shapdir, "background_data.pickle", protocol='pickle')

        # mldir - load model and get Task info
        _, _, task_cfg, _, model_path = get_configs_and_model(mldir)
        task = Task(task_cfg[utils_section]['task'])
        model = find_and_load(mldir, osp.basename(model_path), protocol='pickle')

        if check_unlogging:
            _check_unlogging(unlog, task)

        if unlog:
            model = Unlogger(model)

        # now we can load classes_order from shapdir
        if task == Task.CLASSIFICATION:
            classes_order = find_and_load(shapdir, 'classes_order.npy', protocol='numpy')
        elif task == Task.REGRESSION:
            classes_order = None
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        # make explainer
        if task == Task.CLASSIFICATION:
            e = shap.KernelExplainer(model.predict_proba, background_data, link=link)
        elif task == Task.REGRESSION:
            e = shap.KernelExplainer(model.predict, background_data, link=link)
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        # put everything into object
        # TODO: To się zepsuje, jeżeli ktoś będzie grzebał w bebechach i podmieniał pola
        self._repr_params = {'shapdir': shapdir, 'mldir': mldir,
                             'origin': origin, 'check_unlogging': check_unlogging}

        self.explainer = e
        self.link = link
        self.background_data = background_data

        self.model = model
        self.unlog = unlog
        self.origin = make_origin(origin)
        self.classes_order = classes_order

        self.task = task

    def __str__(self, ):
        return f"{self.origin}, link={self.link}, unlog={self.unlog}"

    def __repr__(self, ):
        return f"SHAPCalculator(**{repr(self._repr_params)})"

    def shap_values(self, x):
        assert isinstance(x, (np.ndarray, Sample)), f"{type(x)}"
        try:
            assert self.origin == x.s_vals_origin, f"Origin mismatch: {self.origin} != {x.s_vals_origin}"
            assert np.all(self.classes_order == x.classes_order), f"Classes order mismatch: {self.classes_order} != {x.classes_order}"
            f_vals = x.f_vals
        except AttributeError:
            # it's a numpy array
            f_vals = x

        # tuplujemy f_vals bo np.array is not hashable
        return self._shap_values(tuple(f_vals))

    @lru_cache(maxsize=None)
    def _shap_values(self, x):
        # TODO: możliwe, że w regresji obudowanie przez np.array będzie przeszkadzać
        # ale w klasyfikacji jest konieczne
        return np.array(self.explainer.shap_values(np.array(x)))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
