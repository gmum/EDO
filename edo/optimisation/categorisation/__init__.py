import numpy as np
from collections import namedtuple


class Region(namedtuple('Region', ['majority', 'purity', 'n_correct', 'indices', 'start', 'end'])):
    def __repr__(self):
        return "Region(" + ', '.join([f"{attr}={repr(getattr(self, attr))}" for attr in self._asdict()]) + ")"

    def __str__(self):
        return "Region(" + ', '.join([f"{attr}={np.round(getattr(self, attr), 3)}" for attr in self.as_dict()]) + ")"

    def as_dict(self):
        """Same as self._asdict() but without `indices`."""
        d = self._asdict()
        del d['indices']
        return d


SeparationResult = namedtuple('SeparationResult', ['score', 'thresholds', 'regions', 'params'])       # well-separated
HighImpactResult = namedtuple('HighImpactResult', ['score', 'loss_region', 'gain_region', 'params'])  # high impact
UnimportantResult = namedtuple('UnimportantResult', ['score', 'params'])                              # unimportant
RandomRule = namedtuple('RandomRule', [])                                                             # random


def result_as_dict(result):
    """
    Human-friendly version of self._asdict()
    :param result: one of SeparationResult, HighImpactResult, UnimportantResult or RandomRule
    :return: dictionary describing `result`
    """
    d = result._asdict()
    if isinstance(result, SeparationResult):
        d['type'] = 'SeparationResult'
        d['regions'] = [r.as_dict() for r in d['regions']]
    elif isinstance(result, HighImpactResult):
        d['type'] = 'HighImpactResult'
        d['loss_region'] = d['loss_region'].as_dict()
        d['gain_region'] = d['gain_region'].as_dict()
    elif isinstance(result, UnimportantResult):
        d['type'] = 'UnimportantResult'
    elif isinstance(result, RandomRule):
        d['type'] = 'RandomRule'
        d['score'] = None  # for convenience when using Rule.as_dict()
    else:
        raise NotImplementedError(f"{type(result)}")
    return d


from ... import Task, TASK_ERROR_MSG
from .well_separated import two_way_separation
from .high_impact import high_impact as _high_impact
from .unimportant import unimportant as _unimportant


def _calculate(f_vals, s_vals, func, kwargs, task):
    """
    Common API to calculate any category.
    :param f_vals: numpy.array [samples]: matrix of feature values
    :param s_vals: numpy.array [(classes x) samples]: matrix of SHAP values
    :param func: function: a function to call
    :param kwargs: arguments to `func`
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :return: the result of calling `func` (for each class)
    """
    if task == Task.CLASSIFICATION:
        r = [func(f_vals, s_vals[c, :], **kwargs) for c in range(s_vals.shape[0])]
    elif task == Task.REGRESSION:
        r = func(f_vals, s_vals, **kwargs)
    else:
        raise ValueError(TASK_ERROR_MSG(task))
    return r


def well_separated(feature_values, shap_values, task, n_groups, min_purity):
    """
    Describe feature as well-separated.
    :param feature_values: numpy.array [samples]: matrix of feature values
    :param shap_values: numpy.array [(classes x) samples]: matrix of SHAP values
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :param n_groups: int: number of well-separated groups, currently only two groups are supported
    :param min_purity: float: minimal required purity of each region
    :return: List[SeparationResult]: all optimal solutions
    """
    assert n_groups == 2, NotImplementedError("Currently only `n_groups` = 2 is supported.")
    func = two_way_separation  # NOTE: in the future, call different functions based on n_groups
    return _calculate(feature_values, shap_values, func, {'min_purity': min_purity}, task)


def high_impact(feature_values, shap_values, task, gamma, metric):
    """
    Describe feature as high impact.
    :param feature_values: numpy.array [samples]: matrix of feature values
    :param shap_values: numpy.array [(classes x) samples]: matrix of SHAP values
    :param task: task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :param gamma: float: minimal SHAP value of high impact samples, must be positive
    :param metric: str: which metric to use to calculate the score, must be `absolute`, `ratio` or `purity`
    :return: HighImpactResult
    """
    func = _high_impact
    return _calculate(feature_values, shap_values, func, {'gamma': gamma, 'metric': metric}, task)


def unimportant(feature_values, shap_values, task, niu, metric):
    """
    Describe feature as unimportant.
    :param feature_values: numpy.array [samples]: matrix of feature values
    :param shap_values: numpy.array [(classes x) samples]: matrix of SHAP values
    :param task: task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :param niu: float: maximal SHAP value of unimportant samples, must be positive
    :param metric: str: which metric to use to calculate the score, must be `ratio` or `absolute`
    :return: UnimportantResult
    """
    func = _unimportant
    return _calculate(feature_values, shap_values, func, {'niu': niu, 'metric': metric}, task)


