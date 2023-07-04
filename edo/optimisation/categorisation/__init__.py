import numpy as np
from collections import namedtuple


class Region(namedtuple('Region', ['majority', 'purity', 'n_correct', 'indices', 'start', 'end'])):
    def __repr__(self, ):
        return "Region(" + ', '.join([f"{attr}={repr(getattr(self, attr))}" for attr in self._asdict()]) + ")"

    def __str__(self):
        return "Region(" + ', '.join([f"{attr}={np.round(getattr(self, attr), 3)}" for attr in self._asdict() if attr not in ['indices'] ]) + ")"

    def as_dict(self):
        d = self._asdict()
        del d['indices']
        return d


SeparationResult = namedtuple('SeparationResult', ['score', 'thresholds', 'regions', 'params'])
HighImpactResult = namedtuple('HighImpactResult', ['score', 'loss_region', 'gain_region', 'params'])
UnimportantResult = namedtuple('UnimportantResult', ['score', 'params'])
RandomRule = namedtuple('RandomRule', [])  # for rules derived randomly

# from .categorisation import well_separated, high_impact, unimportant


def result_as_dict(result):
    d = result._asdict()

    if isinstance(result, HighImpactResult):
        d['type'] = 'HighImpactResult'
        d['loss_region'] = d['loss_region'].as_dict()
        d['gain_region'] = d['gain_region'].as_dict()
    elif isinstance(result, SeparationResult):
        d['type'] = 'SeparationResult'
        d['regions'] = [r.as_dict() for r in d['regions']]
    elif isinstance(result, RandomRule):
        d['type'] = 'RandomRule'
        d['score'] = None  # dla wygody w używaniu rule.as_dict
    else:
        raise NotImplementedError(f"{type(result)}")
    return d


from ... import Task, TASK_ERROR_MSG
from .high_impact import high_impact as _high_impact
from .unimportant import unimportant as _unimportant
from .well_separated import two_way_separation


def _calculate(f_vals, s_vals, func, kwargs, task):
    if task == Task.CLASSIFICATION:
        r = [func(f_vals, s_vals[c, :], **kwargs) for c in range(s_vals.shape[0])]
    elif task == Task.REGRESSION:
        r = func(f_vals, s_vals, **kwargs)
    else:
        raise ValueError(TASK_ERROR_MSG(task))
    return r


def well_separated(feature_values, shap_values, task):
    # call different function based on n_way
    func = two_way_separation
    return _calculate(feature_values, shap_values, func, {}, task)


def high_impact(feature_values, shap_values, task, gamma, metric):
    func = _high_impact
    return _calculate(feature_values, shap_values, func, {'gamma': gamma, 'metric': metric}, task)


def unimportant(feature_values, shap_values, task, miu, metric):
    func = _unimportant
    return _calculate(feature_values, shap_values, func, {'miu': miu, 'metric': metric}, task)


