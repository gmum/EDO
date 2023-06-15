from collections import namedtuple
import numpy as np


class Region(namedtuple('Region', ['majority', 'purity', 'n_correct', 'indices', 'start', 'end'])):
    def __repr__(self, ):
        return "Region(" + ', '.join([f"{attr}={repr(getattr(self, attr))}" for attr in self._asdict()]) + ")"

    def __str__(self):
        return "Region(" + ', '.join([f"{attr}={np.round(getattr(self, attr), 3)}" for attr in self._asdict() if attr not in ['indices'] ]) + ")"

SeparationResult = namedtuple('SeparationResult', ['score', 'thresholds', 'regions', 'params'])
HighImpactResult = namedtuple('HighImpactResult', ['score', 'loss_region', 'gain_region', 'params'])
UnimportantResult = namedtuple('UnimportantResult', ['score', 'params'])
RandomRule = namedtuple('RandomRule', [])  # for rules derived randomly

from .categorisation import well_separated, high_impact, unimportant


def result_as_dict(result):
    d = result._asdict()

    if isinstance(result, HighImpactResult):
        d['type'] = 'HighImpactResult'
        d['loss_region'] = region_as_dict(d['loss_region'])
        d['gain_region'] = region_as_dict(d['gain_region'])
    elif isinstance(result, SeparationResult):
        d['type'] = 'SeparationResult'
        d['regions'] = [region_as_dict(r) for r in d['regions']]
    elif isinstance(result, RandomRule):
        d['type'] = 'RandomRule'
        d['score'] = None  # dla wygody w używaniu rule.as_dict
    else:
        raise NotImplementedError(f"{type(result)}")
    return d


def region_as_dict(region):
    d = region._asdict()
    del d['indices']
    return d
