from collections import defaultdict

import numpy as np

from .categorisation import well_separated, high_impact, unimportant
from ._check import validate_task
from .._check import validate_shapes, validate_index
from .. import make_origin, Task, deduce_task, TASK_ERROR_MSG


def make_features(features, samples, feature_values, shap_values, classes_order, origin, task):
    # features - indices of features we are interested in
    # samples - indices of samples we are interested in
    # feature_values - matrix of feature values
    # shap_values - matrix of SHAP values
    # classes_order -
    # origin - where do SHAP values come from
    # task -

    my_features = []

    for ftr_index in features:
        if task == Task.CLASSIFICATION:
            f_vals = feature_values[samples, ftr_index]
            s_vals = shap_values[:, samples, ftr_index]
        elif task == Task.REGRESSION:
            raise NotImplementedError
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        ft = Feature(f_vals, s_vals, origin=origin, ftr_index=ftr_index,
                     classes_order=classes_order, task=task,
                     name=f"F{ftr_index}")
        my_features.append(ft)

    return my_features


class Feature(object):
    """
    TODO: doksy
    origin: Origin
        inf. about the model for which the rule is derived
    ftr_index: int
        przekazywane do Rule, żeby było wiadomo, na którym indeksie ma być podmieniana wartość
    """

    def __init__(self, f_vals, s_vals, origin, ftr_index, classes_order=None, task=None, name=None):
        validate_shapes(f_vals, s_vals, classes_order)
        validate_index(ftr_index)
        task = deduce_task(task, s_vals)
        validate_task(task, s_vals)
        if task == Task.CLASSIFICATION:
            assert len(classes_order) == s_vals.shape[0],\
                f"classes_order length {len(classes_order)} does not match s_vals shape {s_vals.shape}"
        elif task == Task.REGRESSION:
            assert classes_order is None, f"Task is {task} but classes_order is {classes_order}"
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        # PROPERTIES GIVEN AS ARGUMENTS
        self.name = name
        self._feature_values = f_vals
        self._shap_values = s_vals  # classes x samples
        self.s_vals_origin = make_origin(origin)
        self.ftr_index = ftr_index
        self._classes_order = classes_order
        self._task = task
        self._nsamples = self._feature_values.shape[0]

        # # BELOW ARE CACHE PROPERTIES - THEY ARE FILLED WITH CONTENT WHEN THE CONTENT IS CALCULATED
        # categorisation cache
        self._well_separated = defaultdict(dict)  # {n_way: {min_purity: [SeparationResult,...]}}
        self._high_impact = defaultdict(dict)  # {metric: {gamma: [HighImpactResult,...]}}
        self._selectively_important = defaultdict(dict)  # {metric: {miu: [SelectivelyImportantResult,...]}}
        self._unimportant = defaultdict(dict)  # {metric: {miu: [UnimportantResult,...]}}

        # other cache
        self._importance = np.mean(np.abs(self._shap_values))

    def __str__(self, ):
        return f"Name: {self.name}, n_samples: {self._nsamples}, {self._task}"

    def __repr__(self, ):
        return f"Feature({repr(self._feature_values)}, {repr(self._shap_values)}, origin={repr(self.s_vals_origin)}, ftr_index={repr(self.ftr_index)}, classes_order={repr(self._classes_order)}, task={self._task}, name={self.name})"

    def info(self, ):
        print(
            f"Name: {self.name}, origin: {self.s_vals_origin}, ftr_index: {self.ftr_index}, n_samples: {self._nsamples}, {self._task}, classes_order={self._classes_order}, importance: {np.round(self._importance, 3)}")

        for categ in [self._well_separated, self._high_impact, self._selectively_important, self._unimportant]:
            for k1 in sorted(categ.keys()):
                print(f"{k1}:")
                for k2 in categ[k1]:
                    print('\t', k2, ':')
                    for res in categ[k1][k2]:
                        print("\t\t", res)
                    print('\n')
                print('\n')

    def well_separated(self, n_groups, min_purity=None):
        # min purity is important only for three_way_separation
        params = {} if min_purity is None else {'min_purity': min_purity}
        try:
            return self._well_separated[n_groups][min_purity]
        except KeyError:
            self._well_separated[n_groups][min_purity] = well_separated(self._feature_values, self._shap_values,
                                                                        self._task, n_groups, params)
        finally:
            return self._well_separated[n_groups][min_purity]

    def high_impact(self, gamma, metric='ratio'):
        try:
            return self._high_impact[metric][gamma]
        except KeyError:
            self._high_impact[metric][gamma] = high_impact(self._feature_values, self._shap_values, self._task,
                                                           gamma, metric)
        finally:
            return self._high_impact[metric][gamma]


    def unimportant(self, miu, metric):
        try:
            return self._unimportant[metric][miu]
        except KeyError:
            self._unimportant[metric][miu] = unimportant(self._feature_values, self._shap_values, self._task, miu,
                                                         metric)
        finally:
            return self._unimportant[metric][miu]

    def importance(self, ):
        return self._importance
