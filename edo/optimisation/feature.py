from collections import defaultdict

import numpy as np

from .categorisation import well_separated, high_impact, unimportant
from .._check import validate_shapes, validate_index, validate_task
from .. import make_origin, Task, deduce_task, TASK_ERROR_MSG


def make_features(feature_indices, sample_indices, feature_values, shap_values, classes_order, origin, task):
    """
    Create Feature objects.
    :param feature_indices: List: indices of features to use
    :param sample_indices: List: indices of samples to use
    :param feature_values: numpy.array [samples x features]: matrix of feature values
    :param shap_values: numpy.array [(classes x) samples x features]: matrix of SHAP values
    :param classes_order: numpy.array: classes order from the model used to calculate SHAP values or None in the case of
                                       regressors
    :param origin: Origin: description of the model used to calculate SHAP values
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :return: List[Feature]: list of Feature objects
    """
    my_features = []
    for ftr_index in feature_indices:
        if task == Task.CLASSIFICATION:
            f_vals = feature_values[sample_indices, ftr_index]
            s_vals = shap_values[:, sample_indices, ftr_index]
        elif task == Task.REGRESSION:
            raise NotImplementedError
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        ft = Feature(f_vals, s_vals, origin=origin, ftr_index=ftr_index, classes_order=classes_order, task=task,
                     name=f"F{ftr_index}")
        my_features.append(ft)

    return my_features


class Feature(object):
    def __init__(self, f_vals, s_vals, origin, ftr_index, classes_order=None, task=None, name=None):
        """
        Features are used to derive Rules.
        :param f_vals: numpy.array [samples]: matrix of feature values
        :param s_vals: numpy.array [(classes x) samples]: matrix of SHAP values
        :param origin: Origin: description of model used to calculate SHAP values
        :param ftr_index: index of this feature
        :param classes_order: numpy.array: classes order from model used to calculate SHAP values or None if the model
                                           is a regressor; default: None
        :param task: Task: is the model used to calculate SHAP values a classifier or a regressor if None then
                           deduced using s_vals; default: None
        :param name: str: human-friendly identifier of the feature; default: None
        """
        validate_shapes(f_vals, s_vals, classes_order)
        validate_index(ftr_index)
        task = deduce_task(task, s_vals)
        validate_task(task, s_vals)
        if task == Task.CLASSIFICATION:
            assert len(classes_order) == s_vals.shape[0], \
                f"classes_order length ({len(classes_order)}) does not match s_vals shape: {s_vals.shape}"
        elif task == Task.REGRESSION:
            assert classes_order is None, f"Task is {task} but classes_order is {classes_order}"
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        # PROPERTIES GIVEN IN ARGUMENTS
        self.name = name
        self._feature_values = f_vals
        self._shap_values = s_vals  # classes x samples
        self.s_vals_origin = make_origin(origin)
        self.ftr_index = ftr_index
        self._classes_order = classes_order
        self._task = task
        self._nsamples = self._feature_values.shape[0]

        # CACHED PROPERTIES
        self._well_separated = defaultdict(dict)  # {n_groups: {min_purity: [SeparationResult,...]}}
        self._high_impact = defaultdict(dict)     # {metric: {gamma: [HighImpactResult,...]}}
        self._unimportant = defaultdict(dict)     # {metric: {niu: [UnimportantResult,...]}}

    def __str__(self):
        return f"Name: {self.name}, n_samples: {self._nsamples}, {self._task}"

    def __repr__(self):
        return f"Feature({repr(self._feature_values)}, {repr(self._shap_values)}, {repr(self.s_vals_origin)}, {repr(self.ftr_index)}, classes_order={repr(self._classes_order)}, task={self._task}, name={self.name})"

    def info(self):
        """Human-readable detailed information about the object."""
        print(
            f"Name: {self.name}, origin: {self.s_vals_origin}, ftr_index: {self.ftr_index}, n_samples: {self._nsamples}, {self._task}, classes_order={self._classes_order}")

        for category in [self._well_separated, self._high_impact, self._unimportant]:
            for k1 in sorted(category.keys()):
                print(f"{k1}:")  # TODO: "k1" czy "k1: category[k1]?"
                for k2 in category[k1]:
                    print('\t', k2, ':')  # TODO: j.w.
                    for res in category[k1][k2]:
                        print("\t\t", res)
                    print('\n')
                print('\n')
        return

    def well_separated(self, n_groups=2, min_purity=None):
        """
        Calculate all optimal separations of this feature (category: well-separated).
        :param n_groups: int: number of well-separated groups, currently only two groups are supported
        :param min_purity: float: minimal required purity of each region
        :return: List[SeparationResult]: all optimal solutions
        """
        assert n_groups == 2, NotImplementedError("Currently only `n_groups` = 2 is supported.")

        try:
            return self._well_separated[n_groups][min_purity]
        except KeyError:
            self._well_separated[n_groups][min_purity] = well_separated(self._feature_values, self._shap_values,
                                                                        self._task, n_groups, min_purity)
        return self._well_separated[n_groups][min_purity]

    def high_impact(self, gamma, metric='ratio'):
        """
        Find loss and gain regions of this feature (category: high impact).
        :param gamma: float: minimal SHAP value of high impact samples, must be positive
        :param metric: str: which metric to use to calculate the score, must be `absolute`, `ratio` or `purity`
        :return: HighImpactResult
        """
        try:
            return self._high_impact[metric][gamma]
        except KeyError:
            self._high_impact[metric][gamma] = high_impact(self._feature_values, self._shap_values, self._task, gamma,
                                                           metric)
        return self._high_impact[metric][gamma]

    def unimportant(self, niu, metric):
        """
        Calculate importance score for this feature (category: unimportant).
        :param niu: float: maximal SHAP value of unimportant samples, must be positive
        :param metric: str: which metric to use to calculate the score, must be `ratio` or `absolute`
        :return: UnimportantResult
        """
        try:
            return self._unimportant[metric][niu]
        except KeyError:
            self._unimportant[metric][niu] = unimportant(self._feature_values, self._shap_values, self._task, niu,
                                                         metric)
        return self._unimportant[metric][niu]
