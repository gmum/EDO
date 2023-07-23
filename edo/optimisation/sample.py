import numpy as np
from copy import deepcopy

from . import optimise, ExtendedHistoryRecord
from .._check import validate_shapes
from .. import make_origin, Task, TASK_ERROR_MSG


def make_samples(sample_indices, feature_values, shap_values, smiles_order, origin, classes_order, task):
    """
    Create Sample objects.
    :param sample_indices: List: indices of samples to use
    :param feature_values: numpy.array [samples x features]: matrix of feature values
    :param shap_values: numpy.array [(classes x) samples x features]: matrix of SHAP values
    :param smiles_order: numpy.array [samples]: matrix with SMILES
    :param origin: Origin: description of the model used to calculate SHAP values
    :param classes_order: numpy.array: classes order from the model used to calculate SHAP values or None in the case of
                                       regressors
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :return: List[Samples]: list of Sample objects
    """
    my_samples = []
    for idx in sample_indices:
        if task == Task.CLASSIFICATION:
            f_vals = feature_values[idx, :]
            s_vals = shap_values[:, idx, :]
        elif task == Task.REGRESSION:
            raise NotImplementedError
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        s = Sample(f_vals, s_vals, origin, classes_order, smiles_order[idx])
        my_samples.append(s)
    return my_samples


class Sample(object):
    def __init__(self, feature_values, shap_values, origin, classes_order=None, smiles=None):
        """
        Sample that can be optimised (e.x. a molecule).
        :param feature_values: numpy.array[features]: matrix of feature values
        :param shap_values: shap_values: numpy.array[(classes x) features]: matrix of SHAP values
        :param origin: Origin: description of the model used to calculate SHAP values
        :param classes_order: numpy.array[classes] or None: classes order from the model used to calculate SHAP values
                                                            or None in the case of regressors; default: None
        :param smiles: str: SMILES; default: None
        """
        validate_shapes(feature_values, shap_values, classes_order=classes_order)

        self.original_f_vals = feature_values   # original values
        self.f_vals = deepcopy(feature_values)  # current values, possibly updated
        self.original_s_vals = shap_values
        self.s_vals = deepcopy(shap_values)
        self.origin = make_origin(origin)
        self.classes_order = classes_order
        self.smiles = smiles
        self.history = []

    def __str__(self):
        return f"{self.smiles} ({self.origin})"

    def __repr__(self):
        return f"Sample({repr(self.original_f_vals)}, {repr(self.original_s_vals)}, {repr(self.origin)}, {repr(self.classes_order)}, {repr(self.smiles)})"

    def update(self, rule, skip_criterion_check=False, shap_calculator=None, extended_history=False):
        """
        Call edo.optimisation.optimise and return self to allow chaining (sample.update(rule1).update(rule2) )
        :param rule: Rule: rule to apply
        :param skip_criterion_check: boolean: if True then rule can be applied even if criterion is not satisfied;
                                              default: False
        :param shap_calculator: SHAPCalculator or None: a model to calculate SHAP values of the optimised sample,
                                if SHAP values should not be recalculated use None; default: None
        :param extended_history: if True will use ExtendedHistoryRecord instead of HistoryRecord; default: False
        :return: Sample: self
        """
        optimise(self, rule, skip_criterion_check, shap_calculator, extended_history)
        return self

    def print_history(self):
        """
        Print history of updates - each rule that was applied (successfully or not) on the sample.
        :return: None
        """
        for rec in self.history:
            result = 'applied' if rec.success else 'skipped'
            # print(f"{result}    {rec.rule.name} ({rec.rule.origin}): {rec.rule.action} ({rec.rule.criterion})")
            print(f"{result}    {rec.rule.name} ({rec.rule})")
        print()  # empty string at the end
        return

    def number_of_applied_changes(self):
        """
        Get the number of times a rule was successfully applied.
        :return: int: number of changes
        """
        return len([rec for rec in self.history if rec.success])

    def get_s_vals(self, new_f_vals):
        """
        Search history to check if SHAP values for these feature values are already calculated.
        :param new_f_vals: np.array[features]: matrix of feature values
        :return: np.array[features] or None: SHAP values for new_f_vals or None if they are not found in the history
        """
        for entry in reversed(self.history):
            if not isinstance(entry, ExtendedHistoryRecord) or np.any(entry.f_vals != new_f_vals):
                continue
            else:
                return deepcopy(entry.s_vals)
        if np.all(self.original_f_vals == new_f_vals):
            return deepcopy(self.original_s_vals)
        return None
