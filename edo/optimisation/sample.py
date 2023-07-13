import numpy as np
from copy import deepcopy

from . import optimise, ExtendedHistoryRecord
from .._check import validate_shapes
from .. import make_origin, Task, TASK_ERROR_MSG


def make_samples(samples, feature_values, shap_values, smiles_order, origin, classes_order, task):
    # samples - indices of samples we are interested in
    # feature_values - matrix of feature values
    # shap_values - matrix of SHAP values
    # smiles_order - array of SMILES
    # origin - where do SHAP values come from

    my_samples = []
    for smi_idx in samples:
        if task == Task.CLASSIFICATION:
            f_vals = feature_values[smi_idx, :]
            s_vals = shap_values[:, smi_idx, :]
        elif task == Task.REGRESSION:
            raise NotImplementedError
        else:
            raise ValueError(TASK_ERROR_MSG(task))

        s = Sample(f_vals, s_vals, origin, classes_order, smiles_order[smi_idx])
        my_samples.append(s)
    return my_samples


class Sample(object):
    """
    Sample that can be optimised (e.x. a molecule)

    feature_values: np.array (1 dimension)
        feature values, this is what will be optimised
    shap_values: np.array (1 or 2 dimensions)
        SHAP values, used to check if Sample is eligible for optimisation
    shap_values_origin: Origin
        information about the model that calculated `shap_values`
    classes_order: TODO uzupełnić
    smiles: str or None
        TODO: this assumes that Sample is in fact a molecule, it would be better to
              remove smiles from Sample and make a class Molecule(Sample) that only adds smiles
    """

    def __init__(self, feature_values, shap_values, shap_values_origin, classes_order=None, smiles=None):
        validate_shapes(feature_values, shap_values, classes_order=classes_order)

        self.original_f_vals = feature_values  # original values
        self.f_vals = deepcopy(feature_values)  # current values, possibly updated
        self.original_s_vals = shap_values
        self.s_vals = deepcopy(shap_values)
        self.s_vals_origin = make_origin(shap_values_origin)
        self.classes_order = classes_order
        self.smiles = smiles
        self.history = []

    def __str__(self):
        return f"{self.smiles} ({self.s_vals_origin})"

    def __repr__(self):
        return f"Sample({repr(self.original_f_vals)}, {repr(self.original_s_vals)}, {repr(self.s_vals_origin)}, {repr(self.classes_order)}, {repr(self.smiles)})"

    def update(self, rule, skip_criterion_check=False, shap_calculator=None, extended_history=False):
        """
        Call optimise and return self to allow chaining (sample.update(rule1).update(rule2) )
        """
        optimise(self, rule, skip_criterion_check, shap_calculator, extended_history)
        return self

    def print_history(self):
        for rec in self.history:
            result = 'applied' if rec.success else 'skipped'
            # print(f"{result}    {rec.rule.name} ({rec.rule.origin}): {rec.rule.action} ({rec.rule.criterion})")
            print(f"{result}    {rec.rule.name}")
        print()  # empty string at the end
        return

    def number_of_applied_changes(self):
        return len([rec for rec in self.history if rec.success])

    def get_s_vals(self, new_f_vals):
        # if SHAP values for these feature values are already calculated then they are returned, otherwise None
        for entry in reversed(self.history):
            if not isinstance(entry, ExtendedHistoryRecord) or np.any(entry.f_vals != new_f_vals):
                continue
            else:
                return entry.s_vals  # TODO: na wszelki wypadek upewnić się, że to jest przekazywane przez referencję albo dać deepcopy
        if np.all(self.original_f_vals == new_f_vals):
            return self.original_s_vals  # TODO: j.w.
        return None
