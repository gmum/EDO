import numpy as np

from enum import Enum
from collections import namedtuple

# global random generator to ensure reproducibility
RNG = np.random.default_rng(0)


def set_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)
    return


def get_random_generator():
    return RNG


class Goal(Enum):
    MAXIMISATION = 'maximisation'
    MINIMISATION = 'minimisation'


# HistoryRecord stores which Rule was applied and if successfully
HistoryRecord = namedtuple('HistoryRecord', ['success', 'rule'])
# ExtendedHistoryRecord additionally stores information about new feature and SHAP values after the update
ExtendedHistoryRecord = namedtuple('ExtendedHistoryRecord', ['success', 'rule', 'f_vals', 's_vals'])


# TODO: poniższa forma dokumentacji jest niedomyślna dla Pycharma ale lepiej się czyta. Może zmienić też w innych plikach?
def optimise(sample, rule, skip_criterion_check=False, shap_calculator=None, extended_history=False):
    """
    This function is called by both Sample.update and Rule.apply to provide a shared implementation.
    It checks if rule can be applied and changes sample inplace by:
    - updating its feature values (if rule can be applied) and
    - updating its history (always) and
    - updating its SHAP values (if update_shap is not None).
    
    sample: Sample
        sample to be optimised
    rule: Rule
        rule to apply
    skip_criterion_check: boolean
        if True then rule can be applied even if criterion is not satisfied; default: False
    shap_calculator: SHAPCalculator or None
        a model to calculate SHAP values of the optimised sample, if SHAP values should not be recalculated use None;
        default: None
    extended_history: boolean
        if True will use ExtendedHistoryRecord instead of HistoryRecord; default: False
    """

    if rule.can_be_applied(sample, skip_criterion_check):
        new_f_vals = rule.action.do(sample.f_vals)
        assert (new_f_vals != sample.f_vals).any(), 'Rule was applied but nothing changed!'
        sample.f_vals = new_f_vals
        success = True
    else:
        new_f_vals = None
        success = False

    # In some scenarios only the last rule asks to update SHAP values. Even if this rule was not applied, a previous one
    # could have changed the feature values and so SHAP values must be updated even if success==False.
    new_shaps = None
    if shap_calculator is not None:
        new_shaps = sample.get_s_vals(new_f_vals)  # maybe we already have it calculated?
        if new_shaps is None:
            new_shaps = shap_calculator.shap_values(sample)
        sample.s_vals = new_shaps

    if not extended_history:
        sample.history.append(HistoryRecord(success, rule))
    else:
        sample.history.append(ExtendedHistoryRecord(success, rule, new_f_vals, new_shaps))
