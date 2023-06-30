from enum import Enum
from collections import namedtuple

import numpy as np


RNG = np.random.default_rng(0)


def set_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)


def get_random_generator():
    return RNG


# HistoryRecord stores which Rule was applied to Sample and if successfully
# ex. HistoryRecord(False, Rule) - there was an attempt to apply Rule but it was not applied
HistoryRecord = namedtuple('HistoryRecord', ['success', 'rule'])
# After updating f_vals, we should calculate new s_vals as well. Let's keep count.
ExtendedHistoryRecord = namedtuple('ExtendedHistoryRecord', ['success', 'rule', 'f_vals', 's_vals'])


# do we want to maximise or minimise the predicted value?
class Goal(Enum):
    MAXIMISATION = 'maximisation'
    MINIMISATION = 'minimisation'


def optimise(sample, rule, skip_criterion_check=False, update_shap=None, extended_history=False):
    """
    Optimise sample using rule.
    This function is called by both Sample.update and Rule.apply to provide a shared implementation.
    Check if rule can be applied and then change sample inplace by
    - updating its feature values (if rule can be applied) and
    - updating its history (always) and
    - updating its SHAP values (is update_shap is not None).
    
    sample: Sample
        sample to be optimised
    rule: Rule
        rule to apply
    skip_criterion_check: boolean
        if True then rule can be applied even if criterion is not satisfied; default: False
        Usecase: if the rule has a different origin than SHAP values of the sample
        then checking the criterion does not make sense bc SHAP values might be completely different.
    update_shap: SHAPCalculator or None
    extended_history: use ExtendedHistoryRecord instead of HistoryRecord?
    
    """
    assert sample.s_vals_origin.representation == rule.origin.representation, f"Representation of the sample must be the same for which the rule is derived. {sample.s_vals_origin.representation} != {rule.origin.representation}."

    # Rule.can_be_applied checks:
    # self.action.is_possible(sample) and (skip_criterion_check or self.criterion.is_satisfied(sample))
    if rule.can_be_applied(sample, skip_criterion_check):
        new_f_vals = rule.action.do(sample.f_vals)
        assert (new_f_vals != sample.f_vals).any(), 'Rule was applied but nth changed!'
        sample.f_vals = new_f_vals
        success = True
    else:
        new_f_vals = None
        success = False

    # Why do we always do this?
    # When applying several rules at once, only the last one asks to calculate SHAPs.
    # Now, the last one might have been unsuccessful but the previous one COULD change f_vals.
    # That's why we always do it.
    new_shaps = None
    if update_shap is not None:
        # TODO: ze względów optymalizacyjnych powinniśmy sprawdzać,
        # czy od ostatniego liczenia SHAPów, zmieniło się f_vals.
        new_shaps = update_shap.shap_values(sample)
        sample.s_vals = new_shaps

    if not extended_history:
        sample.history.append(HistoryRecord(success, rule))
    else:
        sample.history.append(ExtendedHistoryRecord(success, rule, new_f_vals, new_shaps))

# # # # # # # # # # # # # # # #
# # # H O W   T O   U S E # # #
# # # # # # # # # # # # # # # #
#
# from edo import make_origin
# from edo.optimisation.sample import make_samples
# from edo.shap_analysis.feature import make_features
#
# # load stuff
# task = Task.CLASSIFICATION
# m1_origin = make_origin(('human', 'random', 'krfp', task.value, 'trees'))
# model, shapcalculator = load_model(results_dir, m1_origin, check_unlogging=True)
#
# mldir = find_experiment(self.results_dir, 'ml', m1_origin)
# train, test = load_train_test(mldir)
# present_features = list(get_present_features(train[0], 0.1))
#
# shap_m1 = find_experiment(self.results_dir, 'shap', m1_origin)
# shap_smis, shap_x, shap_true_ys, classes_order, shap_vals = load_shap_files(shap_m1, task, check_unlogging=True)
#
# tr_preds, test_preds = load_predictions(mldir, self.task)
# tr_smis = index_of_smiles(shap_smis, tr_preds.index.tolist())
# te_smis = index_of_smiles(shap_smis, test_preds.index.tolist())
#
# # make Samples
# my_samples = make_samples(te_smis, shap_x, shap_vals, shap_smis, m1_origin, classes_order, task)
# # make Features
# my_features = make_features(present_features, tr_smis, shap_x, shap_vals, classes_order, m1_origin, task)
#
# # derive Rules
# hi_params = {'gamma': 0.001, 'metric': 'ratio'}
# well_separated_rules = []
# high_impact_rules = []
# random_rules = []
# for ft in self.my_features:
#     well_separated_rules.extend(derive_well_separated_two_way_rules(ft, task))
#     high_impact_rules.extend(derive_high_impact_rules(ft, hi_params, task))
#     random_rules.extend(derive_random_rules_sample(ft, task))
# all_rules = well_separated_rules + high_impact_rules + random_rules
#
# # optimise
# for sample in my_samples:
#     for rule in my_rules:
#         sample.update(rule, skip_criterion_check=False, update_shap=shapcalculator, extended_history=True)
