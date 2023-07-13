from . import get_random_generator

"""Scenarios for optimisation"""


def optimise(samples, rules, at_once=1, n_times=1, shap_calculator=None, skip_criterion_check=False,
             extended_history=False):
    """
    Default scenario in which each sample is optimised with at_once * n_times rules that are randomly selected from
    the set of rules that can be applied. Importantly, if `at_once` > 1 then some of the rules might not be applied
    successfully. Such rules are included in sample's history even though the sample's feature values do not change.
    SHAP values are recalculated (if shap_calculator != None) after each update cycle (i.e. n_times times).
    :param samples: List[Sample]: samples to optimise
    :param rules: List[Rule]: rules to optimise with
    :param at_once: int: number of rules to apply in each update cycle; default: 1
    :param n_times: int: number of update cycles, each update cycle applies at_once rules; default: 1
    :param shap_calculator: SHAPCalculator or None: a model to calculate SHAP values of the optimised sample,
                            if SHAP values should not be recalculated use None; default None
    :param skip_criterion_check: boolean: if True will skip criterion check while making a list of all possible rules;
                                          default: False
    :param extended_history: boolean: if True will use ExtendedHistoryRecord instead of HistoryRecord; default: False
    :return: None (samples are modified inplace)
    """
    rng = get_random_generator()  # ensure reproducibility
    for sample in samples:
        # cycle:
        for i in range(n_times):
            possible_rules = [r for r in rules if r.can_be_applied(sample, skip_criterion_check)]
            if len(possible_rules) == 0:
                break
            chosen = rng.choice(possible_rules, at_once)  # randomly select `at_once` rules
            # update sample with `at_once` rules at once and only then update SHAP values.
            for chosen_rule in chosen[:-1]:
                sample.update(chosen_rule, shap_calculator=None, skip_criterion_check=skip_criterion_check,
                              extended_history=extended_history)
            sample.update(chosen[-1], shap_calculator=shap_calculator, skip_criterion_check=skip_criterion_check,
                          extended_history=extended_history)
    return None
