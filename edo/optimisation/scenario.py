from . import get_random_generator

"""Scenarios for optimisation"""


# TODO: tests!


def optimise(samples, rules, at_once=1, n_times=1, update_shap=None,
             # basic scenario
             skip_criterion_check=False, extended_history=False):
    # związek ma w sumie zaaplikowanych at_once * n_times reguł
    # SHAPy są liczone n_times razy (jeśli są liczone)
    # samples: samples to optimise
    # rules: rules to optimise with
    # at_once - ile reguł na raz aplikujemy
    # n_times - ile razy updejtujemy
    # update_shap: SHAPCalculator or None
    # skip_criterion_check: skip criterion check while making a list of all possible rules?
    # extended_history: use ExtendedHistoryRecord instead of HistoryRecord?

    rng = get_random_generator()
    for sample in samples:
        for i in range(n_times):
            possible_rules = [r for r in rules if r.can_be_applied(sample, skip_criterion_check)]
            if len(possible_rules) == 0:
                continue
            chosen = rng.choice(possible_rules, at_once)
            for chosen_rule in chosen[:-1]:
                sample.update(chosen_rule, update_shap=None,
                              skip_criterion_check=skip_criterion_check,
                              extended_history=extended_history)
            # OK, so this looks weird but makes sense.
            # We update sample with `at_once` rules at once
            # and only then update SHAPs.
            sample.update(chosen[-1], update_shap=update_shap,
                          skip_criterion_check=skip_criterion_check,
                          extended_history=extended_history)
