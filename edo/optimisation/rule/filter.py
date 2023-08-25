import numpy as np
from copy import deepcopy
from collections import defaultdict

from ... import Task, no_print
from ..categorisation import SeparationResult, HighImpactResult


# # # # # # # # # # # # # # # # # # # # # # #
# B Y   C O N D I T I O N   A N D   G O A L #
# # # # # # # # # # # # # # # # # # # # # # #

def filter_rules(rules, goal=None, cls_name=None, condition=None):
    """
    Filter rules by their goal, class they are derived for or by condition they must satisfy.
    :param rules: Iterable[Rule]: rules to filter
    :param goal: Goal or None: should the rules maximise or minimise the predicted value
    :param cls_name: int, str or None:
    :param condition: function[Rule -> boolean] or None:
    :return: List[Rule]: subset of rules
    """
    filtered = deepcopy(rules)
    if goal is not None:
        filtered = [r for r in filtered if r.goal == goal]
    if cls_name is not None:
        filtered = [r for r in filtered if r.cls_name == cls_name]
    if condition is not None:
        filtered = [r for r in filtered if condition(r)]

    return filtered


def condition_well_separated(rule, min_score):
    """
    Condition for well separated rules:
    - if rule derivation is based on a SeparationResult then the rule is kept only if score >= min_score
    - if rule derivation is based on a different result then the rule is kept
    Only rules based on SeparationResult can be filtered out.

    :param rule: Rule: rule for which condition should be checked
    :param min_score: float: minimal separation quality
    :return: False if `rule` is based on a SeparationResult and its score < min_score; True otherwise
    """
    if not isinstance(rule.derivation[0], SeparationResult):
        return True  # rules other than well_separated are kept
    else:
        return rule.derivation[0].score >= min_score


def condition_high_impact(rule, min_score):
    """
    Condition for high impact rules:
    - if rule derivation is based on a HighImpactResult then the rule is kept only if score >= min_score
    - if rule derivation is based on a different result then the rule is kept
    Only rules based on HighImpactResult can be filtered out.

    :param rule: Rule: rule for which condition should be checked
    :param min_score: float: minimal impact score
    :return: False if `rule` is based on a HighImpactResult and its score < min_score; True otherwise
    """
    if not isinstance(rule.derivation[0], HighImpactResult):
        return True  # rules other than high_impact are kept
    else:
        return rule.derivation[0].score >= min_score


# # # # # # # # # # # # # # #
# B Y   I M P O R T A N C E #
# # # # # # # # # # # # # # #

def filter_out_unimportant(rules, features, params, max_ratio, task):
    """
    Filter rules by the importance of features used to derive them.
    :param rules: Iterable[Rule]: rules to filter
    :param features: Iterable[Feature]: features used to derive rules
    :param params: {'niu': float, 'metric': str}: params for Feature.unimportant() to calculate unimportance score
    :param max_ratio: maximal allowed unimportance score
    :param task: Task: are the rules derived using a regression or a classification model
    :return: List[Rule]: rules that are derived on sufficiently important features
    """
    unimportant = _get_unimportnant(features, params, max_ratio, task)
    important_rules = [r for r in rules if (r.ftr_idx, r.cls_idx) not in unimportant]
    return important_rules


def _get_unimportnant(features, params, max_ratio, task):
    """
    Find unimportant features.
    :param features: Iterable[Feature]: features to filter
    :param params: {'niu': float, 'metric': str}: params for Feature.unimportant() to calculate unimportance score
    :param max_ratio: maximal allowed unimportance score
    :param task: Task: are the rules derived using a regression or a classification model
    :return: List[Tuple[feature index:int, class_index:int, str or None]]: tuples that identify unimportant features
    NOTE: features are identified by their index and class for which SHAP values are derived: (ft.ftr_index, cls_idx).
    """
    unimportant = []
    for ft in features:
        info = ft.unimportant(**params)
        if task == Task.CLASSIFICATION:
            for cls_idx, cls_info in enumerate(info):
                if cls_info.score >= max_ratio:
                    unimportant.append((ft.ftr_index, cls_idx))
        elif task == Task.REGRESSION:
            raise NotImplementedError
            if info.score >= max_ratio:
                unimportant.append((ft.ftr_index, None))
        else:
            raise ValueError(TASK_ERROR_MSG(task))
    return unimportant


# # # # # # # # # # # # # # # # # #
# B Y   C O N T R A D I C T I O N #
# # # # # # # # # # # # # # # # # #

def filter_contradictory_soft(rules):
    """
    Remove contradictory rules. This is done by:
    1. grouping rules that can contradict one another,
    2. from each group removing all rules that contradict the highest number of other rules,
    3. repeating step 2 until no two rules contradict one another.

    :param rules: Iterable[Rule]: rules to filter
    :return: List[Rule]: a subset of rules such that no two rules contradict one another
    """
    # group rules that can contradict one another
    g = _group_rules(rules)

    good_rules = []
    for k in g.keys():
        group = g[k]

        # repeat removing rules until no two rules contradict one another
        while True:
            # for each rule count how many other rules in contradicts
            n_group = []
            for rule in group:
                n_contradictions = len([other for other in group if rule.contradicts(other)])
                n_group.append((rule, n_contradictions))

            # list of contradiction counts
            ens = [n for r, n in n_group]
            if sum(ens) == 0:  # no two rules contradict one another
                break

            # remove rules that contradict the highest number of other rules
            group = [r for r, n in n_group if n < max(ens)]

        good_rules.extend(group)
    return good_rules


def contradictory_rules_stats(rules, print_func=no_print):
    """
    Print a summary of rules that includes:
    - the number of rules,
    - the number of groups and the total number of rules in groups,
      - groups consist of rules that can be compared, i.e. share the origin and operate on the same feature,
      - a group must contain at least two rules,
    - the number of rules that contradict at least one other rule.
    :param rules: List[Rule]: rules to summarise
    :param print_func: a function to print with; default: edo.no_print (prints nothing)
    :return: None
    """
    g = _group_rules(rules)
    groups = [g[k] for k in g.keys() if len(g[k]) > 1]

    n_contradictory_rules = 0
    for group in groups:
        for rule in group:
            if np.any([rule.contradicts(other) for other in group]):
                n_contradictory_rules += 1

    print_func(f"All rules: {len(rules)}, among them")
    print_func(f"{sum([len(g) for g in groups])} rules in {len(groups)} groups*")
    print_func(f"{n_contradictory_rules} contradictory rules")
    print_func("* a group must consist of at least two rules")
    return None


def _group_rules(rules):
    """
    Group rules by their origin and the index of feature they modify so that rules that can be compared are in the same
    group.
    :param rules: Iterable[Rule]: rules to group
    :return: {(feature_index, origin):[rule1, ...]}: a dictionary with (feature_index, origin) as keys
                                                     and lists of rules as values
    """
    groups = defaultdict(list)
    for r in rules:
        groups[(r.ftr_idx, r.origin)].append(r)
    return groups
