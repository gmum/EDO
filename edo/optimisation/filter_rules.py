from copy import deepcopy
from collections import defaultdict

from .. import Task
from ..shap_analysis.categorisation import SeparationResult, HighImpactResult


# # # # # # # # # # # # # # # # # # # # # # #
# B Y   C O N D I T I O N   A N D   G O A L #
# # # # # # # # # # # # # # # # # # # # # # #

def filter_rules(rules, goal=None, cls_name=None, condition=None):
    # returns rules that optimise given goal
    # (for a given class (identified by cls_name not its index!))
    # and satisfy a given condition

    # must make a new list so that the given list is not updated!
    filtered = deepcopy(rules)
    if goal is not None:
        filtered = [r for r in rules if r.goal == goal]
    if cls_name is not None:
        filtered = [r for r in filtered if r.cls_name == cls_name]
    if condition is not None:
        filtered = [r for r in filtered if condition(r)]

    return filtered


# TODO: dwa poniższe można przepisać jako universal condition i po prostu podawać im klasę
def condition_well_separated(rule, min_score):
    # (only rules based on SeparationResult can be filtered out)
    # if rule derivation is based on a SeparationResult then the rule is kept only if score >= min_score
    # if rule derivation is based on a different result then the rule is kept
    if not isinstance(rule.derivation[0], SeparationResult):
        return True  # rules other than well_separated are kept
    else:
        return rule.derivation[0].score >= min_score


def condition_high_impact(rule, min_score):
    # (only rules based on HighImpactResult can be filtered out)
    # if rule derivation is based on a HighImpactResult then the rule is kept only if score >= min_score
    # if rule derivation is based on a different result then the rule is kept
    if not isinstance(rule.derivation[0], HighImpactResult):
        return True  # rules other than high_impact are kept
    else:
        return rule.derivation[0].score >= min_score


# # # # # # # # # # # # # # # #
# B Y   U N I M P O R T A N T #
# # # # # # # # # # # # # # # #

def filter_out_unimportant(rules, features, params, max_ratio, task):
    """

    :param rules: list of rules
    :param features: list of features
    :param params: params for calculating unimportance (miu and metric)
    :param max_ratio: maximal percentage of unimportant samples allowed
    :param task: task
    :return: important rules
    """
    unimportant = _get_unimportnant(features, params, max_ratio, task)
    important_rules = [r for r in rules if (r.ftr_idx, r.cls_idx) not in unimportant]
    return important_rules


def _get_unimportnant(features, params, max_ratio, task):
    """

    :param features:
    :param params:
    :param max_ratio:
    :param task:
    :return: list of tuples (feature index, class index) that identifies important features
    """
    # TODO identyfikuje cechoklasy po (ft.ftr_index, cls_idx)
    # zwraca listę tupli
    unimportant = []
    for ft in features:
        info = ft.unimportant(**params)
        if task == Task.CLASSIFICATION:
            # TODO uwaga! ten kawałek identyfikuje cechoklasy po ft.ftr_index, cls_idx
            for cls_idx, cls_info in enumerate(info):
                if cls_info.score >= max_ratio:
                    unimportant.append((ft.ftr_index, cls_idx))
        elif task == Task.REGRESSION:
            raise NotImplementedError
            # TODO: napisane na sucho, never run or tested
            if info.score >= max_ratio:
                unimportant.append((ft.ftr_index, None))
        else:
            raise ValueError(TASK_ERROR_MSG(task))
    return unimportant


# # # # # # # # # # # # # # # # # #
# B Y   C O N T R A D I C T I O N #
# # # # # # # # # # # # # # # # # #

def filter_contradictive_soft(rules):
    # grupuje cechy, które mogą być porównywane
    # dla każdej cechy w grupie wylicza z iloma innymi jest niezgodna
    # usuwa rulę, która jest niezgodna z największą liczbą rul w grupie
    # powtarza, aż w grupie nie będzie contradictionów
    g = _group_rules(rules)
    good_rules = []

    for k in g.keys():
        group = g[k]

        # remove rebel rules
        while True:
            n_group = []
            for ra in group:
                n_contradictions = len([1 for rb in group if ra.contradicts(rb)])
                n_group.append((ra, n_contradictions))

            ens = [n for r, n in n_group]
            if sum(ens) == 0:
                break

            group = [r for r, n in n_group if n < max(ens)]

        good_rules.extend(group)
    return good_rules


def rebel_rules_stats(rules, print_func=None):
    g = _group_rules(rules)

    n_condratictive_rules = 0
    for k in g.keys():
        for ra in g[k]:
            n_contradictions = len([1 for rb in g[k] if ra.contradicts(rb)])
            if n_contradictions != 0:
                n_condratictive_rules += 1

    if print_func is not None:
        print_func(f"Wszystkich reguł: {len(rules)}, z czego")
        print_func(f"{sum([len(g[k]) for k in g.keys() if len(g[k])>1])} reguł w {len(g.keys())} grupach*")
        print_func(f"{n_condratictive_rules} condratictive rules")
        print_func("* przez grupę rozumiemy conajmniej dwie rule")


def _group_rules(rules):
    # groups rules that can be compared by their
    # (r.ftr_idx, r.origin)
    # returns a dictionary (ftr_idx, origin): [rule1, ...]
    groups = defaultdict(list)
    for r in rules:
        groups[(r.ftr_idx, r.origin)].append(r)
    return groups
