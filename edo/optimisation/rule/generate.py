import operator

from . import Rule
from ..categorisation import RandomRule
from .. import Goal, get_random_generator
from ... import Task, TASK_ERROR_MSG


def derive_well_separated_rules(feature, task):
    """
    Derive well-separated rules for feature. Rules for both maximisation and minimisation are derived.
    :param feature: Feature: feature for which the rules are derived
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :return: List[Rule]: list of the derived rules
    """
    rules = []
    info = feature.well_separated()
    ftr_params = {'origin': feature.s_vals_origin, 'feature_index': feature.ftr_index}

    if task == Task.CLASSIFICATION:
        for cls_idx, cls_equivalent_solutions in enumerate(info):
            # well_separated gives all optimal solutions
            for sol in cls_equivalent_solutions:
                left_reg, right_reg = sol.regions
                if left_reg.majority == 0 and right_reg.majority == 1:
                    setup = [('add', Goal.MAXIMISATION, operator.lt),
                             ('remove', Goal.MINIMISATION, operator.gt)]
                elif left_reg.majority == 1 and right_reg.majority == 0:
                    setup = [('add', Goal.MINIMISATION, operator.gt),
                             ('remove', Goal.MAXIMISATION, operator.lt)]
                else:
                    # this should never happen
                    msg = f"In well separated result left_reg.majority={left_reg.majority} and right_reg.majority={right_reg.majority}"
                    raise RuntimeError(msg)

                cls_params = {'class_index': cls_idx,
                              'class_name': feature._classes_order[cls_idx],
                              'criterion_reference_value': sol.thresholds}

                for (action, goal, relation) in setup:
                    individual_params = {'action': action, 'goal': goal,
                                         'criterion_relation': relation,
                                         'derivation': sol}

                    rules.append(Rule(**ftr_params, **cls_params, **individual_params))

    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return rules


def derive_high_impact_rules(feature, params, task):
    """
    Derive high impact rules for feature. Rules for both maximisation and minimisation are derived.
    :param feature: Feature: feature for which the rules are derived
    :param params: {'gamma':float, 'metric':str}: parameters of high impact rules
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :return: List[Rule]: list of the derived rules
    """
    rules = []
    info = feature.high_impact(**params)
    ftr_params = {'origin': feature.s_vals_origin,
                  'feature_index': feature.ftr_index}

    if task == Task.CLASSIFICATION:
        for cls_idx, cls_info in enumerate(info):
            ref_val = cls_info.params['gamma']
            loss_reg, gain_reg = cls_info.loss_region, cls_info.gain_region
            if loss_reg.majority == gain_reg.majority:
                continue
            elif loss_reg.majority == (0,) or gain_reg.majority == (1,):
                setup = [('add', Goal.MAXIMISATION, operator.lt, ref_val),
                         ('remove', Goal.MINIMISATION, operator.gt, -ref_val)]
            elif loss_reg.majority == (1,) or gain_reg.majority == (0,):
                setup = [('add', Goal.MINIMISATION, operator.gt, -ref_val),
                         ('remove', Goal.MAXIMISATION, operator.lt, ref_val)]
            else:
                # this should never happen
                msg = f"In high impact result loss_region.majority={loss_reg.majority} and gain_region.majority={gain_reg.majority}"
                raise RuntimeError(msg)

            cls_params = {'class_index': cls_idx,
                          'class_name': feature._classes_order[cls_idx]}

            for (action, goal, relation, rv) in setup:
                individual_params = {'action': action, 'goal': goal,
                                     'criterion_relation': relation,
                                     'criterion_reference_value': rv,
                                     'derivation': cls_info}

                rules.append(Rule(**ftr_params, **cls_params, **individual_params))

    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return rules


# # # # # # # # # # # # # #
# R A N D O M   R U L E S #
# # # # # # # # # # # # # #

def always_satisfied(a, b):
    """
    clever rules have criteria based on functions from operator module random rules have this
    :param a: anything
    :param b: anything
    :return: True
    """
    return True


def derive_random_rules_sample(feature, task):
    """
    Derive random rules for feature. Rules for both maximisation and minimisation are derived.
    :param feature: Feature: feature for which the rules are derived
    :param params: {'gamma':float, 'metric':str}: parameters of high impact rules
    :param task: Task: is the model used to calculate SHAP values a classifier or a regressor
    :return: List[Rule]: list of the derived rules
    """
    rng = get_random_generator()  # ensure reproducibility
    ftr_params = {'origin': feature.s_vals_origin, 'feature_index': feature.ftr_index}

    actions = ['add', 'remove']
    goals = [Goal.MAXIMISATION, Goal.MINIMISATION]

    individual_params = []
    if task == Task.CLASSIFICATION:
        for cls_idx in range(len(feature._classes_order)):
            action = rng.choice(actions, 1)[0]
            goal = rng.choice(goals, 1)[0]
            # criterion_reference_value doesn't matter because the relation is always_satisfied
            params = {'action': action, 'goal': goal, 'criterion_relation': always_satisfied,
                      'criterion_reference_value': 0,
                      'derivation': RandomRule(),
                      'class_index': cls_idx,
                      'class_name': feature._classes_order[cls_idx],
                      }
            individual_params.append(params)

    elif task == Task.REGRESSION:
        raise NotImplementedError  # Napisane na sucho, nigdy nie odpalone.
        action = rng.choice(actions, 1)[0]
        goal = rng.choice(goals, 1)[0]
        # criterion ref_point doesn't matter because the relation is always_satisfied
        params = {'action': action, 'goal': goal, 'criterion_relation': always_satisfied,
                  'criterion_reference_value': 0,
                  'derivation': RandomRule(),
                  'class_index': None,
                  'class_name': None,
                  }
        individual_params.append(params)
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    rules = [Rule(**ftr_params, **ind) for ind in individual_params]
    return rules
