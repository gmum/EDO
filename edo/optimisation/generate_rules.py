import itertools
import operator

from . import Goal, get_random_generator
from .rule import Rule
from .. import Task, TASK_ERROR_MSG
from ..shap_analysis.categorisation import RandomRule


def derive_well_separated_two_way_rules(ftr, task):
    rules = []
    info = ftr.well_separated(2)
    ftr_params = {'origin': ftr.s_vals_origin,
                  'feature_index': ftr.ftr_index}

    if task == Task.CLASSIFICATION:
        for cls_idx, cls_equivalent_solutions in enumerate(info):
            # well separated returns all optimal solutions
            for sol in cls_equivalent_solutions:
                left_reg, right_reg = sol.regions
                if left_reg.majority == 0 and right_reg.majority == 1:
                    setup = [('add', Goal.MAXIMISATION, operator.lt),
                             ('remove', Goal.MINIMISATION, operator.gt)]
                elif left_reg.majority == 1 and right_reg.majority == 0:
                    setup = [('add', Goal.MINIMISATION, operator.gt),
                             ('remove', Goal.MAXIMISATION, operator.lt)]
                else:
                    msg = f"In two-way well separated result left_reg.majority={left_reg.majority} and right_reg.majority={right_reg.majority}"
                    raise RuntimeError(msg)

                cls_params = {'class_index': cls_idx,
                              'class_name': ftr._classes_order[cls_idx],
                              'criterion_reference_point': sol.thresholds}

                for (action, goal, relation) in setup:
                    individual_params = {'action': action, 'goal': goal,
                                         'criterion_relation': relation,
                                         'derivation': (sol, goal)}

                    rules.append(Rule(**ftr_params, **cls_params, **individual_params))

    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return rules


def derive_high_impact_rules(ftr, params, task):
    rules = []
    info = ftr.high_impact(**params)
    ftr_params = {'origin': ftr.s_vals_origin,
                  'feature_index': ftr.ftr_index}

    if task == Task.CLASSIFICATION:
        for cls_idx, cls_info in enumerate(info):
            ref_val = cls_info.params['gamma']
            loss_reg, gain_reg = cls_info.loss_region, cls_info.gain_region
            if loss_reg.majority == gain_reg.majority:
                continue  # wersja uproszczona, nie bawimy się w to
            # poniższe zakomentowuję, bo włącza się, gdy choć jeden region jest 'inpure'
            # - jak rzeczywiście jest inpure, to możemy go zignorować i patrzeć tylko na ten drugi
            # - jeżeli jest pusty, to piszemy, że majority = (0, 1), a w sumie pusty nam nie przeszkadza
            ## elif len(loss_reg.majority) + len(gain_reg.majority) > 2:
            ## równa liczba sampli w którymś regionie, majority = (0, 1)
            ## continue  # wersja uproszczona, nie bawimy się w to
            elif loss_reg.majority == (0,) or gain_reg.majority == (1,):
                setup = [('add', Goal.MAXIMISATION, operator.lt, ref_val),
                         ('remove', Goal.MINIMISATION, operator.gt, -ref_val)]
            elif loss_reg.majority == (1,) or gain_reg.majority == (0,):
                setup = [('add', Goal.MINIMISATION, operator.gt, -ref_val),
                         ('remove', Goal.MAXIMISATION, operator.lt, ref_val)]
            else:
                msg = f"In high impact result loss_region.majority={loss_reg.majority} and gain_region.majority={gain_reg.majority}"
                raise RuntimeError(msg)

            cls_params = {'class_index': cls_idx,
                          'class_name': ftr._classes_order[cls_idx]}

            for (action, goal, relation, rv) in setup:
                individual_params = {'action': action, 'goal': goal,
                                     'criterion_relation': relation,
                                     'criterion_reference_point': rv,
                                     'derivation': (cls_info, goal)}

                rules.append(Rule(**ftr_params, **cls_params, **individual_params))

    elif task == Task.REGRESSION:
        raise NotImplementedError
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return rules


# # # # # # # # # # # # # #
# R A N D O M   R U L E S #
# # # # # # # # # # # # # #

# clever rules have criteria based on functions from operator
# random rules have this
def always_satisfied(a, b):
    return True


def derive_random_rules_all(ftr, task):
    # dla każdej cechoklasy wszystkie możliwe rule
    ftr_params = {'origin': ftr.s_vals_origin,
                  'feature_index': ftr.ftr_index}

    setup_params = []
    s_iter = itertools.product(['add', 'remove'],
                               [Goal.MAXIMISATION, Goal.MINIMISATION],
                               [always_satisfied, ])

    for (action, goal, relation) in s_iter:
        params = {'action': action, 'goal': goal,
                  'criterion_relation': relation,
                  'derivation': (RandomRule(), goal)}
        setup_params.append(params)

    individual_params = []
    if task == Task.CLASSIFICATION:
        for cls_idx in range(len(ftr._classes_order)):
            # criterion ref_point nie ma znaczenia
            # bo relation=always_satified
            params = {'class_index': cls_idx,
                      'class_name': ftr._classes_order[cls_idx],
                      'criterion_reference_point': 0}
            individual_params.append(params)

    elif task == Task.REGRESSION:
        raise NotImplementedError  # Napisane na sucho, nigdy nie odpalone.
        params = {'class_index': None, 'class_name': None,
                  'criterion_reference_point': 0}
        individual_params.append(params)
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    p_iter = itertools.product(individual_params, setup_params)
    rules = [Rule(**ftr_params, **ind, **stp) for ind, stp in p_iter]
    return rules


def derive_random_rules_sample(ftr, task):
    # jedna cechoklasa -> jedna rula

    rng = get_random_generator()
    ftr_params = {'origin': ftr.s_vals_origin,
                  'feature_index': ftr.ftr_index}

    actions = ['add', 'remove']
    goals = [Goal.MAXIMISATION, Goal.MINIMISATION]

    individual_params = []
    if task == Task.CLASSIFICATION:
        for cls_idx in range(len(ftr._classes_order)):
            # criterion ref_point nie ma znaczenia
            # bo relation=always_satified
            a = rng.choice(actions, 1)[0]
            g = rng.choice(goals, 1)[0]
            params = {'action': a, 'goal': g,
                      'criterion_relation': always_satisfied,
                      'derivation': (RandomRule(), g),
                      'class_index': cls_idx,
                      'class_name': ftr._classes_order[cls_idx],
                      'criterion_reference_point': 0}
            individual_params.append(params)

    elif task == Task.REGRESSION:
        raise NotImplementedError  # Napisane na sucho, nigdy nie odpalone.
        a = rng.choice(actions, 1)[0]
        g = rng.choice(goals, 1)[0]
        params = {'action': a, 'goal': g,
                  'criterion_relation': always_satisfied,
                  'derivation': (RandomRule(), g),
                  'class_index': None, 'class_name': None,
                  'criterion_reference_point': 0}
        individual_params.append(params)
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    rules = [Rule(**ftr_params, **ind) for ind in individual_params]
    return rules
