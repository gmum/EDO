import numpy as np
import pandas as pd

from collections import Counter

from .. import Task, TASK_ERROR_MSG, no_print
from .categorisation import SeparationResult, HighImpactResult, RandomRule
from .utils import _get_pred, get_predictions_before_after


def optimisation_stats(features, rules, samples, samples_for_evaluation, print_func=no_print):
    """
    Statistics of the optimisation procedure: number of features, samples, etc.
    :param features: indices of features that were available for optimisation
    :param rules: Iterable[Rule]: rules that were available for optimisation
    :param samples: Iterable[Sample]: samples that were available for optimisation
    :param samples_for_evaluation: Iterable[Sample]: samples that actually changed during optimisation
    :param print_func: a function to print with; default: edo.no_print (prints nothing)
    :return: dictionary with statistics
    """
    n_feats = len(features)
    n_samples = len(samples)
    n_samp_eval = len(samples_for_evaluation)

    print_func(f'#features: {n_feats}')
    print_func(f'#samples: {n_samples}')
    print_func(f'    #optimised: {n_samp_eval}')

    desc = {'n_features': n_feats,
            'n_samples_for_optimisation': n_samples,
            'n_evaluated_samples': n_samp_eval}

    desc.update(rule_stats(rules, print_func))
    desc.update(applied_changes_stats(samples, samples_for_evaluation, print_func))
    return desc


def rule_stats(rules, print_func=no_print):
    """
    Statistics of rules available during optimisation: number of rules in each category, action they perform, etc.
    :param rules: Iterable[Rule]: rules that were available for optimisation
    :param print_func: a function to print with; default: edo.no_print (prints nothing)
    :return: dictionary with statistics
    """
    n_rules = len(rules)
    ws_rules = len([r for r in rules if isinstance(r.derivation[0], SeparationResult)])
    hi_rules = len([r for r in rules if isinstance(r.derivation[0], HighImpactResult)])
    rnd_rules = len([r for r in rules if isinstance(r.derivation[0], RandomRule)])
    assert ws_rules + hi_rules + rnd_rules == n_rules

    target = Counter([(r.cls_name, r.goal) for r in rules])
    target = [(f'{goal.name} of class {cls}', target[cls, goal]) for cls, goal in target]
    target = dict(target)

    action = dict(Counter([r.action.desc for r in rules]))

    print_func(f'#rules: {n_rules}')
    print_func(f'    #well separated: {ws_rules}')
    print_func(f'    #high impact: {hi_rules}')
    print_func(f'    #random: {rnd_rules}')
    for t in target:
        print_func(f'    {t}: {target[t]}')
    print_func(f'    #action: {action}')

    desc = {'n_rules': n_rules, 'n_ws_rules': ws_rules, 'n_hi_rules': hi_rules, 'n_rnd_rules': rnd_rules,
            'n_rules_target': target, 'n_rules_action': action}
    return desc


def applied_changes_stats(samples, samples_for_evaluation, print_func=no_print):
    """
    Statistics of samples available for optimisation: number of samples, mean number of applied changes, etc.
    :param samples: Iterable[Sample]: samples that were available for optimisation
    :param samples_for_evaluation: Iterable[Sample]: samples that actually changed during optimisation
    :param print_func: a function to print with; default: edo.no_print (prints nothing)
    :return: dictionary with statistics
    """
    mean_all = mean_number_of_applied_rules(samples)
    mean_changed = mean_number_of_applied_rules(samples_for_evaluation)

    print_func('mean number of applied changes')
    print_func(f'among all: {mean_all}')
    print_func(f'among changed: {mean_changed}')

    return {'n_applied_rules_among_all': mean_all,
            'n_applied_rules_among_changed': mean_changed}


def mean_number_of_applied_rules(samples):
    n_applied_rules = [s.number_of_applied_changes() for s in samples]
    return np.mean(n_applied_rules)


def evaluate_stability_optimisation(samples, model, task, print_func=no_print):
    """
    Calculate optimisation metrics for samples using model. This function is specific for optimisation of metabolic
    stability, i.e. it assumes that probability of low stability class (0) is minimised and probability of high
    stability class (2) is maximised.
    :param samples: Iterable[Sample]: samples to evaluate
    :param model: sklearn-like model to evaluate with
    :param task: Task: is the model classifier of regressor
    :param print_func: a function to print with; default: edo.no_print (prints nothing)
    :return: a dictionary with scores if task is regression or two dictionaries if the task is classification
    """
    if len(samples) == 0:
        if task == Task.CLASSIFICATION:
            return {}, {}
        elif task == Task.REGRESSION:
            return {}
        else:
            raise ValueError(TASK_ERROR_MSG(task))

    before, after = get_predictions_before_after(samples, model, task)
    if task == Task.CLASSIFICATION:
        # NOTE: we assume that classes_order == [0, 1, 2]
        # minimisation of probability of low stability class
        should_be_lower = after[:, 0]
        should_be_higher = before[:, 0]
        scores_class_unstable = calculate_optimisation_metrics(before, after, should_be_lower, should_be_higher, task,
                                                               print_func)

        # maximisation of probability of high stability class
        should_be_lower = before[:, 2]
        should_be_higher = after[:, 2]
        scores_class_stable = calculate_optimisation_metrics(before, after, should_be_lower, should_be_higher, task,
                                                             print_func)

        return scores_class_unstable, scores_class_stable
    elif task == Task.REGRESSION:
        scores = calculate_optimisation_metrics(before, after, before, after, task, print_func)
        return scores
    else:
        raise ValueError(TASK_ERROR_MSG(task))


def calculate_optimisation_metrics(before, after, should_be_lower, should_be_higher, task, print_func=no_print):
    """
    Calculate success rate, mean change and mean class jump
    :param before: predictions before optimisation
    :param after: predictions after optimisation
    :param should_be_lower: predictions that are supposed to have lower values
    :param should_be_higher: predictions that are supposed to have higher values
    :param task: Task: are the predictions calculated by classifier of regressor
    :param print_func: a function to print with; default: edo.no_print (prints nothing)
    :return:  dictionary with statistics
    """
    sr = success_rate(should_be_lower, should_be_higher)
    mc = mean_change(should_be_lower, should_be_higher)
    scores = {'success_rate': sr, 'mean_change': mc}
    print_func(f'success rate: {sr}')
    print_func(f'mean change: {mc}')

    if task == Task.CLASSIFICATION:
        mcj = mean_class_jump(before, after)
        scores.update({'mean_class_jump': mcj})
        print_func(f'mean class jump: {mcj}')

    return scores


def success_rate(smaller, bigger):
    """
    How many times values that were supposed to get smaller got smaller divided by the number of samples.
    :param smaller: values that are supposed to be smaller
    :param bigger: values that are supposed to be bigger
    :return: float: success rate
    """
    assert smaller.shape == bigger.shape, f'Shape mismatch {smaller.shape} != {bigger.shape}'
    return np.sum(smaller < bigger) / len(smaller)


def mean_change(smaller, bigger):
    """
    Mean change in the right direction.
    :param smaller: values that are supposed to be smaller
    :param bigger: values that are supposed to be bigger
    :return: float: mean change
    """
    return np.mean(bigger - smaller)


def mean_class_jump(before, after):
    """
    Mean class increase i.e. prediction was changed to a class with a higher index.
    :param before: predictions before optimisation
    :param after: predictions after optimisation
    :return: float: mean class jump
    """
    # NOTE: we assume that classes_order == [0, 1, 2]
    class_before = np.argmax(before, axis=1)
    class_after = np.argmax(after, axis=1)
    return np.mean(class_after - class_before)


def get_history(samples, model, task):
    """
    Create pandas.DataFrame in which each row describes the prediction change introduced by applying a single rule to
    a sample. This function is specific for optimisation of metabolic stability, i.e. it assumes that the classes order
    is low stability class (0), medium stability class (1), high stability class (2).
    :param samples: Iterable[Sample]: samples
    :param model: sklearn-like model to calculate predictions
    :param task: Task: is the model classifier of regressor
    :return: pandas.DataFrame with prediction change introduced by each rule.
    """
    initial_preds = _get_pred(np.array([s.original_f_vals for s in samples]), model, task)
    results = []
    for s, pred_before in zip(samples, initial_preds):
        preds_history = _get_pred(np.array([entry.f_vals for entry in s.history]), model, task)

        for entry, pred_after in zip(s.history, preds_history):
            entry_desc = {'smiles': s.smiles, 'success': entry.success}
            entry_desc.update(entry.rule.as_dict(compact=True))

            if entry.success is False:
                results.append(entry_desc)
                continue

            pred_change = pred_after - pred_before
            desc = {'pred_change': pred_change}
            if task == Task.CLASSIFICATION:
                # NOTE: we assume that classes_order == [0, 1, 2]
                low, med, high = pred_change.flatten()
                desc = {'pred_change_low': low, 'pred_change_med': med, 'pred_change_high': high}

            entry_desc.update(desc)
            results.append(entry_desc)

            pred_before = pred_after

    df = pd.DataFrame.from_dict(results)
    return df
