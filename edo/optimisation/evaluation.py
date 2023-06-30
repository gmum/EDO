import numpy as np
import pandas as pd

from collections import Counter

from .. import Task, TASK_ERROR_MSG
from .categorisation import SeparationResult, HighImpactResult, RandomRule
from .utils import get_predictions_before_after, _get_pred_single_sample


def optimisation_stats(features, rules, samples, samples_for_evaluation, print_func=None):
    # statistics of optimisation procedure
    # features - indices of features that were used
    # rules - rules that were used
    # samples - samples to optimise
    # samples_for_evaluation - samples to evaluate
    # (ex. a subset of `samples` such that only samples which actually undergone some change are present)
    n_feats = len(features)
    n_samples = len(samples)
    n_samp_eval = len(samples_for_evaluation)

    if print_func is not None:
        print_func(f'#features: {n_feats}')
        print_func(f'#samples: {n_samples}')
        print_func(f'    #optimised: {n_samp_eval}')

    desc = {'n_features': n_feats,
            'n_samples_for_optimisation': n_samples,
            'n_evaluated_samples': n_samp_eval}

    desc.update(rule_stats(rules, print_func))
    desc.update(applied_changes_stats(samples, samples_for_evaluation, print_func))
    return desc


def rule_stats(rules, print_func=None):
    # statistics on rules used for optimisation
    n_rules = len(rules)
    ws_rules = len([r for r in rules if isinstance(r.derivation[0], SeparationResult)])
    hi_rules = len([r for r in rules if isinstance(r.derivation[0], HighImpactResult)])
    rnd_rules = len([r for r in rules if isinstance(r.derivation[0], RandomRule)])
    assert ws_rules + hi_rules + rnd_rules == n_rules

    target = Counter([(r.cls_name, r.goal) for r in rules])
    target = [(f'{goal.name} of class {cls}', target[cls, goal]) for cls, goal in target]
    target = dict(target)

    action = dict(Counter([r.action.desc for r in rules]))

    if print_func is not None:
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


def applied_changes_stats(samples, samples_for_evaluation, print_func=None):
    # statistics on changes applied during optimisation
    # samples - samples to optimise
    # samples_for_evaluation - samples to evaluate
    mean_all = mean_number_of_applied_rules(samples)
    mean_changed = mean_number_of_applied_rules(samples_for_evaluation)

    if print_func is not None:
        print_func('mean number of applied changes')
        print_func(f'among all: {mean_all}')
        print_func(f'among changed: {mean_changed}')

    return {'n_applied_rules_among_all': mean_all,
            'n_applied_rules_among_changed': mean_changed}


def mean_number_of_applied_rules(samples):
    n_applied_rules = [s.number_of_applied_changes() for s in samples]
    return np.mean(n_applied_rules)


def evaluate_optimisation(samples, model, task, print_func=None):
    # how effective was optimisation
    if len(samples)==0:
        if task == Task.CLASSIFICATION:
            return {}, {}
        elif task == Task.REGRESSION:
            return {}
        else:
            raise ValueError(TASK_ERROR_MSG(task))

    # TODO: uwaga, zakładamy, że goal to maksymalizacja
    before, after = get_predictions_before_after(samples, model, task)
    if task == Task.CLASSIFICATION:
        # TODO: zakładamy, że klasę 0 chcemy zminimalizować, a klasę 2 zmaksymalizować
        # TODO uwaga! ten kawałek zakłada, że classes_order = [0, 1, 2]

        # chcemy, aby prawdopodobieństwo unstable zmalało
        should_be_lower = after[:, 0]  # klasa unstable po optymalizacji powinna być mniejsza
        should_be_higher = before[:, 0]  # klasa unstable przed optymalizacją powinna być większa
        scores_class_unstable = _calculate_stats(before, after, should_be_lower, should_be_higher, task, print_func)

        # ponadto chcemy, aby prawdopodobieństwo klasy stable się zwiększyło
        should_be_lower = before[:, 2]  # klasa stable przed optymalizacją powinna być mniejsza
        should_be_higher = after[:, 2]  # klasa stable po optymalizacji powinna być większa
        scores_class_stable = _calculate_stats(before, after, should_be_lower, should_be_higher, task, print_func)

        return scores_class_unstable, scores_class_stable
    elif task == Task.REGRESSION:
        scores = _calculate_stats(before, after, before, after, task, print_func)
        return scores
    else:
        raise ValueError(TASK_ERROR_MSG(task))


def _calculate_stats(before, after, should_be_lower, should_be_higher, task, print_func=None):
    sr = success_rate(should_be_lower, should_be_higher)
    mc = mean_change(should_be_lower, should_be_higher)
    scores = {'succes_rate': sr, 'mean_change': mc}
    if print_func is not None:
        print_func(f'succes rate: {sr}')
        print_func(f'mean change: {mc}')

    if task == Task.CLASSIFICATION:
        mcj = mean_class_jump(before, after)
        scores.update({'mean_class_jump': mcj})
        if print_func is not None:
            print_func(f'mean class jump: {mcj}')

    return scores


def success_rate(smaller, bigger):
    # ile razy zmalało, to co miało zmaleć
    # TODO: na razie wystarczy, że zmaleje o epsilon, ale możnaby ustawić jakiś threshold
    assert smaller.shape == bigger.shape, 'Shape mismatch'
    return np.sum(smaller < bigger) / len(smaller)


def mean_change(smaller, bigger):
    # o ile średnio zmalało, to co miało zmaleć
    return np.mean(bigger - smaller)


def mean_class_jump(before, after):
    # średni skok w dobrą stronę
    # TODO uwaga! ten kawałek zakłada, że classes_order = [0, 1, 2]
    # TODO: zakładamy, że klasę 0 chcemy zminimalizować, a klasę 2 zmaksymalizować
    class_before = np.argmax(before, axis=1)
    class_after = np.argmax(after, axis=1)
    return np.mean(class_after - class_before)


# TODO: robienie tego per sample i per wpis jest potencjalnie powolne!
def evaluate_history(samples, model, task):
    # przechodzi przez historię każdego sampla i sprawdza wpływ każdej kolejnej zmiany
    # (podstawowa ewaluacja bierze tylko oryginalny związek i ostateczny)
    results = []
    for s in samples:
        preds_before = _get_pred_single_sample(s.original_f_vals, model, task)

        for entry in s.history:
            entry_desc = {'smiles': s.smiles, 'success': entry.success}
            entry_desc.update(entry.rule.as_dict(compact=True))

            if entry.success is not True:
                results.append(entry_desc)
                continue

            preds_after = _get_pred_single_sample(entry.f_vals, model, task)

            # TODO: poniekąd zakładamy, że goal to maksymalizacja
            pred_change = preds_after - preds_before  # o ile wzrosło
            desc = {'pred_change': pred_change}
            if task == Task.CLASSIFICATION:
                # TODO uwaga! ten kawałek zakłada, że classes_order = [0, 1, 2]
                low, med, high = pred_change.flatten()
                desc = {'pred_change_low': low,
                        'pred_change_med': med,
                        'pred_change_high': high}

            entry_desc.update(desc)
            results.append(entry_desc)

            preds_before = preds_after

    df = pd.DataFrame.from_dict(results)
    return df
