import numpy as np
from . import HighImpactResult, Region
from .utils import n_zeros_ones, purity, majority
from ..._check import validate_shapes, assert_binary, assert_strictly_positive_threshold


def high_impact(f_vals, s_vals, gamma, metric='ratio'):
    params = {'gamma': gamma, 'metric': metric}

    allowed_metrics = ['absolute', 'ratio', 'purity']

    assert metric in allowed_metrics, f"metric must be one of {allowed_metrics} is {metric}."
    assert_strictly_positive_threshold(gamma)
    validate_shapes(f_vals, s_vals, classes_order=None)
    assert_binary(f_vals)

    high_loss_indices = s_vals <= -gamma
    high_gain_indices = s_vals >= gamma
    indices = [high_loss_indices, high_gain_indices]

    region_f_vals = tuple(f_vals[i] for i in indices)

    zeros_ones = tuple(n_zeros_ones(reg) for reg in region_f_vals)
    purities = tuple(purity(a=reg) for reg in region_f_vals)
    majorities = tuple(majority(n0, n1) for n0, n1 in zeros_ones)
    n_correct = tuple(max(n0, n1) for n0, n1 in zeros_ones)

    loss_region, gain_region = [Region(m, p, n, i, s, e) for m, p, n, i, s, e
                                in zip(majorities, purities, n_correct, indices, (-np.inf, gamma), (-gamma, np.inf))]

    n_samples = len(s_vals)
    n_hi_correct = np.sum(n_correct)
    n_hi_samples = sum(len(r) for r in region_f_vals)
    if metric == 'absolute':
        # n correctly classified in both high impact regions
        score = n_hi_correct
    elif metric == 'ratio':
        # percentage of correctly classified high impact samples among ALL samples
        score = n_hi_correct / n_samples
    elif metric == 'purity':
        # overall purity of high impact regions
        score = np.nan if n_hi_samples == 0 else n_hi_correct / n_hi_samples
    else:
        raise ValueError(f"metric must be one of {allowed_metrics} is {metric}.")

    return HighImpactResult(score, loss_region, gain_region, params)
