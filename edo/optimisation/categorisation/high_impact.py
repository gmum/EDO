import numpy as np
from . import HighImpactResult, Region
from .utils import n_zeros_ones, purity, majority
from ..._check import validate_shapes, assert_binary, assert_strictly_positive_threshold


def high_impact(f_vals, s_vals, gamma, metric='ratio'):
    """
    Describe feature as high impact.
    :param f_vals: numpy.array [samples]: matrix of binary feature values
    :param s_vals: numpy.array [samples]: matrix of SHAP values
    :param gamma: float: minimal SHAP value of high impact samples, must be positive
    :param metric: str: which metric to use to calculate the score, must be `absolute`, `ratio` or `purity`;
                        default: `ratio`
    :return: HighImpactResult
    """
    params = {'gamma': gamma, 'metric': metric}

    allowed_metrics = ['absolute', 'ratio', 'purity']
    assert metric in allowed_metrics, f"metric must be one of {allowed_metrics} is {metric}."
    assert_strictly_positive_threshold(gamma)
    validate_shapes(f_vals, s_vals, classes_order=None)
    assert_binary(f_vals)

    high_loss_indices = s_vals <= -gamma
    high_gain_indices = s_vals >= gamma
    indices = [high_loss_indices, high_gain_indices]   # indices of samples in each region
    region_f_vals = tuple(f_vals[i] for i in indices)  # feature values of samples in each region

    zeros_ones = tuple(n_zeros_ones(reg) for reg in region_f_vals)
    purities = tuple(purity(a=reg) for reg in region_f_vals)
    majorities = tuple(majority(n0, n1) for n0, n1 in zeros_ones)
    n_correct = tuple(max(n0, n1) for n0, n1 in zeros_ones)

    loss_region, gain_region = [Region(m, p, n, i, s, e) for m, p, n, i, s, e
                                in zip(majorities, purities, n_correct, indices, (-np.inf, gamma), (-gamma, np.inf))]

    # calculate the score
    n_samples = len(s_vals)                            # number of all samples
    n_hi_samples = sum(len(r) for r in region_f_vals)  # number of samples in each region
    n_hi_correct = np.sum(n_correct)                   # number of samples correctly assigned to each region
    if metric == 'absolute':
        score = n_hi_correct              # number of correctly assigned high impact samples
    elif metric == 'ratio':
        score = n_hi_correct / n_samples  # percentage of correctly assigned high impact samples among ALL samples
    elif metric == 'purity':
        # overall purity of high impact regions calculated as the percentage of correctly assigned high impact samples
        # among high impact samples
        score = np.nan if n_hi_samples == 0 else n_hi_correct / n_hi_samples
    else:
        raise ValueError(f"metric must be one of {allowed_metrics} is {metric}.")

    return HighImpactResult(score, loss_region, gain_region, params)
