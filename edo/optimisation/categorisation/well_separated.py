import numpy as np

from . import SeparationResult, Region
from .utils import purity
from ..._check import validate_shapes, assert_binary


def _make_regions(majority_vals, score_purity_n, indices, thresholds):
    return tuple(Region(m, p, n, i, s, e)
                 for m, (_, p, n), i, s, e in zip(majority_vals, score_purity_n, indices, [-np.inf, ] + thresholds,
                                                  thresholds + [np.inf, ]))


def two_way_separation(f_vals, s_vals, min_purity=None):
    """
    Describe feature as well-separated.
    :param f_vals: numpy.array [samples]: matrix of binary feature values
    :param s_vals: numpy.array [samples]: matrix of SHAP values
    :param min_purity: None: currently not used, might be implemented in the future, must be None; default: None
    :return: List[SeparationResult]: all optimal solutions
    """
    params = {'n_groups': 2, 'min_purity': min_purity}

    assert min_purity is None, f"`min_purity` must be None is {min_purity}"
    validate_shapes(f_vals, s_vals, classes_order=None)
    assert_binary(f_vals)

    def _majority_values(l0, l1, r0, r1):
        # l0 - left zeros, r1 - right ones, ...
        if l0 + r1 > l1 + r0:
            values = (0, 1)
        elif l0 + r1 < l1 + r0:
            values = (1, 0)
        else:
            # equal number of ones and zeros on both sides
            values = (None, None)
        return values

    def _n_correct_per_cluster(l0, l1, r0, r1):
        if l0 + r1 > l1 + r0:
            return (l0, r1)
        else:
            return (l1, r0)

    sorted_indices = np.argsort(s_vals)
    n_samples = len(sorted_indices)

    # initialise variables
    n_ones, n_zeros = np.sum(f_vals), len(f_vals) - np.sum(f_vals)
    assert n_ones + n_zeros == len(f_vals)

    # initialise with values corresponding to such threshold that all samples are in gain region and loss region is empty
    best_thresholds = [-1]  # index of the right-most sample in the loss region
    maj_values = [_majority_values(0, 0, n_zeros, n_ones), ]
    purities = [(purity(0, 0), purity(n_zeros, n_ones)), ]
    max_correct = max(n_ones, n_zeros)
    n_correct_per_cluster = [(0, max_correct), ]

    j = 0                 # index of the right-most sample in the loss region
    while j < n_samples:  # check threshold between each two samples
        while j + 1 < n_samples and s_vals[sorted_indices[j]] == s_vals[sorted_indices[j + 1]]:
            # two samples can have equal SHAP values, there can be no threshold between them
            j += 1

        left_ones = np.sum(f_vals[sorted_indices[:j + 1]])
        left_zeros = j + 1 - left_ones
        right_ones = np.sum(f_vals[sorted_indices[j + 1:]])
        right_zeros = len(f_vals) - j - 1 - right_ones

        assert left_zeros + right_zeros == n_zeros, AssertionError(f'{n_zeros} != {left_zeros} + {right_zeros}')
        assert left_ones + right_ones == n_ones, AssertionError(f'{n_ones} != {left_ones} + {right_ones}')

        n_correct = max(left_zeros + right_ones, left_ones + right_zeros)

        if n_correct == max_correct:  # is this solution as good as the current optimal?
            best_thresholds.append(sorted_indices[j])
            maj_values.append(_majority_values(left_zeros, left_ones, right_zeros, right_ones))
            purities.append((purity(left_zeros, left_ones), purity(right_zeros, right_ones)))
            n_correct_per_cluster.append((_n_correct_per_cluster(left_zeros, left_ones, right_zeros, right_ones)))

        elif n_correct > max_correct:  # is this solution better than the current optimal?
            best_thresholds = [sorted_indices[j], ]
            maj_values = [_majority_values(left_zeros, left_ones, right_zeros, right_ones), ]
            purities = [(purity(left_zeros, left_ones), purity(right_zeros, right_ones))]
            n_correct_per_cluster = [(_n_correct_per_cluster(left_zeros, left_ones, right_zeros, right_ones))]
            max_correct = n_correct

        j += 1

    global_purity = max_correct / n_samples
    if -1 in best_thresholds:  # putting all samples in the gain region is optimal
        best_thresholds = np.hstack(([-np.inf], s_vals[best_thresholds[1:]]))  # set the correct threshold value
    else:
        best_thresholds = s_vals[best_thresholds]

    return [SeparationResult(global_purity, t,
                             [Region(m[0], p[0], n[0], None, -np.inf, t),
                              Region(m[1], p[1], n[1], None, t, np.inf)],
                             params)
            for t, m, p, n in zip(best_thresholds, maj_values, purities, n_correct_per_cluster)]
