import numpy as np
from functools import reduce
from . import SeparationResult, Region
from .utils import n_zeros_ones, purity, majority

from ..._check import validate_shapes, assert_binary


def _make_regions(majority_vals, score_purity_n, indices, thresholds):
    return tuple(Region(m, p, n, i, s, e) 
            for m, (_, p, n), i, s, e in zip(majority_vals,
                                             score_purity_n,
                                             indices,
                                             [-np.inf, ]+thresholds,
                                             thresholds+[np.inf, ]
                                            ))


def three_way_separation(f_vals, s_vals, min_purity=0.5):
    params = {'min_purity': min_purity}

    validate_shapes(f_vals, s_vals, classes_order=None)
    assert_binary(f_vals)
    
    def _score3(zeros, ones):
        p = purity(zeros, ones)
        n_correctly = max(zeros, ones)
        return p*n_correctly, p, n_correctly
        
    def _score_aglomerator(*args):
        return np.mean(args)  # srednia
        
    
    sorted_indices = np.argsort(s_vals)
    n_samples = len(s_vals)
    best_score = -1
    thresholds,  = [(-1, -1)]
    regions = []
    #purities, majorities, n_correct = [(-1, -1, -1)], [(-1, -1, -1)], [(-1, -1, -1)]
    
    # left/right is the index of the right_most sample in the interval
    # intervals cannot be empty
    for left in range(n_samples-2):
        # equal shap values on both sides of the threshold?
        if s_vals[sorted_indices[left]] == s_vals[sorted_indices[left+1]]:
            continue
            
        for right in range(1+left, n_samples-1):
            # equal shap values on both sides of the threshold?
            if s_vals[sorted_indices[right]] == s_vals[sorted_indices[right+1]]:
                continue
            
            l_indices = sorted_indices[:left+1]
            m_indices = sorted_indices[left+1:right+1]
            r_indices = sorted_indices[right+1:]
            indices = (l_indices, m_indices, r_indices)
            
            zeros_ones = tuple(n_zeros_ones(f_vals[i]) for i in indices)
            
            majority_vals = tuple(majority(n0, n1, min_purity) for n0, n1 in zeros_ones)
            # two neighbouring intervals cannot have the same label
            if majority_vals[0]==majority_vals[1] or majority_vals[1]==majority_vals[2]:
                continue
            
            score_purity_n = tuple(_score3(n0, n1) for n0, n1 in zeros_ones)
            all_score = _score_aglomerator(*[rec[0] for rec in score_purity_n])

            if all_score == best_score:
                thresholds.append((s_vals[sorted_indices[left]], s_vals[sorted_indices[right]]))
                regions.append(_make_regions(majority_vals, score_purity_n,
                                             indices, list(thresholds[-1])))
            elif all_score > best_score:
                best_score = all_score
                thresholds = [(s_vals[sorted_indices[left]], s_vals[sorted_indices[right]])]
                regions = [_make_regions(majority_vals, score_purity_n,
                                         indices, list(thresholds[-1])), ]

    return [SeparationResult(best_score, t, r, params) for t, r in zip(thresholds, regions)]



def two_way_separation(f_vals, s_vals):
    """
    feature_values - representation, 1-dimensional array
    shaps - shap values, 1-dimensional array
    """
    params = {}
    
    validate_shapes(f_vals, s_vals, classes_order=None)
    assert_binary(f_vals)

    def _majority_values(l0, l1, r0, r1):
        # l0 - left zeros, r1 - right ones, ...
        if l0+r1 > l1+r0:
            values = (0, 1)
        elif l0+r1 < l1+r0:
            values = (1, 0)
        else:
            # equal number of ones and zeros on both sides
            values = (None, None)
        return values
    
    def _n_correct_per_cluster(l0, l1, r0, r1):
        if l0+r1 > l1 + r0:
            return (l0, r1)
        else:
            return(l1, r0)
    
    sorted_indices = np.argsort(s_vals)
    n_samples = len(sorted_indices)
    
    # initialise variables
    n_ones, n_zeros = np.sum(f_vals), len(f_vals)-np.sum(f_vals)
    assert n_ones + n_zeros == len(f_vals)

    best_thresholds = [-1]
    maj_values = [_majority_values(0, 0, n_zeros, n_ones), ]
    purities = [(purity(0, 0), purity(n_zeros, n_ones)), ]
    max_correct = max(n_ones, n_zeros)
    n_correct_per_cluster = [(0, max_correct), ]

    j = 0
    while j <= n_samples-1:
        while j+1 <= n_samples-1 and s_vals[sorted_indices[j]] == s_vals[sorted_indices[j+1]]:
            j+=1
        
        left_ones = np.sum(f_vals[sorted_indices[:j+1]])
        left_zeros = j+1-left_ones
        right_ones = np.sum(f_vals[sorted_indices[j+1:]])
        right_zeros = len(f_vals)-j-1-right_ones
        
        assert left_zeros + right_zeros == n_zeros, AssertionError(f'{n_zeros} != {left_zeros} + {right_zeros}')
        assert left_ones + right_ones == n_ones, AssertionError(f'{n_ones} != {left_ones} + {right_ones}')

        n_correct = max(left_zeros+right_ones, left_ones+right_zeros)

        # did it get better?
        if n_correct == max_correct:
            best_thresholds.append(sorted_indices[j])
            maj_values.append(_majority_values(left_zeros, left_ones, right_zeros, right_ones))
            purities.append((purity(left_zeros, left_ones), purity(right_zeros, right_ones)))
            n_correct_per_cluster.append((_n_correct_per_cluster(left_zeros, left_ones, right_zeros, right_ones)))
        
        elif n_correct > max_correct:
            best_thresholds = [sorted_indices[j], ]
            maj_values = [_majority_values(left_zeros, left_ones, right_zeros, right_ones), ]
            purities = [(purity(left_zeros, left_ones), purity(right_zeros, right_ones)) ]
            n_correct_per_cluster = [(_n_correct_per_cluster(left_zeros, left_ones, right_zeros, right_ones))]
            max_correct = n_correct
        
        j+=1
    
    global_purity = max_correct/n_samples
    # best_thresholds: indeks ostatniego związku po lewej
    if -1 in best_thresholds:
        best_thresholds = np.hstack(([-np.inf], s_vals[best_thresholds[1:]]))
    else:
        best_thresholds = s_vals[best_thresholds]
    
    return [SeparationResult(global_purity, t,
                             [Region(m[0], p[0], n[0], None, -np.inf, t),
                              Region(m[1], p[1], n[1], None, t, np.inf)],
                             params)
            for t, m, p, n in zip(best_thresholds, maj_values, purities, n_correct_per_cluster)]
