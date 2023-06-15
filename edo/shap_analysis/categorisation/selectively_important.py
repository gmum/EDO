import numpy as np
from . import SelectivelyImportantResult, Region
from .utils import n_zeros_ones, purity, majority
from ..._check import validate_shapes, assert_binary, assert_strictly_positive_threshold


def selectively_important(f_vals, s_vals, miu, metric='ratio'):
    params = {'miu': miu, 'metric': metric}
    
    assert metric in ['ratio', 'absolute'], f"metric must be `absolute` or `ratio` is {metric}."
    assert_strictly_positive_threshold(miu)
    validate_shapes(f_vals, s_vals, classes_order=None)
    assert_binary(f_vals)
    
    negative_importance_indices = s_vals <= -miu
    positive_importance_indices = s_vals >= miu
    global_importance_indices = np.abs(s_vals) >= miu
    indices = [negative_importance_indices, positive_importance_indices, global_importance_indices]
    
    zeros_ones = tuple(n_zeros_ones(f_vals[i]) for i in indices)
    majorities = tuple(majority(zeros, ones) for zeros, ones in zeros_ones)
    purities = tuple(purity(zeros, ones) for zeros, ones in zeros_ones)
    n_correct = tuple(max(zeros, ones) for zeros, ones in zeros_ones)
    
    negative_region, positive_region, global_region = tuple(Region(m, p, n, i, s, e) for m, p, n, i, s, e
                                             in zip(majorities, purities,
                                                    n_correct, indices,
                                                    [-np.inf, miu, None],
                                                    [-miu, np.inf, None]
                                                    )
                                            )
    
    if metric=='absolute':
        score = global_region.n_correct
    elif metric=='ratio':
        score = global_region.purity
    else:
        raise ValueError(f"metric must be `absolute` or `ratio` is {metric}.")
    
    return SelectivelyImportantResult(score, negative_region, positive_region, global_region, params)
