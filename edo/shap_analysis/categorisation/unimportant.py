import numpy as np
from . import UnimportantResult
from ..._check import validate_shapes, assert_strictly_positive_threshold


def unimportant(f_vals, s_vals, miu, metric='ratio'):
    # f_vals are not used; they're there for consistency with other functions
    params = {'miu': miu, 'metric': metric}

    assert metric in ['ratio', 'absolute'], f"metric must be `absolute` or `ratio` is {metric}."
    assert_strictly_positive_threshold(miu)
    validate_shapes(f_vals, s_vals, classes_order=None)

    unimportant_indices = np.abs(s_vals) < miu
    n_unimportant = np.sum(unimportant_indices)
    if metric == 'absolute':
        score = n_unimportant
    elif metric == 'ratio':
        score = n_unimportant / len(s_vals)
    else:
        raise ValueError(f"metric must be `absolute` or `ratio` is {metric}.")

    return UnimportantResult(score, params)
