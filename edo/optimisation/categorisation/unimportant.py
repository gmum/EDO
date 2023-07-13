import numpy as np
from . import UnimportantResult
from ..._check import validate_shapes, assert_strictly_positive_threshold


def unimportant(f_vals, s_vals, niu, metric='ratio'):
    """
    Describe feature as unimportant.
    :param f_vals: not used; present only to assure consistency with other functions in the module
    :param s_vals: numpy.array [samples]: matrix of SHAP values
    :param niu: float: maximal SHAP value of unimportant samples, must be positive
    :param metric: str: which metric to use to calculate the score, must be `ratio` or `absolute`; default: `ratio`
    :return: UnimportantResult
    """
    params = {'niu': niu, 'metric': metric}

    assert metric in ['ratio', 'absolute'], f"metric must be `absolute` or `ratio` is {metric}."
    assert_strictly_positive_threshold(niu)
    validate_shapes(f_vals, s_vals, classes_order=None)

    unimportant_indices = np.abs(s_vals) < niu
    n_unimportant = np.sum(unimportant_indices)
    if metric == 'absolute':
        score = n_unimportant
    elif metric == 'ratio':
        score = n_unimportant / len(s_vals)
    else:
        raise ValueError(f"metric must be `absolute` or `ratio` is {metric}.")

    return UnimportantResult(score, params)
