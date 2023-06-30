import numpy as np
from ..._check import assert_binary


def n_zeros_ones(a):
    assert_binary(a)
    ones = np.sum(a)
    return len(a) - ones, ones


def purity(zeros=None, ones=None, a=None):
    msg = f"Provide either zeros and ones or a. Is zeros: {zeros}, ones: {ones}, a: {a}."
    assert (zeros is None) == (ones is None), msg
    assert (zeros is None) != (a is None), msg

    if a is not None:
        assert_binary(a)
        zeros = len(a[a == 0])
        ones = len(a[a == 1])

    n = zeros + ones
    p = max(zeros, ones) / n if n != 0 else np.nan

    return p


def majority(zeros, ones, min_purity=0.5):
    if purity(zeros, ones) < min_purity:
        return (0, 1)

    if zeros > ones:
        return (0,)
    elif zeros < ones:
        return (1,)
    else:
        return (0, 1)
