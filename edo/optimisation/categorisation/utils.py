import numpy as np
from ..._check import assert_binary


def n_zeros_ones(a):
    """
    Calculate number of zeros and ones in a binary vector.
    :param a: numpy.array [samples]: a vector of binary values
    :return: (int, int): number of zeros, number of ones
    """
    assert_binary(a)
    ones = np.sum(a)
    return len(a) - ones, ones


def purity(zeros=None, ones=None, a=None):
    """
    Calculate purity given zeros and ones or a binary vector.
    :param zeros: int: number of zeros; default None
    :param ones: int: number of ones; default None
    :param a: numpy.array [samples]: a vector of binary values; default None
    :return: float: purity; or numpy.nan if the number of samples is 0
    """
    msg = f"Provide either `zeros` and `ones` or `a`. Given zeros: {zeros}, ones: {ones}, a: {a}."
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
    """
    Calculate majority value given zeros and ones or indicate that the purity is smaller than required.
    :param zeros: int: number of zeros
    :param ones: int: number of ones
    :param min_purity: float: minimal required purity; default: 0.5
    :return: Tuple[majority value: int] or (0, 1) if purity is smaller than required
                                                     or the number of ones and zeros are equal
    """
    if purity(zeros, ones) < min_purity or zeros == ones:
        return (0, 1)
    return (0,) if zeros > ones else (1,)
