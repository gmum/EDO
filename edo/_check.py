import numpy as np

from edo import Task


def assert_binary(a):
    assert set(np.unique(a)).issubset({0, 1}), f"Values in `a` must be in {{0, 1}}, are {np.unique(a)}."


def assert_strictly_positive_threshold(t):
    assert t > 0, f"`threshold` must be strictly positive, is {t}."


def validate_shapes(f_vals, s_vals, classes_order=None):
    """
    Are feature values and SHAP values of a given sample in a good shape?
    Is the shape of classes_order correct?
    :param f_vals: np.array: feature values
    :param s_vals: np.array: SHAP values
    :param classes_order: order of classes
    """
    assert len(f_vals.shape) == 1, f"`f_vals` must be 1-dimensional array, is {f_vals.shape}."
    assert f_vals.shape[0] == s_vals.shape[-1],\
        f"Number of samples in `f_vals` and `s_vals` must be equal but {f_vals.shape[0]} != {s_vals.shape[-1]}."

    # we don't make any assumptions about the task...
    assert len(s_vals.shape) in (1, 2),\
        f"`s_vals` must be 1- (regression) or 2-dimensional array (classification), is {s_vals.shape}."
    # ...unless classes_order is given
    if classes_order is not None:
        assert len(s_vals.shape) == 2, f"For classification `s_vals` must be 2-dimensional array, is {s_vals.shape}."
        assert len(classes_order) == s_vals.shape[0],\
            f"Length of `classes_order` must be equal to `s_vals.shape[0]` but {len(classes_order)} != {s_vals.shape[0]}."


def validate_index(i, size=None):
    """Is `i` a correct nonnegative index for an Iterable of such size?"""
    assert isinstance(i, int) or isinstance(i, np.int_), TypeError(f"index must be an integer, is {type(i)}.")
    assert i >= 0, ValueError(f"index must be nonnegative, is {i}.")
    if size is not None:
        assert i < size, ValueError(f"index must be smaller than {size}, is {i}.")


def validate_task(task, s_vals):
    # TODO: dwa razy sprawdzamy to samo, choose one
    assert (task == Task.CLASSIFICATION and len(s_vals.shape) == 2) or (task == Task.REGRESSION and len(s_vals.shape) == 1), f"`s_vals.shape` and task mismatch. `s_vals` must be 1- (regression) or 2-dimensional array (classification), is {s_vals.shape} and task {task}."
    assert (len(s_vals.shape), task) in [(1, Task.REGRESSION), (2, Task.CLASSIFICATION)], f"`s_vals.shape` and `task` mismatch. `s_vals` must be 1- (regression) or 2-dimensional array (classification), is {s_vals.shape} and task is {task}."
