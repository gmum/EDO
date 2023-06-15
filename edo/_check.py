import numpy as np


def assert_binary(a):
    assert set(np.unique(a)).issubset(set([0,1])), f"Values in `a` must be in {set([0, 1])}, are {np.unique(a)}."
    
    
def assert_strictly_positive_threshold(t):
    assert t > 0, f"`threshold` must be strictly positive, is {t}."


def validate_shapes(f_vals, s_vals, classes_order=None, class_index=None):
    assert len(f_vals.shape) == 1, f"`f_vals` must be 1-dimensional array, is {f_vals.shape}."
    assert f_vals.shape[0] == s_vals.shape[-1], f"Number of samples in `f_vals` and `s_vals` must be equal, is {f_vals.shape[0]} != {s_vals.shape[-1]}."
    
    # we don't make any assumption about the task...
    assert len(s_vals.shape) in (1, 2), f"`s_vals` must be 1- (regression) or 2-dimensional array (classification), is {s_vals.shape}."
    # ...unles classes_order/index is given 
    if classes_order is not None or class_index is not None:
        assert len(s_vals.shape) == 2, f"For classification `s_vals` must be 2-dimensional array, is {s_vals.shape}."
    
    if classes_order is not None:
        assert len(classes_order) == s_vals.shape[0], f"Length of `classes_order` must be equal to `s_vals.shape[0]`, {len(classes_order)} != {s_vals.shape[0]}."
    
    if class_index is not None:
        # negative indices start from -1 so we need to substract 1 to use "<" operator
        abs_class_idx = class_index if class_index>=0 else abs(class_index)-1
        assert  abs_class_idx < s_vals.shape[0], f"Insufficient number of classes. `s_vals.shape[0]`={s_vals.shape[0]}, `class_index`={class_index}."
        

def validate_index(i, max_val=None):
    assert isinstance(i, int) or isinstance(i, np.int_), TypeError(f"index must be integers, not {type(i)}.")
    assert i >= 0, ValueError(f"index must be nonnegative, is {i}.")
    if max_val is not None:
        assert i<max_val, ValueError(f"index must be smaller than {max_val}, is {i}.")

