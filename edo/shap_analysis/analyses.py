import os.path as osp
import json

import numpy as np
import pandas as pd

from ..config import utils_section
from .. import Task


def tanimoto_similarity(a, b):
    """Tanimoto similarity for binary fingerprints.
    a: matrix of shape (samples, fingerprint lenght)
    b: a single fingerprint
    The other way around should work as well.
    """
    et = np.sum(np.logical_and(a>0, b>0), axis=1)
    vel = np.sum(np.logical_or(a>0, b>0), axis=1)
    return et/vel


def get_n_most_important_features(data_dir, shap_vals, repr_cfg, n=9):
    """
    Return SMARTS of the n most important features based on SHAP values.
    """

    indices = list(reversed(list(np.abs(shap_vals).argsort())[-n:]))

    representation = repr_cfg[utils_section]["fingerprint"]
    if 'krfp' == representation:
        df = pd.read_csv(osp.join(data_dir, 'list_of_features_KlekFP.csv'), sep='\t', index_col=0)
    elif 'maccs' == representation:
        df = pd.read_csv(osp.join(data_dir, 'list_of_features_MACCSFP.csv'), sep='\t', index_col=0)
        #df['explanation'] = df.apply(lambda row: row.explanation.split("'")[1], axis=1)  # workaround
        df = df.drop([col for col in df.columns if col != 'explanation'], axis=1)  # these columns contain only NaNs
    else:
        raise ValueError(f"Unknown representation: {representation}. Known representations are `krfp` and `maccs`.")

    df_dict = df.loc[indices].to_dict('index')
    df_dict['order'] = [int(i) for i in indices]  # JSON cannot int64
    return json.dumps(df_dict)


def find_optimal_separation_point(shaps, X, feature_order, feature_index, task, class_index=None):
    """
    shaps - shap values,
    X - representation,
    features - features order,
    feature_index - name of feature for which the analysis is done,
    task,
    class_index - if task is classification, then for which class should the analysis be done
    
    """
    feat_idx = feature_order.index(feature_index)  # indeks cechy o nazwie "feature_index"
    this_values = X[:, feat_idx]
    assert set(this_values) == set([0, 1]), AssertionError(f"{set(this_values)} != {0, 1}")
    
    if task == Task.CLASSIFICATION:
        this_shaps = shaps[class_index, :, feat_idx]
    elif task == Task.REGRESSION:
        this_shaps = shaps[:, feat_idx]
    else:
        raise ValueError(f"Unknown task: {task}. Known tasks are `regression` and `classification`.") 
    sorted_indices = np.argsort(this_shaps)
    n_samples = len(sorted_indices)
    
    # initialise variables
    n_ones, n_zeros = np.sum(this_values), len(this_values)-np.sum(this_values)
    assert n_ones + n_zeros == len(this_values)

    best_thresholds = [-1]
    max_correct = max(n_ones, n_zeros)

    j = 0
    while j <= n_samples-1:
        while j+1 <= n_samples-1 and this_shaps[sorted_indices[j]] == this_shaps[sorted_indices[j+1]]:
            j+=1
        
        left_ones = np.sum(this_values[sorted_indices[:j+1]])
        left_zeros = j+1-left_ones
        right_ones = np.sum(this_values[sorted_indices[j+1:]])
        right_zeros = len(this_values)-j-1-right_ones
        
        assert left_zeros + right_zeros == n_zeros, AssertionError(f'{n_zeros} != {left_zeros} + {right_zeros}')
        assert left_ones + right_ones == n_ones, AssertionError(f'{n_ones} != {left_ones} + {right_ones}')

        n_correct = max(left_zeros+right_ones, left_ones+right_zeros)

        # did it get better?
        if n_correct == max_correct:
            best_thresholds.append(sorted_indices[j])
        elif n_correct > max_correct:
            best_thresholds = [sorted_indices[j], ]
            max_correct = n_correct
        
        j+=1
    
    purity = max_correct/len(this_values)
    # best_thresholds: indeks ostatniego związku po lewej
    if -1 in best_thresholds:
        values = np.hstack(([-np.inf], this_shaps[best_thresholds[1:]]))  # this_shaps[sorted_indices[-1]]
    else:
        values = this_shaps[best_thresholds]
    return max_correct, purity, values


def situation_at_threshold(sv, cecha, shaps, X, feature_order, task, class_index=None, print_func=print):
    """
    sv - threshold shap values
    cecha - name of feature that we analyse
    shaps - shap values
    X - representation of molecules
    feature_order - ordering of features across numpy arrays (shaps and X)
    task
    class_index - in case of classification, index of class that we're interested in
    """
    
    feat_idx = feature_order.index(cecha)  # indeks cechy o nazwie "cecha"
    bits = X[:, feat_idx]
    assert set(bits) == set([0, 1])  # czy mamy tylko zera i jedynki?
    
    if task == Task.CLASSIFICATION:
        this_shaps = shaps[class_index, :, feat_idx]
    elif task == Task.REGRESSION:
        this_shaps = shaps[:, feat_idx]
    else:
        raise ValueError(f"Unknown task: {task}. Known tasks are `regression` and `classification`.") 
    
    sorted_indices = np.argsort(this_shaps)
    sorted_shaps = this_shaps[sorted_indices]
        
    sorted_bits = bits[sorted_indices]
    
    if sv == -np.inf:
        i = -1
    else:
        i = np.max(np.where(sorted_shaps<=sv))

    left_ones = np.sum(sorted_bits[:i+1])
    left_zeros = (i+1) - left_ones
    
    right_ones = np.sum(sorted_bits[i+1:])
    right_zeros = (len(bits) - (i+1)) - right_ones
    n_correct = max(left_zeros+right_ones, left_ones+right_zeros)
    
    assert left_ones + right_ones == np.sum(bits)
    assert left_zeros + right_zeros == len(bits) - np.sum(bits)
    
    purity = n_correct/len(bits)
    
    if print_func is not None:
        print_func(left_zeros, right_zeros)
        print_func(left_ones, right_ones)
        print_func(n_correct, np.round(purity, 2), '\n')
    
    return left_zeros, left_ones, right_zeros, right_ones