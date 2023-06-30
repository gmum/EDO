import os.path as osp
import numpy as np
import pandas as pd

from edo import Task
from edo.config import UTILS, CSV
from edo.utils import get_configs_and_model

from utils import load_shap_files, load_ml_files, get_smiles_true_predicted, get_smiles_correct, Stability

##############################################
### Kamila zupelnie straszna implementacja ###
##############################################

import dataclasses
from enum import Enum


def get_feature_index(feature_order, feature_name):
    if type(feature_order) is np.ndarray:
        feature_order = feature_order.tolist()

    return feature_order.index(feature_name)


def get_x_axis(shaps, task, feature_index, class_index):
    if task == Task.REGRESSION:
        assert (len(shaps.shape) == 2) and (class_index is None)
        return shaps[:, feature_index]
    elif task == Task.CLASSIFICATION:
        assert (len(shaps.shape) == 3) and (class_index is not None)
        return shaps[class_index, :, feature_index]
    else:
        assert False


def get_y_axis(X, feature_index):
    return X[:, feature_index]


@dataclasses.dataclass
class SPoint:
    x: float
    zeroes: int
    ones: int


def make_s_point(x, y):
    if y >= 0.5:
        return SPoint(x, zeroes=0, ones=1)
    else:
        return SPoint(x, zeroes=1, ones=0)


def update_s_point(s_point, y):
    if y < 0.5:
        s_point.zeroes += 1
    else:
        s_point.ones += 1


def make_s_point_list(x_axis, y_axis):
    assert x_axis.shape == y_axis.shape
    xy_list = sorted(np.column_stack((x_axis, y_axis)).tolist())

    result = []
    last = SPoint(float('-inf'), 0, 0)

    for x, y in xy_list:
        if last.x == x:
            update_s_point(last, y)
        else:
            last = make_s_point(x, y)
            result.append(last)

    return result


class SeparationType(Enum):
    ZEROES_ON_LEFT = 0
    ONES_ON_LEFT = 1


@dataclasses.dataclass
class Separation:
    x: float
    score: int
    type: SeparationType


def update_separation(separation, x, add, subtract):
    separation.x = x
    separation.score += add
    separation.score -= subtract


def update_best_separation(best, current):
    best_score = best[0].score

    if current.score < best_score:
        return

    if current.score > best_score:
        best.clear()

    best.append(dataclasses.replace(current))


def calculate_best_separations(s_points):
    total_zeroes = sum(p.zeroes for p in s_points)
    total_ones = sum(p.ones for p in s_points)

    best_with_zeroes_on_left = [Separation(float('-inf'), total_ones, SeparationType.ZEROES_ON_LEFT)]
    best_with_ones_on_left = [Separation(float('-inf'), total_zeroes, SeparationType.ONES_ON_LEFT)]

    current_with_zeroes_on_left = dataclasses.replace(best_with_zeroes_on_left[0])
    current_with_ones_on_left = dataclasses.replace(best_with_ones_on_left[0])

    for p in s_points:
        update_separation(current_with_zeroes_on_left, p.x, add=p.zeroes, subtract=p.ones)
        update_best_separation(best_with_zeroes_on_left, current_with_zeroes_on_left)

        update_separation(current_with_ones_on_left, p.x, add=p.ones, subtract=p.zeroes)
        update_best_separation(best_with_ones_on_left, current_with_ones_on_left)

    assert current_with_zeroes_on_left.score == total_zeroes
    assert current_with_ones_on_left.score == total_ones

    zeroes_on_left_score = best_with_zeroes_on_left[0].score
    ones_on_left_score = best_with_ones_on_left[0].score

    if zeroes_on_left_score > ones_on_left_score:
        return best_with_zeroes_on_left
    if zeroes_on_left_score < ones_on_left_score:
        return best_with_ones_on_left

    return best_with_zeroes_on_left + best_with_ones_on_left


def find_optimal_separation_point(shaps, X, feature_order, feature_name, task, class_index=None, extras=False):
    """
    shaps - shap values: shaps[[class,]molecule,feature_index]
    X - representation: X[molecule,feature_index]
    feature_order - features order: list of feature names
    feature_name - name of feature for which the analysis is done,
    task - either REGRESSION (class_index is None) or CLASSIFICATION,
    class_index - if task is classification, then for which class should the analysis be done
    """

    feature_index = get_feature_index(feature_order, feature_name)
    x_axis = get_x_axis(shaps, task, feature_index, class_index)
    y_axis = get_y_axis(X, feature_index)

    s_points = make_s_point_list(x_axis, y_axis)
    # return calculate_best_separations(s_points)
    separations = calculate_best_separations(s_points)
    assert np.all([separations[0].score == s.score for s in separations])

    if extras:
        return separations
    else:
        return separations[0].score, separations[0].score/X.shape[0], [s.x for s in separations]


if __name__=="__main__":
    data_path = '/home/pocha/dane_doktorat'
    
    exp = 'h-ma-c-trees'
    some_model = osp.join(data_path, 'ml', exp)
    some_shaps = osp.join(data_path, 'shap', exp)

    exp = 'h-ma-r-trees'
    some_model = osp.join(data_path, 'ml', exp)
    some_shaps = osp.join(data_path, 'shap', exp)

    data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle = get_configs_and_model(some_model)
    x_train, x_test, smiles_train, smiles_test = load_ml_files(some_model)
    task = Task(task_cfg[UTILS]['task'])
    shap_cfg, smiles_order, X_full, morgan_repr, true_ys, preds, classes_order, expected_values, shap_values, background_data = load_shap_files(some_shaps, task)

    smiles_true_predicted_df = get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order)
    
    separations = find_optimal_separation_point(shap_values, X_full, list(range(166)), 10, task, None)
    print(separations)
