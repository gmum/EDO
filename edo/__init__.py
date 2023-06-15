from enum import Enum
from collections import namedtuple

from ._errors import TASK_ERROR_MSG


class Task(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


def deduce_task(task, s_vals=None):
    if task is None:
        if 2 == len(s_vals.shape):
            task = Task.CLASSIFICATION
        else:
            task = Task.REGRESSION
    elif isinstance(task, str):
        task = Task(task)
    elif isinstance(task, Task):
        task = task
    else:
        raise TypeError(f"`task` must be string, Task or None, is {type(task)}.")
    return task


# Origin identifies a ML model
# ex. Origin('human', 'random', 'KRFP', 'classification', 'SVM') is
# an SVM classifier trained on randomly splitted human data represented with KRFP
# note: hypothetically, there can be more than one model with such a specification, though not in our project
Origin = namedtuple('Origin', ['dataset', 'split', 'representation', 'task', 'model'])


def make_origin(origin):
    if not isinstance(origin, Origin):
        dataset, split, representation, task, model = origin

        mod = model.lower()
        models = ['nb', 'svm', 'trees', 'knn']
        assert mod in models, f'`model` {model} not in {models}'

        fp = representation.lower()
        fps = ['kr', 'krfp', 'maccs', 'ma', 'padel', 'pubfp', 'mo128', 'mo512', 'mo1024']
        assert fp in fps, f'`fingerprint` {representation} not in {fps}'

        if fp in ['krfp', 'maccs'] and mod != 'knn':
            fp = fp[:2]
        elif mod == 'knn' and fp == 'kr':
            fp = 'krfp'
        elif mod == 'knn' and fp == 'ma':
            fp = 'maccs'

        splits = {'r': 'random', 'random': 'random',
                  's': 'scaffold', 'scaffold': 'scaffold'}
        datasets = {'h': 'h', 'human': 'h',
                    'r': 'r', 'rat': 'r'}
        tasks = {'r': 'r', 'reg': 'r', 'regression': 'r',
                 'c': 'c', 'cls': 'c', 'classification': 'c'}

        split = splits[split.lower()]
        data = datasets[dataset.lower()]
        task = tasks[task.lower()]

        origin = Origin(data, split, fp, task, mod)
    return origin
