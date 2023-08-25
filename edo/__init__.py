from enum import Enum
from collections import namedtuple

no_print = lambda x: None  # print nothing


class Task(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


def TASK_ERROR_MSG(t):
    return f"Unknown task: {t}. Known tasks are `Task.REGRESSION` and `Task.CLASSIFICATION`."


def deduce_task(task, s_vals=None):
    """
    Deduce task and return as Task.
    :param task: Task, str or None: task description
    :param s_vals: np.array or None: SHAP values for a single feature/sample
    :return: Task
    """
    if task is None:
        if 2 == len(s_vals.shape):
            task = Task.CLASSIFICATION
        elif 1 == len(s_vals.shape):
            task = Task.REGRESSION
        else:
            raise ValueError(f"`s_vals.shape` must be 1 or 2 not {s_vals.shape}.")
    elif isinstance(task, str):
        task = Task(task)
    elif isinstance(task, Task):
        task = task
    else:
        raise TypeError(f"`task` must be string, Task or None, is {type(task)}.")
    return task


# Origin identifies a ML model and can be used to load the corresponding data (edo.optimisation.utils.find_experiment)
# ex. Origin('human', 'random', 'KRFP', 'classification', 'SVM') is
# an SVM classifier trained on randomly split human data represented with KRFP
# NOTE: hypothetically, there can be more than one model with such a specification, though not in our project
Origin = namedtuple('Origin', ['dataset', 'split', 'representation', 'task', 'model'])


def make_origin(origin):
    """
    Create Origin object.
    :param origin: Origin, Tuple[str] or List[str]: origin description
    :return: Origin
    """
    if not isinstance(origin, Origin):
        dataset, split, representation, task, model = origin

        mod = model.lower()
        models = ['nb', 'svm', 'trees', 'knn']
        assert mod in models, f'`model` {model} not in {models}'

        fp = representation.lower()
        fps = ['kr', 'krfp', 'maccs', 'ma', 'padel', 'pubfp', 'mo128', 'mo512', 'mo1024']
        assert fp in fps, f'`representation` {representation} not in {fps}'

        # we used a different abbreviation when calculating KNN models. This is a workaround.
        if fp in ['krfp', 'maccs'] and mod != 'knn':
            fp = fp[:2]
        elif mod == 'knn' and fp == 'kr':
            fp = 'krfp'
        elif mod == 'knn' and fp == 'ma':
            fp = 'maccs'
        else:
            pass

        splits = {'r': 'random', 'random': 'random', 's': 'scaffold', 'scaffold': 'scaffold'}
        datasets = {'h': 'h', 'human': 'h', 'r': 'r', 'rat': 'r'}
        tasks = {'r': 'r', 'reg': 'r', 'regression': 'r', 'c': 'c', 'cls': 'c', 'classification': 'c'}

        split = splits[split.lower()]
        data = datasets[dataset.lower()]
        task = tasks[task.lower()]

        origin = Origin(data, split, fp, task, mod)
    return origin
