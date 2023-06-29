import os
import os.path as osp
import sys
import time
import types
import logging
import numpy as np
from sklearn.metrics._scorer import _BaseScorer
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.metrics import precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from .data import unlog_stability


def debugger_decorator(func):
    """Print input params and the returned values."""

    def wrapper(*args, **kwargs):
        print(f'\nCalling {func} with params:')
        for a in args:
            print(a)
        for k in kwargs:
            print(f'{k}: {kwargs[k]}')
        returned_values = func(*args, **kwargs)
        print(f'\nReturned values are: {returned_values}\n')
        return returned_values

    return wrapper


class LoggerWrapper:
    def __init__(self, path='.'):
        """
        Wrapper for logging.
        Allows to replace sys.stderr.write so that error massages are redirected do sys.stdout and also saved in a file.
        use: logger = LoggerWrapper(); sys.stderr.write = logger.log_errors
        :param: path: directory to create log file
        """
        # count spaces so that the output is nicely indented
        self.trailing_spaces = 0

        # create the log file
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.filename = osp.join(path, f'{timestamp}.log')
        try:
            os.mknod(self.filename)
        except FileExistsError:
            pass

        # configure logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.filename,
                            filemode='w')
        formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')

        # make a handler to redirect everything to std.out
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        self.console = logging.StreamHandler(sys.stdout)
        self.console.setLevel(logging.INFO)
        self.console.setFormatter(formatter)
        self.logger.addHandler(self.console)

    def log_errors(self, msg):
        msg = msg.strip('\n')  # don't add extra enters

        if msg == ' ' * len(msg):  # if you only get spaces: don't print them, but do remember
            self.trailing_spaces += len(msg)
        elif len(msg) > 1:
            self.logger.error(' ' * self.trailing_spaces + msg)
            self.trailing_spaces = 0


# # # WRAPPERS FOR SKLEARN SCORERS
def get_scorer(scoring):
    """wrapper for sklearn.metrics.get_scorer to use precision and recall with average=None, and confusion_matrix"""
    if 'precision_none' == scoring:
        return make_scorer(precision_score, greater_is_better=True, needs_proba=False, needs_threshold=False,
                           average=None)
    elif 'recall_none' == scoring:
        return make_scorer(recall_score, greater_is_better=True, needs_proba=False, needs_threshold=False, average=None)
    elif 'confusion_matrix' == scoring:
        return make_scorer(confusion_matrix, greater_is_better=True, needs_proba=False, needs_threshold=False)
    else:
        return sklearn_get_scorer(scoring)


class NanSafeScorer(_BaseScorer):
    """Wrapper for sklearn scorers that returns the worst possible value instead of nan"""

    def __init__(self, scorer):
        assert isinstance(scorer, _BaseScorer)
        super().__init__(scorer._score_func, scorer._sign, scorer._kwargs)
        self._scorer = scorer

        if any([fname in self._score_func.__name__ for fname in ['roc_auc']]):
            self._worst = 0
        elif 'error' in self._score_func.__name__:
            self._worst = np.inf if self._sign > 0 else -np.inf
        else:
            raise NotImplementedError(f'Unimplemented _score_func {self._score_func.__name__}.')

    def __call__(self, estimator, X, y_true, sample_weight=None):
        try:
            return self._scorer(estimator, X, y_true, sample_weight)
        except ValueError:
            return self._worst


# # # WRAPPERS FOR SKLEARN MODELS
def adapt_to_classification(model, cutoffs, **kwargs):
    """
    Adapts a regression model to perform classification.
    :param model: sklearn-like regression model
    :param cuttoffs: function: a function that changes regression into classification
    :param kwargs: params for cuttoffs function
    :return: model
    """

    model.old_predict = model.predict
    model.cutoffs = cutoffs
    model.cutoffs_kwargs = kwargs

    def new_predict(self, X):
        y = self.old_predict(X)
        y = self.cutoffs(y, self.cutoffs_kwargs)
        return y

    model.predict = types.MethodType(new_predict, model)
    return model


class Unloger(object):
    """Unlogs predictions of regression models when calculating SHAP values (for classifiers it makes no sense)."""

    def __init__(self, model):
        if isinstance(model, Pipeline):
            if not isinstance(model.steps[-1][1], RegressorMixin):
                raise TypeError(f"`model` must be a regressor, is {type(model.steps[-1][1])}.")
        elif not isinstance(model, RegressorMixin):
            raise TypeError(f"`model` must be a regressor, is {type(model)}.")

        self.model = model
        self.unlog = unlog_stability
        self.MemoryError = False

    def predict(self, X):
        try:
            return self.unlog(self.model.predict(X))
        except MemoryError:
            if not self.MemoryError:
                self.MemoryError = True
                print("MemoryError workaround: this is gonna be sloooooow...")
            return self.unlog(np.array([self.model.predict(x.reshape(1, -1)) for x in X]).reshape(-1))
