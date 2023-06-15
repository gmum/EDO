import os

# changing default tempdir for Prometheus
import tempfile
tempfile.tempdir = os.getenv('SCRATCH_LOCAL', None)

import sys
import logging
import neptune
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from metstab_pred.src.config import utils_section, csv_section, metrics_section, force_classification_metrics_section
from metstab_pred.src.config import parse_data_config, parse_representation_config, parse_task_config
from metstab_pred.src.data import load_data
from metstab_pred.src.utils import force_classification, get_scorer
from metstab_pred.src.savingutils import save_configs, save_as_json, save_predictions, LoggerWrapper, pickle_and_log_artifact


# usage
# python scripts/training/train-1nn.py configs/data/rat.cfg configs/repr/maccs.cfg configs/task/classification.cfg saving_dir

n_args = 1 + 4
neptune.init('lamiane/metstab-knn')
version_tag = "A"


if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: {sys.argv[0]} data.cfg representation.cfg task.cfg main_saving_directory")
        quit(1)

    # set global saving subdir for this experiment and create it
    saving_dir = sys.argv[4]
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:-1]}')

    # Load configs
    data_cfg = parse_data_config(sys.argv[1])
    repr_cfg = parse_representation_config(sys.argv[2])
    task_cfg = parse_task_config(sys.argv[3])
    save_configs(sys.argv[1:-1], saving_dir)

    nexp = neptune.create_experiment(name=saving_dir,
                                     params={'dataset': data_cfg[utils_section]['dataset'],
                                             'model': 'KNN', 'neighbours': 1,
                                             'fingerprint': repr_cfg[utils_section]['fingerprint'],
                                             'morgan_nbits': repr_cfg[utils_section]['morgan_nbits'],
                                             'task': task_cfg[utils_section]['task'],
                                             'data_cfg':data_cfg, 'repr_cfg': repr_cfg, 'task_cfg': task_cfg,
                                            },
                                     tags=['metstabpred',] + sys.argv[1:-1] + [version_tag],
                                     upload_source_files=os.path.join(os.path.dirname(os.path.realpath(__file__)), '*.py')
                                     )

    # load data (and change to classification if needed)
    x, y, cv_split, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

    # # # saving dataset just in case
    objects = [x, y, smiles, test_x, test_y, test_smiles]
    fnames = ['x', 'y', 'smiles', 'test_x', 'test_y', 'test_smiles']
    for obj, fname in zip(objects, fnames):
        pickle_and_log_artifact(obj, saving_dir, fname, nexp)

    # change y in case of classification
    if 'classification' == task_cfg[utils_section]['task']:
        log_scale = True if 'log' == data_cfg[csv_section]['scale'].lower().strip() else False
        y = task_cfg[utils_section]['cutoffs'](y, log_scale)
        test_y = task_cfg[utils_section]['cutoffs'](test_y, log_scale)

    # define model
    if 'classification' == task_cfg[utils_section]['task']:
        model = KNeighborsClassifier(n_neighbors=1, metric='jaccard')
    elif 'regression' == task_cfg[utils_section]['task']:
        model = KNeighborsRegressor(n_neighbors=1, metric='jaccard')
    else:
        raise ValueError(f"Task must be `regression` or `classification`. Is {task_cfg[utils_section]['task']}.")
    
    # no CV because no grid search
    _ = model.fit(x, y)
    
    # SAVING RESULTS

    # save the model itself
    pickle_and_log_artifact(model, saving_dir, 'model', nexp)

    # save predictions
    save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, saving_dir)

    # SAVING SCORES
    all_scores = {}

    # # main score
    all_scores['model_train_score'] = model.score(x, y)
    all_scores['model_test_score'] = model.score(test_x, test_y)

    # # additional scores
    for score_name in task_cfg[metrics_section].values():
        scorer = get_scorer(score_name)
        try:
            all_scores[f'test_{score_name}'] = scorer(model, test_x, test_y)
        except RuntimeError:
            all_scores[f'test_{score_name}'] = 'RuntimeError'

    # # for regression models we perform dummy classification
    if force_classification_metrics_section in task_cfg:
        # change data and model to work with classification
        log_scale = True if 'log' == data_cfg[csv_section]['scale'].lower().strip() else False
        y = task_cfg[utils_section]['cutoffs'](y, log_scale)
        test_y = task_cfg[utils_section]['cutoffs'](test_y, log_scale)
        model = force_classification(model, task_cfg[utils_section]['cutoffs'], log_scale=log_scale)
        save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, os.path.join(saving_dir, 'forced_classification'))

        predictions = model.predict(test_x)
        for score_name in task_cfg[force_classification_metrics_section].values():
            scorer = get_scorer(score_name)
            try:
                all_scores[f'forced_classification_test_{score_name}'] = scorer(model, test_x, test_y)
            except RuntimeError:
                all_scores[f'forced_classification_test_{score_name}'] = 'RuntimeError'

    logger_wrapper.logger.info(all_scores)
    save_as_json(all_scores, saving_dir, 'best_model_scores.json', nexp=nexp)
    _ = [nexp.log_metric(k,v) for k,v in all_scores.items() if isinstance(v, float)]
