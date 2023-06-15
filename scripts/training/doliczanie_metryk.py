import os
import sys
import time
import logging
import pickle
import neptune
import numpy as np
from sklearn.metrics import confusion_matrix

from metstab_pred.src.data import load_data
from metstab_pred.src.config import utils_section, csv_section, metrics_section, force_classification_metrics_section
from metstab_pred.src.utils import force_classification, get_scorer, find_and_load, get_configs_and_model
from metstab_pred.src.savingutils import save_configs, save_as_json, save_predictions, LoggerWrapper, pickle_and_log_artifact

# usage
# python doliczanie_metryk.py directory run_id

n_args = 1 + 2
project = neptune.init('lamiane/metstab-pred-scaffold')

if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: {sys.argv[0]} directory neptune_run_id")
        quit(1)

    # set data directory, logger and neptune run
    saving_dir = sys.argv[1]
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    nexp = project.get_experiments(id=sys.argv[2])[0]
    
    # load data and model
    data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle = get_configs_and_model(saving_dir)
    x, y, cv_split, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)

    # SAVING RESULTS
    save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, saving_dir)

    # SAVING SCORES
    all_scores = {}

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
