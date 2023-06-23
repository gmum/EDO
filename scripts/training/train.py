import os
import sys
import time

import tpot
tpot.decorators.MAX_EVAL_SECS = 120


import edo.training.grid as grid
from edo.config import utils_section, csv_section, metrics_section, force_classification_metrics_section
from edo.config import parse_model_config, parse_data_config, parse_representation_config, parse_task_config, parse_tpot_config
from edo.data import load_data
from edo.utils import force_classification, get_scorer, NanSafeScorer
from edo.savingutils import save_configs, save_as_json, save_predictions, LoggerWrapper, save_as_pickle

tpot.decorators.MAX_EVAL_SECS = 120  # just in case

# usage
# python scripts/training/train.py configs/model/svm.cfg configs/data/rat.cfg configs/repr/maccs.cfg configs/task/classification.cfg configs/tpot-med.cfg saving_dir

n_args = 1 + 6
wrap_score = True



if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: {sys.argv[0]} model.cfg data.cfg representation.cfg task.cfg tpot.cfg main_saving_directory")
        quit(1)

    # set global saving subdir for this experiment and create it
    saving_dir = sys.argv[6]
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:-1]}')

    # Load configs
    model_cfg = parse_model_config(sys.argv[1])
    data_cfg = parse_data_config(sys.argv[2])
    repr_cfg = parse_representation_config(sys.argv[3])
    task_cfg = parse_task_config(sys.argv[4])
    tpot_cfg = parse_tpot_config(sys.argv[5])
    save_configs(sys.argv[1:-1], saving_dir)


    # load data (and change to classification if needed)
    x, y, cv_split, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])

    # # # saving dataset just in case
    objects = [x, y, smiles, test_x, test_y, test_smiles]
    fnames = ['x', 'y', 'smiles', 'test_x', 'test_y', 'test_smiles']
    for obj, fname in zip(objects, fnames):
        save_as_pickle(obj, saving_dir, fname)

    # change y in case of classification
    if 'classification' == task_cfg[utils_section]['task']:
        log_scale = True if 'log' == data_cfg[csv_section]['scale'].lower().strip() else False
        y = task_cfg[utils_section]['cutoffs'](y, log_scale)
        test_y = task_cfg[utils_section]['cutoffs'](test_y, log_scale)

    # grid space and tpot configuration
    grid_space = grid.get_grid(task_cfg[utils_section]['task'], **model_cfg[utils_section])

    n_jobs = tpot_cfg[utils_section]['n_jobs']
    max_time_mins = tpot_cfg[utils_section]['max_time_mins']
    minimal_number_of_models = tpot_cfg[utils_section]['minimal_number_of_models']


    scorer = get_scorer(task_cfg[utils_section]['metric'])
    if wrap_score:
        scorer = NanSafeScorer(scorer)

    tpot_model_kwargs = { # constants
        'generations': None,
        'random_state': 666,
        'warm_start': False,
        'use_dask': True,
        'memory': 'auto',
        'verbosity': 3,
        'cv': cv_split,
        # general setup
        'n_jobs': n_jobs,
        'max_time_mins': max_time_mins,
        'max_eval_time_mins': n_jobs * max_time_mins // minimal_number_of_models,
        # per model setup
        'scoring': scorer,
        'periodic_checkpoint_folder': os.path.join(saving_dir, "./tpot_checkpoints")
         }


    # run experiment
    logger_wrapper.logger.info(f"Starting time {time.strftime('%Y-%m-%d %H:%M')}")
        
    TPOTModel = task_cfg[utils_section]['tpot_model']
    model = TPOTModel(config_dict=grid_space, **tpot_model_kwargs)

    _ = model.fit(x, y)



    # SAVING RESULTS
    timestamp = time.strftime('%Y-%m-%d_%H-%M')
    logger_wrapper.logger.info(f"End time {timestamp}")

    model.export(os.path.join(saving_dir, f'{timestamp}-best_model_script.py'))

    save_as_json(model.evaluated_individuals_, saving_dir, 'evaluated_individuals.json')

    # save the model itself
    save_as_pickle(model.fitted_pipeline_, saving_dir, 'model')

    # save predictions
    save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, saving_dir)

    # SAVING SCORES
    all_scores = {}

    # # main score
    all_scores['grid_mean_cv_score'] = model._optimized_pipeline_score
    all_scores['grid_test_score'] = model.score(test_x, test_y)

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
    save_as_json(all_scores, saving_dir, 'best_model_scores.json')
