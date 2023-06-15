import os
import sys
import logging
import time
import warnings

import shap
import pickle
import neptune
import numpy as np

from metstab_pred.src.data import load_data, Unlogger
from metstab_pred.src.config import parse_shap_config, utils_section
from metstab_pred.src.utils import get_configs_and_model, find_and_load
from metstab_pred.src.savingutils import save_configs, save_as_json, pickle_and_log_artifact, save_npy_and_log_artifact, LoggerWrapper

neptune.init('lamiane/metstab-shap')
version_tag = "C"
tags = ['metstab-shap', version_tag]

n_args = 1 + 3

if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: python {sys.argv[0]} input_directory saving_directory shap.cfg")
        quit(1)

    data_dir = sys.argv[1]
    saving_dir = sys.argv[2]

    # set global saving dir for this experiment and create it
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:]}')

    # we know this warning, we skip it
    #     log = logging.getLogger('shap')
    #     level = log.level
    #     log.setLevel(logging.ERROR)
    #     warnings.simplefilter("ignore")

    # load shap configuration
    shap_cfg = parse_shap_config(sys.argv[3])
    k = shap_cfg[utils_section]["k"]
    link = shap_cfg[utils_section]["link"]
    unlog = shap_cfg[utils_section]["unlog"]
    assert isinstance(unlog, bool), f"Bool must be bool, `unlog` is {type(unlog)}."
    if unlog:
        tags.append('unlogged')
    save_configs([sys.argv[3], ], saving_dir)

    # make neptune experiment
    nexp = neptune.create_experiment(name=saving_dir,
                                     params={'n_background_samples': k, 'link': link, 'unlog': unlog,
                                             'source dir': data_dir, 'out dir': saving_dir},
                                     tags=tags,
                                     upload_source_files=os.path.join(os.path.dirname(os.path.realpath(__file__)), '*.py'))


    # load other configs
    data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle = get_configs_and_model(data_dir)
    nexp.log_text('model pickle', model_pickle)
    nexp.set_property('dataset', os.path.splitext(os.path.basename(data_cfg[utils_section]['test']))[0].split('-')[0])
    nexp.set_property('model', model_cfg[utils_section]['model'])
    nexp.set_property('task', task_cfg[utils_section]['task'])
    nexp.set_property('fingerprint', repr_cfg[utils_section]['fingerprint'])
    nexp.set_property('morgan_nbits', repr_cfg[utils_section]['morgan_nbits'])
    
    if unlog and task_cfg[utils_section]['task']=='classification':
        raise ValueError('Unlogging for classification does not make sense!')
    
    # load and concatenate data
    if "fp" not in repr_cfg[utils_section]['fingerprint'] and 'padel' not in repr_cfg[utils_section]['fingerprint']:
        # MACCS or Morgan, we can calculate it for the sake of simplicity
        x, y, _, test_x, test_y, smiles, test_smiles = load_data(data_cfg, **repr_cfg[utils_section])
    else:
        # KRFP, PubFP or PaDEL, we want to save time
        x = find_and_load(data_dir, '-x.pickle', protocol='pickle')
        y = find_and_load(data_dir, '-y.pickle', protocol='pickle')
        smiles = find_and_load(data_dir, '-smiles.pickle', protocol='pickle')
        test_x = find_and_load(data_dir, '-test_x.pickle', protocol='pickle')
        test_y = find_and_load(data_dir, '-test_y.pickle', protocol='pickle')
        test_smiles = find_and_load(data_dir, '-test_smiles.pickle', protocol='pickle')
        
    # we care no more about the data split so we merge it back
    X_full = np.concatenate((x, test_x), axis=0)
    y_full = np.concatenate((y, test_y), axis=0)
    smiles_full = np.concatenate((smiles, test_smiles), axis=0)

    # some smiles are present in the data multiple times
    # removing abundant smiles will speed up calculating shap values
    unique_indices = [np.where(smiles_full == sm)[0][0] for sm in np.unique(smiles_full)]
    X_full = X_full[unique_indices, :]
    y_full = y_full[unique_indices]
    smiles_full = smiles_full[unique_indices]

    objects = [X_full, y_full, smiles_full]
    fnames = ['X_full', 'true_ys', 'smiles']
    for obj, fname in zip(objects , fnames):
        save_npy_and_log_artifact(obj, saving_dir, fname, allow_pickle=False, nexp=nexp)

    # load model
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
    
    if unlog:
        model = Unlogger(model)

    # calculating SHAP values
    nexp.log_text('shap start', time.strftime('%Y-%m-%d %H:%M'))
    background_data = shap.kmeans(X_full, k)
    if 'classification' == task_cfg[utils_section]['task']:
        e = shap.KernelExplainer(model.predict_proba, background_data, link=link)
    else:
        e = shap.KernelExplainer(model.predict, background_data, link=link)  # regression
    sv = e.shap_values(X_full)
    nexp.log_text('shap end', time.strftime('%Y-%m-%d %H:%M'))

    # saving results
    pickle_and_log_artifact(background_data, saving_dir, 'background_data', nexp)
    save_npy_and_log_artifact(sv, saving_dir, 'SHAP_values', allow_pickle=False, nexp=nexp)
    save_npy_and_log_artifact(e.expected_value, saving_dir, 'expected_values', allow_pickle=False, nexp=nexp)

    if 'classification' == task_cfg[utils_section]['task']:
        save_npy_and_log_artifact(model.classes_, saving_dir, 'classes_order', allow_pickle=False, nexp=nexp)
        preds = model.predict_proba(X_full)
    else:
        preds = model.predict(X_full)
    save_npy_and_log_artifact(preds, saving_dir, 'predictions', allow_pickle=False, nexp=nexp)

    nexp.log_text('main end', time.strftime('%Y-%m-%d %H:%M'))
