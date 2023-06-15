import os
import sys
import logging
import time
import warnings

import shap
import pickle
import neptune
import numpy as np

from metstab_pred.src.data import Unlogger
from metstab_pred.src.config import parse_shap_config, utils_section
from metstab_pred.src.utils import get_configs_and_model, find_and_load
from metstab_pred.src.savingutils import save_npy_and_log_artifact, LoggerWrapper

neptune.init('lamiane/metstab-shap')
version_tag = "C"
modus_operandi_tag = "minion"
tags = ['metstab-shap', version_tag, modus_operandi_tag]

n_args = 1 + 4


if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: python {sys.argv[0]} input_directory saving_directory shap.cfg part_id")
        quit(1)

    data_dir = sys.argv[1]
    saving_dir = sys.argv[2]
    part_id = sys.argv[4]

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
    link = shap_cfg[utils_section]["link"]
    unlog = shap_cfg[utils_section]["unlog"]
    assert isinstance(unlog, bool), f"Bool must be bool, `unlog` is {type(unlog)}."
    if unlog:
        tags.append('unlogged')

    # make neptune experiment
    nexp = neptune.create_experiment(name=saving_dir,
                                     params={'n_background_samples': shap_cfg[utils_section]["k"],
                                             'link': link, 'unlog': unlog,
                                             'part_id': part_id,
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
    
    # load model, data and background data
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
        
    if unlog:
        model = Unlogger(model)
        
    X_part = find_and_load(saving_dir, f'X_part_{part_id}.npy', protocol='numpy')
    background_data = find_and_load(saving_dir, "background_data.pickle", protocol='pickle')

    # calculating SHAP values
    nexp.log_text('shap start', time.strftime('%Y-%m-%d %H:%M'))
    if 'classification' == task_cfg[utils_section]['task']:
        e = shap.KernelExplainer(model.predict_proba, background_data, link=link)
    else:
        e = shap.KernelExplainer(model.predict, background_data, link=link)  # regression
    sv = e.shap_values(X_part)
    nexp.log_text('shap end', time.strftime('%Y-%m-%d %H:%M'))

    # saving results
    save_npy_and_log_artifact(sv, saving_dir, f'SHAP_values_part_{part_id}', allow_pickle=False, nexp=nexp)
    save_npy_and_log_artifact(e.expected_value, saving_dir, f'expected_values_part_{part_id}', allow_pickle=False, nexp=nexp)
