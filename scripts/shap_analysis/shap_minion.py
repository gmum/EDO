import sys

import shap
import pickle

from edo.wrappers import Unloger, LoggerWrapper
from edo.config import parse_shap_config, UTILS
from edo.utils import get_configs_and_model, find_and_load
from edo.savingutils import save_as_np


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


    # load shap configuration
    shap_cfg = parse_shap_config(sys.argv[3])
    link = shap_cfg[UTILS]["link"]
    unlog = shap_cfg[UTILS]["unlog"]
    assert isinstance(unlog, bool), f"Bool must be bool, `unlog` is {type(unlog)}."

    # load other configs
    data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle = get_configs_and_model(data_dir)

    # load model, data and background data
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
        
    if unlog:
        model = Unloger(model)
        
    X_part = find_and_load(saving_dir, f'X_part_{part_id}.npy', protocol='numpy')
    background_data = find_and_load(saving_dir, "background_data.pickle", protocol='pickle')

    # calculating SHAP values
    if 'classification' == task_cfg[UTILS]['task']:
        e = shap.KernelExplainer(model.predict_proba, background_data, link=link)
    else:
        e = shap.KernelExplainer(model.predict, background_data, link=link)  # regression
    sv = e.shap_values(X_part)

    # saving results
    save_as_np(sv, saving_dir, f'SHAP_values_part_{part_id}', allow_pickle=False, nexp=nexp)
    save_as_np(e.expected_value, saving_dir, f'expected_values_part_{part_id}', allow_pickle=False, nexp=nexp)
