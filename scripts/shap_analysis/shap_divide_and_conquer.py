import os
import sys
import logging

import shap
import pickle
import neptune
import numpy as np

from metstab_pred.src.data import load_data, Unlogger
from metstab_pred.src.config import parse_shap_config, utils_section
from metstab_pred.src.utils import get_configs_and_model, find_and_load
from metstab_pred.src.savingutils import save_configs, pickle_and_log_artifact, save_npy_and_log_artifact, LoggerWrapper

neptune.init('lamiane/metstab-shap')
version_tag = "D"
modus_operandi_tag = "d&c"
tags=['metstab-shap', version_tag, modus_operandi_tag]
n_parts = 30

n_args = 1 + 3  # + 1 optional parameter


if __name__=='__main__':
    if (len(sys.argv) - n_args) not in [0, 1]:
        print(f"Usage: python {sys.argv[0]} input_directory saving_directory shap.cfg [n_parts (optional)]")
        quit(1)

    data_dir = sys.argv[1]
    saving_dir = sys.argv[2]
    try:
        n_parts = int(sys.argv[4])
    except IndexError:
        pass

    # set global saving dir for this experiment and create it
    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:]}')

    # load shap configuration
    shap_cfg = parse_shap_config(sys.argv[3])
    k = shap_cfg[utils_section]["k"]
    unlog = shap_cfg[utils_section]["unlog"]
    assert isinstance(unlog, bool), f"Bool must be bool, `unlog` is {type(unlog)}."
    if unlog:
        tags.append('unlogged')
    save_configs([sys.argv[3], ], saving_dir)

    # make neptune experiment
    nexp = neptune.create_experiment(name=saving_dir,
                                     params={'n_background_samples': k,
                                             'link': shap_cfg[utils_section]["link"], 'unlog': unlog,
                                             'source dir': data_dir, 'out dir': saving_dir,
                                             'n_parts': n_parts},
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

    # calculating background data using the whole dataset
    background_data = shap.kmeans(X_full, k)
    pickle_and_log_artifact(background_data, saving_dir, 'background_data', nexp)

    # chunking the dataset (#DIVIDE)
    n_samples = X_full.shape[0]
    part_size = int(np.floor(n_samples/n_parts))
    remaining = n_samples - (part_size*n_parts)
    assert remaining < n_parts
    
    sizes = [part_size,] * n_parts
    update = ([1,] * remaining) + ([0,] * (n_parts-remaining))
    assert len(sizes) == len(update)

    sizes = np.array(sizes) + np.array(update)
    assert np.sum(sizes) == n_samples
    print(f"Chunk sizes: {sizes}.")
    
    start = 0
    for i, size in enumerate(sizes):
        save_npy_and_log_artifact(X_full[start:start+size, :], saving_dir, f'X_part_{i+1}', allow_pickle=False, nexp=nexp)
        start = start+size
    assert start == n_samples

    # classes order and predictions will be the same regardless of part
    # saving it once and for the whole dataset:
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
    
    if unlog:
        model = Unlogger(model)

    if 'classification' == task_cfg[utils_section]['task']:
        save_npy_and_log_artifact(model.classes_, saving_dir, 'classes_order', allow_pickle=False, nexp=nexp)
        preds = model.predict_proba(X_full)
    else:
        preds = model.predict(X_full)

    save_npy_and_log_artifact(preds, saving_dir, 'predictions_part', allow_pickle=False, nexp=nexp)

    print("Finished.")
