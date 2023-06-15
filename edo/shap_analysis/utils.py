import os
import os.path as osp
import time

import numpy as np
import pandas as pd

from copy import deepcopy
from collections.abc import Iterable

from rdkit import Chem

from ..utils import find_and_load
from ..config import utils_section, csv_section, parse_shap_config
from ..data import preprocess_dataset

from .. import Task, TASK_ERROR_MSG


DATA_DIR = '/home/pocha/dane_phd'

# TODO: tests!


def load_ml_files(directory):
    x_train = find_and_load(directory, '-x.pickle', protocol='pickle')
    x_test = find_and_load(directory, '-test_x.pickle', protocol='pickle')
    
    smiles_train = find_and_load(directory, '-smiles.pickle', protocol='pickle')
    smiles_test = find_and_load(directory, '-test_smiles.pickle', protocol='pickle')
    
    return x_train, x_test, smiles_train, smiles_test


def load_shap_files(directory, task, check_unlogging=True):
    shap_cfg = parse_shap_config([osp.join(directory, f) for f in os.listdir(directory) if 'shap' in f and 'cfg' in f][0])
    if check_unlogging and task==Task.REGRESSION:
        assert shap_cfg[utils_section]["unlog"], f"{directory} contains SHAP values for an estimator that was not unlogged!"

    smiles_order = find_and_load(directory, 'canonised.npy', protocol='numpy')
    X_full = find_and_load(directory, 'X_full.npy', protocol='numpy')
    morgan_repr = find_and_load(directory, "morgans.npy", protocol='numpy')
    true_ys = find_and_load(directory, 'true_ys.npy', protocol='numpy')
    preds = find_and_load(directory, 'predictions', protocol='numpy')
    expected_values = find_and_load(directory, 'expected_values.npy', protocol='numpy')
    shap_values = find_and_load(directory, 'SHAP_values.npy', protocol='numpy')
    background_data = find_and_load(directory, 'background_data.pickle', protocol='pickle')

    if task == Task.CLASSIFICATION:
        classes_order = find_and_load(directory, 'classes_order.npy', protocol='numpy')
    elif task == Task.REGRESSION:
        classes_order = None
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    return shap_cfg, smiles_order, X_full, morgan_repr, true_ys, preds, classes_order, expected_values, shap_values, background_data


def canonise(smiles):
    """Return canonic form of smiles"""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)


def index_of_smiles(smiles_order, smi):
    """Return index of a given SMILE or None."""
    try:
        if isinstance(smi, str):
            # all indices of smiles
            return np.where(smiles_order == smi)[0]
        elif isinstance(smi, Iterable):
            # TODO: lepsze obsłużenie IndexErrora
            # list comprehension because the order is important
            # one index per smiles
            return [np.where(smiles_order == s)[0][0] for s in smi]
    except IndexError:
        # element not found in the array
        return None


def get_chembl(smi, data_dir):
    """Get CHEMBL of (canonic) smiles if we have it in our dataset.
    Returns None if smiles was not found."""
    df = pd.read_csv(osp.join(data_dir, 'smiles_chembl.csv'), index_col=0)
    try:
        chembl = df.loc[smi].CHEMBL_ID
    except KeyError:
        chembl = None
    return chembl


def represent_compound(compound, repr_cfg, data_cfg, tmpdirname):
    """
    Get a representation for a single compound.
    """
    timestamp = time.time()
    fname = osp.join(tmpdirname, f'{timestamp}-smiles.csv')
    df = pd.DataFrame([[compound['canonic_smiles'], -1]], columns=['SMILES', 'STABILITY_SCORE'])
    df.to_csv(fname, sep=data_cfg[csv_section]['delimiter'], header=data_cfg[csv_section]['skip_line'])

    # this will never happen with our data, byt might happen if someone added new datasets
    if 1 != data_cfg[csv_section]['smiles_index'] or 2 != data_cfg[csv_section]['y_index']:
        data_cfg = deepcopy(data_cfg)
        data_cfg[csv_section]['smiles_index'] = 1
        data_cfg[csv_section]['y_index'] = 2

    x_r, _, smiles = preprocess_dataset(fname, data_cfg, **repr_cfg[utils_section])
    return x_r, smiles

