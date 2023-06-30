import os
import os.path as osp
import collections
import json
import pickle
from typing import Iterable

import numpy as np
from .config import parse_data_config, parse_representation_config, parse_task_config, parse_model_config


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


def usv(it):
    """Unpack single value with guarantees"""
    assert isinstance(it, collections.Iterable)
    if len(it) == 1:
        return it[0]
    elif len(it) == 0:
        return None
    else:
        raise ValueError(f'len(it)={len(it)}')


def get_all_subfolders(path, extend=False):
    """
    List all subdirectories in path.
    path: str: path
    extend: boolean: return absolute paths?
    """
    subfolders = [folder for folder in os.listdir(path) if osp.isdir(osp.join(path, folder))]
    if extend:
        subfolders = [osp.join(path, f) for f in subfolders]
    return subfolders


def get_all_files(path, extend=False):
    """
    List all files in path.
    path: str: path
    extend: boolean: return absolute paths?
    """
    files = [folder for folder in os.listdir(path) if osp.isfile(osp.join(path, folder))]
    if extend:
        files = [osp.join(path, f) for f in files]
    return files


def get_configs_and_model(folder_path):
    """Load configs from folder_path and determine a path to the pickled model"""
    configs = [cfg for cfg in get_all_files(folder_path, extend=True) if cfg.endswith('.cfg')]
    data_cfg = parse_data_config(usv([dc for dc in configs if 'rat' in dc or 'human' in dc]))
    repr_cfg = parse_representation_config(
        usv([rc for rc in configs if rc.endswith(('maccs.cfg', 'padel.cfg', 'fp.cfg')) or 'morgan' in rc]))
    task_cfg = parse_task_config(usv([tc for tc in configs if 'regression' in tc or 'classification' in tc]))

    try:
        model_cfg = parse_model_config(usv([mc for mc in configs if mc.endswith(('nb.cfg', 'svm.cfg', 'trees.cfg'))]))
    except TypeError as err:
        # knn has no config file
        if 'knn' in folder_path:
            model_cfg = None
        else:
            raise err

    model_pickle = usv([pkl for pkl in get_all_files(folder_path, extend=True) if pkl.endswith('model.pickle')])

    return data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle


def find_and_load(directory, pattern, protocol='numpy'):
    """Scan the directory to find a filename matching the pattern and load it using numpy, pickle or json protocol."""
    fname = usv([osp.join(directory, f) for f in os.listdir(directory) if pattern in f])
    if protocol == 'numpy':
        arr = np.load(fname, allow_pickle=False)
    elif protocol == 'pickle':
        with open(fname, 'rb') as f:
            arr = pickle.load(f)
    elif protocol == 'json':
        with open(fname, 'r') as f:
            arr = json.load(f)
    else:
        raise NotImplementedError(f"Protocol must be `numpy`, `pickle` or `json`. Is {protocol}.")
    return arr
