import os
import time
import json
import pickle
import shutil
import os.path as osp
import numpy as np


def get_timestamp():
    return time.strftime('%Y-%m-%d_%H-%M')


def make_directory(where, dname):
    """Make directory called dname at where location unless it already exists"""
    try:
        os.makedirs(osp.join(where, dname))
    except FileExistsError:
        pass
    return


def save_predictions(x, y, cv_split, test_x, test_y, smiles, test_smiles, model, saving_dir):
    """Calculate predictions of a model on each fold and save in saving_dir"""

    timestamp = get_timestamp()

    def _save_to_file(smis, true_label, predicted_label, predicted_probabilities, filename):
        if predicted_probabilities is None:
            predicted_probabilities = [None, ] * len(smis)
        try:
            os.makedirs(saving_dir)
        except FileExistsError:
            pass
        with open(osp.join(saving_dir, f"{timestamp}-{filename}"), 'w') as fid:
            fid.write('smiles\ttrue\tpredicted\tclass_probabilities\n')
            for sm, true, pred, proba in zip(smis, true_label, predicted_label, predicted_probabilities):
                fid.write(f"{sm}\t{true}\t{pred}\t{proba}\n")

    # training data
    for idx, (_, indices) in enumerate(cv_split):
        this_x = x[indices]
        this_y = y[indices]
        this_smiles = smiles[indices]
        predicted = model.predict(this_x)
        try:
            proba = model.predict_proba(this_x)
        except (AttributeError, RuntimeError):
            proba = None
        _save_to_file(this_smiles, this_y, predicted, proba, f'train-{idx}.predictions')

    # test data
    predicted = model.predict(test_x)
    try:
        proba = model.predict_proba(test_x)
    except (AttributeError, RuntimeError):
        proba = None
    _save_to_file(test_smiles, test_y, predicted, proba, f'test.predictions')


def save_configs(cfgs_list, directory):
    # copy configs from cfgs_list to directory
    timestamp = get_timestamp()
    for config_file in cfgs_list:
        filename = f"{timestamp}-{osp.basename(config_file)}"
        shutil.copyfile(config_file, osp.join(directory, filename))
    return


def save_as_json(obj, saving_dir, filename):
    # save as json using timestamp
    timestamp = get_timestamp()

    # change numpy arrays to json-edible format
    if isinstance(obj, dict):
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                obj[key] = obj[key].tolist()

    with open(osp.join(saving_dir, f'{timestamp}-{filename}'), 'w') as f:
        json.dump(obj, f, indent=2)
    return


def save_as_pickle(obj, saving_dir, filename):
    # save as pickle using timestamp
    timestamp = get_timestamp()
    with open(osp.join(saving_dir, f'{timestamp}-{filename}.pickle'), 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    return


def save_as_np(obj, saving_dir, filename, allow_pickle):
    # save as numpy using timestamp
    timestamp = get_timestamp()
    np.save(osp.join(saving_dir, f'{timestamp}-{filename}.npy'), obj, allow_pickle=allow_pickle)
    return
