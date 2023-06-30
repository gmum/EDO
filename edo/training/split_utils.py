import os
import pandas as pd

"""A set of functions for working on raw data from Sabina."""

# CONSTANTS
DATA_DIR = os.path.abspath(os.path.join("..", "..", "data"))  # must be set for everyone independently
HUMAN = "t12_human_data_all.csv"
RAT = "t12_rat_data_all.csv"
MOUSE = "t12_mouse_data_all.csv"

SMILES = "SMILES"
STABILITY_SCORE = "STABILITY_SCORE"
CHEMBL_ID = "CHEMBL_ID"
HR = "HR"
SOURCE = "SOURCE"
T12 = "T12"  # human only


def load_data(data_id):
    """
    Load dataset and set column names for later use.
    :param data_id: str: dataset id, must include 'human', 'rat' or 'mouse'
    :return: pd.DataFrame: dataset with renamed columns
    """
    if 'human' in data_id.lower():
        return pd.read_csv(os.path.join(DATA_DIR, HUMAN), header=None,
                           names=(SMILES, SOURCE, HR, STABILITY_SCORE, T12, CHEMBL_ID))
    elif 'rat' in data_id.lower():
        return pd.read_csv(os.path.join(DATA_DIR, RAT), header=None,
                           names=(SMILES, SOURCE, HR, STABILITY_SCORE, CHEMBL_ID))
    elif 'mouse' in data_id.lower():
        return pd.read_csv(os.path.join(DATA_DIR, MOUSE), header=None,
                           names=(SMILES, SOURCE, HR, STABILITY_SCORE, CHEMBL_ID))
    else:
        raise ValueError('data_id not recognised')


def clean_data(data):
    """
    Select rows based on a defined set of rules. Select only required columns.
    :param data: pd.DataFrame: data loaded with load_data
    :return: pd.DataFrame with SMILES, stability score and CHEMBL ID
    """
    # 1. restricting to 'Liver microsomes', 'Liver microsome', and 'Liver'
    liver_like_source = ['Liver', 'Liver microsome', 'Liver microsomes']
    liver_filter = data[SOURCE].isin(liver_like_source)
    cleaned_data = data[liver_filter]

    # 2. restricting to t1/2
    if T12 in cleaned_data:
        # it's in if, 'cause only humans have information about it
        t12_filter = cleaned_data[T12].isin(['T1/2', ])
        cleaned_data = cleaned_data[t12_filter]

    # 3. restricting to hr
    hr_filter = cleaned_data[HR].isin(['hr', ])
    cleaned_data = cleaned_data[hr_filter]

    # 4. we only care about SMILES,stability score, and CHEMBL ID
    cleaned_data = cleaned_data[[SMILES, STABILITY_SCORE, CHEMBL_ID]]

    return cleaned_data


def save_split_data(folds, test, saving_directory=DATA_DIR, prefix=''):
    """
    Save split datasets.
    :param folds: list: pd.DataFrames with cross-validation folds
    :param test: pd.DataFrame: test set
    :param saving_directory: where should the data be stored
    :param prefix: str: file name prefix - helps to identify the dataset, the splitting method, etc
    :return: None
    """
    test.to_csv(os.path.join(saving_directory, f"{prefix}-test.csv"))

    for idx, fold in enumerate(folds):
        fold.to_csv(os.path.join(saving_directory, f"{prefix}-fold-{idx+1}.csv"))
    return
