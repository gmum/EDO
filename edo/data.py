# code from gmum/MLinPL2019_cheminfo_workshops and gmum/geo-gcn
import os
import os.path as osp
from functools import lru_cache

import hashlib
import tempfile
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

try:
    from padelpy import padeldescriptor  # required to calculate KRFP, PubFP, PaDEL 1D&2D
except ModuleNotFoundError:
    print("PADELPY NOT FOUND.")

from .config import csv_section, utils_section

DATA = 'DATA'
test = 'test'


def load_data(data_config, fingerprint, morgan_nbits=None):
    datasets = []
    indices = []
    this_start = 0
    for path in sorted(data_config[DATA].values()):
        x, y, smiles = preprocess_dataset(path=path, data_config=data_config,
                                          fingerprint=fingerprint, morgan_nbits=morgan_nbits)
        datasets.append((x, y, smiles))
        indices.append((this_start, this_start+len(y)))
        this_start += len(y)

    x = np.vstack([el[0] for el in datasets])
    y = np.hstack([el[1] for el in datasets])
    smiles = np.hstack([el[2] for el in datasets])

    cv_split = get_cv_split(indices)

    # test set
    test_x, test_y, test_smiles = preprocess_dataset(path=data_config[utils_section][test],
                                                     data_config=data_config,
                                                     fingerprint=fingerprint,
                                                     morgan_nbits=morgan_nbits)

    return x, y, cv_split, test_x, test_y, smiles, test_smiles


def load_data_from_df(dataset_paths, smiles_index, y_index, skip_line=False, delimiter=',', scale=None, average=None):
    """
    Load multiple files from csvs, concatenate and return smiles and ys
    :param dataset_paths: list: paths to csv files with data
    :param smiles_index: int: index of the column with smiles
    :param y_index: int: index of the column with the label
    :param skip_line: boolean: True if the first line of the file contains column names, False otherwise
    :param delimiter: delimeter used in csv
    :param scale: should y be scaled? (useful with skewed distributions of y)
    :param average: if the same SMILES appears multiple times how should its values be averaged?
    :return: (smiles, labels) - np.arrays
    """

    # column names present in files?
    header = 0 if skip_line else None

    # reading all the files
    dfs = []
    for data_path in dataset_paths:
        dfs.append(pd.read_csv(data_path, delimiter=delimiter, header=header))

    # merging
    data_df = pd.concat(dfs)

    # scaling
    if scale is not None:
        if 'sqrt' == scale.lower().strip():
            data_df.iloc[:, y_index] = np.sqrt(data_df.iloc[:, y_index])
        elif 'log' == scale.lower().strip():
            data_df.iloc[:, y_index] = np.log(1 + data_df.iloc[:, y_index])
        else:
            raise NotImplementedError(f"Scale {scale} is not implemented.")

    # averaging when one smiles has multiple values
    if average is not None:
        smiles_col = data_df.iloc[:, smiles_index].name
        y_col = data_df.iloc[:, y_index].name

        data_df = data_df.loc[:, [smiles_col, y_col]]  # since now: smiles is 0, y_col is 1, dropping other columns
        smiles_index = 0
        y_index = 1
        if 'median' == average.lower().strip():
            data_df[y_col] = data_df[y_col].groupby(data_df[smiles_col]).transform('median')
        else:
            raise NotImplementedError(f"Averaging {average} is not implemented.")

    # breaking into x and y
    data_df = data_df.values
    data_x = data_df[:, smiles_index]
    data_y = data_df[:, y_index]

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    return data_x, data_y


def preprocess_dataset(path, data_config, fingerprint, morgan_nbits=None):
    if fingerprint == 'morgan':
        assert morgan_nbits is not None, 'Parameter `morgan_nbits` must be set when using Morgan fingerprint.'

    smiles, labels = load_data_from_df([path,], **data_config[csv_section])
    x = []
    y = []
    calculated_smiles = []

    # for other representations we go smiles by smiles because some make rdkit throw errors
    for this_smiles, this_label in zip(smiles, labels):
        try:
            mol = Chem.MolFromSmiles(this_smiles)
            if fingerprint == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=morgan_nbits)
                fp = [int(i) for i in fp.ToBitString()]
            elif fingerprint == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
                fp = np.array(fp)[1:]  # index 0 is unset
            elif fingerprint == 'krfp':
                fp = krfp(this_smiles)
            elif fingerprint == 'padel':
                fp = padel_1D2D(this_smiles)
            elif fingerprint == 'pubfp':
                fp = pubfp(this_smiles)
            else:
                # unknown fingerprint
                raise ValueError(f"Only `morgan`, `maccs`, `krfp`, `padel` and `pubfp` are accepted values. Found: {fingerprint}")
            x.append(fp)
            y.append(this_label)
            calculated_smiles.append(this_smiles)
        except Exception as e:
            print(f'For smiles {this_smiles} the error is:')
            print(e, '\n')
    return np.array(x), np.array(y), calculated_smiles


def padel_assert(pattern_hash, pattern_filepath):
    """    check if pattern_filename has the proper content.    """
    # padel here means PaDEL program, not PaDEL fingeprprint
    with open(pattern_filepath, 'r') as desc_file:
        desc_file_content = desc_file.read()
        
    m = hashlib.md5()
    m.update(desc_file_content.encode('utf-8'))
    assert m.hexdigest() == pattern_hash, f"File {pattern_filepath} has improper content."


def padel_calculate(smi, padel_kwargs):
    """calculate representation of smi with padel"""
    # on prometheus we should use SCRATCH, everywhere else the default location is fine
    with tempfile.TemporaryDirectory(dir=os.getenv('SCRATCH', None)) as tmpdirname:
        smi_file = os.path.join(tmpdirname, "molecules.smi")
        with open(smi_file, 'w') as sf:
            sf.write(smi)
        out = os.path.join(tmpdirname, "out.csv")
        padeldescriptor(mol_dir=smi_file, d_file=out, retainorder=True, threads=5, **padel_kwargs)
        fp = pd.read_csv(out).values[:,1:].reshape((-1)).astype(int)
        return fp


def padel_1D2D(smi):
    """"calculate PaDEL 1D&2D descriptor of smi"""
    pattern_filepath = osp.join(osp.dirname(osp.realpath(__file__)), 'descriptors_padelfp.xml')
    pattern_hash = 'fb1788d709b5ce54fc546f671456c962'
    padel_assert(pattern_hash, pattern_filepath)

    padel_kwargs = {'fingerprints': False, 'd_2d': True, 'descriptortypes': pattern_filepath}
    fp = padel_calculate(smi, padel_kwargs)
    return fp


def pubfp(smi):
    """"calculate PubChem fingerprint of smi"""
    pattern_filepath = osp.join(osp.dirname(osp.realpath(__file__)), 'descriptors_pubfp.xml')
    pattern_hash = '04ac1eb1f136aaafb5e858edb2ee67de'
    padel_assert(pattern_hash, pattern_filepath)
    
    fp = padel_calculate(smi, {'fingerprints': True, 'descriptortypes': pattern_filepath})
    return fp

@lru_cache(maxsize=None)
def krfp(smi):
    """"calculate Klekota-Roth fingeprint of smi"""
    pattern_filepath = osp.join(osp.dirname(osp.realpath(__file__)), 'descriptors_krfp.xml')
    pattern_hash = 'f6145f57ff346599b907b044316c4e71'
    padel_assert(pattern_hash, pattern_filepath)
    
    fp = padel_calculate(smi, {'fingerprints': True, 'descriptortypes': pattern_filepath})
    return fp


def get_cv_split(indices):
    iterator = []
    for val_indices in indices:
        train_indices = []
        for idxs in [list(range(*i)) for i in indices if i != val_indices]:
            train_indices.extend(idxs)
        val_indices = list(range(*val_indices))

        assert len(train_indices) + len(val_indices) == len(set(train_indices + val_indices))

        iterator.append((np.array(train_indices), np.array(val_indices)))
    return iterator


def log_stability(values):
    if isinstance(values, (list, tuple)):
        return [np.log(1+v) for v in values]
    else:
        # for int, float, np.array it'll work, for else - IDK
        return np.log(1+values)


def unlog_stability(values):
    if isinstance(values, (list, tuple)):
        return [np.exp(v)-1 for v in values]
    else:
        return np.exp(values) - 1
    

class Unlogger(object):
    """Unloggs values of regressors when calculating SHAP (for classifiers it makes no sense)."""
    def __init__(self, model):
        if isinstance(model, Pipeline):
            if not isinstance(model.steps[-1][1], RegressorMixin):
                raise TypeError(f"`model` must be a regressor, is {type(model.steps[-1][1])}.")
        elif not isinstance(model, RegressorMixin):
            raise TypeError(f"`model` must be a regressor, is {type(model)}.")
            
        self.model = model
        self.unlog = unlog_stability
        self.MemoryError = False
    
    def predict(self, X):
        try:
            return self.unlog(self.model.predict(X))
        except MemoryError:
            if not self.MemoryError:
                self.MemoryError = True
                print("MemoryError workaround: this is gonna be sloooooow...")
            return self.unlog(np.array([self.model.predict(x.reshape(1, -1)) for x in X]).reshape(-1))


def cutoffs_metstabon(values, log_scale):
    """Changes regression to classification
    according to cutoffs from MetStabOn - Online Platform for Metabolic Stability Predictions (Podlewska & Kafel)
    values - np.array of metabolic stabilities
    log_scale - boolean indicating if the stability values are in log-scale (True) or not (False)
    """

    # y <= 0.6 - low
    # 0.6 < y <= 2.32 - medium
    # 2.32 < y - high

    low = 0
    medium = 1
    high = 2

    bottom_threshold = 0.6
    top_threshold = 2.32

    if log_scale:
        bottom_threshold = log_stability(bottom_threshold)
        top_threshold = log_stability(top_threshold)

    if isinstance(values, np.ndarray):
        classification = np.ones(values.shape, dtype=int)
        classification[values<=bottom_threshold] = low
        classification[values>top_threshold] = high
    elif isinstance(values, float):
        if values <= bottom_threshold:
            return low
        else:
            return medium if values <= top_threshold else high
    else:
        raise NotImplementedError(f"Supported types for `values` are numpy.ndarray and float, is {type(values)}")

    return classification
