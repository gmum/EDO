# from: github.com/chemprop/chemprop
# https://github.com/chemprop/chemprop/blob/2ae05928f386fcf3306ce2491a8fc6a1f03655ec/chemprop/data/scaffold.py

import os.path as osp

import logging
import pandas as pd
from tqdm import tqdm
from random import Random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import metstab_pred.src.training.split_utils as msu


def make_mol(s: str, keep_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.
    
    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize = False)
        Chem.SanitizeMol(mol, sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    else:
        mol = Chem.MolFromSmiles(s)
    return mol


def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h = False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = mol, includeChirality = include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total = len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def load_precomputed_scaffold(mols, mols_df, path):
    """
    Load scaffold clusters from a file.
    :param mols: list of SMILEs to calculate scaffold for
    :param mols_df: pandas.DataFrame with the same SMILES as in mols and their CHEMBL_ID
    :param path: path to csv file with cluster assignment based on scaffold, must include CHEMBL_ID
    """
    precomputed = pd.read_csv(path)
    precomputed = mols_df.join(precomputed.set_index('CHEMBL_ID'), on='CHEMBL_ID')

    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(list(mols)), total = len(mols)):
        row = precomputed[precomputed.SMILES==mol]
        scaffold = row.cluster_index.values[0]
        scaffolds[scaffold].add(i)

    return scaffolds


def scaffold_split(data: List[str],
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = True,
                   seed: int = 0,
                   logger: logging.Logger = None,
                   precomputed_scaffold_kwargs = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits a list of smiles by scaffold so that no molecules sharing a scaffold are in different splits.
    :param data: A list of smiles.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param balanced: Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    :param seed: Random seed for shuffling when doing balanced splitting.
    :param logger: A logger for recording output.
    :param precomputed_scaffold: path to csv file with precomputed scaffolds
    :return: A tuple of lists with indices that define the train, validation, and test splits of the data.
    """
    
    assert abs(1 - sum(sizes)) < 1e-05

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    if precomputed_scaffold_kwargs is None:
        scaffold_to_indices = scaffold_to_smiles(data, use_indices=True)
    else:
        scaffold_to_indices = load_precomputed_scaffold(data, **precomputed_scaffold_kwargs)
    
    # Seed randomness
    random = Random(seed)

    # sort scaffolds (represented as sets of indices)
    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    # assign scaffolds to train/validation/test
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')
    else:
        print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')

    if logger is not None:
        log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return train, val, test


if __name__ == "__main__":
    
    def make_scaffold_split(data, seed, precomputed=None):
        # 10% is test, 3 equal folds is rest
        if precomputed is not None:
            precomputed['mols_df'] = data

        folds2_3, fold1, test = scaffold_split(list(data.SMILES),[0.6, 0.3, 0.1], precomputed_scaffold_kwargs=precomputed, seed=seed)
        fold2, _, fold3 = scaffold_split(folds2_3,[0.5, 0, 0.5], precomputed_scaffold_kwargs=precomputed, seed=seed)

        print(len(fold1), len(fold2), len(fold3), len(test))

        # apply split to DataFrame
        fold1_df = data[data.SMILES.isin(fold1)]
        fold2_df = data[data.SMILES.isin(fold2)]
        fold3_df = data[data.SMILES.isin(fold3)]
        test_df = data[data.SMILES.isin(test)]

        folds = [fold1_df, fold2_df, fold3_df]

        assert sum(df.shape[0] for df in folds + [test_df, ]) == data.shape[0]

        return folds, test_df
    
    
    human_seed = 19657
    human_data = msu.clean_data(msu.load_data(msu.HUMAN))
    human_precomputed_path = osp.join("..", "..", 'data', 'scaffold_clustering_from_Sabina', 'human_all_data_clustered-2021-12.csv')
    human_folds, human_test = make_scaffold_split(human_data, human_seed, precomputed={'path': human_precomputed_path})
    msu.save_split_data(human_folds, human_test, saving_directory=msu.DATA_DIR, prefix=f"human_scaffold_{human_seed}")

    rat_seed = 84741
    rat_data = msu.clean_data(msu.load_data(msu.RAT))
    rat_precomputed_path = osp.join("..", "..", 'data', 'scaffold_clustering_from_Sabina', 'rat_all_clustered_data-2021-12.csv')
    rat_folds, rat_test = make_scaffold_split(rat_data, rat_seed, precomputed={'path': rat_precomputed_path})
    msu.save_split_data(rat_folds, rat_test, saving_directory=msu.DATA_DIR, prefix=f"rat_scaffold_{rat_seed}")
    
    print("Done")
