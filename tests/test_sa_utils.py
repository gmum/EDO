import unittest

import os.path as osp

import numpy as np

from edo.utils import get_all_subfolders
from edo.optimisation.utils import load_train_test

from edo.shap_analysis.utils import index_of_smiles


class TestIndexOfSmiles(unittest.TestCase):
    def setUp(self):
        self.directory = '/home/pocha/dane_phd/random_split/ml'
        self.n = 100

    def test_index_of_smiles(self):
        print("Testing index_of_smiles...")
        all_models = get_all_subfolders(self.directory, extend=True)
        for mldir in all_models:

            (_, _, smiles_train), (_, _, smiles_test) = load_train_test(mldir)
            smiles = np.concatenate((smiles_train, smiles_test))

            for i in np.random.choice(range(smiles.shape[0]), size=self.n):
                # for a single SMI we get all indices where it appears
                i_smi = smiles[i]
                i_smi_idx = index_of_smiles(smiles, i_smi)
                assert i in i_smi_idx, f"{i}!={i_smi_idx}, ({i_smi})"
                assert (smiles[i_smi_idx] == i_smi).all(), f"{smiles[i_smi_idx]} != {i_smi}"

                # for more SMIs we get one index per SMI
                j = i // 2  # we need another index
                j_smi = smiles[j]
                ij_smis = [i_smi, j_smi]
                ij_smis_idx = index_of_smiles(smiles, ij_smis)
                assert i_smi == smiles[ij_smis_idx[0]], f"{i_smi} = {smiles[ij_smis_idx[0]]}"
                assert j_smi == smiles[ij_smis_idx[1]], f"{j_smi} = {smiles[ij_smis_idx[1]]}"


if __name__ == "__main__":
    unittest.main()
    print("Done.")
