# import os.path as osp
#
import unittest
import numpy as np

from edo.utils import get_all_subfolders, index_of_smiles
from edo.optimisation.utils import load_train_test

from edo._check import assert_binary, assert_strictly_positive_threshold


class Test_check(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.max_size = 1000

    def test_assert_binary(self):
        print("Testing assert_binary...")
        for i in range(self.n):
            arr = np.random.permutation(
                [0] * np.random.randint(0, self.max_size) + [1] * np.random.randint(0, self.max_size))
            assert_binary(arr)  # to powinno przejść

            arr1 = np.random.randint(low=-self.max_size, high=0, size=np.random.randint(1, self.max_size))  # negative
            arr2 = np.random.randint(low=2, high=self.max_size,
                                     size=np.random.randint(1, self.max_size))  # positive >= 2
            zeros = np.zeros(np.random.randint(self.max_size))
            ones = np.ones(np.random.randint(self.max_size))

            with self.assertRaises(AssertionError):
                assert_binary(arr1)  # only negative

            with self.assertRaises(AssertionError):
                assert_binary(arr2)  # only positive >= 2

            with self.assertRaises(AssertionError):
                # anything in range [-max_size, max_size) but definitely nonbinary
                assert_binary(np.random.permutation(np.concatenate((arr1, arr2, zeros, ones))))

    def test_assert_strictly_positive_threshold(self):
        print("Testing assert_strictly_positive_threshold...")
        for i in range(self.n):
            pos = np.random.randint(low=1, high=self.max_size)  # positive
            assert_strictly_positive_threshold(pos)  # to powinno przejść

            with self.assertRaises(AssertionError):
                assert_strictly_positive_threshold(0)

            with self.assertRaises(AssertionError):
                neg = np.random.randint(low=-self.max_size, high=0)  # [-max_size, 0)
                assert_strictly_positive_threshold(neg)


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


if __name__ == '__main__':
    unittest.main()
