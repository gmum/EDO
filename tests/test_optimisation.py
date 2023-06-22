import os.path as osp
import unittest
import itertools

import numpy as np

from tqdm import tqdm

from edo import make_origin
from edo.optimisation.utils import find_experiment
from edo.optimisation import set_seed, get_random_generator


class TestFindingExperiments(unittest.TestCase):
    """
    Tests:
    - make_origin
    - find_experiment
    """

    def setUp(self):
        self.results_dir = '/home/pocha/shared-lnk/results/pocha/full_clean_data/2022-12-full-clean-data'

    def test_make_origin(self):
        datasets = ['h', 'H', 'human', 'Human', 'HUMAN', 'hUmaN', 'r', 'R', 'rat', 'Rat', 'RAT']
        splits = ['r', 'R', 'random', 'Random', 'RANDOM', 's', 'scaffold', 'Scaffold', 'SCAFFOLD']
        reprs = ['krfp', 'KR', 'kr', 'MACCS', 'maccs', 'MA', 'ma', 'padel', 'PaDEL', 'pubfp', 'PubFP', 'mo128', 'mo512',
                 'mo1024']
        tasks = ['r', 'R', 'reg', 'REG', 'Regression', 'c', 'CLS', 'CLASSIFICATION']
        models = ['NB', 'SVM', 'Trees', 'kNN']

        for origin_tuple in tqdm(itertools.product(datasets, splits, reprs, tasks, models),
                                 desc="Testing finding experiments..."):
            if origin_tuple[-1] == 'NB' and 'r' in origin_tuple[-2].lower():
                # no regression for naive Bayes
                continue
            # print(origin_tuple)
            origin = make_origin(origin_tuple)
            # print(origin)
            exp_path = find_experiment(self.results_dir, 'ml', origin)
            self.assertTrue(osp.exists(exp_path))


class TestRandomGenerator(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.max_size = 1000

    def generate_random_stuff(self, magic_number):

        set_seed(magic_number)
        rng = get_random_generator()
        arr1 = rng.integers(self.max_size, size=self.max_size)
        arr2 = rng.random(self.max_size)
        arr3 = rng.permutation(arr1 + arr2)
        arr4 = rng.choice(arr1 + arr2, size=magic_number)
        return arr1, arr2, arr3, arr4


    def test_random_generator(self):
        print("Testing reproducibility of random generator...")
        for i in range(self.n):
            seed = np.random.randint(self.max_size)
            sample1 = self.generate_random_stuff(seed)
            sample2 = self.generate_random_stuff(seed)

            for s1, s2 in zip(sample1, sample2):
                self.assertTrue(np.all(s1 == s2), f'{s1}!={s2}')



if __name__ == '__main__':
    unittest.main()
