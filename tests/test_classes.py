import unittest

import numpy as np

from edo import make_origin, Task, TASK_ERROR_MSG
from edo.data import cutoffs_metstabon
from edo.utils import index_of_smiles
from edo.optimisation.feature import make_features
from edo.optimisation.utils import get_present_features
from edo.optimisation.categorisation import SeparationResult, HighImpactResult, RandomRule
from edo.optimisation import Goal
from edo.optimisation.utils import load_train_test, load_shap_files, load_predictions, load_model, find_experiment
from edo.optimisation.rule.generate import derive_well_separated_two_way_rules, derive_high_impact_rules
from edo.optimisation.rule.generate import derive_random_rules_sample
from edo.optimisation.sample import make_samples

from edo.optimisation.utils import get_predictions_before_after, get_predictions_before_after_slow
from edo.optimisation.utils import filter_correct_predictions_only, group_samples, intersection_list, difference_list
from edo.optimisation.rule.filter import filter_rules, condition_well_separated, condition_high_impact
from edo.optimisation.rule.filter import filter_out_unimportant


class TestOptimisationUtils(unittest.TestCase):
    def setUp(self):
        self.small_n = 10
        self.n = 100
        self.big_n = 1000
        self.results_dir = '/home/pocha/shared-lnk/results/pocha/full_clean_data/2022-12-full-clean-data'
        self.check_unlogging = True
        self.make_features_rules_samples()
        self.make_regression_model()

    def make_features_rules_samples(self):
        hi_params = {'gamma': 0.001, 'metric': 'ratio'}
        self.task = Task.CLASSIFICATION
        m1_origin = make_origin(('human', 'random', 'krfp', self.task.value, 'trees'))
        self.model, self.shapcalculator = load_model(self.results_dir, m1_origin, self.check_unlogging)

        mldir = find_experiment(self.results_dir, 'ml', m1_origin)
        train, test = load_train_test(mldir)
        present_features = list(get_present_features(train[0], 0.1))

        shap_m1 = find_experiment(self.results_dir, 'shap', m1_origin)
        shap_smis, shap_x, shap_true_ys, classes_order, shap_vals = load_shap_files(shap_m1, self.task,
                                                                                    self.check_unlogging)

        self.tr_preds, self.test_preds = load_predictions(mldir, self.task)
        tr_smis = index_of_smiles(shap_smis, self.tr_preds.index.tolist())
        te_smis = index_of_smiles(shap_smis, self.test_preds.index.tolist())

        self.my_samples = make_samples(te_smis, shap_x, shap_vals, shap_smis, m1_origin, classes_order, self.task)
        self.my_features = make_features(present_features, tr_smis, shap_x, shap_vals, classes_order, m1_origin,
                                         self.task)

        self.well_separated_rules = []
        self.high_impact_rules = []
        self.random_rules = []
        for ft in self.my_features:
            self.well_separated_rules.extend(derive_well_separated_two_way_rules(ft, self.task))
            self.high_impact_rules.extend(derive_high_impact_rules(ft, hi_params, self.task))
            self.random_rules.extend(derive_random_rules_sample(ft, self.task))

    def make_regression_model(self):
        self.reg_task = Task.REGRESSION
        m1_origin = make_origin(('human', 'random', 'krfp', self.task.value, 'trees'))
        self.reg_model, self.reg_shapcalculator = load_model(self.results_dir, m1_origin, self.check_unlogging)

    def test_get_predictions(self):
        print("Testing get_predictions_before_after...")
        for model, task in ((self.model, self.task), (self.reg_model, self.reg_task)):
            fast_before, fast_after = get_predictions_before_after(self.my_samples, model, task)
            slow_before, slow_after = get_predictions_before_after_slow(self.my_samples, model, task)

            if task == Task.CLASSIFICATION:
                new_shape = (slow_after.shape[0], slow_after.shape[-1])
            elif task == Task.REGRESSION:
                new_shape = (slow_before.shape[0])
            else:
                raise ValueError(TASK_ERROR_MSG(task))

            slow_before = slow_before.reshape(new_shape)
            slow_after = slow_after.reshape(new_shape)

            self.assertTrue(np.all(np.isclose(slow_before, fast_before)))
            self.assertTrue(np.all(np.isclose(slow_after, fast_after)))

    def test_filter_correct_predictions_only(self):
        print("Testing filter_correct_predictions_only...")
        for df in [self.tr_preds, self.test_preds]:
            filtered_smis = filter_correct_predictions_only(df, self.task)
            self.assertEqual(filtered_smis, sorted(filtered_smis),
                             msg=f"Returned result is not sorted: {filtered_smis}")

            for smi in df.index.tolist():
                smi_row = df.loc[smi]
                smi_true = smi_row.loc['true']
                smi_predicted = smi_row.loc['predicted']
                # TODO: te warunki pewnie można jakoś uprościć
                if smi in filtered_smis:
                    # prediction should be correct
                    self.assertEqual(smi_true, smi_predicted, msg=f"{smi_true} != {smi_predicted} ({smi})")

                    if self.task == Task.CLASSIFICATION:
                        # TODO uwaga! ten kawałek zakłada, że classes_order = [0, 1, 2] (CHYBA)
                        # tak samo jak w testowanej funkcji
                        argmax_class = np.argmax([smi_row.loc[c] for c in ['zero', 'one', 'two']])
                        self.assertEqual(smi_true, argmax_class, msg=f'{smi_true} != {argmax_class} ({smi})')
                else:
                    # prediction should not be correct
                    if self.task == Task.CLASSIFICATION:
                        argmax_class = np.argmax([smi_row.loc[c] for c in ['zero', 'one', 'two']])
                        self.assertTrue((smi_true != smi_predicted) or (smi_true != argmax_class))
                    elif self.task == Task.REGRESSION:
                        self.assertNotEqual(smi_true, smi_predicted, msg=f"{smi_true} == {smi_predicted} ({smi})")
                    else:
                        raise ValueError(TASK_ERROR_MSG(self.task))

    def test_group_samples(self):
        print("Testing group_samples...")
        for df in [self.tr_preds, self.test_preds]:
            groups = group_samples(df, self.task)
            ref_vals = [0, 1, 2]

            for smi_list, ref_val in zip(groups, ref_vals):
                self.assertEqual(smi_list, sorted(smi_list), msg=f"Result is not sorted: {smi_list}")

                for smi in smi_list:
                    smi_true = df.loc[smi, 'true']
                    if self.task == Task.REGRESSION:
                        smi_true = cutoffs_metstabon(smi_true, self.unlog)

                    self.assertEqual(smi_true, ref_val, msg=f"{smi_true} != {ref_val} ({smi})")

    def test_intersection_list(self):
        print('Testing intersection_list...')
        for n in range(self.n):
            # up to self.small_n lists of max size self.n and values in [0, self.big_n)
            arrays = [np.random.randint(1 + np.random.randint(self.big_n), size=1 + np.random.randint(self.n)) for i in
                      range(1 + np.random.randint(self.small_n))]
            all_values = set(np.hstack(arrays))
            intersection = intersection_list(*arrays)
            self.assertTrue(set(intersection).issubset(all_values))
            self.assertEqual(intersection, sorted(intersection), msg=f"Result is not sorted: {intersection}")
            for v in all_values:
                if v in intersection:
                    self.assertTrue(np.all([v in arr for arr in arrays]), msg=f"{v} is not present in all {arrays}")
                else:
                    self.assertFalse(np.all([v in arr for arr in arrays]), msg=f"{v} is present in all {arrays}")

    def test_difference_list(self):
        print('Testing difference_list...')
        for n in range(self.n):
            # up to self.small_n lists of max size self.n and values in [0, self.big_n)
            arrays = [np.random.randint(1 + np.random.randint(self.big_n), size=1 + np.random.randint(self.n)) for i in
                      range(1 + np.random.randint(self.small_n))]
            ref_arr = arrays[0]
            all_values = set([v for arr in arrays for v in arr])
            difference = difference_list(*arrays)

            self.assertTrue(set(difference).issubset(ref_arr))
            self.assertEqual(difference, sorted(difference), msg=f"Result is not sorted: {difference}")

            for v in all_values:
                if v in difference:
                    self.assertIn(v, ref_arr)  # zbedne, bo sprawdzamy issubset wczesniej
                    self.assertFalse(np.any([v in arr for arr in arrays[1:]]))
                else:
                    # not present in ref_arr OR present in ref_arr and in any others
                    self.assertTrue(v not in ref_arr or (v in ref_arr and np.any([v in arr for arr in arrays[1:]])))

    def test_filter_rules(self):
        print("Testing filter_rules...")
        all_rules = self.well_separated_rules + self.high_impact_rules + self.random_rules
        # TODO: equality is checked based on dict representation. This is not optimal.
        all_rules_d = [r.as_dict() for r in all_rules]

        # test filtering on goal
        minimisation_rules = filter_rules(all_rules, goal=Goal.MINIMISATION)
        maximisation_rules = filter_rules(all_rules, goal=Goal.MAXIMISATION)
        self.assertEqual(len(all_rules), len(maximisation_rules) + len(minimisation_rules),
                         msg=f"Length mismatch {len(all_rules)} != {len(maximisation_rules)} + {len(minimisation_rules)}")

        for goal, rules in ((Goal.MINIMISATION, minimisation_rules), (Goal.MAXIMISATION, maximisation_rules)):
            for rule in rules:
                self.assertEqual(rule.goal, goal, msg=f"{rule.goal} != {goal}")
                self.assertIn(rule.as_dict(), all_rules_d)

        # test filtering on class
        low_rules = filter_rules(all_rules, cls_name=0)
        med_rules = filter_rules(all_rules, cls_name=1)
        high_rules = filter_rules(all_rules, cls_name=2)
        self.assertEqual(len(all_rules), len(low_rules) + len(med_rules) + len(high_rules),
                         msg=f"Length mismatch {len(all_rules)} != {len(low_rules)} + {len(med_rules)} + {len(high_rules)}")
        for cls, rules in ((0, low_rules), (1, med_rules), (2, high_rules)):
            for rule in rules:
                self.assertEqual(rule.cls_name, cls)
                self.assertIn(rule.as_dict(), all_rules_d)

        # test filtering on condition
        for n in range(self.n):
            score = np.random.rand()
            ws_condition = lambda r: condition_well_separated(r, score)
            # ws_filtered = filter_rules(all_rules, ws_condition)
            ws_filtered_d = [r.as_dict() for r in filter_rules(all_rules, condition=ws_condition)]
            hi_condition = lambda r: condition_high_impact(r, score)
            # hi_filtered = filter_rules(all_rules, hi_condition)
            hi_filtered_d = [r.as_dict() for r in filter_rules(all_rules, condition=hi_condition)]

            for rule in all_rules:
                rule_d = rule.as_dict()
                derivation = rule.derivation[0]
                if isinstance(derivation, RandomRule):
                    self.assertIn(rule_d, ws_filtered_d)
                    self.assertIn(rule_d, hi_filtered_d)
                elif isinstance(derivation, SeparationResult):
                    self.assertEqual(derivation.score >= score, rule_d in ws_filtered_d)
                    self.assertIn(rule_d, hi_filtered_d)
                elif isinstance(derivation, HighImpactResult):
                    self.assertEqual(derivation.score >= score, rule_d in hi_filtered_d)
                    self.assertIn(rule_d, ws_filtered_d)
                else:
                    raise NotImplementedError(f"Derivation type {type(derivation)} is not implemented!")

    @unittest.skip("Nie wiem na razie, jak to zrobić.")
    def test_filter_out_unimportant(self):
        print("Testing filter_out_unimportant...")
        all_rules = self.well_separated_rules + self.high_impact_rules + self.random_rules
        all_s_vals = np.array([s.s_vals for s in self.my_samples]).flatten()
        s_vals_max = np.max(all_s_vals)
        s_vals_min = np.min(all_s_vals)
        s_vals_range = s_vals_max - s_vals_min

        for n in range(self.n):
            max_ratio = np.random.rand()
            miu = (np.random.rand() * s_vals_range) + s_vals_min
            only_important = filter_out_unimportant(all_rules, self.my_features, {'miu': miu}, max_ratio, self.task)
            for rule in all_rules:
                pass  # na razie nie wiem, jak to zrobić


if __name__ == '__main__':
    unittest.main()
