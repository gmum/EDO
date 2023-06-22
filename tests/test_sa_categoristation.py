import operator
import numpy as np
from edo.shap_analysis.feature import Feature

from edo.shap_analysis.categorisation.utils import n_zeros_ones, purity, majority


def test_n_zeros_ones(n=1000, max_size=1000):
    print("Testing n_zeros_ones...")

    for i in range(n):
        n_zeros = np.random.randint(0, max_size)
        n_ones = np.random.randint(0, max_size)
        arr = np.random.permutation([0] * n_zeros + [1] * n_ones)
        ans = n_zeros_ones(arr)

        assert (n_zeros, n_ones) == ans, f"{(n_zeros, n_ones)} != {ans}"


def test_purity(n=1000, max_s=100):
    print("Testing purity...")

    for i in range(n):
        pur = np.random.randint(low=50, high=100 + 1)  # high is exclusive
        # we want to test on a variable number of samples
        s = int(np.random.randint(1, max_s) * (1 + np.random.rand(1)))
        n_more = s * pur
        n_less = s * (100 - pur)
        arr0 = np.random.permutation([0] * n_more + [1] * n_less)  # more zeros than ones
        arr1 = np.random.permutation([1] * n_more + [0] * n_less)  # more ones than zeros

        # giving number of ones and number of zeros
        assert pur / 100 == purity(n_more, n_less), f"{pur/100} != {purity(n_more, n_less)}, ({n_more, n_less})"
        assert pur / 100 == purity(n_less, n_more), f"{pur/100} != {purity(n_less, n_more)}, ({n_less, n_more})"
        # giving an array
        assert pur / 100 == purity(a=arr0), f"{pur/100} != {purity(a=arr0)}"
        assert pur / 100 == purity(a=arr1), f"{pur/100} != {purity(a=arr1)}"


def test_majority(n=1000, max_s=100):
    print("Testing majority...")
    for i in range(n):
        max_pur = np.random.randint(52, 100)
        pur = np.random.randint(low=51, high=max_pur)
        # we want n_more and n_less to sum to a variable number of samples
        s = int(np.random.randint(1, max_s) * (1 + np.random.rand(1)))
        n_more = s * pur
        n_less = s * (100 - pur)

        # pur is higher than min_purity=0.5
        assert (0,) == majority(n_more, n_less, min_purity=0.5), f"(0,) != {majority(n_more, n_less, min_purity=0.5)}"
        assert (1,) == majority(n_less, n_more, min_purity=0.5), f"(1,) != {majority(n_less, n_more, min_purity=0.5)}"
        assert (0, 1) == majority(n_less, n_less,
                                  min_purity=0.5), f"(0, 1) != {majority(n_less, n_less, min_purity=0.5)}"
        assert (0, 1) == majority(n_more, n_more,
                                  min_purity=0.5), f"(0, 1) != {majority(n_more, n_more, min_purity=0.5)}"

        # pur is lower than min_purity
        min_purity = (pur + (0.5 * (100 - pur))) / 100
        assert (0, 1) == majority(n_more, n_less,
                                  min_purity=min_purity), f"(0, 1) != {majority(n_more, n_less, min_purity=min_purity)} ({min_purity})"
        assert (0, 1) == majority(n_less, n_more,
                                  min_purity=min_purity), f"(0, 1) != {majority(n_less, n_more, min_purity=min_purity)} ({min_purity})"
        assert (0, 1) == majority(n_less, n_less,
                                  min_purity=min_purity), f"(0, 1) != {majority(n_less, n_less, min_purity=min_purity)} ({min_purity})"
        assert (0, 1) == majority(n_more, n_more,
                                  min_purity=min_purity), f"(0, 1) != {majority(n_more, n_more, min_purity=min_purity)} ({min_purity})"


"""
Below, there are dummy tests for categories:
- high-gain,
- unimportant.

Tests for well-separated features are more general and done in test_shap_analysis.py
"""

N_SAMPLES = 6  # number of samples in dummy features


def dummy_features():
    origin = ('h', 'r', 'mo128', 'c', 'nb')
    # feature definitions
    f1_vals = np.array([0, 0, 0, 0, 1, 1])
    f1_shap = np.array([-1, 0, 0, 1, 2, 3])

    f2_vals = np.array([1, 1, 0, 1, 0, 0])
    f2_shap = np.array([-2, -1, 0, 1, 2, 3])

    f3_vals = np.array([0, 0, 1, 1, 0, 0])
    f3_shap = np.array([-2, -1, 0, 0, 1, 2])

    f4_vals = np.array([0, 1, 0, 1, 0, 1])
    f4_shap = np.array([-2, -1, 0, 0, 1, 2])

    f5_vals = np.array([1, 0, 0, 1, 1, 1])
    f5_shap = np.array([-1, 0, 1, 2, 2, 3])

    f6_vals = np.array([1, 0, 1, 0, 0, 1])
    f6_shap = np.array([-1, 0, 1, 2, 2, 3])

    f7_vals = np.array([0, 0, 1, 1, 0, 1])
    f7_shap = np.array([-1, 0, 0, 1, 1, 2])

    f8_vals = np.array([1, 0, 1, 0, 1, 1])
    f8_shap = np.array([-2, -1, 0, 1, 2, 3])

    features = [(f1_vals, f1_shap), (f2_vals, f2_shap), (f3_vals, f3_shap), (f4_vals, f4_shap), (f5_vals, f5_shap),
                (f6_vals, f6_shap), (f7_vals, f7_shap), (f8_vals, f8_shap)]
    features = [Feature(*f, ftr_index=i, origin=origin, name=f"F{i}") for i, f in enumerate(features, 1)]
    return features


def high_impact_answers():
    answers = {(1, 'absolute'): [3, 4, 4, 2, 4, 3, 3, 3],
               (1, 'ratio'): [3 / 6, 4 / 6, 4 / 6, 2 / 6, 4 / 6, 3 / 6, 3 / 6, 3 / 6],
               (1, 'purity'): [3 / 4, 4 / 5, 1, 2 / 4, 4 / 5, 3 / 5, 3 / 4, 3 / 5],

               (2, 'absolute'): [2, 3, 2, 2, 3, 2, 1, 3],
               (2, 'ratio'): [2 / 6, 3 / 6, 2 / 6, 2 / 6, 3 / 6, 2 / 6, 1 / 6, 3 / 6],
               (2, 'purity'): [1, 1, 1, 1, 1, 2 / 3, 1, 1],

               (3, 'absolute'): [1, 1, 0, 0, 1, 1, 0, 1],
               (3, 'ratio'): [1 / 6, 1 / 6, 0, 0, 1 / 6, 1 / 6, 0, 1 / 6],
               (3, 'purity'): [1, 1, np.nan, np.nan, 1, 1, np.nan, 1],

               (4, 'absolute'): [0, 0, 0, 0, 0, 0],
               (4, 'ratio'): [0, 0, 0, 0, 0, 0],
               (4, 'purity'): [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
               }

    return answers


def unimportant_answers():
    answers = {(1, 'absolute'): [2, 1, 2, 2, 1, 1, 2, 1],
               (2, 'absolute'): [4, 3, 4, 4, 3, 3, 5, 3],
               (3, 'absolute'): [5, 5, 6, 6, 5, 5, 6, 5],
               (4, 'absolute'): [6, 6, 6, 6, 6, 6, 6, 6],
               }

    for miu, metric in list(answers.keys()):
        answers[miu, 'ratio'] = [i / N_SAMPLES for i in answers[miu, 'absolute']]

    return answers


def _test_region_indices(region, s_vals, threshold, lower_than=True):
    # Checks if condition (<= or >= than threshold) is
    # - satisfied for all samples in the region
    # - satisfied for none samples outside of the region
    # s_vals - SHAP values of ALL samples
    if lower_than:
        thr = -threshold
        rel = operator.le
    else:
        thr = threshold
        rel = operator.ge

    assert np.all(rel(s_vals[region.indices], thr)), f"Unsatisfied {s_vals[region.indices]} {rel} {thr}"
    assert np.all(np.logical_or(region.indices, np.logical_not(
        rel(s_vals, thr)))), f"Some samples satisfy (SHAP val {rel} {thr}) but are not included in the region."


def _test_region_metrics(region, f_vals):
    zeros_ones = n_zeros_ones(f_vals)
    assert max(zeros_ones) == region.n_correct, f"Incorrect n_correct {max(zeros_ones)} != {region.n_correct}"
    assert majority(*zeros_ones) == region.majority, f"Incorrect majority {majority(*zeros_ones)} != {region.majority}"
    assert np.isclose(purity(a=f_vals), region.purity,
                      equal_nan=True), f"Incorrect purity {purity(a=f_vals)} != {region.purity}"


def test_high_impact_regions(features):
    print("Testing consistency of high impact regions...")
    for ftr in features:
        # print(f"   Testing {ftr.name}...")
        f_vals = ftr._feature_values
        s_vals = ftr._shap_values

        for chosen_metric in ftr._high_impact.values():
            for solution in chosen_metric.values():
                gamma, metric = solution.params['gamma'], solution.params['metric']
                loss_reg, gain_reg = solution.loss_region, solution.gain_region

                _test_region_indices(loss_reg, s_vals, gamma, lower_than=True)
                _test_region_indices(gain_reg, s_vals, gamma, lower_than=False)

                _test_region_metrics(loss_reg, f_vals[loss_reg.indices])
                _test_region_metrics(gain_reg, f_vals[gain_reg.indices])

                # is range for each region correct?
                assert loss_reg.start == -np.inf and loss_reg.end == -gamma, f"Incorrect range {loss_reg.start} != {-np.inf} or {loss_reg.end} != {-gamma}"
                assert gain_reg.start == gamma and gain_reg.end == np.inf, f"Incorrect range {gain_reg.start} != {gamma} or {gain_reg.end} != {np.inf}"

                # is score correct?
                n_correct = loss_reg.n_correct + gain_reg.n_correct
                n_samples = len(f_vals)
                n_hg_samples = np.sum(loss_reg.indices) + np.sum(gain_reg.indices)

                if metric == 'absolute':
                    ref_score = n_correct
                elif metric == 'ratio':
                    ref_score = n_correct / n_samples
                elif metric == 'purity':
                    ref_score = n_correct / n_hg_samples if n_hg_samples !=0 else np.nan
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                assert np.isclose(solution.score, ref_score,
                                  equal_nan=True), f"Incorrect score {solution.score} != {ref_score}; metric: {metric}."


def iterate_and_test_equality(features, answers, category):
    # for dummy features check if reference answers are correct
    print(f"Testing {category}...")
    for thr, metric in answers.keys():
        for ftr, ref_score in zip(features, answers[(thr, metric)]):
            score = getattr(ftr, category)(thr, metric).score
            assert np.isclose(score, ref_score,
                              equal_nan=True), f"Feature {ftr.name}: score for threshold {thr} and metric {metric} should be {ref_score} is {score}."


if __name__ == "__main__":
    test_n_zeros_ones()
    test_purity()
    test_majority()

    features = dummy_features()

    iterate_and_test_equality(features, high_impact_answers(), 'high_impact')
    iterate_and_test_equality(features, unimportant_answers(), 'unimportant')

    test_high_impact_regions(features)

    print("Done.")
