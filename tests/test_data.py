import os.path as osp
import numpy as np

from edo.utils import get_all_subfolders, get_configs_and_model, find_and_load, usv
from edo.optimisation.utils import load_train_test

from edo.data import log_stability, unlog_stability
from edo.wrappers import Unloger


def test_log_unlog_stability(n=100, m=200):
    print("Testing log unlog stability...")

    def _change_range(x, m=m):
        x = 1 - x  # [0,1) -> (0,1]
        x = x * m  # (0,1] -> (0,200]
        return x

    def _test(x):
        assert np.isclose(x, log_stability(unlog_stability(x))).all(), f"Log-unlog doesn't work for {x}"
        assert np.isclose(x, unlog_stability(log_stability(x))).all(), f"Unlog-log doesn't work for {x}"

    # 1D np.array
    a = _change_range(np.random.random_sample(n))
    _test(a)

    # 2D np.array
    a = _change_range(np.random.random_sample((n, n)))
    _test(a)

    # list/tuple of floats
    a = _change_range(np.random.random_sample(n))
    _test(list(a))
    _test(tuple(a))

    # list/tuple of ints
    a = _change_range(np.random.random_sample(n))
    _test([int(a) + 1 for a in a])
    _test(tuple(int(a) + 1 for a in a))

    # int or float
    for i in range(n):
        a = usv(_change_range(np.random.random_sample(1)))
        _test(float(a))
        _test(int(a) + 1)


def test_Unlogger(exp_dir):
    print("Testing Unlogger...")
    all_models = get_all_subfolders(exp_dir, extend=True)
    for mldir in all_models:
        if '-r-' not in osp.basename(mldir):
            continue

        print('\t', osp.basename(mldir))
        (x_train, _, _), (x_test, _, _) = load_train_test(mldir)
        _, _, _, _, model_pickle = get_configs_and_model(mldir)

        just_model = find_and_load(mldir, osp.basename(model_pickle), protocol='pickle')
        unlg_model = find_and_load(mldir, osp.basename(model_pickle), protocol='pickle')
        unlg_model = Unloger(unlg_model)

        for x in [x_train, x_test]:
            just_preds = just_model.predict(x)
            unlg_preds = unlg_model.predict(x)

            assert np.isclose(just_preds, log_stability(unlg_preds)).all()
            assert np.isclose(unlog_stability(just_preds), unlg_preds).all()


if __name__ == "__main__":
    directory = '/home/pocha/dane_phd/random_split/ml'

    test_log_unlog_stability()
    test_Unlogger(directory)

    print("Done.")
