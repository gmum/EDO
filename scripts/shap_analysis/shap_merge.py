import os
import sys

import numpy as np

from edo.utils import find_and_load
from edo.savingutils import save_npy_and_log_artifact, LoggerWrapper


n_args = 1 + 1


if __name__=='__main__':
    if len(sys.argv) != n_args:
        print(f"Usage: python {sys.argv[0]} working_directory")
        quit(1)

    work_dir = sys.argv[1]

    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    logger_wrapper = LoggerWrapper(work_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:]}')

    def sorting_func(x):
        return int(x.split('.')[0].split('_')[-1])

    # concatenating X and checking equality
    files = sorted([os.path.join(work_dir, f) for f in os.listdir(work_dir) if 'X_part_' in f], key=sorting_func)
    arrays = [np.load(f, allow_pickle=False) for f in files]
    X_from_parts = np.vstack(arrays)
    X_full = find_and_load(work_dir, '-X_full.npy', protocol='numpy')
    assert np.all(X_from_parts == X_full), f"X_from_parts is not equal to X_full. The difference is: {np.sum(np.abs(X_full-X_from_parts))}."

    # expected_values should be equal
    files = sorted([os.path.join(work_dir, f) for f in os.listdir(work_dir) if 'expected_values_part_' in f], key=sorting_func)
    arrays = [np.load(f, allow_pickle=False) for f in files]
    for i in range(len(arrays)):
        assert np.all(arrays[0] == arrays[i]), f"Expected values 0 are not equal to expected values {i}."
    save_npy_and_log_artifact(arrays[0], work_dir, 'expected_values', allow_pickle=False)

    # concatenating shap values and saving
    files = sorted([os.path.join(work_dir, f) for f in os.listdir(work_dir) if 'SHAP_values_part_' in f], key=sorting_func)
    arrays = [np.load(f, allow_pickle=False) for f in files]
    if 3==len(arrays[0].shape):
        # classification
        shap_vals_from_parts = np.concatenate(arrays, axis=1)
    else:
        # regression
        shap_vals_from_parts = np.concatenate(arrays, axis=0)
    save_npy_and_log_artifact(shap_vals_from_parts, work_dir, 'SHAP_values', allow_pickle=False)
