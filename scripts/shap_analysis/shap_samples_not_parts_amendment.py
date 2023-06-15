import os
import neptune
import numpy as np
from metstab_pred.src.utils import find_and_load as fal
from metstab_pred.src.savingutils import save_npy_and_log_artifact

# neptune configuration
neptune.init('lamiane/metstab-shap')
version_tag = "C"
modus_operandi_tag = "samples not parts"
tags=['metstab-shap', version_tag, modus_operandi_tag]

# constants
n_samples = 2
path = 'shap/h-pubfp-r-svm'
X_full = fal(path, '2022-06-07_13-52-X_full.npy')
smiles = fal(path, '2022-06-07_13-52-smiles.npy')

# make neptune experiment
nexp = neptune.create_experiment(name=path,
                                 params={'out dir': path},
                                 tags=tags,
                                 upload_source_files=os.path.join(os.path.dirname(os.path.realpath(__file__)), '*.py'))


parts = [f for f in os.listdir(path) if 'SHAP' in f]
ids = [int(p.split('_')[-1].split('.')[0]) for p in parts]
assert set(ids) == set(range(1, 1200))


n_calculated = sum([fal(path, p).shape[0] for p in parts])

i = len(parts)
start = n_calculated
end = start + n_samples

while end <= len(X_full):
    save_npy_and_log_artifact(X_full[start:end, :], path, f'X_part_{i+1}', allow_pickle=False, nexp=nexp)
    i = i+1
    start = end
    end = start + n_samples

# last part goes untill the end
if start < X_full.shape[0]:
    save_npy_and_log_artifact(X_full[start:, :], path, f'X_part_{i+1}', allow_pickle=False, nexp=nexp)


# concatenating X and checking equality
def sorting_func(x):
    return int(x.split('.')[0].split('_')[-1])

files = sorted([os.path.join(path, f) for f in os.listdir(path) if 'X_part_' in f], key=sorting_func)
arrays = [np.load(f, allow_pickle=False) for f in files]
X_from_parts = np.vstack(arrays)
X_full = fal(path, '-X_full.npy', protocol='numpy')
assert np.all(X_full.shape == X_from_parts.shape), f"Shape mismatch! {X_full.shape} != {X_from_parts.shape}."
assert np.all(X_from_parts == X_full), f"X_from_parts is not equal to X_full. The difference is: {np.sum(np.abs(X_full-X_from_parts))}."

print('Success!')
