import os
import sys
import os.path as osp
import numpy as np

from edo.utils import find_and_load


def natural_sort_key_giver(s):
    return int(s.split('_')[-1].split('.')[0])


res_dir = sys.argv[1]


for heree in [h for h in os.listdir(res_dir) if osp.isdir(osp.join(res_dir, h))]:
    print('\n\t', f"Processing {heree}...")
    here = osp.join(res_dir, heree)  

    # partial results might be in their own directory
    partials_dir = [d for d in os.listdir(here) if osp.isdir(osp.join(here, d)) and 'partial' in d]
    assert len(partials_dir) <= 1
    if len(partials_dir) == 1:
        partials_dir = osp.join(here, partials_dir[0])
    else:
        partials_dir = here
    
    try:
        X_full = find_and_load(here, 'X_full')
        smiles = find_and_load(here, 'smiles.npy')
        ys = find_and_load(here, 'true_ys')
        shaps_full = find_and_load(here, 'SHAP_values.npy')
    except IndexError:
        print(f"{heree} not a results directory?")
        continue

    try:
        x_parts = sorted([f for f in os.listdir(partials_dir) if 'X_part' in f], key=natural_sort_key_giver)
        shap_parts = sorted([f for f in os.listdir(partials_dir) if 'SHAP_values_part' in f], key=natural_sort_key_giver)
        
        x_parts_ids = set([natural_sort_key_giver(fname) for fname in x_parts])
        shap_parts_ids = set([natural_sort_key_giver(fname) for fname in shap_parts])
        assert x_parts_ids == shap_parts_ids, f"Subtasks {x_parts_ids.difference(shap_parts_ids)} are missing. ({here})"
        
        x_parts = [find_and_load(partials_dir, f) for f in x_parts]
        X_from_parts = np.concatenate(x_parts, axis=0)
        
        shap_parts = [find_and_load(partials_dir, f) for f in shap_parts]
        if 3 == len(shaps_full.shape):
            shap_from_parts = np.concatenate(shap_parts, axis=1)
        else:
            shap_from_parts = np.concatenate(shap_parts, axis=0)
            
    except ValueError:
        print(f"{osp.basename(here)} was not divided.")
        continue


    assert X_full.shape[0] == smiles.shape[0]       # z tym nie powinno być problemu
    assert X_full.shape[0] == ys.shape[0]           # z tym nie powinno być problemu
    assert np.all(X_full == X_from_parts)           # to jest sprawdzane w merge
    assert X_full.shape[0] == shaps_full.shape[-2]   
    assert np.all(shap_from_parts == shaps_full)    # z tym nie powinno być problemu
    
    try:
        predictions = find_and_load(here, 'predictions')
        assert X_full.shape[0] == predictions.shape[0]  # z tym nie powinno być problemu
    except IndexError:
        print(f"{heree} brak predictions.")
    print(f"{heree} jest OK.")
