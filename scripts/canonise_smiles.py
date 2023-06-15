import os
import os.path as osp
import sys
import numpy as np
from rdkit import Chem
from metstab_pred.src.utils import find_and_load
from metstab_pred.src.savingutils import save_npy_and_log_artifact as save_npy

if __name__=="__main__":
    res_dir = sys.argv[1]

    for directory in [osp.join(res_dir, d) for d in os.listdir(res_dir) if osp.isdir(osp.join(res_dir, d))]:
        print(directory)
        if len([os.path.join(directory, f) for f in os.listdir(directory) if '-canonised.npy' in f]) > 0:
            print("Already calculated. Skipping.\n")
            continue
        smiles_order = find_and_load(directory, 'smiles.npy', protocol='numpy')
        canonised = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in smiles_order]
        assert len(smiles_order)==len(canonised)
        save_npy(np.array(canonised), directory, 'canonised', allow_pickle=False)
        print("Calculated and saved.\n")

    print("Success.")
