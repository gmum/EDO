import os
import sys
import shutil

import os.path as osp
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from metstab_pred.src.utils import find_and_load
from metstab_pred.src.config import parse_representation_config, utils_section
from metstab_pred.src.savingutils import save_npy_and_log_artifact

    
def calculate_morgan(smi, morgan_nbits):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=morgan_nbits)
    fp = [int(i) for i in fp.ToBitString()]
    return fp
    
    
if __name__=="__main__":
    mor_cfg = sys.argv[1]  # morgan configuration file
    res_dir = sys.argv[2]  # folder with shap folders
    morgan_cfg = parse_representation_config(mor_cfg)
    mor_bits = morgan_cfg[utils_section]['morgan_nbits']

    for directory in [osp.join(res_dir, d) for d in os.listdir(res_dir) if osp.isdir(osp.join(res_dir, d))]:
        print(directory)
        
        try:
            _ = find_and_load(directory, 'morgans.npy', protocol='numpy')
            print("Already calculated. Skipping.\n")
            continue
        except IndexError:
            pass
        
        smiles_order = find_and_load(directory, 'smiles.npy', protocol='numpy')
        morganised = [calculate_morgan(smi, mor_bits) for smi in smiles_order]

        assert len(smiles_order)==len(morganised)
        save_npy_and_log_artifact(np.array(morganised), directory, 'morgans', allow_pickle=False)
        shutil.copyfile(mor_cfg, osp.join(directory, f'MORGAN_{mor_bits}.cfg'))
        print("Calculated and saved.\n")

    print("Success.")
