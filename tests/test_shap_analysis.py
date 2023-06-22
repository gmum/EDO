import os
import sys

import os.path as osp
import numpy as np
import pandas as pd

from tqdm import tqdm

from edo import Task, TASK_ERROR_MSG
from edo.utils import get_configs_and_model
from edo.config import utils_section, csv_section
from edo.data import unlog_stability

from edo.shap_analysis import Category
from edo.shap_analysis.utils import load_shap_files, load_ml_files
from edo.shap_analysis.preprocessing import get_smiles_true_predicted, get_smiles_correct, get_smiles_stability_value
from edo.shap_analysis.preprocessing import get_present_features, filter_samples
from edo.shap_analysis.preprocessing import e, enough

from edo.shap_analysis.analyses import find_optimal_separation_point, situation_at_threshold
from edo.shap_analysis.categorisation import well_separated
from edo.shap_analysis.categorisation.utils import purity
from edo.shap_analysis.categorisation.test_separation_point import find_optimal_separation_point as kfind, SeparationType


def test_get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order):
    smiles_true_predicted_df = get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order)
    alternate = np.array(list(zip(smiles_order, true_ys, preds)), dtype='object')
    
    for i in range(len(alternate)):
        al_smi = alternate[i][0]
        al_true = alternate[i][1]
        al_pred = alternate[i][2]

        smiles_true_predicted_df.loc[al_smi].true == al_true

        if task == Task.REGRESSION:
            assert smiles_true_predicted_df.loc[al_smi].predicted == al_pred, print(smiles_true_predicted_df.loc[al_smi].predicted - al_pred)
        elif task == Task.CLASSIFICATION:
            class_columns = [Category(i).name for i in  classes_order]
            assert all(smiles_true_predicted_df.loc[al_smi, class_columns] == al_pred), print(smiles_true_predicted_df.loc[al_smi].predicted - al_pred)
        else:
            raise ValueError(TASK_ERROR_MSG(task))


def test_get_smiles_correct(smiles_true_predicted_df, task, task_cfg, data_cfg, classes_order):
    correct_smiles = get_smiles_correct(smiles_true_predicted_df, task, task_cfg, data_cfg, classes_order)
    
    if data_cfg[csv_section]['scale'] != 'log':
        raise NotImplementedError
    log_scale = True

    correct_smis = []

    if task == Task.REGRESSION:
        for smi in smiles_true_predicted_df.index:
            true_h = unlog_stability(smiles_true_predicted_df.loc[smi].true)
            predicted_h = unlog_stability(smiles_true_predicted_df.loc[smi].predicted)

            if (1-e)*true_h <= predicted_h <= (1+e)* true_h or (predicted_h >= enough and true_h >= enough):
                correct_smis.append(smi)
        
    elif task == Task.CLASSIFICATION:
        for smi in smiles_true_predicted_df.index:
            true_class = task_cfg[utils_section]['cutoffs'](smiles_true_predicted_df.loc[smi].true, log_scale)
            predicted_class = np.argmax([smiles_true_predicted_df.loc[smi].UNSTABLE,
                                         smiles_true_predicted_df.loc[smi].MEDIUM,
                                         smiles_true_predicted_df.loc[smi].STABLE])

            if true_class == predicted_class:
                correct_smis.append(smi)
        
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    assert set(correct_smis) == correct_smiles, correct_smiles.symmetric_difference(set(correct_smis))


def test_get_smiles_stability_value(smiles_true_predicted_df, data_cfg, task_cfg):
    low, med, high = get_smiles_stability_value(smiles_true_predicted_df, data_cfg, task_cfg)

    if data_cfg[csv_section]['scale'] != 'log':
        raise NotImplementedError
    log_scale = True

    low_smis, med_smis, high_smis = [], [], []

    for smi in smiles_true_predicted_df.index:
        true_class = task_cfg[utils_section]['cutoffs'](smiles_true_predicted_df.loc[smi].true, log_scale)

        if true_class == 0:
            low_smis.append(smi)
        elif true_class == 1:
            med_smis.append(smi)
        else:
            high_smis.append(smi)

    assert low == set(low_smis)
    assert med == set(med_smis)
    assert high == set(high_smis)


def test_get_present_features(x_train, threshold):
    satisfied = get_present_features(x_train, threshold)
    
    features = []
    for i in range(x_train.shape[1]):
        this_feature = x_train[:, i]
        non_zero_elements = np.sum(this_feature.astype(bool))

        if (non_zero_elements/x_train.shape[0] >= threshold) and (non_zero_elements/x_train.shape[0] <= 1- threshold):
            features.append(i)
            
    assert sorted(list(set(features))) == satisfied


def test_filter_samples(my_smis, my_feats, X_full, X_full_df, shap_values, task, smiles_order):
    to_analyse_X, to_analyse_df, to_analyse_shaps, smi_order, mol_indices, feature_order = filter_samples(my_smis, my_feats, X_full, shap_values, task, smiles_order)

    assert np.all(smiles_order[mol_indices] == smi_order)

    # shape
    assert np.all(to_analyse_X.shape[-2:] == (len(my_smis), len(my_feats)))
    assert set(to_analyse_df.columns) == set(my_feats)
    assert to_analyse_df.shape[0] == len(my_smis)
    assert np.all(to_analyse_shaps.shape[-2:] == (len(my_smis), len(my_feats)))

    # representation correctness
    for new_feat, old_feat in enumerate(feature_order):
        for new_mol, old_mol in enumerate(mol_indices):
            assert X_full[old_mol, old_feat] == to_analyse_X[new_mol, new_feat], f"{old_mol}-{old_feat}  {new_mol}-{new_feat}"

    for smi, f in zip(to_analyse_df.index, to_analyse_df.columns):
        assert to_analyse_df.loc[smi, f] == X_full_df.loc[smi, f], f"feature {f}\t{smi}"

    # shap values correctness
    if task == Task.CLASSIFICATION:
        for i in range(shap_values.shape[0]):
            for new_mol, old_mol in enumerate(mol_indices):
                for new_f, old_f in enumerate(feature_order):
                    assert to_analyse_shaps[i, new_mol, new_f] == shap_values[i, old_mol, old_f]
    elif task == Task.REGRESSION:
        for new_mol, old_mol in enumerate(mol_indices):
            for new_f, old_f in enumerate(feature_order):
                assert to_analyse_shaps[new_mol, new_f] == shap_values[old_mol, old_f]
    else:
        raise ValueError(TASK_ERROR_MSG(task))


def test_find_separation_point_analyses(shap_values, X_full, feature_order, task):
    if task == Task.CLASSIFICATION:
        classes = range(shap_values.shape[0])
    elif task == Task.REGRESSION:
        classes = (None, )
    else:
        raise ValueError(TASK_ERROR_MSG(task))

    for class_idx in classes:
        for i in feature_order:
            max_correct, purity, best_thresholds = find_optimal_separation_point(shap_values, X_full, feature_order,
                                                                                 i, task, class_idx)
            kmax_correct, kpurity, kbest_thresholds = kfind(shap_values, X_full, feature_order, i, task, class_idx)

            assert max_correct == kmax_correct, AssertionError(f'{max_correct} != {kmax_correct}')

            best_thresholds = np.array(sorted(best_thresholds))
            kbest_thresholds = np.array(sorted(kbest_thresholds))
            # TODO: why not best_thresholds == kbest_thresholds?
            assert len(best_thresholds) == len(kbest_thresholds), AssertionError(f'{best_thresholds} != {kbest_thresholds}')
            assert set(best_thresholds) == set(kbest_thresholds), AssertionError(f'{best_thresholds} != {kbest_thresholds}')


def test_find_separation_point_well_separated(shap_values, X_full, feature_order, task):
    for f in feature_order:
        feat_idx = feature_order.index(f)  # indeks cechy o nazwie "f"
        
        if task == Task.REGRESSION:
            separation_results = [well_separated(X_full[:,feat_idx], shap_values[:,feat_idx], task, n_way=2), ]
            classes = [None, ]
        elif task == Task.CLASSIFICATION:
            separation_results = well_separated(X_full[:,feat_idx], shap_values[:,:,feat_idx], task, n_way=2)
            classes = range(len(separation_results))
        else:
            raise ValueError(f"Unknown task: {task}. Known tasks are `regression` and `classification`.")

        reference_result = []
        for c in classes:
            ref = kfind(shap_values, X_full, feature_order, f, task, class_index=c, extras=True)
            reference_result.append(ref)
            
        for my_results, ref, c in zip(separation_results, reference_result, classes):
            # checking score
            my_score = set([my.score for my in my_results])
            ref_score = set([kk.score/len(X_full) for kk in ref])
            assert my_score == ref_score, f"{my_score} != {ref_score}"
            
            # checking thresholds and their corresponding group tagging
            my_tre_val = dict([(my.thresholds, tuple(r.majority for r in my.regions) ) for my in my_results])
            ref_tr_val = dict([(kk.x, (0,1) if kk.type == SeparationType.ZEROES_ON_LEFT else (1,0)) for kk in ref])
            assert my_tre_val == ref_tr_val, f"{my_tre_val} != {ref_tr_val}"

            # checking purities
            for my in my_results:
                l0, l1, r0, r1 = situation_at_threshold(my.thresholds, f, shap_values, X_full, feature_order, task, class_index=c, print_func=None)

                mines = tuple(r.purity for r in my.regions)
                reference = (purity(l0, l1), purity(r0, r1))
                assert np.all(np.isclose(mines, reference, equal_nan=True)), f"{mines} != {reference}"

        

def all_tests(some_model, some_shaps):
    # data preparation
    data_cfg, repr_cfg, task_cfg, model_cfg, model_pickle = get_configs_and_model(some_model)
    x_train, x_test, smiles_train, smiles_test = load_ml_files(some_model)
    task = Task(task_cfg[utils_section]['task'])
    shap_cfg, smiles_order, X_full, morgan_repr, true_ys, preds, classes_order, expected_values, shap_values, background_data = load_shap_files(some_shaps, task)
    X_full_df = pd.DataFrame(X_full, columns=list(range(X_full.shape[1])), index=smiles_order)

    # smiles_true_predicted
    test_get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order)
    smiles_true_predicted_df = get_smiles_true_predicted(smiles_order, true_ys, preds, task, classes_order)
    
    # smiles and feature filters
    test_get_smiles_correct(smiles_true_predicted_df, task, task_cfg, data_cfg, classes_order)
    test_get_smiles_stability_value(smiles_true_predicted_df, data_cfg, task_cfg)
    _ = [test_get_present_features(x_train, t) for t in [0.1, 0.25, 0.55, 0.9]]
    
    # filter_samples
    correct_smiles = get_smiles_correct(smiles_true_predicted_df, task, task_cfg, data_cfg, classes_order)
    low, med, high = get_smiles_stability_value(smiles_true_predicted_df, data_cfg, task_cfg)
    for t in [0.15, 0.2]:
        satisfied = get_present_features(x_train, t)
        for stab_class in [low, med, high]:
            this_mols = correct_smiles.intersection(stab_class)
            test_filter_samples(this_mols, satisfied, X_full, X_full_df, shap_values, task, smiles_order)
            
    # find optimal separation point (analyses)
    my_feats = get_present_features(x_train, 0.01)
    to_analyse_X, to_analyse_df, to_analyse_shaps, smi_order, mol_indices, feature_order = filter_samples(smiles_order, my_feats, X_full, shap_values, task, smiles_order)
    
    test_find_separation_point_analyses(to_analyse_shaps, to_analyse_X, feature_order, task)
#     if task == Task.CLASSIFICATION:
#         for c in classes_order:
#             test_find_separation_point_analyses(to_analyse_shaps, to_analyse_X, feature_order, task, c)
#     elif task == Task.REGRESSION:
#         test_find_separation_point_analyses(to_analyse_shaps, to_analyse_X, feature_order, task, None)
#     else:
#         raise ValueError(TASK_ERROR_MSG(task))
        
        
    # find optimal separation point (well_separated)
    test_find_separation_point_well_separated(to_analyse_shaps, to_analyse_X, feature_order, task)

if __name__=="__main__":
    directory = '/home/pocha/dane_phd/random_split/'
    
    pbar = tqdm(os.listdir(osp.join(directory, 'ml')))
    for exp in pbar:
        pbar.set_description("Running tests... %s" % exp)
        some_model = osp.join(directory, 'ml', exp)
        some_shaps = osp.join(directory, 'shap', exp)
        all_tests(some_model, some_shaps)
    
    print("Done.")
