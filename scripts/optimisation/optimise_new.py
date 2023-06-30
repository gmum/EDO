import os.path as osp
import sys
import time
import argparse

import numpy as np
import pandas as pd

from edo import make_origin, Task, TASK_ERROR_MSG
from edo.savingutils import make_directory, save_as_json, get_timestamp
from edo.wrappers import LoggerWrapper

from edo.shap_analysis.preprocessing import get_present_features
from edo.shap_analysis.utils import index_of_smiles

from edo.optimisation import Goal, set_seed, get_random_generator

from edo.optimisation.utils import find_experiment, load_train_test, load_predictions, load_shap_files, load_model, \
    difference_list
from edo.optimisation.utils import filter_correct_predictions_only, group_samples, intersection_list
from edo.optimisation.filter_rules import condition_well_separated, condition_high_impact

from edo.shap_analysis.feature import make_features
from edo.optimisation.generate_rules import derive_well_separated_two_way_rules, derive_high_impact_rules
from edo.optimisation.generate_rules import derive_random_rules_sample
from edo.optimisation.filter_rules import filter_rules, filter_out_unimportant
from edo.optimisation.filter_rules import filter_contradictive_soft, rebel_rules_stats
from edo.optimisation.sample import make_samples
from edo.optimisation.scenario import optimise

from edo.optimisation.evaluation import rule_stats, optimisation_stats, evaluate_optimisation, evaluate_history

"""
# both correctly and incorrectly predicted samples are optimised
"""

###############
check_unlogging = True  # always check unlogging of models
pprint = True  # print all stats

if __name__ == "__main__":

    # ARGPARSE  # TODO: ustalić sensowniejszą kolejność argumentów
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('split', type=str)
    parser.add_argument('fp', type=str)
    parser.add_argument('task', type=str)
    parser.add_argument('m1', type=str, help='model that delivers SHAPs for rule derivation')
    parser.add_argument('mic', type=str, help='independent classifier for evaluation')
    parser.add_argument('mir', type=str, help='independent regression model for evaluation')

    parser.add_argument('at_once', type=int)
    parser.add_argument('n_times', type=int)
    parser.add_argument('pf_ratio', type=float,
                        help='derive rules only for features which are present and absent in at least pf_ratio samples in the training set')
    parser.add_argument('ws_min_score', type=float,
                        help='minimal score for well separated features used to derive rules')
    parser.add_argument('hi_gamma', type=float, help='parameter gamma for high impact features')
    parser.add_argument('hi_metric', type=str, help='metric for high impact features')
    parser.add_argument('hi_min_score', type=float, help='minimal score for high impact features used to derive rules')
    parser.add_argument('unimp_miu', type=float, help='parameter miu for unimportant features')
    parser.add_argument('unimp_metric', type=str, help='metric for unimportant features')
    parser.add_argument('unimp_max_ratio', type=float,
                        help='maximal ratio of unimportant samples in features used to derive rules')

    parser.add_argument('seed', type=int)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('saving_dir', type=str)

    parser.add_argument('--baseline', action='store_true',
                        help="derive random rules instead of rules based on SHAP values")
    parser.add_argument('--no_contradictive', action='store_true', help="filter out contradictive rules")
    parser.add_argument('--skip_criterion_check', action='store_true',
                        help="only check feature value when selecting candidate rules")
    parser.add_argument('--update_shap', action='store_true', help='update SHAP values after updating the molecule')
    parser.add_argument('--debug', action='store_true', help='optimise only three samples not all of them')


    # TODO: poniższe argumenty trzeba dodać do opisu eksperymentu
    parser.add_argument('--use_incorrect', action='store_true',
                        help='include samples for which prediction is incorrect in rule derivation')

    parser.add_argument('--low', type=str, nargs='*', const='low', default='low',
                        help='a list of rule groups to optimise samples of low stability, available groups: `low` (default), `med`, `high`, `all`')
    parser.add_argument('--med', type=str, nargs='*', const='med', default='med',
                        help='a list of rule groups to optimise samples of medium stability, available groups: `low`, `med` (default), `high`, `all`')
    parser.add_argument('--high', type=str, nargs='*', const='high', default='high',
                        help='a list of rule groups to optimise samples of high stability, available groups: `low`, `med`, `high` (default), `all`')

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    rng = get_random_generator()
    seed = args.seed  # used later for saving results

    # define models
    task = Task(args.task)
    task_reg = Task.REGRESSION  # independent regressor evaluator is always a regressor

    m1_origin = make_origin((args.dataset, args.split, args.fp, task.value, args.m1))
    mic_origin = make_origin((args.dataset, args.split, args.fp, task.value, args.mic))
    mir_origin = make_origin((args.dataset, args.split, args.fp, task_reg.value, args.mir))

    # define experiment setup
    at_once = args.at_once
    n_times = args.n_times

    pf_ratio = args.pf_ratio

    ws_min_score = args.ws_min_score
    hi_params = {'gamma': args.hi_gamma, 'metric': args.hi_metric}
    hi_min_score = args.hi_min_score
    unimp_params = {'miu': args.unimp_miu, 'metric': args.unimp_metric}
    unimp_max_ratio = args.unimp_max_ratio

    condition_ws = lambda x: condition_well_separated(x, ws_min_score)  # hacksy
    condition_hi = lambda x: condition_high_impact(x, hi_min_score)

    # in directory, out directory
    results_dir = args.results_dir
    saving_dir = args.saving_dir
    expname = f'{int(time.time())}-{at_once}-{n_times}-{seed}'
    make_directory(saving_dir, expname)
    saving_dir = osp.join(saving_dir, expname)

    # optional experiment setup (with default values)
    baseline = args.baseline  # random rules or regular rules
    no_contradictive = args.no_contradictive  # filter our contradictive rules?
    # assert not (baseline and no_contradictive), "Filtering out contradictive rules does not make sense for baseline"
    skip_criterion_check = args.skip_criterion_check
    update_shap = args.update_shap
    debug = args.debug  # optimise only three samples
    correct_only = not args.use_incorrect  # include samples for which prediction is incorrect in rule derivation

    # LOGGER
    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    timestamp = get_timestamp()
    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(timestamp)
    logger_wrapper.logger.info(f'Running {sys.argv}')
    logger_wrapper.logger.info(f'with params: {vars(args)}')
    pprint = logger_wrapper.logger.info if pprint else None

    # # # # # # # #
    # S K R Y P T #
    # # # # # # # #

    # LOAD STUFF

    ml_m1 = find_experiment(results_dir, 'ml', m1_origin)
    shap_m1 = find_experiment(results_dir, 'shap', m1_origin)

    train, test = load_train_test(ml_m1)
    present_features = get_present_features(train[0], pf_ratio)

    tr_preds, test_preds = load_predictions(ml_m1, task)

    if correct_only:
        train_correct_smi = filter_correct_predictions_only(tr_preds, task)
    else:
        train_smi = sorted(tr_preds.index.tolist())  # use all samples to derive rules
    test_correct_smi = filter_correct_predictions_only(test_preds, task)
    test_incorrect_smi = difference_list(test_preds.index.tolist(), test_correct_smi)

    groups_train_samples = group_samples(tr_preds, task)

    # unstable molecules use unstable rules
    # medium and stable molecules use rules derived on ALL samples
    # low, _, _ = groups_train_samples
    # groups_train_samples = [low, sorted(tr_preds.index.tolist()), sorted(tr_preds.index.tolist())]
    # group_train_names = ['unstable', 'all', 'all']

    #####
    low, med, high = groups_train_samples
    alles = sorted(tr_preds.index.tolist())
    rule_groups = {'low': low, 'med': med, 'high': high, 'all': alles}

    assert all([rg in rule_groups.keys() for rg in args.low+args.med+args.high]),\
        f"Available rule groups are: `low`, `med`, `high`, `all`, given: {args.low, args.med, args.high}"

    for_low = [(rule_groups[rg], rg) for rg in args.low]
    for_med = [(rule_groups[rg], rg) for rg in args.med]
    for_high = [(rule_groups[rg], rg) for rg in args.high]

    groups_train = [for_low, for_med, for_high]
    ####

    groups_test_samples = group_samples(test_preds, task)
    group_test_names = ['unstable', 'medium', 'stable']

    shap_smis, shap_x, shap_true_ys, classes_order, shap_vals = load_shap_files(shap_m1, task, check_unlogging)
    assert np.all(classes_order == [0, 1, 2]), NotImplementedError(
        "Musimy sklepać reindeksowanie jak classes order jest inny niż domyślny")

    # model, który liczył SHAPy
    shapator, shap_giver = load_model(results_dir, m1_origin, check_unlogging)
    if not update_shap:
        shap_giver = None

    # niezależny model
    mic, _ = load_model(results_dir, mic_origin, check_unlogging)
    assert np.all(
        shapator.classes_ == mic.classes_), f"Classes order mismatch {shapator.classes_}!={mic.classes_}"

    # regressor
    mir, mir_unlogged = load_model(results_dir, mir_origin, check_unlogging)

    # DO THINGS

    group_scores = {}

    # for group_tr, group_te, name_tr, name_te in zip(groups_train_samples, groups_test_samples, group_train_names,
    #                                                 group_test_names):
    #     logger_wrapper.logger.info(f"{name_te} samples will be optimised with rules derived on {name_tr}")

    ####
    for tr, group_te, name_te in zip(groups_train, groups_test_samples, group_test_names):
        for group_tr, name_tr in tr:

            logger_wrapper.logger.info(f"{name_te} samples will be optimised with rules derived on {name_tr}")
    ####

            # # # # # # # # # # # # # # # # #
            # C R E A T E   F E A T U R E S #
            # # # # # # # # # # # # # # # # #
            tr_smis = intersection_list(train_correct_smi, group_tr)  # wybieramy smilesy
            tr_smis = index_of_smiles(shap_smis, tr_smis)  # bierzemy ich indeksy
            my_features = make_features(present_features, tr_smis, shap_x, shap_vals, classes_order, m1_origin, task)

            # # # # # # # # # # # # # #
            # D E R I V E   R U L E S #
            # # # # # # # # # # # # # #
            all_rules = []
            for ft in my_features:
                if not baseline:
                    r1 = derive_well_separated_two_way_rules(ft, task)
                    r2 = derive_high_impact_rules(ft, hi_params, task)
                    all_rules.extend(r1 + r2)
                else:
                    all_rules.extend(derive_random_rules_sample(ft, task))

            logger_wrapper.logger.info('All')
            rules_all = rule_stats(all_rules, print_func=pprint)

            # # # # # # # # # # # # # #
            # F I L T E R   R U L E S #
            # # # # # # # # # # # # # #

            # filtering based on class and goal
            max_stab_rules = filter_rules(all_rules, Goal.MAXIMISATION, cls_name=2)  # maximise stable class probability
            min_unstab_rules = filter_rules(all_rules, Goal.MINIMISATION, cls_name=0)
            my_rules = max_stab_rules + min_unstab_rules
            logger_wrapper.logger.info('After filtering based on class and goal')
            rules_class_goaled = rule_stats(my_rules, print_func=pprint)

            # filtering based on conditions
            rules_conditioned = {}
            if not baseline:
                my_rules = filter_rules(my_rules, condition=condition_ws)
                my_rules = filter_rules(my_rules, condition=condition_hi)
                logger_wrapper.logger.info('After filtering based on conditions')
                rules_conditioned = rule_stats(my_rules, print_func=pprint)

            # filtering out unimportant
            my_rules = filter_out_unimportant(my_rules, my_features, unimp_params, unimp_max_ratio, task=task)
            logger_wrapper.logger.info('After filtering out unimportant rules')
            rules_importante = rule_stats(my_rules, print_func=pprint)

            # filtering out contradictive rules (SOFT VERSION)
            rules_noncontradictive = {}
            if no_contradictive:
                rebel_rules_stats(my_rules, print_func=pprint)
                logger_wrapper.logger.info(' ')
                my_rules = filter_contradictive_soft(my_rules)
                rebel_rules_stats(my_rules, print_func=pprint)
                logger_wrapper.logger.info('After filtering out contradictive rules')
                rules_noncontradictive = rule_stats(my_rules, print_func=pprint)

            rule_history = {'rules_all': rules_all,
                            'rules_class_goaled': rules_class_goaled,
                            'rules_conditioned': rules_conditioned,
                            'rules_importante': rules_importante,
                            'rules_noncontradictive': rules_noncontradictive}
            d = [rula.as_dict() for rula in my_rules]
            rules_df = pd.DataFrame.from_dict(d)
            rules_df.to_csv(osp.join(saving_dir, f'{timestamp}-{name_tr}-rules.csv'))  # TODO: można dodać info o tym dla jakich sampli

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # C R E A T E   S A M P L E S   F O R   O P T I M I S A T I O N #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            for c_smis, c_name in ((test_correct_smi, 'correctly_predicted'),
                                   (test_incorrect_smi, 'incorrectly_predicted')):
                logger_wrapper.logger.info(f'Optimisation of {c_name} {name_te} compounds')

                te_smis = intersection_list(c_smis, group_te)  # wybieramy smilesy
                te_smis = index_of_smiles(shap_smis, te_smis)  # bierzemy ich indeksy
                my_samples = make_samples(te_smis, shap_x, shap_vals, shap_smis, m1_origin, classes_order, task)
                assert len(my_samples) == len(te_smis)

                # # # # # # # # # #
                # O P T I M I S E #
                # # # # # # # # # #
                if debug:
                    my_samples = my_samples[:3]
                optimise(my_samples, my_rules, at_once=at_once, n_times=n_times,
                         update_shap=shap_giver, skip_criterion_check=skip_criterion_check,
                         extended_history=True)

                # # # # # # # # # #
                # E V A L U A T E #
                # # # # # # # # # #
                samples_to_evaluate = [s for s in my_samples if s.number_of_applied_changes() > 0]
                opis_exp = optimisation_stats(present_features, my_rules, my_samples, samples_to_evaluate,
                                              print_func=pprint)

                scores = {}
                for model, model_name, model_task in ((shapator, 'shapator', task), (mic, 'independent', task),
                                                      (mir, 'regressor_logged', task_reg), (mir_unlogged, 'regressor_unlogged', task_reg)):

                    my_scores = evaluate_optimisation(samples_to_evaluate, model, model_task, print_func=pprint)
                    if task == Task.CLASSIFICATION:
                        scores[f"{model_name}_unstable"] = my_scores[0]
                        scores[f"{model_name}_stable"] = my_scores[1]
                    elif task == Task.REGRESSION:
                        scores[model_name] = my_scores
                    else:
                        raise ValueError(TASK_ERROR_MSG(task))

                    df = evaluate_history(samples_to_evaluate, model, model_task)
                    df.to_csv(
                        osp.join(saving_dir, f'{timestamp}-history-R-{name_tr}-S-{name_te}-{c_name}-{model_name}.csv'))


                # # na razie modelem, który liczył shapy
                # shapator_scores_class_unstable, shapator_scores_class_stable = evaluate_optimisation(samples_for_evaluation,
                #                                                                                      shapator, task,
                #                                                                                      print_func=pprint)
                # shapator_df = evaluate_history(samples_for_evaluation, shapator, task)
                # shapator_df.to_csv(
                #     osp.join(saving_dir, f'{timestamp}-history-R-{name_tr}-S-{name_te}-{c_name}-shapator.csv'))
                #
                # # a teraz osobnym modelem
                # mic_scores_class_unstable, mic_scores_class_stable = evaluate_optimisation(samples_for_evaluation,
                #                                                                            mic, task, print_func=pprint)
                # mic_df = evaluate_history(samples_for_evaluation, mic, task)
                # mic_df.to_csv(
                #     osp.join(saving_dir, f'{timestamp}-history-R-{name_tr}-S-{name_te}-{c_name}-independator.csv'))
                #
                # # a teraz regresorem
                # mir_scores = evaluate_optimisation(samples_for_evaluation, mir, task_reg, print_func=pprint)
                # mir_df = evaluate_history(samples_for_evaluation, mir, task_reg)
                # mir_df.to_csv(
                #     osp.join(saving_dir, f'{timestamp}-history-R-{name_tr}-S-{name_te}-{c_name}-regressor-logged.csv'))
                #
                # # i odlogowanym regressorem
                # reg_unlogged_scores = evaluate_optimisation(samples_for_evaluation, mir_unlogged, task_reg,
                #                                             print_func=pprint)
                # reg_unlogged_df = evaluate_history(samples_for_evaluation, mir_unlogged, task_reg)
                # reg_unlogged_df.to_csv(
                #     osp.join(saving_dir, f'{timestamp}-history-R-{name_tr}-S-{name_te}-{c_name}-regressor-unlogged.csv'))

                group_results = {'n_train_samples': len(tr_smis),
                                 # 'shapator_unstable': shapator_scores_class_unstable,
                                 # 'shapator_stable': shapator_scores_class_stable,
                                 # 'independent_unstable': mic_scores_class_unstable,
                                 # 'independent_stable': mic_scores_class_stable,
                                 # 'regressor_logged': mir_scores,
                                 # 'regressor_unlogged': reg_unlogged_scores,
                                 'rule_type': name_tr,
                                 'optimised_samples_type': name_te,
                                 'optimised_samples_pred': c_name
                                 }
                group_results.update(scores)
                group_results.update(opis_exp)
                group_results.update(rule_history)
                group_scores[f'R-{name_tr}-S-{name_te}-{c_name}_scores'] = group_results

    results = {'shapator': m1_origin._asdict(),
               'independator': mic_origin._asdict(),
               'regressor': mir_origin._asdict(),
               'at_once': at_once, 'n_times': n_times,
               'pf_ratio': pf_ratio,
               'ws_min_score': ws_min_score,
               'hi_params': hi_params, 'hi_min_score': hi_min_score,
               'unimp_params': unimp_params, 'unimp_max_ratio': unimp_max_ratio,
               'seed': seed, 'results_dir': results_dir, 'timestamp': timestamp,
               'baseline': baseline, 'no_contradictive': no_contradictive,
               'skip_criterion_check': skip_criterion_check, 'update_shap': update_shap,
               'check_unlogging': check_unlogging, 'debug': debug
               }
    results.update(group_scores)
    save_as_json(results, saving_dir, 'stats.json')