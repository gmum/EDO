import os.path as osp
from configparser import ConfigParser
from distutils.util import strtobool

try:
    from tpot import TPOTRegressor, TPOTClassifier
except ModuleNotFoundError:
    print("TPOT NOT FOUND.")

UTILS = "UTILS"
CSV = "CSV"
DATA = 'DATA'
METRICS = "METRICS"
ADAPTED_CLS_METRICS = "ADAPTED_CLS_METRICS"


def str_to_bool(val):
    # distutils.util.strtobool returns zeros and ones instead of bool.
    v = strtobool(val)
    if v == 0:
        return False
    elif v == 1:
        return True
    else:
        raise ValueError


def load_config(fpath):
    """Load configuration file."""
    config = ConfigParser()
    with open(fpath, 'r') as f:
        config.read_file(f)
    return config._sections


def parse_model_config(config_path):
    """
    Load and parse model configuration file in which:
    model: a human-readable name of the model.
    """
    config = load_config(config_path)
    return config


def parse_shap_config(config_path):
    """
    Load and parse SHAP configuration file in which:
    k: size of the background dataset
    link: from SHAP docs: `The link function used to map between the output units of the model and the SHAP value units.`
    unlog: True if predictions should be unloged before calculating SHAP values, else False (uses wrappers.Unloger)
    """
    config = load_config(config_path)
    config[UTILS]["k"] = int(config[UTILS]["k"])
    try:
        config[UTILS]["unlog"] = str_to_bool(config[UTILS]["unlog"])
    except KeyError:
        config[UTILS]["unlog"] = False
    return config


def parse_data_config(config_path):
    """
    Load and parse dataset configuration file in which:
    dataset: human-readable name of the dataset
    test: path to CSV file with the test samples
    [DATA] foldN: path to CSV file with samples of the n-th fold

    Parameters in section CSV are used in function data.load_csvs.
    [CSV] smiles_index: index of column with SMILES
    [CSV] y_index: index of column with labels
    [CSV] delimiter: delimiter used in the CSV files
    [CSV] skip_line: True if the first line of each file contains column names, False otherwise
    [CSV] scale: how should the labels be scaled?
    [CSV] average: if the same SMILES appears multiple times how should its labels be averaged?
    """
    config = load_config(config_path)

    # make paths absolute
    root = osp.abspath(osp.join('..', osp.dirname(__file__)))
    config[UTILS]['test'] = osp.join(root, config[UTILS]['test'])
    for key in sorted(config[DATA]):
        config[DATA][key] = osp.join(root, config[DATA][key])

    config[CSV]["smiles_index"] = int(config[CSV]["smiles_index"])
    config[CSV]["y_index"] = int(config[CSV]["y_index"])
    config[CSV]["skip_line"] = str_to_bool(config[CSV]["skip_line"])

    if config[CSV]["delimiter"] == '\\t' or config[CSV]["delimiter"] == 'tab':
        config[CSV]["delimiter"] = '\t'

    return config


def parse_representation_config(config_path):
    """
    Load and parse representation configuration file in which:
    fingerprint: name of the fingerprint
    morgan_nbits: number of bits in Morgan fingerprint
    These parameters are used in function data.load_and_preprocess.
    """
    config = load_config(config_path)

    if config[UTILS]['morgan_nbits'] == "None":
        config[UTILS]['morgan_nbits'] = None
    else:
        config[UTILS]['morgan_nbits'] = int(config[UTILS]['morgan_nbits'])

    return config


def parse_task_config(config_path):
    """
    Load and parse task configuration file (regression or classification) in which:
    [UTILS] tpot_model: TPOTClassifier or TPOTRegressor depending on the task
    [UTILS] task: human-readable name of the task
    [UTILS] cutoffs: cutoffs for changing regression to classification
    [UTILS] metric: metric to use during hyperparameter search

    [METRICS] metric_N: metrics to calculate on test data using the best model

    [FORCE_CLASSIFICATION_METRICS] metric_N: metrics to calculate on test data using the best regression model
                                             after changing its predictions to classification
    """
    config = load_config(config_path)

    try:
        if config[UTILS]['tpot_model'] == 'TPOTClassifier':
            config[UTILS]['tpot_model'] = TPOTClassifier
        elif config[UTILS]['tpot_model'] == 'TPOTRegressor':
            config[UTILS]['tpot_model'] = TPOTRegressor
        else:
            raise ValueError(
                f"TPOT models are TPOTClassifier and TPOTRegressor but {config[UTILS]['tpot_model']} was given")
    except NameError:
        print("TPOT NOT FOUND. Task config has strings instead of classes.")

    if 'cutoffs' in config[UTILS]:
        if config[UTILS]['cutoffs'] == 'metstabon':
            from .data import cutoffs_metstabon
            config[UTILS]['cutoffs'] = cutoffs_metstabon
        else:
            raise NotImplementedError("Only metstabon cutoffs are implemented.")

    return config


def parse_tpot_config(config_path):
    """
    Load and parse TPOT configuration file in which:
    n_jobs: number of parallel jobs
    max_time_mins: time allowed to search for best hyperparameters (in minutes)
    minimal_number_of_models: minimal number of hyperparameter configurations that should be evaluated
    """
    config = load_config(config_path)
    config[UTILS]['n_jobs'] = int(config[UTILS]['n_jobs'])
    config[UTILS]['max_time_mins'] = int(config[UTILS]['max_time_mins'])
    config[UTILS]['minimal_number_of_models'] = int(config[UTILS]['minimal_number_of_models'])

    return config
