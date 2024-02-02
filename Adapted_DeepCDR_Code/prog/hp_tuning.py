"""Implements hyperparameter tuning for architecture adaption as described in Section 3: Methods of the report"""

import os.path
import json
import optuna
import gc
import sys
import math
import tensorflow_addons as tfa
import sklearn.model_selection as sk
import keras.callbacks as cb
from datetime import datetime, timedelta
from util.DataLoader import *
from model import KerasMultiSourceGCNModel
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from scipy.stats import pearsonr

ROOT_PATH = "../"
sys.path.append(ROOT_PATH)
from run_DeepCDR import loadAndEvalModel, savePredictions

EXP_NAME = "all_out_strict"
EXP_VERSION = "Version_1"
DPATH = "../data"
DRUG_INFO = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv' % DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt' % DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat' % DPATH
Drug_feature_file_random = '%s/GDSC/drug_graph_feat_random' % DPATH

LOG_DIR_BASE = "../logs/"

Genomic_mutation_file = '../data/CCLE/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = '../data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

VAL_PATH = "../data/TuningSplits/val.csv"
TEST_PATH = "../data/TuningSplits/test.csv"
TRAIN_PATH = "../data/TuningSplits/train.csv"

VAL_PATH_DRUG_OUT = "../data/TuningSplits/drug_out/val.csv"
TEST_PATH_DRUG_OUT = "../data/TuningSplits/drug_out/test.csv"
TRAIN_PATH_DRUG_OUT = "../data/TuningSplits/drug_out/train.csv"

VAL_PATH_CELL_OUT = "../data/TuningSplits/cell_out/val.csv"
TEST_PATH_CELL_OUT = "../data/TuningSplits/cell_out/test.csv"
TRAIN_PATH_CELL_OUT = "../data/TuningSplits/cell_out/train.csv"

VAL_PATH_ALL_OUT = "../data/TuningSplits/all_out/val.csv"
TEST_PATH_ALL_OUT = "../data/TuningSplits/all_out/test.csv"
TRAIN_PATH_ALL_OUT = "../data/TuningSplits/all_out/train.csv"

VAL_PATH_ALL_OUT_STRICT = "../data/TuningSplits/all_out_strict/val.csv"
TEST_PATH_ALL_OUT_STRICT = "../data/TuningSplits/all_out_strict/test.csv"
TRAIN_PATH_ALL_OUT_STRICT = "../data/TuningSplits/all_out_strict/train.csv"


VAL_PATH_DEBUG = "../data/TuningSplits/debug_val.csv"
TEST_PATH_DEBUG = "../data/TuningSplits/debug_test.csv"
TRAIN_PATH_DEBUG = "../data/TuningSplits/debug_train.csv"

PCC = tfa.metrics.PearsonsCorrelation()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LOG_HPARAMS = "../logs/hparams"


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def optuna_objective(trial):
    """
    Defines options for hyperparameters and selects hyperparameters based on
    the options that performed well in previous runs. Starts training
    afterwards and returns the value of the objective function.

    See https://medium.com/optuna/using-optuna-to-optimize-pytorch-lightning-hyperparameters-d9e04a481585

    Args:
        trial: trial object, stores performance of hyperparameters
        from previous runs

    Returns:
        Validation loss of last epoch
    """

    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
    if not gpu_id:
        raise ValueError('No gpu specified! Please select "export CUDA_VISIBLE_DEVICES=<device_id>')

    # Define fixed parameters
    no_randomise = {"mutation": False, "methylation": False, "expression": False, "drug": False}

    params = {
        "k": 1,
        "ratio_test_set": 0.05,
        "leaveOut": "normal",
        "debug_mode": False,
        "consider_ratio": False,
        "mul": False,
        "group_by_tissue": False,
        "save_split": False,
        "randomise": no_randomise,
        "hp_tuning": True,
        "patience": 10,
        "max_epoch": 100,
        "unit_list": [256, 256, 256]
    }

    """
            "nb_attn_head_gexpr":8,
            "nb_attn_head_mut": 8,
            "nb_attn_head_methy": 8,

            "key_dim_mut": 64,
            "key_dim_methy": 64,
            "key_dim_gexpr": 64,

            "pos_attn_gexpr": 2,
            "pos_attn_mut": 3,
            "pos_attn_methy": 2,

            "loss": "mean_squared_error",
            "use_mut": True,
            "use_methy": True,
            "use_gexp": True,
            "use_bn": True,
            "use_relu": True,
            "use_GMP": True
    """

    # Define hyperparameter options and ranges
    lr_min = 1e-5
    lr_max = 1e-2
    nb_attn_head = [4, 6, 8]
    pos_attn_gexpr = [-1, 0, 1, 2]
    pos_attn_mut = [-1, 0, 1, 2, 3]
    pos_attn_methy = [-1, 0, 1, 2]
    key_dim = [32, 40, 48, 56, 64, 70, 76, 82, 90]
    loss = ["mean_squared_error", "mean_squared_logarithmic_error"]

    # Let optuna select hyperparameters based on options defined above
    params["lr"] = trial.suggest_float("lr", lr_min, lr_max, log=True)

    params["nb_attn_head_gexpr"] = trial.suggest_categorical("nb_attn_head_gexpr", nb_attn_head)
    params["nb_attn_head_methy"] = trial.suggest_categorical("nb_attn_head_methy", nb_attn_head)
    params["nb_attn_head_mut"] = trial.suggest_categorical("nb_attn_head_mut", nb_attn_head)
    params["pos_attn_gexpr"] = trial.suggest_categorical("pos_attn_gexpr", pos_attn_gexpr)
    params["pos_attn_mut"] = trial.suggest_categorical("pos_attn_mut", pos_attn_mut)
    params["pos_attn_methy"] = trial.suggest_categorical("pos_attn_methy", pos_attn_methy)
    params["loss"] = trial.suggest_categorical("loss", loss)
    params["use_mut"] = trial.suggest_categorical("use_mut", (True, False))
    params["use_gexp"] = trial.suggest_categorical("use_gexp", (True, False))
    params["use_methy"] = trial.suggest_categorical("use_methy", (True, False))
    params["use_bn"] = trial.suggest_categorical("use_bn", (True, False))
    params["use_GMP"] = trial.suggest_categorical("use_GMP", (True, False))
    params["use_relu"] = trial.suggest_categorical("use_relu", (True, False))
    params["key_dim_mut"] = trial.suggest_categorical("key_dim_mut", key_dim)
    params["key_dim_methy"] = trial.suggest_categorical("key_dim_mut", key_dim)
    params["key_dim_gexpr"] = trial.suggest_categorical("key_dim_mut", key_dim)

    with open(os.path.join(LOG_HPARAMS, EXP_VERSION, datetime.now().strftime("%d.%m-%H:%M")), 'w') as file:
        writer = csv.writer(file)
        for key, value in params.items():
            writer.writerow([key, value])

    print(params.items())
    try:
        val_loss = train_attention_model(params, experiment_name=EXP_NAME, experiment_version=EXP_VERSION)
        return val_loss
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        return math.inf

def load_indices(path: str):
    indices = []
    with open(path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the header
        next(csvreader)
        for row in csvreader:
            indices.append(tuple(row))
    return indices


def getData(params, data_paths):
    """
    Load the data for training, validation and testing.
    Args:
        params: Parameter as configured for hyperparameter optimization
        data_paths: paths to sample ids for training, validation and testing data sets

    Returns: train, validation and test data
    """
    # Add the parent directory to sys.path
    sys.path.append(ROOT_PATH)
    from DeepCDR.prog.run_DeepCDR import MetadataGenerateOriginal
    from run_DeepCDR import MetadataGenerate
    if params["randomise"]:
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = MetadataGenerate(DRUG_INFO,
                                                                                                        Cell_line_info_file,
                                                                                                        Genomic_mutation_file,
                                                                                                        Drug_feature_file_random if
                                                                                                        params[
                                                                                                            "randomise"][
                                                                                                            "drug"] else Drug_feature_file,
                                                                                                        Gene_expression_file,
                                                                                                        Methylation_file,
                                                                                                        randomise=
                                                                                                        params[
                                                                                                            "randomise"],
                                                                                                        debug_mode=
                                                                                                        params[
                                                                                                            "debug_mode"])


    else:
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = MetadataGenerateOriginal(
            DRUG_INFO,
            Cell_line_info_file,
            Genomic_mutation_file,
            Drug_feature_file,
            Gene_expression_file,
            Methylation_file,
            False)

    data_test_idx = load_indices(TEST_PATH_DEBUG if params["debug_mode"] else data_paths[2])
    data_val_idx = load_indices(VAL_PATH_DEBUG if params["debug_mode"] else data_paths[1])
    data_train_idx = load_indices(TRAIN_PATH_DEBUG if params["debug_mode"] else data_paths[0])

    train_data = {}
    val_data = {}
    test_data = {}
    # Extract features for train, test and val
    test_data["X_drug_data_test"], test_data["X_mutation_data_test"], test_data["X_gexpr_data_test"], test_data[
        "X_methylation_data_test"], test_data["Y_test"], test_data["cancer_type_test_list"] = FeatureExtract(
        data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom=False)

    val_data["X_drug_data_val"], val_data["X_mutation_data_val"], val_data["X_gexpr_data_val"], val_data[
        "X_methylation_data_val"], \
        val_data["Y_val"], val_data["cancer_type_val_list"] = FeatureExtract(
        data_val_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom=False)

    train_data["X_drug_data_train"], train_data["X_mutation_data_train"], train_data["X_gexpr_data_train"], train_data[
        "X_methylation_data_train"], \
        train_data["Y_train"], train_data["cancer_type_train_list"] = FeatureExtract(
        data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom=False)

    return train_data, val_data, test_data


def getTestData():
    """
    Loads data for model testing
    Returns: test data
    """
    from DeepCDR.prog.run_DeepCDR import MetadataGenerateOriginal
    mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = MetadataGenerateOriginal(DRUG_INFO,
                                                                                                            Cell_line_info_file,
                                                                                                            Genomic_mutation_file,
                                                                                                            Drug_feature_file,
                                                                                                            Gene_expression_file,
                                                                                                            Methylation_file,
                                                                                                            False)

    data_test_idx = load_indices(TEST_PATH)
    test_data = {}
    # Extract features for train, test and val
    test_data["X_drug_data_test"], test_data["X_mutation_data_test"], test_data["X_gexpr_data_test"], test_data[
        "X_methylation_data_test"], test_data["Y_test"], test_data["cancer_type_test_list"] = FeatureExtract(
        data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom=False)
    return test_data


def createData(debug: False):
    """
    Creates csv files for train, test and validation data
    Args:
        debug: wether model is debuged or not
    """
    from DeepCDR.prog.run_DeepCDR import MetadataGenerateOriginal
    if debug:
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = MetadataGenerate(DRUG_INFO,
                                                                                                        Cell_line_info_file,
                                                                                                        Genomic_mutation_file,
                                                                                                        Drug_feature_file,
                                                                                                        Gene_expression_file,
                                                                                                        Methylation_file,
                                                                                                        randomise={
                                                                                                            "mutation": False,
                                                                                                            "methylation": False,
                                                                                                            "expression": False,
                                                                                                            "drug": False},
                                                                                                        debug_mode=True)
    else:
        mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = MetadataGenerateOriginal(
            DRUG_INFO,
            Cell_line_info_file,
            Genomic_mutation_file,
            Drug_feature_file,
            Gene_expression_file,
            Methylation_file,
            False)
    data_train_idx, data_test_idx = sk.train_test_split(data_idx, test_size=0.1, random_state=42)
    data_test_idx, data_val_idx = sk.train_test_split(data_test_idx, test_size=0.5, random_state=42)

    df_train = pd.DataFrame(data_train_idx, columns=["cellline", "drug", "ic50", "tissue"])
    df_test = pd.DataFrame(data_test_idx, columns=["cellline", "drug", "ic50", "tissue"])
    df_val = pd.DataFrame(data_val_idx, columns=["cellline", "drug", "ic50", "tissue"])

    df_train.to_csv(TRAIN_PATH_DEBUG if debug else TRAIN_PATH, index=False)
    df_test.to_csv(TEST_PATH_DEBUG if debug else TEST_PATH, index=False)
    df_val.to_csv(VAL_PATH_DEBUG if debug else VAL_PATH, index=False)


def train_attention_model(params, experiment_name="", experiment_version=None,
                          data_paths=[TRAIN_PATH, VAL_PATH, TEST_PATH], checkpoint_dir="", test=False):
    """
    Trains a single attention model.
    Args:
        params: parameters for hyperparameter optimisation
        experiment_name: name of current experiment
        experiment_version: version of current run
        data_paths: list of paths to training, validation and test data
        checkpoint_dir: directory to save model checkpoint

    Returns:
        History of model training
    """
    log_dir = os.path.join(LOG_DIR_BASE, experiment_name, experiment_version, datetime.now().strftime("%d.%m-%H:%M:%S"))

    print("Load Data...")
    train_data, val_data, test_data = getData(params, data_paths)

    print("Build Model...")
    model = KerasMultiSourceGCNModel(params["use_mut"], params["use_gexp"], params["use_methy"]).createMaster(
        train_data["X_drug_data_train"][0][0].shape[-1],
        train_data["X_mutation_data_train"].shape[-2],
        train_data["X_gexpr_data_train"].shape[-1],
        train_data["X_methylation_data_train"].shape[-1],
        params)

    optimizer = Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss=params["loss"],

                  metrics=['mse', PCC] if params["loss"] == "mean_squared_error" else ['msle', PCC])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=params["patience"], verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        f'../checkpoint/hp_tuning/{checkpoint_dir}{experiment_name}_{experiment_version}_{datetime.now().strftime("%d.%m-%H:%M:%S")}.h5',
        monitor='val_loss',
        save_best_only=True)
    tensorboard = cb.TensorBoard(log_dir=log_dir)
    callbacks = [checkpoint, tensorboard, earlyStopping, ClearMemory()]
    X_drug_feat_data_train = [item[0] for item in train_data["X_drug_data_train"]]
    X_drug_adj_data_train = [item[1] for item in train_data["X_drug_data_train"]]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)  # nb_instance * Max_stom * Max_stom

    X_drug_adj_data_val, X_drug_feat_data_val = getDrugMatrices(val_data["X_drug_data_val"])
    validation_data = (
        [X_drug_feat_data_val, X_drug_adj_data_val, val_data["X_mutation_data_val"], val_data["X_gexpr_data_val"],
         val_data["X_methylation_data_val"]], val_data["Y_val"])

    train_start = time.time()
    print("Start Training")
    history = model.fit(x=[X_drug_feat_data_train, X_drug_adj_data_train, train_data["X_mutation_data_train"],
                           train_data["X_gexpr_data_train"],
                           train_data["X_methylation_data_train"]], y=train_data["Y_train"], batch_size=64,
                        epochs=params["max_epoch"],
                        validation_data=validation_data,
                        callbacks=callbacks)
    print(f"Time to train last model: {str(timedelta(seconds=time.time() - train_start))}")
    metric = 'val_mse' if params["loss"] == "mean_squared_error" else 'val_msle'
    best_epoch = np.argmin(history.history[metric]) + 1
    stopped_epoch = len(history.history[metric])
    print(f"Best Epoch: {best_epoch} \n Stopped after epoch: {stopped_epoch}")
    return history.history[metric][-1]


def test(checkpoint, params):
    """
    Tests model performance on test data and prints its PCC to console
    Args:
        checkpoint: checkpoint of the model to be tested
        params: hyperparameters of the model to be tested
    """
    data = getTestData()
    X_drug_adj_data_test, X_drug_feat_data_test = getDrugMatrices(data["X_drug_data_test"])

    model = KerasMultiSourceGCNModel(params["use_mut"], params["use_gexp"], params["use_methy"]).createMaster(
        data["X_drug_data_test"][0][0].shape[-1],
        data["X_mutation_data_test"].shape[-2],
        data["X_gexpr_data_test"].shape[-1],
        data["X_methylation_data_test"].shape[-1],
        params)

    # Loads the weights
    model.load_weights(checkpoint)
    Y_pred = model.predict(
        [X_drug_feat_data_test, X_drug_adj_data_test, params["X_mutation_data_test"], params["X_gexpr_data_test"],
         params["X_methylation_data_test"]])
    overall_pcc = pearsonr(Y_pred[:, 0], params["Y_test"])[0]
    print("The overall Pearson's correlation is %.4f." % overall_pcc)


def optuna_optimization():
    """
    Creates an optuna study and minimize the objective
    """
    study = optuna.create_study(direction="minimize")
    seconds_per_day = 24 * 60 * 60
    study.optimize(optuna_objective, n_trials=300, timeout=7 * seconds_per_day)


def pred_ensemble(test_data_path, save_path_dir_base, model_paths: list):
    single_result = []
    for index, model in enumerate(model_paths):
        savepath = os.path.join(save_path_dir_base, f"model_{str(index)}")
        Y_test, Y_pred, data_test_idx = loadAndEvalModel(savePath=savepath, test_data_path=test_data_path,
                                                         modelpath=model[0], params_path=model[1], save=True)
        single_result.append(Y_pred)
    ensemble_pred = np.mean(single_result, axis=0)
    savePredictions(Y_test, ensemble_pred, data_test_idx, os.path.join(save_path_dir_base, "ensemble"))
    overall_pcc = pearsonr(ensemble_pred[:, 0], Y_test)[0]
    print(f"The PCC for the ensemble is: {overall_pcc}")
    return ensemble_pred


if __name__ == '__main__':
    # createData(debug=True)
    # optuna_optimization()
    baseline_params = "../logs/hparams/baseline.json"

    # Version 2
    models_v2 = [
        ("../checkpoint/hp_tuning/fine_tuning_Version_2_25.09-13:41.h5",
         "../logs/hparams/Version_2/18.09-08:43.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_2_25.09-17:37.h5",
         "../logs/hparams/Version_2/18.09-19:59.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_2_25.09-22:23.h5",
         "../logs/hparams/Version_2/19.09-02:15.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_2_26.09-03:05.h5",
         "../logs/hparams/Version_2/19.09-04:01.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_2_26.09-07:33.h5",
         "../logs/hparams/Version_2/19.09-04:21.json")
        # ("", "../logs/hparams/baseline.json")
    ]
    ### Version1
    models_v1 = [
        ("../checkpoint/hp_tuning/fine_tuning_Version_1_23.09-22:45.h5",
         "../logs/hparams/Version_1/18.09-08:43.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_1_23.09-23:17.h5",
         "../logs/hparams/Version_1/18.09-19:59.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_1_23.09-23:17.h5",
         "../logs/hparams/Version_1/19.09-02:15.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_1_24.09-00:16.h5",
         "../logs/hparams/Version_1/19.09-04:01.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_1_24.09-00:38.h5",
         "../logs/hparams/Version_1/19.09-04:21.json"),
        ("", "../logs/hparams/baseline.json")
    ]

    models_v3 = [
        ("../checkpoint/hp_tuning/fine_tuning_Version_3_26.09-11:53.h5",
         "../logs/hparams/Version_3/18.09-08:43.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_3_26.09-15:18.h5",
         "../logs/hparams/Version_3/18.09-19:59.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_3_26.09-20:36.h5",
         "../logs/hparams/Version_3/19.09-02:15.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_3_27.09-03:57.h5",
         "../logs/hparams/Version_3/19.09-04:01.json"),
        ("../checkpoint/hp_tuning/fine_tuning_Version_1_24.09-00:38.h5",
         "../logs/hparams/Version_1/19.09-04:21.json")
        # ("", "../logs/hparams/baseline.json")
    ]

    models_drug = [
        (
        "../checkpoint/hp_tuning/drug_out/drug_outdrug_out_Version_1_27.09-20:25.h5",
        "../logs/hparams/Version_1/18.09-08:43.json"),
        (
        "../checkpoint/hp_tuning/drug_out/drug_outdrug_out_Version_1_27.09-20:31.h5",
        "../logs/hparams/Version_1/18.09-19:59.json"),
        (
        "../checkpoint/hp_tuning/drug_out/drug_outdrug_out_Version_1_27.09-20:40.h5",
        "../logs/hparams/Version_1/19.09-02:15.json"),
        (
        "../checkpoint/hp_tuning/drug_out/drug_outdrug_out_Version_1_27.09-20:51.h5",
        "../logs/hparams/Version_1/19.09-04:01.json"),
        (
        "../checkpoint/hp_tuning/drug_out/drug_outdrug_out_Version_1_27.09-20:57.h5",
        "../logs/hparams/Version_1/19.09-04:21.json")
        # ("", "../logs/hparams/baseline.json")
    ]

    models_cell = [
        (
            "../checkpoint/hp_tuning/cell_out/cell_out_Version_1_27.09-23:23.h5",
            "../logs/hparams/Version_1/18.09-08:43.json"),
        (
            "../checkpoint/hp_tuning/cell_out/cell_out_Version_1_27.09-23:31.h5",
            "../logs/hparams/Version_1/18.09-19:59.json"),
        (
            "../checkpoint/hp_tuning/cell_out/cell_out_Version_1_27.09-23:44.h5",
            "../logs/hparams/Version_1/19.09-02:15.json"),
        (
            "../checkpoint/hp_tuning/cell_out/cell_out_Version_1_27.09-23:52.h5",
            "../logs/hparams/Version_1/19.09-04:01.json"),
        (
            "../checkpoint/hp_tuning/cell_out/cell_out_Version_1_28.09-00:01.h5",
            "../logs/hparams/Version_1/19.09-04:21.json")
        # ("", "../logs/hparams/baseline.json")
    ]

    models_all_out = [
        (
            "../checkpoint/hp_tuning/all_out/drug_out_Version_1_28.09-19:43.h5",
            "../logs/hparams/Version_1/18.09-08:43.json"),
        (
            "../checkpoint/hp_tuning/all_out/drug_out_Version_1_28.09-20:02.h5",
            "../logs/hparams/Version_1/18.09-19:59.json"),
        (
            "../checkpoint/hp_tuning/all_out/drug_out_Version_1_28.09-20:17.h5",
            "../logs/hparams/Version_1/19.09-02:15.json"),
        (
            "../checkpoint/hp_tuning/all_out/drug_out_Version_1_28.09-20:36.h5",
            "../logs/hparams/Version_1/19.09-04:01.json"),
        (
            "../checkpoint/hp_tuning/all_out/drug_out_Version_1_28.09-20:46.h5",
            "../logs/hparams/Version_1/19.09-04:21.json")
        # ("", "../logs/hparams/baseline.json")
    ]

    models_all_out_strict = [
        (
            "../checkpoint/hp_tuning/all_out_strict/all_out_strict_Version_1_29.09-08:34:51.h5",
            "../logs/hparams/Version_1/18.09-08:43.json"),
        (
            "../checkpoint/hp_tuning/all_out_strict/all_out_strict_Version_1_29.09-08:35:26.h5",
            "../logs/hparams/Version_1/18.09-19:59.json"),
        (
            "../checkpoint/hp_tuning/all_out_strict/all_out_strict_Version_1_29.09-08:35:56.h5",
            "../logs/hparams/Version_1/19.09-02:15.json"),
        (
            "../checkpoint/hp_tuning/all_out_strict/all_out_strict_Version_1_29.09-08:36:31.h5",
            "../logs/hparams/Version_1/19.09-04:01.json"),
        (
            "../checkpoint/hp_tuning/all_out_strict/all_out_strict_Version_1_29.09-08:37:05.h5",
            "../logs/hparams/Version_1/19.09-04:21.json")
        # ("", "../logs/hparams/baseline.json")
    ]

    paths_drug_out = [TRAIN_PATH_DRUG_OUT, VAL_PATH_DRUG_OUT, TEST_PATH_DRUG_OUT]
    paths_cell_out = [TRAIN_PATH_CELL_OUT, VAL_PATH_CELL_OUT, TEST_PATH_CELL_OUT]
    paths_all_out = [TRAIN_PATH_ALL_OUT, VAL_PATH_ALL_OUT, TEST_PATH_ALL_OUT]
    paths_all_out_strict = [TRAIN_PATH_ALL_OUT_STRICT, VAL_PATH_ALL_OUT_STRICT, TEST_PATH_ALL_OUT_STRICT]

    trainings = {"drug_out/": paths_drug_out, "cell_out/": paths_cell_out, "all_out/": paths_all_out,
                 "all_out_strict/": paths_all_out_strict}

    # for checkpoint, params in models_v1:
    #   train_attention_model(json.load(open(params)), experiment_version=EXP_VERSION, experiment_name=EXP_NAME, checkpoint_dir = "all_out_strict/", data_paths=paths_all_out_strict)

    save_path = "../data/TuningSplits/preds/all_out_strict"
    test_split_path = "../data/TuningSplits/all_out_strict/test.csv"

    baselinepath = "../checkpoint/hp_tuning/all_out_strict/baseline.h5"
    baseline_save = '../data/TuningSplits/preds/all_out_strict/baseline'
    Y_test, Y_pred, data_test_idx = loadAndEvalModel(savePath=baseline_save, test_data_path=test_split_path,
                                                     modelpath=baselinepath, params_path=baseline_params, save=True)

    # pred_ensemble(test_data_path=test_split_path, save_path_dir_base = save_path, model_paths = models_all_out_strict)
