"Run and train DeepCDR model."

import gc
import glob
import json
import math
from datetime import timedelta
from util.DataLoader import *
from util.DataGenerator import *
from util.DataPlotter import *
from sys import getsizeof
import pandas as pd
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import tensorflow as tf
from scipy.stats import pearsonr
from model import KerasMultiSourceGCNModel
import keras.callbacks as cb
import argparse

####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, default=True, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, default=True, help='use methylation or not')

parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
# hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256, 256, 256],
                    help='unit list for GCN')

parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

LOG_DIR_BASE = "../logs/training"
LOG_DIR_BASE_TUNE = "../logs/hp_tuning"

if len(tf.config.list_physical_devices('GPU')) == 0:
    raise SystemError('GPU device not found')

use_mut, use_gexp, use_methy = args.use_mut, args.use_gexp, args.use_methy
israndom = args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut') + '_' + (
    'with_gexp' if use_gexp else 'without_gexp') + '_' + ('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str, args.unit_list)) + '_' + ('bn' if args.use_bn else 'no_bn') + '_' + (
    'relu' if args.use_relu else 'tanh') + '_' + ('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' + GCN_deploy

####################################Constants Settings###########################
TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ']
DPATH = '../data'
Drug_info = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv' % DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt' % DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat' % DPATH
Drug_feature_file_random = '%s/GDSC/drug_graph_feat_random' % DPATH

Genomic_mutation_file = '../data/CCLE/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = '../data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
# Gene_expression_file = '../data/CCLE/umap_10000_40_2.csv'
Methylation_file = '../data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

Drug_info_permutation = '../data/Randomised/drug_permutation.csv'
Drug_info_randomisation = '../data/Randomised/drug_randomisation.csv'

CHECKPOINT = "../checkpoint/normal/best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP_21.12-15:51.h5"

DRUG_SHAPE = 75
MUTATION_SHAPE = 34673
# for testing with other expression inputs, change this value here
EXPR_SHAPE = 697
METHYLATION_SHAPE = 808
UNIT_LIST = [256, 256, 256]
USE_RELU = True
USE_BN = True
USE_GMP = None
USE_MUT = True
USE_GEXP = True
USE_METHY = True


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def ModelTraining(model, X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train,
                  Y_train, validation_data, leaveOut, logdir, params):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=None,
                                                amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=params["patience"], verbose=1, restore_best_weights=True)
    save_path = f'../checkpoint/{leaveOut}/best_DeepCDR_{model_suffix}_{datetime.now().strftime("%d.%m-%H:%M")}'
    checkpoint = ModelCheckpoint(save_path + ".h5", monitor='val_loss', save_best_only=True)
    # dump params dict to have more information about the model params
    with open(save_path + '.json', 'w') as f:
        json.dump(params, f, indent=4)
    tensorboard = cb.TensorBoard(log_dir=logdir)

    callbacks = [checkpoint, tensorboard, earlyStopping, ClearMemory()]
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)  # nb_instance * Max_stom * Max_stom

    train_start = time.time()
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        tf.config.experimental.get_memory_info('GPU:0')['current']
    history = model.fit(x=[X_drug_feat_data_train, X_drug_adj_data_train, X_mutation_data_train, X_gexpr_data_train,
                           X_methylation_data_train], y=Y_train, batch_size=64, epochs=params["max_epoch"],
                        validation_data=validation_data,
                        callbacks=callbacks)
    print(f"Time to train last model: {str(timedelta(seconds=time.time() - train_start))}")
    best_epoch = np.argmin(history.history['val_mse']) + 1
    stopped_epoch = len(history.history['val_mse'])
    return model, history, stopped_epoch, best_epoch


def ModelEvaluate(model, X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test,
                  cancer_type_test_list, file_path):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)  # nb_instance * Max_stom * Max_stom
    Y_pred = model.predict(
        [X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test])
    overall_pcc = pearsonr(Y_pred[:, 0], Y_test)[0]
    print("The overall Pearson's correlation is %.4f." % overall_pcc)
    return overall_pcc


def analyzeDrugLevel(path, group_attribute, sort_attribute, options=None):
    """
    Analyze the model performance grouped by drug
    Args:
        path: data containing predicted value and ground truth for all drug-cell-line-pairs to be considered
        sort_attribute: attribute to sort drugs by
        options: optional list of drugs to consider
    """
    df = pd.read_csv(path)
    if options is None: options = df['drug'].unique()
    df = df[df['drug'].isin(options)]
    # plotIC50BoxPlot(df, group_attribute, sort_attribute)
    plotIC50DotPlot(df, group_attribute)


def savePredictions(Y_test, Y_pred, data_test_idx, path):
    """
    Saves predictions of models
    Args:
        Y_test: ground truth
        Y_pred: predictions
        data_test_idx: ids of test samples
        path: path to save file to
    """
    df = pd.DataFrame(list(zip(Y_test, Y_pred[:, 0], [item[1] for item in data_test_idx], abs(Y_test - Y_pred[:, 0]))),
                      columns=["gt", "pred", "drug", "diff"])
    df["cellline"] = [item[0] for item in data_test_idx]
    df["tissue"] = [item[3] for item in data_test_idx]
    df.to_csv(path, index=False)


def loadAndEvalModel(savePath, test_data_path, modelpath, params_path, zero_Cellline=False, zero_Drug=False,
                     save=False):
    """
    Loads and evaluates a trained model
    Args:
        savePath: path to save the predictions to
        test_data_path: path to test data
        modelpath: path to model
        params_path: path to hyperparameters of the model
        zero_Cellline: whether cell line information should be set to zero
        zero_Drug: whether drug input should be set to zero
        save: whether results should be saved or not
    """

    X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list, model, data_test_idx = getTestData(
        test_data_path, modelpath, params_path)

    if zero_Cellline:
        X_mutation_data_test.fill(0)
        X_methylation_data_test.fill(0)
        X_gexpr_data_test.fill(0)
    if zero_Drug:
        X_drug_feat_data_test = [np.zeros(item[0].shape) for item in X_drug_data_test]
        X_drug_adj_data_test = [np.zeros(item[1].shape) for item in X_drug_data_test]
    else:
        X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
        X_drug_adj_data_test = [item[1] for item in X_drug_data_test]

    X_drug_feat_data_test = np.array(X_drug_feat_data_test)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)  # nb_instance * Max_stom * Max_stom

    Y_pred = model.predict(
        [X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test])
    if save:
        savePredictions(Y_test, Y_pred, data_test_idx, savePath)

    overall_pcc = pearsonr(Y_pred[:, 0], Y_test)[0]
    print("The overall Pearson's correlation of the individual model is %.4f." % overall_pcc)
    return Y_test, Y_pred, data_test_idx


def MetadataGenerateOriginal(Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Drug_feature_file,
                             Gene_expression_file, Methylation_file, filtered):
    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    # map cellline --> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        # if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load demap cell lines genomic mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    cell_line_id_set = list(mutation_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    # load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])

    # only keep overlapped cell lines
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]

    # load methylation
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
    # methylation_feature = methylation_feature.drop('ACH-001190')

    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    # filter experiment data
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[
                                    each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
    return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx


def mean_subtraction(data_idx):
    drug_means = {}
    for drug in set([x[1] for x in data_idx]):
        rows = [x for x in data_idx if x[1] == drug]
        mean = np.mean([x[2] for x in rows])
        drug_means[drug] = mean
        rows = [(x[0], x[1], x[2] - mean, x[3]) for x in rows]
        data_idx = [x for x in data_idx if x[1] != drug] + rows
    return data_idx, drug_means


def getTestData(filename: str, modelpath: str, model_params_path):
    """
    Loads test data
    Args:
        filename: name under which test data should be stored
        modelpath: path to model
        model_params_path: path to model parameters
    Returns: all input and other attributes needed to test a model
    """
    params = json.load(open(model_params_path))
    model = KerasMultiSourceGCNModel(params['use_mut'], params['use_gexp'], params['use_methy']).createMaster(
        DRUG_SHAPE,
        MUTATION_SHAPE,
        EXPR_SHAPE,
        METHYLATION_SHAPE,
        params)

    # Loads the weights
    model.load_weights(modelpath)
    # Load test data
    mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx_orig = MetadataGenerateOriginal(Drug_info,
                                                                                                     Cell_line_info_file,
                                                                                                     Genomic_mutation_file,
                                                                                                     Drug_feature_file,
                                                                                                     Gene_expression_file,
                                                                                                     Methylation_file,
                                                                                                     False)
    data_test_idx = []
    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the header
        next(csvreader)
        for row in csvreader:
            if row[0] == 'ACH-001190':
                continue
            data_test_idx.append([row[0], row[1], float(row[2]), row[3]])

    if params['subtract_mean']:
        tmp1 = {(t[0], t[1]) for t in data_test_idx}
        tmp2 = {(t[0], t[1]) for t in data_idx_orig}

        intersect = tmp1.intersection(tmp2)
        data_train_idx = [x for x in data_idx_orig if (x[0], x[1]) not in intersect]
        # remove drug means from train data as well as from test data
        data_train_idx, drug_means = mean_subtraction(data_train_idx)
        data_test_idx = [(item[0], item[1], item[2] - drug_means[item[1]], item[3]) for item in data_test_idx]

    # Extract features for training and test
    X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list = FeatureExtract(
        data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom=False)

    return X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list, model, data_test_idx


def singleTrainingRun(data_train_idx, data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature,
                      foldIdx, params):
    """
    Trains a single DeepCDR model
    Args:
        data_train_idx: list of indices of training samples
        data_test_idx: list of indices of test samples
        drug_feature: drug input
        mutation_feature: mutation input
        gexpr_feature: gene expression input
        methylation_feature: methylation input
        foldIdx: ID of current fold
        params: model parameters

    Returns:
        statisics of the model performance, after which the training was stopped and the epoch of the best validation performance.

    """
    log_dir = os.path.join(LOG_DIR_BASE, "debug" if params["debug_mode"] else "no_debug", params["leaveOut"],
                           "consider_ratio" if params["consider_ratio"] else "no ratio", f"fold_{foldIdx}",
                           datetime.now().strftime("%d.%m-%H:%M"))
    print('log_dir: ', log_dir)
    # Extract features for training and test
    print("Extract features for training and test")
    X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train, Y_train, cancer_type_train_list = FeatureExtract(
        data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom)
    X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list = FeatureExtract(
        data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom)
    X_drug_adj_data_test, X_drug_feat_data_test = getDrugMatrices(X_drug_data_test)
    data_objects = [X_drug_adj_data_test, X_drug_feat_data_test, X_drug_data_train, X_mutation_data_train,
                    X_gexpr_data_train, X_methylation_data_train, Y_train, cancer_type_train_list, X_drug_data_test,
                    X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list]

    data_size = sum([getsizeof(obj) for obj in data_objects])
    print(f"Size of Data set: {data_size / 1000000} MB")

    validation_data = (
        [X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test],
        Y_test)

    print("build model")
    model = KerasMultiSourceGCNModel(use_mut, use_gexp, use_methy).createMaster(X_drug_data_train[0][0].shape[-1],
                                                                                X_mutation_data_train.shape[-2],
                                                                                X_gexpr_data_train.shape[-1],
                                                                                X_methylation_data_train.shape[-1],
                                                                                params
                                                                                )
    print('Begin training...')

    model, history, earlystop, best_epoch = ModelTraining(model, X_drug_data_train, X_mutation_data_train,
                                                          X_gexpr_data_train,
                                                          X_methylation_data_train, Y_train, validation_data,
                                                          params["leaveOut"], log_dir, params)
    stats = ModelEvaluate(model, X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test,
                          Y_test, cancer_type_test_list, '%s/DeepCDR_%s.log' % (DPATH, model_suffix))

    print("Clean session...")
    tf.keras.backend.clear_session()

    plotStatistic(history, foldIdx, params["leaveOut"], "Regression", params["debug_mode"])
    return stats, earlystop, best_epoch


def compare_tuples(t1, t2):
    if t1[0] != t2[0] or t1[1] != t2[1] or t1[3] != t2[3]:
        return False
    if isinstance(t1[2], float) and isinstance(t2[2], float):
        if not math.isclose(t1[2], t2[2], rel_tol=1e-09, abs_tol=1e-09):
            print("exists")
            return False
    return True


def runKFoldCV(params, train_data_path):
    """
    Trains k models in a cross fold validation and saves their performance to file system.
    Args:
        params: parameters for models to train
    """
    validationScores = []

    # Drug_info_file = Drug_info_randomisation if params["randomise"]["drug"] else Drug_info
    # Drug_feature_file = Drug_feature_file_random
    Drug_info_file = Drug_info_permutation if params["randomise"]["drug"] else Drug_info

    mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx_orig = MetadataGenerate(Drug_info_file,
                                                                                                    Cell_line_info_file,
                                                                                                    Genomic_mutation_file,
                                                                                                    Drug_feature_file,
                                                                                                    Gene_expression_file,
                                                                                                    Methylation_file,
                                                                                                    randomise=params[
                                                                                                        "randomise"],
                                                                                                    debug_mode=params[
                                                                                                        "debug_mode"])

    # this is the bit that causes problems, comment it out to test stuff.
    # be sure to rename data_idx_orig to data_idx in that case
    data_idx = []
    with open(train_data_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        # Skip the header
        next(csvreader)
        for row in csvreader:
            if row[0] == 'ACH-001190':
                continue
            data_idx.append(tuple(row))

    print("Read from train data:", len(data_idx))
    # tmp1 = {(t[0], t[1]) for t in data_idx}
    # tmp2 = {(t[0], t[1]) for t in data_idx_orig}
    #
    # intersect = tmp1.intersection(tmp2)
    # data_idx = [x for x in data_idx if (x[0], x[1]) in intersect]
    data_idx = data_idx_orig
    print("New data_idx:", len(data_idx))
    print(f"Using {Gene_expression_file} expression file")
    splits = getSplits(params, data_idx)
    for index, split in enumerate(splits):

        print(f"Training for fold Nr. {index}")
        data_train_idx, data_test_idx = [data_idx[idx] for idx in split[0]], [data_idx[idx] for idx in split[1]]
        if params['subtract_mean']:
            # data_train_idx: (cell_line, drug, ic50, cancer_type)
            data_train_idx, drug_means = mean_subtraction(data_train_idx)
            # remove means from test data as well from drug_means
            data_test_idx = [(item[0], item[1], item[2] - drug_means[item[1]], item[3]) for item in data_test_idx]

        print(f"len training: {len(data_train_idx)}, len test: {len(data_test_idx)}")
        # check whether folds were constructed properly
        if params["leaveOut"] == "drug_out":
            assert set([item[1] for item in data_train_idx]).isdisjoint(set([item[1] for item in data_test_idx]))
        elif params["leaveOut"] == "celline_out":
            assert set([item[0] for item in data_train_idx]).isdisjoint(set([item[0] for item in data_test_idx]))
        elif params["leaveOut"] == "all_out":
            assert set([item[1] for item in data_train_idx]).isdisjoint(
                set([item[1] for item in data_test_idx])) and set([item[0] for item in data_train_idx]).isdisjoint(
                set([item[0] for item in data_test_idx]))

        validationScores.append(
            singleTrainingRun(data_train_idx, data_test_idx, drug_feature, mutation_feature, gexpr_feature,
                              methylation_feature, index, params))
        if index == params["k"] - 1:
            break

    date_time = datetime.now().strftime("%d.%m.%Y-%H:%M")
    if params["debug_mode"]:
        if not os.path.isdir("Result kfv/Regression/debug/"):
            os.makedirs("Result kfv/Regression/debug/")
        fp = open(fr'Result kfv/Regression/debug/{params["leaveOut"]}_{date_time}', 'w')
    else:
        if not os.path.isdir("Result kfv/Regression/no_debug/"):
            os.makedirs("Result kfv/Regression/no_debug/")
        fp = open(
            fr'Result kfv/Regression/no_debug/{params["leaveOut"]}_ratio_{params["consider_ratio"]}_mul_{params["mul"]}_{date_time}',
            'w')
    for idx, element in enumerate(validationScores):
        fp.write(
            f"Model {idx}, validation Scores (Pearson's) : {element[0]}, stopped after epoch: {element[1]}, best epoch: {element[2]} \n\n")
    fp.close()
    print(
        f'The validation scores for the {params["k"]} folds are (mse, early stopping, best epoch): {validationScores}')


if __name__ == '__main__':
    params = {
        "k": 5,
        "ratio_test_set": 0.05,
        # this is important
        "leaveOut": "normal",
        "debug_mode": False,
        "consider_ratio": True,
        "mul": False,
        "group_by_tissue": False,
        "save_split": False,
        "randomise": {"mutation": False, "methylation": False, "expression": False, "drug": False},
        "hp_tuning": True,
        "patience": 10,
        "max_epoch": 100,
        "use_mut": True,
        "use_gexp": True,
        "use_methy": True,
        "use_bn": True,
        "use_GMP": True,
        "unit_list": [256, 256, 256],
        "use_relu": True,
        "pos_attn_gexpr": -1,
        "pos_attn_mut": -1,
        "pos_attn_methy": -1,
        "lr": 0.001,
        "nb_attn_head_gexpr": 8,
        "nb_attn_head_mut": 8,
        "nb_attn_head_methy": 8,
        "loss": "mse",
        "subtract_mean": True,
        "used_dataset": Gene_expression_file,
    }

    path = "../data/test_data.csv"
    # get the checkpoint file that was edited last
    test = True
    if test:
        # CHECKPOINT = "/nfs/home/students/e.albrecht/tmp/DeepCDR_latest_CUDA/Adapted_DeepCDR_Code/checkpoint/normal/
        # best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP_22.12-08:24.h5"
        print("Testing")
        for file in sorted(glob.glob("../checkpoint/normal/*.h5"), key=os.path.getmtime, reverse=True)[:5]:
            CHECKPOINT = file
            print("Using checkpoint:", CHECKPOINT)
            loadAndEvalModel(path, '../data/FixedSplits/normal_test.csv', CHECKPOINT, CHECKPOINT[:-3] + ".json",
                            zero_Cellline=False, zero_Drug=False, save=True)
    else:
        print("Training")
        runKFoldCV(params, "../data/FixedSplits/normal_train.csv")
