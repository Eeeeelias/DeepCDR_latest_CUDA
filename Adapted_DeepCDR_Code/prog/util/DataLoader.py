"""Handles correct loading and initialization of input data.

Helps to calucalte drug adjacency and feature matrix. Furthermore, handles data splitting.
"""

import numpy as np
import pandas as pd
import random
import csv
import os
import time
import hickle as hkl
import scipy.sparse as sp
from sklearn.model_selection import KFold, GroupKFold

##############################################################################

TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ']
Max_atoms = 100

Genomic_mutation_file_debug = '../data/CCLE/Sorted/Genomic_mutation_file_sorted.csv'
Gene_expression_file_debug = '../data/CCLE/Sorted/Gene_expression_file_sorted.csv'
Methylation_file_debug = '../data/CCLE/Sorted/Methylation_file_sorted.csv'

Genomic_mutation_file_random = '../data/Randomised/Row/genomic_mutation.csv'
Gene_expression_file_random = '../data/Randomised/Row/genomic_expression.csv'
Methylation_file_random = '../data/Randomised/Row/genomic_methylation.csv'

Cancer_response_exp_file = '../data/CCLE/GDSC_IC50.csv'
MAX_ROWS_DEBUG = 50

##############################################################################

def getDrugMatrices(X_drug_data_test):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)  # nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)  # nb_instance * Max_stom * Max_stom
    return X_drug_adj_data_test, X_drug_feat_data_test

# split into training and test set
def DataSplit(data_idx, ratio=0.95):
    """Takes a list of data indices and splits them into train and test set according to ratio.

    Args:
        data_idx: list of all data indices to be split
        ratio: ratio of train set

    Returns:
        data_train_idx: data indices for train set
        data_test_idx: data indices for test set
    """
    data_train_idx, data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx, data_test_idx


def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix


def CalculateGraphFeat(feat_mat, adj_list, israndom):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms, feat_mat.shape[-1]), dtype='float32')
    adj_mat = np.zeros((Max_atoms, Max_atoms), dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms, feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms - feat_mat.shape[0])
    feat[:feat_mat.shape[0], :] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    adj_ = adj_mat[:len(adj_list), :len(adj_list)]
    adj_2 = adj_mat[len(adj_list):, len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2
    return [feat, adj_mat]


def FeatureExtract(data_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature, israndom):
    cancer_type_list = []

    nb_instance = len(data_idx)
    nb_mutation_feature = mutation_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    mutation_data = np.zeros((nb_instance, 1, nb_mutation_feature, 1), dtype='float32')
    gexpr_data = np.zeros((nb_instance, nb_gexpr_features), dtype='float32')
    methylation_data = np.zeros((nb_instance, nb_methylation_features), dtype='float32')
    target = np.zeros(nb_instance, dtype='float32')
    for idx in range(nb_instance):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
        # modify
        feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
        # fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list, israndom)
        # randomlize X A
        mutation_data[idx, 0, :, 0] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx, :] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx, :] = methylation_feature.loc[cell_line_id].values
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
    return drug_data, mutation_data, gexpr_data, methylation_data, target, cancer_type_list


def split_all_out(data_idx, k):
    """
    Performs data split under the condition, that neither drugs nor cell lines are shared across training and test set.
    Args:
        data_idx: List of data indices to be split
        k: Number of folds to be split

    Returns:
        K data splits
    """
    print("start data splitting")
    drugs = list(set([item[1] for item in data_idx]))
    drugs = random.sample(drugs, k)
    splits = []
    for i in range(len(drugs)):
        test_split = [sample for sample in data_idx if sample[1] == drugs[i]]
        test_celllines = list(set(item[0] for item in test_split))
        excld_train = [sample for sample in data_idx if sample[0] in test_celllines]
        train_split = list(set(data_idx) - set(excld_train))

        splits.append(
            [[data_idx.index(sample) for sample in train_split], [data_idx.index(sample) for sample in test_split]])
        print("finished split %d" % i)
    return splits


def MetadataGenerate(Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Drug_feature_file,
                     Gene_expression_file, Methylation_file, randomise, debug_mode=False):
    start = time.time()

    if debug_mode:
        mutation_feature = pd.read_csv(Genomic_mutation_file_debug, sep=',', header=0, index_col=[0])
        gexpr_feature = pd.read_csv(Gene_expression_file_debug, sep=',', header=0, index_col=[0])
        methylation_feature = pd.read_csv(Methylation_file_debug, sep=',', header=0, index_col=[0])

        common_indices = mutation_feature.index.intersection(gexpr_feature.index).intersection(
            methylation_feature.index)
        mutation_feature = mutation_feature.loc[list(common_indices)]
        methylation_feature = methylation_feature.loc[list(common_indices)]
        gexpr_feature = gexpr_feature.loc[list(common_indices)]

    else:
        mutation_feature = pd.read_csv(Genomic_mutation_file_random if randomise["mutation"] else Genomic_mutation_file,
                                       sep=',', header=0, index_col=[0])
        gexpr_feature = pd.read_csv(Gene_expression_file_random if randomise["expression"] else Gene_expression_file,
                                    sep=',', header=0, index_col=[0])
        methylation_feature = pd.read_csv(Methylation_file_random if randomise["methylation"] else Methylation_file,
                                          sep=',', header=0, index_col=[0])
        if randomise['mutation']:
            print("Using randomised mutation data")
        if randomise['expression']:
            print("Using randomised expression data")
        if randomise['methylation']:
            print("Using randomised methylation data")
        if randomise['drug']:
            print("Using randomised drug data")

        mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]

    # remove cell line 'ACH-001190' from all features due to it not being in the original feature data?
    # How did it get here? Only god may know
    # mutation_feature = mutation_feature.drop('ACH-001190')
    # gexpr_feature = gexpr_feature.drop('ACH-001190')
    methylation_feature = methylation_feature.drop('ACH-001190')

    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    if randomise["drug"]:
        drugid2pubchemid = {item[0]: item[1] for item in rows}
    else:
        drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    # map cellline --> cancer type
    cellline2cancertype = {}
    counter = 0
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        # if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label
        if debug_mode and counter == MAX_ROWS_DEBUG: break
        counter += 1

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    counter = 0
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
        counter += 1
        if debug_mode and counter == MAX_ROWS_DEBUG: break

    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    if debug_mode:
        experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0], nrows=MAX_ROWS_DEBUG)
    else:
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
    print(f"preprocessing time: {time.time() - start}")
    return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx


def getSplits(params, data_idx):
    """
    Splits the data into folds according to configuration in params.

    Args:
        params: list of parameters from model configuration. Among others contain the splitting mode (leave drug out or similar) and the size of the rest set
        data_idx: list of all data samples to be considered

    Returns:
        data splits according to configuration in params (e.g. leave drug out, leave cell out, number of folds)
    """
    if params["leaveOut"] == "normal":
        if params["group_by_tissue"]:
            tissue_types = [item[3] for item in data_idx]
            kf = GroupKFold(n_splits=int(1 / params["ratio_test_set"])) if params["consider_ratio"] else GroupKFold(
                n_splits=params["k"])
            splits = kf.split(data_idx, groups=tissue_types)
        else:
            kf = KFold(n_splits=int(1 / params["ratio_test_set"]), shuffle=True) if params["consider_ratio"] else KFold(
                n_splits=params["k"])
            splits = kf.split(data_idx)
    elif params["leaveOut"] == "all_out":
        splits = split_all_out(data_idx, params["k"])
    else:
        groups = {
            "drug_out": [item[1] for item in data_idx],
            "cellline_out": [item[0] for item in data_idx],
        }
        kf = GroupKFold(n_splits=int(1 / params["ratio_test_set"])) if params["consider_ratio"] else GroupKFold(
            n_splits=params["k"])
        splits = kf.split(data_idx, groups=groups.get(params["leaveOut"]))
    if params["save_split"]:
        path_train = f'../data/FixedSplits/{params["leaveOut"]}_train.csv'
        path_test = f'../data/FixedSplits/{params["leaveOut"]}_test.csv'
        for split in splits:
            data_train_idx, data_test_idx = [data_idx[idx] for idx in split[0]], [data_idx[idx] for idx in split[1]]
            df_train = pd.DataFrame(data_train_idx, columns=["cellline", "drug", "ic50", "tissue"])
            df_test = pd.DataFrame(data_test_idx, columns=["cellline", "drug", "ic50", "tissue"])
            df_train.to_csv(path_train, index=False)
            df_test.to_csv(path_test, index=False)
            break

    return splits

