"""Script to split data into train, validation and test set for hyperparameter tuning"""

import os.path
import pandas as pd
import random
from DataLoader import *
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DPATH = '../data'
Drug_info = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv' % DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt' % DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat' % DPATH
Drug_feature_file_random = '%s/GDSC/drug_graph_feat_random' % DPATH

Genomic_mutation_file = '../data/CCLE/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = '../data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '../data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

random.seed(10)
no_randomise = {"mutation": False, "methylation": False, "expression": False, "drug": False}
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.05263, random_state=42)
mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = MetadataGenerate(Drug_info,
                                                                                                Cell_line_info_file,
                                                                                                Genomic_mutation_file,
                                                                                                Drug_feature_file,
                                                                                                Gene_expression_file,
                                                                                                Methylation_file,
                                                                                                randomise=no_randomise,
                                                                                                debug_mode=False)

drug_labels = [item[1] for item in data_idx]
cell_labels = [item[0] for item in data_idx]

# split drugs out
base_dir = "../data/TuningSplits/drug_out"
left, test_drugs_out = next(gss_test.split(data_idx, groups=drug_labels))

drug_labels_val = [item[1] for index, item in enumerate(data_idx) if index not in test_drugs_out]
train_drugs_out, val_drugs_out = next(gss_val.split(left, groups=drug_labels_val))

df_train = pd.DataFrame([data_idx[left[idx]] for idx in train_drugs_out],
                        columns=["cellline", "drug", "ic50", "tissue"])
df_val = pd.DataFrame([data_idx[left[idx]] for idx in val_drugs_out], columns=["cellline", "drug", "ic50", "tissue"])
df_test = pd.DataFrame([data_idx[idx] for idx in test_drugs_out], columns=["cellline", "drug", "ic50", "tissue"])

df_train.to_csv(os.path.join(base_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(base_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(base_dir, "test.csv"), index=False)

# split cell out
base_dir = "../data/TuningSplits/cell_out"
left, test_cell_out = next(gss_test.split(data_idx, groups=cell_labels))
cell_labels_val = [item[0] for index, item in enumerate(data_idx) if index not in test_cell_out]
train_cell_out, val_cell_out = next(gss_val.split(left, groups=cell_labels_val))

df_train = pd.DataFrame([data_idx[left[idx]] for idx in train_cell_out], columns=["cellline", "drug", "ic50", "tissue"])
df_val = pd.DataFrame([data_idx[left[idx]] for idx in val_cell_out], columns=["cellline", "drug", "ic50", "tissue"])
df_test = pd.DataFrame([data_idx[idx] for idx in test_cell_out], columns=["cellline", "drug", "ic50", "tissue"])

df_train.to_csv(os.path.join(base_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(base_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(base_dir, "test.csv"), index=False)

# split all out
drugs = (list(set([item[1] for item in data_idx])))
random.shuffle(drugs)
drugs_train = drugs[:round(len(drugs) * 0.01)]
drugs_val = drugs[round(len(drugs) * 0.01) + 1:round(len(drugs) * 0.02)]

train_all_out_strict = [sample for sample in data_idx if sample[1] in drugs_train]
train_celllines = list(set(item[0] for item in train_all_out_strict))
excld = [sample for sample in data_idx if sample[0] in train_celllines]
left = list(set(data_idx) - set(excld))

###### split strict
base_dir = "../data/TuningSplits/all_out_strict"
val_all_out_strict = [sample for sample in left if sample[1] in drugs_val]
val_celllines = list(set(item[0] for item in val_all_out_strict))
excld_test = [sample for sample in left if sample[0] in val_celllines]

test_all_out = list(set(left) - set(excld_test))

df_train = pd.DataFrame(train_all_out_strict, columns=["cellline", "drug", "ic50", "tissue"])
df_val = pd.DataFrame(val_all_out_strict, columns=["cellline", "drug", "ic50", "tissue"])
df_test = pd.DataFrame(test_all_out, columns=["cellline", "drug", "ic50", "tissue"])

assert set([item[1] for item in train_all_out_strict]).isdisjoint(set([item[1] for item in test_all_out]))
assert set([item[0] for item in train_all_out_strict]).isdisjoint(set([item[0] for item in test_all_out]))
assert set([item[1] for item in train_all_out_strict]).isdisjoint(set([item[1] for item in val_all_out_strict]))
assert set([item[0] for item in train_all_out_strict]).isdisjoint(set([item[0] for item in val_all_out_strict]))
assert set([item[1] for item in val_all_out_strict]).isdisjoint(set([item[1] for item in test_all_out]))
assert set([item[0] for item in val_all_out_strict]).isdisjoint(set([item[0] for item in test_all_out]))

df_train.to_csv(os.path.join(base_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(base_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(base_dir, "test.csv"), index=False)

######
base_dir = "../data/TuningSplits/all_out"

test_celllines = list(set(item[0] for item in test_all_out))
test_drugs = list(set(item[1] for item in test_all_out))
excld = [sample for sample in data_idx if sample[0] in test_celllines or sample[1] in test_drugs]
left = list(set(data_idx) - set(excld))
train_all_out, val_all_out = train_test_split(left, test_size=0.03)

assert set([item[1] for item in train_all_out]).isdisjoint(set([item[1] for item in test_all_out]))
assert set([item[0] for item in train_all_out]).isdisjoint(set([item[0] for item in test_all_out]))
assert set([item[1] for item in val_all_out]).isdisjoint(set([item[1] for item in test_all_out]))
assert set([item[0] for item in val_all_out]).isdisjoint(set([item[0] for item in test_all_out]))

df_train = pd.DataFrame(train_all_out, columns=["cellline", "drug", "ic50", "tissue"])
df_val = pd.DataFrame(val_all_out, columns=["cellline", "drug", "ic50", "tissue"])

df_train.to_csv(os.path.join(base_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(base_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(base_dir, "test.csv"), index=False)
