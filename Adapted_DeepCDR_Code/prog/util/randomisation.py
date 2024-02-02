"""Methods to randomize different kinds of input data"""

import pandas as pd
import random
import csv
import numpy as np
import random

PATH_METHYLATION = "../data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv"
PATH_EXPRESSION = "../data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv"
PATH_MUTATION = "../data/CCLE/genomic_mutation_34673_demap_features.csv"

TARGET_METHYLATION_ROW = "../data/Randomised/Row/genomic_methylation.csv"
TARGET_MUTATION_ROW = "../data/Randomised/Row/genomic_mutation.csv"
TARGET_EXPRESSION_ROW = "../data/Randomised/Row/genomic_expression.csv"

TARGET_METHYLATION_COL = "../data/Randomised/Column/genomic_methylation.csv"
TARGET_MUTATION_COL = "../data/Randomised/Column/genomic_mutation.csv"
TARGET_EXPRESSION_COL = "../data/Randomised/Column/genomic_expression.csv"

TARGET_METHYLATION_ROW = "../data/Randomised/genomic_methylation.csv"
TARGET_MUTATION_ROW = "../data/Randomised/genomic_mutation.csv"
TARGET_EXPRESSION_ROW = "../data/Randomised/genomic_expression.csv"


PATH_ORIGINAL_DRUG = '../data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
PATH_DRUG_RANDOMISED = '../data/random_pubchem_ids.csv'

TARGET_DRUG_PERMUTATION = "../data/Randomised/drug_permutation.csv"
TARGET_DRUG_RANDOMISED = "../data/Randomised/drug_randomisation.csv"


def randomise_columns(src_path, target_path):
    """
    Randomizes a dataframe by permuting columns
    Args:
        src_path: path to the dataframe to be randomized
        target_path: path to the location where randomized dataframe should be saved.
    """
    df_original = pd.read_csv(src_path, sep=',')
    columns_old = df_original.columns.tolist()
    cols = df_original.columns.tolist()
    first_col = cols[0]
    cols = cols[1:]
    random.shuffle(cols)
    cols.insert(0, first_col)
    print(len([i for i, j in zip(columns_old, cols) if i == j]))
    df = df_original[cols]
    df.to_csv(target_path, index=False)


def randomise_rows(src_path, target_path):
    """
        Randomizes a dataframe by permuting rows
        Args:
            src_path: path to the dataframe to be randomized
            target_path: path to the location where randomized dataframe should be saved.
        """
    np.random.seed = 0
    df_original = pd.read_csv(src_path, sep=',')
    df = pd.read_csv(src_path, sep=',')
    print(df.head())
    for column_name in df.columns[1:len(df.columns)]:
        # Get the shuffled index
        shuffled_index = np.random.permutation(df.index)
        # Reorder the values in the desired column based on the shuffled index
        df[column_name] = df[column_name].iloc[shuffled_index].reset_index(drop=True)
    equal_values_count = (df_original.drop([0]) == df.drop([0])).values.sum()
    relative_equal_values = "{:.2%}".format(equal_values_count / df.drop([0]).size)
    print(f"Randomised dataframe contains {equal_values_count} ({relative_equal_values}) similar entries")
    df.to_csv(target_path, index=False)


def permute_drug(source, target):
    """
    Permutes the identifier of drug information
    Args:
        source: location of original mapping of drug id to puchem id. The latter yields the drug information.
        target: path to which permuted mapping should be saved to
    """
    drugid2pubchemid = load_drugid2pubchemid(source)

    vals = list(drugid2pubchemid.values())
    random.shuffle(vals)
    shuffledPairs = (zip(drugid2pubchemid, vals))
    shuffledPairs = sorted(shuffledPairs, key=lambda x: x[0])
    save_file(target, shuffledPairs)


def randomise_drug(source_origin, source_random, target):
    """
    Maps drug IDs occuring in the dataset to random pubchem IDs of different drugs.
    Args:
        source_origin: source of old mapping from drug ID to puchem ID
        source_random: source of random pubchem IDs that do not appear in the original dataset
        target: location to save randomised mapping to
    """
    drugid2pubchemid = load_drugid2pubchemid(source_origin)

    reader = csv.reader(open(source_random, 'r'))
    for row in reader:
        random_vals = row

    randomised_pairs = (zip(drugid2pubchemid, random_vals[:len(drugid2pubchemid)]))
    randomised_pairs = sorted(randomised_pairs, key=lambda x: x[0])
    save_file(target, randomised_pairs)


def load_drugid2pubchemid(source):
    """
    Loads a list that maps drug IDs to their corresponding puchem ID

    Args:
        source: location of the file

    Returns: loaded csv file
    """
    reader = csv.reader(open(source, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}
    return drugid2pubchemid


def save_file(target, data):
    """
    Writes data to a target destination
    Args:
        target: location to save data to
        data: data to be saved
    """
    with open(target, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    # randomise_rows(PATH_METHYLATION, TARGET_METHYLATION_ROW)
    # randomise_rows(PATH_MUTATION, TARGET_MUTATION_ROW)
    # randomise_rows(PATH_EXPRESSION, TARGET_EXPRESSION_ROW)
    # permute_drug(PATH_DRUG_PERMUTATION, TARGET_DRUG_PERMUTATION)
    # randomise_drug(PATH_ORIGINAL_DRUG, PATH_DRUG_RANDOMISED, TARGET_DRUG_RANDOMISED)
    randomise_columns(PATH_METHYLATION, TARGET_METHYLATION_COL)
    randomise_columns(PATH_MUTATION, TARGET_MUTATION_COL)
    randomise_columns(PATH_EXPRESSION, TARGET_EXPRESSION_COL)
