#get drug features using Deepchem library
import os
import deepchem as dc
from rdkit import Chem
import numpy as np
import hickle as hkl
import csv

import random

from rdkit import Chem
import os
import csv


def generatePubChemIDs(original_IDs, target_path):
	random.seed = 9
	# PubChem Database contains 115,668,416 compounds
	random_IDs = set(random.sample(range(115668416), 500))
	reader = csv.reader(open(original_IDs, 'r'))
	rows = {item[5] for item in reader}
	random_IDs = random_IDs - rows
	with open(target_path, "w", newline="") as f:
		write = csv.writer(f)
		write.writerow(list(random_IDs))


def convert_sdf_to_hkl(sdf_file, target_dir, target_pubchem_ids):
	"""Retrieve SMILES for each Molecule from SDF file"""
	suppl = Chem.SDMolSupplier(sdf_file)
	pubchem_IDs = []
	with open("../data/random_pubchem_smiles.txt", 'w') as f:
		for mol in suppl:
			pubchem_id = mol.GetProp('PUBCHEM_COMPOUND_CID')
			smi = Chem.MolToSmiles(mol)

			f.write(f"{pubchem_id}\t{smi}\n")
			pubchem_IDs.append(pubchem_id)
	# safe list of IDs of used moleculs
	with open(target_pubchem_ids, "w") as f:
		wr = csv.writer(f)
		wr.writerow(pubchem_IDs)


def convert_smiles_to_hkl(drug_smiles_file, save_dir):
	for item in open(drug_smiles_file).readlines():
		length = item.split('\t')
		if len(length) < 2:
			print('stop')
	pubchemid2smile = {item.split('\t')[0]: item.split('\t')[1] for item in open(drug_smiles_file).readlines()}
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	molecules = []
	for each in pubchemid2smile.keys():
		print(each)
		molecules = []
		molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
		featurizer = dc.feat.graph_features.ConvMolFeaturizer()
		mol_object = featurizer.featurize(molecules)
		features = mol_object[0].atom_features
		degree_list = mol_object[0].deg_list
		adj_list = mol_object[0].canon_adj_list
		hkl.dump([features, adj_list, degree_list], '%s/%s.hkl' % (save_dir, each))


if __name__ == '__main__':
	input_sdf_file = '../data/PubChem_structures.sdf'
	target_dir = "../data/GDSC/unprocessed_random_drugs/"

	target_path_pubchem_ids = "../data/random_pubchem_ids.csv.csv"
	original_Drugs = '../data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'

	# generatePubChemIDs(original_Drugs, target_path_pubchem_ids)
	"""

	if not os.path.exists(input_sdf_file):
		print("Error: SDF file not found.")
	else:
		convert_sdf_to_hkl(input_sdf_file, target_dir, target_path_pubchem_ids)
	"""
	drug_smiles_file = '../data/223drugs_pubchem_smiles.txt'
	save_dir = '../data/GDSC/drug_graph_feat'

	drug_smiles_file_random = '../data/random_pubchem_smiles.txt'
	save_dir_random = '../data/GDSC/drug_graph_feat_random'

	# source_file = '../data/random_pubchem_ids.csv'
	convert_smiles_to_hkl(drug_smiles_file_random, save_dir_random)
