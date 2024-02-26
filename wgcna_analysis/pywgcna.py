#!/usr/bin/env python

import PyWGCNA
import pandas as pd

csv = '/cmnfs/home/students/a.schmucklermann/CCLE_depMap_18Q4_TPM_v2.csv'

pyWGCNA_5xFAD = PyWGCNA.WGCNA(name='CCLE_minmodule10', species='human', minModuleSize=10, geneExpPath=csv, outputPath='/cmnfs/home/students/a.schmucklermann/', save=True)
pyWGCNA_5xFAD = pyWGCNA_5xFAD.runWGCNA()
pyWGCNA_5xFAD.saveWGCNA()


#pyWGCNA_5xFAD = PyWGCNA.readWGCNA('/cmnfs/home/students/a.schmucklermann/test.p')
#pyWGCNA_5xFAD.top_n_hub_genes(moduleName="coral", n=10).to_csv('/cmnfs/home/students/a.schmucklermann/top_10_hub_genes.csv')
#pyWGCNA_5xFAD.geneExpr.to_df().to_csv('/cmnfs/home/students/a.schmucklermann/genes.csv')
#pyWGCNA_5xFAD.analyseWGCNA()
