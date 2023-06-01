#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:27:45 2022

@author: simonameiler
"""

import sys
import scipy as sp
import numpy as np
import pandas as pd
import copy as cp
import logging
import seaborn as sns
import matplotlib.pyplot as plt

#Load Climada modules
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.engine.unsequa import UncOutput, CalcDeltaImpact, Calc

    
LOGGER = logging.getLogger(__name__)

###########################################################################
############### A: define constants, functions, paths #####################
###########################################################################

# define paths
unsequa_dir = SYSTEM_DIR/"unsequa"
res_dir = SYSTEM_DIR/"results"

res = 300
ref_year = 2005
region = ['AP', 'IO', 'SH', 'WP']
period = [2050, 2090]
N_samples = 2**11

# make dictionary of unsequa output
output_dict= {}
for reg in region:
    for per in period:
        unsequa_str = f"unsequa_TC_{per}_{reg}_0{res}as_MIT_{N_samples}_v3.hdf5"
        output_imp = UncOutput.from_hdf5(unsequa_dir.joinpath(unsequa_str))
        output_dict[str(reg)+'_'+str(per)] = output_imp

# make dataframe of all results over regions and periods

#output_df = output_imp.samples_df
output_df = cp.deepcopy(output_imp.samples_df)
for reg in region:
    for per in period:
        output_df[str(reg)+'_'+str(per)+'_EAD_unc'] = output_dict[
            str(reg)+'_'+str(per)].aai_agg_unc_df
        output_df[str(reg)+'_'+str(per)+'_rp100_unc'] = output_dict[
            str(reg)+'_'+str(per)].freq_curve_unc_df.rp100

#%%
# add values for climate sensitivity to each GCM
# make dictionary of climate model and value for TCR
TCR_dict = {1: 2.0, 
            2: 2.22, 
            3: 2.30, 
            4: 1.50, 
            5: 2.35, 
            6: 1.55, 
            7: 1.64, 
            8: 1.67, 
            9: 2.77}

ECS_dict = {1: 5.15,
            2: 4.90,
            3: 4.26,
            4: 2.87,
            5: 4.70,
            6: 2.60,
            7: 2.98,
            8: 3.13,
            9: 5.36}
# make a new column for TCR and copy the gc_model values
output_df['TCR'] = output_df['gc_model']
output_df['ECS'] = output_df['gc_model']
# loop through gc_model values and replace each TCR value with the respective number in the TCR_dict
for g in range(1,10):
    output_df.loc[output_df['gc_model'] == g, 'TCR'] = TCR_dict[g]
    output_df.loc[output_df['gc_model'] == g, 'ECS'] = ECS_dict[g]
    
#%%
# sns.scatterplot(data=output_df, x="TCR", y='AP_2050_EAD_unc')

# corr_dict = {}
# for i, risk_metric in enumerate(['AP_2050_EAD_unc',
#     'AP_2050_rp100_unc', 'AP_2090_EAD_unc', 'AP_2090_rp100_unc',
#     'IO_2050_EAD_unc', 'IO_2050_rp100_unc', 'IO_2090_EAD_unc',
#     'IO_2090_rp100_unc', 'SH_2050_EAD_unc', 'SH_2050_rp100_unc',
#     'SH_2090_EAD_unc', 'SH_2090_rp100_unc', 'WP_2050_EAD_unc',
#     'WP_2050_rp100_unc', 'WP_2090_EAD_unc', 'WP_2090_rp100_unc']):
#     corr = output_df["TCR"].corr(output_df[risk_metric])
#     corr_dict[risk_metric] = corr
#     ax = sns.scatterplot(data=output_df, x="TCR", y=risk_metric)
    
#%%
from scipy.stats import spearmanr

# Calculate Spearman's rank correlation coefficient and p-value
corr_dict_spearman = {}
for i, risk_metric in enumerate(['AP_2050_EAD_unc',
    'AP_2050_rp100_unc', 'AP_2090_EAD_unc', 'AP_2090_rp100_unc',
    'IO_2050_EAD_unc', 'IO_2050_rp100_unc', 'IO_2090_EAD_unc',
    'IO_2090_rp100_unc', 'SH_2050_EAD_unc', 'SH_2050_rp100_unc',
    'SH_2090_EAD_unc', 'SH_2090_rp100_unc', 'WP_2050_EAD_unc',
    'WP_2050_rp100_unc', 'WP_2090_EAD_unc', 'WP_2090_rp100_unc']):
    rho, pval = spearmanr(output_df["TCR"], output_df[risk_metric])
    corr_dict_spearman[risk_metric] = rho, pval
    # Print the results
    print("Spearman's correlation coefficient:", rho)
    print("p-value:", pval)
    
    
#%%
from scipy.stats import kendalltau
corr_dict_kendall = {}
for i, risk_metric in enumerate(['AP_2050_EAD_unc',
    'AP_2050_rp100_unc', 'AP_2090_EAD_unc', 'AP_2090_rp100_unc',
    'IO_2050_EAD_unc', 'IO_2050_rp100_unc', 'IO_2090_EAD_unc',
    'IO_2090_rp100_unc', 'SH_2050_EAD_unc', 'SH_2050_rp100_unc',
    'SH_2090_EAD_unc', 'SH_2090_rp100_unc', 'WP_2050_EAD_unc',
    'WP_2050_rp100_unc', 'WP_2090_EAD_unc', 'WP_2090_rp100_unc']):
    tau, pval = kendalltau(output_df["TCR"], output_df[risk_metric])
    corr_dict_kendall[risk_metric] = tau, pval
    # Print the results
    print("Kendall's tau correlation coefficient:", tau)
    print("p-value:", pval)
    
#%%
# make dataframe of all EAD values over all basins
EAD_df_2050 = pd.concat([output_df['AP_2050_EAD_unc'], 
                         output_df['IO_2050_EAD_unc'], 
                         output_df['SH_2050_EAD_unc'], 
                         output_df['WP_2050_EAD_unc']])

EAD_df_2090 = pd.concat([output_df['AP_2090_EAD_unc'], 
                         output_df['IO_2090_EAD_unc'], 
                         output_df['SH_2090_EAD_unc'], 
                         output_df['WP_2090_EAD_unc']])

rp100_df_2050 = pd.concat([output_df['AP_2050_rp100_unc'], 
                         output_df['IO_2050_rp100_unc'], 
                         output_df['SH_2050_rp100_unc'], 
                         output_df['WP_2050_rp100_unc']])

rp100_df_2090 = pd.concat([output_df['AP_2090_rp100_unc'], 
                         output_df['IO_2090_rp100_unc'], 
                         output_df['SH_2090_rp100_unc'], 
                         output_df['WP_2090_rp100_unc']])

TCR_df = pd.concat([output_df["TCR"], 
                    output_df["TCR"], 
                    output_df["TCR"], 
                    output_df["TCR"]])

#%%
spearman_dict_global = dict()
for i,risk in enumerate([EAD_df_2050, EAD_df_2090, rp100_df_2050, rp100_df_2090]):
    rho, pval = spearmanr(TCR_df, risk)
    spearman_dict_global[i] = [rho, pval]
    
#%%
corr_dict_spearman = {}
for i, risk_metric in enumerate(['AP_2050_EAD_unc',
    'AP_2050_rp100_unc', 'AP_2090_EAD_unc', 'AP_2090_rp100_unc',
    'IO_2050_EAD_unc', 'IO_2050_rp100_unc', 'IO_2090_EAD_unc',
    'IO_2090_rp100_unc', 'SH_2050_EAD_unc', 'SH_2050_rp100_unc',
    'SH_2090_EAD_unc', 'SH_2090_rp100_unc', 'WP_2050_EAD_unc',
    'WP_2050_rp100_unc', 'WP_2090_EAD_unc', 'WP_2090_rp100_unc']):
    rho, pval = spearmanr(output_df["ECS"], output_df[risk_metric])
    corr_dict_spearman[risk_metric] = [rho, pval]
    # Print the results
    print("Spearman's correlation coefficient:", rho)
    print("p-value:", pval)

#%%
df = pd.read_excel('/Users/simonameiler/Desktop/haz_freq_corr.xlsx', sheet_name='Sheet2')

spearman_dict_global_fi = dict()
for i,risk in enumerate(["freq_2050", "freq_2090", "int_2050", "int_2090"]):
    rho, pval = spearmanr(df["TCR"], df[risk])
    spearman_dict_global_fi[i] = [rho, pval]