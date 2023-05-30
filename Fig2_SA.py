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
from matplotlib.patches import Patch

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

#%% rearrange dataframe for plotting
lst = ['mn_exp',
       'ssp_exp',
       'gdp_model',
       'HE_fut',
       'HE_base',
       'ssp_haz',
       'gc_model',
       'wind_model',
       'v_half']

sens1_df_dict = {}
sensT_df_dict = {}
conf1_df_dict = {}
confT_df_dict = {}
for reg in region:
    df1_S1 = output_dict[str(reg)+'_2050'].get_sensitivity(salib_si='S1')
    df1_S1_conf = output_dict[str(reg)+'_2050'].get_sensitivity(salib_si='S1_conf')
    df2_S1 = output_dict[str(reg)+'_2090'].get_sensitivity(salib_si='S1')
    df2_S1_conf = output_dict[str(reg)+'_2090'].get_sensitivity(salib_si='S1_conf')
    
    dfS1 = df1_S1[["param","aai_agg","rp100"]]
    dfS1["aai_agg_2090"] = df2_S1.aai_agg
    dfS1["rp100_2090"] = df2_S1.rp100
    dfS1[""] = df2_S1.aai_agg - df2_S1.aai_agg
    dfS1 = dfS1.reindex(columns=["param", "aai_agg", "aai_agg_2090", "", "rp100", "rp100_2090"])
    dfS1 = dfS1.set_index("param")
    dfS1_reord = dfS1.reindex(lst)
    dfS1_conf = df1_S1_conf[["param","aai_agg","rp100"]]
    dfS1_conf["aai_agg_2090"] = df2_S1_conf.aai_agg
    dfS1_conf["rp100_2090"] = df2_S1_conf.rp100
    dfS1_conf[""] = df2_S1.aai_agg - df2_S1.aai_agg
    dfS1_conf = dfS1_conf.set_index("param")
    dfS1_conf = dfS1_conf.reindex(columns=["param", "aai_agg", "aai_agg_2090", "", "rp100", "rp100_2090"])
    dfS1_conf_reord = dfS1_conf.reindex(lst)
    sens1_df_dict[reg] = dfS1_reord
    conf1_df_dict[reg] = dfS1_conf_reord


    df1_ST = output_dict[str(reg)+'_2050'].get_sensitivity(salib_si='ST')
    df1_ST_conf = output_dict[str(reg)+'_2050'].get_sensitivity(salib_si='ST_conf')
    df2_ST = output_dict[str(reg)+'_2090'].get_sensitivity(salib_si='ST')
    df2_ST_conf = output_dict[str(reg)+'_2090'].get_sensitivity(salib_si='ST_conf')
    
    dfST = df1_ST[["param","aai_agg","rp100"]]
    dfST["aai_agg_2090"] = df2_ST.aai_agg
    dfST["rp100_2090"] = df2_ST.rp100
    dfST[""] = df2_ST.aai_agg - df2_ST.aai_agg
    dfST = dfST.reindex(columns=["param", "aai_agg", "aai_agg_2090", "", "rp100", "rp100_2090"])
    dfST = dfST.set_index("param")
    dfST_reord = dfST.reindex(lst)
    dfST_conf = df1_ST_conf[["param","aai_agg","rp100"]]
    dfST_conf["aai_agg_2090"] = df2_S1_conf.aai_agg
    dfST_conf["rp100_2090"] = df2_ST_conf.rp100
    dfST_conf[""] = df2_S1.aai_agg - df2_S1.aai_agg
    dfST_conf = dfST_conf.set_index("param")
    dfST_conf = dfST_conf.reindex(columns=["param", "aai_agg", "aai_agg_2090", "", "rp100", "rp100_2090"])
    dfST_conf_reord = dfST_conf.reindex(lst)
    sensT_df_dict[reg] = dfST_reord
    confT_df_dict[reg] = dfST_conf_reord


#%% make plot of both, S1 and ST
# lst = ['mn_exp',
#        'ssp_exp',
#        'gdp_model',
#        'HE_fut',
#        'HE_base',
#        'ssp_haz',
#        'gc_model',
#        'wind_model',
#        'v_half']

# labels_list_1 = ["a)", "c)", "e)", "g)"]
# labels_list_2 = ["b)", "d)", "f)", "h)"]
# y = np.arange(len(lst))
# colPalette_d = sns.hls_palette(n_colors=len(y), l=0.3, s=1.)
# colPalette_l = sns.hls_palette(n_colors=len(y), l=0.7, s=1.)

# legend_elements = [Patch(edgecolor='darkgrey', facecolor='darkgrey',
#                          label="\u0394 EAD 2050 (%)"),
#                    Patch(hatch='///', edgecolor='k', facecolor='lightgrey',
#                          label="\u0394 EAD 2090 (%)"),
#                    Patch(hatch='xx', edgecolor='k', facecolor='darkgrey',
#                          label="\u0394 rp100 2050 (%)"),
#                    Patch(hatch='**', edgecolor='k', facecolor='lightgrey',
#                          label="\u0394 rp100 2090 (%)")]

# fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(12,18), sharex=True, sharey=True)
# plt.subplots_adjust(wspace = 0.1)
# #fig.tight_layout()
# #plt.suptitle("First order sensitivity", x=0.1, y=0.98)
# #plt.suptitle("Total order sensitivity", x=0.6, y=0.98)
# #ax = ax.flatten()

# height = 0.2
# for i in range(2):
#     for r, reg in enumerate(region):
#         mid_cent_a_S1 = ax[r,0].barh(y+0.45, sens1_df_dict[reg].aai_agg, xerr=conf1_df_dict[reg].aai_agg, 
#                                 color=colPalette_d, height=height)
#         end_cent_a_S1 = ax[r,0].barh(y+0.25, sens1_df_dict[reg].aai_agg_2090, xerr=conf1_df_dict[reg].aai_agg_2090, 
#                                 color=colPalette_l, hatch='///', height=height)
#         mid_cent_rp_S1 = ax[r,0].barh(y-0.05, sens1_df_dict[reg].rp100, xerr=conf1_df_dict[reg].rp100,
#                                 height=height, color=colPalette_d, hatch='xx')
#         end_cent_rp_S1 = ax[r,0].barh(y-0.25, sens1_df_dict[reg].rp100_2090, xerr=conf1_df_dict[reg].rp100_2090,
#                                 height=height, color=colPalette_l, hatch='**')
#         mid_cent_a_ST = ax[r,1].barh(y+0.45, sensT_df_dict[reg].aai_agg, xerr=confT_df_dict[reg].aai_agg, 
#                                 color=colPalette_d, label="\u0394 EAD (%)", height=height)
#         end_cent_a_ST = ax[r,1].barh(y+0.25, sensT_df_dict[reg].aai_agg_2090, xerr=confT_df_dict[reg].aai_agg_2090, 
#                                 color=colPalette_l, hatch='///', height=height)      
#         mid_cent_rp_ST = ax[r,1].barh(y-0.05, sensT_df_dict[reg].rp100, xerr=confT_df_dict[reg].rp100,
#                                 height=height, hatch='xx',color=colPalette_d)
#         end_cent_rp_ST = ax[r,1].barh(y-0.25, sensT_df_dict[reg].rp100_2090, xerr=confT_df_dict[reg].rp100_2090,
#                                 height=height, color=colPalette_l, hatch='**')
#         #ax[r,i].text(0.9, 0.5, reg, transform=ax[r,i].transAxes, fontsize=16)
#         ax[0,0].set_title("Frist-order sensitivity", fontsize=16)
#         ax[0,1].set_title("Total-order sensitivity", fontsize=16)
#         ax[r,0].set_ylabel(reg, fontsize=16)
#         #ax[0,1].set_title("Total order sensitivity", fontsize=16)
#         ax[r,0].text(-0.35, 1.05, labels_list_1[r], transform=ax[r,0].transAxes, fontsize=16, fontweight="bold")
#         ax[r,1].text(-0.1, 1.05, labels_list_2[r], transform=ax[r,1].transAxes, fontsize=16, fontweight="bold")
#         ax[r,i].set_yticks(y)
#         ax[r,i].set_yticklabels(lst, rotation=0)
#         ax[2,1].legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1.5))
#         ax[r,i].spines['right'].set_visible(False)
#         ax[r,i].spines['top'].set_visible(False)
#         ax[r,i].spines['left'].set_visible(False)
#         ax[r,i].spines['bottom'].set_visible(True)

# save_fig_str = "SA_TC_risk_MIT.png"
# plt.savefig(res_dir.joinpath(save_fig_str), dpi=300, facecolor='w', 
#             edgecolor='w', orientation='portrait', papertype=None, 
#             format='png', bbox_inches='tight', pad_inches=0.1) 

#%% make plot of both, S1 and ST - no confidence bars
lst = ['mn_exp',
       'ssp_exp',
       'gdp_model',
       'HE_fut',
       'HE_base',
       'ssp_haz',
       'gc_model',
       'wind_model',
       'v_half']

labels_list_1 = ["a)", "c)", "e)", "g)"]
labels_list_2 = ["b)", "d)", "f)", "h)"]
y = np.arange(len(lst))
colPalette_d = sns.hls_palette(n_colors=len(y), l=0.3, s=1.)
colPalette_l = sns.hls_palette(n_colors=len(y), l=0.7, s=1.)

legend_elements = [Patch(edgecolor='darkgrey', facecolor='darkgrey',
                         label="\u0394 EAD 2050 (%)"),
                   Patch(hatch='///', edgecolor='k', facecolor='lightgrey',
                         label="\u0394 EAD 2090 (%)"),
                   Patch(hatch='xx', edgecolor='k', facecolor='darkgrey',
                         label="\u0394 rp100 2050 (%)"),
                   Patch(hatch='**', edgecolor='k', facecolor='lightgrey',
                         label="\u0394 rp100 2090 (%)")]

fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(12,18), sharex=True, sharey=True)
plt.subplots_adjust(wspace = 0.1)

height = 0.2
for i in range(2):
    for r, reg in enumerate(region):
        mid_cent_a_S1 = ax[r,0].barh(y+0.45, sens1_df_dict[reg].aai_agg,
                                color=colPalette_d, height=height)
        end_cent_a_S1 = ax[r,0].barh(y+0.25, sens1_df_dict[reg].aai_agg_2090, 
                                color=colPalette_l, hatch='///', height=height)
        mid_cent_rp_S1 = ax[r,0].barh(y-0.05, sens1_df_dict[reg].rp100, 
                                height=height, color=colPalette_d, hatch='xx')
        end_cent_rp_S1 = ax[r,0].barh(y-0.25, sens1_df_dict[reg].rp100_2090, 
                                height=height, color=colPalette_l, hatch='**')
        mid_cent_a_ST = ax[r,1].barh(y+0.45, sensT_df_dict[reg].aai_agg, 
                                color=colPalette_d, label="\u0394 EAD (%)", height=height)
        end_cent_a_ST = ax[r,1].barh(y+0.25, sensT_df_dict[reg].aai_agg_2090, 
                                color=colPalette_l, hatch='///', height=height)      
        mid_cent_rp_ST = ax[r,1].barh(y-0.05, sensT_df_dict[reg].rp100, 
                                height=height, hatch='xx',color=colPalette_d)
        end_cent_rp_ST = ax[r,1].barh(y-0.25, sensT_df_dict[reg].rp100_2090, 
                                height=height, color=colPalette_l, hatch='**')
        #ax[r,i].text(0.9, 0.5, reg, transform=ax[r,i].transAxes, fontsize=16)
        ax[0,0].set_title("Frist-order sensitivity", fontsize=16)
        ax[0,1].set_title("Total-order sensitivity", fontsize=16)
        ax[r,0].set_ylabel(reg, fontsize=16)
        #ax[0,1].set_title("Total order sensitivity", fontsize=16)
        ax[r,0].text(-0.35, 1.05, labels_list_1[r], transform=ax[r,0].transAxes, fontsize=16, fontweight="bold")
        ax[r,1].text(-0.1, 1.05, labels_list_2[r], transform=ax[r,1].transAxes, fontsize=16, fontweight="bold")
        ax[r,i].set_yticks(y)
        ax[r,i].set_yticklabels(lst, rotation=0)
        ax[3,1].legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0, -0.3))
        ax[r,i].spines['right'].set_visible(False)
        ax[r,i].spines['top'].set_visible(False)
        ax[r,i].spines['left'].set_visible(False)
        ax[r,i].spines['bottom'].set_visible(True)

# save_fig_str = "SA_TC_risk_MIT.png"
# plt.savefig(res_dir.joinpath(save_fig_str), dpi=300, facecolor='w', 
#             edgecolor='w', orientation='portrait', papertype=None, 
#             format='png', bbox_inches='tight', pad_inches=0.1) 





