#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:27:45 2022

@author: simonameiler
"""


import numpy as np
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

#Load Climada modules
from climada.engine.unsequa import UncOutput

DATA = Path('./data')

LOGGER = logging.getLogger(__name__)

###########################################################################
############### A: define constants, functions, paths #####################
###########################################################################

# define paths
unsequa_dir = DATA
res_dir = DATA

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


import matplotlib.pyplot as plt
class Handler(object):
    def __init__(self, color1, color2):
        self.color1=color1
        self.color2=color2
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = plt.Rectangle([x0, y0], width, height, facecolor=self.color1,
                                   edgecolor='k', transform=handlebox.get_transform())
        patch2 = plt.Rectangle([x0+width/2., y0], width/2., height, facecolor=self.color2,
                                   edgecolor='k', transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        handlebox.add_artist(patch2)
        return patch

for r, reg in enumerate(region):
    sens1_df_dict[reg][sens1_df_dict[reg] < 0] = 0

#fig, axes = plt.subplots(figsize=(10, 15), ncols=2, layout='constrained', sharey=True)
fig, axes = plt.subplots(figsize=(10, 15), ncols=2, sharey=True)
fig.subplots_adjust(bottom=0.20, wspace=0.07)
bars_aai = []

small_shift = 0.15
bar_height = 0.25
large_shift = 1 #must be integer

output_btt = ['rp100_2090', 'aai_agg_2090', 'rp100', 'aai_agg']

region_btt = region[::-1]

sens_names = [
    'Exposure distribution',
    'SSP exposures',
    'GDP model',
    'Event subsampling future',
    'Event subsampling base',
    'SSP hazard',
    'GCM',
    'Wind model',
    'Vulnerability steepness']

for ax, sens_dict in zip(axes, [sens1_df_dict, sensT_df_dict]):
    #bottom
    for r, reg in enumerate(region_btt):
        val_tot = 0
        for i, val in enumerate(sens_dict[reg][output_btt[0]]):
            bar = ax.barh(r-small_shift, width=val, height=bar_height, left=val_tot, color=colPalette_d[i])
            val_tot += val
            bars_aai.append(bar)
        val_tot = 0
        for i, val in enumerate(sens_dict[reg][output_btt[1]]):
            ax.barh(r+small_shift, width=val, height=bar_height, left=val_tot, color=colPalette_l[i])
            val_tot += val
    #top
    for r, reg in enumerate(region_btt):
        val_tot = 0
        for i, val in enumerate(sens_dict[reg][output_btt[2]]):
            ax.barh(r-small_shift+large_shift+4 , width=val, height=bar_height, left=val_tot, color=colPalette_d[i])
            val_tot += val
        val_tot = 0
        for i, val in enumerate(sens_dict[reg][output_btt[3]]):
            ax.barh(r+small_shift+large_shift+4 , width=val, height=bar_height, left=val_tot, color=colPalette_l[i])
            val_tot += val

axes[0].annotate('Average Annual Impact', xytext=(0.5, 4), xy=(0.8, 5), arrowprops=dict(facecolor='black', shrink=0.05))
axes[1].annotate('Return Period 100', xytext=(0.6, 4), xy=(1.2, 4.8), arrowprops=dict(facecolor='black', shrink=0.05))


plt.setp(axes[0], yticks=range(2*len(region)+large_shift), yticklabels=region_btt  + large_shift*[''] + region_btt )
axes[0].set_xlabel('S1')
axes[1].set_xlabel('ST')

axes[0].text(-0.15, 1.5, '2090', size='large', rotation='vertical')
axes[0].text(-0.15, 6.5, '2050', size='large', rotation='vertical')

handles = [plt.Rectangle((0,0), 1,1) for i in range(len(colPalette_d)+1)]
hmap = dict(zip(handles, [Handler(col1, col2) for col1, col2 in zip(colPalette_l, colPalette_d)]))
axes[0].legend(handles=handles, labels=sens_names, handler_map=hmap, ncol=4, loc='center left',
             bbox_to_anchor=(0, -0.2),fancybox=False, shadow=False, fontsize='medium')



# fig, ax = plt.subplots(figsize=(10, 15))
# bars_aai = []
# for r, reg in enumerate(region):
#     val_tot = 0
#     for i, val in enumerate(sens1_df_dict[reg]['aai_agg']):
#         bar = ax.barh(r/2, width=val, height=0.3, left=val_tot, color=colPalette_l[i])
#         val_tot += val
#         bars_aai.append(bar)
# for r, reg in enumerate(region):
#     val_tot = 0
#     for i, val in enumerate(sens1_df_dict[reg]['rp100']):
#         bar = ax.barh(r/2+4, width=val, height=0.3, left=val_tot, color=colPalette_l[i])
#         val_tot += val
# for r, reg in enumerate(region):
#     val_tot = 0
#     for i, val in enumerate(sens1_df_dict[reg]['aai_agg_2090']):
#         bar = ax.barh(r/2+8, width=val, height=0.3, left=val_tot, color=colPalette_l[i])
#         val_tot += val
# for r, reg in enumerate(region):
#     val_tot = 0
#     for i, val in enumerate(sens1_df_dict[reg]['rp100_2090']):
#         bar = ax.barh(r/2+12, width=val, height=0.3, left=val_tot, color=colPalette_l[i])
#         val_tot += val
# plt.setp(ax, yticks=[0,0.5,1,1.5], yticklabels=region)
# plt.setp(ax, yticks=np.array([0,0.5,1,1.5])+4, yticklabels=region)
# ax.set_xlabel('S1')
# ax.legend(bars_aai, sens1_df_dict['AP'].index.values, loc='center right')
# ax.text(-0.1, 4, 'RP100_2050', size='medium', rotation='vertical')
# ax.text(-0.1, 0, 'AAI_AGG_2050', size='medium', rotation='vertical')
# ax.text(-0.1, 8, 'AAI_AGG_2090', size='medium', rotation='vertical')
# ax.text(-0.1, 12, 'RP_2090', size='medium', rotation='vertical')

