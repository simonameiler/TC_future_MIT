"""
Adapted for code repository on 2023-06-22

description: Figure 1 - plotting of TC risk change drivers
             Supplementary Information Table S1 and S2 - values describing the 
             boxplots of Figure 1 in more detail.

@author: simonameiler
"""

import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

#Load Climada modules
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.engine.unsequa import UncOutput

    
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
deltas = ["", "_expconst", "_hazconst"]
delta_dict = {"": "total",
              "_expconst": "CC",
              "_hazconst": "SOC"}
N_samples = 2**11

# make dictionary of unsequa output
output_dict= {}
for reg in region:
    for delta in deltas:
        for per in period:
            if delta == "":
                unsequa_str = f"unsequa_TC_{per}_{reg}_0{res}as_MIT_2048_v3.hdf5"
                output_imp = UncOutput.from_hdf5(unsequa_dir.joinpath(unsequa_str))
                output_dict[str(reg)+'_'+str(per)+'_'+str(delta_dict[delta])] = output_imp
            else:
                unsequa_str = f"unsequa_TC_{per}_{reg}_0{res}as_MIT_{N_samples}_v3{delta}.hdf5"
                output_imp = UncOutput.from_hdf5(unsequa_dir.joinpath(unsequa_str))
                output_dict[str(reg)+'_'+str(per)+'_'+str(delta_dict[delta])] = output_imp
                
#%%
delta_df_dict = {}
for reg in region:
    delta_df_list = []
    for delta in deltas:
        for per in period:
            df = pd.DataFrame()
            df['EAD'] = output_dict[str(reg)+'_'+str(per)+'_'+str(delta_dict[delta])].aai_agg_unc_df
            df['rp100'] = output_dict[str(reg)+'_'+str(per)+'_'+str(delta_dict[delta])].freq_curve_unc_df.rp100
            df['delta'] = delta_dict[delta]
            df['year'] = per
            delta_df_list.append(df)

    delta_df = pd.concat(delta_df_list)
    df_cc = delta_df[delta_df.delta == 'CC']
    df_soc = delta_df[delta_df.delta == 'SOC']
    df_sum = df_cc + df_soc
    df_sum.delta = 'sum'
    df_sum.year[df_sum.year == 4100] = 2050
    df_sum.year[df_sum.year == 4180] = 2090
    delta_df = pd.concat([delta_df, df_sum])
    delta_df_dict[reg] = delta_df

#%% wold map incl. region boundaries and results

BASIN_BOUNDS = {
    # North Atlantic/Eastern Pacific Basin
    'AP': [-180.0, -20.0, 0.0, 65.0],

    # Indian Ocean Basin
    'IO': [30.0, 100.0, 0.0, 40.0],

    # Southern Hemisphere Basin
    'SH': [-180.0, 180.0, -60.0, 0.0],

    # Western Pacific Basin
    'WP': [100.0, 180.0, 0.0, 65.0],
}



labels_dict = {(0,0): 'b)',
               (0,1): 'c)',
               (0,2): 'd)',
               (0,3): 'e)',
               (1,0): 'f)',
               (1,1): 'g)',
               (1,2): 'h)',
               (1,3): 'i)'}

metric = ["EAD", "rp100"]

fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(12,6), sharex=True, sharey=True)
for m, met in enumerate(metric):
    for r, reg in enumerate(['AP', 'IO', 'SH', 'WP']):
        sns.boxplot(data=delta_df_dict[reg], x="delta", hue="year", y=met, 
                    order=["CC","SOC","sum", "total"], width=.4, showfliers = False, ax=ax[m,r])
        ax[m,r].get_legend().remove()
        ax[m,r].set_yscale('symlog')
        sns.despine()
        ax[m,r].set(xlabel="", ylabel="")
        ax[0,r].text(0.05, 0.9, f"{reg}", transform=ax[0,r].transAxes, fontsize=12)
        ax[1,r].text(0.05, 0.9, f"{reg}", transform=ax[1,r].transAxes, fontsize=12)
        ax[m,r].text(-0.1, 1.05, labels_dict[m,r], transform=ax[m,r].transAxes, 
                         fontsize=16, fontweight="bold")
        ax[m,0].set(xlabel="", ylabel=f"\u0394 {met} (%)")
        ax[m,r].axhline(0, ls='dotted', color='k')
        #ax[m,r].set_ylim(bottom=10**-2, top=10**2)
handles, labels = ax[1,1].get_legend_handles_labels()
ax[1,3].legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1, 1.25))



ax = plt.gcf().add_axes([0.25, 1, 0.5, 0.5], projection=ccrs.Robinson())
ax.text(0.05, 0.9, "a)", transform=ax.transAxes, fontsize=16, fontweight="bold")
ax.text(0.1, 0.7, "North Atlantic/\nEastern Pacific (AP)", transform=ax.transAxes, fontsize=12, multialignment="center")
ax.text(0.6, 0.6, "North Indian\nOcean (IO)", transform=ax.transAxes, fontsize=12, multialignment="center")
ax.text(0.35, 0.3, "Southern Hemisphere (SH)", transform=ax.transAxes, fontsize=12)
ax.text(0.78, 0.65, "North Western\nPacific (WP)", transform=ax.transAxes, fontsize=12, multialignment="center")

patches = []
#colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, basin in enumerate(BASIN_BOUNDS.keys()):
    lonmin, lonmax, latmin, latmax = BASIN_BOUNDS[basin]
    height = latmax - latmin
    if lonmax >= lonmin:
        width = lonmax - lonmin
    else:
        # basin crossing the -180/180 longitude threshold
        width = (180 - lonmin) + (lonmax - (-180))

    ax.set_global()
    
    p = mpatches.Rectangle(xy=[lonmin, latmin], width=width, height=height,
                           facecolor=(0,1,0, 0.0),
                           edgecolor=(0,0,0, 1.0),
                           linewidth=1.5,
                           linestyle="dashed",
                           transform=ccrs.PlateCarree(),
                           label=basin,)
    ax.add_patch(p)
    patches.append(p)
#ax.legend(ncol=4, loc='lower left')
#ax.gridlines()
ax.coastlines(color='grey')

save_fig_str = "delta_TC_risk.png"
plt.savefig(res_dir.joinpath(save_fig_str), dpi=300, facecolor='w', 
            edgecolor='w', orientation='portrait', papertype=None, 
            format='png', bbox_inches='tight', pad_inches=0.1) 

#%% create csv files for boxplots values - Supplementary Tables S1, S2
for m, met in enumerate(metric):
    for r, reg in enumerate(['AP', 'IO', 'SH', 'WP']):
        save_csv_str = f"{reg}_{met}_describe_sum.csv"
        delta_df_dict[reg].groupby(["delta","year"])[met].describe().to_csv(res_dir.joinpath(save_csv_str))