"""
Adapted for code repository on 2023-06-22

description: Figure 2 - plotting of sensitivity indices

@author: simonameiler
"""

import sys
import scipy as sp
import numpy as np
import pandas as pd
import copy as cp
import copy
import logging

#Load Climada modules
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.hazard import TropCyclone

import climada.util.coordinates as u_coord
from climada.entity.exposures.litpop import LitPop
from climada.engine.unsequa import CalcDeltaImpact, InputVar
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone

def main(region, period):
    
    LOGGER = logging.getLogger(__name__)
    
    ###########################################################################
    ############### A: define constants, functions, paths #####################
    ###########################################################################
    
    # define paths
    haz_dir = SYSTEM_DIR/"hazard/future"
    unsequa_dir = SYSTEM_DIR/"unsequa"
    
    res = 300
    ref_year = 2005
    N_samples = 2**10
    region = str(region) # AP, IO, SH, WP
    period = str(period) # 20thcal (1995-2014), cal (2041-2060), _2cal (2081-2100)
    
    # translate period in the hazard object naming (K. Emanuel) to ref_year for
    # exposure scaling
    year_from_per = {'20thcal': 2005,
                     'cal': 2050,
                     '_2cal': 2090}
    
    
    ###########################################################################
    ########## B: load and define hazard, exposure, impf_sets #################
    ###########################################################################
    
    # load hazard
    # make list
    h1_min, h1_max = (1, 9)
    h2_min, h2_max = (1, 3)
    h3_min, h3_max = (1, 1)
    
    model_key = {1: 'cesm2', 
                 2: 'cnrm6', 
                 3: 'ecearth6', 
                 4: 'fgoals', 
                 5: 'ipsl6', 
                 6: 'miroc6', 
                 7: 'mpi6', 
                 8: 'mri6', 
                 9: 'ukmo6'}
    
    ssp_haz_key = {1: 'ssp245', 
                   2: 'ssp370',
                   3: 'ssp585'}
    
    wind_model_key = {1: '',
                      2: 'ER11_'}
    
    # present climate
    tc_haz_base_dict = {}
    for h1 in range(h1_min, h1_max+1):
        for h3 in range(h3_min, h3_max+1):
            haz_base_str = f"TC_{region}_0{res}as_{wind_model_key[h3]}MIT_{model_key[h1]}_20thcal.hdf5"
            tc_haz_base = TropCyclone.from_hdf5(haz_dir.joinpath(haz_base_str))
            tc_haz_base.check()
            tc_haz_base_dict[str(wind_model_key[h3])+str(model_key[h1])] = tc_haz_base
    
    # future climate
    tc_haz_fut_dict = {}
    for h1 in range(h1_min, h1_max+1):
        for h2 in range(h2_min, h2_max+1):
            for h3 in range(h3_min, h3_max+1):
                haz_fut_str = f"TC_{region}_0{res}as_{wind_model_key[h3]}MIT_{model_key[h1]}_{ssp_haz_key[h2]}{period}.hdf5"
                tc_haz_fut = TropCyclone.from_hdf5(haz_dir.joinpath(haz_fut_str))
                tc_haz_fut.check()
                tc_haz_fut_dict[str(wind_model_key[h3])+str(model_key[h1])+'_'+str(ssp_haz_key[h2])] = tc_haz_fut

        
    # store frequency information
    for h, haz in tc_haz_fut_dict.items():
        print(h, haz.frequency.sum())
    
if __name__ == "__main__":
    main(*sys.argv[1:])
