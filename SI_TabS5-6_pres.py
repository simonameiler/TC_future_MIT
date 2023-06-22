"""
Adapted for code repository on 2023-06-22

description: Figure 2 - plotting of sensitivity indices

@author: simonameiler
"""

import sys
import numpy as np

#Load Climada modules
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.hazard import TropCyclone


def main(region):

    
    ###########################################################################
    ############### A: define constants, functions, paths #####################
    ###########################################################################
    
    # define paths
    haz_dir = SYSTEM_DIR/"hazard/future"
    
    res = 300
    region = str(region) # AP, IO, SH, WP
    
    
    ###########################################################################
    ########## B: load and define hazard, exposure, impf_sets #################
    ###########################################################################
    
    # load hazard
    # make list
    h1_min, h1_max = (1, 9)
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

        
    # store frequency information
    for h, haz in tc_haz_base_dict.items():
        print(h, haz.frequency.sum())

    # store intensity information
    for h, haz in tc_haz_base_dict.items():  
        print(h, np.max(haz.intensity, axis=1).mean())
    
if __name__ == "__main__":
    main(*sys.argv[1:])
