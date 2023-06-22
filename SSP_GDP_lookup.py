"""
Adapted for code repository on 2023-06-22

description: Figure 2 - plotting of sensitivity indices

@author: evelynm; adapted @simonameiler
"""

import pandas as pd
import logging

import climada.util.coordinates as u_coord
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

def get_pop_scen(country, year, ssp):
    """
    Lookup function for a country's growth factor in year X, relative to the 
    base year 2020, according to an SSP scenario and a modelling source.
    
    Annual growth factors were calculated from the SSP public database (v2.0)
    Samir KC, Wolfgang Lutz,
    The human core of the shared socioeconomic pathways: Population scenarios 
    by age, sex and level of education for all countries to 2100,
    Global Environmental Change,
    Volume 42,
    2017,
    Pages 181-192,
    ISSN 0959-3780,
    https://doi.org/10.1016/j.gloenvcha.2014.06.004.
    Selection: 1. Region - all countries, 2. Scenarios - POP, 
    3. Variable - Population (growth Total)
    
    Parameters
    ----------
    country : str
        iso3alpha (e.g. 'JPN'), or English name (e.g. 'Switzerland')
    year : int
        The yer for which to get a GDP projection for. Any among [2020, 2099].
    ssp : int
        The SSP scenario for which to get a GDP projecion for. Any amon [1, 5].
        
    Returns
    -------
    float
        The country's GDP growth relative to the year 2020, according to chosen
        SSP scenario and source.
    
    Example
    -------
    get_pop_scen('Switzerland', 2067, 2)
    """
    
    ssp_long = f'SSP{str(ssp)}'
    
    try:
        iso3a = u_coord.country_to_iso(country, representation="alpha3")
    except LookupError:
        LOGGER.error('Country not identified: %s.', country)
        return None

    df_csv = pd.read_csv(SYSTEM_DIR.joinpath('ssps_pop_annual.csv'),
                         usecols = ['Scenario', 'Region', str(year)])

    sel_bool = ((df_csv.Scenario == ssp_long) &
                (df_csv.Region == iso3a))
    
    return df_csv.loc[sel_bool][str(year)].values[0]

