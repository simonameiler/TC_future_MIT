"""
Adapted for code repository on 2023-06-22

description: Uncertainty and sensitivity analysis - main analysis

@author: simonameiler
"""

import sys
import scipy as sp
import numpy as np
import pandas as pd
import copy as cp
import logging

#Load Climada modules
from climada.util.constants import SYSTEM_DIR # loads default directory paths for data
from climada.hazard import Hazard

import climada.util.coordinates as u_coord
from climada.entity import Exposures
from climada.engine.unsequa import InputVar, CalcDeltaImpact
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone

def main(region, period):
    
    LOGGER = logging.getLogger(__name__)
    
    ###########################################################################
    ############### A: define constants, functions, paths #####################
    ###########################################################################
    
    # define paths
    haz_dir = SYSTEM_DIR/"hazard/future"
    unsequa_dir = SYSTEM_DIR/"unsequa"
    #SYSTEM_DIR = Path('./data')
    
    res = 300
    ref_year = 2005
    N_samples = 2**11
    region = str(region) # AP, IO, SH, WP
    period = str(period) # 20thcal (1995-2014), cal (2041-2060), _2cal (2081-2100)
    
    # translate period in the hazard object naming (K. Emanuel) to ref_year for
    # exposure scaling
    year_from_per = {'20thcal': 2005,
                     'cal': 2050,
                     '_2cal': 2090}
    
    # 32 world regions used by the PIK model to scale GDP factors according to SSPs
    # cf: https://tntcat.iiasa.ac.at/SspDb/dsd?Action=htmlpage&page=about#regiondefs
    PIK_32regions = {'R32AUNZ': ['Australia', 'New Zealand'],
                     'R32BRA': ['Brazil'],
                     'R32CAN': ['Canada'],
                     'R32CAS': ['Armenia', 'Azerbaijan', 'Georgia', 'Kazakhstan', 
                                'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan'],
                     'R32CHN': ['China', 'Hong Kong', 'Macao'],
                     'R32EEU': ['Albania', 'Bosnia and Herzegovina', 'Croatia', 
                                'Montenegro', 'Serbia', 'MKD'],
                     'R32EEU-FSU': ['Belarus', 'Moldova', 'Ukraine'],
                     'R32EFTA': ['Iceland', 'Norway', 'Switzerland'],
                     'R32EU12-H': ['Cyprus', 'Czech Republic', 'Estonia', 'Hungary', 
                                   'Malta', 'Poland', 'Slovakia', 'Slovenia'],
                     'R32EU12-M': ['Bulgaria', 'Latvia', 'Lithuania', 'Romania'],
                     'R32EU15': ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 
                                 'Germany', 'Greece', 'Ireland', 'Italy', 'Luxembourg', 
                                 'Netherlands', 'Portugal', 'Spain', 'Sweden', 'United Kingdom'],
                     'R32IDN': ['Indonesia'],
                     'R32IND': ['India'],
                     'R32JPN': ['Japan'],
                     'R32KOR': ['KOR'],
                     'R32LAM-L': ['Belize', 'Guatemala', 'Haiti', 'Honduras', 'Nicaragua'],
                     'R32LAM-M': ['Antigua and Barbuda', 'Argentina', 'Bahamas', 
                                  'Barbados', 'Bermuda', 'Bolivia', 'Chile', 'Colombia', 
                                  'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 
                                  'Ecuador', 'El Salvador', 'French Guiana', 'Grenada', 
                                  'Guadeloupe', 'Guyana', 'Jamaica', 'Martinique', 
                                  'Netherlands Antilles', 'Panama', 'Paraguay', 'Peru', 
                                  'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 
                                  'Suriname', 'Trinidad and Tobago', 'Uruguay', 'Venezuela'], 
                     'R32MEA-H': ['Bahrain', 'Israel', 'Kuwait', 'Oman', 'Qatar', 
                                  'Saudi Arabia', 'United Arab Emirates'],
                     'R32MEA-M': ['IRN', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 
                                  'Syrian Arab Republic', 'Yemen'],
                     'R32MEX': ['Mexico'],
                     'R32NAF': ['Algeria', 'Egypt', 'Libya', 'Morocco', 'Tunisia', 
                                'Western Sahara'],
                     'R32OAS-CPA': ['Cambodia', 'LAO', 'Mongolia', 'Vietnam'],
                     'R32OAS-L': ['Bangladesh', 'PRK', 'Fiji', 'FSM', 'Myanmar', 
                                  'Nepal', 'Papua New Guinea', 'Philippines', 'Samoa', 
                                  'Solomon Islands', 'Timor-Leste', 'Tonga', 'Vanuatu'],
                     'R32OAS-M': ['Bhutan', 'Brunei Darussalam', 'French Polynesia', 
                                  'Guam', 'Malaysia', 'Maldives', 'New Caledonia', 
                                  'Singapore', 'Sri Lanka', 'Thailand'],
                     'R32PAK': ['Pakistan', 'Afghanistan'],
                     'R32RUS': ['Russian Federation'],
                     'R32SAF': ['South Africa'],
                     'R32SSA-L': ['Benin', 'Burkina Faso', 'Burundi', 'Cameroon', 
                                  'Cabo Verde', 'Central African Republic', 'Chad', 
                                  'Comoros', 'Congo', 'CIV', 'COD', 
                                  'Djibouti', 'Eritrea', 'Ethiopia', 'Gambia', 'Ghana', 
                                  'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 
                                  'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mozambique', 
                                  'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 
                                  'Senegal', 'Sierra Leone', 'Somalia', 'South Sudan', 
                                  'Sudan', 'SWZ', 'Togo', 'Uganda', 'United Republic of Tanzania', 
                                  'Zambia', 'Zimbabwe'],
                     'R32SSA-M': ['Angola', 'Botswana', 'Equatorial Guinea', 'Gabon', 
                                  'Mauritius', 'Mayotte', 'Namibia', 'Réunion', 'Seychelles'],
                     'R32TUR': ['Turkey'],
                     'R32TWN': ['Taiwan'],
                     'R32USA': ['Puerto Rico', 'VIR', 'USA']
                         }
    # function to return iso3 code for countries for the PIK_32regions dictionary
    def key_return(X):
        for key, value in PIK_32regions.items():
            if X == u_coord.country_to_iso(value, representation="alpha3"):
                return key
            if isinstance(u_coord.country_to_iso(value, representation="alpha3"), 
                          list) and X in u_coord.country_to_iso(value, representation="alpha3"):
                return key
        return "Key doesnt exist"
    
    # function to retrieve GDP growth factors per country, year, SSP, model
    def get_gdp_scen(country, year, ssp, model):
        """
        Lookup function for a country's growth factor in year X, relative to the 
        base year 2020, according to an SSP scenario and a modelling source.
        
        Annual growth factors were calculated from the SSP public database (v2.0)
        Keywan Riahi, Detlef P. van Vuuren, Elmar Kriegler, Jae Edmonds, 
        Brian C. O’Neill, Shinichiro Fujimori, Nico Bauer, Katherine Calvin, 
        Rob Dellink, Oliver Fricko, Wolfgang Lutz, Alexander Popp, 
        Jesus Crespo Cuaresma, Samir KC, Marian Leimbach, Leiwen Jiang, Tom Kram, 
        Shilpa Rao, Johannes Emmerling, Kristie Ebi, Tomoko Hasegawa, Petr Havlík, 
        Florian Humpenöder, Lara Aleluia Da Silva, Steve Smith, Elke Stehfest, 
        Valentina Bosetti, Jiyong Eom, David Gernaat, Toshihiko Masui, Joeri Rogelj,
        Jessica Strefler, Laurent Drouet, Volker Krey, Gunnar Luderer, Mathijs Harmsen,
        Kiyoshi Takahashi, Lavinia Baumstark, Jonathan C. Doelman, Mikiko Kainuma, 
        Zbigniew Klimont, Giacomo Marangoni, Hermann Lotze-Campen, Michael Obersteiner,
        Andrzej Tabeau, Massimo Tavoni.
        The Shared Socioeconomic Pathways and their energy, land use, and
        greenhouse gas emissions implications: An overview, Global Environmental
        Change, Volume 42, Pages 153-168, 2017,
        ISSN 0959-3780, DOI:110.1016/j.gloenvcha.2016.05.009
        Selection: 1. Region - all countries, 2. Scenarios - GDP (PIK, IIASA, OECD),
        3. Variable - GDP (growth PPP)
        
        Parameters
        ----------
        country : str
            iso3alpha (e.g. 'JPN'), or English name (e.g. 'Switzerland')
        year : int
            The yer for which to get a GDP projection for. Any among [2020, 2099].
        ssp : int
            The SSP scenario for which to get a GDP projecion for. Any amon [1, 5].
        model : str
            The model source from which the GDP projections have been calculated. 
            Either IIASA, PIK or OECD.
            
        Returns
        -------
        float
            The country's GDP growth relative to the year 2020, according to chosen
            SSP scenario and source.
        
        Example
        -------
        get_gdp_scen('Switzerland', 2067, 2)
        """
        ssp_long = f'SSP{str(ssp)}'
        
        model_long = {'IIASA': 'IIASA GDP',
                      'OECD': 'OECD Env-Growth',
                      'PIK': 'PIK GDP-32'}   
        
        if (model == 'IIASA' or model == 'OECD'):
            try:
                iso3a = u_coord.country_to_iso(country, representation="alpha3")
            except LookupError:
                LOGGER.error('Country not identified: %s.', country)
                return None
        
            df_csv = pd.read_csv(SYSTEM_DIR.joinpath('ssps_gdp_annual.csv'),
                                 usecols = ['Model', 'Scenario', 'Region', str(year)])
        
            sel_bool = ((df_csv.Model == model_long[model]) & 
                        (df_csv.Scenario == ssp_long) &
                        (df_csv.Region == iso3a))
        
        elif model == 'PIK':

            df_csv = pd.read_csv(SYSTEM_DIR.joinpath('ssps_gdp_annual.csv'),
                                 usecols = ['Model', 'Scenario', 'Region', str(year)])
        
            sel_bool = ((df_csv.Model == model_long[model]) & 
                        (df_csv.Scenario == ssp_long) &
                        (df_csv.Region == key_return(country)))
        
        return df_csv.loc[sel_bool][str(year)].values[0]


    # function to update exposure total value with GDP growth factor
    def exp_gdp_scen(exposure, year, ssp, model):
        
        exp_future = cp.deepcopy(exposure)
        cntry_list_iso = np.unique(exposure.gdf.iso_code).tolist()
        for country in cntry_list_iso:
            try:
                exp_future.gdf.value[exp_future.gdf.iso_code==country] = \
                exp_future.gdf.value[exp_future.gdf.iso_code==country] * \
                get_gdp_scen(country, year, ssp, model)
            except IndexError:
                LOGGER.error('Country not identified: %s.', country)            
        
        return exp_future
    
    ###########################################################################
    ########## B: load and define hazard, exposure, impf_sets #################
    ###########################################################################
    
    # load hazard
    # make list
    h1_min, h1_max = (1, 9)
    h2_min, h2_max = (1, 3)
    h3_min, h3_max = (1, 2)
    
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
            tc_haz_base = Hazard.from_hdf5(haz_dir.joinpath(haz_base_str))
            tc_haz_base.check()
            tc_haz_base_dict[str(wind_model_key[h3])+str(model_key[h1])] = tc_haz_base
    
    # future climate
    tc_haz_fut_dict = {}
    for h1 in range(h1_min, h1_max+1):
        for h2 in range(h2_min, h2_max+1):
            for h3 in range(h3_min, h3_max+1):
                haz_fut_str = f"TC_{region}_0{res}as_{wind_model_key[h3]}MIT_{model_key[h1]}_{ssp_haz_key[h2]}{period}.hdf5"
                tc_haz_fut = Hazard.from_hdf5(haz_dir.joinpath(haz_fut_str))
                tc_haz_fut.check()
                tc_haz_fut_dict[str(wind_model_key[h3])+str(model_key[h1])+'_'+str(ssp_haz_key[h2])] = tc_haz_fut

        
    # load exposure
    # present
    e1_min, e1_max = (1, 5)
    e2_min, e2_max = (1, 3)
    e3_min, e3_max = (1, 9)
    
    # make dictionary of exposures for different SSPs and models
    gdp_key = {1: 'IIASA',
               2: 'OECD',
               3: 'PIK'}

    mn_key = {1: [0.5, 0.5],
              2: [0.5, 1.0],
              3: [0.5, 1.5], 
              4: [1.0, 0.5],
              5: [1.0, 1.0],
              6: [1.0, 1.5],
              7: [1.5, 0.5],
              8: [1.5, 1.0],
              9: [1.5, 1.5]}
    
    # set exposure region_id to code regions for impact function mapping
    iso3n_per_region = impf_id_per_region = ImpfSetTropCyclone.get_countries_per_region()[2]
    code_regions = {'NA1': 1, 'NA2': 2, 'NI': 3, 'OC': 4, 'SI': 5, 'WP1': 6, \
                    'WP2': 7, 'WP3': 8, 'WP4': 9, 'ROW': 10}
    
    exp_base_dict = {}
    for e3 in range(e3_min, e3_max+1):
        [m, n] = mn_key[e3]
        ent_str = f"litpop_0300as_{ref_year}_{region}_{m}-{n}.hdf5"
        exp_base = Exposures.from_hdf5(SYSTEM_DIR.joinpath(ent_str))
        exp_base.assign_centroids(tc_haz_base)
        exp_base.value_unit = 'USD'
        
        # match exposure with correspoding impact function
        for calibration_region in impf_id_per_region:
            for country_iso3n in iso3n_per_region[calibration_region]:
                exp_base.gdf.loc[exp_base.gdf.region_id==country_iso3n, 'impf_TC'] = code_regions[calibration_region]
   
        # get iso3alpha codes from exposure region_ids
        exp_natids = np.unique(exp_base.gdf.region_id).tolist()
        #exp_iso = u_coord.country_to_iso(exp_natids, representation="alpha3")

        # add iso_code to exposure gdf
        for natid in exp_natids:
            exp_base.gdf.loc[exp_base.gdf.region_id==natid, 'iso_code'] = \
            u_coord.country_to_iso(natid, representation="alpha3")
    
        exp_base_dict[str(mn_key[e3])] = exp_base
    
    exp_fut_dict = {}
    for e3 in range(e3_min, e3_max+1):
        for e1 in range(e1_min, e1_max+1):
            for e2 in range(e2_min, e2_max+1):
                exp_fut = exp_gdp_scen(exp_base_dict[str(mn_key[e3])], year_from_per[period], e1, gdp_key[e2])
                exp_fut_dict['SSP'+str(e1)+'_'+gdp_key[e2]+'_'+str(mn_key[e3])] = exp_fut
    
    ###########################################################################
    ############## C: define input variables and parameters ###################
    ###########################################################################
    
    # hazard
    # present    
    def haz_base_func(gc_model, wind_model, HE_base, tc_haz_base_dict=tc_haz_base_dict):
        haz_base = tc_haz_base_dict[
            str(wind_model_key[wind_model])+str(model_key[gc_model])]

        rng = np.random.RandomState(int(HE_base))
        rnd_ids = [rng.choice(np.arange(500*n+1, 500*(n+1)+1), int(n_ev), replace=False) for n in range(20)]
        event_ids = np.concatenate(rnd_ids).tolist()
        haz_base = haz_base.select(event_id=event_ids)
        
        return haz_base
        
    
    haz_base_distr = {"gc_model": sp.stats.randint(low=h1_min, high=h1_max+1),
                      "wind_model": sp.stats.randint(low=h3_min, high=h3_max+1),
                      "HE_base": sp.stats.randint(0, 2**32 - 1)}
    
    n_ev = 400 #80% of all events are selected for samples of every year
    haz_base_iv = InputVar(haz_base_func, haz_base_distr)
    
    # future
    def haz_fut_func(gc_model, ssp_haz, wind_model, HE_fut, tc_haz_fut_dict=tc_haz_fut_dict):
        haz_fut = tc_haz_fut_dict[
            str(wind_model_key[wind_model])+str(model_key[gc_model])+'_'+str(ssp_haz_key[ssp_haz])]
        
        rng = np.random.RandomState(int(HE_fut))
        rnd_ids = [rng.choice(np.arange(500*n+1, 500*(n+1)+1), int(n_ev), replace=False) for n in range(20)]
        event_ids = np.concatenate(rnd_ids).tolist()
        haz_fut = haz_fut.select(event_id=event_ids)
        
        return haz_fut
    
    haz_fut_distr = {"gc_model": sp.stats.randint(low=h1_min, high=h1_max+1),
                     "ssp_haz": sp.stats.randint(low=h2_min, high=h2_max+1),
                     "wind_model": sp.stats.randint(low=h3_min, high=h3_max+1),
                     "HE_fut": sp.stats.randint(0, 2**32 - 1)}
    
    haz_fut_iv = InputVar(haz_fut_func, haz_fut_distr)
    
    # exposure
    # present   
    def exp_base_func(mn_exp, exp_base_dict=exp_base_dict):
        return exp_base_dict[str(mn_key[mn_exp])]
    
    # derive distribution from GDP growth scaling factors; see exp_scale_future.py
    exp_base_distr = {"mn_exp": sp.stats.randint(low=e3_min, high=e3_max+1)}
    
    exp_base_iv = InputVar(exp_base_func, exp_base_distr)
    
    # future    
    def exp_fut_func(ssp_exp, gdp_model, mn_exp, exp_fut_dict=exp_fut_dict):
        return exp_fut_dict['SSP'+str(int(ssp_exp))+'_'+gdp_key[gdp_model]+'_'+str(mn_key[mn_exp])]
    
    # derive distribution from GDP growth scaling factors; see exp_scale_future.py
    exp_fut_distr = {"ssp_exp": sp.stats.randint(low=e1_min, high=e1_max+1),
                     "gdp_model": sp.stats.randint(low=e2_min, high=e2_max+1),
                     "mn_exp": sp.stats.randint(low=e3_min, high=e3_max+1)}
    
    exp_fut_iv = InputVar(exp_fut_func, exp_fut_distr)

    def impf_func(v_half):
        impf_set_MED = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(
            calibration_approach='EDR', q=v_half)
        return impf_set_MED

    impf_distr = {"v_half": sp.stats.uniform(.25, .75)}
    impf_iv = InputVar(impf_func, impf_distr)
    
    ###########################################################################
    ####################### D: calculate uncertainty ##########################
    ###########################################################################

    calc_imp = CalcDeltaImpact(exp_base_iv, impf_iv, haz_base_iv, 
                               exp_fut_iv, impf_iv, haz_fut_iv)
    
 
    output_imp = calc_imp.make_sample(N=N_samples)
    output_imp.get_samples_df()
    output_imp = calc_imp.uncertainty(output_imp, calc_at_event=True)
    
    ###########################################################################
    ####################### D: calculate sensitivity ##########################
    ###########################################################################
    
    output_imp = calc_imp.sensitivity(output_imp)

    output_imp.to_hdf5(unsequa_dir.joinpath(
        f"unsequa_TC_{year_from_per[period]}_{region}_0{res}as_MIT_{N_samples}_v3.hdf5"))
    
if __name__ == "__main__":
    main(*sys.argv[1:])
