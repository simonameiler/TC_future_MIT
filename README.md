# Uncertainties and sensitivities in the quantification of future tropical cyclone risk
These scripts reproduce the main results of the paper:

*Simona Meiler(1,2), Alessio Ciullo(1,2), Chahan M. Kropf(1,2),
Kerry Emanuel(3), and David N. Bresch(1,2):
**Uncertainties and sensitivities in the quantification of
future tropical cyclone risk***

Publication status: [under revision](https://doi.org/10.21203/rs.3.rs-2703613/v1).

(1) Institute for Environmental Decisions, ETH Zurich, Switzerland

(2) Federal Office of Meteorology and Climatology MeteoSwiss, Switzerland

(3) Lorenz Center, Massachusetts Institute of Technology, Cambridge, Massachusetts, USA

Contact: [Simona Meiler](simona.meiler@usys.ethz.ch)


## Content:

#### Centroids.py
Python script to generate the centroids files.

#### Exposure_vary_baseline.py
Python script to generate the baseline Exposure files for different exponent combinations of the LitPop method.

#### Hazard_windfield_calc.py
Python scripts to load TC track sets (for present and the two future periods, various GCMs and emission scenarios) 
and calculate the 2D windfields using two wind models. The output hdf5 files are the hazard sets, which are further 
used for the uncertainty and sensitivity analysis `UA_SA*`.
Note that this step requires a computer cluster and that the output files are large (multiple GB per file).

#### SSP_GDP_scenarios_preprocessing.py
Python script which converts GDP growth factors downloaded from the SSP public database and stored in `iamc_db.csv`
into annual growth factors for each SSP scenario and country (`ssps_gdp_annual.csv`). Needed for the `UA_SA*` Python scripts.

#### UA_SA*.py
Python scripts to run the publication's central uncertainty and sensitivity analyses. Files are named after their primary
analysis focus: `abs` refers to UA/SA for absolute TC risk estimate in the future, `CC` and `SOC` are to assess climate
change and socio-economic development independently, `main` yields UA/SA results for the total TC risk increase.
Note that this step requires a computer cluster.

#### data
CSV and Excel files with GDP growth factors from the SSP public database. `iamc_db.csv` is needed for `SSP_GDP_scenarios_preprocessing.py`
and yields `ssps_gdp_annual.csv`, which is used in the `UA_SA*` Python scripts.
HDF5 files containing the output of the uncertainty and sensitivity analyses described above. All files required
to reproduce the figures of the publication are provided.

#### Fig*.py
Python scripts named according to their Figure number in the publication; can be used to reproduce the figures.
Figure numbers starting with S produce outputs and results for the Supplementary Material and contain
code analogous to their main text counterparts.

#### SI_TabS*.py
Python script named according to their Table number in the publication's Supplementary Information to produce the
respective values.

## Requirements
Requires:
* Python 3.8+ environment (best to use conda for CLIMADA repository)
* _CLIMADA_ repository version 3.3.1+:
        https://wcr.ethz.ch/research/climada.html
        https://github.com/CLIMADA-project/climada_python
* branch: feature/unsequa_delta_climate

## ETH cluster
Computationally demanding calculations were run on the [Euler cluster of ETH Zurich](https://scicomp.ethz.ch/wiki/Euler).

## Documentation:
Publication: submitted to **Nature Communications**

Documentation for CLIMADA is available on Read the Docs:
* [online (recommended)](https://climada-python.readthedocs.io/en/stable/)
* [PDF file](https://buildmedia.readthedocs.org/media/pdf/climada-python/stable/climada-python.pdf)

If script fails, revert CLIMADA version to release v3.3.1:
* from [GitHub](https://github.com/CLIMADA-project/climada_python/releases/tag/v3.1.2)

## History

Created on 22 June 2023

-----

www.wcr.ethz.ch
