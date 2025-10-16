# Strategy for identifying Vera C. Rubin Observatory kilonova candidates with Fink broker

Code to reproduce results of "Strategy for identifying Vera C. Rubin Observatory kilonova candidates for targeted gravitational-wave searches" (Stevenson, Möller, Powell 2025) [![arXiv](https://img.shields.io/badge/arxiv-astro--ph%2F1901.06384-red)](https://arxiv.org/abs/2510.12932).



- `0.analysis_estimates_ocean_oneyear.ipynb`: KNe estimates for one year of Rubin LSST
- `1a.analysis_broker_ocean.ipynb`: results from broker alerts for ocean observing strategy (v.5)
- `1b.analysis_broker_obsv43.ipynb`: results from broker alerts for v4.3 observing strategy (not used in the paper)
- `2.analysis_fink_process_alerts.ipynb`: results from FInk ZTF filter strategies
-` 3.analysis_proprietary_ocean.ipynb`: results for KNe searches using both public alerts and proprietary data from Rubin.

Environment configured with uv and mise using the `pyproject.toml` adn `mise.toml` files.


## Simulations: 
Simulations were generated using SNANA within the PIPPIN framework. We use the observing strategy v5.0 (Ocean) for Rubin LSST and Kasen and Bulla KN models.
- SIMS_KN_OCEAN_250_NDET: Ocean, rate 250 /yr/GPc³  power law, yes detection efficiency, NGEN_UNIT: 5 (5 year), 20 ranseeds (by 20 times)
- SIMS_KN_OCEAN_ONEYEAR_250_NDET: Ocean, rate 250 /yr/GPc³  power law, yes detection efficiency, NGEN_UNIT: 1 (1 year), 20 ranseeds (by 20 times)