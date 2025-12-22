# 🧬 Under Construction  Czech FOI Mortality Data Analysis – Overview

This repository contains scripts, datasets, and plots for analyzing Czech-FOI mortality data, with a focus on understanding vaccine effectiveness (VE) while accounting for biases in the data.  

**Main Goal:**  
To obtain **unbiased and fair comparisons** between vaccinated (VX) and unvaccinated (UVX) individuals using RMST (Restricted Mean Survival Time).  
The scripts explore how vaccine effectiveness can be estimated, check for biases in the Methodes analysing the data, and simulate different scenarios.

---

## Key Concept: RMST (Restricted Mean Survival Time)

RMST is a way to measure **how long people survive during a fixed follow-up period**.  

- It calculates the **average survival time** up to a cutoff (e.g., 2 years).  
- It allows fair comparison between groups even if follow-up times differ.  
- The difference in RMST shows how much longer, on average, one group survives compared to another.  

Simple analogy: “RMST tells us the average number of days people survived during the study period. Comparing RMST between groups shows how much longer, on average, one group lived compared to another

---

## Repository Structure

### Scripts  
All scripts are in the [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts):

| Script | Purpose |
|--------|---------|
| `AA) Export AG ALL from Czech FOI.py` | Exports raw age-group-specific mortality data from the Czech FOI dataset. |
| `AA) real_data_sim_dose_reclassified_uvx_as_vx.py` | Performs **Mortality-Conditioned Stochastic Imputation (MCSI)** to simulate reclassification of unvaccinated deaths as vaccinated and generates diagnostic plots. |
| `AA) simulate deaths doseschedule and bias all AG.py` | Simulates death and vaccination schedules across age groups to explore potential biases. |
| `AC) hernan_style_poold_logistics_RMST.py` | Implements RMST analysis using pooled logistic regression (Hernán-style approach). |
| `AE) Empirical_dynamic_CC_RMST.py` | Computes RMST in an **empirical case-control framework**, dynamically adjusting for covariates. |
| `AF) Empirical vs causal comparison RMST.py` | Compares RMST estimates using empirical versus causal methods to assess bias. |

### Data  
The input datasets are in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra) and contain age-specific CSV mortality files used by the scripts.  

### Plots & Logs  
All interactive plots and detailed epidemiological audit logs are in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results). These visualize vaccination timing, synthetic assignments, exposure durations, and cumulative person-time curves.

**Author:** AI / Drifitng  
**Date:** 2025-11-02  
**Environment:** Python ≥ 3.10  
**Raw Dataset used:** Vesely_106_202403141131_AG10.csv (~136k individuals Czech-FOI 1.9 GB). 
<br>True science must be reproducible, which requires the code and the data. Otherwise it is dogmatic. 
