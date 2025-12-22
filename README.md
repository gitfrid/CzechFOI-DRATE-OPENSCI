# 🧬 Under Construction  Czech FOI Mortality Data Analysis – Overview

This repository contains scripts, datasets, Result Logs and plots for analyzing Czech-FOI mortality data, with a focus on understanding vaccine effectiveness (VE) while accounting for biases in the data.  

**Main Goal:**  
Explore differnt Methodes to fairly compare how long vaccinated (VX) and unvaccinated (UVX) people lived during the study period using RMST (Restricted Mean Survival Time). The scripts examine how vaccine effectiveness is calculated, identify possible biases in the analysis methods, and test how different assumptions or scenarios can change the results.

---

## Key Concept: RMST (Restricted Mean Survival Time)

RMST is a way to measure **how long people survive during a fixed follow-up period**.  

- It calculates the **average survival time** up to a cutoff (e.g., 2 years).  
- It allows comparison between groups even if follow-up times differ.  
- The difference in RMST shows how much longer, on average, one group survives compared to another.  

Simple analogy: RMST tells us the average number of days people survived during the study period. Comparing between groups RMST shows how much longer, on average, one group lived compared to another

---

## Repository Structure

### Scripts  
All scripts are in the [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts):

| Script | Purpose |
|--------|---------|
| `AA) Export AG ALL from Czech FOI.py` | Exports raw age-group-specific mortality data from the Czech FOI dataset. For differnt age-groups|
| `AA) real_data_sim_dose_reclassified_uvx_as_vx.py` | Performs **Mortality-Conditioned Stochastic Imputation (MCSI)** creates a Dataset per AG containg reclassification of unvaccinated deaths as vaccinated used for senitivity analyses of differnt Methodes|
| `AA) simulate deaths doseschedule and bias all AG.py` | creates a Dataset per AG containg with simulated constant desth rate death and simulated real vaccination schedule across age groups to explore potential biases of differnt Methodes |
| `AC) hernan_style_poold_logistics_RMST.py` | Implements RMST analysis using pooled logistic regression (Hernán-style approach). |
| `AE) Empirical_dynamic_CC_RMST.py` | Computes RMST in as descriptive **empirical case-control framework**, dynamically adjusting for covariates. |
| `AF) Empirical vs causal comparison RMST.py` | Compares RMST estimates using empirical versus causal methods to assess bias. |

### Data  
The input datasets are in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra) and contain age-specific CSV mortality files used by the scripts.  

### Plots & Epidomilogical Result Logs  
All interactive plots and detailed epidemiological audit logs are in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results). These visualize vaccination timing, synthetic assignments, exposure durations, and cumulative person-time curves.

**Author:** AI / Drifitng
**Environment:** Python ≥ 3.10  
**Raw Dataset used:** Vesely_106_202403141131_AG10.csv (~136k individuals Czech-FOI 1.9 GB). 
<br>True science must be reproducible, which requires the code and the data. Otherwise it is dogmatic. 
