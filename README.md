# <img src="https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Plot%20Results/under%20construction.png" width="40"> Czech FOI Mortality Data Analysis  

This repository analyzes Czech FOI mortality data to evaluate **vaccine effectiveness (VE)** and potential biases. The project compares survival times of vaccinated (VX) and unvaccinated (UVX) individuals using **Restricted Mean Survival Time (RMST)**. Simulations are used to detect biases and test the sensitivity of statistical methods.  

Empirical approaches estimate VE directly from observed data but may be biased, while causal models use hypothetical scenarios to adjust for confounders, relying on model assumptions for validity. RMST measures the **average survival time** within a fixed follow-up period and shows differences in days lived between groups.  

The repository includes Python scripts for data processing, RMST analysis, and simulations, as well as input datasets, plots, and logs—all organized for reproducibility.

---

## Concept: RMST (Restricted Mean Survival Time)

**RMST** measures the **average survival time** during a fixed study period (e.g., 2 years).  

- Shows the **average number of days a person lived** during the study.  
- Allows fair comparison even if follow-up times differ between groups.  
- The difference in RMST between groups tells you **how many more (or fewer) days, on average, one group survived**.  

**Analogy:**  
Imagine asking: *“On average, how many days did each person live during the study?”* RMST gives a simple, intuitive answer to compare groups.

---


### DataScripts & Data Overview

All scripts are located in the [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts):

- [AA) Export AG ALL from Czech FOI.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20Export%20AG%20ALL%20from%20Czech%20FOI.py)  
  Exports raw age-group-specific mortality data from the Czech FOI dataset into individual CSV files.

- [AA) real_data_sim_dose_reclassified_uvx_as_vx.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20real_data_sim_dose_reclassified_uvx_as_vx.py)  
  Performs Mortality-Conditioned Stochastic Imputation (MCSI) to reclassify a fraction of unvaccinated deaths as vaccinated for sensitivity analyses.

- [AA) simulate deaths doseschedule and bias all AG.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doseschedule%20and%20bias%20all%20AG.py)  
  Simulates deaths and vaccination schedules across age groups to explore potential biases and test method accuracy.

- [AC) hernan_style_poold_logistics_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan_style_poold_logistics_RMST.py)  
  Implements RMST analysis using pooled logistic regression (Target Trial emulation).

- [AE) Empirical_dynamic_CC_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AE%29%20Empirical_dynamic_CC_RMST.py)  
  Computes RMST in a descriptive empirical case-control framework with dynamic covariate adjustment.

- [AF) Empirical vs causal comparison RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AF%29%20Empirical%20vs%20causal%20comparison%20RMST.py)  
  Compares RMST estimates from empirical versus causal methods to assess potential bias.

---

### Data
Input and processed datasets are stored in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra). This includes age-specific CSV mortality files created and used by the scripts.

### Plots & Logs
Interactive plots and detailed epidemiological audit logs are stored in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results).  
They visualize vaccination timing, synthetic dose assignments, exposure durations, and cumulative person-time curves.

**Raw Dataset (not included):** Vesely_106_202403141131.csv (~1.9 GB) [Download via Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)  
Science that does not share anonymized data or code, risks becoming dogmatic.  

**Disclaimer:** This repository is for methodological exploration and demonstration only and is **not intended for making causal claims**.  
