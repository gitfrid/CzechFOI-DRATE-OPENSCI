# 🧬 Czech FOI Mortality Data Analysis – Overview

This repository contains scripts, datasets, result logs, and plots for analyzing Czech FOI mortality data, with a focus on understanding **vaccine effectiveness (VE)** while accounting for biases in the data.  

**Main Goal:**  
Explore different methods to fairly compare how long **vaccinated (VX)** and **unvaccinated (UVX)** people lived during the study period using **RMST (Restricted Mean Survival Time)**.  
The scripts examine how vaccine effectiveness is calculated, identify potential biases in analytical methods, and simulate alternative scenarios to understand how assumptions impact results.

---

## Key Concept: RMST (Restricted Mean Survival Time)

RMST measures **how long people survive during a fixed follow-up period** (for example, 2 years).  

- It calculates the **average survival time** up to a cutoff.  
- It allows comparison between groups even if follow-up times differ.  
- Differences in RMST show how much longer, on average, one group survived compared to another.  

**Simple analogy:**  
“On average, how many days did people live during the study period?”  
Comparing groups, RMST shows the average additional (or fewer) days one group lived **per person** within the same study period.

---

## Repository Structure

### Scripts  
All scripts are in the [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts):

- **`AA) Export AG ALL from Czech FOI.py`**  
  Exports raw age-group-specific mortality data from the Czech FOI dataset (one CSV-File per AG).

- **`AA) real_data_sim_dose_reclassified_uvx_as_vx.py`**  
  Performs **Mortality-Conditioned Stochastic Imputation (MCSI)** to reclassify unvaccinated deaths as vaccinated, used for sensitivity analyses (one CSV-File per AG).

- **`AA) simulate deaths doseschedule and bias all AG.py`**  
  Simulates deaths with constant death rates and vaccination schedules across age groups to explore potential biases of different methods (one CSV-File per AG).

- **`AC) hernan_style_poold_logistics_RMST.py`**  
  Implements RMST analysis using pooled logistic regression (Hernán-style approach).

- **`AE) Empirical_dynamic_CC_RMST.py`**  
  Computes RMST in a descriptive **empirical case-control framework**, dynamically adjusting for covariates.

- **`AF) Empirical vs causal comparison RMST.py`**  
  Compares RMST estimates using empirical versus causal methods to assess potential bias.

---

### Data  
The input datasets are in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra) and contain age-specific CSV mortality files created and used by the scripts.

### Plots & Epidemiological Result Logs  
Interactive plots and detailed epidemiological audit logs are in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results).  
They visualize vaccination timing, synthetic dose assignments, exposure durations, and cumulative person-time curves.

---

**Author:** AI / Drifting  
**Environment:** Python ≥ 3.10  
**Raw Dataset - not contained:** Vesely_106_202403141131.csv (Raw Data of all AG, 1.9 GB) download from a [Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)

**Disclaimer:** This repository is for **exploration and methodological demonstration**, not for making causal claims.
