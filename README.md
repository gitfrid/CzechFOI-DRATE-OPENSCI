# <img src="https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/under%20construction.png" width="40"> Czech FOI Mortality Data Analysis – Overview 

This repository contains scripts, datasets, logs, and plots for analyzing Czech FOI mortality data, with a focus on evaluating vaccine effectiveness (VE) and identifying potential biases.

**Main Goal:**  
The project aims to fairly compare the survival time of vaccinated (VX) and unvaccinated (UVX) individuals during the study period using **RMST (Restricted Mean Survival Time)**. The scripts explore how vaccine effectiveness is calculated, detect data biases, and use simulations to test the robustness of statistical methods.

---

## Key Concept: RMST (Restricted Mean Survival Time)

RMST measures the average survival time within a fixed follow-up period (for example, 2 years).

- It provides the **average number of days a person survived** during the study period.  
- It allows comparison between groups even when follow-up times differ.  
- Differences in RMST show how many additional (or fewer) days, on average, one group survived compared to another.  

**Simple analogy:**  
On average, how many days did each person live during the study period?  
When comparing groups, RMST shows the average difference in survival days per person.

---

## Repository Structure

### Python Scripts
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

---

**Author:** AI / Drifting  **Environment:** Python ≥ 3.10  

**Raw Dataset (not included):** `Vesely_106_202403141131.csv` (~1.9 GB) [Download via Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)  
Science that does not share anonymized data or code, risks becoming dogmatic.  

**Disclaimer:** This repository is for exploration and methodological demonstration only and is not intended for making causal claims.
