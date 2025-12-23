# <img src="https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Plot%20Results/under%20construction.png" width="40"> Czech FOI Mortality Data Analysis  

This repository analyzes Czech FOI mortality data to evaluate vaccine effectiveness (VE) and potential biases using Restricted Mean Survival Time (RMST). The project compares survival times of vaccinated (VX) and unvaccinated (UVX) individuals and uses simulations to examine how data structure and assumptions may influence results.  

Empirical approaches estimate VE directly from observed data, while causal models emulate hypothetical scenarios to adjust for confounding. RMST measures the average survival time within a fixed follow‑up period and expresses differences in days lived between groups.

---

## Scientific Motivation

Traditional VE metrics (such as hazard ratios) rely on strong assumptions and can be distorted when follow‑up differs between groups. RMST provides a more intuitive alternative by summarizing **how long people lived**, on average, during a fixed period. This makes comparisons clearer and less sensitive to model assumptions.

Simulations complement this by testing how misclassification, timing artifacts, or structural biases could influence RMST‑based estimates.

---
## Workflow Overview
Raw FOI Data  (all Age Goups)
    ↓  
Data Export (per Age Group) — creates the real‑world dataset  
    ↓  
Simulations (per Age Group) — creates bias-check and misclassification-sensitivity dataset  
    ↓  
RMST Estimation (Empirical and Causal) — applied to all datasets  
    ↓  
Comparison of emprical vs causal Methods and Results — across all datasets  
    ↓  
Plots, Logs, Interpretation

---

## Concept: RMST (Restricted Mean Survival Time)

RMST measures the average survival time during a fixed study period (e.g., 2 years).  

- Shows the average number of days a person lived during follow‑up  
- Works even when follow‑up differs between groups  
- RMST differences represent how many more (or fewer) days one group survived on average  

Analogy: *“On average, how many days did each person live during the study?”*

---

### DataScripts & Data Overview

All scripts are located in the [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts):

- [AA) Export AG ALL from Czech FOI.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20Export%20AG%20ALL%20from%20Czech%20FOI.py)  
  Exports raw age‑group‑specific mortality data into individual CSV files.

- [AA) real_data_sim_dose_reclassified_uvx_as_vx.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20real_data_sim_dose_reclassified_uvx_as_vx.py)  
  Performs Mortality‑Conditioned Stochastic Imputation (MCSI) to test misclassification sensitivity.

- [AA) simulate deaths doseschedule and bias all AG.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doseschedule%20and%20bias%20all%20AG.py)  
  Simulates deaths and vaccination schedules to explore potential biases.

- [AC) hernan_style_poold_logistics_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan_style_pooled_logistics_RMST.py)  
  RMST analysis using pooled logistic regression (Target Trial emulation).
  
  Estimates the causal effect of vaccination on survival by asking a counterfactual question: What would the average survival time have been if everyone in the study population had been vaccinated versus if no one had been vaccinated?

- [AE) Empirical_dynamic_CC_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AE%29%20Empirical_dynamic_CC_RMST.py)  
  Empirical RMST estimation with dynamic exposure classification.
  
  Asking a descriptive question: What survival difference was observed under real-world vaccination rollout with time-varying exposure, using purely descriptive empirical data?

- [AF) Empirical vs causal comparison RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AF%29%20Empirical%20vs%20causal%20comparison%20RMST.py)  
  Compares empirical and causal RMST estimates.

  This script compares emperical observed, bias-minimized, and causal estimates of survival differences between vaccinated and unvaccinated individuals using RMST.
  It asks how survival differed in reality under time-varying vaccination exposure, how that difference changes when empirical selection biases are reduced via clone–censor methods,
  and how both compare to explicit counterfactual causal estimates. The resulting ΔΔRMST decompositions quantify the contributions of selection bias and causal modeling assumptions.

---

### Data

Input and processed datasets are stored in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra).  
These include age‑specific CSV mortality files created by the scripts.

---

### Plots & Logs

Plots and audit logs are stored in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results).  
They visualize vaccination timing, synthetic dose assignments, exposure durations, and cumulative person‑time curves.

---

**Raw Dataset (not included):**  
Vesely_106_202403141131.csv (~1.9 GB) [Download via Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)    

Science that does not share anonymized data or the used code risks becoming dogmatic.

---

**Disclaimer:**  
This repository is for methodological exploration only and is not intended for making causal claims.
