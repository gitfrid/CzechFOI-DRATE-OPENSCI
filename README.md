# <img src="https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Plot%20Results/under%20construction.png" width="40"> Czech FOI Mortality Data Analysis  

This repository analyzes Czech FOI mortality data to evaluate vaccine effectiveness (VE) and potential biases using Restricted Mean Survival Time (RMST). The project compares survival times of vaccinated (VX) and unvaccinated (UVX) individuals and uses simulations to examine how data structure and assumptions may influence results.  

Empirical approaches estimate VE directly from observed data, while causal models emulate hypothetical scenarios and can adjust for confounding. RMST measures the average survival time within a fixed follow‑up period and expresses differences in days lived between groups.

---

## Scientific Motivation

Traditional VE metrics (such as hazard ratios) rely on strong assumptions and can be distorted when follow‑up differs between groups. RMST provides a more intuitive alternative by summarizing **how long people lived**, on average, during a fixed period. This makes comparisons clearer and less sensitive to model assumptions.

Simulations complement this by testing how misclassification, timing artifacts, or structural biases could influence RMST‑based estimates.

---
## Workflow Overview
Raw FOI Data (all Age Goups)  
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

### Scripts Overview

All scripts are located in the [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts):

- [AA) Export AG ALL from Czech FOI.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20Export%20AG%20ALL%20from%20Czech%20FOI.py)  
  Exports raw age‑group‑specific mortality data into individual CSV files.

- [AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.py)  
  This script simulates conservative, calendar-consistent reclassification of a fixed fraction of “unvaccinated” individuals whose dose dates are missing, 
  in order to test how sensitive VE and RMST estimates are to plausible exposure misclassification—without introducing immortal time bias or negative exposure artifacts.

  It answers the question: What happens to VE and RMST estimates when a small fraction of unvaccinated are plausibly reclassified as vaccinated based on the observed rollout?

- [AA) simulate deaths doseschedule and bias all AG.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doseschedule%20and%20bias%20all%20AG.py)  
  Simulates deaths and vaccination schedules to explore potential biases.

  It answers the question: Do RMST and survival-analysis methods falsely detect vaccine effects when deaths are simulated under a true null effect (HR = 1) but real vaccination schedules are retained?

- [AC) hernan_style_poold_logistics_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan_style_pooled_logistics_RMST.py)  
  Causal model based RMST analysis using pooled logistic regression **(Target Trial emulation scientific Gold Standard )**.
  
  Estimates the causal effect of vaccination on survival by asking the Hypothetical counterfactual question:  
  **What would the average survival time have been if everyone in the study population had been vaccinated versus if no one had been vaccinated?**

- [AE) Empirical_dynamic_CC_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AE%29%20Empirical_dynamic_CC_RMST.py)  
  Empirical RMST estimation with dynamic exposure classification and clone–censor design.  
  This script computes non-parametric, empirical time-to-event summaries using individual-level data and discrete-time hazards.  
  **No regression models, parametric assumptions, or covariates are used.**

  It addresses two complementary descriptive questions:  
  **What survival difference was observed under real-world vaccination rollout with time-varying exposure?  
  How does this observed historical difference compare to a protocol-fixed (clone–censor) construction that removes immortal time by design?**

- [AF) Empirical vs causal comparison RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AF%29%20Empirical%20vs%20causal%20comparison%20RMST.py)  
  Compares empirical and causal RMST estimates. (under construction)

  This script compares emperical observed, bias-minimized, and causal estimates of survival differences between vaccinated and unvaccinated individuals using RMST.
  It asks how survival differed in reality under time-varying vaccination exposure, how that difference changes when empirical selection biases are reduced via clone–censor methods,
  and how both compare to explicit counterfactual causal estimates. The resulting ΔΔRMST decompositions quantify the contributions of selection bias and causal modeling assumptions.

### Experimental RMST Scripts

- [AE) empirical_landmark_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/experimental/AE%29%20empirical_landmark_RMST.py)  
  Empirical RMST estimation using a fixed‑time landmark approach. (Results seems strongly biasd -  contais bug: wrong observation window)
  
  Asking the descriptive question: How did survival differ between vaccinated and unvaccinated individuals when exposure status is frozen at a chosen landmark day, avoiding time‑varying classification?
  
- [AE) C.S. Peirce evidence weighted rmst.py (Exploratory)](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/experimental/AE%29%20C.S.%20Pierce%20evidence%20weighted%20directional%20rmst.py)  
  C.S. Peirce inspired Evidence-Weighted RMST Script uses an Information-Theoretic Surprisal-Filter to separate real survival signals from statistical noise.
  While standard models treat every day of data as equal, this script weights daily results by their statistical certainty $$I(t) = \text{sign}(\Delta S(t)) \times -\ln(p(t))$$, prioritizing high-evidence days over sparse-data flukes.

  It answers: How much of the observed survival benefit is a robust, proven signal rather than a statistical coincidence?

  **Related Wiki Pages:**  [Simple Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Simple-Explanation)  [Methodical Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Methode-Paper)

---

### Data

Input and processed datasets are stored in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra).  
These include age‑specific CSV mortality files created by the scripts.

---

### Result Plots & Logs

Plots and epidemiological logs are stored in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results).  
They visualize vaccination timing, synthetic dose assignments, exposure durations, and cumulative person‑time curves.



**Related Wiki Pages:**  [Age 70 Mortality Analysis Results](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Age-70-Mortality-Analysis-Results) 
[Plot-Files preview](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/HTML-Plot-Files-shown-as-PNG)

---

**Raw Dataset (not included):**  
Vesely_106_202403141131.csv (~1.9 GB) [Download via Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)    

Science that does not share anonymized data or the used code risks becoming dogmatic.

---

**Author:**  
AI/Drifting - 2025 

---

**Disclaimer:**  
This repository is for methodological exploration only and is not intended for making causal claims.
May contain subtle errors of a methodological, logical, mathematical, or coding nature.
