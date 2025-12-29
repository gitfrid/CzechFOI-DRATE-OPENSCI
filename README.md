# Czech FOI Mortality Data Analysis

This repository is divided into two complementary domains:

- **Metrological Forensic Auditing** — to assess data integrity and detect structural bias  
- **RMST Methodological Research** — to evaluate and compare survival estimation techniques

---

## PART I: Data Integrity Audit & Bias Quantification

This analysis functions as a **forensic data audit**, not as a vaccine efficacy study.  
Its purpose is to evaluate whether the large survival benefits reported in early, crude Czech population comparisons reflect **biological protection** or are predominantly explained by **systematic cohort-selection bias**.

The results provide strong evidence that **a substantial fraction—likely the majority—of the initially reported survival advantage arises from Healthy Vaccinee Bias (HVB)** rather than from a causal mortality reduction.

---

### Audit Objective

To distinguish genuine biological effects from **metrological artifacts** introduced by cohort construction, timing, and baseline risk differences.

The central test is a *bias-detection hypothesis*:

> If estimated survival gains scale linearly with an age group’s **background mortality**, the signal is indicative of cohort selection effects rather than biological efficacy.

---

### Access to Reproducible Code

- **Primary Script:**  
  **[AG) metrological calibration.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AG%29%20metrological%20calibration.py)**

### Access to Input Data

The script reads the Czech-FOI (Freedom of Information request) raw mortality dataset Vesely_106_202403141131.csv directly:
[Download Link](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)
This dataset provides the age-group-specific mortality records that underpin the RMST gain calculations and bias detection analyses.


---

### Primary Forensic Findings

#### 1. Linearity Between Survival Gain and Background Mortality
- **Observation:** Restricted Mean Survival Time (RMST) gains increase almost linearly with age-specific background mortality across cohorts.
- **Interpretation:**  
  Biological interventions typically produce **relative risk reductions**. In contrast, a near-linear scaling of **absolute time gains** with baseline mortality is a well-known signature of **Healthy Vaccinee Bias**, where healthier individuals are preferentially vaccinated.

This pattern strongly suggests that the observed effect magnitude is driven by **pre-existing frailty differences**, not treatment efficacy.

---

#### 2. Temporal Sensitivity and Signal Erosion
- **Observation:** Observation: When a lag period is introduced (0 → 42 days), estimated RMST gains decline monotonically across all older age groups, except AG100, which shows unstable behavior.
- **Interpretation:**  
  This erosion indicates that a significant portion of the apparent survival benefit originates from **early post-vaccination exclusion of the most frail individuals**, rather than from delayed biological protection.

The persistence of erosion across increasing lags is inconsistent with a dominant causal vaccine effect and consistent with **time-dependent selection bias**.

---

### Visual Forensic Evidence

![Scientific Forensic Analysis](Plot%20Results/AG%29%20metrological%20calibration/scientific_forensic_analysis_20251228_224832.png)

---

### Bias Erosion Summary (Lag Sensitivity Analysis)

| Age Group | Raw Gain (Lag 0) | Clean Gain (Lag 42) | Signal Erosion |
| :--- | :---: | :---: | :---: |
| **60+** | 19.45 h | 11.05 h | **43.2%** |
| **70+** | 50.11 h | 28.72 h | **42.7%** |
| **80+** | 116.63 h | 63.64 h | **45.4%** |
| **90+** | 117.04 h | 65.09 h | **44.4%** |

**Key implication:**  
Across older age groups, approximately **40–45% of the initially reported survival gain disappears** once temporal bias is addressed. Given the additional linearity with background mortality, this strongly indicates that **most of the original crude effect size is non-causal**.

**Reproducible Output:**  
The full numerical results, including cohort-wise RMST gains across all lag specifications, are available as a plain-text forensic output file:

**[Scientific Metrological Forensic Analysis (TXT)](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Plot%20Results/AG%29%20metrological%20calibration/scientific_metrological_forensic_analysis.txt)**

---

### Audit Conclusion

Taken together, the linear mortality scaling and the substantial lag-induced erosion provide **convergent forensic evidence** that the large survival benefits observed in early Czech population comparisons are **predominantly attributable to Healthy Vaccinee Bias**.

Any remaining residual signal after bias correction should be interpreted cautiously and **cannot be assumed to represent causal vaccine efficacy without independent confirmation**.

---

> **Further Technical Detail:**  
> A full breakdown of model assumptions, robustness checks, and biological plausibility constraints is available in the  
> **[Technical Forensic Wiki](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki)**.

<br>

---

<br>

## **PART II: RMST Research & Methodological Comparisons**

### **Scientific Motivation**
While Part I audits the data, Part II analyzes how different **mathematical models** interpret that data. This section compares **real-world empirical observations** against **Target Trial causal emulations**.

Traditional VE metrics (such as hazard ratios) rely on strong assumptions and can be distorted when follow‑up differs between groups. **Restricted Mean Survival Time (RMST)** provides an intuitive alternative by summarizing **how long people lived**, on average, during a fixed period, making comparisons clearer and less sensitive to model assumptions.

* **Key Resource:** **[Full Wiki Results](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki)**
* **Note:** Scripts can be modified for all age groups (**0-113**) using the **Vesely_106_202403141131.csv** (~1.9 GB) dataset.

<br>

### **Workflow Overview**
```text
Raw FOI Data (All Age Groups)
      ↓
Data Export (One file per Age Group) → Real-world dataset
      ↓
Simulations (Bias-check HR=1 & Misclassification-sensitivity dataset)
      ↓
RMST Estimation (Empirical, Causal, or Experimental methods)
      ↓
Comparison of Results (Empirical vs. Causal ΔΔRMST)
      ↓
Plots, Logs, & Interpretation
```

## Concept: Restricted Mean Survival Time (RMST)

RMST measures the **average survival time** during a fixed follow-up period (e.g., 2 years).

- Represents the average number of days lived during follow-up  
- Robust to unequal follow-up durations  
- Differences correspond to days (or hours) gained or lost on average  

**Analogy:**  
*“On average, how many days did each person live during the study?”*

---

## Scripts Overview

All scripts are located in the  
[Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts).

### Data Preparation & Simulation

- **[AA) Export AG ALL from Czech FOI.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20Export%20AG%20ALL%20from%20Czech%20FOI.py)**  
  Exports raw age-group-specific mortality data into individual CSV files.

- **[AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.py)**  
  Simulates conservative, calendar-consistent reclassification of a fixed fraction of individuals recorded as unvaccinated whose dose dates are missing.  
  Designed to test sensitivity of VE and RMST estimates to plausible exposure misclassification without introducing immortal-time or negative-exposure artifacts.

  **Question addressed:**  
  *How do VE and RMST estimates change if a small fraction of unvaccinated individuals are plausibly reclassified as vaccinated based on the observed rollout?*

- **[AA) simulate deaths doseschedule and bias all AG.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doseschedule%20and%20bias%20all%20AG.py)**  
  Simulates deaths and vaccination schedules under a true null effect (HR = 1) while preserving real rollout timing.

  **Question addressed:**  
  *Do RMST and survival-analysis methods falsely detect vaccine effects when no causal effect exists?*

---

### Causal & Empirical RMST Estimation

- **[AC) hernan_style_pooled_logistics_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan_style_pooled_logistics_RMST.py)**  
  Causal RMST estimation using pooled logistic regression (Hernán style).  
  **Target-Trial emulation — methodological gold standard**, except that covariates are intentionally omitted by design.

  **Counterfactual question:**  
  *What would the average survival time have been if everyone had been vaccinated versus if no one had been vaccinated?*

- **[AE) Empirical_dynamic_CC_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AE%29%20Empirical_dynamic_CC_RMST.py)**  
  Empirical RMST estimation using dynamic exposure classification and a clone–censor design.  
  **Purely descriptive:** no regression models, parametric assumptions, or covariates.

  Addresses two complementary questions:  
  - What survival difference was observed under real-world rollout with time-varying exposure?  
  - How does this compare to a protocol-fixed clone–censor construction that removes immortal time by design?

---

### Experimental RMST Scripts

- **[AE) empirical_landmark_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/experimental/AE%29%20empirical_landmark_RMST.py)**  
  **Empirical Landmark-Conditional ΔRMST**

  Empirical landmark-conditional difference in restricted mean survival time (ΔRMST).  
  Design: sequential target-trial emulation with eligibility defined by survival to each landmark.

  Primary analysis is ITT-like (no post-landmark censoring).  
  Sensitivity analysis applies per-protocol censoring at crossover (uncorrected for informative censoring).

  **Question addressed:**  
  *Among individuals who have survived to day t, what is the difference in expected remaining survival between those already vaccinated and those not yet vaccinated (post-landmark prognosis)?*

- **[AE) C.S. Peirce evidence-weighted RMST.py (Exploratory)](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/experimental/AE%29%20C.S.%20Pierce%20evidence%20weighted%20directional%20rmst.py)**  
  **C.S. Peirce-inspired Evidence-Weighted RMST**

  Applies an information-theoretic surprisal filter to separate robust survival signals from statistical noise.  
  Daily contributions are weighted by statistical certainty: I(t) = sign(ΔS(t)) × -ln(p(t))

  High-evidence days dominate the estimate, reducing sensitivity to sparse-data fluctuations.

  **Question addressed:**  
  *How much of the observed survival benefit represents a robust signal rather than a statistical coincidence?*

  **Related Wiki Pages:**  
  - [Simple Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Simple-Explanation)  
  - [Methodical Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Methode-Paper)

---

## Data

All primary input and processed datasets are stored in the  
[Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra).

Three datasets (illustrated for age group 70) are used:

- **Real-World Data:** official Czech FOI mortality records  
- **Null-Effect Simulation (HR = 1):** validates that methods do not produce false positives  
- **Bias Stress-Test:** 5 % UVX → VX reclassification to assess misclassification sensitivity

---

## Result Plots & Logs

Plots and epidemiological logs are stored in the  
[Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results).

Observation windows are defined to ensure data completeness and temporal stability.

---

**Raw dataset (not included):**  
Vesely_106_202403141131.csv (~1.9 GB)  
[Available via Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)

> **Science that does not share anonymized data or code risks becoming dogmatic.**

---

**Author:** AI / Drifting 2025-12  
Environment details in [requirements.txt](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/tool%20scripts/Version%20Verification.txt)

---

**Disclaimer:**  
This repository is for methodological exploration only and does not make causal claims.  
May contain subtle methodological, logical, mathematical, or coding errors.
