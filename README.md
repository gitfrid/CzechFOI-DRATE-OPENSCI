# Czech FOI Mortality Data Analysis

This repository is divided into two complementary domains:

- **Metrological Forensic Auditing** — assesses data integrity and quantifies structural bias  
- **RMST Methodological Research** — evaluates and compares survival estimation techniques

---

## PART I: Data Integrity Audit & Bias Quantification

# **Update note: "Model update in progress: Implementing Poisson rate-modeling and time-varying covariate alignment to further eliminate immortal time bias (v10.0 coming soon)."**

This is a **forensic data audit**, not a vaccine efficacy study.  
Its purpose is to evaluate whether large survival advantages in early Czech population comparisons reflect **true biological protection** or are predominantly explained by **systematic cohort-selection bias (Healthy Vaccinee Bias, HVB)**.

The analysis provides strong evidence that **a substantial fraction—likely the majority—of initially reported survival gains arises from selection bias**, not causal mortality reduction.

### Audit Objective

To distinguish biological effects from **metrological artifacts** introduced by cohort construction, timing, and baseline health differences.

**Bias-detection hypothesis:**

> If estimated survival gains scale linearly with an age group’s **background mortality**, the signal reflects cohort selection rather than biological efficacy.

### Access to Reproducible Code

- **Primary Script:**  
  [AG) metrological calibration.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AG%29%20metrological%20calibration.py)

### Access to Input Data

- Czech FOI raw mortality dataset: Vesely_106_202403141131.csv  
[Download Link](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)  

The dataset provides age-specific mortality records used for RMST gain calculations and bias detection analyses.

---

### Primary Forensic Findings

#### 1. Linearity Between RMST Gain and Background Mortality
- **Observation:** RMST gains increase nearly linearly with age-specific background mortality.  
- **Interpretation:** Absolute gains scaling with baseline mortality is a hallmark of **Healthy Vaccinee Bias**, reflecting pre-existing frailty differences rather than causal protection.

#### 2. Temporal Sensitivity and Residual Signals
- **Observation:** Introducing a lag period (0 → 42 days) reduces RMST gains substantially across most age groups.  
- **Interpretation:** The bias signal is particularly strong in the **immediate post-vaccination period** (first days to weeks), when no biological vaccine effect on mortality is plausible, yet apparent survival advantages are largest — a classic hallmark of **Healthy Vaccinee Bias**. Residual positive signals after lag adjustment remain smaller, heterogeneous, and **cannot be assumed causal** without further independent validation.

---

### Visual Forensic Evidence

![Scientific Forensic Analysis](https://raw.githubusercontent.com/gitfrid/CzechFOI-DRATE-OPENSCI/main/Plot%20Results/AG%29%20metrological%20calibration/scientific_forensic_analysis_20251228_224832.png)

*(Left: RMST gain vs. background mortality · Right: RMST gain as a function of lag period by age group)*

---

### Bias Erosion Summary (Lag Sensitivity Analysis)

| Age Group | Raw Gain (Lag 0) | Clean Gain (Lag 42) | Signal Erosion |
|-----------|-----------------|-------------------|----------------|
| 60+       | 19.45 h         | 11.05 h           | 43.2%          |
| 70+       | 50.11 h         | 28.72 h           | 42.7%          |
| 80+       | 116.63 h        | 63.64 h           | 45.4%          |
| 90+       | 117.04 h        | 65.09 h           | 44.4%          |

**Reproducible Output:**  
[Plain-text scientific forensic summary](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Plot%20Results/AG%29%20metrological%20calibration/scientific_metrological_forensic_analysis.txt)

---

### Audit Conclusion

The linear mortality scaling and substantial lag-induced erosion provide **convergent evidence** that most of the apparent survival advantage arises from **Healthy Vaccinee Bias**.  

Residual positive RMST differences remain, but their magnitude is smaller, inconsistent across age strata, and **cannot be interpreted as causal** without independent validation.

> **Further Detail:** Full methodology, robustness checks, and assumptions are in the  
> [Technical Forensic Wiki](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki).

---

## PART II: RMST Research & Methodological Comparisons

This section examines **how mathematical models interpret observed mortality data** and compares empirical estimates to target-trial emulations.

### Scientific Motivation

- RMST summarizes **average survival time** over a fixed horizon (e.g., 180 days)  
- Provides **absolute survival differences** in hours or days  
- Less sensitive to model assumptions than hazard ratios  
- Enables direct bias quantification

*Full Wiki Results:* [Scientific Metrological Forensic Analysis (Parts I–III)](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki)

---

### RMST Concept

- Measures **average survival time** during fixed follow-up  
- Robust to unequal follow-up  
- Differences reflect absolute hours/days gained or lost  

**Analogy:** "How many hours, on average, did each person survive during the observation period?"

---

### Scripts Overview

- [Py Scripts folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Py%20Scripts)  

**Key Analysis Scripts:**

- Data preparation & simulation: AA) scripts  
- Empirical & causal RMST estimation: AC) and AE) scripts  
- Experimental approaches: Peircean evidence-weighted RMST  

**Peircean RMST:** separates robust survival signal from statistical noise, weighting daily contributions by certainty.  

Related Wiki pages:  
- [Simple Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Simple-Explanation)  
- [Method Paper](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Methode-Paper)

---

### Data

Primary and processed datasets stored in [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra)  

- Real-world Czech FOI mortality data  
- Null-effect simulations (HR=1)  
- Bias stress-tests (5% unvaccinated → vaccinated reclassification)

---

### Result Plots & Logs

Stored in [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results)  

Observation windows defined for **completeness and stability**.

---

**Raw dataset (~1.9 GB):** Vesely_106_202403141131.csv  
[Request via FOI](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz)

---

**Author:** AI / Drifting 2025-12  
**Environment:** [requirements.txt](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/tool%20scripts/Version%20Verification.txt)

**Disclaimer:**  
Methodological exploration only; no causal claims. May contain coding, mathematical, or logical limitations.
