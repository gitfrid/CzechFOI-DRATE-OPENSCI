# Czech FOI Mortality Data Analysis (Scripts are prototypes)

This repository is divided into two complementary domains:

- **Bias-diagnostic target trial emulation** - test compatibility of observed mortality patterns
- **RMST Methodological Research** — evaluates and compares survival estimation techniques

---

## PART I: Explorative Target Trial Emulation Diagnostics – Czech Mortality Data (Age & Sex Only)

**Framework**: Hernán-style target trial emulation with restricted mean survival time (RMST) estimation  
**Data**: Czech national mortality registry (age, sex, death date, first-dose date)  
**Core Goal**: Perform rigorous diagnostic testing of observed all-cause mortality patterns around first-dose vaccination dates, specifically to evaluate whether the data are compatible with the **sharp null hypothesis** — i.e., no causal effect of vaccination on overall mortality (conditional on age and sex).

The pipeline systematically investigates both **positive lags** (post-vaccination mortality patterns) and **negative lags** (pre-vaccination mortality patterns). Negative lags serve as a powerful falsification test: under the sharp null, mortality patterns before the actual vaccination date should show no systematic difference compared to placebo-matched or simulated controls. Any significant deviation in negative lags strongly suggests residual selection bias (e.g., healthy vaccinee effect) rather than a true vaccine effect.

**Important Note**: This is purely an exploratory diagnostic tool — results are not causal estimates due to limited confounding control (age & sex only). Strong deviations in negative lags indicate the presence of unmeasured confounding and undermine causal interpretation of any positive-lag findings.

All code: [AG) HVE Target Trial RMST Diagnostic Core.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AG%29%20HVE%20Target%20Trial%20RMST%20Diagnostic%20Core.py)  
Data source: [Czech-FOI Dataset Vesely_106_202403141131.csv](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz) (~1.9 GB, not included in repo)

Further methodology, robustness checks, and interpretation:  
[Technical Wiki →](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki)

<br>
<br>

---

## PART II: RMST Research & Methodological Comparisons

## Motivation

While Part I audits the *data*, Part II analyzes how different **mathematical models** interpret that data. This section compares "real-world" empirical observations against "Target Trial" causal emulations.

Traditional VE metrics (such as hazard ratios) rely on strong assumptions and can be distorted when follow‑up differs between groups. RMST provides a more intuitive alternative by summarizing **how long people lived**, on average, during a fixed period. This makes comparisons clearer and less sensitive to model assumptions.

Simulations complement this by testing how misclassification, timing artifacts, or structural biases could influence RMST‑based estimates.
**The scripts can be easily modified to obtain results for all age groups 0-113.

**Results see Related:**  [WiKi](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki)


---
## Workflow Overview
Raw FOI Data (all Age Goups)  
    ↓  
Data Export (one file per Age Group) — creates the real‑world dataset  
    ↓  
Simulations (per Age Group) — creates a HR=1 bias-check and a misclassification-sensitivity dataset  
    ↓  
RMST Estimation (per Age Group) - using scientific empirical or causal Methodes or experimental Methodes — applied to all three datasets  
    ↓  
Comparison of emprical vs causal Methods and Results — across all three datasets  
    ↓  
Plots, Logs, Interpretation - for all three datasets

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

### Helper Scripts

- [AA) Export AG ALL from Czech FOI.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20Export%20AG%20ALL%20from%20Czech%20FOI.py)  
  Exports raw age‑group‑specific mortality data into individual CSV files.

- [AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.py)  
  This script simulates conservative, calendar-consistent reclassification of a fixed fraction of “unvaccinated” individuals whose dose dates are missing, 
  in order to test how sensitive VE and RMST estimates are to plausible exposure misclassification—without introducing immortal time bias or negative exposure artifacts.

  It answers the question: What happens to VE and RMST estimates when a small fraction of unvaccinated are plausibly reclassified as vaccinated based on the observed rollout?

- [AA) simulate deaths doseschedule and bias all AG.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AA%29%20simulate%20deaths%20doseschedule%20and%20bias%20all%20AG.py)  
  Simulates deaths and vaccination schedules to explore potential biases.

  It answers the question: Do RMST and survival-analysis methods falsely detect vaccine effects when deaths are simulated under a true null effect (HR = 1) but real vaccination schedules are retained?

### Scientific RSTM Scripts (prototypes)

     
-  [AC) hernan style sequential trial tte pooled logistic rmst.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan%20style%20sequential%20trial%20tte%20pooled%20logistic%20rmst.py)  
   Implements a sequential (multi-time-zero) target trial emulation using pooled logistic regression to estimate the RMST difference (Hernán style) **except covariates (HVE-adjustment) are not used by design**.
   
   Instead of picking just one starting day, the script re-checks eligibility **every single day** throughout the entire time period.
   Each day can be a new possible "start" for vaccination. It estimates the **pooled average effect** of initiating vaccination on a given eligible calendar day t versus not initiating exactly on that day (but possibly later),
   averaged across all observed eligible days t in the data.
   
   → This provides a causal contrast under the real-world observed timing and delays of vaccination initiation, while avoiding immortal time bias.
   
   
-  [AC) hernan style tte pooled logistic rmst.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan%20style%20tte%20pooled%20logistic%20rmst.py)  
   Classic single-time-zero target trial emulation using pooled logistic regression for RMST estimation.
   <br>(Hernán style) **except covariates are not used by design**
   
   Imagine we pick **one single day** at the beginning as the starting point for everyone.
     
   We ask:"If everyone had started their first vaccine dose on that one day (versus nobody ever getting vaccinated), how much longer would people have lived on average?"

   → This is the classic, straightforward way to do this kind of analysis (causal model based).

- [AC) hernan_style_poold_logistics_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AC%29%20hernan_style_pooled_logistics_RMST.py)  
  Causal model based RMST analysis using pooled logistic regression (Hernán style)
  <br>**Contains flaws !** replaced by **AC) hernan style sequential trial tte pooled logistic rmst.py** above!
    
  Estimates the causal effect of vaccination on survival by asking the Hypothetical counterfactual question:  
  What would the average survival time have been if everyone in the study population had been vaccinated versus if no one had been vaccinated?

- [AE) Empirical_dynamic_CC_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AE%29%20Empirical_dynamic_CC_RMST.py)  
  Empirical RMST estimation with dynamic exposure classification and clone–censor design.  
  **Descriptional no regression models, parametric assumptions, or covariates are used.**
  
  This script computes non-parametric, empirical time-to-event summaries using individual-level data and discrete-time hazards.  

  It addresses two complementary descriptive questions:  
  What survival difference was observed under real-world vaccination rollout with time-varying exposure?  
  How does this observed historical difference compare to a protocol-fixed (clone–censor) construction that removes immortal time by design?


### Experimental RMST Scripts

- [AE) empirical_landmark_RMST.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/experimental/AE%29%20empirical_landmark_RMST.py)  
  Empirical Landmark-Conditional ΔRMST
  
  Empirical landmark-conditional difference in restricted mean survival time (ΔRMST).
  Design: Sequential target-trial emulation with eligibility defined by survival to each landmark.
  Primary analysis ia ITT-like (no post-landmark censoring). Sensitivity analysis is Per-protocol censoring at crossover (uncorrected for informative censoring).
  
  Answers the question: Among people who have survived to day t, what is the difference in expected remaining survival between those already vaccinated and those not yet vaccinated (post-landmark prognosis)?
  
  
- [AE) C.S. Peirce evidence weighted rmst.py (Exploratory)](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/experimental/AE%29%20C.S.%20Pierce%20evidence%20weighted%20directional%20rmst.py)  
  C.S. Peirce inspired Evidence-Weighted RMST
  
  Uses an Information-Theoretic Surprisal-Filter to separate real survival signals from statistical noise.
  While standard models treat every day of data as equal, this script weights daily results by their statistical certainty $$I(t) = \text{sign}(\Delta S(t)) \times -\ln(p(t))$$, prioritizing high-evidence days over sparse-data flukes.

  It answers: How much of the observed survival benefit is a robust, proven signal rather than a statistical coincidence?
  **Related Wiki Pages:**  [Simple Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Simple-Explanation)  [Methodical Explanation](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Peircean-Evidence%E2%80%91Weighted-RMST-%E2%80%90-Methode-Paper)



- [AG) Clone Censor weight RMST Stress-Test simulation.py](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/AG%29%20Clone%20Censor%20weight%20RMST%20Stress-Test%20simulation.py)
  IPW RMST Validation — Stress-Test Script

  This script implements a simulation-based stress test of a **clone-censor-weighting (CCW)** restricted mean survival time (RMST) pipeline under realistic violations of target trial emulation assumptions.
  It evaluates the numerical robustness, directional fidelity, weight stability, and diagnostic performance of the method when sequential ignorability is challenged by:
  - Calendar-time mortality waves and phased vaccination rollout
  - Age/sex prioritization
  - Latent unmeasured health confounding affecting both treatment timing and outcome hazard

  Two scenarios are tested:
  - **Null** (true HR ≈ 1) — checks for absence of spurious signals
  - **Protective** (true HR ≈ 0.7) — assesses recovery of a true negative effect
  
  The pipeline uses strategy-specific IPCW, calendar- + relative-time B-splines in both propensity and hazard models, stabilized weights with clipping and truncation,
  pooled logistic regression (CLogLog link) on person-day rows, bootstrapped confidence intervals, and placebo falsification sampled from the numerator model.
  Results are aggregated over a t₀ grid weighted by eligibility size.

  The stress test demonstrates if CCW-RMST **fails gracefully** (directionally informative, stable diagnostics, negative lags ≈ 0, placebo centered near 0)
  under epidemiologically plausible departures from ideal conditions — providing strong support for its use as a diagnostic tool in observational vaccine effectiveness studies with limited measured confounders.



> [Technical Wiki](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/IPW-RMST-Validation-Stress%E2%80%90Test).

<br>
<br>
---

### Data

All primary input and processed datasets are hosted in the [Terra folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Terra). 

This repository includes three distinct datafiles (only for Age-Group 70) used as input for the methodes:

* **Real-World Data:** Age-specific mortality CSV files containing the raw, official Czech FOI data.
* **Null Hypothesis (HR=1) Simulation:** A synthetic dataset with a constant Hazard Ratio of 1.0 and simulated real dose schedule , used to validate that the methodologies do not produce false-positive signals.
* **Stress-Test (Bias Simulation):** The reclassified real dataset where **5% of the Unvaccinated (UVX)** cohort is intentionally shifted to the **Vaccinated (VX)** cohort to measure the impact (sensitivity) of potential misclassification bias.

---

**Raw Czech-FOI Dataset (not included repository):**  
Vesely_106_202403141131.csv (~1.9 GB) [Download via Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz) 

**Science that does not share anonymized data or the used code risks becoming dogmatic. - Thank's for request**

---

### Result Plots & Logs

Plots and epidemiological logs are stored in the [Plot Results folder](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/tree/main/Plot%20Results).  
They visualize vaccination timing, synthetic dose assignments, exposure durations, and cumulative person‑time curves.

While raw data tracking begins on 2020-01-01, each individual's active observation period begins at Observation Start (the date of the first dose administered within the age cohort) and concludes at Observation End (the final record date minus a 30-day safety buffer). This window establishes the stable interval used to evaluate vaccination strategies and survival outcomes while ensuring data completeness and stability

**Related Wiki Pages:**  
[Age 70 Mortality Analysis Results](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Age-70-Mortality-Analysis-Results) -> 
[Result-Files](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/Statistical-Analysis-Result-Files-AG70) -> 
[Plot-Files preview](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/wiki/HTML-Plot-Files-shown-as-PNG)


---

**Author:** AI/Drifting 2025-12   
To reproduce the analysis environment, install the dependencies listed in the [requirements.txt](https://github.com/gitfrid/CzechFOI-DRATE-OPENSCI/blob/main/Py%20Scripts/tool%20scripts/Version%20Verification.txt) file. 

---

**Disclaimer:**  
This repository is for methodological exploration only and is not intended for making causal claims.
<br>May contain subtle errors of a methodological, logical, mathematical, or coding nature.
**Disclaimer:**  
Methodological exploration only; no causal claims. May contain coding, mathematical, or logical limitations.
