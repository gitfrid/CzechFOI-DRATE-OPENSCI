# 🧬 Under Construction
Data driven empirical landmark analysis

Hypothesis: The only purely factual, empirical analysis is conditional on survival to vaccination (or to a landmark day). 
<br>Everything else requires counterfactual modeling.

**Estimand and design:**

Estimates empirical differences in restricted mean survival time (ΔRMST) between individuals vaccinated by each landmark day and those not yet vaccinated, conditional on survival to the landmark. 
For each landmark t, constructed risk sets of survivors, computed daily hazards as observed deaths divided by those at risk, obtained survival via the product‑limit estimator, and integrated survival over a fixed horizon τ to obtain RMST. 
Report ΔRMST = RMST(vaccinated by t) − RMST(not yet vaccinated by t).

**Inference:**

Confidence intervals were obtained via nonparametric bootstrap resampling of individuals without parametric modeling assumptions.

**Scope and limitations:**
This is a descriptive, empirical analysis. It does not estimate a causal effect from a common baseline and does not adjust for time‑varying confounding. Results are conditional on survival to each landmark and may reflect selection and calendar‑time composition.

**AE) empirical_landmark_rmst_bootstrap.py**
a descriptive, empirical analysis - doesn't fully remove all biases - don't use covariants or causal modelling! 

**Author:** AI / Drifitng  
**Date:** 2025-11-02  
**Environment:** Python ≥ 3.10  
**Raw Dataset used:** Vesely_106_202403141131_AG10.csv (~136k individuals Czech-FOI 1.9 GB)

## 
