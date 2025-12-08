# 🧬 OpenSAFELY Time-Dependent Cox Model — Age 70 Analysis

**Author:** Your Name  
**Date:** 2025-11-02  
**Environment:** Python ≥ 3.10  
**Dataset:** Vesely_106_202403141131_AG70.csv (~136k individuals)

---

## 📘 Overview

This repository implements a **time-dependent Cox proportional hazards model** in the OpenSAFELY framework to estimate **vaccine effectiveness (VE)** and **life-days saved (LDS)** among individuals aged 70.  
The model follows the causal inference principles outlined by **Miguel Hernán and Robins (2020)** and mirrors OpenSAFELY’s published analytic pipeline for time-varying exposures.

Key features:
- Time-varying vaccination exposure  
- Non-proportional hazards check via interaction (`vaccinated × time_mid_scaled`)  
- Kaplan–Meier survival curves for vaccinated vs. unvaccinated  
- Life-/Days-Saved (LDS) with 95% bootstrap CI  
- Publication-ready outputs (HTML, CSV, TXT)

---

## 📊 Outputs

| Output | Description | Example |
|--------|--------------|----------|
| [`FA_OpenSAFELY_TimeDepCox_AG70_KM.html`](./Plot%20Results/FA_OpenSAFELY_TimeDepCox_AG70_KM.html) | Interactive Kaplan–Meier survival plot (vaccinated vs unvaccinated) | [📈 View KM Plot](./Plot%20Results/FA_OpenSAFELY_TimeDepCox_AG70_KM.html) |
| [`FA_OpenSAFELY_TimeDepCox_AG70.TXT`](./Plot%20Results/FA_OpenSAFELY_TimeDepCox_AG70.TXT) | Full Cox model summary, PH check, and LDS results | [📄 View Results](./Plot%20Results/FA_OpenSAFELY_TimeDepCox_AG70.TXT) |
| [`FA_OpenSAFELY_TimeDepCox_AG70_Cox_summary.csv`](./Plot%20Results/FA_OpenSAFELY_TimeDepCox_AG70_Cox_summary.csv) | Machine-readable coefficients and statistics | [📑 CSV summary](./Plot%20Results/FA_OpenSAFELY_TimeDepCox_AG70_Cox_summary.csv) |

---

## ⚙️ Quick Start

```bash
# Clone repository
git clone https://github.com/YourUsername/OpenSAFELY-TimeDepCox-AG70.git
cd OpenSAFELY-TimeDepCox-AG70

# Install dependencies
pip install -r requirements.txt

# Run analysis
python FA_OpenSAFELY_TimeDepCox_AG70.py
