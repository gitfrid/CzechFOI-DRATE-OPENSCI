#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from scipy.integrate import simpson
import plotly.graph_objects as go
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

"""
Title: Hernán-Style RMST Estimation Using Cox Time-Varying Model with ID-Level Bootstrap
Author: [drifting]
Date: 2025-12
Version: 5

Description
-----------
This script implements a Hernán-style causal RMST analysis using a Cox 
Time-Varying model with cloning and censoring. Marginal survival for 
vaccinated vs unvaccinated strategies is reconstructed from TTE data.
ID-level bootstrap is used for ΔRMST confidence intervals.

Features:
    • Cloning of treated and untreated strategies
    • CoxTimeVaryingFitter for marginal hazard estimation
    • Survival curve reconstruction from baseline hazard
    • Restricted Mean Survival Time (RMST) via Simpson integration
    • ID-level Bootstrap CI for ΔRMST (sequential, safe)
    • Epidemiological crude summaries (rates, RR, VE)
"""

# ======================================================================
# Configuration
# ======================================================================

AGE = 70

#INPUT = Path(fr"C:\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\CzechFOI-DRATE-OPENSCI\Plot Results\AC) HERNAN_Cox_Time_Varying_RMST\AC) HERNAN_Cox_Time_Varying_RMST_SIM")
INPUT = Path(fr"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\CzechFOI-DRATE-OPENSCI\Plot Results\AC) HERNAN_Cox_Time_Varying_RMST\AC) HERNAN_Cox_Time_Varying_RMST")

STUDY_START = pd.Timestamp("2020-01-01")
AGE_REFERENCE_YEAR = 2023

# Bootstrap settings
N_BOOT_REFIT = 5
RANDOM_SEED = 12345
SAFETY_BUFFER_DAYS = 30

# Seed RNGs
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ======================================================================
# Logging
# ======================================================================

OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.txt"
log_fh = open(log_path, "w", encoding="utf-8")
def log(msg: str):
    print(msg)
    log_fh.write(msg + "\n")

# ======================================================================
# Helper Functions
# ======================================================================

def compute_rmst(times: np.ndarray, surv: np.ndarray) -> float:
    """Compute RMST via Simpson integration."""
    if len(times) <= 1:
        return 0.0
    return float(simpson(surv, x=np.asarray(times)))

def summarize_rates(pt_u, ev_u, pt_v, ev_v):
    """Compute crude rates, RR, VE."""
    rate_u = ev_u / pt_u if pt_u > 0 else np.nan
    rate_v = ev_v / pt_v if pt_v > 0 else np.nan
    rr = rate_v / rate_u if rate_u > 0 else np.nan
    ve = (1 - rr) * 100 if np.isfinite(rr) else np.nan
    log("\n=== Epidemiological Summary ===")
    log(f"Unvaccinated: events={ev_u:,}, person-time={pt_u:.0f} days, rate={rate_u:.6e}")
    log(f"Vaccinated:   events={ev_v:,}, person-time={pt_v:.0f} days, rate={rate_v:.6e}")
    log(f"Rate ratio = {rr:.3f} → VE ≈ {ve:.1f}%")
    return dict(rate_unvacc=rate_u, rate_vacc=rate_v, rr=rr, ve=ve)

def annotate_effect(fig, label, value, ci=None, ve=None):
    """Add ΔRMST annotation to Plotly figure."""
    txt = f"{label} = {value:.2f} days"
    if ci is not None and all(np.isfinite(ci)):
        txt += f"<br>95% CI: {ci[0]:.2f}–{ci[1]:.2f}"
    if ve is not None and np.isfinite(ve):
        txt += f"<br>VE ≈ {ve:.1f}%"
    fig.add_annotation(
        text=txt,
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="black",
        align="right"
    )

def marginal_survival(tv_df: pd.DataFrame, ctv: CoxTimeVaryingFitter, treat_val: int):
    """Compute marginal survival curve for a fixed vaccination strategy."""
    tv_copy = tv_df.copy()
    tv_copy['vaccinated'] = treat_val
    lp = np.zeros(len(tv_copy))
    for cov in ctv.params_.index:
        lp += tv_copy.get(cov, 0) * ctv.params_[cov]
    surv_values = []
    baseline_h = ctv.baseline_cumulative_hazard_
    for t in baseline_h.index.values:
        H0_t = baseline_h.loc[t].values[0]
        surv_values.append(np.mean(np.exp(-H0_t * np.exp(lp))))
    return baseline_h.index.values, np.array(surv_values)

# ======================================================================
# Load Data
# ======================================================================

log(f"Loading CSV: {INPUT}")
raw = pd.read_csv(INPUT, dtype=str)
raw.columns = raw.columns.str.strip()
date_cols = ['DatumUmrti'] + [c for c in raw.columns if c.startswith('Datum_')]
for c in date_cols:
    raw[c] = pd.to_datetime(raw[c], errors='coerce')
raw['age'] = AGE_REFERENCE_YEAR - pd.to_numeric(raw['Rok_narozeni'], errors='coerce')
raw = raw[raw['age'] == AGE].copy()
if raw.empty:
    raise SystemExit(f"No subjects found for AGE={AGE}")
log(f"Subjects after age filter: {raw.shape[0]}")

# ======================================================================
# Study Window
# ======================================================================

dose_cols = [c for c in raw.columns if c.startswith("Datum_") and c != "DatumUmrti"]
last_dose = raw[dose_cols].max(axis=1, skipna=True).max(skipna=True) if dose_cols else pd.NaT
last_death = raw["DatumUmrti"].max(skipna=True)
valid_days = [(d - STUDY_START).days for d in [last_dose, last_death] if pd.notna(d)]
EXOGENOUS_STUDY_END_DAY = min(valid_days) - SAFETY_BUFFER_DAYS
FIRST_VAX_DAY = (raw["Datum_1"].min(skipna=True) - STUDY_START).days
LAST_OBS_DAY = EXOGENOUS_STUDY_END_DAY

# ======================================================================
# Cloning & Censoring
# ======================================================================

raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
raw["first_dose_day"] = (raw["Datum_1"] - STUDY_START).dt.days
tv_records = []

for _, row in raw.iterrows():
    death = row["death_day"]
    first = row["first_dose_day"]

    # Untreated clone
    start_u = FIRST_VAX_DAY
    stop_u = min(first if pd.notna(first) else LAST_OBS_DAY,
                 death if pd.notna(death) else LAST_OBS_DAY,
                 LAST_OBS_DAY)
    if stop_u > start_u:
        tv_records.append({'id': row['Rok_narozeni'], 'start': start_u, 'stop': stop_u,
                           'event': int(pd.notna(death) and death <= stop_u), 'vaccinated': 0})
    # Treated clone
    if pd.notna(first):
        start_v = max(first, FIRST_VAX_DAY)
        stop_v = min(death if pd.notna(death) else LAST_OBS_DAY, LAST_OBS_DAY)
        if stop_v > start_v:
            tv_records.append({'id': row['Rok_narozeni'], 'start': start_v, 'stop': stop_v,
                               'event': int(pd.notna(death) and death >= start_v and death <= stop_v),
                               'vaccinated': 1})

tv = pd.DataFrame(tv_records)
tv = tv[tv['stop'] > tv['start']].copy()
tv['person_time'] = tv['stop'] - tv['start']

# ======================================================================
# Epidemiology Summary
# ======================================================================

pt_un = tv.loc[tv['vaccinated']==0, 'person_time'].sum()
ev_un = tv.loc[tv['vaccinated']==0, 'event'].sum()
pt_tr = tv.loc[tv['vaccinated']==1, 'person_time'].sum()
ev_tr = tv.loc[tv['vaccinated']==1, 'event'].sum()
epi = summarize_rates(pt_un, ev_un, pt_tr, ev_tr)

# ======================================================================
# Cox Time-Varying Fit
# ======================================================================

ctv = CoxTimeVaryingFitter()
ctv.fit(tv, id_col='id', start_col='start', stop_col='stop', event_col='event', show_progress=True)
log("CoxTimeVaryingFitter summary:")
log(str(ctv.summary))

# ======================================================================
# Marginal Survival & ΔRMST
# ======================================================================

times_u, Su = marginal_survival(tv, ctv, 0)
times_v, Sv = marginal_survival(tv, ctv, 1)

common_times = np.arange(FIRST_VAX_DAY, LAST_OBS_DAY + 1)
Su = np.interp(common_times, times_u, Su)
Sv = np.interp(common_times, times_v, Sv)

delta_rmst = compute_rmst(common_times, Sv - Su)
log(f"ΔRMST (Vaccinated – Unvaccinated, Cox TV) = {delta_rmst:.2f} days")

# ======================================================================
# Sequential ID-Level Bootstrap ΔRMST (safe, no parallel refit)
# ======================================================================

# Weekly thinned grid
common_times_weekly = np.arange(FIRST_VAX_DAY, LAST_OBS_DAY + 1, 7)
ids = tv['id'].unique()

boot_refit = []
for _ in range(N_BOOT_REFIT):
    sample_ids = np.random.choice(ids, size=len(ids), replace=True)
    df_b = pd.concat([tv[tv['id']==sid] for sid in sample_ids], ignore_index=True)
    # Compute ΔRMST using original Cox fit (safe)
    t_u, Su_b = marginal_survival(df_b, ctv, 0)
    t_v, Sv_b = marginal_survival(df_b, ctv, 1)
    Su_b = np.interp(common_times_weekly, t_u, Su_b)
    Sv_b = np.interp(common_times_weekly, t_v, Sv_b)
    boot_refit.append(compute_rmst(common_times_weekly, Sv_b - Su_b))

boot_refit = np.array(boot_refit, float)
valid_refit = boot_refit[np.isfinite(boot_refit)]

if valid_refit.size:
    ci_low, ci_high = np.percentile(valid_refit, [2.5, 97.5])
else:
    ci_low, ci_high = np.nan, np.nan

log(f"ΔRMST 95% CI (sequential bootstrap, weekly grid): [{ci_low:.2f}, {ci_high:.2f}]")

# ======================================================================
# Survival Plot
# ======================================================================

fig = go.Figure()
fig.add_trace(go.Scatter(x=common_times, y=Su, mode='lines', name='Unvaccinated', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=common_times, y=Sv, mode='lines', name='Vaccinated', line=dict(color='green')))
fig.update_layout(
    title=f"Hernán-Style Cox TV: Marginal Survival (AGE={AGE})",
    xaxis_title="Days since global first-vaccination day",
    yaxis_title="Survival probability",
    template="plotly_white",
    yaxis=dict(range=[0, 1.05])
)
annotate_effect(fig, "ΔRMST", delta_rmst, ci=(ci_low, ci_high), ve=epi['ve'])

plot_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.html"
fig.write_html(str(plot_path))
log(f"Saved survival plot → {plot_path}")

# ======================================================================
# Finalize
# ======================================================================

log("Finished. Results printed to console and written to log.")
log_fh.close()
