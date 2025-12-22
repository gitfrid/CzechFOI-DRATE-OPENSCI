#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Historical vs Clone–Censor RMST Analysis (Descriptive + Bias Diagnostic)
======================================================================

This script performs a fully empirical time-to-event analysis using 
individual-level data to compute and compare three distinct quantities:

1) Historical (Naive) ΔRMST
   -------------------------
   A purely descriptive estimate of what ACTUALLY happened historically. 
   Individuals contribute person-time to the unvaccinated pool until the 
   day of vaccination, then transition to the vaccinated pool. 
   
   Reflects real-world behavior, rollout timing, frailty selection, 
   and immortal time effects exactly as they occurred in the population.

2) Clone–Censor (CC) ΔRMST
   -----------------------
   A bias-minimized empirical association estimate constructed via 
   cloning and artificial censoring at protocol deviation. 

   Approximates a hypothetical per-protocol comparison using observed 
   hazards, effectively mitigating immortal time bias by ensuring 
   treatment assignment is fixed at t=0 of the rollout.

3) ΔΔRMST(t) = CC ΔRMST − Historical ΔRMST
   ----------------------------------------
   A diagnostic quantity measuring the magnitude and timing of 
   selection bias (e.g., healthy vaccinee effects). 
   
   Quantifies the "bias gap": the distance between the observed historical 
   association and the bias-minimized construction.

4) Hazard Ratio (HR) Diagnostic
   ----------------------------
   A daily smoothed (7-day rolling) empirical Hazard Ratio comparison. 
   Used to identify if bias is concentrated during specific phases 
   (e.g., early rollout vs. late follow-up).

Outputs:
--------
• CSV: Detailed daily survival curves, RMST, ΔRMST, and ΔΔRMST.
• HTML (Interactive Plotly):
    - ΔRMST Comparison (Historical vs. CC with 95% CI)
    - Survival Curves (CC Model)
    - ΔΔRMST (Bias Diagnostic Magnitude)
    - Daily Hazard Ratio (7-day Smoothed Diagnostic)
• LOG: High-fidelity summary of person-days, deaths, and VE(τ).

Author:  drifting + AI Date:    2025-12-22 Version: 1
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# =====================================================================
# Configuration
# =====================================================================
AGE = 70
STUDY_START = pd.Timestamp("2020-01-01")
AGE_REF_YEAR = 2023

IMMUNITY_LAG = 0
SAFETY_BUFFER = 30
SPARSE_RISK_THRESHOLD = 10
BOOTSTRAP_REPS = 30
RANDOM_SEED = 12345

# =====================================================================
# INPUT / OUTPUT
# =====================================================================

# --- real Czech FOI data ---
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) emperical_dynamic_CC_RMST\AE emperical_dynamic_CC_RMST")

# --- real data with 20% UVX→VX reclassification (sensitivity) ---
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) emperical_dynamic_CC_RMST\AE emperical_dynamic_CC_RMST_RECLASSIFIED")

# --- simulated HR=1 null dataset ---
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) emperical_dynamic_CC_RMST\AE emperical_dynamic_CC_RMST_SIM")

OUT_BASE = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.name}_AG{AGE}")
OUT_BASE.parent.mkdir(parents=True, exist_ok=True)

CSV_OUT    = OUT_BASE.with_suffix(".csv")
LOG_OUT    = OUT_BASE.with_suffix(".log")
PLOT_DRMST = OUT_BASE.with_name(f"{OUT_BASE.name}_Delta_RMST.html")
PLOT_SURV  = OUT_BASE.with_name(f"{OUT_BASE.name}_Survival.html")
PLOT_DIFF  = OUT_BASE.with_name(f"{OUT_BASE.name}_DeltaDelta_RMST.html")
PLOT_HR    = OUT_BASE.with_name(f"{OUT_BASE.name}_Hazard_Ratio.html")

# =====================================================================
# Logging Setup
# =====================================================================
def setup_logger(path):
    lg = logging.getLogger("rmst_analysis")
    lg.setLevel(logging.INFO)
    lg.handlers = []
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    lg.addHandler(fh)
    return lg

logger = setup_logger(LOG_OUT)

def log(msg):
    print(msg)
    logger.info(msg)

# =====================================================================
# Core Computational Functions (Untouched)
# =====================================================================
def rmst_curve(S, t):
    out = np.zeros_like(t, dtype=float)
    for i in range(1, len(t)):
        out[i] = simpson(S[: i + 1], x=t[: i + 1])
    return out

def compute_daily_events(df, start_day, end_day):
    window = int(end_day - start_day)
    events = np.zeros(window, dtype=int)
    diff = np.zeros(window + 1, dtype=int)
    si = np.clip((df["start"].values - start_day).astype(int), 0, window)
    ei = np.clip((df["stop"].values - start_day).astype(int), 0, window)
    np.add.at(diff, si, 1)
    np.add.at(diff, ei, -1)
    risk = np.cumsum(diff)[:window]
    ev_idx = ei[df["event"].values.astype(bool)] - 1
    ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < window)]
    np.add.at(events, ev_idx, 1)
    return events, risk

def preprocess_raw(raw):
    raw = raw.copy()
    raw.columns = raw.columns.str.strip()
    for c in ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")
    raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw = raw[raw["age"] == AGE].reset_index(drop=True)
    raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
    raw["vax_day"] = (raw["Datum_1"] - STUDY_START).dt.days
    raw.loc[raw["death_day"] < 0, "death_day"] = np.nan
    raw.loc[raw["vax_day"] < 0, "vax_day"] = np.nan
    return raw

def build_clones(raw):
    FIRST = int(raw["vax_day"].min(skipna=True))
    last_obs = min(x for x in [raw["death_day"].max(skipna=True), raw["vax_day"].max(skipna=True)] if pd.notna(x)) - SAFETY_BUFFER
    if last_obs <= FIRST: raise ValueError("Insufficient follow-up window")
    t = np.arange(int(last_obs - FIRST))
    clones = []
    for _, r in raw.iterrows():
        d, f = r["death_day"], r["vax_day"]
        su, eu = FIRST, min(x for x in [f, d, last_obs] if pd.notna(x))
        if eu > su: clones.append((0, su, eu, int(pd.notna(d) and d <= eu)))
        if pd.notna(f):
            sv, ev = max(f + IMMUNITY_LAG, FIRST), min(x for x in [d, last_obs] if pd.notna(x))
            if ev > sv: clones.append((1, sv, ev, int(pd.notna(d) and sv <= d <= ev)))
    clones_df = pd.DataFrame(clones, columns=["vaccinated", "start", "stop", "event"])
    return clones_df.astype({"vaccinated": "int8", "event": "int8"}), t, FIRST, last_obs

def compute_cc_rmst(clones, t, FIRST, last_obs):
    ev_v, r_v = compute_daily_events(clones[clones.vaccinated == 1], FIRST, last_obs)
    ev_u, r_u = compute_daily_events(clones[clones.vaccinated == 0], FIRST, last_obs)
    haz_v = np.where(r_v > SPARSE_RISK_THRESHOLD, ev_v / r_v, 0.0)
    haz_u = np.where(r_u > SPARSE_RISK_THRESHOLD, ev_u / r_u, 0.0)
    S_v, S_u = np.cumprod(1 - haz_v), np.cumprod(1 - haz_u)
    RMST_v, RMST_u = rmst_curve(S_v, t), rmst_curve(S_u, t)
    return S_v, S_u, RMST_v, RMST_u, RMST_v - RMST_u, r_v, r_u, haz_v, haz_u

def compute_historical_rmst(raw, FIRST, last_obs, t):
    days = np.arange(FIRST, int(last_obs))
    n_days = len(days)
    r_v, r_u, ev_v, ev_u = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
    vax_days, death_days = raw["vax_day"].values, raw["death_day"].values
    for i, day in enumerate(days):
        r_v[i] = np.sum((vax_days <= day) & ((np.isnan(death_days)) | (death_days > day)))
        r_u[i] = np.sum(((np.isnan(vax_days)) | (vax_days > day)) & ((np.isnan(death_days)) | (death_days > day)))
        ev_v[i] = np.sum((vax_days <= day) & (death_days == day))
        ev_u[i] = np.sum(((np.isnan(vax_days)) | (vax_days > day)) & (death_days == day))
    haz_v, haz_u = np.where(r_v > SPARSE_RISK_THRESHOLD, ev_v / r_v, 0.0), np.where(r_u > SPARSE_RISK_THRESHOLD, ev_u / r_u, 0.0)
    S_v, S_u = np.cumprod(1 - haz_v), np.cumprod(1 - haz_u)
    RMST_v, RMST_u = rmst_curve(S_v, t), rmst_curve(S_u, t)
    return S_v, S_u, RMST_v, RMST_u, RMST_v - RMST_u, r_v, r_u, haz_v, haz_u

def bootstrap_once(i, raw):
    np.random.seed(RANDOM_SEED + i)
    raw_b = raw.sample(frac=1, replace=True).reset_index(drop=True)
    clones_b, t_b, FIRST_b, last_obs_b = build_clones(raw_b)
    res = compute_cc_rmst(clones_b, t_b, FIRST_b, last_obs_b)
    return res[4] # Delta_cc

# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    log("=== INITIALIZING EPIDEMIOLOGICAL BIAS DIAGNOSTIC ===")
    raw = preprocess_raw(pd.read_csv(INPUT, low_memory=False))
    
    clones, t, FIRST, last_obs = build_clones(raw)
    S_v_cc, S_u_cc, RMST_v_cc, RMST_u_cc, Delta_cc, r_v_cc, r_u_cc, h_v_cc, h_u_cc = compute_cc_rmst(clones, t, FIRST, last_obs)
    S_v_hist, S_u_hist, RMST_v_hist, RMST_u_hist, Delta_hist, r_v_hist, r_u_hist, h_v_hist, h_u_hist = compute_historical_rmst(raw, FIRST, last_obs, t)

    Delta_diff = Delta_cc - Delta_hist
    VE_cc = 1 - (1 - S_v_cc[-1]) / (1 - S_u_cc[-1]) if S_u_cc[-1] < 1 else np.nan

    # Calculate Empirical HR (7-day smoothed)
    def smooth_hr(h_v, h_u):
        hv_s = pd.Series(h_v).rolling(7, center=True).mean()
        hu_s = pd.Series(h_u).rolling(7, center=True).mean()
        return hv_s / hu_s

    hr_cc = smooth_hr(h_v_cc, h_u_cc)
    hr_hist = smooth_hr(h_v_hist, h_u_hist)

    log(f"Running bootstrap ({BOOTSTRAP_REPS} reps)...")
    with ProcessPoolExecutor() as ex:
        boots = list(tqdm(ex.map(bootstrap_once, range(BOOTSTRAP_REPS), [raw]*BOOTSTRAP_REPS), total=BOOTSTRAP_REPS))
    boot_arr = np.array(boots)
    lo, hi = np.nanpercentile(boot_arr, 2.5, axis=0), np.nanpercentile(boot_arr, 97.5, axis=0)

    # ------------------ Final summary logging ------------------
    log("\n" + "="*74)
    log(f"EPIDEMIOLOGICAL SUMMARY: AG{AGE}")
    log("="*74)
    log(f"Historical ΔRMST: {Delta_hist[-1]:+.2f} days")
    log(f"Clone-Censor ΔRMST: {Delta_cc[-1]:+.2f} days (95% CI: {lo[-1]:.2f}, {hi[-1]:.2f})")
    log(f"Bias Diagnostic (ΔΔRMST): {Delta_diff[-1]:+.2f} days")
    log("="*74)

    # ------------------ Plotly Visualizations ------------------
    # Plot 1: Delta RMST Comparison
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=Delta_hist, name="Historical (Reality)", line=dict(color='gray', dash='dash')))
    fig1.add_trace(go.Scatter(x=t, y=Delta_cc, name="CC (Bias-Minimized)", line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=t, y=hi, fill=None, mode='lines', line_color='rgba(0,0,255,0)', showlegend=False))
    fig1.add_trace(go.Scatter(x=t, y=lo, fill='tonexty', mode='lines', line_color='rgba(0,0,255,0)', fillcolor='rgba(0,0,255,0.1)', name="95% CI"))
    fig1.update_layout(template="plotly_white", title=f"AG{AGE}: Accumulated Life-Days Gained/Lost", xaxis_title="Days since Start", yaxis_title="ΔRMST (Days)")
    fig1.write_html(PLOT_DRMST)

    # Plot 2: Survival Curves
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=S_u_cc, name="Unvax (CC)", line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=t, y=S_v_cc, name="Vax (CC)", line=dict(color='green')))
    fig2.update_layout(template="plotly_white", title=f"AG{AGE}: Survival Probabilities (CC Model)", yaxis_range=[S_u_cc.min()*0.99, 1.0])
    fig2.write_html(PLOT_SURV)

    # Plot 3: Bias Diagnostic
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=Delta_diff, fill='tozeroy', name="ΔΔRMST Bias", line=dict(color='purple')))
    fig3.update_layout(template="plotly_white", title=f"AG{AGE}: Selection Bias Magnitude Over Time", yaxis_title="Difference (Days)")
    fig3.write_html(PLOT_DIFF)

    # Plot 4: Hazard Ratio Diagnostic
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=t, y=hr_hist, name="Historical HR", line=dict(color='gray', dash='dot')))
    fig4.add_trace(go.Scatter(x=t, y=hr_cc, name="Clone-Censor HR", line=dict(color='black', width=2)))
    fig4.add_hline(y=1.0, line_dash="dash", line_color="red")
    fig4.update_layout(template="plotly_white", title=f"AG{AGE}: Daily Hazard Ratio (7-day Smooth)", yaxis_type="log", yaxis_title="Hazard Ratio (V vs U)")
    fig4.write_html(PLOT_HR)

    # Save Data
    results_df = pd.DataFrame({"t": t, "Delta_hist": Delta_hist, "Delta_CC": Delta_cc, "Delta_diff": Delta_diff, "HR_CC": hr_cc})
    results_df.to_csv(CSV_OUT, index=False)
    log(f"Results saved to {CSV_OUT}")