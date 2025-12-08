#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.integrate import simpson
from joblib import Parallel, delayed
import plotly.graph_objects as go

"""
Title: Empirical Risk-Set Expansion with Kaplan–Meier & ΔRMST
Author: [drifting]
Date: 2025-12
Version: 1

Description
-----------
This script implements an empirical risk-set expansion analysis for vaccinated vs 
unvaccinated subjects, with Kaplan–Meier survival estimation and restricted mean 
survival time (ΔRMST) computation. Bootstrap inference is included for ΔRMST CI.

Features:
    • Empirical risk-set expansion for each vaccinated index
    • Kaplan–Meier survival curves for VX and UVX
    • ΔRMST computation via Simpson integration
    • Parallelized bootstrap for CI
    • Epidemiological summaries (person-time, rates, VE)
    • Publication-quality interactive Plotly visualization
"""

# ======================================================================
# Configuration
# ======================================================================

AGE = 70

#INPUT = Path(r"C:\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(r"C:\CzechFOI-DRATE-OPENSCI\Plot Results\AE) empirical risk set expansion\AE) Vesely_106_202403141131_empirical-CCW_SIM")
INPUT = Path(fr"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(r"C:\CzechFOI-DRATE-OPENSCI\Plot Results\AE) empirical risk set expansion\AE) Vesely_106_202403141131_empirical-CCW")

STUDY_START = pd.Timestamp("2020-01-01")
REFERENCE_YEAR = 2023
HORIZON_DAYS = 700
T_GRID_STEP = 5
N_BOOT = 25
RANDOM_SEED = 42
UVX_SAMPLE_PER_VX = 50
SAFETY_BUFFER_DAYS = 30
N_CORES = -1  # all CPU cores

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Logging
# ======================================================================

log_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.txt"
log_fh = open(log_path, "w", encoding="utf-8")
def log(msg: str):
    print(msg)
    log_fh.write(msg + "\n")

# ======================================================================
# Helper Functions
# ======================================================================

def compute_rmst(times: np.ndarray, surv: np.ndarray) -> float:
    """Compute restricted mean survival time (RMST) via Simpson integration."""
    if len(times) <= 1:
        return 0.0
    return float(simpson(surv, x=np.asarray(times)))

def summarize_rates(pt_uvx, ev_uvx, pt_vx, ev_vx):
    """Compute crude rates, rate ratio, and VE; log summary."""
    rate_uvx = ev_uvx / pt_uvx if pt_uvx > 0 else np.nan
    rate_vx = ev_vx / pt_vx if pt_vx > 0 else np.nan
    rr = rate_vx / rate_uvx if rate_uvx > 0 else np.nan
    ve = (1 - rr) * 100 if np.isfinite(rr) else np.nan

    log("\n=== Epidemiological Summary ===")
    log(f"Unvaccinated: events={ev_uvx:,}, person-time={pt_uvx:.0f} days, rate={rate_uvx:.6e}")
    log(f"Vaccinated:   events={ev_vx:,}, person-time={pt_vx:.0f} days, rate={rate_vx:.6e}")
    log(f"Rate ratio = {rr:.3f} → VE ≈ {ve:.1f}%")

    return dict(rate_unvacc=rate_uvx, rate_vacc=rate_vx, rr=rr, ve=ve)

def annotate_effect(fig, label, value, ci=None, ve=None):
    txt = f"{label} = {value:.2f} days"
    if ci is not None and all(np.isfinite(ci)):
        txt += f"<br>95% CI: {ci[0]:.2f}–{ci[1]:.2f}"
    if ve is not None and np.isfinite(ve):
        txt += f"<br>VE ≈ {ve:.1f}%"
    fig.add_annotation(
        text=txt, xref="paper", yref="paper",
        x=0.98, y=0.02, showarrow=False,
        bgcolor="rgba(255,255,255,0.85)", bordercolor="black", align="right"
    )

# ======================================================================
# Data Loading & Preparation
# ======================================================================

def read_and_prepare(path: Path):
    df = pd.read_csv(path, parse_dates=True, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
    df['age'] = REFERENCE_YEAR - df['birth_year']
    df = df[df['age'] == AGE].copy()
    df['death_day'] = (pd.to_datetime(df['datumumrti'], errors='coerce') - STUDY_START).dt.days
    dose_cols = [f'datum_{i}' for i in range(1, 8)]
    for c in dose_cols:
        if c in df.columns:
            df[c + '_day'] = (pd.to_datetime(df[c], errors='coerce') - STUDY_START).dt.days
    df['first_dose_day'] = df[[c+'_day' for c in dose_cols if c+'_day' in df.columns]].min(axis=1, skipna=True)
    df['last_dose_day'] = df[[c+'_day' for c in dose_cols if c+'_day' in df.columns]].max(axis=1, skipna=True)
    return df

# ======================================================================
# Study Window
# ======================================================================

def compute_study_window(df: pd.DataFrame):
    last_dose_day = df['last_dose_day'].max(skipna=True)
    last_death_day = df['death_day'].max(skipna=True)
    valid_days = [d for d in [last_dose_day, last_death_day] if not pd.isna(d)]
    if not valid_days:
        raise SystemExit("No valid study end.")
    EXOGENOUS_STUDY_END_DAY = min(valid_days) - SAFETY_BUFFER_DAYS
    FIRST_VAX_DAY = int(df['first_dose_day'].min(skipna=True))
    LAST_OBS_DAY = int(EXOGENOUS_STUDY_END_DAY)
    WINDOW_LENGTH = LAST_OBS_DAY - FIRST_VAX_DAY
    log(f"Study start day: {FIRST_VAX_DAY}, end day: {LAST_OBS_DAY}, duration: {WINDOW_LENGTH}")
    return FIRST_VAX_DAY, LAST_OBS_DAY, WINDOW_LENGTH

# ======================================================================
# Empirical Risk-Set Expansion
# ======================================================================

def riskset_expansion(df: pd.DataFrame, horizon=HORIZON_DAYS):
    vx_df = df[df['first_dose_day'].notna()].reset_index(drop=True)
    uvx_df = df.copy().reset_index(drop=True)
    INF = 1e9
    rng = np.random.default_rng(RANDOM_SEED)
    recs = []
    for i, t_index in enumerate(vx_df['first_dose_day']):
        eligible = np.where(
            (uvx_df['death_day'].fillna(INF) > t_index) &
            (uvx_df['first_dose_day'].fillna(INF) > t_index)
        )[0]
        if not len(eligible):
            continue
        sampled = rng.choice(eligible, min(len(eligible), UVX_SAMPLE_PER_VX), replace=False)
        vx_row = vx_df.iloc[i]
        end_vx = min(vx_row['death_day'] if pd.notna(vx_row['death_day']) else INF,
                     vx_row['first_dose_day'] + horizon)
        recs.append({'group':'VX','follow_time':end_vx - vx_row['first_dose_day'],'event':pd.notna(vx_row['death_day']) and vx_row['death_day'] <= end_vx})
        for u_idx in sampled:
            u = uvx_df.iloc[u_idx]
            end_uv = min(u['death_day'] if pd.notna(u['death_day']) else INF,
                         vx_row['first_dose_day'] + horizon)
            recs.append({'group':'UVX','follow_time':end_uv - vx_row['first_dose_day'],'event':pd.notna(u['death_day']) and u['death_day'] <= end_uv})
    rs_df = pd.DataFrame(recs)
    return rs_df[rs_df['group']=='VX'], rs_df[rs_df['group']=='UVX']

# ======================================================================
# Kaplan–Meier & ΔRMST
# ======================================================================

def pooled_km_and_rmst(vx: pd.DataFrame, uvx: pd.DataFrame, t_grid: np.ndarray):
    if vx.empty or uvx.empty:
        return np.ones_like(t_grid), np.ones_like(t_grid), 0.0
    kmf = KaplanMeierFitter()
    kmf.fit(vx['follow_time'], event_observed=vx['event'])
    s_vx = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])
    kmf.fit(uvx['follow_time'], event_observed=uvx['event'])
    s_uvx = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])
    delta_rmst = compute_rmst(t_grid, s_vx - s_uvx)
    return s_uvx, s_vx, delta_rmst

# ======================================================================
# Parallel Bootstrap
# ======================================================================

def paired_bootstrap_rmst(vx: pd.DataFrame, uvx: pd.DataFrame, t_grid: np.ndarray, n_boot=N_BOOT):
    n = len(vx)
    if n==0: return np.array([])
    def one_iter(seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0,n,n)
        _, _, r = pooled_km_and_rmst(vx.iloc[idx], uvx.iloc[idx], t_grid)
        return r
    return np.array(Parallel(n_jobs=N_CORES)(delayed(one_iter)(RANDOM_SEED+i) for i in range(n_boot)))

# ======================================================================
# Diagnostics
# ======================================================================

def compute_epi(vx: pd.DataFrame, uvx: pd.DataFrame):
    pt_vx, ev_vx = vx['follow_time'].sum(), vx['event'].sum()
    pt_uvx, ev_uvx = uvx['follow_time'].sum(), uvx['event'].sum()
    return summarize_rates(pt_uvx, ev_uvx, pt_vx, ev_vx)

# ======================================================================
# Plotting
# ======================================================================

def plot_survival(vx, uvx, t_grid, delta_rmst, ci, ve, output_html):
    kmf = KaplanMeierFitter()
    kmf.fit(vx['follow_time'], event_observed=vx['event'])
    s_vx = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])
    kmf.fit(uvx['follow_time'], event_observed=uvx['event'])
    s_uvx = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_grid, y=s_vx, name='Vaccinated', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=t_grid, y=s_uvx, name='Unvaccinated', line=dict(color='blue')))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([s_vx, s_uvx[::-1]]),
        fill='toself', fillcolor='rgba(0,200,0,0.2)', line=dict(color='rgba(255,255,255,0)'), name='ΔRMST area'
    ))
    annotate_effect(fig, "ΔRMST", delta_rmst, ci=ci, ve=ve['ve'])
    fig.update_layout(title=f"Kaplan–Meier Survival (AGE={AGE}) — Empirical Risk-Set Expansion",
                      xaxis_title="Days since index", yaxis_title="Survival probability",
                      yaxis=dict(range=[0,1.05]), template='plotly_white')
    fig.write_html(output_html)
    log(f"Plot saved: {output_html}")

# ======================================================================
# Main Execution
# ======================================================================

def main():
    t0 = pd.Timestamp.now()
    log(f"Loading CSV: {INPUT}")
    df = read_and_prepare(INPUT)
    log(f"Subjects after AGE filter: {len(df)}")

    FIRST_VAX_DAY, LAST_OBS_DAY, _ = compute_study_window(df)
    vx, uvx = riskset_expansion(df)
    log(f"Risk-set expansion: VX={len(vx)}, UVX={len(uvx)}")

    t_grid = np.arange(0, HORIZON_DAYS+1, T_GRID_STEP)
    s_uvx, s_vx, delta_rmst = pooled_km_and_rmst(vx, uvx, t_grid)
    boot_vals = paired_bootstrap_rmst(vx, uvx, t_grid)
    ci = np.percentile(boot_vals,[2.5,97.5]) if len(boot_vals) else (np.nan,np.nan)
    ve = compute_epi(vx, uvx)

    log(f"ΔRMST = {delta_rmst:.4f} days, 95% CI: {ci[0]:.4f}–{ci[1]:.4f}, VE ≈ {ve['ve']:.1f}%")
    plot_survival(vx, uvx, t_grid, delta_rmst, ci, ve, f"{OUTPUT_BASE}_AG{AGE}.html")

    log(f"Summary saved to: {OUTPUT_BASE}_AG{AGE}.txt")
    log_fh.close()

if __name__ == "__main__":
    main()
