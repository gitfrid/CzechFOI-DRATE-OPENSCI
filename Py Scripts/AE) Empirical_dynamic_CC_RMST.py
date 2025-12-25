#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical Time-to-Event Comparison Using Historical Switching
and Clone–Censor Design (RMST-Based Diagnostic Analysis)
================================================================
Key rules (December 2025 – strict version):
- Time 0 = day of FIRST vaccination in the entire cohort
- Absolute study end = last_obs = last relevant event date - SAFETY_BUFFER
- NOTHING is calculated, plotted or considered after last_obs
- No extrapolation, no padding, no late tail analysis

Author: AI-assisted / gitfried   Version: 1.8   Date: Dec 2025
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# =====================================================================
# Configuration
# =====================================================================
AGE = 70
STUDY_START = pd.Timestamp("2020-01-01")
AGE_REF_YEAR = 2023

IMMUNITY_LAG = 0
SAFETY_BUFFER = 30
BOOTSTRAP_REPS = 100
RANDOM_SEED = 12345
SPARSE_RISK_THRESHOLD = 10

# =====================================================================
# INPUT / OUTPUT
# =====================================================================

# --- real Czech FOI dataset per AG  ---
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) emperical_dynamic_CC_RMST\AE emperical_dynamic_CC_RMST")

# --- simulated dataset - real data with 5% death or alive UVX→VX reclassification (sensitivity) ---
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_DeathOrAlive_reclassified_PCT5_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) emperical_dynamic_CC_RMST\AE emperical_dynamic_CC_RMST_DeathOrAlive_RECLASSIFIED")

# --- simulated null dataset - HR=1 with simulated real dose schedule  ---
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
# Top-Level Functions (Required for Windows multiprocessing spawn)
# =====================================================================

def compute_discrete_rmst(S):
    """Calculates cumulative sum of survival for daily RMST."""
    return np.cumsum(S)

def compute_daily_events(df, start_day, end_day):
    """Efficiently counts daily risk and events using difference arrays."""
    window = int(end_day - start_day) + 1
    events = np.zeros(window, dtype=int)
    diff = np.zeros(window + 1, dtype=int)

    si = np.clip((df["start"].values - start_day).astype(int), 0, window)
    ei = np.clip((df["stop"].values - start_day).astype(int), 0, window)

    np.add.at(diff, si, 1)
    np.add.at(diff, ei, -1)
    risk = np.cumsum(diff)[:window]

    ev_mask = df["event"].values.astype(bool)
    ev_idx = ei[ev_mask] - 1
    valid = (ev_idx >= 0) & (ev_idx < window)
    np.add.at(events, ev_idx[valid], 1)

    return events, risk

def build_clones_and_time(raw):
    """Refined clone building with strict time-zero anchoring."""
    FIRST = int(raw["vax_day"].dropna().min())
    death_max = raw["death_day"].dropna().max()
    vax_max   = raw["vax_day"].dropna().max()
    last_relevant = min(x for x in [death_max, vax_max] if pd.notna(x))
    last_obs = last_relevant - SAFETY_BUFFER

    if last_obs <= FIRST:
        raise ValueError(f"No follow-up after safety buffer (FIRST={FIRST}, last_obs={last_obs})")

    t = np.arange(FIRST, int(last_obs) + 1)
    clones = []
    for r in raw.itertuples():
        d, f = r.death_day, r.vax_day
        # Unvaccinated clone (Historical source)
        eu = min(x for x in [f, d, last_obs] if pd.notna(x))
        if eu > FIRST:
            clones.append((0, FIRST, eu, int(pd.notna(d) and d <= eu)))
        # Vaccinated clone
        if pd.notna(f):
            sv = max(f + IMMUNITY_LAG, FIRST)
            ev = min(x for x in [d, last_obs] if pd.notna(x))
            if ev > sv:
                clones.append((1, sv, ev, int(pd.notna(d) and sv <= d <= ev)))

    clones_df = pd.DataFrame(clones, columns=["vaccinated", "start", "stop", "event"])
    return clones_df.astype({"vaccinated":"int8", "event":"int8"}), t, FIRST, last_obs

def compute_rmst_components(data, t_full, FIRST, is_historical=False):
    """Core hazard and RMST engine."""
    end_day = t_full[-1]
    if is_historical:
        n_days = len(t_full)
        r_v, r_u, ev_v, ev_u = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
        vax_days, death_days = data["vax_day"].values, data["death_day"].values
        for i, day in enumerate(t_full):
            alive = np.isnan(death_days) | (death_days >= day)
            r_v[i] = np.sum((vax_days <= day) & alive)
            r_u[i] = np.sum(((np.isnan(vax_days)) | (vax_days > day)) & alive)
            ev_v[i] = np.sum((vax_days <= day) & (death_days == day))
            ev_u[i] = np.sum(((np.isnan(vax_days)) | (vax_days > day)) & (death_days == day))
    else:
        ev_v, r_v = compute_daily_events(data[data.vaccinated==1], FIRST, end_day)
        ev_u, r_u = compute_daily_events(data[data.vaccinated==0], FIRST, end_day)

    haz_v = np.full_like(r_v, np.nan, dtype=float)
    haz_u = np.full_like(r_u, np.nan, dtype=float)
    mask_v = r_v > SPARSE_RISK_THRESHOLD
    mask_u = r_u > SPARSE_RISK_THRESHOLD
    
    haz_v[mask_v] = ev_v[mask_v] / r_v[mask_v]
    haz_u[mask_u] = ev_u[mask_u] / r_u[mask_u]

    S_v = np.minimum(np.cumprod(np.where(np.isnan(haz_v), 1.0, 1.0 - haz_v)), 1.0)
    S_u = np.minimum(np.cumprod(np.where(np.isnan(haz_u), 1.0, 1.0 - haz_u)), 1.0)

    RMST_v = compute_discrete_rmst(S_v)
    RMST_u = compute_discrete_rmst(S_u)

    return S_v, S_u, RMST_v, RMST_u, (RMST_v - RMST_u), r_v, r_u, haz_v, haz_u

def bootstrap_survival_and_delta(args):
    """Worker function for cluster bootstrap."""
    i, raw, t_full, FIRST, global_trunc_limit = args
    np.random.seed(RANDOM_SEED + i)
    indices = np.random.choice(len(raw), len(raw), replace=True)
    raw_b = raw.iloc[indices].reset_index(drop=True)
    try:
        clones_b, _, _, _ = build_clones_and_time(raw_b)
        Sv_b, Su_b, _, _, delta_b, *_ = compute_rmst_components(clones_b, t_full, FIRST, is_historical=False)
        return (Sv_b[:global_trunc_limit], 
                Su_b[:global_trunc_limit], 
                delta_b[:global_trunc_limit])
    except:
        return None

# =====================================================================
# Main execution
# =====================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    def setup_logger(path):
        lg = logging.getLogger("rmst_analysis")
        lg.setLevel(logging.INFO)
        lg.handlers = []
        fh = logging.FileHandler(path, mode="w", encoding="utf-8-sig")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        lg.addHandler(fh)
        return lg

    logger = setup_logger(LOG_OUT)
    def log(msg): print(msg); logger.info(msg)

    log("STARTING STRICT RMST BIAS DIAGNOSTIC ANALYSIS WITH PLOTS")
    log(f"Age group: {AGE} | Safety buffer: {SAFETY_BUFFER} | Bootstrap reps: {BOOTSTRAP_REPS}")

    def preprocess_raw(raw):
        raw = raw.copy()
        raw.columns = raw.columns.str.strip()
        for c in ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]:
            raw[c] = pd.to_datetime(raw[c], errors="coerce")
        raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
        raw = raw[raw["age"] == AGE].reset_index(drop=True)
        raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
        raw["vax_day"]   = (raw["Datum_1"]     - STUDY_START).dt.days
        raw.loc[raw["death_day"] < 0, "death_day"] = np.nan
        raw.loc[raw["vax_day"]   < 0, "vax_day"]   = np.nan
        return raw

    df_raw = preprocess_raw(pd.read_csv(INPUT, low_memory=False))
    clones, t_full, FIRST, last_obs = build_clones_and_time(df_raw)

    log("Computing main analysis arms...")
    S_v_cc, S_u_cc, R_v_cc, R_u_cc, Delta_cc_full, rv_cc, ru_cc, hv_cc, hu_cc = compute_rmst_components(clones, t_full, FIRST, False)
    S_v_hi, S_u_hi, R_v_hi, R_u_hi, Delta_hi_full, rv_hi, ru_hi, hv_hi, hu_hi = compute_rmst_components(df_raw, t_full, FIRST, True)

    trunc_cc = np.argmax(np.minimum(rv_cc, ru_cc) <= SPARSE_RISK_THRESHOLD) or len(t_full)
    trunc_hi = np.argmax(np.minimum(rv_hi, ru_hi) <= SPARSE_RISK_THRESHOLD) or len(t_full)
    g_idx = min(trunc_cc, trunc_hi)
    log(f"Global truncation at index {g_idx} (CC@{trunc_cc}, Hist@{trunc_hi})")

    t_days = t_full[:g_idx] - FIRST
    Delta_cc = Delta_cc_full[:g_idx]
    Delta_hi = Delta_hi_full[:g_idx]
    Sv_cc_p, Su_cc_p = S_v_cc[:g_idx], S_u_cc[:g_idx]
    Sv_hi_p, Su_hi_p = S_v_hi[:g_idx], S_u_hi[:g_idx]
    hv_cc_p, hu_cc_p = hv_cc[:g_idx], hu_cc[:g_idx]
    hv_hi_p, hu_hi_p = hv_hi[:g_idx], hu_hi[:g_idx]

    log(f"Running bootstrap ({BOOTSTRAP_REPS} reps)...")
    with ProcessPoolExecutor() as executor:
        boot_results = list(tqdm(executor.map(bootstrap_survival_and_delta, 
                                             zip(range(BOOTSTRAP_REPS), repeat(df_raw), repeat(t_full), repeat(FIRST), repeat(g_idx))), 
                                 total=BOOTSTRAP_REPS))
    
    boot_results = [r for r in boot_results if r is not None]
    b_Sv = np.array([r[0] for r in boot_results])
    b_Su = np.array([r[1] for r in boot_results])
    b_D  = np.array([r[2] for r in boot_results])

    Sv_lo, Sv_hi = np.nanpercentile(b_Sv, [2.5, 97.5], axis=0)
    Su_lo, Su_hi = np.nanpercentile(b_Su, [2.5, 97.5], axis=0)
    D_lo, D_hi_ci = np.nanpercentile(b_D, [2.5, 97.5], axis=0)

    log("Generating Plotly visualisations...")
    # 1. Survival Plot
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(x=t_days, y=Sv_hi_p, name="Vacc (Hist)", line=dict(dash='dash', color='lightgreen')))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Su_hi_p, name="Unvax (Hist)", line=dict(dash='dash', color='lightcoral')))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Sv_hi, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Sv_lo, fill='tonexty', fillcolor='rgba(0,128,0,0.1)', line_color='rgba(0,0,0,0)', name="Vacc CC 95% CI"))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Sv_cc_p, name="Vacc (CC)", line=dict(color='green', width=3)))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Su_hi, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Su_lo, fill='tonexty', fillcolor='rgba(200,0,0,0.1)', line_color='rgba(0,0,0,0)', name="Unvax CC 95% CI"))
    fig_surv.add_trace(go.Scatter(x=t_days, y=Su_cc_p, name="Unvax (CC)", line=dict(color='red', width=3)))
    fig_surv.update_layout(title=f"Survival AG{AGE}: Historical vs Clone-Censor", template="plotly_white", yaxis_range=[0.8, 1.01])
    fig_surv.write_html(PLOT_SURV)

    # 2. Delta RMST Plot
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=t_days, y=D_hi_ci, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig_delta.add_trace(go.Scatter(x=t_days, y=D_lo, fill='tonexty', fillcolor='rgba(0,0,255,0.1)', line_color='rgba(0,0,0,0)', name="CC 95% CI"))
    fig_delta.add_trace(go.Scatter(x=t_days, y=Delta_hi, name="Historical ΔRMST", line=dict(dash='dash', color='gray')))
    fig_delta.add_trace(go.Scatter(x=t_days, y=Delta_cc, name="Clone-Censor ΔRMST", line=dict(color='blue', width=3)))
    fig_delta.update_layout(title=f"Delta RMST AG{AGE}: Days Gained (Empirical)", template="plotly_white")
    fig_delta.write_html(PLOT_DRMST)

    # 3. Bias Diagnostic (Delta-Delta)
    fig_bias = go.Figure()
    fig_bias.add_trace(go.Scatter(x=t_days, y=Delta_cc - Delta_hi, fill='tozeroy', name="Bias (ΔΔRMST)", line=dict(color='purple')))
    fig_bias.update_layout(title=f"Bias Diagnostic: Clone-Censor vs Historical AG{AGE}", template="plotly_white")
    fig_bias.write_html(PLOT_DIFF)

    # 4. Hazard Ratio Plot (Smoothed)
    hr_cc = pd.Series(hv_cc_p).rolling(7, center=True).mean() / pd.Series(hu_cc_p).rolling(7, center=True).mean()
    fig_hr = go.Figure()
    fig_hr.add_trace(go.Scatter(x=t_days, y=hr_cc, name="Smoothed CC HR", line=dict(color='black', width=2)))
    fig_hr.add_hline(y=1.0, line_dash="dash", line_color="red")
    fig_hr.update_layout(title=f"Hazard Ratio (7-day window) AG{AGE}", template="plotly_white", yaxis_type="log")
    fig_hr.write_html(PLOT_HR)

    # --- ADVANCED EPIDEMIOLOGICAL SUMMARY ---
    log("\n" + "="*90)
    log("EXTENDED EPIDEMIOLOGICAL & METHODOLOGICAL SUMMARY")
    log("="*90)
    n_total = len(df_raw)
    n_vax = df_raw["vax_day"].notna().sum()
    n_deaths = df_raw["death_day"].notna().sum()
    
    log(f"Cohort size (age group {AGE}):           {n_total:,d} individuals")
    log(f"Ever vaccinated (≥1 dose):               {n_vax:,d} ({n_vax/n_total:6.1%})")
    log(f"Total deaths observed:                   {n_deaths:,d} ({n_deaths/n_total:6.1%})")
    
    final_day = int(t_days[-1])
    log(f"\nResults at day {final_day} since first vaccination in cohort:")
    log(f"  Historical ΔRMST:                        {Delta_hi[-1]:+8.2f} days")
    log(f"  Clone–Censor ΔRMST:                      {Delta_cc[-1]:+8.2f} days")
    log(f"    95% bootstrap CI (n={BOOTSTRAP_REPS}):          [{D_lo[-1]:+6.2f}, {D_hi_ci[-1]:+6.2f}] days")
    log(f"  ΔΔRMST (Bias/Selection Diagnostic):      {Delta_cc[-1]-Delta_hi[-1]:+8.2f} days")

    inc_v = 1 - Sv_cc_p[-1]
    inc_u = 1 - Su_cc_p[-1]
    ve = (1 - (inc_v / inc_u)) if inc_u > 0 else 0
    log(f"\nCrude Analysis (CC Design at day {final_day}):")
    log(f"  Crude Vaccine Effectiveness (1-RR):      {ve:+8.1%}")
    log(f"  Risk set (Vaccinated arm):               {rv_cc[g_idx-1]:6.0f}")
    log(f"  Risk set (Unvaccinated arm):             {ru_cc[g_idx-1]:6.0f}")
    log("="*90)