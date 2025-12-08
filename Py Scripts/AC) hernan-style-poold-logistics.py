#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Hernán-Style RMST Estimation with Empirical Bootstrap
Author: [drifting]
Date: 2025-12
Version: 3 (header + comments)

Description
-----------
This script implements a causal RMST (restricted mean survival time)
analysis following Hernán-style cloning and pooled logistic regression,
with empirical ID-level bootstrap for confidence intervals. The analysis
includes epidemiological summaries (person-time, incidence rates, VE)
and interactive Plotly visualization.

Features:
    • Hernán-style cloning for vaccinated/unvaccinated strategies
    • Pooled logistic regression for marginal hazards
    • Reconstruction of survival curves from hazards
    • ΔRMST computation via Simpson integration
    • Empirical bootstrap for ΔRMST CI (ID-level resampling)
    • Epidemiological crude summaries (person-time, rates, RR, VE)
    • Interactive Plotly survival curve plot with annotations
"""

# ======================================================================
# Imports and Configuration
# ======================================================================

from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import simpson
import plotly.graph_objects as go
from joblib import Parallel, delayed

# ---------------------- User-Defined Parameters ----------------------

AGE = 70

#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) HERNAN_poold_logistics_RMST\AC) HERNAN_poold_logistics_RMST_SIM")
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) HERNAN_poold_logistics_RMST\AC) HERNAN_poold_logistics_RMST")

STUDY_START = pd.Timestamp("2020-01-01")
AGE_REF_YEAR = 2023
N_BOOT = 200                 # Number of bootstrap replicates
RANDOM_SEED = 12345
SAFETY_BUFFER = 30           # Days trimmed from study end
N_CORES = -1                 # Number of parallel cores (-1 = all cores)

# Seed RNGs for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output folder if missing

# ======================================================================
# Logging
# ======================================================================

OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
log_fh = open(OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.txt", "w", encoding="utf-8")

def log(msg: str):
    print(msg)
    log_fh.write(msg + "\n")

# ======================================================================
# Helper Functions
# ======================================================================

def compute_rmst(times: np.ndarray, surv: np.ndarray) -> float:
    """Compute restricted mean survival time (RMST) using Simpson integration."""
    return float(simpson(surv, x=np.asarray(times))) if len(times) > 1 else 0.0

def summarize_rates(pt_u, ev_u, pt_v, ev_v):
    """
    Compute crude incidence rates, rate ratio, and vaccine effectiveness.
    Returns dictionary with rate_unvacc, rate_vacc, rr, ve.
    """
    rate_u = ev_u / pt_u if pt_u > 0 else np.nan
    rate_v = ev_v / pt_v if pt_v > 0 else np.nan
    rr = rate_v / rate_u if rate_u > 0 else np.nan
    ve = (1 - rr) * 100 if np.isfinite(rr) else np.nan

    log(f"\n=== Epidemiological Summary ===")
    log(f"Unvaccinated: events={ev_u}, PT={pt_u:.0f}, rate={rate_u:.6e}")
    log(f"Vaccinated:   events={ev_v}, PT={pt_v:.0f}, rate={rate_v:.6e}")
    log(f"Rate ratio = {rr:.3f} → VE ≈ {ve:.1f}%")

    return dict(rate_unvacc=rate_u, rate_vacc=rate_v, rr=rr, ve=ve)

def compute_daily_events(df: pd.DataFrame, start_day: int, end_day: int):
    """
    Aggregate daily number of events and number at risk between start_day and end_day.
    Returns arrays: events, risk.
    """
    window = end_day - start_day
    events = np.zeros(window, int)
    risk = np.zeros(window, int)
    for _, row in df.iterrows():
        s = max(int(row["start"]), start_day)
        e = min(int(row["stop"]), end_day)
        if e <= s: continue
        s_idx, e_idx = s - start_day, e - start_day
        risk[s_idx:e_idx] += 1
        if row["event"] == 1:
            events[e_idx - 1] += 1
    return events, risk

def fit_pooled_logistic(agg: pd.DataFrame):
    """
    Fit pooled logistic regression model:
    - Dependent: daily probability of event
    - Independent: vaccinated indicator + time dummies
    Returns fitted GLM and column order for predictions.
    """
    df = agg[agg["at_risk"] > 0].copy()
    p = (df["events"] / df["at_risk"]).clip(1e-9, 1-1e-9)
    w = df["at_risk"].astype(float)
    t_dummies = pd.get_dummies(df["day"].astype("category"), prefix="t", drop_first=True).astype(float)
    X = pd.concat([t_dummies, df[["vaccinated"]].astype(float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    glm = sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w)
    return glm.fit(method="newton", maxiter=200), list(X.columns)

def predict_survival(res, cols, time_days, vacc):
    """
    Compute survival curve from pooled logistic hazards:
    S(t) = product_{s<=t} (1 - hazard_s)
    """
    Xp = pd.get_dummies(pd.Series(time_days).astype("category"), prefix="t", drop_first=True).astype(float)
    Xp["vaccinated"] = float(vacc)
    Xp = sm.add_constant(Xp, has_constant="add")
    for c in cols:
        if c not in Xp: Xp[c] = 0.0
    Xp = Xp[cols].astype(float)
    hazard = 1 / (1 + np.exp(-(Xp @ res.params[cols])))
    return np.cumprod(1 - hazard)


# ======================================================================
# Load and Prepare Data
# ======================================================================

log(f"Loading CSV: {INPUT}")
raw = pd.read_csv(INPUT, dtype=str)
raw.columns = raw.columns.str.strip()

# Convert relevant columns to datetime
date_cols = ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]
for c in date_cols: raw[c] = pd.to_datetime(raw[c], errors="coerce")

# Age filter
raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
raw = raw[raw["age"] == AGE].copy()
if raw.empty: raise SystemExit(f"No subjects for AGE={AGE}")
log(f"Subjects after age filter: {raw.shape[0]}")

# Study window determination
dose_cols = [c for c in raw.columns if c.startswith("Datum_") and c != "DatumUmrti"]
last_dose = raw[dose_cols].max(axis=1, skipna=True).max(skipna=True) if dose_cols else pd.NaT
last_death = raw["DatumUmrti"].max(skipna=True)
EXO_END_DAY = min([(d - STUDY_START).days for d in [last_dose,last_death] if pd.notna(d)]) - SAFETY_BUFFER
FIRST_VAX_DAY = (raw["Datum_1"].min(skipna=True) - STUDY_START).days
LAST_OBS_DAY = EXO_END_DAY
time_days = np.arange(0, LAST_OBS_DAY - FIRST_VAX_DAY)

raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
raw["first_dose_day"] = (raw["Datum_1"] - STUDY_START).dt.days


# ======================================================================
# Cloning for Hernán-Style Analysis
# ======================================================================

clones = []
for _, r in raw.iterrows():
    death, first = r["death_day"], r["first_dose_day"]

    # Unvaccinated clone (censor at vaccination)
    start_u = FIRST_VAX_DAY
    stop_u = min(first if pd.notna(first) else LAST_OBS_DAY,
                 death if pd.notna(death) else LAST_OBS_DAY,
                 LAST_OBS_DAY)
    if stop_u > start_u:
        clones.append((r["Rok_narozeni"], 0, start_u, stop_u, int(pd.notna(death) and death <= stop_u)))

    # Vaccinated clone (starts at first dose)
    if pd.notna(first):
        start_v = max(first, FIRST_VAX_DAY)
        stop_v = min(death if pd.notna(death) else LAST_OBS_DAY, LAST_OBS_DAY)
        if stop_v > start_v:
            clones.append((r["Rok_narozeni"], 1, start_v, stop_v, int(pd.notna(death) and death >= start_v and death <= stop_v)))

clones_df = pd.DataFrame(clones, columns=["id","vaccinated","start","stop","event"])


# ======================================================================
# Daily Aggregation
# ======================================================================

ev_v, risk_v = compute_daily_events(clones_df[clones_df["vaccinated"]==1], FIRST_VAX_DAY, LAST_OBS_DAY)
ev_u, risk_u = compute_daily_events(clones_df[clones_df["vaccinated"]==0], FIRST_VAX_DAY, LAST_OBS_DAY)

agg_df = pd.DataFrame({
    "day": np.concatenate([time_days, time_days]),
    "vaccinated": np.concatenate([np.ones_like(time_days), np.zeros_like(time_days)]),
    "events": np.concatenate([ev_v, ev_u]),
    "at_risk": np.concatenate([risk_v, risk_u])
})


# ======================================================================
# Fit Pooled Logistic Regression and Compute Survival
# ======================================================================

res, cols = fit_pooled_logistic(agg_df)
S_v = predict_survival(res, cols, time_days, 1)
S_u = predict_survival(res, cols, time_days, 0)
delta_rmst = compute_rmst(time_days,S_v) - compute_rmst(time_days,S_u)
log(f"ΔRMST (Vaccinated – Unvaccinated): {delta_rmst:.2f} days")


# ======================================================================
# Empirical Bootstrap for ΔRMST CI
# ======================================================================

id_groups = {pid: df for pid, df in clones_df.groupby("id")}
ids = np.array(list(id_groups.keys()))

def bootstrap_rmst(_):
    sample_ids = np.random.choice(ids, len(ids), replace=True)
    df_b = pd.concat([id_groups[sid] for sid in sample_ids], ignore_index=True)
    ev_vb, risk_vb = compute_daily_events(df_b[df_b["vaccinated"]==1], FIRST_VAX_DAY,LAST_OBS_DAY)
    ev_ub, risk_ub = compute_daily_events(df_b[df_b["vaccinated"]==0], FIRST_VAX_DAY,LAST_OBS_DAY)
    agg_b = pd.DataFrame({
        "day": np.concatenate([time_days,time_days]),
        "vaccinated": np.concatenate([np.ones_like(time_days), np.zeros_like(time_days)]),
        "events": np.concatenate([ev_vb,ev_ub]),
        "at_risk": np.concatenate([risk_vb,risk_ub])
    })
    res_b,_ = fit_pooled_logistic(agg_b)
    S_vb = predict_survival(res_b, cols, time_days, 1)
    S_ub = predict_survival(res_b, cols, time_days, 0)
    return compute_rmst(time_days,S_vb)-compute_rmst(time_days,S_ub)

boot_vals = Parallel(n_jobs=N_CORES, backend="threading")(delayed(bootstrap_rmst)(i) for i in range(N_BOOT))
boot_vals = np.array(boot_vals)[~np.isnan(boot_vals)]
ci_low, ci_high = np.percentile(boot_vals,[2.5,97.5]) if len(boot_vals)>=20 else (np.nan,np.nan)
log(f"ΔRMST 95% CI: [{ci_low:.2f},{ci_high:.2f}]")


# ======================================================================
# Epidemiology: Person-Time and Crude VE
# ======================================================================

clones_df["pt"] = clones_df["stop"] - clones_df["start"]
pt_u = clones_df.loc[clones_df["vaccinated"]==0,"pt"].sum()
pt_v = clones_df.loc[clones_df["vaccinated"]==1,"pt"].sum()
ev_u = clones_df.loc[clones_df["vaccinated"]==0,"event"].sum()
ev_v = clones_df.loc[clones_df["vaccinated"]==1,"event"].sum()
epi = summarize_rates(pt_u,ev_u,pt_v,ev_v)


# ======================================================================
# Plotting: Survival Curves
# ======================================================================

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_days, y=S_u, mode="lines", name="Unvaccinated", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=time_days, y=S_v, mode="lines", name="Vaccinated", line=dict(color="green")))
fig.update_layout(
    title=f"Hernán-Style Marginal Survival (AGE={AGE})",
    xaxis_title="Days since first vaccination",
    yaxis_title="Survival probability",
    template="plotly_white",
    yaxis=dict(range=[0,1.05])
)
annotation = f"ΔRMST = {delta_rmst:.2f} days"
if np.isfinite(ci_low): annotation += f"<br>95% CI: {ci_low:.2f}–{ci_high:.2f}"
if np.isfinite(epi['ve']): annotation += f"<br>VE ≈ {epi['ve']:.1f}%"
fig.add_annotation(
    text=annotation, xref="paper", yref="paper",
    x=0.98, y=0.02, showarrow=False,
    bgcolor="rgba(255,255,255,0.85)", bordercolor="black", align="right"
)
fig.write_html(OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.html")
log(f"Saved plot → {OUTPUT_BASE.name}_AG{AGE}.html")


# ======================================================================
# Finalization
# ======================================================================

log("Finished Hernán-style RMST analysis with empirical bootstrap.")
log_fh.close()
