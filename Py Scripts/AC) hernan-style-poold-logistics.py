#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import simpson
from scipy.special import expit
import plotly.graph_objects as go
from joblib import Parallel, delayed
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

"""
Title: Hernán-Style RMST Estimation with Empirical Bootstrap (Clone-Aware)
Author: [drifting]
Date: 2025-12
Version: 8

Description
-----------
This script implements causal RMST (restricted mean survival time) analysis
using Hernán-style cloning and pooled logistic regression, with an empirical
ID-level bootstrap for confidence intervals. Includes epidemiological summaries
(person-time, incidence rates, VE) and interactive Plotly survival plots.

Features:
- Unique ID per subject for bootstrap.
- Clone-aware ΔRMST estimation.
- Precomputed daily events and GLM matrices for speed.
- Parallelized bootstrap with safe logging.
- Numerical stability for hazards and survival curves.
"""

# ---------------------- User-Defined Parameters ----------------------

AGE = 70

# Input / Output paths
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) HERNAN_poold_logistics_RMST\AC) HERNAN_poold_logistics_RMST_SIM")
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) HERNAN_poold_logistics_RMST\AC) HERNAN_poold_logistics_RMST")

STUDY_START = pd.Timestamp("2020-01-01")
AGE_REF_YEAR = 2023

# Bootstrap parameters
N_BOOT = 25                  # ≥200 for stable CI
BOOT_SUBSAMPLE = 0.5          # Fraction of IDs per replicate
RANDOM_SEED = 12345
SAFETY_BUFFER = 30            # Days trimmed from study end
N_CORES = -1                  # Number of parallel cores (-1 = all cores)

# Optional verbose logging
VERBOSE = False

# Seed RNGs for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ======================================================================
# Logging setup
# ======================================================================
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.txt"

def log(msg: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")

# ======================================================================
# Helper functions
# ======================================================================

def compute_rmst(times: np.ndarray, surv: np.ndarray) -> float:
    """Compute RMST using Simpson integration."""
    return float(simpson(surv, x=np.asarray(times))) if len(times) > 1 else 0.0

def summarize_rates(pt_u, ev_u, pt_v, ev_v):
    """Compute crude incidence rates, rate ratio, and VE."""
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
    """Vectorized daily events and at-risk counts."""
    window = int(end_day - start_day)
    if window <= 0:
        return np.zeros(0, int), np.zeros(0, int)

    events = np.zeros(window, int)
    diff = np.zeros(window + 1, int)

    starts_raw = pd.to_numeric(df["start"], errors="coerce").to_numpy()
    stops_raw = pd.to_numeric(df["stop"], errors="coerce").to_numpy()
    starts_raw = np.where(np.isnan(starts_raw), start_day, starts_raw)
    stops_raw = np.where(np.isnan(stops_raw), start_day, stops_raw)

    starts_idx = np.clip((starts_raw - start_day).astype(int), 0, window)
    stops_idx  = np.clip((stops_raw - start_day).astype(int), 0, window)

    if starts_idx.size > 0:
        np.add.at(diff, starts_idx, 1)
        np.add.at(diff, stops_idx, -1)
    risk = np.cumsum(diff)[:window]

    event_mask = (df["event"].to_numpy() == 1)
    if event_mask.any():
        event_indices = stops_idx[event_mask] - 1
        valid = (event_indices >= 0) & (event_indices < window)
        if np.any(valid):
            np.add.at(events, event_indices[valid], 1)

    return events, risk

# ======================================================================
# Load and prepare data
# ======================================================================

log(f"Loading CSV: {INPUT}")
raw = pd.read_csv(INPUT, dtype=str)
raw.columns = raw.columns.str.strip()

date_cols = ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]
for c in date_cols: raw[c] = pd.to_datetime(raw[c], errors="coerce")

raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
raw = raw[raw["age"] == AGE].copy()
if raw.empty: raise SystemExit(f"No subjects for AGE={AGE}")
log(f"Subjects after age filter: {raw.shape[0]}")

raw = raw.reset_index(drop=True)
raw["subject_id"] = np.arange(len(raw))

dose_cols = [c for c in raw.columns if c.startswith("Datum_") and c != "DatumUmrti"]
last_dose = raw[dose_cols].max(axis=1, skipna=True).max(skipna=True) if dose_cols else pd.NaT
last_death = raw["DatumUmrti"].max(skipna=True)
EXO_END_DAY = min([(d - STUDY_START).days for d in [last_dose, last_death] if pd.notna(d)]) - SAFETY_BUFFER
FIRST_VAX_DAY = (raw["Datum_1"].min(skipna=True) - STUDY_START).days
LAST_OBS_DAY = EXO_END_DAY
time_days = np.arange(0, LAST_OBS_DAY - FIRST_VAX_DAY)

raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
raw["first_dose_day"] = (raw["Datum_1"] - STUDY_START).dt.days

# ======================================================================
# Clone creation for Hernán-style analysis
# ======================================================================

clones = []
for _, r in raw.iterrows():
    death, first = r["death_day"], r["first_dose_day"]

    # Unvaccinated arm
    start_u = FIRST_VAX_DAY
    stop_u = min(first if pd.notna(first) else LAST_OBS_DAY,
                 death if pd.notna(death) else LAST_OBS_DAY,
                 LAST_OBS_DAY)
    if stop_u > start_u:
        clones.append((r["subject_id"], 0, start_u, stop_u, int(pd.notna(death) and death <= stop_u)))

    # Vaccinated arm
    if pd.notna(first):
        start_v = max(first, FIRST_VAX_DAY)
        stop_v = min(death if pd.notna(death) else LAST_OBS_DAY, LAST_OBS_DAY)
        if stop_v > start_v:
            clones.append((r["subject_id"], 1, start_v, stop_v, int(pd.notna(death) and death >= start_v and death <= stop_v)))

clones_df = pd.DataFrame(clones, columns=["id", "vaccinated", "start", "stop", "event"])

# ======================================================================
# Aggregate & GLM for initial RMST
# ======================================================================

ev_v, risk_v = compute_daily_events(clones_df[clones_df["vaccinated"]==1], FIRST_VAX_DAY, LAST_OBS_DAY)
ev_u, risk_u = compute_daily_events(clones_df[clones_df["vaccinated"]==0], FIRST_VAX_DAY, LAST_OBS_DAY)

agg_df = pd.DataFrame({
    "day": np.concatenate([time_days, time_days]),
    "vaccinated": np.concatenate([np.ones_like(time_days), np.zeros_like(time_days)]),
    "events": np.concatenate([ev_v, ev_u]),
    "at_risk": np.concatenate([risk_v, risk_u])
})

def fit_pooled_logistic(agg: pd.DataFrame):
    df = agg[agg["at_risk"] > 0].copy()
    p = (df["events"] / df["at_risk"]).clip(1e-9, 1-1e-9)
    w = df["at_risk"].astype(float)
    t_dummies = pd.get_dummies(df["day"].astype("category"), prefix="t", drop_first=True).astype(float)
    X = pd.concat([t_dummies, df[["vaccinated"]].astype(float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    glm = sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w)
    return glm.fit(method="newton", maxiter=200), list(X.columns)

res, cols = fit_pooled_logistic(agg_df)
def predict_survival(res, cols, time_days, vacc):
    Xp = pd.get_dummies(pd.Series(time_days).astype("category"), prefix="t", drop_first=True).astype(float)
    Xp["vaccinated"] = float(vacc)
    Xp = sm.add_constant(Xp, has_constant="add")
    for c in cols:
        if c not in Xp: Xp[c] = 0.0
    Xp = Xp[cols].astype(float)
    hazard = expit(Xp @ res.params[cols])
    return np.cumprod(1 - hazard)

S_v = predict_survival(res, cols, time_days, 1)
S_u = predict_survival(res, cols, time_days, 0)
delta_rmst = compute_rmst(time_days, S_v) - compute_rmst(time_days, S_u)
log(f"ΔRMST (Vaccinated – Unvaccinated): {delta_rmst:.2f} days")

# ======================================================================
# Clone-Aware Bootstrap for ΔRMST (clone-aware, Fast)
# ======================================================================

# Precompute daily events per ID
id_groups = {pid: df for pid, df in clones_df.groupby("id")}
ids = np.array(list(id_groups.keys()))
log(f"Unique IDs in clones_df: {len(ids)}")

precomputed = {
    pid: {arm: compute_daily_events(df[df["vaccinated"] == arm], FIRST_VAX_DAY, LAST_OBS_DAY)
          for arm in (0,1)}
    for pid, df in id_groups.items()
}

base_cols = pd.get_dummies(pd.Categorical(time_days), prefix="t", drop_first=True).columns.tolist()
full_cols = ["const"] + base_cols + ["vaccinated"]

Xp_base = (
    pd.get_dummies(pd.Series(time_days).astype("category"), prefix="t", drop_first=True)
    .astype(np.float32)
    .pipe(lambda X: sm.add_constant(X, has_constant="add"))
    .reindex(columns=full_cols, fill_value=0.0)
)

def fit_glm(events, risk, days, vaccinated):
    mask = risk > 0
    if not mask.any():
        return None
    p = (events[mask] / risk[mask]).clip(1e-9, 1 - 1e-9)
    w = risk[mask].astype(np.float32)

    X_time = pd.get_dummies(pd.Series(days[mask]).astype("category"),
                            prefix="t", drop_first=True).astype(np.float32)
    for col in base_cols:
        if col not in X_time.columns:
            X_time[col] = 0.0

    X = (
        pd.concat([X_time.reset_index(drop=True),
                   pd.Series(vaccinated[mask], name="vaccinated")], axis=1)
        .pipe(lambda X: sm.add_constant(X, has_constant="add"))
        .reindex(columns=full_cols, fill_value=0.0)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w).fit(
            method="lbfgs", maxiter=200
        )

def survival_curve(model, vacc_val):
    Xp = Xp_base.copy()
    Xp["vaccinated"] = float(vacc_val)
    hazard = expit(Xp @ model.params)
    hazard = np.clip(hazard, 1e-12, 1 - 1e-12)
    return np.cumprod(1.0 - hazard)

def bootstrap_once(i, rng):
    sample_ids = rng.choice(ids, max(int(len(ids) * BOOT_SUBSAMPLE), 1), replace=True)
    L = len(time_days)

    ev_vb = np.zeros(L, dtype=int); risk_vb = np.zeros(L, dtype=int)
    ev_ub = np.zeros(L, dtype=int); risk_ub = np.zeros(L, dtype=int)

    for sid in sample_ids:
        ev0, risk0 = precomputed[sid][0]
        ev1, risk1 = precomputed[sid][1]
        ev_ub += ev0; risk_ub += risk0
        ev_vb += ev1; risk_vb += risk1

    if risk_vb.sum() == 0 or risk_ub.sum() == 0:
        log(f"Replicate {i}: empty arm detected")
        return np.nan

    days = np.concatenate([time_days, time_days])
    vaccinated = np.concatenate([np.ones(L), np.zeros(L)])
    events = np.concatenate([ev_vb, ev_ub])
    risk = np.concatenate([risk_vb, risk_ub])

    try:
        model = fit_glm(events, risk, days, vaccinated)
        if model is None:
            return np.nan
        #if i < 5:
            #log(f"Replicate {i} convergence info: {model.mle_retvals}")

    except Exception as e:
        log(f"Replicate {i} failed: {e}")
        return np.nan

    S_vb = survival_curve(model, 1.0)
    S_ub = survival_curve(model, 0.0)
    rmst_diff = compute_rmst(time_days, S_vb) - compute_rmst(time_days, S_ub)
    return rmst_diff if not np.isnan(rmst_diff) else np.nan

def run_bootstrap():
    rng = np.random.RandomState(RANDOM_SEED)
    results = Parallel(n_jobs=N_CORES, backend="threading")(
        delayed(bootstrap_once)(i, rng) for i in range(N_BOOT)
    )
    vals = np.array(results, dtype=float)
    valid = vals[~np.isnan(vals)]

    if valid.size > 0:
        log(f"Bootstrap ΔRMST stats: min={valid.min():.3f}, max={valid.max():.3f}, std={valid.std():.4f}")

    failed = [i for i, v in enumerate(results) if np.isnan(v)]
    log(f"Bootstrap replicates failed: {len(failed)}")
    if failed:
        log(f"Failed replicate indices: {failed}")

    if valid.size >= 20:
        ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan
    log(f"ΔRMST 95% CI (percentile): [{ci_low:.2f}, {ci_high:.2f}]")

    return valid, (ci_low, ci_high), failed

boot_vals, (ci_low, ci_high), failed_reps = run_bootstrap()

# ======================================================================
# Epidemiology: Person-Time and Crude VE
# ======================================================================

clones_df["pt"] = clones_df["stop"] - clones_df["start"]
pt_u = clones_df.loc[clones_df["vaccinated"]==0,"pt"].sum()
pt_v = clones_df.loc[clones_df["vaccinated"]==1,"pt"].sum()
ev_u = clones_df.loc[clones_df["vaccinated"]==0,"event"].sum()
ev_v = clones_df.loc[clones_df["vaccinated"]==1,"event"].sum()
epi = summarize_rates(pt_u, ev_u, pt_v, ev_v)

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
