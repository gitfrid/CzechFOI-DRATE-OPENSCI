#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Target Trial Emulation of a Dynamic Vaccination Strategy
using Pooled Logistic Regression and RMST

This script emulates a two-arm target trial following
Hernán & Robins (2008, 2020):

Dynamic strategies:
    A = 0: Never vaccinate
    A = 1: Dynamic vaccination strategy: for each individual,
           treatment starts on their actual first-dose date;
           prior to that day, they are considered unvaccinated.

Key features:
    • Clone-based emulation with artificial censoring (intervals: (start, stop])
    • Daily discrete-time hazard modeling via pooled logistic regression
    • Natural cubic spline for baseline hazard
    • Time-varying treatment effects (spline × vaccination)
    • Standardized survival curves S_v(t), S_u(t)
    • Restricted mean survival time:
          RMST_v(t) = ∫₀ᵗ S_v(u) du
          RMST_u(t) = ∫₀ᵗ S_u(u) du
          ΔRMST(t) = RMST_v(t) − RMST_u(t)
      Vaccine effectiveness at τ:
          VE(τ) = 1 − CI_v(τ)/CI_u(τ)
      Number needed to treat per life-year:
          NNT_year = 365 / ΔRMST(τ)
    • Subject-level cluster bootstrap for uncertainty (95% percentile CIs)
      CIs for ΔRMST(t) and survival curves are pointwise 95% bootstrap intervals

Note:
    No confounders are included by design. This implementation
    is intended for methodological illustration or simulation.

Author: AI / drifting Date: 2025-12 Version: 1.0
"""

from __future__ import annotations
import random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.integrate import simpson
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import plotly.graph_objects as go
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ===================== PARAMETERS =====================

AGE = 70
AGE_REF_YEAR = 2023
STUDY_START = pd.Timestamp("2020-01-01")

# Input / Output paths (adapt to your local environment)

# real world Czech-FOI data
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) hernan_style_poold_logistics_RMST\AC) hernan_style_poold_logistics_RMST")

# simulated dataset HR=1 with simulated real dose schedule (sensitivity test if the used methode is bias free - null hypothesis)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) hernan_style_poold_logistics_RMST\AC) hernan_style_poold_logistics_RMST_SIM")

# real data with hypothetical 5% uvx deaths reclassified as vx (sensitivity test for missclassifcation)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_PTC1_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) hernan_style_poold_logistics_RMST\AC) hernan_style_poold_logistics_RMST_RECLASSIFIED")

N_BOOT = 30               # number of bootstrap replications
BOOT_SUBSAMPLE = 0.4      # fraction of subjects per bootstrap sample
RANDOM_SEED = 12345
N_CORES = 4               # set to -1 to use all cores,
SAFETY_BUFFER = 30        # days cut-off to avoid sparse tail
TIME_DF = 4               # spline degrees of freedom for time

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}.txt"

def log(msg: str):
    """Log message to console and to text file."""
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ===================== HELPERS =====================

def compute_rmst(t, S):
    """Compute RMST up to max(t) using Simpson integration."""
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    return float(simpson(S, x=t)) if len(t) > 1 else 0.0

def rmst_curve(S, t):
    """Compute RMST(t) curve over time."""
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    return np.array([simpson(S[:i+1], x=t[:i+1]) for i in range(len(t))])

def compute_daily_events(df, start_day, end_day):
    """
    Aggregate clone intervals into daily events and at-risk counts.
    Returns events and risk arrays.
    """
    window = int(end_day - start_day)
    events = np.zeros(window, int)
    diff = np.zeros(window + 1, int)

    starts = df["start"].to_numpy(dtype=float)
    stops = df["stop"].to_numpy(dtype=float)

    si = np.clip((starts - start_day).astype(int), 0, window)
    ei = np.clip((stops - start_day).astype(int), 0, window)

    np.add.at(diff, si, 1)
    np.add.at(diff, ei, -1)
    risk = np.cumsum(diff)[:window]

    ev_idx = ei[df["event"].to_numpy(dtype=int) == 1] - 1
    ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < window)]
    np.add.at(events, ev_idx, 1)

    return events, risk

def natural_cubic_spline_basis(x, df):
    """
    Construct natural cubic spline basis for time variable x
    for numeric stability and consistent dtype.
    """
    x = np.asarray(x, float)
    q = np.linspace(0, 1, df + 1)[1:-1]
    knots = np.quantile(x, q)
    kmin, kmax = x.min(), x.max()

    def d(z, k):
        return np.maximum(z - k, 0) ** 3

    cols = {"time_lin": x}
    for j, k in enumerate(knots, 1):
        cols[f"time_s{j}"] = (
            d(x, k)
            - d(x, kmax) * (kmax - k) / (kmax - kmin)
            + d(x, kmin) * (k - kmin) / (kmax - kmin)
        )
    return pd.DataFrame(cols).astype("float32")

# ===================== DATA PREP =====================

def load_raw(path):
    """
    Load raw CSV, filter by age, compute death_day and first_dose_day.
    """
    log(f"Loading CSV: {path}")
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

    # Parse date columns
    for c in ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")

    raw["Rok_narozeni"] = pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw["age"] = AGE_REF_YEAR - raw["Rok_narozeni"]
    raw = raw[raw["age"] == AGE].copy()
    raw.reset_index(drop=True, inplace=True)
    raw["subject_id"] = np.arange(len(raw))

    raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
    raw["first_dose_day"] = (raw["Datum_1"] - STUDY_START).dt.days

    log(f"Subjects after age filter: {len(raw)}")
    log(f"Deaths: {raw['death_day'].notna().sum()}, Vaccinated: {raw['first_dose_day'].notna().sum()}")
    return raw

def build_clones(raw):
    """
    Construct strategy-specific clones (A=0 and A=1).

    A=0: from FIRST_VAX_DAY until min(vaccination, death, last_obs).
    A=1: from first dose until min(death, last_obs). Only for vaccinated.
    """
    FIRST = int(raw.loc[raw["first_dose_day"].notna(), "first_dose_day"].min())
    last_obs = min(raw["death_day"].max(), raw["first_dose_day"].max()) - SAFETY_BUFFER
    last_obs = float(last_obs)
    window = int(last_obs - FIRST)
    if window <= 0:
        raise ValueError(f"No follow-up window: FIRST={FIRST}, last_obs={last_obs}")

    t = np.arange(window, dtype=int)

    clones = []
    for _, r in raw.iterrows():
        sid = int(r["subject_id"])
        d = float(r["death_day"]) if pd.notna(r["death_day"]) else np.nan
        f = float(r["first_dose_day"]) if pd.notna(r["first_dose_day"]) else np.nan

        # Strategy A = 0
        su = FIRST
        eu_candidates = [x for x in [f, d, last_obs] if not np.isnan(x)]
        if eu_candidates:
            eu = min(eu_candidates)
            if eu > su:
                clones.append((sid, 0, su, eu, int(not np.isnan(d) and d <= eu)))

        # Strategy A = 1
        if not np.isnan(f):
            sv = max(f, FIRST)
            ev = min(d if not np.isnan(d) else last_obs, last_obs)
            if ev > sv:
                clones.append((sid, 1, sv, ev, int(not np.isnan(d) and sv <= d <= ev)))

    df = pd.DataFrame(clones, columns=["id", "vaccinated", "start", "stop", "event"])
    df = df.astype({"id": "int32", "vaccinated": "int32", "event": "int32"})

    log(f"Total clone intervals: {len(df)}")
    log(f"  Vaccinated intervals:   {int((df['vaccinated'] == 1).sum())}")
    log(f"  Unvaccinated intervals: {int((df['vaccinated'] == 0).sum())}")
    log(f"FIRST_VAX_DAY: {FIRST}")
    log(f"last_obs: {last_obs}")
    log(f"Analysis window length: {len(t)} days")

    return df, t, FIRST, last_obs

def aggregate(clones, t, FIRST, last_obs):
    """
    Daily aggregation by vaccination arm.
    """
    ev_v, r_v = compute_daily_events(clones[clones.vaccinated == 1], FIRST, last_obs)
    ev_u, r_u = compute_daily_events(clones[clones.vaccinated == 0], FIRST, last_obs)

    # Basic epidemiological summaries
    pt_v = r_v.sum()
    pt_u = r_u.sum()
    e_v = ev_v.sum()
    e_u = ev_u.sum()
    rate_v = e_v / pt_v * 100_000 if pt_v > 0 else np.nan
    rate_u = e_u / pt_u * 100_000 if pt_u > 0 else np.nan

    log(f"Person-time (vaccinated):   {pt_v:,} person-days")
    log(f"Person-time (unvaccinated): {pt_u:,} person-days")
    log(f"Deaths (vaccinated):        {e_v}")
    log(f"Deaths (unvaccinated):      {e_u}")
    log(f"Crude mortality rate V: {rate_v:.2f} per 100,000 person-days" if pt_v > 0 else "Crude rate V: NA")
    log(f"Crude mortality rate U: {rate_u:.2f} per 100,000 person-days" if pt_u > 0 else "Crude rate U: NA")

    agg = pd.DataFrame({
        "day": np.concatenate([t, t]),
        "vaccinated": np.concatenate([np.ones_like(t), np.zeros_like(t)]),
        "events": np.concatenate([ev_v, ev_u]),
        "risk": np.concatenate([r_v, r_u]),
    })
    return agg

# ===================== MODEL FITTING =====================

def fit_plr(agg, spline_by_day, log_details: bool = True):
    """
    Fit pooled logistic regression with spline-based time and time×treatment effects.

    Outcome: daily hazard (events/risk) with freq_weights = risk.
    """
    df = agg[agg.risk > 0].copy()
    p = (df.events / df.risk).clip(1e-9, 1 - 1e-9).astype("float32")
    w = df.risk.astype("float32")
    df["vaccinated"] = df["vaccinated"].astype("float32")

    S = spline_by_day.loc[df.day].reset_index(drop=True)
    X = S.copy()
    X["vaccinated"] = df["vaccinated"]

    # Interaction terms: spline × vaccination
    inter = S.mul(df["vaccinated"].to_numpy(dtype=np.float32)[:, None])
    inter.columns = [f"{c}_x_vacc" for c in inter.columns]

    X = pd.concat([X, inter], axis=1)
    X.insert(0, "const", 1.0)
    X = X.astype("float32")

    if log_details:
        log(f"GLM rows used: {len(X)}; predictors: {X.shape[1]}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w).fit()

    if log_details:
        log(f"GLM converged: {m.converged}")
    return m

def predict_S(m, t, A, spline_by_day):
    """
    Predict survival curve S(t | A) from fitted pooled logistic regression.
    """
    S = spline_by_day.loc[t].reset_index(drop=True)
    X = S.copy()
    X["vaccinated"] = float(A)

    inter = S.mul(float(A))
    inter.columns = [f"{c}_x_vacc" for c in inter.columns]

    X = pd.concat([X, inter], axis=1)
    X.insert(0, "const", 1.0)
    X = X.reindex(columns=m.params.index, fill_value=0.0).astype("float32")

    haz = expit(X.to_numpy(dtype=np.float32) @ m.params.to_numpy(dtype=np.float32))
    haz = np.clip(haz, 1e-9, 1 - 1e-9)
    S_t = np.cumprod(1 - haz)
    return S_t

# ===================== BOOTSTRAP HELPERS =====================

def precompute_bootstrap(clones, t, FIRST, last_obs):
    """
    Precompute per-ID daily events and risk for bootstrap efficiency.
    """
    ids = clones.id.unique()
    groups = {i: g for i, g in clones.groupby("id")}
    pre = {
        i: {
            a: compute_daily_events(groups[i][groups[i].vaccinated == a], FIRST, last_obs)
            for a in (0, 1)
        }
        for i in ids
    }
    return ids, pre

def bootstrap_once(i, ids, pre, t, spline_by_day):
    """
    Single bootstrap replicate:
      - Resample subjects with replacement (subsample size = BOOT_SUBSAMPLE * n_ids)
      - Aggregate daily events and risk
      - Refit GLM and compute S_v(t), S_u(t)
    """
    rng = np.random.default_rng(RANDOM_SEED + i)
    samp = rng.choice(ids, int(len(ids) * BOOT_SUBSAMPLE), replace=True)

    ev_v = np.zeros_like(t, dtype=int)
    ev_u = np.zeros_like(t, dtype=int)
    r_v = np.zeros_like(t, dtype=int)
    r_u = np.zeros_like(t, dtype=int)

    for sid in samp:
        e0, r0 = pre[sid][0]
        e1, r1 = pre[sid][1]
        ev_u += e0
        r_u += r0
        ev_v += e1
        r_v += r1

    agg_b = pd.DataFrame({
        "day": np.concatenate([t, t]),
        "vaccinated": np.concatenate([np.ones_like(t), np.zeros_like(t)]),
        "events": np.concatenate([ev_v, ev_u]),
        "risk": np.concatenate([r_v, r_u]),
    })

    try:
        m = fit_plr(agg_b, spline_by_day, log_details=False)
        Sv_b = predict_S(m, t, 1, spline_by_day)
        Su_b = predict_S(m, t, 0, spline_by_day)
        return Sv_b, Su_b
    except Exception:
        return None

# ===================== MAIN EXECUTION =====================

def main():
    # 1) Load and preprocess raw data
    raw = load_raw(INPUT)

    # 2) Build strategy-specific clones and analysis time grid
    clones, t, FIRST, last_obs = build_clones(raw)

    # 3) Aggregate daily risk and events by arm
    agg = aggregate(clones, t, FIRST, last_obs)

    # 4) Construct spline basis for time
    spline_by_day = natural_cubic_spline_basis(t, TIME_DF)
    spline_by_day["day"] = t
    spline_by_day.set_index("day", inplace=True)

    # 5) Fit pooled logistic regression model
    m = fit_plr(agg, spline_by_day, log_details=True)

    # 6) Predict survival curves under A=1 and A=0
    Sv = predict_S(m, t, 1, spline_by_day)
    Su = predict_S(m, t, 0, spline_by_day)

    # 7) Compute RMST curves and ΔRMST(t)
    RMST_v_t = rmst_curve(Sv, t)
    RMST_u_t = rmst_curve(Su, t)
    Delta_t = RMST_v_t - RMST_u_t

    tau = t[-1]
    RMST_v_tau = RMST_v_t[-1]
    RMST_u_tau = RMST_u_t[-1]
    Delta_tau = Delta_t[-1]

    log(f"Final-time RMST_v(τ={tau}): {RMST_v_tau:.2f}")
    log(f"Final-time RMST_u(τ={tau}): {RMST_u_tau:.2f}")
    log(f"Final-time ΔRMST(τ={tau}):  {Delta_tau:.2f} days")

    # 8) Compute VE(τ)
    Sv_tau = Sv[-1]
    Su_tau = Su[-1]
    CI_v_tau = 1 - Sv_tau
    CI_u_tau = 1 - Su_tau
    VE_tau = np.nan if CI_u_tau == 0 else 1 - (CI_v_tau / CI_u_tau)

    log(f"Survival at τ={tau}: S_v={Sv_tau:.4f}, S_u={Su_tau:.4f}")
    log(f"Cumulative incidence at τ={tau}: CI_v={CI_v_tau:.4f}, CI_u={CI_u_tau:.4f}")
    log(f"Vaccine effectiveness VE(τ={tau}): {VE_tau:+.3%}" if not np.isnan(VE_tau) else "VE(τ): NA")

    # 9) NNT per life-year gained
    if Delta_tau > 0:
        NNT_year = 365.0 / Delta_tau
        log(f"NNT per life-year gained (τ={tau}): {NNT_year:.2f}")
    else:
        NNT_year = np.nan
        log("NNT per life-year gained: not defined (ΔRMST ≤ 0).")

    # 10) Bootstrap: subject-level resampling
    ids, pre = precompute_bootstrap(clones, t, FIRST, last_obs)

    boot_surv = []
    log(f"Starting bootstrap ({N_BOOT} replications, subsample={BOOT_SUBSAMPLE:.2f})")
    with tqdm_joblib(tqdm(total=N_BOOT, desc="Bootstrap", mininterval=0.2)):
        boot_raw = Parallel(n_jobs=N_CORES, backend="loky")(
            delayed(bootstrap_once)(i, ids, pre, t, spline_by_day) for i in range(N_BOOT)
        )

    for b in boot_raw:
        if b is not None:
            boot_surv.append(b)

    n_success = len(boot_surv)
    log(f"Bootstrap successful replicates: {n_success}/{N_BOOT}")
    if n_success == 0:
        log("No valid bootstrap replicates; aborting CI computation.")
        return

    boot_Sv = np.array([Sv_b for Sv_b, Su_b in boot_surv])
    boot_Su = np.array([Su_b for Sv_b, Su_b in boot_surv])

    # 11) Compute bootstrap RMST(t) and ΔRMST(t)
    boot_RMST_v = np.array([rmst_curve(Sv_b, t) for Sv_b in boot_Sv])
    boot_RMST_u = np.array([rmst_curve(Su_b, t) for Su_b in boot_Su])
    boot_Delta_t = boot_RMST_v - boot_RMST_u

    # Pointwise bootstrap at τ
    Delta_tau_boot = boot_Delta_t[:, -1]
    RMST_v_tau_boot = boot_RMST_v[:, -1]
    RMST_u_tau_boot = boot_RMST_u[:, -1]

    Sv_tau_boot = boot_Sv[:, -1]
    Su_tau_boot = boot_Su[:, -1]
    CI_v_tau_boot = 1 - Sv_tau_boot
    CI_u_tau_boot = 1 - Su_tau_boot
    VE_tau_boot = np.where(CI_u_tau_boot == 0, np.nan, 1 - CI_v_tau_boot / CI_u_tau_boot)
    NNT_year_boot = np.where(Delta_tau_boot > 0, 365.0 / Delta_tau_boot, np.nan)

    # 12) Compute 95% percentile CIs
    Delta_lo_tau, Delta_hi_tau = np.nanpercentile(Delta_tau_boot, [2.5, 97.5])
    VE_lo_tau, VE_hi_tau = np.nanpercentile(
        VE_tau_boot[~np.isnan(VE_tau_boot)], [2.5, 97.5]
    ) if np.any(~np.isnan(VE_tau_boot)) else (np.nan, np.nan)
    if np.any(~np.isnan(NNT_year_boot)):
        NNT_lo, NNT_hi = np.nanpercentile(NNT_year_boot[~np.isnan(NNT_year_boot)], [2.5, 97.5])
    else:
        NNT_lo, NNT_hi = (np.nan, np.nan)

    log(f"ΔRMST(τ={tau}) 95% CI: [{Delta_lo_tau:.2f}, {Delta_hi_tau:.2f}] days")
    log(f"VE(τ={tau}) 95% CI:     [{VE_lo_tau:.3%}, {VE_hi_tau:.3%}]" if not np.isnan(VE_lo_tau) else "VE(τ) CI: NA")
    if not np.isnan(NNT_lo):
        log(f"NNT per life-year gained 95% CI: [{NNT_lo:.2f}, {NNT_hi:.2f}]")
    else:
        log("NNT CI: NA")

    # 13) Pointwise bootstrap CIs for curves
    Delta_lo, Delta_hi = np.percentile(boot_Delta_t, [2.5, 97.5], axis=0)
    RMST_v_lo, RMST_v_hi = np.percentile(boot_RMST_v, [2.5, 97.5], axis=0)
    RMST_u_lo, RMST_u_hi = np.percentile(boot_RMST_u, [2.5, 97.5], axis=0)
    Sv_lo, Sv_hi = np.percentile(boot_Sv, [2.5, 97.5], axis=0)
    Su_lo, Su_hi = np.percentile(boot_Su, [2.5, 97.5], axis=0)

    # ===================== PLOT 1: ΔRMST(t) =====================

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=t, y=Delta_hi, line=dict(width=0), showlegend=False))
    fig_delta.add_trace(go.Scatter(
        x=t, y=Delta_lo, fill="tonexty",
        fillcolor="rgba(0, 100, 200, 0.2)",
        line=dict(width=0), showlegend=False
    ))
    fig_delta.add_trace(go.Scatter(
        x=t, y=Delta_t,
        mode="lines",
        line=dict(color="black", width=2),
        name="ΔRMST(t)"
    ))
    fig_delta.add_hline(y=0, line=dict(color="gray", dash="dash"))

    fig_delta.update_layout(
        title="Cumulative difference in restricted mean survival time, ΔRMST(t)",
        xaxis_title="Time since index (days)",
        yaxis_title="ΔRMST(t) in days",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60)
    )

    # τ-specific annotation for ΔRMST(τ) with CI
    fig_delta.add_annotation(
        x=tau,
        y=Delta_tau,
        text=(
            f"ΔRMST(τ={tau}) = {Delta_tau:.2f} days<br>"
            f"95% CI [{Delta_lo_tau:.2f}, {Delta_hi_tau:.2f}]"
        ),
        showarrow=True,
        arrowhead=2,
        ax=-40,
        ay=-40,
        bgcolor="white"
    )

    delta_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}_DeltaRMST_t.html"
    fig_delta.write_html(delta_path)

    caption_delta = (
        "Figure 1. Cumulative difference in restricted mean survival time, ΔRMST(t), "
        "between vaccination and no-vaccination strategies over follow-up time. "
        "The solid line shows the point estimate from a pooled logistic regression "
        "with spline-based time-varying effects; the shaded area denotes the 95% "
        "bootstrap confidence band based on cluster resampling of individuals. "
        "The horizontal dashed line at zero corresponds to no difference in RMST."
    )
    log(caption_delta)

    # ===================== PLOT 2: RMST_v(t), RMST_u(t) =====================

    fig_rmst = go.Figure()
    fig_rmst.add_trace(go.Scatter(x=t, y=RMST_v_hi, line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(
        x=t, y=RMST_v_lo, fill="tonexty",
        fillcolor="rgba(0, 150, 0, 0.2)",
        line=dict(width=0), showlegend=False
    ))
    fig_rmst.add_trace(go.Scatter(x=t, y=RMST_u_hi, line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(
        x=t, y=RMST_u_lo, fill="tonexty",
        fillcolor="rgba(200, 0, 0, 0.2)",
        line=dict(width=0), showlegend=False
    ))

    fig_rmst.add_trace(go.Scatter(
        x=t, y=RMST_v_t,
        mode="lines",
        line=dict(color="green", width=2),
        name="RMST_v(t)"
    ))
    fig_rmst.add_trace(go.Scatter(
        x=t, y=RMST_u_t,
        mode="lines",
        line=dict(color="red", width=2),
        name="RMST_u(t)"
    ))

    fig_rmst.update_layout(
        title="Restricted mean survival time curves, by vaccination strategy",
        xaxis_title="Time since index (days)",
        yaxis_title="RMST(t) in days",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60)
    )

    rmst_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}_RMST_vu_t.html"
    fig_rmst.write_html(rmst_path)

    caption_rmst = (
        "Figure 2. Restricted mean survival time curves, RMST_v(t) and RMST_u(t), "
        "for vaccination and no-vaccination strategies. Solid lines denote the "
        "point estimates obtained from the pooled logistic model; shaded bands "
        "represent 95% bootstrap confidence intervals based on cluster resampling. "
        "These curves describe the cumulative expected survival time under each "
        "strategy, without directly displaying their difference."
    )
    log(caption_rmst)

    # ===================== PLOT 3: SURVIVAL CURVES S_v(t), S_u(t) =====================

    fig_surv = go.Figure()

    # Vaccinated CI
    fig_surv.add_trace(go.Scatter(x=t, y=Sv_hi, line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(
        x=t, y=Sv_lo, fill="tonexty",
        fillcolor="rgba(0, 150, 0, 0.2)",
        line=dict(width=0), showlegend=False
    ))

    # Unvaccinated CI
    fig_surv.add_trace(go.Scatter(x=t, y=Su_hi, line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(
        x=t, y=Su_lo, fill="tonexty",
        fillcolor="rgba(200, 0, 0, 0.2)",
        line=dict(width=0), showlegend=False
    ))

    # Point estimates
    fig_surv.add_trace(go.Scatter(
        x=t, y=Sv,
        mode="lines",
        line=dict(color="green", width=2),
        name="Vaccinated"
    ))
    fig_surv.add_trace(go.Scatter(
        x=t, y=Su,
        mode="lines",
        line=dict(color="red", width=2),
        name="Unvaccinated"
    ))

    fig_surv.update_layout(
        title="Standardized survival curves, by vaccination strategy",
        xaxis_title="Time since index (days)",
        yaxis_title="Survival probability",
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60)
    )

    surv_path = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}_Survival_CI.html"
    fig_surv.write_html(surv_path)

    caption_surv = (
        "Figure 3. Standardized survival curves for vaccination and no-vaccination "
        "strategies, obtained from a pooled logistic regression with spline-based "
        "time-varying effects. Solid lines show the estimated marginal survival "
        "probabilities under each strategy; shaded areas correspond to 95% "
        "bootstrap confidence bands with cluster resampling. These curves are "
        "descriptive and complement the causal contrast shown in Figure 1."
    )
    log(caption_surv)

    # ===================== FINAL SUMMARY BLOCK =====================

    log("Summary:")
    log(f"  ΔRMST(τ={tau} days) = {Delta_tau:.2f} days "
        f"[95% CI: {Delta_lo_tau:.2f}, {Delta_hi_tau:.2f}]")
    log(f"  VE(τ={tau} days)    = {VE_tau:+.1%} "
        f"[95% CI: {VE_lo_tau:.1%}, {VE_hi_tau:.1%}]" if not np.isnan(VE_lo_tau) else "  VE: NA")
    if not np.isnan(NNT_year):
        log(f"  NNT (per life-year) = {NNT_year:.2f} "
            f"[95% CI: {NNT_lo:.2f}, {NNT_hi:.2f}]")
    else:
        log("  NNT (per life-year): NA")

    log("Finished. Three HTML plots generated:")
    log(f"  1) ΔRMST(t):    {delta_path}")
    log(f"  2) RMST_v/u(t): {rmst_path}")
    log(f"  3) Survival CI: {surv_path}")

if __name__ == "__main__":
    main()
