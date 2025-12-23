#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical vs causal dynamic RMST comparison with Historical baseline

This script:

1) Loads raw CSV and filters for a specified age group.
2) Computes "pure historical" RMST curves (ΔRMST_History) without clone-censor.
3) Builds clone-censor data for dynamic vaccination regime:
   - A=0: never vaccinate
   - A=1: vaccinate on actual first-dose day + optional lag
4) Computes empirical dynamic RMST (ΔRMST_CC) using clone-censor, nonparametric hazards.
5) Computes causal dynamic RMST (ΔRMST_Causal, all vs none) using pooled logistic regression.
6) Computes mixed-world causal effect under observed coverage (ΔRMST_Causal_mix).
7) Computes ΔΔRMST decompositions:
   - ΔRMST_CC − ΔRMST_History → selection / immortal-time bias
   - ΔRMST_Causal − ΔRMST_CC → causal model contribution
8) Plots ΔRMST curves, ΔΔRMST decompositions, and survival curves (vaccinated/unvaccinated for all).
9) Saves unified CSV and clean log.

Author: AI / drifting Date: 2025-12 Vers: 1
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.integrate import simpson
import plotly.graph_objects as go

# ===================== CONFIG =====================
AGE = 70
AGE_REF_YEAR = 2023
STUDY_START = pd.Timestamp("2020-01-01")

SAFETY_BUFFER = 30  # tail cut-off
TIME_DF = 4         # spline df for pooled logistic
IMMUNITY_LAG = 0    # Optional: lag vaccinated start

# Input / Output paths

# real world Czech-FOI data
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AF) Empirical_vs_causal_comparison_RMST\AF) Empirical_vs_causal_comparison")

# simulated dataset HR=1 with simulated real dose schedule (sensitivity test if the used methode is bias free - null hypothesis)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AF) Empirical_vs_causal_comparison_RMST\AF) Empirical_vs_causal_comparison_SIM")

# real data with hypothetical 5% uvx deaths reclassified as vx (sensitivity test for missclassifcation)
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_PTC5_uvx_as_vx_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AF) Empirical_vs_causal_comparison_RMST\AF) Empirical_vs_causal_comparison_RECLASSIFIED")

OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.csv")
PLOT_DELTA_OUT = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_ΔRMST_AG{AGE}.html")
PLOT_DD_OUT = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_ΔΔRMST_AG{AGE}.html")
PLOT_RMST_OUT = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_RMST_AG{AGE}.html")
PLOT_SURV_OUT = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_Survival_AG{AGE}.html")
LOG_OUT = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.log")

# ===================== LOGGING =====================
def log(msg: str):
    """Write message to console and log file."""
    print(msg)
    with open(LOG_OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ===================== HELPERS =====================
def rmst_curve(S, t):
    """RMST(t) via Simpson integration."""
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    out = np.zeros_like(t, dtype=float)
    for i in range(len(t)):
        out[i] = simpson(S[:i+1], x=t[:i+1]) if i > 0 else 0.0
    return out

def compute_daily_events(df, start_day, end_day):
    """Daily events and at-risk counts."""
    window = int(end_day - start_day)
    events = np.zeros(window, int)
    diff = np.zeros(window + 1, int)
    starts = df["start"].to_numpy(float)
    stops = df["stop"].to_numpy(float)
    si = np.clip((starts - start_day).astype(int), 0, window)
    ei = np.clip((stops - start_day).astype(int), 0, window)
    np.add.at(diff, si, 1)
    np.add.at(diff, ei, -1)
    risk = np.cumsum(diff)[:window]
    ev_idx = ei[df["event"].to_numpy(int) == 1] - 1
    ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < window)]
    np.add.at(events, ev_idx, 1)
    return events, risk

def natural_cubic_spline_basis(x, df):
    """Natural cubic spline basis for pooled logistic."""
    x = np.asarray(x, float)
    q = np.linspace(0, 1, df + 1)[1:-1]
    knots = np.quantile(x, q)
    kmin, kmax = x.min(), x.max()
    def d(z, k): return np.maximum(z - k, 0) ** 3
    cols = {"time_lin": x}
    for j, k in enumerate(knots, 1):
        cols[f"time_s{j}"] = d(x, k) - d(x, kmax)*(kmax-k)/(kmax-kmin) + d(x, kmin)*(k-kmin)/(kmax-kmin)
    return pd.DataFrame(cols).astype("float32")

# ===================== DATA PREP =====================
def load_raw(path: Path):
    log(f"Loading raw data from: {path}")
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()
    for c in ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")
    raw["Rok_narozeni"] = pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw["age"] = AGE_REF_YEAR - raw["Rok_narozeni"]
    raw = raw[raw["age"] == AGE].copy()
    raw.reset_index(drop=True, inplace=True)
    raw["subject_id"] = np.arange(len(raw))
    raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
    raw["first_dose_day"] = (raw["Datum_1"] - STUDY_START).dt.days
    raw.loc[raw["death_day"] < 0, "death_day"] = np.nan
    raw.loc[raw["first_dose_day"] < 0, "first_dose_day"] = np.nan
    log(f"Subjects after age filter: {len(raw)}")
    log(f"  Deaths: {raw['death_day'].notna().sum()}")
    log(f"  Ever vaccinated (any time): {raw['first_dose_day'].notna().sum()}")
    return raw

def build_clones(raw: pd.DataFrame):
    """Construct clone-censor intervals for dynamic vaccination."""
    FIRST = int(raw.loc[raw["first_dose_day"].notna(), "first_dose_day"].min())
    last_obs = min(raw["death_day"].max(), raw["first_dose_day"].max()) - SAFETY_BUFFER
    last_obs = float(last_obs)
    t = np.arange(int(last_obs - FIRST), dtype=int)
    clones = []
    for _, r in raw.iterrows():
        sid = int(r["subject_id"])
        d = float(r["death_day"]) if pd.notna(r["death_day"]) else np.nan
        f = float(r["first_dose_day"]) if pd.notna(r["first_dose_day"]) else np.nan
        # A=0 never vaccinate
        su = FIRST
        eu_candidates = [x for x in [f, d, last_obs] if not np.isnan(x)]
        if eu_candidates:
            eu = min(eu_candidates)
            if eu > su: clones.append((sid, 0, su, eu, int(not np.isnan(d) and d <= eu)))
        # A=1 vaccinate actual + lag
        if not np.isnan(f):
            sv = max(f + IMMUNITY_LAG, FIRST)
            ev = min(d if not np.isnan(d) else last_obs, last_obs)
            if ev > sv: clones.append((sid, 1, sv, ev, int(not np.isnan(d) and sv <= d <= ev)))
    df = pd.DataFrame(clones, columns=["id","vaccinated","start","stop","event"]).astype({"id":"int32","vaccinated":"int32","event":"int32"})
    log(f"Clone intervals constructed: {len(df)}")
    log(f"  Vaccinated intervals:   {(df['vaccinated']==1).sum()}")
    log(f"  Unvaccinated intervals: {(df['vaccinated']==0).sum()}")
    log(f"FIRST_VAX_DAY (obs_start): {FIRST}")
    log(f"last_obs (obs_end):       {last_obs}")
    log(f"Analysis window length (days): {len(t)}")
    log(f"Using IMMUNITY_LAG: {IMMUNITY_LAG} days")
    return df, t, FIRST, last_obs

# ===================== EMPIRICAL DYNAMIC RMST =====================
def empirical_dynamic_rmst(clones, t, FIRST, last_obs):
    ev_v, r_v = compute_daily_events(clones[clones.vaccinated==1], FIRST, last_obs)
    ev_u, r_u = compute_daily_events(clones[clones.vaccinated==0], FIRST, last_obs)
    haz_v = np.where(r_v>10, ev_v/r_v, 0.0)
    haz_u = np.where(r_u>10, ev_u/r_u, 0.0)
    S_v = np.cumprod(1.0 - haz_v)
    S_u = np.cumprod(1.0 - haz_u)
    RMST_v = rmst_curve(S_v, t)
    RMST_u = rmst_curve(S_u, t)
    Delta = RMST_v - RMST_u
    return S_v, S_u, RMST_v, RMST_u, Delta, ev_v, r_v, ev_u, r_u

# ===================== HISTORY RMST =====================
def history_rmst(raw, t, FIRST, last_obs):
    """Compute purely descriptive ΔRMST_History (no clone-censor)."""
    events_v = np.zeros_like(t)
    events_u = np.zeros_like(t)
    risk_v = np.zeros_like(t)
    risk_u = np.zeros_like(t)
    for i, day in enumerate(t):
        mask_v = raw["first_dose_day"].notna() & (raw["first_dose_day"] <= FIRST + day)
        mask_u = raw["first_dose_day"].isna() | (raw["first_dose_day"] > FIRST + day)
        risk_v[i] = mask_v.sum()
        risk_u[i] = mask_u.sum()
        events_v[i] = ((raw["death_day"] == FIRST + day) & mask_v).sum()
        events_u[i] = ((raw["death_day"] == FIRST + day) & mask_u).sum()
    haz_v = np.where(risk_v>0, events_v/risk_v, 0.0)
    haz_u = np.where(risk_u>0, events_u/risk_u, 0.0)
    S_v = np.cumprod(1.0 - haz_v)
    S_u = np.cumprod(1.0 - haz_u)
    RMST_v = rmst_curve(S_v, t)
    RMST_u = rmst_curve(S_u, t)
    Delta = RMST_v - RMST_u
    return S_v, S_u, RMST_v, RMST_u, Delta

# ===================== CAUSAL POOLED LOGISTIC =====================
def fit_plr(agg, spline_by_day):
    df = agg[agg.risk>0].copy()
    p = (df.events/df.risk).clip(1e-9,1-1e-9).astype("float32")
    w = df.risk.astype("float32")
    df["vaccinated"] = df["vaccinated"].astype("float32")
    S = spline_by_day.loc[df.day].reset_index(drop=True)
    X = S.copy()
    X["vaccinated"] = df["vaccinated"]
    inter = S.mul(df["vaccinated"].to_numpy(dtype=np.float32)[:, None])
    inter.columns = [f"{c}_x_vacc" for c in inter.columns]
    X = pd.concat([X, inter], axis=1)
    X.insert(0,"const",1.0)
    m = sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w).fit()
    return m

def predict_S_plr(m, t, A, spline_by_day):
    S = spline_by_day.loc[t].reset_index(drop=True)
    X = S.copy()
    X["vaccinated"] = float(A)
    inter = S.mul(float(A))
    inter.columns = [f"{c}_x_vacc" for c in inter.columns]
    X = pd.concat([X, inter], axis=1)
    X.insert(0,"const",1.0)
    X = X.reindex(columns=m.params.index, fill_value=0.0).astype("float32")
    haz = expit(X.to_numpy(dtype=np.float32) @ m.params.to_numpy(dtype=np.float32))
    S_t = np.cumprod(1.0 - np.clip(haz,1e-9,1-1e-9))
    return S_t

def causal_dynamic_rmst(ev_v, r_v, ev_u, r_u, t):
    agg = pd.DataFrame({
        "day": np.concatenate([t,t]),
        "vaccinated": np.concatenate([np.ones_like(t), np.zeros_like(t)]),
        "events": np.concatenate([ev_v, ev_u]),
        "risk": np.concatenate([r_v, r_u])
    })
    spline_by_day = natural_cubic_spline_basis(t, TIME_DF)
    spline_by_day["day"] = t
    spline_by_day.set_index("day", inplace=True)
    m = fit_plr(agg, spline_by_day)
    Sv = predict_S_plr(m, t, 1, spline_by_day)
    Su = predict_S_plr(m, t, 0, spline_by_day)
    RMST_v = rmst_curve(Sv, t)
    RMST_u = rmst_curve(Su, t)
    Delta = RMST_v - RMST_u
    return Sv, Su, RMST_v, RMST_u, Delta

# ===================== MAIN =====================
def main():
    if LOG_OUT.exists(): LOG_OUT.unlink()
    log(f"=== Empirical vs causal dynamic RMST comparison with History ===")
    log(f"Age group: {AGE}, Study start: {STUDY_START.date()}, Input: {INPUT}")

    raw = load_raw(INPUT)
    clones, t, FIRST, last_obs = build_clones(raw)

    # -------- Historical (pure descriptive) ----------
    S_v_h, S_u_h, RMST_v_h, RMST_u_h, Delta_h = history_rmst(raw, t, FIRST, last_obs)
    log("Computed ΔRMST_History (pure descriptive, no clone-censor)")

    # -------- Empirical CC ----------
    S_v_cc, S_u_cc, RMST_v_cc, RMST_u_cc, Delta_cc, ev_v, r_v, ev_u, r_u = empirical_dynamic_rmst(clones, t, FIRST, last_obs)
    log("Computed ΔRMST_CC (empirical clone-censor)")

    # -------- Causal ----------
    Sv_c, Su_c, RMST_v_c, RMST_u_c, Delta_c = causal_dynamic_rmst(ev_v, r_v, ev_u, r_u, t)
    log("Computed ΔRMST_Causal (100% vs 0%, causal model)")

    # -------- Mixed world ----------
    p = raw["first_dose_day"].notna().mean()
    RMST_v_mix = RMST_u_c + p*Delta_c
    RMST_u_mix = RMST_u_c.copy()
    Delta_mix = p*Delta_c
    S_v_mix = p*Sv_c + (1-p)*Su_c
    S_u_mix = Su_c.copy()
    log(f"Computed ΔRMST_Causal_mix (attributable under observed coverage p={p:.3f})")

    # -------- ΔΔRMST decomposition ----------
    Delta_cc_hist = Delta_cc - Delta_h
    Delta_c_cc = Delta_c - Delta_cc

    # -------- Save CSV ----------
    df = pd.DataFrame({
        "t": t,
        "Delta_History": Delta_h,
        "Delta_CC": Delta_cc,
        "Delta_Causal": Delta_c,
        "Delta_Causal_mix": Delta_mix,
        "Delta_CC_minus_History": Delta_cc_hist,
        "Delta_Causal_minus_CC": Delta_c_cc,
        "RMST_v_History": RMST_v_h,
        "RMST_u_History": RMST_u_h,
        "RMST_v_CC": RMST_v_cc,
        "RMST_u_CC": RMST_u_cc,
        "RMST_v_Causal": RMST_v_c,
        "RMST_u_Causal": RMST_u_c,
        "RMST_v_Mix": RMST_v_mix,
        "RMST_u_Mix": RMST_u_mix,
        "S_v_History": S_v_h,
        "S_u_History": S_u_h,
        "S_v_CC": S_v_cc,
        "S_u_CC": S_u_cc,
        "S_v_Causal": Sv_c,
        "S_u_Causal": Su_c,
        "S_v_Mix": S_v_mix,
        "S_u_Mix": S_u_mix
    })
    df.to_csv(CSV_OUT, index=False)
    log(f"All ΔRMST, RMST, survival curves saved to CSV: {CSV_OUT}")

    # -------- ΔRMST plot ----------
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=t, y=Delta_h, mode="lines", line=dict(color="black", width=2), name="ΔRMST_History"))
    fig_delta.add_trace(go.Scatter(x=t, y=Delta_cc, mode="lines", line=dict(color="blue", width=2, dash="dash"), name="ΔRMST_CC"))
    fig_delta.add_trace(go.Scatter(x=t, y=Delta_c, mode="lines", line=dict(color="cyan", width=2, dash="dot"), name="ΔRMST_Causal"))
    fig_delta.add_trace(go.Scatter(x=t, y=Delta_mix, mode="lines", line=dict(color="green", width=2), name="ΔRMST_Causal_mix"))
    fig_delta.update_layout(title="ΔRMST Comparison", xaxis_title="Days", yaxis_title="ΔRMST")
    fig_delta.write_html(PLOT_DELTA_OUT)
    log(f"ΔRMST plot saved: {PLOT_DELTA_OUT}")

    # -------- ΔΔRMST plot ----------
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=t, y=Delta_cc_hist, mode="lines", line=dict(color="purple", width=2), name="ΔRMST_CC - ΔRMST_History"))
    fig_dd.add_trace(go.Scatter(x=t, y=Delta_c_cc, mode="lines", line=dict(color="orange", width=2), name="ΔRMST_Causal - ΔRMST_CC"))
    fig_dd.update_layout(title="ΔΔRMST Decomposition", xaxis_title="Days", yaxis_title="ΔΔRMST")
    fig_dd.write_html(PLOT_DD_OUT)
    log(f"ΔΔRMST decomposition plot saved: {PLOT_DD_OUT}")

    # -------- Survival plot ----------
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(x=t, y=S_v_h, mode="lines", line=dict(color="black", width=2), name="S_v_History"))
    fig_surv.add_trace(go.Scatter(x=t, y=S_u_h, mode="lines", line=dict(color="gray", width=2), name="S_u_History"))
    fig_surv.add_trace(go.Scatter(x=t, y=S_v_cc, mode="lines", line=dict(color="blue", width=2, dash="dash"), name="S_v_CC"))
    fig_surv.add_trace(go.Scatter(x=t, y=S_u_cc, mode="lines", line=dict(color="red", width=2, dash="dash"), name="S_u_CC"))
    fig_surv.add_trace(go.Scatter(x=t, y=Sv_c, mode="lines", line=dict(color="cyan", width=2, dash="dot"), name="S_v_Causal"))
    fig_surv.add_trace(go.Scatter(x=t, y=Su_c, mode="lines", line=dict(color="magenta", width=2, dash="dot"), name="S_u_Causal"))
    fig_surv.add_trace(go.Scatter(x=t, y=S_v_mix, mode="lines", line=dict(color="green", width=2), name="S_v_Mix"))
    fig_surv.add_trace(go.Scatter(x=t, y=S_u_mix, mode="lines", line=dict(color="orange", width=2), name="S_u_Mix"))
    fig_surv.update_layout(title="Survival curves: Historical, Empirical CC, Causal, Mixed-world",
                           xaxis_title="Days since FIRST_VAX_DAY (obs_start)",
                           yaxis_title="Survival probability",
                           template="plotly_white")
    fig_surv.write_html(PLOT_SURV_OUT)
    log(f"Survival plot saved: {PLOT_SURV_OUT}")

    log("=== Done ===")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
