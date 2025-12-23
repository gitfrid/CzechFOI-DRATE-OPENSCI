#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
C.S. Peirce style emperical Evidence-Weighted RMST Analysis
---------------------------------------

This script implements a strategy-based, time-varying, empirical Peircean analysis
of restricted mean survival time (RMST) for a dynamic vaccination strategy.

Key features:
    - Target-trial style design with clone-based strategies:
        A = 0: never vaccinate
        A = 1: dynamic vaccination at the individual's actual first-dose date
    - Immortal time bias correction via clone construction
    - Empirical hazards and survival for each strategy (no regression modeling)
    - Peircean evidence weighting using daily hypergeometric surprisal:
          I(t) = -log10 p(t)
      where p(t) tests random allocation of deaths between strategies
    - Peirce evidence-weighted survival curves
    - RMST-like contrasts:
          ΔRMST_empirical(t)  = ∑ [S_v(t) - S_u(t)]
          ΔRMST_weighted(t)   = ∑ [S_v(t) - S_u(t)] * I(t)
    - Cluster bootstrap over subjects for:
        * Peirce survival curves (95% pointwise CIs)
        * ΔRMST empirical and weighted curves (95% pointwise CIs)
    - Preprint-ready plots and CSV summary

Author: AI / inspierd by C.S. Peirce & Hernan / Assisted gitfrid  Date:2025  Version 1.0
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# =====================================================================
# CONFIGURATION
# =====================================================================

AGE = 70
AGE_REF_YEAR = 2023
STUDY_START = pd.Timestamp("2020-01-01")
SAFETY_BUFFER = 30

# -------------------- INPUT / OUTPUT --------------------

# real Czech-FOI Data for specific AG
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce evidence weighted RMST\AE) C.S. Pierce evidence weighted RMST")

# simulated dataset HR=1 with simulated real dose schedule
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce evidence weighted RMST\AE) C.S. Pierce evidence weighted RMST_SIM")

# real data with 5% uvx deaths reclassified as vx
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_PTC5_uvx_as_vx_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce evidence weighted RMST\AE) C.S. Pierce evidence weighted RMST_RECLASSIFIED")


# Bootstrap settings
N_BOOT = 30
BOOT_SUBSAMPLE = 0.40
RANDOM_SEED = 12345

# ---------------------------------------------------------------------
# Derived paths and logging setup
# ---------------------------------------------------------------------
base_name = OUTPUT_BASE.name
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)

log_path = OUTPUT_BASE.parent / f"{base_name}_AG{AGE}_log.txt"
_log_fh = open(log_path, "w", encoding="utf-8")


def log(msg: str):
    """Log to console and to log file."""
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


# =====================================================================
# DATA LOADING
# =====================================================================

def load_raw(path: Path) -> pd.DataFrame:
    """
    Load raw CSV, filter by age, compute death_day and first_dose_day.
    Assumes:
        - 'Rok_narozeni' (year of birth)
        - 'DatumUmrti'  (date of death)
        - 'Datum_1'     (date of first dose)
    """
    log(f"Loading CSV: {path}")
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

    # Parse date columns
    date_cols = ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]
    for c in date_cols:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")

    raw["Rok_narozeni"] = pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw["age"] = AGE_REF_YEAR - raw["Rok_narozeni"]
    raw = raw[raw["age"] == AGE].copy()
    raw.reset_index(drop=True, inplace=True)
    raw["subject_id"] = np.arange(len(raw), dtype=np.int32)

    raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
    raw["first_dose_day"] = (raw["Datum_1"] - STUDY_START).dt.days

    log(f"Subjects after age filter: {len(raw)}")
    log(f"Deaths: {raw['death_day'].notna().sum()}, "
        f"Vaccinated: {raw['first_dose_day'].notna().sum()}")
    return raw


# =====================================================================
# CLONE CONSTRUCTION (STRATEGY A=0, A=1)
# =====================================================================

def build_clones(raw: pd.DataFrame):
    """
    Construct strategy-specific clones (A=0 and A=1).

    A=0: from FIRST_VAX_DAY until min(vaccination, death, last_obs).
    A=1: from first dose until min(death, last_obs). Only for vaccinated.

    Returns:
        clones: DataFrame with columns [id, vaccinated, start, stop, event]
        t:      numpy array of analysis days (0..T-1)
        FIRST:  integer, first vaccination day in dataset
        last_obs: float, last observation day (tail trimmed by SAFETY_BUFFER)
    """
    FIRST = int(raw.loc[raw["first_dose_day"].notna(), "first_dose_day"].min())
    last_obs = float(
        min(raw["death_day"].max(), raw["first_dose_day"].max()) - SAFETY_BUFFER
    )

    window = int(last_obs - FIRST)
    if window <= 0:
        raise ValueError(f"No follow-up window: FIRST={FIRST}, last_obs={last_obs}")

    t = np.arange(window, dtype=int)

    clones = []
    for _, r in raw.iterrows():
        sid = int(r["subject_id"])
        d = float(r["death_day"]) if pd.notna(r["death_day"]) else np.nan
        f = float(r["first_dose_day"]) if pd.notna(r["first_dose_day"]) else np.nan

        # Strategy A = 0 (never vaccinate)
        su = FIRST
        eu_candidates = [x for x in (f, d, last_obs) if not np.isnan(x)]
        if eu_candidates:
            eu = min(eu_candidates)
            if eu > su:
                event_u = int(not np.isnan(d) and d <= eu)
                clones.append((sid, 0, su, eu, event_u))

        # Strategy A = 1 (dynamic vaccination)
        if not np.isnan(f):
            sv = max(f, FIRST)
            ev = min(d if not np.isnan(d) else last_obs, last_obs)
            if ev > sv:
                event_v = int(not np.isnan(d) and sv <= d <= ev)
                clones.append((sid, 1, sv, ev, event_v))

    df = pd.DataFrame(
        clones, columns=["id", "vaccinated", "start", "stop", "event"]
    ).astype({"id": "int32", "vaccinated": "int32", "event": "int32"})

    log(f"Total clone intervals: {len(df)}")
    log(f"  Vaccinated intervals:   {int((df['vaccinated'] == 1).sum())}")
    log(f"  Unvaccinated intervals: {int((df['vaccinated'] == 0).sum())}")
    log(f"FIRST_VAX_DAY: {FIRST}")
    log(f"last_obs: {last_obs}")
    log(f"Analysis window length: {len(t)} days")

    return df, t, FIRST, last_obs


# =====================================================================
# DAILY AGGREGATION OF CLONES
# =====================================================================

def compute_daily_events(df: pd.DataFrame, start_day: float, end_day: float):
    """
    Aggregate clone intervals into daily events and at-risk counts
    over [start_day, end_day).
    """
    window = int(end_day - start_day)
    events = np.zeros(window, dtype=int)
    diff = np.zeros(window + 1, dtype=int)

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


def aggregate(clones: pd.DataFrame, t: np.ndarray, FIRST: int, last_obs: float):
    """
    Daily aggregation by vaccination strategy.

    Returns:
        agg: DataFrame with columns [day, vaccinated, events, risk]
    """
    ev_v, r_v = compute_daily_events(
        clones[clones.vaccinated == 1], FIRST, last_obs
    )
    ev_u, r_u = compute_daily_events(
        clones[clones.vaccinated == 0], FIRST, last_obs
    )

    log(f"Person-time (vaccinated):   {r_v.sum():,} person-days")
    log(f"Person-time (unvaccinated): {r_u.sum():,} person-days")
    log(f"Deaths (vaccinated):        {ev_v.sum()}")
    log(f"Deaths (unvaccinated):      {ev_u.sum()}")

    agg = pd.DataFrame({
        "day": np.concatenate([t, t]),
        "vaccinated": np.concatenate([np.ones_like(t), np.zeros_like(t)]),
        "events": np.concatenate([ev_v, ev_u]),
        "risk": np.concatenate([r_v, r_u]),
    })

    return agg

# =====================================================================
# PEIRCEAN HYPERGEOMETRIC EVIDENCE
# =====================================================================

def hypergeom_p_value(N_v: int, N_u: int, D: int, k: int) -> float:
    """
    Two-sided hypergeometric p-value for daily allocation of D deaths
    between vaccinated (N_v at risk) and unvaccinated (N_u at risk).

    N_v + N_u = total at risk that day
    D         = total deaths that day
    k         = vaccinated deaths that day
    """
    M = N_v + N_u
    if M <= 0 or D <= 0:
        return 1.0

    rv = hypergeom(M, D, N_v)
    p_lower = rv.cdf(k)
    p_upper = rv.sf(k - 1)
    return 2 * min(p_lower, p_upper)


# =====================================================================
# PEIRCEAN EMPIRICAL STRATEGY-BASED ANALYSIS
# =====================================================================

def peircean_empirical_from_agg(agg: pd.DataFrame, t: np.ndarray):
    """
    Peircean-style empirical, strategy-based analysis using aggregated clone data.

    Input:
        agg: DataFrame with columns:
             - day: 0..T-1 (relative to FIRST_VAX_DAY)
             - vaccinated: 1 for strategy A=1 (dynamic vaccinate), 0 for A=0 (never)
             - events: deaths that day under that strategy
             - risk: persons at risk that day under that strategy
        t:   array of days 0..T-1

    Output:
        dict with keys:
          - S_v_emp, S_u_emp: empirical survival curves
          - DeltaS: S_v_emp - S_u_emp
          - I_t: Peircean evidence per day
          - Delta_RMST_empirical_peircean
          - Delta_RMST_weighted_peircean
          - h_v, h_u: empirical hazards
          - S_v_p, S_u_p: Peirce evidence-weighted survival curves
    """
    agg_v = agg[agg["vaccinated"] == 1].sort_values("day")
    agg_u = agg[agg["vaccinated"] == 0].sort_values("day")

    if not np.array_equal(agg_v["day"].values, t) or not np.array_equal(agg_u["day"].values, t):
        raise ValueError("Days in agg do not match time grid t for both strategies.")

    r_v = agg_v["risk"].to_numpy(dtype=float)
    e_v = agg_v["events"].to_numpy(dtype=float)
    r_u = agg_u["risk"].to_numpy(dtype=float)
    e_u = agg_u["events"].to_numpy(dtype=float)

    # Empirical hazards
    h_v = np.where(r_v > 0, e_v / r_v, 0.0)
    h_u = np.where(r_u > 0, e_u / r_u, 0.0)

    # Empirical survival curves
    S_v_emp = np.empty_like(h_v)
    S_u_emp = np.empty_like(h_u)

    S_v_emp[0] = 1.0 - h_v[0]
    S_u_emp[0] = 1.0 - h_u[0]
    for i in range(1, len(t)):
        S_v_emp[i] = S_v_emp[i - 1] * (1.0 - h_v[i])
        S_u_emp[i] = S_u_emp[i - 1] * (1.0 - h_u[i])

    # Pointwise difference
    DeltaS = S_v_emp - S_u_emp

    # Peircean evidence per day
    I_t = np.zeros_like(h_v)
    for i in range(len(t)):
        N_v = int(r_v[i])
        N_u = int(r_u[i])
        D = int(e_v[i] + e_u[i])
        k = int(e_v[i])

        if D == 0 or (N_v + N_u) == 0:
            I_t[i] = 0.0
            continue

        p_val = hypergeom_p_value(N_v, N_u, D, k)
        p_val = max(p_val, 1e-16)
        I_t[i] = -np.log10(p_val)

    # Peirce evidence-weighted hazards and survival
    w = I_t / (np.max(I_t) if np.max(I_t) > 0 else 1.0)

    h_v_p = h_v * w
    h_u_p = h_u * w

    S_v_p = np.empty_like(h_v_p)
    S_u_p = np.empty_like(h_u_p)
    S_v_p[0] = 1.0 - h_v_p[0]
    S_u_p[0] = 1.0 - h_u_p[0]
    for i in range(1, len(t)):
        S_v_p[i] = S_v_p[i - 1] * (1.0 - h_v_p[i])
        S_u_p[i] = S_u_p[i - 1] * (1.0 - h_u_p[i])

    # Peircean RMST-like curves
    Delta_RMST_empirical_peircean = np.cumsum(DeltaS)
    Delta_RMST_weighted_peircean = np.cumsum(DeltaS * I_t)

    return {
        "S_v_emp": S_v_emp,
        "S_u_emp": S_u_emp,
        "DeltaS": DeltaS,
        "I_t": I_t,
        "Delta_RMST_empirical_peircean": Delta_RMST_empirical_peircean,
        "Delta_RMST_weighted_peircean": Delta_RMST_weighted_peircean,
        "h_v": h_v,
        "h_u": h_u,
        "S_v_p": S_v_p,
        "S_u_p": S_u_p,
    }


# =====================================================================
# CLUSTER BOOTSTRAP FOR PEIRCE SURVIVAL & ΔRMST
# =====================================================================

def bootstrap_peirce(raw: pd.DataFrame,
                     B: int,
                     subsample: float,
                     seed: int = 12345):
    """
    Cluster bootstrap for Peirce survival curves and ΔRMST curves.

    raw: original subject-level DataFrame
    B: number of bootstrap replications
    subsample: fraction of subjects per bootstrap
    Returns:
        dict with:
          - t: time grid
          - Sv_p_lo, Sv_p_hi, Su_p_lo, Su_p_hi: 95% CI bands for Peirce survival
          - RMST_emp_lo, RMST_emp_hi: 95% CI bands for empirical ΔRMST(t)
          - RMST_w_lo, RMST_w_hi: 95% CI bands for weighted ΔRMST(t)
    """
    rng = np.random.default_rng(seed)
    ids = raw["subject_id"].unique()
    n = len(ids)

    # Build base time grid from full data
    clones_full, t_full, FIRST, last_obs = build_clones(raw)
    agg_full = aggregate(clones_full, t_full, FIRST, last_obs)
    T = len(t_full)

    Svp_list = []
    Sup_list = []
    RMST_emp_list = []
    RMST_w_list = []

    successful = 0
    for b in range(B):
        m = int(subsample * n)
        sample_ids = rng.choice(ids, size=m, replace=True)
        boot_raw = raw[raw["subject_id"].isin(sample_ids)].copy()

        try:
            clones_b, t_b, FIRST_b, last_obs_b = build_clones(boot_raw)
            if len(t_b) != T:
                # Different window length; skip this replicate
                continue

            agg_b = aggregate(clones_b, t_b, FIRST_b, last_obs_b)
            res_b = peircean_empirical_from_agg(agg_b, t_b)

            Svp_list.append(res_b["S_v_p"])
            Sup_list.append(res_b["S_u_p"])
            RMST_emp_list.append(res_b["Delta_RMST_empirical_peircean"])
            RMST_w_list.append(res_b["Delta_RMST_weighted_peircean"])
            successful += 1
        except Exception:
            continue

    log(f"Bootstrap successful replicates: {successful}/{B}")
    if successful == 0:
        raise RuntimeError("No successful bootstrap replicates for Peirce analysis.")

    Svp_arr = np.vstack(Svp_list)
    Sup_arr = np.vstack(Sup_list)
    RMST_emp_arr = np.vstack(RMST_emp_list)
    RMST_w_arr = np.vstack(RMST_w_list)

    Sv_p_lo = np.percentile(Svp_arr, 2.5, axis=0)
    Sv_p_hi = np.percentile(Svp_arr, 97.5, axis=0)
    Su_p_lo = np.percentile(Sup_arr, 2.5, axis=0)
    Su_p_hi = np.percentile(Sup_arr, 97.5, axis=0)

    RMST_emp_lo = np.percentile(RMST_emp_arr, 2.5, axis=0)
    RMST_emp_hi = np.percentile(RMST_emp_arr, 97.5, axis=0)
    RMST_w_lo = np.percentile(RMST_w_arr, 2.5, axis=0)
    RMST_w_hi = np.percentile(RMST_w_arr, 97.5, axis=0)

    return {
        "t": t_full,
        "Sv_p_lo": Sv_p_lo,
        "Sv_p_hi": Sv_p_hi,
        "Su_p_lo": Su_p_lo,
        "Su_p_hi": Su_p_hi,
        "RMST_emp_lo": RMST_emp_lo,
        "RMST_emp_hi": RMST_emp_hi,
        "RMST_w_lo": RMST_w_lo,
        "RMST_w_hi": RMST_w_hi,
    }

# =====================================================================
# PLOTTING
# =====================================================================

def plot_peircean(t: np.ndarray,
                  S_v_emp: np.ndarray,
                  S_u_emp: np.ndarray,
                  S_v_p: np.ndarray,
                  S_u_p: np.ndarray,
                  Sv_p_lo: np.ndarray,
                  Sv_p_hi: np.ndarray,
                  Su_p_lo: np.ndarray,
                  Su_p_hi: np.ndarray,
                  DeltaS: np.ndarray,
                  I_t: np.ndarray,
                  RMST_emp: np.ndarray,
                  RMST_emp_lo: np.ndarray,
                  RMST_emp_hi: np.ndarray,
                  RMST_w: np.ndarray,
                  RMST_w_lo: np.ndarray,
                  RMST_w_hi: np.ndarray,
                  output_base: Path,
                  age: int):
    """
    Produce two HTML plots:
      1) Survival curves (empirical + Peirce with 95% CI bands)
      2) Peircean ΔRMST-like curves (empirical and weighted, with 95% CI bands)
    """
    base_name = output_base.name

    # ---------- Plot 1: Survival ----------
    fig1 = go.Figure()

    # Empirical survival
    fig1.add_trace(go.Scatter(
        x=t, y=S_v_emp,
        mode="lines", line=dict(color="green", width=2),
        name="S_v(t) dynamic vaccinate"
    ))
    fig1.add_trace(go.Scatter(
        x=t, y=S_u_emp,
        mode="lines", line=dict(color="red", width=2),
        name="S_u(t) never vaccinate"
    ))

    # Peirce survival CIs - vaccinated
    fig1.add_trace(go.Scatter(
        x=t, y=Sv_p_hi,
        line=dict(width=0), showlegend=False
    ))
    fig1.add_trace(go.Scatter(
        x=t, y=Sv_p_lo,
        fill="tonexty",
        fillcolor="rgba(0, 0, 255, 0.15)",
        line=dict(width=0),
        showlegend=False
    ))

    # Peirce survival CIs - unvaccinated
    fig1.add_trace(go.Scatter(
        x=t, y=Su_p_hi,
        line=dict(width=0), showlegend=False
    ))
    fig1.add_trace(go.Scatter(
        x=t, y=Su_p_lo,
        fill="tonexty",
        fillcolor="rgba(128, 0, 128, 0.15)",
        line=dict(width=0),
        showlegend=False
    ))

    # Peirce survival curves
    fig1.add_trace(go.Scatter(
        x=t, y=S_v_p,
        mode="lines",
        line=dict(color="blue", width=2, dash="dot"),
        name="S_v^Peirce(t)"
    ))
    fig1.add_trace(go.Scatter(
        x=t, y=S_u_p,
        mode="lines",
        line=dict(color="purple", width=2, dash="dot"),
        name="S_u^Peirce(t)"
    ))

    fig1.update_layout(
        title=f"Peircean Empirical and Evidence-Weighted Survival Curves (AG{age})",
        xaxis_title="Days since FIRST_VAX_DAY",
        yaxis_title="Survival probability",
        template="plotly_white",
    )

    surv_filename = f"{base_name}_AG{age}_PeirceSurvival_CI.html"
    surv_path = output_base.parent / surv_filename
    fig1.write_html(surv_path)
    log(f"Saved survival plot: {surv_path}")

    # ---------- Plot 2: ΔRMST-like curves ----------
    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.30, 0.25, 0.45],
        vertical_spacing=0.03,
        subplot_titles=[
            "ΔS(t) = S_v(t) - S_u(t)",
            "Peircean evidence I(t) = -log10 p(t)",
            "Peircean RMST-like contrasts (with 95% CIs)"
        ]
    )

    # Panel 1: ΔS(t)
    fig2.add_trace(go.Scatter(
        x=t, y=DeltaS,
        mode="lines", line=dict(color="black", width=2),
        name="ΔS(t)"
    ), row=1, col=1)
    fig2.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=1)

    # Panel 2: I(t)
    fig2.add_trace(go.Scatter(
        x=t, y=I_t,
        mode="lines", line=dict(color="blue", width=2),
        name="I(t)"
    ), row=2, col=1)

    # Panel 3: RMST curves with CIs
    # Empirical RMST band
    fig2.add_trace(go.Scatter(
        x=t, y=RMST_emp_hi,
        line=dict(width=0),
        showlegend=False
    ), row=3, col=1)
    fig2.add_trace(go.Scatter(
        x=t, y=RMST_emp_lo,
        fill="tonexty",
        fillcolor="rgba(0, 150, 0, 0.15)",
        line=dict(width=0),
        showlegend=False
    ), row=3, col=1)

    # Weighted RMST band
    fig2.add_trace(go.Scatter(
        x=t, y=RMST_w_hi,
        line=dict(width=0),
        showlegend=False
    ), row=3, col=1)
    fig2.add_trace(go.Scatter(
        x=t, y=RMST_w_lo,
        fill="tonexty",
        fillcolor="rgba(200, 0, 0, 0.15)",
        line=dict(width=0),
        showlegend=False
    ), row=3, col=1)

    # Empirical RMST curve
    fig2.add_trace(go.Scatter(
        x=t, y=RMST_emp,
        mode="lines",
        line=dict(color="green", width=2),
        name="ΔRMST_empirical(t)"
    ), row=3, col=1)

    # Weighted RMST curve
    fig2.add_trace(go.Scatter(
        x=t, y=RMST_w,
        mode="lines",
        line=dict(color="red", width=2, dash="dot"),
        name="ΔRMST_weighted(t)"
    ), row=3, col=1)

    fig2.update_layout(
        title=f"Peircean Time-Varying RMST-like Contrasts (AG{age})",
        xaxis3_title="Days since FIRST_VAX_DAY",
        template="plotly_white",
    )

    rmst_filename = f"{base_name}_AG{age}_PeirceRMST_CI.html"
    rmst_path = output_base.parent / rmst_filename
    fig2.write_html(rmst_path)
    log(f"Saved RMST plot: {rmst_path}")


# =====================================================================
# CSV SUMMARY
# =====================================================================

def save_summary_csv(FIRST: int,
                     last_obs: float,
                     t: np.ndarray,
                     RMST_emp: np.ndarray,
                     RMST_emp_lo: np.ndarray,
                     RMST_emp_hi: np.ndarray,
                     RMST_w: np.ndarray,
                     RMST_w_lo: np.ndarray,
                     RMST_w_hi: np.ndarray,
                     output_base: Path,
                     age: int):
    """
    Save a one-row CSV summary with final ΔRMST values and their 95% CIs.
    """
    base_name = output_base.name
    T = len(t)
    tau = t[-1]

    summary = {
        "AGE": age,
        "FIRST_VAX_DAY": FIRST,
        "last_obs": last_obs,
        "window_days": T,
        "tau": tau,
        "Delta_RMST_empirical_final": RMST_emp[-1],
        "Delta_RMST_empirical_95CI_low": RMST_emp_lo[-1],
        "Delta_RMST_empirical_95CI_high": RMST_emp_hi[-1],
        "Delta_RMST_weighted_final": RMST_w[-1],
        "Delta_RMST_weighted_95CI_low": RMST_w_lo[-1],
        "Delta_RMST_weighted_95CI_high": RMST_w_hi[-1],
    }

    df = pd.DataFrame([summary])
    csv_filename = f"{base_name}_AG{age}_summary.csv"
    csv_path = output_base.parent / csv_filename
    df.to_csv(csv_path, index=False)
    log(f"Saved summary CSV: {csv_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    log("=== Peircean Evidence-Weighted RMST Analysis ===")

    # Load raw data and build clones
    raw = load_raw(INPUT)
    clones, t, FIRST, last_obs = build_clones(raw)
    agg = aggregate(clones, t, FIRST, last_obs)

    # Peircean empirical and evidence-weighted analysis
    log("\n=== Computing Peircean empirical strategy-based RMST ===")
    res = peircean_empirical_from_agg(agg, t)

    S_v_emp = res["S_v_emp"]
    S_u_emp = res["S_u_emp"]
    DeltaS = res["DeltaS"]
    I_t = res["I_t"]
    RMST_emp = res["Delta_RMST_empirical_peircean"]
    RMST_w = res["Delta_RMST_weighted_peircean"]
    S_v_p = res["S_v_p"]
    S_u_p = res["S_u_p"]

    log(f"Final ΔRMST_empirical_peircean(T): {RMST_emp[-1]:.3f} days")
    log(f"Final ΔRMST_weighted_peircean(T): {RMST_w[-1]:.3f} (days × evidence)")

    # Bootstrap CIs
    log("\n=== Bootstrap CIs for Peirce survival and ΔRMST curves ===")
    boot_res = bootstrap_peirce(
        raw, B=N_BOOT, subsample=BOOT_SUBSAMPLE, seed=RANDOM_SEED
    )

    Sv_p_lo = boot_res["Sv_p_lo"]
    Sv_p_hi = boot_res["Sv_p_hi"]
    Su_p_lo = boot_res["Su_p_lo"]
    Su_p_hi = boot_res["Su_p_hi"]
    RMST_emp_lo = boot_res["RMST_emp_lo"]
    RMST_emp_hi = boot_res["RMST_emp_hi"]
    RMST_w_lo = boot_res["RMST_w_lo"]
    RMST_w_hi = boot_res["RMST_w_hi"]

    # Plots
    plot_peircean(
        t,
        S_v_emp, S_u_emp,
        S_v_p, S_u_p,
        Sv_p_lo, Sv_p_hi, Su_p_lo, Su_p_hi,
        DeltaS, I_t,
        RMST_emp, RMST_emp_lo, RMST_emp_hi,
        RMST_w, RMST_w_lo, RMST_w_hi,
        OUTPUT_BASE,
        AGE,
    )

    # Summary CSV
    save_summary_csv(
        FIRST, last_obs, t,
        RMST_emp, RMST_emp_lo, RMST_emp_hi,
        RMST_w, RMST_w_lo, RMST_w_hi,
        OUTPUT_BASE,
        AGE,
    )

    log("\n=== Finished Peircean Evidence-Weighted RMST Analysis ===")
    _log_fh.close()


if __name__ == "__main__":
    main()
