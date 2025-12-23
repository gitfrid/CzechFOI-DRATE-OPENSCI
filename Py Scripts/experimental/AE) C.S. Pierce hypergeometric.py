#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time‑Varying Peircean RMST‑like Analysis
----------------------------------------

This script computes a *strategy‑based*, *time‑varying*, *empirical* Peircean
ΔRMST‑like contrast using:

    • Dynamic strategies (A=0 never vaccinate, A=1 vaccinate at actual dose day)
    • Clone‑based follow‑up (corrects immortal time bias)
    • Daily empirical hazards from aggregated clone intervals
    • Empirical survival curves S_v(t), S_u(t)
    • Peircean hypergeometric evidence I(t)
    • ΔRMST_empirical_peircean(t) = Σ ΔS(t)
    • ΔRMST_weighted_peircean(t) = Σ ΔS(t)·I(t)

This script contains ONLY the Peircean method.
No pooled logistic regression, no GLM, no modeling.

Author: AI — 2025
"""

# =====================================================================
# IMPORTS
# =====================================================================
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
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce hypergeom vax effect\AE) C.S. Pierce hypergeom vax effect RMST")

# simulated dataset HR=1 with simulated real dose schedule
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce hypergeom vax effect\AE) C.S. Pierce hypergeom vax effect RMST_SIM")

# real data with 5% uvx deaths reclassified as vx
# INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_PTC5_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce hypergeom vax effect\AE) C.S. Pierce hypergeom vax effect RMST_RECLASSIFIED")

# =====================================================================
# LOGGING
# =====================================================================
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_BASE / "peircean_timevarying_log.txt"
log_fh = open(log_path, "w", encoding="utf-8")

def log(msg: str):
    print(msg)
    log_fh.write(msg + "\n")


# =====================================================================
# DATA LOADING
# =====================================================================
def load_raw(path: Path) -> pd.DataFrame:
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


# =====================================================================
# CLONE CONSTRUCTION (STRATEGY A=0, A=1)
# =====================================================================
def build_clones(raw: pd.DataFrame):
    """
    Construct strategy-specific clones (A=0 and A=1).

    A=0: from FIRST_VAX_DAY until min(vaccination, death, last_obs).
    A=1: from first dose until min(death, last_obs). Only for vaccinated.
    """
    FIRST = int(raw.loc[raw["first_dose_day"].notna(), "first_dose_day"].min())
    last_obs = float(min(raw["death_day"].max(), raw["first_dose_day"].max()) - SAFETY_BUFFER)

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
        eu_candidates = [x for x in [f, d, last_obs] if not np.isnan(x)]
        if eu_candidates:
            eu = min(eu_candidates)
            if eu > su:
                clones.append((sid, 0, su, eu, int(not np.isnan(d) and d <= eu)))

        # Strategy A = 1 (dynamic vaccination)
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


# =====================================================================
# DAILY AGGREGATION OF CLONES
# =====================================================================
def compute_daily_events(df, start_day, end_day):
    """
    Aggregate clone intervals into daily events and at-risk counts.
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


def aggregate(clones, t, FIRST, last_obs):
    """
    Daily aggregation by vaccination strategy.
    """
    ev_v, r_v = compute_daily_events(clones[clones.vaccinated == 1], FIRST, last_obs)
    ev_u, r_u = compute_daily_events(clones[clones.vaccinated == 0], FIRST, last_obs)

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
def hypergeom_p_value(N_v, N_u, D, k):
    """
    Two-sided hypergeometric p-value for daily allocation of D deaths
    between vaccinated (N_v at risk) and unvaccinated (N_u at risk).
    """
    M = N_v + N_u
    if M <= 0 or D <= 0:
        return 1.0

    rv = hypergeom(M, D, N_v)
    p_lower = rv.cdf(k)
    p_upper = rv.sf(k - 1)
    return 2 * min(p_lower, p_upper)
# =====================================================================
# PEIRCEAN EMPIRICAL STRATEGY-BASED RMST
# =====================================================================
def peircean_empirical_from_agg(agg, t):
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
        dict with:
          S_v_emp, S_u_emp, DeltaS,
          I_t,
          Delta_RMST_empirical_peircean,
          Delta_RMST_weighted_peircean
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

    # Peircean evidence
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
    }


# =====================================================================
# PLOTTING
# =====================================================================
def plot_peircean(t, S_v, S_u, DeltaS, I_t, RMST_emp, RMST_w, outdir):
    """
    Produce two HTML plots:
      1) Empirical survival curves
      2) Peircean ΔRMST-like curves
    """
    # --- Plot 1: Survival ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=S_v, mode="lines", line=dict(color="green", width=2),
                              name="S_v(t) dynamic vaccinate"))
    fig1.add_trace(go.Scatter(x=t, y=S_u, mode="lines", line=dict(color="red", width=2),
                              name="S_u(t) never vaccinate"))

    fig1.update_layout(
        title="Peircean Empirical Strategy-Based Survival Curves",
        xaxis_title="Days since FIRST_VAX_DAY",
        yaxis_title="Survival probability",
        template="plotly_white",
    )

    surv_path = outdir / "Peircean_Empirical_Survival.html"
    fig1.write_html(surv_path)
    log(f"Saved: {surv_path}")

    # --- Plot 2: ΔRMST-like ---
    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.25, 0.40],
        vertical_spacing=0.03,
        subplot_titles=[
            "ΔS(t) = S_v(t) - S_u(t)",
            "Peircean evidence I(t) = -log10 p(t)",
            "Peircean ΔRMST-like curves"
        ]
    )

    fig2.add_trace(go.Scatter(x=t, y=DeltaS, mode="lines", line=dict(color="black", width=2)), row=1, col=1)
    fig2.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=1)

    fig2.add_trace(go.Scatter(x=t, y=I_t, mode="lines", line=dict(color="blue", width=2)), row=2, col=1)

    fig2.add_trace(go.Scatter(x=t, y=RMST_emp, mode="lines", line=dict(color="green", width=2),
                              name="ΔRMST_empirical"), row=3, col=1)
    fig2.add_trace(go.Scatter(x=t, y=RMST_w, mode="lines", line=dict(color="red", width=2, dash="dot"),
                              name="ΔRMST_weighted"), row=3, col=1)

    fig2.update_layout(
        title="Peircean Time-Varying RMST-like Contrasts",
        xaxis3_title="Days since FIRST_VAX_DAY",
        template="plotly_white",
    )

    rmst_path = outdir / "Peircean_RMST_like.html"
    fig2.write_html(rmst_path)
    log(f"Saved: {rmst_path}")


# =====================================================================
# CSV OUTPUT
# =====================================================================
def save_summary_csv(summary, outdir):
    df = pd.DataFrame([summary])
    path = outdir / "Peircean_summary.csv"
    df.to_csv(path, index=False)
    log(f"Saved summary CSV: {path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    log("=== Time-Varying Peircean RMST Analysis ===")

    raw = load_raw(INPUT)
    clones, t, FIRST, last_obs = build_clones(raw)
    agg = aggregate(clones, t, FIRST, last_obs)

    log("\n=== Computing Peircean empirical strategy-based RMST ===")
    res = peircean_empirical_from_agg(agg, t)

    S_v = res["S_v_emp"]
    S_u = res["S_u_emp"]
    DeltaS = res["DeltaS"]
    I_t = res["I_t"]
    RMST_emp = res["Delta_RMST_empirical_peircean"]
    RMST_w = res["Delta_RMST_weighted_peircean"]

    log(f"Final ΔRMST_empirical_peircean(T): {RMST_emp[-1]:.3f} days")
    log(f"Final ΔRMST_weighted_peircean(T): {RMST_w[-1]:.3f} (days × evidence)")

    summary = {
        "FIRST_VAX_DAY": FIRST,
        "last_obs": last_obs,
        "window_days": len(t),
        "Delta_RMST_empirical_final": RMST_emp[-1],
        "Delta_RMST_weighted_final": RMST_w[-1],
    }
    save_summary_csv(summary, OUTPUT_BASE)

    plot_peircean(t, S_v, S_u, DeltaS, I_t, RMST_emp, RMST_w, OUTPUT_BASE)

    log("\n=== Finished Peircean Time-Varying RMST Analysis ===")
    log_fh.close()


if __name__ == "__main__":
    main()
