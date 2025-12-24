#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
C.S. Peirce-Informed Empirical Evidence-Weighted RMST Analysis (Experimental/Exploratory)
---------------------------------------

This script implements a strategy-based, time-varying, empirical Peircean analysis
of restricted mean survival time (RMST) for a dynamic vaccination strategy.

Enhanced Key Features (Peircean Improvements):
    - Target-trial style design with clone-based strategies:
        A = 0: never vaccinate (control)
        A = 1: dynamic vaccination at the individual's actual first-dose date
    - Immortal time & healthy vaccinee bias correction via clone construction (+14d buffer)
    - Empirical hazards and survival for each strategy (non-parametric, no PH assumptions)
    - Peircean evidence weighting using directional log-odds surprisal:
          I(t) = sign(ΔS(t)) * -ln(p(t))  # Directional, signed evidence
      where p(t) is derived from a rolling-window hypergeometric test (Fisher's exact logic)
    - Peircean error bounds (Greenwood-style) on empirical hazards
    - Abductive flagging for high-surprise days (|I(t)| > threshold)
    - RMST-like contrasts:
          ΔRMST_empirical(t) = Σ [S_v(t) - S_u(t)]
          ΔRMST_weighted(t)  = Σ ([S_v(t) - S_u(t)] * w(t))  where w(t) = |I(t)|/max|I|
    - Enhanced multi-seed consensus bootstrap for robust confidence intervals
    - Preprint-ready directional diagnostics (Green/Red surprisal fills) and cluster reports

Author: AI / Inspired by C.S. Peirce & Hernán / Assisted gitfrid    Date: 2025    Version 1.3
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import logging
import plotly.express as px

# =====================================================================
# CONFIGURATION
# =====================================================================

AGE = 70
AGE_REF_YEAR = 2023
STUDY_START = pd.Timestamp("2020-01-01")
SAFETY_BUFFER = 30

# Input / Output paths (adjust as needed)

# real Czech-FOI Data for specific AG
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce evidence weighted RMST\AE) C.S. Pierce evidence weighted RMST")

# simulated dataset HR=1 with simulated real dose schedule
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce evidence weighted RMST\AE) C.S. Pierce evidence weighted RMST_SIM")

# simulated dataset real data with 5% uvx deaths reclassified as vx
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_PTC5_uvx_as_vx_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) C.S. Pierce evidence weighted RMST\AE) C.S. Pierce evidence weighted RMST_RECLASSIFIED")

# Bootstrap settings
N_BOOT = 5
BOOT_SUBSAMPLE = 0.80
RANDOM_SEED = 12345
N_SEEDS = 3

# Peircean parameters
ROLLING_WINDOW = 7
SURPRISE_THRESHOLD = 3.0
EARLY_DOWNWEIGHT_FACTOR = 0.5

# Derive log file name BEFORE initializing logger
LOG_FILE = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_AG{AGE}_log.txt"

# CRITICAL: Create the folder before the logger tries to touch it
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)

# Initialize Logger (Force clear previous handlers for VS Code stability)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.info

# Test the log immediately
log(f"Logger initialized. Saving to: {LOG_FILE}")

# =====================================================================
# DATA LOADING & CLONE BUILDING
# =====================================================================

def load_raw(path: Path) -> pd.DataFrame:
    log(f"Loading CSV: {path}")
    raw = pd.read_csv(path, dtype=str)
    raw.columns = raw.columns.str.strip()

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
    log(f"Deaths: {raw['death_day'].notna().sum()}, Vaccinated: {raw['first_dose_day'].notna().sum()}")
    return raw

def build_clones(raw: pd.DataFrame):
    vax_mask = raw["first_dose_day"].notna()
    if not vax_mask.any():
        raise ValueError("No vaccination dates found.")

    FIRST = int(raw.loc[vax_mask, "first_dose_day"].min())
    max_day = min(raw["death_day"].max(), raw["first_dose_day"].max())
    last_obs = float(max_day - SAFETY_BUFFER)
    window = int(last_obs - FIRST)
    if window <= 0:
        raise ValueError(f"Window non-positive. FIRST: {FIRST}, last_obs: {last_obs}")

    t = np.arange(window, dtype=int)
    clones = []

    for _, r in raw.iterrows():
        sid = int(r["subject_id"])
        d = float(r["death_day"]) if pd.notna(r["death_day"]) else np.nan
        f = float(r["first_dose_day"]) if pd.notna(r["first_dose_day"]) else np.nan

        # Never-vax clone
        eu = min([x for x in (f, d, last_obs) if not np.isnan(x)])
        if eu > FIRST:
            event_u = int(not np.isnan(d) and d <= eu)
            clones.append((sid, 0, FIRST, eu, event_u))

        # Vaccinated clone (+14 day buffer)
        if not np.isnan(f):
            sv = f + 14
            ev = min([x for x in (d, last_obs) if not np.isnan(x)])
            if ev > sv:
                event_v = int(not np.isnan(d) and sv < d <= ev)
                clones.append((sid, 1, sv, ev, event_v))

    df = pd.DataFrame(clones, columns=["id", "vaccinated", "start", "stop", "event"])
    df = df.astype({"id": "int32", "vaccinated": "int32", "event": "int32"})
    return df, t, FIRST, last_obs

def aggregate(clones: pd.DataFrame, t: np.ndarray, FIRST: int, last_obs: float):
    def get_events_risk(df_group):
        win = len(t)
        events = np.zeros(win, dtype=int)
        diff = np.zeros(win + 1, dtype=int)
        si = np.clip((df_group["start"] - FIRST).astype(int), 0, win)
        ei = np.clip((df_group["stop"] - FIRST).astype(int), 0, win)
        np.add.at(diff, si, 1)
        np.add.at(diff, ei, -1)
        risk = np.cumsum(diff)[:win]
        ev_idx = (ei[df_group["event"] == 1] - 1).astype(int)
        ev_idx = ev_idx[(ev_idx >= 0) & (ev_idx < win)]
        np.add.at(events, ev_idx, 1)
        return events, risk

    ev_v, r_v = get_events_risk(clones[clones.vaccinated == 1])
    ev_u, r_u = get_events_risk(clones[clones.vaccinated == 0])

    log(f"Vacc: {r_v.sum():,} days, {ev_v.sum()} deaths")
    log(f"Unvax: {r_u.sum():,} days, {ev_u.sum()} deaths")

    return pd.DataFrame({
        "day": np.tile(t, 2),
        "vaccinated": np.repeat([1, 0], len(t)),
        "events": np.concatenate([ev_v, ev_u]),
        "risk": np.concatenate([r_v, r_u])
    })

# =====================================================================
# PEIRCEAN CORE
# =====================================================================

def hypergeom_p_value_rolling(agg: pd.DataFrame, t: np.ndarray, window_size: int = ROLLING_WINDOW):
    v = agg[agg["vaccinated"] == 1].sort_values("day")
    u = agg[agg["vaccinated"] == 0].sort_values("day")

    r_v = v["risk"].rolling(window_size, min_periods=1).sum().to_numpy()
    e_v = v["events"].rolling(window_size, min_periods=1).sum().to_numpy()
    r_u = u["risk"].rolling(window_size, min_periods=1).sum().to_numpy()
    e_u = u["events"].rolling(window_size, min_periods=1).sum().to_numpy()

    p_vals = np.ones(len(t))
    for i in range(len(t)):
        M = int(r_v[i] + r_u[i])
        D = int(e_v[i] + e_u[i])
        n = int(r_v[i])
        k = int(e_v[i])
        if D > 0 and M > 0:
            rv = hypergeom(M, D, n)
            p_vals[i] = 2 * min(rv.cdf(k), rv.sf(k - 1))

    return p_vals

def peircean_empirical_from_agg(agg: pd.DataFrame, t: np.ndarray):
    v = agg[agg["vaccinated"] == 1].sort_values("day")
    u = agg[agg["vaccinated"] == 0].sort_values("day")

    rv, ev = v["risk"].values.astype(float), v["events"].values.astype(float)
    ru, eu = u["risk"].values.astype(float), u["events"].values.astype(float)

    # 1. Empirical Hazards
    h_v = np.divide(ev, rv, out=np.zeros_like(ev), where=(rv >= 50))
    h_u = np.divide(eu, ru, out=np.zeros_like(eu), where=(ru >= 50))

    # 2. Empirical Survival & Directional Delta
    S_v_emp = np.cumprod(1 - h_v)
    S_u_emp = np.cumprod(1 - h_u)
    DeltaS = S_v_emp - S_u_emp

    # 3. DIRECTIONAL Surprisal (The Fix)
    p_vals = hypergeom_p_value_rolling(agg, t)
    # I(t) = sign(DeltaS) * -ln(p)
    # Positive I_t = Evidence favoring Vax; Negative I_t = Evidence favoring Unvax
    I_t = np.sign(DeltaS) * -np.log(np.maximum(p_vals, 1e-16))

    # 4. Asymmetric Evidence-Weighting
    # We normalize the magnitude but keep the sign for weighting logic
    max_abs_I = np.max(np.abs(I_t))
    w_magnitude = np.abs(I_t) / max_abs_I if max_abs_I > 0 else np.zeros_like(I_t)
    
    # We apply the weights to the hazards to create "Purified" curves
    # These curves only move significantly when the evidence (w) is strong
    S_v_p = np.cumprod(1 - h_v * w_magnitude)
    S_u_p = np.cumprod(1 - h_u * w_magnitude)

    # 5. Error Bounds
    err_v = np.sqrt(np.divide(h_v * (1 - h_v), rv, out=np.zeros_like(h_v), where=(rv > 0)))
    err_u = np.sqrt(np.divide(h_u * (1 - h_u), ru, out=np.zeros_like(h_u), where=(ru > 0)))

    return {
        "S_v_emp": S_v_emp,
        "S_u_emp": S_u_emp,
        "DeltaS": DeltaS,
        "I_t": I_t,  # Now SIGNED
        "abduction_flags": (np.abs(I_t) > SURPRISE_THRESHOLD).astype(int),
        "Delta_RMST_empirical": np.cumsum(DeltaS),
        "Delta_RMST_weighted": np.cumsum(DeltaS * w_magnitude), # Area under the evidence-weighted curve
        "S_v_p": S_v_p,
        "S_u_p": S_u_p,
        "h_v": h_v,
        "h_u": h_u,
        "err_v": err_v,
        "err_u": err_u
    }

# =====================================================================
# BOOTSTRAP (MULTI-SEED)
# =====================================================================

def bootstrap_peirce(raw: pd.DataFrame, B: int = N_BOOT, subsample: float = BOOT_SUBSAMPLE):
    clones_ref, t_full, _, _ = build_clones(raw)
    results_emp = []
    results_w = []

    for s in range(N_SEEDS):
        rng = np.random.default_rng(RANDOM_SEED + s)
        for _ in range(B):
            sample = raw.sample(frac=subsample, replace=True, random_state=rng.integers(0, 1e9))
            clones_b, _, f_b, lo_b = build_clones(sample)
            if clones_b is not None:
                agg_b = aggregate(clones_b, t_full, f_b, lo_b)
                res_b = peircean_empirical_from_agg(agg_b, t_full)
                results_emp.append(res_b["Delta_RMST_empirical"])
                results_w.append(res_b["Delta_RMST_weighted"])

    arr_emp = np.array(results_emp)
    arr_w = np.array(results_w)

    return {
        "emp_lo": np.percentile(arr_emp, 2.5, axis=0),
        "emp_hi": np.percentile(arr_emp, 97.5, axis=0),
        "w_lo": np.percentile(arr_w, 2.5, axis=0),
        "w_hi": np.percentile(arr_w, 97.5, axis=0)
    }

def plot_peircean(t, S_v_emp, S_u_emp, S_v_p, S_u_p, err_v, err_u, DeltaS, I_t,
                  RMST_emp, RMST_w, boot_res, output_base, age):
    base_name = output_base.name

    high_surprise_days = t[np.abs(I_t) > SURPRISE_THRESHOLD]
    plot_days = t[np.argsort(np.abs(I_t))[-20:]] # Top 20 for the visual
    if len(high_surprise_days) > 0:
        log(f"High surprise days detected: {len(high_surprise_days)}")

    # =================================================================
    # FIG 1: Survival Curves with Error Bounds
    # =================================================================
    fig1 = go.Figure()
    
    # Error Bounds (Shaded areas)
    fig1.add_trace(go.Scatter(x=t, y=S_v_emp + err_v, line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=t, y=S_v_emp - err_v, fill="tonexty", fillcolor="rgba(0,255,0,0.1)", name="S_v error"))
    fig1.add_trace(go.Scatter(x=t, y=S_u_emp + err_u, line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=t, y=S_u_emp - err_u, fill="tonexty", fillcolor="rgba(255,0,0,0.1)", name="S_u error"))

    # Empirical Lines (The "Brute Fact")
    fig1.add_trace(go.Scatter(x=t, y=S_v_emp, mode="lines", line=dict(color="darkgreen", width=2.5), name="S_v Empirical"))
    fig1.add_trace(go.Scatter(x=t, y=S_u_emp, mode="lines", line=dict(color="darkred", width=2.5), name="S_u Empirical"))

    # Weighted Lines (The "Purified Signal")
    fig1.add_trace(go.Scatter(x=t, y=S_v_p, mode="lines", line=dict(color="blue", dash="dot"), name="S_v Weighted"))
    fig1.add_trace(go.Scatter(x=t, y=S_u_p, mode="lines", line=dict(color="purple", dash="dot"), name="S_u Weighted"))

    # Add Surprise Flags
    for day in high_surprise_days:
        fig1.add_vline(x=day, line=dict(color="orange", width=0.5, dash="dot"))

    fig1.update_layout(title=f"Peircean Survival Strategy Comparison AG{age}", 
                      xaxis_title="Days", yaxis_title="Survival Probability", 
                      template="plotly_white", yaxis=dict(autorange=True))
    fig1.write_html(OUTPUT_BASE.parent / f"{base_name}_survival_AG{age}.html")
    log("Survival plot saved.")

    # =================================================================
    # FIG 2: RMST & Directional Surprisal
    # =================================================================
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.3, 0.25, 0.45],
                         subplot_titles=["ΔS(t) (Absolute Gap)", "Directional Surprisal I(t)", "Cumulative ΔRMST"])

    fig2.add_trace(go.Scatter(x=t, y=DeltaS, mode="lines", name="ΔS(t)", line=dict(color="black")), row=1, col=1)
    fig2.add_hline(y=0, line_dash="dash", row=1, col=1)

    # Directional Fill for Surprisal (Crucial for Version 1.2)
    fig2.add_trace(go.Scatter(x=t, y=np.where(I_t >= 0, I_t, 0), fill='tozeroy', 
                              fillcolor='rgba(0, 255, 0, 0.3)', line=dict(width=0), name="Vax Benefit"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=t, y=np.where(I_t < 0, I_t, 0), fill='tozeroy', 
                              fillcolor='rgba(255, 0, 0, 0.3)', line=dict(width=0), name="Vax Harm/Waning"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=t, y=I_t, mode="lines", line=dict(color="gray", width=1), showlegend=False), row=2, col=1)

    # RMST Comparison
    fig2.add_trace(go.Scatter(x=t, y=RMST_emp, mode="lines", line=dict(color="green"), name="ΔRMST Empirical"), row=3, col=1)
    fig2.add_trace(go.Scatter(x=t, y=RMST_w, mode="lines", line=dict(color="blue", dash="dot", width=2.5), name="ΔRMST Weighted"), row=3, col=1)

    for day in high_surprise_days:
        fig2.add_vline(x=day, line=dict(color="orange", width=0.5, dash="dot"), row=2, col=1)

    fig2.update_layout(title=f"Peircean RMST Diagnostics AG{age}", xaxis3_title="Days", template="plotly_white")
    fig2.write_html(OUTPUT_BASE.parent / f"{base_name}_rmst_AG{age}.html")
    log("RMST diagnostic plot saved.")
    
# =====================================================================
# ABDUCTION DIAGNOSTICS
# =====================================================================

def create_abduction_table(res: dict, agg: pd.DataFrame, t: np.ndarray, top_n: int = 12):
    # Use the signed I_t from the results
    I_t = res["I_t"]
    
    df = pd.DataFrame({
        'day': t,
        'surprisal_It': I_t,
        'p_value': np.exp(-np.abs(I_t)), # Reconstruction of p-value from I_t magnitude
        'DeltaS': res["DeltaS"]
    })
    
    # Label the direction of evidence
    df['Interpretation'] = np.where(df['surprisal_It'] > 0, 'Strong Vax Benefit', 'Evidence favoring Unvax')
    df.loc[np.abs(df['surprisal_It']) < SURPRISE_THRESHOLD, 'Interpretation'] = 'Low Evidence'

    # Sort by the magnitude of surprise
    top = df.iloc[np.argsort(np.abs(I_t))[::-1]].head(top_n).copy()
    top['calendar_date'] = (STUDY_START + pd.to_timedelta(top['day'], unit='D')).dt.strftime('%Y-%m-%d')
    
    return top[['calendar_date', 'surprisal_It', 'Interpretation', 'DeltaS']]


def generate_peircean_cluster_report(abduction_table: pd.DataFrame):
    """
    Summarizes the top abductive surprises by month and direction 
    to identify temporal 'nodes' of truth.
    """
    if abduction_table.empty:
        return "No significant surprises found to cluster."

    df = abduction_table.copy()
    # Ensure date format and create month column
    df['month'] = pd.to_datetime(df['calendar_date']).dt.to_period('M')
    
    # Group by month and the 'Interpretation' column we added in the previous step
    monthly = df.groupby(['month', 'Interpretation']).size().reset_index(name='surprise_count')
    
    report_text = "=== Monthly Peircean Cluster Analysis ===\n"
    report_text += monthly.to_string(index=False)
    return report_text

# =====================================================================
# MAIN
# =====================================================================

def main():
    log("=== Peircean RMST Analysis v1.2 – Full Directional Pipeline ===")

    # 1. Data Prep
    raw = load_raw(INPUT)
    clones, t, FIRST, last_obs = build_clones(raw)
    agg = aggregate(clones, t, FIRST, last_obs)

    # 2. Directional Analysis
    log("\nComputing Directional Peircean Core...")
    res = peircean_empirical_from_agg(agg, t)

    log(f"Final Empirical ΔRMST: {res['Delta_RMST_empirical'][-1]:.3f} days")
    log(f"Final Weighted ΔRMST:  {res['Delta_RMST_weighted'][-1]:.3f} days")

    # 3. Bootstrap (Optional: only if N_BOOT > 0)
    boot_res = None
    if N_BOOT > 0:
        log(f"\nRunning {N_BOOT} bootstrap iterations...")
        boot_res = bootstrap_peirce(raw)

    # 4. Refined Directional Plots
    log("\nGenerating Evidence-Weighted Diagnostic Plots...")
    plot_peircean(
        t, res["S_v_emp"], res["S_u_emp"], res["S_v_p"], res["S_u_p"],
        res["err_v"], res["err_u"], res["DeltaS"], res["I_t"],
        res["Delta_RMST_empirical"], res["Delta_RMST_weighted"],
        boot_res, OUTPUT_BASE, AGE
    )

    # 5. Abduction Table & Cluster Report
    log("\nGenerating Abduction Diagnostics...")
    tab = create_abduction_table(res, agg, t, top_n=15)
    
    log("\n[Top 15 Abductive Surprises]")
    log(tab.to_string(index=False))

    cluster_report = generate_peircean_cluster_report(tab)
    log("\n" + cluster_report)

    log("\n=== Analysis Complete. Check output folder for HTML diagnostics. ===")

if __name__ == "__main__":
    main()