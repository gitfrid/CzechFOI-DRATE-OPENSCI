#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Empirical Risk-Set Survival Analysis with ΔRMST

OVERVIEW
This script estimates the difference in restricted mean survival time
(ΔRMST) between vaccinated and unvaccinated individuals using a purely
empirical survival-analysis design.

Key features:
• Exact age stratification (no covariates)
• Risk-set expansion anchored at each vaccinated individual's dose date
• Kaplan–Meier survival estimation
• Step-function RMST integration
• Anchor-aware bootstrap inference
• Epidemiological summaries and diagnostic plots
• Fully non-parametric, no Cox models

Author: AI / drifting Date: 2025-12 Version: 4.0
"""

# =====================================================================
# IMPORTS
# =====================================================================
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from joblib import Parallel, delayed
import plotly.graph_objects as go

# =====================================================================
# CONFIGURATION
# =====================================================================
AGE = 70

#INPUT = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) empirical risk set expansion\AE) Vesely_106_202403141131_empirical-CCW_SIM")
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) empirical risk set expansion\AE) Vesely_106_202403141131_empirical-CCW")

STUDY_START = pd.Timestamp("2020-01-01")
REFERENCE_YEAR = 2023
HORIZON_DAYS = 700
UVX_SAMPLE_PER_VX = 50
SAFETY_BUFFER_DAYS = 30
N_BOOT = 30
RANDOM_SEED = 42
N_CORES = 4

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)

# =====================================================================
# LOGGING
# =====================================================================
log_path = OUTPUT_BASE.with_suffix(".txt")
log_fh = open(log_path, "w", encoding="utf-8")
def log(msg: str):
    print(msg)
    log_fh.write(msg + "\n")

# =====================================================================
# DATA PREPARATION
# =====================================================================
def read_and_prepare(path: Path) -> pd.DataFrame:
    """
    Load CSV, filter to age, and compute time scales in days.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Age filter
    df["birth_year"] = pd.to_numeric(df["rok_narozeni"], errors="coerce")
    df["age"] = REFERENCE_YEAR - df["birth_year"]
    df = df[df["age"] == AGE].copy()

    # Death date → day index
    df["death_day"] = (pd.to_datetime(df["datumumrti"], errors="coerce") - STUDY_START).dt.days

    # Vaccination dates
    dose_cols = [f"datum_{i}" for i in range(1, 8)]
    for c in dose_cols:
        if c in df.columns:
            df[c + "_day"] = (pd.to_datetime(df[c], errors="coerce") - STUDY_START).dt.days

    dose_day_cols = [c + "_day" for c in dose_cols if c + "_day" in df.columns]
    df["first_dose_day"] = df[dose_day_cols].min(axis=1, skipna=True)
    df["last_dose_day"] = df[dose_day_cols].max(axis=1, skipna=True)

    # Preserve original row index as person_id
    df["person_id"] = df.index
    return df

# =====================================================================
# STUDY WINDOW
# =====================================================================
def compute_study_window(df: pd.DataFrame):
    """
    Conservative study window avoiding right-edge artifacts.
    """
    last_dose = df["last_dose_day"].max(skipna=True)
    last_death = df["death_day"].max(skipna=True)
    end_day = int(min(last_dose, last_death) - SAFETY_BUFFER_DAYS)
    start_day = int(df["first_dose_day"].min(skipna=True))
    log(f"Study window: day {start_day} → {end_day}")
    return start_day, end_day
# =====================================================================
# RISK-SET EXPANSION
# =====================================================================
def riskset_expansion(df: pd.DataFrame):
    """
    Construct anchor-specific risk sets.

    Each vaccinated individual defines one anchor.
    Unvaccinated individuals may appear multiple times across anchors.
    """
    vx_df = df[df["first_dose_day"].notna()].reset_index(drop=True)
    uvx_df = df.reset_index(drop=True)

    INF = 1e9
    rng = np.random.default_rng(RANDOM_SEED)

    recs_vx, recs_uvx = [], []

    for anchor_id, t0 in enumerate(vx_df["first_dose_day"]):
        eligible = np.where(
            (uvx_df["death_day"].fillna(INF) > t0) &
            (uvx_df["first_dose_day"].fillna(INF) > t0)
        )[0]

        if len(eligible) == 0:
            continue

        sampled = rng.choice(
            eligible,
            size=min(len(eligible), UVX_SAMPLE_PER_VX),
            replace=False
        )

        vx_row = vx_df.iloc[anchor_id]
        end_vx = min(
            vx_row["death_day"] if pd.notna(vx_row["death_day"]) else INF,
            t0 + HORIZON_DAYS
        )

        recs_vx.append({
            "group": "VX",
            "follow_time": end_vx - t0,
            "event": pd.notna(vx_row["death_day"]) and vx_row["death_day"] <= end_vx,
            "anchor_id": anchor_id,
            "person_id": vx_row["person_id"]
        })

        for i in sampled:
            u = uvx_df.iloc[i]
            end_u = min(
                u["death_day"] if pd.notna(u["death_day"]) else INF,
                t0 + HORIZON_DAYS
            )

            recs_uvx.append({
                "group": "UVX",
                "follow_time": end_u - t0,
                "event": pd.notna(u["death_day"]) and u["death_day"] <= end_u,
                "anchor_id": anchor_id,
                "person_id": u["person_id"]
            })

    return pd.DataFrame(recs_vx), pd.DataFrame(recs_uvx)

# =====================================================================
# RMST / KM CALCULATIONS
# =====================================================================
def rmst_from_km(times, surv, tau):
    """
    Exact RMST for right-continuous KM step functions up to tau.
    """
    mask = times <= tau
    times = times[mask]
    surv = surv[mask]

    if len(times) < 2:
        return 0.0

    dt = np.diff(times)
    return float(np.sum(dt * surv[:-1]))

def km_and_delta_rmst(vx: pd.DataFrame, uvx: pd.DataFrame, tau: int):
    """
    Compute anchor-averaged ΔRMST between VX and UVX.
    """
    kmf = KaplanMeierFitter()

    kmf.fit(vx["follow_time"], vx["event"])
    t_vx = kmf.survival_function_.index.values
    s_vx = kmf.survival_function_["KM_estimate"].values

    kmf.fit(uvx["follow_time"], uvx["event"])
    t_uv = kmf.survival_function_.index.values
    s_uv = kmf.survival_function_["KM_estimate"].values

    rmst_vx = rmst_from_km(t_vx, s_vx, tau)
    rmst_uv = rmst_from_km(t_uv, s_uv, tau)

    return rmst_vx - rmst_uv, rmst_vx, rmst_uv

# =====================================================================
# EPIDEMIOLOGICAL SUMMARY
# =====================================================================
def epi_summary(vx: pd.DataFrame, uvx: pd.DataFrame, tau: int):
    """
    Print descriptive epidemiological statistics.
    """
    pt_vx = vx["follow_time"].sum()
    pt_uv = uvx["follow_time"].sum()
    ev_vx = vx["event"].sum()
    ev_uv = uvx["event"].sum()

    rate_vx = ev_vx / pt_vx
    rate_uv = ev_uv / pt_uv
    rr = rate_vx / rate_uv if rate_uv > 0 else np.nan
    ard = rate_uv - rate_vx
    nnt = tau / ard if ard > 0 else np.nan

    log("\n=== Epidemiological Summary ===")
    log(f"Person-time VX:  {pt_vx:.0f} days ({pt_vx/365:.1f} years)")
    log(f"Person-time UVX: {pt_uv:.0f} days ({pt_uv/365:.1f} years)")
    log(f"Deaths VX / UVX: {ev_vx} / {ev_uv}")
    log(f"Mortality rate VX:  {rate_vx:.6e}")
    log(f"Mortality rate UVX: {rate_uv:.6e}")
    log(f"Rate ratio (VX/UVX): {rr:.3f}")
    log(f"Absolute rate difference: {ard:.6e}")
    log(f"Approx. NNT over τ: {nnt:.1f}")
# =====================================================================
# ANCHOR-AWARE BOOTSTRAP FOR ΔRMST
# =====================================================================
def bootstrap_delta_rmst(vx: pd.DataFrame, uvx: pd.DataFrame, tau: int, n_boot=N_BOOT):
    """
    Resample anchors with replacement; for each, include its VX row and all matched UVX rows.
    Returns bootstrap distribution of ΔRMST.
    """
    uvx_groups = uvx.groupby("anchor_id", sort=False)
    anchor_ids = vx["anchor_id"].values
    n = len(anchor_ids)

    def one_boot(seed):
        rng = np.random.default_rng(seed)
        resampled = rng.integers(0, n, n)
        vx_b = vx.iloc[resampled].reset_index(drop=True)

        uvx_list = []
        for aid in vx_b["anchor_id"]:
            if aid in uvx_groups.groups:
                uvx_list.append(uvx_groups.get_group(aid))

        if not uvx_list:
            return np.nan

        uvx_b = pd.concat(uvx_list, ignore_index=True)
        d, _, _ = km_and_delta_rmst(vx_b, uvx_b, tau)
        return d

    seeds = RANDOM_SEED + np.arange(n_boot)
    boot = Parallel(n_jobs=N_CORES)(delayed(one_boot)(s) for s in seeds)
    return np.asarray(boot, dtype=float)

# =====================================================================
# BOOTSTRAP SURVIVAL CURVES / BANDS
# =====================================================================
def bootstrap_survival_curves(vx: pd.DataFrame, uvx: pd.DataFrame, t_grid, n_boot=300):
    """
    Compute pointwise bootstrap bands for VX and UVX survival curves.
    """
    uvx_groups = uvx.groupby("anchor_id", sort=False)
    anchor_ids = vx["anchor_id"].values
    n = len(anchor_ids)

    S_vx, S_uv = [], []
    kmf = KaplanMeierFitter()

    for b in range(n_boot):
        rng = np.random.default_rng(RANDOM_SEED + b)
        res = rng.integers(0, n, n)
        vx_b = vx.iloc[res].reset_index(drop=True)

        uvx_list = []
        for aid in vx_b["anchor_id"]:
            if aid in uvx_groups.groups:
                uvx_list.append(uvx_groups.get_group(aid))

        if not uvx_list:
            continue

        uvx_b = pd.concat(uvx_list, ignore_index=True)

        kmf.fit(vx_b["follow_time"], vx_b["event"])
        s_vx = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_["KM_estimate"])
        kmf.fit(uvx_b["follow_time"], uvx_b["event"])
        s_uv = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_["KM_estimate"])

        S_vx.append(s_vx)
        S_uv.append(s_uv)

    return np.array(S_vx), np.array(S_uv)

# =====================================================================
# REUSE INTENSITY DIAGNOSTICS
# =====================================================================
def reuse_intensity(uvx: pd.DataFrame):
    """
    How often each unvaccinated individual is reused across anchors.
    """
    reuse = uvx.groupby("person_id").size()
    log("\n=== Reuse Intensity Diagnostics ===")
    log(f"Unique UVX persons: {reuse.size}")
    log(f"Mean reuse count: {reuse.mean():.2f}")
    log(f"Median reuse count: {reuse.median():.0f}")
    log(f"95th percentile reuse: {reuse.quantile(0.95):.0f}")
    log(f"Max reuse count: {reuse.max():.0f}")
    return reuse

# =====================================================================
# EFFECTIVE SAMPLE SIZE (ESS)
# =====================================================================
def effective_sample_size(reuse_counts):
    """
    ESS = (sum w)^2 / sum(w^2)
    where w = reuse count per individual
    """
    w = reuse_counts.values.astype(float)
    ess = (w.sum() ** 2) / (w ** 2).sum()
    log(f"Effective sample size (UVX): {ess:.0f}")
    log(f"Nominal UVX rows: {w.sum():.0f}")
    return ess

# =====================================================================
# NEGATIVE CONTROL OUTCOME (POST-HORIZON)
# =====================================================================
def negative_control_post_horizon(vx, uvx, tau):
    """
    Deaths occurring AFTER tau should show ΔRMST ≈ 0.
    """
    vx_nc = vx.copy()
    uvx_nc = uvx.copy()
    vx_nc["event"] = vx_nc["follow_time"] > tau
    uvx_nc["event"] = uvx_nc["follow_time"] > tau
    d, _, _ = km_and_delta_rmst(vx_nc, uvx_nc, tau)
    log(f"Negative control ΔRMST (post-τ): {d:.3f} days")
    return d

# =====================================================================
# PLACEBO (PRE-DOSE) RMST
# =====================================================================
def placebo_pre_dose(df, horizon=180):
    """
    Shift anchor backwards in time; expect ΔRMST ≈ 0.
    """
    df_p = df.copy()
    df_p["first_dose_day"] = df_p["first_dose_day"] - horizon
    vx_p, uvx_p = riskset_expansion(df_p)
    d, _, _ = km_and_delta_rmst(vx_p, uvx_p, horizon)
    log(f"Placebo pre-dose ΔRMST: {d:.3f} days")
    return d

# =====================================================================
# SENSITIVITY TO UVX_SAMPLE_PER_VX
# =====================================================================
def sensitivity_uvx_sampling(df, values=(10, 25, 50, 100)):
    """
    Sensitivity of ΔRMST to number of UVX sampled per anchor.
    """
    out = {}
    global UVX_SAMPLE_PER_VX
    log("\n=== Sensitivity: UVX_SAMPLE_PER_VX ===")
    for v in values:
        UVX_SAMPLE_PER_VX = v
        vx_s, uvx_s = riskset_expansion(df)
        d, _, _ = km_and_delta_rmst(vx_s, uvx_s, HORIZON_DAYS)
        log(f"UVX_SAMPLE_PER_VX={v:3d} → ΔRMST={d:.3f}")
        out[v] = d
    return out
# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================

def plot_survival_with_ci(t_grid, s_vx, s_uv, S_vx_boot, S_uv_boot, delta_rmst, ci, output_html):
    """
    Kaplan–Meier survival curves with bootstrap CI ribbons and ΔRMST area.
    """
    vx_lo, vx_hi = np.percentile(S_vx_boot, [2.5, 97.5], axis=0)
    uv_lo, uv_hi = np.percentile(S_uv_boot, [2.5, 97.5], axis=0)

    fig = go.Figure()

    # CI ribbons
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([vx_hi, vx_lo[::-1]]),
        fill="toself",
        fillcolor="rgba(0,150,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="VX 95% CI"
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([uv_hi, uv_lo[::-1]]),
        fill="toself",
        fillcolor="rgba(0,0,200,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="UVX 95% CI"
    ))

    # Survival curves
    fig.add_trace(go.Scatter(x=t_grid, y=s_vx, name="Vaccinated", line=dict(color="green", width=2)))
    fig.add_trace(go.Scatter(x=t_grid, y=s_uv, name="Unvaccinated", line=dict(color="blue", width=2)))

    # ΔRMST shaded area
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_grid, t_grid[::-1]]),
        y=np.concatenate([s_vx, s_uv[::-1]]),
        fill="toself",
        fillcolor="rgba(0,200,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="ΔRMST"
    ))

    fig.add_annotation(
        text=f"ΔRMST = {delta_rmst:.2f} days<br>95% CI: {ci[0]:.2f}–{ci[1]:.2f}",
        xref="paper", yref="paper", x=0.02, y=0.02,
        showarrow=False, bgcolor="rgba(255,255,255,0.9)", bordercolor="black"
    )

    fig.update_layout(
        title=f"Kaplan–Meier Survival (AGE={AGE})",
        xaxis_title="Days since index (anchor)",
        yaxis_title="Survival probability",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white"
    )

    fig.write_html(output_html)
    log(f"Survival plot saved: {output_html}")


def plot_reuse_intensity(reuse_counts, output_html):
    """
    Histogram of how often unvaccinated individuals are reused.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=reuse_counts.values, nbinsx=50, marker_color="gray"))
    fig.update_layout(
        title="Reuse Intensity of Unvaccinated Individuals",
        xaxis_title="Number of times reused",
        yaxis_title="Number of individuals",
        template="plotly_white"
    )
    fig.write_html(output_html)
    log(f"Reuse intensity plot saved: {output_html}")


# =====================================================================
# PUBLICATION-READY FIGURE CAPTION
# =====================================================================
FIGURE_CAPTION = f"""
Figure: Anchor-averaged Kaplan–Meier survival curves for vaccinated (green)
and unvaccinated (blue) individuals aged {AGE} years.
Risk sets are constructed at each vaccinated individual's dose date,
with unvaccinated controls sampled from those alive and unvaccinated
at that time. Shaded ribbons denote 95% bootstrap confidence intervals.
ΔRMST is shown as the shaded area between curves. Analyses are purely
empirical and non-parametric; no regression models or covariate adjustment.
"""

# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    log("=== Loading data ===")
    df = read_and_prepare(INPUT)
    log(f"Subjects after age filter: {len(df)}")

    compute_study_window(df)

    log("=== Constructing risk sets ===")
    vx, uvx = riskset_expansion(df)
    log(f"Risk-set rows: VX={len(vx)}, UVX={len(uvx)}")

    t_grid = np.arange(0, HORIZON_DAYS + 1)

    # Point estimate ΔRMST
    delta_rmst, rmst_vx, rmst_uv = km_and_delta_rmst(vx, uvx, HORIZON_DAYS)
    log(f"ΔRMST = {delta_rmst:.3f} days")

    # Bootstrap ΔRMST
    boot = bootstrap_delta_rmst(vx, uvx, HORIZON_DAYS)
    boot = boot[np.isfinite(boot)]
    ci = np.percentile(boot, [2.5, 97.5])
    log(f"95% CI: {ci[0]:.3f}–{ci[1]:.3f}")

    # Bootstrap survival curves
    S_vx, S_uv = bootstrap_survival_curves(vx, uvx, t_grid)

    # Point survival curves
    kmf = KaplanMeierFitter()
    kmf.fit(vx["follow_time"], vx["event"])
    s_vx = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_["KM_estimate"])

    kmf.fit(uvx["follow_time"], uvx["event"])
    s_uv = np.interp(t_grid, kmf.survival_function_.index, kmf.survival_function_["KM_estimate"])

    # Epidemiology
    epi_summary(vx, uvx, HORIZON_DAYS)

    # Diagnostics
    reuse = reuse_intensity(uvx)
    effective_sample_size(reuse)
    negative_control_post_horizon(vx, uvx, HORIZON_DAYS)
    placebo_pre_dose(df)
    sensitivity_uvx_sampling(df)

    # Plots
    plot_survival_with_ci(
        t_grid, s_vx, s_uv, S_vx, S_uv, delta_rmst, ci,
        OUTPUT_BASE.with_name("survival_plot.html")
    )
    plot_reuse_intensity(
        reuse,
        OUTPUT_BASE.with_name("reuse_intensity.html")
    )

    log("\n=== FIGURE CAPTION ===")
    log(FIGURE_CAPTION)

    log("=== Analysis complete ===")
    log_fh.close()


if __name__ == "__main__":
    main()
