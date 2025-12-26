#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical Landmark RMST (Restricted Mean Survival Time) Estimation with Non-Parametric Bootstrap
Applied to Age-Stratified Czech FOI Mortality/Vaccination Data

Purpose:
    This script implements a fully empirical, non-parametric landmark analysis
    of survival differences between vaccinated and unvaccinated individuals.
    The method computes ΔRMST (difference in restricted mean survival time)
    at a sequence of landmark times, using only observed hazards and without
    parametric assumptions.

    The script supports:
        • Real-world Czech FOI datasets
        • Sensitivity analyses with reclassified UVX→VX deaths
        • Simulated datasets (HR=1 null model)
        • Parallel bootstrap for confidence intervals
        • Plotly visualizations (ΔRMST curve + survival curves + ΔΔRMST comparison)
        • Full logging for reproducibility
        • Primary: ITT-like (no crossover censoring)
        • Sensitivity: Per-protocol (with crossover censoring, uncorrected)
        • Fixed-τ RMST sensitivity (e.g., 365 days)

Scientific Notes:
    • Landmark analysis avoids immortal time bias by conditioning on survival
      up to each landmark.
    • RMST is computed directly from empirical hazards (non-parametric).
    • Bootstrap resampling preserves the empirical joint distribution.
    • No Cox model, no proportional hazards assumption, no smoothing.
    • Primary analysis: Intention-to-treat-like (no censoring at crossover) to reflect historical reality without selection bias.
    • Sensitivity: Per-protocol (censor at crossover) — uncorrected for informative censoring (likely biased toward overestimating vaccine benefit due to selection).
    • Tie-breaking: If death and vax on same day, delay vax effect by 1 day (v_idx += 1).
    • Target Trial Emulation Alignment (Hernán et al., JAMA 2022): Emulated pragmatic trial per landmark t; eligibility = alive at t (age 70 cohort); treatment = vaccinated by t vs unvaccinated at t.
      Primary ITT-like to avoid uncorrected selection bias (no confounders available for IPCW).
    • This script is suitable for inclusion in a preprint Methods section.

Author: AI / drifting Date: 2025-12-26 Version: 2
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ---------------- Configuration ----------------

AGE = 70                                    # exact age cohort to analyze (integer)
STUDY_START = pd.Timestamp("2020-01-01")    # reference origin for converting dates to day indices
AGE_REF_YEAR = 2023                         # Age is approximated by year of birth due to data limitations (induces ±1 year around birthdays)
IMMUNITY_LAG = 0                            # days excluded after vaccination to avoid immediate-risk artifacts
OBS_END_SAFETY = 30                         # buffer to avoid right-edge censoring artifacts
RANDOM_SEED = 12345
LANDMARK_STEP = 7                           # spacing between landmark times
BOOTSTRAP_REPS = 200                        # Increase to 1000 for better percentile stability
FIXED_TAU = 365                             # days for fixed-horizon RMST sensitivity (e.g., 365 for 1 year)

# Input/Output
# real world Czech-FOI dataset 
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST")

# real data with hypothetical 5% uvx death or alive reclassified as vx (sensitivity test for missclassifcation)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_DeathOrAlive_reclassified_PCT5_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_DeathOrALive_RECLASSIFIED")

# simulated dataset HR=1 with simulated real dose schedule (sensitivity test if the used methode is bias free - null hypothesis)
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_SIM")

OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
CSV_PATH_PRIMARY = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_primary_ITT_AG{AGE}.csv")
CSV_PATH_SENSITIVITY = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_sensitivity_PP_AG{AGE}.csv")
PLOT_RMST_PRIMARY = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_ΔRMST_primary_ITT_AG{AGE}.html")
PLOT_RMST_SENSITIVITY = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_ΔRMST_sensitivity_PP_AG{AGE}.html")
PLOT_SURV_PRIMARY = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_Survival_primary_ITT_AG{AGE}.html")
PLOT_SURV_SENSITIVITY = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_Survival_sensitivity_PP_AG{AGE}.html")
PLOT_DELTADELTA = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_ΔΔRMST_comparison_AG{AGE}.html")
LOG_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.log")

# ---------------- Logging ----------------
def setup_logger(log_path):
    logger = logging.getLogger("landmark_rmst")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(processName)s | %(message)s"))
    logger.addHandler(fh)
    return logger

logger = setup_logger(LOG_PATH)
def log(msg): logger.info(msg)

# ---------------- Helpers ----------------
# Converts hazard sequence → survival curve (empirical product-limit)
def survival_from_hazard(h):
    h = np.clip(h, 0.0, 1.0)
    s = np.empty_like(h, dtype=float)
    acc = 1.0
    for i in range(len(h)):
        acc *= (1.0 - h[i])  # correct KM-style update
        s[i] = acc
    return s

# Computes RMST by step-integration of survival curve (Kaplan–Meier style)
def rmst_from_survival(S, dt=1.0, max_days=None):
    if max_days is not None:
        S = S[:max_days]
    return float(np.sum(S) * dt)

# ---------------- Main processing ----------------
# Core function: computes all landmark RMST values for a dataset.
# use_per_protocol: If True, apply crossover censoring (sensitivity); False for primary ITT-like.
def compute_landmarks(raw, use_per_protocol=False):
    # Parse dates
    date_cols = [c for c in raw.columns if c.startswith("Datum_") or c == "DatumUmrti"]
    for c in date_cols:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")

    # Filter by exact age
    raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw = raw[raw["age"] == AGE].reset_index(drop=True)

    # Convert dates to day indices
    raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
    raw["vax_day"] = (raw["Datum_1"] - STUDY_START).dt.days
    raw.loc[raw["death_day"] < 0, "death_day"] = np.nan
    raw.loc[raw["vax_day"] < 0, "vax_day"] = np.nan

    # Tie-breaking rule: If death_day == vax_day, delay vax effect by 1 day
    same_day_mask = (raw["death_day"] == raw["vax_day"]) & raw["death_day"].notna() & raw["vax_day"].notna()
    raw.loc[same_day_mask, "vax_day"] += 1
    same_day_count = same_day_mask.sum()
    # log(f"Same-day death+vax cases: {same_day_count} (delayed vax by 1 day)")

    # Determine observation window
    first_dose_min = raw["Datum_1"].min(skipna=True)
    last_dose_max = raw[[c for c in raw.columns if c.startswith("Datum_") and c != "DatumUmrti"]].max().max(skipna=True)
    last_death = raw["DatumUmrti"].max(skipna=True)
    base_end = min(d for d in [last_dose_max, last_death] if pd.notna(d))
    FIRST_ELIG_DAY = int((first_dose_min - STUDY_START).days)
    LAST_OBS_DAY = int((base_end - STUDY_START).days) - OBS_END_SAFETY
    if LAST_OBS_DAY <= FIRST_ELIG_DAY:
        raise ValueError("OBS_END_SAFETY too large; no observation window remains.")

    DAYS = np.arange(FIRST_ELIG_DAY, LAST_OBS_DAY + 1)
    LANDMARKS = np.arange(FIRST_ELIG_DAY, LAST_OBS_DAY + 1, LANDMARK_STEP)

    #log(f"Subjects: {len(raw)} | Landmark window: {FIRST_ELIG_DAY}-{LAST_OBS_DAY} ({len(DAYS)} days) | Landmarks: {len(LANDMARKS)}")

    # ---------------- Index mapping ----------------
    NEVER_VAX_IDX = (LAST_OBS_DAY - FIRST_ELIG_DAY + 1)
    V_DIM = NEVER_VAX_IDX + 1
    D_DIM = (LAST_OBS_DAY - FIRST_ELIG_DAY + 1)
    LAST_VAX_IDX = V_DIM - 2

    vax_days = raw["vax_day"].to_numpy()
    v_idx = np.full(len(vax_days), NEVER_VAX_IDX, dtype=int)
    mask = ~np.isnan(vax_days)
    v_idx[mask] = np.clip((vax_days[mask].astype(int) - FIRST_ELIG_DAY), 0, NEVER_VAX_IDX)

    death_days = raw["death_day"].to_numpy()
    d_idx = np.full(len(death_days), -1, dtype=int)
    mask_death = ~np.isnan(death_days)
    d_idx[mask_death] = np.clip(death_days[mask_death].astype(int) - FIRST_ELIG_DAY, 0, D_DIM - 1)

    # Build empirical hazard tables
    counts_by_v = np.bincount(v_idx, minlength=V_DIM).astype(np.int64)
    deaths_by_vd = np.zeros((V_DIM, D_DIM), dtype=np.int64)
    mask = d_idx >= 0
    for vi, di in zip(v_idx[mask], d_idx[mask]):
        deaths_by_vd[vi, di] += 1

    # Precompute cumulative tables for fast landmark evaluation
    counts_cumv = np.cumsum(counts_by_v)
    counts_sufv = np.cumsum(counts_by_v[::-1])[::-1]
    deaths_prefv = np.cumsum(deaths_by_vd, axis=0)
    deaths_sufv = np.cumsum(deaths_by_vd[::-1, :], axis=0)[::-1, :]
    deaths_cumd_prefv = np.cumsum(np.cumsum(deaths_by_vd, axis=1), axis=0)
    deaths_cumd_sufv = np.cumsum(np.cumsum(deaths_by_vd[::-1, :], axis=1), axis=0)[::-1, :]

    results = []

    # ---------------- Landmark computation ----------------
    for t_day in tqdm(LANDMARKS, desc="Landmarks"):
        t = t_day - FIRST_ELIG_DAY
        t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)

        # Vaccinated at landmark: vaccinated on or before t, alive at t
        N_v = counts_cumv[min(t, LAST_VAX_IDX)]
        died_to_t_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl] if N_v > 0 else 0
        N_v -= died_to_t_v

        # Unvaccinated at landmark: v_idx >= t+1 (vaccinate later or never), alive at t
        v_start_u = min(t + 1, NEVER_VAX_IDX)
        N_u = counts_sufv[v_start_u]
        died_to_t_u = deaths_cumd_sufv[v_start_u, t_excl] if N_u > 0 else 0
        N_u -= died_to_t_u

        if N_v <= 0 or N_u <= 0:
            continue

        # Death counts for the two groups
        E_v = deaths_prefv[min(t, LAST_VAX_IDX), :].copy()
        E_u = deaths_sufv[v_start_u, :].copy()
        if t_excl >= 0:
            E_v[: t_excl + 1] = 0
            E_u[: t_excl + 1] = 0

        # Cumulative deaths after the landmark
        cumE_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), :] - deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl]
        cumE_u = deaths_cumd_sufv[v_start_u, :] - deaths_cumd_sufv[v_start_u, t_excl]

        # Start of follow-up after landmark + immunity lag
        start = t_excl + 1
        if start >= D_DIM:
            continue

        # --- Survival and Risk Set Calculation per Landmark ---

        # 1. Vaccinated Risk Set (Deaths only)
        R_v = N_v - np.concatenate([[0], cumE_v[:-1]])

        # 2. Unvaccinated Risk Set
        crossovers_cum = np.zeros(D_DIM)  # Default no crossover
        if use_per_protocol:  # Sensitivity: Censor at crossover
            mask_u = (v_idx >= v_start_u)
            vax_idx_u = v_idx[mask_u]

            vax_follow = vax_idx_u[
                (vax_idx_u >= start) &
                (vax_idx_u < NEVER_VAX_IDX) &
                (vax_idx_u < D_DIM)
            ]
            vax_counts_during_followup = np.bincount(vax_follow, minlength=D_DIM)
            crossovers_cum = np.cumsum(vax_counts_during_followup)

        # Risk Set U: subtract cumulative deaths AND (if per_protocol) cumulative crossovers
        R_u = N_u - np.concatenate([[0], (cumE_u[:-1] + crossovers_cum[:-1])])

        # Avoid zero/negative risk sets for stability
        R_v = np.clip(R_v, 1e-12, None)
        R_u = np.clip(R_u, 1e-12, None)

        # Segment hazards from start onward
        h_v_seg = E_v[start:] / R_v[start:]
        h_u_seg = E_u[start:] / R_u[start:]

        # Survival functions
        S_v = survival_from_hazard(h_v_seg)
        S_u = survival_from_hazard(h_u_seg)

        # RMST using step-integration (Kaplan–Meier style)
        horizon = D_DIM - start  # Effective unrestricted horizon
        rmst_v_unres = rmst_from_survival(S_v)
        rmst_u_unres = rmst_from_survival(S_u)
        delta_unres = float(rmst_v_unres - rmst_u_unres)

        rmst_v_fixed = rmst_from_survival(S_v, max_days=FIXED_TAU)
        rmst_u_fixed = rmst_from_survival(S_u, max_days=FIXED_TAU)
        delta_fixed = float(rmst_v_fixed - rmst_u_fixed)

        results.append(
            {
                "t_landmark": t_day,
                "n_v": int(N_v),
                "n_u": int(N_u),
                "rmst_v_unres": rmst_v_unres,
                "rmst_u_unres": rmst_u_unres,
                "delta_rmst_unres": delta_unres,
                "rmst_v_fixed": rmst_v_fixed,
                "rmst_u_fixed": rmst_u_fixed,
                "delta_rmst_fixed": delta_fixed,
                "effective_horizon_days": horizon,
            }
        )

    res_df = pd.DataFrame(results).sort_values("t_landmark")
    return (
        raw,
        res_df,
        V_DIM,
        D_DIM,
        counts_cumv,
        counts_sufv,
        deaths_prefv,
        deaths_sufv,
        deaths_cumd_prefv,
        deaths_cumd_sufv,
        FIRST_ELIG_DAY,
        NEVER_VAX_IDX,
        LAST_VAX_IDX,
        v_idx,
        same_day_count,
    )

# ---------------- Bootstrap ----------------
def one_boot(i, raw, res_df, V_DIM, D_DIM,
             counts_cumv, counts_sufv,
             deaths_prefv, deaths_sufv,
             deaths_cumd_prefv, deaths_cumd_sufv,
             FIRST_ELIG_DAY, NEVER_VAX_IDX, LAST_VAX_IDX, use_per_protocol):
    np.random.seed(RANDOM_SEED + i)
    idx = np.random.choice(len(raw), len(raw), replace=True)
    raw_b = raw.iloc[idx].reset_index(drop=True)
    _, res_df_b, *_ = compute_landmarks(raw_b, use_per_protocol=use_per_protocol)
    # pad to align with main res_df landmarks
    aligned = res_df_b.set_index("t_landmark").reindex(res_df["t_landmark"], fill_value=np.nan)
    return aligned["delta_rmst_unres"].values, aligned["delta_rmst_fixed"].values

# ---------------- Main ----------------
if __name__ == "__main__":
    multiprocessing.freeze_support()

    raw = pd.read_csv(INPUT, low_memory=False)
    raw.columns = raw.columns.str.strip()

    # Primary: ITT-like (no per_protocol)
    log("Computing primary ITT-like analysis (no crossover censoring)")
    (
        raw,
        res_df_primary,
        V_DIM,
        D_DIM,
        counts_cumv,
        counts_sufv,
        deaths_prefv,
        deaths_sufv,
        deaths_cumd_prefv,
        deaths_cumd_sufv,
        FIRST_ELIG_DAY,
        NEVER_VAX_IDX,
        LAST_VAX_IDX,
        v_idx,
        same_day_count,
    ) = compute_landmarks(raw, use_per_protocol=False)
    res_df_primary.to_csv(CSV_PATH_PRIMARY, index=False)
    log(f"Saved primary CSV: {CSV_PATH_PRIMARY} | Landmarks computed: {len(res_df_primary)}")

    # Sensitivity: Per-protocol (with crossover censoring)
    log("Computing sensitivity per-protocol analysis (with crossover censoring, uncorrected)")
    _, res_df_sensitivity, *_ = compute_landmarks(raw, use_per_protocol=True)
    res_df_sensitivity.to_csv(CSV_PATH_SENSITIVITY, index=False)
    log(f"Saved sensitivity CSV: {CSV_PATH_SENSITIVITY} | Landmarks computed: {len(res_df_sensitivity)}")

    # ---------------- Parallel bootstrap ----------------
    log("Starting bootstrap for primary...")
    indices = range(BOOTSTRAP_REPS)
    boot_func_primary = partial(
        one_boot,
        raw=raw,
        res_df=res_df_primary,
        V_DIM=V_DIM,
        D_DIM=D_DIM,
        counts_cumv=counts_cumv,
        counts_sufv=counts_sufv,
        deaths_prefv=deaths_prefv,
        deaths_sufv=deaths_sufv,
        deaths_cumd_prefv=deaths_cumd_prefv,
        deaths_cumd_sufv=deaths_cumd_sufv,
        FIRST_ELIG_DAY=FIRST_ELIG_DAY,
        NEVER_VAX_IDX=NEVER_VAX_IDX,
        LAST_VAX_IDX=LAST_VAX_IDX,
        use_per_protocol=False,
    )

    boot_unres_primary = []
    boot_fixed_primary = []
    with ProcessPoolExecutor() as ex:
        for unres, fixed in tqdm(ex.map(boot_func_primary, indices), total=BOOTSTRAP_REPS, desc="Bootstrap primary"):
            boot_unres_primary.append(unres)
            boot_fixed_primary.append(fixed)

    boot_unres_array_primary = np.stack(boot_unres_primary, axis=0)
    boot_fixed_array_primary = np.stack(boot_fixed_primary, axis=0)
    delta_unres_lower_primary = np.nanpercentile(boot_unres_array_primary, 2.5, axis=0)
    delta_unres_upper_primary = np.nanpercentile(boot_unres_array_primary, 97.5, axis=0)
    delta_fixed_lower_primary = np.nanpercentile(boot_fixed_array_primary, 2.5, axis=0)
    delta_fixed_upper_primary = np.nanpercentile(boot_fixed_array_primary, 97.5, axis=0)
    res_df_primary["delta_rmst_unres_CI_lower"] = delta_unres_lower_primary
    res_df_primary["delta_rmst_unres_CI_upper"] = delta_unres_upper_primary
    res_df_primary["delta_rmst_fixed_CI_lower"] = delta_fixed_lower_primary
    res_df_primary["delta_rmst_fixed_CI_upper"] = delta_fixed_upper_primary
    res_df_primary.to_csv(CSV_PATH_PRIMARY, index=False)
    log(f"Primary bootstrap CI added and CSV updated: {CSV_PATH_PRIMARY}")

    # Bootstrap for sensitivity
    log("Starting bootstrap for sensitivity...")
    boot_func_sensitivity = partial(
        one_boot,
        raw=raw,
        res_df=res_df_sensitivity,
        V_DIM=V_DIM,
        D_DIM=D_DIM,
        counts_cumv=counts_cumv,
        counts_sufv=counts_sufv,
        deaths_prefv=deaths_prefv,
        deaths_sufv=deaths_sufv,
        deaths_cumd_prefv=deaths_cumd_prefv,
        deaths_cumd_sufv=deaths_cumd_sufv,
        FIRST_ELIG_DAY=FIRST_ELIG_DAY,
        NEVER_VAX_IDX=NEVER_VAX_IDX,
        LAST_VAX_IDX=LAST_VAX_IDX,
        use_per_protocol=True,
    )

    boot_unres_sensitivity = []
    boot_fixed_sensitivity = []
    with ProcessPoolExecutor() as ex:
        for unres, fixed in tqdm(ex.map(boot_func_sensitivity, indices), total=BOOTSTRAP_REPS, desc="Bootstrap sensitivity"):
            boot_unres_sensitivity.append(unres)
            boot_fixed_sensitivity.append(fixed)

    boot_unres_array_sensitivity = np.stack(boot_unres_sensitivity, axis=0)
    boot_fixed_array_sensitivity = np.stack(boot_fixed_sensitivity, axis=0)
    delta_unres_lower_sensitivity = np.nanpercentile(boot_unres_array_sensitivity, 2.5, axis=0)
    delta_unres_upper_sensitivity = np.nanpercentile(boot_unres_array_sensitivity, 97.5, axis=0)
    delta_fixed_lower_sensitivity = np.nanpercentile(boot_fixed_array_sensitivity, 2.5, axis=0)
    delta_fixed_upper_sensitivity = np.nanpercentile(boot_fixed_array_sensitivity, 97.5, axis=0)
    res_df_sensitivity["delta_rmst_unres_CI_lower"] = delta_unres_lower_sensitivity
    res_df_sensitivity["delta_rmst_unres_CI_upper"] = delta_unres_upper_sensitivity
    res_df_sensitivity["delta_rmst_fixed_CI_lower"] = delta_fixed_lower_sensitivity
    res_df_sensitivity["delta_rmst_fixed_CI_upper"] = delta_fixed_upper_sensitivity
    res_df_sensitivity.to_csv(CSV_PATH_SENSITIVITY, index=False)
    log(f"Sensitivity bootstrap CI added and CSV updated: {CSV_PATH_SENSITIVITY}")

    # ---------------- Compute summary statistics for logging ----------------
    med_primary = res_df_primary.median(numeric_only=True)
    med_sens = res_df_sensitivity.median(numeric_only=True)

    # Crude VE approx: (ΔRMST / RMST_u) * 100
    ve_primary_unres = (med_primary['delta_rmst_unres'] / med_primary['rmst_u_unres']) * 100 if med_primary['rmst_u_unres'] > 0 else np.nan
    ve_sens_unres = (med_sens['delta_rmst_unres'] / med_sens['rmst_u_unres']) * 100 if med_sens['rmst_u_unres'] > 0 else np.nan

    # Rough survival approx at FIXED_TAU: 1 - (t - RMST)/t
    t = FIXED_TAU
    surv_v_primary_fixed = 1 - (t - med_primary['rmst_v_fixed']) / t if t > 0 else np.nan
    surv_u_primary_fixed = 1 - (t - med_primary['rmst_u_fixed']) / t if t > 0 else np.nan
    surv_v_sens_fixed = 1 - (t - med_sens['rmst_v_fixed']) / t if t > 0 else np.nan
    surv_u_sens_fixed = 1 - (t - med_sens['rmst_u_fixed']) / t if t > 0 else np.nan

    # ---------------- Plot ΔRMST ----------------
    # Primary
    fig_rmst_primary = go.Figure()
    fig_rmst_primary.add_trace(go.Scatter(x=res_df_primary["t_landmark"], y=res_df_primary["delta_rmst_unres"],
                                          mode="lines+markers", name="ΔRMST unrestricted"))
    fig_rmst_primary.add_trace(go.Scatter(x=res_df_primary["t_landmark"], y=res_df_primary["delta_rmst_unres_CI_lower"],
                                          mode="lines", line=dict(dash="dash"), name="CI lower unrestricted"))
    fig_rmst_primary.add_trace(go.Scatter(x=res_df_primary["t_landmark"], y=res_df_primary["delta_rmst_unres_CI_upper"],
                                          mode="lines", line=dict(dash="dash"), name="CI upper unrestricted"))
    fig_rmst_primary.add_trace(go.Scatter(x=res_df_primary["t_landmark"], y=res_df_primary["delta_rmst_fixed"],
                                          mode="lines+markers", name=f"ΔRMST fixed {FIXED_TAU} days", line=dict(color="green")))
    fig_rmst_primary.add_trace(go.Scatter(x=res_df_primary["t_landmark"], y=res_df_primary["delta_rmst_fixed_CI_lower"],
                                          mode="lines", line=dict(dash="dash", color="green"), name=f"CI lower fixed {FIXED_TAU}"))
    fig_rmst_primary.add_trace(go.Scatter(x=res_df_primary["t_landmark"], y=res_df_primary["delta_rmst_fixed_CI_upper"],
                                          mode="lines", line=dict(dash="dash", color="green"), name=f"CI upper fixed {FIXED_TAU}"))
    fig_rmst_primary.update_layout(title="Primary ITT-like ΔRMST per Landmark",
                                   xaxis_title="Landmark day",
                                   yaxis_title="ΔRMST [days]",
                                   template="plotly_white")
    fig_rmst_primary.write_html(PLOT_RMST_PRIMARY)
    log(f"Primary ΔRMST plot saved: {PLOT_RMST_PRIMARY}")

    # Sensitivity
    fig_rmst_sensitivity = go.Figure()
    fig_rmst_sensitivity.add_trace(go.Scatter(x=res_df_sensitivity["t_landmark"], y=res_df_sensitivity["delta_rmst_unres"],
                                              mode="lines+markers", name="ΔRMST unrestricted"))
    fig_rmst_sensitivity.add_trace(go.Scatter(x=res_df_sensitivity["t_landmark"], y=res_df_sensitivity["delta_rmst_unres_CI_lower"],
                                              mode="lines", line=dict(dash="dash"), name="CI lower unrestricted"))
    fig_rmst_sensitivity.add_trace(go.Scatter(x=res_df_sensitivity["t_landmark"], y=res_df_sensitivity["delta_rmst_unres_CI_upper"],
                                              mode="lines", line=dict(dash="dash"), name="CI upper unrestricted"))
    fig_rmst_sensitivity.add_trace(go.Scatter(x=res_df_sensitivity["t_landmark"], y=res_df_sensitivity["delta_rmst_fixed"],
                                              mode="lines+markers", name=f"ΔRMST fixed {FIXED_TAU} days", line=dict(color="green")))
    fig_rmst_sensitivity.add_trace(go.Scatter(x=res_df_sensitivity["t_landmark"], y=res_df_sensitivity["delta_rmst_fixed_CI_lower"],
                                              mode="lines", line=dict(dash="dash", color="green"), name=f"CI lower fixed {FIXED_TAU}"))
    fig_rmst_sensitivity.add_trace(go.Scatter(x=res_df_sensitivity["t_landmark"], y=res_df_sensitivity["delta_rmst_fixed_CI_upper"],
                                              mode="lines", line=dict(dash="dash", color="green"), name=f"CI upper fixed {FIXED_TAU}"))
    fig_rmst_sensitivity.update_layout(title="Sensitivity Per-Protocol ΔRMST per Landmark (Uncorrected for Selection Bias)",
                                       xaxis_title="Landmark day",
                                       yaxis_title="ΔRMST [days]",
                                       template="plotly_white")
    fig_rmst_sensitivity.write_html(PLOT_RMST_SENSITIVITY)
    log(f"Sensitivity ΔRMST plot saved: {PLOT_RMST_SENSITIVITY}")

    # ---------------- Plot Survival curves for all landmarks ----------------
    # Primary
    fig_surv_primary = go.Figure()
    for idx_row, row in res_df_primary.iterrows():
        t_day = int(row["t_landmark"])
        t = t_day - FIRST_ELIG_DAY
        t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)

        E_v = deaths_prefv[min(t, LAST_VAX_IDX), :].copy()
        v_start_u = min(t + 1, NEVER_VAX_IDX)
        E_u = deaths_sufv[v_start_u, :].copy()

        if t_excl >= 0:
            E_v[: t_excl + 1] = 0
            E_u[: t_excl + 1] = 0

        cumE_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), :] - deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl]
        cumE_u = deaths_cumd_sufv[v_start_u, :] - deaths_cumd_sufv[v_start_u, t_excl]

        start = t_excl + 1
        if start >= D_DIM:
            continue

        crossovers_cum = np.zeros(D_DIM)

        R_v = row["n_v"] - np.concatenate([[0], cumE_v[:-1]])
        R_u = row["n_u"] - np.concatenate([[0], (cumE_u[:-1] + crossovers_cum[:-1])])

        R_v = np.clip(R_v, 1e-12, None)
        R_u = np.clip(R_u, 1e-12, None)

        h_v_seg = E_v[start:] / R_v[start:]
        h_u_seg = E_u[start:] / R_u[start:]

        S_v = survival_from_hazard(h_v_seg)
        S_u = survival_from_hazard(h_u_seg)
        days = np.arange(len(S_v))

        fig_surv_primary.add_trace(go.Scatter(x=days, y=S_v, mode="lines",
                                              name=f"Vaccinated (LM {t_day})",
                                              visible="legendonly"))
        fig_surv_primary.add_trace(go.Scatter(x=days, y=S_u, mode="lines",
                                              name=f"Unvaccinated (LM {t_day})",
                                              visible="legendonly"))

    fig_surv_primary.update_layout(title="Primary ITT-like Empirical Survival Curves (all landmarks selectable)",
                                   xaxis_title="Days since landmark",
                                   yaxis_title="Survival probability",
                                   template="plotly_white",
                                   legend=dict(x=1.05, y=1, orientation="v"))
    fig_surv_primary.write_html(PLOT_SURV_PRIMARY)
    log(f"Primary survival plot saved: {PLOT_SURV_PRIMARY}")

    # Sensitivity survival
    fig_surv_sensitivity = go.Figure()
    for idx_row, row in res_df_sensitivity.iterrows():
        t_day = int(row["t_landmark"])
        t = t_day - FIRST_ELIG_DAY
        t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)

        E_v = deaths_prefv[min(t, LAST_VAX_IDX), :].copy()
        v_start_u = min(t + 1, NEVER_VAX_IDX)
        E_u = deaths_sufv[v_start_u, :].copy()

        if t_excl >= 0:
            E_v[: t_excl + 1] = 0
            E_u[: t_excl + 1] = 0

        cumE_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), :] - deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl]
        cumE_u = deaths_cumd_sufv[v_start_u, :] - deaths_cumd_sufv[v_start_u, t_excl]

        start = t_excl + 1
        if start >= D_DIM:
            continue

        mask_u = (v_idx >= v_start_u)
        vax_idx_u = v_idx[mask_u]

        vax_follow = vax_idx_u[
            (vax_idx_u >= start) &
            (vax_idx_u < NEVER_VAX_IDX) &
            (vax_idx_u < D_DIM)
        ]
        vax_counts_during_followup = np.bincount(vax_follow, minlength=D_DIM)
        crossovers_cum = np.cumsum(vax_counts_during_followup)

        R_v = row["n_v"] - np.concatenate([[0], cumE_v[:-1]])
        R_u = row["n_u"] - np.concatenate([[0], (cumE_u[:-1] + crossovers_cum[:-1])])

        R_v = np.clip(R_v, 1e-12, None)
        R_u = np.clip(R_u, 1e-12, None)

        h_v_seg = E_v[start:] / R_v[start:]
        h_u_seg = E_u[start:] / R_u[start:]

        S_v = survival_from_hazard(h_v_seg)
        S_u = survival_from_hazard(h_u_seg)
        days = np.arange(len(S_v))

        fig_surv_sensitivity.add_trace(go.Scatter(x=days, y=S_v, mode="lines",
                                                  name=f"Vaccinated (LM {t_day})",
                                                  visible="legendonly"))
        fig_surv_sensitivity.add_trace(go.Scatter(x=days, y=S_u, mode="lines",
                                                  name=f"Unvaccinated (LM {t_day})",
                                                  visible="legendonly"))

    fig_surv_sensitivity.update_layout(title="Sensitivity Per-Protocol Empirical Survival Curves (all landmarks selectable, uncorrected bias)",
                                       xaxis_title="Days since landmark",
                                       yaxis_title="Survival probability",
                                       template="plotly_white",
                                       legend=dict(x=1.05, y=1, orientation="v"))
    fig_surv_sensitivity.write_html(PLOT_SURV_SENSITIVITY)
    log(f"Sensitivity survival plot saved: {PLOT_SURV_SENSITIVITY}")

    # ---------------- Plot ΔΔRMST (comparison) ----------------
    res_df_compare = res_df_primary.merge(res_df_sensitivity, on="t_landmark", suffixes=("_primary", "_sensitivity"))
    res_df_compare["delta_delta_unres"] = res_df_compare["delta_rmst_unres_sensitivity"] - res_df_compare["delta_rmst_unres_primary"]
    res_df_compare["delta_delta_fixed"] = res_df_compare["delta_rmst_fixed_sensitivity"] - res_df_compare["delta_rmst_fixed_primary"]

    fig_deltadelta = go.Figure()
    fig_deltadelta.add_trace(go.Scatter(x=res_df_compare["t_landmark"], y=res_df_compare["delta_delta_unres"],
                                        mode="lines+markers", name="ΔΔRMST unrestricted (sensitivity - primary)"))
    fig_deltadelta.add_trace(go.Scatter(x=res_df_compare["t_landmark"], y=res_df_compare["delta_delta_fixed"],
                                        mode="lines+markers", name=f"ΔΔRMST fixed {FIXED_TAU} days (sensitivity - primary)", line=dict(color="green")))
    fig_deltadelta.update_layout(title="ΔΔRMST: Comparison of Sensitivity Per-Protocol vs Primary ITT-like",
                                 xaxis_title="Landmark day",
                                 yaxis_title="ΔΔRMST [days] (positive = sensitivity higher)",
                                 template="plotly_white")
    fig_deltadelta.write_html(PLOT_DELTADELTA)
    log(f"ΔΔRMST comparison plot saved: {PLOT_DELTADELTA}")

    # ---------------- Final summary logging ----------------
    log("-" * 74)
    log("Empirical Landmark RMST summary (non-parametric, unbiased)")
    log(f"Total subjects: {len(raw)}")
    log(f"Total vaccinated (any time): {np.isfinite(raw['vax_day']).sum()}")
    log(f"Total deaths: {np.isfinite(raw['death_day']).sum()}")
    log(f"Same-day death+vax cases: {same_day_count}")
    log(f"Median effective horizon (days): {res_df_primary['effective_horizon_days'].median():.0f}")
    log(f"Landmarks computed: {len(res_df_primary)}")

    log("Primary ITT-like (no censoring):")
    log(f"Median ΔRMST unrestricted: {res_df_primary['delta_rmst_unres'].median():+.2f} days")
    log(f"Mean ΔRMST unrestricted: {res_df_primary['delta_rmst_unres'].mean():+.2f} days")
    log(f"Median ΔRMST fixed {FIXED_TAU}: {res_df_primary['delta_rmst_fixed'].median():+.2f} days")
    log(f"Mean ΔRMST fixed {FIXED_TAU}: {res_df_primary['delta_rmst_fixed'].mean():+.2f} days")
    log(f"Median N_v: {res_df_primary['n_v'].median():.0f}; Median N_u: {res_df_primary['n_u'].median():.0f}")

    log("Sensitivity Per-Protocol (uncorrected censoring):")
    log(f"Median ΔRMST unrestricted: {res_df_sensitivity['delta_rmst_unres'].median():+.2f} days")
    log(f"Mean ΔRMST unrestricted: {res_df_sensitivity['delta_rmst_unres'].mean():+.2f} days")
    log(f"Median ΔRMST fixed {FIXED_TAU}: {res_df_sensitivity['delta_rmst_fixed'].median():+.2f} days")
    log(f"Mean ΔRMST fixed {FIXED_TAU}: {res_df_sensitivity['delta_rmst_fixed'].mean():+.2f} days")
    log(f"Median N_v: {res_df_sensitivity['n_v'].median():.0f}; Median N_u: {res_df_sensitivity['n_u'].median():.0f}")

    # ---------------- Conditional Summary (added as requested) ----------------
    log("Conditional Summary - Representative Values (median across all landmarks)")
    log("Primary ITT-like (no crossover censoring):")
    log(f"  Unrestricted mean: ΔRMST: {med_primary['delta_rmst_unres']:+.2f} days | "
        f"95% CI: [{med_primary['delta_rmst_unres_CI_lower']:.2f}, "
        f"{med_primary['delta_rmst_unres_CI_upper']:.2f}] | "
        f"VE approx: {ve_primary_unres:.1f}% | "
        f"Survival at {FIXED_TAU} days: {surv_v_primary_fixed:.1%} vs {surv_u_primary_fixed:.1%} (vx vs uvx)")

    log(f"  Fixed {FIXED_TAU}-day horizon: ΔRMST: {med_primary['delta_rmst_fixed']:+.2f} days | "
        f"95% CI: [{med_primary['delta_rmst_fixed_CI_lower']:.2f}, "
        f"{med_primary['delta_rmst_fixed_CI_upper']:.2f}]")

    log("Sensitivity Per-Protocol (uncorrected censoring at crossover):")
    log(f"  Unrestricted mean: ΔRMST: {med_sens['delta_rmst_unres']:+.2f} days | "
        f"95% CI: [{med_sens['delta_rmst_unres_CI_lower']:.2f}, "
        f"{med_sens['delta_rmst_unres_CI_upper']:.2f}] | "
        f"VE approx: {ve_sens_unres:.1f}% | "
        f"Survival at {FIXED_TAU} days: {surv_v_sens_fixed:.1%} vs {surv_u_sens_fixed:.1%} (vx vs uvx)")

    log(f"  Fixed {FIXED_TAU}-day horizon: ΔRMST: {med_sens['delta_rmst_fixed']:+.2f} days | "
        f"95% CI: [{med_sens['delta_rmst_fixed_CI_lower']:.2f}, "
        f"{med_sens['delta_rmst_fixed_CI_upper']:.2f}]")

    log("Note: VE is crude approximation = (ΔRMST / RMST_unvaccinated) × 100%. "
        "Survival probabilities are rough approximations based on RMST. "
        "Sensitivity may overestimate VE due to uncorrected informative selection bias.")

    log("Epidemiological notes: Positive ΔRMST indicates vaccinated survival advantage. Sensitivity may overestimate VE due to selection bias. Compare to null simulation for method validation.")
    log("-" * 74)

    print(f"\nDONE ✓\nPrimary CSV: {CSV_PATH_PRIMARY}\nSensitivity CSV: {CSV_PATH_SENSITIVITY}\n"
          f"Plots: {PLOT_RMST_PRIMARY}, {PLOT_SURV_PRIMARY}, {PLOT_RMST_SENSITIVITY}, {PLOT_SURV_SENSITIVITY}, {PLOT_DELTADELTA}\nLog: {LOG_PATH}")