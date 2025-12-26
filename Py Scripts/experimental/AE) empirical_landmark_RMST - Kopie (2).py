#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical Landmark RMST (Restricted Mean Survival Time) Estimation with Non‑Parametric Bootstrap
Applied to Age‑Stratified Czech FOI Mortality/Vaccination Data

Purpose:
    This script implements a fully empirical, non‑parametric landmark analysis
    of survival differences between vaccinated and unvaccinated individuals.
    The method computes ΔRMST (difference in restricted mean survival time)
    at a sequence of landmark times, using only observed hazards and without
    parametric assumptions.

    The script supports:
        • Real-world Czech FOI datasets
        • Sensitivity analyses with reclassified UVX→VX deaths
        • Simulated datasets (HR=1 null model)
        • Parallel bootstrap for confidence intervals
        • Plotly visualizations (ΔRMST curve + survival curves)
        • Full logging for reproducibility

Scientific Notes:
    • Landmark analysis avoids immortal time bias by conditioning on survival
      up to each landmark.
    • RMST is computed directly from empirical hazards (non‑parametric).
    • Bootstrap resampling preserves the empirical joint distribution.
    • No Cox model, no proportional hazards assumption, no smoothing.
    • This script is suitable for inclusion in a preprint Methods section.

Author: AI / drifting Date: 2025-12 Version: 1
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
# Study parameters, dataset selection, and output paths.
# AGE: exact age cohort to analyze (integer)
# STUDY_START: reference origin for converting dates to day indices
# IMMUNITY_LAG: days excluded after vaccination to avoid immediate-risk artifacts
# OBS_END_SAFETY: buffer to avoid right-edge censoring artifacts
# LANDMARK_STEP: spacing between landmark times
# BOOTSTRAP_REPS: number of bootstrap replicates for CI estimation

AGE = 70
STUDY_START = pd.Timestamp("2020-01-01")
AGE_REF_YEAR = 2023
IMMUNITY_LAG = 0
OBS_END_SAFETY = 30
RANDOM_SEED = 12345
LANDMARK_STEP = 30
BOOTSTRAP_REPS = 200

# Input/Output
# real world Czech-FOI dataset 
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST")

# real data with hypothetical 5% uvx death or alive reclassified as vx (sensitivity test for missclassifcation)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_DeathOrAlive_reclassified_PCT5_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_DeathOrALive_RECLASSIFIED")

# simulated dataset HR=1 with simulated real dose schedule (sensitivity test if the used methode is bias free - null hypothesis)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_SIM")

OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.csv")
PLOT_RMST_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_ΔRMST_AG{AGE}.html")
PLOT_SURV_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_Survival_AG{AGE}.html")
LOG_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.log")

# ---------------- Logging ----------------
# Creates a dedicated logger for reproducible analysis output.
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
def rmst_from_survival(S, dt=1.0):
    # For discrete daily time, RMST is just the sum of survival probabilities.
    # This correctly measures the area of the 'staircase' rectangles.
    return float(np.sum(S) * dt)

# ---------------- Main processing ----------------
# Core function: computes all landmark RMST values for a dataset.
# This is the scientific heart of the method.
def compute_landmarks(raw):
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

    log(f"Subjects: {len(raw)} | Landmark window: {FIRST_ELIG_DAY}-{LAST_OBS_DAY} ({len(DAYS)} days) | Landmarks: {len(LANDMARKS)}")

    # ---------------- Index mapping ----------------
    # Converts vaccination/death days into discrete indices for hazard tables.
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

        # 2. Unvaccinated Risk Set (Deaths + Crossover Censoring)
        #    People unvaccinated at landmark t: v_idx >= v_start_u
        mask_u = (v_idx >= v_start_u)
        vax_idx_u = v_idx[mask_u]

        # Count vaccinations among the 'Unvaccinated' group occurring during follow-up
        # Only those who:
        #   - were unvaccinated at the landmark
        #   - get their first dose during follow-up (start ..)
        #   - are not NEVER_VAX_IDX
        #   - lie within the observed death grid [0, D_DIM-1]
        vax_follow = vax_idx_u[
            (vax_idx_u >= start) &
            (vax_idx_u < NEVER_VAX_IDX) &
            (vax_idx_u < D_DIM)
        ]
        vax_counts_during_followup = np.bincount(vax_follow, minlength=D_DIM)
        crossovers_cum = np.cumsum(vax_counts_during_followup)

        # Risk Set U: subtract cumulative deaths AND cumulative crossovers
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
        rmst_v = rmst_from_survival(S_v)
        rmst_u = rmst_from_survival(S_u)
        delta = float(rmst_v - rmst_u)

        results.append(
            {
                "t_landmark": t_day,
                "n_v": int(N_v),
                "n_u": int(N_u),
                "rmst_v": rmst_v,
                "rmst_u": rmst_u,
                "delta_rmst": delta,
            }
        )

    res_df = pd.DataFrame(results).sort_values("t_landmark")
    res_df.to_csv(CSV_PATH, index=False)
    log(f"Saved CSV: {CSV_PATH} | Landmarks computed: {len(res_df)}")
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
    )

# ---------------- Bootstrap ----------------
# Non‑parametric bootstrap: resample subjects with replacement,
# recompute all landmarks, and extract ΔRMST for CI estimation.
def one_boot(i, raw, res_df, V_DIM, D_DIM,
             counts_cumv, counts_sufv,
             deaths_prefv, deaths_sufv,
             deaths_cumd_prefv, deaths_cumd_sufv,
             FIRST_ELIG_DAY, NEVER_VAX_IDX, LAST_VAX_IDX):
    np.random.seed(RANDOM_SEED + i)
    idx = np.random.choice(len(raw), len(raw), replace=True)
    raw_b = raw.iloc[idx].reset_index(drop=True)
    _, res_df_b, *_ = compute_landmarks(raw_b)
    # pad to align with main res_df landmarks
    return res_df_b.set_index("t_landmark").reindex(res_df["t_landmark"], fill_value=np.nan)["delta_rmst"].values

# ---------------- Main ----------------
if __name__ == "__main__":
    multiprocessing.freeze_support()

    raw = pd.read_csv(INPUT, low_memory=False)
    raw.columns = raw.columns.str.strip()
    (
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
    ) = compute_landmarks(raw)

    # ---------------- Parallel bootstrap ----------------
    log("Starting bootstrap...")
    indices = range(BOOTSTRAP_REPS)
    boot_func = partial(
        one_boot,
        raw=raw,
        res_df=res_df,
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
    )

    boot_deltas = []
    with ProcessPoolExecutor() as ex:
        for res in tqdm(ex.map(boot_func, indices), total=BOOTSTRAP_REPS, desc="Bootstrap"):
            boot_deltas.append(res)

    boot_array = np.stack(boot_deltas, axis=0)
    delta_lower = np.nanpercentile(boot_array, 2.5, axis=0)
    delta_upper = np.nanpercentile(boot_array, 97.5, axis=0)
    res_df["delta_rmst_CI_lower"] = delta_lower
    res_df["delta_rmst_CI_upper"] = delta_upper
    res_df.to_csv(CSV_PATH, index=False)
    log(f"Bootstrap CI added and CSV updated: {CSV_PATH}")

    # ---------------- Plot ΔRMST ----------------
    fig_rmst = go.Figure()
    fig_rmst.add_trace(go.Scatter(x=res_df["t_landmark"], y=res_df["delta_rmst"],
                                  mode="lines+markers", name="ΔRMST"))
    fig_rmst.add_trace(go.Scatter(x=res_df["t_landmark"], y=res_df["delta_rmst_CI_lower"],
                                  mode="lines", line=dict(dash="dash"), name="CI lower"))
    fig_rmst.add_trace(go.Scatter(x=res_df["t_landmark"], y=res_df["delta_rmst_CI_upper"],
                                  mode="lines", line=dict(dash="dash"), name="CI upper"))
    fig_rmst.update_layout(title="ΔRMST per Landmark",
                           xaxis_title="Landmark day",
                           yaxis_title="ΔRMST [days]",
                           template="plotly_white")
    fig_rmst.write_html(PLOT_RMST_PATH)
    log(f"ΔRMST plot saved: {PLOT_RMST_PATH}")

    # ---------------- Plot Survival curves for all landmarks ----------------
    fig_surv = go.Figure()
    for idx_row, row in res_df.iterrows():
        t_day = int(row["t_landmark"])
        t = t_day - FIRST_ELIG_DAY
        t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)

        # Death counts for both groups
        E_v = deaths_prefv[min(t, LAST_VAX_IDX), :].copy()
        v_start_u = min(t + 1, NEVER_VAX_IDX)
        E_u = deaths_sufv[v_start_u, :].copy()

        # Remove deaths before landmark + lag
        if t_excl >= 0:
            E_v[: t_excl + 1] = 0
            E_u[: t_excl + 1] = 0

        # Cumulative deaths after landmark
        cumE_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), :] - deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl]
        cumE_u = deaths_cumd_sufv[v_start_u, :] - deaths_cumd_sufv[v_start_u, t_excl]

        # Start of follow-up
        start = t_excl + 1
        if start >= D_DIM:
            continue

        # APPLY CROSSOVER CENSORING (same as in compute_landmarks)
        mask_u = (v_idx >= v_start_u)
        vax_idx_u = v_idx[mask_u]

        vax_follow = vax_idx_u[
            (vax_idx_u >= start) &
            (vax_idx_u < NEVER_VAX_IDX) &
            (vax_idx_u < D_DIM)
        ]
        vax_counts_during_followup = np.bincount(vax_follow, minlength=D_DIM)
        crossovers_cum = np.cumsum(vax_counts_during_followup)

        # Risk sets
        R_v = row["n_v"] - np.concatenate([[0], cumE_v[:-1]])
        R_u = row["n_u"] - np.concatenate([[0], (cumE_u[:-1] + crossovers_cum[:-1])])

        R_v = np.clip(R_v, 1e-12, None)
        R_u = np.clip(R_u, 1e-12, None)

        # Hazards
        h_v_seg = E_v[start:] / R_v[start:]
        h_u_seg = E_u[start:] / R_u[start:]

        # Survival curves
        S_v = survival_from_hazard(h_v_seg)
        S_u = survival_from_hazard(h_u_seg)
        days = np.arange(len(S_v))

        fig_surv.add_trace(go.Scatter(x=days, y=S_v, mode="lines",
                                      name=f"Vaccinated (LM {t_day})",
                                      visible="legendonly"))
        fig_surv.add_trace(go.Scatter(x=days, y=S_u, mode="lines",
                                      name=f"Unvaccinated (LM {t_day})",
                                      visible="legendonly"))

    fig_surv.update_layout(title="Empirical Survival Curves (all landmarks selectable)",
                           xaxis_title="Days since landmark",
                           yaxis_title="Survival probability",
                           template="plotly_white",
                           legend=dict(x=1.05, y=1, orientation="v"))
    fig_surv.write_html(PLOT_SURV_PATH)
    log(f"Survival plot saved: {PLOT_SURV_PATH}")

    # ---------------- Final summary logging ----------------
    log("-" * 74)
    log("Empirical Landmark RMST summary (non-parametric, unbiased)")
    log(f"Total subjects: {len(raw)}")
    log(f"Total vaccinated (any time): {np.isfinite(raw['vax_day']).sum()}")
    log(f"Total deaths: {np.isfinite(raw['death_day']).sum()}")
    log(f"Landmarks computed: {len(res_df)}")
    log(f"Median ΔRMST: {res_df['delta_rmst'].median():+.2f} days")
    log(f"Mean ΔRMST: {res_df['delta_rmst'].mean():+.2f} days")
    log(f"Median N_v: {res_df['n_v'].median():.0f}; Median N_u: {res_df['n_u'].median():.0f}")
    log("-" * 74)

    print(f"\nDONE ✓\nCSV: {CSV_PATH}\nPlots: {PLOT_RMST_PATH}, {PLOT_SURV_PATH}\nLog: {LOG_PATH}")
