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
    • Vaccination is treated as a non-absorbing state; competing risks between
      vaccination and death are not explicitly modeled (implicit in the landmark approach).
    • This script is suitable for inclusion in a preprint Methods section.

Time scale:
    • Day 0 = first day any subject received Dose 1 (Datum_1.min()).
    • Study end (in days) = min(last Dose 1 date, last death date) - first_dose_min - OBS_END_SAFETY.
    • All event times (vaccination, death), hazard tables, risk sets, and
      landmark times are expressed relative to first_dose_min.

Author: AI / drifting
Date: 2025-12
Version: 1 (adapted to first_dose_min time origin)
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
# AGE_REF_YEAR: reference year to compute age from year of birth
# IMMUNITY_LAG: days excluded after vaccination to avoid immediate-risk artifacts
# OBS_END_SAFETY: buffer (days) to avoid right-edge censoring artifacts
# LANDMARK_STEP: spacing between landmark times (in days)
# BOOTSTRAP_REPS: number of bootstrap replicates for CI estimation

AGE = 70
AGE_REF_YEAR = 2023
IMMUNITY_LAG = 0
OBS_END_SAFETY = 30
RANDOM_SEED = 12345
LANDMARK_STEP = 30
BOOTSTRAP_REPS = 20

# Input/Output
# real world Czech-FOI dataset
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST")

# real data with hypothetical 5% uvx death or alive reclassified as vx (sensitivity test for misclassification)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_DeathOrAlive_reclassified_PCT5_uvx_as_vx_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_DeathOrALive_RECLASSIFIED")

# simulated dataset HR=1 with simulated real dose schedule (sensitivity test if the used method is bias-free - null hypothesis)
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
        acc *= (1.0 - h[i])
        s[i] = acc
    return s

# Computes RMST by numerical integration of survival curve
def rmst_from_survival(S, dt=1.0):
    return float(np.trapz(S, dx=dt))

# ---------------- Main processing ----------------
# Core function: computes all landmark RMST values for a dataset.
# This is the scientific heart of the method.
def compute_landmarks(raw, first_dose_min=None, LAST_OBS_DAY=None):
    # Parse dates
    date_cols = [c for c in raw.columns if c.startswith("Datum_") or c == "DatumUmrti"]
    for c in date_cols:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")

    # Filter by exact age
    raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw = raw[raw["age"] == AGE].reset_index(drop=True)

    if raw.empty:
        raise ValueError(f"No subjects with age == {AGE} found after filtering.")

    # ---------------- Define entry day (for left-truncation / late entry) ----------------
    # IMPORTANT: Define entry_day as the day the subject enters observation (relative to first_dose_min)
    # If your dataset has a column for first observation date (e.g., 'Datum_entry'), use:
    # raw["entry_day"] = (raw["Datum_entry"] - first_dose_min).dt.days
    # Otherwise, assume entry_day = 0 for all (no late entry) or min(vax_day, death_day) as proxy
    raw["entry_day"] = 0  # CHANGE THIS IF YOU HAVE A BETTER COLUMN / PROXY FOR ENTRY TIME
    raw["entry_day"] = raw["entry_day"].fillna(0).clip(lower=0).astype(int)

    # ---------------- Define study time origin and window ----------------
    # Study start: first observed Dose 1 in this age cohort
    if first_dose_min is None:
        first_dose_min = raw["Datum_1"].min(skipna=True)
    if pd.isna(first_dose_min):
        raise ValueError("No Datum_1 (Dose 1) dates found; cannot define study start.")

    # Study end (calendar): minimum of last Dose 1 date and last death date
    # (ensures overlapping period where both processes are observed)
    last_first_dose = raw["Datum_1"].max(skipna=True)
    last_death = raw["DatumUmrti"].max(skipna=True)
    if pd.isna(last_first_dose) or pd.isna(last_death):
        raise ValueError("Insufficient data to define study end (need both Dose 1 and death dates).")

    base_end = min(last_first_dose, last_death)

    # Relative time bounds (in days) from first_dose_min
    FIRST_ELIG_DAY = 0
    if LAST_OBS_DAY is None:
        LAST_OBS_DAY = int((base_end - first_dose_min).days) - OBS_END_SAFETY
    if LAST_OBS_DAY <= FIRST_ELIG_DAY:
        raise ValueError(
            f"OBS_END_SAFETY too large or window too short; "
            f"FIRST_ELIG_DAY={FIRST_ELIG_DAY}, LAST_OBS_DAY={LAST_OBS_DAY}"
        )

    # Convert dates to day indices relative to first_dose_min
    raw["death_day"] = (raw["DatumUmrti"] - first_dose_min).dt.days
    raw["vax_day"] = (raw["Datum_1"] - first_dose_min).dt.days

    # Remove events before study start or entry day (safe-guard)
    raw.loc[raw["death_day"] < raw["entry_day"], "death_day"] = np.nan
    raw.loc[raw["vax_day"] < raw["entry_day"], "vax_day"] = np.nan

    # Observation window (relative days)
    DAYS = np.arange(FIRST_ELIG_DAY, LAST_OBS_DAY + 1)
    LANDMARKS = np.arange(FIRST_ELIG_DAY, LAST_OBS_DAY + 1, LANDMARK_STEP)

    log(
        f"Subjects: {len(raw)} | "
        f"Study start (Dose 1 min): {first_dose_min.date()} | "
        f"Study end (base_end): {base_end.date()} | "
        f"Landmark window: {FIRST_ELIG_DAY}-{LAST_OBS_DAY} ({len(DAYS)} days) | "
        f"Landmarks: {len(LANDMARKS)}"
    )

    # ---------------- Epidemiologic sanity checks ----------------
    log("Running epidemiologic data checks...")

    # Basic counts
    n_total = len(raw)
    n_vax = raw["vax_day"].notna().sum()
    n_unvax = n_total - n_vax
    n_deaths = raw["death_day"].notna().sum()

    log(f"Total subjects: {n_total}")
    log(f"Vaccinated: {n_vax} ({n_vax/n_total*100:.1f}%)")
    log(f"Unvaccinated: {n_unvax} ({n_unvax/n_total*100:.1f}%)")
    log(f"Deaths observed: {n_deaths} ({n_deaths/n_total*100:.1f}%)")

    # Check for missing dates
    missing_vax = raw["Datum_1"].isna().sum()
    missing_death = raw["DatumUmrti"].isna().sum()
    log(f"Missing vaccination dates: {missing_vax}")
    log(f"Missing death dates: {missing_death}")

    # Check for impossible sequences
    n_vax_after_death = ((raw["vax_day"] > raw["death_day"]) & raw["death_day"].notna()).sum()
    n_death_before_entry = (raw["death_day"] < raw["entry_day"]).sum()
    n_vax_before_entry = (raw["vax_day"] < raw["entry_day"]).sum()

    log(f"Vaccination AFTER death (impossible): {n_vax_after_death}")
    log(f"Deaths before entry (should be 0): {n_death_before_entry}")
    log(f"Vaccinations before entry (should be 0): {n_vax_before_entry}")

    # Check follow-up window
    if LAST_OBS_DAY <= 0:
        log("ERROR: LAST_OBS_DAY <= 0 — no usable follow-up window.")
        raise ValueError("Follow-up window collapsed.")

    log(f"Follow-up window length: {LAST_OBS_DAY} days")

    # Time-to-event summaries
    if n_vax > 0:
        log(f"Median time to vaccination: {np.nanmedian(raw['vax_day']):.1f} days")
    if n_deaths > 0:
        log(f"Median time to death: {np.nanmedian(raw['death_day']):.1f} days")

    # Person-time calculations (adjusted for entry)
    raw["effective_start"] = raw["entry_day"]
    raw["death_or_censor"] = raw["death_day"].fillna(LAST_OBS_DAY)
    pt_v = ((raw["death_or_censor"] - raw["vax_day"].clip(lower=raw["effective_start"])).clip(lower=0)).sum()
    pt_u = ((raw["vax_day"].fillna(raw["death_or_censor"]) - raw["effective_start"]).clip(lower=0)).sum()

    log(f"Person-time unvaccinated: {pt_u:.1f} person-days")
    log(f"Person-time vaccinated:   {pt_v:.1f} person-days")

    # Incidence rates
    if pt_u > 0:
        d_u = raw.loc[(raw["vax_day"].isna() | (raw["vax_day"] > raw["death_day"])) & raw["death_day"].notna()].shape[0]
        ir_u = (d_u / pt_u) * 1e5
        log(f"Incidence rate unvaccinated: {ir_u:.2f} deaths per 100k person-days")
    if pt_v > 0:
        d_v = raw.loc[raw["vax_day"].notna() & (raw["death_day"] > raw["vax_day"]) & raw["death_day"].notna()].shape[0]
        ir_v = (d_v / pt_v) * 1e5
        log(f"Incidence rate vaccinated:   {ir_v:.2f} deaths per 100k person-days")

    # Vaccination coverage curve summary
    if n_vax > 0:
        log(f"Vaccination coverage by study end: {n_vax/n_total*100:.1f}%")
        log(f"50% coverage reached at day: {np.nanpercentile(raw['vax_day'], 50):.1f}")

    # Death distribution by vaccination status
    d_v = ((raw["death_day"].notna()) & (raw["vax_day"].notna()) & (raw["death_day"] > raw["vax_day"])).sum()
    d_u = ((raw["death_day"].notna()) & ((raw["vax_day"].isna()) | (raw["death_day"] <= raw["vax_day"]))).sum()
    log(f"Deaths among vaccinated:   {d_v} ({d_v/n_deaths*100:.1f}%)")
    log(f"Deaths among unvaccinated: {d_u} ({d_u/n_deaths*100:.1f}%)")

    # Landmark coverage check
    log("Checking landmark sample sizes...")
    for lm in LANDMARKS[:10]:  # only first 10 to avoid log spam
        alive_at_lm = ((raw["entry_day"] <= lm) & ((raw["death_day"].isna()) | (raw["death_day"] > lm))).sum()
        log(f"LM {lm:4d}: alive subjects = {alive_at_lm}")

    log("Epidemiologic checks completed.")


    # ---------------- Precompute dimensions (used in per-landmark recomputation) ----------------
    NEVER_VAX_IDX = (LAST_OBS_DAY - FIRST_ELIG_DAY + 1)  # index for "never vaccinated up to last obs"
    V_DIM = NEVER_VAX_IDX + 1                            # 0..LAST_OBS_DAY + "never vax" bucket
    D_DIM = (LAST_OBS_DAY - FIRST_ELIG_DAY + 1)          # 0..LAST_OBS_DAY
    LAST_VAX_IDX = V_DIM - 2                             # last day index (not the never-vax bucket)

    results = []
    surv_data = {}  # To store per-landmark survival curves for plotting

    # ---------------- Landmark computation (with per-landmark recomputation for left-truncation) ----------------
    for t_day in tqdm(LANDMARKS, desc="Landmarks"):
        # t_day is already relative to first_dose_min (0..LAST_OBS_DAY)
        t = t_day  # relative index
        # Exclusion window (e.g. immunity lag after vaccination)
        t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)

        # Eligible subjects at landmark: entered <= t and alive > t
        eligible_mask = (raw["entry_day"] <= t) & ((raw["death_day"].isna()) | (raw["death_day"] > t))
        eligible_raw = raw[eligible_mask].copy()

        if len(eligible_raw) == 0:
            continue

        # Extract relative day arrays for eligible
        vax_days = eligible_raw["vax_day"].to_numpy()
        death_days = eligible_raw["death_day"].to_numpy()

        # Vaccination indices
        v_idx = np.full(len(vax_days), NEVER_VAX_IDX, dtype=int)  # default: never vaccinated (within window)
        mask_v = ~np.isnan(vax_days)
        mask_v &= (vax_days >= FIRST_ELIG_DAY) & (vax_days <= LAST_OBS_DAY)
        v_idx[mask_v] = vax_days[mask_v].astype(int)

        # Death indices
        d_idx = np.full(len(death_days), -1, dtype=int)  # default: no death within window (censored)
        mask_d = ~np.isnan(death_days)
        mask_d &= (death_days >= FIRST_ELIG_DAY) & (death_days <= LAST_OBS_DAY)
        d_idx[mask_d] = death_days[mask_d].astype(int)

        # Build empirical hazard tables for this landmark's eligible set
        counts_by_v = np.bincount(v_idx, minlength=V_DIM).astype(np.int64)
        deaths_by_vd = np.zeros((V_DIM, D_DIM), dtype=np.int64)
        mask_obs_death = d_idx >= 0
        for vi, di in zip(v_idx[mask_obs_death], d_idx[mask_obs_death]):
            deaths_by_vd[vi, di] += 1

        # Precompute cumulative tables for this landmark
        counts_cumv = np.cumsum(counts_by_v)                             # cumulative by vaccination day
        counts_sufv = np.cumsum(counts_by_v[::-1])[::-1]                 # suffix sums
        deaths_prefv = np.cumsum(deaths_by_vd, axis=0)                   # cumulative in v-d plane
        deaths_sufv = np.cumsum(deaths_by_vd[::-1, :], axis=0)[::-1, :]  # suffix sums in v-d plane
        deaths_cumd_prefv = np.cumsum(np.cumsum(deaths_by_vd, axis=1), axis=0)
        deaths_cumd_sufv = np.cumsum(np.cumsum(deaths_by_vd[::-1, :], axis=1), axis=0)[::-1, :]

        # Vaccinated group at landmark t:
        # Individuals vaccinated at or before t who are still alive at t_excl.
        N_v = counts_cumv[min(t, LAST_VAX_IDX)]
        died_to_t_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl] if N_v > 0 else 0
        N_v -= died_to_t_v

        # Unvaccinated group at landmark t:
        # Individuals whose vaccination occurs after t (or never) and still alive at t_excl.
        v_start_u = min(t + 1, NEVER_VAX_IDX)
        N_u = counts_sufv[v_start_u]
        died_to_t_u = deaths_cumd_sufv[v_start_u, t_excl] if N_u > 0 else 0
        N_u -= died_to_t_u

        if N_v <= 0 or N_u <= 0:
            continue

        # Event counts (deaths) by follow-up day, for each group, conditional on survival to t_excl
        E_v = deaths_prefv[min(t, LAST_VAX_IDX), :].copy()
        E_u = deaths_sufv[v_start_u, :].copy()
        if t_excl >= 0:
            E_v[: t_excl + 1] = 0
            E_u[: t_excl + 1] = 0

        # Cumulative deaths after t_excl to compute risk sets over follow-up
        cumE_v = deaths_cumd_prefv[min(t, LAST_VAX_IDX), :] - deaths_cumd_prefv[min(t, LAST_VAX_IDX), t_excl]
        cumE_u = deaths_cumd_sufv[v_start_u, :] - deaths_cumd_sufv[v_start_u, t_excl]

        # Risk sets over follow-up
        R_v = N_v - np.concatenate([[0], cumE_v[:-1]])
        R_u = N_u - np.concatenate([[0], cumE_u[:-1]])
        R_v = np.clip(R_v, 1e-10, None)  # Adjusted clipping for stability
        R_u = np.clip(R_u, 1e-10, None)

        # Follow-up segment starts after t_excl
        start = t_excl + 1
        if start >= D_DIM:
            continue

        # Hazards and survival from landmark forward
        h_v_seg = E_v[start:] / R_v[start:]
        h_u_seg = E_u[start:] / R_u[start:]
        S_v = survival_from_hazard(h_v_seg)
        S_u = survival_from_hazard(h_u_seg)
        rmst_v = rmst_from_survival(S_v)
        rmst_u = rmst_from_survival(S_u)
        delta = float(rmst_v - rmst_u)

        results.append(
            {
                "t_landmark": t_day,    # days since first_dose_min
                "n_v": int(N_v),
                "n_u": int(N_u),
                "rmst_v": rmst_v,
                "rmst_u": rmst_u,
                "delta_rmst": delta,
            }
        )

        # Save survival curves for plotting
        surv_data[t_day] = {
            'S_v': S_v,
            'S_u': S_u,
            'days': np.arange(len(S_v))
        }

    res_df = pd.DataFrame(results).sort_values("t_landmark")
    res_df.to_csv(CSV_PATH, index=False)
    log(f"Saved CSV: {CSV_PATH} | Landmarks computed: {len(res_df)}")

    return (
        raw,
        res_df,
        V_DIM,
        D_DIM,
        FIRST_ELIG_DAY,
        NEVER_VAX_IDX,
        LAST_VAX_IDX,
        first_dose_min,
        LAST_OBS_DAY,
        surv_data,
    )

# ---------------- Bootstrap ----------------
# Non‑parametric bootstrap: resample subjects with replacement,
# recompute all landmarks, and extract ΔRMST for CI estimation.
def one_boot(i, raw, res_df, V_DIM, D_DIM,
             FIRST_ELIG_DAY, NEVER_VAX_IDX, LAST_VAX_IDX,
             first_dose_min, LAST_OBS_DAY):
    np.random.seed(RANDOM_SEED + i)
    idx = np.random.choice(len(raw), len(raw), replace=True)
    raw_b = raw.iloc[idx].reset_index(drop=True)
    # Recompute landmarks on bootstrap sample, but with FIXED time origin and window from main
    _, res_df_b, *_ = compute_landmarks(raw_b, first_dose_min=first_dose_min, LAST_OBS_DAY=LAST_OBS_DAY)
    # Pad to align with main res_df landmarks
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
        FIRST_ELIG_DAY,
        NEVER_VAX_IDX,
        LAST_VAX_IDX,
        first_dose_min,
        LAST_OBS_DAY,
        surv_data,
    ) = compute_landmarks(raw)

    # ---------------- Parallel bootstrap ----------------
    log("Starting bootstrap...")
    np.random.seed(RANDOM_SEED)  # For reproducibility across runs
    indices = range(BOOTSTRAP_REPS)
    boot_func = partial(
        one_boot,
        raw=raw,
        res_df=res_df,
        V_DIM=V_DIM,
        D_DIM=D_DIM,
        FIRST_ELIG_DAY=FIRST_ELIG_DAY,
        NEVER_VAX_IDX=NEVER_VAX_IDX,
        LAST_VAX_IDX=LAST_VAX_IDX,
        first_dose_min=first_dose_min,
        LAST_OBS_DAY=LAST_OBS_DAY,
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

    # ---------------- VE, CI, and Survival Summary ----------------
    log("Computing VE, CI, and survival summary...")

    # Use the last landmark (max follow-up)
    last_row = res_df.iloc[-1]

    delta = last_row["delta_rmst"]
    ci_low = last_row["delta_rmst_CI_lower"]
    ci_up = last_row["delta_rmst_CI_upper"]

    # VE = ΔRMST / RMST_unvaccinated
    rmst_u = last_row["rmst_u"]
    if rmst_u > 0:
        VE = delta / rmst_u
    else:
        VE = np.nan

    # Restricted mean survival fraction at end of follow-up
    # (Previously misnamed as "survival probability")
    max_follow = last_row["t_landmark"]  # days since first dose
    mean_surv_frac_v = last_row["rmst_v"] / max_follow if max_follow > 0 else np.nan
    mean_surv_frac_u = last_row["rmst_u"] / max_follow if max_follow > 0 else np.nan

    log(
        f"ΔRMST: {delta:+.2f} days | "
        f"95% CI: [{ci_low:+.2f}, {ci_up:+.2f}] | "
        f"VE: {VE*100:+.1f}% | "
        f"Restricted mean survival fraction: {mean_surv_frac_v*100:.1f}% vs {mean_surv_frac_u*100:.1f}%"
    )

    # ---------------- Plot ΔRMST ----------------
    fig_rmst = go.Figure()
    fig_rmst.add_trace(go.Scatter(
        x=res_df["t_landmark"],
        y=res_df["delta_rmst"],
        mode="lines+markers",
        name="ΔRMST"
    ))
    fig_rmst.add_trace(go.Scatter(
        x=res_df["t_landmark"],
        y=res_df["delta_rmst_CI_lower"],
        mode="lines",
        line=dict(dash="dash"),
        name="CI lower"
    ))
    fig_rmst.add_trace(go.Scatter(
        x=res_df["t_landmark"],
        y=res_df["delta_rmst_CI_upper"],
        mode="lines",
        line=dict(dash="dash"),
        name="CI upper"
    ))
    fig_rmst.update_layout(
        title="ΔRMST per Landmark (days since first Dose 1)",
        xaxis_title="Landmark day (relative to first Dose 1)",
        yaxis_title="ΔRMST [days]",
        template="plotly_white"
    )
    fig_rmst.write_html(PLOT_RMST_PATH)
    log(f"ΔRMST plot saved: {PLOT_RMST_PATH}")

    # ---------------- Plot Survival curves for all landmarks ----------------
    fig_surv = go.Figure()
    for t_day in res_df["t_landmark"]:
        if t_day in surv_data:
            sd = surv_data[t_day]
            fig_surv.add_trace(go.Scatter(
                x=sd['days'],
                y=sd['S_v'],
                mode="lines",
                name=f"Vaccinated (LM {t_day})",
                visible="legendonly"
            ))
            fig_surv.add_trace(go.Scatter(
                x=sd['days'],
                y=sd['S_u'],
                mode="lines",
                name=f"Unvaccinated (LM {t_day})",
                visible="legendonly"
            ))

    fig_surv.update_layout(
        title="Empirical Survival Curves (all landmarks selectable)",
        xaxis_title="Days since landmark",
        yaxis_title="Survival probability",
        template="plotly_white",
        legend=dict(x=1.05, y=1, orientation="v")
    )
    fig_surv.write_html(PLOT_SURV_PATH)
    log(f"Survival plot saved: {PLOT_SURV_PATH}")

    # ---------------- Final summary logging ----------------
    log("-" * 74)
    log("Empirical Landmark RMST summary (non-parametric, unbiased, time origin = first Dose 1)")
    log(f"Total subjects: {len(raw)}")
    log(f"Total vaccinated (any time within window): {np.isfinite(raw['vax_day']).sum()}")
    log(f"Total deaths (within window): {np.isfinite(raw['death_day']).sum()}")
    log(f"Landmarks computed: {len(res_df)}")
    log(f"Median ΔRMST: {res_df['delta_rmst'].median():+.2f} days")
    log(f"Mean ΔRMST: {res_df['delta_rmst'].mean():+.2f} days")
    log(f"Median N_v: {res_df['n_v'].median():.0f}; Median N_u: {res_df['n_u'].median():.0f}")
    log("-" * 74)

    print(
        f"\nDONE ✓\n"
        f"CSV:   {CSV_PATH}\n"
        f"Plots: {PLOT_RMST_PATH}, {PLOT_SURV_PATH}\n"
        f"Log:   {LOG_PATH}\n"
    )