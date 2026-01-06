#!/usr/bin/env python3
"""
Bias-Necessity RMST Audit Pipeline

This script implements a bias-necessity falsification pipeline for vaccination-mortality registry data.
The goal is to test if observed mortality patterns are compatible with biology alone or require selection bias on latent health.

Key Components:
- Pre-vaccination mortality checks for temporal falsification
- Lagged RMST with negative controls
- Vectorized placebo simulation from empirical hazards to test selection sufficiency

Important Notes:
- This is falsification, not causal estimation. No effect sizes claimed.
- Run in quick_test mode first; set to False for full data.
- Outputs: CSVs (summaries, diagnostics), PNG plots, warnings.log
- Dependencies: numpy, pandas, matplotlib, seaborn, tqdm, lifelines

Patched Features:
- Canonical date normalization
- Reproducible RNG for placebo sims
- Merged alignment checks and normalization
- Vectorized placebo with memory logging and chunk fallback
- Safer IF denominator handling

Author: AI / Drifting assistence   Date: January 2026 Version 1.0
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from datetime import datetime
from lifelines import KaplanMeierFitter
import os
import ctypes

# ==================== SLEEP PREVENTION Windows11 ====================
# Prevents Windows from sleeping during long runs
if os.name == 'nt':
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print(">>> Windows Sleep Prevention: ACTIVE")
    except Exception as e:
        print(f">>> Could not set Sleep Prevention: {e}")
# ==================== END SLEEP PREVENTION ====================

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -------------------------
# Configuration
# -------------------------
# Define configuration parameters as a dataclass for easy access
@dataclass
class Config:
    input_path: Path = Path(r"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131.csv")
    out_dir: Path = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AG) bias necessety rmst audit")
    study_start: pd.Timestamp = pd.Timestamp("2020-01-01")
    study_end: pd.Timestamp = pd.Timestamp("2023-12-31")
    age_ref_year: int = 2023
    pre_vax_window_days: int = 30
    rmst_tau_days: int = 30
    vacc_quantiles: int = 10
    age_bins: tuple = (0, 40, 60, 80, 200)
    restrict_pre_vax_comparator_to_eventual_vaccinators: bool = True
    figsize: tuple = (10, 6)
    bootstrap_seed: int = 12345
    small_n_threshold: int = 50
    min_y_threshold: int = 20  # Suppress IF variance if min_Y < this

    quick_test: bool = True
    sample_frac: float = 0.03
    lag_min: int = -30
    lag_max: int = 90
    n_placebo_sims: int = 1
    bootstrap_reps: int = 100

CFG = Config()
CFG.out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Logging + Reproducibility
# -------------------------
# Functions for logging messages and warnings to console and file
def log(msg: str):
    print(f"[{pd.Timestamp.now().isoformat()}] {msg}")

def log_warning(msg: str):
    with open(CFG.out_dir / "warnings.log", "a", encoding="utf-8") as f:
        f.write(f"[{pd.Timestamp.now().isoformat()}] {msg}\n")
    log(f"[WARNING] {msg}")

def save_run_info(cfg, row_count: int):
    info = f"""Run started: {datetime.now().isoformat()}
Input file: {cfg.input_path}
Rows after sampling & cleaning: {row_count:,}
Quick test mode: {cfg.quick_test}
Sample fraction: {cfg.sample_frac}
Lag range: {cfg.lag_min} to {cfg.lag_max} days
RMST tau: {cfg.rmst_tau_days} days
Placebo simulations: {cfg.n_placebo_sims} (note: simulated only among never-vaccinated; eventual vaccinators differ in frailty — consider sensitivity)
Bootstrap reps: {cfg.bootstrap_reps}
Seed: {cfg.bootstrap_seed}
"""
    (cfg.out_dir / "run_info.txt").write_text(info, encoding="utf-8")
    log("Reproducibility info saved.")

# -------------------------
# Data loading
# -------------------------
# Load, clean, and prepare the dataset
def load_and_prepare_data(cfg: Config = CFG) -> pd.DataFrame:
    log(f"Loading CSV: {cfg.input_path}")
    raw = pd.read_csv(cfg.input_path, dtype=str)
    raw.columns = raw.columns.str.strip()

    required = ["Datum_1", "DatumUmrti", "Rok_narozeni"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if cfg.quick_test:
        raw = raw.sample(frac=cfg.sample_frac, random_state=cfg.bootstrap_seed).reset_index(drop=True)
        log(f"Quick test: sampled {cfg.sample_frac*100:.1f}% of data ({len(raw):,} rows)")

    # Canonicalize all date columns used later (timezone-naive, normalized)
    date_cols = ["Datum_1", "DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]
    for c in date_cols:
        if c in raw.columns:
            raw[c] = pd.to_datetime(raw[c], errors="coerce").dt.tz_localize(None).dt.normalize()

    raw["Rok_narozeni"] = pd.to_numeric(raw.get("Rok_narozeni", np.nan), errors="coerce")
    raw["age_ref"] = cfg.age_ref_year - raw["Rok_narozeni"]

    raw = raw[(raw["age_ref"] >= 0) & (raw["age_ref"] <= 120)].reset_index(drop=True)

    raw["study_start"] = pd.to_datetime(cfg.study_start).tz_localize(None)
    raw["study_end"] = pd.to_datetime(cfg.study_end).tz_localize(None)

    raw["death_day"] = (raw["DatumUmrti"] - cfg.study_start).dt.days.astype(float)
    raw["first_dose_day"] = (raw["Datum_1"] - cfg.study_start).dt.days.astype(float)

    raw["subject_id"] = np.arange(len(raw))

    sex_col = next((c for c in raw.columns if "pohlav" in c.lower() or c.lower() == "pohlavi"), None)
    if sex_col:
        raw["sex"] = raw[sex_col].str.upper().map({'M': 0, 'Z': 1, 'F': 1}).fillna(0).astype(int)
    else:
        raw["sex"] = 0

    # Vaccination date bounds warning
    if (raw["Datum_1"] < cfg.study_start).any():
        log_warning("Some Datum_1 < study_start — clamping will apply")
    if (raw["Datum_1"] > cfg.study_end).any():
        log_warning("Some Datum_1 > study_end — may be censored early")

    raw = raw.reset_index(drop=True)  # Clean index

    log(f"Total subjects after cleaning: {len(raw):,}")
    return raw

# -------------------------
# Pre-vaccination mortality
# -------------------------
# Analyze mortality before vaccination to detect selection bias
def pre_vaccination_mortality_analysis(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    vacc_col = "Datum_1"
    death_col = "DatumUmrti"
    vaccinated = df[~df[vacc_col].isna()].copy()
    log(f"[Pre-vax] Vaccinated count: {len(vaccinated):,}")

    vaccinated = vaccinated.sort_values(vacc_col).reset_index(drop=True)
    try:
        vaccinated["q"] = pd.qcut(vaccinated[vacc_col].rank(method="first"), q=cfg.vacc_quantiles, labels=False) + 1
    except Exception as e:
        log(f"qcut failed: {e}. Falling back to cut on ranks.")
        vaccinated["q"] = pd.cut(vaccinated[vacc_col].rank(method="first"), bins=cfg.vacc_quantiles, labels=False) + 1

    summaries = []
    for q in sorted(vaccinated["q"].unique()):
        sub = vaccinated[vaccinated["q"] == q].copy()
        T_q = sub[vacc_col].median()
        window_start = T_q - pd.to_timedelta(cfg.pre_vax_window_days, unit="D")
        window_end = T_q

        sub["died_in_pre"] = (~sub[death_col].isna()) & (sub[death_col] >= window_start) & (sub[death_col] < window_end)
        vax_rate = sub["died_in_pre"].mean()

        at_risk = df[(df["study_start"] <= T_q) & ((df[death_col].isna()) | (df[death_col] > window_start))]
        if cfg.restrict_pre_vax_comparator_to_eventual_vaccinators:
            comp = at_risk[(~at_risk[vacc_col].isna()) & (at_risk[vacc_col] >= T_q)].copy()
        else:
            comp = at_risk[(at_risk[vacc_col].isna()) | (at_risk[vacc_col] >= T_q)].copy()

        comp["died_in_pre"] = (~comp[death_col].isna()) & (comp[death_col] >= window_start) & (comp[death_col] < window_end)
        comp_rate = comp["died_in_pre"].mean()

        summaries.append({
            "quantile": int(q),
            "T_q": T_q,
            "vax_pre_mortality": float(vax_rate),
            "comp_pre_mortality": float(comp_rate),
            "rate_diff": float(vax_rate - comp_rate),
            "n_vax": int(len(sub)),
            "n_comp": int(len(comp))
        })

    summary_df = pd.DataFrame(summaries).sort_values("T_q")
    summary_df.to_csv(cfg.out_dir / "pre_vax_summary.csv", index=False)

    plt.figure(figsize=cfg.figsize)
    plt.plot(summary_df["T_q"], summary_df["vax_pre_mortality"], marker="o", label="Vaccinated")
    plt.plot(summary_df["T_q"], summary_df["comp_pre_mortality"], marker="o", label="Comparator")
    plt.xlabel("Vaccination time quantile")
    plt.ylabel(f"Mortality in {cfg.pre_vax_window_days} days before")
    plt.title("Pre-vaccination mortality")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "pre_vax_mortality.png", dpi=300)
    plt.close()

    return summary_df

# -------------------------
# Streaming RMST computation (low memory)
# -------------------------
# Compute RMST with IF variance for real and placebo data
def compute_rmst_if_or_bootstrap_with_km(df: pd.DataFrame, cfg: Config = CFG):
    df = df.reset_index(drop=True)  # Ensure clean, unique index

    tau = cfg.rmst_tau_days
    results = []
    diag_rows = []
    rng = np.random.default_rng(cfg.bootstrap_seed)

    rel_days = range(cfg.lag_min, cfg.lag_max + 1)

    for rel_day in tqdm(rel_days, desc="Streaming RMST per rel_day"):
        calendar_day = df["vacc_date"] + pd.to_timedelta(rel_day, unit="D")
        valid_mask = (calendar_day >= df["study_start"]) & (calendar_day <= df["study_end"]) & \
                     ((df["death_date"].isna()) | (df["death_date"] > calendar_day))
        risk_set = df[valid_mask].copy()
        if len(risk_set) == 0:
            results.append({"rel_day": int(rel_day), "rmst": np.nan, "n": 0, "rmst_se": np.nan,
                            "rmst_ci_lower": np.nan, "rmst_ci_upper": np.nan, "method": None, "rmst_km": np.nan})
            diag_rows.append({"rel_day": int(rel_day), "n": 0, "median_censor_time": np.nan,
                              "prop_censored_before_tau": np.nan})
            continue

        risk_set["calendar_day"] = calendar_day[valid_mask]

        raw_ttd = (risk_set["death_date"] - risk_set["calendar_day"]).dt.days.fillna(np.inf).astype(float)
        censor_time = (risk_set["study_end"] - risk_set["calendar_day"]).dt.days.astype(float)
        obs_time = np.minimum(np.minimum(raw_ttd, censor_time), tau)
        event = ((raw_ttd <= censor_time) & (raw_ttd <= tau)).astype(bool)
        n = len(obs_time)

        raw_ttd = np.asarray(raw_ttd)
        censor_time = np.asarray(censor_time)
        obs_time = np.asarray(obs_time)
        event = np.asarray(event)

        median_censor = float(np.median(censor_time))
        prop_censored = float(np.mean(censor_time < tau))
        diag_rows.append({"rel_day": int(rel_day), "n": n, "median_censor_time": median_censor,
                          "prop_censored_before_tau": prop_censored})

        # KM RMST
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(obs_time, event)
            sf = kmf.survival_function_.reset_index()
            timeline_col, surv_col = sf.columns[0], sf.columns[1]
            sf = sf.rename(columns={timeline_col: 'timeline', surv_col: 'surv'})
            if sf.empty:
                rmst_km = np.nan
            else:
                if sf['timeline'].iloc[0] > 0:
                    sf = pd.concat([pd.DataFrame({'timeline': [0.], 'surv': [1.]}), sf], ignore_index=True)
                if tau not in sf['timeline'].values:
                    last_surv = sf[sf['timeline'] < tau]['surv'].iloc[-1] if any(sf['timeline'] < tau) else 1.
                    sf = pd.concat([sf, pd.DataFrame({'timeline': [float(tau)], 'surv': [last_surv]})], ignore_index=True)
                sf = sf.sort_values('timeline').reset_index(drop=True)
                deltas = sf['timeline'].diff().fillna(0).values[1:]
                rmst_km = np.sum(sf['surv'].values[:-1] * deltas)
        except Exception as e:
            log(f"KM failed at rel_day {rel_day}: {e}")
            rmst_km = np.nan

        # RMST estimation
        se = np.nan
        lower = np.nan
        upper = np.nan
        method = "none"
        min_Y = np.nan
        if n < cfg.small_n_threshold:
            if cfg.bootstrap_reps < 2:
                rmst_val = float(np.mean(obs_time))
                se = 0.0
                lower = upper = rmst_val
                log_warning(f"Bootstrap skipped at rel_day {rel_day} (reps={cfg.bootstrap_reps})")
            else:
                boot_stats = [np.mean(obs_time[rng.integers(0, n, n)]) for _ in range(cfg.bootstrap_reps)]
                rmst_val = float(np.mean(obs_time))
                se = float(np.std(boot_stats, ddof=1))
                lower = rmst_val - 1.96 * se
                upper = rmst_val + 1.96 * se
            method = "bootstrap"
        else:
            obs_time_days = np.floor(obs_time).astype(int)
            event_mask = event & (obs_time_days <= tau)
            event_times = np.sort(np.unique(obs_time_days[event_mask]))
            if len(event_times) == 0:
                rmst_val = float(np.mean(obs_time))
                method = "mean"
                log_warning(f"No events at rel_day {rel_day}: using mean(obs_time)")
            else:
                Y = np.array([np.sum(obs_time_days >= t) for t in event_times], dtype=float)
                d = np.array([np.sum((obs_time_days == t) & event) for t in event_times], dtype=float)

                # Safer denominator handling: avoid dividing by zero or tiny counts
                Y_safe = Y.copy()
                Y_safe[Y_safe < 1] = np.nan
                dh = np.divide(d, Y_safe, out=np.zeros_like(d, dtype=float), where=~np.isnan(Y_safe))

                S_after = np.cumprod(1 - dh)
                grid = np.concatenate(([0.], event_times, [tau]))
                S_on_interval = [1.] + list(S_after)
                deltas = grid[1:] - grid[:-1]
                rmst_val = float(np.sum(S_on_interval * deltas))

                m = len(event_times)
                T_matrix = np.repeat(obs_time_days[:, None], m, axis=1)
                E_matrix = np.repeat(event[:, None], m, axis=1)
                event_times_matrix = np.repeat(event_times[None, :], n, axis=0)
                if T_matrix.shape != (n, m):
                    log_warning(f"T_matrix shape unexpected at rel_day {rel_day}: {T_matrix.shape} vs {(n,m)}")
                    rmst_val = float(np.mean(obs_time))
                    se = np.nan
                    method = "IF_shape_mismatch_fallback"
                else:
                    DeltaN = ((T_matrix == event_times_matrix) & E_matrix).astype(float)
                    Y_i = (T_matrix >= event_times_matrix).astype(float)
                    denom = Y.copy()
                    denom[denom == 0] = np.nan
                    A_raw = (DeltaN - Y_i * dh[None, :])
                    A = np.where(np.isnan(denom[None, :]), 0.0, A_raw / denom[None, :])
                    cumA = np.cumsum(A, axis=1)
                    phi_matrix = - (cumA * S_after[None, :])
                    if phi_matrix.shape != (n, m):
                        log_warning(f"phi_matrix shape unexpected at rel_day {rel_day}: {phi_matrix.shape} vs {(n,m)}")
                        rmst_val = float(np.mean(obs_time))
                        se = np.nan
                        method = "IF_shape_mismatch_fallback"
                    else:
                        phi_rmst = np.sum(phi_matrix * deltas[1:][None, :], axis=1)
                        var_phi = np.var(phi_rmst, ddof=1) if n > 1 else 0.0
                        var_phi = 0.0 if not np.isfinite(var_phi) else var_phi
                        se = float(np.sqrt(var_phi / n)) if n > 0 else 0.0
                        lower = rmst_val - 1.96 * se
                        upper = rmst_val + 1.96 * se
                        method = "IF"

                        min_Y = float(np.nanmin(Y)) if len(Y) > 0 else np.nan
                        if not np.isnan(min_Y) and min_Y < cfg.min_y_threshold:
                            se = np.nan
                            lower = np.nan
                            upper = np.nan
                            method = "IF_suppressed_small_Y"
                            log_warning(f"Suppressed IF variance at rel_day {rel_day}: min_Y = {min_Y:.0f} < {cfg.min_y_threshold}")

        if prop_censored > 0.5:
            se = np.nan
            lower = np.nan
            upper = np.nan
            method = f"{method}_censored"
            log_warning(f"Heavy censoring at rel_day {rel_day}: suppressed CI (prop={prop_censored:.2f})")

        if rmst_km > 0:
            rel_diff = abs(rmst_val - rmst_km) / rmst_km
            if rel_diff > 0.05:
                log_warning(f"KM-IF discrepancy at rel_day {rel_day}: {rel_diff:.3f} (n={n}, median_censor={median_censor:.1f}, prop_censored={prop_censored:.2f})")
        elif rmst_km == 0 and rmst_val > 0:
            log_warning(f"KM=0 but RMST>0 at rel_day {rel_day} (n={n}, median_censor={median_censor:.1f}, prop_censored={prop_censored:.2f})")

        results.append({
            "rel_day": int(rel_day),
            "rmst": float(rmst_val),
            "n": int(n),
            "rmst_se": float(se),
            "rmst_ci_lower": float(lower),
            "rmst_ci_upper": float(upper),
            "method": method,
            "rmst_km": float(rmst_km)
        })

        diag_rows[-1]["min_Y"] = min_Y if not np.isnan(min_Y) else np.nan

    res_df = pd.DataFrame(results).sort_values("rel_day")
    diag_df = pd.DataFrame(diag_rows).sort_values("rel_day")
    diag_df["flag_median_censor_lt_tau"] = diag_df["median_censor_time"] < tau
    diag_df["flag_prop_censored_gt_50pct"] = diag_df["prop_censored_before_tau"] > 0.5
    diag_df["flag_unreliable"] = (diag_df["n"] < cfg.small_n_threshold) | (diag_df["flag_prop_censored_gt_50pct"])

    return res_df, diag_df

# -------------------------
# Hazard estimation
# -------------------------
# Estimate daily vaccination hazards for placebo simulation
def estimate_calendar_uptake_hazard(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    if df["Datum_1"].isna().all():
        raise ValueError("No vaccination dates found — cannot estimate uptake hazard")

    df = df.copy()
    bins = cfg.age_bins
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]
    min_v = df["Datum_1"].min()
    max_v = df["Datum_1"].max()
    if pd.isna(min_v) or pd.isna(max_v):
        raise ValueError("Vaccination date range is empty or NaT")

    days = pd.date_range(min_v.normalize(), max_v.normalize(), freq="D")
    records = []
    for day in tqdm(days, desc="Uptake hazard"):
        at_risk = df[(df["study_start"] <= day) & ((df["DatumUmrti"].isna()) | (df["DatumUmrti"] > day))]
        if at_risk.empty:
            continue
        at_risk = at_risk.copy()  # Avoid chained assignment warning
        at_risk["age_at_day"] = day.year - at_risk["Rok_narozeni"]
        at_risk["age_group"] = pd.cut(at_risk["age_at_day"], bins=bins, labels=labels, right=False)
        at_risk["vacc_today"] = at_risk["Datum_1"].dt.normalize() == day
        grp = at_risk.groupby(["age_group", "sex"], dropna=False).agg(
            n_at_risk=("subject_id", "count"),
            n_vacc=("vacc_today", "sum")
        ).reset_index()
        grp["calendar_day"] = day
        grp["hazard"] = grp["n_vacc"] / grp["n_at_risk"].replace(0, np.nan)
        records.append(grp)

    if len(records) == 0:
        raise RuntimeError("No hazard records generated — check input dates and study window")
    hazard_df = pd.concat(records, ignore_index=True)
    hazard_df.to_csv(cfg.out_dir / "hazard_df.csv", index=False)

    # Hazard sanity assert
    if hazard_df['hazard'].max() == 0 or hazard_df['hazard'].isna().all():
        raise RuntimeError("All hazards zero or NaN — check age_group labels and date normalization")

    # Hazard group assert
    expected_labels = set([f"[{cfg.age_bins[i]},{cfg.age_bins[i+1]})" for i in range(len(cfg.age_bins)-1)])
    hazard_groups = set(hazard_df['age_group'].astype(str).unique())
    if not hazard_groups.issuperset(expected_labels):
        raise RuntimeError("Age group labels in hazard_df do not match expected labels; check age_bins and normalization")

    return hazard_df

# -------------------------
# Vectorized Placebo simulation (compact, faster)
# -------------------------
# Simulate placebo vaccination dates using vectorized hazards
def simulate_placebo_vectorized(df: pd.DataFrame, hazard_df: pd.DataFrame, cfg: Config = CFG, seed: int = None) -> pd.DataFrame:
    """
    Vectorized placebo assignment:
    - Build hazard matrix indexed by (age_group_label, sex) x days
    - For never-vaccinated subjects, map them to a group index and draw uniforms
      to find the first day where U < hazard(group, day).
    - Fallback to chunked processing if memory estimate too large.
    """
    rng = np.random.default_rng(seed or cfg.bootstrap_seed)

    # Work only with never-vaccinated subjects
    never_df = df[df["Datum_1"].isna()].copy().reset_index(drop=True)
    if never_df.empty:
        # Nothing to simulate
        return pd.DataFrame({"subject_id": [], "placebo_vacc_date": []})

    # Prepare day list and group labels
    days = np.array(sorted(pd.to_datetime(hazard_df["calendar_day"]).dt.normalize().unique()))
    n_days = len(days)
    bins = cfg.age_bins
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]

    # Build group keys present in hazard_df (canonicalized)
    hazard_df = hazard_df.copy()
    hazard_df["age_group_str"] = hazard_df["age_group"].astype(str)
    group_keys = sorted(hazard_df[["age_group_str", "sex"]].drop_duplicates().apply(tuple, axis=1).tolist())

    # Map group key -> row index
    group_to_idx = {g: i for i, g in enumerate(group_keys)}
    n_groups = len(group_keys)

    # Build hazard matrix shape (n_groups, n_days), fill with 0.0
    hazard_matrix = np.zeros((n_groups, n_days), dtype=float)

    # Build mapping from day to column index (canonicalized)
    day_to_col = {pd.Timestamp(d).normalize(): i for i, d in enumerate(days)}

    # Fill hazard_matrix using canonical keys
    for _, row in hazard_df.iterrows():
        g = (str(row["age_group"]), int(row["sex"]))
        # Use canonicalized age_group_str if available
        if (row.get("age_group_str", None) is not None):
            g = (row["age_group_str"], int(row["sex"]))
        if g not in group_to_idx:
            # skip groups not in mapping (shouldn't happen)
            continue
        gi = group_to_idx[g]
        di = day_to_col.get(pd.to_datetime(row["calendar_day"]).normalize(), None)
        if di is None:
            continue
        hazard_matrix[gi, di] = float(row["hazard"]) if not pd.isna(row["hazard"]) else 0.0

    # Compute subject-level group index using age at study_start (fast, consistent with earlier df_light)
    never_df["age_at_ref"] = cfg.study_start.year - never_df["Rok_narozeni"]
    never_df["age_group"] = pd.cut(never_df["age_at_ref"], bins=bins, labels=labels, right=False).astype(str)
    never_df["sex_int"] = never_df["sex"].astype(int)

    # Map subject to group index; if group missing in hazard groups, assign -1 (no hazard)
    subj_group_keys = list(zip(never_df["age_group"].tolist(), never_df["sex_int"].tolist()))
    subj_group_idx = np.array([group_to_idx.get(k, -1) for k in subj_group_keys], dtype=int)

    n_subj = len(never_df)

    # Memory guard: estimate size of boolean matrix n_subj * n_days
    est_cells = int(n_subj) * int(n_days)
    log(f"Placebo vectorized estimate: subjects={n_subj}, days={n_days}, cells={est_cells}")
    max_cells = int(5e7)  # ~50 million booleans ~50 MB; adjust to machine RAM
    if est_cells <= max_cells:
        # Vectorized path: build hazards_subjects (n_subj, n_days) by advanced indexing
        valid_mask = subj_group_idx >= 0
        hazards_subjects = np.zeros((n_subj, n_days), dtype=float)
        if valid_mask.any():
            hazards_subjects[valid_mask, :] = hazard_matrix[subj_group_idx[valid_mask], :]

        # Draw uniforms and compare
        rand = rng.random((n_subj, n_days))
        vax_mask = rand < hazards_subjects  # True where vaccinated on that day

        # For each subject, find first True index; if none, set -1
        any_vax = vax_mask.any(axis=1)
        first_idx = np.where(any_vax, vax_mask.argmax(axis=1), -1)

        # Map indices to dates
        placebo_dates = []
        for idx in first_idx:
            if idx == -1:
                placebo_dates.append(pd.NaT)
            else:
                placebo_dates.append(pd.to_datetime(days[int(idx)]).normalize())

        never_df["placebo_vacc_date"] = placebo_dates
    else:
        # Chunked fallback to avoid memory blowup
        log_warning(f"Vectorized assignment too large (cells={est_cells}), using chunked fallback")
        chunk_size = max(1, int(max_cells // n_days))
        placebo_dates = [pd.NaT] * n_subj
        for start in tqdm(range(0, n_subj, chunk_size), desc="Placebo sim chunks"):
            end = min(n_subj, start + chunk_size)
            idxs = np.arange(start, end)
            valid_mask = subj_group_idx[idxs] >= 0
            hazards_chunk = np.zeros((len(idxs), n_days), dtype=float)
            if valid_mask.any():
                hazards_chunk[valid_mask, :] = hazard_matrix[subj_group_idx[idxs[valid_mask]], :]
            rand = rng.random((len(idxs), n_days))
            vax_mask = rand < hazards_chunk
            any_vax = vax_mask.any(axis=1)
            first_idx = np.where(any_vax, vax_mask.argmax(axis=1), -1)
            for i_local, fi in enumerate(first_idx):
                if fi == -1:
                    placebo_dates[start + i_local] = pd.NaT
                else:
                    placebo_dates[start + i_local] = pd.to_datetime(days[int(fi)]).normalize()
        never_df["placebo_vacc_date"] = placebo_dates

    # Return DataFrame with subject_id and placebo_vacc_date
    out = never_df[["subject_id", "placebo_vacc_date"]].copy()
    return out

# -------------------------
# Main pipeline
# -------------------------
# Run the full bias-necessity audit pipeline
def run_pipeline(cfg: Config = CFG):
    df = load_and_prepare_data(cfg)
    df = df.reset_index(drop=True)  # Ensure clean index
    df["Datum_1_original"] = df["Datum_1"].copy()  # Preserve original vaccination status
    save_run_info(cfg, len(df))

    pre_vax_summary = pre_vaccination_mortality_analysis(df, cfg)

    # Real vaccinated
    vaccinated = df[~df["Datum_1"].isna()].copy()
    vaccinated = vaccinated.reset_index(drop=True)  # Clean index
    vaccinated["vacc_date"] = vaccinated["Datum_1"].dt.normalize()  # Explicit normalization
    vaccinated["death_date"] = vaccinated["DatumUmrti"]
    log("[RMST] Real vaccination...")
    rmst_real, diag_real = compute_rmst_if_or_bootstrap_with_km(vaccinated, cfg)
    rmst_real.to_csv(cfg.out_dir / "rmst_real.csv", index=False)
    diag_real.to_csv(cfg.out_dir / "rel_day_diagnostics.csv", index=False)

    log("[Placebo] Hazard estimation...")
    hazard_df = estimate_calendar_uptake_hazard(df, cfg)

    log(f"[Placebo] {cfg.n_placebo_sims} simulations (vectorized)...")
    # Run vectorized simulation n_placebo_sims times, saving wide table
    sims = []
    seeds = []
    rng_global = np.random.default_rng(cfg.bootstrap_seed)
    for s in range(cfg.n_placebo_sims):
        seed = int(rng_global.integers(0, 2**31 - 1))
        seeds.append(seed)
        sim_out = simulate_placebo_vectorized(df, hazard_df, cfg, seed=seed)
        sim_out = sim_out.rename(columns={"placebo_vacc_date": f"placebo_vacc_date_sim{s}"})
        sims.append(sim_out)

    # Merge sims into wide table by subject_id using safe merge
    merged = pd.DataFrame({"subject_id": df["subject_id"].values})
    for sim in sims:
        merged = merged.merge(sim, on="subject_id", how="left")

    merged = merged.reset_index(drop=True)

    # Defensive checks on merged wide placebo table
    if merged["subject_id"].duplicated().any():
        dup_ids = merged["subject_id"][merged["subject_id"].duplicated()].unique()
        raise RuntimeError(f"Duplicate subject_id in merged placebo table: {len(dup_ids)} duplicates")
    if len(merged) != len(df):
        raise RuntimeError(f"Length mismatch: merged ({len(merged)}) vs df ({len(df)})")

    # Normalize placebo date columns
    for c in merged.columns:
        if c.startswith("placebo_vacc_date_sim"):
            merged[c] = pd.to_datetime(merged[c], errors="coerce").dt.normalize()

    n_assigned = merged[[c for c in merged.columns if c.startswith("placebo_vacc_date_sim")]].notna().sum().sum()
    log(f"Total placebo assignments across sims: {int(n_assigned)}")

    merged.to_csv(cfg.out_dir / "placebo_sim_dates_wide.csv", index=False)

    with open(cfg.out_dir / "run_info.txt", "a", encoding="utf-8") as f:
        f.write("\nPlacebo simulation seeds: " + ", ".join(map(str, seeds)) + "\n")

    # Placebo nonempty check
    if merged[f'placebo_vacc_date_sim0'].notna().sum() == 0:
        raise RuntimeError("No placebo vaccinations assigned in sim 1 — check hazard lookup and age_group mapping")

    # Compute RMST for each placebo sim
    rmst_placebo_list = []
    for s in range(cfg.n_placebo_sims):
        col = f"placebo_vacc_date_sim{s}"
        # Create df_sim by mapping the wide column into Datum_1 for everyone (explicit alignment)
        place_map = merged.set_index("subject_id")[col]
        df_sim = df.copy().reset_index(drop=True)
        df_sim["Datum_1"] = df_sim["subject_id"].map(place_map)
        # Ensure Datum_1 is normalized datetimes or NaT
        df_sim["Datum_1"] = pd.to_datetime(df_sim["Datum_1"], errors="coerce").dt.normalize()

        never_vacc_mask = df_sim["Datum_1_original"].isna() & df_sim["Datum_1"].notna()
        df_sim_placebo = df_sim.loc[never_vacc_mask].copy().reset_index(drop=True)

        if len(df_sim_placebo) == 0:
            raise RuntimeError(f"No placebo vaccinations assigned in sim {s+1}. Check hazard lookup and simulation.")

        log(f"[Placebo sim {s+1}] Processing {len(df_sim_placebo):,} never-vaccinated individuals with placebo dates")

        df_sim_placebo["vacc_date"] = df_sim_placebo["Datum_1"].dt.normalize()
        df_sim_placebo["death_date"] = df_sim_placebo["DatumUmrti"]
        rmst_p, _ = compute_rmst_if_or_bootstrap_with_km(df_sim_placebo, cfg)
        rmst_p = rmst_p.rename(columns={"rmst": f"rmst_sim{s}"})
        rmst_placebo_list.append(rmst_p[["rel_day", f"rmst_sim{s}"]])

    # Merge placebo RMSTs across sims
    all_rel_days = sorted(set().union(*(set(d["rel_day"]) for d in rmst_placebo_list)))
    rmst_placebo_merged = pd.DataFrame({"rel_day": all_rel_days})
    for s_df in rmst_placebo_list:
        rmst_placebo_merged = rmst_placebo_merged.merge(s_df, on="rel_day", how="left")

    sim_cols = [c for c in rmst_placebo_merged.columns if c.startswith("rmst_sim")]
    vals = rmst_placebo_merged[sim_cols].values
    rmst_placebo_merged["rmst_placebo_mean"] = np.nanmean(vals, axis=1)
    rmst_placebo_merged["rmst_placebo_lower"] = np.nanpercentile(vals, 2.5, axis=1)
    rmst_placebo_merged["rmst_placebo_upper"] = np.nanpercentile(vals, 97.5, axis=1)
    rmst_placebo_merged.to_csv(cfg.out_dir / "rmst_placebo_sims_summary.csv", index=False)

    rmst_placebo_summary = rmst_placebo_merged[["rel_day", "rmst_placebo_mean", "rmst_placebo_lower", "rmst_placebo_upper"]].rename(
        columns={"rmst_placebo_mean": "rmst", "rmst_placebo_lower": "rmst_ci_lower", "rmst_placebo_upper": "rmst_ci_upper"}
    )

    plot_df = pd.merge(
        rmst_real[["rel_day", "rmst", "rmst_ci_lower", "rmst_ci_upper"]].rename(columns={"rmst": "rmst_real", "rmst_ci_lower": "real_lower", "rmst_ci_upper": "real_upper"}),
        rmst_placebo_summary.rename(columns={"rmst": "rmst_placebo", "rmst_ci_lower": "placebo_lower", "rmst_ci_upper": "placebo_upper"}),
        on="rel_day",
        how="inner"
    ).dropna(subset=["rmst_real", "rmst_placebo"])

    unreliable_rel_days = diag_real[diag_real["flag_unreliable"]]["rel_day"].unique()

    plt.figure(figsize=cfg.figsize)
    plt.plot(plot_df["rel_day"], plot_df["rmst_real"], label="Real vaccination", color="C0")
    plt.fill_between(plot_df["rel_day"], plot_df["real_lower"], plot_df["real_upper"], color="C0", alpha=0.2)
    plt.plot(plot_df["rel_day"], plot_df["rmst_placebo"], label="Placebo mean", color="C1")
    plt.fill_between(plot_df["rel_day"], plot_df["placebo_lower"], plot_df["placebo_upper"], color="C1", alpha=0.2)
    for unreliable in unreliable_rel_days:
        plt.axvspan(unreliable - 0.5, unreliable + 0.5, color='grey', alpha=0.3)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Days since vaccination / placebo")
    plt.ylabel(f"RMST over next {cfg.rmst_tau_days} days")
    plt.title("Real vs Placebo RMST curves")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "rmst_real_vs_placebo_mean.png", dpi=300)
    plt.close()

    plot_df["rmst_diff"] = plot_df["rmst_real"] - plot_df["rmst_placebo"]
    plot_df.to_csv(cfg.out_dir / "rmst_difference_real_minus_placebo_mean.csv", index=False)
    plt.figure(figsize=cfg.figsize)
    plt.plot(plot_df["rel_day"], plot_df["rmst_diff"], color="C2")
    for unreliable in unreliable_rel_days:
        plt.axvspan(unreliable - 0.5, unreliable + 0.5, color='grey', alpha=0.3)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Days since vaccination / placebo")
    plt.ylabel("RMST difference (real - placebo)")
    plt.title("RMST difference")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(cfg.out_dir / "rmst_difference.png", dpi=300)
    plt.close()

    log(f"Pipeline finished successfully.")
    log(f"Results saved in: {cfg.out_dir.resolve()}")
    log("Check run_info.txt and rel_day_diagnostics.csv for quality control.")

if __name__ == "__main__":
    log("Starting bias-necessity RMST pipeline (vectorized placebo) - patched.")
    run_pipeline(CFG)
    log("Done.")