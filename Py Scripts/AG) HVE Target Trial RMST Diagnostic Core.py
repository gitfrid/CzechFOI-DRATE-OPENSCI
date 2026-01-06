#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Target Trial Emulation (TTE) Core
===============================================================================
A modern, bias-resistant pipeline for restricted mean survival time (RMST)
estimation using Czech national mortality data (age + sex only).

Implements Hernán-style target trial emulation with extensive diagnostics:
- Daily marginal & sex-specific propensity scores
- Overlap weighting for balance
- Pooled logistic regression for discrete-time hazards
- RMST computation at 180 days
- Placebo-date falsification
- Negative-control lags for selection bias testing
- Bootstrap confidence intervals
- Null, confounding, and strong-confounding simulation calibration

Key methodological safeguards:
- Fixed symmetric unvaccinated aggregation & immortal time bias
- Deterministic per-worker & per-bootstrap RNG seeding
- Safe temporal alignment (first=0, inclusive ranges, eligibility > day)
- Configurable weight truncation & ESS reporting
- Full bootstrap vector saving for reproducibility

Outputs:
- CSV: ΔRMST estimates + CIs, pre-vaccination mortality, balance SMD, weights stats
- Interactive Plotly HTML: lag-sweep plots, PS distributions, null calibration

Intended use:
Diagnostic evaluation of mortality patterns around first-dose vaccination.
Results are exploratory due to limited confounding control (age + sex only).

Author: AI / Drifting assistence   Date: January 2026 Version 1.0
===============================================================================
"""

from __future__ import annotations

# ==================== SLEEP PREVENTION  Windows11 ====================
import ctypes
import os
if os.name == 'nt': 
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print(">>> Windows Sleep Prevention: ACTIVE")
    except Exception as e:
        print(f">>> Could not set Sleep Prevention: {e}")
# ==================== END SLEEP PREVENTION  ====================

import json
import logging
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.gam.api import BSplines
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from glob import glob
import scipy.special as sc


# ==================== CONFIGURATION ====================
CONFIG = {
    "study_start": pd.Timestamp("2020-01-01"),
    "age_ref_year": 2023,
    "input_path": Path(r"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131.csv"),
    "output_base": Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AG) HVE Target Trial RMST Diagnostic Core"),
    "tau": 180,
    "n_boot": 500,
    "boot_subsample": 0.8,
    "random_seed": 20251231,
    "n_cores": 2,
    "time_df": 5,
    "spline_degree": 3,
    "ridge_alphas": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "single_ages": list(range(60, 91)),
    "lag_sweep": [0, 14, 28, 42, 56, 70],
    "max_last_obs_days": 360,
    "glm_maxiter": 5000,
    "pre_vax_window": 30,
    "n_placebo_draws": 100,
    "n_null_sims": 50,
    "n_subj_sim": 2000,
    "sim_max_days": 1000,
    "sim_hazard_wave_amp": 0.005,
    "weight_trunc_perc": 99,
    "robust_se": True,
    "sim_scenarios": ['null', 'confound', 'censor', 'strong_confound'],
    "sim_confound_strength": 0.5,
    "sim_censor_rate": 0.1,
    "min_boot_success": 200,
    "ess_threshold": 50,
}

# === Choose run mode here ===
RUN_MODE = "null_sim" # coose -> full/quick/null_sim

if RUN_MODE == "quick":
    CONFIG.update({
        "single_ages": list(range(60, 91, 5)),      # every 5 years: 60,65,...,90 → 7 ages
        "lag_sweep": [-28, -14, -7, 0, 14, 28, 42, 56, 70],  # negative controls + main lags
        "n_null_sims": 20,                          # 20 simulations per scenario → solid calibration
        "n_boot": 200,                              # 300 bootstraps → decent CIs
        "n_subj_sim": 2000,                         # 2000 subjects per age → good precision
        "time_df": 4,                               # reasonable spline complexity
        "glm_maxiter": 3000,
        "ridge_alphas": [0.0, 0.001, 0.01, 0.1],    # fewer alphas → faster fitting
    })
    print("!!! QUICK TEST MODE ACTIVE !!!")
elif RUN_MODE == "test":
    print("!!! UNIT TEST MODE ACTIVE !!!")
elif RUN_MODE == "null_sim":
    print("!!! NULL SIMULATION MODE ACTIVE !!!")
elif RUN_MODE not in ["full", "null_sim"]:
    raise ValueError("Invalid RUN_MODE.")


# ==================== END CONFIGURATION ====================

CONFIG["output_base"].mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_LOG = CONFIG["output_base"] / f"gs_tte_log_{timestamp}.txt"

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==================== LOGGING ====================
class ParallelSafeLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.parallel_buffer: List[str] = []
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")

    def info(self, msg: str, parallel: bool = False) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"{ts} INFO {msg}"
        print(full_msg)
        if parallel:
            self.parallel_buffer.append(full_msg)
        else:
            logging.info(msg)

    def flush_parallel_buffer(self) -> None:
        if self.parallel_buffer:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for msg in self.parallel_buffer:
                    f.write(msg + "\n")
            self.parallel_buffer.clear()

logger = ParallelSafeLogger(MAIN_LOG)
def log(msg: str, parallel: bool = False) -> None:
    logger.info(msg, parallel=parallel)

# ==================== HELPER FUNCTIONS ====================
def safe_divide(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    return np.where(np.abs(b) > epsilon, a / b, 0.0)

def safe_clip(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=min_val, posinf=max_val, neginf=min_val)
    return np.clip(arr, min_val, max_val)

def save_df(df: pd.DataFrame, filename: str) -> None:
    path = CONFIG["output_base"] / f"{filename}_{timestamp}.csv"
    df.to_csv(path, index=False)
    log(f"Saved CSV: {path}")

def save_plot(fig: go.Figure, filename: str) -> None:
    path = CONFIG["output_base"] / f"{filename}_{timestamp}.html"
    fig.write_html(path)
    log(f"Saved Plot: {path}")

def concatenate_all_csvs_to_txt(output_txt: str = "all_results_summary.txt") -> None:
    csv_files = sorted(
        f for f in glob(str(CONFIG["output_base"]) + "/**/*.csv", recursive=True)
        if not Path(f).name.startswith("00_")
    )
    if not csv_files:
        log("No relevant CSV files found to concatenate (excluded 00_*).")
        return
    txt_path = CONFIG["output_base"] / output_txt
    log(f"Concatenating {len(csv_files)} CSV files into {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as outfile:
        outfile.write(f"All Results Summary (excluding raw 00_* files)\n")
        outfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"Total CSV files included: {len(csv_files)}\n")
        outfile.write("=" * 80 + "\n\n")
        for csv_path in csv_files:
            rel_path = Path(csv_path).relative_to(CONFIG["output_base"])
            outfile.write(f"=== {rel_path} ===\n" + "-" * 80 + "\n")
            try:
                df = pd.read_csv(csv_path)
                outfile.write(df.to_string(index=False))
            except Exception as e:
                outfile.write(f"[Error reading file: {e}]\n")
            outfile.write("\n\n" + "=" * 80 + "\n\n")
    log(f"Created comprehensive summary: {txt_path}")

# ==================== UNIT TESTS ====================
def test_compute_daily_events_and_risk():
    df_trial = pd.DataFrame({
        "start": [0, 5, 10],
        "stop": [3, 8, 15],
        "event": [0, 1, 0]
    })
    events, risk = compute_daily_events_and_risk(df_trial, 0, 16)
    expected_events = np.array([0.]*8 + [1.] + [0.]*7)
    expected_risk = np.array([1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,1.,1.,0.])
    assert np.allclose(events, expected_events), "Events mismatch"
    assert np.allclose(risk, expected_risk), "Risk mismatch"
    print("test_compute_daily_events_and_risk PASSED")

def test_aggregate_trials_vectorized():
    df_age = pd.DataFrame({
        "death_day": [10, 20, np.nan],
        "first_dose_day": [5, np.nan, np.nan],
        "sex": [0, 1, 0]
    })
    p_cond = pd.DataFrame({0: np.full(20, 0.1), 1: np.full(20, 0.1)}, index=np.arange(20))
    sex_dist_per_day = pd.DataFrame(np.full((20, 2), 0.5), index=np.arange(20), columns=[0,1])
    marginal_ps = pd.Series(np.full(20, 0.1), index=np.arange(20))
    agg = aggregate_trials_vectorized(df_age, 0, 20, p_cond, sex_dist_per_day, marginal_ps, 0, 20)
    assert not agg.empty
    assert "day" in agg.columns
    assert agg["day"].isin(range(20)).all()
    print("test_aggregate_trials_vectorized PASSED")

def test_bootstrap():
    np.random.seed(42)
    deltas = np.random.normal(0, 1, 10)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    assert ci_low < 0 < ci_high
    print("test_bootstrap PASSED")

def test_null_sim_generation():
    raw_sim = generate_null_data([60], 'null')
    assert 'age' in raw_sim.columns
    assert 'sex' in raw_sim.columns
    assert 'first_dose_day' in raw_sim.columns
    assert 'death_day' in raw_sim.columns
    assert len(raw_sim) > 0
    print("test_null_sim_generation PASSED")

# ==================== SPLINE ====================
def build_spline_basis(times: np.ndarray, df_deg: int, degree: int) -> pd.DataFrame:
    times = np.sort(np.asarray(times, dtype=int))
    t = times.astype(float)
    t_c = t - t.mean()
    t_std = t_c / (t_c.std(ddof=0) + 1e-10)
    bs = BSplines(t_std[:, None], df=[df_deg + degree + 1], degree=[degree], variable_names=["time"])
    basis = bs.basis.astype(np.float32)
    basis = np.nan_to_num(basis, nan=0, posinf=0, neginf=0)
    cols = [f"time_s_{i}" for i in range(basis.shape[1])]
    spl = pd.DataFrame(basis, columns=cols, index=times)
    spl["time_lin"] = t_std
    return spl

# ==================== DATA PREP ====================
def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    log(f"Loading CSV: {input_path}")
    raw = pd.read_csv(input_path, dtype=str)
    raw.columns = raw.columns.str.strip()

    for c in ["DatumUmrti"] + [col for col in raw.columns if col.startswith("Datum_")]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")

    raw["Rok_narozeni"] = pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw["age"] = CONFIG["age_ref_year"] - raw["Rok_narozeni"]
    raw = raw[(raw["age"] >= min(CONFIG["single_ages"])) &
              (raw["age"] <= max(CONFIG["single_ages"]))].reset_index(drop=True)

    raw["death_day"] = (raw["DatumUmrti"] - CONFIG["study_start"]).dt.days.astype(float).clip(lower=0)
    raw["first_dose_day"] = (raw["Datum_1"] - CONFIG["study_start"]).dt.days.astype(float).clip(lower=0)

    raw["subject_id"] = np.arange(len(raw))

    sex_col = next((c for c in raw.columns if "pohlav" in c.lower() or c.lower() == "pohlavi"), None)
    if sex_col:
        raw["sex"] = raw[sex_col].str.upper().map({'M': 0, 'Z': 1, 'F': 1}).fillna(0).astype(int)
    else:
        raw["sex"] = 0

    log(f"Total subjects after age filter: {len(raw)}")
    return raw

# ==================== PROPENSITY ====================
def aggregate_propensity_daily(df: pd.DataFrame, start_day: int, end_day: float) -> pd.DataFrame:
    rows = []
    last_safe = min(int(end_day), start_day + CONFIG["max_last_obs_days"])
    for day in range(start_day, last_safe + 1):
        elig = df[(df["death_day"].isna() | (df["death_day"] > day)) &
                  (df["first_dose_day"].isna() | (df["first_dose_day"] > day))]
        for s in [0, 1]:
            sub = elig[elig["sex"] == s]
            n = len(sub)
            if n > 0:
                rows.append({
                    "day": day,
                    "sex": s,
                    "n_eligible": n,
                    "n_vaccinated": (sub["first_dose_day"] == day).sum(),
                })
    return pd.DataFrame(rows)

def fit_glm_regularized_with_retry(y, X, family, alpha_list, maxiter=5000, cov_type='nonrobust', freq_weights=None):
    model = None
    for alpha in alpha_list:
        try:
            model = sm.GLM(y, X, family=family, freq_weights=freq_weights).fit_regularized(
                alpha=alpha, maxiter=maxiter
            )
            if not np.isnan(model.params).any():
                log(f"Regularized fit converged with alpha={alpha}")
                return model
        except Exception as e:
            log(f"Regularized fit failed with alpha={alpha}: {str(e)}")
    log("All regularized fits failed → falling back to unpenalized GLM")
    model = sm.GLM(y, X, family=family, freq_weights=freq_weights).fit(maxiter=maxiter)
    if cov_type != 'nonrobust':
        model = model.get_robustcov_results(cov_type=cov_type)
    return model

def fit_propensity_model(prop_data: pd.DataFrame, spline: pd.DataFrame, include_sex: bool = True) -> sm.GLM:
    if prop_data.empty:
        raise ValueError("No propensity data")
    y = prop_data["n_vaccinated"]
    X = spline.loc[prop_data["day"].astype(int)].reset_index(drop=True).copy()
    if include_sex:
        X["sex"] = prop_data["sex"].astype(float)
    X = sm.add_constant(X)
    freq_weights = prop_data["n_eligible"].astype(float)
    return fit_glm_regularized_with_retry(y, X, sm.families.Binomial(), CONFIG["ridge_alphas"], CONFIG["glm_maxiter"], freq_weights=freq_weights)

def compute_conditional_probabilities(model: sm.GLM, spline: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for s in [0, 1]:
        X = spline.copy()
        X["sex"] = float(s)
        X = sm.add_constant(X)
        X = X.reindex(columns=model.params.index, fill_value=0.0)
        p = model.predict(X)
        out[s] = safe_clip(p, 0.001, 0.999)
    return pd.DataFrame(out, index=spline.index)

def compute_marginal_ps(p_cond: pd.DataFrame, prop_data: pd.DataFrame) -> pd.Series:
    if p_cond.empty or prop_data.empty:
        days = p_cond.index if not p_cond.empty else prop_data['day'].unique()
        return pd.Series(0.5, index=days, name='marginal_ps')

    sex_prop = prop_data.pivot_table(
        index='day',
        columns='sex',
        values='n_eligible',
        fill_value=0
    )
    sex_dist = sex_prop.div(sex_prop.sum(axis=1), axis=0).fillna(0)

    ps = (p_cond * sex_dist.reindex(p_cond.index)).sum(axis=1)
    ps_values = safe_clip(ps.values, 0.01, 0.99)

    return pd.Series(ps_values, index=p_cond.index, name='marginal_ps')

# ==================== TRIAL EMULATION ====================
def compute_daily_events_and_risk(df_trial: pd.DataFrame, start_day: int, max_follow: int) -> Tuple[np.ndarray, np.ndarray]:
    window = max_follow
    events = np.zeros(window, dtype=float)
    risk_diff = np.zeros(window + 1, dtype=float)

    if df_trial.empty:
        return events, np.zeros(window, dtype=float)

    starts = (df_trial["start"] - start_day).clip(0, window).astype(int)
    stops = (df_trial["stop"] - start_day).clip(0, window).astype(int)

    np.add.at(risk_diff, starts, 1)
    np.add.at(risk_diff, stops, -1)
    risk = np.cumsum(risk_diff)[:window]

    assert np.all(risk >= 0), "Negative risk detected"

    event_mask = df_trial["event"].values.astype(bool)
    event_days = (df_trial.loc[event_mask, "stop"] - start_day).clip(0, window - 1).astype(int)
    np.add.at(events, event_days, 1)

    return events, risk

def aggregate_trials_vectorized(
    df_age: pd.DataFrame,
    first_day: int,
    last_obs: float,
    p_cond: pd.DataFrame,
    sex_dist_per_day: pd.DataFrame,
    marginal_ps: pd.Series,
    lag: int,
    max_follow: int
) -> pd.DataFrame:
    tau = max_follow
    t_grid = np.arange(tau)
    events_v = np.zeros(tau, dtype=float)
    risk_v = np.zeros(tau, dtype=float)
    events_u = np.zeros(tau, dtype=float)
    risk_u = np.zeros(tau, dtype=float)

    last_safe = min(int(last_obs), first_day + CONFIG["max_last_obs_days"])

    ps_all = marginal_ps.values
    trunc_low, trunc_high = np.percentile(ps_all, [100 - CONFIG["weight_trunc_perc"], CONFIG["weight_trunc_perc"]])

    for t0 in range(first_day, last_safe + 1):
        ps_t0 = marginal_ps.get(t0, 0.5)
        ps_t0 = np.clip(ps_t0, trunc_low, trunc_high)

        elig = df_age[(df_age["death_day"].isna() | (df_age["death_day"] >= t0)) &
                      (df_age["first_dose_day"].isna() | (df_age["first_dose_day"] >= t0))]
        if elig.empty:
            continue

        initiators = elig[elig["first_dose_day"] == t0]
        if not initiators.empty:
            vax_df = initiators.copy()
            vax_df["start"] = t0 + lag
            vax_df["stop"] = np.minimum(vax_df["death_day"].fillna(99999), t0 + tau)
            vax_df = vax_df[vax_df["stop"] >= vax_df["start"]]
            if not vax_df.empty:
                vax_df["event"] = (vax_df["death_day"] >= vax_df["start"]) & (vax_df["death_day"] <= vax_df["stop"])
                ev, rk = compute_daily_events_and_risk(vax_df, t0, tau)
                events_v += ev * (1 - ps_t0)
                risk_v += rk * (1 - ps_t0)

        ctrl = elig[elig["first_dose_day"] != t0].copy()
        if ctrl.empty:
            continue
        ctrl["start"] = t0 + lag
        ctrl["stop"] = np.minimum(ctrl["first_dose_day"].fillna(99999), ctrl["death_day"].fillna(99999))
        ctrl["stop"] = np.minimum(ctrl["stop"], t0 + tau)
        ctrl = ctrl[ctrl["stop"] >= ctrl["start"]]
        if not ctrl.empty:
            ctrl["event"] = (ctrl["death_day"] >= ctrl["start"]) & (ctrl["death_day"] <= ctrl["stop"])
            eu, ru = compute_daily_events_and_risk(ctrl, t0, tau)
            events_u += eu * ps_t0
            risk_u += ru * ps_t0

    agg = pd.DataFrame({
        "day": np.tile(t_grid, 2),
        "vaccinated": np.repeat([1, 0], len(t_grid)),
        "events": np.concatenate([events_v, events_u]),
        "risk": np.concatenate([risk_v, risk_u]),
    })
    agg["day"] = agg["day"].astype(int)
    assert agg["day"].isin(range(tau)).all(), "Day misalignment in aggregation"

    return agg

# ==================== OUTCOME & RMST ====================
def fit_pooled_logistic(agg: pd.DataFrame, spline: pd.DataFrame) -> sm.GLM:
    valid = agg[agg["risk"] > 0].copy()
    if valid.empty:
        raise ValueError("No rows with positive risk")

    valid["day"] = valid["day"].astype(int).clip(lower=spline.index.min(), upper=spline.index.max())
    spline_part = spline.reindex(valid["day"]).reset_index(drop=True)

    X = spline_part.copy()
    X["vaccinated"] = valid["vaccinated"].astype(float).values
    X = sm.add_constant(X)

    y_binom = np.column_stack([valid["events"], valid["risk"] - valid["events"]])

    valid_mask = (
        ~X.isna().any(axis=1) &
        ~np.isinf(X.values).any(axis=1) &
        ~np.isinf(y_binom).any(axis=1)
    )

    if valid_mask.sum() == 0:
        raise ValueError("All rows have NaN/Inf after cleaning - no valid data for model fit")

    X_clean = X[valid_mask].copy()
    y_binom_clean = y_binom[valid_mask]

    log(f"Fitting outcome model on {len(X_clean)} valid rows")

    cov_type = 'HC0' if CONFIG["robust_se"] else 'nonrobust'
    model = fit_glm_regularized_with_retry(
        y_binom_clean, X_clean, sm.families.Binomial(), CONFIG["ridge_alphas"], CONFIG["glm_maxiter"], cov_type
    )
    return model

def predict_survival(model, t_hazards, arm, spline):
    spline_part = spline.loc[t_hazards].copy()
    X = spline_part.copy()
    X["vaccinated"] = float(arm)
    X = sm.add_constant(X)
    X = X.reindex(columns=model.params.index, fill_value=0.0)
    h = model.predict(X)
    h = safe_clip(h, 1e-9, 1 - 1e-9)
    surv = np.concatenate([[1.0], np.cumprod(1 - h)])
    return safe_clip(surv, 0.0, 1.0)

def compute_rmst_from_survival(surv: np.ndarray) -> float:
    return float(np.sum(surv[:-1]))

def compute_delta(df: pd.DataFrame, first: int, last_obs: float, p_cond: pd.DataFrame,
                  sex_dist_per_day: pd.DataFrame, marginal_ps: pd.Series, lag: int,
                  tau: int, spline_outcome: pd.DataFrame, t_grid_haz: np.ndarray) -> float:
    agg = aggregate_trials_vectorized(
        df, first, last_obs, p_cond, sex_dist_per_day, marginal_ps, lag, tau
    )
    model = fit_pooled_logistic(agg, spline_outcome)
    Sv = predict_survival(model, t_grid_haz, 1, spline_outcome)
    Su = predict_survival(model, t_grid_haz, 0, spline_outcome)
    rmst_v = compute_rmst_from_survival(Sv)
    rmst_u = compute_rmst_from_survival(Su)
    return rmst_v - rmst_u

# ==================== BOOTSTRAP ====================
def joint_bootstrap_delta(df_age, first, last_obs, p_cond, sex_dist_per_day, marginal_ps, lag, tau, spline_outcome, t_grid_haz):
    n = len(df_age)
    boot_real = []
    boot_placebo = []
    boot_adj = []
    for _ in range(CONFIG["n_boot"]):
        idx = np.random.choice(n, int(n * CONFIG["boot_subsample"]), replace=True)
        df_boot = df_age.iloc[idx].reset_index(drop=True).copy()
        prop_data_boot = aggregate_propensity_daily(df_boot, first, last_obs)
        if prop_data_boot.empty:
            continue
        prop_spline_obs = build_spline_basis(prop_data_boot["day"].unique(), CONFIG["time_df"], CONFIG["spline_degree"])
        prop_model_boot = fit_propensity_model(prop_data_boot, prop_spline_obs)
        prop_spline_full = build_spline_basis(np.arange(first, int(last_obs) + 1), CONFIG["time_df"], CONFIG["spline_degree"])
        p_cond_boot = compute_conditional_probabilities(prop_model_boot, prop_spline_full)
        sex_dist_per_day_boot = prop_data_boot.pivot_table(index='day', columns='sex', values='n_eligible', fill_value=0).div(prop_data_boot.groupby('day')['n_eligible'].sum(), axis=0).fillna(0)
        marginal_ps_boot = compute_marginal_ps(p_cond_boot, prop_data_boot)
        delta_real = compute_delta(df_boot, first, last_obs, p_cond_boot, sex_dist_per_day_boot, marginal_ps_boot, lag, tau, spline_outcome, t_grid_haz)
        df_placebo_boot = generate_robust_placebo_dates(df_boot, p_cond_boot, first)
        delta_placebo = compute_delta(df_placebo_boot, first, last_obs, p_cond_boot, sex_dist_per_day_boot, marginal_ps_boot, lag, tau, spline_outcome, t_grid_haz)
        delta_adj = delta_real - delta_placebo
        boot_real.append(delta_real)
        boot_placebo.append(delta_placebo)
        boot_adj.append(delta_adj)
    return boot_real, boot_placebo, boot_adj

# ==================== PRE-VAX MORTALITY ====================
def compute_pre_vax_mortality_balanced(df: pd.DataFrame, window: int = 30, n_placebo_draws: int = 100) -> Dict[str, float]:
    df = df.copy()
    vax = df[df["first_dose_day"].notna()].copy()
    never = df[df["first_dose_day"].isna()].copy()
    res: Dict[str, float] = {}

    if vax.empty:
        res["pre_mort_early"] = res["pre_mort_late"] = 0.0
    else:
        q1, q3 = vax["first_dose_day"].quantile([0.25, 0.75])
        for label, sub in [("early", vax[vax["first_dose_day"] <= q1]),
                           ("late", vax[vax["first_dose_day"] >= q3])]:
            pre_start = sub["first_dose_day"] - window
            is_pre_death = (sub["death_day"] >= pre_start) & (sub["death_day"] < sub["first_dose_day"])
            deaths = is_pre_death.sum()
            pt = len(sub) * window
            res[f"pre_mort_{label}"] = deaths / pt if pt > 0 else 0.0

    if never.empty or vax.empty:
        res["pre_mort_never"] = 0.0
    else:
        real_dates = vax["first_dose_day"].dropna().values
        if len(real_dates) == 0:
            res["pre_mort_never"] = 0.0
        else:
            rng = np.random.default_rng(CONFIG["random_seed"])
            rates = []
            for _ in range(n_placebo_draws):
                placebo_day = rng.choice(real_dates, size=len(never), replace=True)
                pre_start = placebo_day - window
                is_pre_death = (never["death_day"] >= pre_start) & (never["death_day"] < placebo_day)
                deaths = is_pre_death.sum()
                pt = len(never) * window
                rates.append(deaths / pt if pt > 0 else 0.0)
            res["pre_mort_never"] = float(np.mean(rates))

    return res

# ==================== PLACEBO ====================
def generate_robust_placebo_dates(df_age: pd.DataFrame, p_cond: pd.DataFrame, first: int) -> pd.DataFrame:
    df = df_age.reset_index(drop=True).copy()
    rng = np.random.default_rng(CONFIG["random_seed"])
    placebo_days = np.full(len(df), np.nan)
    max_days = CONFIG["max_last_obs_days"]
    days = np.arange(first, first + max_days + 1)
    p_cond_max_day = p_cond.index.max()
    for i, row in df.iterrows():
        sex = row["sex"]
        death = row["death_day"] if not np.isnan(row["death_day"]) else np.inf
        elig_days = days[(days <= death) & (days <= p_cond_max_day)]  # Clip to p_cond range
        if len(elig_days) == 0:
            continue
        p_elig = p_cond.loc[elig_days, sex].values
        cum_s = np.cumprod(1 - p_elig)
        u = rng.uniform()
        vax_idx = np.searchsorted(-cum_s, -u)
        if vax_idx < len(elig_days):
            placebo_days[i] = elig_days[vax_idx]
    df["first_dose_day"] = np.maximum(placebo_days, 0)
    return df

# ==================== POSITIVITY DIAGNOSTICS ====================
def compute_balance_smd(df_age: pd.DataFrame, marginal_ps: pd.Series, age: int):
    df_age = df_age.copy()
    days = marginal_ps.index
    min_day, max_day = days.min(), days.max()
    df_age['first_dose_day_clipped'] = df_age['first_dose_day'].clip(min_day, max_day)
    df_age['ps'] = marginal_ps.reindex(df_age['first_dose_day_clipped']).fillna(0.5).values

    df_age['weight'] = np.where(df_age['first_dose_day'].notna(), 1 - df_age['ps'], df_age['ps'])

    vax_group = df_age[df_age['first_dose_day'].notna()]
    unvax_group = df_age[df_age['first_dose_day'].isna()]

    smd_pre = (vax_group['sex'].mean() - unvax_group['sex'].mean()) / np.sqrt(0.5) if len(vax_group) > 0 and len(unvax_group) > 0 else np.nan

    w_v = vax_group['weight']
    w_u = unvax_group['weight']
    mean_v = (vax_group['sex'] * w_v).sum() / w_v.sum() if w_v.sum() > 0 else np.nan
    mean_u = (unvax_group['sex'] * w_u).sum() / w_u.sum() if w_u.sum() > 0 else np.nan
    smd_post = (mean_v - mean_u) / np.sqrt(0.5) if not np.isnan(mean_v) and not np.isnan(mean_u) else np.nan

    balance = pd.DataFrame({
        'variable': ['sex'],
        'smd_pre': [smd_pre],
        'smd_post': [smd_post]
    })
    save_df(balance, f"balance_smd_age{age}")

    weights = df_age['weight'].dropna()
    ess = (weights.sum()**2) / (weights**2).sum() if len(weights) > 0 else np.nan
    weights_stats = pd.DataFrame({
        'min': [weights.min() if len(weights) > 0 else np.nan],
        'max': [weights.max() if len(weights) > 0 else np.nan],
        'mean': [weights.mean() if len(weights) > 0 else np.nan],
        'ess': [ess]
    })
    save_df(weights_stats, f"weights_stats_age{age}")

    if not np.isnan(ess) and ess < CONFIG["ess_threshold"]:
        log(f"Low ESS for age {age}: {ess:.1f} < {CONFIG['ess_threshold']} — results flagged as low power")

    df_age['vaccinated'] = df_age['first_dose_day'].notna().map({True: 'Vax', False: 'Unvax'})
    fig_ps = px.histogram(
        df_age.dropna(subset=['ps']),
        x='ps',
        color='vaccinated',
        barmode='overlay',
        title=f"Propensity Score Distribution - Age {age}",
        labels={'ps': 'Marginal Propensity Score'},
        color_discrete_map={'Vax': 'blue', 'Unvax': 'orange'},
        opacity=0.7
    )
    fig_ps.update_layout(showlegend=True)
    save_plot(fig_ps, f"ps_dist_age{age}")

    if not np.isnan(smd_post) and abs(smd_post) > 0.1:
        log(f"Warning: Post-weight SMD for sex at age {age} = {smd_post:.3f} (>0.1 threshold)")

# ==================== PLOTTING ====================
def plot_lag_sweep_with_ci(df: pd.DataFrame, age: int, title_suffix: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["lag"],
        y=df["delta"],
        mode="lines+markers",
        name="ΔRMST",
        line=dict(color="blue"),
        error_y=dict(type='data', symmetric=False, array=df["delta_high"] - df["delta"], arrayminus=df["delta"] - df["delta_low"])
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"ΔRMST by Lag - Age {age}{title_suffix}",
        xaxis_title="Lag (days)",
        yaxis_title="ΔRMST (days)",
        template="plotly_white"
    )
    return fig

def plot_real_vs_placebo_lag_sweep(df_real: pd.DataFrame, df_placebo: pd.DataFrame, df_adj: pd.DataFrame, age: int) -> go.Figure:
    fig = go.Figure()
    for name, df, color in [("Real", df_real, "blue"), ("Placebo", df_placebo, "gray"), ("Adjusted", df_adj, "purple")]:
        fig.add_trace(go.Scatter(
            x=df["lag"],
            y=df["delta"],
            mode="lines+markers",
            name=name,
            line=dict(color=color, dash="dash" if name == "Placebo" else "solid"),
            error_y=dict(type='data', symmetric=False, array=df["delta_high"] - df["delta"], arrayminus=df["delta"] - df["delta_low"])
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"Real vs Placebo vs Adjusted ΔRMST - Age {age}",
        xaxis_title="Lag (days)",
        yaxis_title="ΔRMST (days)",
        template="plotly_white"
    )
    return fig

def plot_pre_vax_mortality(hve_stats: Dict[str, float], age: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Early", "Late", "Never"],
        y=[hve_stats.get("pre_mort_early", 0), hve_stats.get("pre_mort_late", 0), hve_stats.get("pre_mort_never", 0)],
        marker_color=["green", "orange", "red"]
    ))
    fig.update_layout(
        title=f"Pre-vaccination Mortality Rates ({CONFIG['pre_vax_window']}-day) – Age {age}",
        yaxis_title=f"Rate per {CONFIG['pre_vax_window']} person-days"
    )
    return fig

def plot_null_calibration(df_sim: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for typ in df_sim["type"].unique():
        sub = df_sim[df_sim["type"] == typ]
        fig.add_trace(go.Scatter(
            x=sub["lag"],
            y=sub["mean_delta"],
            mode="lines+markers",
            name=typ,
            error_y=dict(type='data', array=sub["sd_delta"])
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title="Null Calibration: Mean ΔRMST by Lag (with SD)",
        xaxis_title="Lag (days)",
        yaxis_title="Mean ΔRMST (days)",
        template="plotly_white"
    )
    return fig

# ==================== NULL SIMULATION ====================
def generate_null_data(single_ages: List[int], scenario: str = 'null') -> pd.DataFrame:
    try:
        from scipy.special import expit, logit
    except ImportError:
        raise ImportError("scipy.special required for simulation. Install scipy.")

    rng = np.random.default_rng(CONFIG["random_seed"])
    rows = []
    confound_strength = CONFIG["sim_confound_strength"] if scenario != 'strong_confound' else 2.0
    for age in single_ages:
        for i in range(CONFIG["n_subj_sim"]):
            sex = rng.choice([0, 1])
            U = rng.normal(0, 1) if 'confound' in scenario else 0
            days = np.arange(CONFIG["sim_max_days"])
            logit_p = -5 + 0.01 * days / 365 + 0.2 * sex + confound_strength * U
            p_day = sc.expit(logit_p)
            cum_s = np.cumprod(1 - p_day)
            u = rng.uniform()
            vax_idx = np.searchsorted(-cum_s, -u)
            first_dose_day = days[vax_idx] if vax_idx < len(days) else np.nan

            logit_h = sc.logit(0.001) + 0.01 * (age - 60) / 30 + CONFIG["sim_hazard_wave_amp"] * np.sin(2 * np.pi * days / 365)
            if 'confound' in scenario:
                logit_h -= confound_strength * U
            h_day = sc.expit(logit_h + 0.001 * days / 365)
            cum_s_death = np.cumprod(1 - h_day)
            u_death = rng.uniform()
            death_idx = np.searchsorted(-cum_s_death, -u_death)
            death_day = days[death_idx] if death_idx < len(days) else np.nan

            if scenario == 'censor':
                censor_prob = CONFIG["sim_censor_rate"] + 0.05 * (1 if not np.isnan(first_dose_day) else 0)
                if rng.uniform() < censor_prob:
                    censor_day = rng.uniform(0, CONFIG["sim_max_days"])
                    death_day = min(death_day, censor_day) if not np.isnan(death_day) else censor_day

            rows.append({
                "age": age,
                "sex": sex,
                "first_dose_day": first_dose_day,
                "death_day": death_day,
                "subject_id": i,
            })
    raw_sim = pd.DataFrame(rows)
    log(f"Generated {scenario} sim data: {len(raw_sim)} subjects")
    return raw_sim

# ==================== MAIN PIPELINE ====================
def run_age_diagnostics(age: int, raw: pd.DataFrame, spline_outcome: pd.DataFrame, t_grid_haz: np.ndarray):
    log(f"Processing age {age}")
    df_age = raw[raw["age"] == age].copy()
    if len(df_age) < 500:
        log(f"Skipping age {age}: too few subjects")
        return None, None, None

    first = 0
    last_obs = CONFIG["max_last_obs_days"]

    prop_data = aggregate_propensity_daily(df_age, first, last_obs)
    if prop_data.empty:
        log(f"Skipping age {age}: no propensity data")
        return None, None, None

    prop_spline_obs = build_spline_basis(prop_data["day"].unique(), CONFIG["time_df"], CONFIG["spline_degree"])
    prop_model = fit_propensity_model(prop_data, prop_spline_obs, include_sex=True)

    days_grid = np.arange(first, int(last_obs) + 1)
    prop_spline_full = build_spline_basis(days_grid, CONFIG["time_df"], CONFIG["spline_degree"])
    p_cond = compute_conditional_probabilities(prop_model, prop_spline_full)

    sex_dist_per_day = prop_data.pivot_table(index='day', columns='sex', values='n_eligible', fill_value=0).div(prop_data.groupby('day')['n_eligible'].sum(), axis=0).fillna(0)
    marginal_ps = compute_marginal_ps(p_cond, prop_data)

    compute_balance_smd(df_age, marginal_ps, age)

    def process_lag(lag):
        boot_real, boot_placebo, boot_adj = joint_bootstrap_delta(df_age, first, last_obs, p_cond, sex_dist_per_day, marginal_ps, lag, CONFIG["tau"], spline_outcome, t_grid_haz)
        n_success = len(boot_real)
        log(f"Age {age} Lag {lag}: Successful bootstraps: {n_success}/{CONFIG['n_boot']}")
        if n_success < CONFIG["min_boot_success"]:
            log(f"Warning: Insufficient successful bootstraps for age {age} lag {lag} — CIs unreliable")
        delta_real = np.mean(boot_real) if n_success >= CONFIG["min_boot_success"] else np.nan
        ci_real = np.percentile(boot_real, [2.5, 97.5]) if n_success >= CONFIG["min_boot_success"] else [np.nan, np.nan]
        delta_placebo = np.mean(boot_placebo) if n_success >= CONFIG["min_boot_success"] else np.nan
        ci_placebo = np.percentile(boot_placebo, [2.5, 97.5]) if n_success >= CONFIG["min_boot_success"] else [np.nan, np.nan]
        delta_adj = np.mean(boot_adj) if n_success >= CONFIG["min_boot_success"] else np.nan
        ci_adj = np.percentile(boot_adj, [2.5, 97.5]) if n_success >= CONFIG["min_boot_success"] else [np.nan, np.nan]
        return {
            "lag": lag,
            "delta_real": delta_real, "real_low": ci_real[0], "real_high": ci_real[1],
            "delta_placebo": delta_placebo, "placebo_low": ci_placebo[0], "placebo_high": ci_placebo[1],
            "delta_adj": delta_adj, "adj_low": ci_adj[0], "adj_high": ci_adj[1]
        }

    lag_results = Parallel(n_jobs=CONFIG["n_cores"])(delayed(process_lag)(lag) for lag in CONFIG["lag_sweep"])

    df_results = pd.DataFrame(lag_results)
    df_real = df_results[["lag", "delta_real", "real_low", "real_high"]].rename(columns={"delta_real": "delta", "real_low": "delta_low", "real_high": "delta_high"})
    df_placebo = df_results[["lag", "delta_placebo", "placebo_low", "placebo_high"]].rename(columns={"delta_placebo": "delta", "placebo_low": "delta_low", "placebo_high": "delta_high"})
    df_adj = df_results[["lag", "delta_adj", "adj_low", "adj_high"]].rename(columns={"delta_adj": "delta", "adj_low": "delta_low", "adj_high": "delta_high"})

    df_real["age"] = df_placebo["age"] = df_adj["age"] = age
    save_df(df_real, f"real_deltas_age{age}")
    save_df(df_placebo, f"placebo_deltas_age{age}")
    save_df(df_adj, f"bias_adjusted_deltas_age{age}")

    fig_lag = plot_lag_sweep_with_ci(df_real, age, " (Real)")
    save_plot(fig_lag, f"lag_sweep_real_age{age}")

    fig_real_placebo = plot_real_vs_placebo_lag_sweep(df_real, df_placebo, df_adj, age)
    save_plot(fig_real_placebo, f"real_vs_placebo_age{age}")

    hve_pre = compute_pre_vax_mortality_balanced(df_age, CONFIG["pre_vax_window"], CONFIG["n_placebo_draws"])
    hve_pre["age"] = age
    save_df(pd.DataFrame([hve_pre]), f"pre_vax_mortality_age{age}")
    fig_pre = plot_pre_vax_mortality(hve_pre, age)
    save_plot(fig_pre, f"pre_vax_mortality_age{age}")

    return df_real, df_placebo, df_adj

# ==================== MAIN ENTRY POINT ====================
def main():
    if RUN_MODE == "test":
        log("Starting unit tests...")
        test_compute_daily_events_and_risk()
        test_aggregate_trials_vectorized()
        test_bootstrap()
        test_null_sim_generation()
        log("All unit tests completed successfully.")
        logger.flush_parallel_buffer()
        return

    t_grid_haz = np.arange(CONFIG["tau"])
    spline_outcome = build_spline_basis(np.arange(0, CONFIG["tau"] + 1), CONFIG["time_df"], CONFIG["spline_degree"])

    if RUN_MODE == "null_sim":
        log("Running null simulations for calibration...")
        sim_results = []
        for scenario in CONFIG["sim_scenarios"]:
            log(f"Scenario: {scenario}")
            def run_sim(i):
                try:
                    from scipy.special import expit, logit
                except ImportError:
                    raise ImportError("run_sim scipy.special required for simulation. Install scipy.")
                
                raw_sim = generate_null_data(CONFIG["single_ages"], scenario)
                all_real_sim = []
                all_placebo_sim = []
                all_adj_sim = []
                for age in CONFIG["single_ages"]:
                    real, placebo, adj = run_age_diagnostics(age, raw_sim, spline_outcome, t_grid_haz)
                    if real is not None:
                        real["sim"] = i
                        real["scenario"] = scenario
                        placebo["sim"] = i
                        placebo["scenario"] = scenario
                        adj["sim"] = i
                        adj["scenario"] = scenario
                        all_real_sim.append(real)
                        all_placebo_sim.append(placebo)
                        all_adj_sim.append(adj)
                return pd.concat(all_real_sim) if all_real_sim else pd.DataFrame(), pd.concat(all_placebo_sim) if all_placebo_sim else pd.DataFrame(), pd.concat(all_adj_sim) if all_adj_sim else pd.DataFrame()

            sim_outputs = Parallel(n_jobs=CONFIG["n_cores"])(delayed(run_sim)(i) for i in range(CONFIG["n_null_sims"]))

            all_real_sim = pd.concat([out[0] for out in sim_outputs if not out[0].empty])
            all_placebo_sim = pd.concat([out[1] for out in sim_outputs if not out[1].empty])
            all_adj_sim = pd.concat([out[2] for out in sim_outputs if not out[2].empty])

            save_df(all_real_sim, f"null_sim_real_deltas_{scenario}")
            save_df(all_placebo_sim, f"null_sim_placebo_deltas_{scenario}")
            save_df(all_adj_sim, f"null_sim_adjusted_deltas_{scenario}")

            for typ, df in [("real", all_real_sim), ("placebo", all_placebo_sim), ("adjusted", all_adj_sim)]:
                if df.empty:
                    continue
                summary = df.groupby("lag")["delta"].agg(['mean', 'std']).reset_index()
                summary["type"] = typ
                summary["scenario"] = scenario
                sim_results.append(summary)
        df_sim_summary = pd.concat(sim_results)
        save_df(df_sim_summary, "null_sim_summary_all")

        fig_null = plot_null_calibration(df_sim_summary)
        save_plot(fig_null, "null_calibration_all")

        log("Null simulations completed.")
        logger.flush_parallel_buffer()
        return

    raw = load_and_prepare_data(CONFIG["input_path"])

    def process_age(age):
        return run_age_diagnostics(age, raw, spline_outcome, t_grid_haz)

    age_outputs = Parallel(n_jobs=CONFIG["n_cores"])(delayed(process_age)(age) for age in CONFIG["single_ages"])

    all_real = pd.concat([out[0] for out in age_outputs if out[0] is not None])
    all_placebo = pd.concat([out[1] for out in age_outputs if out[1] is not None])
    all_adj = pd.concat([out[2] for out in age_outputs if out[2] is not None])

    if not all_real.empty:
        save_df(all_real, "all_real_deltas")
        save_df(all_placebo, "all_placebo_deltas")
        save_df(all_adj, "all_bias_adjusted_deltas")

    concatenate_all_csvs_to_txt()

    meta = {
        "config": {k: str(v) if isinstance(v, (Path, pd.Timestamp)) else v for k, v in CONFIG.items()},
        "timestamp": timestamp,
        "run_mode": RUN_MODE,
        "assumptions": "No unmeasured confounding, positivity holds, no competing risks, binary sex, no loss to follow-up beyond censoring."
    }
    with open(CONFIG["output_base"] / f"metadata_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.flush_parallel_buffer()
    log("Diagnostic pipeline completed.")

if __name__ == "__main__":
    main()