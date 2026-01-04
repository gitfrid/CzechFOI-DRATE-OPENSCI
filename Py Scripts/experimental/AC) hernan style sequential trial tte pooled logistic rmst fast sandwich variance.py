#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hernán-style sequential target-trial emulation (Design B) - Sandwich Variance Version
Implements calendar-time sequential trials with grace period, pooled logistic regression,
RMST estimation, empirical sandwich variance + delta method, and IPW for artificial censoring.
Exploratory analysis — no confounder (HVE) adjustment by design.

Scientific notes (for methods/documentation):
- Estimand: Pooled effect of initiating vaccination at calendar day t vs not initiating,
  averaged across eligible t, with 14-day grace period in treatment definition.
  IPW corrects artificial censoring from cloning.
- Confounding: Only time trends (B-splines) + censoring; NO adjustment for healthy-vaccinee
  bias, indication, or other confounders.
- Grace period: Deaths during grace attributed to unvaccinated (A=0).
- IPW: Marginal (empirical) p_init; positivity monitored via weight diagnostics.
- Variance: Empirical sandwich (HC0) for GLM params, delta method for ΔRMST CI.

Author: AI / Drifting assistence   Date: January 2026 Version 1.1
"""

from __future__ import annotations
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.api import BSplines
from scipy.special import expit
from scipy.integrate import simpson
from tqdm.auto import tqdm
import logging
import plotly.graph_objects as go
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ===================== TOP-LEVEL PARAMETERS =====================

AGE = 70
DATA_SET = "real"

DATA_CONFIG = {
    "real": {
        "input": r"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{age}.csv",
        "suffix": ""
    },
    "sim": {
        "input": r"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{age}.csv",
        "suffix": "_SIM"
    },
    "reclassified": {
        "input": r"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_DeathOrAlive_reclassified_PCT5_uvx_as_vx_AG{age}.csv",
        "suffix": "_RECLASSIFIED"
    }
}

selected = DATA_CONFIG[DATA_SET]
INPUT = Path(selected["input"].format(age=AGE))
OUTPUT_BASE = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) hernan style sequential trial poold logistics RMST") / f"AC) hernan style sequential trial poold logistics RMST Fast {selected['suffix']}"

CONFIG = {
    "age_ref_year": 2023,
    "study_start": pd.Timestamp("2020-01-01"),
    "input_path": INPUT,
    "output_base": OUTPUT_BASE,
    "random_seed": 12345,
    "safety_buffer": 30,
    "time_df": 4,
    "spline_degree": 3,
    "tau": 90,
    "grace_period": 0,  # days
    "ipw_max": 1e6,  # For capping
    "ipw_trim_quantile": 0.99,  # For trimming (e.g., truncate at 99th percentile)
    "delta_eps_scale": 1e-6,  # For delta method
}

CONFIG["output_base"].parent.mkdir(parents=True, exist_ok=True)
MAIN_LOG = CONFIG["output_base"].parent / f"{CONFIG['output_base'].name}_AG{AGE}.txt"

np.random.seed(CONFIG["random_seed"])

# ===================== LOGGING =====================

def log(msg: str, timestamp: bool = True):
    ts = datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds") if timestamp else ""
    line = f"{ts}  {msg}" if ts else msg
    print(line)
    with open(MAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ===================== NUMERIC HELPERS =====================

def compute_rmst(t: np.ndarray, S: np.ndarray) -> float:
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    return float(simpson(S, x=t)) if len(t) > 1 else 0.0

def rmst_curve(S: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.array([compute_rmst(t[:i+1], S[:i+1]) for i in range(len(t))])

def compute_daily_events(df: pd.DataFrame, start_day: int, end_day: float) -> tuple[np.ndarray, np.ndarray]:
    window = int(end_day - start_day)
    events = np.zeros(window, dtype=float)
    diff = np.zeros(window + 1, dtype=int)

    if df.empty:
        return events, np.zeros_like(events)

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

# ===================== SPLINE BASIS =====================

def build_spline_basis(t: np.ndarray, df_deg: int = 4, degree: int = 3) -> pd.DataFrame:
    days = np.asarray(t, dtype=float)
    days_centered = days - days.mean()
    days_std = days_centered / (days_centered.std(ddof=0) + 1e-10)
    df_basis = df_deg + degree + 1
    bs = BSplines(
        x=days_std[:, None],
        df=[df_basis],
        degree=[degree],
        variable_names=["time"]
    )
    basis_matrix = bs.basis.astype("float32")
    columns = [f"time_s{i}" for i in range(basis_matrix.shape[1])]
    spline_df = pd.DataFrame(basis_matrix, columns=columns, index=days)
    spline_df["time_lin"] = days_std
    return spline_df

# ===================== DATA LOADING & PREPARATION =====================

def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    log(f"Loading CSV: {input_path}")
    raw = pd.read_csv(input_path, dtype=str)
    raw.columns = raw.columns.str.strip()

    for c in ["DatumUmrti"] + [c for c in raw.columns if c.startswith("Datum_")]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")

    raw["Rok_narozeni"] = pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw["age"] = CONFIG["age_ref_year"] - raw["Rok_narozeni"]
    raw = raw[raw["age"] == AGE].copy()
    raw.reset_index(drop=True, inplace=True)
    raw["subject_id"] = np.arange(len(raw))

    raw["death_day"] = (raw["DatumUmrti"] - CONFIG["study_start"]).dt.days
    raw["first_dose_day"] = (raw["Datum_1"] - CONFIG["study_start"]).dt.days

    raw["death_day"] = pd.to_numeric(raw["death_day"], errors="coerce").astype(float)
    raw["first_dose_day"] = pd.to_numeric(raw["first_dose_day"], errors="coerce").astype(float)

    log(f"Subjects after age filter: {len(raw)}")
    log(f"Deaths: {raw['death_day'].notna().sum()}, Vaccinated: {raw['first_dose_day'].notna().sum()}")
    return raw

# ===================== AGGREGATION =====================

def aggregate_daily_updated(raw: pd.DataFrame, first: int, last_obs: float, p_init: np.ndarray, grace: int, t_grid: np.ndarray) -> pd.DataFrame:
    max_window = len(t_grid)
    events_u_total = np.zeros(max_window, dtype=float)
    risk_u_total = np.zeros(max_window, dtype=float)
    events_v_total = np.zeros(max_window, dtype=float)
    risk_v_total = np.zeros(max_window, dtype=float)

    last_day_int = int(last_obs)

    weights_used = []

    death_np = raw['death_day'].to_numpy()
    first_np = raw['first_dose_day'].to_numpy()
    id_np = raw['subject_id'].to_numpy()

    for start_day in tqdm(range(first, last_day_int), desc="Aggregating trials"):
        max_k = min(max_window, last_day_int - start_day)
        if max_k <= 0:
            continue

        end_for_compute = start_day + max_k

        eligible_mask = (
            (np.isnan(death_np) | (death_np >= start_day)) &
            (np.isnan(first_np) | (first_np >= start_day))
        )
        eligible = raw[eligible_mask]

        initiators_mask = (eligible['first_dose_day'] == start_day)
        initiators = eligible[initiators_mask]
        non_initiators = eligible[~initiators_mask]

        # A=0: Non-initiators
        if not non_initiators.empty:
            df_non = non_initiators.copy()
            df_non['start'] = float(start_day)
            df_non['stop'] = np.minimum(
                df_non['first_dose_day'].fillna(last_obs),
                df_non['death_day'].fillna(last_obs)
            ).clip(upper=last_obs)
            df_non['event'] = (
                (df_non['death_day'] >= df_non['start']) &
                (df_non['death_day'] <= df_non['stop'])
            ).astype(int)
            events_non, risk_non = compute_daily_events(df_non, start_day, end_for_compute)
        else:
            events_non = risk_non = np.zeros(max_k, dtype=float)

        # A=0: Grace period from initiators
        events_grace = risk_grace = np.zeros(max_k, dtype=float)
        if grace > 0 and not initiators.empty:
            df_grace = initiators.copy()
            df_grace['start'] = float(start_day)
            grace_end = start_day + grace
            death_or_end = df_grace['death_day'].fillna(last_obs)
            df_grace['stop'] = np.minimum(grace_end, death_or_end).clip(upper=last_obs)
            df_grace['event'] = (
                (df_grace['death_day'] >= df_grace['start']) &
                (df_grace['death_day'] <= df_grace['stop'])
            ).astype(int)
            events_grace, risk_grace = compute_daily_events(df_grace, start_day, end_for_compute)

        # A=1: Vaccinated after grace
        events_v_trial = risk_v_trial = np.zeros(max_k, dtype=float)
        vax_start = start_day + grace
        if not initiators.empty and vax_start <= last_day_int:
            vax_mask = (initiators['death_day'].isna() | (initiators['death_day'] >= vax_start))
            df_vax = initiators[vax_mask].copy()
            if not df_vax.empty:
                df_vax['start'] = float(vax_start)
                df_vax['stop'] = df_vax['death_day'].fillna(last_obs).clip(upper=last_obs)
                df_vax['event'] = (
                    (df_vax['death_day'] >= df_vax['start']) &
                    (df_vax['death_day'] <= df_vax['stop'])
                ).astype(int)
                events_v_trial, risk_v_trial = compute_daily_events(df_vax, start_day, end_for_compute)

        # IPW with trimming
        idx0 = start_day - first
        ipws = []
        for k in range(max_k):
            if k == 0:
                ipw = 1.0
            else:
                s_idx = max(0, idx0)
                e_idx = min(len(p_init), idx0 + k)
                if s_idx >= e_idx:
                    ipw = 1.0
                else:
                    probs = 1.0 - p_init[s_idx:e_idx]
                    log_probs = np.sum(np.log(np.clip(probs, 1e-12, 1.0)))
                    surv = float(np.exp(log_probs))
                    ipw = 1.0 / surv if surv > 1e-12 else 0.0
            ipws.append(ipw)

        ipws_arr = np.array(ipws)
        if np.any(ipws_arr > 1.0):
            trim_threshold = np.quantile(ipws_arr[ipws_arr > 1.0], CONFIG["ipw_trim_quantile"])
            ipws_arr = np.clip(ipws_arr, None, min(trim_threshold, CONFIG["ipw_max"]))

        for k in range(max_k):
            ipw = ipws_arr[k]
            if ipw > 1.0:
                weights_used.append(ipw)

            events_u_total[k] += events_non[k] * ipw + events_grace[k]
            risk_u_total[k] += risk_non[k] * ipw + risk_grace[k]
            events_v_total[k] += events_v_trial[k]
            risk_v_total[k] += risk_v_trial[k]

    agg = pd.DataFrame({
        "day": np.concatenate([t_grid, t_grid]),
        "vaccinated": np.concatenate([np.ones_like(t_grid), np.zeros_like(t_grid)]),
        "events": np.concatenate([events_v_total, events_u_total]),
        "risk": np.concatenate([risk_v_total, risk_u_total]),
    })

    if weights_used:
        w = np.array(weights_used)
        log(f"IPW diagnostics: mean={w.mean():.2f}, median={np.median(w):.2f}, max={w.max():.1e}")

    return agg

# ===================== MODELING =====================

def fit_pooled_logistic(agg: pd.DataFrame, spline: pd.DataFrame, log_details: bool = True) -> sm.GLM:
    df = agg[agg.risk > 0].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No positive-risk days")

    p = (df.events / df.risk).clip(1e-9, 1 - 1e-9).astype("float32")
    w = df.risk.astype("float32")
    df["vaccinated"] = df["vaccinated"].astype("float32")

    S = spline.loc[df.day].reset_index(drop=True)
    X = S.copy()
    X["vaccinated"] = df["vaccinated"]
    inter = S.mul(df["vaccinated"].to_numpy(dtype=np.float32)[:, None])
    inter.columns = [f"{c}_x_vacc" for c in inter.columns]
    X = pd.concat([X, inter], axis=1)
    X.insert(0, "const", 1.0)
    X = X.astype("float32")

    if log_details:
        log(f"GLM rows: {len(X)}, predictors: {X.shape[1]}")

    model = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w).fit()

        # Apply robust sandwich covariance (fixed for newer statsmodels)
        if model is not None:
            try:
                model = model._get_robustcov_results(cov_type='HC0')
            except AttributeError:
                # Fallback for older versions
                try:
                    model = model.get_robustcov_results(cov_type='HC0')
                except Exception as e:
                    log(f"Warning: Could not apply robust covariance: {str(e)}")
                    # Continue with non-robust model

    except Exception as e:
        log(f"Critical error in GLM fitting: {str(e)}")
        raise RuntimeError("GLM fitting failed completely. Check data for NaN/Inf values or zero variance.")

    if model is None:
        raise RuntimeError("No valid GLM model was fitted. All attempts failed.")

    if log_details:
        log(f"GLM converged: {model.converged}")

    return model

def predict_survival(model: sm.GLM, t: np.ndarray, A: int, spline: pd.DataFrame) -> np.ndarray:
    S = spline.loc[t].reset_index(drop=True)
    X = S.copy()
    X["vaccinated"] = float(A)
    inter = S.mul(float(A))
    inter.columns = [f"{c}_x_vacc" for c in inter.columns]
    X = pd.concat([X, inter], axis=1)
    X.insert(0, "const", 1.0)
    X = X.reindex(columns=model.params.index, fill_value=0.0).astype("float32")

    hazard = expit(X.to_numpy(dtype=np.float32) @ model.params.to_numpy(dtype=np.float32))
    hazard = np.clip(hazard, 1e-9, 1 - 1e-9)
    return np.cumprod(1 - hazard)

# ===================== RESULTS & PLOTS =====================

def compute_estimands(Sv: np.ndarray, Su: np.ndarray, t: np.ndarray) -> dict:
    rmst_v = rmst_curve(Sv, t)
    rmst_u = rmst_curve(Su, t)
    delta = rmst_v - rmst_u
    tau = t[-1]
    return {
        "tau": tau,
        "rmst_v_tau": rmst_v[-1],
        "rmst_u_tau": rmst_u[-1],
        "delta_tau": delta[-1],
        "ve_tau": 1 - (1 - Sv[-1]) / (1 - Su[-1]) if Su[-1] != 1 else np.nan,
        "rmst_v": rmst_v,
        "rmst_u": rmst_u,
        "delta": delta,
        "sv": Sv,
        "su": Su,
    }

def plot_and_save(estimands: dict, asymp_results: dict, output_base: Path):
    t = np.arange(len(estimands["delta"]))
    delta = estimands["delta"]
    rmst_v = estimands["rmst_v"]
    rmst_u = estimands["rmst_u"]
    sv = estimands["sv"]
    su = estimands["su"]
    tau = estimands["tau"]
    delta_tau = estimands["delta_tau"]
    delta_lo = asymp_results["ci_low"]
    delta_hi = asymp_results["ci_high"]

    # Plot 1: ΔRMST(t) - point curve, CI only at tau
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=t, y=delta, mode="lines", line=dict(color="black", width=2), name="ΔRMST(t)"))
    fig_delta.add_hline(y=0, line=dict(color="gray", dash="dash"))
    fig_delta.add_annotation(x=tau, y=delta_tau,
                            text=f"ΔRMST(τ={tau}) = {delta_tau:.2f} days<br>95% CI [{delta_lo:.2f}, {delta_hi:.2f}]",
                            showarrow=True, arrowhead=2, ax=-40, ay=-40, bgcolor="white")
    fig_delta.update_layout(title="ΔRMST(t)", xaxis_title="Days", yaxis_title="ΔRMST(t) (days)", template="plotly_white")
    fig_delta.write_html(output_base.parent / f"{output_base.name}_DeltaRMST.html")

    # Plot 2: RMST curves - points only
    fig_rmst = go.Figure()
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v, mode="lines", line=dict(color="green", width=2), name="RMST_v(t)"))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u, mode="lines", line=dict(color="red", width=2), name="RMST_u(t)"))
    fig_rmst.update_layout(title="Restricted Mean Survival Time", xaxis_title="Days", yaxis_title="RMST(t) (days)", template="plotly_white")
    fig_rmst.write_html(output_base.parent / f"{output_base.name}_RMST_curves.html")

    # Plot 3: Survival curves - points only
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(x=t, y=sv, mode="lines", line=dict(color="green", width=2), name="Vaccinated"))
    fig_surv.add_trace(go.Scatter(x=t, y=su, mode="lines", line=dict(color="red", width=2), name="Unvaccinated"))
    fig_surv.update_layout(title="Standardized Survival Curves", xaxis_title="Days", yaxis_title="Survival", template="plotly_white")
    fig_surv.write_html(output_base.parent / f"{output_base.name}_Survival.html")

    log("All plots saved as HTML (point estimates; CI for ΔRMST(τ) only).")

# ===================== MAIN EXECUTION =====================
# ===================== DELTA METHOD FOR RMST CI =====================

def delta_method_rmst_ci(model: sm.GLM, t: np.ndarray, spline: pd.DataFrame, 
                         eps_scale: float = 1e-6) -> dict:
    """
    Compute asymptotic SE and CI for ΔRMST using delta method + numerical gradient.
    Uses the robust covariance from sandwich estimator.
    """
    params = model.params.copy()
    cov = model.cov_params()  # This uses the HC0 robust cov we applied earlier
    n_params = len(params)
    
    def compute_delta_rmst(p: np.ndarray) -> float:
        original_params = model.params.copy()
        model.params = pd.Series(p, index=model.params.index)
        
        Sv = predict_survival(model, t, 1, spline)
        Su = predict_survival(model, t, 0, spline)
        rmst_v = compute_rmst(t, Sv)
        rmst_u = compute_rmst(t, Su)
        delta = rmst_v - rmst_u
        
        model.params = original_params
        return delta
    
    delta_hat = compute_delta_rmst(params.values)
    
    # Numerical central difference gradient
    grad = np.zeros(n_params)
    for i in range(n_params):
        eps = eps_scale * max(1e-8, abs(params.iloc[i]))  # Adaptive eps for stability
        p_plus = params.values.copy()
        p_plus[i] += eps
        delta_plus = compute_delta_rmst(p_plus)
        
        p_minus = params.values.copy()
        p_minus[i] -= eps
        delta_minus = compute_delta_rmst(p_minus)
        
        grad[i] = (delta_plus - delta_minus) / (2 * eps)
    
    var_delta = np.dot(grad, np.dot(cov, grad))
    se_delta = np.sqrt(max(0.0, var_delta))
    
    z = stats.norm.ppf(0.975)  # ~1.96 for 95%
    ci_low = delta_hat - z * se_delta
    ci_high = delta_hat + z * se_delta
    
    return {
        'delta_tau': delta_hat,
        'se_delta': se_delta,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

def main():
    log("Starting Hernán-style target trial emulation (sandwich variance version)")
    log(f"Dataset: {DATA_SET} | Age: {AGE} | Grace: {CONFIG['grace_period']} days")

    raw = load_and_prepare_data(CONFIG["input_path"])

    first = int(raw.loc[raw["first_dose_day"].notna(), "first_dose_day"].min())
    last_obs = min(raw["death_day"].max(), raw["first_dose_day"].max()) - CONFIG["safety_buffer"]
    last_obs = float(last_obs)

    if last_obs <= first:
        raise ValueError(f"No follow-up window: first={first}, last_obs={last_obs}")

    log(f"Study window: first={first}, last_obs={last_obs}")

    t_grid = np.arange(0, int(last_obs - first), dtype=int)

    # Compute marginal p_init
    p_init = np.zeros(len(t_grid), dtype=float)
    for i, s in enumerate(range(first, int(last_obs))):
        eligible = ((raw['death_day'].isna() | (raw['death_day'] >= s)) &
                    (raw['first_dose_day'].isna() | (raw['first_dose_day'] >= s))).sum()
        initiates = (raw['first_dose_day'] == s).sum()
        p_init[i] = initiates / eligible if eligible > 0 else 0.0
    log("Computed marginal p_init for IPW")

    spline = build_spline_basis(t_grid, df_deg=CONFIG["time_df"], degree=CONFIG["spline_degree"])
    agg = aggregate_daily_updated(raw, first, last_obs, p_init, CONFIG["grace_period"], t_grid)

    model = fit_pooled_logistic(agg, spline)

    Sv = predict_survival(model, t_grid, 1, spline)
    Su = predict_survival(model, t_grid, 0, spline)

    estimands = compute_estimands(Sv, Su, t_grid)
    log(f"Main results: ΔRMST(τ={estimands['tau']}) = {estimands['delta_tau']:.2f} days")
    log(f"VE(τ={estimands['tau']}): {estimands['ve_tau']:+.3%}")

    asymp_results = delta_method_rmst_ci(model, t_grid, spline, CONFIG["delta_eps_scale"])

    log("Sandwich + delta method results:")
    log(f"ΔRMST(τ) = {asymp_results['delta_tau']:.2f} days")
    log(f"SE = {asymp_results['se_delta']:.2f}")
    log(f"95% asymptotic CI: [{asymp_results['ci_low']:.2f}, {asymp_results['ci_high']:.2f}]")

    plot_and_save(estimands, asymp_results, CONFIG["output_base"])

    log("Analysis complete (sandwich variance adapted).")

if __name__ == "__main__":
    main()