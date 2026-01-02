#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hernán-style sequential target-trial emulation (Design B) - Fully fixed version
Implements calendar-time sequential trials with grace period, pooled logistic regression,
RMST estimation, cluster bootstrap, and IPW for artificial censoring.
Exploratory analysis — no confounder adjustment by design.

Scientific notes (for methods/documentation):
- Estimand: Pooled effect of initiating vaccination at calendar day t vs not initiating,
  averaged across eligible t, with 14-day grace period in treatment definition.
  IPW corrects artificial censoring from cloning.
- Confounding: Only time trends (B-splines) + censoring; NO adjustment for healthy-vaccinee
  bias, indication, or other confounders.
- Grace period: Deaths during grace attributed to unvaccinated (A=0).
- IPW: Marginal (empirical) p_init; positivity monitored via weight diagnostics.
"""

from __future__ import annotations
import random
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.api import BSplines
from scipy.special import expit
from scipy.integrate import simpson
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import logging
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import plotly.graph_objects as go

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

if DATA_SET not in DATA_CONFIG:
    raise ValueError(f"Unknown DATA_SET '{DATA_SET}'")

selected = DATA_CONFIG[DATA_SET]
INPUT = Path(selected["input"].format(age=AGE))
OUTPUT_BASE = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) hernan style sequential trial poold logistics RMST") / f"AC) hernan style sequential trial poold logistics RMST{selected['suffix']}"

CONFIG = {
    "age_ref_year": 2023,
    "study_start": pd.Timestamp("2020-01-01"),
    "input_path": INPUT,
    "output_base": OUTPUT_BASE,
    "n_boot": 2,  # Increase to 200+ for final runs
    "boot_subsample": 0.4,
    "random_seed": 12345,
    "n_cores": 4,
    "safety_buffer": 30,
    "time_df": 4,
    "spline_degree": 3,
    "debug_single_rep": False,
    "tau": 90,
    "grace_period": 0,  # days
}

CONFIG["output_base"].parent.mkdir(parents=True, exist_ok=True)
MAIN_LOG = CONFIG["output_base"].parent / f"{CONFIG['output_base'].name}_AG{AGE}.txt"
BOOT_LOG_DIR = CONFIG["output_base"].parent / "bootstrap_logs"
BOOT_LOG_DIR.mkdir(parents=True, exist_ok=True)

random.seed(CONFIG["random_seed"])
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

    # CRITICAL: Coerce to float early to avoid object dtype issues
    raw["death_day"] = pd.to_numeric(raw["death_day"], errors="coerce").astype(float)
    raw["first_dose_day"] = pd.to_numeric(raw["first_dose_day"], errors="coerce").astype(float)

    log(f"Subjects after age filter: {len(raw)}")
    log(f"Deaths: {raw['death_day'].notna().sum()}, Vaccinated: {raw['first_dose_day'].notna().sum()}")
    if len(raw) == 0:
        raise ValueError("No subjects after age filtering")
    return raw

# ===================== AGGREGATION - FULLY FIXED VERSION =====================

def aggregate_daily_updated(raw: pd.DataFrame, first: int, last_obs: float, p_init: np.ndarray, grace: int, t_grid: np.ndarray) -> pd.DataFrame:
    max_window = len(t_grid)
    events_u_total = np.zeros(max_window, dtype=float)
    risk_u_total = np.zeros(max_window, dtype=float)
    events_v_total = np.zeros(max_window, dtype=float)
    risk_v_total = np.zeros(max_window, dtype=float)

    last_day_int = int(last_obs)
    if len(p_init) < (last_day_int - first):
        raise ValueError("p_init length must be at least last_obs - first")

    weights_used = []  # for diagnostics

    # Precompute NumPy arrays for vectorized eligibility
    death_np = raw['death_day'].to_numpy()
    first_np = raw['first_dose_day'].to_numpy()
    id_np = raw['subject_id'].to_numpy()

    for start_day in tqdm(range(first, last_day_int), desc="Aggregating trials"):
        max_k = min(max_window, last_day_int - start_day)
        if max_k <= 0:
            continue

        end_for_compute = start_day + max_k

        # Vectorized eligibility
        eligible_mask = (
            (np.isnan(death_np) | (death_np >= start_day)) &
            (np.isnan(first_np) | (first_np >= start_day))
        )
        eligible_ids = id_np[eligible_mask]
        if len(eligible_ids) == 0:
            continue

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
            events_non = np.zeros(max_k, dtype=float)
            risk_non = np.zeros(max_k, dtype=float)

        # A=0: Grace period from initiators
        events_grace = np.zeros(max_k, dtype=float)
        risk_grace = np.zeros(max_k, dtype=float)

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
        events_v_trial = np.zeros(max_k, dtype=float)
        risk_v_trial = np.zeros(max_k, dtype=float)

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

        # Safe IPW accumulation
        idx0 = start_day - first
        for k in range(max_k):
            if k == 0:
                ipw = 1.0
            else:
                s_idx = max(0, idx0)
                e_idx = min(len(p_init), idx0 + k)  # exclusive
                if s_idx >= e_idx:
                    ipw = 1.0
                else:
                    probs = 1.0 - p_init[s_idx:e_idx]
                    log_probs = np.sum(np.log(np.clip(probs, 1e-12, 1.0)))
                    surv = float(np.exp(log_probs))
                    if surv < 1e-12:
                        ipw = 0.0
                    else:
                        ipw = min(1.0 / surv, 1e6)
            if ipw > 1.0:
                weights_used.append(ipw)

            events_u_total[k] += events_non[k] * ipw + events_grace[k]
            risk_u_total[k] += risk_non[k] * ipw + risk_grace[k]
            events_v_total[k] += events_v_trial[k]
            risk_v_total[k] += risk_v_trial[k]

    # Summaries + diagnostics
    pt_v = risk_v_total.sum()
    pt_u = risk_u_total.sum()
    e_v = events_v_total.sum()
    e_u = events_u_total.sum()

    rate_v = e_v / pt_v * 100_000 if pt_v > 0 else np.nan
    rate_u = e_u / pt_u * 100_000 if pt_u > 0 else np.nan

    log(f"Person-time V: {pt_v:,.2f} | U: {pt_u:,.2f}")
    log(f"Deaths V: {e_v:,.2f} | U: {e_u:,.2f}")
    log(f"Crude rate V: {rate_v:.2f} | U: {rate_u:.2f} per 100,000 pd")

    if weights_used:
        w = np.array(weights_used)
        log(f"IPW diagnostics: mean={w.mean():.2f}, median={np.median(w):.2f}, "
            f"max={w.max():.1e}, % capped={(w >= 1e6).mean()*100:.1f}%")
    else:
        log("No IPW weights >1 applied")

    agg = pd.DataFrame({
        "day": np.concatenate([t_grid, t_grid]),
        "vaccinated": np.concatenate([np.ones_like(t_grid), np.zeros_like(t_grid)]),
        "events": np.concatenate([events_v_total, events_u_total]),
        "risk": np.concatenate([risk_v_total, risk_u_total]),
    })
    return agg

# ===================== MODELING =====================

def fit_pooled_logistic(agg: pd.DataFrame, spline: pd.DataFrame, log_details: bool = True, start_params=None) -> sm.GLM:
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = sm.GLM(p, X, family=sm.families.Binomial(), freq_weights=w).fit(start_params=start_params)

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
# ===================== BOOTSTRAP =====================

def bootstrap_once(i: int, raw: pd.DataFrame, t: np.ndarray, spline: pd.DataFrame, first: int, last_obs: float, grace: int, start_params=None) -> tuple | None:
    try:
        rng = np.random.default_rng(CONFIG["random_seed"] + i)
        ids = raw["subject_id"].to_numpy()
        n_draw = len(ids) if CONFIG["boot_subsample"] >= 1.0 else int(round(len(ids) * CONFIG["boot_subsample"]))
        samp = rng.choice(ids, n_draw, replace=True)
        # Preserve multiplicity with concat
        raw_b = pd.concat([raw.loc[raw['subject_id'] == sid].copy() for sid in samp], ignore_index=True)

        # Recompute p_init for bootstrap sample
        p_init_b = np.zeros(len(t), dtype=float)
        for j, s in enumerate(range(first, int(last_obs))):
            eligible_mask = (raw_b['death_day'].isna() | (raw_b['death_day'] >= s)) & \
                            (raw_b['first_dose_day'].isna() | (raw_b['first_dose_day'] >= s))
            eligible = eligible_mask.sum()
            initiates = (raw_b['first_dose_day'] == s).sum()
            p_init_b[j] = initiates / eligible if eligible > 0 else 0.0

        agg_b = aggregate_daily_updated(raw_b, first, last_obs, p_init_b, grace, t)
        m = fit_pooled_logistic(agg_b, spline, log_details=False, start_params=start_params)
        log(f"Bootstrap rep {i}: GLM converged={m.converged}")  # Added diagnostic
        Sv_b = predict_survival(m, t, 1, spline)
        Su_b = predict_survival(m, t, 0, spline)
        return Sv_b, Su_b
    except Exception as e:
        log(f"Bootstrap replicate {i} failed: {str(e)}")
        return None

def run_bootstrap(raw: pd.DataFrame, t: np.ndarray, spline: pd.DataFrame, first: int, last_obs: float, grace: int, start_params=None) -> list:
    log(f"Starting bootstrap ({CONFIG['n_boot']} reps, subsample={CONFIG['boot_subsample']:.2f})")
    if CONFIG["debug_single_rep"]:
        test = bootstrap_once(0, raw, t, spline, first, last_obs, grace, start_params)
        return [test] if test else []

    boot_raw = Parallel(n_jobs=CONFIG["n_cores"], backend="loky")(
        delayed(bootstrap_once)(i, raw, t, spline, first, last_obs, grace, start_params)
        for i in range(CONFIG["n_boot"])
    )
    successful = [b for b in boot_raw if b is not None]
    log(f"Bootstrap: {len(successful)}/{CONFIG['n_boot']} successful")
    return successful

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

def plot_and_save(estimands: dict, boot_results: list, output_base: Path):
    t = np.arange(len(estimands["delta"]))
    delta = estimands["delta"]
    rmst_v = estimands["rmst_v"]
    rmst_u = estimands["rmst_u"]
    sv = estimands["sv"]
    su = estimands["su"]
    tau = estimands["tau"]
    delta_tau = estimands["delta_tau"]

    if len(boot_results) == 0:
        boot_Sv = np.array([sv])
        boot_Su = np.array([su])
    else:
        boot_Sv = np.array([r[0] for r in boot_results])
        boot_Su = np.array([r[1] for r in boot_results])

    boot_rmst_v = np.array([rmst_curve(Sv_b, t) for Sv_b in boot_Sv])
    boot_rmst_u = np.array([rmst_curve(Su_b, t) for Su_b in boot_Su])
    boot_delta = boot_rmst_v - boot_rmst_u

    delta_lo, delta_hi = np.percentile(boot_delta, [2.5, 97.5], axis=0)
    rmst_v_lo, rmst_v_hi = np.percentile(boot_rmst_v, [2.5, 97.5], axis=0)
    rmst_u_lo, rmst_u_hi = np.percentile(boot_rmst_u, [2.5, 97.5], axis=0)
    sv_lo, sv_hi = np.percentile(boot_Sv, [2.5, 97.5], axis=0)
    su_lo, su_hi = np.percentile(boot_Su, [2.5, 97.5], axis=0)

    # Plot 1: ΔRMST(t)
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=t, y=delta_hi, line=dict(width=0), showlegend=False))
    fig_delta.add_trace(go.Scatter(x=t, y=delta_lo, fill="tonexty", fillcolor="rgba(0,100,200,0.2)", line=dict(width=0), showlegend=False))
    fig_delta.add_trace(go.Scatter(x=t, y=delta, mode="lines", line=dict(color="black", width=2), name="ΔRMST(t)"))
    fig_delta.add_hline(y=0, line=dict(color="gray", dash="dash"))
    fig_delta.add_annotation(x=tau, y=delta_tau,
                            text=f"ΔRMST(τ={tau}) = {delta_tau:.2f} days<br>95% CI [{delta_lo[-1]:.2f}, {delta_hi[-1]:.2f}]",
                            showarrow=True, arrowhead=2, ax=-40, ay=-40, bgcolor="white")
    fig_delta.update_layout(title="ΔRMST(t)", xaxis_title="Days", yaxis_title="ΔRMST(t) (days)", template="plotly_white")
    fig_delta.write_html(output_base.parent / f"{output_base.name}_DeltaRMST.html")

    # Plot 2: RMST curves
    fig_rmst = go.Figure()
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v_hi, line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v_lo, fill="tonexty", fillcolor="rgba(0,150,0,0.2)", line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u_hi, line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u_lo, fill="tonexty", fillcolor="rgba(200,0,0,0.2)", line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v, mode="lines", line=dict(color="green", width=2), name="RMST_v(t)"))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u, mode="lines", line=dict(color="red", width=2), name="RMST_u(t)"))
    fig_rmst.update_layout(title="Restricted Mean Survival Time", xaxis_title="Days", yaxis_title="RMST(t) (days)", template="plotly_white")
    fig_rmst.write_html(output_base.parent / f"{output_base.name}_RMST_curves.html")

    # Plot 3: Survival curves
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(x=t, y=sv_hi, line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=sv_lo, fill="tonexty", fillcolor="rgba(0,150,0,0.2)", line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=su_hi, line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=su_lo, fill="tonexty", fillcolor="rgba(200,0,0,0.2)", line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=sv, mode="lines", line=dict(color="green", width=2), name="Vaccinated"))
    fig_surv.add_trace(go.Scatter(x=t, y=su, mode="lines", line=dict(color="red", width=2), name="Unvaccinated"))
    fig_surv.update_layout(title="Standardized Survival Curves", xaxis_title="Days", yaxis_title="Survival", template="plotly_white")
    fig_surv.write_html(output_base.parent / f"{output_base.name}_Survival.html")

    log("All plots saved as HTML.")

# ===================== MAIN EXECUTION =====================

def main():
    log("Starting Hernán-style target trial emulation (fixed version)")
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
    start_params = model.params  # For bootstrap warm start

    Sv = predict_survival(model, t_grid, 1, spline)
    Su = predict_survival(model, t_grid, 0, spline)

    estimands = compute_estimands(Sv, Su, t_grid)
    log(f"Main results: ΔRMST(τ={estimands['tau']}) = {estimands['delta_tau']:.2f} days")
    log(f"VE(τ={estimands['tau']}): {estimands['ve_tau']:+.3%}")

    boot_results = run_bootstrap(raw, t_grid, spline, first, last_obs, CONFIG["grace_period"], start_params)

    plot_and_save(estimands, boot_results, CONFIG["output_base"])

    log("Analysis complete.")

if __name__ == "__main__":
    main()