#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Design A (Hernán-style) with Design B optimizations:
- Vectorized per-subject precompute (contiguous arrays)
- Numba-accelerated IRLS for pooled logistic
- Threaded precompute with progress bar
- m-out-of-n bootstrap subsampling
- Timing logs for heavy steps
"""

from __future__ import annotations
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import numba as nb
from scipy.integrate import simpson
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
AGE = 70
DATA_SET = "real"  # "real", "sim", "reclassified"

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
OUTPUT_BASE = Path(
    r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AC) hernan style fixed time poold logistics RMST"
) / f"AC) hernan style fixed time poold logistics RMST{selected['suffix']}"

CONFIG = {
    "age_ref_year": 2023,
    "study_start": pd.Timestamp("2020-01-01"),
    "input_path": INPUT,
    "output_base": OUTPUT_BASE,

    # Performance / bootstrap
    "n_boot": 200,            # final: 200+, dev: 20
    "boot_subsample": 0.10,   # m-out-of-n subsample fraction
    "random_seed": 12345,
    "n_cores": 4,

    "safety_buffer": 30,
    "time_df": 2,             # spline df (time basis)
    "debug_single_rep": False,
    "grace_period": 0,
    "strict": False,
}

CONFIG["output_base"].parent.mkdir(parents=True, exist_ok=True)
MAIN_LOG = CONFIG["output_base"].parent / f"{CONFIG['output_base'].name}_AG{AGE}.txt"
BOOT_LOG_DIR = CONFIG["output_base"].parent / "bootstrap_logs"
BOOT_LOG_DIR.mkdir(parents=True, exist_ok=True)

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------- LOGGING ----------------
def log(msg: str, timestamp: bool = True):
    if timestamp:
        ts = datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")
        line = f"{ts}  {msg}"
    else:
        line = msg
    print(line)
    with open(MAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------- Numba IRLS (Numba warmup included) ----------------
@nb.njit
def _logit(x):
    return 1.0 / (1.0 + np.exp(-x))

@nb.njit
def _irls_logistic(X, y, w, max_iter=50, tol=1e-8):
    """
    Numba-compiled IRLS for weighted logistic regression.
    X: (n, p) float64
    y: (n,) float64  (observed proportions events/risk)
    w: (n,) float64  (freq weights = risk)
    Returns beta (p,) float64
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _logit(eta)
        v = mu * (1.0 - mu)
        for i in range(n):
            if v[i] < 1e-12:
                v[i] = 1e-12
        z = eta + (y - mu) / v
        W = w * v
        A = np.zeros((p, p), dtype=np.float64)
        b = np.zeros(p, dtype=np.float64)
        for j in range(p):
            for k in range(p):
                s = 0.0
                for i in range(n):
                    s += X[i, j] * W[i] * X[i, k]
                A[j, k] = s
            s2 = 0.0
            for i in range(n):
                s2 += X[i, j] * W[i] * z[i]
            b[j] = s2
        try:
            beta_new = np.linalg.solve(A, b)
        except Exception:
            for d in range(p):
                A[d, d] += 1e-8
            beta_new = np.linalg.solve(A, b)
        maxdiff = 0.0
        for j in range(p):
            diff = abs(beta_new[j] - beta[j])
            if diff > maxdiff:
                maxdiff = diff
        beta = beta_new
        if maxdiff < tol:
            return beta
    return beta

def _numba_warmup():
    X = np.ones((5, 6), dtype=np.float64)
    y = np.full(5, 0.1, dtype=np.float64)
    w = np.ones(5, dtype=np.float64)
    try:
        _irls_logistic(X, y, w, max_iter=1)
    except Exception:
        pass

# warm up JIT early
_numba_warmup()

# ---------------- HELPERS ----------------
def compute_rmst(t: np.ndarray, S: np.ndarray) -> float:
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    return float(simpson(S, x=t)) if len(t) > 1 else 0.0

def rmst_curve(S: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    return np.array([compute_rmst(t[:i+1], S[:i+1]) for i in range(len(t))])

def compute_daily_events(df: pd.DataFrame, start_day: int, end_day: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate interval data into daily events and risk for [start_day, end_day).
    """
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

def natural_cubic_spline_basis(x: np.ndarray, df: int) -> pd.DataFrame:
    x = np.asarray(x, float)
    q = np.linspace(0, 1, df + 1)[1:-1]
    knots = np.quantile(x, q) if len(q) > 0 else np.array([])
    kmin, kmax = x.min(), x.max()
    def d(z, k):
        return np.maximum(z - k, 0) ** 3
    cols = {"time_lin": x}
    denom = kmax - kmin if kmax > kmin else 1.0
    for j, k in enumerate(knots, 1):
        cols[f"time_s{j}"] = (
            d(x, k)
            - d(x, kmax) * (kmax - k) / denom
            + d(x, kmin) * (k - kmin) / denom
        )
    return pd.DataFrame(cols).astype("float32")

def build_spline_basis(t: np.ndarray, df_deg: int) -> pd.DataFrame:
    spline = natural_cubic_spline_basis(t, df_deg)
    spline["day"] = t
    spline.set_index("day", inplace=True)
    return spline

# ---------------- DATA LOAD & CLONES ----------------
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
    if len(raw) == 0:
        raise ValueError("No subjects after age filtering")
    return raw

def build_clones_and_time_grid(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, int, float]:
    first = int(raw.loc[raw["first_dose_day"].notna(), "first_dose_day"].min())
    last_obs = min(raw["death_day"].max(), raw["first_dose_day"].max()) - CONFIG["safety_buffer"]
    last_obs = float(last_obs)
    window = int(last_obs - first)
    if window <= 0:
        raise ValueError(f"No follow-up window: first={first}, last_obs={last_obs}")
    t = np.arange(window, dtype=int)
    grace = int(CONFIG["grace_period"])
    log(f"Using grace period of {grace} days for vaccinated arm")
    log(f"Follow-up window length: {window} days")
    clones = []
    skipped_preindex_doses = 0
    for _, r in raw.iterrows():
        sid = int(r["subject_id"])
        d = float(r["death_day"]) if pd.notna(r["death_day"]) else np.nan
        f = float(r["first_dose_day"]) if pd.notna(r["first_dose_day"]) else np.nan
        eligible = (np.isnan(f) or f >= first) and (np.isnan(d) or d >= first)
        if not eligible:
            continue
        su = float(first)
        eu_cands = [x for x in [f, d, last_obs] if not np.isnan(x)]
        if eu_cands:
            eu = min(eu_cands)
            if eu > su:
                clones.append((sid, 0, su, eu, int(not np.isnan(d) and d <= eu)))
        if not np.isnan(f) and f >= first:
            dose_day = float(f)
            if grace > 0:
                grace_start = dose_day
                grace_end = min(dose_day + grace, d if not np.isnan(d) else last_obs, last_obs)
                if grace_end > grace_start:
                    clones.append((sid, 0, grace_start, grace_end, int(not np.isnan(d) and grace_start <= d <= grace_end)))
            v_start = dose_day + grace
            v_end = min(d if not np.isnan(d) else last_obs, last_obs)
            if v_end > v_start:
                clones.append((sid, 1, v_start, v_end, int(not np.isnan(d) and v_start <= d <= v_end)))
        else:
            if not np.isnan(f) and f < first:
                skipped_preindex_doses += 1
    df = pd.DataFrame(clones, columns=["id", "vaccinated", "start", "stop", "event"])
    df = df.astype({"id": "int32", "vaccinated": "int32", "event": "int32"})
    log(f"Total clone intervals: {len(df)}")
    log(f"  Vaccinated intervals:   {(df['vaccinated'] == 1).sum()}")
    log(f"  Unvaccinated intervals: {(df['vaccinated'] == 0).sum()}")
    log(f"Analysis window: {len(t)} days (day {first} to {last_obs})")
    if skipped_preindex_doses > 0:
        msg = f"Skipped {skipped_preindex_doses} observed doses before index (first_dose_day < first)"
        log(msg)
        if CONFIG["strict"]:
            raise RuntimeError(msg)
    return df, t, first, last_obs

# ---------------- VECTORIZED PRECOMPUTE ----------------
def precompute_bootstrap_vectorized(clones: pd.DataFrame, t: np.ndarray, first: int, last_obs: float, n_jobs: int = 4):
    """
    Returns:
      ids_sorted,
      events_arm0 (n_ids, window) float32,
      risk_arm0   (n_ids, window) float32,
      events_arm1 (n_ids, window) float32,
      risk_arm1   (n_ids, window) float32
    """
    ids = np.unique(clones.id.to_numpy())
    ids_sorted = np.sort(ids)
    n_ids = len(ids_sorted)
    window = len(t)
    events_arm0 = np.zeros((n_ids, window), dtype=np.float32)
    risk_arm0   = np.zeros((n_ids, window), dtype=np.float32)
    events_arm1 = np.zeros((n_ids, window), dtype=np.float32)
    risk_arm1   = np.zeros((n_ids, window), dtype=np.float32)
    id_to_idx = {int(sid): idx for idx, sid in enumerate(ids_sorted)}
    groups = clones.groupby("id")
    def _worker_fill(sid):
        g = groups.get_group(sid)
        ev0, r0 = compute_daily_events(g[g.vaccinated == 0], first, last_obs)
        ev1, r1 = compute_daily_events(g[g.vaccinated == 1], first, last_obs)
        return int(sid), ev0.astype(np.float32), r0.astype(np.float32), ev1.astype(np.float32), r1.astype(np.float32)
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker_fill)(sid) for sid in tqdm(ids_sorted, desc="Precompute vectorized", mininterval=0.5)
    )
    for sid, ev0, r0, ev1, r1 in results:
        idx = id_to_idx[int(sid)]
        L0 = min(window, ev0.shape[0])
        events_arm0[idx, :L0] = ev0[:L0]
        risk_arm0[idx, :L0] = r0[:L0]
        L1 = min(window, ev1.shape[0])
        events_arm1[idx, :L1] = ev1[:L1]
        risk_arm1[idx, :L1] = r1[:L1]
    return ids_sorted, events_arm0, risk_arm0, events_arm1, risk_arm1

# ---------------- AGGREGATION & MODELING ----------------
def aggregate_daily(clones: pd.DataFrame, t: np.ndarray, first: int, last_obs: float) -> pd.DataFrame:
    ev_v, r_v = compute_daily_events(clones[clones.vaccinated == 1], first, last_obs)
    ev_u, r_u = compute_daily_events(clones[clones.vaccinated == 0], first, last_obs)
    pt_v = r_v.sum()
    pt_u = r_u.sum()
    e_v = ev_v.sum()
    e_u = ev_u.sum()
    rate_v = e_v / pt_v * 100_000 if pt_v > 0 else np.nan
    rate_u = e_u / pt_u * 100_000 if pt_u > 0 else np.nan
    log(f"Person-time V: {pt_v:,} | U: {pt_u:,}")
    log(f"Deaths V: {e_v} | U: {e_u}")
    log(f"Crude rate V: {rate_v:.2f} | U: {rate_u:.2f} per 100,000 pd")
    log(f"Min risk: {min(r_v.min(), r_u.min())}, Max risk: {max(r_v.max(), r_u.max())}")
    agg = pd.DataFrame({
        "day": np.concatenate([t, t]),
        "vaccinated": np.concatenate([np.ones_like(t), np.zeros_like(t)]),
        "events": np.concatenate([ev_v, ev_u]),
        "risk": np.concatenate([r_v, r_u]),
    })
    return agg

def fit_pooled_logistic(agg: pd.DataFrame, spline: pd.DataFrame, log_details: bool = True):
    df = agg[agg.risk > 0].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No positive-risk days")
    y = (df.events / df.risk).clip(1e-9, 1 - 1e-9).astype("float64").to_numpy()
    w = df.risk.astype("float64").to_numpy()
    A = df.vaccinated.astype("float64").to_numpy()
    S = spline.loc[df.day].reset_index(drop=True).astype("float64")
    S_arr = S.to_numpy()
    n = S_arr.shape[0]
    k = S_arr.shape[1]
    X_no_intercept = np.empty((n, 2 * k + 1), dtype=np.float64)
    X_no_intercept[:, :k] = S_arr
    X_no_intercept[:, k] = A
    X_no_intercept[:, k + 1:] = S_arr * A[:, None]
    X = np.empty((n, X_no_intercept.shape[1] + 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = X_no_intercept
    if log_details:
        log(f"GLM rows: {len(X)}, predictors: {X.shape[1]}")
    beta = _irls_logistic(X, y, w)
    colnames = ["const"] + list(S.columns) + ["vaccinated"] + [f"{c}_x_vacc" for c in S.columns]
    return (beta, colnames)

def predict_survival(model, t: np.ndarray, A: int, spline: pd.DataFrame) -> np.ndarray:
    beta, colnames = model
    S = spline.loc[t].reset_index(drop=True).astype("float64")
    S_arr = S.to_numpy()
    n = S_arr.shape[0]
    k = S_arr.shape[1]
    vacc = np.full(n, float(A), dtype=np.float64)
    X_no_intercept = np.empty((n, 2 * k + 1), dtype=np.float64)
    X_no_intercept[:, :k] = S_arr
    X_no_intercept[:, k] = vacc
    X_no_intercept[:, k + 1:] = S_arr * vacc[:, None]
    X = np.empty((n, X_no_intercept.shape[1] + 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = X_no_intercept
    eta = X @ beta
    haz = 1.0 / (1.0 + np.exp(-eta))
    haz = np.clip(haz, 1e-9, 1 - 1e-9)
    S_curve = np.cumprod(1 - haz)
    if np.any(np.diff(S_curve) > 1e-6):
        msg = f"Non-monotone survival curve detected for A={A}"
        log(msg)
        if CONFIG["strict"]:
            raise RuntimeError(msg)
    return S_curve

# ---------------- BOOTSTRAP ----------------
def bootstrap_once(i: int, ids: np.ndarray, events_arm0: np.ndarray, risk_arm0: np.ndarray,
                   events_arm1: np.ndarray, risk_arm1: np.ndarray, t: np.ndarray, spline: pd.DataFrame) -> tuple | None:
    log_path = BOOT_LOG_DIR / f"worker_{i:04d}.txt"
    try:
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(f"replicate {i} start\n")
            rng = np.random.default_rng(CONFIG["random_seed"] + i)
            n_draw = len(ids) if CONFIG["boot_subsample"] >= 1.0 else int(max(1, round(len(ids) * CONFIG["boot_subsample"])))
            samp = rng.choice(ids, n_draw, replace=True)
            flog.write(f"replicate {i}: sampling {n_draw} ids\n")
            samp_idx = np.searchsorted(ids, samp)
            ev_v = events_arm1[samp_idx].sum(axis=0)
            r_v  = risk_arm1[samp_idx].sum(axis=0)
            ev_u = events_arm0[samp_idx].sum(axis=0)
            r_u  = risk_arm0[samp_idx].sum(axis=0)
            agg_b_days = np.concatenate([t, t])
            agg_b_vacc = np.concatenate([np.ones_like(t), np.zeros_like(t)])
            agg_b_events = np.concatenate([ev_v, ev_u])
            agg_b_risk = np.concatenate([r_v, r_u])
            agg_b = pd.DataFrame({
                "day": agg_b_days,
                "vaccinated": agg_b_vacc,
                "events": agg_b_events,
                "risk": agg_b_risk,
            })
            flog.write(f"replicate {i}: fitting PLR\n")
            m = fit_pooled_logistic(agg_b, spline, log_details=False)
            flog.write(f"replicate {i}: GLM fitted\n")
            Sv_b = predict_survival(m, t, 1, spline)
            Su_b = predict_survival(m, t, 0, spline)
            flog.write(f"replicate {i} finished successfully\n")
            return Sv_b, Su_b
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(f"replicate {i} failed: {str(e)}\n")
        return None

def run_bootstrap(ids: np.ndarray, events_arm0: np.ndarray, risk_arm0: np.ndarray,
                  events_arm1: np.ndarray, risk_arm1: np.ndarray, t: np.ndarray, spline: pd.DataFrame) -> list:
    log(f"Starting bootstrap ({CONFIG['n_boot']} reps, subsample={CONFIG['boot_subsample']:.2f})")
    if CONFIG["debug_single_rep"]:
        test = bootstrap_once(0, ids, events_arm0, risk_arm0, events_arm1, risk_arm1, t, spline)
        return [test] if test else []
    with parallel_backend("threading", n_jobs=CONFIG["n_cores"]):
        with tqdm_joblib(tqdm(total=CONFIG["n_boot"], desc="Bootstrap", mininterval=0.5, disable=False)):
            boot_raw = Parallel(n_jobs=CONFIG["n_cores"], prefer="threads")(
                delayed(bootstrap_once)(i, ids, events_arm0, risk_arm0, events_arm1, risk_arm1, t, spline)
                for i in range(CONFIG["n_boot"])
            )
    successful = [b for b in boot_raw if b is not None]
    log(f"Bootstrap: {len(successful)}/{CONFIG['n_boot']} successful")
    if len(successful) == 0:
        log("No valid replicates — skipping CIs")
    return successful

# ---------------- RESULTS & PLOTTING ----------------
def compute_estimands(Sv: np.ndarray, Su: np.ndarray, t: np.ndarray) -> dict:
    rmst_v = rmst_curve(Sv, t)
    rmst_u = rmst_curve(Su, t)
    delta = rmst_v - rmst_u
    tau = t[-1]
    delta_tau = delta[-1]
    sv_tau = Sv[-1]
    su_tau = Su[-1]
    ci_v_tau = 1 - sv_tau
    ci_u_tau = 1 - su_tau
    ve_tau = np.nan if ci_u_tau == 0 else 1 - (ci_v_tau / ci_u_tau)
    nnt_year = 365.0 / delta_tau if delta_tau > 0 else np.nan
    return {
        "tau": tau,
        "delta_tau": delta_tau,
        "ve_tau": ve_tau,
        "nnt_year": nnt_year,
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

    # Bootstrap CIs
    if len(boot_results) == 0:
        delta_lo = delta_hi = rmst_v_lo = rmst_v_hi = rmst_u_lo = rmst_u_hi = sv_lo = sv_hi = su_lo = su_hi = np.full_like(t, np.nan)
        Delta_lo_tau = Delta_hi_tau = VE_lo_tau = VE_hi_tau = NNT_lo = NNT_hi = np.nan
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

        Delta_lo_tau = delta_lo[-1]
        Delta_hi_tau = delta_hi[-1]

        VE_boot = np.array([
            1 - (1 - Sv_b[-1]) / (1 - Su_b[-1]) if Su_b[-1] != 1 else np.nan
            for Sv_b, Su_b in zip(boot_Sv, boot_Su)
        ])
        VE_lo_tau = np.nanpercentile(VE_boot, 2.5)
        VE_hi_tau = np.nanpercentile(VE_boot, 97.5)

        NNT_boot = np.array([365.0 / d if d > 0 else np.nan for d in boot_delta[:, -1]])
        NNT_lo = np.nanpercentile(NNT_boot, 2.5)
        NNT_hi = np.nanpercentile(NNT_boot, 97.5)

    # ---------------- ΔRMST(t) ----------------
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=t, y=delta_hi, line=dict(width=0), showlegend=False))
    fig_delta.add_trace(go.Scatter(x=t, y=delta_lo, fill="tonexty",
                                   fillcolor="rgba(0,100,200,0.2)", line=dict(width=0), showlegend=False))
    fig_delta.add_trace(go.Scatter(x=t, y=delta, mode="lines",
                                   line=dict(color="black", width=2), name="ΔRMST(t)"))
    fig_delta.add_hline(y=0, line=dict(color="gray", dash="dash"))
    fig_delta.add_annotation(
        x=tau, y=delta_tau,
        text=f"ΔRMST(τ={tau}) = {delta_tau:.2f} days<br>95% CI [{Delta_lo_tau:.2f}, {Delta_hi_tau:.2f}]",
        showarrow=True, arrowhead=2, ax=-40, ay=-40, bgcolor="white"
    )
    fig_delta.update_layout(
        title="ΔRMST(t)",
        xaxis_title="Days",
        yaxis_title="ΔRMST(t) (days)",
        template="plotly_white"
    )
    fig_delta.write_html(output_base.parent / f"{output_base.name}_DeltaRMST.html")

    # ---------------- RMST curves ----------------
    fig_rmst = go.Figure()
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v_hi, line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v_lo, fill="tonexty",
                                  fillcolor="rgba(0,150,0,0.2)", line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u_hi, line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u_lo, fill="tonexty",
                                  fillcolor="rgba(200,0,0,0.2)", line=dict(width=0), showlegend=False))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_v, mode="lines",
                                  line=dict(color="green", width=2), name="RMST_v(t)"))
    fig_rmst.add_trace(go.Scatter(x=t, y=rmst_u, mode="lines",
                                  line=dict(color="red", width=2), name="RMST_u(t)"))
    fig_rmst.update_layout(
        title="Restricted Mean Survival Time",
        xaxis_title="Days",
        yaxis_title="RMST(t) (days)",
        template="plotly_white"
    )
    fig_rmst.write_html(output_base.parent / f"{output_base.name}_RMST_curves.html")

    # ---------------- Survival curves ----------------
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(x=t, y=sv_hi, line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=sv_lo, fill="tonexty",
                                  fillcolor="rgba(0,150,0,0.2)", line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=su_hi, line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=su_lo, fill="tonexty",
                                  fillcolor="rgba(200,0,0,0.2)", line=dict(width=0), showlegend=False))
    fig_surv.add_trace(go.Scatter(x=t, y=sv, mode="lines",
                                  line=dict(color="green", width=2), name="Vaccinated"))
    fig_surv.add_trace(go.Scatter(x=t, y=su, mode="lines",
                                  line=dict(color="red", width=2), name="Unvaccinated"))
    fig_surv.update_layout(
        title="Standardized Survival Curves",
        xaxis_title="Days",
        yaxis_title="Survival",
        template="plotly_white"
    )
    fig_surv.write_html(output_base.parent / f"{output_base.name}_Survival.html")

    log("All plots saved as HTML.")

# ---------------- MAIN ----------------
def main():
    log("Starting Design A (vectorized + Numba IRLS + threaded precompute)")
    raw = load_and_prepare_data(CONFIG["input_path"])
    clones, t, first, last_obs = build_clones_and_time_grid(raw)
    spline = build_spline_basis(t, CONFIG["time_df"])
    agg = aggregate_daily(clones, t, first, last_obs)

    # Fit main model
    model = fit_pooled_logistic(agg, spline)
    Sv = predict_survival(model, t, 1, spline)
    Su = predict_survival(model, t, 0, spline)

    log(f"S_v(0)={Sv[0]:.4f}, S_v(τ)={Sv[-1]:.4f}")
    log(f"S_u(0)={Su[0]:.4f}, S_u(τ)={Su[-1]:.4f}")

    estimands = compute_estimands(Sv, Su, t)
    log(f"Main results: ΔRMST(τ={estimands['tau']}) = {estimands['delta_tau']:.2f} days")
    log(f"VE(τ={estimands['tau']}): {estimands['ve_tau']:+.3%}")

    # Vectorized precompute
    t0 = time.time()
    ids, events_arm0, risk_arm0, events_arm1, risk_arm1 = precompute_bootstrap_vectorized(
        clones, t, first, last_obs, n_jobs=CONFIG["n_cores"]
    )
    log(f"precompute_bootstrap_vectorized took {time.time() - t0:.1f} s")

    # Bootstrap
    boot_results = run_bootstrap(
        ids, events_arm0, risk_arm0, events_arm1, risk_arm1, t, spline
    )

    # 🔥 Plotting (this is the missing line)
    plot_and_save(estimands, boot_results, CONFIG["output_base"])

    log("Analysis complete.")


if __name__ == "__main__":
    main()
