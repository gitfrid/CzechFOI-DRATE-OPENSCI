#!/usr/bin/env python3
"""
IPW RMST Validation — Stress-Test (Clone Censor weight RMST Stress-Test simulation)
- Scenarios: Null (HR≈1) and Protective (HR≈0.7)
- Realistic DGP: calendar mortality waves, age/sex rollout, latent health confounder
- Clone-Censor-Weighting with strategy-specific IPCW
- Calendar + relative splines in treatment & hazard models
- Stabilized weights (clipped factors, truncation)
- Pooled logistic (CLogLog) on person-day rows, freq_weights
- Placebo falsification via simulated vaccination from numerator treatment model
- Weighted t₀ averaging by n_eligible
- Diagnostics: ESS, SMDs, positivity

Author: AI / Drifting assistance   Date: January 2026 Version 1.1 (methodologically aligned)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.gam.api import BSplines
from joblib import Parallel, delayed
from pathlib import Path
import logging, time, math

# ---------------- CONFIG ----------------
C = {
    "start": pd.Timestamp("2020-01-01"),
    "ref": 2023,
    "N": 2000,
    "tau": 90,
    "max_t": 240,
    "ages": [60],                 # single age group for validation
    # NOTE: lags are now >= 0 to ensure attainable strategies in this setup.
    # If you want negative lags for falsification, add them explicitly and
    # treat those as non-causal checks.
    "lags": [0, 14],
    "spline_df": 5,
    "spline_deg": 3,
    "alpha": 1e-4,
    "n_boot": 50,                 # small for validation; increase for final runs
    "seed": 2026,
    "min_group": 30,
    "n_jobs": -1,
    "weight_trunc_pct": 99.5,
    "t0_quantiles": [0.1,0.3,0.5,0.7,0.9],
    "out_dir": Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AG) Clone Censor weight RMST Stress-Test simulation")
}

C["out_dir"].mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(message)s",
                    handlers=[logging.FileHandler(C["out_dir"] / "validation.log"),
                              logging.StreamHandler()])

np.random.seed(C['seed'])

# ============================================================
# Synthetic data generator (realistic)
# ============================================================
def generate_synthetic(N, scenario='A'):
    """
    Generate cohort with:
     - calendar vaccination rollout waves (age/sex prioritization)
     - calendar mortality wave (sinusoidal + Gaussian wave)
     - latent health score affecting both vaccination timing and death hazard
       (higher health -> healthier -> more likely to vaccinate early AND lower death hazard)
     - scenario 'A' null (vac effect HR=1), 'B' protective (HR~0.7)
    """
    rows = []
    max_t = C['max_t']
    t = np.arange(0, max_t + 1)

    # calendar vaccination baseline: logistic rollout with a mid-wave and age prioritization
    base_vacc_curve = 1 / (1 + np.exp(-(t - 100) / 20))  # increases over time
    # add a supply wave bump
    base_vacc_curve += 0.2 * np.exp(-((t - 140)**2) / (2 * 25**2))
    # normalize to probabilities per day (small)
    vacc_prob_t_base = 0.01 * base_vacc_curve / base_vacc_curve.max()

    # calendar mortality baseline: seasonal + epidemic wave
    seasonal = 0.0006 * (1 + 0.3 * np.sin(2 * np.pi * t / 180))
    epidemic = 0.0015 * np.exp(-((t - 110)**2) / (2 * 30**2))
    base_hazard_t = seasonal + epidemic  # time-varying baseline hazard

    # scenario effect
    vac_hr = 1.0 if scenario == 'A' else 0.7

    for i in range(N):
        # baseline covariates
        age = 60
        sex = np.random.binomial(1, 0.5)

        # latent health score: higher -> healthier
        # healthier individuals vaccinate earlier and have lower death hazard
        health = np.random.normal(0, 1)

        # age/sex modifies vaccination propensity: older prioritized slightly, sex effect small
        age_factor = 1.0 + 0.05 * (age - 60)
        sex_factor = 1.0 + 0.05 * (1 - sex)  # sex=0 slightly higher uptake

        # per-day vaccination probability:
        #   base * modifiers * health effect (healthier -> higher uptake)
        health_vacc_factor = np.exp(0.2 * health)  # healthier → more likely to vaccinate
        vacc_prob_t = vacc_prob_t_base * age_factor * sex_factor * health_vacc_factor
        vacc_prob_t = np.clip(vacc_prob_t, 0, 0.5)

        # sample vaccination day: first day with Bernoulli trial
        v_day = np.nan
        for day in range(0, max_t + 1):
            if np.random.uniform() < vacc_prob_t[day]:
                v_day = day
                break

        # death hazard per day depends on calendar baseline, age, sex, and health;
        # after vaccination hazard multiplies by vac_hr
        death_day = np.nan
        for day in range(0, max_t + 1):
            # healthier (higher health) → lower hazard
            health_hazard_factor = np.exp(-0.3 * health)
            h = base_hazard_t[day] \
                * (1.0 + 0.02 * (age - 60)) \
                * (1.0 + 0.05 * sex) \
                * health_hazard_factor

            if (not np.isnan(v_day)) and day >= v_day:
                h *= vac_hr

            if np.random.uniform() < h:
                death_day = day
                break

        rows.append({
            'Rok_narozeni': C['ref'] - age,
            'Pohlavikod': 1 if sex == 0 else 2,
            'DatumUmrti': (C['start'] + pd.Timedelta(days=int(death_day))).strftime('%Y-%m-%d') if not np.isnan(death_day) else '',
            'Datum_1': (C['start'] + pd.Timedelta(days=int(v_day))).strftime('%Y-%m-%d') if not np.isnan(v_day) else ''
        })

    df = pd.DataFrame(rows)
    df['age'] = (C['ref'] - pd.to_numeric(df['Rok_narozeni'], errors='coerce')).astype(int)
    df['v_t'] = (pd.to_datetime(df['Datum_1'], errors='coerce') - C['start']).dt.days
    df['d_t'] = (pd.to_datetime(df['DatumUmrti'], errors='coerce') - C['start']).dt.days
    df['sex'] = df['Pohlavikod'].map({1:0,2:1}).fillna(0).astype(int)
    return df

# ============================================================
# spline helpers
# ============================================================
def make_calendar_spline(max_t, extra_tau=C['tau']):
    max_idx = max_t + extra_tau + max(abs(l) for l in C['lags'])
    t_range = np.arange(0, max_idx + 1)
    s_norm = (t_range - t_range.mean()) / (t_range.std() + 1e-10)
    spl = BSplines(s_norm[:, None], df=[C['spline_df'] + 3], degree=[C['spline_deg']])
    spl_o = pd.DataFrame(spl.basis.astype(np.float32), index=t_range)
    return spl_o

def make_relative_spline(tau, buffer=30):
    """
    Create relative-time spline basis covering 0 to tau + buffer days.
    Buffer ensures no KeyError even if day_rel reaches tau (full follow-up day).
    """
    max_rel = tau + buffer
    r = np.arange(0, max_rel + 1)  # inclusive: 0 .. tau+buffer
    s_norm = (r - r.mean()) / (r.std() + 1e-10)
    spl = BSplines(s_norm[:, None], df=[C['spline_df'] + 3], degree=[C['spline_deg']])
    spl_r = pd.DataFrame(spl.basis.astype(np.float32), index=r)
    return spl_r

# ============================================================
# clone-censor and IPCW pipeline
# ============================================================
def construct_clones(df_base, t0, tau, lag):
    """
    Construct clones representing two static strategies over [t0, t0+tau]:

      Strategy A (clone='A'):
        - Vaccinate exactly at day v_idx = t0 + lag, if alive and uncensored.
        - Non-adherence:
            * Vaccinate at any day != v_idx within [t0, t0+tau].
            * Fail to vaccinate by v_idx (i.e., still unvaccinated at v_idx).
        - Non-adherence is handled by censoring at the time of deviation.

      Strategy B (clone='B'):
        - Never vaccinate during [t0, t0+tau].
        - Non-adherence: any vaccination in [t0, t0+tau]; censor at that day.

    Assumption for this setup:
      - All individuals are unvaccinated at t0 (enforced by eligibility criteria).
      - lag >= 0, so v_idx >= t0 (no unattainable "vaccinated before t0" regime).
    """
    v_idx = t0 + lag

    if v_idx < t0 or v_idx > t0 + tau or v_idx > C['max_t']:
        # Strategy A not well-defined or outside follow-up window for this t0,lag
        return pd.DataFrame()

    # Eligibility: alive and unvaccinated at t0
    elig = df_base[(df_base['d_t'].isna() | (df_base['d_t'] >= t0)) &
                   (df_base['v_t'].isna() | (df_base['v_t'] >= t0))].copy()
    if elig.empty:
        return pd.DataFrame()

    max_day = min(C['max_t'], t0 + tau)
    clones = []

    for idx, r in elig.iterrows():
        vacc_obs = int(r['v_t']) if not pd.isna(r['v_t']) and r['v_t'] <= C['max_t'] else np.nan
        death_obs = int(r['d_t']) if not pd.isna(r['d_t']) and r['d_t'] <= C['max_t'] else np.nan

        for clone in ['A','B']:
            for d_rel in range(0, (max_day - t0) + 1):
                gday = t0 + d_rel

                # event indicator
                event = 1.0 if (not np.isnan(death_obs) and gday == death_obs) else 0.0

                # observed vaccination status
                vacc_status_obs = 1.0 if (not np.isnan(vacc_obs) and vacc_obs <= gday) else 0.0
                A_day = 1.0 if (not np.isnan(vacc_obs) and vacc_obs == gday) else 0.0

                # clone-specific censoring
                C_day = 0.0
                if clone == 'A':
                    # Strategy A: vaccinate exactly at v_idx
                    if not np.isnan(vacc_obs):
                        # early or late vaccination
                        if vacc_obs != v_idx and gday == vacc_obs and vacc_obs >= t0 and vacc_obs <= t0 + tau:
                            C_day = 1.0
                    else:
                        # never vaccinated: non-adherence at v_idx
                        if gday == v_idx:
                            C_day = 1.0
                else:
                    # Strategy B: never vaccinate in [t0, t0+tau]
                    if not np.isnan(vacc_obs) and vacc_obs >= t0 and vacc_obs <= t0 + tau and gday == vacc_obs:
                        C_day = 1.0

                clones.append({
                    'id': idx,
                    'clone': clone,
                    'day_rel': float(d_rel),
                    'global_day': int(gday),
                    'event': float(event),
                    'vacc_status_obs': float(vacc_status_obs),
                    'A_day': float(A_day),
                    'C_day': float(C_day),
                    'age': int(r['age']),
                    'sex': int(r['sex'])
                })

    return pd.DataFrame(clones)

# ============================================================
# treatment & censoring models
# ============================================================
def fit_models_per_clone(df_clone, spl_cal, spl_rel, covariates):
    df = df_clone.copy()

    df['A_day'] = df['A_day'].astype(float)
    df['C_day'] = df['C_day'].astype(float)
    df['A_lag1'] = df.groupby('id')['A_day'].shift(1).fillna(0.0).astype(float)

    days = df['global_day'].astype(int).values
    X_cal = spl_cal.loc[days].reset_index(drop=True)
    days_rel = df['day_rel'].astype(int).values
    X_rel = spl_rel.loc[days_rel].reset_index(drop=True)
    X_base = df[covariates].reset_index(drop=True)

    # Treatment models
    X_den_t = pd.concat([sm.add_constant(X_cal), X_rel, X_base, df[['A_lag1']].reset_index(drop=True)], axis=1)
    y_t = df['A_day'].astype(float).values
    try:
        m_den_t = sm.GLM(y_t, X_den_t, family=sm.families.Binomial()).fit()
    except Exception:
        m_den_t = sm.GLM(y_t, X_den_t, family=sm.families.Binomial()).fit_regularized(alpha=C['alpha'], L1_wt=0)
    p_den_t = m_den_t.predict(X_den_t)

    X_num_t = pd.concat([sm.add_constant(X_cal), X_rel, X_base], axis=1)
    try:
        m_num_t = sm.GLM(y_t, X_num_t, family=sm.families.Binomial()).fit()
    except Exception:
        m_num_t = sm.GLM(y_t, X_num_t, family=sm.families.Binomial()).fit_regularized(alpha=C['alpha'], L1_wt=0)
    p_num_t = m_num_t.predict(X_num_t)

    # Censoring models
    y_c = df['C_day'].astype(float).values
    X_den_c = X_den_t.copy()
    try:
        m_den_c = sm.GLM(y_c, X_den_c, family=sm.families.Binomial()).fit()
    except Exception:
        m_den_c = sm.GLM(y_c, X_den_c, family=sm.families.Binomial()).fit_regularized(alpha=C['alpha'], L1_wt=0)
    p_den_c = m_den_c.predict(X_den_c)

    X_num_c = X_num_t.copy()
    try:
        m_num_c = sm.GLM(y_c, X_num_c, family=sm.families.Binomial()).fit()
    except Exception:
        m_num_c = sm.GLM(y_c, X_num_c, family=sm.families.Binomial()).fit_regularized(alpha=C['alpha'], L1_wt=0)
    p_num_c = m_num_c.predict(X_num_c)

    eps = 1e-6
    df['p_den_t'] = np.clip(p_den_t, eps, 1 - eps)
    df['p_num_t'] = np.clip(p_num_t, eps, 1 - eps)
    df['p_den_c'] = np.clip(p_den_c, eps, 1 - eps)
    df['p_num_c'] = np.clip(p_num_c, eps, 1 - eps)
    return df, (m_den_t, m_num_t, m_den_c, m_num_c)

# ============================================================
# stabilized weights
# ============================================================
def compute_sw(df_clone):
    df = df_clone.copy()

    treat_factor = np.where(df['A_day'] == 1.0,
                            df['p_num_t'] / df['p_den_t'],
                            (1.0 - df['p_num_t']) / (1.0 - df['p_den_t']))
    cens_factor = np.where(df['C_day'] == 1.0,
                           df['p_num_c'] / df['p_den_c'],
                           (1.0 - df['p_num_c']) / (1.0 - df['p_den_c']))

    factor = np.clip(treat_factor * cens_factor, 1e-8, 1e8)
    df['factor'] = factor
    df['sw_raw'] = df.groupby('id')['factor'].cumprod()

    vals = df['sw_raw'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        df['sw'] = df['sw_raw']
        return df

    max_w = float(np.nanpercentile(vals, C['weight_trunc_pct']))
    df['sw'] = np.clip(df['sw_raw'], 1e-6, max_w)
    return df

# ============================================================
# outcome model (single pooled logistic across clones)
# ============================================================
def fit_pooled_logistic(df_all, spl_cal, spl_rel, covariates, tau):
    """
    Fit a single weighted pooled logistic model across both clones,
    as in a standard g-method setup. Clones differ in censoring & weights,
    not in the outcome surface itself.
    """
    df = df_all.copy()
    df = df[df['day_rel'] < tau].copy()
    if df.empty:
        return None, None

    days_cal = df['global_day'].astype(int).values
    X_cal = spl_cal.loc[days_cal].reset_index(drop=True)
    days_rel = df['day_rel'].astype(int).values
    X_rel = spl_rel.loc[days_rel].reset_index(drop=True)
    X_cov = df[['vacc_status_obs'] + covariates].reset_index(drop=True)

    X = pd.concat([sm.add_constant(X_cal), X_rel, X_cov], axis=1)
    y = df['event'].astype(float).values
    weights = df['sw'].astype(float).values

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial(sm.families.links.CLogLog()))
        res = model.fit(freq_weights=weights)
    except Exception:
        res = model.fit_regularized(alpha=C['alpha'], L1_wt=0)

    return res, X.columns

# ============================================================
# RMST prediction under a strategy
# ============================================================
def build_vaccination_path(days_rel, tau, lag, strategy):
    """
    Build a time-varying vaccination path for days_rel=0..tau,
    consistent with the strategy definition.

    Strategy A: vaccinate exactly at relative day = lag (since v_idx = t0 + lag).
    Strategy B: never vaccinate during [0, tau].
    """
    if strategy == 'B':
        return np.zeros_like(days_rel, dtype=float)

    # Strategy A
    vacc_path = np.zeros_like(days_rel, dtype=float)
    # vaccinate at day_rel == lag
    vacc_path[days_rel >= lag] = 1.0
    return vacc_path

def predict_rmst(res, X_cols, spl_cal, spl_rel, tau, mean_cov_row, t0, lag, strategy):
    """
    Predict RMST over days 0..tau starting from t0 under a given strategy:
      - strategy='A': vaccinate at t0+lag
      - strategy='B': never vaccinate during follow-up
    """
    days_rel = np.arange(0, tau + 1)
    days_cal = t0 + days_rel

    X_cal = spl_cal.loc[days_cal].reset_index(drop=True)
    X_rel = spl_rel.loc[days_rel].reset_index(drop=True)

    vacc_path = build_vaccination_path(days_rel, tau, lag, strategy)

    cov_df = pd.DataFrame({
        'vacc_status_obs': vacc_path,
        'age': mean_cov_row['age'],
        'sex': mean_cov_row['sex']
    })

    X_pred = pd.concat([sm.add_constant(X_cal), X_rel, cov_df.reset_index(drop=True)], axis=1)

    # Align columns
    for c in X_cols:
        if c not in X_pred.columns:
            X_pred[c] = 0.0
    X_pred = X_pred[X_cols]

    h = res.predict(X_pred)
    h = np.clip(h, 1e-12, 1 - 1e-12)

    S = np.cumprod(1 - h)
    rmst = float(np.sum(S))
    return rmst

# ============================================================
# estimate delta for t0 and lag
# ============================================================
def estimate_delta_t0_lag(df_base, t0, lag, tau, spl_cal, spl_rel, covariates):
    clones = construct_clones(df_base, t0, tau, lag)
    if clones.empty:
        return None

    dfs = []
    for clone in ['A','B']:
        sub = clones[clones['clone']==clone].copy()
        if sub.empty:
            return None
        df_models, _ = fit_models_per_clone(sub, spl_cal, spl_rel, covariates)
        df_sw = compute_sw(df_models)
        dfs.append(df_sw)

    df_all = pd.concat(dfs, ignore_index=True)

    # diagnostics
    # use final weights aggregated across clones to compute ESS
    last_w = df_all.groupby(['id','clone'])['sw'].last().unstack(fill_value=np.nan).mean(axis=1).dropna().values
    ess = (np.sum(last_w)**2)/np.sum(last_w**2) if len(last_w)>0 else np.nan
    pos_low = float((df_all['p_den_t'] < 0.01).mean())
    pos_high = float((df_all['p_den_t'] > 0.99).mean())

    # SMDs at baseline (day_rel==0)
    base = df_all[df_all['day_rel']==0]
    smd_age = np.nan; smd_sex = np.nan
    if not base.empty:
        a = base[base['clone']=='A']; b = base[base['clone']=='B']
        if len(a)>0 and len(b)>0:
            wa = a['sw']; wb = b['sw']

            # age
            xa = a['age']; xb = b['age']
            ma = np.average(xa, weights=wa) if wa.sum()>0 else np.nan
            mb = np.average(xb, weights=wb) if wb.sum()>0 else np.nan
            pooled_sd = math.sqrt(((xa.var(ddof=1) if len(xa)>1 else 0) +
                                   (xb.var(ddof=1) if len(xb)>1 else 0))/2) if (len(xa)>1 or len(xb)>1) else 0.0
            smd_age = (ma-mb)/pooled_sd if pooled_sd>0 else 0.0

            # sex
            xa = a['sex']; xb = b['sex']
            ma = np.average(xa, weights=wa) if wa.sum()>0 else np.nan
            mb = np.average(xb, weights=wb) if wb.sum()>0 else np.nan
            pooled_sd = math.sqrt(((xa.var(ddof=1) if len(xa)>1 else 0) +
                                   (xb.var(ddof=1) if len(xb)>1 else 0))/2) if (len(xa)>1 or len(xb)>1) else 0.0
            smd_sex = (ma-mb)/pooled_sd if pooled_sd>0 else 0.0

    # Fit a single pooled logistic outcome model across clones
    if df_all['id'].nunique() < C['min_group']:
        return None

    res, X_cols = fit_pooled_logistic(df_all, spl_cal, spl_rel, covariates, tau)
    if res is None:
        return None

    # Use a common baseline covariate profile (e.g., mean age/sex among eligibles at t0)
    mean_age = float(df_all[df_all['day_rel']==0]['age'].mean())
    mean_sex = float(df_all[df_all['day_rel']==0]['sex'].mean())
    mean_cov_row = {'age': mean_age, 'sex': mean_sex}

    rmst_A = predict_rmst(res, X_cols, spl_cal, spl_rel, tau,
                          mean_cov_row, t0=t0, lag=lag, strategy='A')
    rmst_B = predict_rmst(res, X_cols, spl_cal, spl_rel, tau,
                          mean_cov_row, t0=t0, lag=lag, strategy='B')

    delta = rmst_A - rmst_B
    n_eligible = int(df_all[df_all['day_rel']==0]['id'].nunique())

    return {'t0': int(t0), 'lag': int(lag), 'delta': float(delta),
            'rmst_A': float(rmst_A), 'rmst_B': float(rmst_B),
            'n_eligible': n_eligible, 'ess': float(ess),
            'smd_age': float(smd_age), 'smd_sex': float(smd_sex),
            'pos_low': pos_low, 'pos_high': pos_high}

# ============================================================
# placebo sampling (simulate vaccination)
# ============================================================
def placebo_delta_t0_lag(df_base, t0, lag, tau, spl_cal, spl_rel, covariates, rng):
    """
    Placebo: simulate vaccination times from the numerator treatment model,
    overwrite v_t, and re-run the full estimation pipeline.
    This is a falsification device, not a canonical g-method.
    """
    clones = construct_clones(df_base, t0, tau, lag)
    if clones.empty:
        return None

    sim_vacc = {}
    for clone in ['A','B']:
        sub = clones[clones['clone']==clone].copy()
        if sub.empty:
            return None

        df_models, _ = fit_models_per_clone(sub, spl_cal, spl_rel, covariates)
        p = df_models['p_num_t'].values
        A_sim = rng.binomial(1, p)
        df_models['A_sim'] = A_sim

        first = df_models[df_models['A_sim']==1].groupby('id')['global_day'].min().to_dict()
        for k,v in first.items():
            if k not in sim_vacc:
                sim_vacc[k] = v

    df_sim = df_base.copy()
    df_sim['v_t_sim'] = df_sim.index.map(lambda i: sim_vacc.get(i, np.nan))
    df_sim2 = df_sim.copy()
    df_sim2['v_t'] = df_sim2['v_t_sim']

    return estimate_delta_t0_lag(df_sim2, t0, lag, tau, spl_cal, spl_rel, covariates)

# ============================================================
# bootstrap wrapper (placebo inside each replicate)
# ============================================================
def bootstrap_age(df_age, scenario_name):
    rng_master = np.random.default_rng(C['seed'])

    spl_cal = make_calendar_spline(C['max_t'])
    spl_rel = make_relative_spline(C['tau'], buffer=30)
    covariates = ['age','sex']

    vt = df_age['v_t'].dropna()
    if len(vt) < 10:
        t0_grid = [60, 90, 120, 150, 180]
    else:
        qs = np.quantile(vt, C['t0_quantiles'])
        t0_grid = [int(max(30, min(C['max_t']-C['tau'], int(q)))) for q in qs]

    logging.info(f"{scenario_name} t0_grid: {t0_grid}")

    seeds = [int(C['seed'] + i) for i in range(C['n_boot'])]

    def one_boot(bseed):
        rng = np.random.default_rng(int(bseed))
        ids = df_age.index.to_numpy()
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        df_boot = df_age.loc[sampled_ids].reset_index(drop=True)

        out_per_lag = []
        for lag in C['lags']:
            t0_vals = []
            t0_weights = []
            t0_placebos = []

            for t0 in t0_grid:
                res = estimate_delta_t0_lag(df_boot, t0, lag, C['tau'], spl_cal, spl_rel, covariates)
                if res is not None:
                    t0_vals.append(res['delta'])
                    t0_weights.append(res['n_eligible'])

                    p = placebo_delta_t0_lag(df_boot, t0, lag, C['tau'], spl_cal, spl_rel, covariates, rng)
                    if p is not None:
                        t0_placebos.append(p['delta'])

            if not t0_vals:
                continue

            w = np.array(t0_weights, dtype=float)
            vals = np.array(t0_vals, dtype=float)
            wsum = w.sum()
            mean_delta = float(np.sum(vals * w) / wsum) if wsum>0 else float(np.mean(vals))
            placebo_mean = float(np.mean(t0_placebos)) if t0_placebos else 0.0

            out_per_lag.append({'lag': lag,
                                'delta_mean': mean_delta,
                                'placebo_mean': placebo_mean})
        return out_per_lag

    parallel = Parallel(n_jobs=C['n_jobs'])
    boot_results = parallel(delayed(one_boot)(s) for s in seeds)

    from collections import defaultdict
    agg = defaultdict(list)
    for br in boot_results:
        for item in br:
            agg[item['lag']].append((item['delta_mean'], item['placebo_mean']))

    out = []
    for lag, vals in agg.items():
        deltas = np.array([v[0] for v in vals], dtype=float)
        placebos = np.array([v[1] for v in vals], dtype=float)
        out.append({'scenario': scenario_name, 'lag': int(lag),
                    'delta_mean': float(np.mean(deltas)),
                    'delta_lo': float(np.percentile(deltas, 2.5)),
                    'delta_hi': float(np.percentile(deltas, 97.5)),
                    'placebo_mean': float(np.mean(placebos)),
                    'placebo_lo': float(np.percentile(placebos,2.5)),
                    'placebo_hi': float(np.percentile(placebos,97.5)),
                    'n_boot': len(deltas)})
    return out

# ============================================================
# run validation for both scenarios
# ============================================================
def run_validation():
    logging.info("Generating synthetic data (A null, B protective)")
    dfA = generate_synthetic(C['N'], scenario='A')
    dfB = generate_synthetic(C['N'], scenario='B')

    dfA_age = dfA[dfA['age']==60].reset_index(drop=True)
    dfB_age = dfB[dfB['age']==60].reset_index(drop=True)
    logging.info(f"Scenario A N_age60={len(dfA_age)}; Scenario B N_age60={len(dfB_age)}")

    resA = bootstrap_age(dfA_age, 'A_null')
    resB = bootstrap_age(dfB_age, 'B_protective')

    dfA_out = pd.DataFrame(resA)
    dfB_out = pd.DataFrame(resB)

    dfA_out.to_csv(C['out_dir'] / "MASTER_SUMMARY_A.csv", index=False)
    dfB_out.to_csv(C['out_dir'] / "MASTER_SUMMARY_B.csv", index=False)

    logging.info("Validation complete. CSVs saved.")
    print("\n=== Validation Summary (Scenario A: Null) ===")
    if not dfA_out.empty:
        print(dfA_out.to_string(index=False))
    else:
        print("No results for Scenario A")

    print("\n=== Validation Summary (Scenario B: Protective) ===")
    if not dfB_out.empty:
        print(dfB_out.to_string(index=False))
    else:
        print("No results for Scenario B")

    print("\nOutputs saved to:", C['out_dir'].resolve())

if __name__ == "__main__":
    start = time.time()
    run_validation()
    logging.info(f"Elapsed {(time.time()-start)/60:.1f} minutes")
