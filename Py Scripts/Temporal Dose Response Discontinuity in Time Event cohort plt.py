#!/usr/bin/env python3
"""
TDR-DiT: Descriptive Event-Time Hazard Study Around Vaccination Eligibility Dates
Final version: cohort-stratified plots, fixed length mismatch via merge, programmatic reference bin, descriptive framing

Supports staggered eligibility across age cohorts (Czech rollout phases).
Descriptive only – not causal inference on vaccine effects.

Author: Grok (finalized 2026-02-05)
Version: 1.9 – Cohort-stratified, GitHub-ready
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
import os
import gc
import logging
import json
import sys
import platform
from datetime import datetime
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson
import ctypes

# Sleep prevention (Windows)
if os.name == 'nt':
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print(">>> Windows Sleep Prevention: ACTIVE")
    except Exception as e:
        print(f">>> Could not set Sleep Prevention: {e}")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("tdr_dit_final")
warnings.filterwarnings("ignore")

# Configuration – full population (0–112)
@dataclass
class Config:
    input_path: Path = Path(r"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131.csv")
    out_dir: Path = Path(r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\TDR_DiT_COHORT")
    study_start: pd.Timestamp = pd.Timestamp("2020-01-01")
    fixed_study_end: pd.Timestamp = pd.Timestamp("2023-12-31")
    use_dynamic_study_end: bool = True
    study_end_buffer_days: int = 90
    quick_test: bool = False
    sample_frac: float = 1.0
    lag_min: int = -60
    lag_max: int = 120
    age_bins: tuple = (0, 113)  # Full population
    bin_size_days: int = 7      # Weekly bins
    ref_bin_days: int = -14     # Target reference bin

CFG = Config()
CFG.out_dir.mkdir(parents=True, exist_ok=True)

# Cohort-specific eligibility dates (Czech rollout phases)
COHORT_ELIGIBILITY = {
    (80, 120): pd.Timestamp("2021-01-15"),  # 80+
    (70, 80): pd.Timestamp("2021-03-01"),   # 70–79
    (60, 70): pd.Timestamp("2021-05-01"),   # 60–69
    (18, 60): pd.Timestamp("2021-06-01"),   # 18–59
    (12, 18): pd.Timestamp("2021-09-01"),   # Adolescents
    (5, 12): pd.Timestamp("2021-12-01"),    # Children
    (0, 5): pd.Timestamp("2022-01-01"),     # Young children (low/no early vax)
}

# Utilities
def write_run_info(cfg: Config, row_count: int, cohort_label: str = ""):
    info = {
        "run_started": datetime.now().isoformat(),
        "input_path": str(cfg.input_path),
        "rows": int(row_count),
        "age_bins": f"{cfg.age_bins[0]}-{cfg.age_bins[1]}",
        "cohort": cohort_label,
        "lag_range": f"{cfg.lag_min}..{cfg.lag_max}",
        "bin_size_days": cfg.bin_size_days,
        "notes": [
            "Descriptive event-time hazard analysis around cohort-specific vaccination eligibility dates.",
            "Not causal vaccination effect – eligibility ≠ treatment.",
            "Risk sets approximate survivors to bin start.",
            "No explicit counterfactual after eligibility in single cohort; staggered across age groups.",
            "Pre-trends tested via joint F-test on leads."
        ],
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        }
    }
    with open(cfg.out_dir / f"run_info_{cohort_label}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    log.info(f"Run info saved for {cohort_label}")

# Data loading
def load_and_prepare_data(cfg: Config = CFG) -> Tuple[pd.DataFrame, pd.Timestamp]:
    log.info(f"Loading CSV: {cfg.input_path}")
    raw = pd.read_csv(cfg.input_path, dtype=str, low_memory=True)
    raw.columns = raw.columns.str.strip()

    required = ["Datum_1", "DatumUmrti", "Rok_narozeni"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if cfg.quick_test:
        raw = raw.sample(frac=cfg.sample_frac, random_state=12345).reset_index(drop=True)

    # Parse dates
    for c in ["Datum_1", "DatumUmrti"]:
        if c in raw.columns:
            raw[c] = pd.to_datetime(raw[c], errors="coerce").dt.normalize()

    raw["Rok_narozeni"] = pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
    raw["age_at_2021"] = 2021 - raw["Rok_narozeni"]

    # Filter to age bins
    raw = raw[raw["age_at_2021"].between(cfg.age_bins[0], cfg.age_bins[1]-1)].reset_index(drop=True)
    if len(raw) == 0:
        raise ValueError(f"No subjects in age range {cfg.age_bins[0]}-{cfg.age_bins[1]-1}")

    # Assign eligibility based on age
    raw["eligibility_date"] = np.nan
    for (min_age, max_age), elig_date in COHORT_ELIGIBILITY.items():
        mask = raw["age_at_2021"].between(min_age, max_age-1)
        raw.loc[mask, "eligibility_date"] = elig_date

    raw = raw[raw["eligibility_date"].notna()].reset_index(drop=True)

    raw["study_start"] = cfg.study_start.normalize()
    if cfg.use_dynamic_study_end:
        max_death = raw["DatumUmrti"].max()
        study_end = pd.to_datetime(max_death).normalize() + pd.Timedelta(days=cfg.study_end_buffer_days) if pd.notna(max_death) else pd.to_datetime(cfg.fixed_study_end).normalize()
    else:
        study_end = pd.to_datetime(cfg.fixed_study_end).normalize()

    raw["study_end"] = study_end.normalize()

    raw["subject_id"] = np.arange(len(raw))

    # Sex mapping
    sex_col = next((c for c in raw.columns if "pohlav" in c.lower() or c.lower() in ["pohlavi", "sex"]), None)
    if sex_col:
        raw["sex"] = raw[sex_col].astype(str).str.upper().map({'M': 0, 'Z': 1, 'F': 1}).fillna(0).astype(np.int8)
    else:
        raw["sex"] = 0

    # Censor date
    raw["censor_date"] = raw["DatumUmrti"].fillna(raw["study_end"])

    log.info(f"Loaded {len(raw):,} subjects in age range {cfg.age_bins[0]}-{cfg.age_bins[1]-1}")
    return raw, study_end

# TDR-DiT analysis function (per cohort)
def analyze_cohort(df_cohort: pd.DataFrame, cohort_label: str, cfg: Config = CFG):
    log.info(f"Analyzing cohort: {cohort_label}")

    # Ensure eligibility
    df_cohort['eligibility_date'] = pd.to_datetime(df_cohort['eligibility_date'])

    # Compute rel_day per subject
    df_cohort['censor_rel_day'] = (df_cohort['censor_date'] - df_cohort['eligibility_date']).dt.days
    df_cohort['death_rel_day'] = (df_cohort['DatumUmrti'] - df_cohort['eligibility_date']).dt.days

    # Weekly bins
    bins = np.arange(cfg.lag_min, cfg.lag_max + cfg.bin_size_days, cfg.bin_size_days)
    agg_list = []
    for bin_start in tqdm(bins, desc=f"Aggregating {cohort_label}"):
        bin_end = bin_start + cfg.bin_size_days
        n_at_risk = (df_cohort['censor_rel_day'] > bin_start).sum()
        n_deaths = ((df_cohort['death_rel_day'] >= bin_start) & (df_cohort['death_rel_day'] < bin_end)).sum()
        agg_list.append({
            'rel_bin': bin_start,
            'n_at_risk': n_at_risk,
            'n_deaths': n_deaths
        })

    agg = pd.DataFrame(agg_list)

    # Dummies
    dummies = pd.get_dummies(agg['rel_bin'], prefix='lag').astype(float)
    agg = pd.concat([agg, dummies], axis=1)

    # Programmatic reference bin
    pre_bins = agg[agg['rel_bin'] < 0]['rel_bin']
    if not pre_bins.empty:
        ref_bin = pre_bins.max()
        ref_col = f'lag_{ref_bin}'
        if ref_col in agg.columns:
            agg = agg.drop(columns=[ref_col])
        log.info(f"Reference bin for {cohort_label}: {ref_bin} days")
    else:
        log.warning(f"No pre-0 bins for {cohort_label}; no reference dropped")

    # Poisson GLM
    X_cols = [c for c in agg.columns if c.startswith('lag_')]
    X = sm.add_constant(agg[X_cols])
    offset = np.log(agg['n_at_risk'] + 1e-10)
    model = GLM(agg['n_deaths'], X, family=Poisson(), offset=offset).fit()

    log.info(f"Model summary for {cohort_label}:\n{model.summary()}")

    # Pre-trends test
    lead_cols = [c for c in X_cols if int(c.split('_')[-1]) < 0]
    pre_trends_p = np.nan
    if lead_cols:
        wald_test = model.wald_test(" = ".join(lead_cols) + " = 0")
        pre_trends_p = wald_test.pvalue
        log.info(f"Pre-trends joint F-test for {cohort_label}: p = {pre_trends_p:.4f}")

    # Extract coefs/ses from modeled columns
    coefs = model.params[X_cols]
    ses = model.bse[X_cols]
    times = [int(c.split('_')[-1]) for c in X_cols]

    # Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=coefs, mode='markers+lines', name='Coef',
        error_y=dict(type='data', array=1.96*ses, visible=True)
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Eligibility", annotation_position="top")
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"Descriptive Event-Time Hazard Study: {cohort_label}",
        xaxis_title=f"Days from Eligibility ({cfg.bin_size_days}-day bins)",
        yaxis_title="Log Hazard Ratio",
        hovermode="x unified"
    )
    
    fig.write_html(
    cfg.out_dir / f"event_study_{cohort_label}.html",
    config={
        "toImageButtonOptions": {
            "format": "svg",
            "filename": "tdr_dit_event_study",
            "height": 600,
            "width": 900,
            "scale": 1
        }
    })

    # Results DF with merge for alignment
    results_df = pd.DataFrame({
        'rel_bin': times,
        'coef_log_hr': coefs.values,
        'se': ses.values
    })
    results_df['ci_lower'] = results_df['coef_log_hr'] - 1.96 * results_df['se']
    results_df['ci_upper'] = results_df['coef_log_hr'] + 1.96 * results_df['se']

    results_df = results_df.merge(
        agg[['rel_bin', 'n_at_risk', 'n_deaths']],
        on='rel_bin',
        how='left'
    )

    results_df.to_csv(cfg.out_dir / f"results_{cohort_label}.csv", index=False)

    log.info(f"Completed {cohort_label}. Plot: event_study_{cohort_label}.html")

# Main: Stratified loop over cohorts
if __name__ == "__main__":
    log.info("Starting descriptive event-time hazard study")
    df, study_end = load_and_prepare_data(CFG)

    # Loop over cohorts
    for (min_age, max_age), elig_date in COHORT_ELIGIBILITY.items():
        cohort_label = f"{min_age}-{max_age}"
        df_cohort = df[df["age_at_2021"].between(min_age, max_age-1)].copy()
        if len(df_cohort) == 0:
            log.info(f"No subjects in {cohort_label}; skipping")
            continue
        write_run_info(CFG, len(df_cohort), cohort_label)
        analyze_cohort(df_cohort, cohort_label, CFG)

    log.info("All cohorts analyzed. Done.")