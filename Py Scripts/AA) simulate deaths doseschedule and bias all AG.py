#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import CoxTimeVaryingFitter

"""
Title: Simulation of COVID-19 Vaccine Doses and Deaths with Bias Assessment
Author: [drifting]
Date: 2025-12
Version: 1.1

Description
-----------
This script simulates individual-level COVID-19 vaccination and death data 
based on real vaccination sequences, to evaluate potential biases in hazard 
ratio estimation methodes. The workflow generates synthetic death events while preserving 
observed vaccination schedules and allows comparison between simulated and 
real dose patterns.

Methods
-------
1. Loads real vaccination and death dates for each age group.
2. Estimates crude death rates for simulation.
3. Simulates death days under HR=1 assumption using a Bernoulli process.
4. Assigns vaccination doses from real sequences while censoring after death.
5. Aggregates daily dose counts and compares simulated vs real distributions.
6. Generates inter-dose interval histograms for simulated vs real data.
7. Validates Cox proportional hazards assumptions using minimal HR checks.

Outputs
-------
- Simulated individual-level CSVs per age group.
- Daily dose counts CSVs and plots.
- Inter-dose interval histograms.
- Validation of HR estimates for vaccinated vs unvaccinated.

Notes
-----
- Assumes START_DATE = 2020-01-01 as reference.
- Ages processed: 0–113 (AG0 to AG113).
- Uses fixed random seed for reproducibility.
- Designed for exploratory analysis and bias evaluation, not causal inference.
"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FOLDER = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
OUTPUT_FOLDER = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
PLOT_OUTPUT_FOLDER = r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AA) simulate deaths doseschedule and bias"

START_DATE = pd.Timestamp('2020-01-01')
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]  # columns for vaccine doses
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS
BASE_RNG_SEED = 42
# AGES = range(0, 114)  # AG0 to AG113
AGES = [70]
np.random.seed(BASE_RNG_SEED)

# Create folder if it doesn't exist
os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def parse_dates(df):
    """Convert all relevant columns to datetime format."""
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def estimate_death_rate(df):
    """Estimate crude death rate from observed deaths, clipped to avoid extremes."""
    return float(np.clip(df['DatumUmrti'].notna().sum() / len(df), 1e-4, 0.999))

def simulate_deaths_hr1(n, end_measure, death_rate):
    """Simulate death days using Bernoulli sampling under HR=1."""
    rng = np.random.default_rng(BASE_RNG_SEED)
    will_die = rng.random(n) < death_rate
    death_days = np.full(n, np.nan)
    death_days[will_die] = rng.integers(0, end_measure + 1, size=will_die.sum())
    return death_days

def per_day_counts(series_dates, end_measure=None):
    """Count number of events per day up to optional end_measure."""
    days = (series_dates - START_DATE).dt.days.dropna().astype(int)
    if end_measure is not None:
        days = days[days <= end_measure]
    return days.value_counts().sort_index()

def assign_doses_from_sequences(df_real, death_days, START_DATE, END_MEASURE):
    """
    Assign doses from real sequences while censoring after simulated death.
    Returns simulated DataFrame and per-dose matching summary.
    """
    df_sim = pd.DataFrame()
    df_sim['Rok_narozeni'] = df_real['Rok_narozeni'].reset_index(drop=True)
    for col in DOSE_DATE_COLS:
        df_sim[col] = pd.NaT
    seq_library = df_real[DOSE_DATE_COLS].to_dict(orient='records')
    for idx in range(len(df_sim)):
        seq = seq_library[idx]
        for col in DOSE_DATE_COLS:
            dt = seq[col]
            if pd.isna(dt):
                continue
            day_int = (dt - START_DATE).days
            if np.isnan(death_days[idx]) or death_days[idx] > day_int:
                df_sim.at[idx, col] = dt
    df_sim['death_day'] = death_days
    df_sim['DatumUmrti'] = pd.NaT
    died_mask = ~np.isnan(death_days)
    if died_mask.any():
        df_sim.loc[died_mask, 'DatumUmrti'] = START_DATE + pd.to_timedelta(
            df_sim.loc[died_mask, 'death_day'].astype(int), unit='D'
        )
    sim_totals = df_sim[DOSE_DATE_COLS].notna().sum().to_dict()
    real_totals = df_real[DOSE_DATE_COLS].notna().sum().to_dict()
    match_rows = []
    for col in DOSE_DATE_COLS:
        match_rows.append({
            "dose": col,
            "real": int(real_totals.get(col, 0)),
            "sim": int(sim_totals.get(col, 0)),
            "shortfall": max(0, int(real_totals.get(col, 0)) - int(sim_totals.get(col, 0)))
        })
    match_df = pd.DataFrame(match_rows)
    return df_sim, match_df

def save_daily_dose_counts(df_sim, df_real, DOSE_DATE_COLS, END_MEASURE, output_csv):
    """Aggregate daily dose counts and save as CSV."""
    all_days = range(0, END_MEASURE + 1)
    rows = []
    for col in DOSE_DATE_COLS:
        sim_counts = per_day_counts(df_sim[col], end_measure=END_MEASURE)
        real_counts = per_day_counts(df_real[col], end_measure=END_MEASURE)
        for day in all_days:
            rows.append({
                "day": day,
                "dose": col,
                "sim": int(sim_counts.get(day, 0)),
                "real": int(real_counts.get(day, 0))
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Daily dose counts saved: {output_csv}")

def plot_daily_doses(df_sim, df_real, DOSE_DATE_COLS, END_MEASURE, output_html):
    """Plot daily counts for simulated vs real doses."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for col in DOSE_DATE_COLS:
        sim_counts = per_day_counts(df_sim[col], end_measure=END_MEASURE)
        real_counts = per_day_counts(df_real[col], end_measure=END_MEASURE)
        fig.add_trace(go.Scatter(x=sim_counts.index, y=sim_counts.values,
                                 mode='lines', name=f"Sim {col}"))
        fig.add_trace(go.Scatter(x=real_counts.index, y=real_counts.values,
                                 mode='lines', name=f"Real {col}", line=dict(dash='dot')))
    fig.update_layout(title=f"Daily Dose Counts: Real vs Simulated",
                      xaxis_title="Days since 2020-01-01",
                      yaxis_title="Number of doses",
                      template="plotly_white", height=600)
    fig.write_html(output_html)
    print(f"Daily dose plot saved: {output_html}")

def plot_intervals_comparison(df_sim, df_real, DOSE_DATE_COLS, output_html):
    """Plot histograms of inter-dose intervals comparing simulated vs real."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for i in range(len(DOSE_DATE_COLS)-1):
        dose1, dose2 = DOSE_DATE_COLS[i], DOSE_DATE_COLS[i+1]
        mask_real = df_real[dose1].notna() & df_real[dose2].notna()
        mask_sim = df_sim[dose1].notna() & df_sim[dose2].notna()
        days_real = (df_real.loc[mask_real, dose2] - df_real.loc[mask_real, dose1]).dt.days
        days_sim = (df_sim.loc[mask_sim, dose2] - df_sim.loc[mask_sim, dose1]).dt.days
        fig.add_trace(go.Histogram(x=days_real, name=f"Real {dose2}-{dose1}", opacity=0.5))
        fig.add_trace(go.Histogram(x=days_sim, name=f"Sim {dose2}-{dose1}", opacity=0.5))
    fig.update_layout(barmode='overlay',
                      title="Inter-dose Interval Histograms: Real vs Simulated",
                      xaxis_title="Days between consecutive doses",
                      yaxis_title="Count",
                      template="plotly_white", height=600)
    fig.write_html(output_html)
    print(f"Inter-dose interval comparison saved: {output_html}")

def validate_hr_is_one(df_sim, min_vax_count=5, min_event_count=1):
    """
    Fit a minimal Cox proportional hazards model to validate HR ≈ 1.
    Skips fitting if insufficient vaccinated subjects or events.
    """
    df_sim = df_sim.copy()
    df_sim['first_dose_day'] = df_sim[[c for c in DOSE_DATE_COLS]].apply(
        lambda r: (r - START_DATE).dt.days.min(), axis=1
    )
    df_sim['death_day'] = df_sim['death_day'].fillna(df_sim['death_day'].max())
    df_sim['t0'] = 0
    df_sim['t1'] = df_sim['death_day']
    df_sim['event'] = np.where(df_sim['DatumUmrti'].notna(), 1, 0)
    df_sim['vaccinated'] = np.where(df_sim['first_dose_day'].notna(), 1, 0)

    n_vax = df_sim['vaccinated'].sum()
    n_events = df_sim['event'].sum()
    if n_vax < min_vax_count or n_events < min_event_count:
        print(f"[VALIDATION] Skipping Cox model: only {n_vax} vaccinated and {n_events} events.")
        return

    tv = df_sim[['Rok_narozeni', 't0', 't1', 'event', 'vaccinated']].rename(columns={'Rok_narozeni': 'id'})
    tv['id'] = tv['id'].astype(str)
    mask_zero_length = (tv['t0'] == tv['t1']) & (tv['event'] == 1)
    if mask_zero_length.any():
        tv.loc[mask_zero_length, 't1'] += 0.5
        print(f"[VALIDATION] Added 0.5 days to {mask_zero_length.sum()} zero-length intervals for Cox model.")
    cox = CoxTimeVaryingFitter()
    try:
        cox.fit(tv, id_col='id', start_col='t0', stop_col='t1', event_col='event')
        hr = np.exp(cox.params_['vaccinated'])
        print(f"[VALIDATION] Estimated HR (vaccinated vs unvaccinated): {hr:.3f}")
    except Exception as e:
        print(f"[VALIDATION] Cox model failed: {e}")

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def run_sim_bias_all_ages():
    """Main loop: simulate deaths and doses for all age groups."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOT_OUTPUT_FOLDER, exist_ok=True)

    for age in AGES:
        input_csv = os.path.join(INPUT_FOLDER, f"Vesely_106_202403141131_AG{age}.csv")
        if not os.path.exists(input_csv):
            print(f"[WARNING] Input file not found for AG{age}: {input_csv}")
            continue
        print(f"\n=== Processing {input_csv} ===")

        df = pd.read_csv(input_csv, usecols=NEEDED_COLS, dtype=str)
        df = parse_dates(df)

        if df['DatumUmrti'].notna().any():
            END_MEASURE = int(((df['DatumUmrti'] - START_DATE).dt.days).max(skipna=True))
        else:
            END_MEASURE = 1533  # default window if no deaths

        death_rate = estimate_death_rate(df)
        n = len(df)
        death_days = simulate_deaths_hr1(n, END_MEASURE, death_rate)

        df_sim, perday_match_df = assign_doses_from_sequences(df, death_days, START_DATE, END_MEASURE)

        # Output filenames per age group
        sim_output_csv = os.path.join(OUTPUT_FOLDER, f"AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{age}.csv")
        perday_match_csv = os.path.join(PLOT_OUTPUT_FOLDER, f"PerDayMatch_AG{age}.csv")
        daily_dose_csv = os.path.join(PLOT_OUTPUT_FOLDER, f"DailyDose_AG{age}.csv")
        plot_html = os.path.join(PLOT_OUTPUT_FOLDER, f"DailyDose_AG{age}.html")
        interval_html = os.path.join(PLOT_OUTPUT_FOLDER, f"IntervalHist_AG{age}.html")

        final_cols = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS + ['death_day']
        df_sim[final_cols].to_csv(sim_output_csv, index=False)
        perday_match_df.to_csv(perday_match_csv, index=False)
        save_daily_dose_counts(df_sim, df, DOSE_DATE_COLS, END_MEASURE, daily_dose_csv)
        plot_daily_doses(df_sim, df, DOSE_DATE_COLS, END_MEASURE, plot_html)
        plot_intervals_comparison(df_sim, df, DOSE_DATE_COLS, interval_html)
        validate_hr_is_one(df_sim)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_sim_bias_all_ages()
