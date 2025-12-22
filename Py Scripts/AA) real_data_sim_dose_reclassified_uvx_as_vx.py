"""
Stochastic Sensitivity Analysis for Vaccine Effectiveness
PROJECT: Czech FOI (Freedom of Information) Mortality Data Analysis

PURPOSE:
This script creates a dataset in which a percentage of unvaccinated (UVX) deaths 
are reclassified as vaccinated (VX). It performs a sensitivity analysis to explore 
the impact of this reclassification on estimates of vaccine effectiveness (VE) 
and Restricted Mean Survival Time (RMST).

SCIENTIFIC SPECIFICATIONS:
- Method: Mortality-Conditioned Stochastic Imputation (MCSI)
- Strategy Tracking: Empirical (Local) vs. Fallback assignments
- Exposure Audit: Mean and Quantile (25th, 50th, 75th)
- Cumulative person-time curves normalized for transparency
- Robustness: Deprecation-safe `.astype('int64')` for KS-Divergence

Author: AI / Drifting Date: 12.2025 Version: 1.0
"""

import pandas as pd
import numpy as np
import os
import logging
from scipy.stats import ks_2samp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# RESEARCH CONFIGURATION
# ============================================================

RECLASSIFY_PERCENTAGE = 0.05 
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

INPUT_DIR  = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
OUTPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"

LOG_PATH = (
    r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results"
    r"\AA) real_data_sim_dose_reclassified_uvx_as_vx"
    r"\AA) real_data_sim_dose_reclassified_uvx_as_vx.txt"
)
PLOT_BASE_PATH = LOG_PATH.replace(".txt", ".html")

def logmsg(msg: str):
    """Print and log message."""
    print(msg)
    logging.info(msg)

# ============================================================
# ENHANCED MCSI ALGORITHM WITH STRATEGY QUANTILES
# ============================================================
def reclassify_age_group(age: int):
    ptc_label = f"PTC{int(RECLASSIFY_PERCENTAGE * 100)}"
    input_csv = os.path.join(INPUT_DIR, f"Vesely_106_202403141131_AG{age}.csv")
    output_csv = os.path.join(OUTPUT_DIR, f"AA) real_data_sim_dose_reclassified_{ptc_label}_uvx_as_vx_AG{age}.csv")

    if not os.path.exists(input_csv):
        logmsg(f"File not found for Age {age}. Skipping.")
        return

    # -------------------------------
    # 1. Load Data
    # -------------------------------
    df = pd.read_csv(input_csv, low_memory=False)
    dose_cols = [c for c in df.columns if c.startswith("Datum_")]
    for c in dose_cols + ["DatumUmrti"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    cohort_start = df[dose_cols].min().min()
    
    # -------------------------------
    # 2. Benchmarks (Observed Deaths)
    # -------------------------------
    real_vx_deaths_df = df.dropna(subset=["Datum_1", "DatumUmrti"]).copy()
    real_days_vac = (real_vx_deaths_df["DatumUmrti"] - real_vx_deaths_df["Datum_1"]).dt.days
    
    uvx_death_pool = df[(df["DatumUmrti"] >= cohort_start) & (df["Datum_1"].isna())]
    n_select = max(1, int(RECLASSIFY_PERCENTAGE * len(uvx_death_pool)))

    # -------------------------------
    # 3. Imputation with Strategy Tracking
    # -------------------------------
    all_real_doses = pd.concat([df[c] for c in dose_cols]).dropna()
    selected_indices = np.random.choice(uvx_death_pool.index, size=n_select, replace=False)
    
    assigned_dates, synth_death_dates, strategies = [], [], []
    
    for idx in selected_indices:
        death_date = df.at[idx, "DatumUmrti"]
        window_start = max(death_date - pd.Timedelta(days=120), cohort_start)
        local_pool = all_real_doses[(all_real_doses >= window_start) & (all_real_doses < death_date)]

        if not local_pool.empty:
            dose_date = pd.Timestamp(np.random.choice(local_pool.values))
            strategies.append("Empirical")
        else:
            offset = np.random.randint(30, 91)
            dose_date = max(cohort_start, death_date - pd.Timedelta(days=offset))
            strategies.append("Fallback")

        df.at[idx, "Datum_1"] = dose_date
        assigned_dates.append(dose_date)
        synth_death_dates.append(death_date)

    # -------------------------------
    # 4. Save Modified Dataset
    # -------------------------------
    df.to_csv(output_csv, index=False)
    
    # -------------------------------
    # 5. Exposure Quantiles
    # -------------------------------
    synth_days_vac = (pd.Series(synth_death_dates) - pd.Series(assigned_dates)).dt.days
    r_q = real_days_vac.quantile([0.25, 0.5, 0.75]).to_dict()
    s_q = synth_days_vac.quantile([0.25, 0.5, 0.75]).to_dict()

    # Per-strategy quantiles
    strategy_df = pd.DataFrame({'days': synth_days_vac, 'strategy': strategies})
    strat_quantiles = strategy_df.groupby('strategy')['days'].quantile([0.25,0.5,0.75]).unstack()
    
    # Kolmogorov-Smirnov Divergence
    ks_stat, ks_p = ks_2samp(
        real_vx_deaths_df["Datum_1"].dropna().astype('int64'), 
        pd.Series(assigned_dates).astype('int64')
    )

    # -------------------------------
    # 6. Detailed Scientific Log
    # -------------------------------
    logmsg("="*85)
    logmsg(f"SCIENTIFIC AUDIT: AGE {age} | {ptc_label}")
    logmsg(f"  Population N:         {len(df)}")
    logmsg(f"  Reclassified N:       {n_select} ({n_select/len(df):.2%} of total cohort)")
    logmsg(f"  Assignment Strategy:  {pd.Series(strategies).value_counts().to_dict()}")
    logmsg(f"  Mean Exposure R/S:    {real_days_vac.mean():.1f} / {synth_days_vac.mean():.1f} days")
    logmsg(f"  Quantile R (25/50/75): {r_q[0.25]:.0f} / {r_q[0.5]:.0f} / {r_q[0.75]:.0f} days")
    logmsg(f"  Quantile S (25/50/75): {s_q[0.25]:.0f} / {s_q[0.5]:.0f} / {s_q[0.75]:.0f} days")
    for strat in strat_quantiles.index:
        logmsg(f"  Exposure {strat} (25/50/75): {strat_quantiles.loc[strat, 0.25]:.0f} / {strat_quantiles.loc[strat, 0.5]:.0f} / {strat_quantiles.loc[strat, 0.75]:.0f} days")
    logmsg(f"  Injected Person-Days: {synth_days_vac.sum()} total days")
    logmsg(f"  Divergence (KS-D):    {ks_stat:.4f} (p={ks_p:.2e})")
    logmsg("="*85)

    create_diagnostic_plot(real_vx_deaths_df["Datum_1"], pd.Series(assigned_dates), 
                           real_days_vac, synth_days_vac, age, ptc_label)

# ============================================================
# 7. Diagnostic Plotting with Normalized Cumulative Curves
# ============================================================
def create_diagnostic_plot(real_doses, synth_doses, real_days, synth_days, age, ptc):
    """Generates the dual-pane diagnostic report with cumulative person-time curves."""
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=("Vaccination Rollout (Density)", "Exposure Accumulation")
    )

    # LEFT Subplot: Probability Density & CDF
    fig.add_trace(go.Histogram(x=real_doses, name="Real VX Deaths", marker_color='rgba(100, 149, 237, 0.4)', histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Histogram(x=synth_doses, name="Synth VX Deaths", marker_color='rgba(255, 99, 71, 0.6)', histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.sort(real_doses), y=np.linspace(0, 1, len(real_doses)), name="Real CDF", line=dict(color='blue')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=np.sort(synth_doses), y=np.linspace(0, 1, len(synth_doses)), name="Synth CDF", line=dict(color='red', dash='dot')), row=1, col=1, secondary_y=True)

    # RIGHT Subplot: Boxplots & Normalized Cumulative Exposure
    fig.add_trace(go.Box(y=real_days, name="Real Days", marker_color='blue', boxpoints='outliers'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Box(y=synth_days, name="Synth Days", marker_color='red', boxpoints='outliers'), row=1, col=2, secondary_y=False)
    
    # Cumulative curves normalized
    fig.add_trace(go.Scatter(y=np.sort(real_days).cumsum()/np.sort(real_days).cumsum().max(), 
                             name="Real Cumul. Exposure", line=dict(color='blue', width=1, dash='solid')), row=1, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(y=np.sort(synth_days).cumsum()/np.sort(synth_days).cumsum().max(), 
                             name="Synth Cumul. Exposure", line=dict(color='red', width=1, dash='dash')), row=1, col=2, secondary_y=True)

    fig.update_layout(
        title=f"Distributional Audit: Age {age} ({ptc})",
        barmode='overlay', template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Normalized Cumulative Exposure", row=1, col=2, secondary_y=True)
    
    fig.write_html(PLOT_BASE_PATH.replace(".html", f"_{ptc}_AG{age}.html"))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s | %(message)s")
    reclassify_age_group(70)
