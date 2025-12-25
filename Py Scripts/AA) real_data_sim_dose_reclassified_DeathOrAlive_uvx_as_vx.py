"""
Stochastic Sensitivity Analysis for Vaccine Effectiveness
PROJECT: Czech FOI (Freedom of Information) Mortality Data Analysis

PURPOSE
-------
Simulates exposure misclassification from *unknown vaccination dates*. A fixed
percentage of individuals classified as unvaccinated (UVX) are assumed to have
been vaccinated, but their dose dates are missing. These are reclassified as
vaccinated (VX) using conservative, calendar-consistent imputation.

The analysis evaluates the robustness of vaccine effectiveness (VE) and RMST
estimates to non-differential exposure misclassification.

SCIENTIFIC RATIONALE
-------------------
- Hypothesis: Some vaccinated individuals are misclassified as unvaccinated
  due to missing vaccination dates.
- Design: Conservative, local calendar window imputation prior to death/censor.
- Imputation strategy: Local empirical doses or bounded fallback.
- Ensures strictly positive exposure (death_date - dose_date > 0).

SCIENTIFIC SPECIFICATIONS
-------------------------
- Method: Exposure-Misclassification Sensitivity Analysis (EMSA)
- Strategy tracking: Empirical vs fallback
- Exposure audit: Mean, 25th, 50th, 75th quantiles
- Distributional diagnostics: KS divergence
- Visualization: Density, CDF, normalized cumulative exposure

Author: AI-assisted (methodological specification by user)
Version: 2.0
Date: 12.2025
"""

import pandas as pd
import numpy as np
import os
import logging
from scipy.stats import ks_2samp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURATION
# ============================================================

RECLASSIFY_PERCENTAGE = 0.05
RANDOM_SEED = 12345
LOCAL_WINDOW_DAYS = 120
FALLBACK_MIN_DAYS = 30
FALLBACK_MAX_DAYS = 90

np.random.seed(RANDOM_SEED)

INPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
OUTPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"

LOG_PATH = (
    r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results"
    r"\AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx"
    r"\AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.txt"
)
PLOT_BASE_PATH = LOG_PATH.replace(".txt", ".html")

# ============================================================
# LOGGING
# ============================================================

def logmsg(msg: str):
    print(msg)
    logging.info(msg)

# ============================================================
# CORE SIMULATION
# ============================================================

def reclassify_age_group(age: int):
    ptc_label = f"PCT{int(RECLASSIFY_PERCENTAGE * 100)}"

    input_csv = os.path.join(INPUT_DIR, f"Vesely_106_202403141131_AG{age}.csv")
    output_csv = os.path.join(
        OUTPUT_DIR,
        f"AA) real_data_sim_dose_DeathOrAlive_reclassified_{ptc_label}_uvx_as_vx_AG{age}.csv"
    )

    if not os.path.exists(input_csv):
        logmsg(f"File not found for Age {age}. Skipping.")
        return

    # -------------------------------
    # 1. Load and prepare data
    # -------------------------------
    df = pd.read_csv(input_csv, low_memory=False)
    dose_cols = [c for c in df.columns if c.startswith("Datum_")]
    for c in dose_cols + ["DatumUmrti"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    obs_start = df[dose_cols].min().min()
    obs_end = df["DatumUmrti"].max()

    # -------------------------------
    # 2. Identify misclassified pool (UVX)
    # -------------------------------
    unknown_pool = df[
        df["Datum_1"].isna() &
        df["DatumUmrti"].notna() &
        (df["DatumUmrti"] >= obs_start) &
        (df["DatumUmrti"] <= obs_end)
    ].copy()

    n_select = max(1, int(RECLASSIFY_PERCENTAGE * len(unknown_pool)))
    selected_indices = np.random.choice(unknown_pool.index, size=n_select, replace=False)

    # -------------------------------
    # 3. Reference vaccination calendar
    # -------------------------------
    all_real_doses = pd.concat([df[c] for c in dose_cols]).dropna().sort_values()

    assigned_dates, followup_end_dates, strategies = [], [], []

    # -------------------------------
    # 4. Local imputation with strictly positive exposure
    # -------------------------------
    for idx in selected_indices:
        death_date = df.at[idx, "DatumUmrti"]
        end_date = death_date if pd.notna(death_date) else obs_end

        window_start = max(obs_start, end_date - pd.Timedelta(days=LOCAL_WINDOW_DAYS))
        window_end = end_date - pd.Timedelta(days=1)  # strictly before death

        # Local empirical doses
        local_pool = all_real_doses[(all_real_doses >= window_start) & (all_real_doses <= window_end)]

        if not local_pool.empty:
            dose_date = pd.Timestamp(np.random.choice(local_pool.values))
            strategy = "Empirical"
        else:
            if (window_end - window_start).days > 0:
                offset_days = np.random.randint(0, (window_end - window_start).days + 1)
                dose_date = window_start + pd.Timedelta(days=offset_days)
            else:
                dose_date = window_start
            strategy = "Fallback"

        # Ensure strictly positive exposure
        assert (end_date - dose_date).days > 0, "Negative or zero exposure detected!"

        df.at[idx, "Datum_1"] = dose_date
        assigned_dates.append(dose_date)
        followup_end_dates.append(end_date)
        strategies.append(strategy)

    # -------------------------------
    # 5. Save modified dataset
    # -------------------------------
    df.to_csv(output_csv, index=False)

    # -------------------------------
    # 6. Exposure audits
    # -------------------------------
    real_vx = df.dropna(subset=["Datum_1", "DatumUmrti"])
    real_days = (real_vx["DatumUmrti"] - real_vx["Datum_1"]).dt.days
    synth_days = (pd.Series(followup_end_dates) - pd.Series(assigned_dates)).dt.days

    r_q = real_days.quantile([0.25, 0.5, 0.75]).to_dict()
    s_q = synth_days.quantile([0.25, 0.5, 0.75]).to_dict()

    strategy_df = pd.DataFrame({"days": synth_days, "strategy": strategies})
    strat_quantiles = strategy_df.groupby("strategy")["days"].quantile([0.25, 0.5, 0.75]).unstack()

    ks_stat, ks_p = ks_2samp(real_vx["Datum_1"].dropna().astype("int64"),
                             pd.Series(assigned_dates).astype("int64"))

    # -------------------------------
    # 7. Scientific log
    # -------------------------------
    logmsg("=" * 90)
    logmsg(f"EXPOSURE MISCLASSIFICATION AUDIT — AGE {age}")
    logmsg(f"Misclassification fraction: {RECLASSIFY_PERCENTAGE:.1%}")
    logmsg(f"Total cohort size:          {len(df)}")
    logmsg(f"Reclassified individuals:   {n_select}")
    logmsg(f"Assignment strategies:      {pd.Series(strategies).value_counts().to_dict()}")
    logmsg(f"Mean exposure (real/synth): {real_days.mean():.1f} / {synth_days.mean():.1f} days")
    logmsg(f"Quantiles real (25/50/75):  {r_q[0.25]:.0f} / {r_q[0.5]:.0f} / {r_q[0.75]:.0f}")
    logmsg(f"Quantiles synth (25/50/75): {s_q[0.25]:.0f} / {s_q[0.5]:.0f} / {s_q[0.75]:.0f}")
    for strat in strat_quantiles.index:
        q = strat_quantiles.loc[strat]
        logmsg(f"  {strat} exposure (25/50/75): {q[0.25]:.0f} / {q[0.5]:.0f} / {q[0.75]:.0f}")
    logmsg(f"Injected person-days:       {synth_days.sum():.0f}")
    logmsg(f"KS divergence (dose dates): {ks_stat:.4f} (p={ks_p:.2e})")
    logmsg("=" * 90)

    create_diagnostic_plot(real_vx["Datum_1"], pd.Series(assigned_dates),
                           real_days, synth_days, age,
                           f"UVX_UNKNWN_{ptc_label}")

# ============================================================
# DIAGNOSTIC PLOTTING
# ============================================================

def create_diagnostic_plot(real_doses, synth_doses, real_days, synth_days, age, label):
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=("Vaccination Date Distribution", "Exposure Accumulation")
    )

    # Density + CDF
    fig.add_trace(go.Histogram(
        x=real_doses, name="Observed VX", histnorm="probability density",
        marker_color="rgba(70,130,180,0.4)"
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=synth_doses, name="Imputed VX", histnorm="probability density",
        marker_color="rgba(220,20,60,0.6)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=np.sort(real_doses),
        y=np.linspace(0, 1, len(real_doses)),
        name="Observed CDF",
        line=dict(color="blue")
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=np.sort(synth_doses),
        y=np.linspace(0, 1, len(synth_doses)),
        name="Imputed CDF",
        line=dict(color="red", dash="dot")
    ), row=1, col=1, secondary_y=True)

    # Exposure accumulation
    fig.add_trace(go.Box(
        y=real_days, name="Observed Exposure", marker_color="blue"
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=synth_days, name="Imputed Exposure", marker_color="red"
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        y=np.sort(real_days).cumsum() / np.sort(real_days).cumsum().max(),
        name="Observed cumulative",
        line=dict(color="blue")
    ), row=1, col=2, secondary_y=True)

    fig.add_trace(go.Scatter(
        y=np.sort(synth_days).cumsum() / np.sort(synth_days).cumsum().max(),
        name="Imputed cumulative",
        line=dict(color="red", dash="dash")
    ), row=1, col=2, secondary_y=True)

    fig.update_layout(
        title=f"Exposure Misclassification Diagnostic — Age {age} ({label})",
        template="plotly_white",
        barmode="overlay",
        legend=dict(orientation="h", y=1.05)
    )

    fig.write_html(
        PLOT_BASE_PATH.replace(".html", f"_{label}_AG{age}.html")
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(message)s"
    )

    reclassify_age_group(70)
