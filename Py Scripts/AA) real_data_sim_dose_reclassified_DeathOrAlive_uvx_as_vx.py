
"""
Stochastic Sensitivity Analysis for Vaccine Effectiveness
PROJECT: Czech FOI (Freedom of Information) Mortality Data Analysis

PURPOSE
-------
Simulates exposure misclassification from *unknown vaccination dates*. A fixed
percentage of individuals classified as unvaccinated (UVX) are assumed to have
been vaccinated, but their dose dates are missing. These are reclassified as
vaccinated (VX) using conservative, calendar-consistent imputation.

This analysis evaluates the robustness of vaccine effectiveness (VE) and RMST
estimates to non-differential exposure misclassification (Population-Wide),
under a fixed global observation window shared by VX and UVX.

SCIENTIFIC RATIONALE
-------------------
- Hypothesis: Some vaccinated individuals are misclassified as unvaccinated
  due to missing vaccination dates (e.g., registry errors).
- Design: Conservative, global calendar window imputation prior to death or study end.
- Imputation strategy: Empirical doses when possible, bounded fallback otherwise.
- Ensures strictly positive exposure (end_date - dose_date > 0).

SCIENTIFIC SPECIFICATIONS
-------------------------
- Method: Exposure-Misclassification Sensitivity Analysis (EMSA)
- Exposure window: Fixed global observation window (obs_start, obs_end)
- Exposure definition: Dose_1 (Datum_1) only
- Strategy tracking: Empirical vs fallback
- Exposure audit:
    - Mean, 25th, 50th, 75th quantiles (real deaths and all imputed)
- Distributional diagnostics:
    - KS divergence (Dose_1 dates, days since obs_start)
- Epidemiological diagnostics:
    - Exposure for deaths vs survivors (imputed)
    - Person-time for imputed individuals
    - Calendar-week dose counts and vaccination rate (real vs imputed)
- Visualization (most useful plots):
    1) Dose date distribution (real vs imputed) + CDF (2-panel figure, left side)
    2) Exposure distribution and cumulative exposure (2-panel figure, right side)
    3) Calendar-week vaccination counts and rates (real vs imputed)

Author: AI-assisted (methodological specification by user)
Version: 3.3 (Population-Wide, Dose_1 only, extended epidemiological diagnostics)
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

# Kept for reference; no backward-looking window is used.
LOCAL_WINDOW_DAYS = 120

FALLBACK_MIN_DAYS = 30
FALLBACK_MAX_DAYS = 90  # kept for compatibility; not explicitly used

np.random.seed(RANDOM_SEED)

INPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
OUTPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"

LOG_PATH = (
    r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results"
    r"\AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx"
    r"\AA) real_data_sim_dose_reclassified_DeathOrAlive_uvx_as_vx.txt"
)
PLOT_BASE_PATH = LOG_PATH.replace(".txt", ".html")

# Optional: export additional epidemiological tables
EPI_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    r"AA) real_data_sim_dose_DeathOrAlive_reclassified_EPI"
)

# ============================================================
# LOGGING
# ============================================================

def logmsg(msg: str):
    print(msg)
    logging.info(msg)

# ============================================================
# DIAGNOSTIC PLOTTING – CORE 2-PANEL FIGURE
# ============================================================

def create_diagnostic_plot(
    real_doses,
    synth_doses,
    real_days_deaths,
    synth_days_all,
    age,
    label
):
    """
    Core diagnostic figure:

    Left panel:
      - Histogram density of real Dose_1 dates
      - Histogram density of imputed Dose_1 dates
      - CDF of real Dose_1
      - CDF of imputed Dose_1

    Right panel:
      - Boxplot of real exposure (deaths only)
      - Boxplot of imputed exposure (all imputed)
      - Cumulative exposure curve (real deaths)
      - Cumulative exposure curve (imputed all)
    """

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        subplot_titles=("Vaccination Date Distribution", "Exposure Accumulation")
    )

    # LEFT: Density + CDF (calendar dates)
    fig.add_trace(go.Histogram(
        x=real_doses,
        name="Observed VX Dose_1",
        histnorm="probability density",
        marker_color="rgba(70,130,180,0.4)"
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=synth_doses,
        name="Imputed VX Dose_1",
        histnorm="probability density",
        marker_color="rgba(220,20,60,0.6)"
    ), row=1, col=1)

    if len(real_doses) > 0:
        fig.add_trace(go.Scatter(
            x=np.sort(real_doses),
            y=np.linspace(0, 1, len(real_doses)),
            name="Observed CDF",
            line=dict(color="blue")
        ), row=1, col=1, secondary_y=True)

    if len(synth_doses) > 0:
        fig.add_trace(go.Scatter(
            x=np.sort(synth_doses),
            y=np.linspace(0, 1, len(synth_doses)),
            name="Imputed CDF",
            line=dict(color="red", dash="dot")
        ), row=1, col=1, secondary_y=True)

    # RIGHT: Exposure Boxplots + Cumulative Curves
    fig.add_trace(go.Box(
        y=real_days_deaths,
        name="Observed Exposure (Deaths)",
        marker_color="blue"
    ), row=1, col=2)

    fig.add_trace(go.Box(
        y=synth_days_all,
        name="Imputed Exposure (All)",
        marker_color="red"
    ), row=1, col=2)

    if len(real_days_deaths) > 0:
        sorted_real = np.sort(real_days_deaths)
        cum_real = sorted_real.cumsum()
        if cum_real.max() > 0:
            cum_real_norm = cum_real / cum_real.max()
        else:
            cum_real_norm = cum_real
        fig.add_trace(go.Scatter(
            y=cum_real_norm,
            name="Observed Cumulative (Deaths)",
            line=dict(color="blue")
        ), row=1, col=2, secondary_y=True)

    if len(synth_days_all) > 0:
        sorted_synth = np.sort(synth_days_all)
        cum_synth = sorted_synth.cumsum()
        if cum_synth.max() > 0:
            cum_synth_norm = cum_synth / cum_synth.max()
        else:
            cum_synth_norm = cum_synth
        fig.add_trace(go.Scatter(
            y=cum_synth_norm,
            name="Imputed Cumulative (All)",
            line=dict(color="red", dash="dash")
        ), row=1, col=2, secondary_y=True)

    fig.update_layout(
        title=f"Exposure Misclassification Diagnostic (Pop-Wide, Dose_1) — Age {age} ({label})",
        template="plotly_white",
        barmode="overlay",
        legend=dict(orientation="h", y=1.05)
    )

    fig.write_html(PLOT_BASE_PATH.replace(".html", f"_{label}_AG{age}.html"))

# ============================================================
# DIAGNOSTIC PLOTTING – CALENDAR-WEEK VACCINATION RATE
# ============================================================

def create_weekly_vaccination_plot(
    real_week_counts,
    imp_week_counts,
    imp_week_rate,
    age,
    label
):
    """
    Additional essential epidemiological plot:

    - Calendar-week counts of Dose_1:
        * Observed real vaccinations
        * Imputed vaccinations
    - Imputed vaccination fraction of cohort per week (secondary y-axis)
    """

    # Align indices
    all_weeks = sorted(set(real_week_counts.index).union(set(imp_week_counts.index)))
    real_counts_aligned = real_week_counts.reindex(all_weeks, fill_value=0)
    imp_counts_aligned = imp_week_counts.reindex(all_weeks, fill_value=0)
    imp_rate_aligned = imp_week_rate.reindex(all_weeks, fill_value=0.0)

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=("Calendar-Week Vaccination Counts and Rates (Dose_1)")
    )

    fig.add_trace(go.Bar(
        x=all_weeks,
        y=real_counts_aligned.values,
        name="Real Dose_1 (counts)",
        marker_color="rgba(70,130,180,0.7)"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=all_weeks,
        y=imp_counts_aligned.values,
        name="Imputed Dose_1 (counts)",
        marker_color="rgba(220,20,60,0.7)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=all_weeks,
        y=imp_rate_aligned.values,
        name="Imputed Dose_1 (fraction of cohort)",
        mode="lines+markers",
        line=dict(color="black", width=2)
    ), row=1, col=1, secondary_y=True)

    fig.update_layout(
        title=f"Calendar-Week Vaccination Dynamics (Dose_1) — Age {age} ({label})",
        template="plotly_white",
        barmode="group",
        xaxis_title="Calendar Week",
        legend=dict(orientation="h", y=1.1)
    )

    fig.update_yaxes(title_text="Counts", secondary_y=False)
    fig.update_yaxes(title_text="Fraction of Cohort (Imputed)", secondary_y=True)

    fig.write_html(PLOT_BASE_PATH.replace(".html", f"_{label}_WEEKLY_AG{age}.html"))

# ============================================================
# EXTRA EPIDEMIOLOGICAL OUTPUTS
# ============================================================

def compute_extended_epi_outputs(
    df,
    obs_start,
    obs_end,
    assigned_indices,
    assigned_dates,
    strategies,
    age,
    ptc_label
):
    """
    Computes extended epidemiological diagnostics and (optionally) writes them to disk.
    Focused on:
      - Exposure for deaths vs survivors (imputed)
      - Person-time for imputed individuals
      - Calendar-week dose counts and imputed vaccination rate
      - Logging summary stats
    """
    if len(assigned_dates) == 0:
        return None, None, None  # for downstream plotting

    epi_df = pd.DataFrame({
        "idx": assigned_indices,
        "dose_date": assigned_dates,
        "strategy": strategies
    })

    epi_df["dose_date"] = pd.to_datetime(epi_df["dose_date"])
    epi_df = epi_df.merge(
        df[["DatumUmrti"]],
        left_on="idx",
        right_index=True,
        how="left"
    )

    # Define end_point per individual (death if present, otherwise obs_end)
    epi_df["end_point"] = np.where(
        epi_df["DatumUmrti"].notna(),
        epi_df["DatumUmrti"],
        obs_end
    )
    epi_df["end_point"] = pd.to_datetime(epi_df["end_point"])

    # Exposure in days (imputed)
    epi_df["exposure_days"] = (epi_df["end_point"] - epi_df["dose_date"]).dt.days

    # Death vs survivor classification
    epi_df["status"] = np.where(epi_df["DatumUmrti"].notna(), "Death", "Survivor")

    # Person-time (imputed, in years)
    epi_df["pt_years"] = epi_df["exposure_days"] / 365.25

    # Calendar-week dose counts (real vs imputed)
    real_d1 = df["Datum_1"].dropna()
    real_d1_weeks = real_d1.dt.to_period("W").astype(str)
    real_week_counts = real_d1_weeks.value_counts().sort_index()

    imp_weeks = epi_df["dose_date"].dt.to_period("W").astype(str)
    imp_week_counts = imp_weeks.value_counts().sort_index()

    cohort_size = len(df)
    imp_week_rate = imp_week_counts / cohort_size

    # Time-since-dose distributions (for log only)
    time_since_dose_deaths = epi_df.loc[epi_df["status"] == "Death", "exposure_days"]
    time_since_dose_survivors = epi_df.loc[epi_df["status"] == "Survivor", "exposure_days"]

    # Logging extended summaries
    logmsg("--- Extended Epidemiological Outputs (Imputed) ---")
    logmsg(f"Imputed individuals (N):       {len(epi_df)}")
    logmsg(f"  Deaths (imputed)            : {sum(epi_df['status'] == 'Death')}")
    logmsg(f"  Survivors (imputed)         : {sum(epi_df['status'] == 'Survivor')}")

    if not time_since_dose_deaths.empty:
        logmsg(
            "  Exposure (Deaths) mean/median/25-75%: "
            f"{time_since_dose_deaths.mean():.1f} / "
            f"{time_since_dose_deaths.median():.1f} / "
            f"{time_since_dose_deaths.quantile(0.25):.1f}-"
            f"{time_since_dose_deaths.quantile(0.75):.1f} days"
        )

    if not time_since_dose_survivors.empty:
        logmsg(
            "  Exposure (Survivors) mean/median/25-75%: "
            f"{time_since_dose_survivors.mean():.1f} / "
            f"{time_since_dose_survivors.median():.1f} / "
            f"{time_since_dose_survivors.quantile(0.25):.1f}-"
            f"{time_since_dose_survivors.quantile(0.75):.1f} days"
        )

    logmsg(
        f"Total person-time (imputed) ~ {epi_df['pt_years'].sum():.1f} person-years "
        f"(mean {epi_df['pt_years'].mean():.3f} per imputed individual)"
    )

    if not imp_week_rate.empty:
        first_weeks = imp_week_rate.head(3)
        last_weeks = imp_week_rate.tail(3)
        logmsg("Imputed weekly vaccination rate (first 3 weeks):")
        for w, v in first_weeks.items():
            logmsg(f"  Week {w}: {v:.4f} of cohort")
        logmsg("Imputed weekly vaccination rate (last 3 weeks):")
        for w, v in last_weeks.items():
            logmsg(f"  Week {w}: {v:.4f} of cohort")

    # Optional: write extended outputs to disk
    try:
        os.makedirs(EPI_OUTPUT_DIR, exist_ok=True)

        epi_df.to_csv(
            os.path.join(
                EPI_OUTPUT_DIR,
                f"EPI_imputed_individuals_{ptc_label}_AG{age}.csv"
            ),
            index=False
        )

        real_week_counts.to_csv(
            os.path.join(
                EPI_OUTPUT_DIR,
                f"EPI_real_dose1_week_counts_{ptc_label}_AG{age}.csv"
            ),
            header=["n_real_dose1"]
        )

        imp_week_counts.to_csv(
            os.path.join(
                EPI_OUTPUT_DIR,
                f"EPI_imputed_dose1_week_counts_{ptc_label}_AG{age}.csv"
            ),
            header=["n_imputed_dose1"]
        )

        imp_week_rate.to_csv(
            os.path.join(
                EPI_OUTPUT_DIR,
                f"EPI_imputed_dose1_week_rate_{ptc_label}_AG{age}.csv"
            ),
            header=["imputed_fraction_of_cohort"]
        )

    except Exception as e:
        logmsg(f"Warning: could not write extended EPI outputs: {e}")

    # Return for weekly plot
    return real_week_counts, imp_week_counts, imp_week_rate

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

    # Global observation window (fixed for all individuals, VX and UVX)
    obs_start = df[dose_cols].min().min()
    obs_end = df["DatumUmrti"].max()
    if pd.isna(obs_end):
        obs_end = pd.to_datetime("2024-03-14")

    # -------------------------------
    # 2. Identify Population-Wide Pool (Alive + Dead)
    # -------------------------------
    unknown_pool = df[df["Datum_1"].isna()].copy()
    n_select = max(1, int(RECLASSIFY_PERCENTAGE * len(unknown_pool)))
    selected_indices = np.random.choice(unknown_pool.index, size=n_select, replace=False)

    # -------------------------------
    # 3. Reference vaccination calendar (Dose_1 only)
    # -------------------------------
    all_real_dose1 = df["Datum_1"].dropna().sort_values()

    if all_real_dose1.empty:
        logmsg("Warning: No observed Datum_1 in dataset; imputation will be fully fallback-based.")

    assigned_dates = []
    assigned_indices = []
    followup_end_dates = []
    strategies = []

    # -------------------------------
    # 4. Imputation Logic (global window, Dose_1 only)
    # -------------------------------
    for idx in selected_indices:
        death_date = df.at[idx, "DatumUmrti"]
        end_point = death_date if pd.notna(death_date) else obs_end

        window_start = obs_start
        window_end = end_point - pd.Timedelta(days=1)

        if window_end < window_start:
            continue

        if not all_real_dose1.empty:
            local_pool = all_real_dose1[
                (all_real_dose1 >= window_start) & (all_real_dose1 <= window_end)
            ]
        else:
            local_pool = pd.Series([], dtype="datetime64[ns]")

        if not local_pool.empty:
            dose_date = pd.Timestamp(np.random.choice(local_pool.values))
            strategy = "Empirical"
        else:
            window_length_days = (window_end - window_start).days
            offset_days = np.random.randint(0, window_length_days + 1)
            dose_date = window_start + pd.Timedelta(days=offset_days)
            strategy = "Fallback"

        if dose_date >= end_point:
            if (end_point - pd.Timedelta(days=1)) >= window_start:
                dose_date = end_point - pd.Timedelta(days=1)
            else:
                continue

        df.at[idx, "Datum_1"] = dose_date
        assigned_dates.append(dose_date)
        assigned_indices.append(idx)
        followup_end_dates.append(end_point)
        strategies.append(strategy)

    # -------------------------------
    # 5. Save modified dataset
    # -------------------------------
    df.to_csv(output_csv, index=False)

    if len(assigned_dates) == 0:
        logmsg(f"No valid reclassifications for Age {age}.")
        return

    # -------------------------------
    # 6. Exposure audits
    # -------------------------------
    real_vx_deaths = df.dropna(subset=["Datum_1", "DatumUmrti"])
    real_days_deaths = (real_vx_deaths["DatumUmrti"] - real_vx_deaths["Datum_1"]).dt.days

    synth_end = pd.Series(followup_end_dates)
    synth_start = pd.Series(assigned_dates)
    synth_days_all = (synth_end - synth_start).dt.days

    r_q = (
        real_days_deaths.quantile([0.25, 0.5, 0.75]).to_dict()
        if not real_days_deaths.empty
        else {0.25: 0, 0.5: 0, 0.75: 0}
    )
    s_q = synth_days_all.quantile([0.25, 0.5, 0.75]).to_dict()

    strategy_df = pd.DataFrame({"days": synth_days_all, "strategy": strategies})
    strat_quantiles = strategy_df.groupby("strategy")["days"].quantile(
        [0.25, 0.5, 0.75]
    ).unstack()

    if not all_real_dose1.empty:
        real_doses_days = (all_real_dose1 - obs_start).dt.days
        assigned_days = (pd.Series(assigned_dates) - obs_start).dt.days
        ks_stat, ks_p = ks_2samp(real_doses_days, assigned_days)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    # -------------------------------
    # 7. Extended epidemiological outputs
    # -------------------------------
    real_week_counts, imp_week_counts, imp_week_rate = compute_extended_epi_outputs(
        df=df,
        obs_start=obs_start,
        obs_end=obs_end,
        assigned_indices=assigned_indices,
        assigned_dates=assigned_dates,
        strategies=strategies,
        age=age,
        ptc_label=ptc_label
    )

    # -------------------------------
    # 8. Scientific log
    # -------------------------------
    logmsg("=" * 90)
    logmsg(f"POPULATION-WIDE EXPOSURE MISCLASSIFICATION AUDIT — AGE {age}")
    logmsg(f"Misclassification fraction: {RECLASSIFY_PERCENTAGE:.1%}")
    logmsg(f"Total cohort size:          {len(df)}")
    logmsg(f"Reclassified individuals:   {len(assigned_dates)} (Alive + Dead)")
    logmsg(f"Assignment strategies:      {pd.Series(strategies).value_counts().to_dict()}")

    if not real_days_deaths.empty:
        logmsg(
            f"Mean exposure (real deaths / all synth): "
            f"{real_days_deaths.mean():.1f} / {synth_days_all.mean():.1f} days"
        )
        logmsg(
            f"Quantiles real (25/50/75): "
            f"{r_q[0.25]:.0f} / {r_q[0.5]:.0f} / {r_q[0.75]:.0f}"
        )
    else:
        logmsg(
            f"Mean exposure (real deaths unavailable) / all synth: "
            f"NaN / {synth_days_all.mean():.1f} days"
        )
        logmsg("Quantiles real (25/50/75): NaN / NaN / NaN")

    logmsg(
        f"Quantiles synth (25/50/75): "
        f"{s_q[0.25]:.0f} / {s_q[0.5]:.0f} / {s_q[0.75]:.0f}"
    )
    for strat in strat_quantiles.index:
        q = strat_quantiles.loc[strat]
        logmsg(
            f"  {strat} exposure (25/50/75): "
            f"{q[0.25]:.0f} / {q[0.5]:.0f} / {q[0.75]:.0f}"
        )

    if not np.isnan(ks_stat):
        logmsg(
            f"KS divergence (Dose_1 dates, days since obs_start): "
            f"{ks_stat:.4f} (p={ks_p:.2e})"
        )
    else:
        logmsg("KS divergence (Dose_1): not available (no real Dose_1 dates).")

    logmsg("=" * 90)

    # -------------------------------
    # 9. Plots (most useful)
    # -------------------------------
    create_diagnostic_plot(
        real_doses=all_real_dose1 if not all_real_dose1.empty else pd.Series([], dtype="datetime64[ns]"),
        synth_doses=pd.Series(assigned_dates),
        real_days_deaths=real_days_deaths,
        synth_days_all=synth_days_all,
        age=age,
        label=f"UVX_ALL_{ptc_label}"
    )

    if real_week_counts is not None and imp_week_counts is not None and imp_week_rate is not None:
        create_weekly_vaccination_plot(
            real_week_counts=real_week_counts,
            imp_week_counts=imp_week_counts,
            imp_week_rate=imp_week_rate,
            age=age,
            label=f"UVX_ALL_{ptc_label}"
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

    # Example: run for AG70 (only age group in your data)
    reclassify_age_group(70)
