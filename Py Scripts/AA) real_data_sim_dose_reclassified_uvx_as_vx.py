import pandas as pd
import numpy as np
import os
import logging

# === CONFIGURATION ===
INPUT_DIR  = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
OUTPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
LOG_PATH   = r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AA) real_data_sim_dose_reclassified_uvx_as_vx\AA) real_data_sim_dose_reclassified_uvx_as_vx.txt"

STUDY_START = pd.Timestamp("2020-01-01")
RECLASSIFY_PERCENTAGE = 0.2
RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)

# --- Ensure log directory exists ---
log_dir = os.path.dirname(LOG_PATH)
os.makedirs(log_dir, exist_ok=True)

# --- Logging setup ---
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger()

def logmsg(msg):
    print(msg)
    log.info(msg)

# --- Function to process one age group ---
def reclassify_age_group(age):
    input_csv  = os.path.join(INPUT_DIR, f"Vesely_106_202403141131_AG{age}.csv")
    output_csv = os.path.join(OUTPUT_DIR, f"AA) real_data_sim_dose_reclassified_uvx_as_vx_AG{age}.csv")

    if not os.path.exists(input_csv):
        logmsg(f"⚠️ File not found for Age {age}, skipping.")
        return

    logmsg(f"📥 Processing Age {age} → {input_csv}")
    df = pd.read_csv(input_csv)

    # Convert all dose columns to datetime
    dose_cols = [c for c in df.columns if c.startswith("Datum_")]
    for c in dose_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["DatumUmrti"] = pd.to_datetime(df["DatumUmrti"], errors="coerce")

    # Identify uvx deaths (death but no recorded dose)
    uvx_deaths = df[(df["DatumUmrti"].notna()) & (df["Datum_1"].isna())].copy()
    logmsg(f"Age {age}: total uvx deaths = {len(uvx_deaths)}")

    if len(uvx_deaths) == 0:
        logmsg(f"Age {age}: no uvx deaths, skipping reclassification.")
        return

    # Randomly select percentage
    n_select = int(RECLASSIFY_PERCENTAGE * len(uvx_deaths))
    selected_idx = np.random.choice(uvx_deaths.index, size=n_select, replace=False)
    logmsg(f"Age {age}: reclassifying {n_select} (~{int(RECLASSIFY_PERCENTAGE*100)}%) uvx deaths")

    # Global cohort bounds: earliest and latest dose ever
    cohort_first_dose = df[dose_cols].min().min()
    cohort_last_dose  = df[dose_cols].max().max()
    logmsg(f"Age {age}: cohort dose bounds → first={cohort_first_dose}, last={cohort_last_dose}")

    # All real dose dates
    all_real_doses = pd.concat([df[c] for c in dose_cols]).dropna()

    assigned_dates = []
    for idx in selected_idx:
        death_date = df.loc[idx, "DatumUmrti"]

        # Restrict distribution to doses before death
        valid_doses = all_real_doses[(all_real_doses < death_date) &
                                     (all_real_doses >= cohort_first_dose) &
                                     (all_real_doses <= cohort_last_dose)]

        if len(valid_doses) > 0:
            # Empirical distribution (frequency counts)
            value_counts = valid_doses.value_counts(normalize=True)
            dose_date = np.random.choice(value_counts.index, p=value_counts.values)
        else:
            # Fallback: assign 30–90 days before death, clamped to cohort bounds
            offset_days = np.random.randint(30, 91)
            candidate = death_date - pd.Timedelta(days=offset_days)
            if candidate < cohort_first_dose:
                candidate = cohort_first_dose
            if candidate > cohort_last_dose:
                candidate = cohort_last_dose
            dose_date = candidate

        df.loc[idx, "Datum_1"] = dose_date
        assigned_dates.append(dose_date)

    # Save modified dataset
    df.to_csv(output_csv, index=False)
    logmsg(f"💾 Saved reclassified dataset for Age {age} → {output_csv}")

    # Log summary statistics
    if assigned_dates:
        assigned_series = pd.Series(assigned_dates)
        logmsg(f"Age {age}: dose date summary")
        logmsg(f"  Earliest: {assigned_series.min()}")
        logmsg(f"  Latest:   {assigned_series.max()}")
        logmsg(f"  Median:   {assigned_series.median()}")
        logmsg(f"  Mean:     {assigned_series.mean()}")

# --- MAIN LOOP ---
def main():
    for age in range(0, 115):  # loop over ages 0–114
        reclassify_age_group(age)

    logmsg("✅ Done. All age groups processed.")

if __name__ == "__main__":
    main()
