import pandas as pd
import os

"""
====================================================================================
Cohort Age Filtering for Vaccine Evaluation (All AGs)
====================================================================================

This script filters the raw vaccination and mortality dataset by age group.
It parses dates, computes age, and saves a separate CSV for each age cohort
for downstream empirical survival analysis.
"""

# === CONFIGURATION ===
INPUT_CSV = r"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131.csv"
OUTPUT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Terra"
REFERENCE_YEAR = 2023
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FUNCTIONS ===
def parse_dates(df):
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def calculate_age(df):
    df["Age"] = REFERENCE_YEAR - df["Rok_narozeni"].astype(int)
    return df

def format_dates_for_csv(df):
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('')
    return df

# === MAIN ===
def filter_and_save_all_ages():
    print("📥 Loading input CSV...")
    df = pd.read_csv(INPUT_CSV, usecols=NEEDED_COLS, dtype=str)

    print("📆 Parsing dates and calculating age...")
    df = parse_dates(df)
    df = calculate_age(df)

    ages = df["Age"].dropna().unique()
    print(f"🔎 Found {len(ages)} unique ages: {sorted(ages)}")

    ages = [70]  # Only process age 70
    print(f"🔎 Processing only Age {ages[0]}")

    #for age in sorted(ages):
    for age in [70]:
        df_age = df[df["Age"] == age].copy()
        df_age = format_dates_for_csv(df_age)

        output_csv = os.path.join(OUTPUT_DIR, f"Vesely_106_202403141131_AG{age}.csv")
        df_age.to_csv(output_csv, index=False)
        print(f"💾 Saved {len(df_age)} rows for Age {age} → {output_csv}")

    print("✅ Done. All age groups exported.")

# === RUN ===
if __name__ == "__main__":
    filter_and_save_all_ages()
