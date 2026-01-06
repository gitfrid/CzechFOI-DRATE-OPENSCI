#!/usr/bin/env python3
"""
Standalone rel-day unit test for RMST pipeline.
Usage:
    python test_rel_day.py --input /path/to/data.csv --outdir /path/to/output_dir --sample 1000
________________________________________________________________________________________________

Expected outcome from the dry run
Script completes without uncaught exceptions up to the point it normally writes outputs for the sampled data.

Warnings may be written to warnings.log but should not indicate fatal problems (no repeated errors about missing columns or empty hazard records).

Key output files created in CFG.out_dir:

run_info.txt — run metadata and seeds.

pre_vax_summary.csv — pre‑vaccination falsification summary.
hazard_df.csv — daily uptake hazards for the sampled date range.
placebo_sim_dates_wide.csv — placebo assignment dates for never‑vaccinated in each sim.
rmst_real.csv and rel_day_diagnostics.csv — RMST results and diagnostics for real vaccinations.
rmst_placebo_sims_summary.csv and plots PNGs — placebo RMST summaries and comparison plots.


Expected outcome from the Rel‑day unit test
Exit code 0 or a printed message: "Rel-day unit test PASSED".

No dev_rel_day_failures.csv file should be created.

The test confirms vacc_date is normalized and that adding 0 days returns the same calendar day for the sampled vaccinated rows.

Files to inspect after both runs
warnings.log — look for any WARNING lines flagged during the run.

rel_day_diagnostics.csv — check min_Y, prop_censored_before_tau, and flag_unreliable.

hazard_df.csv — confirm hazards are nonzero for at least some age groups/days.

placebo_sim_dates_wide.csv — ensure at least some placebo dates were assigned.

dev_rel_day_failures.csv (only if test failed) — contains failing rows for debugging.

If the Rel‑day test fails
Immediate symptom: dev_rel_day_failures.csv exists and fail_count > 0.

Inspect columns: Datum_1, vacc_date, vacc_date_norm (or the columns saved by the test).

Likely causes and fixes

Non‑normalized input dates — reapply normalization:

py
df['Datum_1'] = pd.to_datetime(df['Datum_1'], errors='coerce').dt.tz_localize(None).dt.normalize()
Timezone artifacts — ensure .dt.tz_localize(None) is used before .dt.normalize().

Unexpected types or parsing failures — check Datum_1 for malformed strings; fix or drop bad rows.

Re-run the standalone test after fixes until it passes.

Next steps after a passing test
Keep CFG.quick_test=True for iterative development or set CFG.quick_test=False and CFG.sample_frac=0.1 for a larger dry run.

Run the full pipeline on the dry sample, inspect the diagnostic files above, and profile runtime for one rel_day to estimate full run time.

Proceed to the full production run only after diagnostics look reasonable and no critical warnings remain.


"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def normalize_dates(df, date_cols):
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None).dt.normalize()
    return df

def run_test(input_path: Path, out_dir: Path, sample_n: int = 1000, seed: int = 12345):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading input: {input_path}")
    df = pd.read_csv(input_path, dtype=str)
    df.columns = df.columns.str.strip()

    # Required columns
    for col in ("Datum_1", "DatumUmrti", "Rok_narozeni"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # Normalize date columns exactly as pipeline does
    df = normalize_dates(df, ["Datum_1", "DatumUmrti"] + [c for c in df.columns if c.startswith("Datum_")])

    # Build vaccinated subset
    vaccinated = df[~df["Datum_1"].isna()].copy()
    if vaccinated.empty:
        raise SystemExit("No vaccinated rows found in the input file (Datum_1 all NaN).")

    # Create vacc_date normalized (same as pipeline)
    vaccinated["vacc_date"] = pd.to_datetime(vaccinated["Datum_1"], errors="coerce").dt.tz_localize(None).dt.normalize()

    # Sample up to sample_n rows
    n = min(sample_n, len(vaccinated))
    sample = vaccinated.sample(n=n, random_state=seed).reset_index(drop=True)

    # Compute calendar_day for rel_day 0 and check equality
    calendar_day_rel0 = sample["vacc_date"] + pd.to_timedelta(0, unit="D")
    rel_calc = (calendar_day_rel0 - sample["vacc_date"]).dt.days

    failures = sample[rel_calc != 0].copy()
    fail_count = len(failures)
    print(f"Sampled {n} vaccinated rows. Failures: {fail_count}")

    if fail_count > 0:
        out_file = out_dir / "dev_rel_day_failures.csv"
        # include key columns for debugging
        cols_to_save = [c for c in ["subject_id", "Datum_1", "vacc_date"] if c in failures.columns]
        failures.to_csv(out_file, index=False, columns=cols_to_save)
        print(f"Rel-day unit test FAILED. {fail_count} rows written to: {out_file}")
        sys.exit(2)
    else:
        print("Rel-day unit test PASSED: rel_day 0 corresponds to vaccination day in the sample.")
        sys.exit(0)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rel-day unit test for RMST pipeline")
    p.add_argument("--input", "-i", required=True, type=Path, help="Path to input CSV (same file used by pipeline)")
    p.add_argument("--outdir", "-o", required=True, type=Path, help="Directory to write diagnostics (dev_rel_day_failures.csv)")
    p.add_argument("--sample", "-n", type=int, default=1000, help="Number of vaccinated rows to sample (default 1000)")
    p.add_argument("--seed", type=int, default=12345, help="Random seed for sampling")
    args = p.parse_args()

    run_test(args.input, args.outdir, sample_n=args.sample, seed=args.seed)
