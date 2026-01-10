# Quick code snippet you can run in a new cell after both runs
import pandas as pd
import numpy as np

folder_original = fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\1 AG) bias necessety rmst audit_QUICK_TEST"
folder_optimized = fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AG) bias necessety rmst audit_QUICK_TEST"

files_to_compare = [
    'rmst_real.csv',
    'rmst_placebo_sims_summary.csv',
    'pre_vax_summary.csv',
    'hazard_df.csv',
    'rel_day_diagnostics.csv',  # add if exists
    'rmst_difference_real_minus_placebo_mean.csv'
]

for file in files_to_compare:
    try:
        df_orig = pd.read_csv(f"{folder_original}/{file}")
        df_opt  = pd.read_csv(f"{folder_optimized}/{file}")
        
        print(f"\n=== {file} ===")
        
        # Sort by key column if present
        key_col = next((c for c in ['rel_day', 'quantile', 'calendar_day'] if c in df_orig.columns), None)
        if key_col:
            df_orig = df_orig.sort_values(key_col).reset_index(drop=True)
            df_opt  = df_opt.sort_values(key_col).reset_index(drop=True)
        
        # Auto-detect numeric columns (including bool → int)
        numeric_cols = []
        for col in df_orig.columns:
            try:
                orig_series = pd.to_numeric(df_orig[col], errors='coerce')
                opt_series  = pd.to_numeric(df_opt[col],  errors='coerce')
                if not (orig_series.isna().all() and opt_series.isna().all()):
                    df_orig[col] = orig_series
                    df_opt[col]  = opt_series
                    numeric_cols.append(col)
            except:
                pass  # Skip if coercion fails
        
        if not numeric_cols:
            print("No numeric columns found to compare.")
            continue
        
        # Compute diff only on numeric columns
        diff = df_orig[numeric_cols] - df_opt[numeric_cols]
        abs_diff = diff.abs()
        
        print("Maximum absolute differences:")
        print(abs_diff.max())
        
        print("\nMean absolute differences:")
        print(abs_diff.mean())
        
        # Sample side-by-side (first numeric column)
        sample_col = numeric_cols[0] if numeric_cols else None
        if sample_col and key_col:
            comparison = pd.DataFrame({
                key_col: df_orig[key_col],
                f'orig_{sample_col}': df_orig[sample_col],
                f'opt_{sample_col}': df_opt[sample_col],
                'diff': diff[sample_col]
            })
            print(f"\nSample side-by-side (first 10 rows, using '{sample_col}'):")
            print(comparison.head(10))
        elif sample_col:
            print(f"\nFirst numeric column '{sample_col}' sample (first 10):")
            print(df_orig[sample_col].head(10))
        
        # Shape check
        if df_orig.shape != df_opt.shape:
            print(f"WARNING: Shapes differ! Orig: {df_orig.shape}, Opt: {df_opt.shape}")
    
    except FileNotFoundError as e:
        print(f"Skipping {file}: {e}")
    except Exception as e:
        print(f"Error comparing {file}: {e}")