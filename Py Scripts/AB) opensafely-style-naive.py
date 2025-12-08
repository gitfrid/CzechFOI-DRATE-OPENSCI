#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from scipy.integrate import simpson
import plotly.graph_objects as go
import random

# ---------------- CONFIG ----------------
INPUT = r"C:\CzechFOI-DRATE-OPENSCI\Terra\FG) case3_sim_deaths_sim_real_doses_with_constraint.csv"
OUTPUT = r"C:\CzechFOI-DRATE-OPENSCI\Plot Results\AB) OpenSAFELY_style_naive_RMST\AB) OpenSAFELY_style_naive_RMST_AG70_SIM"
#INPUT = r"C:\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG70.csv"
#OUTPUT = r"C:\CzechFOI-DRATE-OPENSCI\Plot Results\AB) OpenSAFELY_style_naive_RMST\AB) OpenSAFELY_style_naive_RMST_AG70"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

STUDY_START = pd.Timestamp('2020-01-01')
AGE_REFERENCE_YEAR = 2023
AGE = 70
N_BOOT = 3
RANDOM_SEED = 12345
SAFETY_BUFFER_DAYS = 30

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------- HELPERS ----------------
def ensure_dt(s):
    return pd.to_datetime(s, errors='coerce')

def rmst(times, surv):
    times = np.asarray(times, dtype=float)
    surv = np.asarray(surv, dtype=float)
    return float(simpson(surv, x=times)) if len(times) > 1 else 0.0

# ---------------- LOGGING ----------------
log_file = f"{OUTPUT}.txt"
fh = open(log_file, "w", encoding="utf-8")
def log(msg):
    print(msg)
    fh.write(str(msg) + "\n")

# ---------------- LOAD DATA ----------------
log(f"Loading CSV: {INPUT}")
raw = pd.read_csv(INPUT, dtype=str)
raw.columns = raw.columns.str.strip()
for c in ['DatumUmrti'] + [f'Datum_{i}' for i in range(1, 8)]:
    if c in raw.columns:
        raw[c] = ensure_dt(raw[c])

if 'Rok_narozeni' not in raw.columns:
    raise ValueError('CSV must contain Rok_narozeni')

raw['age'] = AGE_REFERENCE_YEAR - pd.to_numeric(raw['Rok_narozeni'], errors='coerce')
raw = raw[raw['age'] == AGE].copy()
if raw.empty:
    raise SystemExit(f'No subjects at AGE {AGE}')
log(f"Number of subjects after AGE filter (AGE={AGE}): {raw.shape[0]}")

# ---------------- EXOGENOUS STUDY END & STUDY WINDOW ----------------
dose_cols = [c for c in raw.columns if c.startswith("Datum_") and c != "DatumUmrti"]
last_dose = raw[dose_cols].max(axis=1, skipna=True).max(skipna=True) if dose_cols else pd.NaT
last_death = raw['DatumUmrti'].max(skipna=True)

last_dose_day, last_death_day = ((d - STUDY_START).days if pd.notna(d) else None for d in (last_dose, last_death))
valid_days = [d for d in [last_dose_day, last_death_day] if d is not None]
if not valid_days:
    raise SystemExit("Cannot determine study end: no valid last dose or death.")

EXOGENOUS_STUDY_END_DAY = min(valid_days) - SAFETY_BUFFER_DAYS
first_dose_date = raw['Datum_1'].min(skipna=True)
FIRST_VAX_DAY = (first_dose_date - STUDY_START).days
LAST_OBS_DAY  = EXOGENOUS_STUDY_END_DAY
WINDOW_LENGTH = LAST_OBS_DAY - FIRST_VAX_DAY

log(f"Study start day (first dose observed): {FIRST_VAX_DAY} (date: {first_dose_date.date()})")
log(f"Study end day (exogenous study end): {LAST_OBS_DAY} (date: {(STUDY_START + pd.Timedelta(days=LAST_OBS_DAY)).date()})")
log(f"Observation duration (days): {WINDOW_LENGTH}")
log(f"Last recorded vaccine dose day: {last_dose_day} (date: {last_dose.date() if pd.notna(last_dose) else 'NA'})")
log(f"Last recorded death day: {last_death_day} (date: {last_death.date() if pd.notna(last_death) else 'NA'})")

# ---------------- DEFINE PERSON-TIME ----------------
raw['death_day'] = (raw['DatumUmrti'] - STUDY_START).dt.days
raw['first_dose_day'] = (raw['Datum_1'] - STUDY_START).dt.days
raw['end_day'] = raw['death_day'].fillna(LAST_OBS_DAY).clip(upper=LAST_OBS_DAY)

# ---------------- VECTORISED TV TABLE ----------------
ids = raw.index.to_numpy()
death_day = raw['death_day'].to_numpy()
first_dose_day = raw['first_dose_day'].to_numpy()
end_day = raw['end_day'].to_numpy()

tv_records = []

for i, pid in enumerate(ids):
    death = death_day[i]
    first = first_dose_day[i] if not pd.isna(first_dose_day[i]) else None
    person_end = end_day[i]

    # Unvaccinated interval
    start_u = FIRST_VAX_DAY
    stop_u = min(first, person_end) if first is not None else person_end
    if stop_u > start_u:
        event_u = int((not pd.isna(death)) and (death <= stop_u))
        tv_records.append({'id': pid, 'start': start_u, 'stop': stop_u, 'event': event_u, 'vaccinated': 0})

    # Vaccinated interval
    if first is not None and first <= person_end:
        start_v = first
        stop_v = person_end
        if stop_v > start_v:
            event_v = int((not pd.isna(death)) and (death <= stop_v) and (death >= start_v))
            tv_records.append({'id': pid, 'start': start_v, 'stop': stop_v, 'event': event_v, 'vaccinated': 1})

tv = pd.DataFrame.from_records(tv_records)
tv = tv[tv['stop'] > tv['start']].copy()
if tv.empty:
    raise SystemExit('No person-time after building tv-table')

tv_path = f"OUTDIR.parquet"
tv.to_parquet(tv_path, index=False)
log(f"Saved TV table to {tv_path} (rows: {tv.shape[0]})")

# ---------------- PERSON-TIME AND EVENTS / EPIDEMIOLOGICAL SUMMARY ----------------
tv['person_time'] = tv['stop'] - tv['start']

pt_un = float(tv[tv['vaccinated'] == 0]['person_time'].sum())
ev_un = int(tv[tv['vaccinated'] == 0]['event'].sum())
pt_tr = float(tv[tv['vaccinated'] == 1]['person_time'].sum())
ev_tr = int(tv[tv['vaccinated'] == 1]['event'].sum())

rate_un = ev_un / pt_un if pt_un > 0 else float('nan')
rate_tr = ev_tr / pt_tr if pt_tr > 0 else float('nan')
rr = rate_tr / rate_un if rate_un > 0 else float('nan')
ve = (1 - rr) * 100 if rr == rr else float('nan')

log("\n=== Epidemiological summary ===")
log(f"Unvaccinated: events={ev_un:,}, person-time={pt_un:,.0f} days, crude rate={rate_un:.6e}")
log(f"Vaccinated:   events={ev_tr:,}, person-time={pt_tr:,.0f} days, crude rate={rate_tr:.6e}")
log(f"Rate ratio: {rr:.3f}  →  VE ≈ {ve:.1f}%")

# ---------------- COX TIME-VARYING ----------------
ctv = CoxTimeVaryingFitter()
ctv.fit(tv, id_col='id', start_col='start', stop_col='stop', event_col='event', show_progress=False)
log("CoxTimeVaryingFitter summary:")
log(ctv.summary)

# ---------------- MARGINAL SURVIVAL ----------------
km_v = KaplanMeierFitter().fit(tv[tv['vaccinated']==1]['stop'], tv[tv['vaccinated']==1]['event'])
km_uv = KaplanMeierFitter().fit(tv[tv['vaccinated']==0]['stop'], tv[tv['vaccinated']==0]['event'])

common_times = np.union1d(km_v.survival_function_.index.values, km_uv.survival_function_.index.values)
Sv = np.interp(common_times, km_v.survival_function_.index.values, km_v.survival_function_.iloc[:,0])
Su = np.interp(common_times, km_uv.survival_function_.index.values, km_uv.survival_function_.iloc[:,0])

mask_follow = (common_times >= FIRST_VAX_DAY) & (common_times <= LAST_OBS_DAY)
common_times = common_times[mask_follow]
Sv = Sv[mask_follow]
Su = Su[mask_follow]

delta_rmst = rmst(common_times, Sv - Su)
log(f"Delta RMST (Vaccinated - Unvaccinated) = {delta_rmst:.3f} days")

# ---------------- VECTORISED BOOTSTRAP ----------------
id_groups = {pid: df_sub for pid, df_sub in tv.groupby('id')}
id_list = np.array(list(id_groups.keys()))
boot_rmst = np.zeros(N_BOOT)

for i in range(N_BOOT):
    sampled_ids = np.random.choice(id_list, size=len(id_list), replace=True)
    boot_df = pd.concat([id_groups[pid] for pid in sampled_ids], ignore_index=True)
    km_vb = KaplanMeierFitter().fit(boot_df[boot_df['vaccinated']==1]['stop'], boot_df[boot_df['vaccinated']==1]['event'])
    km_uvb = KaplanMeierFitter().fit(boot_df[boot_df['vaccinated']==0]['stop'], boot_df[boot_df['vaccinated']==0]['event'])
    Svb = np.interp(common_times, km_vb.survival_function_.index.values, km_vb.survival_function_.iloc[:,0])
    Sub = np.interp(common_times, km_uvb.survival_function_.index.values, km_uvb.survival_function_.iloc[:,0])
    boot_rmst[i] = rmst(common_times, Svb - Sub)

ci_low, ci_high = np.percentile(boot_rmst, [2.5, 97.5])
log(f"95% bootstrap CI for delta RMST: [{ci_low:.3f}, {ci_high:.3f}]")

# ---------------- KM PLOT ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=common_times, y=Su, mode='lines', name='Unvaccinated'))
fig.add_trace(go.Scatter(x=common_times, y=Sv, mode='lines', name='Vaccinated'))
fig.update_layout(title=f'OpenSAFELY-style: Marginal KM (AGE={AGE}, Vectorized + Fast)',
                  xaxis_title='Days', yaxis_title='Survival')

km_path = f"{OUTPUT}.html"
fig.write_html(km_path)
log(f"Saved KM plot to {km_path}")

# ---------------- CLOSE LOG ----------------
log(f"Finished. All results printed to console and saved to {log_file}")
fh.close()
