#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold-standard empirical landmark analysis
Outputs:
- CSV with per-landmark ΔRMST(t) and summaries
- Plotly HTML figure:
    Row 1: ΔRMST(t) across landmarks
    Row 2: survival curves (vx vs uvx) per landmark with shaded area (ΔRMST)
- Log file with detailed epidemiological summary
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging, logging.handlers, multiprocessing

# ---------------- Configuration ----------------
AGE = 80
STUDY_START = pd.Timestamp("2020-01-01")
AGE_REF_YEAR = 2023
IMMUNITY_LAG = 0         # set >0 if you want post-vax immunity delay (in days)
OBS_END_SAFTY = 30
RANDOM_SEED = 12345
LANDMARK_STEP = 30       # evaluate every 30 days

# Input/Output (toggle INPUT lines as needed)
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) case3_sim_deaths_sim_real_doses_with_constraint_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_SIM")
#INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\Vesely_106_202403141131_AG{AGE}.csv")
#OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST")
INPUT = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Terra\AA) real_data_sim_dose_reclassified_uvx_as_vx_AG{AGE}.csv")
OUTPUT_BASE = Path(fr"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST\AE) Landmark_full_emperical_RMST_RECLASSIFIED")


OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH =  OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.txt")
CSV_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.csv")
PLOT_PATH = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.stem}_AG{AGE}.html")

# ---------------- Logging ----------------
log_queue = multiprocessing.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(queue_handler)
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(message)s")
file_handler.setFormatter(formatter)
listener = logging.handlers.QueueListener(log_queue, file_handler)
listener.start()
def log(msg): logger.info(msg)

# ---------------- Helpers ----------------
def survival_from_hazard(h):
    h = np.clip(h, 0.0, 1.0)
    s = np.empty_like(h, dtype=float)
    acc = 1.0
    for i in range(len(h)):
        acc *= (1.0 - h[i])
        s[i] = acc
    return s

def rmst_from_survival(S, dt=1.0):
    return float(np.trapezoid(S, dx=dt))

# ---------------- Load & preprocess ----------------
np.random.seed(RANDOM_SEED)
log(f"Loading input: {INPUT}")
raw = pd.read_csv(INPUT, low_memory=False)
raw.columns = raw.columns.str.strip()

# Parse dates (matches your old script)
date_cols = [c for c in raw.columns if c.startswith("Datum_") or c == "DatumUmrti"]
for c in date_cols:
    raw[c] = pd.to_datetime(raw[c], errors="coerce")

# Age filter (exact age group)
raw["age"] = AGE_REF_YEAR - pd.to_numeric(raw["Rok_narozeni"], errors="coerce")
raw = raw[raw["age"] == AGE].reset_index(drop=True)

# Construct analysis day variables aligned to study start
raw["death_day"] = (raw["DatumUmrti"] - STUDY_START).dt.days
raw["vax_day"]   = (raw["Datum_1"]     - STUDY_START).dt.days
raw.loc[raw["death_day"] < 0, "death_day"] = np.nan
raw.loc[raw["vax_day"]   < 0, "vax_day"]   = np.nan

# Study window from observed data (gold standard: data-driven)
first_dose_min = raw["Datum_1"].min(skipna=True)
last_dose_max = raw[[c for c in raw.columns if c.startswith("Datum_") and c != "DatumUmrti"]].max().max(skipna=True)
last_death = raw["DatumUmrti"].max(skipna=True)
base_end = min(d for d in [last_dose_max, last_death] if pd.notna(d))
FIRST_ELIG_DAY = int((first_dose_min - STUDY_START).days)
LAST_OBS_DAY   = max(1, int((base_end - STUDY_START).days) - OBS_END_SAFTY)
DAYS = np.arange(FIRST_ELIG_DAY, LAST_OBS_DAY + 1)
LANDMARKS = np.arange(FIRST_ELIG_DAY, LAST_OBS_DAY + 1, LANDMARK_STEP)

log(f"Subjects: {len(raw)}")
log(f"Landmark window: {FIRST_ELIG_DAY}..{LAST_OBS_DAY} ({len(DAYS)} days)")
log(f"Evaluating landmarks every {LANDMARK_STEP} days ({len(LANDMARKS)} landmarks)")

# ---------------- Index mapping for fast empirical counts ----------------
NEVER_VAX_IDX = (LAST_OBS_DAY - FIRST_ELIG_DAY + 1)
V_DIM = NEVER_VAX_IDX + 1                 # vaccination index dimension (0..NEVER_VAX_IDX)
D_DIM = (LAST_OBS_DAY - FIRST_ELIG_DAY + 1)  # death day index dimension (0..D_DIM-1)

vax_days = raw["vax_day"].to_numpy()

# vax_days is a NumPy array, so handle NaN with np.where
v_idx = np.where(np.isnan(vax_days),
                 NEVER_VAX_IDX,
                 np.clip(vax_days.astype(int) - FIRST_ELIG_DAY, 0, NEVER_VAX_IDX))


death_days = raw["death_day"].to_numpy()
d_idx = np.full(len(death_days), -1, dtype=int)
mask_death = ~np.isnan(death_days)
d_idx[mask_death] = np.clip(death_days[mask_death].astype(int) - FIRST_ELIG_DAY, 0, D_DIM - 1)

# ---------------- Precompute counts (prefix/suffix for landmark splits) ----------------
counts_by_v = np.bincount(v_idx, minlength=V_DIM).astype(np.int64)

deaths_by_vd = np.zeros((V_DIM, D_DIM), dtype=np.int64)
mask = (d_idx >= 0)
for vi, di in zip(v_idx[mask], d_idx[mask]):
    deaths_by_vd[vi, di] += 1

counts_cumv = np.cumsum(counts_by_v)                  # cumulative vaccinated by index
counts_sufv = np.cumsum(counts_by_v[::-1])[::-1]      # suffix counts (not-yet-vaccinated segment)
deaths_prefv = np.cumsum(deaths_by_vd, axis=0)        # deaths cum over v prior
deaths_sufv  = np.cumsum(deaths_by_vd[::-1, :], axis=0)[::-1, :]  # deaths cum over v suffix
deaths_cumd  = np.cumsum(deaths_by_vd, axis=1)        # deaths cum over days
deaths_cumd_prefv = np.cumsum(deaths_cumd, axis=0)    # deaths cum over v then days (prefix)
deaths_cumd_sufv  = np.cumsum(deaths_cumd[::-1, :], axis=0)[::-1, :]  # deaths cum over v then days (suffix)

# ---------------- Compute per-landmark ΔRMST ----------------
results = []
log("Computing empirical ΔRMST(t) over landmarks...")

for t_day in tqdm(LANDMARKS, desc="Landmarks"):
    # Index position of the landmark
    t = t_day - FIRST_ELIG_DAY
    t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)  # optional immunity lag exclusion

    # Risk set sizes at landmark, conditional on alive (exclude those who died up to t_excl)
    N_v = counts_cumv[min(t, V_DIM - 2)]  # vaccinated by t (index t maps to <= v_idx t)
    died_to_t_v = deaths_cumd_prefv[min(t, V_DIM - 2), t_excl] if N_v > 0 else 0
    N_v -= died_to_t_v

    v_start_u = min(t + 1, V_DIM - 1)  # first index that corresponds to "not yet vaccinated by t"
    N_u = counts_sufv[v_start_u]
    died_to_t_u = deaths_cumd_sufv[v_start_u, t_excl] if N_u > 0 else 0
    N_u -= died_to_t_u

    # If either group has no risk set, skip landmark
    if N_v <= 0 or N_u <= 0:
        continue

    # Event sequences after landmark (exclude early interval up to t_excl)
    E_v = deaths_prefv[min(t, V_DIM - 2), :].copy()
    E_u = deaths_sufv[v_start_u, :].copy()
    if t_excl >= 0:
        E_v[:(t_excl + 1)] = 0
        E_u[:(t_excl + 1)] = 0

    # Cumulative events to build risk set over time since landmark
    cumE_v = deaths_cumd_prefv[min(t, V_DIM - 2), :] - deaths_cumd_prefv[min(t, V_DIM - 2), t_excl]
    cumE_u = deaths_cumd_sufv[v_start_u, :]           - deaths_cumd_sufv[v_start_u, t_excl]

    R_v = N_v - np.concatenate([[0], cumE_v[:-1]])
    R_u = N_u - np.concatenate([[0], cumE_u[:-1]])
    R_v = np.clip(R_v, 1e-12, None)
    R_u = np.clip(R_u, 1e-12, None)

    # Segment start and horizon
    start = t_excl + 1
    if start >= D_DIM:
        continue

    h_v_seg = E_v[start:] / R_v[start:]
    h_u_seg = E_u[start:] / R_u[start:]

    # Empirical survival from discrete hazards
    S_v = survival_from_hazard(h_v_seg)
    S_u = survival_from_hazard(h_u_seg)

    # RMST is area under survival; ΔRMST is group difference
    rmst_v = rmst_from_survival(S_v)
    rmst_u = rmst_from_survival(S_u)
    delta = float(rmst_v - rmst_u)

    # Summaries for logging
    total_events_v = int(E_v[start:].sum())
    total_events_u = int(E_u[start:].sum())

    results.append({
        "t_landmark": t_day,
        "n_v": int(N_v),
        "n_u": int(N_u),
        "events_v": total_events_v,
        "events_u": total_events_u,
        "rmst_v": rmst_v,
        "rmst_u": rmst_u,
        "delta_rmst": delta,
        "horizon_len": int(len(h_v_seg))
    })

    log(f"Landmark {t_day}: "
        f"N_v={N_v}, N_u={N_u}, "
        f"events_v={total_events_v}, events_u={total_events_u}, "
        f"rmst_v={rmst_v:.2f}, rmst_u={rmst_u:.2f}, "
        f"ΔRMST={delta:+.2f}, horizon={len(h_v_seg)}")

# ---------------- Collect results ----------------
res_df = pd.DataFrame(results).sort_values("t_landmark")
res_df.to_csv(CSV_PATH, index=False)
log(f"Saved per-landmark results CSV: {CSV_PATH}")
log(f"Landmarks computed: {len(res_df)}")

# ---------------- Helper: reconstruct survival curves at a landmark ----------------
def survival_curves_at_landmark(t_day):
    """
    Reconstruct empirical survival curves S_v and S_u using precomputed arrays,
    at a given landmark t_day. Returns x (days since landmark), S_v, S_u.
    """
    t = t_day - FIRST_ELIG_DAY
    t_excl = min(D_DIM - 1, t + IMMUNITY_LAG)

    N_v = counts_cumv[min(t, V_DIM - 2)]
    died_to_t_v = deaths_cumd_prefv[min(t, V_DIM - 2), t_excl] if N_v > 0 else 0
    N_v -= died_to_t_v

    v_start_u = min(t + 1, V_DIM - 1)
    N_u = counts_sufv[v_start_u]
    died_to_t_u = deaths_cumd_sufv[v_start_u, t_excl] if N_u > 0 else 0
    N_u -= died_to_t_u

    if N_v <= 0 or N_u <= 0:
        return np.array([0]), np.array([1.0]), np.array([1.0])

    E_v = deaths_prefv[min(t, V_DIM - 2), :].copy()
    E_u = deaths_sufv[v_start_u, :].copy()
    if t_excl >= 0:
        E_v[:(t_excl + 1)] = 0
        E_u[:(t_excl + 1)] = 0

    cumE_v = deaths_cumd_prefv[min(t, V_DIM - 2), :] - deaths_cumd_prefv[min(t, V_DIM - 2), t_excl]
    cumE_u = deaths_cumd_sufv[v_start_u, :]           - deaths_cumd_sufv[v_start_u, t_excl]

    R_v = N_v - np.concatenate([[0], cumE_v[:-1]])
    R_u = N_u - np.concatenate([[0], cumE_u[:-1]])
    R_v = np.clip(R_v, 1e-12, None)
    R_u = np.clip(R_u, 1e-12, None)

    start = t_excl + 1
    if start >= D_DIM:
        return np.array([0]), np.array([1.0]), np.array([1.0])

    h_v_seg = E_v[start:] / R_v[start:]
    h_u_seg = E_u[start:] / R_u[start:]
    S_v = survival_from_hazard(h_v_seg)
    S_u = survival_from_hazard(h_u_seg)

    x = np.arange(len(h_v_seg))  # integer days since landmark
    return x, S_v, S_u

# ---------------- Plot: ΔRMST(t) + shaded survival curves ----------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=False,
    specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    row_heights=[0.45, 0.55],
    vertical_spacing=0.10
)

# Panel 1: ΔRMST(t)
fig.add_trace(
    go.Scatter(
        x=res_df["t_landmark"],
        y=res_df["delta_rmst"],
        name="ΔRMST(t)",
        mode="lines+markers",
        line=dict(color="darkgreen", width=2),
        marker=dict(size=6)
    ),
    row=1, col=1
)
fig.update_yaxes(title_text="ΔRMST(t) [days]", row=1, col=1)
fig.update_xaxes(title_text="Landmark day", row=1, col=1)

# Panel 2: Survival curves per landmark with shaded ΔRMST area
for rec in results:
    t_day = rec["t_landmark"]
    x, S_v, S_u = survival_curves_at_landmark(t_day)
    if len(x) <= 1:
        continue

    name_base = f"KM @ day {t_day}"

    # Vaccinated curve
    fig.add_trace(
        go.Scatter(
            x=x, y=S_v, mode="lines",
            name=f"{name_base} | Vaccinated",
            line=dict(color="steelblue"),
            visible="legendonly",
            hovertemplate=(
                f"Landmark day: {t_day}<br>"
                "Days since landmark: %{x}<br>"
                "Survival (vaccinated): %{y:.3f}<extra></extra>"
            )
        ),
        row=2, col=1
    )

    # Not-yet-vaccinated curve
    fig.add_trace(
        go.Scatter(
            x=x, y=S_u, mode="lines",
            name=f"{name_base} | Not yet vaccinated",
            line=dict(color="orange"),
            visible="legendonly",
            hovertemplate=(
                f"Landmark day: {t_day}<br>"
                "Days since landmark: %{x}<br>"
                "Survival (unvaccinated): %{y:.3f}<extra></extra>"
            )
        ),
        row=2, col=1
    )

    # Shaded area between curves (ΔRMST visual)
    x_poly = np.concatenate([x, x[::-1]])
    y_poly = np.concatenate([S_v, S_u[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_poly, y=y_poly,
            name=f"{name_base} | Area between (ΔRMST)",
            fill="toself", mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(34,139,34,0.25)",
            visible="legendonly",
            hovertemplate=(
                f"Landmark day: {t_day}<br>"
                "Days since landmark: %{x}<br>"
                "Area equals ΔRMST up to horizon<extra></extra>"
            )
        ),
        row=2, col=1
    )

fig.update_yaxes(title_text="Survival probability (empirical)", range=[0,1], row=2, col=1)
fig.update_xaxes(title_text="Days since landmark", row=2, col=1)

fig.update_layout(
    title="Gold-standard Empirical Landmark: ΔRMST(t) and Survival Curves (shaded ΔRMST area)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
)

fig.write_html(PLOT_PATH)
log(f"Saved interactive HTML: {PLOT_PATH}")

# ---------------- Final summary logging ----------------
log("-"*74)
log("Empirical Landmark RMST summary (non-parametric, unbiased)")
log(f"Total subjects: {len(raw)}")
log(f"Landmark window: {FIRST_ELIG_DAY}..{LAST_OBS_DAY} ({len(DAYS)} days)")
log(f"Total vaccinated (any time): {np.isfinite(raw['vax_day']).sum()}")
log(f"Total deaths: {np.isfinite(raw['death_day']).sum()}")
log(f"Landmarks computed (valid risk sets): {len(res_df)}")
log(f"Median ΔRMST: {res_df['delta_rmst'].median():+.2f} days")
log(f"Mean ΔRMST: {res_df['delta_rmst'].mean():+.2f} days")
log(f"Median N_v: {res_df['n_v'].median():.0f}; Median N_u: {res_df['n_u'].median():.0f}")
log(f"Median events_v: {res_df['events_v'].median():.0f}; Median events_u: {res_df['events_u'].median():.0f}")
log("-"*74)

print(f"\nDONE ✓\nCSV: {CSV_PATH}\nPlot: {PLOT_PATH}\nLog: {LOG_PATH}")
listener.stop()
