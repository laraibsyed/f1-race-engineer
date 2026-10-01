
import glob
import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

tqdm.pandas()
sns.set_theme(style="whitegrid")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")
FEATURES_GLOB = f"{CACHE_DIR}/clean/features/*/*/*/laps_features.csv"

_stage_t0 = {}

def stage_start(label: str):
    _stage_t0[label] = time.time()
    print(f"\n[{label}] starting...")

def stage_end(label: str):
    print(f"[{label}] done in {time.time() - _stage_t0[label]:.1f}s")

TIMEDELTA_COLS = [
    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "PitInTime", "PitOutTime", "LapStartTime",
]

def coerce_timedelta_cols(frame: pd.DataFrame) -> pd.DataFrame:
    for col in TIMEDELTA_COLS:
        if col in frame.columns and not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = pd.to_timedelta(frame[col], errors="coerce").dt.total_seconds()
    return frame

def load_all_features(pattern: str = FEATURES_GLOB) -> pd.DataFrame:
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"No files matched {pattern}. Check GCS_CACHE_DIR or run the pipeline's "
            "coverage_check.py first to populate the local cache."
        )
    frames = []

    for p in tqdm(paths, desc="Loading sessions", unit="session"):
        parts = p.split(os.sep)
        year, race, session = parts[-4], parts[-3], parts[-2]
        try:
            df = pd.read_csv(p, dtype={"TrackStatus": str})
        except UnicodeDecodeError:
            df = pd.read_csv(p, dtype={"TrackStatus": str}, encoding="latin-1")
        df = coerce_timedelta_cols(df)
        df["season"] = int(year)
        df["race"] = race
        df["session"] = session
        frames.append(df)
    master = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(paths)} sessions, {len(master):,} driver-lap rows "
          f"(LapTime now in seconds, not Timedelta strings).")
    return master

stage_start("0. Load")
df = load_all_features()
stage_end("0. Load")

race_df = df[df["session"].isin(["R", "S"])].copy()

stage_start("1. Lap time distributions")
clean_pace = race_df[
    (~race_df["is_pit_in"]) & (~race_df["is_pit_out"])
    & (~race_df["is_out_lap"]) & (~race_df["is_in_lap"])
    & (~race_df["is_outlier_laptime"]) & (~race_df["is_missing_laptime"])
].copy()

fig1a, ax1a = plt.subplots(figsize=(9, 6))
sns.boxplot(data=clean_pace, x="Compound", y="LapTime", ax=ax1a)
ax1a.set_title("Lap time distribution by compound")
ax1a.set_ylabel("Lap time (s)")
ax1a.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("eda_1a_laptime_by_compound.png", dpi=120)
plt.show()

team_order = clean_pace.groupby("Team")["LapTime"].median().sort_values().index
fig1b, ax1b = plt.subplots(figsize=(12, 6))
sns.boxplot(data=clean_pace, x="Team", y="LapTime", order=team_order, ax=ax1b)
ax1b.set_title("Lap time distribution by team")
ax1b.set_ylabel("Lap time (s)")
ax1b.tick_params(axis="x", rotation=75)
plt.tight_layout()
plt.savefig("eda_1b_laptime_by_team.png", dpi=120)
plt.show()

top_circuits = clean_pace["race"].value_counts().head(12).index
for circuit in top_circuits:
    circuit_data = clean_pace[clean_pace["race"] == circuit]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=circuit_data, x="Compound", y="LapTime")
    plt.title(f"Lap time distribution by compound — {circuit}")
    plt.ylabel("Lap time (s)")
    plt.tight_layout()
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", circuit).strip("_")
    plt.savefig(f"eda_1c_laptime_{safe_name}.png", dpi=120)
    plt.show()
stage_end("1. Lap time distributions")

stage_start("2. Degradation curves")
deg = clean_pace.dropna(subset=["degradation_rate", "tyre_age"])

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=deg, x="tyre_age", y="degradation_rate", hue="Compound",
    estimator="mean", errorbar=("ci", 95), ax=ax,
)
ax.set_title("Degradation rate vs tyre age, by compound (all circuits)")
plt.tight_layout()
plt.savefig("eda_2_degradation_by_compound.png", dpi=120)
plt.show()

circuit_deg = (
    deg.groupby(["race", "Compound"])["degradation_rate"]
    .mean().reset_index()
    .pivot(index="race", columns="Compound", values="degradation_rate")
)
plt.figure(figsize=(8, 10))
sns.heatmap(circuit_deg, cmap="rocket_r", annot=False)
plt.title("Mean degradation rate: circuit x compound")
plt.tight_layout()
plt.savefig("eda_2_degradation_heatmap.png", dpi=120)
plt.show()
stage_end("2. Degradation curves")

stage_start("3. SC/VSC frequency")
sc_vsc = race_df.groupby(["race"]).agg(
    sc_laps=("is_sc_lap", "sum"),
    vsc_laps=("is_vsc_lap", "sum"),
    total_laps=("LapNumber", "count"),
).reset_index()
sc_vsc["sc_pct"] = sc_vsc["sc_laps"] / sc_vsc["total_laps"] * 100
sc_vsc["vsc_pct"] = sc_vsc["vsc_laps"] / sc_vsc["total_laps"] * 100

by_season = race_df.groupby("season").agg(
    sc_events=("is_sc_deployed_lap", "sum"),
    vsc_events=("is_vsc_deployed_lap", "sum"),
).reset_index()

fig3a, ax3a = plt.subplots(figsize=(10, 6))
by_season.set_index("season")[["sc_events", "vsc_events"]].plot(kind="bar", ax=ax3a)
ax3a.set_title("SC / VSC deployments per season")
ax3a.set_ylabel("Number of events")
plt.tight_layout()
plt.savefig("eda_3a_sc_vsc_by_season.png", dpi=120)
plt.show()

race_df["race_pct"] = race_df["LapNumber"] / race_df.groupby("race")["LapNumber"].transform("max")
trigger_pct = race_df[race_df["is_sc_deployed_lap"] | race_df["is_vsc_deployed_lap"]]["race_pct"]
plt.figure(figsize=(9, 6))
sns.histplot(trigger_pct, bins=20)
plt.title("SC/VSC trigger timing (% race distance)")
plt.xlabel("Fraction of race distance")
plt.tight_layout()
plt.savefig("eda_3b_sc_vsc_trigger_timing.png", dpi=120)
plt.show()
stage_end("3. SC/VSC frequency")

def compute_undercut_attempts(race_group: pd.DataFrame) -> pd.DataFrame:
    race_group = race_group.sort_values(["Driver", "LapNumber"])
    pit_laps = race_group[race_group["is_pit_in"]][["Driver", "LapNumber", "Position", "Stint"]]
    attempts = []
    for _, row in pit_laps.iterrows():
        lap_state = race_group[race_group["LapNumber"] == row["LapNumber"]]
        ahead = lap_state[lap_state["Position"] == row["Position"] - 1]
        if ahead.empty:
            continue
        rival = ahead.iloc[0]["Driver"]
        rival_pit_soon = race_group[
            (race_group["Driver"] == rival)
            & (race_group["LapNumber"].between(row["LapNumber"], row["LapNumber"] + 1))
            & (race_group["is_pit_in"])
        ]
        if not rival_pit_soon.empty:
            continue
        later = race_group[
            (race_group["Driver"] == row["Driver"])
            & (race_group["LapNumber"] == row["LapNumber"] + 3)
        ]
        rival_later = race_group[
            (race_group["Driver"] == rival)
            & (race_group["LapNumber"] == row["LapNumber"] + 3)
        ]
        if later.empty or rival_later.empty:
            continue
        success = later.iloc[0]["Position"] < rival_later.iloc[0]["Position"]
        attempts.append({
            "tyre_age_at_pit": row["Stint"],
            "position_at_pit": row["Position"],
            "success": success,
        })
    return pd.DataFrame(attempts)

stage_start("4. Undercut success rate")
if "Position" in race_df.columns:
    race_groups = list(race_df.groupby(["season", "race"]))
    undercut_results = pd.concat(
        [compute_undercut_attempts(g) for _, g in tqdm(race_groups, desc="Scanning races for undercuts")],
        ignore_index=True,
    )
    print(f"Undercut attempts found: {len(undercut_results)}")
    print(undercut_results.groupby(pd.cut(undercut_results["position_at_pit"], [0, 5, 10, 20]))
          ["success"].mean())

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=undercut_results, x=pd.cut(undercut_results["position_at_pit"], [0, 3, 6, 10, 20]),
        y="success",
    )
    plt.title("Undercut success rate by track position band")
    plt.ylabel("Success rate")
    plt.tight_layout()
    plt.savefig("eda_4_undercut.png", dpi=120)
    plt.show()
else:
    print("SKIPPED — 'Position' column not found in laps_features.csv. "
          "Re-check fastf1 laps.csv export before running this section.")
stage_end("4. Undercut success rate")

stage_start("5. Pit window distribution")
compound_thresh = deg.groupby("Compound")["degradation_rate"].quantile(0.75)

def first_crossover_lap(stint_group: pd.DataFrame) -> float:
    thresh = compound_thresh.get(stint_group["Compound"].iloc[0], np.nan)
    over = stint_group[stint_group["degradation_rate"] > thresh]
    return over["tyre_age"].min() if not over.empty else np.nan

proxy = (
    deg.groupby(["season", "race", "Driver", "Stint"])
    .progress_apply(first_crossover_lap, include_groups=False)
    .reset_index(name="proxy_optimal_tyre_age")
)
actual_pit_age = (
    race_df[race_df["is_pit_in"]]
    .groupby(["season", "race", "Driver", "Stint"])["tyre_age"].max()
    .reset_index(name="actual_pit_tyre_age")
)
pit_compare = proxy.merge(actual_pit_age, on=["season", "race", "Driver", "Stint"]).dropna()
pit_compare["delta_laps"] = pit_compare["actual_pit_tyre_age"] - pit_compare["proxy_optimal_tyre_age"]

plt.figure(figsize=(8, 5))
sns.histplot(pit_compare["delta_laps"], bins=30)
plt.axvline(0, color="black", linestyle="--")
plt.title("Actual pit lap minus degradation-crossover proxy\n(+ve = pitted later than proxy suggests)")
plt.tight_layout()
plt.savefig("eda_5_pit_window.png", dpi=120)
plt.show()
stage_end("5. Pit window distribution")

stage_start("6. Correlation matrix")
numeric_cols = [
    "LapTime", "tyre_age", "compound_encoded", "stint_number",
    "gap_to_leader", "gap_to_car_ahead", "degradation_rate",
    "fuel_load_estimate", "track_temp_c",
]
numeric_cols = [c for c in numeric_cols if c in race_df.columns]
corr = race_df[numeric_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature correlation matrix (Race/Sprint sessions)")
plt.tight_layout()
plt.savefig("eda_6_correlation.png", dpi=120)
plt.show()
stage_end("6. Correlation matrix")
