#!/usr/bin/env python3
"""
Pull weather_cleaned.csv (or raw weather.csv fallback), time-aligned to the
same lap windows already inspected in laps_features.csv - step 3, still
read-only, no fix.

Standalone. Does not modify hermes_master.py or any weather module file.

WHY THIS IS NECESSARY (confirmed by the previous step, not assumed):
'Rainfall' does not exist as a column in laps_features.csv at all - it lives
in weather.csv / weather_cleaned.csv, a separate, TIME-indexed file
hermes_master.py never loads for its per-lap decisions. This pulls that
real file directly and aligns it to the same three lap windows already
inspected, by real session time (laps_features.csv's own Time column), not
by guessing a lap-to-time offset.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", ".")).resolve()
HERMES_MASTER_PATH = Path(os.environ.get("HERMES_MASTER_PATH", "system/HERMES/trees/master.py"))
SEASON, RACE, SESSION = 2024, "British_Grand_Prix", "R"
D1, D2 = "VER", "PER"

WINDOWS = [
    ("wet crossover (lap 19)", range(15, 23)),
    ("wet crossover (lap 26)", range(22, 30)),
    ("drying crossover (laps 37-38)", range(34, 42)),
]
PAD_SECONDS = 90  # pad either side of the lap window's own time span


def die(msg):
    print(msg)
    sys.exit(1)


if not HERMES_MASTER_PATH.exists():
    die(f"Could not find {HERMES_MASTER_PATH.resolve()} - set HERMES_MASTER_PATH.")

spec = importlib.util.spec_from_file_location("hermes_master_under_test", HERMES_MASTER_PATH)
hm = importlib.util.module_from_spec(spec)
sys.modules["hermes_master_under_test"] = hm
spec.loader.exec_module(hm)

laps_path = hm.find_laps_features(SEASON, RACE, SESSION)
if laps_path is None:
    die(f"laps_features.csv not found for {SEASON} {RACE} {SESSION}.")
laps = pd.read_csv(laps_path)

weather = hm.load_weather(SEASON, RACE, SESSION)
if weather is None:
    die(f"No weather_cleaned.csv or raw weather.csv found for {SEASON} {RACE} {SESSION} - "
        f"checked CACHE_DIR/clean/fastf1/.../weather_cleaned.csv and "
        f"CACHE_DIR/raw/fastf1/.../weather.csv. If your weather file lives somewhere else, "
        f"edit hm.load_weather's candidate paths or point CACHE_DIR at the right root.")

print("=" * 100)
print(f"weather file columns found: {list(weather.columns)}")
print("=" * 100)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


laps_time_col = find_col(laps, ["Time", "LapStartTime"])
weather_time_col = find_col(weather, ["time_seconds", "Time"])
if laps_time_col is None or weather_time_col is None:
    print(f"laps columns available: {list(laps.columns)}")
    die("Could not find a usable time column on both sides - cannot align by time without one.")
print(f"aligning laps.{laps_time_col!r} against weather.{weather_time_col!r}")


def to_seconds(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


laps["_time_s"] = to_seconds(laps[laps_time_col])
weather["_time_s"] = to_seconds(weather[weather_time_col])

rain_cols = [c for c in weather.columns if any(kw in c.lower() for kw in
             ("rain", "onset", "wet"))]
print(f"rain-related columns in weather data: {rain_cols}")
if not rain_cols:
    print("WARNING: no rain-related column found at all in this weather file - if this is the "
          "raw weather.csv fallback (not weather_cleaned.csv), is_rain_onset/is_rain_end won't "
          "exist yet (they're added by your cleaning pipeline) - only the raw Rainfall flag will.")

context_cols = [c for c in weather.columns if c.lower() in ("airtemp", "tracktemp", "humidity")]
show_cols = ["_time_s"] + [c for c in dict.fromkeys(rain_cols + context_cols) if c in weather.columns]

for label, lap_range in WINDOWS:
    print("\n" + "=" * 100)
    print(f"WINDOW: {label}")
    print("=" * 100)
    for driver in (D1, D2):
        driver_laps = laps[(laps["Driver"] == driver) & (laps["LapNumber"].isin(lap_range))]
        if driver_laps.empty or driver_laps["_time_s"].isna().all():
            print(f"\n--- {driver}: no valid lap-time data in this window ---")
            continue
        t0 = driver_laps["_time_s"].min() - PAD_SECONDS
        t1 = driver_laps["_time_s"].max() + PAD_SECONDS
        print(f"\n--- {driver}: real session time window {t0:.0f}s-{t1:.0f}s "
              f"(laps {lap_range.start}-{lap_range.stop - 1}) ---")
        w = weather[(weather["_time_s"] >= t0) & (weather["_time_s"] <= t1)].sort_values("_time_s")
        if w.empty:
            print("  (no weather samples in this time range - check sampling density/time alignment)")
        else:
            print(w[show_cols].to_string(index=False))

        for _, lap_row in driver_laps.iterrows():
            lap_t = lap_row["_time_s"]
            if pd.isna(lap_t) or weather["_time_s"].isna().all():
                continue
            nearest_idx = (weather["_time_s"] - lap_t).abs().idxmin()
            nearest = weather.loc[nearest_idx]
            rain_vals = {c: nearest.get(c) for c in rain_cols}
            print(f"    lap {int(lap_row['LapNumber'])} (t={lap_t:.0f}s) -> nearest weather sample "
                  f"(t={nearest['_time_s']:.0f}s, delta={abs(nearest['_time_s'] - lap_t):.0f}s): {rain_vals}")

print("\n" + "=" * 100)
print("No fix applied. This is the REAL, time-aligned weather signal - compare it against")
print("the earlier laps_features.csv-only pull to see exactly what Tier 1 is currently blind")
print("to, and whether the real Rainfall/is_rain_onset/is_rain_end sequence actually confirms")
print("a wet/dry transition at these laps at all before scoping any integration fix.")
print("=" * 100)