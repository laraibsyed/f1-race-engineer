""

import os
import sys
from typing import Optional
import pandas as pd
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drying_line import (
    evaluate_drying_crossover, LapTimeSample, MIN_SECONDS_SINCE_RAIN_STOPPED,
    PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS,
)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")

RACE = {
    "name": "2022 Monaco Grand Prix",
    "season": "2022",
    "race_folder": "Monaco_Grand_Prix",
    "session": "R",

    "real_switches": {

        "GAS": (4, "INTERMEDIATE"),
        "VET": (6, "INTERMEDIATE"),
        "TSU": (6, "INTERMEDIATE"),
        "HAM": (16, "INTERMEDIATE"),
        "PER": (16, "INTERMEDIATE"),
        "NOR": (18, "INTERMEDIATE"),
        "LEC": (18, "INTERMEDIATE"),
        "VER": (18, "INTERMEDIATE"),
    },
}

class CachedBucket:
    ""

    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=CACHE_DIR):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def _local_path(self, blob_path):
        return os.path.join(self.cache_dir, blob_path)

    def read_csv(self, blob_path):
        local_path = self._local_path(blob_path)
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        blob = self.bucket.blob(blob_path)
        data = blob.download_as_bytes()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        import io
        return pd.read_csv(io.BytesIO(data))

def parse_timedelta_seconds(series):
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

def find_rain_end_time(weather_df: pd.DataFrame) -> list:
    ""
    w = weather_df.copy()
    if "time_seconds" not in w.columns:
        w["time_seconds"] = parse_timedelta_seconds(w["Time"])
    w = w.sort_values("time_seconds").reset_index(drop=True)

    min_sustained_dry_seconds = 150

    sustained_rain_ends = []
    rain_end_indices = w.index[w["is_rain_end"] == True].tolist()
    for idx in rain_end_indices:
        end_time = w.loc[idx, "time_seconds"]

        future = w[w["time_seconds"] > end_time]
        next_rain_onset = future[future["Rainfall"] == True]
        if next_rain_onset.empty:

            sustained_rain_ends.append(end_time)
        else:
            next_onset_time = next_rain_onset["time_seconds"].iloc[0]
            if (next_onset_time - end_time) >= min_sustained_dry_seconds:
                sustained_rain_ends.append(end_time)

    return sustained_rain_ends

def most_recent_rain_end_before(rain_end_times: list, at_time_seconds: float) -> float:
    ""
    if not rain_end_times:
        return None
    eligible = [t for t in rain_end_times if t <= at_time_seconds]
    if not eligible:
        return None
    return max(eligible)

def build_driver_compound_state(laps_df: pd.DataFrame) -> pd.DataFrame:
    ""
    df = laps_df.copy()
    df["time_seconds"] = parse_timedelta_seconds(df["Time"])
    df["lap_time_seconds"] = parse_timedelta_seconds(df["LapTime"])

    exclude_cols = ["is_pit_in", "is_pit_out", "is_sc_lap", "is_vsc_lap"]
    exclusion_mask = pd.Series(False, index=df.index)
    for col in exclude_cols:
        if col in df.columns:
            exclusion_mask |= df[col].fillna(False).astype(bool)
    df["is_valid_reference_lap"] = ~exclusion_mask & df["lap_time_seconds"].notna()

    return df

def find_representative_switch_lap(laps_df: pd.DataFrame, driver: str,
                                    switch_lap: int, compound: str,
                                    max_laps_forward: int = 4) -> Optional[pd.Series]:
    ""
    driver_laps = laps_df[
        (laps_df["Driver"] == driver) & (laps_df["Compound"] == compound)
    ].sort_values("LapNumber")

    candidates = driver_laps[
        (driver_laps["LapNumber"] >= switch_lap)
        & (driver_laps["LapNumber"] <= switch_lap + max_laps_forward)
    ]
    for _, row in candidates.iterrows():

        is_transition_lap = bool(row.get("is_pit_in", False)) or bool(row.get("is_pit_out", False))
        has_valid_time = pd.notna(row.get("lap_time_seconds"))
        if not is_transition_lap and has_valid_time:
            return row
    return None

def run_backtest():
    print(f"\n{'='*70}")
    print(f"  {RACE['name']}")
    print(f"{'='*70}")

    bucket = CachedBucket()
    weather_path = f"clean/fastf1/{RACE['season']}/{RACE['race_folder']}/{RACE['session']}/weather_cleaned.csv"
    laps_path = f"clean/features/{RACE['season']}/{RACE['race_folder']}/{RACE['session']}/laps_features.csv"

    try:
        weather_df = bucket.read_csv(weather_path)
    except Exception as e:
        print(f"  Could not load weather file at {weather_path}: {e}")
        return
    try:
        laps_df = bucket.read_csv(laps_path)
    except Exception as e:
        print(f"  Could not load laps file at {laps_path}: {e}")
        return

    rain_end_times = find_rain_end_time(weather_df)
    if not rain_end_times:
        print("  No is_rain_end event found in weather data -- cannot run "
              "the drying-crossover backtest (this module needs a wet-to-dry "
              "transition, which requires rain to have been detected and "
              "then stopped in the weather file).")
        return
    print(f"  Rain stop events detected at session times: "
          f"{[f'{t:.0f}s' for t in rain_end_times]}")
    print(f"  ({len(rain_end_times)} separate is_rain_end events -- likely "
          f"sensor toggling on residual drizzle rather than {len(rain_end_times)} "
          f"genuinely separate rain spells; using the most recent one before "
          f"each lap, not a single global value -- see find_rain_end_time() docstring)")

    laps_df = build_driver_compound_state(laps_df)
    total_laps = int(laps_df["LapNumber"].max())

    print(f"  total laps: {total_laps}")
    print(f"  gate threshold: {MIN_SECONDS_SINCE_RAIN_STOPPED}s since rain stopped")
    print(f"  performance evidence threshold: {PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS}s "
          f"(switched driver must be this much faster than wet-tyre reference median)")

    print(f"\n  Lap-by-lap check for each driver with a documented real switch:\n")
    header = f"  {'Driver':<8}{'Lap':<6}{'SecSinceRainEnd':<18}{'GateOpen':<10}{'DriverLapTime':<15}{'RefMedian':<12}{'Delta':<9}{'ModuleState'}"
    print(header)

    results_summary = []

    for driver, (real_switch_lap, target_compound) in RACE["real_switches"].items():
        driver_laps = laps_df[laps_df["Driver"] == driver].sort_values("LapNumber")
        if driver_laps.empty:
            print(f"  {driver:<8} -- no lap data found at all, skipping")
            continue

        switch_row = find_representative_switch_lap(
            laps_df, driver, real_switch_lap, target_compound
        )
        if switch_row is None:
            print(f"  {driver:<8}{real_switch_lap:<6}-- no representative "
                  f"(non-transition) lap found on {target_compound} shortly "
                  f"after the documented switch, skipping")
            continue

        evaluated_lap = int(switch_row["LapNumber"])
        lap_note = "" if evaluated_lap == real_switch_lap else f" (evaluated lap {evaluated_lap}, not the pit-in lap {real_switch_lap})"

        lap_time_seconds_at_switch = switch_row["time_seconds"]
        applicable_rain_end = most_recent_rain_end_before(rain_end_times, lap_time_seconds_at_switch)
        if applicable_rain_end is None:
            print(f"  {driver:<8}{real_switch_lap:<6}-- this lap happened before ANY "
                  f"is_rain_end event -- rain hadn't stopped yet by this clock, "
                  f"skipping (check race_date/session alignment if this is unexpected)")
            continue
        seconds_since_rain_end = lap_time_seconds_at_switch - applicable_rain_end

        same_lap_others = laps_df[
            (laps_df["LapNumber"] == evaluated_lap)
            & (laps_df["Driver"] != driver)
            & (laps_df["is_valid_reference_lap"])
            & (laps_df["Compound"].isin(["WET", "INTERMEDIATE"]))
        ]
        reference_laps = [
            LapTimeSample(driver=r["Driver"], lap_time_seconds=r["lap_time_seconds"],
                           compound=r["Compound"])
            for _, r in same_lap_others.iterrows()
        ]

        switched_sample = LapTimeSample(
            driver=driver, lap_time_seconds=switch_row["lap_time_seconds"],
            compound=switch_row["Compound"],
        )

        result = evaluate_drying_crossover(
            seconds_since_rain_end=seconds_since_rain_end,
            switched_driver_lap=switched_sample,
            reference_laps=reference_laps if reference_laps else None,
        )

        ref_median = "n/a"
        if result.performance_delta_seconds is not None and reference_laps:
            ref_median = f"{switch_row['lap_time_seconds'] - result.performance_delta_seconds:.2f}"

        delta_display = f"{result.performance_delta_seconds:.2f}" if result.performance_delta_seconds is not None else "n/a"

        print(f"  {driver:<8}{evaluated_lap:<6}{seconds_since_rain_end:<18.0f}"
              f"{str(result.gate_open):<10}{switch_row['lap_time_seconds']:<15.2f}"
              f"{ref_median:<12}{delta_display:<9}{result.state.value}{lap_note}")

        results_summary.append((driver, real_switch_lap, result))

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    n_confirmed = sum(1 for _, _, r in results_summary if r.state.value == "CONSIDER_DRIER_TYRE")
    n_total = len(results_summary)
    print(f"  Module reached CONSIDER_DRIER_TYRE for {n_confirmed}/{n_total} "
          f"documented real switches, evaluated AT the lap the real switch happened.")
    print(f"  This checks: 'if this module had been watching at the moment "
          f"a real driver switched, would it have agreed the evidence was "
          f"there?' It does NOT check whether the module would have flagged "
          f"the switch EARLIER or LATER than the real driver -- that would "
          f"need scanning every lap before the real switch too, which is a "
          f"natural next step if this result looks promising.")
    print(f"  Known limitation: reference-lap filtering excludes in/out/SC/VSC "
          f"laps but does NOT control for driver skill or car pace differences "
          f"(see module docstring) -- a driver who is simply faster than the "
          f"field regardless of tyre could trigger a false positive here.")

if __name__ == "__main__":
    run_backtest()
