""

import io
import json
import os
import sys
import urllib.request
import urllib.parse

import pandas as pd
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossover import evaluate_crossover, PERTURBATION_SETS, RaceStage
from openmeteo import CIRCUIT_COORDS

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")
OPENMETEO_CACHE_DIR = os.path.join(CACHE_DIR, "openmeteo_historical")

SEVERITY_CAP_MM_HR = 7.6

def fetch_historical_openmeteo(race_name: str, date_str: str) -> dict:
    ""
    cache_file = os.path.join(OPENMETEO_CACHE_DIR, f"{race_name}_{date_str}_v2.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    if race_name not in CIRCUIT_COORDS:
        raise ValueError(f"'{race_name}' not in CIRCUIT_COORDS (fetch_openmeteo_weather.py) "
                          f"-- check spelling matches your bucket folder naming exactly.")
    lat, lon = CIRCUIT_COORDS[race_name]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "precipitation,rain,temperature_2m,relative_humidity_2m",
        "timezone": "UTC",

    }
    url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    os.makedirs(OPENMETEO_CACHE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data

def build_continuous_rain_signal(openmeteo_data: dict) -> pd.DataFrame:
    ""
    hourly = openmeteo_data["hourly"]
    df = pd.DataFrame({
        "hour_timestamp": pd.to_datetime(hourly["time"]),
        "rain_mm": hourly["rain"],
    })
    df["severity_0_100"] = (df["rain_mm"] / SEVERITY_CAP_MM_HR * 100).clip(upper=100)
    return df

RACES = [
    {
        "name": "2021 Belgian Grand Prix (Spa)",
        "season": "2021",
        "race_folder": "Belgian_Grand_Prix",
        "session": "R",
        "race_date": "2021-08-29",
        "race_start_utc": "13:00",

        "actual_inter_lap": None,

    },
    {
        "name": "2022 Monaco Grand Prix",
        "season": "2022",
        "race_folder": "Monaco_Grand_Prix",
        "session": "R",
        "race_date": "2022-05-29",
        "race_start_utc": "13:00",

        "actual_inter_lap": None,

    },
    {
        "name": "2021 Russian Grand Prix (Sochi)",
        "season": "2021",
        "race_folder": "Russian_Grand_Prix",
        "session": "R",
        "race_date": "2021-09-26",
        "race_start_utc": "12:00",

        "actual_inter_lap": 50,
    },
]

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
    return pd.to_timedelta(series).dt.total_seconds()

def build_lap_rain_timeline(weather_df: pd.DataFrame, laps_df: pd.DataFrame,
                             openmeteo_signal: pd.DataFrame = None,
                             race_date: str = None, race_start_utc: str = None) -> pd.DataFrame:
    ""
    weather_df = weather_df.copy()
    if "time_seconds" not in weather_df.columns:
        weather_df["time_seconds"] = parse_timedelta_seconds(weather_df["Time"])
    weather_df = weather_df.sort_values("time_seconds")

    laps_df = laps_df.copy()
    laps_df["time_seconds"] = parse_timedelta_seconds(laps_df["Time"])
    lap_timeline = (
        laps_df.groupby("LapNumber")["time_seconds"]
        .min()
        .reset_index()
        .sort_values("LapNumber")
    )

    total_laps = int(lap_timeline["LapNumber"].max())

    merged = pd.merge_asof(
        lap_timeline.sort_values("time_seconds"),
        weather_df[["time_seconds", "Rainfall", "is_rain_onset"]].sort_values("time_seconds"),
        on="time_seconds",
        direction="backward",
    )
    merged["rain_probability_proxy"] = merged["Rainfall"].map({True: 100.0, False: 0.0})

    merged["rain_severity_openmeteo"] = None
    if openmeteo_signal is not None:
        if not (race_date and race_start_utc):
            print("  WARNING: race_date/race_start_utc not set in RACES -- cannot "
                  "reconstruct calendar time to join Open-Meteo's hourly data. "
                  "Falling back to FastF1 binary proxy only for this race.")
        else:
            session_start = pd.Timestamp(f"{race_date}T{race_start_utc}:00", tz="UTC")
            merged["calendar_time"] = session_start + pd.to_timedelta(merged["time_seconds"], unit="s")

            om = openmeteo_signal.copy()
            om["hour_timestamp"] = pd.to_datetime(om["hour_timestamp"], utc=True)
            om = om.sort_values("hour_timestamp")

            joined = pd.merge_asof(
                merged.sort_values("calendar_time"), om[["hour_timestamp", "severity_0_100"]],
                left_on="calendar_time", right_on="hour_timestamp",
                direction="nearest", tolerance=pd.Timedelta("2h"),
            )
            merged = merged.merge(
                joined[["LapNumber", "severity_0_100"]], on="LapNumber", how="left"
            )
            merged["rain_severity_openmeteo"] = merged["severity_0_100"]
            merged = merged.drop(columns=["severity_0_100"])

    merged["total_laps"] = total_laps
    return merged.sort_values("LapNumber").reset_index(drop=True)

def run_backtest_for_race(bucket: CachedBucket, race: dict):
    print(f"\n{'='*70}")
    print(f"  {race['name']}")
    print(f"{'='*70}")

    weather_path = f"clean/fastf1/{race['season']}/{race['race_folder']}/{race['session']}/weather_cleaned.csv"
    laps_path = f"clean/features/{race['season']}/{race['race_folder']}/{race['session']}/laps_features.csv"

    try:
        weather_df = bucket.read_csv(weather_path)
    except Exception as e:
        print(f"  Could not load weather file at {weather_path}: {e}")
        print(f"  (check the path matches your actual bucket structure)")
        return
    try:
        laps_df = bucket.read_csv(laps_path)
    except Exception as e:
        print(f"  Could not load laps file at {laps_path}: {e}")
        print(f"  Trying alternate path clean/fastf1/.../laps_flagged.csv ...")
        laps_path = f"clean/fastf1/{race['season']}/{race['race_folder']}/{race['session']}/laps_flagged.csv"
        try:
            laps_df = bucket.read_csv(laps_path)
        except Exception as e2:
            print(f"  Also failed: {e2}")
            print(f"  Skipping this race -- fix the path in RACES and re-run.")
            return

    n_rain_readings = int(weather_df["Rainfall"].sum()) if "Rainfall" in weather_df.columns else 0
    if n_rain_readings == 0:
        print(f"  No rain readings found in weather data for this session -- "
              f"policy will trivially return STAY_SLICKS every lap. "
              f"Check this is really meant to be a wet race, or check the file path.")

    USE_OPENMETEO_SEVERITY = False
    openmeteo_signal = None
    if USE_OPENMETEO_SEVERITY and race.get("race_date"):
        try:
            om_data = fetch_historical_openmeteo(race["race_folder"], race["race_date"])
            openmeteo_signal = build_continuous_rain_signal(om_data)
            print(f"  Open-Meteo historical data fetched for {race['race_date']} "
                  f"(cached at {OPENMETEO_CACHE_DIR})")
        except Exception as e:
            print(f"  Could not fetch Open-Meteo historical data: {e}")

    timeline = build_lap_rain_timeline(
        weather_df, laps_df, openmeteo_signal,
        race_date=race.get("race_date"), race_start_utc=race.get("race_start_utc"),
    )
    total_laps = int(timeline["total_laps"].iloc[0])
    using_continuous = timeline["rain_severity_openmeteo"].notna().any()

    print(f"  total laps: {total_laps} | rain readings in weather file (FastF1): {n_rain_readings}")
    print(f"  signal in use for policy evaluation: "
          f"{'Open-Meteo continuous severity (0-100)' if using_continuous else 'FastF1 binary Rainfall flag (0%/100% only)'}")
    if not using_continuous:
        print(f"  NOTE: threshold sets (baseline/stricter/looser) cannot diverge "
              f"meaningfully with a binary signal -- see module docstring for why "
              f"the continuous Open-Meteo approach was tried and rejected.")
    if race.get("actual_inter_lap"):
        print(f"  actual pit-for-inters lap (ground truth, from race report): "
              f"lap {race['actual_inter_lap']}")
    else:
        print(f"  actual_inter_lap not set in RACES dict -- add it to compare "
              f"flagged laps against what really happened")

    print(f"\n  Lap-by-lap policy output (only showing laps where ANY threshold "
          f"set flags MONITOR or higher):\n")
    header = f"  {'Lap':<5}{'OMSeverity%':<12}{'FF1Proxy%':<11}"
    for name in PERTURBATION_SETS:
        header += f"{name:<14}"
    print(header)

    any_flagged = False
    for _, row in timeline.iterrows():
        lap_num = int(row["LapNumber"])
        laps_remaining = total_laps - lap_num

        om_severity = row["rain_severity_openmeteo"]
        ff1_proxy = row["rain_probability_proxy"]
        rain_prob = om_severity if pd.notna(om_severity) else ff1_proxy
        if pd.isna(rain_prob):
            continue

        results_per_set = {}
        row_has_flag = False
        for set_name, thresholds in PERTURBATION_SETS.items():
            result = evaluate_crossover(rain_prob, laps_remaining, total_laps, thresholds)
            results_per_set[set_name] = result.state.value
            if result.state.value != "STAY_SLICKS":
                row_has_flag = True

        if row_has_flag:
            any_flagged = True
            om_display = f"{om_severity:.0f}" if pd.notna(om_severity) else "n/a"
            ff1_display = f"{ff1_proxy:.0f}" if pd.notna(ff1_proxy) else "n/a"
            line = f"  {lap_num:<5}{om_display:<12}{ff1_display:<11}"
            for name in PERTURBATION_SETS:
                line += f"{results_per_set[name]:<14}"
            print(line)

    if using_continuous:
        disagreements = timeline[
            (timeline["rain_severity_openmeteo"].fillna(0) < 5) &
            (timeline["rain_probability_proxy"].fillna(0) == 100)
        ]
        if not disagreements.empty:
            disagreement_laps = disagreements["LapNumber"].astype(int).tolist()
            print(f"\n  ⚠ Cross-check: FastF1's on-track sensor recorded rain on "
                  f"lap(s) {disagreement_laps}, but Open-Meteo's reanalysis shows "
                  f"near-zero rain at that hour. Worth checking manually -- could be "
                  f"a genuinely very localised shower the circuit-wide reanalysis "
                  f"model smooths over, or a sensor blip (as seen in Abu Dhabi 2018).")

    if not any_flagged:
        print("  (no laps flagged by any threshold set -- policy stayed on "
              "STAY_SLICKS throughout, consistent with a dry session or a "
              "very brief/negligible rain reading)")

    onset_rows = timeline_with_onset = None
    onset_lap = None
    if "is_rain_onset" in weather_df.columns:
        w = weather_df.copy()
        if "time_seconds" not in w.columns:
            w["time_seconds"] = parse_timedelta_seconds(w["Time"])
        onset_reading = w[w["is_rain_onset"] == True]
        if not onset_reading.empty:
            onset_time_seconds = onset_reading["time_seconds"].iloc[0]

            laps_before_onset = timeline[timeline["time_seconds"] <= onset_time_seconds]
            onset_lap = int(laps_before_onset["LapNumber"].max()) if not laps_before_onset.empty else int(timeline["LapNumber"].min())
            print(f"\n  >> Sensor onset check: weather.csv's is_rain_onset fired at "
                  f"lap {onset_lap} (session time {onset_time_seconds:.0f}s).")

    if race.get("actual_inter_lap") and any_flagged:
        actual_lap = race["actual_inter_lap"]

        def _row_rain_prob(row):
            om = row["rain_severity_openmeteo"]
            return om if pd.notna(om) else row["rain_probability_proxy"]

        flagged_laps = [
            int(row["LapNumber"]) for _, row in timeline.iterrows()
            if not pd.isna(_row_rain_prob(row))
            and evaluate_crossover(
                _row_rain_prob(row),
                total_laps - int(row["LapNumber"]),
                total_laps,
            ).state.value != "STAY_SLICKS"
        ]
        if flagged_laps:
            first_flag = min(flagged_laps)
            gap = actual_lap - first_flag
            direction = "before" if gap > 0 else "after" if gap < 0 else "exactly at"
            print(f"\n  >> Policy (baseline) first flagged lap {first_flag}; "
                  f"real pit-for-inters was lap {actual_lap} "
                  f"-> policy flagged {abs(gap)} lap(s) {direction} the real switch.")
            if onset_lap is not None:
                policy_lag = first_flag - onset_lap
                print(f"  >> Sensor-to-policy lag: onset detected at lap {onset_lap}, "
                      f"policy flagged at lap {first_flag} "
                      f"({'0 laps -- policy reacts the instant the sensor does' if policy_lag == 0 else f'{policy_lag} lap(s) of lag between sensor onset and policy flag'}).")
                if policy_lag == 0:
                    print(f"  >> This means the full {abs(gap)}-lap gap to the real pit lap "
                          f"is NOT policy lag -- it's the gap between 'sensor first detects "
                          f"rain' and 'driver actually pits'. That's a real strategic delay "
                          f"(e.g. waiting to confirm it's not a false alarm, or track still "
                          f"marginal for inters), not a weakness in this module's logic.")
            print(f"  >> Note: FastF1's binary Rainfall flag can only mark WHEN rain "
                  f"was detected, not its build-up (\"a few drops\" vs \"falling hard\") -- "
                  f"real strategists reacted to the progressive intensification described "
                  f"in race commentary, which this proxy collapses into a single instant. "
                  f"Treat this gap as an upper bound on the policy's real responsiveness.")

def main():
    bucket = CachedBucket()
    for race in RACES:
        run_backtest_for_race(bucket, race)

    print(f"\n{'='*70}")
    print("  Reminder: rain_probability here is a 0/100 PROXY from FastF1's "
          "Rainfall flag (is it raining right now), NOT a real forecast "
          "probability. This tests the policy's reaction speed/threshold "
          "behaviour once rain has actually started -- see module docstring.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
