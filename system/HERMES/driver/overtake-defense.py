
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CLOSE_FOLLOWING_SECONDS = 1.0

def race_name_underscore_to_space(race: str) -> str:
    ""
    return race.replace("_", " ")

def extract_lap_summary(json_bytes: bytes) -> dict:
    ""
    data = json.loads(json_bytes)
    tel = data["tel"]

    raw_distances = tel["DistanceToDriverAhead"]
    raw_speeds = tel["speed"]
    raw_driver_ahead = tel["DriverAhead"]

    mean_speed_kmh = float(np.mean([s for s in raw_speeds if s is not None])) if raw_speeds else np.nan

    valid_mask = [d not in (None, "None") for d in raw_distances]
    if not any(valid_mask):

        return {"driver_ahead": None, "min_gap_seconds": np.inf, "mean_speed_kmh": mean_speed_kmh}

    distances = np.array([raw_distances[i] for i in range(len(raw_distances)) if valid_mask[i]], dtype=float)
    speeds_at_valid = np.array([raw_speeds[i] for i in range(len(raw_speeds)) if valid_mask[i]], dtype=float)
    driver_ahead_at_valid = [raw_driver_ahead[i] for i in range(len(raw_driver_ahead)) if valid_mask[i]]

    if len(distances) == 0:
        return {"driver_ahead": None, "min_gap_seconds": np.inf, "mean_speed_kmh": mean_speed_kmh}

    min_idx = int(np.argmin(distances))
    min_distance_m = float(distances[min_idx])
    speed_at_min_kmh = float(speeds_at_valid[min_idx]) if speeds_at_valid[min_idx] > 0 else \
        float(np.mean(speeds_at_valid[speeds_at_valid > 0]) or 1.0)
    speed_at_min_mps = speed_at_min_kmh / 3.6
    gap_seconds_at_min = min_distance_m / speed_at_min_mps if speed_at_min_mps > 0 else np.inf

    valid_ahead = [d for d in driver_ahead_at_valid if d not in (None, "None")]
    driver_ahead_mode = max(set(valid_ahead), key=valid_ahead.count) if valid_ahead else None

    return {
        "driver_ahead": driver_ahead_mode,
        "min_gap_seconds": gap_seconds_at_min,
        "mean_speed_kmh": mean_speed_kmh,
    }

def list_race_telemetry_files(bucket, season: int, race_underscore: str, session: str = "Race") -> list:
    race_space = race_name_underscore_to_space(race_underscore)
    prefix = f"raw/tracinginsights/{season}/{race_space}/{session}/"
    return [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith("_tel.json")]

def build_lap_summaries_for_race(bucket, season: int, race_underscore: str, session: str = "Race") -> pd.DataFrame:
    ""
    files = list_race_telemetry_files(bucket, season, race_underscore, session)
    print(f"[load] {len(files)} telemetry files for {race_underscore} {season} {session}")

    rows = []
    for path in files:
        parts = path.split("/")
        driver = parts[-2]
        lap_number = int(parts[-1].replace("_tel.json", ""))
        blob = bucket.blob(path)
        try:
            summary = extract_lap_summary(blob.download_as_bytes())
        except Exception as e:
            print(f"[warn] failed to parse {path}: {e}")
            continue
        if summary is None:
            continue
        summary.update({"driver": driver, "lap_number": lap_number})
        rows.append(summary)

    return pd.DataFrame(rows)

def compute_cumulative_elapsed_time(laps_features_df: pd.DataFrame) -> pd.DataFrame:
    ""
    df = laps_features_df.copy()
    df["LapTime_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df = df.sort_values(["Driver", "LapNumber"])
    df["cumulative_elapsed_seconds"] = df.groupby("Driver")["LapTime_seconds"].cumsum()
    df["lap_start_elapsed_seconds"] = df["cumulative_elapsed_seconds"] - df["LapTime_seconds"]
    df["lap_mid_elapsed_seconds"] = (df["lap_start_elapsed_seconds"] + df["cumulative_elapsed_seconds"]) / 2
    return df[["Driver", "LapNumber", "lap_mid_elapsed_seconds"]].rename(
        columns={"Driver": "driver", "LapNumber": "lap_number"})

def find_time_aligned_lap(defender_code: str, target_elapsed_seconds: float,
                           lap_summaries_with_time: pd.DataFrame):
    ""
    defender_rows = lap_summaries_with_time[lap_summaries_with_time["driver"] == defender_code]
    if defender_rows.empty:
        return None
    idx = (defender_rows["lap_mid_elapsed_seconds"] - target_elapsed_seconds).abs().idxmin()
    return defender_rows.loc[idx]

def _is_unknown(value) -> bool:
    ""
    return isinstance(value, float) and np.isnan(value)

def reconstruct_overtakes_and_defenses(lap_summaries: pd.DataFrame,
                                         pit_laps: set,
                                         laps_features_df: pd.DataFrame,
                                         close_following_seconds: float = CLOSE_FOLLOWING_SECONDS) -> pd.DataFrame:
    ""
    lap_summaries = lap_summaries.sort_values(["driver", "lap_number"]).reset_index(drop=True)
    elapsed = compute_cumulative_elapsed_time(laps_features_df)
    lap_summaries = lap_summaries.merge(elapsed, on=["driver", "lap_number"], how="left")

    events = []

    for driver, g in lap_summaries.groupby("driver"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            lap_now, lap_next = g.loc[i, "lap_number"], g.loc[i + 1, "lap_number"]
            if lap_next != lap_now + 1:
                continue
            if (driver, lap_now) in pit_laps or (driver, lap_next) in pit_laps:
                continue

            ahead_now, ahead_next = g.loc[i, "driver_ahead"], g.loc[i + 1, "driver_ahead"]
            if _is_unknown(ahead_now) or _is_unknown(ahead_next):
                continue
            if ahead_now is not None and ahead_now != ahead_next:
                events.append({"driver": driver, "lap": lap_now, "event": "overtake_made", "rival": ahead_now})

    close_followers = lap_summaries[lap_summaries["min_gap_seconds"] < close_following_seconds]
    for _, follower_row in close_followers.iterrows():
        follower = follower_row["driver"]
        defender = follower_row["driver_ahead"]
        follower_lap_number = follower_row["lap_number"]
        follower_elapsed = follower_row["lap_mid_elapsed_seconds"]
        if defender is None or pd.isna(defender) or pd.isna(follower_elapsed):
            continue

        defender_lap_row = find_time_aligned_lap(defender, follower_elapsed, lap_summaries)
        if defender_lap_row is None:
            continue

        next_lap_follower = lap_summaries[(lap_summaries["driver"] == follower)
                                           & (lap_summaries["lap_number"] == follower_lap_number + 1)]
        if next_lap_follower.empty:
            continue
        still_behind_same_car = next_lap_follower["driver_ahead"].iloc[0] == defender
        events.append({"driver": defender, "lap": defender_lap_row["lap_number"],
                        "event": "defense_held" if still_behind_same_car else "defense_lost",
                        "rival": follower})

    return pd.DataFrame(events)

def run_proof_of_concept(bucket, season: int, race: str, session: str = "Race"):
    ""
    print(f"\n{'='*70}\n=== Proof of concept: {race} {season} ===\n{'='*70}")

    pit_path = f"clean/features/{season}/{race}/R/laps_features.csv"
    pit_blob = bucket.blob(pit_path)
    pit_local = f"/tmp/pit_check_{season}_{race}.csv"
    pit_blob.download_to_filename(pit_local)
    pit_df = pd.read_csv(pit_local, usecols=["Driver", "DriverNumber", "LapNumber", "LapTime",
                                              "is_pit_in", "is_pit_out"])

    pit_laps = set(
        pit_df.loc[pit_df["is_pit_in"] | pit_df["is_pit_out"], ["Driver", "LapNumber"]]
        .itertuples(index=False, name=None)
    )
    print(f"[pit laps] {len(pit_laps)} pit in/out laps excluded from overtake detection")

    lap_summaries = build_lap_summaries_for_race(bucket, season, race, session)
    print(f"[summary] {len(lap_summaries)} driver-laps summarized (BEFORE id-mapping fix)")

    number_to_code = dict(zip(pit_df["DriverNumber"].astype(str), pit_df["Driver"]))

    def _map_driver_ahead(value):
        if value is None:
            return None
        return number_to_code.get(str(value), np.nan)

    lap_summaries["driver_ahead"] = lap_summaries["driver_ahead"].apply(_map_driver_ahead)
    n_real = lap_summaries["driver_ahead"].apply(lambda x: x is not None and not _is_unknown(x)).sum()
    n_confirmed_none = lap_summaries["driver_ahead"].apply(lambda x: x is None).sum()
    n_unknown = lap_summaries["driver_ahead"].apply(_is_unknown).sum()
    print(f"[fix] driver_ahead mapped - {n_real} real driver codes, {n_confirmed_none} confirmed "
          f"'no one ahead' laps, {n_unknown} genuinely unmapped")

    events = reconstruct_overtakes_and_defenses(lap_summaries, pit_laps, laps_features_df=pit_df)
    print(f"[events] {len(events)} total events reconstructed")
    print(events["event"].value_counts().to_string())

    print(f"\n=== Per-driver summary: {race} {season} ===")
    per_driver = events.groupby(["driver", "event"]).size().unstack(fill_value=0)
    print(per_driver.to_string())

    out_path = f"overtake_defense_events_{season}_{race}.csv"
    events.to_csv(out_path, index=False)
    print(f"[save] {out_path}")

    return events, per_driver

if __name__ == "__main__":
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    TEST_CASES = [
        (2018, "Abu_Dhabi_Grand_Prix"),
        (2019, "Monaco_Grand_Prix"),
        (2023, "Bahrain_Grand_Prix"),
    ]

    results = {}
    for season, race in TEST_CASES:
        try:
            events, per_driver = run_proof_of_concept(bucket, season, race)
            results[(season, race)] = events
        except Exception as e:
            print(f"\n[ERROR] {race} {season} failed: {type(e).__name__}: {e}")
            print("[note] a failure on a NEW race/era is exactly what this multi-race test is "
                  "for - a real, previously-unseen edge case, not a reason to skip investigating")

    print(f"\n{'='*70}\n=== CROSS-RACE SANITY CHECK ===\n{'='*70}")
    print("(Monaco should show noticeably LOWER total overtakes than Abu Dhabi/Bahrain,")
    print(" and noticeably HIGHER defense_held rates - real-world knowledge about")
    print(" street-circuit overtaking difficulty. If it doesn't, investigate before trusting.)")
    for (season, race), events in results.items():
        if events.empty:
            continue
        counts = events["event"].value_counts()
        total_overtakes = counts.get("overtake_made", 0)
        total_defenses = counts.get("defense_held", 0) + counts.get("defense_lost", 0)
        defense_hold_rate = counts.get("defense_held", 0) / total_defenses if total_defenses > 0 else float("nan")
        print(f"  {race} {season}: {total_overtakes} overtakes, "
              f"{defense_hold_rate*100:.1f}% defense-hold rate ({total_defenses} total defense events)")
    print("\n[save] overtake_defense_events_poc.csv")
