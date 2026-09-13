"""
Driver Taxonomy — Sampled-Race Aggression/Defense Scoring
================================================================
Builds real, data-derived aggression_level and defensive_strength scores
per driver (matching driver_archetypes.xlsx's convention), using a SAMPLED
set of races per driver rather than the full 197,909-file archive - a
deliberate scope decision given the project timeline. Sound because the
underlying reconstruction logic has already been validated to generalize
correctly across 3 genuinely different race types (2018 Abu Dhabi/Track,
2019 Monaco/Street, 2023 Bahrain/Track post-2022-regs) - Monaco correctly
showed far fewer overtakes and a far higher defense-hold rate, matching real
motorsport knowledge, including one specific, checkable match (Hamilton's
famous defensive drive at 2019 Monaco).

DUPLICATES the core reconstruction logic from overtake_defense_poc.py rather
than importing it - that file's name has a hyphen (overtake-defense.py) and
can't be imported as a Python module directly. Keep both in sync if either
changes.

SAMPLING STRATEGY: for each driver with >=20 career races, sample
N_RACES_PER_DRIVER races, STRATIFIED by circuit_type (real values confirmed
from circuit_taxonomy.xlsx: "Track" or "Street", not assumed) so a driver's
sample isn't accidentally all-one-circuit-character by chance - Monaco vs.
Bahrain just proved that would badly distort results.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CLOSE_FOLLOWING_SECONDS = 1.0
MIN_CAREER_RACES = 20    # matches the Second Driver Logic diagram's own archetype-fallback threshold
CIRCUIT_TAXONOMY_PATH = r"src\taxanomy\circuit_taxonomy.xlsx"

# Same circuit_id <-> race name mapping already built and used for the Cox
# model - reused here rather than rebuilt, for consistency.
CIRCUIT_ID_TO_RACE_NAMES = {
    "MEL": ["Australian_Grand_Prix"], "BAH": ["Bahrain_Grand_Prix", "Sakhir_Grand_Prix"],
    "CHN": ["Chinese_Grand_Prix"], "AZR": ["Azerbaijan_Grand_Prix"], "SPN": ["Spanish_Grand_Prix"],
    "MON": ["Monaco_Grand_Prix"], "CAN": ["Canadian_Grand_Prix"], "FRA": ["French_Grand_Prix"],
    "AUS": ["Austrian_Grand_Prix", "Styrian_Grand_Prix"],
    "UK": ["British_Grand_Prix", "70th_Anniversary_Grand_Prix"],
    "GER": ["German_Grand_Prix"], "HUN": ["Hungarian_Grand_Prix"], "BEL": ["Belgian_Grand_Prix"],
    "ITA": ["Italian_Grand_Prix"], "SIN": ["Singapore_Grand_Prix"], "RUS": ["Russian_Grand_Prix"],
    "JPN": ["Japanese_Grand_Prix"], "TEX": ["United_States_Grand_Prix"],
    "MEX": ["Mexican_Grand_Prix", "Mexico_City_Grand_Prix"],
    "BRA": ["Brazilian_Grand_Prix", "São_Paulo_Grand_Prix"],
    "AUH": ["Abu_Dhabi_Grand_Prix"], "IMO": ["Emilia_Romagna_Grand_Prix"],
    "IST": ["Turkish_Grand_Prix"], "DUT": ["Dutch_Grand_Prix"], "QTR": ["Qatar_Grand_Prix"],
    "KSA": ["Saudi_Arabian_Grand_Prix"], "MIA": ["Miami_Grand_Prix"], "LAS": ["Las_Vegas_Grand_Prix"],
}


class CachedBucket:
    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=os.environ.get("GCS_CACHE_DIR", "./gcs_cache")):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def read_csv(self, blob_path, **kwargs):
        local_path = os.path.join(self.cache_dir, blob_path)
        if os.path.exists(local_path):
            return pd.read_csv(local_path, **kwargs)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob = self.bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        return pd.read_csv(local_path, **kwargs)

    def list_blob_names(self, prefix):
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]


# ---------------------------------------------------------------------------
# Step 1: who qualifies, and what races has each driver actually run?
# ---------------------------------------------------------------------------
def build_driver_race_participation(bucket: CachedBucket) -> pd.DataFrame:
    """Scans every Race-session results.csv (small files, ~20 rows each -
    much lighter than telemetry) to build a full participation history."""
    paths = [p for p in bucket.list_blob_names("raw/fastf1/") if p.endswith("/R/results.csv")]
    rows = []
    for p in paths:
        parts = p.split("/")
        season, race = int(parts[2]), parts[3]
        df = bucket.read_csv(p, usecols=["Abbreviation"])
        for driver in df["Abbreviation"].dropna().unique():
            rows.append({"season": season, "race": race, "driver": driver})
    return pd.DataFrame(rows)


def get_circuit_type_lookup() -> dict:
    """race_underscore -> "Track"/"Street", via circuit_id mapping."""
    taxonomy = pd.read_excel(CIRCUIT_TAXONOMY_PATH)[["circuit_id", "circuit_type"]]
    race_to_type = {}
    for _, row in taxonomy.iterrows():
        for race_name in CIRCUIT_ID_TO_RACE_NAMES.get(row["circuit_id"], []):
            race_to_type[race_name] = row["circuit_type"]
    return race_to_type


RACES_PER_SEASON = 6     # global sample size per season - gives ~50-60 total races regardless
                          # of driver count, vs. the old up-to-468 driver-race pairs
MIN_RACES_WITH_DATA = 5  # a driver whose career barely overlaps the global sample gets flagged,
                          # not silently trusted - same small-sample caution as elsewhere in this project


def build_global_race_sample(participation: pd.DataFrame, race_to_type: dict,
                               races_per_season: int = RACES_PER_SEASON, seed: int = 42) -> list:
    """
    ARCHITECTURAL FIX: the previous design sampled races PER DRIVER, meaning
    two drivers who happened to share a sampled race caused that race's
    ~900-1500 telemetry files to be downloaded and processed TWICE - once per
    driver - despite every race's data already containing all ~20 drivers on
    track. This wasted most of the actual runtime. Fixed here: one GLOBAL,
    stratified sample of races (per season, by circuit_type), each processed
    EXACTLY ONCE, with every driver present in that race credited from the
    same single pass - a driver's effective sample size is just "however many
    of these global races their career happened to overlap," which naturally
    scales with career length without any per-driver sampling logic at all.
    """
    rng = random.Random(seed)
    races_with_type = participation.copy()
    races_with_type["circuit_type"] = races_with_type["race"].map(race_to_type)
    unique_races = races_with_type[["season", "race", "circuit_type"]].drop_duplicates().dropna(subset=["circuit_type"])

    sampled = []
    for season, season_group in unique_races.groupby("season"):
        for circuit_type, type_group in season_group.groupby("circuit_type"):
            proportion = len(type_group) / len(season_group)
            n_from_this_type = max(1, round(races_per_season * proportion))
            pool = list(type_group[["season", "race"]].itertuples(index=False, name=None))
            sampled.extend(rng.sample(pool, min(n_from_this_type, len(pool))))

    return sampled


# ---------------------------------------------------------------------------
# Step 2: the validated reconstruction logic, duplicated from overtake_defense_poc.py
# ---------------------------------------------------------------------------
def race_name_underscore_to_space(race: str) -> str:
    return race.replace("_", " ")


def extract_lap_summary(json_bytes: bytes) -> dict:
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

    return {"driver_ahead": driver_ahead_mode, "min_gap_seconds": gap_seconds_at_min, "mean_speed_kmh": mean_speed_kmh}


def build_lap_summaries_for_race(bucket: CachedBucket, season: int, race_underscore: str,
                                   session: str = "Race") -> pd.DataFrame:
    race_space = race_name_underscore_to_space(race_underscore)
    prefix = f"raw/tracinginsights/{season}/{race_space}/{session}/"
    files = [b for b in bucket.list_blob_names(prefix) if b.endswith("_tel.json")]

    rows = []
    for path in files:
        parts = path.split("/")
        driver, lap_number = parts[-2], int(parts[-1].replace("_tel.json", ""))
        blob = bucket.bucket.blob(path)
        try:
            summary = extract_lap_summary(blob.download_as_bytes())
        except Exception:
            continue
        if summary is None:
            continue
        summary.update({"driver": driver, "lap_number": lap_number})
        rows.append(summary)
    return pd.DataFrame(rows)


def _is_unknown(value) -> bool:
    return isinstance(value, float) and np.isnan(value)


def compute_cumulative_elapsed_time(laps_features_df: pd.DataFrame) -> pd.DataFrame:
    df = laps_features_df.copy()
    df["LapTime_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df = df.sort_values(["Driver", "LapNumber"])
    df["cumulative_elapsed_seconds"] = df.groupby("Driver")["LapTime_seconds"].cumsum()
    df["lap_start_elapsed_seconds"] = df["cumulative_elapsed_seconds"] - df["LapTime_seconds"]
    df["lap_mid_elapsed_seconds"] = (df["lap_start_elapsed_seconds"] + df["cumulative_elapsed_seconds"]) / 2
    return df[["Driver", "LapNumber", "lap_mid_elapsed_seconds"]].rename(
        columns={"Driver": "driver", "LapNumber": "lap_number"})


def find_time_aligned_lap(defender_code: str, target_elapsed_seconds: float, lap_summaries_with_time: pd.DataFrame):
    defender_rows = lap_summaries_with_time[lap_summaries_with_time["driver"] == defender_code]
    if defender_rows.empty:
        return None
    idx = (defender_rows["lap_mid_elapsed_seconds"] - target_elapsed_seconds).abs().idxmin()
    return defender_rows.loc[idx]


def reconstruct_overtakes_and_defenses(lap_summaries: pd.DataFrame, pit_laps: set,
                                         laps_features_df: pd.DataFrame,
                                         close_following_seconds: float = CLOSE_FOLLOWING_SECONDS) -> pd.DataFrame:
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
        follower, defender = follower_row["driver"], follower_row["driver_ahead"]
        follower_lap_number, follower_elapsed = follower_row["lap_number"], follower_row["lap_mid_elapsed_seconds"]
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
                        "event": "defense_held" if still_behind_same_car else "defense_lost", "rival": follower})

    return pd.DataFrame(events)


def process_one_race(bucket: CachedBucket, season: int, race: str) -> pd.DataFrame:
    """Quiet version of overtake_defense_poc.py's run_proof_of_concept - no
    per-race printing, just returns events, for use inside the aggregation loop."""
    pit_path = f"clean/features/{season}/{race}/R/laps_features.csv"
    pit_df = bucket.read_csv(pit_path, usecols=["Driver", "DriverNumber", "LapNumber", "LapTime",
                                                  "is_pit_in", "is_pit_out"])
    pit_laps = set(pit_df.loc[pit_df["is_pit_in"] | pit_df["is_pit_out"], ["Driver", "LapNumber"]]
                   .itertuples(index=False, name=None))

    lap_summaries = build_lap_summaries_for_race(bucket, season, race)
    if lap_summaries.empty:
        return pd.DataFrame()

    number_to_code = dict(zip(pit_df["DriverNumber"].astype(str), pit_df["Driver"]))

    def _map_driver_ahead(value):
        if value is None:
            return None
        return number_to_code.get(str(value), np.nan)

    lap_summaries["driver_ahead"] = lap_summaries["driver_ahead"].apply(_map_driver_ahead)
    return reconstruct_overtakes_and_defenses(lap_summaries, pit_laps, laps_features_df=pit_df)


# ---------------------------------------------------------------------------
# Step 3: aggregate across the sample, normalize into archetype-style scores
# ---------------------------------------------------------------------------
def normalize_to_archetype_range(series: pd.Series, low: float = 0.85, high: float = 1.15) -> pd.Series:
    """Min-max scale into the same 0.85-1.15 range driver_archetypes.xlsx uses,
    so these scores are directly comparable to/interchangeable with the
    manual archetype fallback values for drivers below the race threshold."""
    if series.max() == series.min():
        return pd.Series(1.0, index=series.index)
    return low + (series - series.min()) / (series.max() - series.min()) * (high - low)


if __name__ == "__main__":
    bucket = CachedBucket()

    print("[load] scanning all Race-session results.csv for driver participation history...")
    participation = build_driver_race_participation(bucket)
    print(f"[load] {len(participation)} driver-race entries across "
          f"{participation['race'].nunique()} circuits, {participation['season'].nunique()} seasons")

    race_counts = participation.groupby("driver").size().sort_values(ascending=False)
    qualifying_drivers = set(race_counts[race_counts >= MIN_CAREER_RACES].index)
    print(f"\n[scope] {len(qualifying_drivers)}/{race_counts.shape[0]} drivers have "
          f">= {MIN_CAREER_RACES} career races - real scores computed for these, "
          f"archetype fallback for the rest")

    race_to_type = get_circuit_type_lookup()

    global_sample = build_global_race_sample(participation, race_to_type)
    print(f"\n[sample] {len(global_sample)} races in the GLOBAL sample (processed once each, "
          f"regardless of how many drivers qualify) - was up to {len(qualifying_drivers) * 12} "
          f"under the old per-driver design")

    # Accumulate raw counts per driver across every race they actually appear
    # in from the global sample - one pass per race, not one pass per driver.
    driver_counts = {}  # driver -> {"overtakes": n, "defense_held": n, "defense_lost": n, "races_with_data": n}

    for i, (season, race) in enumerate(global_sample, 1):
        print(f"\n[{i}/{len(global_sample)}] processing {race} {season} ...")
        try:
            events = process_one_race(bucket, season, race)
        except Exception as e:
            print(f"  [warn] failed: {type(e).__name__}: {e} - skipping this race entirely")
            continue
        if events.empty:
            print("  [warn] no events reconstructed for this race - skipping")
            continue

        counts_per_driver = events.groupby(["driver", "event"]).size().unstack(fill_value=0)
        for driver, row in counts_per_driver.iterrows():
            if driver not in qualifying_drivers:
                continue  # not enough career races to trust a real score - archetype fallback instead
            stats = driver_counts.setdefault(driver, {"overtakes": 0, "defense_held": 0,
                                                         "defense_lost": 0, "races_with_data": 0})
            stats["overtakes"] += row.get("overtake_made", 0)
            stats["defense_held"] += row.get("defense_held", 0)
            stats["defense_lost"] += row.get("defense_lost", 0)
            stats["races_with_data"] += 1
        print(f"  [done] {len(counts_per_driver)} drivers credited from this race")

    driver_stats = []
    for driver, stats in driver_counts.items():
        total_defenses = stats["defense_held"] + stats["defense_lost"]
        driver_stats.append({
            "driver": driver,
            "races_with_data": stats["races_with_data"],
            "thin_sample": stats["races_with_data"] < MIN_RACES_WITH_DATA,
            "overtakes_per_race": stats["overtakes"] / stats["races_with_data"],
            "defense_hold_rate": stats["defense_held"] / total_defenses if total_defenses > 0 else np.nan,
        })

    n_thin = sum(1 for d in qualifying_drivers if d not in driver_counts) + \
        sum(1 for s in driver_stats if s["thin_sample"])
    print(f"\n[coverage] {len(driver_stats)}/{len(qualifying_drivers)} qualifying drivers got at "
          f"least some data from the global sample; {n_thin} have thin coverage "
          f"(<{MIN_RACES_WITH_DATA} races) - flagged in the output, not silently trusted")

    stats_df = pd.DataFrame(driver_stats)
    stats_df["aggression_level"] = normalize_to_archetype_range(stats_df["overtakes_per_race"])
    stats_df["defensive_strength"] = normalize_to_archetype_range(stats_df["defense_hold_rate"])

    print("\n" + "=" * 70)
    print("=== Real, data-derived driver scores (global race sample) ===")
    print("=" * 70)
    print(stats_df.sort_values("aggression_level", ascending=False).to_string(index=False))

    stats_df.to_csv("driver_taxonomy_sampled_scores.csv", index=False)
    print("\n[save] driver_taxonomy_sampled_scores.csv")