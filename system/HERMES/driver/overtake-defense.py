"""
Overtake / Defense Event Reconstruction — Single-Race Proof of Concept
===========================================================================
Builds the real data behind aggression_level (overtakes made) and
defensive_strength (successful defenses when under pressure) from raw
tracinginsights per-lap telemetry - the 197,909-file archive that has NOT
been aggregated yet (only 200 races have the clean telemetry_by_lap.csv
version, which lacks DriverAhead/DistanceToDriverAhead entirely).

SCOPED DELIBERATELY TO ONE RACE FIRST. This is the largest new data-
engineering task this session by a wide margin - validate the detection
logic works correctly on one race's ~1,000-1,400 files before committing to
processing the full historical archive, the same "small then scale"
discipline that made every other module in this project reliable.

KEY FACTS CONFIRMED FROM REAL DATA before building anything (not assumed):
  - DistanceToDriverAhead is in METRES (confirmed range 103-412m on a real
    sample lap), not a time gap - normalized here using that lap's own
    speed_mean to get an approximate time gap, so a "close" threshold means
    the same thing at Monaco and at Monza.
  - drs=0 for an entire lap is NOT necessarily a bug - it correlates with
    genuinely not being close enough to the car ahead (confirmed on the same
    real sample), consistent with real DRS activation rules.
  - Raw tracinginsights paths use race names WITH SPACES
    ("Abu Dhabi Grand Prix"), differing from laps_features.csv's underscore
    convention ("Abu_Dhabi_Grand_Prix") - handled explicitly via
    race_name_underscore_to_space(), not assumed to match.

CLOSE-FOLLOWING THRESHOLD: CLOSE_FOLLOWING_SECONDS = 1.0 - ASSUMPTION,
matches real DRS activation rules (within 1 second) as a natural, defensible
choice, but flagged as a threshold that could use its own sensitivity check
later, same as every other calibrated constant this project.
"""

import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CLOSE_FOLLOWING_SECONDS = 1.0  # ASSUMPTION - matches real DRS activation range


def race_name_underscore_to_space(race: str) -> str:
    """laps_features.csv convention -> raw tracinginsights path convention."""
    return race.replace("_", " ")


def extract_lap_summary(json_bytes: bytes) -> dict:
    """
    From one raw per-driver-per-lap telemetry file, extract lightweight
    scalar summaries only - NOT the full time-series - keeping this tractable
    across a potentially huge number of files.

    "None" (literal string) appears in DistanceToDriverAhead whenever there's
    no car ahead at all - not just in DriverAhead. Confirmed as a real,
    expected case, not a data bug: HAM's race-winning laps (leading
    unchallenged for long stretches) hit this exactly, both just after his
    early overtake into the lead and while cruising to the finish. Filtering
    only DriverAhead and not DistanceToDriverAhead caused np.array(dtype=
    float) to crash outright on 8 of HAM's laps in the first real run.
    """
    data = json.loads(json_bytes)
    tel = data["tel"]

    raw_distances = tel["DistanceToDriverAhead"]
    raw_speeds = tel["speed"]
    raw_driver_ahead = tel["DriverAhead"]

    mean_speed_kmh = float(np.mean([s for s in raw_speeds if s is not None])) if raw_speeds else np.nan

    valid_mask = [d not in (None, "None") for d in raw_distances]
    if not any(valid_mask):
        # No valid distance-to-ahead reading ANYWHERE in this lap - driver
        # had no car ahead for the entire lap (leading/unchallenged).
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
    """Downloads and summarizes every driver-lap file for one race. This is
    the part that scales to the full archive later - kept as its own
    function so scaling up is just calling this in a loop over races."""
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
    """
    Per (Driver, LapNumber): cumulative elapsed RACE TIME at the end of that
    lap - a TRUE cross-driver time reference, unlike raw LapNumber. A driver
    a lap down will show a LARGER cumulative_elapsed_seconds for the SAME raw
    LapNumber than the leader, correctly reflecting they reached "lap N"
    later in real time. Same technique as
    nlp_pit_correlation_validation.py's estimate_lap_start_times() - reused
    deliberately, same root problem (raw lap number isn't a time reference
    once someone falls behind), different symptom.
    """
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
    """
    Among the defender's OWN laps, finds whichever lap's elapsed time is
    CLOSEST to target_elapsed_seconds - the true time-aligned lap, which may
    NOT equal the follower's raw lap_number if the defender is lapped.
    """
    defender_rows = lap_summaries_with_time[lap_summaries_with_time["driver"] == defender_code]
    if defender_rows.empty:
        return None
    idx = (defender_rows["lap_mid_elapsed_seconds"] - target_elapsed_seconds).abs().idxmin()
    return defender_rows.loc[idx]


def _is_unknown(value) -> bool:
    """True only for NaN - an unmapped real car number we couldn't identify,
    genuine uncertainty. NOT true for None, which is a CONFIRMED 'no car
    ahead' signal (e.g. taking the lead) and should participate normally in
    comparisons, not be treated as equally untrustworthy as real NaN garbage."""
    return isinstance(value, float) and np.isnan(value)


def reconstruct_overtakes_and_defenses(lap_summaries: pd.DataFrame,
                                         pit_laps: set,
                                         laps_features_df: pd.DataFrame,
                                         close_following_seconds: float = CLOSE_FOLLOWING_SECONDS) -> pd.DataFrame:
    """
    lap_summaries: driver, lap_number, driver_ahead, min_gap_seconds
    pit_laps: set of (driver, lap_number) that were pit in/out laps - EXCLUDED
    from overtake detection, since a position change caused by a pit stop is
    not an on-track pass.

    For each driver, an OVERTAKE MADE is recorded when their driver_ahead
    changes from lap N to N+1 (neither driver pitted around this transition) -
    they are no longer behind the same car, implying they passed.

    For each driver being followed closely (another driver's driver_ahead ==
    them, with min_gap_seconds < close_following_seconds), a DEFENSE HELD is
    recorded if they are STILL that follower's driver_ahead next lap, a
    DEFENSE LOST if not.

    FIXED: cross-driver "who is near whom" matching now uses cumulative
    elapsed TIME, not raw lap_number - grouping by raw lap_number silently
    assumed every driver was on the same lap at the same moment, which is
    false for anyone a lap down. Confirmed as a real issue on the 2018 Abu
    Dhabi proof-of-concept run, where GRO and MAG both finished "+1 Lap".
    """
    lap_summaries = lap_summaries.sort_values(["driver", "lap_number"]).reset_index(drop=True)
    elapsed = compute_cumulative_elapsed_time(laps_features_df)
    lap_summaries = lap_summaries.merge(elapsed, on=["driver", "lap_number"], how="left")

    events = []

    for driver, g in lap_summaries.groupby("driver"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            lap_now, lap_next = g.loc[i, "lap_number"], g.loc[i + 1, "lap_number"]
            if lap_next != lap_now + 1:
                continue  # missing lap in between - don't guess across a gap
            if (driver, lap_now) in pit_laps or (driver, lap_next) in pit_laps:
                continue  # pit-stop-caused reshuffle, not an on-track pass

            ahead_now, ahead_next = g.loc[i, "driver_ahead"], g.loc[i + 1, "driver_ahead"]
            if _is_unknown(ahead_now) or _is_unknown(ahead_next):
                continue  # genuine uncertainty (unmapped real driver) - don't guess, don't count
            if ahead_now is not None and ahead_now != ahead_next:
                events.append({"driver": driver, "lap": lap_now, "event": "overtake_made", "rival": ahead_now})

    # Defenses: for EVERY close-following row (not grouped by raw lap_number
    # anymore), find the defender's TIME-ALIGNED lap directly.
    close_followers = lap_summaries[lap_summaries["min_gap_seconds"] < close_following_seconds]
    for _, follower_row in close_followers.iterrows():
        follower = follower_row["driver"]
        defender = follower_row["driver_ahead"]
        follower_lap_number = follower_row["lap_number"]
        follower_elapsed = follower_row["lap_mid_elapsed_seconds"]
        if defender is None or pd.isna(defender) or pd.isna(follower_elapsed):
            continue

        # What lap number was the DEFENDER actually on at this same real moment?
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
    """
    All the fixes found on the 2018 Abu Dhabi baseline, generalized into a
    reusable function: (1) driver-number vs driver-code translation, (2)
    time-aligned cross-driver matching for lapped traffic, (3) "None"
    filtering for both DriverAhead and DistanceToDriverAhead, (4) preserving
    confirmed None distinctly from genuinely-unmapped NaN.
    """
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

    # Three deliberately different test cases: the already-validated baseline,
    # a street circuit famous for being extremely hard to overtake on (real-
    # world expectation: LOW overtake counts, HIGH defense-held rates - a
    # genuinely different distribution shape, not just more of the same), and
    # a post-2022 ground-effect-era race with different aero/DRS dynamics.
    TEST_CASES = [
        (2018, "Abu_Dhabi_Grand_Prix"),   # baseline, already validated above
        (2019, "Monaco_Grand_Prix"),       # street circuit - expect LOW overtakes
        (2023, "Bahrain_Grand_Prix"),      # post-2022 regs - different era
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