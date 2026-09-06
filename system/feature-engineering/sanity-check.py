"""
sanity_check_features.py

Post-engineering sanity check on clean/features/.../laps_features.csv:
    1. Row count parity vs laps_flagged.csv (engineering shouldn't drop/add rows)
    2. compound_encoded coverage -- flags any Compound string that didn't
       map to COMPOUND_ORDER (e.g. unmapped 2018 legacy names like
       HYPERSOFT/SUPERSOFT/ULTRASOFT, if that cleaning step never actually
       got wired into clean_laps.py)
    3. Race-only features (gap_to_leader, gap_to_car_ahead, fuel_load_estimate)
       are fully null in FP/Q sessions and mostly populated in R/S sessions
    4. degradation_rate and tyre_age basic range sanity

Read-only, doesn't modify anything.

Usage:
    python sanity_check_features.py
    python sanity_check_features.py --year 2024
"""

import argparse
import io
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = Path(os.getenv("GCS_CACHE_DIR", "./gcs_cache"))

LAPS_PREFIX = "clean/fastf1/"
FEATURES_PREFIX = "clean/features/"
RACE_LIKE_SESSIONS = {"R", "S"}
KNOWN_COMPOUNDS = {"WET", "INTERMEDIATE", "HARD", "MEDIUM", "SOFT", "HYPERSOFT", "SUPERSOFT", "ULTRASOFT"}
# These are FastF1's own markers for test/undetermined tyres, not a real
# hardness class -- compound_encoded is SUPPOSED to be null for these.
# Not a bug, don't flag it as one.
EXPECTED_NULL_COMPOUNDS = {"TEST", "TEST_UNKNOWN", "UNKNOWN"}
VALID_TRACK_TEMP_BUCKETS = {"cool", "warm", "hot", "extreme"}


class CachedBucket:
    def __init__(self, bucket_name: str, cache_dir: Path = CACHE_DIR):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _local_path(self, blob_path: str) -> Path:
        return self.cache_dir / blob_path

    def download_as_bytes(self, blob_path: str) -> bytes:
        local_path = self._local_path(blob_path)
        if local_path.exists():
            return local_path.read_bytes()
        data = self.bucket.blob(blob_path).download_as_bytes()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return data

    def list_blob_names(self, prefix: str, retries: int = 3, timeout: int = 120) -> list[str]:
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return [b.name for b in self.client.list_blobs(self.bucket.name, prefix=prefix, timeout=timeout)]
            except Exception as e:
                last_error = e
                print(f"    [warn] list_blobs attempt {attempt}/{retries} failed ({e}), retrying...")
                time.sleep(2 * attempt)
        raise last_error


bucket = CachedBucket(BUCKET_NAME)


def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception:
            return None


def find_sessions(year: str) -> list[tuple[str, str, str]]:
    prefix = f"{FEATURES_PREFIX}{year}/" if year != "all" else FEATURES_PREFIX
    print(f"[scan] Listing {prefix} ...")
    paths = bucket.list_blob_names(prefix)
    sessions = set()
    for p in paths:
        if p.endswith("laps_features.csv"):
            parts = p.rstrip("/").split("/")
            if len(parts) >= 4:
                sessions.add((parts[-4], parts[-3], parts[-2]))
    print(f"[scan] Found {len(sessions)} sessions.")
    return sorted(sessions)


def check_session(year: str, race: str, session: str) -> dict:
    result = {"year": year, "race": race, "session": session, "issues": []}

    feat_path = f"{FEATURES_PREFIX}{year}/{race}/{session}/laps_features.csv"
    laps_path = f"{LAPS_PREFIX}{year}/{race}/{session}/laps_flagged.csv"

    try:
        feat_df = read_csv_robust(bucket.download_as_bytes(feat_path))
        laps_df = read_csv_robust(bucket.download_as_bytes(laps_path))
    except Exception as e:
        result["issues"].append(f"ERROR reading files: {type(e).__name__}: {e}")
        return result

    if feat_df is None or laps_df is None:
        result["issues"].append("ERROR: unreadable CSV")
        return result

    # 1. row parity
    if len(feat_df) != len(laps_df):
        result["issues"].append(f"ROW COUNT MISMATCH: laps_flagged={len(laps_df)}, laps_features={len(feat_df)}")

    # 2. compound mapping coverage
    if "Compound" in feat_df.columns and "compound_encoded" in feat_df.columns:
        has_compound = feat_df["Compound"].notna()
        unmapped = feat_df.loc[has_compound & feat_df["compound_encoded"].isna(), "Compound"].unique()
        # only flag genuinely unexpected unmapped strings -- TEST/UNKNOWN/etc
        # are supposed to be null, not a data quality problem
        genuinely_unexpected = set(unmapped) - EXPECTED_NULL_COMPOUNDS
        if genuinely_unexpected:
            result["issues"].append(f"UNMAPPED COMPOUNDS: {sorted(genuinely_unexpected)}")

    # 3. race-only feature null pattern
    race_only_cols = [c for c in ("gap_to_leader", "gap_to_car_ahead", "fuel_load_estimate") if c in feat_df.columns]
    if race_only_cols:
        if session not in RACE_LIKE_SESSIONS:
            still_populated = {c: feat_df[c].notna().sum() for c in race_only_cols if feat_df[c].notna().any()}
            if still_populated:
                result["issues"].append(f"NOT NULLED in non-race session: {still_populated}")
        else:
            all_null = {c: True for c in race_only_cols if feat_df[c].isna().all()}
            if all_null:
                result["issues"].append(f"UNEXPECTEDLY ALL-NULL in race session: {list(all_null.keys())}")

    # 4. tyre_age / degradation_rate range sanity
    if "tyre_age" in feat_df.columns:
        min_age = feat_df["tyre_age"].min()
        max_age = feat_df["tyre_age"].max()
        if pd.notna(min_age) and min_age < 1:
            result["issues"].append(f"tyre_age has values < 1 (min={min_age})")
        if pd.notna(max_age) and max_age > 80:
            result["issues"].append(f"tyre_age has suspiciously high max ({max_age})")

    if "degradation_rate" in feat_df.columns:
        pct_null = round(feat_df["degradation_rate"].isna().mean() * 100, 1)
        result["degradation_pct_null"] = pct_null

    # 5. weather coverage -- how much of this session actually got a
    # matched track temperature, vs weather.csv being missing/unmatched
    if "track_temp_c" in feat_df.columns:
        pct_null_weather = round(feat_df["track_temp_c"].isna().mean() * 100, 1)
        result["track_temp_pct_null"] = pct_null_weather
        if pct_null_weather == 100.0:
            result["issues"].append("NO WEATHER DATA (track_temp_c 100% null -- weather.csv likely missing)")

        # range sanity -- track temps outside this are almost certainly a
        # parsing/unit bug, not real F1 conditions anywhere on Earth
        temp_vals = pd.to_numeric(feat_df["track_temp_c"], errors="coerce").dropna()
        if len(temp_vals) and (temp_vals.min() < 0 or temp_vals.max() > 70):
            result["issues"].append(f"track_temp_c OUT OF PLAUSIBLE RANGE: min={temp_vals.min()}, max={temp_vals.max()}")
    else:
        result["issues"].append("MISSING track_temp_c COLUMN (re-run engineer_features.py --force?)")

    if "track_temp_bucket" in feat_df.columns:
        observed_buckets = set(feat_df["track_temp_bucket"].dropna().astype(str).unique())
        unexpected_buckets = observed_buckets - VALID_TRACK_TEMP_BUCKETS
        if unexpected_buckets:
            result["issues"].append(f"UNEXPECTED track_temp_bucket VALUES: {sorted(unexpected_buckets)}")

    # 6. SC/VSC event-marker columns present, and cross-checked against
    # the broader is_sc_lap/is_vsc_lap flags built earlier from TrackStatus.
    # These are two independent derivations (race control messages vs
    # TrackStatus codes) -- if they disagree, one of them has a real bug.
    sc_vsc_cols = ["is_sc_deployed_lap", "is_sc_ending_lap", "is_vsc_deployed_lap",
                   "is_vsc_ending_lap", "is_sc_through_pit_lane_lap"]
    missing_sc_vsc_cols = [c for c in sc_vsc_cols if c not in feat_df.columns]
    if missing_sc_vsc_cols:
        result["issues"].append(f"MISSING SC/VSC EVENT COLUMNS: {missing_sc_vsc_cols}")
    else:
        result["issues"].extend(
            cross_check_sc_vsc_events(feat_df, "is_sc_deployed_lap", "is_sc_ending_lap", "is_sc_lap", "SC")
        )
        result["issues"].extend(
            cross_check_sc_vsc_events(feat_df, "is_vsc_deployed_lap", "is_vsc_ending_lap", "is_vsc_lap", "VSC")
        )

    return result


def cross_check_sc_vsc_events(df: pd.DataFrame, deployed_col: str, ending_col: str,
                               active_col: str, label: str) -> list[str]:
    """
    For every lap where a deployment or ending event fired, confirm the
    broader active_col (is_sc_lap/is_vsc_lap, built from TrackStatus in
    clean_laps.py) agrees that something was actually happening that lap.
    Doesn't try to validate the full period between events -- just the
    event laps themselves, which is the strongest signal without assuming
    a fragile exact-pairing model between deploy/end events.
    """
    issues = []
    if active_col not in df.columns:
        return [f"Cannot cross-check {label}: {active_col} column missing"]

    for event_col, event_label in [(deployed_col, "deployed"), (ending_col, "ending")]:
        event_laps = sorted(df.loc[df[event_col], "LapNumber"].dropna().unique())
        for lap in event_laps:
            lap_rows = df[df["LapNumber"] == lap]
            if not lap_rows[active_col].any():
                issues.append(
                    f"{label} {event_label} fired at lap {int(lap)} but {active_col} is False "
                    f"for every row on that lap -- message-based and TrackStatus-based flags disagree"
                )
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="all")
    args = parser.parse_args()

    sessions = find_sessions(args.year)
    print(f"\nChecking {len(sessions)} sessions ...\n")

    results = []
    all_unmapped_compounds = set()
    for i, (y, r, s) in enumerate(sessions):
        res = check_session(y, r, s)
        results.append(res)
        for issue in res["issues"]:
            print(f"  [{y}/{r}/{s}] {issue}")
            if issue.startswith("UNMAPPED COMPOUNDS"):
                compounds_str = issue.split(": ", 1)[1]
                all_unmapped_compounds.update(eval(compounds_str))
        if (i + 1) % 150 == 0:
            print(f"    ...{i + 1}/{len(sessions)} checked")

    df = pd.DataFrame(results)
    df.to_csv("feature_sanity_report.csv", index=False)

    n_with_issues = sum(1 for r in results if r["issues"])
    n_no_weather = sum(1 for r in results if any(i.startswith("NO WEATHER DATA") for i in r["issues"]))
    print("\n" + "=" * 60)
    print("FEATURE SANITY SUMMARY")
    print("=" * 60)
    print(f"Total sessions checked: {len(sessions)}")
    print(f"Sessions with issues:   {n_with_issues}")
    print(f"  of which, sessions with NO weather data at all: {n_no_weather}")
    if all_unmapped_compounds:
        print(f"\nUNMAPPED COMPOUND STRINGS FOUND ACROSS DATASET: {sorted(all_unmapped_compounds)}")
        print("These need adding to COMPOUND_ORDER in engineer_features.py and a re-run.")
    else:
        print("\nNo unmapped compound strings found -- compound_encoded covers everything present.")
    print(f"\nFull report: feature_sanity_report.csv")