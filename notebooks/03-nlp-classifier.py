import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")
RADIOS_CSV = r"data\external\team-radios\classified_radios.csv"
PIT_LOOKAHEAD_LAPS = 3

class CachedBucket:
    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=CACHE_DIR):
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

def load_classified_radios(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    bad_rows = df["message_timestamp"] == "message_timestamp"
    if bad_rows.any():
        print(f"[warn] {bad_rows.sum()} embedded duplicate header row(s) found and dropped")
        df = df[~bad_rows].copy()
    df["message_timestamp"] = pd.to_datetime(df["message_timestamp"], format="ISO8601", utc=True)
    df["season"] = df["race_id"].str.split("_", n=1).str[0].astype(int)
    df["race"] = df["race_id"].str.split("_", n=1).str[1]
    df["racing_number"] = df["racing_number"].astype(int)
    return df[df["category"] == "tyre_feedback"].copy()

def load_race_laps(bucket: CachedBucket, seasons: set, races: set) -> pd.DataFrame:
    ""
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv") or "/R/" not in p:
            continue
        parts = p.split("/")
        season, race = int(parts[2]), parts[3]
        if season not in seasons or race not in races:
            continue
        df = bucket.read_csv(p)
        df["Season"], df["Race"] = season, race
        frames.append(df)
    if not frames:
        raise RuntimeError("No matching race sessions found - check season/race name overlap "
                            "between classified_radios.csv and your GCS bucket structure")
    return pd.concat(frames, ignore_index=True)

def estimate_lap_start_times(laps: pd.DataFrame, radios: pd.DataFrame) -> pd.DataFrame:
    ""
    laps = laps.copy()
    laps["LapTime_seconds"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

    race_start = radios.groupby("race_id")["message_timestamp"].min().rename("race_start_estimate")
    laps["race_id"] = laps["Season"].astype(str) + "_" + laps["Race"]
    laps = laps.merge(race_start, left_on="race_id", right_index=True, how="left")

    laps = laps.sort_values(["Season", "Race", "DriverNumber", "LapNumber"])
    cum_prior = laps.groupby(["Season", "Race", "DriverNumber"])["LapTime_seconds"].cumsum() \
        - laps["LapTime_seconds"].fillna(0)
    laps["LapStartDate_estimated"] = laps["race_start_estimate"] + pd.to_timedelta(cum_prior, unit="s")
    return laps

def match_messages_to_nearest_lap(tyre_feedback: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    ""
    required_cols = {"DriverNumber", "LapNumber", "is_pit_in", "LapTime"}
    missing = required_cols - set(laps.columns)
    if missing:
        raise KeyError(f"laps_features.csv is missing expected column(s): {missing} - "
                        f"check the real column names in your data and update this script")

    laps = estimate_lap_start_times(laps, tyre_feedback)
    time_col = "LapStartDate_estimated"

    matched_rows = []
    n_skipped_no_laps = 0
    n_skipped_no_valid_timestamp = 0
    for _, msg in tyre_feedback.iterrows():
        driver_laps = laps[
            (laps["Season"] == msg["season"]) & (laps["Race"] == msg["race"])
            & (laps["DriverNumber"] == msg["racing_number"])
        ].sort_values("LapNumber")
        if driver_laps.empty:
            n_skipped_no_laps += 1
            continue

        driver_laps_valid_time = driver_laps.dropna(subset=[time_col])
        if driver_laps_valid_time.empty:
            n_skipped_no_valid_timestamp += 1
            continue

        time_diffs = (driver_laps_valid_time[time_col] - msg["message_timestamp"]).abs()
        nearest_idx = time_diffs.idxmin()
        nearest_lap_number = driver_laps_valid_time.loc[nearest_idx, "LapNumber"]
        pit_laps = frozenset(driver_laps.loc[driver_laps["is_pit_in"], "LapNumber"])

        matched_rows.append({
            "season": msg["season"], "race": msg["race"], "driver_id": msg["driver_id"],
            "stress_level": msg["stress_level"], "matched_lap": nearest_lap_number,
            "pit_laps": pit_laps,
        })

    if n_skipped_no_laps or n_skipped_no_valid_timestamp:
        print(f"[warn] skipped {n_skipped_no_laps} messages (no laps found for that driver/race), "
              f"{n_skipped_no_valid_timestamp} messages (all laps had NaT LapStartDate)")

    return pd.DataFrame(matched_rows)

def compute_pit_rate_for_lookahead(matched: pd.DataFrame, lookahead_laps: int) -> pd.DataFrame:
    ""
    matched = matched.copy()
    matched["pitted_within_lookahead"] = matched.apply(
        lambda r: any(r["matched_lap"] <= p <= r["matched_lap"] + lookahead_laps for p in r["pit_laps"]),
        axis=1,
    )
    summary = matched.groupby("stress_level")["pitted_within_lookahead"].agg(["mean", "count"])
    return summary.rename(columns={"mean": "pit_rate", "count": "n_messages"})

def two_prop_z_test(p1, n1, p2, n2):
    ""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p2 - p1) / se

    from math import erf
    p_val = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))
    return z, p_val

LOOKAHEAD_WINDOWS_TO_SWEEP = [1, 3, 5, 7]

if __name__ == "__main__":
    print("[load] pulling classified_radios.csv ...")
    tyre_feedback = load_classified_radios(RADIOS_CSV)
    print(f"[load] {len(tyre_feedback)} tyre_feedback messages "
          f"({tyre_feedback['stress_level'].value_counts().to_dict()})")

    seasons = set(tyre_feedback["season"].unique())
    races = set(tyre_feedback["race"].unique())
    print(f"[load] pulling matching race sessions from GCS ({len(seasons)} seasons, {len(races)} distinct races)...")
    bucket = CachedBucket()
    laps = load_race_laps(bucket, seasons, races)
    print(f"[load] {len(laps)} lap rows loaded")

    print("\n[note] LapStartDate is real only for 2020 (telemetry=True vs. telemetry=False at "
          "ingestion - see prior root-cause finding). Using the estimated-time fallback "
          "(cumulative LapTime + race-start proxy) for ALL seasons instead, so no season needs "
          "to be excluded - see estimate_lap_start_times() docstring for the known limitations.")

    matched = match_messages_to_nearest_lap(tyre_feedback, laps)
    print(f"[match] {len(matched)}/{len(tyre_feedback)} messages matched to a lap")

    print("\n=== SWEEP: pit rate + significance vs. informational control, by lookahead window ===")
    sweep_rows = []
    for window in LOOKAHEAD_WINDOWS_TO_SWEEP:
        summary = compute_pit_rate_for_lookahead(matched, window)
        print(f"\n--- lookahead = {window} lap(s) ---")
        print(summary.to_string())

        if "informational" not in summary.index:
            continue
        control_rate = summary.loc["informational", "pit_rate"]
        control_n = summary.loc["informational", "n_messages"]
        for level in ["medium", "high"]:
            if level not in summary.index:
                continue
            rate = summary.loc[level, "pit_rate"]
            n = summary.loc[level, "n_messages"]
            z, p_val = two_prop_z_test(control_rate, control_n, rate, n)
            lift = rate - control_rate
            sig = "**SIGNIFICANT**" if p_val < 0.05 else "not significant"
            print(f"  {level} vs. control: {'+' if lift >= 0 else ''}{lift:.3f} lift, "
                  f"p={p_val:.3f} ({sig})")
            sweep_rows.append({"lookahead_laps": window, "stress_level": level,
                                "pit_rate": rate, "n_messages": n, "lift_vs_control": lift,
                                "p_value": p_val})

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("stress_signal_lookahead_sweep.csv", index=False)
    print(f"\n[save] stress_signal_lookahead_sweep.csv")

    print("\n=== SUMMARY: any window reach significance? ===")
    if sweep_df["p_value"].min() < 0.05:
        print(sweep_df[sweep_df["p_value"] < 0.05].to_string(index=False))
    else:
        print("No lookahead window (1/3/5/7 laps) produced a statistically significant "
              "difference from the informational control, for either medium or high stress. "
              "This is a real, reportable finding: not an artifact of choosing the wrong window.")
