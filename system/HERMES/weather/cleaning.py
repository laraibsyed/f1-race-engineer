"""
batch_clean_weather.py
-----------------------
Batch-runs clean_weather.py's cleaning logic across every session in
raw/fastf1/<year>/<race>/<session>/weather.csv and writes
clean/fastf1/<year>/<race>/<session>/weather_cleaned.csv

Resumable by design (matches CachedBucket pattern from the rest of the pipeline):
  - Skip-if-output-exists: if weather_cleaned.csv already exists in GCS, skip it.
    So if this dies halfway through, just re-run the same command — it picks up
    where it left off instead of re-cleaning everything.
  - Per-session try/except: one bad/missing weather.csv doesn't kill the whole run.
  - Write-through: every cleaned file goes to GCS AND local gcs_cache at the same time.
  - --force flag bypasses the skip check if you ever need to fully re-run.

Usage:
    python batch_clean_weather.py                # normal resumable run
    python batch_clean_weather.py --force         # re-clean everything, ignore existing outputs
    python batch_clean_weather.py --year 2018     # only one season
"""

import argparse
import io
import os
import sys
import traceback

import pandas as pd
from google.cloud import storage

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")


class CachedBucket:
    """Same pattern used across the rest of the pipeline (clean_laps.py, align_telemetry.py etc).
    Reads check local disk first, writes go to GCS + local cache (write-through)."""

    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=CACHE_DIR):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def _local_path(self, blob_path):
        return os.path.join(self.cache_dir, blob_path)

    def exists(self, blob_path):
        local_path = self._local_path(blob_path)
        if os.path.exists(local_path):
            return True
        return self.bucket.blob(blob_path).exists()

    def read_csv(self, blob_path):
        local_path = self._local_path(blob_path)
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        blob = self.bucket.blob(blob_path)
        data = blob.download_as_bytes()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")

    def write_csv(self, blob_path, df):
        local_path = self._local_path(blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_filename(local_path)

    def list_blob_names(self, prefix):
        # always live, never cached (per bucket_setup.md convention)
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Same logic as clean_weather.py — kept in sync manually, see note at bottom of file."""
    df = df.copy()

    df["time_seconds"] = pd.to_timedelta(df["Time"]).dt.total_seconds()

    if df["Rainfall"].dtype == object:
        df["Rainfall"] = df["Rainfall"].map({"True": True, "False": False}).astype(bool)
    else:
        df["Rainfall"] = df["Rainfall"].astype(bool)

    prev_rainfall = df["Rainfall"].shift(1)
    df["is_rain_onset"] = (df["Rainfall"] == True) & (prev_rainfall == False)   # noqa: E712
    df["is_rain_end"] = (df["Rainfall"] == False) & (prev_rainfall == True)     # noqa: E712
    df.loc[df.index[0], ["is_rain_onset", "is_rain_end"]] = False

    df["rainfall_pct_session"] = df["Rainfall"].expanding().mean() * 100
    df["sample_gap_seconds"] = df["time_seconds"].diff()

    rain_group = (df["Rainfall"] != df["Rainfall"].shift()).cumsum()
    streak_len_seconds = df.groupby(rain_group)["time_seconds"].transform(
        lambda x: x.max() - x.min()
    )
    df["is_short_rain_streak"] = df["Rainfall"] & (streak_len_seconds < 600)

    original_cols = ["Time", "AirTemp", "Humidity", "Pressure", "Rainfall",
                      "TrackTemp", "WindDirection", "WindSpeed"]
    derived_cols = ["time_seconds", "is_rain_onset", "is_rain_end",
                     "rainfall_pct_session", "sample_gap_seconds",
                     "is_short_rain_streak"]
    return df[[c for c in original_cols if c in df.columns] + derived_cols]


def find_weather_sessions(bucket: CachedBucket, year_filter=None):
    """List every raw/fastf1/<year>/<race>/<session>/weather.csv path in the bucket."""
    prefix = "raw/fastf1/"
    all_blobs = bucket.list_blob_names(prefix)
    weather_paths = [b for b in all_blobs if b.endswith("/weather.csv")]
    if year_filter:
        weather_paths = [p for p in weather_paths if f"/raw/fastf1/{year_filter}/" in f"/{p}"]
    return sorted(weather_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-clean even if output exists")
    parser.add_argument("--year", type=str, default=None, help="Only process one season")
    args = parser.parse_args()

    bucket = CachedBucket()

    print(f"[batch_clean_weather] scanning raw/fastf1/ for weather.csv files"
          f"{' (year=' + args.year + ')' if args.year else ''}...")
    weather_paths = find_weather_sessions(bucket, year_filter=args.year)
    print(f"[batch_clean_weather] found {len(weather_paths)} weather.csv files")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    failed_sessions = []

    for i, raw_path in enumerate(weather_paths, start=1):
        # raw/fastf1/<year>/<race>/<session>/weather.csv -> clean/fastf1/<year>/<race>/<session>/weather_cleaned.csv
        session_dir = raw_path.replace("raw/fastf1/", "", 1).rsplit("/weather.csv", 1)[0]
        out_path = f"clean/fastf1/{session_dir}/weather_cleaned.csv"

        if not args.force and bucket.exists(out_path):
            n_skipped += 1
            print(f"[{i}/{len(weather_paths)}] SKIP  (already exists) {session_dir}")
            continue

        try:
            raw_df = bucket.read_csv(raw_path)
            cleaned_df = clean_weather(raw_df)
            bucket.write_csv(out_path, cleaned_df)

            n_blips = int(cleaned_df["is_short_rain_streak"].sum())
            flag_note = f" ⚠ {n_blips} short rain streak reading(s)" if n_blips else ""
            print(f"[{i}/{len(weather_paths)}] OK    {session_dir}{flag_note}")
            n_done += 1

        except Exception as e:
            n_failed += 1
            failed_sessions.append(session_dir)
            print(f"[{i}/{len(weather_paths)}] FAIL  {session_dir} -> {e}")
            traceback.print_exc(file=sys.stdout)
            continue  # keep going, don't let one bad session kill the run

    print("\n[batch_clean_weather] run complete")
    print(f"  cleaned: {n_done} | skipped (already done): {n_skipped} | failed: {n_failed}")
    if failed_sessions:
        print("  failed sessions:")
        for s in failed_sessions:
            print(f"    - {s}")
        print("  re-run the same command to retry these (successful ones will be skipped)")


if __name__ == "__main__":
    main()

# NOTE: clean_weather() logic here is duplicated from clean_weather.py, not imported.
# If you tweak the cleaning logic in clean_weather.py (e.g. change the rain streak
# threshold), copy the change here too, or these will drift out of sync. Same
# duplication pattern the rest of this pipeline already uses (CachedBucket is
# duplicated per-script rather than shared), so at least it's consistent.