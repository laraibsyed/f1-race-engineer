""

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
    ""

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

        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]

def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    ""
    df = df.copy()

    df["time_seconds"] = pd.to_timedelta(df["Time"]).dt.total_seconds()

    if df["Rainfall"].dtype == object:
        df["Rainfall"] = df["Rainfall"].map({"True": True, "False": False}).astype(bool)
    else:
        df["Rainfall"] = df["Rainfall"].astype(bool)

    prev_rainfall = df["Rainfall"].shift(1)
    df["is_rain_onset"] = (df["Rainfall"] == True) & (prev_rainfall == False)
    df["is_rain_end"] = (df["Rainfall"] == False) & (prev_rainfall == True)
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
    ""
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
            continue

    print("\n[batch_clean_weather] run complete")
    print(f"  cleaned: {n_done} | skipped (already done): {n_skipped} | failed: {n_failed}")
    if failed_sessions:
        print("  failed sessions:")
        for s in failed_sessions:
            print(f"    - {s}")
        print("  re-run the same command to retry these (successful ones will be skipped)")

if __name__ == "__main__":
    main()

