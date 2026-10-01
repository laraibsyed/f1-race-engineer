
import io
import os

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

SAMPLE_SESSIONS = [
    ("2018", "Abu_Dhabi_Grand_Prix", "R"),
    ("2018", "Australian_Grand_Prix", "R"),
    ("2021", "British_Grand_Prix", "R"),
    ("2024", "Abu_Dhabi_Grand_Prix", "R"),
    ("2024", "Singapore_Grand_Prix", "R"),
    ("2025", "Abu_Dhabi_Grand_Prix", "R"),
]

def find_weather_file(year: str, race: str, session: str) -> str | None:
    prefix = f"raw/fastf1/{year}/{race}/{session}/"
    for b in client.list_blobs(BUCKET_NAME, prefix=prefix):
        if b.name.endswith("weather.csv"):
            return b.name
    return None

def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception:
            return None

def inspect_weather(year: str, race: str, session: str):
    print("=" * 70)
    print(f"{year}/{race}/{session}")
    print("=" * 70)

    path = find_weather_file(year, race, session)
    if not path:
        print(f"  NOT FOUND under raw/fastf1/{year}/{race}/{session}/")
        return

    print(f"  Found: {path}")
    data = bucket.blob(path).download_as_bytes()
    df = read_csv_robust(data)
    if df is None or df.empty:
        print("  Empty or unreadable.")
        return

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print()
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        pct_null = round(series.isna().mean() * 100, 1)
        non_null = series.dropna()
        if pd.api.types.is_numeric_dtype(series) and len(non_null):
            rng = f"min={non_null.min()}, max={non_null.max()}, mean={round(non_null.mean(), 2)}"
        elif len(non_null):
            uniques = non_null.unique()
            rng = f"unique: {sorted(map(str, uniques))[:10]}" if len(uniques) <= 10 else f"{len(uniques)} unique values"
        else:
            rng = "N/A (all null)"
        print(f"    {col:20s} dtype={dtype:10s} null={pct_null:5.1f}%  {rng}")

    print()
    print("  First 3 rows:")
    print(df.head(3).to_string())

    laps_prefix = f"raw/fastf1/{year}/{race}/{session}/"
    laps_path = None
    for b in client.list_blobs(BUCKET_NAME, prefix=laps_prefix):
        if b.name.endswith("laps.csv"):
            laps_path = b.name
            break
    if laps_path and "Time" in df.columns:
        laps_df = read_csv_robust(bucket.blob(laps_path).download_as_bytes())
        if laps_df is not None and "Time" in laps_df.columns:
            weather_time = pd.to_timedelta(df["Time"], errors="coerce").dt.total_seconds()
            laps_time = pd.to_timedelta(laps_df["Time"], errors="coerce").dt.total_seconds()
            print()
            print(f"  weather.csv 'Time' range (parsed as timedelta->seconds): "
                  f"{weather_time.min()} to {weather_time.max()}")
            print(f"  laps.csv    'Time' range (parsed as timedelta->seconds): "
                  f"{laps_time.min()} to {laps_time.max()}")
            print("  (If these ranges overlap/match in scale, same clock -- merge_asof will work "
                  "the same way align_telemetry.py used it. If wildly different, investigate further "
                  "before building track_temp_bucket.)")

    print()

if __name__ == "__main__":
    for year, race, session in SAMPLE_SESSIONS:
        inspect_weather(year, race, session)
