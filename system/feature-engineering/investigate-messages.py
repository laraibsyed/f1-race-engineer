"""
investigate_messages.py

Diagnostic-only, no cleaning/writing. Answers before building the SC/VSC
race-control-message join:
    1. Does raw/fastf1/.../messages.csv exist, what columns does it have?
    2. Is there already a native Lap/LapNumber field, or do we need to
       time-match against laps.csv like we did for weather/telemetry?
    3. What does actual SC/VSC message text look like -- exact wording,
       and whether it's consistent across seasons (message wording is
       exactly the kind of thing that's drifted before in this project).
    4. Do these messages agree with the is_sc_lap/is_vsc_lap flags we
       already built from TrackStatus in clean_laps.py, or do they add
       information (e.g. exact deployment/ending lap vs TrackStatus's
       per-lap flag)?

Requirements:
    pip install google-cloud-storage pandas python-dotenv --break-system-packages

Usage:
    python investigate_messages.py
"""

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
    ("2018", "Australian_Grand_Prix", "R"),
    ("2021", "Azerbaijan_Grand_Prix", "R"),   # known SC-heavy race
    ("2022", "Australian_Grand_Prix", "R"),   # known SC-heavy race
    ("2023", "Qatar_Grand_Prix", "S"),         # known chaotic sprint
    ("2024", "Abu_Dhabi_Grand_Prix", "R"),
]

SC_VSC_KEYWORDS = ["SAFETY CAR", "VIRTUAL SAFETY CAR", "VSC"]


def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception:
            return None


def find_file(year: str, race: str, session: str, filename: str) -> str | None:
    prefix = f"raw/fastf1/{year}/{race}/{session}/"
    for b in client.list_blobs(BUCKET_NAME, prefix=prefix):
        if b.name.endswith(filename):
            return b.name
    return None


def inspect_session(year: str, race: str, session: str):
    print("=" * 70)
    print(f"{year}/{race}/{session}")
    print("=" * 70)

    path = find_file(year, race, session, "messages.csv")
    if not path:
        print(f"  NOT FOUND under raw/fastf1/{year}/{race}/{session}/")
        return

    print(f"  Found: {path}")
    df = read_csv_robust(bucket.blob(path).download_as_bytes())
    if df is None or df.empty:
        print("  Empty or unreadable.")
        return

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print()

    for col in df.columns:
        series = df[col]
        pct_null = round(series.isna().mean() * 100, 1)
        non_null = series.dropna()
        if len(non_null) and not pd.api.types.is_numeric_dtype(series):
            uniques = non_null.unique()
            preview = sorted(map(str, uniques))[:8] if len(uniques) <= 30 else f"{len(uniques)} unique values"
        elif len(non_null):
            preview = f"min={non_null.min()}, max={non_null.max()}"
        else:
            preview = "N/A"
        print(f"    {col:20s} null={pct_null:5.1f}%  {preview}")

    # find the actual message text column -- usually called 'Message'
    text_col = next((c for c in df.columns if c.lower() == "message"), None)
    if text_col is None:
        print("\n  No obvious 'Message' text column found -- inspect columns above manually.")
        return

    print(f"\n  SC/VSC-related rows (matching on '{text_col}' column):")
    mask = df[text_col].astype(str).str.upper().apply(lambda s: any(k in s for k in SC_VSC_KEYWORDS))
    sc_vsc_rows = df[mask]
    if sc_vsc_rows.empty:
        print("    None found in this session.")
    else:
        print(sc_vsc_rows.to_string())

    print()


if __name__ == "__main__":
    for year, race, session in SAMPLE_SESSIONS:
        inspect_session(year, race, session)