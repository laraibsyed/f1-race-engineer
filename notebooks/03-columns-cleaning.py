"""
investigate_fastf1_issues.py

Root-cause checks for the open flags from the fastf1 CSV audit, BEFORE
writing any cleaning logic. Answers:

1. TrackStatus: is min=1/max=12675 a real code, or a parsing bug?
   (FastF1's TrackStatus should normally be a short code like '1','2','4','6','7')

2. The 100%-null columns in laps/results (LapStartDate, Position, Q1/Q2/Q3,
   GridPosition, ClassifiedPosition, Status, Points, Laps, Time):
   are these null EVERYWHERE (all years), or just in the specific files
   that happened to get sampled? Checks a recent year (2024) vs an old
   year (2018) directly to see if it's a coverage gap (old seasons lack
   this data) or a genuine ingestion bug (broken everywhere).

3. Compound: confirms exactly which years use old vs new tyre naming,
   so the cleaning/mapping logic is based on fact, not assumption.

This does NOT fix anything -- it's diagnostic only. Once you see the
output, we decide where each fix belongs.

Requirements:
    pip install google-cloud-storage pandas python-dotenv --break-system-packages

Usage:
    python investigate_fastf1_issues.py
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


def read_csv(path: str) -> pd.DataFrame:
    data = bucket.blob(path).download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def find_one_file(prefix: str, filename: str) -> str | None:
    for b in client.list_blobs(BUCKET_NAME, prefix=prefix):
        if b.name.endswith(f"/{filename}"):
            return b.name
    return None


def check_track_status():
    print("=" * 70)
    print("CHECK 1: TrackStatus -- real code or parsing bug?")
    print("=" * 70)
    for year in ["2018", "2024"]:
        path = find_one_file(f"raw/fastf1/{year}/", "laps.csv")
        if not path:
            print(f"  No laps.csv found under {year}, skipping.")
            continue
        df = read_csv(path)
        if "TrackStatus" not in df.columns:
            print(f"  {year}: no TrackStatus column.")
            continue
        vc = df["TrackStatus"].value_counts().head(10)
        print(f"\n  {year} ({path}):")
        print(f"    dtype: {df['TrackStatus'].dtype}")
        print(f"    top values:\n{vc.to_string()}")
        print(f"    max value: {df['TrackStatus'].max()}, sample of raw values: {df['TrackStatus'].head(10).tolist()}")


def check_null_columns():
    print("\n" + "=" * 70)
    print("CHECK 2: 100%-null columns -- coverage gap or ingestion bug?")
    print("=" * 70)
    cols_to_check = {
        "laps": ["LapStartDate", "Position"],
        "results": ["GridPosition", "Q1", "Q2", "Q3", "Position", "ClassifiedPosition", "Status", "Points", "Laps", "Time"],
    }
    for table, cols in cols_to_check.items():
        for year in ["2018", "2024"]:
            path = find_one_file(f"raw/fastf1/{year}/", f"{table}.csv")
            if not path:
                continue
            df = read_csv(path)
            print(f"\n  {table}.csv, {year} ({path}):")
            for col in cols:
                if col not in df.columns:
                    print(f"    {col}: COLUMN NOT PRESENT AT ALL")
                    continue
                pct_null = round(df[col].isna().mean() * 100, 1)
                print(f"    {col}: {pct_null}% null (n={len(df)})")


def check_compound_naming():
    print("\n" + "=" * 70)
    print("CHECK 3: Compound naming by year")
    print("=" * 70)
    for year in ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
        path = find_one_file(f"raw/fastf1/{year}/", "laps.csv")
        if not path:
            continue
        df = read_csv(path)
        if "Compound" not in df.columns:
            continue
        uniques = sorted(df["Compound"].dropna().unique().tolist())
        print(f"  {year}: {uniques}")


def check_null_columns_by_session_type():
    """Follow-up check: confirm null columns populate in Q/R sessions, not just FP1."""
    print("\n" + "=" * 70)
    print("CHECK 2b: same null columns, but in Q and R sessions (not FP1)")
    print("=" * 70)
    cols_to_check = ["GridPosition", "Q1", "Q2", "Q3", "Position", "ClassifiedPosition", "Status", "Points", "Laps", "Time"]
    for session in ["Q", "R"]:
        for year in ["2018", "2024"]:
            candidates = [b.name for b in client.list_blobs(BUCKET_NAME, prefix=f"raw/fastf1/{year}/")
                          if b.name.endswith(f"/{session}/results.csv")]
            if not candidates:
                print(f"\n  {year} {session}: no results.csv found under this session type.")
                continue
            path = candidates[0]
            df = read_csv(path)
            print(f"\n  results.csv, {year} {session} ({path}):")
            for col in cols_to_check:
                if col not in df.columns:
                    print(f"    {col}: COLUMN NOT PRESENT")
                    continue
                pct_null = round(df[col].isna().mean() * 100, 1)
                print(f"    {col}: {pct_null}% null (n={len(df)})")


if __name__ == "__main__":
    check_track_status()
    check_null_columns()
    check_compound_naming()
    check_null_columns_by_session_type()
    print("\n" + "=" * 70)
    print("Done. Review the output above before writing any cleaning logic.")
    print("=" * 70)