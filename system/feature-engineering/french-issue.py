"""
investigate_french_gp_2022_vsc.py

One-off diagnostic: 2022 French GP Race, lap 53 -- messages.csv says VSC
was deployed on this lap, but is_vsc_lap (derived from TrackStatus in
clean_laps.py) is False for every row on that lap. Since French GP 2022
is a 53-lap race, lap 53 is the final lap -- checking whether this is a
last-lap TrackStatus recording cutoff, an off-by-one in lap numbering
between messages.csv and laps.csv, or something else.

Requirements:
    pip install google-cloud-storage pandas python-dotenv --break-system-packages
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

YEAR, RACE, SESSION = "2022", "French_Grand_Prix", "R"


def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")


if __name__ == "__main__":
    # 1. Raw messages.csv -- what does it actually say around lap 53?
    msg_path = f"raw/fastf1/{YEAR}/{RACE}/{SESSION}/messages.csv"
    messages = read_csv_robust(bucket.blob(msg_path).download_as_bytes())
    print("=" * 70)
    print("Race control messages, laps 50-53:")
    print("=" * 70)
    nearby = messages[(messages["Lap"] >= 50) & (messages["Lap"] <= 53)]
    print(nearby[["Time", "Category", "Message", "Status", "Lap"]].to_string(index=False))

    # 2. Cleaned laps -- TrackStatus and is_vsc_lap around lap 53
    laps_path = f"clean/fastf1/{YEAR}/{RACE}/{SESSION}/laps_flagged.csv"
    laps = read_csv_robust(bucket.blob(laps_path).download_as_bytes())
    print()
    print("=" * 70)
    print("laps_flagged.csv, laps 50-53 (one row per driver, showing unique TrackStatus/is_vsc_lap):")
    print("=" * 70)
    nearby_laps = laps[(laps["LapNumber"] >= 50) & (laps["LapNumber"] <= 53)]
    print(nearby_laps.groupby("LapNumber").agg(
        n_rows=("Driver", "count"),
        unique_trackstatus=("TrackStatus", lambda x: sorted(x.astype(str).unique())),
        any_vsc_lap=("is_vsc_lap", "any"),
    ).to_string())

    print()
    print(f"Max LapNumber in laps_flagged.csv: {laps['LapNumber'].max()}")
    print(f"Max Lap in messages.csv: {messages['Lap'].max()}")