"""
Follow-up discovery: what's actually IN telemetry_by_lap.csv and results.csv?
=================================================================================
The blob count alone doesn't tell us if telemetry_by_lap.csv is a nicely
aggregated one-row-per-lap file (easy to use) or something else. And we
haven't looked at results.csv's actual columns yet, despite confirming it
exists. Check both directly before deciding what's realistically buildable
for aggression_level / pressure_risk_tolerance / defensive_strength.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")

if __name__ == "__main__":
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    print("=== telemetry_by_lap.csv structure ===")
    tel_path = "clean/tracinginsights/2018/Abu_Dhabi_Grand_Prix/R/telemetry_by_lap.csv"
    blob = bucket.blob(tel_path)
    local = "/tmp/sample_telemetry_by_lap.csv"
    blob.download_to_filename(local)
    df = pd.read_csv(local)
    print(f"[shape] {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[columns] {df.columns.tolist()}")
    print(f"\n[sample rows]")
    print(df.head(10).to_string())

    print("\n\n=== results.csv structure ===")
    res_path = "raw/fastf1/2018/Abu_Dhabi_Grand_Prix/R/results.csv"
    blob2 = bucket.blob(res_path)
    local2 = "/tmp/sample_results.csv"
    blob2.download_to_filename(local2)
    df2 = pd.read_csv(local2)
    print(f"[shape] {df2.shape[0]} rows, {df2.shape[1]} columns")
    print(f"[columns] {df2.columns.tolist()}")
    print(f"\n[sample rows]")
    print(df2.head(10).to_string())

    print("\n\n=== How many total tracinginsights blobs are RAW per-lap-per-driver vs clean aggregated? ===")
    raw_tel = list(bucket.list_blobs(prefix="raw/tracinginsights/"))
    clean_tel = list(bucket.list_blobs(prefix="clean/tracinginsights/"))
    print(f"[raw/tracinginsights/] {len(raw_tel)} blobs")
    print(f"[clean/tracinginsights/] {len(clean_tel)} blobs")
    if raw_tel:
        print(f"  raw sample: {raw_tel[0].name}")
    if clean_tel:
        print(f"  clean sample: {clean_tel[0].name}")