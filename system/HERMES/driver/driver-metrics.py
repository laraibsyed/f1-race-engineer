"""
Data Discovery — What Exists for aggression_level / pressure_risk_tolerance /
defensive_strength?
================================================================================
Before building anything for these three metrics, find out what's ACTUALLY in
the bucket rather than assume. Lists top-level structure, checks for
standings/results tables, and checks whether tracinginsights telemetry
(known to have drs/DriverAhead/DistanceToDriverAhead fields, confirmed from
the sample JSON shared earlier) is raw per-lap files or already aggregated.
"""

import os
from collections import Counter
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")


def top_level_prefixes(bucket, prefix="", delimiter="/"):
    """Lists immediate 'folders' under a prefix, without descending into them -
    cheap way to see bucket structure without listing everything."""
    iterator = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    blobs = list(iterator)  # must consume to populate .prefixes
    return list(iterator.prefixes), [b.name for b in blobs]


if __name__ == "__main__":
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    print("=== Top-level bucket structure ===")
    prefixes, files = top_level_prefixes(bucket)
    for p in sorted(prefixes):
        print(f"  [folder] {p}")
    for f in sorted(files):
        print(f"  [file]   {f}")

    print("\n=== Looking inside 'raw/' and 'clean/' for anything standings/results/telemetry-like ===")
    for base in ["raw/", "clean/"]:
        sub_prefixes, sub_files = top_level_prefixes(bucket, prefix=base)
        print(f"\n--- {base} ---")
        for p in sorted(sub_prefixes):
            print(f"  [folder] {p}")
        for f in sorted(sub_files)[:10]:
            print(f"  [file]   {f}")

    print("\n=== Checking specifically for standings/results/points data ===")
    keywords = ["standings", "results", "points", "championship"]
    all_blobs = list(bucket.list_blobs())
    matches = [b.name for b in all_blobs if any(k in b.name.lower() for k in keywords)]
    if matches:
        print(f"[found] {len(matches)} matching blobs, showing first 15:")
        for m in matches[:15]:
            print(f"  {m}")
    else:
        print("[not found] no blob names contain 'standings', 'results', 'points', or 'championship'")

    print("\n=== Checking tracinginsights telemetry: raw per-lap files, or aggregated? ===")
    tel_blobs = [b.name for b in all_blobs if "tracinginsights" in b.name.lower() and "tel" in b.name.lower()]
    print(f"[found] {len(tel_blobs)} telemetry-related blobs")
    if tel_blobs:
        print("Sample paths:")
        for t in tel_blobs[:5]:
            print(f"  {t}")
        # crude check: if there are thousands of small per-lap files, it's raw
        if len(tel_blobs) > 1000:
            print(f"[note] {len(tel_blobs)} files strongly suggests RAW per-lap telemetry, "
                  f"not pre-aggregated - processing this for overtakes/DRS would be a "
                  f"substantial new data-engineering task, not a quick reuse")

    print("\n=== Checking laps_features.csv columns directly (one sample file) ===")
    sample_paths = [b.name for b in all_blobs if b.name.endswith("laps_features.csv")]
    if sample_paths:
        import pandas as pd
        sample = sample_paths[0]
        print(f"[sample] {sample}")
        blob = bucket.blob(sample)
        local = "/tmp/sample_laps_features.csv"
        blob.download_to_filename(local)
        df = pd.read_csv(local, nrows=5)
        print("Columns:", df.columns.tolist())