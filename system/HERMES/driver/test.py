
import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
SEASON, RACE = 2018, "Belgian_Grand_Prix"
DRIVERS_TO_CHECK = ["LEC", "ERI", "ALO", "VAN", "HUL", "SAI"]

class CachedBucket:
    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=os.environ.get("GCS_CACHE_DIR", "./gcs_cache")):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def read_csv(self, blob_path, **kwargs):
        local_path = os.path.join(self.cache_dir, blob_path)
        if os.path.exists(local_path):
            return pd.read_csv(local_path, **kwargs)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.bucket.blob(blob_path).download_to_filename(local_path)
        return pd.read_csv(local_path, **kwargs)

if __name__ == "__main__":
    bucket = CachedBucket()
    path = f"clean/features/{SEASON}/{RACE}/R/laps_features.csv"
    print(f"[load] {path}")
    df = bucket.read_csv(path, usecols=["Driver", "Team"])

    print(f"\n=== Raw (Driver, Team) pairs for {SEASON} {RACE} ===")
    present = df[df["Driver"].isin(DRIVERS_TO_CHECK)][["Driver", "Team"]].drop_duplicates()
    if present.empty:
        print("  [!] NONE of these six drivers appear in this file's Driver column at all.")
        print("  -> genuine data gap for this race, not a naming mismatch. Document, don't fix.")
    else:
        print(present.to_string(index=False))

    missing_entirely = [d for d in DRIVERS_TO_CHECK if d not in df["Driver"].unique()]
    if missing_entirely:
        print(f"\n[!] Drivers with NO rows at all in this file: {missing_entirely}")
        print("    -> genuine absence from source data for this race -- document as a gap.")

    found = [d for d in DRIVERS_TO_CHECK if d in df["Driver"].unique()]
    if found:
        print(f"\n[ok] Drivers WITH rows in this file: {found}")
        print("Team values used (compare these against the taxonomy's expected team names,")
        print("e.g. RBR_ALIASES-style sets elsewhere in this project, for an exact-match check):")
        print(df[df["Driver"].isin(found)]["Team"].value_counts().to_string())

    print("\n=== All distinct Team values in this race file (for spotting a one-off sponsor variant) ===")
    print(df["Team"].value_counts().to_string())
