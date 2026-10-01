
import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")

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

    def list_blob_names(self, prefix):
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]

def load_all_laps_features(bucket: CachedBucket) -> pd.DataFrame:
    ""
    paths = [p for p in bucket.list_blob_names("clean/features/") if p.endswith("/R/laps_features.csv")]
    frames = []
    for p in paths:
        df = bucket.read_csv(p, dtype={"TrackStatus": str})
        parts = p.split("/")
        df["Season"] = int(parts[2])
        df["Race"] = parts[3]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full

def join_profiles_to_laps(laps: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    ""
    before = len(laps)

    profiles_renamed = profiles.rename(columns={
        "driver": "Driver", "season": "Season", "target_race": "Race",
    })

    merged = laps.merge(
        profiles_renamed[["Driver", "Season", "Race", "prior_career_races",
                           "tyre_management_pti", "consistency_factor_pti", "profile_source"]],
        on=["Driver", "Season", "Race"],
        how="left",
        validate="many_to_one",
    )

    after = len(merged)
    if after != before:
        raise AssertionError(
            f"Row count changed on join ({before} -> {after}) -- "
            "duplicate (Driver, Season, Race) keys in the profile CSV, stop and investigate."
        )
    print(f"[ok] lap row count unchanged by join ({before}).")

    unmatched_mask = merged["profile_source"].isna()
    n_unmatched = unmatched_mask.sum()
    if n_unmatched:
        unmatched_pairs = (
            merged.loc[unmatched_mask, ["Driver", "Season", "Race"]]
            .drop_duplicates()
            .sort_values(["Season", "Driver"])
        )
        print(f"[warn] {n_unmatched} lap rows ({len(unmatched_pairs)} unique driver/race pairs) "
              "had no matching profile row -- driver-code mismatch, or a race the profile "
              "script didn't see. NOT silently dropped; investigate before using these rows "
              "in the ablation.")
        print(unmatched_pairs.to_string(index=False))
    else:
        print("[ok] every lap row matched a profile row.")

    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-csv", default="driver_rolling_profiles.csv")
    parser.add_argument("--output", default="laps_features_with_driver_profiles.csv")
    args = parser.parse_args()

    bucket = CachedBucket()

    print("[load] pulling laps_features.csv archive ...")
    laps = load_all_laps_features(bucket)
    print(f"[load] {len(laps)} lap rows across {laps['Season'].nunique()} seasons")

    print(f"[load] reading {args.profiles_csv} ...")
    profiles = pd.read_csv(args.profiles_csv)
    print(f"[load] {len(profiles)} profile rows "
          f"({(profiles['profile_source'] == 'rolling').sum()} rolling, "
          f"{(profiles['profile_source'] == 'fallback').sum()} fallback)")

    merged = join_profiles_to_laps(laps, profiles)

    merged.to_csv(args.output, index=False)
    print(f"\n[save] {args.output} -- {len(merged)} rows, "
          f"{merged['tyre_management_pti'].notna().sum()} with a driver profile attached")
