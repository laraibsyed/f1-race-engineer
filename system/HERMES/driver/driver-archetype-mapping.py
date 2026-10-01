import argparse
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
MIN_CAREER_RACES = 20
ROOKIE_RACE_CUTOFF = 10

SELF_INFLICTED_RATE_THRESHOLD = 0.08

POINTS_PER_RACE_THRESHOLD = 2.0

ARCHETYPES_PATH = "src/taxanomy/drivers_archetypes.xlsx"

SELF_INFLICTED_KEYWORDS = ["accident", "collision", "spun off", "spin", "damage", "off track"]

def classify_status(status: str) -> str:
    s = str(status).lower()
    return "self_inflicted" if any(k in s for k in SELF_INFLICTED_KEYWORDS) else "other"

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

def load_all_results(bucket: CachedBucket) -> pd.DataFrame:
    paths = [p for p in bucket.list_blob_names("raw/fastf1/") if p.endswith("/R/results.csv")]
    frames = []
    for p in paths:
        parts = p.split("/")
        season, race = int(parts[2]), parts[3]
        df = bucket.read_csv(p, usecols=["Abbreviation", "Points", "Status"])
        df["season"], df["race"] = season, race
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rookie-race-cutoff", type=int, default=ROOKIE_RACE_CUTOFF)
    parser.add_argument("--master-csv", default="driver_taxonomy_master_final.csv",
                         help="The already-built 6-metric CSV for qualifying (>=20 race) drivers")
    parser.add_argument("--output", default="driver_taxonomy_complete_roster.csv")
    args = parser.parse_args()

    bucket = CachedBucket()
    print("[load] scanning all results.csv (Points, Status)...")
    results = load_all_results(bucket)
    results["status_class"] = results["Status"].apply(classify_status)

    per_driver = results.groupby("Abbreviation").agg(
        career_races=("season", "count"),
        avg_points_per_race=("Points", "mean"),
    )
    self_inflicted_counts = results[results["status_class"] == "self_inflicted"].groupby("Abbreviation").size()
    per_driver["self_inflicted_dnf_rate"] = (self_inflicted_counts / per_driver["career_races"]).fillna(0)

    sub20 = per_driver[per_driver["career_races"] < MIN_CAREER_RACES].copy()
    print(f"[scope] {len(sub20)} drivers below {MIN_CAREER_RACES} career races need archetype fallback")

    rookie_mask = sub20["career_races"] < args.rookie_race_cutoff
    print(f"[scope] {rookie_mask.sum()} true rookies (<{args.rookie_race_cutoff} races), "
          f"{(~rookie_mask).sum()} more experienced ({args.rookie_race_cutoff}-{MIN_CAREER_RACES-1} races)")

    def assign_archetype(row, is_rookie):
        if is_rookie:
            return "rookie_aggressive" if row["self_inflicted_dnf_rate"] > SELF_INFLICTED_RATE_THRESHOLD else "rookie_conservative"
        return "junior_high_potential" if row["avg_points_per_race"] > POINTS_PER_RACE_THRESHOLD else "senior_backmarker"

    sub20["archetype"] = [assign_archetype(row, rookie_mask.loc[d]) for d, row in sub20.iterrows()]

    print("\n=== Archetype assignments (SANITY-CHECK these against drivers you know) ===")
    print(sub20[["career_races", "self_inflicted_dnf_rate", "avg_points_per_race", "archetype"]]
          .sort_values("career_races", ascending=False).to_string())

    archetypes_df = pd.read_excel(ARCHETYPES_PATH)
    print(f"\n[load] {ARCHETYPES_PATH} columns: {archetypes_df.columns.tolist()}")
    archetypes_df = archetypes_df.set_index("archetype")

    metric_cols = ["aggression_level", "tyre_management", "consistency_factor",
                   "wet_weather_skill", "pressure_risk_tolerance", "defensive_strength"]
    fallback_rows = []
    for driver, row in sub20.iterrows():
        archetype_row = archetypes_df.loc[row["archetype"]]
        fallback_rows.append({
            "driver": driver, "source": "archetype_fallback", "archetype": row["archetype"],
            "career_races": row["career_races"],
            **{m: archetype_row[m] for m in metric_cols if m in archetype_row}
        })
    fallback_df = pd.DataFrame(fallback_rows)

    master = pd.read_csv(args.master_csv)
    master["source"] = "real_computed"
    master["archetype"] = None

    combined = pd.concat([master, fallback_df], ignore_index=True, sort=False)
    combined.to_csv(args.output, index=False)
    print(f"\n[save] {args.output} - {len(combined)} total drivers "
          f"({len(master)} real_computed + {len(fallback_df)} archetype_fallback)")
