
import argparse
import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
MIN_CAREER_RACES = 20
MIN_RACES_WITH_DATA = 15
MIN_PRESSURE_RACES = 9

PRESSURE_RACES_PER_SEASON = 3

SELF_INFLICTED_KEYWORDS = ["accident", "collision", "spun off", "spin", "damage", "off track"]
MECHANICAL_KEYWORDS = ["engine", "gearbox", "hydraulic", "electrical", "electronic", "power unit",
                        "power loss", "brakes", "suspension", "transmission", "clutch", "fuel",
                        "overheating", "wheel", "exhaust", "water pressure", "water leak",
                        "water pump", "oil leak", "turbo", "steering", "radiator", "battery",
                        "driveshaft", "differential", "cooling", "mechanical", "tyre",
                        "vibrations", "undertray"]

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

def classify_status(status: str) -> str:
    ""
    s = str(status).lower()
    if any(k in s for k in SELF_INFLICTED_KEYWORDS):
        return "self_inflicted"
    if any(k in s for k in MECHANICAL_KEYWORDS):
        return "mechanical"
    return "finished_or_other"

def load_all_results(bucket: CachedBucket) -> pd.DataFrame:
    ""
    paths = [p for p in bucket.list_blob_names("raw/fastf1/") if p.endswith("/R/results.csv")]
    frames = []
    for p in paths:
        parts = p.split("/")
        season, race = int(parts[2]), parts[3]
        df = bucket.read_csv(p, usecols=["Abbreviation", "TeamName", "Points", "Status",
                                          "ClassifiedPosition", "GridPosition"])
        df["season"], df["race"] = season, race
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def normalize_to_archetype_range(series, low=0.85, high=1.15, fit_mask=None, winsorize_pct=0.05):
    ""
    fit_values = series[fit_mask] if fit_mask is not None else series
    fit_values = fit_values.dropna()
    if fit_values.empty:
        return pd.Series(1.0, index=series.index)
    lo, hi = fit_values.quantile(winsorize_pct), fit_values.quantile(1 - winsorize_pct)
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    scaled = low + (series - lo) / (hi - lo) * (high - low)
    return scaled.clip(low, high)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-status", action="store_true",
                         help="Print every unique Status value in the archive and STOP - "
                              "run this FIRST, before trusting SELF_INFLICTED_KEYWORDS/"
                              "MECHANICAL_KEYWORDS above at all.")
    parser.add_argument("--min-races-with-data", type=int, default=MIN_RACES_WITH_DATA)
    parser.add_argument("--min-pressure-races", type=int, default=MIN_PRESSURE_RACES)
    parser.add_argument("--pressure-races-per-season", type=int, default=PRESSURE_RACES_PER_SEASON)
    parser.add_argument("--output", default="driver_pressure_risk_tolerance.csv")
    args = parser.parse_args()

    bucket = CachedBucket()
    print("[load] scanning all results.csv (Points, Status, ClassifiedPosition, GridPosition)...")
    results = load_all_results(bucket)
    print(f"[load] {len(results)} rows across {results['race'].nunique()} circuits, "
          f"{results['season'].nunique()} seasons")

    if args.inspect_status:
        print("\n=== EVERY unique Status value in the archive (verify keyword lists against this) ===")
        for v, count in results["Status"].value_counts().items():
            tag = classify_status(v)
            print(f"  [{tag:>18}] {v!r}  (n={count})")
        print("\n[inspect-status] stopping here - re-run without this flag once the keyword "
              "lists above are confirmed to actually match real values")
        raise SystemExit(0)

    results["status_class"] = results["Status"].apply(classify_status)

    race_counts = results.groupby("Abbreviation").size()
    qualifying_drivers = set(race_counts[race_counts >= MIN_CAREER_RACES].index)
    print(f"[scope] {len(qualifying_drivers)} qualifying drivers (>= {MIN_CAREER_RACES} career races)")

    season_race_order = (results[["season", "race"]].drop_duplicates()
                          .sort_values(["season", "race"]))

    season_race_order["race_rank_in_season"] = season_race_order.groupby("season").cumcount(ascending=False)
    pressure_races = set(season_race_order[season_race_order["race_rank_in_season"] < args.pressure_races_per_season]
                         [["season", "race"]].itertuples(index=False, name=None))
    results["is_pressure_race"] = results.apply(lambda r: (r["season"], r["race"]) in pressure_races, axis=1)
    print(f"[caveat] 'last 3 races of the season' is ordered ALPHABETICALLY by race name here, "
          f"NOT by actual calendar date - results.csv doesn't carry a round number. This is a "
          f"real limitation, not fixed - if you have a round-number source elsewhere, redo this "
          f"with the true calendar order before trusting the pressure-race split.")

    accum = {}
    for (season, race), race_df in results.groupby(["season", "race"]):
        by_team = race_df.groupby("TeamName")["Abbreviation"].apply(list).to_dict()
        for _, row in race_df.iterrows():
            driver = row["Abbreviation"]
            if driver not in qualifying_drivers:
                continue
            teammates = [d for d in by_team.get(row["TeamName"], []) if d != driver]
            if not teammates:
                continue
            teammate_rows = race_df[race_df["Abbreviation"].isin(teammates)]

            stats = accum.setdefault(driver, {"pressure_points_delta": [], "self_inflicted": 0,
                                                "mechanical": 0, "total_races": 0})
            stats["total_races"] += 1
            if row["status_class"] == "self_inflicted":
                stats["self_inflicted"] += 1
            elif row["status_class"] == "mechanical":
                stats["mechanical"] += 1

            if row["is_pressure_race"] and not teammate_rows.empty:
                teammate_points = teammate_rows["Points"].mean()
                stats["pressure_points_delta"].append(row["Points"] - teammate_points)

    rows = []
    for driver, stats in accum.items():
        n = stats["total_races"]
        n_pressure = len(stats["pressure_points_delta"])
        rows.append({
            "driver": driver,
            "races_with_data": n,
            "pressure_races_with_data": n_pressure,
            "thin_sample": n < args.min_races_with_data,
            "thin_pressure_sample": n_pressure < args.min_pressure_races,
            "self_inflicted_dnf_rate": stats["self_inflicted"] / n if n else np.nan,
            "mechanical_dnf_rate": stats["mechanical"] / n if n else np.nan,
            "avg_pressure_points_delta": float(np.mean(stats["pressure_points_delta"])) if n_pressure else np.nan,
        })

    df = pd.DataFrame(rows)
    reliable = ~df["thin_sample"] & ~df["thin_pressure_sample"]

    def zscore(s, mask):
        ref = s[mask].dropna()
        return (s - ref.mean()) / ref.std() if ref.std() else pd.Series(0.0, index=s.index)

    clutch_z = zscore(df["avg_pressure_points_delta"], reliable)
    risk_z = zscore(df["self_inflicted_dnf_rate"], reliable)
    composite = clutch_z - risk_z

    df["pressure_risk_tolerance"] = normalize_to_archetype_range(composite, fit_mask=reliable)

    print(f"\n[coverage] {len(df)}/{len(qualifying_drivers)} qualifying drivers got data, "
          f"{df['thin_sample'].sum()} flagged thin_sample, "
          f"{df['thin_pressure_sample'].sum()} flagged thin_pressure_sample")
    df = df.sort_values("pressure_risk_tolerance", ascending=False)
    df.to_csv(args.output, index=False)
    print(f"\n[save] {args.output}")
