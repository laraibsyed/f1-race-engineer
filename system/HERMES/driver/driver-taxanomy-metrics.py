
import argparse
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
MIN_CAREER_RACES = 20

MIN_RACES_WITH_DATA = 15

MIN_WET_RACES = 3

CHECKPOINT_DIR = os.environ.get("METRICS_CHECKPOINT_DIR", "./metrics_checkpoints")

EXCLUDE_FLAG_COLUMNS = ["is_pit_in", "is_pit_out", "is_sc_lap", "is_vsc_lap",
                         "is_missing_laptime", "is_outlier_laptime", "is_out_lap", "is_in_lap"]
WET_COMPOUNDS = {"WET", "INTERMEDIATE"}

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

def checkpoint_path(season, race):
    return os.path.join(CHECKPOINT_DIR, f"{season}_{race}.json")

def save_checkpoint(season, race, per_driver: dict, status="ok"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(checkpoint_path(season, race), "w") as f:
        json.dump({"status": status, "data": per_driver}, f)

def load_checkpoint(season, race):
    path = checkpoint_path(season, race)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload["data"] if payload["status"] == "ok" else {}

def build_driver_race_participation(bucket: CachedBucket) -> pd.DataFrame:
    ""
    paths = [p for p in bucket.list_blob_names("raw/fastf1/") if p.endswith("/R/results.csv")]
    rows = []
    for p in paths:
        parts = p.split("/")
        season, race = int(parts[2]), parts[3]
        df = bucket.read_csv(p, usecols=["Abbreviation"])
        for driver in df["Abbreviation"].dropna().unique():
            rows.append({"season": season, "race": race, "driver": driver})
    return pd.DataFrame(rows)

def list_feature_race_files(bucket: CachedBucket) -> list:
    ""
    paths = [p for p in bucket.list_blob_names("clean/features/") if p.endswith("/R/laps_features.csv")]
    out = []
    for p in paths:
        parts = p.split("/")
        out.append((int(parts[2]), parts[3]))
    return out

def process_one_race(bucket: CachedBucket, season: int, race: str) -> dict:
    ""
    path = f"clean/features/{season}/{race}/R/laps_features.csv"
    cols = ["Driver", "Team", "LapTime", "Compound", "degradation_rate"] + EXCLUDE_FLAG_COLUMNS
    df = bucket.read_csv(path, usecols=lambda c: c in cols)

    exclude_mask = pd.Series(False, index=df.index)
    for col in EXCLUDE_FLAG_COLUMNS:
        if col in df.columns:
            exclude_mask |= df[col].fillna(False).astype(bool)
    clean = df[~exclude_mask].copy()
    if clean.empty:
        return {}

    clean["LapTime_seconds"] = pd.to_timedelta(clean["LapTime"], errors="coerce").dt.total_seconds()
    clean = clean.dropna(subset=["LapTime_seconds"])

    per_driver = {}
    for driver, g in clean.groupby("Driver"):
        team = g["Team"].mode().iloc[0] if not g["Team"].mode().empty else None
        mean_degradation = float(g["degradation_rate"].mean()) if "degradation_rate" in g else np.nan
        cv = float(g["LapTime_seconds"].std() / g["LapTime_seconds"].mean()) if g["LapTime_seconds"].mean() else np.nan

        wet_laps = g[g["Compound"].isin(WET_COMPOUNDS)] if "Compound" in g else pd.DataFrame()
        wet_laptime = float(wet_laps["LapTime_seconds"].mean()) if not wet_laps.empty else None

        per_driver[driver] = {"team": team, "degradation": mean_degradation,
                               "consistency_cv": cv, "wet_laptime": wet_laptime}
    return per_driver

def teammate_relative_deltas(per_driver: dict) -> dict:
    ""
    by_team = {}
    for driver, stats in per_driver.items():
        by_team.setdefault(stats["team"], []).append(driver)

    deltas = {}
    for driver, stats in per_driver.items():
        teammates = [d for d in by_team.get(stats["team"], []) if d != driver]
        if not teammates:
            continue

        def teammate_mean(field):
            vals = [per_driver[t][field] for t in teammates if per_driver[t][field] is not None
                     and not (isinstance(per_driver[t][field], float) and np.isnan(per_driver[t][field]))]
            return float(np.mean(vals)) if vals else None

        d = {}
        tm_deg = teammate_mean("degradation")
        if stats["degradation"] is not None and not np.isnan(stats["degradation"]) and tm_deg is not None:
            d["delta_degradation"] = stats["degradation"] - tm_deg

        tm_cv = teammate_mean("consistency_cv")
        if stats["consistency_cv"] is not None and not np.isnan(stats["consistency_cv"]) and tm_cv is not None:
            d["delta_cv"] = stats["consistency_cv"] - tm_cv

        tm_wet = teammate_mean("wet_laptime")
        if stats["wet_laptime"] is not None and tm_wet is not None:
            d["delta_wet_laptime"] = stats["wet_laptime"] - tm_wet

        if d:
            deltas[driver] = d
    return deltas

def normalize_to_archetype_range(series: pd.Series, low=0.85, high=1.15, fit_mask=None,
                                  winsorize_pct: float = 0.05) -> pd.Series:
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
    parser.add_argument("--test-races", type=int, default=None,
                         help="Process only the first N race files - use this FIRST to sanity-check "
                              "column names/values against real data before running the full archive.")
    parser.add_argument("--min-races-with-data", type=int, default=MIN_RACES_WITH_DATA)
    parser.add_argument("--min-wet-races", type=int, default=MIN_WET_RACES)
    parser.add_argument("--output", default="driver_simple_metrics.csv")
    args = parser.parse_args()

    bucket = CachedBucket()

    print("[load] scanning results.csv for career race participation...")
    participation = build_driver_race_participation(bucket)
    race_counts = participation.groupby("driver").size()
    qualifying_drivers = set(race_counts[race_counts >= MIN_CAREER_RACES].index)
    print(f"[scope] {len(qualifying_drivers)} qualifying drivers (>= {MIN_CAREER_RACES} career races)")

    race_files = list_feature_race_files(bucket)
    if args.test_races:
        race_files = race_files[:args.test_races]
        print(f"[test mode] limited to first {args.test_races} race files - "
              f"VERIFY the output looks sane before removing --test-races")
    print(f"[scope] {len(race_files)} race files to process")

    accum = {}
    n_skipped = 0
    for i, (season, race) in enumerate(race_files, 1):
        cached = load_checkpoint(season, race)
        if cached is not None:
            n_skipped += 1
            deltas = cached
        else:
            print(f"[{i}/{len(race_files)}] processing {race} {season} ...")
            try:
                per_driver = process_one_race(bucket, season, race)
                deltas = teammate_relative_deltas(per_driver) if per_driver else {}
            except Exception as e:
                print(f"  [warn] failed: {type(e).__name__}: {e} - skipping, not checkpointed (will retry)")
                continue
            save_checkpoint(season, race, deltas, status="ok" if deltas else "empty")

        for driver, d in deltas.items():
            if driver not in qualifying_drivers:
                continue
            stats = accum.setdefault(driver, {"delta_degradation": [], "delta_cv": [], "delta_wet_laptime": []})
            for key in ("delta_degradation", "delta_cv", "delta_wet_laptime"):
                if key in d:
                    stats[key].append(d[key])

    print(f"\n[resume] {n_skipped}/{len(race_files)} races already checkpointed and skipped")

    rows = []
    for driver, stats in accum.items():
        n_deg = len(stats["delta_degradation"])
        n_cv = len(stats["delta_cv"])
        n_wet = len(stats["delta_wet_laptime"])
        rows.append({
            "driver": driver,
            "races_with_data": max(n_deg, n_cv),
            "wet_races_with_data": n_wet,
            "thin_sample": max(n_deg, n_cv) < args.min_races_with_data,
            "thin_wet_sample": n_wet < args.min_wet_races,
            "avg_delta_degradation": float(np.mean(stats["delta_degradation"])) if n_deg else np.nan,
            "avg_delta_cv": float(np.mean(stats["delta_cv"])) if n_cv else np.nan,
            "avg_delta_wet_laptime": float(np.mean(stats["delta_wet_laptime"])) if n_wet else np.nan,
        })

    df = pd.DataFrame(rows)
    reliable = ~df["thin_sample"]
    reliable_wet = ~df["thin_wet_sample"]

    df["tyre_management"] = normalize_to_archetype_range(-df["avg_delta_degradation"], fit_mask=reliable)
    df["consistency_factor"] = normalize_to_archetype_range(-df["avg_delta_cv"], fit_mask=reliable)
    df["wet_weather_skill"] = normalize_to_archetype_range(-df["avg_delta_wet_laptime"], fit_mask=reliable_wet)

    print(f"\n[coverage] {len(df)}/{len(qualifying_drivers)} qualifying drivers got at least some data")
    print(f"[coverage] {df['thin_sample'].sum()} flagged thin_sample, {df['thin_wet_sample'].sum()} flagged thin_wet_sample")

    df = df.sort_values("tyre_management", ascending=False)
    df.to_csv(args.output, index=False)
    print(f"\n[save] {args.output}")
