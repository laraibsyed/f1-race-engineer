"""
Driver Taxonomy - pressure_risk_tolerance (the 6th and final driver_archetypes.xlsx metric)
=============================================================================================
UNLIKE the other five metrics, this one's exact methodology was never precisely specified
anywhere in the project handoffs - only "buildable from results.csv (Points for championship-gap
context, Status for DNF/incident tracking)". The design below is a first attempt, not a
previously-validated approach. Two teammate-relative components (same pattern as the other
simple metrics):

  1. CLUTCH PERFORMANCE: a driver's finishing position/points relative to their teammate,
     specifically in each season's LAST 3 races (a proxy for "championship pressure" without
     needing to compute full running championship standings) vs. the rest of the season.
     Positive = performs BETTER than teammate specifically when stakes are highest.

  2. SELF-INFLICTED DNF RATE: retirements classified from the `Status` column as driver-caused
     (accident, collision, spun off) vs. mechanical (engine, gearbox, hydraulics, electrical),
     relative to teammate's rate. Higher = more prone to risk-taking that ends in a self-caused DNF.

CRITICAL - RUN --inspect-status FIRST: the STATUS_KEYWORDS classification below is a guess based
on typical F1 terminology, NOT verified against the real, actual unique Status values in this
bucket's results.csv. Never trust a schema/value-set assumption without checking real data first
- the exact lesson this project has hit repeatedly. Run with --inspect-status to print every
unique Status value found before trusting the DNF classification at all.
"""

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
MIN_PRESSURE_RACES = 9   # ASSUMPTION - 3 full seasons' worth of "pressure races" (3/season).
                          # thin_sample alone (based on TOTAL races) doesn't catch this: a driver
                          # can clear 15+ total races while having only 6-8 actual pressure-race
                          # observations feeding the clutch-performance half of the composite -
                          # exactly the small-sample distortion from Bug #6, on a different column
                          # this time. ANT/HAD/BEA/KUB/LAW/COL all had 6-8 pressure races and
                          # clustered suspiciously near the top before this was added.
PRESSURE_RACES_PER_SEASON = 3  # ASSUMPTION - last N races of a season treated as "high pressure".
                                 # Not empirically derived - a real championship-gap calculation
                                 # (points behind the leader with races remaining) would be more
                                 # accurate but requires full standings reconstruction per round,
                                 # out of scope for this simpler-metrics pass. Documented, not hidden.

# ASSUMPTION - verified against REAL Status values from --inspect-status output on the actual
# archive (3590 rows, 9 seasons). Two categories left DELIBERATELY unclassified, not missed:
#   - "Retired" (185 rows - the single largest status!) has no attributable cause in this
#     dataset at all. Excluding it from both categories is the honest choice, but it's a real,
#     acknowledged accuracy limitation - a large share of DNFs simply can't be attributed here.
#   - "Disqualified" (16) is a technical/rules infringement, not really "risk-taking" in a
#     driving sense - left out of self_inflicted deliberately, a judgment call open to revision.
#   - "Puncture" (6) could be driver-induced (aggressive kerb use) or bad luck from debris -
#     genuinely ambiguous, left unclassified rather than guessed either way.
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
    """Returns 'self_inflicted', 'mechanical', or 'finished_or_other'. ASSUMPTION-based -
    see module docstring. A status matching neither keyword list (e.g. 'Finished', 'Lapped',
    or a real-world phrasing this guess didn't anticipate) falls into 'finished_or_other' and
    is correctly excluded from the DNF-rate calculation rather than silently miscounted."""
    s = str(status).lower()
    if any(k in s for k in SELF_INFLICTED_KEYWORDS):
        return "self_inflicted"
    if any(k in s for k in MECHANICAL_KEYWORDS):
        return "mechanical"
    return "finished_or_other"


def load_all_results(bucket: CachedBucket) -> pd.DataFrame:
    """Scans every results.csv in the archive - small files, cheap to load in full,
    same as the career-participation scan reused across every script in this project."""
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
    """Same winsorized, fit_mask-protected scaler as the other five metrics."""
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

    # Mark each season's last N races as "pressure" races
    season_race_order = (results[["season", "race"]].drop_duplicates()
                          .sort_values(["season", "race"]))  # NOTE: alphabetical, not calendar
                                                                # order within a season - see caveat below
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
    reliable = ~df["thin_sample"] & ~df["thin_pressure_sample"]  # BOTH need to be reliable -
        # a driver with plenty of total races but too few pressure races (or vice versa) still
        # shouldn't anchor the scale, since the composite genuinely needs both halves to be trustworthy

    # Composite: clutch performance (higher = better under pressure) minus a risk-taking
    # penalty (higher self-inflicted DNF rate = more risk-prone). Both z-scored on the
    # reliable subset before combining so neither dominates purely from differing raw scales.
    def zscore(s, mask):
        ref = s[mask].dropna()
        return (s - ref.mean()) / ref.std() if ref.std() else pd.Series(0.0, index=s.index)

    clutch_z = zscore(df["avg_pressure_points_delta"], reliable)
    risk_z = zscore(df["self_inflicted_dnf_rate"], reliable)
    composite = clutch_z - risk_z  # documented, adjustable formula - NOT a validated weighting,
                                     # just a reasonable starting combination. Revisit once you
                                     # see whether this ranking matches known drivers' reputations.
    df["pressure_risk_tolerance"] = normalize_to_archetype_range(composite, fit_mask=reliable)

    print(f"\n[coverage] {len(df)}/{len(qualifying_drivers)} qualifying drivers got data, "
          f"{df['thin_sample'].sum()} flagged thin_sample, "
          f"{df['thin_pressure_sample'].sum()} flagged thin_pressure_sample")
    df = df.sort_values("pressure_risk_tolerance", ascending=False)
    df.to_csv(args.output, index=False)
    print(f"\n[save] {args.output}")