"""
Archetype Fallback - drivers with < MIN_CAREER_RACES (20) real races
=======================================================================
DESIGN DECISION (documented, not accidental): this uses each driver's FULL career history to
assign an archetype, matching how the five REAL computed metrics (aggression_level,
tyre_management, consistency_factor, wet_weather_skill, pressure_risk_tolerance,
defensive_strength) are also built from each qualifying driver's entire career, not a rolling
window. An earlier version of this script made archetype assignment leak-free/rolling
(point-in-time, no future data) while the five real metrics stayed static - which was actually
the WRONG inconsistency: it held rookies to a stricter no-leakage standard than established
drivers, when if anything a rookie's eventual full career is the MORE informative signal about
who they are as a driver, not less.

JUSTIFICATION FOR WHY STATIC IS FINE HERE (for the dissertation write-up): the Second Driver
Problem diagram's "rolling t-1, no leakage" principle applies to LAYER 3 (live, lap-by-lap
hierarchy awareness during a race) - genuinely time-sensitive, in-race state. Driver PROFILES
(Layer 2 - aggression_level, archetype, etc.) are a season-level PRIOR, not live race state -
using a driver's full career to characterize who they are, then feeding that stable profile into
a live strategy engine, is a standard and defensible design, not a leakage bug. Given project
timeline constraints, full-career/static is used consistently across all six metrics AND the
archetype fallback - this is a scoped, deliberate trade-off, not an oversight.

HEURISTIC (still a genuine judgment call, no more-detailed rule existed anywhere in the
project - checked against the Second Driver Problem diagram, which confirms the <20-race branch
exists but doesn't specify HOW to choose among the four specific archetypes):

  AXIS 1 - experience within the sub-20 bracket:
    < ROOKIE_RACE_CUTOFF (10) career races  -> "true rookie" bucket
    >= ROOKIE_RACE_CUTOFF (10) career races -> "more experienced, still building" bucket

  AXIS 2 - performance signal, split at a FIXED threshold (see constants above) - NOT a
  within-population median, because the sub-20 population is small enough (often single digits)
  that a median-based split degenerates: a bucket of size 1 is mathematically guaranteed to fall
  "below its own median", and a rookie bucket where most drivers have exactly 0 self-inflicted
  DNFs (near-certain off 1-7 races) collapses into one label regardless of real differences:
    True rookie bucket    -> self_inflicted_dnf_rate
                              above SELF_INFLICTED_RATE_THRESHOLD -> rookie_aggressive, else rookie_conservative
    Experienced bucket    -> avg_points_per_race
                              above POINTS_PER_RACE_THRESHOLD -> junior_high_potential, else senior_backmarker

SANITY-CHECK BEFORE TRUSTING: once run, check a few drivers you know personally against their
assigned archetype - this heuristic has never been checked against real output.
"""

import argparse
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
MIN_CAREER_RACES = 20
ROOKIE_RACE_CUTOFF = 10  # ASSUMPTION - the split between "true rookie" and "more experienced
                          # but still sub-threshold" - not empirically derived, adjust freely
SELF_INFLICTED_RATE_THRESHOLD = 0.08  # ASSUMPTION - FIXED, not a within-population median.
    # With only a handful of sub-20 drivers at any given time (5 in this run: 4 rookies + 1
    # more-experienced), a median-based split degenerates badly - most 1-7 race rookies have
    # EXACTLY 0 self-inflicted DNFs by pure chance, collapsing the whole rookie bucket into one
    # label, and a bucket of size 1 (as DEV was) is mathematically guaranteed to fall on the
    # "below median" side of ITS OWN VALUE regardless of real performance. A fixed threshold
    # avoids this: a driver's classification now depends only on their own rate, never on how
    # many other thin-sample drivers happen to exist alongside them right now.
POINTS_PER_RACE_THRESHOLD = 2.0  # ASSUMPTION - same fix, same reasoning, for the junior_high_
    # potential/senior_backmarker split. Revisit both numbers once you see real distributions
    # across a larger pool (e.g. once more drivers eventually clear the sub-20 threshold).
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

    # ---- Pull in the generic archetype metric values ----
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

    # ---- Combine with the real, qualifying-driver master table ----
    master = pd.read_csv(args.master_csv)
    master["source"] = "real_computed"
    master["archetype"] = None

    combined = pd.concat([master, fallback_df], ignore_index=True, sort=False)
    combined.to_csv(args.output, index=False)
    print(f"\n[save] {args.output} - {len(combined)} total drivers "
          f"({len(master)} real_computed + {len(fallback_df)} archetype_fallback)")