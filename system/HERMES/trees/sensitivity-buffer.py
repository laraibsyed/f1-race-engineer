"""
DEADLINE_BUFFER_LAPS Sensitivity Sweep
==========================================
gate_tree_tier2.py's deadline_approaching() flags urgency when the mandatory
2-compound rule isn't yet satisfied and laps_remaining_in_race <= buffer.
DEADLINE_BUFFER_LAPS=5 was a placeholder - not specified anywhere in the MDP
doc. This grounds the choice in real historical behaviour instead of a guess.

Method: for every (Season, Race), Race sessions only (mandatory compound rule
is a race rule, not a sprint one), for every driver who does NOT get the
wet-race exception (never ran INTERMEDIATE/WET all race), find the lap on
which their SECOND distinct dry compound was first used - the lap they became
rule-compliant - and compute how many laps remained in the race at that
point. This produces a real distribution of "how much margin do actual
strategies carry when they comply." The buffer should sit near the tight end
of that distribution - large enough to rarely misfire on ordinary two-stop
strategies (which comply early, with laps to spare), small enough to still
catch genuinely late compliance before it becomes a problem.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")


class CachedBucket:
    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=CACHE_DIR):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def read_csv(self, blob_path, **kwargs):
        local_path = os.path.join(self.cache_dir, blob_path)
        if os.path.exists(local_path):
            return pd.read_csv(local_path, **kwargs)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob = self.bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        return pd.read_csv(local_path, **kwargs)

    def list_blob_names(self, prefix):
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]


DRY_COMPOUNDS = {"HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"WET", "INTERMEDIATE"}


def load_race_sessions(bucket: CachedBucket) -> pd.DataFrame:
    """Race sessions ONLY - the mandatory compound rule is a race rule."""
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv"):
            continue
        if "/R/" not in p:
            continue
        df = bucket.read_csv(p)
        parts = p.split("/")
        df["Season"] = int(parts[2])
        df["Race"] = parts[3]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_compliance_margins(laps: pd.DataFrame) -> pd.DataFrame:
    """
    For every driver-race, finds compliance_lap (lap of 2nd distinct dry
    compound) and laps_remaining_at_compliance. Excludes wet-exception drivers
    (ran INTERMEDIATE/WET at any point - the rule doesn't apply to them) and
    non-compliant drivers (never reached a 2nd dry compound - DNF or a real
    rule violation, either way not informative about normal margin).
    """
    rows = []
    for (season, race), race_laps in laps.groupby(["Season", "Race"]):
        total_race_laps = race_laps["LapNumber"].max()
        for driver, g in race_laps.groupby("Driver"):
            g = g.sort_values("LapNumber")
            compounds_seen = set()
            wet_exception = False
            compliance_lap = None
            for _, row in g.iterrows():
                compound = row["Compound"]
                if compound in WET_COMPOUNDS:
                    wet_exception = True
                elif compound in DRY_COMPOUNDS:
                    compounds_seen.add(compound)
                    if len(compounds_seen) >= 2 and compliance_lap is None:
                        compliance_lap = row["LapNumber"]
            if wet_exception or compliance_lap is None:
                continue
            rows.append({
                "season": season, "race": race, "driver": driver,
                "compliance_lap": compliance_lap, "total_race_laps": total_race_laps,
                "laps_remaining_at_compliance": total_race_laps - compliance_lap,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling Race session laps_features.csv ...")
    laps = load_race_sessions(bucket)
    print(f"[load] {len(laps)} raw rows")

    margins = compute_compliance_margins(laps)
    print(f"[compute] {len(margins)} driver-races with valid dry-compound compliance "
          f"(wet-exception and non-compliant cases excluded)")
    margins.to_csv("deadline_buffer_compliance_margins.csv", index=False)

    print("\n=== Distribution of laps_remaining_at_compliance ===")
    print(margins["laps_remaining_at_compliance"]
          .describe(percentiles=[.01, .05, .10, .25, .50]).to_string())

    candidate_buffers = list(range(1, 21))
    sweep_rows = [{"buffer_laps": b,
                    "pct_races_flagged": (margins["laps_remaining_at_compliance"] <= b).mean() * 100}
                   for b in candidate_buffers]
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("deadline_buffer_sensitivity.csv", index=False)

    print("\n=== Sweep: % of historical races that would trigger 'deadline approaching' ===")
    print(sweep_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["buffer_laps"], sweep_df["pct_races_flagged"], marker="o")
    ax.axvline(5, color="grey", linestyle="--", alpha=0.6, label="Original placeholder (5 laps)")
    ax.set_xlabel("Candidate DEADLINE_BUFFER_LAPS value")
    ax.set_ylabel("% of historical compliant races flagged as urgent")
    ax.set_title("Deadline buffer sensitivity - grounded in real compliance timing")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("deadline_buffer_sensitivity.png", dpi=150)
    print("\n[save] deadline_buffer_sensitivity.png / .csv")