"""
SC/VSC Empirical Prior — Leave-One-Race-Out Validation
==========================================================
Answers the checklist's validation question honestly: the empirical prior is
a historical BASE RATE, not a live crash detector - it cannot "predict" a
specific race's specific incident. What it CAN be validated for: does the
lap-level structure (not just circuit-level average) carry real signal -
across many real historical incidents, is the predicted probability at the
ACTUAL incident lap systematically higher than at a random other lap in the
same race?

LEAVE-ONE-RACE-OUT, not in-sample: with only 5-8 races per circuit, a single
incident can meaningfully inflate its own lap's probability estimate in the
full prior - evaluating a race against a prior that includes its own data
would be circular. Each race is evaluated against a prior rebuilt WITHOUT it.

Same paired-comparison + AUC approach as the Cox model's ROC validation
earlier in this project - consistent methodology across the codebase.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Reused from sc_vsc_probability_model.py - duplicated deliberately so this
# validation script is self-contained, same pattern as the rest of this project.
MIN_RACES_FOR_RELIABLE_PRIOR = 4
SMOOTHING_WINDOW = 5


def compute_race_lap_extents(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.groupby(["Season", "Race"])["LapNumber"].max().reset_index(name="total_race_laps")


def compute_deployment_by_lap(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.groupby(["Season", "Race", "LapNumber"]).agg(
        sc_deployed=("is_sc_deployed_lap", "any"),
        vsc_deployed=("is_vsc_deployed_lap", "any"),
    ).reset_index()


def build_prior_for_circuit(extents: pd.DataFrame, deployment: pd.DataFrame, circuit: str,
                              exclude_season: int = None) -> pd.DataFrame:
    """Same logic as sc_vsc_probability_model.build_empirical_prior, but for
    ONE circuit at a time, with an optional season excluded (leave-one-out)."""
    race_extents = extents[extents["Race"] == circuit]
    race_deployment = deployment[deployment["Race"] == circuit]
    if exclude_season is not None:
        race_extents = race_extents[race_extents["Season"] != exclude_season]
        race_deployment = race_deployment[race_deployment["Season"] != exclude_season]

    if race_extents.empty:
        return pd.DataFrame(columns=["lap_number", "p_either_smoothed"])

    max_lap = int(race_extents["total_race_laps"].max())
    rows = []
    for lap in range(1, max_lap + 1):
        n_reach = (race_extents["total_race_laps"] >= lap).sum()
        if n_reach == 0:
            continue
        lap_rows = race_deployment[race_deployment["LapNumber"] == lap]
        n_sc = lap_rows["sc_deployed"].sum()
        n_vsc = lap_rows["vsc_deployed"].sum()
        rows.append({"lap_number": lap, "n_races_reaching_lap": n_reach,
                      "p_sc_raw": n_sc / n_reach, "p_vsc_raw": n_vsc / n_reach})

    prior = pd.DataFrame(rows).sort_values("lap_number")
    if prior.empty:
        return prior
    prior["p_sc_smoothed"] = prior["p_sc_raw"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    prior["p_vsc_smoothed"] = prior["p_vsc_raw"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    prior["p_either_smoothed"] = 1 - (1 - prior["p_sc_smoothed"]) * (1 - prior["p_vsc_smoothed"])
    return prior


def run_leave_one_out_validation(laps: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    extents = compute_race_lap_extents(laps)
    deployment = compute_deployment_by_lap(laps)

    circuit_race_counts = extents.groupby("Race")["Season"].nunique()
    eligible_circuits = circuit_race_counts[circuit_race_counts >= MIN_RACES_FOR_RELIABLE_PRIOR].index

    rows = []
    for circuit in eligible_circuits:
        circuit_extents = extents[extents["Race"] == circuit]
        circuit_deployment = deployment[deployment["Race"] == circuit]

        for season in circuit_extents["Season"].unique():
            race_deploy = circuit_deployment[circuit_deployment["Season"] == season]
            incident_laps = race_deploy.loc[race_deploy["sc_deployed"] | race_deploy["vsc_deployed"], "LapNumber"].tolist()
            if not incident_laps:
                continue  # this race had no SC/VSC at all - nothing to validate against

            loo_prior = build_prior_for_circuit(extents, deployment, circuit, exclude_season=season)
            if loo_prior.empty:
                continue

            total_laps_this_race = int(circuit_extents.loc[circuit_extents["Season"] == season, "total_race_laps"].iloc[0])
            non_incident_laps = [l for l in range(1, total_laps_this_race + 1) if l not in incident_laps]
            if not non_incident_laps:
                continue
            control_lap = rng.choice(non_incident_laps)

            for incident_lap in incident_laps:
                p_incident = loo_prior.loc[loo_prior["lap_number"] == incident_lap, "p_either_smoothed"]
                p_control = loo_prior.loc[loo_prior["lap_number"] == control_lap, "p_either_smoothed"]
                if p_incident.empty or p_control.empty:
                    continue
                rows.append({"circuit": circuit, "season": season, "incident_lap": incident_lap,
                             "control_lap": control_lap, "p_incident": p_incident.iloc[0],
                             "p_control": p_control.iloc[0]})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from google.cloud import storage

    load_dotenv()
    BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
    CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")

    class CachedBucket:
        def __init__(self):
            self.client = storage.Client()
            self.bucket = self.client.bucket(BUCKET_NAME)

        def read_csv(self, blob_path, **kwargs):
            local_path = os.path.join(CACHE_DIR, blob_path)
            if os.path.exists(local_path):
                return pd.read_csv(local_path, **kwargs)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob = self.bucket.blob(blob_path)
            blob.download_to_filename(local_path)
            return pd.read_csv(local_path, **kwargs)

        def list_blob_names(self, prefix):
            return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]

    bucket = CachedBucket()
    print("[load] pulling Race session laps_features.csv ...")
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv") or "/R/" not in p:
            continue
        parts = p.split("/")
        df = bucket.read_csv(p, usecols=lambda c: c in {
            "LapNumber", "Driver", "is_sc_deployed_lap", "is_vsc_deployed_lap"})
        df["Season"], df["Race"] = int(parts[2]), parts[3]
        frames.append(df)
    laps = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(laps)} rows")

    results = run_leave_one_out_validation(laps)
    print(f"\n[validate] {len(results)} real (incident_lap, control_lap) pairs, "
          f"leave-one-race-out, across {results['circuit'].nunique()} circuits")

    pct_incident_higher = (results["p_incident"] > results["p_control"]).mean()
    print(f"\n[result] incident lap has HIGHER predicted P(SC) than control lap: "
          f"{pct_incident_higher*100:.1f}% of the time (50% = no better than chance)")

    labels = [1] * len(results) + [0] * len(results)
    scores = results["p_incident"].tolist() + results["p_control"].tolist()
    auc = roc_auc_score(labels, scores)
    print(f"[result] AUC (incident vs. control, pooled): {auc:.3f} (0.5 = no discrimination)")

    results.to_csv("sc_probability_loo_validation.csv", index=False)
    print("\n[save] sc_probability_loo_validation.csv")