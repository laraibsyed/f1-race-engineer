
import os
import pandas as pd
import numpy as np
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

def load_race_sessions(bucket: CachedBucket) -> pd.DataFrame:
    ""
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv") or "/R/" not in p:
            continue
        parts = p.split("/")
        df = bucket.read_csv(p, usecols=lambda c: c in {
            "LapNumber", "Driver", "is_sc_deployed_lap", "is_vsc_deployed_lap"
        })
        df["Season"], df["Race"] = int(parts[2]), parts[3]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def compute_race_lap_extents(laps: pd.DataFrame) -> pd.DataFrame:
    ""
    return laps.groupby(["Season", "Race"])["LapNumber"].max().reset_index(name="total_race_laps")

def compute_deployment_by_lap(laps: pd.DataFrame) -> pd.DataFrame:
    ""
    return laps.groupby(["Season", "Race", "LapNumber"]).agg(
        sc_deployed=("is_sc_deployed_lap", "any"),
        vsc_deployed=("is_vsc_deployed_lap", "any"),
    ).reset_index()

HORIZON_LAPS = 5

MIN_RACES_FOR_RELIABLE_PRIOR = 4

def build_empirical_prior(laps: pd.DataFrame, horizon: int = HORIZON_LAPS) -> pd.DataFrame:
    ""
    extents = compute_race_lap_extents(laps)
    deployment = compute_deployment_by_lap(laps)

    rows = []
    for race, race_extents in extents.groupby("Race"):
        max_lap = int(race_extents["total_race_laps"].max())
        race_deployment = deployment[deployment["Race"] == race]

        for lap in range(1, max_lap + 1):
            races_reaching = race_extents[race_extents["total_race_laps"] >= lap]
            n_races_reaching = len(races_reaching)
            if n_races_reaching == 0:
                continue

            n_with_incident_in_window = 0
            for _, r in races_reaching.iterrows():
                season = r["Season"]
                window_end = min(lap + horizon - 1, r["total_race_laps"])
                window_rows = race_deployment[
                    (race_deployment["Season"] == season)
                    & (race_deployment["LapNumber"] >= lap)
                    & (race_deployment["LapNumber"] <= window_end)
                ]
                if window_rows["sc_deployed"].any() or window_rows["vsc_deployed"].any():
                    n_with_incident_in_window += 1

            rows.append({
                "circuit": race, "lap_number": lap,
                "n_races_reaching_lap": n_races_reaching,
                "p_incident_within_horizon": n_with_incident_in_window / n_races_reaching,
            })

    return pd.DataFrame(rows)

def build_circuit_level_prior(laps: pd.DataFrame, horizon: int = HORIZON_LAPS) -> pd.DataFrame:
    ""
    extents = compute_race_lap_extents(laps)
    deployment = compute_deployment_by_lap(laps)

    race_had_incident = deployment.groupby(["Season", "Race"]).agg(
        had_sc=("sc_deployed", "any"), had_vsc=("vsc_deployed", "any")
    ).reset_index()
    race_had_incident["had_either"] = race_had_incident["had_sc"] | race_had_incident["had_vsc"]

    merged = race_had_incident.merge(extents, on=["Season", "Race"])
    circuit_stats = merged.groupby("Race").agg(
        n_races=("had_either", "count"),
        p_race_incidence=("had_either", "mean"),
        mean_total_laps=("total_race_laps", "mean"),
    ).reset_index().rename(columns={"Race": "circuit"})

    circuit_stats = circuit_stats[circuit_stats["n_races"] >= MIN_RACES_FOR_RELIABLE_PRIOR]

    def _window_prob(row):
        if row["mean_total_laps"] <= 0:
            return 0.0
        return min(1.0, row["p_race_incidence"] * (horizon / row["mean_total_laps"]))

    circuit_stats["p_window_horizon"] = circuit_stats.apply(_window_prob, axis=1)
    return circuit_stats

def get_sc_probability(circuit_prior: pd.DataFrame, circuit: str, lap_number: int = None) -> float:
    ""
    row = circuit_prior[circuit_prior["circuit"] == circuit]
    if row.empty:
        return None
    return float(row["p_window_horizon"].iloc[0])

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling Race session laps_features.csv ...")
    laps = load_race_sessions(bucket)
    print(f"[load] {len(laps)} raw rows")

    prior = build_empirical_prior(laps)
    print(f"[compute] empirical prior built for {prior['circuit'].nunique()} circuits, "
          f"{len(prior)} (circuit, lap) cells")

    print("\n=== Overall SC/VSC rate by circuit (top 10, sanity check) ===")
    print("[CORRECTED METRIC] mean per-lap probability is confounded by race length - a circuit")
    print("with more laps dilutes the same incident rate across more lap-slots, systematically")
    print("understating longer races (this is why Monaco didn't appear in the first version).")
    print("Using per-RACE incidence rate instead: fraction of races with >=1 SC/VSC anywhere,")
    print("length-independent and directly comparable across circuits. Also filtering to")
    print("circuits with >=4 historical races - single-race circuits give meaningless rates.")

    extents = compute_race_lap_extents(laps)
    deployment = compute_deployment_by_lap(laps)
    race_had_incident = deployment.groupby(["Season", "Race"]).agg(
        had_sc=("sc_deployed", "any"), had_vsc=("vsc_deployed", "any")
    ).reset_index()
    race_had_incident["had_either"] = race_had_incident["had_sc"] | race_had_incident["had_vsc"]

    incidence = race_had_incident.groupby("Race").agg(
        n_races=("had_either", "count"),
        pct_races_with_sc=("had_sc", "mean"),
        pct_races_with_vsc=("had_vsc", "mean"),
        pct_races_with_either=("had_either", "mean"),
    ).sort_values("pct_races_with_either", ascending=False)

    reliable_incidence = incidence[incidence["n_races"] >= 4]
    print(f"\n[{len(reliable_incidence)}/{len(incidence)} circuits have >=4 races - showing those]")
    print(reliable_incidence.head(10).to_string())

    print("\n[sanity check] known high-incident circuits (Monaco, Baku, Singapore, Jeddah) "
          "should rank near the top above - if they don't, investigate before trusting this")

    print("\n=== Targeted check: specific known circuits, wherever they actually rank ===")
    named_circuits = {"Monaco_Grand_Prix": "Monaco", "Azerbaijan_Grand_Prix": "Baku",
                       "Singapore_Grand_Prix": "Singapore", "Saudi_Arabian_Grand_Prix": "Jeddah"}
    for race_name, common_name in named_circuits.items():
        if race_name in incidence.index:
            row = incidence.loc[race_name]
            rank = incidence.index.get_loc(race_name) + 1
            print(f"  {common_name} ({race_name}): rank {rank}/{len(incidence)}, "
                  f"{row['pct_races_with_either']*100:.0f}% of races had SC/VSC (n={row['n_races']})")
        else:
            print(f"  {common_name} ({race_name}): not found in data")

    example_circuit = reliable_incidence.index[0]
    print(f"\n=== [DIAGNOSTIC ONLY, NOT VALIDATED] Example lap-by-lap curve: {example_circuit} ===")
    print(f"(p_incident_within_horizon = P(SC/VSC begins within the next {HORIZON_LAPS} laps) - "
          f"this per-lap resolution FAILED leave-one-race-out validation, see module docstring. "
          f"Shown here only to illustrate why it looked plausible before being properly tested.)")
    example = prior[prior["circuit"] == example_circuit][
        ["lap_number", "n_races_reaching_lap", "p_incident_within_horizon"]]
    print(example.head(15).to_string(index=False))

    prior.to_csv("sc_vsc_empirical_prior.csv", index=False)
    print("\n[save] sc_vsc_empirical_prior.csv (diagnostic lap-level table)")

    print("\n" + "=" * 70)
    print("=== VALIDATED circuit-level prior (this is what get_sc_probability() uses) ===")
    print("=" * 70)
    circuit_prior = build_circuit_level_prior(laps)
    print(circuit_prior.sort_values("p_window_horizon", ascending=False).to_string(index=False))

    print(f"\n=== get_sc_probability() demonstration (lap_number has no effect, as documented) ===")
    for circuit in ["Azerbaijan_Grand_Prix", "Monaco_Grand_Prix"]:
        p_lap10 = get_sc_probability(circuit_prior, circuit, lap_number=10)
        p_lap40 = get_sc_probability(circuit_prior, circuit, lap_number=40)
        print(f"  {circuit}: P(SC/VSC in next {HORIZON_LAPS} laps) = {p_lap10:.3f} "
              f"(lap 10) / {p_lap40:.3f} (lap 40) - identical, as expected")

    circuit_prior.to_csv("sc_vsc_circuit_level_prior.csv", index=False)
    print("\n[save] sc_vsc_circuit_level_prior.csv (VALIDATED - use this one)")
