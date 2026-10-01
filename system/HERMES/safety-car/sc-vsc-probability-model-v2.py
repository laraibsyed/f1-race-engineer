
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

load_dotenv()

HORIZON_LAPS = 5
MIN_RACES_FOR_RELIABLE_PRIOR = 4
N_BOOTSTRAP = 1000

class CachedBucket:
    def __init__(self, bucket_name=os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket"),
                 cache_dir=os.environ.get("GCS_CACHE_DIR", "./gcs_cache")):
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
    return pd.concat(frames, ignore_index=True)

def compute_race_lap_extents(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.groupby(["Season", "Race"])["LapNumber"].max().reset_index(name="total_race_laps")

def compute_deployment_by_lap(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.groupby(["Season", "Race", "LapNumber"]).agg(
        sc_deployed=("is_sc_deployed_lap", "any"),
        vsc_deployed=("is_vsc_deployed_lap", "any"),
    ).reset_index()

def build_loo_prior(extents: pd.DataFrame, deployment: pd.DataFrame, circuit: str,
                     exclude_season: int, horizon: int = HORIZON_LAPS) -> pd.DataFrame:
    ""
    race_extents = extents[(extents["Race"] == circuit) & (extents["Season"] != exclude_season)]
    race_deployment = deployment[(deployment["Race"] == circuit) & (deployment["Season"] != exclude_season)]
    if race_extents.empty:
        return pd.DataFrame(columns=["lap_number", "p_incident_within_horizon"])

    max_lap = int(race_extents["total_race_laps"].max())
    rows = []
    for lap in range(1, max_lap + 1):
        races_reaching = race_extents[race_extents["total_race_laps"] >= lap]
        n_reach = len(races_reaching)
        if n_reach == 0:
            continue
        n_incident = 0
        for _, r in races_reaching.iterrows():
            window_end = min(lap + horizon - 1, r["total_race_laps"])
            wr = race_deployment[
                (race_deployment["Season"] == r["Season"])
                & (race_deployment["LapNumber"] >= lap) & (race_deployment["LapNumber"] <= window_end)
            ]
            if wr["sc_deployed"].any() or wr["vsc_deployed"].any():
                n_incident += 1
        rows.append({"lap_number": lap, "p_incident_within_horizon": n_incident / n_reach})
    return pd.DataFrame(rows)

def build_full_validation_dataset(laps: pd.DataFrame, horizon: int = HORIZON_LAPS) -> pd.DataFrame:
    ""
    extents = compute_race_lap_extents(laps)
    deployment = compute_deployment_by_lap(laps)

    circuit_race_counts = extents.groupby("Race")["Season"].nunique()
    eligible_circuits = circuit_race_counts[circuit_race_counts >= MIN_RACES_FOR_RELIABLE_PRIOR].index

    all_rows = []
    for circuit in eligible_circuits:
        circuit_extents = extents[extents["Race"] == circuit]
        circuit_deployment = deployment[deployment["Race"] == circuit]

        for season in circuit_extents["Season"].unique():
            loo_prior = build_loo_prior(extents, deployment, circuit, exclude_season=season, horizon=horizon)
            if loo_prior.empty:
                continue
            loo_lookup = loo_prior.set_index("lap_number")["p_incident_within_horizon"]

            total_laps = int(circuit_extents.loc[circuit_extents["Season"] == season, "total_race_laps"].iloc[0])
            race_deploy = circuit_deployment[circuit_deployment["Season"] == season]

            for lap in range(1, total_laps + 1):
                if lap not in loo_lookup.index:
                    continue
                window_end = min(lap + horizon - 1, total_laps)
                window_rows = race_deploy[(race_deploy["LapNumber"] >= lap) & (race_deploy["LapNumber"] <= window_end)]
                actual = int(window_rows["sc_deployed"].any() or window_rows["vsc_deployed"].any())
                all_rows.append({"circuit": circuit, "season": season, "lap": lap,
                                  "predicted_p": loo_lookup.loc[lap], "actual": actual})

    return pd.DataFrame(all_rows)

def race_level_bootstrap_auc(dataset: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, seed: int = 42) -> tuple:
    ""
    rng = np.random.default_rng(seed)
    race_keys = dataset[["circuit", "season"]].drop_duplicates().to_records(index=False).tolist()

    aucs = []
    for _ in range(n_bootstrap):
        sampled_keys = rng.choice(len(race_keys), size=len(race_keys), replace=True)
        pieces = []
        for idx in sampled_keys:
            circuit, season = race_keys[idx]
            pieces.append(dataset[(dataset["circuit"] == circuit) & (dataset["season"] == season)])
        resampled = pd.concat(pieces, ignore_index=True)
        if resampled["actual"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(resampled["actual"], resampled["predicted_p"]))

    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling Race session laps_features.csv ...")
    laps = load_race_sessions(bucket)
    print(f"[load] {len(laps)} rows")

    dataset = build_full_validation_dataset(laps)
    print(f"\n[validate] {len(dataset)} (lap, predicted, actual) rows across "
          f"{dataset[['circuit','season']].drop_duplicates().shape[0]} races, "
          f"{dataset['circuit'].nunique()} circuits")
    print(f"[validate] base rate: {dataset['actual'].mean()*100:.1f}% of laps have an "
          f"SC/VSC starting within the next {HORIZON_LAPS} laps")

    auc = roc_auc_score(dataset["actual"], dataset["predicted_p"])
    brier = brier_score_loss(dataset["actual"], dataset["predicted_p"])
    print(f"\n[result] AUC: {auc:.3f} (0.5 = no discrimination)")
    print(f"[result] Brier score: {brier:.4f} (lower is better; a model predicting the "
          f"base rate everywhere scores {dataset['actual'].mean()*(1-dataset['actual'].mean()):.4f})")

    print("\n[bootstrap] resampling races (not individual laps) for a clustering-aware CI ...")
    ci_low, ci_high = race_level_bootstrap_auc(dataset)
    print(f"[result] AUC 95% CI (race-level bootstrap, {N_BOOTSTRAP} resamples): [{ci_low:.3f}, {ci_high:.3f}]")
    if ci_low > 0.5:
        print("  -> CI excludes 0.5: genuine evidence of real discrimination, even accounting for clustering")
    else:
        print("  -> CI includes 0.5: NOT statistically distinguishable from no discrimination once "
              "race-level clustering is properly accounted for")

    print("\n=== Calibration table (does 'predicted X%' actually happen ~X% of the time?) ===")
    frac_positive, mean_predicted = calibration_curve(dataset["actual"], dataset["predicted_p"], n_bins=10, strategy="quantile")
    calib_df = pd.DataFrame({"mean_predicted": mean_predicted, "observed_frequency": frac_positive})
    print(calib_df.to_string(index=False))

    dataset.to_csv("sc_probability_full_validation.csv", index=False)
    print("\n[save] sc_probability_full_validation.csv")
