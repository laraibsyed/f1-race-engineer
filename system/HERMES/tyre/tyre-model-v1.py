""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

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

RBR_ALIASES = {
    "Red Bull Racing", "Red Bull Racing Honda", "Red Bull Racing RBPT",
    "Oracle Red Bull Racing", "Red Bull",
}

def load_all_teams_laps(bucket: CachedBucket) -> pd.DataFrame:
    ""
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv"):
            continue
        if "/R/" not in p and "/S/" not in p:
            continue

        df = bucket.read_csv(p, dtype={"TrackStatus": str})

        parts = p.split("/")
        df["Season"] = int(parts[2])
        df["Race"] = parts[3]
        df["Session"] = parts[4]

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["is_rbr"] = full["Team"].isin(RBR_ALIASES)
    return full

MIN_STINT_LENGTH = 5

RED_FLAG_RESTART_BUFFER = 2

def get_red_flag_affected_laps(df: pd.DataFrame, buffer: int = RED_FLAG_RESTART_BUFFER) -> set:
    ""
    red_flag_mask = df["TrackStatus"].astype(str).str.contains("5", na=False)
    red_flag_laps = df.loc[red_flag_mask, ["Season", "Race", "Session", "LapNumber"]].drop_duplicates()

    affected = set()
    for _, row in red_flag_laps.iterrows():
        for offset in range(buffer + 1):
            affected.add((row["Season"], row["Race"], row["Session"], row["LapNumber"] + offset))
    return affected

def filter_valid_laps(df: pd.DataFrame) -> pd.DataFrame:
    ""
    red_flag_affected = get_red_flag_affected_laps(df)
    lap_keys = list(zip(df["Season"], df["Race"], df["Session"], df["LapNumber"]))
    is_red_flag_affected = pd.Series(lap_keys, index=df.index).isin(red_flag_affected)

    mask = (
        (~df["is_pit_in"].astype(bool))
        & (~df["is_pit_out"].astype(bool))
        & (~df["is_out_lap"].astype(bool))
        & (~df["is_in_lap"].astype(bool))
        & (~df["is_missing_laptime"].astype(bool))
        & (~df["is_outlier_laptime"].astype(bool))
        & (~df["is_sc_lap"].astype(bool))
        & (~df["is_vsc_lap"].astype(bool))
        & (~is_red_flag_affected)
    )
    clean = df[mask].dropna(subset=["tyre_age", "degradation_rate", "Compound"])

    stint_lengths = clean.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapNumber"].transform("count")
    clean = clean[stint_lengths >= MIN_STINT_LENGTH]

    return clean

LAPTIME_OUTLIER_Z_THRESH = 4.0

def filter_global_degradation_outliers(df: pd.DataFrame, z_thresh: float = LAPTIME_OUTLIER_Z_THRESH) -> pd.DataFrame:
    ""
    grp = df.groupby(["Compound", "Race"])["degradation_rate"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())

    mad_safe = mad.replace(0, np.nan)
    robust_z = 0.6745 * (df["degradation_rate"] - med) / mad_safe
    is_outlier = robust_z.abs().gt(z_thresh).fillna(False)

    return df[~is_outlier]

MIN_ROWS_PER_GROUP = 20

def fit_baseline_regression(train_df: pd.DataFrame) -> dict:
    models = {}
    for (compound, circuit), g in train_df.groupby(["Compound", "Race"]):
        if len(g) < MIN_ROWS_PER_GROUP:
            continue
        X = g[["tyre_age"]].values
        y = g["degradation_rate"].values
        models[(compound, circuit)] = LinearRegression().fit(X, y)
    return models

def evaluate(models: dict, test_df: pd.DataFrame, rbr_only: bool = True) -> pd.DataFrame:
    ""
    if rbr_only:
        test_df = test_df[test_df["is_rbr"]]

    rows = []
    for (compound, circuit), model in models.items():
        g = test_df[(test_df["Compound"] == compound) & (test_df["Race"] == circuit)]
        if g.empty:
            continue
        preds = model.predict(g[["tyre_age"]].values)
        rmse = root_mean_squared_error(g["degradation_rate"], preds)
        rows.append({"compound": compound, "circuit": circuit, "n_test_rows": len(g), "rmse": rmse})
    return pd.DataFrame(rows)

def run_random_split(df: pd.DataFrame, test_frac=0.2, seed=42) -> pd.DataFrame:
    ""
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    models = fit_baseline_regression(train_df)
    results = evaluate(models, test_df)
    results["split_method"] = "random"
    return results

def run_fixed_split(df: pd.DataFrame, cutoff=2024) -> pd.DataFrame:
    ""
    train_df = df[df["Season"] < cutoff]
    test_df = df[df["Season"] >= cutoff]
    models = fit_baseline_regression(train_df)
    results = evaluate(models, test_df)
    results["split_method"] = "fixed_2018_2023_train"
    return results

def run_expanding_window(df: pd.DataFrame, last_complete_season=2025) -> pd.DataFrame:
    ""
    all_results = []
    for test_year in range(2021, last_complete_season + 1):
        train_df = df[df["Season"] < test_year]
        test_df = df[df["Season"] == test_year]
        models = fit_baseline_regression(train_df)
        results = evaluate(models, test_df)
        results["split_method"] = f"expanding_window_test_{test_year}"
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling ALL TEAMS' laps_features.csv from clean/features/ ...")
    raw = load_all_teams_laps(bucket)
    print(f"[load] {len(raw)} raw rows across {raw['Season'].nunique()} seasons, "
          f"{raw['is_rbr'].sum()} of which are RBR")

    clean = filter_valid_laps(raw)
    n_after_structural_filters = len(clean)
    print(f"[filter] {n_after_structural_filters} rows remain after excluding pit/out/in/missing/outlier/SC/VSC/red-flag laps "
          f"({clean['is_rbr'].sum()} RBR)")

    clean = filter_global_degradation_outliers(clean)
    print(f"[filter] {len(clean)} rows remain after global degradation-rate outlier pass "
          f"(removed {n_after_structural_filters - len(clean)} additional rows, {clean['is_rbr'].sum()} RBR)")

    random_results = run_random_split(clean)
    fixed_results = run_fixed_split(clean)
    expanding_results = run_expanding_window(clean)

    all_results = pd.concat([random_results, fixed_results, expanding_results], ignore_index=True)

    summary = (
        all_results.groupby("split_method")["rmse"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_rmse", "median": "median_rmse",
                          "std": "std_rmse", "count": "n_groups_evaluated"})
        .sort_values("mean_rmse")
    )

    print("\n=== SUMMARY: mean + median RMSE by split method ===")
    print("(a big gap between mean and median means a few groups are dominating the mean - check those before trusting it)")
    print(summary.to_string())

    out_path = "tyre_regression_v1_results.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\n[save] full per-group results written to {out_path} - use this for your comparison plots")

    N_WORST = 10
    worst_groups = all_results.sort_values("rmse", ascending=False).head(N_WORST)
    print(f"\n=== TOP {N_WORST} WORST GROUPS ACROSS ALL SPLIT METHODS ===")
    print("(check each of these with the drill-down below before trusting any fold's mean)")
    print(worst_groups.to_string(index=False))

    top = worst_groups.iloc[0]
    debug_rows = clean[
        (clean["Compound"] == top["compound"])
        & (clean["Race"] == top["circuit"])
    ].sort_values(["Season", "Driver", "LapNumber"])

    cols_to_show = [c for c in ["Season", "Driver", "Stint", "LapNumber", "TyreLife", "tyre_age",
                                 "LapTime", "degradation_rate", "TrackStatus",
                                 "is_pit_in", "is_pit_out"] if c in debug_rows.columns]
    print(f"\n=== DRILL-DOWN: worst group = {top['compound']}/{top['circuit']} "
          f"(rmse={top['rmse']:.3f}, split={top['split_method']}) ===")
    print(debug_rows[cols_to_show].to_string(index=False))
