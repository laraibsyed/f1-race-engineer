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
LAPTIME_OUTLIER_Z_THRESH = 4.0

def get_red_flag_affected_laps(df: pd.DataFrame, buffer: int = RED_FLAG_RESTART_BUFFER) -> set:
    red_flag_mask = df["TrackStatus"].astype(str).str.contains("5", na=False)
    red_flag_laps = df.loc[red_flag_mask, ["Season", "Race", "Session", "LapNumber"]].drop_duplicates()
    affected = set()
    for _, row in red_flag_laps.iterrows():
        for offset in range(buffer + 1):
            affected.add((row["Season"], row["Race"], row["Session"], row["LapNumber"] + offset))
    return affected

def filter_valid_laps(df: pd.DataFrame, min_stint_length: int = MIN_STINT_LENGTH) -> pd.DataFrame:
    red_flag_affected = get_red_flag_affected_laps(df)
    lap_keys = list(zip(df["Season"], df["Race"], df["Session"], df["LapNumber"]))
    is_red_flag_affected = pd.Series(lap_keys, index=df.index).isin(red_flag_affected)
    mask = (
        (~df["is_pit_in"].astype(bool)) & (~df["is_pit_out"].astype(bool))
        & (~df["is_out_lap"].astype(bool)) & (~df["is_in_lap"].astype(bool))
        & (~df["is_missing_laptime"].astype(bool)) & (~df["is_outlier_laptime"].astype(bool))
        & (~df["is_sc_lap"].astype(bool)) & (~df["is_vsc_lap"].astype(bool))
        & (~is_red_flag_affected)
    )
    clean = df[mask].dropna(subset=["tyre_age", "degradation_rate", "Compound"])
    stint_lengths = clean.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapNumber"].transform("count")
    return clean[stint_lengths >= min_stint_length]

def filter_global_degradation_outliers(df: pd.DataFrame, z_thresh: float = LAPTIME_OUTLIER_Z_THRESH) -> pd.DataFrame:
    grp = df.groupby(["Compound", "Race"])["degradation_rate"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    mad_safe = mad.replace(0, np.nan)
    robust_z = 0.6745 * (df["degradation_rate"] - med) / mad_safe
    is_outlier = robust_z.abs().gt(z_thresh).fillna(False)
    return df[~is_outlier]

DRY_COMPOUNDS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]

def add_pace_loss_and_era(clean_laps: pd.DataFrame) -> pd.DataFrame:
    ""
    df = clean_laps[clean_laps["Compound"].isin(DRY_COMPOUNDS)].copy()
    df["LapTime_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

    baseline = df.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapTime_seconds"].transform("min")
    df["pace_loss_seconds"] = df["LapTime_seconds"] - baseline

    df["regulation_era"] = pd.cut(
        df["Season"], bins=[-np.inf, 2021, 2025, np.inf],
        labels=["2018-2021", "2022-2025", "2026+"],
    )
    return df

def check_multicollinearity(df: pd.DataFrame) -> None:
    numeric_predictors = df[["tyre_age", "fuel_load_estimate", "stint_number"]].dropna()
    temp_dummies = pd.get_dummies(df.loc[numeric_predictors.index, "track_temp_bucket"],
                                   prefix="temp", drop_first=True).astype(float)
    X = pd.concat([numeric_predictors, temp_dummies], axis=1)

    print("\n=== CORRELATION MATRIX (predictors) ===")
    print(X.corr().round(2).to_string())

    print("\n=== VARIANCE INFLATION FACTOR (VIF > 5-10 = concerning) ===")
    for col in X.columns:
        others = X.drop(columns=[col])
        model = LinearRegression().fit(others, X[col])
        r2 = model.score(others, X[col])
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        print(f"  {col:>20}: VIF = {vif:.2f}")

MIN_ROWS_PER_GROUP = 40

def build_design_matrix(df: pd.DataFrame, temp_dummy_columns: list) -> pd.DataFrame:
    ""
    temp_dummies = pd.get_dummies(df["track_temp_bucket"], prefix="temp", drop_first=True)
    temp_dummies = temp_dummies.reindex(columns=temp_dummy_columns, fill_value=0).astype(float)
    X = pd.concat([df[["tyre_age", "fuel_load_estimate", "stint_number"]].reset_index(drop=True),
                   temp_dummies.reset_index(drop=True)], axis=1)
    return X

def fit_baseline_regression(train_df: pd.DataFrame, temp_dummy_columns: list) -> dict:
    models = {}
    for (compound, circuit, era), g in train_df.groupby(["Compound", "Race", "regulation_era"], observed=True):
        g = g.dropna(subset=["tyre_age", "fuel_load_estimate", "stint_number",
                              "track_temp_bucket", "pace_loss_seconds"])
        if len(g) < MIN_ROWS_PER_GROUP:
            continue
        X = build_design_matrix(g, temp_dummy_columns)
        y = g["pace_loss_seconds"].reset_index(drop=True)
        models[(compound, circuit, era)] = LinearRegression().fit(X, y)
    return models

def evaluate(models: dict, test_df: pd.DataFrame, temp_dummy_columns: list, rbr_only: bool = True) -> pd.DataFrame:
    if rbr_only:
        test_df = test_df[test_df["is_rbr"]]

    rows = []
    for (compound, circuit, era), model in models.items():
        g = test_df[(test_df["Compound"] == compound) & (test_df["Race"] == circuit)
                     & (test_df["regulation_era"] == era)].dropna(
            subset=["tyre_age", "fuel_load_estimate", "stint_number",
                    "track_temp_bucket", "pace_loss_seconds"])
        if g.empty:
            continue
        X = build_design_matrix(g, temp_dummy_columns)
        preds = model.predict(X)
        rmse = root_mean_squared_error(g["pace_loss_seconds"], preds)
        rows.append({"compound": compound, "circuit": circuit, "era": era,
                      "n_test_rows": len(g), "rmse": rmse})
    return pd.DataFrame(rows)

def run_random_split(df: pd.DataFrame, temp_dummy_columns: list, test_frac=0.2, seed=42) -> pd.DataFrame:
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    models = fit_baseline_regression(train_df, temp_dummy_columns)
    results = evaluate(models, test_df, temp_dummy_columns)
    results["split_method"] = "random"
    return results

def run_fixed_split(df: pd.DataFrame, temp_dummy_columns: list, cutoff=2024) -> pd.DataFrame:
    train_df = df[df["Season"] < cutoff]
    test_df = df[df["Season"] >= cutoff]
    models = fit_baseline_regression(train_df, temp_dummy_columns)
    results = evaluate(models, test_df, temp_dummy_columns)
    results["split_method"] = "fixed_2018_2023_train"
    return results

def run_expanding_window(df: pd.DataFrame, temp_dummy_columns: list, last_complete_season=2025) -> pd.DataFrame:
    all_results = []
    for test_year in range(2021, last_complete_season + 1):
        train_df = df[df["Season"] < test_year]
        test_df = df[df["Season"] == test_year]
        models = fit_baseline_regression(train_df, temp_dummy_columns)
        results = evaluate(models, test_df, temp_dummy_columns)
        results["split_method"] = f"expanding_window_test_{test_year}"
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling ALL TEAMS' laps_features.csv ...")
    raw = load_all_teams_laps(bucket)
    print(f"[load] {len(raw)} raw rows, {raw['is_rbr'].sum()} RBR")

    clean = filter_valid_laps(raw)
    clean = filter_global_degradation_outliers(clean)
    print(f"[filter] {len(clean)} rows after structural + outlier cleaning")

    modeling_df = add_pace_loss_and_era(clean)
    print(f"[filter] {len(modeling_df)} rows after excluding WET/INTERMEDIATE "
          f"(dry compounds only: {DRY_COMPOUNDS})")

    check_multicollinearity(modeling_df)

    temp_dummy_columns = sorted(pd.get_dummies(
        modeling_df["track_temp_bucket"], prefix="temp", drop_first=True).columns.tolist())
    print(f"\n[info] temp dummy columns: {temp_dummy_columns}")

    random_results = run_random_split(modeling_df, temp_dummy_columns)
    fixed_results = run_fixed_split(modeling_df, temp_dummy_columns)
    expanding_results = run_expanding_window(modeling_df, temp_dummy_columns)

    all_results = pd.concat([random_results, fixed_results, expanding_results], ignore_index=True)

    summary = (
        all_results.groupby("split_method")["rmse"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_rmse", "median": "median_rmse",
                          "std": "std_rmse", "count": "n_groups_evaluated"})
        .sort_values("mean_rmse")
    )
    print("\n=== SUMMARY: mean + median RMSE (seconds) by split method ===")
    print(summary.to_string())

    print("\n=== RMSE by regulation era (aggregated across all split methods) ===")
    print(all_results.groupby("era", observed=True)["rmse"].agg(["mean", "median", "count"]).to_string())

    out_path = "tyre_regression_v2_results.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")

    N_WORST = 10
    worst_groups = all_results.sort_values("rmse", ascending=False).head(N_WORST)
    print(f"\n=== TOP {N_WORST} WORST GROUPS ===")
    print(worst_groups.to_string(index=False))
