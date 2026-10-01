
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def load_rbr_laps(bucket: CachedBucket) -> pd.DataFrame:
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
    return full[full["Team"].isin(RBR_ALIASES)].copy()

MIN_STINT_LENGTH = 5
RED_FLAG_RESTART_BUFFER = 2

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
    clean = clean[stint_lengths >= min_stint_length]

    return clean

def filter_global_degradation_outliers(df: pd.DataFrame, z_thresh: float) -> pd.DataFrame:
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

def evaluate(models: dict, test_df: pd.DataFrame) -> pd.DataFrame:
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
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    models = fit_baseline_regression(train_df)
    results = evaluate(models, test_df)
    results["split_method"] = "random"
    return results

def run_fixed_split(df: pd.DataFrame, cutoff=2024) -> pd.DataFrame:
    train_df = df[df["Season"] < cutoff]
    test_df = df[df["Season"] >= cutoff]
    models = fit_baseline_regression(train_df)
    results = evaluate(models, test_df)
    results["split_method"] = "fixed_2018_2023_train"
    return results

def run_expanding_window(df: pd.DataFrame, last_complete_season=2025) -> pd.DataFrame:
    all_results = []
    for test_year in range(2021, last_complete_season + 1):
        train_df = df[df["Season"] < test_year]
        test_df = df[df["Season"] == test_year]
        models = fit_baseline_regression(train_df)
        results = evaluate(models, test_df)
        results["split_method"] = f"expanding_window_test_{test_year}"
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)

Z_THRESHOLDS_TO_TEST = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling RBR laps_features.csv ...")
    raw = load_rbr_laps(bucket)
    print(f"[load] {len(raw)} raw RBR rows")

    structurally_clean = filter_valid_laps(raw, min_stint_length=MIN_STINT_LENGTH)
    print(f"[filter] {len(structurally_clean)} rows after structural filters "
          f"(MIN_STINT_LENGTH={MIN_STINT_LENGTH}, fixed for this sweep)\n")

    sweep_rows = []
    for z_thresh in Z_THRESHOLDS_TO_TEST:
        clean = filter_global_degradation_outliers(structurally_clean, z_thresh=z_thresh)

        random_results = run_random_split(clean)
        fixed_results = run_fixed_split(clean)
        expanding_results = run_expanding_window(clean)
        all_results = pd.concat([random_results, fixed_results, expanding_results], ignore_index=True)

        row = {
            "z_thresh": z_thresh,
            "n_rows_remaining": len(clean),
            "pct_rows_remaining": len(clean) / len(structurally_clean) * 100,
            "mean_rmse": all_results["rmse"].mean(),
            "median_rmse": all_results["rmse"].median(),
            "std_rmse": all_results["rmse"].std(),
            "n_groups_evaluated": len(all_results),
        }
        sweep_rows.append(row)
        print(f"[z_thresh={z_thresh:>4}] rows={row['n_rows_remaining']:>6} "
              f"({row['pct_rows_remaining']:.1f}%)  "
              f"mean_rmse={row['mean_rmse']:.4f}  median_rmse={row['median_rmse']:.4f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("z_threshold_sensitivity.csv", index=False)
    print("\n[save] z_threshold_sensitivity.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(sweep_df["z_thresh"], sweep_df["mean_rmse"], marker="o", label="Mean RMSE")
    ax1.plot(sweep_df["z_thresh"], sweep_df["median_rmse"], marker="s", label="Median RMSE")
    ax1.axvline(4.0, color="grey", linestyle="--", alpha=0.6, label="Chosen threshold (4.0)")
    ax1.set_ylabel("RMSE (degradation_rate)")
    ax1.set_title("Tyre degradation model — RMSE vs. global outlier z-threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(sweep_df["z_thresh"], sweep_df["pct_rows_remaining"], marker="^", color="grey")
    ax2.axvline(4.0, color="grey", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Robust z-score threshold — laps beyond this many MAD units are excluded")
    ax2.set_ylabel("% of structurally-filtered rows retained")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("z_threshold_sensitivity.png", dpi=150)
    print("[save] z_threshold_sensitivity.png")
