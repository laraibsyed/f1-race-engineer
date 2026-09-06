"""
Stint-Length Cutoff Sensitivity Check
=======================================
Answers the question: how sensitive is the tyre degradation model's RMSE to
MIN_STINT_LENGTH? Sweeps the cutoff (1-10 laps), re-fits + re-evaluates the
SAME V1 model (same three train/test splits as tyre_regression_v1.py) at each
value, and plots RMSE vs. cutoff - so "drop stints shorter than 5 laps" is
justified by where the curve actually flattens out, not just asserted.

Produces:
  - stint_length_sensitivity.csv   (raw numbers per cutoff, for your appendix)
  - stint_length_sensitivity.png   (the dissertation figure)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the file
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

load_dotenv()

# ---------------------------------------------------------------------------
# 0. CachedBucket (same pattern as your other scripts)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 1. Load (once) - identical to tyre_regression_v1.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Filter - min_stint_length is now a PARAMETER, not a fixed constant
# ---------------------------------------------------------------------------
RED_FLAG_RESTART_BUFFER = 2  # laps after a red-flag event to also exclude, session-wide


def get_red_flag_affected_laps(df: pd.DataFrame, buffer: int = RED_FLAG_RESTART_BUFFER) -> set:
    """
    Same logic as tyre_regression_v1.py - the restart lap after a red flag
    carries a NORMAL TrackStatus (the flag itself has already ended), so a
    single-lap code-only check misses it. Confirmed via 2020 Italian GP:
    race "restarted on Lap 28 of 53" - exactly matches ALB's anomalous row.
    Kept in sync with tyre_regression_v1.py deliberately, so this stint-length
    sweep isn't contaminated by an unrelated, uncontrolled artifact sitting
    identically at every cutoff value.
    """
    red_flag_mask = df["TrackStatus"].astype(str).str.contains("5", na=False)
    red_flag_laps = df.loc[red_flag_mask, ["Season", "Race", "Session", "LapNumber"]].drop_duplicates()

    affected = set()
    for _, row in red_flag_laps.iterrows():
        for offset in range(buffer + 1):
            affected.add((row["Season"], row["Race"], row["Session"], row["LapNumber"] + offset))
    return affected


def filter_valid_laps(df: pd.DataFrame, min_stint_length: int) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# 3. Model + splits - identical to tyre_regression_v1.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Sweep MIN_STINT_LENGTH and plot
# ---------------------------------------------------------------------------
CUTOFFS_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling RBR laps_features.csv (once, then re-filtered per cutoff) ...")
    raw = load_rbr_laps(bucket)
    print(f"[load] {len(raw)} raw RBR rows\n")

    sweep_rows = []
    for cutoff in CUTOFFS_TO_TEST:
        clean = filter_valid_laps(raw, min_stint_length=cutoff)

        random_results = run_random_split(clean)
        fixed_results = run_fixed_split(clean)
        expanding_results = run_expanding_window(clean)
        all_results = pd.concat([random_results, fixed_results, expanding_results], ignore_index=True)

        row = {
            "min_stint_length": cutoff,
            "n_rows_remaining": len(clean),
            "pct_rows_remaining": len(clean) / len(raw) * 100,
            "mean_rmse": all_results["rmse"].mean(),
            "median_rmse": all_results["rmse"].median(),
            "std_rmse": all_results["rmse"].std(),
            "n_groups_evaluated": len(all_results),
        }
        sweep_rows.append(row)
        print(f"[cutoff={cutoff:>2}] rows={row['n_rows_remaining']:>6} "
              f"({row['pct_rows_remaining']:.1f}%)  "
              f"mean_rmse={row['mean_rmse']:.4f}  median_rmse={row['median_rmse']:.4f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("stint_length_sensitivity.csv", index=False)
    print("\n[save] stint_length_sensitivity.csv")

    # --- Plot: RMSE vs cutoff on top, % data retained on bottom ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(sweep_df["min_stint_length"], sweep_df["mean_rmse"], marker="o", label="Mean RMSE")
    ax1.plot(sweep_df["min_stint_length"], sweep_df["median_rmse"], marker="s", label="Median RMSE")
    ax1.axvline(5, color="grey", linestyle="--", alpha=0.6, label="Chosen cutoff (5 laps)")
    ax1.set_ylabel("RMSE (degradation_rate)")
    ax1.set_title("Tyre degradation model — RMSE vs. minimum stint-length cutoff")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(sweep_df["min_stint_length"], sweep_df["pct_rows_remaining"], marker="^", color="grey")
    ax2.axvline(5, color="grey", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Minimum stint length (laps) — stints shorter than this are excluded")
    ax2.set_ylabel("% of filtered rows retained")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("stint_length_sensitivity.png", dpi=150)
    print("[save] stint_length_sensitivity.png")