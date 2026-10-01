""

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

MIN_SEGMENT_LENGTH = 3
MIN_IMPROVEMENT = 0.20
MIN_SLOPE_RATIO = 2.0
MIN_STEP_SECONDS = 0.3

def _fit_line_sse(x: np.ndarray, y: np.ndarray):
    ""
    if len(x) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    preds = slope * x + intercept
    sse = float(np.sum((y - preds) ** 2))
    return sse, float(slope)

def detect_cliff(x: np.ndarray, y: np.ndarray):
    ""
    n = len(x)
    if n < 2 * MIN_SEGMENT_LENGTH:
        return None, False, {"reason": "stint too short to scan"}

    whole_sse, _ = _fit_line_sse(x, y)

    best = {"total_sse": np.inf, "split": None, "pre_slope": None, "post_slope": None}
    for i in range(MIN_SEGMENT_LENGTH, n - MIN_SEGMENT_LENGTH):
        pre_sse, pre_slope = _fit_line_sse(x[:i], y[:i])
        post_sse, post_slope = _fit_line_sse(x[i:], y[i:])
        total_sse = pre_sse + post_sse
        if total_sse < best["total_sse"]:
            best.update(total_sse=total_sse, split=i, pre_slope=pre_slope, post_slope=post_slope)

    if best["split"] is None:
        return None, False, {"reason": "no valid split found"}

    i = best["split"]
    improvement = (whole_sse - best["total_sse"]) / whole_sse if whole_sse > 0 else 0.0
    slope_ratio = (best["post_slope"] / best["pre_slope"]) if best["pre_slope"] > 0 else np.inf

    pre_tail_median = float(np.median(y[max(0, i - MIN_SEGMENT_LENGTH):i]))
    post_head_median = float(np.median(y[i:i + MIN_SEGMENT_LENGTH]))
    local_step = post_head_median - pre_tail_median

    is_cliff = (
        improvement >= MIN_IMPROVEMENT
        and best["post_slope"] > 0
        and slope_ratio >= MIN_SLOPE_RATIO
        and local_step >= MIN_STEP_SECONDS
    )
    cliff_tyre_age = x[i] if is_cliff else None

    diagnostics = {
        "improvement": improvement,
        "pre_slope": best["pre_slope"],
        "post_slope": best["post_slope"],
        "slope_ratio": slope_ratio,
        "local_step_seconds": local_step,
        "split_index": i,
    }
    return cliff_tyre_age, is_cliff, diagnostics

def compute_true_stint_lengths(raw_laps: pd.DataFrame) -> dict:
    ""
    group_cols = ["Season", "Race", "Session", "Driver", "Stint"]
    lengths = raw_laps.dropna(subset=["tyre_age"]).groupby(group_cols)["tyre_age"].max()
    return lengths.to_dict()

def build_stint_table(laps: pd.DataFrame, true_stint_lengths: dict = None) -> pd.DataFrame:
    laps = laps.copy()
    laps["LapTime_seconds"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

    group_cols = ["Season", "Race", "Session", "Driver", "Stint"]
    rows = []
    for keys, g in laps.groupby(group_cols):
        g = g.sort_values("LapNumber")
        x = g["tyre_age"].values
        y = g["LapTime_seconds"].values

        cliff_age, is_cliff, diag = detect_cliff(x, y)

        true_n_laps = true_stint_lengths.get(keys) if true_stint_lengths else None

        rows.append({
            "season": keys[0], "race": keys[1], "session": keys[2],
            "driver": keys[3], "stint": keys[4],
            "team": g["Team"].iloc[0],
            "is_rbr": bool(g["is_rbr"].iloc[0]),
            "compound": g["Compound"].iloc[0],
            "circuit": keys[1],
            "n_laps_cleaned": len(g),
            "n_laps_true": true_n_laps,
            "duration": cliff_age if is_cliff else x.max(),
            "event": int(is_cliff),
            "cliff_tyre_age": cliff_age,
            "slope_ratio": diag.get("slope_ratio"),
            "improvement": diag.get("improvement"),

            "track_temp_bucket": g["track_temp_bucket"].mode().iloc[0]
                if g["track_temp_bucket"].notna().any() else np.nan,
            "fuel_load_estimate": g["fuel_load_estimate"].mean(),
            "stint_number": g["stint_number"].iloc[0] if "stint_number" in g.columns else np.nan,
            "regulation_era": ("2018-2021" if keys[0] <= 2021
                                else "2022-2025" if keys[0] <= 2025
                                else "2026+"),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling ALL TEAMS' laps_features.csv ...")
    raw = load_all_teams_laps(bucket)
    print(f"[load] {len(raw)} raw rows, {raw['is_rbr'].sum()} of which are RBR")

    true_stint_lengths = compute_true_stint_lengths(raw)

    clean = filter_valid_laps(raw)
    clean = filter_global_degradation_outliers(clean)
    print(f"[filter] {len(clean)} rows after full cleaning pipeline ({clean['is_rbr'].sum()} RBR)")

    stint_table = build_stint_table(clean, true_stint_lengths)
    stint_table.to_csv("cliff_detection_stints.csv", index=False)

    n_stints = len(stint_table)
    n_rbr_stints = stint_table["is_rbr"].sum()
    n_cliffs = stint_table["event"].sum()
    n_rbr_cliffs = stint_table.loc[stint_table["is_rbr"], "event"].sum()
    print(f"\n[result] {n_stints} stints analysed ({n_rbr_stints} RBR)")
    print(f"[result] {n_cliffs} cliffs detected ({n_cliffs / n_stints * 100:.1f}%) "
          f"— {n_rbr_cliffs} of those are RBR ({n_rbr_cliffs / n_rbr_stints * 100:.1f}% of RBR stints)")
    print(f"[result] {n_stints - n_cliffs} censored (no cliff observed within the stint)")

    print("\n=== CLIFF RATE BY COMPOUND (ALL TEAMS) ===")
    print(stint_table.groupby("compound")["event"].agg(["mean", "count"])
          .rename(columns={"mean": "cliff_rate", "count": "n_stints"})
          .sort_values("cliff_rate", ascending=False).to_string())

    print("\n=== CLIFF RATE BY COMPOUND (RBR ONLY, for comparison) ===")
    rbr_stints = stint_table[stint_table["is_rbr"]]
    print(rbr_stints.groupby("compound")["event"].agg(["mean", "count"])
          .rename(columns={"mean": "cliff_rate", "count": "n_stints"})
          .sort_values("cliff_rate", ascending=False).to_string())

    print("\n[save] cliff_detection_stints.csv (all teams, with is_rbr flag)")

    laps_seconds = clean.copy()
    laps_seconds["LapTime_seconds"] = pd.to_timedelta(laps_seconds["LapTime"]).dt.total_seconds()

    cliff_examples = stint_table[stint_table["event"] == 1].sample(
        min(4, n_cliffs), random_state=42
    ) if n_cliffs > 0 else pd.DataFrame()

    if not cliff_examples.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (_, row) in zip(axes.flat, cliff_examples.iterrows()):
            stint_laps = laps_seconds[
                (laps_seconds["Season"] == row["season"]) & (laps_seconds["Race"] == row["race"])
                & (laps_seconds["Driver"] == row["driver"]) & (laps_seconds["Stint"] == row["stint"])
            ].sort_values("LapNumber")
            ax.plot(stint_laps["tyre_age"], stint_laps["LapTime_seconds"], marker="o")
            ax.axvline(row["cliff_tyre_age"], color="red", linestyle="--", label="detected cliff")
            ax.set_title(f"{row['season']} {row['race']} {row['driver']} ({row['compound']}, "
                         f"{row['team']})", fontsize=9)
            ax.set_xlabel("tyre_age")
            ax.set_ylabel("LapTime (s)")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("cliff_detection_examples.png", dpi=150)
        print("[save] cliff_detection_examples.png - visually confirm these actually look like cliffs")
    else:
        print("[warn] no cliffs detected at all - check MIN_SLOPE_RATIO / MIN_IMPROVEMENT thresholds")
