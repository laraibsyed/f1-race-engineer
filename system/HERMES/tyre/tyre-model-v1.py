"""
Tyre Degradation — Regression V1 (basic)
==========================================
Single predictor (tyre_age), fit separately per (compound, circuit), on RBR laps only.

Runs THREE train/test approaches on the exact same model + data so the results are
directly comparable in your dissertation:
    1. RANDOM SPLIT        - naive row-level shuffle (included specifically to show why
                              it's the wrong approach for time-ordered data)
    2. FIXED SPLIT          - train 2018-2023, test 2024-2026 (2026 is PARTIAL - flagged)
    3. EXPANDING WINDOW     - Round 1..4 + Holdout, matches your diagram exactly

KNOWN V1 LIMITATION (flagged on purpose, fix in v2):
    Target = degradation_rate, which is pace_loss / tyre_age. Predictor = tyre_age.
    This means tyre_age partially appears on both sides of the equation. Fine for a
    basic first pass, but the v2 conversation should revisit target = raw pace loss
    (LapTime vs stint baseline) instead, so tyre_age is a clean, un-entangled predictor.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

load_dotenv()  # must run BEFORE storage.Client() is instantiated, or GOOGLE_APPLICATION_CREDENTIALS
               # (and anything else in .env) never makes it into os.environ in time

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
# 1. Load + filter
# ---------------------------------------------------------------------------
def load_all_teams_laps(bucket: CachedBucket) -> pd.DataFrame:
    """
    Race + Sprint only — degradation_rate is null/meaningless in FP/Q.

    SCOPE CHANGE: loads ALL teams now, not just RBR. Tyre degradation physics is
    assumed to depend on compound and circuit, not on which team is running the
    tyre - car-specific effects (e.g. downforce level affecting tyre stress) are
    a secondary factor not modelled here. This is a deliberate assumption, made
    explicit here rather than done silently, because RBR-only data produced
    dangerously thin samples at high tyre_age (as few as 1 stint at risk by
    tyre_age 50 for MEDIUM - see the survival-model at-risk table).

    An `is_rbr` column is added so downstream scripts can still report/validate
    specifically on RBR, even though training uses the full grid.
    """
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv"):
            continue
        if "/R/" not in p and "/S/" not in p:
            continue
        # TrackStatus MUST be read as str, not int - same bug your clean_laps.py
        # already hit once (int parsing gives nonsense values like max=12675,
        # because it's really a string of concatenated single-digit codes).
        df = bucket.read_csv(p, dtype={"TrackStatus": str})

        # Season/Race pulled from the path itself: clean/features/<year>/<race>/<session>/laps_features.csv
        parts = p.split("/")
        df["Season"] = int(parts[2])
        df["Race"] = parts[3]
        df["Session"] = parts[4]

        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["is_rbr"] = full["Team"].isin(RBR_ALIASES)
    return full


MIN_STINT_LENGTH = 5  # stints shorter than this can't produce a reliable median
                       # for is_outlier_laptime to compare against - see the PER
                       # 2022 British GP case: a 2-lap stint where the one bad lap
                       # itself drags the median up enough to hide from its own
                       # 1.5x-median outlier check. Below this length, drop the
                       # whole stint rather than trust the per-lap flags alone.


RED_FLAG_RESTART_BUFFER = 2  # laps after a red-flag event to also exclude, session-wide


def get_red_flag_affected_laps(df: pd.DataFrame, buffer: int = RED_FLAG_RESTART_BUFFER) -> set:
    """
    Returns a set of (Season, Race, Session, LapNumber) keys to exclude - the
    lap where TrackStatus contains code 5 (red flag), PLUS the next `buffer`
    laps, SESSION-WIDE (every driver, not just whichever row happened to
    record the code).

    Why session-wide + buffer, not just "drop rows with code 5":
    A red flag stops the entire race for everyone, but the restart lap itself
    - the one that's actually anomalous (standing start, bunched field, cold
    tyres) - typically carries a NORMAL TrackStatus, since the red flag has
    already ended by the time racing resumes. Confirmed via two real cases:
      - 2022 British GP: lap-1 red flag (Zhou's crash), PER's restart-stint
        lap 2 was anomalously slow (149.3s) despite normal TrackStatus.
      - 2020 Italian GP: lap-26 red flag (Leclerc's crash), race "restarted
        on Lap 28 of 53" (RaceFans, 2020) - matches ALB's anomalous row
        EXACTLY: LapNumber 28, TrackStatus=1 (looks clean), LapTime 119.9s
        vs. a ~85s stint baseline.
    A single-driver, single-lap, code-only filter misses both of these.
    """
    red_flag_mask = df["TrackStatus"].astype(str).str.contains("5", na=False)
    red_flag_laps = df.loc[red_flag_mask, ["Season", "Race", "Session", "LapNumber"]].drop_duplicates()

    affected = set()
    for _, row in red_flag_laps.iterrows():
        for offset in range(buffer + 1):
            affected.add((row["Season"], row["Race"], row["Session"], row["LapNumber"] + offset))
    return affected


def filter_valid_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that would poison the degradation fit:
      - pit in/out, out/in laps: not representative racing pace
      - missing/outlier laptimes: data quality
      - SC/VSC laps: pace reflects bunching, not tyre wear
      - red flag laps AND the restart lap(s) that follow, session-wide - see
        get_red_flag_affected_laps() docstring for why the restart lap matters
        more than the flagged lap itself.
      - laps belonging to very short stints: the per-stint median used by
        is_outlier_laptime can self-mask on a 2-3 lap stint (see MIN_STINT_LENGTH
        comment above) - a real limitation, not something the existing flag
        catches on its own.

    TrackStatus is a string of concatenated single-digit codes (same pattern your
    clean_laps.py already uses for is_sc_lap/is_vsc_lap - code 4 = SC, 6/7 = VSC).
    Code 5 = red flag. Built here rather than as a pre-existing column since the
    pipeline hasn't engineered is_red_flag_lap yet - worth adding there properly
    later so this filter doesn't have to re-derive it in every script.

    Flag, not delete, is your pipeline's rule elsewhere - but for MODEL FITTING
    specifically these rows should not be in the training data.
    """
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


LAPTIME_OUTLIER_Z_THRESH = 4.0  # robust (MAD-based) z-score cutoff


def filter_global_degradation_outliers(df: pd.DataFrame, z_thresh: float = LAPTIME_OUTLIER_Z_THRESH) -> pd.DataFrame:
    """
    Final catch-all pass, run AFTER filter_valid_laps(). Targets incidents with
    NO TrackStatus signature at all - e.g. 2021 PER's British GP lap 5 (107.5s
    vs a ~91s stint baseline, TrackStatus completely clean - a one-off incident
    with nothing to filter on), or a local yellow-flag zone that never escalates
    to SC/VSC (TrackStatus contains "2" but not 4/6/7, so is_sc_lap/is_vsc_lap
    never fires - see 2025 TSU/VER laps 43-45).

    Uses median absolute deviation (MAD), computed per (Compound, Race) GLOBALLY
    across every season/stint at once - not per single stint like
    is_outlier_laptime - so one anomalous lap can't inflate its own baseline the
    way a short 2-3 lap stint's median could (the exact mechanism behind the
    British GP 2022 and Italian GP 2020 fixes earlier).

    This is deliberately a blunt, generic pass rather than another targeted fix:
    past this point, individual incidents stop sharing a common cause worth
    writing bespoke rules for, and a general robust-statistics filter is more
    defensible than an ever-growing list of one-off exclusions tailored to
    this exact dataset.
    """
    grp = df.groupby(["Compound", "Race"])["degradation_rate"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())

    mad_safe = mad.replace(0, np.nan)  # MAD=0 groups: can't compute a meaningful ratio, don't flag
    robust_z = 0.6745 * (df["degradation_rate"] - med) / mad_safe
    is_outlier = robust_z.abs().gt(z_thresh).fillna(False)

    return df[~is_outlier]


# ---------------------------------------------------------------------------
# 3. Model: basic linear fit per (compound, circuit)
# ---------------------------------------------------------------------------
MIN_ROWS_PER_GROUP = 20  # below this, skip the group rather than trust a noisy fit


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
    """
    RMSE per (compound, circuit) group that exists in BOTH train and test.

    rbr_only=True (default): the MODEL was fit on all teams (for statistical
    power - see load_all_teams_laps), but validation is still scoped to RBR
    specifically, since that's the actual deliverable. Set False to check
    all-grid generalisation instead.
    """
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


# ---------------------------------------------------------------------------
# 3. THREE train/test approaches
# ---------------------------------------------------------------------------
def run_random_split(df: pd.DataFrame, test_frac=0.2, seed=42) -> pd.DataFrame:
    """
    Naive row-level shuffle. Included deliberately to demonstrate the leakage
    problem: rows from the same race/stint end up on both sides, so the model
    can effectively see part of a stint it's "predicting" on.
    """
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    models = fit_baseline_regression(train_df)
    results = evaluate(models, test_df)
    results["split_method"] = "random"
    return results


def run_fixed_split(df: pd.DataFrame, cutoff=2024) -> pd.DataFrame:
    """
    Train 2018-2023, test 2024-2026.
    NOTE: 2026 is a PARTIAL season as of this run (current season, still in progress).
    That's mixed into the same test bucket as 2024/2025 here because that's what
    was asked for - but call this out explicitly in the write-up as a limitation:
    the 2024-2026 test set doesn't have equal-sized seasons in it.
    """
    train_df = df[df["Season"] < cutoff]
    test_df = df[df["Season"] >= cutoff]
    models = fit_baseline_regression(train_df)
    results = evaluate(models, test_df)
    results["split_method"] = "fixed_2018_2023_train"
    return results


def run_expanding_window(df: pd.DataFrame, last_complete_season=2025) -> pd.DataFrame:
    """
    Round 1: train 2018-2020 / test 2021
    Round 2: train 2018-2021 / test 2022
    Round 3: train 2018-2022 / test 2023
    Round 4: train 2018-2023 / test 2024
    Holdout: train 2018-2024 / test 2025
    Matches your diagram exactly. Stops at 2025 on purpose - see last message
    re: not folding an incomplete 2026 season into a "normal" test fold.
    """
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
# 4. Run all three, compare
# ---------------------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # DIAGNOSTIC: automatically surface the worst-RMSE groups across ALL
    # split methods, instead of manually hunting through the CSV each time.
    # -----------------------------------------------------------------
    N_WORST = 10
    worst_groups = all_results.sort_values("rmse", ascending=False).head(N_WORST)
    print(f"\n=== TOP {N_WORST} WORST GROUPS ACROSS ALL SPLIT METHODS ===")
    print("(check each of these with the drill-down below before trusting any fold's mean)")
    print(worst_groups.to_string(index=False))

    # Drill-down: pull every remaining row for the single worst group so you can
    # see exactly what's happening, same way we diagnosed the British GP case.
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