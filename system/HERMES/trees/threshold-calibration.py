"""
Tier 3 Threshold Calibration
================================
Grounds CLIFF_PROBABILITY_THRESHOLD, PACE_LOSS_THRESHOLD_SECONDS, and
TYRE_AGE_TRIGGER_RATIO in real historical data instead of leaving them as
guessed placeholders - same treatment DEADLINE_BUFFER_LAPS got.

Three separate analyses, since these three thresholds measure genuinely
different things:

  1. TYRE_AGE_TRIGGER_RATIO - what fraction through a stint do real cliffs
     actually occur at? (cliff_detection_stints.csv directly)

  2. PACE_LOSS_THRESHOLD_SECONDS - what counts as "notably elevated" pace
     loss vs. typical, using the real distribution from Regression V2's
     cleaned lap data.

  3. CLIFF_PROBABILITY_THRESHOLD - an actual ROC validation of the fitted
     Cox model: at various checkpoints before a real cliff, does the model's
     predicted cliff_probability_next_5_laps correctly separate "cliff
     coming soon" from "not yet"? This is the most important of the three,
     since it's the one most directly tied to real tyre-model output rather
     than a strategic judgement call.

HONEST LIMITATION for Part 3: only stints with event=1 (a real detected
cliff) are used to build positive/negative checkpoint labels. Censored
stints are deliberately EXCLUDED from this specific validation - their true
label near the end of observation is unknowable (the informative-censoring
problem from the survival analysis work: a team may have pitted right
before a cliff would have happened). Using them would risk circularity, not
add a genuine ground truth.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.cloud import storage
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, roc_auc_score

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")
STINTS_CSV = "cliff_detection_stints.csv"
TAXONOMY_PATH = r"src\taxanomy\circuit_taxonomy.xlsx"


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


# =============================================================================
# PART 1: TYRE_AGE_TRIGGER_RATIO
# =============================================================================
def calibrate_tyre_age_ratio():
    print("\n" + "=" * 70)
    print("PART 1: TYRE_AGE_TRIGGER_RATIO")
    print("=" * 70)

    stints = pd.read_csv(STINTS_CSV)
    events = stints[stints["event"] == 1].copy()
    events = events[events["n_laps_true"] > 0]
    events["cliff_ratio"] = events["cliff_tyre_age"] / events["n_laps_true"]

    print(f"[data] {len(events)} real cliff events with a valid ratio")
    print("\nDistribution of cliff_tyre_age / n_laps (how far through the stint the cliff hit):")
    print(events["cliff_ratio"].describe(percentiles=[.05, .10, .25, .50]).to_string())

    p10 = events["cliff_ratio"].quantile(0.10)
    print(f"\n[recommendation] 10th percentile = {p10:.3f} - a trigger ratio here would have "
          f"given advance warning ahead of 90% of real historical cliffs")
    print(f"[compare] current placeholder TYRE_AGE_TRIGGER_RATIO = 0.80")
    return p10


# =============================================================================
# PART 2: PACE_LOSS_THRESHOLD_SECONDS
# =============================================================================
DRY_COMPOUNDS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]


def load_all_teams_laps(bucket):
    paths = bucket.list_blob_names("clean/features/")
    frames = []
    for p in paths:
        if not p.endswith("laps_features.csv") or ("/R/" not in p and "/S/" not in p):
            continue
        df = bucket.read_csv(p, dtype={"TrackStatus": str})
        parts = p.split("/")
        df["Season"], df["Race"], df["Session"] = int(parts[2]), parts[3], parts[4]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


MIN_STINT_LENGTH_P2 = 5
RED_FLAG_RESTART_BUFFER_P2 = 2
LAPTIME_OUTLIER_Z_THRESH_P2 = 4.0


def _get_red_flag_affected_laps(df):
    red_flag_mask = df["TrackStatus"].astype(str).str.contains("5", na=False)
    red_flag_laps = df.loc[red_flag_mask, ["Season", "Race", "Session", "LapNumber"]].drop_duplicates()
    affected = set()
    for _, row in red_flag_laps.iterrows():
        for offset in range(RED_FLAG_RESTART_BUFFER_P2 + 1):
            affected.add((row["Season"], row["Race"], row["Session"], row["LapNumber"] + offset))
    return affected


def _filter_valid_laps(df):
    """IDENTICAL to tyre_regression_v2.py's filter_valid_laps - reused here
    deliberately so this calibration isn't measuring contaminated data."""
    red_flag_affected = _get_red_flag_affected_laps(df)
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
    return clean[stint_lengths >= MIN_STINT_LENGTH_P2]


def _filter_global_degradation_outliers(df):
    """IDENTICAL to tyre_regression_v2.py's filter_global_degradation_outliers."""
    grp = df.groupby(["Compound", "Race"])["degradation_rate"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    mad_safe = mad.replace(0, np.nan)
    robust_z = 0.6745 * (df["degradation_rate"] - med) / mad_safe
    is_outlier = robust_z.abs().gt(LAPTIME_OUTLIER_Z_THRESH_P2).fillna(False)
    return df[~is_outlier]


def calibrate_pace_loss_threshold(bucket):
    print("\n" + "=" * 70)
    print("PART 2: PACE_LOSS_THRESHOLD_SECONDS")
    print("=" * 70)

    raw = load_all_teams_laps(bucket)
    raw = _filter_valid_laps(raw)
    raw = _filter_global_degradation_outliers(raw)
    print(f"[filter] {len(raw)} rows after reusing the SAME cleaning pipeline as Regression V2 "
          f"(short-stint, red-flag-buffer, global MAD outlier)")

    df = raw[raw["Compound"].isin(DRY_COMPOUNDS)].copy()
    df["LapTime_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    baseline = df.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapTime_seconds"].transform("min")
    df["pace_loss_seconds"] = df["LapTime_seconds"] - baseline
    df = df[df["pace_loss_seconds"].notna() & (df["pace_loss_seconds"] >= 0)]

    print(f"[data] {len(df)} dry-compound laps with a valid pace_loss_seconds")
    print("\nOverall pace_loss_seconds distribution:")
    print(df["pace_loss_seconds"].describe(percentiles=[.50, .75, .90, .95]).to_string())

    p90 = df["pace_loss_seconds"].quantile(0.90)
    print(f"\n[recommendation] 90th percentile = {p90:.3f}s - a lap this far above baseline is "
          f"genuinely unusual (top 10% of all dry-compound laps), a defensible 'notably elevated' cutoff")
    print(f"[compare] current placeholder PACE_LOSS_THRESHOLD_SECONDS = 1.0")
    return p90


# =============================================================================
# PART 3: CLIFF_PROBABILITY_THRESHOLD - ROC validation of the Cox model
# =============================================================================
MIN_STINT_LENGTH = 5
RED_FLAG_RESTART_BUFFER = 2
LAPTIME_OUTLIER_Z_THRESH = 4.0
CHECKPOINT_HORIZON = 5   # matches cliff_horizon_laps in tyre_life_projection.py
NEGATIVE_CHECKPOINT_BUFFER = 10  # checkpoints this far before the event = definitely "not yet"

CIRCUIT_ID_TO_RACE_NAMES = {
    "MEL": ["Australian_Grand_Prix"], "BAH": ["Bahrain_Grand_Prix", "Sakhir_Grand_Prix"],
    "CHN": ["Chinese_Grand_Prix"], "AZR": ["Azerbaijan_Grand_Prix"], "SPN": ["Spanish_Grand_Prix"],
    "MON": ["Monaco_Grand_Prix"], "CAN": ["Canadian_Grand_Prix"], "FRA": ["French_Grand_Prix"],
    "AUS": ["Austrian_Grand_Prix", "Styrian_Grand_Prix"],
    "UK": ["British_Grand_Prix", "70th_Anniversary_Grand_Prix"],
    "GER": ["German_Grand_Prix"], "HUN": ["Hungarian_Grand_Prix"], "BEL": ["Belgian_Grand_Prix"],
    "ITA": ["Italian_Grand_Prix"], "SIN": ["Singapore_Grand_Prix"], "RUS": ["Russian_Grand_Prix"],
    "JPN": ["Japanese_Grand_Prix"], "TEX": ["United_States_Grand_Prix"],
    "MEX": ["Mexican_Grand_Prix", "Mexico_City_Grand_Prix"],
    "BRA": ["Brazilian_Grand_Prix", "São_Paulo_Grand_Prix"],
    "AUH": ["Abu_Dhabi_Grand_Prix"], "IMO": ["Emilia_Romagna_Grand_Prix"],
    "IST": ["Turkish_Grand_Prix"], "DUT": ["Dutch_Grand_Prix"], "QTR": ["Qatar_Grand_Prix"],
    "KSA": ["Saudi_Arabian_Grand_Prix"], "MIA": ["Miami_Grand_Prix"], "LAS": ["Las_Vegas_Grand_Prix"],
}


def fit_cox_model():
    """Refits the exact same Cox model as survival_model_v2.py."""
    stints = pd.read_csv(STINTS_CSV)
    stints = stints[stints["session"] == "R"]
    stints = stints[~stints["compound"].isin(["WET"])]

    event_counts = stints.groupby("compound")["event"].sum()
    too_few = event_counts[event_counts < 15].index.tolist()
    stints = stints[~stints["compound"].isin(too_few)]

    race_to_id = {r: cid for cid, races in CIRCUIT_ID_TO_RACE_NAMES.items() for r in races}
    stints["circuit_id"] = stints["race"].map(race_to_id)
    taxonomy = pd.read_excel(TAXONOMY_PATH)[["circuit_id", "circuit_degredation"]]
    stints = stints.merge(taxonomy, on="circuit_id", how="left")
    stints["circuit_degredation_ordinal"] = stints["circuit_degredation"].map(
        {"low": 0, "medium": 1, "high": 2})

    required = ["duration", "event", "compound", "track_temp_bucket", "fuel_load_estimate",
                "stint_number", "regulation_era", "circuit_degredation_ordinal"]
    stints = stints.dropna(subset=required)

    design = pd.get_dummies(
        stints[["duration", "event", "compound", "track_temp_bucket", "regulation_era"]],
        columns=["track_temp_bucket", "regulation_era"], drop_first=True,
    )
    design["fuel_load_estimate"] = stints["fuel_load_estimate"].values
    design["stint_number"] = stints["stint_number"].values
    design["circuit_degredation_ordinal"] = stints["circuit_degredation_ordinal"].values
    design["compound"] = stints["compound"].values

    temp_dummy_columns = [c for c in design.columns if c.startswith("track_temp_bucket_")]
    era_dummy_columns = [c for c in design.columns if c.startswith("regulation_era_")]

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(design, duration_col="duration", event_col="event", strata=["compound"])
    return cph, stints, temp_dummy_columns, era_dummy_columns


def calibrate_cliff_probability_threshold():
    print("\n" + "=" * 70)
    print("PART 3: CLIFF_PROBABILITY_THRESHOLD (ROC validation)")
    print("=" * 70)

    cph, stints, temp_dummy_columns, era_dummy_columns = fit_cox_model()
    events = stints[stints["event"] == 1].copy()
    print(f"[data] {len(events)} real cliff events available for checkpoint generation")

    rows = []
    for _, s in events.iterrows():
        duration = s["duration"]
        base_row = {col: 0.0 for col in temp_dummy_columns + era_dummy_columns}
        temp_col = f"track_temp_bucket_{s['track_temp_bucket']}"
        era_col = f"regulation_era_{s['regulation_era']}"
        if temp_col in base_row:
            base_row[temp_col] = 1.0
        if era_col in base_row:
            base_row[era_col] = 1.0
        base_row["fuel_load_estimate"] = s["fuel_load_estimate"]
        base_row["stint_number"] = s["stint_number"]
        base_row["circuit_degredation_ordinal"] = s["circuit_degredation_ordinal"]
        base_row["compound"] = s["compound"]

        # Positive checkpoints: within CHECKPOINT_HORIZON laps of the real cliff
        for offset in range(1, CHECKPOINT_HORIZON + 1):
            checkpoint_age = duration - offset
            if checkpoint_age <= 0:
                continue
            rows.append({**base_row, "checkpoint_age": checkpoint_age, "true_label": 1})

        # Negative checkpoints: well before the event, definitely "not yet"
        checkpoint_age = duration - NEGATIVE_CHECKPOINT_BUFFER
        if checkpoint_age > 0:
            rows.append({**base_row, "checkpoint_age": checkpoint_age, "true_label": 0})

    checkpoints = pd.DataFrame(rows)
    print(f"[data] {len(checkpoints)} checkpoints generated "
          f"({(checkpoints['true_label'] == 1).sum()} positive, "
          f"{(checkpoints['true_label'] == 0).sum()} negative)")

    predicted_probs = []
    n_failures = 0
    first_error = None
    for record in checkpoints.to_dict("records"):  # to_dict avoids the iterrows() dtype trap below
        t0 = record["checkpoint_age"]
        t1 = t0 + CHECKPOINT_HORIZON
        design_row = pd.DataFrame([{k: v for k, v in record.items()
                                     if k not in ("checkpoint_age", "true_label")}])
        try:
            survival_fn = cph.predict_survival_function(design_row, times=[t0, t1])
            s0, s1 = survival_fn.iloc[0, 0], survival_fn.iloc[1, 0]
            predicted_probs.append(1 - (s1 / s0) if s0 > 0 else np.nan)
        except Exception as e:
            predicted_probs.append(np.nan)
            n_failures += 1
            if first_error is None:
                first_error = e

    if n_failures > 0:
        print(f"[warn] {n_failures}/{len(checkpoints)} checkpoints failed to predict - "
              f"first error: {type(first_error).__name__}: {first_error}")

    checkpoints["predicted_probability"] = predicted_probs
    checkpoints = checkpoints.dropna(subset=["predicted_probability"])

    if checkpoints.empty or checkpoints["true_label"].nunique() < 2:
        print("[error] no usable checkpoints survived, or only one class present - "
              "cannot compute ROC curve. See the [warn] line above for the real cause.")
        return None

    fpr, tpr, thresholds = roc_curve(checkpoints["true_label"], checkpoints["predicted_probability"])
    auc = roc_auc_score(checkpoints["true_label"], checkpoints["predicted_probability"])
    print(f"\n[roc] AUC = {auc:.3f}")

    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    print(f"[roc] best threshold (Youden's J): {best_threshold:.3f} "
          f"(TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
    print(f"[compare] current placeholder CLIFF_PROBABILITY_THRESHOLD = 0.30")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color="red", zorder=5,
               label=f"Best threshold ({best_threshold:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cox model: cliff_probability_next_5_laps ROC validation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cliff_probability_roc.png", dpi=150)
    print("[save] cliff_probability_roc.png")

    return best_threshold


if __name__ == "__main__":
    bucket = CachedBucket()
    tyre_age_ratio = calibrate_tyre_age_ratio()
    pace_loss_threshold = calibrate_pace_loss_threshold(bucket)
    cliff_prob_threshold = calibrate_cliff_probability_threshold()

    print("\n" + "=" * 70)
    print("SUMMARY - recommended values vs. placeholders")
    print("=" * 70)
    print(f"TYRE_AGE_TRIGGER_RATIO:      placeholder=0.80  ->  recommended={tyre_age_ratio:.3f}")
    print(f"PACE_LOSS_THRESHOLD_SECONDS: placeholder=1.0   ->  recommended={pace_loss_threshold:.3f}")
    if cliff_prob_threshold is not None:
        print(f"CLIFF_PROBABILITY_THRESHOLD: placeholder=0.30  ->  recommended={cliff_prob_threshold:.3f}")
    else:
        print("CLIFF_PROBABILITY_THRESHOLD: could not be computed - see [error]/[warn] lines above")