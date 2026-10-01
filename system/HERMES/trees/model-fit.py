""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
from lifelines import CoxPHFitter

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
        self.bucket.blob(blob_path).download_to_filename(local_path)
        return pd.read_csv(local_path, **kwargs)

    def list_blob_names(self, prefix):
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]

RBR_ALIASES = {
    "Red Bull Racing", "Red Bull Racing Honda", "Red Bull Racing RBPT",
    "Oracle Red Bull Racing", "Red Bull",
}
DRY_COMPOUNDS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
MIN_ROWS_PER_GROUP = 40
MIN_STINT_LENGTH = 5
RED_FLAG_RESTART_BUFFER = 2
LAPTIME_OUTLIER_Z_THRESH = 4.0
MIN_EVENTS_PER_COMPOUND = 15
EXCLUDED_COMPOUNDS_COX = ["WET"]

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
TAXONOMY_PATH = "src/taxanomy/circuit_taxonomy.xlsx"

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

def add_pace_loss_and_era(clean_laps: pd.DataFrame) -> pd.DataFrame:
    df = clean_laps[clean_laps["Compound"].isin(DRY_COMPOUNDS)].copy()
    df["LapTime_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    baseline = df.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapTime_seconds"].transform("min")
    df["pace_loss_seconds"] = df["LapTime_seconds"] - baseline
    df["regulation_era"] = pd.cut(
        df["Season"], bins=[-np.inf, 2021, 2025, np.inf],
        labels=["2018-2021", "2022-2025", "2026+"],
    )
    return df

def build_regression_design_matrix(df: pd.DataFrame, temp_dummy_columns: list) -> pd.DataFrame:
    temp_dummies = pd.get_dummies(df["track_temp_bucket"], prefix="temp", drop_first=True)
    temp_dummies = temp_dummies.reindex(columns=temp_dummy_columns, fill_value=0).astype(float)
    X = pd.concat([df[["tyre_age", "fuel_load_estimate", "stint_number"]].reset_index(drop=True),
                   temp_dummies.reset_index(drop=True)], axis=1)
    return X

def fit_regression_v2_models(modeling_df: pd.DataFrame, temp_dummy_columns: list) -> dict:
    from sklearn.linear_model import LinearRegression
    models = {}
    skipped = 0
    for (compound, circuit, era), g in modeling_df.groupby(["Compound", "Race", "regulation_era"], observed=True):
        g = g.dropna(subset=["tyre_age", "fuel_load_estimate", "stint_number",
                              "track_temp_bucket", "pace_loss_seconds"])
        if len(g) < MIN_ROWS_PER_GROUP:
            skipped += 1
            continue
        X = build_regression_design_matrix(g, temp_dummy_columns)
        y = g["pace_loss_seconds"].reset_index(drop=True)
        models[(compound, circuit, era)] = LinearRegression().fit(X, y)
    print(f"[regression] fitted {len(models)} (compound, circuit, era) models, "
          f"skipped {skipped} groups below MIN_ROWS_PER_GROUP={MIN_ROWS_PER_GROUP}")
    return models

def load_cliff_detection_stints(bucket: CachedBucket, path: str) -> pd.DataFrame:
    ""
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[cox] {path} not found locally, trying GCS at processed/{path} ...")
    return bucket.read_csv(f"processed/{path}")

def prepare_cox_data(stints: pd.DataFrame, bucket: CachedBucket) -> pd.DataFrame:
    if "session" in stints.columns:
        stints = stints[stints["session"] == "R"]
    stints = stints[~stints["compound"].isin(EXCLUDED_COMPOUNDS_COX)]

    event_counts = stints.groupby("compound")["event"].sum().sort_values(ascending=False)
    too_few = event_counts[event_counts < MIN_EVENTS_PER_COMPOUND].index.tolist()
    if too_few:
        print(f"[cox] excluding {too_few} -- fewer than {MIN_EVENTS_PER_COMPOUND} events")
        stints = stints[~stints["compound"].isin(too_few)]

    race_to_id = {r: cid for cid, races in CIRCUIT_ID_TO_RACE_NAMES.items() for r in races}
    stints["circuit_id"] = stints["race"].map(race_to_id)

    taxonomy = pd.read_excel(TAXONOMY_PATH)[["circuit_id", "circuit_degredation"]]
    stints = stints.merge(taxonomy, on="circuit_id", how="left")
    stints = stints.dropna(subset=["circuit_degredation"])

    severity_map = {"low": 0, "medium": 1, "high": 2}
    stints["circuit_degredation_ordinal"] = stints["circuit_degredation"].map(severity_map)
    return stints

def build_cox_design_matrix(stints: pd.DataFrame) -> tuple:
    required = ["duration", "event", "compound", "track_temp_bucket", "fuel_load_estimate",
                "stint_number", "regulation_era", "circuit_degredation_ordinal"]
    df = stints.dropna(subset=required).copy()

    design = pd.get_dummies(
        df[["duration", "event", "compound", "track_temp_bucket", "regulation_era"]],
        columns=["track_temp_bucket", "regulation_era"], drop_first=True,
        prefix={"track_temp_bucket": "temp", "regulation_era": "regulation_era"},
    )
    design["fuel_load_estimate"] = df["fuel_load_estimate"].values
    design["stint_number"] = df["stint_number"].values
    design["circuit_degredation_ordinal"] = df["circuit_degredation_ordinal"].values
    design["compound"] = df["compound"].values

    temp_dummy_columns = [c for c in design.columns if c.startswith("temp_")]
    era_dummy_columns = [c for c in design.columns if c.startswith("regulation_era_")]
    return design, temp_dummy_columns, era_dummy_columns

def fit_cox_v2_model(design: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(design, duration_col="duration", event_col="event", strata=["compound"])
    print(f"[cox] fitted, stratified by compound: {design['compound'].unique().tolist()}, "
          f"concordance={cph.concordance_index_:.3f}")
    return cph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliff-stints-csv", default="cliff_detection_stints.csv",
                         help="Path to cliff_detection_stints.csv, produced by cliff_detection.py")
    parser.add_argument("--output", default="tyre_life_models.pkl")
    args = parser.parse_args()

    bucket = CachedBucket()

    print("[load] pulling ALL TEAMS' laps_features.csv for Regression V2 ...")
    raw = load_all_teams_laps(bucket)
    clean = filter_valid_laps(raw)
    clean = filter_global_degradation_outliers(clean)
    modeling_df = add_pace_loss_and_era(clean)
    print(f"[regression] {len(modeling_df)} rows after full cleaning pipeline")

    reg_temp_dummy_columns = sorted(pd.get_dummies(
        modeling_df["track_temp_bucket"], prefix="temp", drop_first=True).columns.tolist())
    reg_models = fit_regression_v2_models(modeling_df, reg_temp_dummy_columns)

    print(f"\n[load] reading {args.cliff_stints_csv} for Cox V2 ...")
    stints_raw = load_cliff_detection_stints(bucket, args.cliff_stints_csv)
    stints = prepare_cox_data(stints_raw, bucket)
    cox_design, cox_temp_dummy_columns, era_dummy_columns = build_cox_design_matrix(stints)
    cph = fit_cox_v2_model(cox_design)

    if set(reg_temp_dummy_columns) != set(cox_temp_dummy_columns):
        raise AssertionError(
            f"temp dummy columns still differ after prefix fix -- Regression V2: "
            f"{reg_temp_dummy_columns}, Cox: {cox_temp_dummy_columns}. "
            f"build_tyre_life_projection() requires ONE shared list; a mismatch here means "
            f"either model would silently one-hot-encode temperature wrong. Stop and "
            f"investigate rather than union them -- a union pads each model's design "
            f"matrix with permanently-zero columns it was never trained on."
        )
    temp_dummy_columns = reg_temp_dummy_columns
    print(f"[check] temp dummy columns match across both models: {temp_dummy_columns}")

    payload = {
        "reg_models": reg_models,
        "cph": cph,
        "temp_dummy_columns": temp_dummy_columns,
        "era_dummy_columns": era_dummy_columns,
    }
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[save] {args.output} -- {len(reg_models)} regression models, "
          f"1 Cox model, {len(temp_dummy_columns)} temp dummy columns, "
          f"{len(era_dummy_columns)} era dummy columns")
    print("\n[next] load this pickle in the tree instead of calling fit_all_models() per lap:")
    print("""
    import pickle
    from tyre_life_projection import build_tyre_life_projection

    with open("tyre_life_models.pkl", "rb") as f:
        models = pickle.load(f)

    projection = build_tyre_life_projection(
        reg_models=models["reg_models"], cph=models["cph"],
        temp_dummy_columns=models["temp_dummy_columns"],
        era_dummy_columns=models["era_dummy_columns"],
        compound=..., circuit=..., regulation_era=..., tyre_age=...,
        fuel_load_estimate=..., stint_number=..., track_temp_bucket=...,
        circuit_degredation_ordinal=...,
    )
    """)
