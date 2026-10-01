
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
    return pd.concat(frames, ignore_index=True)

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

CIRCUIT_ID_TO_RACE_NAMES = {
    "MEL": ["Australian_Grand_Prix"],
    "BAH": ["Bahrain_Grand_Prix", "Sakhir_Grand_Prix"],
    "CHN": ["Chinese_Grand_Prix"],
    "AZR": ["Azerbaijan_Grand_Prix"],
    "SPN": ["Spanish_Grand_Prix"],
    "MON": ["Monaco_Grand_Prix"],
    "CAN": ["Canadian_Grand_Prix"],
    "FRA": ["French_Grand_Prix"],
    "AUS": ["Austrian_Grand_Prix", "Styrian_Grand_Prix"],
    "UK": ["British_Grand_Prix", "70th_Anniversary_Grand_Prix"],
    "GER": ["German_Grand_Prix"],
    "HUN": ["Hungarian_Grand_Prix"],
    "BEL": ["Belgian_Grand_Prix"],
    "ITA": ["Italian_Grand_Prix"],
    "SIN": ["Singapore_Grand_Prix"],
    "RUS": ["Russian_Grand_Prix"],
    "JPN": ["Japanese_Grand_Prix"],
    "TEX": ["United_States_Grand_Prix"],
    "MEX": ["Mexican_Grand_Prix", "Mexico_City_Grand_Prix"],
    "BRA": ["Brazilian_Grand_Prix", "São_Paulo_Grand_Prix"],
    "AUH": ["Abu_Dhabi_Grand_Prix"],
    "IMO": ["Emilia_Romagna_Grand_Prix"],
    "IST": ["Turkish_Grand_Prix"],
    "DUT": ["Dutch_Grand_Prix"],
    "QTR": ["Qatar_Grand_Prix"],
    "KSA": ["Saudi_Arabian_Grand_Prix"],
    "MIA": ["Miami_Grand_Prix"],
    "LAS": ["Las_Vegas_Grand_Prix"],
}

KNOWN_GAPS = ["Portuguese_Grand_Prix", "Tuscan_Grand_Prix", "Eifel_Grand_Prix"]

def compute_sc_vsc_probability(raw_laps: pd.DataFrame) -> pd.DataFrame:
    ""
    session_flags = raw_laps.groupby(["Season", "Race", "Session"]).agg(
        any_sc=("is_sc_lap", "any"), any_vsc=("is_vsc_lap", "any")
    ).reset_index()
    per_circuit = session_flags.groupby("Race").agg(
        sc_probability_computed=("any_sc", "mean"),
        vsc_probability_computed=("any_vsc", "mean"),
        n_sessions=("any_sc", "count"),
    ).reset_index()
    return per_circuit

TAXONOMY_PATH = r"src\taxanomy\circuit_taxonomy.xlsx"
OUTPUT_PATH = r"src\taxanomy\circuit_taxonomy_updated.xlsx"

if __name__ == "__main__":
    bucket = CachedBucket()
    print("[load] pulling ALL TEAMS' laps_features.csv ...")
    raw = load_all_teams_laps(bucket)
    print(f"[load] {len(raw)} raw rows")

    print("\n[decision] circuit_degredation is NOT being computed or overwritten - see "
          "circuit_severity_investigation_notes.md. Four empirical approaches were tried and "
          "each failed for a distinct, documented reason; the manual values are being kept "
          "deliberately. Only sc_probability_pct/vsc_probability_pct are updated below.")
    sc_vsc = compute_sc_vsc_probability(raw)
    print(f"\n[computed] SC/VSC probability for {len(sc_vsc)} Race names found in your data")

    gaps_present = [g for g in KNOWN_GAPS if g in sc_vsc["Race"].values]
    if gaps_present:
        print(f"[gap] these races have NO taxonomy row and are NOT added automatically "
              f"(can't derive circuit_type/overtaking_difficulty from lap data): {gaps_present}")

    race_to_id = {}
    for cid, race_names in CIRCUIT_ID_TO_RACE_NAMES.items():
        for r in race_names:
            race_to_id[r] = cid
    sc_vsc["circuit_id"] = sc_vsc["Race"].map(race_to_id)

    unmapped = sc_vsc[sc_vsc["circuit_id"].isna() & ~sc_vsc["Race"].isin(KNOWN_GAPS)]
    if not unmapped.empty:
        print(f"[warn] Race names with NO mapping and NOT in KNOWN_GAPS - check "
              f"CIRCUIT_ID_TO_RACE_NAMES: {unmapped['Race'].tolist()}")

    per_id = sc_vsc.dropna(subset=["circuit_id"]).groupby("circuit_id").agg(
        sc_probability_computed=("sc_probability_computed", "mean"),
        vsc_probability_computed=("vsc_probability_computed", "mean"),
        n_sessions=("n_sessions", "sum"),
    ).reset_index()

    taxonomy = pd.read_excel(TAXONOMY_PATH)
    print(f"\n[load] {len(taxonomy)} rows from {TAXONOMY_PATH}")

    merged = taxonomy.merge(per_id, on="circuit_id", how="left")

    no_data = merged[merged["sc_probability_computed"].isna()]
    if not no_data.empty:
        print(f"[warn] taxonomy circuit_ids with NO matching lap data found: "
              f"{no_data['circuit_id'].tolist()} - old manual values kept for these")

    merged["sc_probability_pct"] = merged["sc_probability_computed"].combine_first(
        merged["sc_probability_pct"])
    merged["vsc_probability_pct"] = merged["vsc_probability_computed"].combine_first(
        merged["vsc_probability_pct"])

    final = merged.drop(columns=["sc_probability_computed", "vsc_probability_computed", "n_sessions"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final.to_excel(OUTPUT_PATH, index=False)
    print(f"\n[save] {OUTPUT_PATH} - review this against your original before replacing it")
    print("\n=== BEFORE vs AFTER (sc_probability_pct) ===")
    print("(low n_sessions = treat the percentage with real caution - see German/Istanbul/Vegas GPs)")
    comparison = taxonomy[["circuit_id", "sc_probability_pct"]].merge(
        merged[["circuit_id", "sc_probability_pct", "n_sessions"]], on="circuit_id", suffixes=("_old", "_new"))
    print(comparison.to_string(index=False))
