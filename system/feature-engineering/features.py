"""
engineer_features.py

Feature engineering pass #1 (per Master Checklist):
    tyre_age, compound_encoded, stint_number, gap_to_leader, gap_to_car_ahead
Plus, since they don't need any new data source:
    degradation_rate, fuel_load_estimate

Deliberately NOT included here: track_temp_bucket -- needs weather.csv,
which hasn't been profiled yet (raw/fastf1/.../weather.csv likely exists
per column_profile.py's original scope, but its real schema is unverified).
Given how many times a schema guess has caused rework this project, that's
a separate investigate-first step, not bolted on blind here.

Reads clean/fastf1/<year>/<race>/<session>/laps_flagged.csv, writes
clean/features/<year>/<race>/<session>/laps_features.csv (superset of
laps_flagged.csv's columns, plus the new engineered ones).

Requirements:
    pip install google-cloud-storage pandas python-dotenv --break-system-packages

Usage:
    python engineer_features.py --session "2024/Abu_Dhabi_Grand_Prix/R"
    python engineer_features.py --year 2024
    python engineer_features.py --year all
"""

import argparse
import io
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = Path(os.getenv("GCS_CACHE_DIR", "./gcs_cache"))

LAPS_PREFIX = "clean/fastf1/"
FEATURES_PREFIX = "clean/features/"

# Ordinal compound encoding, softest -> hardest. Direction is a choice, not
# a fact -- documented here so it's not ambiguous downstream. Raw Compound
# string is kept in output too, nothing is lost.
COMPOUND_ORDER = {"WET": 0, "INTERMEDIATE": 1, "HARD": 2, "MEDIUM": 3, "SOFT": 4}

# Simplifying assumption, NOT measured telemetry -- FastF1/tracinginsights
# don't expose actual fuel load. This approximates a full-tank start and
# linear burn to empty by the final lap. Good enough as a proxy feature;
# revisit with per-year/per-circuit fuel consumption data if precision
# matters later.
FUEL_START_KG = 110.0


class CachedBucket:
    def __init__(self, bucket_name: str, cache_dir: Path = CACHE_DIR):
        print(f"[init] Connecting to GCS bucket '{bucket_name}' ...")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] Connected.")

    def _local_path(self, blob_path: str) -> Path:
        return self.cache_dir / blob_path

    def download_as_bytes(self, blob_path: str) -> bytes:
        local_path = self._local_path(blob_path)
        if local_path.exists():
            return local_path.read_bytes()
        data = self.bucket.blob(blob_path).download_as_bytes()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return data

    def upload_from_string(self, blob_path: str, content: str, content_type: str = "text/csv"):
        self.bucket.blob(blob_path).upload_from_string(content, content_type=content_type)
        local_path = self._local_path(blob_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")

    def exists(self, blob_path: str) -> bool:
        if self._local_path(blob_path).exists():
            return True
        return self.bucket.blob(blob_path).exists(timeout=30)

    def list_blob_names(self, prefix: str, retries: int = 3, timeout: int = 120) -> list[str]:
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return [b.name for b in self.client.list_blobs(self.bucket.name, prefix=prefix, timeout=timeout)]
            except Exception as e:
                last_error = e
                print(f"    [warn] list_blobs attempt {attempt}/{retries} failed ({e}), retrying...")
                time.sleep(2 * attempt)
        raise last_error


bucket = CachedBucket(BUCKET_NAME)


def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception:
            return None


def to_seconds(series: pd.Series) -> pd.Series:
    """Handles both timedelta-string columns and already-numeric seconds."""
    out = pd.to_timedelta(series, errors="coerce").dt.total_seconds()
    if out.isna().all():
        out = pd.to_numeric(series, errors="coerce")
    return out


def add_tyre_age_and_compound(df: pd.DataFrame) -> pd.DataFrame:
    if "TyreLife" in df.columns and df["TyreLife"].notna().any():
        df["tyre_age"] = df["TyreLife"]
    else:
        # fall back: rank within (Driver, Stint), 1-indexed
        df["tyre_age"] = df.groupby(["Driver", "Stint"]).cumcount() + 1

    df["compound_encoded"] = df["Compound"].map(COMPOUND_ORDER)
    df["stint_number"] = df["Stint"]
    return df


def add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    gap_to_leader / gap_to_car_ahead, computed within each LapNumber group
    using cumulative session Time. This is the standard lap-based proxy for
    interval, NOT a true same-instant gap -- a lapped car's "same LapNumber"
    isn't literally the same moment on track as the leader's. Good enough
    for strategy features; flag this caveat if exact real-time gaps ever
    matter (e.g. actual pit-window decisions), since that needs continuous
    timing data, not lap-level.
    """
    df["_time_sec"] = to_seconds(df["Time"]) if "Time" in df.columns else pd.NA

    df["gap_to_leader"] = pd.NA
    df["gap_to_car_ahead"] = pd.NA

    for lap_num, group in df.groupby("LapNumber"):
        valid = group["_time_sec"].notna()
        if valid.sum() < 1:
            continue
        sorted_group = group[valid].sort_values("_time_sec")
        leader_time = sorted_group["_time_sec"].iloc[0]
        df.loc[sorted_group.index, "gap_to_leader"] = sorted_group["_time_sec"] - leader_time
        df.loc[sorted_group.index, "gap_to_car_ahead"] = sorted_group["_time_sec"].diff().fillna(0)

    df = df.drop(columns=["_time_sec"])
    return df


def add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pace-loss-per-lap-of-tyre-age proxy. Baseline = fastest CLEAN lap
    (excludes pit/out/in/SC/VSC/missing/outlier flagged laps, using the
    flags already built by clean_laps.py) within each (Driver, Stint).
    degradation_rate = (this lap's time - baseline) / tyre_age.
    Positive = losing time vs fresh-tyre pace; near-zero/negative on the
    baseline lap itself and early in a stint is expected.
    """
    df["_laptime_sec"] = to_seconds(df["LapTime"]) if "LapTime" in df.columns else pd.NA

    clean_flag_cols = ["is_pit_in", "is_pit_out", "is_out_lap", "is_in_lap",
                        "is_sc_lap", "is_vsc_lap", "is_missing_laptime", "is_outlier_laptime"]
    available_flags = [c for c in clean_flag_cols if c in df.columns]
    if available_flags:
        is_clean = ~df[available_flags].any(axis=1)
    else:
        is_clean = df["_laptime_sec"].notna()

    df["degradation_rate"] = pd.NA
    for (driver, stint), idx in df.groupby(["Driver", "Stint"]).groups.items():
        stint_df = df.loc[idx]
        clean_times = stint_df.loc[is_clean.loc[idx] & stint_df["_laptime_sec"].notna(), "_laptime_sec"]
        if clean_times.empty:
            continue
        baseline = clean_times.min()
        tyre_age_safe = stint_df["tyre_age"].clip(lower=1)
        df.loc[idx, "degradation_rate"] = (stint_df["_laptime_sec"] - baseline) / tyre_age_safe

    df = df.drop(columns=["_laptime_sec"])
    return df


def add_fuel_load_estimate(df: pd.DataFrame) -> pd.DataFrame:
    if "LapNumber" not in df.columns:
        df["fuel_load_estimate"] = pd.NA
        return df
    total_laps = df["LapNumber"].max()
    if pd.isna(total_laps) or total_laps <= 0:
        df["fuel_load_estimate"] = pd.NA
        return df
    fraction_remaining = 1 - (df["LapNumber"] - 1) / total_laps
    df["fuel_load_estimate"] = (FUEL_START_KG * fraction_remaining).clip(lower=0)
    return df


def engineer_session(year: str, race: str, session: str, force: bool = False) -> bool:
    out_path = f"{FEATURES_PREFIX}{year}/{race}/{session}/laps_features.csv"
    if not force and bucket.exists(out_path):
        print(f"[skip] {year}/{race}/{session} -- already exists (use --force to redo).")
        return True

    laps_path = f"{LAPS_PREFIX}{year}/{race}/{session}/laps_flagged.csv"
    print(f"[engineer] {year}/{race}/{session}: loading {laps_path} ...")
    try:
        df = read_csv_robust(bucket.download_as_bytes(laps_path))
    except Exception as e:
        print(f"    [error] Could not load laps_flagged.csv: {type(e).__name__}: {e}")
        return False

    if df is None or df.empty:
        print(f"    [error] laps_flagged.csv empty or unreadable.")
        return False

    required = {"Driver", "Stint", "LapNumber", "Compound"}
    missing = required - set(df.columns)
    if missing:
        print(f"    [error] Missing required columns {missing}, skipping.")
        return False

    df = add_tyre_age_and_compound(df)
    df = add_gaps(df)
    df = add_degradation_rate(df)
    df = add_fuel_load_estimate(df)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    bucket.upload_from_string(out_path, buf.getvalue(), content_type="text/csv")
    print(f"    Wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    return True


def list_sessions(year: str) -> list[tuple[str, str, str]]:
    prefix = f"{LAPS_PREFIX}{year}/" if year != "all" else LAPS_PREFIX
    print(f"[list] Listing {prefix} ...")
    paths = bucket.list_blob_names(prefix)
    sessions = set()
    for p in paths:
        if p.endswith("laps_flagged.csv"):
            parts = p.rstrip("/").split("/")
            if len(parts) >= 4:
                sessions.add((parts[-4], parts[-3], parts[-2]))
    print(f"[list] Found {len(sessions)} sessions.")
    return sorted(sessions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="e.g. '2024/Abu_Dhabi_Grand_Prix/R'")
    parser.add_argument("--year", help="e.g. 2024, or 'all'")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.session:
        y, r, s = args.session.strip("/").split("/")
        engineer_session(y, r, s, force=args.force)
    elif args.year:
        sessions = list_sessions(args.year)
        n_failed = 0
        failed = []
        for y, r, s in sessions:
            try:
                ok = engineer_session(y, r, s, force=args.force)
                if not ok:
                    n_failed += 1
                    failed.append(f"{y}/{r}/{s}")
            except Exception as e:
                n_failed += 1
                failed.append(f"{y}/{r}/{s}")
                print(f"[error] {y}/{r}/{s} FAILED: {type(e).__name__}: {e}")
        print(f"\nDone. {len(sessions) - n_failed}/{len(sessions)} succeeded.")
        if failed:
            print("Failed sessions (re-run same command to retry just these):")
            for f in failed:
                print(f"    {f}")
    else:
        raise SystemExit("Provide --session or --year")