
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
TELEMETRY_PREFIX = "clean/tracinginsights/"
TELEMETRY_AVAILABLE_SESSIONS = {"R", "S"}

class CachedBucket:
    ""

    def __init__(self, bucket_name: str, cache_dir: Path = CACHE_DIR):
        print(f"[init] Connecting to GCS bucket '{bucket_name}' ...")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] Connected. Local cache dir: {self.cache_dir.resolve()}")

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

    def list_blob_names(self, prefix: str, retries: int = 3, timeout: int = 120) -> list[str]:
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return [b.name for b in self.client.list_blobs(self.bucket.name, prefix=prefix, timeout=timeout)]
            except Exception as e:
                last_error = e
                print(f"    [warn] list_blobs(prefix={prefix!r}) attempt {attempt}/{retries} failed "
                      f"({type(e).__name__}: {e}), retrying...")
                time.sleep(2 * attempt)
        raise last_error

    def exists(self, blob_path: str) -> bool:
        if self._local_path(blob_path).exists():
            return True
        return self.bucket.blob(blob_path).exists(timeout=30)

bucket = CachedBucket(BUCKET_NAME)

def read_csv_robust(data: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception:
            return None

def find_cleaned_sessions(year: str) -> list[tuple[str, str, str]]:
    prefix = f"{LAPS_PREFIX}{year}/" if year != "all" else LAPS_PREFIX
    print(f"[scan] Listing {prefix} ...")
    paths = bucket.list_blob_names(prefix)
    sessions = set()
    for p in paths:
        if p.endswith("laps_flagged.csv"):
            parts = p.rstrip("/").split("/")
            if len(parts) >= 4:
                sessions.add((parts[-4], parts[-3], parts[-2]))
    print(f"[scan] Found {len(sessions)} cleaned sessions.")
    return sorted(sessions)

def check_session(year: str, race: str, session: str) -> dict:
    result = {
        "year": year, "race": race, "session": session,
        "expects_telemetry": session in TELEMETRY_AVAILABLE_SESSIONS,
        "telemetry_exists": None,
        "laps_lapcount": None,
        "telemetry_lapcount": None,
        "lapcount_mismatch_pct": None,
        "status": None,
    }

    if not result["expects_telemetry"]:
        result["status"] = "skipped (no telemetry expected)"
        return result

    telemetry_path = f"{TELEMETRY_PREFIX}{year}/{race}/{session}/telemetry_by_lap.csv"
    exists = bucket.exists(telemetry_path)
    result["telemetry_exists"] = exists

    if not exists:
        result["status"] = "MISSING"
        return result

    laps_path = f"{LAPS_PREFIX}{year}/{race}/{session}/laps_flagged.csv"
    try:
        laps_df = read_csv_robust(bucket.download_as_bytes(laps_path))
        tel_df = read_csv_robust(bucket.download_as_bytes(telemetry_path))
    except Exception as e:
        result["status"] = f"ERROR reading files ({type(e).__name__}: {e})"
        return result

    if laps_df is None or tel_df is None or "LapNumber" not in laps_df.columns or "LapNumber" not in tel_df.columns:
        result["status"] = "ERROR (missing LapNumber column)"
        return result

    n_laps = laps_df["LapNumber"].nunique()
    n_tel_laps = tel_df["LapNumber"].nunique()
    result["laps_lapcount"] = n_laps
    result["telemetry_lapcount"] = n_tel_laps
    mismatch_pct = round(abs(n_laps - n_tel_laps) / n_laps * 100, 1) if n_laps else None
    result["lapcount_mismatch_pct"] = mismatch_pct

    if mismatch_pct is not None and mismatch_pct > 20:
        result["status"] = f"SUSPICIOUS (lap count off by {mismatch_pct}%)"
    else:
        result["status"] = "ok"

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="all", help="e.g. 2024, or 'all'")
    args = parser.parse_args()

    sessions = find_cleaned_sessions(args.year)
    print(f"\nChecking {len(sessions)} sessions (only Race/Sprint expect telemetry) ...\n")

    results = []
    for i, (y, r, s) in enumerate(sessions):
        res = check_session(y, r, s)
        results.append(res)
        if res["expects_telemetry"] and res["status"] != "ok":
            print(f"  [{res['status']}] {y}/{r}/{s}")
        if (i + 1) % 100 == 0:
            print(f"    ...{i + 1}/{len(sessions)} checked")

    df = pd.DataFrame(results)
    df.to_csv("telemetry_coverage_report.csv", index=False)

    expected = df[df["expects_telemetry"]]
    n_expected = len(expected)
    n_missing = (expected["status"] == "MISSING").sum()
    n_suspicious = expected["status"].str.startswith("SUSPICIOUS", na=False).sum()
    n_error = expected["status"].str.startswith("ERROR", na=False).sum()
    n_ok = (expected["status"] == "ok").sum()

    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print(f"Total cleaned sessions found:      {len(sessions)}")
    print(f"Sessions expecting telemetry (R/S): {n_expected}")
    print(f"  -> ok:                            {n_ok}")
    print(f"  -> MISSING telemetry entirely:    {n_missing}")
    print(f"  -> SUSPICIOUS (lap count mismatch): {n_suspicious}")
    print(f"  -> ERROR (couldn't verify):       {n_error}")
    print(f"\nFull report written to telemetry_coverage_report.csv")

    if n_missing or n_suspicious or n_error:
        print("\nSessions needing attention:")
        problem_rows = expected[expected["status"] != "ok"]
        print(problem_rows[["year", "race", "session", "status", "laps_lapcount", "telemetry_lapcount"]]
              .to_string(index=False))
