"""
flatten_telemetry.py

Flattens tracinginsights telemetry JSON (one file = one driver's session,
containing ~15 parallel arrays aligned by index) into a proper long-format
table: one row per timestamp per driver.

Input shape (per file):
    {
      "tel.time": [0.0, 0.032, 0.151, ...],
      "tel.speed": [222.3, 223.0, 227.0, ...],
      "tel.rpm": [...],
      ...
      "tel.dataKey": "2018-Abu Dhabi Grand Prix-Race-ALO-10"   <- single value, not array
    }

Output shape (long format):
    time | speed | rpm | throttle | brake | drs | ... | driver_code | driver_number | year | race | session

This script does TWO things:
1. A "profile" mode: flattens a sample of files and profiles the REAL
   per-sample values (actual speed range, RPM range etc.), not just
   array lengths -- this is what was missing from column_profile.py.
2. A "convert" mode: flattens specific file(s)/session(s) into an actual
   parquet/csv you can use downstream, since flattening all 197,909 files
   into one object would be huge -- do this per-session or per-race as needed.

Requirements:
    pip install google-cloud-storage pandas gcsfs python-dotenv --break-system-packages

Usage:
    # Profile real value ranges across a sample (recommended first step)
    python flatten_telemetry.py --mode profile --files-per-year 3

    # Flatten one specific session into a single parquet file
    python flatten_telemetry.py --mode convert --session "2018/Abu Dhabi Grand Prix/Race"
"""

import argparse
import io
import json
import os
import re
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

TELEMETRY_PREFIX = "raw/tracinginsights/"


def extract_year(path: str) -> str | None:
    m = re.search(r"/((?:19|20)\d{2})/", path)
    return m.group(1) if m else None


def parse_driver_file_path(path: str) -> dict:
    """
    raw/tracinginsights/<year>/<race>/<session>/<driver_code>/<driver_number>_tel.json
    """
    parts = path.rstrip("/").split("/")
    filename = parts[-1]
    driver_number = filename.split("_")[0]
    return {
        "year": parts[-5] if len(parts) >= 5 else None,
        "race": parts[-4] if len(parts) >= 4 else None,
        "session": parts[-3] if len(parts) >= 3 else None,
        "driver_code": parts[-2],
        "driver_number": driver_number,
    }


def find_fields(obj: dict, prefix: str = "") -> tuple[dict, dict]:
    """
    Recursively walks a (possibly nested) JSON dict and returns
    (array_fields, scalar_fields) with dot-notation keys, e.g.
    {"tel": {"time": [...]}} -> array_fields = {"tel.time": [...]}.
    Handles any nesting depth instead of assuming arrays sit at top level.
    """
    array_fields, scalar_fields = {}, {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, list):
            array_fields[key] = v
        elif isinstance(v, dict):
            nested_arrays, nested_scalars = find_fields(v, prefix=f"{key}.")
            array_fields.update(nested_arrays)
            scalar_fields.update(nested_scalars)
        else:
            scalar_fields[key] = v
    return array_fields, scalar_fields


def flatten_one_file(path: str) -> pd.DataFrame | None:
    """Reads one driver telemetry JSON and returns a long-format DataFrame, or None on failure."""
    try:
        data = bucket.blob(path).download_as_bytes()
        obj = json.loads(data)
    except Exception as e:
        print(f"    Skipped {path}: {e}")
        return None

    meta = parse_driver_file_path(path)

    if not isinstance(obj, dict):
        print(f"    Unexpected JSON top-level type ({type(obj)}) in {path}, skipping.")
        return None

    array_fields, scalar_fields = find_fields(obj)

    if not array_fields:
        print(f"    No array fields found in {path}, skipping.")
        return None

    lengths = {k: len(v) for k, v in array_fields.items()}
    max_len = max(lengths.values())
    mismatched = {k: n for k, n in lengths.items() if n != max_len}
    if mismatched:
        print(f"    WARNING: array length mismatch in {path}: {mismatched} (expected {max_len}). "
              f"Truncating/padding to shortest common length.")
        min_len = min(lengths.values())
        array_fields = {k: v[:min_len] for k, v in array_fields.items()}

    df = pd.DataFrame(array_fields)
    df.columns = [c.replace("tel.", "") for c in df.columns]

    # --- Known data-quality fixes, applied here because they're part of
    # producing a CORRECT flattened row, not just a reshaped one. Each is
    # documented in the data dictionary as a known source-data issue. ---

    # 1. DriverAhead: missing values come through as the literal string
    #    "None" rather than a real null, which silently breaks null-checks
    #    and any downstream groupby/merge on this column.
    if "DriverAhead" in df.columns:
        df["DriverAhead"] = df["DriverAhead"].replace("None", pd.NA)

    # 2. throttle: a value of exactly 104 is a documented sensor
    #    placeholder/error code, not a real throttle percentage (which
    #    should be 0-100). Null it out rather than silently keeping a
    #    physically impossible value in the dataset.
    if "throttle" in df.columns:
        invalid_throttle = df["throttle"] > 100
        if invalid_throttle.any():
            df.loc[invalid_throttle, "throttle"] = pd.NA

    for k, v in scalar_fields.items():
        df[k.replace("tel.", "")] = v
    for k, v in meta.items():
        df[k] = v

    return df


def sample_paths_across_years(files_per_year: int) -> list[str]:
    print("Listing telemetry files (this takes a few minutes)...")
    all_paths = [b.name for b in client.list_blobs(BUCKET_NAME, prefix=TELEMETRY_PREFIX) if b.name.lower().endswith(".json")]
    print(f"  Found {len(all_paths)} total.")

    session_groups = defaultdict(list)
    for p in all_paths:
        parts = p.rstrip("/").split("/")
        session_folder = "/".join(parts[:-2])
        session_groups[session_folder].append(p)

    sessions_by_year = defaultdict(list)
    for session_folder in session_groups:
        sessions_by_year[extract_year(session_folder)].append(session_folder)

    sampled_files = []
    for year, sessions in sorted(sessions_by_year.items()):
        for session_folder in sessions[:files_per_year]:
            sampled_files.extend(session_groups[session_folder][:2])  # 2 drivers per sampled session
    return sampled_files


def run_profile(files_per_year: int):
    sample_paths = sample_paths_across_years(files_per_year)
    print(f"Flattening {len(sample_paths)} driver-files for real-value profiling...")

    flattened = [flatten_one_file(p) for p in sample_paths]
    flattened = [df for df in flattened if df is not None]
    if not flattened:
        print("Nothing flattened, aborting.")
        return

    combined = pd.concat(flattened, ignore_index=True)
    print(f"\nFlattened to {len(combined)} timestamp-level rows across {len(sample_paths)} driver-files.")

    rows = []
    for col in combined.columns:
        series = combined[col]
        pct_null = round(series.isna().mean() * 100, 2)
        dtype = str(series.dtype)
        non_null = series.dropna()

        if pd.api.types.is_numeric_dtype(series):
            rng = f"min={non_null.min()}, max={non_null.max()}, mean={round(non_null.mean(), 3)}" if len(non_null) else "N/A"
        else:
            uniques = non_null.unique()
            rng = f"unique values: {sorted(map(str, uniques))}" if len(uniques) <= 15 else f"{len(uniques)} unique values"

        rows.append({
            "column": col, "dtype": dtype, "pct_null": pct_null,
            "range_or_unique_values": rng, "example_value": non_null.iloc[0] if len(non_null) else None,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv("telemetry_flattened_profile.csv", index=False)
    print("\nWritten to telemetry_flattened_profile.csv")
    print(result_df.to_string(index=False))


def run_convert(session_prefix: str, out_path: str):
    full_prefix = f"{TELEMETRY_PREFIX}{session_prefix}/"
    print(f"Listing driver files under {full_prefix} ...")
    paths = [b.name for b in client.list_blobs(BUCKET_NAME, prefix=full_prefix) if b.name.lower().endswith(".json")]
    print(f"  Found {len(paths)} driver files for this session.")

    flattened = [flatten_one_file(p) for p in paths]
    flattened = [df for df in flattened if df is not None]
    if not flattened:
        print("Nothing flattened.")
        return

    combined = pd.concat(flattened, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"\nDone. {len(combined)} rows across {combined['driver_code'].nunique()} drivers written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["profile", "convert"], required=True)
    parser.add_argument("--files-per-year", type=int, default=3, help="[profile mode] sessions sampled per year")
    parser.add_argument("--session", help="[convert mode] e.g. '2018/Abu Dhabi Grand Prix/Race'")
    parser.add_argument("--out", default="telemetry_session.parquet", help="[convert mode] output path")
    args = parser.parse_args()

    if args.mode == "profile":
        run_profile(args.files_per_year)
    else:
        if not args.session:
            raise SystemExit("--session is required in convert mode, e.g. --session '2018/Abu Dhabi Grand Prix/Race'")
        run_convert(args.session, args.out)