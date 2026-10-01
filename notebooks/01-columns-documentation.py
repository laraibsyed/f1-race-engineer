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

def list_files(prefix: str, exts: tuple[str, ...]) -> list[str]:
    return [b.name for b in client.list_blobs(BUCKET_NAME, prefix=prefix) if b.name.lower().endswith(exts)]

def extract_year(path: str) -> str | None:
    m = re.search(r"/((?:19|20)\d{2})/", path)
    return m.group(1) if m else None

def sample_across_years(paths: list[str], files_per_year: int = 3) -> list[str]:
    ""
    by_year = defaultdict(list)
    for p in paths:
        by_year[extract_year(p)].append(p)
    sampled = []
    for year, year_paths in sorted(by_year.items()):
        sampled.extend(year_paths[:files_per_year])
    return sampled

def profile_csv_dataset(dataset_name: str, paths: list[str], files_per_year: int) -> pd.DataFrame:
    sample_paths = sample_across_years(paths, files_per_year)
    print(f"  Sampling {len(sample_paths)} files across years for '{dataset_name}'...")

    dfs = []
    for p in sample_paths:
        try:
            data = bucket.blob(p).download_as_bytes()
            df = pd.read_csv(io.BytesIO(data))
            dfs.append(df)
        except Exception as e:
            print(f"    Skipped {p}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    rows = []
    for col in combined.columns:
        series = combined[col]
        pct_null = round(series.isna().mean() * 100, 2)
        dtype = str(series.dtype)
        non_null = series.dropna()

        has_unhashable = len(non_null) > 0 and non_null.apply(lambda v: isinstance(v, (list, dict))).any()

        if has_unhashable:
            avg_len = non_null.apply(lambda v: len(v) if isinstance(v, (list, dict)) else None).dropna().mean()
            rng = f"NESTED FIELD (list/dict per cell) -- needs flattening before use. avg length={round(avg_len, 1) if pd.notna(avg_len) else 'N/A'}"
        elif pd.api.types.is_numeric_dtype(series):
            rng = f"min={non_null.min()}, max={non_null.max()}, mean={round(non_null.mean(), 3)}" if len(non_null) else "N/A (all null)"
        else:
            try:
                uniques = non_null.unique()
                if len(uniques) <= 15:
                    rng = f"unique values: {sorted(map(str, uniques))}"
                else:
                    rng = f"{len(uniques)} unique values (too many to list)"
            except TypeError:
                rng = "UNHASHABLE TYPE -- inspect manually"

        example = non_null.iloc[0] if len(non_null) else None
        if isinstance(example, (list, dict)):
            example = str(example)[:200]

        rows.append({
            "source": "fastf1",
            "dataset": dataset_name,
            "column": col,
            "dtype": dtype,
            "pct_null": pct_null,
            "range_or_unique_values": rng,
            "example_value": example,
            "n_files_sampled": len(dfs),
            "n_rows_sampled": len(combined),
        })
    return pd.DataFrame(rows)

def profile_telemetry(prefix: str, files_per_year: int) -> pd.DataFrame:
    ""
    print("Listing telemetry files (this can take a few minutes given file count)...")
    paths = list_files(prefix, (".json",))
    print(f"  Found {len(paths)} telemetry files total.")

    session_groups = defaultdict(list)
    for p in paths:
        parts = p.rstrip("/").split("/")
        session_folder = "/".join(parts[:-2])
        session_groups[session_folder].append(p)

    sessions_by_year = defaultdict(list)
    for session_folder in session_groups:
        year = extract_year(session_folder)
        sessions_by_year[year].append(session_folder)

    sample_files = []
    for year, sessions in sorted(sessions_by_year.items()):
        for session_folder in sessions[:files_per_year]:

            sample_files.extend(session_groups[session_folder][:2])

    print(f"  Sampling {len(sample_files)} driver-files across years for telemetry schema...")

    records = []
    for p in sample_files:
        try:
            data = bucket.blob(p).download_as_bytes()
            obj = json.loads(data)
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict):
                records.append(obj)
        except Exception as e:
            print(f"    Skipped {p}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)
    rows = []
    for col in df.columns:
        series = df[col]
        pct_null = round(series.isna().mean() * 100, 2)
        dtype = str(series.dtype)
        non_null = series.dropna()

        has_unhashable = len(non_null) > 0 and non_null.apply(lambda v: isinstance(v, (list, dict))).any()

        if has_unhashable:

            avg_len = non_null.apply(lambda v: len(v) if isinstance(v, (list, dict)) else None).dropna().mean()
            rng = f"NESTED FIELD (list/dict per cell) -- needs flattening before use. avg length={round(avg_len, 1) if pd.notna(avg_len) else 'N/A'}"
        elif pd.api.types.is_numeric_dtype(series):
            rng = f"min={non_null.min()}, max={non_null.max()}, mean={round(non_null.mean(), 3)}" if len(non_null) else "N/A (all null)"
        else:
            try:
                uniques = non_null.unique()
                if len(uniques) <= 15:
                    rng = f"unique values: {sorted(map(str, uniques))}"
                else:
                    rng = f"{len(uniques)} unique values (too many to list)"
            except TypeError:
                rng = "UNHASHABLE TYPE -- inspect manually"

        example = non_null.iloc[0] if len(non_null) else None
        if isinstance(example, (list, dict)):
            example = str(example)[:200]

        rows.append({
            "source": "tracinginsights",
            "dataset": "telemetry",
            "column": col,
            "dtype": dtype,
            "pct_null": pct_null,
            "range_or_unique_values": rng,
            "example_value": example,
            "n_files_sampled": len(sample_files),
            "n_rows_sampled": len(df),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=BUCKET_NAME)
    parser.add_argument("--source", choices=["fastf1", "telemetry", "both"], default="both")
    parser.add_argument("--files-per-year", type=int, default=3, help="Files/sessions sampled per year")
    parser.add_argument("--out", default="data_dictionary.csv")
    args = parser.parse_args()

    all_results = []

    if args.source in ("fastf1", "both"):
        print("Listing fastf1 CSV files...")
        csv_paths = list_files("raw/fastf1/", (".csv",))
        by_table = defaultdict(list)
        for p in csv_paths:
            table = p.rstrip("/").split("/")[-1][:-4]
            by_table[table].append(p)

        for table, paths in sorted(by_table.items()):
            print(f"Profiling '{table}'...")
            all_results.append(profile_csv_dataset(table, paths, args.files_per_year))

    if args.source in ("telemetry", "both"):
        print("Profiling telemetry...")
        all_results.append(profile_telemetry("raw/tracinginsights/", args.files_per_year))

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(args.out, index=False)
    print(f"\nDone. Data dictionary written to {args.out} ({len(final)} column entries)")
    print(final[["source", "dataset", "column", "dtype", "pct_null"]].to_string(index=False))
