import argparse
import io
import json
import os
import time
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

DEFAULT_BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")

client = storage.Client()

def list_files(bucket_name: str, prefix: str, exts: tuple[str, ...], limit: int | None = None) -> list[str]:
    ""
    print(f"Listing files in gs://{bucket_name}/{prefix} (extensions: {exts}) ...")
    t0 = time.time()
    paths = []
    seen = 0
    for b in client.list_blobs(bucket_name, prefix=prefix):
        seen += 1
        if seen % 20000 == 0:
            print(f"  ...scanned {seen} blobs, kept {len(paths)} matches so far ({time.time()-t0:.1f}s)")
        if b.name.lower().endswith(exts):
            paths.append(b.name)
        if limit and len(paths) >= limit:
            print(f"  Hit limit {limit}, stopping early.")
            break
    print(f"Done: {len(paths)} matching files found in {time.time()-t0:.1f}s")
    return paths

def group_csv_by_table(paths: list[str]) -> dict[str, list[str]]:
    ""
    groups = defaultdict(list)
    for path in paths:
        filename = path.rstrip("/").split("/")[-1]
        dataset_name = filename[:-4] if filename.lower().endswith(".csv") else filename
        groups[dataset_name].append(path)
    return groups

def sample_and_check_csv_schema(bucket_name: str, paths: list[str], sample_size: int = 5) -> dict:
    bucket = client.bucket(bucket_name)
    if len(paths) <= sample_size:
        sample_paths = paths
    else:
        step = len(paths) // sample_size
        sample_paths = [paths[i] for i in range(0, len(paths), step)][:sample_size]

    schemas = {}
    for p in sample_paths:
        data = bucket.blob(p).download_as_bytes()
        df = pd.read_csv(io.BytesIO(data), nrows=5)
        schemas[p] = tuple(df.columns)

    unique_schemas = set(schemas.values())
    consistent = len(unique_schemas) == 1
    return {
        "columns": sorted(unique_schemas)[0] if consistent else None,
        "schema_consistent": consistent,
        "n_distinct_schemas": len(unique_schemas),
    }

def estimate_csv_row_count(bucket_name: str, path: str, exact: bool = False) -> int:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.reload()
    if exact:
        data = blob.download_as_bytes()
        return len(pd.read_csv(io.BytesIO(data)))
    sample = blob.download_as_bytes(start=0, end=20000)
    sample_df = pd.read_csv(io.BytesIO(sample), on_bad_lines="skip")
    if len(sample_df) == 0:
        return 0
    avg_row_bytes = len(sample) / len(sample_df)
    return int(blob.size / avg_row_bytes) if avg_row_bytes else 0

def build_fastf1_inventory(bucket_name: str, prefix: str, exact_counts: bool, limit: int | None) -> pd.DataFrame:
    paths = list_files(bucket_name, prefix, (".csv",), limit)
    groups = group_csv_by_table(paths)
    print(f"\nfastf1 CSV tables found: {sorted(groups.keys())}")

    rows = []
    for dataset_name, file_paths in sorted(groups.items()):
        print(f"Processing '{dataset_name}' ({len(file_paths)} files)...")
        schema_info = sample_and_check_csv_schema(bucket_name, file_paths)
        count_sample = file_paths[:10]
        total_rows_sample = sum(estimate_csv_row_count(bucket_name, p, exact_counts) for p in count_sample)
        avg_rows = total_rows_sample / len(count_sample) if count_sample else 0
        rows.append({
            "source": "fastf1",
            "dataset_name": dataset_name,
            "file_count": len(file_paths),
            "session_count": len(file_paths),
            "example_path": file_paths[0],
            "schema_consistent_across_sample": schema_info["schema_consistent"],
            "n_distinct_schemas_in_sample": schema_info["n_distinct_schemas"],
            "columns": ", ".join(schema_info["columns"]) if schema_info["columns"] else "MIXED - inspect manually",
            "avg_rows_per_file": round(avg_rows),
            "est_total_rows": int(avg_rows * len(file_paths)),
            "row_count_method": "exact" if exact_counts else "estimated",
        })
    return pd.DataFrame(rows)

def group_json_by_session(paths: list[str]) -> dict[str, list[str]]:
    ""
    groups = defaultdict(list)
    for path in paths:
        parts = path.rstrip("/").split("/")
        session_folder = "/".join(parts[:-2])
        groups[session_folder].append(path)
    return groups

def sample_and_check_json_schema(bucket_name: str, paths: list[str], sample_size: int = 3) -> dict:
    ""
    bucket = client.bucket(bucket_name)
    sample_paths = paths[:sample_size] if len(paths) >= sample_size else paths

    schemas = {}
    for p in sample_paths:
        data = bucket.blob(p).download_as_bytes()
        obj = json.loads(data)
        if isinstance(obj, dict):
            keys = tuple(sorted(obj.keys()))
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            keys = tuple(sorted(obj[0].keys()))
        else:
            keys = ("UNRECOGNISED_JSON_SHAPE",)
        schemas[p] = keys

    unique_schemas = set(schemas.values())
    consistent = len(unique_schemas) == 1
    return {
        "columns": sorted(unique_schemas)[0] if consistent else None,
        "schema_consistent": consistent,
        "n_distinct_schemas": len(unique_schemas),
    }

def build_telemetry_inventory(bucket_name: str, prefix: str, limit: int | None) -> pd.DataFrame:
    paths = list_files(bucket_name, prefix, (".json",), limit)
    session_groups = group_json_by_session(paths)
    print(f"\ntracinginsights telemetry: {len(paths)} driver-files across {len(session_groups)} sessions")

    all_paths_sorted = sorted(paths)
    sample_size = min(8, len(all_paths_sorted))
    step = max(1, len(all_paths_sorted) // sample_size) if sample_size else 1
    cross_session_sample = [all_paths_sorted[i] for i in range(0, len(all_paths_sorted), step)][:sample_size]
    schema_info = sample_and_check_json_schema(bucket_name, cross_session_sample)

    files_per_session = [len(v) for v in session_groups.values()]
    avg_files_per_session = sum(files_per_session) / len(files_per_session) if files_per_session else 0

    row = {
        "source": "tracinginsights",
        "dataset_name": "telemetry",
        "file_count": len(paths),
        "session_count": len(session_groups),
        "example_path": paths[0] if paths else None,
        "schema_consistent_across_sample": schema_info["schema_consistent"],
        "n_distinct_schemas_in_sample": schema_info["n_distinct_schemas"],
        "columns": ", ".join(schema_info["columns"]) if schema_info["columns"] else "MIXED - inspect manually",
        "avg_rows_per_file": None,
        "est_total_rows": None,
        "row_count_method": "not computed (JSON, per-driver files) -- see avg_files_per_session instead",
        "avg_files_per_session": round(avg_files_per_session, 1),
    }
    return pd.DataFrame([row])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory FastF1 + tracinginsights data stored in GCS.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_NAME)
    parser.add_argument("--fastf1-prefix", default="raw/fastf1/", help="Prefix for fastf1 CSV tables")
    parser.add_argument("--telemetry-prefix", default="raw/tracinginsights/", help="Prefix for tracinginsights JSON telemetry")
    parser.add_argument("--exact-counts", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Cap files scanned per source, for quick tests")
    parser.add_argument("--out", default="inventory.csv")
    args = parser.parse_args()

    fastf1_df = build_fastf1_inventory(args.bucket, args.fastf1_prefix, args.exact_counts, args.limit)
    telemetry_df = build_telemetry_inventory(args.bucket, args.telemetry_prefix, args.limit)

    combined = pd.concat([fastf1_df, telemetry_df], ignore_index=True)
    combined.to_csv(args.out, index=False)

    print(f"\nDone. Inventory written to {args.out}\n")
    print(combined[["source", "dataset_name", "file_count", "session_count", "schema_consistent_across_sample"]].to_string(index=False))
