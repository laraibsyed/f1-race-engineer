"""
align_telemetry.py

Step: Align telemetry samples to lap-level granularity.

Takes raw tracinginsights telemetry and the corresponding
clean/fastf1/.../laps_flagged.csv (from clean_laps.py), and produces ONE
ROW PER DRIVER PER LAP with aggregated telemetry stats -- ready to join
against laps_flagged.csv for feature engineering (tyre_age, degradation
rate, etc. all need lap-level numbers, not raw timestamp streams).

How the alignment actually works (confirmed via debug run, 2024 Abu Dhabi
Race -- this contradicted the original assumption, see below):
    Each tracinginsights JSON file is NOT a whole driver-session stream.
    It's already exactly ONE DRIVER'S ONE LAP. Evidence: 'time' resets to
    ~0.0 at the start of every file (matches a single lap duration, not a
    ~90min session); file count for a session matches the session's total
    lap count almost exactly (1035 files == 1035 laps_flagged.csv rows for
    2024 Abu Dhabi Race); and the 'dataKey' scalar field inside each file
    (e.g. "2024-Abu Dhabi Grand Prix-Race-ALB-10") ends in the SAME number
    as the filename ("10_tel.json") -- and that number is the LAP NUMBER,
    not the driver number (ALB's real car number is 23).

    So there's no time-matching/merge_asof needed at all -- each file
    already tells you its own lap number via the filename. We just parse
    it and aggregate that file's samples directly. laps_flagged.csv is
    still loaded, but only for a sanity check (does the lap count line up),
    not for the join itself.

Requirements:
    pip install google-cloud-storage pandas gcsfs python-dotenv --break-system-packages

Usage:
    # One session
    python align_telemetry.py --session "2024/Abu_Dhabi_Grand_Prix/R"

    # Every session in a year
    python align_telemetry.py --year 2024

    # Everything (slow -- this is the full 2018-2025 span)
    python align_telemetry.py --year all
"""

import argparse
import io
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = Path(os.getenv("GCS_CACHE_DIR", "./gcs_cache"))

TELEMETRY_PREFIX = "raw/tracinginsights/"
RAW_LAPS_PREFIX = "clean/fastf1/"  # read the already-cleaned laps, not raw

# tracinginsights only stores telemetry for Race (and Sprint, on sprint
# weekends) -- no FP1/FP2/FP3/Qualifying telemetry exists at all. Sessions
# not in this set are skipped outright in bulk mode rather than wasting a
# GCS list call and printing a confusing "0 files found" for every one.
TELEMETRY_AVAILABLE_SESSIONS = {"R", "S"}

SESSION_CODE_MAP = {
    "R": "Race",
    "S": "Sprint",
}


def to_telemetry_race_name(fastf1_race: str) -> str:
    return fastf1_race.replace("_", " ")


def to_telemetry_session_name(fastf1_session: str) -> str:
    return SESSION_CODE_MAP.get(fastf1_session, fastf1_session)


# --------------------------------------------------------------------------
# Local disk cache wrapper around GCS (same pattern as clean_laps.py, so
# anything already downloaded there -- e.g. laps_flagged.csv -- is reused
# here with zero extra GCS calls).
# --------------------------------------------------------------------------
class CachedBucket:
    def __init__(self, bucket_name: str, cache_dir: Path = CACHE_DIR):
        print(f"[init] Connecting to GCS bucket '{bucket_name}' ...")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()  # protects the counters below across threads
        print(f"[init] Connected. Local cache dir: {self.cache_dir.resolve()}")

    def _local_path(self, blob_path: str) -> Path:
        return self.cache_dir / blob_path

    def download_as_bytes(self, blob_path: str) -> bytes:
        local_path = self._local_path(blob_path)
        if local_path.exists():
            with self._lock:
                self._hits += 1
            return local_path.read_bytes()
        with self._lock:
            self._misses += 1
        data = self.bucket.blob(blob_path).download_as_bytes()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return data

    def list_blobs(self, prefix: str = ""):
        return self.client.list_blobs(self.bucket.name, prefix=prefix)

    def upload_from_string(self, blob_path: str, content: str, content_type: str = "text/csv"):
        self.bucket.blob(blob_path).upload_from_string(content, content_type=content_type)
        local_path = self._local_path(blob_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")

    def exists(self, blob_path: str) -> bool:
        """Cheap existence check -- local cache first, then GCS. Used to
        skip sessions that already have output from a previous run."""
        if self._local_path(blob_path).exists():
            return True
        return self.bucket.blob(blob_path).exists()

    def stats(self):
        total = self._hits + self._misses
        pct_cached = round(self._hits / total * 100, 1) if total else 0
        print(f"Cache: {self._hits} hits, {self._misses} misses ({pct_cached}% served from disk)")


bucket = CachedBucket(BUCKET_NAME)


def read_csv_robust(data: bytes, path: str) -> pd.DataFrame | None:
    """
    Some laps_flagged.csv files (accented race names like São_Paulo,
    non-ASCII driver/event fields from older seasons) aren't clean UTF-8.
    Try UTF-8 first (correct in the vast majority of cases), fall back to
    latin-1 (never raises -- worst case a character renders slightly off,
    but it won't crash the whole multi-hour run over one file).
    """
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        print(f"    [warn] {path}: not valid UTF-8, retrying with latin-1 encoding.")
        try:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        except Exception as e:
            print(f"    [error] {path}: failed even with latin-1 fallback ({e}), skipping this session.")
            return None


# --------------------------------------------------------------------------
# Telemetry flattening (same logic as flatten_telemetry.py)
# --------------------------------------------------------------------------
def extract_year(path: str) -> str | None:
    m = re.search(r"/((?:19|20)\d{2})/", path)
    return m.group(1) if m else None


def parse_driver_file_path(path: str) -> dict:
    """
    raw/tracinginsights/<year>/<race>/<session>/<driver_code>/<lap_number>_tel.json

    IMPORTANT: the leading number in the filename is the LAP NUMBER, not
    the driver number (confirmed via the 'dataKey' field inside each file,
    e.g. "2024-Abu Dhabi Grand Prix-Race-ALB-10" -- the trailing 10 matches
    the filename, and ALB's real car number is 23). Each file is exactly
    one driver's one lap -- 'time' resets to 0.0 at the start of every
    file -- not a whole-session stream as originally assumed.
    """
    parts = path.rstrip("/").split("/")
    filename = parts[-1]
    lap_number = filename.split("_")[0]
    return {
        "year": parts[-5] if len(parts) >= 5 else None,
        "race": parts[-4] if len(parts) >= 4 else None,
        "session": parts[-3] if len(parts) >= 3 else None,
        "driver_code": parts[-2],
        "lap_number": int(lap_number) if lap_number.isdigit() else None,
    }


def find_fields(obj: dict, prefix: str = "") -> tuple[dict, dict]:
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
    try:
        data = bucket.download_as_bytes(path)
        obj = json.loads(data)
    except Exception as e:
        print(f"    Skipped {path}: {e}")
        return None

    meta = parse_driver_file_path(path)
    if not isinstance(obj, dict):
        return None

    array_fields, scalar_fields = find_fields(obj)
    if not array_fields:
        return None

    lengths = {k: len(v) for k, v in array_fields.items()}
    max_len = max(lengths.values())
    if any(n != max_len for n in lengths.values()):
        min_len = min(lengths.values())
        array_fields = {k: v[:min_len] for k, v in array_fields.items()}

    df = pd.DataFrame(array_fields)
    df.columns = [c.replace("tel.", "") for c in df.columns]

    if "DriverAhead" in df.columns:
        df["DriverAhead"] = df["DriverAhead"].replace("None", pd.NA)
    if "throttle" in df.columns:
        invalid_throttle = df["throttle"] > 100
        if invalid_throttle.any():
            df.loc[invalid_throttle, "throttle"] = pd.NA

    for k, v in scalar_fields.items():
        df[k.replace("tel.", "")] = v
    for k, v in meta.items():
        df[k] = v

    return df


# --------------------------------------------------------------------------
# Lap boundaries
# --------------------------------------------------------------------------
def load_laps_for_session(year: str, race: str, session: str) -> pd.DataFrame | None:
    """Reads clean/fastf1/<year>/<race>/<session>/laps_flagged.csv.
    Used only for a post-hoc sanity check (lap count comparison) now that
    the actual telemetry-to-lap join comes from the filename, not time
    matching -- LapStartTime_sec is computed but no longer load-bearing."""
    path = f"{RAW_LAPS_PREFIX}{year}/{race}/{session}/laps_flagged.csv"
    print(f"[laps] Looking for {path} ...")
    try:
        data = bucket.download_as_bytes(path)
    except Exception as e:
        print(f"    No laps_flagged.csv found at {path} ({type(e).__name__}: {e}) "
              f"-- run clean_laps.py for this session first.")
        return None
    print(f"[laps] Found it ({len(data)} bytes).")

    laps = read_csv_robust(data, path)
    if laps is None:
        return None
    if "LapStartTime" not in laps.columns or "Driver" not in laps.columns or "LapNumber" not in laps.columns:
        print(f"    {path} missing required columns (LapStartTime/Driver/LapNumber), skipping.")
        return None

    laps["LapStartTime_sec"] = pd.to_timedelta(laps["LapStartTime"], errors="coerce").dt.total_seconds()
    # fallback: if that produced all-NaT (e.g. already stored as plain seconds), try numeric parse
    if laps["LapStartTime_sec"].isna().all():
        laps["LapStartTime_sec"] = pd.to_numeric(laps["LapStartTime"], errors="coerce")

    return laps.dropna(subset=["LapStartTime_sec"]).sort_values("LapStartTime_sec")


# --------------------------------------------------------------------------
# Alignment + aggregation
# --------------------------------------------------------------------------
TELEMETRY_AGG = {
    "speed": ["mean", "max"],
    "rpm": ["mean", "max"],
    "throttle": ["mean"],
    "brake": ["mean"],  # fraction of samples with brake applied (0/1 or %) -> mean = time-on-brake proxy
    "drs": ["mean"],
    "gear": ["mean"],
}


def aggregate_one_lap_file(tel_df: pd.DataFrame, meta: dict) -> dict | None:
    """
    Each file is already exactly one driver's one lap -- no time-matching
    needed. Just aggregate this file's samples into one summary row.
    """
    if meta["lap_number"] is None:
        return None

    agg_spec = {col: funcs for col, funcs in TELEMETRY_AGG.items() if col in tel_df.columns}
    if not agg_spec:
        return None

    row = {"LapNumber": meta["lap_number"], "n_samples": len(tel_df)}
    for col, funcs in agg_spec.items():
        for func in funcs:
            row[f"{col}_{func}"] = tel_df[col].agg(func)

    for col in ("year", "race", "session", "driver_code"):
        row[col] = meta.get(col)

    return row


def process_one_file(path: str) -> tuple[str, dict | None, str]:
    """
    Runs in a worker thread: download + flatten + aggregate one file.
    Returns (path, row_or_None, status) where status is one of
    'ok', 'flatten_failed', 'no_lap_number', 'no_agg_columns'.
    """
    tel_df = flatten_one_file(path)
    if tel_df is None:
        return path, None, "flatten_failed"

    meta = parse_driver_file_path(path)
    row = aggregate_one_lap_file(tel_df, meta)
    if row is None:
        status = "no_lap_number" if meta["lap_number"] is None else "no_agg_columns"
        return path, None, status

    return path, row, "ok"


def align_session(year: str, race: str, session: str, ti_race: str | None = None,
                   ti_session: str | None = None, workers: int = 20, force: bool = False):
    if session not in TELEMETRY_AVAILABLE_SESSIONS and ti_session is None:
        print(f"[align] Skipping {year}/{race}/{session} -- no telemetry exists for this session type "
              f"(only Race/Sprint have telemetry).")
        return

    out_path = f"clean/tracinginsights/{year}/{race}/{session}/telemetry_by_lap.csv"
    if not force and bucket.exists(out_path):
        print(f"[align] Skipping {year}/{race}/{session} -- {out_path} already exists "
              f"(use --force to redo it).")
        return

    print(f"[align] Starting session {year}/{race}/{session}")
    laps = load_laps_for_session(year, race, session)
    if laps is None:
        return
    print(f"[align] Loaded {len(laps)} lap rows.")

    ti_race = ti_race or to_telemetry_race_name(race)
    ti_session = ti_session or to_telemetry_session_name(session)
    prefix = f"{TELEMETRY_PREFIX}{year}/{ti_race}/{ti_session}/"
    print(f"[align] Listing telemetry under {prefix} ...")
    driver_files = [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.lower().endswith(".json")]
    print(f"[align] Found {len(driver_files)} telemetry files.")
    if not driver_files:
        print(f"    No telemetry files under {prefix} -- the session/race name guess may be wrong. "
              f"Check the real folder name in your bucket and re-run with --ti-race / --ti-session to override.")
        return

    print(f"  Aligning {len(driver_files)} driver-files for {year}/{race}/{session} "
          f"using {workers} threads ...")

    # First 3 files sequentially with full debug printing (cheap since
    # they're likely already cached from earlier test runs), rest in
    # parallel via a thread pool -- these are network-bound downloads,
    # so threads (not processes) are the right tool: the GIL releases
    # during I/O waits, letting many downloads happen concurrently.
    debug_files = driver_files[:3]
    rest_files = driver_files[3:]

    results = []
    status_counts = defaultdict(int)

    for path in debug_files:
        print(f"  [debug] --- {path} ---")
        tel_df = flatten_one_file(path)
        if tel_df is None:
            status_counts["flatten_failed"] += 1
            continue
        meta = parse_driver_file_path(path)
        print(f"    [debug] parsed meta from filename: {meta}")
        print(f"    [debug] tel_df columns: {list(tel_df.columns)}, {len(tel_df)} samples")
        row = aggregate_one_lap_file(tel_df, meta)
        if row is None:
            status_counts["no_lap_number" if meta["lap_number"] is None else "no_agg_columns"] += 1
            continue
        status_counts["ok"] += 1
        results.append(row)

    if rest_files:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one_file, p): p for p in rest_files}
            n_done = 0
            for future in as_completed(futures):
                _, row, status = future.result()
                status_counts[status] += 1
                if row is not None:
                    results.append(row)
                n_done += 1
                if n_done % 200 == 0:
                    print(f"    ...{n_done}/{len(rest_files)} done")

    print(f"  [summary] flatten_failed={status_counts['flatten_failed']}, "
          f"no_lap_number={status_counts['no_lap_number']}, "
          f"no_agg_columns={status_counts['no_agg_columns']}, succeeded={status_counts['ok']}")

    if not results:
        print(f"    Nothing aligned for {year}/{race}/{session}")
        return

    combined = pd.DataFrame(results)

    # sanity check: does the set of lap numbers we aggregated roughly match
    # what's in laps_flagged.csv? (won't be 1:1 -- laps has ALL drivers'
    # laps, this is filtered to whichever drivers had telemetry files --
    # but a wildly different count is worth knowing about.)
    print(f"  [sanity] laps_flagged.csv has {laps['LapNumber'].nunique()} unique lap numbers; "
          f"aligned telemetry covers {combined['LapNumber'].nunique()} unique lap numbers.")

    out_path = f"clean/tracinginsights/{year}/{race}/{session}/telemetry_by_lap.csv"
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    bucket.upload_from_string(out_path, buf.getvalue(), content_type="text/csv")
    print(f"    Wrote {out_path} ({len(combined)} driver-lap rows)")


def list_sessions(year: str) -> list[tuple[str, str, str]]:
    """Returns (year, race, session) tuples in fastf1-style naming, by
    walking clean/fastf1/ -- i.e. only sessions you've already run
    clean_laps.py on are candidates for alignment."""
    prefix = f"{RAW_LAPS_PREFIX}{year}/" if year != "all" else RAW_LAPS_PREFIX
    paths = [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith("laps_flagged.csv")]
    sessions = set()
    for p in paths:
        parts = p.rstrip("/").split("/")
        if len(parts) >= 4:
            sessions.add((parts[-4], parts[-3], parts[-2]))
    return sorted(sessions)


if __name__ == "__main__":
    print("[main] Script started.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="fastf1-style, e.g. '2024/Abu_Dhabi_Grand_Prix/R'")
    parser.add_argument("--year", help="e.g. 2024, or 'all'")
    parser.add_argument("--ti-race", help="Override the guessed tracinginsights race folder name (e.g. 'Abu Dhabi Grand Prix')")
    parser.add_argument("--ti-session", help="Override the guessed tracinginsights session folder name (e.g. 'Race')")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent download threads (default 20)")
    parser.add_argument("--force", action="store_true", help="Redo sessions even if telemetry_by_lap.csv already exists")
    args = parser.parse_args()
    print(f"[main] Args: session={args.session}, year={args.year}, workers={args.workers}, force={args.force}")

    try:
        if args.session:
            y, r, s = args.session.strip("/").split("/")
            align_session(y, r, s, ti_race=args.ti_race, ti_session=args.ti_session,
                          workers=args.workers, force=args.force)
        elif args.year:
            sessions = list_sessions(args.year)
            print(f"Found {len(sessions)} sessions for year={args.year}")
            n_failed = 0
            failed_sessions = []
            for y, r, s in sessions:
                try:
                    align_session(y, r, s, ti_race=args.ti_race, ti_session=args.ti_session,
                                  workers=args.workers, force=args.force)
                except Exception as e:
                    # One bad session (bad encoding, malformed file, transient
                    # network error, etc.) should NOT take down a multi-hour
                    # run over hundreds of sessions -- log it and move on.
                    n_failed += 1
                    failed_sessions.append(f"{y}/{r}/{s}")
                    print(f"[error] Session {y}/{r}/{s} FAILED, skipping and continuing: "
                          f"{type(e).__name__}: {e}")
            if failed_sessions:
                print(f"\n[main] {n_failed} session(s) failed and were skipped:")
                for fs in failed_sessions:
                    print(f"    {fs}")
                print("Re-run the same --year command to retry only these -- everything else "
                      "will be skipped automatically since their output already exists.")
        else:
            raise SystemExit("Provide --session 'YEAR/Race/Session' or --year YEAR (or --year all)")
    except Exception:
        import traceback
        print("[main] CRASHED -- full traceback below:")
        traceback.print_exc()
    finally:
        bucket.stats()
        print("[main] Script finished.")