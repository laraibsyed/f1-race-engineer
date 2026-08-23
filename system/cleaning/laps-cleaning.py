"""
clean_laps.py

Step: Handle missing laps + outlier lap times.
Flags (does NOT delete) pit-in/out laps, SC/VSC laps, and statistical
outlier lap times, so downstream modelling can choose to include/exclude
them explicitly rather than losing the data silently.

Includes a built-in local disk cache (CachedBucket) so re-running this
script -- or any later script that reuses CachedBucket -- doesn't
re-download raw files from GCS every time. Cleaned output is written
through to GCS AND the local cache, so downstream steps can read the
cleaned files straight off disk too.

Requirements:
    pip install google-cloud-storage pandas gcsfs python-dotenv --break-system-packages

Usage:
    python clean_laps.py --year 2024          # one year
    python clean_laps.py --year all           # everything, writes per-file
"""

import argparse
import io
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = Path(os.getenv("GCS_CACHE_DIR", "./gcs_cache"))

# FastF1 TrackStatus codes (per-lap string, can contain several codes
# concatenated if status changed mid-lap) -- source: FastF1 docs / laps.track_status
SC_CODE = "4"
VSC_CODES = {"6", "7"}

# Rolling stint-relative bounds. A lap more than this factor slower than
# the median of "clean" laps in the same stint is flagged as an outlier.
# (Loose enough that genuine slow-but-valid laps -- e.g. traffic -- aren't
# wrongly binned with problems, but tight enough to catch scoring glitches.)
OUTLIER_FACTOR = 1.5


# --------------------------------------------------------------------------
# Local disk cache wrapper around GCS. Reads check disk first and only hit
# GCS on a miss; writes go to GCS AND get mirrored locally (write-through),
# so a file this script just produced can be read back without a round trip.
# --------------------------------------------------------------------------
class CachedBucket:
    def __init__(self, bucket_name: str, cache_dir: Path = CACHE_DIR):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _local_path(self, blob_path: str) -> Path:
        return self.cache_dir / blob_path

    def download_as_bytes(self, blob_path: str) -> bytes:
        local_path = self._local_path(blob_path)
        if local_path.exists():
            self._hits += 1
            return local_path.read_bytes()

        self._misses += 1
        data = self.bucket.blob(blob_path).download_as_bytes()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return data

    def list_blobs(self, prefix: str = ""):
        # Listing is cheap and changes as new data lands, so this always
        # hits GCS live -- only the actual file bytes get cached.
        return self.client.list_blobs(self.bucket.name, prefix=prefix)

    def upload_from_string(self, blob_path: str, content: str, content_type: str = "text/csv"):
        self.bucket.blob(blob_path).upload_from_string(content, content_type=content_type)
        local_path = self._local_path(blob_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content)

    def stats(self):
        total = self._hits + self._misses
        pct_cached = round(self._hits / total * 100, 1) if total else 0
        print(f"Cache: {self._hits} hits, {self._misses} misses ({pct_cached}% served from disk)")


bucket = CachedBucket(BUCKET_NAME)


# --------------------------------------------------------------------------
# Cleaning logic
# --------------------------------------------------------------------------
def list_laps_files(year: str) -> list[str]:
    prefix = f"raw/fastf1/{year}/" if year != "all" else "raw/fastf1/"
    return [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith("/laps.csv")]


def read_csv(path: str) -> pd.DataFrame:
    data = bucket.download_as_bytes(path)
    # TrackStatus MUST be read as string -- see investigate_fastf1_issues.py
    # finding (int parsing mangles multi-code laps into huge nonsense values).
    return pd.read_csv(io.BytesIO(data), dtype={"TrackStatus": str})


def flag_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    df["is_pit_in"] = df["PitInTime"].notna() if "PitInTime" in df.columns else False
    df["is_pit_out"] = df["PitOutTime"].notna() if "PitOutTime" in df.columns else False
    return df


def flag_sc_vsc_laps(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["TrackStatus"].fillna("")
    df["is_sc_lap"] = ts.apply(lambda s: SC_CODE in s)
    df["is_vsc_lap"] = ts.apply(lambda s: any(c in s for c in VSC_CODES))
    return df


def flag_missing_laptime(df: pd.DataFrame) -> pd.DataFrame:
    df["is_missing_laptime"] = df["LapTime"].isna() if "LapTime" in df.columns else True
    return df


def flag_out_in_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags out-laps and in-laps by STINT POSITION, not just PitOutTime/
    PitInTime presence. This catches the Qualifying pattern where a driver
    leaves the garage, does a slow prep lap, a flying lap, then a slow
    lap back -- without a full pit-stop sequence being recorded every time.

    is_out_lap: first lap of a (Driver, Stint) group, OR has PitOutTime.
    is_in_lap:  last lap of a (Driver, Stint) group, OR has PitInTime.
    """
    df["is_out_lap"] = df["is_pit_out"].copy()
    df["is_in_lap"] = df["is_pit_in"].copy()

    if "Driver" not in df.columns or "Stint" not in df.columns:
        return df  # can't determine stint boundaries, fall back to pit flags only

    grouped = df.groupby(["Driver", "Stint"]).groups
    for _, idx in grouped.items():
        df.loc[idx[0], "is_out_lap"] = True
        df.loc[idx[-1], "is_in_lap"] = True

    return df


def flag_outlier_laptimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags a lap as an outlier if it's much slower than the median of
    'clean' laps (no pit in/out, no out/in-lap, no SC/VSC, not missing)
    within the same driver + stint. Baseline is computed from clean laps
    only, and out/in laps are excluded from being flagged here too --
    their slowness is already explained, so double-tagging them as
    statistical outliers would just be noise.
    """
    df["is_outlier_laptime"] = False

    if "LapTime" not in df.columns or "Driver" not in df.columns or "Stint" not in df.columns:
        return df  # can't compute without these, leave unflagged

    lap_seconds = pd.to_timedelta(df["LapTime"], errors="coerce").dt.total_seconds()
    clean_mask = (
        ~df["is_pit_in"] & ~df["is_pit_out"] & ~df["is_out_lap"] & ~df["is_in_lap"]
        & ~df["is_sc_lap"] & ~df["is_vsc_lap"] & lap_seconds.notna()
    )

    not_out_in = ~df["is_out_lap"] & ~df["is_in_lap"]
    for (driver, stint), group_idx in df[clean_mask].groupby(["Driver", "Stint"]).groups.items():
        stint_median = lap_seconds.loc[group_idx].median()
        if pd.isna(stint_median):
            continue
        # only laps that aren't already explained as out/in laps are eligible
        # to be flagged -- otherwise a slow in-lap gets double-counted as an
        # "unexplained" outlier on top of being a known in-lap.
        eligible_idx = df[(df["Driver"] == driver) & (df["Stint"] == stint) & not_out_in].index
        too_slow = lap_seconds.loc[eligible_idx] > (stint_median * OUTLIER_FACTOR)
        df.loc[eligible_idx[too_slow.fillna(False)], "is_outlier_laptime"] = True

    return df


def clean_one_file(path: str) -> pd.DataFrame:
    df = read_csv(path)
    df = flag_pit_laps(df)
    df = flag_sc_vsc_laps(df)
    df = flag_missing_laptime(df)
    df = flag_out_in_laps(df)
    df = flag_outlier_laptimes(df)
    df["source_file"] = path
    return df


def write_back(df: pd.DataFrame, path: str):
    out_path = path.replace("raw/fastf1/", "clean/fastf1/").replace("laps.csv", "laps_flagged.csv")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    bucket.upload_from_string(out_path, buf.getvalue(), content_type="text/csv")
    print(f"  Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="all", help="e.g. 2024, or 'all'")
    args = parser.parse_args()

    paths = list_laps_files(args.year)
    print(f"Found {len(paths)} laps.csv files for year={args.year}")

    summary = []
    for p in paths:
        print(f"Cleaning {p} ...")
        df = clean_one_file(p)
        write_back(df, p)
        summary.append({
            "file": p,
            "n_laps": len(df),
            "pct_pit_in": round(df["is_pit_in"].mean() * 100, 1),
            "pct_pit_out": round(df["is_pit_out"].mean() * 100, 1),
            "pct_sc": round(df["is_sc_lap"].mean() * 100, 1),
            "pct_vsc": round(df["is_vsc_lap"].mean() * 100, 1),
            "pct_out_lap": round(df["is_out_lap"].mean() * 100, 1),
            "pct_in_lap": round(df["is_in_lap"].mean() * 100, 1),
            "pct_missing_laptime": round(df["is_missing_laptime"].mean() * 100, 1),
            "pct_outlier": round(df["is_outlier_laptime"].mean() * 100, 1),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("clean_laps_summary.csv", index=False)
    print(f"\nSummary written to clean_laps_summary.csv ({len(summary_df)} files)")
    print(summary_df.to_string(index=False))
    bucket.stats()