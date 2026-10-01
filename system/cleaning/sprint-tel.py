
import argparse
import io
import os
from pathlib import Path

import fastf1
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
BUCKET_NAME = os.getenv("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = Path(os.getenv("GCS_CACHE_DIR", "./gcs_cache"))
FASTF1_CACHE_DIR = Path("./fastf1_cache")
FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))

KNOWN_GAPS = [
    ("2018", "Australian_Grand_Prix", "R"),
    ("2018", "Bahrain_Grand_Prix", "R"),
    ("2021", "British_Grand_Prix", "S"),
    ("2021", "Italian_Grand_Prix", "S"),
    ("2021", "São_Paulo_Grand_Prix", "S"),
    ("2023", "Austrian_Grand_Prix", "S"),
    ("2023", "Azerbaijan_Grand_Prix", "S"),
    ("2023", "Belgian_Grand_Prix", "S"),
    ("2023", "Qatar_Grand_Prix", "S"),
    ("2023", "São_Paulo_Grand_Prix", "S"),
    ("2023", "United_States_Grand_Prix", "S"),
    ("2024", "Austrian_Grand_Prix", "S"),
    ("2024", "Chinese_Grand_Prix", "S"),
    ("2024", "Miami_Grand_Prix", "S"),
    ("2024", "Qatar_Grand_Prix", "S"),
    ("2024", "São_Paulo_Grand_Prix", "S"),
    ("2024", "United_States_Grand_Prix", "S"),
    ("2025", "Belgian_Grand_Prix", "S"),
    ("2025", "Miami_Grand_Prix", "S"),
    ("2025", "Qatar_Grand_Prix", "S"),
    ("2025", "São_Paulo_Grand_Prix", "S"),
    ("2025", "United_States_Grand_Prix", "S"),
    ("2026", "Canadian_Grand_Prix", "S"),
    ("2026", "Chinese_Grand_Prix", "S"),
    ("2026", "Miami_Grand_Prix", "S"),
]

class CachedBucket:
    ""

    def __init__(self, bucket_name: str, cache_dir: Path = CACHE_DIR):
        print(f"[init] Connecting to GCS bucket '{bucket_name}' ...")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] Connected.")

    def _local_path(self, blob_path: str) -> Path:
        return self.cache_dir / blob_path

    def upload_from_string(self, blob_path: str, content: str, content_type: str = "text/csv"):
        self.bucket.blob(blob_path).upload_from_string(content, content_type=content_type)
        local_path = self._local_path(blob_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")

    def exists(self, blob_path: str) -> bool:
        if self._local_path(blob_path).exists():
            return True
        return self.bucket.blob(blob_path).exists(timeout=30)

bucket = CachedBucket(BUCKET_NAME)

def backfill_session(year: str, race: str, session: str, force: bool = False) -> bool:
    out_path = f"clean/tracinginsights/{year}/{race}/{session}/telemetry_by_lap.csv"
    if not force and bucket.exists(out_path):
        print(f"[skip] {year}/{race}/{session} -- already exists, use --force to redo.")
        return True

    race_name = race.replace("_", " ")
    print(f"[fastf1] Loading {year} {race_name} {session} ...")
    try:
        sess = fastf1.get_session(int(year), race_name, session)
        sess.load(telemetry=True, weather=False, messages=False)
    except Exception as e:
        print(f"    [error] Could not load session: {type(e).__name__}: {e}")
        return False

    laps = sess.laps
    if laps is None or laps.empty:
        print(f"    [error] No laps returned by FastF1 for this session.")
        return False

    drivers = laps["Driver"].dropna().unique()
    print(f"    Found {len(laps)} laps across {len(drivers)} drivers, pulling car telemetry per lap ...")

    rows = []
    n_no_telemetry = 0
    for driver_code in drivers:
        driver_laps = laps.pick_driver(driver_code)
        for _, lap in driver_laps.iterrows():
            try:
                car_data = lap.get_car_data()
            except Exception:
                n_no_telemetry += 1
                continue
            if car_data is None or car_data.empty:
                n_no_telemetry += 1
                continue

            row = {
                "LapNumber": lap["LapNumber"],
                "n_samples": len(car_data),
                "speed_mean": car_data["Speed"].mean() if "Speed" in car_data else None,
                "speed_max": car_data["Speed"].max() if "Speed" in car_data else None,
                "rpm_mean": car_data["RPM"].mean() if "RPM" in car_data else None,
                "rpm_max": car_data["RPM"].max() if "RPM" in car_data else None,
                "throttle_mean": car_data["Throttle"].mean() if "Throttle" in car_data else None,
                "brake_mean": car_data["Brake"].mean() if "Brake" in car_data else None,
                "drs_mean": car_data["DRS"].mean() if "DRS" in car_data else None,
                "gear_mean": car_data["nGear"].mean() if "nGear" in car_data else None,
                "year": year,
                "race": race,
                "session": session,
                "driver_code": driver_code,
            }
            rows.append(row)

    print(f"    Aggregated {len(rows)} driver-laps ({n_no_telemetry} laps had no telemetry available).")
    if not rows:
        print(f"    [error] Nothing to write for {year}/{race}/{session}.")
        return False

    combined = pd.DataFrame(rows)
    print(f"    [sanity] laps has {laps['LapNumber'].nunique()} unique lap numbers; "
          f"aggregated telemetry covers {combined['LapNumber'].nunique()} unique lap numbers.")

    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    bucket.upload_from_string(out_path, buf.getvalue(), content_type="text/csv")
    print(f"    Wrote {out_path} ({len(combined)} driver-lap rows)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="Only backfill known gaps for this year")
    parser.add_argument("--force", action="store_true", help="Redo even if output already exists")
    args = parser.parse_args()

    targets = [g for g in KNOWN_GAPS if args.year is None or g[0] == args.year]
    print(f"Backfilling {len(targets)} known-gap session(s) via FastF1 ...\n")

    succeeded, failed = [], []
    for year, race, session in targets:
        ok = backfill_session(year, race, session, force=args.force)
        (succeeded if ok else failed).append(f"{year}/{race}/{session}")
        print()

    print("=" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed sessions (re-run this script to retry -- only these will be attempted, "
              "successful ones are skipped automatically):")
        for f in failed:
            print(f"    {f}")
