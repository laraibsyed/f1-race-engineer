import os
import subprocess
import time
from google.cloud import storage
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- CONFIG ---
BUCKET_NAME = "f1-race-engineer-bucket"
CLONE_DIR = "data\\tracinginsights"  # local temp folder
YEARS = range(2018, 2027)

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

failed_files = []

def blob_exists(gcs_path):
    return bucket.blob(gcs_path).exists()

def upload_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path, content_type="application/json")
    print(f"  Uploaded: {gcs_path}")

def clone_repo(year, dest):
    url = f"https://github.com/TracingInsights/{year}.git"
    print(f"  Cloning {url} into {dest}...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(result.stderr)
    print(f"  Cloned successfully.")

def upload_year(year, repo_dir):
    tel_files = list(Path(repo_dir).rglob("*_tel.json"))
    print(f"  Found {len(tel_files)} telemetry files")

    for local_path in tel_files:
        # Build GCS path preserving folder structure
        relative = local_path.relative_to(repo_dir)
        gcs_path = f"raw/tracinginsights/{year}/{relative.as_posix()}"

        if blob_exists(gcs_path):
            print(f"  Already exists, skipping: {gcs_path}")
            continue

        try:
            upload_to_gcs(str(local_path), gcs_path)
        except Exception as e:
            print(f"  FAILED {gcs_path}: {e}")
            failed_files.append((year, str(relative), str(e)))

# --- MAIN ---
os.makedirs(CLONE_DIR, exist_ok=True)

for year in YEARS:
    print(f"\n{'='*50}")
    print(f"  YEAR: {year}")
    print(f"{'='*50}")

    repo_dir = Path(CLONE_DIR) / str(year)

    # Clone if not already cloned
    if repo_dir.exists():
        print(f"  Repo already cloned at {repo_dir}, skipping clone.")
    else:
        try:
            clone_repo(year, repo_dir)
        except Exception as e:
            print(f"  Could not clone {year}: {e}")
            continue

    # Upload all telemetry files
    upload_year(year, repo_dir)

    print(f"  Year {year} done.")

# --- SUMMARY ---
print(f"\n{'='*50}")
if failed_files:
    print(f"FAILED FILES ({len(failed_files)}):")
    for f in failed_files:
        print(f"  {f}")
else:
    print("All files downloaded successfully!")