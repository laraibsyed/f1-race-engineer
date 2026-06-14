import fastf1
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv
import re
import io
import time

load_dotenv()

# --- CONFIG ---
BUCKET_NAME = "f1-race-engineer-bucket"
CACHE_DIR = "data\\raw"

# --- SPECIFY WHAT TO DOWNLOAD ---
YEAR = 2021
ROUND = 19  # round number, or use event name string e.g. "Monaco Grand Prix"
SESSIONS = ["FP1", "SQ", "S", "Q", "R"]

PAUSE_BETWEEN_SESSIONS = 1

failed_sessions = []

fastf1.Cache.enable_cache(CACHE_DIR)
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def sanitise_name(name):
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w]", "", name)
    return name

def upload_df_to_gcs(df, gcs_path):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    print(f"  Uploaded: {gcs_path}")

def blob_exists(gcs_path):
    return bucket.blob(gcs_path).exists()

def download_session(year, round_num, session_name):
    session = fastf1.get_session(year, round_num, session_name)

    event_name = sanitise_name(session.event["EventName"])
    base_path = f"raw/fastf1/{year}/{event_name}/{session_name}"

    if blob_exists(f"{base_path}/laps.csv"):
        print(f"  Already exists, skipping: {base_path}")
        return

    session.load(
        telemetry=False,
        weather=True,
        laps=True,
        messages=True
    )

    laps_df     = session.laps
    weather_df  = session.weather_data
    messages_df = session.race_control_messages
    results_df  = session.results

    if not laps_df.empty:
        upload_df_to_gcs(laps_df,     f"{base_path}/laps.csv")
    if not weather_df.empty:
        upload_df_to_gcs(weather_df,  f"{base_path}/weather.csv")
    if not messages_df.empty:
        upload_df_to_gcs(messages_df, f"{base_path}/messages.csv")
    if not results_df.empty:
        upload_df_to_gcs(results_df,  f"{base_path}/results.csv")

# --- MAIN ---
print(f"\n{'='*50}")
print(f"  Downloading: {YEAR} Round {ROUND}")
print(f"{'='*50}")

for s in SESSIONS:
    print(f"\n  -> {YEAR} R{ROUND} {s}")
    try:
        download_session(YEAR, ROUND, s)
    except Exception as e:
        print(f"  FAILED {YEAR} R{ROUND} {s}: {e}")
        failed_sessions.append((YEAR, ROUND, s, str(e)))

    print(f"  Pausing {PAUSE_BETWEEN_SESSIONS}s...")
    time.sleep(PAUSE_BETWEEN_SESSIONS)

# --- SUMMARY ---
print(f"\n{'='*50}")
if failed_sessions:
    print("FAILED SESSIONS:")
    for f in failed_sessions:
        print(f"  {f}")
else:
    print("All sessions downloaded successfully!")