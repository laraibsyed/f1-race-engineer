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
YEARS = range(2025, 2027)
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]

PAUSE_BETWEEN_SESSIONS = 1
PAUSE_BETWEEN_ROUNDS = 10
PAUSE_BETWEEN_YEARS = 60

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

    # Build path before loading — get_session() already has event metadata
    event_name = sanitise_name(session.event["EventName"])
    base_path = f"raw/fastf1/{year}/{event_name}/{session_name}"

    # Check GCS FIRST — skip download entirely if already there
    if blob_exists(f"{base_path}/laps.csv"):
        print(f"  Already exists, skipping: {base_path}")
        return

    # Only load if we actually need to
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

# --- MAIN LOOP ---
for year in YEARS:
    print(f"\n{'='*50}")
    print(f"  YEAR: {year}")
    print(f"{'='*50}")

    try:
        schedule = fastf1.get_event_schedule(year)
        rounds = schedule[schedule['EventFormat'] != 'testing']['RoundNumber'].tolist()
    except Exception as e:
        print(f"  Could not get schedule for {year}: {e}")
        continue

    for round_num in rounds:
        print(f"\n  Round {round_num}")
        for s in SESSIONS:
            print(f"    -> {year} R{round_num} {s}")
            try:
                download_session(year, round_num, s)
            except Exception as e:
                print(f"    FAILED {year} R{round_num} {s}: {e}")
                failed_sessions.append((year, round_num, s, str(e)))

            print(f"    Pausing {PAUSE_BETWEEN_SESSIONS}s...")
            time.sleep(PAUSE_BETWEEN_SESSIONS)

        print(f"  Round done. Pausing {PAUSE_BETWEEN_ROUNDS}s before next round...")
        time.sleep(PAUSE_BETWEEN_ROUNDS)

    print(f"Year {year} done. Pausing {PAUSE_BETWEEN_YEARS}s before next year...")
    time.sleep(PAUSE_BETWEEN_YEARS)

# --- SUMMARY ---
print(f"\n{'='*50}")
if failed_sessions:
    print("FAILED SESSIONS:")
    for f in failed_sessions:
        print(f"  {f}")
else:
    print("All sessions downloaded successfully!")