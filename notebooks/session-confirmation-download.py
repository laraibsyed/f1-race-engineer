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

YEAR = 2026
SPRINT_ROUNDS = [2, 4, 5, 9]
SESSIONS = ["FP1", "SQ", "S", "Q", "R"]  # 2023 sprint format

PAUSE_BETWEEN_SESSIONS = 1
PAUSE_BETWEEN_ROUNDS = 10

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

    try:
        session.load(
            telemetry=False,
            weather=True,
            laps=True,
            messages=True
        )
    except Exception as e:
        print(f"  Warning: full load failed ({e}), retrying without weather...")
        session = fastf1.get_session(year, round_num, session_name)
        session.load(
            telemetry=False,
            weather=False,
            laps=True,
            messages=True
        )

    try:
        laps_df = session.laps
    except Exception:
        laps_df = pd.DataFrame()

    try:
        weather_df = session.weather_data
    except Exception:
        weather_df = pd.DataFrame()

    try:
        messages_df = session.race_control_messages
    except Exception:
        messages_df = pd.DataFrame()

    try:
        results_df = session.results
    except Exception:
        results_df = pd.DataFrame()

    if not laps_df.empty:
        upload_df_to_gcs(laps_df,     f"{base_path}/laps.csv")
    if not weather_df.empty:
        upload_df_to_gcs(weather_df,  f"{base_path}/weather.csv")
    if not messages_df.empty:
        upload_df_to_gcs(messages_df, f"{base_path}/messages.csv")
    if not results_df.empty:
        upload_df_to_gcs(results_df,  f"{base_path}/results.csv")

# --- MAIN ---
for round_num in SPRINT_ROUNDS:
    print(f"\n{'='*50}")
    print(f"  Downloading: {YEAR} Round {round_num}")
    print(f"{'='*50}")

    for s in SESSIONS:
        print(f"\n  -> {YEAR} R{round_num} {s}")
        try:
            download_session(YEAR, round_num, s)
        except Exception as e:
            print(f"  FAILED {YEAR} R{round_num} {s}: {e}")
            failed_sessions.append((YEAR, round_num, s, str(e)))

        print(f"  Pausing {PAUSE_BETWEEN_SESSIONS}s...")
        time.sleep(PAUSE_BETWEEN_SESSIONS)

    print(f"  Round done. Pausing {PAUSE_BETWEEN_ROUNDS}s...")
    time.sleep(PAUSE_BETWEEN_ROUNDS)

# --- SUMMARY ---
print(f"\n{'='*50}")
if failed_sessions:
    print("FAILED SESSIONS:")
    for f in failed_sessions:
        print(f"  {f}")
else:
    print("All sessions downloaded successfully!")