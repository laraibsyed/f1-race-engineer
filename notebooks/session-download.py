import fastf1
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv
import re
import io
import time

load_dotenv()

BUCKET_NAME = "f1-race-engineer-bucket"
CACHE_DIR = "data\\raw"

PAUSE_BETWEEN_SESSIONS = 10

failed_sessions = []

fastf1.Cache.enable_cache(CACHE_DIR)
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

TO_REDOWNLOAD = [

    (2019, 17, "FP3"),
    (2021, 15, "FP3"),

    (2020,  2, "FP3"),
    (2020, 11, "FP1"),
    (2020, 11, "FP2"),
]

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

def download_session(year, round_num, session_name):
    session = fastf1.get_session(year, round_num, session_name)
    event_name = sanitise_name(session.event["EventName"])
    base_path = f"raw/fastf1/{year}/{event_name}/{session_name}"

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

print(f"Redownloading {len(TO_REDOWNLOAD)} sessions...\n")

for year, round_num, session_name in TO_REDOWNLOAD:
    print(f"  -> {year} R{round_num:02d} {session_name}")
    try:
        download_session(year, round_num, session_name)
    except Exception as e:
        print(f"  FAILED {year} R{round_num} {session_name}: {e}")
        failed_sessions.append((year, round_num, session_name, str(e)))

    print(f"  Pausing {PAUSE_BETWEEN_SESSIONS}s...")
    time.sleep(PAUSE_BETWEEN_SESSIONS)

print(f"\n{'='*50}")
if failed_sessions:
    print("FAILED SESSIONS:")
    for f in failed_sessions:
        print(f"  {f}")
else:
    print("All sessions redownloaded successfully!")
