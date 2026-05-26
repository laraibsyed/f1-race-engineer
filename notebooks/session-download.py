import fastf1
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv
import re
import io

load_dotenv()

# --- CONFIG ---
BUCKET_NAME = "f1-race-engineer-bucket"
CACHE_DIR = "data\\raw"
YEAR = 2020
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]
failed_sessions = []

fastf1.Cache.enable_cache(CACHE_DIR)
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def sanitise_name(name):
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w]", "", name)
    return name

def get_telemetry_df(session):
    all_telemetry = []
    for _, lap in session.laps.iterlaps():
        try:
            telemetry = lap.get_telemetry()
            telemetry["Driver"] = lap["Driver"]
            telemetry["LapNumber"] = lap["LapNumber"]
            all_telemetry.append(telemetry)
        except Exception as e:
            print(f"Telemetry failed for {lap['Driver']} lap {lap['LapNumber']}: {e}")
            continue
    if all_telemetry:
        return pd.concat(all_telemetry, ignore_index=True)
    return pd.DataFrame()

def upload_df_to_gcs(df, gcs_path):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    print(f"Uploaded: {gcs_path}")

def download_session(year, round_num, session_name):
    session = fastf1.get_session(year, round_num, session_name)
    session.load(
        telemetry=(session_name in ["Q", "R"] and year >= 2020),
        weather=True,
        laps=True,
        messages=True
    )

    event_name = sanitise_name(session.event["EventName"])
    base_path = f"{year}/{event_name}/{session_name}"

    laps_df = session.laps
    weather_df = session.weather_data
    messages_df = session.race_control_messages
    results_df = session.results

    if not laps_df.empty:
        upload_df_to_gcs(laps_df, f"raw/fastf1/{base_path}/laps.csv")
    if not weather_df.empty:
        upload_df_to_gcs(weather_df, f"raw/fastf1/{base_path}/weather.csv")
    if not messages_df.empty:
        upload_df_to_gcs(messages_df, f"raw/fastf1/{base_path}/messages.csv")
    if not results_df.empty:
        upload_df_to_gcs(results_df, f"raw/fastf1/{base_path}/results.csv")

    if session_name in ["Q", "R"] and year >= 2020:
        telemetry_df = get_telemetry_df(session)
        if not telemetry_df.empty:
            upload_df_to_gcs(telemetry_df, f"raw/fastf1/{base_path}/telemetry.csv")

# --- MAIN LOOP ---
schedule = fastf1.get_event_schedule(YEAR)
rounds = schedule[schedule['EventFormat'] != 'testing']['RoundNumber'].tolist()

for round_num in rounds:
    for s in SESSIONS:
        try:
            download_session(YEAR, round_num, s)
        except Exception as e:
            print(f"Failed Year {YEAR} Round {round_num} {s}: {e}")
            failed_sessions.append((YEAR, round_num, s, str(e)))

if failed_sessions:
    print("\n--- FAILED SESSIONS ---")
    for f in failed_sessions:
        print(f)