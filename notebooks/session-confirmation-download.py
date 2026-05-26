import fastf1
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv
import io

load_dotenv()

# --- CONFIG ---
BUCKET_NAME = "f1-race-engineer-bucket"
CACHE_DIR = "data\\raw"          # local fastf1 cache (unavoidable)
YEAR = 2018
ROUND = 14                     # Australia
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]
failed_sessions = []

fastf1.Cache.enable_cache(CACHE_DIR)
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

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
    """Convert df to CSV bytes and stream to GCS — no local file needed"""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    print(f"Uploaded: {gcs_path}")

def download_session(year, round_num, session_name):
    session = fastf1.get_session(year, round_num, session_name)
    
    session.load(telemetry=(session_name not in ["FP1", "FP2", "FP3"]), weather=True, laps=True, messages=True)
    event_name = session.event["EventName"].replace(" ", "_")
    base_path = f"{year}/{event_name}/{session_name}"

    try:
        laps_df = session.laps
    except:
        laps_df = pd.DataFrame()

    try:
        weather_df = session.weather_data
    except:
        weather_df = pd.DataFrame()

    try:
        messages_df = session.race_control_messages
    except:
        messages_df = pd.DataFrame()

    try:
        results_df = session.results
    except:
        results_df = pd.DataFrame()

    if not laps_df.empty:
        upload_df_to_gcs(laps_df, f"raw/fastf1/{base_path}/laps.csv")
    if not weather_df.empty:
        upload_df_to_gcs(weather_df, f"raw/fastf1/{base_path}/weather.csv")
    if not messages_df.empty:
        upload_df_to_gcs(messages_df, f"raw/fastf1/{base_path}/messages.csv")
    if not results_df.empty:
        upload_df_to_gcs(results_df, f"raw/fastf1/{base_path}/results.csv")

    if session_name in ["Q", "R"]:
        telemetry_df = get_telemetry_df(session)
        if not telemetry_df.empty:
            upload_df_to_gcs(telemetry_df, f"raw/fastf1/{base_path}/telemetry.csv")

for s in SESSIONS:
    try:
        download_session(YEAR, ROUND, s)
    except Exception as e:
        print(f"Failed {s}: {e}")
        failed_sessions.append(s)