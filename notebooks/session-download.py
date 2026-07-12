import fastf1
from google.cloud import storage
from dotenv import load_dotenv
import re

load_dotenv()

BUCKET_NAME = "f1-race-engineer-bucket"
CACHE_DIR = "data\\raw"
YEARS = range(2019, 2026)  # 2018-2025 only

fastf1.Cache.enable_cache(CACHE_DIR)
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def sanitise_name(name):
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w]", "", name)
    return name

def blob_exists(gcs_path):
    return bucket.blob(gcs_path).exists()

SESSION_NAME_MAP = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Race": "R",
    "Sprint Shootout": "SQ",
    "Sprint Qualifying": "SQ",
    # Sprint race name varies by year
    "Sprint": "Sprint",   # 2021-2022 stored as "Sprint"
    "Sprint Race": "S",   # if this appears
}

def get_actual_sessions(year, round_num):
    event = fastf1.get_event(year, round_num)
    sessions = []
    for i in range(1, 6):
        session_name = event.get(f"Session{i}")
        if session_name and str(session_name) not in (None, "None", ""):
            short = SESSION_NAME_MAP.get(session_name)
            if short:
                sessions.append(short)
            else:
                print(f"  WARNING: Unknown session name '{session_name}' — add to map")
    return sessions

EXPECTED_FILES = ["laps.csv", "results.csv", "weather.csv", "messages.csv"]

missing = []
incomplete = []
ok = []

for year in YEARS:
    print(f"\n{'='*50}")
    print(f"  YEAR: {year}")
    print(f"{'='*50}")

    try:
        schedule = fastf1.get_event_schedule(year)
        events = schedule[schedule['EventFormat'] != 'testing']
    except Exception as e:
        print(f"  Could not get schedule for {year}: {e}")
        continue

    for _, event in events.iterrows():
        round_num = int(event['RoundNumber'])
        event_name = sanitise_name(event['EventName'])

        try:
            sessions = get_actual_sessions(year, round_num)
        except Exception as e:
            print(f"  Could not get sessions for {year} R{round_num}: {e}")
            continue

        print(f"\n  R{round_num:02d} {event_name} — sessions: {sessions}")

        for s in sessions:
            base_path = f"raw/fastf1/{year}/{event_name}/{s}"

            missing_files = [f for f in EXPECTED_FILES if not blob_exists(f"{base_path}/{f}")]

            if len(missing_files) == len(EXPECTED_FILES):
                missing.append((year, round_num, event_name, s))
                print(f"    ❌ MISSING:    {s}")
            elif missing_files:
                incomplete.append((year, round_num, event_name, s, missing_files))
                print(f"    ⚠️  INCOMPLETE: {s} — missing {missing_files}")
            else:
                ok.append((year, round_num, event_name, s))
                print(f"    ✅ OK:         {s}")

# --- SUMMARY ---
print(f"\n{'='*50}")
print(f"  OK:         {len(ok)}")
print(f"  INCOMPLETE: {len(incomplete)}")
print(f"  MISSING:    {len(missing)}")
print(f"{'='*50}")

with open("gcs_audit.txt", "w") as f:
    f.write(f"AUDIT RESULTS: 2018-2025\n")
    f.write(f"{'='*50}\n")
    f.write(f"OK:         {len(ok)}\n")
    f.write(f"INCOMPLETE: {len(incomplete)}\n")
    f.write(f"MISSING:    {len(missing)}\n")
    f.write(f"{'='*50}\n\n")

    f.write("--- MISSING (nothing uploaded) ---\n")
    for item in missing:
        f.write(f"{item[0]} R{item[1]:02d} | {item[2]:<40} | {item[3]}\n")

    f.write("\n--- INCOMPLETE (some files missing) ---\n")
    for item in incomplete:
        f.write(f"{item[0]} R{item[1]:02d} | {item[2]:<40} | {item[3]} | missing: {item[4]}\n")

print("\nAudit saved to gcs_audit.txt")