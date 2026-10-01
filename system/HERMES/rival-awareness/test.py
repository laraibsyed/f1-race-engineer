
import sys
import requests

URL = "https://api.jolpi.ca/ergast/f1/2023/drivers.json?limit=5"

HEADERS = {"User-Agent": "F1RaceEngineerDissertation/1.0 (RivalKnowledgeModule)"}

print(f"Python version: {sys.version}")
print(f"Testing connection to: {URL}")
print(f"With headers: {HEADERS}\n")

try:
    r = requests.get(URL, headers=HEADERS, timeout=30)
    print(f"SUCCESS — status {r.status_code}")
    print(r.text[:500])
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    print("\nIf this also fails, the User-Agent header wasn't the cause — try a")
    print("different network (e.g. phone hotspot) to rule out local firewall/VPN,")
    print("or running this under an older Python (3.11/3.12) via a separate venv to")
    print("rule out a 3.14-specific SSL/networking quirk.")
