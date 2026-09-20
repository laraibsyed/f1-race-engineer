"""
Minimal connectivity test — bypasses FastF1 entirely to isolate whether the earlier
ConnectionResetError is a general network/TLS problem on this machine, or something
specific to FastF1's request session/cache wrapper.

Usage:
    python test_connection.py
"""
import sys
import requests

URL = "https://api.jolpi.ca/ergast/f1/2023/drivers.json?limit=5"

# Jolpica's own docs explicitly require an identifying User-Agent — a generic default
# one may be silently throttled/ignored rather than rejected outright, which would
# look exactly like the timeout seen in the previous run.
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