"""
Championship Standings Helper (v2 — direct requests, not FastF1's Ergast wrapper)
====================================================================================
v1 used fastf1.ergast.Ergast() and failed twice with two different errors (a TLS
connection reset, then a read timeout) even though a plain `requests.get()` with a
proper User-Agent header succeeded immediately and reliably. Rather than keep
debugging FastF1's internal request handling blind, this hits the Jolpica API (the
maintained, drop-in-compatible successor to the now-shut-down Ergast API) directly.

CONFIRMED schema (via live response, 2023 drivers.json): driver objects have a "code"
field that matches FastF1's 3-letter codes exactly (e.g. "code": "ALB" for Albon) — no
guessing needed for that part. Race-schedule and driverStandings response shapes are
the long-stable, documented Ergast/Jolpica schema (unchanged by the Jolpica migration)
but haven't been live-confirmed against your data yet — if get_points_gap silently
returns None a lot, run --inspect for a season and check those two response shapes
specifically before assuming the data itself is missing.

Jolpica's own docs require an identifying User-Agent — sending one is not optional,
it's the likely reason the FastF1 wrapper's requests were timing out/resetting.

Usage:
    python standings.py --inspect --year 2023   # confirm real response shapes first
    python standings.py --year 2023              # build and cache one season's lookups
"""

import os
import json
import time
import argparse
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "F1RaceEngineerDissertation/1.0 (RivalKnowledgeModule)"}
CACHE_DIR = "./checkpoints/rival_knowledge/standings_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_json(path, retries=4):
    """GET with the required User-Agent header, a generous timeout, and exponential
    backoff — the earlier failures were intermittent (one endpoint succeeded, another
    failed immediately after with the same connection-reset pattern seen before the
    User-Agent fix), consistent with rate-limiting or general flakiness on a small,
    volunteer-run API rather than anything wrong in this request itself."""
    url = f"{BASE_URL}/{path}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = 3 * (2 ** attempt)  # 3s, 6s, 12s, ...
                print(f"[standings]   request failed ({e!r}), retrying in {wait}s ...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err!r}")


def build_driver_code_map(year):
    """Maps FastF1 3-letter driver codes (e.g. 'VER') to Jolpica driverIds
    (e.g. 'max_verstappen') using the API's own 'code' field — CONFIRMED live against
    2023 data to match FastF1 codes exactly."""
    cache_path = f"{CACHE_DIR}/{year}_driver_code_map.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    data = _get_json(f"{year}/drivers.json?limit=100")
    drivers = data["MRData"]["DriverTable"]["Drivers"]
    code_map = {d["code"]: d["driverId"] for d in drivers if "code" in d and "driverId" in d}

    with open(cache_path, "w") as f:
        json.dump(code_map, f)
    return code_map


def build_round_lookup(year):
    """Maps a race name (fastf1-style, underscore-separated) to its round number.
    Ergast/Jolpica's raceName is space-separated (e.g. 'Bahrain Grand Prix') — matched
    to your race_fastf1 naming by a straight space<->underscore swap. Verify this holds
    for every race in a season via --inspect before trusting it archive-wide; a mismatch
    here fails silently (points_gap just returns None) rather than erroring loudly."""
    cache_path = f"{CACHE_DIR}/{year}_rounds.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    data = _get_json(f"{year}.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    lookup = {r["raceName"].replace(" ", "_"): int(r["round"]) for r in races}

    with open(cache_path, "w") as f:
        json.dump(lookup, f)
    return lookup


def get_standings_before_round(year, round_num):
    """Cumulative driver points BEFORE the given round (i.e. as of the end of
    round_num - 1). Round 1 or earlier returns empty standings — season hasn't
    started, so no points gap is meaningful yet."""
    cache_path = f"{CACHE_DIR}/{year}_standings_by_round.json"
    all_rounds = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            all_rounds = json.load(f)

    key = str(round_num - 1) if round_num > 1 else "0"
    if key in all_rounds:
        return all_rounds[key]

    if round_num <= 1:
        points = {}
    else:
        data = _get_json(f"{year}/{round_num - 1}/driverStandings.json?limit=100")
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        points = {}
        if lists:
            for entry in lists[0]["DriverStandings"]:
                points[entry["Driver"]["driverId"]] = float(entry["points"])

    all_rounds[key] = points
    with open(cache_path, "w") as f:
        json.dump(all_rounds, f)
    return points


def get_signed_points_gap(year, race_fastf1, code_a, code_b):
    """SIGNED points gap (code_a's points minus code_b's points) BEFORE the race in
    question — direction matters for a real risk policy (are you ahead or behind this
    specific rival?), unlike the absolute value get_points_gap() returns. Returns None
    if any lookup fails."""
    try:
        code_map = build_driver_code_map(year)
        round_lookup = build_round_lookup(year)
        round_num = round_lookup.get(race_fastf1)
        if round_num is None:
            return None
        points = get_standings_before_round(year, round_num)
        id_a, id_b = code_map.get(code_a), code_map.get(code_b)
        if id_a is None or id_b is None:
            return None
        return points.get(id_a, 0.0) - points.get(id_b, 0.0)
    except Exception as e:
        print(f"[standings] Could not compute signed points gap for {year} {race_fastf1} {code_a} vs {code_b}: {e!r}")
        return None


def get_points_gap(year, race_fastf1, code_a, code_b):
    """Points gap between two drivers (by FastF1 3-letter code) BEFORE the race in
    question. Returns None if any lookup fails — never guess, just report unavailable."""
    try:
        code_map = build_driver_code_map(year)
        round_lookup = build_round_lookup(year)
        round_num = round_lookup.get(race_fastf1)
        if round_num is None:
            return None
        points = get_standings_before_round(year, round_num)
        id_a, id_b = code_map.get(code_a), code_map.get(code_b)
        if id_a is None or id_b is None:
            return None
        return abs(points.get(id_a, 0.0) - points.get(id_b, 0.0))
    except Exception as e:
        print(f"[standings] Could not compute points gap for {year} {race_fastf1} {code_a} vs {code_b}: {e!r}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        print(f"Fetching driver info for {args.year} ...")
        data = _get_json(f"{args.year}/drivers.json?limit=100")
        drivers = data["MRData"]["DriverTable"]["Drivers"]
        print(f"{len(drivers)} drivers. First 5:")
        for d in drivers[:5]:
            print(f"  code={d.get('code')!r}, driverId={d.get('driverId')!r}, name={d.get('givenName')} {d.get('familyName')}")

        print(f"\nFetching race schedule for {args.year} ...")
        time.sleep(2)
        sched = _get_json(f"{args.year}.json?limit=100")
        races = sched["MRData"]["RaceTable"]["Races"]
        print(f"{len(races)} races:")
        for r in races:
            print(f"  round={r['round']}, raceName={r['raceName']!r}")

        print(f"\nFetching driver standings after round 1 for {args.year} ...")
        time.sleep(2)
        standings = _get_json(f"{args.year}/1/driverStandings.json?limit=100")
        lists = standings["MRData"]["StandingsTable"]["StandingsLists"]
        if lists:
            print(f"First 5 standings entries:")
            for entry in lists[0]["DriverStandings"][:5]:
                print(f"  position={entry.get('position')}, points={entry.get('points')}, driverId={entry['Driver'].get('driverId')}")
        else:
            print("  No standings lists returned — check the year/round.")
        return

    code_map = build_driver_code_map(args.year)
    print(f"Built driver code map for {args.year} ({len(code_map)} drivers): {code_map}")
    round_lookup = build_round_lookup(args.year)
    print(f"Built round lookup for {args.year} ({len(round_lookup)} races): {round_lookup}")


if __name__ == "__main__":
    main()