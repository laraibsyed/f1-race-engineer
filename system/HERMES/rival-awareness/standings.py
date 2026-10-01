
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
    ""
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
                wait = 3 * (2 ** attempt)
                print(f"[standings]   request failed ({e!r}), retrying in {wait}s ...")
                time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err!r}")

def build_driver_code_map(year):
    ""
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
    ""
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
    ""
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
    ""
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
    ""
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
