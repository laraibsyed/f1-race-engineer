""

import argparse
import json
import sys
import urllib.request
import urllib.parse

CIRCUIT_COORDS = {
    "70th_Anniversary_Grand_Prix": (52.0786, -1.0169),
    "Abu_Dhabi_Grand_Prix": (24.4672, 54.6031),
    "Australian_Grand_Prix": (-37.8497, 144.9680),
    "Austrian_Grand_Prix": (47.2197, 14.7647),
    "Azerbaijan_Grand_Prix": (40.3725, 49.8533),
    "Bahrain_Grand_Prix": (26.0325, 50.5106),
    "Barcelona_Grand_Prix": (41.5700, 2.2611),
    "Belgian_Grand_Prix": (50.4372, 5.9714),
    "Brazilian_Grand_Prix": (-23.7036, -46.6997),
    "British_Grand_Prix": (52.0786, -1.0169),
    "Canadian_Grand_Prix": (45.5000, -73.5228),
    "Chinese_Grand_Prix": (31.3389, 121.2200),
    "Dutch_Grand_Prix": (52.3888, 4.5409),
    "Eifel_Grand_Prix": (50.3356, 6.9475),
    "Emilia_Romagna_Grand_Prix": (44.3439, 11.7167),
    "French_Grand_Prix": (43.2506, 5.7917),
    "German_Grand_Prix": (49.3278, 8.5656),
    "Hungarian_Grand_Prix": (47.5789, 19.2486),
    "Italian_Grand_Prix": (45.6156, 9.2811),
    "Japanese_Grand_Prix": (34.8431, 136.5411),
    "Las_Vegas_Grand_Prix": (36.1147, -115.1728),
    "Mexican_Grand_Prix": (19.4042, -99.0907),
    "Mexico_City_Grand_Prix": (19.4042, -99.0907),
    "Miami_Grand_Prix": (25.9581, -80.2389),
    "Monaco_Grand_Prix": (43.7347, 7.4206),
    "Portuguese_Grand_Prix": (37.2270, -8.6267),
    "Qatar_Grand_Prix": (25.4900, 51.4542),
    "Russian_Grand_Prix": (43.4057, 39.9578),
    "Sakhir_Grand_Prix": (26.0325, 50.5106),
    "Saudi_Arabian_Grand_Prix": (21.6319, 39.1044),
    "Singapore_Grand_Prix": (1.2914, 103.8642),
    "Spanish_Grand_Prix": (41.5700, 2.2611),
    "Styrian_Grand_Prix": (47.2197, 14.7647),
    "São_Paulo_Grand_Prix": (-23.7036, -46.6997),
    "Turkish_Grand_Prix": (40.9517, 29.4050),
    "Tuscan_Grand_Prix": (43.6167, 11.3833),
    "United_States_Grand_Prix": (30.1328, -97.6411),
}

HISTORICAL_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS_HISTORICAL = "temperature_2m,precipitation,rain,relative_humidity_2m,wind_speed_10m"
HOURLY_VARS_FORECAST = "temperature_2m,precipitation_probability,precipitation,rain,relative_humidity_2m,wind_speed_10m"

def fetch_json(url, params):
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    with urllib.request.urlopen(full_url, timeout=30) as resp:
        return json.loads(resp.read().decode())

def get_historical_weather(race_name, date_str):
    ""
    if race_name not in CIRCUIT_COORDS:
        raise ValueError(f"Unknown race name '{race_name}' -- not in CIRCUIT_COORDS. "
                          f"Check spelling matches your bucket's folder naming exactly.")
    lat, lon = CIRCUIT_COORDS[race_name]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": HOURLY_VARS_HISTORICAL,
        "timezone": "auto",
    }
    return fetch_json(HISTORICAL_BASE_URL, params)

def get_forecast_weather(race_name, days=16):
    ""
    if race_name not in CIRCUIT_COORDS:
        raise ValueError(f"Unknown race name '{race_name}' -- not in CIRCUIT_COORDS. "
                          f"Check spelling matches your bucket's folder naming exactly.")
    lat, lon = CIRCUIT_COORDS[race_name]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS_FORECAST,
        "forecast_days": min(days, 16),
        "timezone": "auto",
    }
    return fetch_json(FORECAST_BASE_URL, params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["historical", "forecast"], required=True)
    parser.add_argument("--race", required=True, help="Race name exactly as it appears in your bucket, e.g. Abu_Dhabi_Grand_Prix")
    parser.add_argument("--date", help="YYYY-MM-DD, required for --mode historical")
    parser.add_argument("--days", type=int, default=16, help="Forecast days ahead (max 16), for --mode forecast")
    parser.add_argument("--out", help="Optional output JSON path")
    args = parser.parse_args()

    try:
        if args.mode == "historical":
            if not args.date:
                print("--date YYYY-MM-DD is required for --mode historical")
                sys.exit(1)
            data = get_historical_weather(args.race, args.date)
        else:
            data = get_forecast_weather(args.race, args.days)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Network error reaching Open-Meteo: {e}")
        sys.exit(1)

    print(json.dumps(data, indent=2)[:2000], "...(truncated)" if len(json.dumps(data)) > 2000 else "")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n-> wrote {args.out}")

if __name__ == "__main__":
    main()
