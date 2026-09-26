"""
fetch_openmeteo_weather.py
---------------------------
Pulls weather data from Open-Meteo's free API for F1 circuits, for two jobs:

  1. HISTORICAL FORECAST (--mode historical): what the forecast actually said on a
     past race date. Used to backtest your rain-crossover logic against what a
     strategist would genuinely have seen in real time -- NOT what FastF1's weather.csv
     shows after the fact (that's ground truth, this is the predictive input).

  2. LIVE FORECAST (--mode forecast): precipitation_probability etc for the next
     16 days. This is what actually feeds your live rain-probability -> compound
     crossover threshold logic.

Both endpoints are free, no API key needed (Open-Meteo, 2024). Rate limit is a soft
10,000 requests/day -- fine for this project's scale.

Circuit coordinates: see CIRCUIT_COORDS below. Sourced from F1DB (Overdijk, 2026),
an actively-maintained open F1 database, cross-checked against official session
locations. A few circuits deliberately share coordinates -- see notes inline
(e.g. Mexican_Grand_Prix/Mexico_City_Grand_Prix are the same physical track under
different event-naming years; Sakhir_Grand_Prix/Styrian_Grand_Prix/etc are 2020
covid-calendar one-offs at an existing venue).

Usage:
    python fetch_openmeteo_weather.py --mode historical --race Abu_Dhabi_Grand_Prix --date 2018-11-25
    python fetch_openmeteo_weather.py --mode forecast --race Abu_Dhabi_Grand_Prix
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse

# Sourced from F1DB (Overdijk, 2026) <https://github.com/f1db/f1db>, cross-checked
# against official circuit locations. lat/lon pinned to the circuit itself, not the
# host city, since weather can genuinely differ across a large metro area.
CIRCUIT_COORDS = {
    "70th_Anniversary_Grand_Prix": (52.0786, -1.0169),   # Silverstone (2020 one-off event name)
    "Abu_Dhabi_Grand_Prix": (24.4672, 54.6031),          # Yas Marina Circuit
    "Australian_Grand_Prix": (-37.8497, 144.9680),       # Albert Park Circuit
    "Austrian_Grand_Prix": (47.2197, 14.7647),           # Red Bull Ring
    "Azerbaijan_Grand_Prix": (40.3725, 49.8533),         # Baku City Circuit
    "Bahrain_Grand_Prix": (26.0325, 50.5106),            # Bahrain International Circuit
    "Barcelona_Grand_Prix": (41.5700, 2.2611),           # Circuit de Barcelona-Catalunya (2026 naming)
    "Belgian_Grand_Prix": (50.4372, 5.9714),             # Circuit de Spa-Francorchamps
    "Brazilian_Grand_Prix": (-23.7036, -46.6997),        # Interlagos (pre-2021 naming)
    "British_Grand_Prix": (52.0786, -1.0169),            # Silverstone Circuit
    "Canadian_Grand_Prix": (45.5000, -73.5228),          # Circuit Gilles Villeneuve
    "Chinese_Grand_Prix": (31.3389, 121.2200),           # Shanghai International Circuit
    "Dutch_Grand_Prix": (52.3888, 4.5409),               # Circuit Zandvoort
    "Eifel_Grand_Prix": (50.3356, 6.9475),               # Nurburgring (2020 one-off)
    "Emilia_Romagna_Grand_Prix": (44.3439, 11.7167),     # Imola
    "French_Grand_Prix": (43.2506, 5.7917),              # Circuit Paul Ricard
    "German_Grand_Prix": (49.3278, 8.5656),              # Hockenheimring
    "Hungarian_Grand_Prix": (47.5789, 19.2486),          # Hungaroring
    "Italian_Grand_Prix": (45.6156, 9.2811),             # Monza
    "Japanese_Grand_Prix": (34.8431, 136.5411),          # Suzuka
    "Las_Vegas_Grand_Prix": (36.1147, -115.1728),        # Las Vegas Street Circuit
    "Mexican_Grand_Prix": (19.4042, -99.0907),           # Autodromo Hermanos Rodriguez (pre-2021 naming)
    "Mexico_City_Grand_Prix": (19.4042, -99.0907),       # same track, 2021+ naming
    "Miami_Grand_Prix": (25.9581, -80.2389),             # Miami International Autodrome
    "Monaco_Grand_Prix": (43.7347, 7.4206),              # Circuit de Monaco
    "Portuguese_Grand_Prix": (37.2270, -8.6267),         # Portimao
    "Qatar_Grand_Prix": (25.4900, 51.4542),              # Losail International Circuit
    "Russian_Grand_Prix": (43.4057, 39.9578),            # Sochi Autodrom
    "Sakhir_Grand_Prix": (26.0325, 50.5106),             # Bahrain outer layout, 2020 one-off, same site
    "Saudi_Arabian_Grand_Prix": (21.6319, 39.1044),      # Jeddah Corniche Circuit
    "Singapore_Grand_Prix": (1.2914, 103.8642),          # Marina Bay Street Circuit
    "Spanish_Grand_Prix": (41.5700, 2.2611),             # Circuit de Barcelona-Catalunya (older naming)
    "Styrian_Grand_Prix": (47.2197, 14.7647),            # Red Bull Ring, 2020/21 one-off
    "São_Paulo_Grand_Prix": (-23.7036, -46.6997),        # Interlagos, 2021+ naming
    "Turkish_Grand_Prix": (40.9517, 29.4050),            # Istanbul Park
    "Tuscan_Grand_Prix": (43.6167, 11.3833),             # Mugello, 2020 one-off
    "United_States_Grand_Prix": (30.1328, -97.6411),     # Circuit of the Americas
}

HISTORICAL_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# NOTE: precipitation_probability is a FORECAST-only field (it's a model confidence
# metric, meaningless for a date that's already happened). The historical/archive
# endpoint only returns what was actually measured (rain amount, humidity etc) --
# no probability field exists there. Kept these two var lists separate on purpose.
HOURLY_VARS_HISTORICAL = "temperature_2m,precipitation,rain,relative_humidity_2m,wind_speed_10m"
HOURLY_VARS_FORECAST = "temperature_2m,precipitation_probability,precipitation,rain,relative_humidity_2m,wind_speed_10m"


def fetch_json(url, params):
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    with urllib.request.urlopen(full_url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_historical_weather(race_name, date_str):
    """What actually happened, per Open-Meteo's reanalysis model, for a past date.

    IMPORTANT LIMITATION for your methodology: this is NOT "what a forecast said
    N days before the race" -- that data (archived forecast snapshots) isn't freely
    available anywhere. This is reanalysis: a best-estimate reconstruction of actual
    conditions, blending models with real observations after the fact. Comparable in
    spirit to FastF1's own weather.csv (also a measurement), not to a live forecast.
    Useful for cross-checking/filling gaps in FastF1 weather data (e.g. the 2022
    Emilia Romagna Sprint gap noted in your data_cleaning notes), but NOT a valid
    stand-in for backtesting "would my crossover logic have called this correctly
    in real time" -- that would need precipitation_probability, which only exists
    on the live forecast endpoint. Document this distinction clearly if you use it."""
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
    """Live forecast for the next N days (max 16). This is the actual predictive
    input for the rain-probability -> compound crossover logic."""
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