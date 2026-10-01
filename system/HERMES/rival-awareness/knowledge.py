
import os
import json
import argparse
import pandas as pd
import numpy as np

try:
    from google.cloud import storage
except ImportError:
    storage = None

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")

YEAR = None
RACE_FASTF1 = None
RACE_TI = None
SESSION_FASTF1 = "R"
SESSION_TI = "Race"

MIN_CLEAN_LAPS_FOR_BASELINE = 3

class CachedBucket:
    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=CACHE_DIR):
        if storage is None:
            raise RuntimeError("google-cloud-storage not installed — pip install google-cloud-storage")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def _local_path(self, blob_path):
        return os.path.join(self.cache_dir, blob_path)

    def read_text(self, blob_path):
        local = self._local_path(blob_path)
        if os.path.exists(local):
            with open(local, "r", encoding="utf-8") as f:
                return f.read()
        blob = self.bucket.blob(blob_path)
        content = blob.download_as_text()
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "w", encoding="utf-8") as f:
            f.write(content)
        return content

    def read_csv(self, blob_path, **kwargs):
        local = self._local_path(blob_path)
        if os.path.exists(local):
            return pd.read_csv(local, **kwargs)
        blob = self.bucket.blob(blob_path)
        data = blob.download_as_bytes()
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            f.write(data)
        return pd.read_csv(local, **kwargs)

    def list_blob_names(self, prefix):

        return [b.name for b in self.bucket.list_blobs(prefix=prefix)]

    def exists(self, blob_path):
        ""
        return self.bucket.blob(blob_path).exists()

    def write_csv(self, blob_path, df):
        ""
        local = self._local_path(blob_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        df.to_csv(local, index=False)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_filename(local, content_type="text/csv")

def inspect_one_file(cb):
    prefix = f"raw/tracinginsights/{YEAR}/{RACE_TI}/{SESSION_TI}/"
    names = cb.list_blob_names(prefix)
    if not names:
        print(f"[inspect] Nothing found under {prefix} — check RACE_TI/SESSION_TI naming.")
        return
    sample_path = names[0]
    print(f"[inspect] Sample file: {sample_path}")
    raw = cb.read_text(sample_path)
    data = json.loads(raw)
    if not isinstance(data, dict):
        print(f"[inspect] Unexpected top-level type: {type(data)} — dump first 500 chars:")
        print(raw[:500])
        return

    print(f"[inspect] Top-level keys: {list(data.keys())}")

    payload = data.get("tel", data)
    if isinstance(payload, list):
        print(f"[inspect] 'tel' is a LIST of {len(payload)} items.")
        if payload:
            first = payload[0]
            print(f"[inspect] First item type: {type(first).__name__}")
            if isinstance(first, dict):
                print(f"[inspect] First item keys: {list(first.keys())}")
                for k in ("DriverAhead", "DistanceToDriverAhead"):
                    if k in first:
                        print(f"[inspect] {k} (first item): {first[k]!r}")
                    else:
                        print(f"[inspect] KEY NOT FOUND in first item: {k}")

                for key in ("DriverAhead", "DistanceToDriverAhead"):
                    sample_vals = [item.get(key) for item in payload[:10] if isinstance(item, dict)]
                    print(f"[inspect] {key} across first 10 items: {sample_vals}")
    elif isinstance(payload, dict):
        print(f"[inspect] 'tel' is a DICT with keys: {list(payload.keys())}")
        for k in ("DriverAhead", "DistanceToDriverAhead"):
            if k in payload:
                v = payload[k]
                print(f"[inspect] {k}: type={type(v).__name__}, sample={str(v)[:200]}")
            else:
                print(f"[inspect] KEY NOT FOUND: {k} — check exact naming in this file.")
    else:
        print(f"[inspect] 'tel' is type {type(payload).__name__}: {str(payload)[:300]}")

def load_features(cb):
    path = f"clean/features/{YEAR}/{RACE_FASTF1}/{SESSION_FASTF1}/laps_features.csv"
    df = cb.read_csv(path, dtype={"TrackStatus": str})
    return df

def build_number_to_code(df):
    m = df[["DriverNumber", "Driver"]].drop_duplicates()
    return dict(zip(m["DriverNumber"].astype(str), m["Driver"]))

def lap_time_to_seconds(series):
    ""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

def add_cumulative_time(df):
    ""
    df = df.sort_values(["Driver", "LapNumber"]).copy()
    df["LapTime_s"] = lap_time_to_seconds(df["LapTime"])
    df["cum_time_s"] = df.groupby("Driver")["LapTime_s"].cumsum()
    return df

def extract_driver_ahead(data):
    ""
    tel = data.get("tel", {})
    driver_ahead_raw = tel.get("DriverAhead")
    distance_raw = tel.get("DistanceToDriverAhead")

    if driver_ahead_raw is None:
        return None, None

    if isinstance(driver_ahead_raw, list):

        pairs = zip(driver_ahead_raw, distance_raw if isinstance(distance_raw, list) else [distance_raw] * len(driver_ahead_raw))
        clean_pairs = [(d, dist) for d, dist in pairs if d is not None and str(d) != "None"]
        if not clean_pairs:
            return None, None
        driver_ahead_raw, distance_raw = clean_pairs[-1]

    if distance_raw is None or str(distance_raw) == "None":
        distance_val = None
    else:
        try:
            distance_val = float(distance_raw)
        except (TypeError, ValueError):
            distance_val = None

    return str(driver_ahead_raw), distance_val

def add_behind_columns(opponent_df, features_df):
    ""
    reverse = opponent_df.dropna(subset=["ahead_driver"])[
        ["Driver", "LapNumber", "ahead_driver", "distance_to_ahead_m"]
    ].rename(columns={
        "Driver": "behind_driver",
        "ahead_driver": "Driver",
        "distance_to_ahead_m": "distance_to_behind_m",
    })

    merged = opponent_df.merge(reverse, on=["Driver", "LapNumber"], how="left")

    lookup = features_df[["Driver", "LapNumber", "Compound", "TyreLife"]].rename(
        columns={"Driver": "behind_driver", "Compound": "behind_compound", "TyreLife": "behind_tyre_age"}
    )
    merged = merged.merge(lookup, on=["behind_driver", "LapNumber"], how="left")
    return merged

def build_opponent_table(cb, features_df, number_to_code):
    ""
    checkpoint_path = f"./checkpoints/rival_knowledge/{YEAR}_{RACE_FASTF1}_{SESSION_FASTF1}_opponents.csv"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    OPPONENT_COLUMNS = ["Driver", "LapNumber", "ahead_driver", "distance_to_ahead_m", "ahead_compound", "ahead_tyre_age"]

    if os.path.exists(checkpoint_path):
        print(f"[opponent] Resuming from checkpoint: {checkpoint_path}")
        df = pd.read_csv(checkpoint_path)
        return df if not df.empty else pd.DataFrame(columns=OPPONENT_COLUMNS)

    rows = []
    drivers = features_df["Driver"].unique()
    lookup = features_df.set_index(["Driver", "LapNumber"])

    for i, driver in enumerate(drivers):
        driver_laps = features_df[features_df["Driver"] == driver]["LapNumber"].unique()
        for lap in driver_laps:
            path = f"raw/tracinginsights/{YEAR}/{RACE_TI}/{SESSION_TI}/{driver}/{int(lap)}_tel.json"
            try:
                raw = cb.read_text(path)
            except Exception:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ahead_number, distance = extract_driver_ahead(data)
            ahead_code = number_to_code.get(ahead_number) if ahead_number is not None else None

            ahead_compound, ahead_tyre_age = None, None
            if ahead_code is not None and (ahead_code, lap) in lookup.index:
                ahead_row = lookup.loc[(ahead_code, lap)]
                ahead_compound = ahead_row.get("Compound")
                ahead_tyre_age = ahead_row.get("TyreLife")

            rows.append({
                "Driver": driver,
                "LapNumber": lap,
                "ahead_driver": ahead_code,
                "distance_to_ahead_m": distance,
                "ahead_compound": ahead_compound,
                "ahead_tyre_age": ahead_tyre_age,
            })

        pd.DataFrame(rows, columns=OPPONENT_COLUMNS if not rows else None).to_csv(checkpoint_path, index=False)
        print(f"[opponent] Checkpointed after driver {i+1}/{len(drivers)} ({driver})")

    if not rows:
        print(f"[opponent] WARNING: zero telemetry reads succeeded for this entire race "
              f"— check whether this is a known data gap (like 2018 Bahrain) or a "
              f"race/session naming mismatch before trusting a downstream 0-event result.")
        return pd.DataFrame(columns=OPPONENT_COLUMNS)

    return pd.DataFrame(rows)

MAX_PLAUSIBLE_PIT_STOP_S = 120.0

def compute_pit_lane_loss(df):
    ""
    results = []
    sorted_df = df.sort_values(["Driver", "LapNumber"])
    indexed = sorted_df.set_index(["Driver", "LapNumber"])

    for _, in_row in sorted_df[sorted_df["is_pit_in"].fillna(False).astype(bool)].iterrows():
        driver = in_row["Driver"]
        in_lap = in_row["LapNumber"]
        out_lap = in_lap + 1
        if (driver, out_lap) not in indexed.index:
            continue
        out_row = indexed.loc[(driver, out_lap)]
        if isinstance(out_row, pd.DataFrame):
            out_row = out_row.iloc[0]
        if not bool(out_row.get("is_pit_out", False)):
            continue

        under_sc = bool(in_row.get("is_sc_lap", False) or in_row.get("is_vsc_lap", False)
                         or out_row.get("is_sc_lap", False) or out_row.get("is_vsc_lap", False))
        if under_sc:
            continue

        try:
            duration = (pd.to_timedelta(out_row["PitOutTime"]) - pd.to_timedelta(in_row["PitInTime"])).total_seconds()
        except Exception:
            continue
        if pd.isna(duration):
            continue
        if duration > MAX_PLAUSIBLE_PIT_STOP_S:
            continue

        results.append({"driver": driver, "in_lap": int(in_lap), "out_lap": int(out_lap), "loss_s": duration})

    loss_df = pd.DataFrame(results)
    if loss_df.empty:
        print("[pit_loss] No valid green-flag pit stop pairs found — check is_pit_in/"
              "is_pit_out/PitInTime/PitOutTime exist and aren't all null for this race.")
        return loss_df, None

    p10, p90 = loss_df["loss_s"].quantile([0.10, 0.90])
    winsorized = loss_df["loss_s"].clip(p10, p90)
    circuit_constant_s = winsorized.median()

    return loss_df, circuit_constant_s

    loss_df = pd.DataFrame(results)
    if loss_df.empty:
        print("[pit_loss] No valid pit in/out laps found — check is_pit_in/is_pit_out flags exist and aren't all null.")
        return loss_df, None

    p10, p90 = loss_df["loss_s"].quantile([0.10, 0.90])
    winsorized = loss_df["loss_s"].clip(p10, p90)
    circuit_constant_s = winsorized.median()

    return loss_df, circuit_constant_s

def flag_undercut_windows(opponent_df, features_df, pit_loss_constant_s,
                           window_laps=3, min_tyre_age_gap=3):
    ""
    REFERENCE_SPEED_MPS = 83.0

    merged = opponent_df.merge(
        features_df[["Driver", "LapNumber", "TyreLife", "Compound", "is_pit_in"]],
        on=["Driver", "LapNumber"], how="left"
    )

    merged["distance_to_ahead_m"] = pd.to_numeric(merged["distance_to_ahead_m"], errors="coerce")
    merged["TyreLife"] = pd.to_numeric(merged["TyreLife"], errors="coerce")
    merged["ahead_tyre_age"] = pd.to_numeric(merged["ahead_tyre_age"], errors="coerce")
    merged["approx_gap_s"] = merged["distance_to_ahead_m"] / REFERENCE_SPEED_MPS
    merged["tyre_age_gap"] = merged["TyreLife"] - merged["ahead_tyre_age"]

    pit_laps = features_df.loc[features_df["is_pit_in"].fillna(False).astype(bool), ["Driver", "LapNumber"]]
    decision_window = set()
    for _, row in pit_laps.iterrows():
        for lap in range(int(row["LapNumber"]) - window_laps, int(row["LapNumber"]) + 1):
            decision_window.add((row["Driver"], lap))

    merged["in_pit_decision_window"] = merged.apply(
        lambda r: (r["Driver"], r["LapNumber"]) in decision_window, axis=1
    )

    merged["undercut_opportunity"] = (
        merged["in_pit_decision_window"]
        & (merged["approx_gap_s"] < pit_loss_constant_s)
        & (merged["tyre_age_gap"] >= min_tyre_age_gap)
    )

    return merged

def summarize_undercut_events(flagged_df, features_df=None):
    ""
    events = []
    for driver, g in flagged_df.sort_values(["Driver", "LapNumber"]).groupby("Driver"):
        flagged_laps = g.loc[g["undercut_opportunity"], "LapNumber"].tolist()
        if not flagged_laps:
            continue

        run_start = flagged_laps[0]
        prev = flagged_laps[0]
        for lap in flagged_laps[1:] + [None]:
            if lap is not None and lap == prev + 1:
                prev = lap
                continue

            pit_rows = g.loc[(g["LapNumber"] >= run_start) & (g["LapNumber"] <= prev + 3) & (g["is_pit_in"].fillna(False)), "LapNumber"]
            actual_pit_lap = int(pit_rows.iloc[0]) if not pit_rows.empty else None

            direction = None
            if features_df is not None and actual_pit_lap is not None:
                rival_row = g.loc[g["LapNumber"] == prev, "ahead_driver"]
                rival = rival_row.iloc[0] if not rival_row.empty else None
                if rival is not None and pd.notna(rival):
                    rival_pit_laps = features_df.loc[
                        (features_df["Driver"] == rival) & (features_df["is_pit_in"].fillna(False)),
                        "LapNumber"
                    ].tolist()
                    if rival_pit_laps:
                        nearest = min(rival_pit_laps, key=lambda x: abs(x - actual_pit_lap))
                        if nearest < actual_pit_lap:
                            direction = "overcut"
                        elif nearest > actual_pit_lap:
                            direction = "undercut"
                        else:
                            direction = "ambiguous"
                    else:
                        direction = "ambiguous"
                else:
                    direction = "ambiguous"

            events.append({
                "Driver": driver,
                "window_start_lap": int(run_start),
                "window_end_lap": int(prev),
                "actual_pit_lap": actual_pit_lap,
                "direction": direction,
            })
            if lap is not None:
                run_start = lap
                prev = lap

    return pd.DataFrame(events)

def main():
    global YEAR, RACE_FASTF1, RACE_TI, SESSION_FASTF1, SESSION_TI

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Race year, e.g. 2019")
    parser.add_argument("--race-fastf1", required=True, help="Underscore-style race name, e.g. Monaco_Grand_Prix")
    parser.add_argument("--race-ti", required=True, help="Space-style race name, e.g. Monaco Grand Prix")
    parser.add_argument("--session-fastf1", default="R", help="fastf1 session code (default R)")
    parser.add_argument("--session-ti", default="Race", help="tracinginsights session name (default Race)")
    parser.add_argument("--inspect", action="store_true", help="Inspect one real telemetry file's schema and exit.")
    args = parser.parse_args()

    YEAR = args.year
    RACE_FASTF1 = args.race_fastf1
    RACE_TI = args.race_ti
    SESSION_FASTF1 = args.session_fastf1
    SESSION_TI = args.session_ti

    race_label = f"{YEAR} {RACE_TI}"

    cb = CachedBucket()

    if args.inspect:
        inspect_one_file(cb)
        return

    print(f"[main] ({race_label}) Loading laps_features.csv ...")
    features_df = load_features(cb)
    features_df = add_cumulative_time(features_df)
    number_to_code = build_number_to_code(features_df)

    print(f"[main] ({race_label}) Building per-lap opponent table (this hits raw telemetry per lap per driver — slow first run)...")
    opponent_df = build_opponent_table(cb, features_df, number_to_code)

    print(f"[main] ({race_label}) Computing empirical pit-lane time loss for this circuit...")
    loss_df, circuit_constant_s = compute_pit_lane_loss(features_df)
    print(f"[main] ({race_label}) empirical pit-lane loss constant (winsorized median): {circuit_constant_s:.2f}s"
          if circuit_constant_s is not None else "[main] Could not compute pit-lane loss constant — see warning above.")

    if circuit_constant_s is not None:
        print(f"[main] ({race_label}) Flagging first-pass undercut windows...")
        flagged = flag_undercut_windows(opponent_df, features_df, circuit_constant_s)
        tag = f"{YEAR}_{RACE_FASTF1}_{SESSION_FASTF1}"
        out_path = f"./checkpoints/rival_knowledge/{tag}_undercut_windows.csv"
        flagged.to_csv(out_path, index=False)
        print(f"[main] Done. {flagged['undercut_opportunity'].sum()} candidate undercut-opportunity rows written to {out_path}")

        events = summarize_undercut_events(flagged, features_df=features_df)
        events_path = f"./checkpoints/rival_knowledge/{tag}_undercut_events.csv"
        events.to_csv(events_path, index=False)
        print(f"[main] Collapsed to {len(events)} distinct candidate events, written to {events_path}")
        print(events.to_string(index=False))

if __name__ == "__main__":
    main()
