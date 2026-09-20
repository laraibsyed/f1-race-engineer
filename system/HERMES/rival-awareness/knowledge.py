"""
Rival Knowledge — POC (proof of concept)
==========================================
Race-agnostic: pass --year, --race-fastf1, --race-ti (and optionally --session-fastf1,
--session-ti) to run against any race in the archive. Validated on 2023 Bahrain first
(see inline notes below for what that validation found); now being scaled to more races
per the handoff's recommended approach — validate on one, then a small spread of races
spanning different eras/circuit types, before trusting the full archive.

Scope for this POC (per project handoff, Section 4 + 5):
  1. Per-lap opponent identity tracking — who is ahead, their compound/tyre age, and the
     real distance to them — using the PRECISE raw-telemetry route (DriverAhead /
     DistanceToDriverAhead), not the cheap Position-based proxy.
  2. Empirical, per-circuit pit-lane time loss constant, derived from this race's real
     in-lap/out-lap data (nothing like this existed in the codebase before — it was only
     a "to manually classify" line item in Master_Checklist.pdf).
  3. A first-pass undercut/overcut window flag, combining 1 + 2.

Design decision carried over from the driver taxonomy work: this module's outputs are
LIVE / rolling by nature (an undercut window only means something at a specific lap),
unlike the taxonomy's static season-level profiles. Everything here is built lap-by-lap,
not aggregated across a career.

Reuses 4 bugs already solved in the driver taxonomy's overtake-defense work
(see handoff Section 3) — do not re-debug these from scratch:
  1. DriverAhead is a car NUMBER; everywhere else uses 3-letter CODES.
  2. Raw LapNumber isn't a valid cross-driver "same moment" reference once someone is
     a lap down — match by cumulative elapsed time, not lap number.
  3. DistanceToDriverAhead can be the literal STRING "None" (crashes float conversion).
  4. .astype(str) before a None check turns real None into the string "None", corrupting
     lookups. Always check `is None` first.

CONFIRMED real schema (via --inspect, 2023 Bahrain, ALB lap 10) — do not re-guess this:
  data = {"tel": {"time": [...], "speed": [...], ..., "DriverAhead": [...],
                  "DistanceToDriverAhead": [...], "dataKey": "..."}}
  "tel" is a DICT OF PARALLEL ARRAYS (one array per channel, all the same length — one
  entry per telemetry sample within the lap), not a list of per-timestamp records.
  DriverAhead values are car-number STRINGS or the literal string "None" (e.g. "31",
  "None"). DistanceToDriverAhead is a parallel array of floats.

VALIDATION SO FAR (2023 Bahrain only): 4 checked events all landed on real, confirmed
pit laps (OCO 12 — Autosport-reported undercut; ZHO 32 — real stop with a strong
reactive-undercut tyre-age signal; ZHO 54 — real stop but a fastest-lap gamble, NOT an
undercut, per F1's own race report; GAS 25 — real stop, but the gap was near-zero,
looking more like traffic-forced than a calculated tyre-offset play; BOT 29 — real stop,
plausible but unconfirmed by any source). Net: the detector reliably lands on real pit
laps, but does NOT distinguish undercut intent from other reasons a stop happened to
match the conditions. Treat it as a pit-timing correlation detector, not a confirmed-
undercut classifier, until/unless that's specifically improved.

Usage:
    python rival_knowledge_poc.py --inspect --year 2023 --race-fastf1 Bahrain_Grand_Prix --race-ti "Bahrain Grand Prix"
    python rival_knowledge_poc.py --year 2023 --race-fastf1 Bahrain_Grand_Prix --race-ti "Bahrain Grand Prix"
    python rival_knowledge_poc.py --year 2019 --race-fastf1 Monaco_Grand_Prix --race-ti "Monaco Grand Prix"
    python rival_knowledge_poc.py --year 2018 --race-fastf1 Abu_Dhabi_Grand_Prix --race-ti "Abu Dhabi Grand Prix"
"""

import os
import json
import argparse
import pandas as pd
import numpy as np

try:
    from google.cloud import storage
except ImportError:
    storage = None  # allows --inspect-less dry runs / local testing without the dependency


# --------------------------------------------------------------------------------------
# Config — set from CLI args in main(), not hardcoded, so scaling to more races doesn't
# require hand-editing this file (a source of real bugs elsewhere in this project when
# "the same file" drifted between chat and disk).
# --------------------------------------------------------------------------------------
BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CACHE_DIR = os.environ.get("GCS_CACHE_DIR", "./gcs_cache")

YEAR = None
RACE_FASTF1 = None    # underscore style — fastf1 paths, clean/features/, clean/tracinginsights/
RACE_TI = None         # space style — raw/tracinginsights/ only
SESSION_FASTF1 = "R"
SESSION_TI = "Race"

# Reliability threshold for any per-driver/per-stint statistic (project rule: small samples
# produce misleading extremes under ~5-15 observations).
MIN_CLEAN_LAPS_FOR_BASELINE = 3


# --------------------------------------------------------------------------------------
# CachedBucket — same read-through pattern used throughout the rest of the project
# --------------------------------------------------------------------------------------
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
        # Never cached — listings are cheap, new data can land anytime (project rule).
        return [b.name for b in self.bucket.list_blobs(prefix=prefix)]

    def exists(self, blob_path):
        """True if the blob exists in GCS — checked live against the bucket, not the
        local cache. The whole point is to survive a wiped or fresh local disk (e.g.
        moving this run to a VM partway through the archive): a local-only check would
        say "not done" even when the real result already lives in the bucket."""
        return self.bucket.blob(blob_path).exists()

    def write_csv(self, blob_path, df):
        """Write-through: writes the CSV to the local cache AND uploads it to GCS,
        mirroring the read-through pattern read_csv/read_text already use. This is how
        a finished race's results get backed up as the archive run progresses, so a
        crash (or a deliberate move to a different machine) doesn't lose completed work
        — the next run checks exists() against the bucket, not just local disk."""
        local = self._local_path(blob_path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        df.to_csv(local, index=False)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_filename(local, content_type="text/csv")


# --------------------------------------------------------------------------------------
# Step 0 — inspect a real raw telemetry file before trusting any assumption about it
# --------------------------------------------------------------------------------------
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

    # Real schema (confirmed): everything lives under a "tel" key, not at the top level.
    # Drill into it and report its shape so we know how to walk it.
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
                # Print a few more items' values for these keys so we can see if they vary
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


# --------------------------------------------------------------------------------------
# Step 1 — load laps_features.csv, build number_to_code, add cumulative elapsed time
# --------------------------------------------------------------------------------------
def load_features(cb):
    path = f"clean/features/{YEAR}/{RACE_FASTF1}/{SESSION_FASTF1}/laps_features.csv"
    df = cb.read_csv(path, dtype={"TrackStatus": str})
    return df


def build_number_to_code(df):
    m = df[["DriverNumber", "Driver"]].drop_duplicates()
    return dict(zip(m["DriverNumber"].astype(str), m["Driver"]))


def lap_time_to_seconds(series):
    """LapTime may come through as a timedelta string (e.g. '0 days 00:01:34.123000')
    or already numeric — handle both rather than assuming."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def add_cumulative_time(df):
    """Bug #2 fix: build a per-driver cumulative elapsed-time clock so we can match
    across drivers by real elapsed time, not raw LapNumber (invalid once someone is
    a lap down)."""
    df = df.sort_values(["Driver", "LapNumber"]).copy()
    df["LapTime_s"] = lap_time_to_seconds(df["LapTime"])
    df["cum_time_s"] = df.groupby("Driver")["LapTime_s"].cumsum()
    return df


# --------------------------------------------------------------------------------------
# Step 2 — per-lap opponent identity from raw telemetry (precise route)
# --------------------------------------------------------------------------------------
def extract_driver_ahead(data):
    """Pulls DriverAhead / DistanceToDriverAhead out of one lap's JSON, applying bugs
    #3 and #4 from the handoff.

    Confirmed real schema (via --inspect): everything lives under a top-level "tel" key,
    which is a DICT OF PARALLEL ARRAYS (one array per channel, e.g. tel["speed"],
    tel["DriverAhead"], tel["DistanceToDriverAhead"], all the same length — one entry per
    telemetry sample within the lap), not a list of per-timestamp records."""
    tel = data.get("tel", {})
    driver_ahead_raw = tel.get("DriverAhead")
    distance_raw = tel.get("DistanceToDriverAhead")

    # Bug #4 fix: check `is None` BEFORE any str conversion.
    if driver_ahead_raw is None:
        return None, None

    if isinstance(driver_ahead_raw, list):
        # Take the last non-null, non-"None"-string sample as representative of the lap.
        # (A median/mode could also be argued for — document whichever you pick.)
        pairs = zip(driver_ahead_raw, distance_raw if isinstance(distance_raw, list) else [distance_raw] * len(driver_ahead_raw))
        clean_pairs = [(d, dist) for d, dist in pairs if d is not None and str(d) != "None"]
        if not clean_pairs:
            return None, None
        driver_ahead_raw, distance_raw = clean_pairs[-1]

    # Bug #3 fix: DistanceToDriverAhead can be the literal string "None".
    if distance_raw is None or str(distance_raw) == "None":
        distance_val = None
    else:
        try:
            distance_val = float(distance_raw)
        except (TypeError, ValueError):
            distance_val = None

    return str(driver_ahead_raw), distance_val


def add_behind_columns(opponent_df, features_df):
    """Derives the car BEHIND each driver via reverse lookup on the already-computed
    'ahead' table — satisfies the "track rival ... gap (car ahead + behind)" checklist
    item WITHOUT any new raw telemetry parsing. If Y's ahead_driver == X on a given
    lap, then X's car-behind is Y, and the gap is the same physical distance already
    captured (from the other car's own telemetry sample, not re-derived from scratch).

    Fast, vectorized — no per-row raw file reads, safe to run on cached opponent
    tables for the whole archive without re-scanning telemetry."""
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
    """One row per (Driver, Lap): who's ahead (code), distance to them, their compound
    and tyre age at that lap. Checkpointed to disk as it goes — expensive per-file work,
    project rule is to never batch-checkpoint only at the end."""
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

        # Checkpoint after each driver, not just at the end.
        pd.DataFrame(rows, columns=OPPONENT_COLUMNS if not rows else None).to_csv(checkpoint_path, index=False)
        print(f"[opponent] Checkpointed after driver {i+1}/{len(drivers)} ({driver})")

    # A genuinely empty result (e.g. a race with zero raw telemetry at all, like the
    # documented 2018 Bahrain gap) must still have the right columns — pd.DataFrame([])
    # has NO columns, which crashes flag_undercut_windows's merge() with KeyError
    # on "Driver" further downstream instead of just producing zero flagged laps.
    if not rows:
        print(f"[opponent] WARNING: zero telemetry reads succeeded for this entire race "
              f"— check whether this is a known data gap (like 2018 Bahrain) or a "
              f"race/session naming mismatch before trusting a downstream 0-event result.")
        return pd.DataFrame(columns=OPPONENT_COLUMNS)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Step 3 — empirical pit-lane time loss for this circuit
# --------------------------------------------------------------------------------------
MAX_PLAUSIBLE_PIT_STOP_S = 120.0  # no real pit stop takes 2+ minutes


def compute_pit_lane_loss(df):
    """Pit-lane time loss = the REAL physical duration spent in the pit lane, computed
    directly as PitOutTime - PitInTime for each matched pit-in -> pit-out lap pair.

    REPLACED the earlier out-lap-LapTime-vs-clean-baseline proxy entirely, after direct
    verification (diagnose_pit_loss.py, 2023 British GP) showed the two measures were
    almost perfectly correlated (r=0.99, n=10 clean stops) but the proxy consistently
    UNDERSTATED the real duration by ~7.8s (median real 28.73s vs median proxy 20.94s).
    Using the real timestamp difference directly is simpler, more physically grounded,
    and removes the fragility the proxy depended on: no 'clean lap baseline' is needed
    at all (no MIN_CLEAN_LAPS_FOR_BASELINE dependency, no degenerate-stint failure mode
    like the one that produced 2023 British GP's near-zero constant), and no risk of
    pooling the incompatible pit_in (~0s) / pit_out (~20s) distributions that caused the
    original outlier investigation to go down the wrong path.

    FIXED (post archive-wide analysis): two races produced absurd 1000+ second
    "durations" (2021 Saudi Arabian GP, 2023 Australian GP) — both well-documented
    red-flag races. Best explanation: a driver's PitInTime recorded just before a red
    flag, PitOutTime not recorded until the race resumed, so the timestamp difference
    captures the entire stoppage, not a pit stop. Rather than try to detect red flags
    specifically (no confirmed track-status column semantics to rely on), this applies
    a hard physical plausibility cap: no real pit stop takes 2+ minutes, so anything
    above MAX_PLAUSIBLE_PIT_STOP_S is dropped before it ever reaches the winsorized
    median — a transparent, circuit-agnostic safeguard rather than a guess at the cause.

    Still excludes SC/VSC-affected stops as a conservative safeguard (see note below on
    why this may matter less for the real-duration method than it did for the old proxy)."""
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
        if isinstance(out_row, pd.DataFrame):  # defensive, in case of any duplicate index
            out_row = out_row.iloc[0]
        if not bool(out_row.get("is_pit_out", False)):
            continue  # the following lap wasn't actually the matching pit-out

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
            continue  # almost certainly spans a red flag or other stoppage, not a real stop

        results.append({"driver": driver, "in_lap": int(in_lap), "out_lap": int(out_lap), "loss_s": duration})

    loss_df = pd.DataFrame(results)
    if loss_df.empty:
        print("[pit_loss] No valid green-flag pit stop pairs found — check is_pit_in/"
              "is_pit_out/PitInTime/PitOutTime exist and aren't all null for this race.")
        return loss_df, None

    # Winsorize before taking a summary constant — a single genuine outlier (e.g. a slow
    # unsafe-release lap, like the 41.98s one seen in the British GP verification)
    # shouldn't distort the circuit-level number.
    p10, p90 = loss_df["loss_s"].quantile([0.10, 0.90])
    winsorized = loss_df["loss_s"].clip(p10, p90)
    circuit_constant_s = winsorized.median()

    return loss_df, circuit_constant_s

    loss_df = pd.DataFrame(results)
    if loss_df.empty:
        print("[pit_loss] No valid pit in/out laps found — check is_pit_in/is_pit_out flags exist and aren't all null.")
        return loss_df, None

    # Winsorize before taking a summary constant — a single genuine outlier (e.g. a slow
    # unsafe-release lap) shouldn't distort the circuit-level number (project rule).
    p10, p90 = loss_df["loss_s"].quantile([0.10, 0.90])
    winsorized = loss_df["loss_s"].clip(p10, p90)
    circuit_constant_s = winsorized.median()

    return loss_df, circuit_constant_s


# --------------------------------------------------------------------------------------
# Step 4 — first-pass undercut/overcut window flag
# --------------------------------------------------------------------------------------
def flag_undercut_windows(opponent_df, features_df, pit_loss_constant_s,
                           window_laps=3, min_tyre_age_gap=3):
    """First-pass rule — REVISED after the initial run on real 2023 Bahrain data
    flagged ~37% of all lap-driver rows as "undercut opportunities" (396/1056).
    That was noise, not signal: v1 evaluated every racing lap of the whole race,
    including grid-formation proximity on lap 1, and the gap threshold (~20s of
    pit-lane loss) is almost always satisfied by normal on-track gaps regardless
    of strategy. Confirmed via real output before changing anything, per project
    rule — never guess a schema/behaviour, verify against real data first.

    Fix: restrict evaluation to laps that are an ACTUAL pit-decision point — within
    `window_laps` laps before a driver's own real pit-in event — since that's the
    only point in the race where "is my gap small enough to undercut" is a real
    question. Also require a meaningful tyre-age gap (not just any positive
    difference), since a 1-lap tyre age edge isn't a real undercut lever.

        An undercut opportunity exists for Driver X at Lap N if:
          - Lap N falls within `window_laps` laps before X's own real pit stop, AND
          - the real gap to the car ahead (converted to an approx time gap) is
            smaller than the pit-lane time loss constant, AND
          - X's tyre is at least `min_tyre_age_gap` laps older than the car ahead's.

    Distance-to-time conversion is still a rough placeholder (flat ~83 m/s / 300 km/h
    reference speed) — fine for this POC, replace with a real speed-at-track-position
    figure before trusting this for anything beyond validation.

    STILL NEEDS VALIDATION: check the flagged rows against a real, known undercut
    event from this race before trusting the window_laps / min_tyre_age_gap values —
    they're reasonable starting guesses, not fitted or validated numbers.
    """
    REFERENCE_SPEED_MPS = 83.0

    merged = opponent_df.merge(
        features_df[["Driver", "LapNumber", "TyreLife", "Compound", "is_pit_in"]],
        on=["Driver", "LapNumber"], how="left"
    )

    # Coerce to numeric — mixed None/float columns are object-dtype, and comparisons
    # on a real None (not NaN) throw instead of just evaluating to False.
    merged["distance_to_ahead_m"] = pd.to_numeric(merged["distance_to_ahead_m"], errors="coerce")
    merged["TyreLife"] = pd.to_numeric(merged["TyreLife"], errors="coerce")
    merged["ahead_tyre_age"] = pd.to_numeric(merged["ahead_tyre_age"], errors="coerce")
    merged["approx_gap_s"] = merged["distance_to_ahead_m"] / REFERENCE_SPEED_MPS
    merged["tyre_age_gap"] = merged["TyreLife"] - merged["ahead_tyre_age"]

    # Build, per driver, the set of (Driver, LapNumber) pairs that fall within
    # window_laps laps before that driver's own real pit-in event.
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
    """Collapses consecutive flagged laps per driver into single candidate events.

    VALIDATED against real 2023 Bahrain data: OCO's reported undercut (Autosport —
    "pitted from 12th and cycled out in sixth", a 2-lap undercut on BOT) shows up as
    undercut_opportunity=True on laps 9 and 10, immediately before OCO's real is_pit_in
    at lap 12. Without this grouping, that single real event would be double-counted
    as two rows; ZHO's laps 51-54 (a fastest-lap gamble, NOT a real undercut per F1's
    own race report) would be quadruple-counted. Collapsing to one row per event fixes
    both, and reporting the actual pit lap alongside each event lets you eyeball
    whether the window's timing makes sense.

    Does NOT distinguish genuine undercut intent from other reasons a stop happened to
    match the conditions (see ZHO lap 54 — flagged, but F1's report says the motive was
    a late fastest-lap attempt, not a tyre-offset undercut). That still needs manual
    annotation against race reports; this function only fixes the double-counting.

    NEW: if features_df is provided, adds a 'direction' column — undercut / overcut /
    ambiguous — a simple pit-order sequencing check against the identified rival, not a
    new detection system:
      - 'undercut': the flagged driver's own pit lap comes BEFORE the rival's next real
        pit stop (they pit first, hoping to gain via fresh tyres before the rival
        reacts) — the classic case, confirmed for Ocon/Bahrain.
      - 'overcut': the rival ALREADY pitted before the flagged driver's own stop (the
        flagged driver stayed out longer, gaining via track position/clean air before
        finally pitting) — confirmed for Gasly/Monaco and Magnussen/Abu Dhabi in the
        single-race validation, both real overcuts the original undercut-shaped rule
        happened to also catch.
      - 'ambiguous': no matching rival pit lap found nearby, or the rival's identity
        wasn't stable across the flagged window.
    """
    events = []
    for driver, g in flagged_df.sort_values(["Driver", "LapNumber"]).groupby("Driver"):
        flagged_laps = g.loc[g["undercut_opportunity"], "LapNumber"].tolist()
        if not flagged_laps:
            continue
        # Group consecutive lap numbers into runs
        run_start = flagged_laps[0]
        prev = flagged_laps[0]
        for lap in flagged_laps[1:] + [None]:
            if lap is not None and lap == prev + 1:
                prev = lap
                continue
            # Run [run_start, prev] ended; find the real pit lap it led into, if any
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


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
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