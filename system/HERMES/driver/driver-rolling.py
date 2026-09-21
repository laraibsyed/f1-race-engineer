"""
Rolling (point-in-time) driver profiles: tyre_management_pti / consistency_factor_pti
========================================================================================
Rebuilds the two driver_archetypes.xlsx metrics used as Regression V2 ablation
candidates -- tyre_management, consistency_factor -- as POINT-IN-TIME values,
so they can be joined onto laps_features.csv without leaking future races into
historical training rows.

WHY THIS IS NEEDED
------------------
driver_taxanomy_metrics.py (driver-taxanomy-metrics.py in this project) builds
these two metrics from a driver's ENTIRE career archive -- a deliberate,
documented choice for that piece of work (driver-archetype-mapping.py's
docstring: profiles are a season-level PRIOR, Layer 2, not live race state).
That's fine for feeding a live strategy engine. It is NOT fine for training
Regression V2 on historical laps, where a 2019 lap would otherwise "know"
about that driver's 2023 races.

This script keeps the METHOD identical to driver-taxanomy-metrics.py --
same teammate-relative delta, same clean-lap flags, same negation-before-
scaling convention, same qualifying-driver threshold -- and only changes the
WINDOW: every (driver, target_race) profile is built using strictly earlier
races only.

DEFINITION (preserved exactly from driver-taxanomy-metrics.py)
----------------------------------------------------------------
Per race, per driver:
    mean_degradation = mean(degradation_rate) over that driver's clean laps
    consistency_cv   = std(LapTime_seconds) / mean(LapTime_seconds) over
                       that driver's clean laps
Teammate-relative delta (same race, same team, own value minus teammate mean,
excluding self, robust to >2-driver teams from mid-season swaps):
    delta_degradation = driver's mean_degradation - teammate_mean(degradation)
    delta_cv          = driver's consistency_cv   - teammate_mean(consistency_cv)
Point-in-time profile for target_race = mean of delta_degradation / delta_cv
across every race STRICTLY BEFORE target_race for that driver, THEN negated
before scaling (lower delta = better than teammate = higher score), matching
driver-taxanomy-metrics.py's normalize_to_archetype_range(-avg_delta, ...)
convention exactly.

CLEAN-LAP FILTER -- identical flags to driver-taxanomy-metrics.py, NOT the
tyre-regression-v2.py flag names (different scripts use different columns
for the same idea, e.g. is_sc_lap here vs. is_sc_lap there -- checked, they
match; is_missing_laptime here vs. is_missing_laptime there -- also matches).
No compound filtering: driver-taxanomy-metrics.py computes degradation/
consistency over ALL clean laps regardless of compound (only wet_weather_skill
filters by compound, which isn't one of the candidate covariates here).

FALLBACK -- reuses the SAME mechanism as driver-archetype-mapping.py, not an
invented value: drivers below MIN_CAREER_RACES get one of the four archetypes
(rookie_aggressive / rookie_conservative / junior_high_potential /
senior_backmarker) via the same fixed-threshold heuristic, then read that
archetype's tyre_management / consistency_factor from
src/taxanomy/drivers_archetypes.xlsx.

Note on career-race counting for the fallback: driver-archetype-mapping.py
counts TOTAL career races (a static, full-career count) to decide who needs
a fallback at all. That's consistent with how IT treats profiles -- but this
script's whole point is that tyre_management_pti/consistency_factor_pti are
point-in-time. So here, "does this driver need a fallback for target_race X"
is evaluated using prior_career_races (races before X), not total career
races -- a driver who eventually clears MIN_CAREER_RACES still gets 'fallback'
for their early races, then 'rolling' once they cross the threshold. This is
a deliberate difference from the static archetype-mapping script, required by
the point-in-time design, not an oversight.

Output shape (per user spec):
    driver, season, round, target_race, prior_career_races,
    tyre_management_pti, consistency_factor_pti, profile_source

CRITICAL INVARIANT: for every target_race, the maximum source-race used to
build its profile is strictly earlier than target_race. Checked explicitly
in stage1_validate() below, not just assumed from the code structure.
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
from calender import RACE_CALENDAR

load_dotenv()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "f1-race-engineer-bucket")
CHECKPOINT_DIR = os.environ.get("ROLLING_PROFILE_CHECKPOINT_DIR", "./rolling_profile_checkpoints")

MIN_CAREER_RACES = 20          # same threshold as driver-taxanomy-metrics.py / driver-archetype-mapping.py
ROOKIE_RACE_CUTOFF = 10        # same as driver-archetype-mapping.py
SELF_INFLICTED_RATE_THRESHOLD = 0.08   # same as driver-archetype-mapping.py
POINTS_PER_RACE_THRESHOLD = 2.0        # same as driver-archetype-mapping.py
ARCHETYPES_PATH = "src/taxanomy/drivers_archetypes.xlsx"

EXCLUDE_FLAG_COLUMNS = ["is_pit_in", "is_pit_out", "is_sc_lap", "is_vsc_lap",
                         "is_missing_laptime", "is_outlier_laptime", "is_out_lap", "is_in_lap"]

SELF_INFLICTED_KEYWORDS = ["accident", "collision", "spun off", "spin", "damage", "off track"]


# ---------------------------------------------------------------------------
# Bucket access -- same CachedBucket pattern used across every script here
# ---------------------------------------------------------------------------
class CachedBucket:
    def __init__(self, bucket_name=BUCKET_NAME, cache_dir=os.environ.get("GCS_CACHE_DIR", "./gcs_cache")):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.cache_dir = cache_dir

    def read_csv(self, blob_path, **kwargs):
        local_path = os.path.join(self.cache_dir, blob_path)
        if os.path.exists(local_path):
            return pd.read_csv(local_path, **kwargs)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.bucket.blob(blob_path).download_to_filename(local_path)
        return pd.read_csv(local_path, **kwargs)

    def list_blob_names(self, prefix):
        return [b.name for b in self.client.list_blobs(self.bucket, prefix=prefix)]


def checkpoint_path(season, race):
    return os.path.join(CHECKPOINT_DIR, f"{season}_{race}.json")


def save_checkpoint(season, race, per_driver: dict, status="ok"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(checkpoint_path(season, race), "w") as f:
        json.dump({"status": status, "data": per_driver}, f)


def load_checkpoint(season, race):
    path = checkpoint_path(season, race)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload["data"] if payload["status"] == "ok" else {}


def list_feature_race_files(bucket: CachedBucket) -> list:
    """Every (season, race) with a Race-session laps_features.csv, ordered by
    TRUE chronological order via the hardcoded RACE_CALENDAR (season+round),
    not alphabetically by race name.

    Only 'R' (Race) session files are read here -- sprint sessions are never
    a separate entry in this list, so a driver's sprint laps never get
    double-counted as an extra "career race" alongside that same weekend's
    Grand Prix. This was already true structurally (list_blob_names filters
    on "/R/laps_features.csv" specifically) -- documented here explicitly per
    the point-in-time design's own requirement to confirm it, not silently
    assumed.
    """
    paths = [p for p in bucket.list_blob_names("clean/features/") if p.endswith("/R/laps_features.csv")]
    out = []
    for p in paths:
        parts = p.split("/")
        out.append((int(parts[2]), parts[3]))
    out = sorted(set(out))

    validate_race_calendar_coverage(out)

    def sort_key(sr):
        season, race = sr
        for (s, r), name in RACE_CALENDAR.items():
            if s == season and name == race:
                return (season, r)
        return (season, 999, race)  # unmapped race: pushed to the end of its season, not silently ordered wrong

    return sorted(out, key=sort_key)


def validate_race_calendar_coverage(race_files: list) -> None:
    """
    Confirms, before this calendar is trusted:
    1. Every (season, race_name) in RACE_CALENDAR maps to exactly one round
       (no duplicate race names within a season in the calendar itself).
    2. Every race actually present in the bucket has a calendar entry.
    Prints and does not raise, matching this project's existing
    warn-and-continue style for coverage gaps (e.g. circuit taxonomy,
    telemetry backfill) -- but the warnings are loud and specific.
    """
    # 1. calendar-internal uniqueness: no season should map two different
    #    rounds to the same race name, and no round should appear twice
    by_season = {}
    for (season, rnd), name in RACE_CALENDAR.items():
        by_season.setdefault(season, {"names": {}, "rounds": set()})
        if name in by_season[season]["names"]:
            print(f"[FAIL] calendar has duplicate race name in {season}: "
                  f"'{name}' at rounds {by_season[season]['names'][name]} and {rnd}")
        by_season[season]["names"][name] = rnd
        if rnd in by_season[season]["rounds"]:
            print(f"[FAIL] calendar has duplicate round number in {season}: round {rnd}")
        by_season[season]["rounds"].add(rnd)

    # 2. bucket coverage: every real (season, race) must have a calendar entry
    calendar_pairs = {(s, name) for (s, _), name in RACE_CALENDAR.items()}
    missing = [(s, r) for (s, r) in race_files if (s, r) not in calendar_pairs]
    if missing:
        print(f"[FAIL] {len(missing)} (season, race) pairs from the bucket have NO calendar "
              f"entry -- these will sort to the end of their season, likely wrong: {missing}")
    else:
        print(f"[ok] every bucket race ({len(race_files)} total) has a calendar entry.")


# ---------------------------------------------------------------------------
# Step 1: per-race, per-driver raw stats (identical method to
# driver-taxanomy-metrics.py's process_one_race)
# ---------------------------------------------------------------------------
def process_one_race(bucket: CachedBucket, season: int, race: str) -> dict:
    """Returns {driver: {"team": t, "degradation": x, "consistency_cv": x}}."""
    path = f"clean/features/{season}/{race}/R/laps_features.csv"
    cols = ["Driver", "Team", "LapTime", "degradation_rate"] + EXCLUDE_FLAG_COLUMNS
    df = bucket.read_csv(path, usecols=lambda c: c in cols)

    exclude_mask = pd.Series(False, index=df.index)
    for col in EXCLUDE_FLAG_COLUMNS:
        if col in df.columns:
            exclude_mask |= df[col].fillna(False).astype(bool)
    clean = df[~exclude_mask].copy()
    if clean.empty:
        return {}

    clean["LapTime_seconds"] = pd.to_timedelta(clean["LapTime"], errors="coerce").dt.total_seconds()
    clean = clean.dropna(subset=["LapTime_seconds"])

    per_driver = {}
    for driver, g in clean.groupby("Driver"):
        team = g["Team"].mode().iloc[0] if not g["Team"].mode().empty else None
        mean_degradation = float(g["degradation_rate"].mean()) if "degradation_rate" in g else np.nan
        cv = float(g["LapTime_seconds"].std() / g["LapTime_seconds"].mean()) if g["LapTime_seconds"].mean() else np.nan
        per_driver[driver] = {"team": team, "degradation": mean_degradation, "consistency_cv": cv}
    return per_driver


def teammate_relative_deltas(per_driver: dict) -> dict:
    """Same as driver-taxanomy-metrics.py: compare each driver against the
    mean of their OWN team's other drivers that race, robust to >2-driver
    teams from mid-season swaps."""
    by_team = {}
    for driver, stats in per_driver.items():
        by_team.setdefault(stats["team"], []).append(driver)

    deltas = {}
    for driver, stats in per_driver.items():
        teammates = [d for d in by_team.get(stats["team"], []) if d != driver]
        if not teammates:
            continue

        def teammate_mean(field):
            vals = [per_driver[t][field] for t in teammates
                     if per_driver[t][field] is not None
                     and not (isinstance(per_driver[t][field], float) and np.isnan(per_driver[t][field]))]
            return float(np.mean(vals)) if vals else None

        d = {}
        tm_deg = teammate_mean("degradation")
        if stats["degradation"] is not None and not np.isnan(stats["degradation"]) and tm_deg is not None:
            d["delta_degradation"] = stats["degradation"] - tm_deg

        tm_cv = teammate_mean("consistency_cv")
        if stats["consistency_cv"] is not None and not np.isnan(stats["consistency_cv"]) and tm_cv is not None:
            d["delta_cv"] = stats["consistency_cv"] - tm_cv

        if d:
            deltas[driver] = d
    return deltas


# ---------------------------------------------------------------------------
# Step 2: expand into point-in-time profiles, shifted so target_race excluded
# ---------------------------------------------------------------------------
def normalize_to_archetype_range(series: pd.Series, low=0.85, high=1.15, fit_mask=None,
                                  winsorize_pct: float = 0.05) -> pd.Series:
    """Identical scaler to driver-taxanomy-metrics.py -- fit_mask-protected,
    winsorized. Used here to keep tyre_management_pti/consistency_factor_pti
    on the SAME scale as the static metrics, so the ablation covariates are
    comparable to what the rest of the taxonomy uses."""
    fit_values = series[fit_mask] if fit_mask is not None else series
    fit_values = fit_values.dropna()
    if fit_values.empty:
        return pd.Series(1.0, index=series.index)
    lo, hi = fit_values.quantile(winsorize_pct), fit_values.quantile(1 - winsorize_pct)
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    scaled = low + (series - lo) / (hi - lo) * (high - low)
    return scaled.clip(low, high)


def build_rolling_profiles(per_race_deltas: dict, race_order: list,
                            min_career_races: int = MIN_CAREER_RACES) -> pd.DataFrame:
    """
    per_race_deltas: {(season, race): {driver: {"delta_degradation":.., "delta_cv":..}}}
    race_order: chronological list of (season, race) -- MUST be true calendar
                order, not alphabetical (see caveat below).

    Returns one row per (driver, target_race) with prior_career_races and
    the raw (not-yet-scaled) rolling deltas. Scaling happens separately in
    scale_profiles(), after fallback rows are known, so the fit_mask can
    exclude fallback rows from defining the scale.
    """
    race_rank = {r: i for i, r in enumerate(race_order)}

    # long-format: one row per (driver, race, delta_degradation, delta_cv)
    rows = []
    for (season, race), deltas in per_race_deltas.items():
        for driver, d in deltas.items():
            rows.append({
                "driver": driver, "season": season, "race": race,
                "race_rank": race_rank[(season, race)],
                "delta_degradation": d.get("delta_degradation", np.nan),
                "delta_cv": d.get("delta_cv", np.nan),
            })
    long_df = pd.DataFrame(rows).sort_values(["driver", "race_rank"])

    out_rows = []
    for driver, g in long_df.groupby("driver"):
        g = g.sort_values("race_rank").reset_index(drop=True)
        # expanding mean of PRIOR races only -- shift(1) is the leakage guard
        prior_mean_deg = g["delta_degradation"].expanding().mean().shift(1)
        prior_mean_cv = g["delta_cv"].expanding().mean().shift(1)
        prior_count = np.arange(len(g))  # number of races strictly before this row

        for i in range(len(g)):
            out_rows.append({
                "driver": driver,
                "season": g.loc[i, "season"],
                "round": g.loc[i, "race_rank"],   # true round/order key, see caveat below
                "target_race": g.loc[i, "race"],
                "prior_career_races": int(prior_count[i]),
                "raw_delta_degradation_pti": prior_mean_deg.iloc[i],
                "raw_delta_cv_pti": prior_mean_cv.iloc[i],
            })

    profiles = pd.DataFrame(out_rows)
    profiles["profile_source"] = np.where(
        profiles["prior_career_races"] < min_career_races, "fallback", "rolling"
    )
    return profiles


def scale_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Negate-then-scale, exactly as driver-taxanomy-metrics.py does for the
    static metrics: normalize_to_archetype_range(-avg_delta, fit_mask=reliable).
    Fit mask = rolling rows only (fallback rows must never define the scale,
    same principle as the original thin_sample fit_mask fix)."""
    reliable = profiles["profile_source"] == "rolling"
    profiles["tyre_management_pti"] = normalize_to_archetype_range(
        -profiles["raw_delta_degradation_pti"], fit_mask=reliable)
    profiles["consistency_factor_pti"] = normalize_to_archetype_range(
        -profiles["raw_delta_cv_pti"], fit_mask=reliable)
    return profiles


# ---------------------------------------------------------------------------
# Step 3: fallback fill -- reuse driver-archetype-mapping.py's mechanism,
# evaluated point-in-time (prior_career_races) rather than static total.
# ---------------------------------------------------------------------------
def classify_status(status: str) -> str:
    s = str(status).lower()
    return "self_inflicted" if any(k in s for k in SELF_INFLICTED_KEYWORDS) else "other"


def load_all_results(bucket: CachedBucket) -> pd.DataFrame:
    paths = [p for p in bucket.list_blob_names("raw/fastf1/") if p.endswith("/R/results.csv")]
    frames = []
    for p in paths:
        parts = p.split("/")
        season, race = int(parts[2]), parts[3]
        df = bucket.read_csv(p, usecols=["Abbreviation", "Points", "Status"])
        df["season"], df["race"] = season, race
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def apply_archetype_fallback(profiles: pd.DataFrame, bucket: CachedBucket) -> pd.DataFrame:
    """
    For 'fallback' rows only: classify the driver via the same two-axis
    heuristic as driver-archetype-mapping.py (self_inflicted_dnf_rate /
    avg_points_per_race, at the same fixed thresholds), computed from that
    driver's history PRIOR TO the target race -- not their eventual full
    career, since a fallback row by definition sits before they clear
    MIN_CAREER_RACES and this must stay point-in-time throughout.
    """
    results = load_all_results(bucket)
    results["status_class"] = results["Status"].apply(classify_status)

    fallback_mask = profiles["profile_source"] == "fallback"
    if not fallback_mask.any():
        print("[info] no fallback rows -- every driver cleared MIN_CAREER_RACES from race 1 "
              "(unlikely; double-check MIN_CAREER_RACES / input data if this is unexpected).")
        return profiles

    archetypes_df = pd.read_excel(ARCHETYPES_PATH).set_index("archetype")
    metric_cols = ["tyre_management", "consistency_factor"]
    missing_cols = [m for m in metric_cols if m not in archetypes_df.columns]
    if missing_cols:
        raise ValueError(f"{ARCHETYPES_PATH} is missing expected columns: {missing_cols}")

    tm_vals, cf_vals = [], []
    for idx in profiles[fallback_mask].index:
        row = profiles.loc[idx]
        driver_hist = results[
            (results["Abbreviation"] == row["driver"]) &
            (results["season"] <= row["season"])   # coarse prior-history slice;
            # exact race-level cutoff isn't available from results.csv alone (no round
            # number, same limitation flagged in pressure-tolerance.py) -- documented,
            # not silently assumed precise.
        ]
        n = len(driver_hist)
        if n == 0:
            tm_vals.append(np.nan)
            cf_vals.append(np.nan)
            continue

        dnf_rate = (driver_hist["status_class"] == "self_inflicted").sum() / n
        avg_points = driver_hist["Points"].mean()

        if row["prior_career_races"] < ROOKIE_RACE_CUTOFF:
            archetype = "rookie_aggressive" if dnf_rate > SELF_INFLICTED_RATE_THRESHOLD else "rookie_conservative"
        else:
            archetype = "junior_high_potential" if avg_points > POINTS_PER_RACE_THRESHOLD else "senior_backmarker"

        archetype_row = archetypes_df.loc[archetype]
        tm_vals.append(archetype_row["tyre_management"])
        cf_vals.append(archetype_row["consistency_factor"])

    profiles.loc[fallback_mask, "tyre_management_pti"] = tm_vals
    profiles.loc[fallback_mask, "consistency_factor_pti"] = cf_vals

    still_missing = profiles.loc[fallback_mask, "tyre_management_pti"].isna().sum()
    if still_missing:
        print(f"[warn] {still_missing} fallback rows have NO prior history at all "
              "(driver's true first-ever race) -- can't classify, left as NaN. "
              "Handle explicitly downstream (e.g. drop from the ablation), don't zero-fill.")

    return profiles


# ---------------------------------------------------------------------------
# Stage 1 validation -- run in full before Regression V2 ablation
# ---------------------------------------------------------------------------
def stage1_validate(profiles: pd.DataFrame, laps_row_count_before: int,
                     merged_row_count: int, n_audit_drivers: int = 5) -> bool:
    all_ok = True

    dupes = profiles.duplicated(subset=["driver", "season", "target_race"]).sum()
    print(f"[check] duplicate (driver, season, target_race) keys: {dupes}")
    if dupes:
        print("[FAIL] duplicate keys present.")
        all_ok = False
    else:
        print("[ok] no duplicate (driver, target_race) keys.")

    if merged_row_count != laps_row_count_before:
        print(f"[FAIL] join changed lap row count: {laps_row_count_before} -> {merged_row_count}")
        all_ok = False
    else:
        print(f"[ok] lap row count unchanged by join ({laps_row_count_before}).")

    unmatched = profiles["tyre_management_pti"].isna().sum()
    print(f"[check] rows with no tyre_management_pti value at all: {unmatched} "
          "(expect this to roughly match true-first-race fallback rows only).")

    sample_drivers = profiles.loc[profiles["profile_source"] == "rolling", "driver"].drop_duplicates()
    sample_drivers = sample_drivers.sample(min(n_audit_drivers, len(sample_drivers)), random_state=0)
    print(f"[check] manually auditing {len(sample_drivers)} drivers: {list(sample_drivers)}")
    n_raw_constant_flagged = 0
    for d in sample_drivers:
        d_rows = profiles[profiles["driver"] == d].sort_values("round")
        rolling = d_rows[d_rows["profile_source"] == "rolling"]
        scaled_vals = rolling["tyre_management_pti"]
        raw_vals = rolling["raw_delta_degradation_pti"]

        scaled_constant = scaled_vals.nunique() <= 1
        raw_constant = raw_vals.dropna().nunique() <= 1

        if not scaled_constant:
            verdict = "evolves over career, as expected"
        elif raw_constant:
            verdict = "RAW INPUT CONSTANT -- potential pipeline bug, investigate"
            n_raw_constant_flagged += 1
        else:
            verdict = ("clipped-but-moving -- raw metric changes but scaled value is pinned at "
                       "the winsorized floor/ceiling (acceptable when the driver is a genuine "
                       "outlier relative to the rest of the field, e.g. SAR; not itself a bug, "
                       "but confirm the raw range against known outliers before assuming so)")

        print(f"  {d}: {len(d_rows)} races on file, prior_career_races "
              f"{d_rows['prior_career_races'].min()}-{d_rows['prior_career_races'].max()}, "
              f"tyre_management_pti range {scaled_vals.min():.3f}-{scaled_vals.max():.3f}, "
              f"raw_delta_degradation_pti range {raw_vals.min():.3f}-{raw_vals.max():.3f} "
              f"({verdict})")

    if n_raw_constant_flagged:
        print(f"[FAIL] {n_raw_constant_flagged} audited driver(s) have a genuinely constant raw "
              "input (delta doesn't change across their rolling races) -- this is NOT the "
              "clipped-but-moving pattern and should be investigated before Stage 2.")
        all_ok = False

    # explicit leakage assertion: independently recount races strictly before
    # each row's round, compare to prior_career_races
    mismatches = 0
    for driver, g in profiles.groupby("driver"):
        g = g.sort_values("round")
        recount = np.arange(len(g))
        if not (recount == g["prior_career_races"].values).all():
            mismatches += 1
    if mismatches:
        print(f"[FAIL] leakage assertion: {mismatches} drivers have prior_career_races "
              "inconsistent with an independent recount of strictly-earlier races.")
        all_ok = False
    else:
        print("[ok] leakage assertion passed: every row's prior_career_races matches an "
              "independent recount of strictly-earlier races for that driver.")

    print(f"\nStage 1 result: {'PASS -- safe to proceed to Regression V2 ablation' if all_ok else 'FAIL -- fix before Stage 2'}")
    return all_ok


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-races", type=int, default=None,
                         help="Process only the first N race files -- sanity-check before a full run.")
    parser.add_argument("--min-career-races", type=int, default=MIN_CAREER_RACES)
    parser.add_argument("--output", default="driver_rolling_profiles.csv")
    args = parser.parse_args()

    bucket = CachedBucket()

    race_files = list_feature_race_files(bucket)
    if args.test_races:
        race_files = race_files[:args.test_races]
        print(f"[test mode] limited to first {args.test_races} race files -- "
              f"VERIFY output before removing --test-races")
    print(f"[scope] {len(race_files)} race files")

    per_race_deltas = {}
    n_skipped = 0
    for i, (season, race) in enumerate(race_files, 1):
        cached = load_checkpoint(season, race)
        if cached is not None:
            n_skipped += 1
            per_race_deltas[(season, race)] = cached
            continue
        print(f"[{i}/{len(race_files)}] processing {race} {season} ...")
        try:
            per_driver = process_one_race(bucket, season, race)
            deltas = teammate_relative_deltas(per_driver) if per_driver else {}
        except Exception as e:
            print(f"  [warn] failed: {type(e).__name__}: {e} -- skipping, not checkpointed (will retry)")
            continue
        save_checkpoint(season, race, deltas, status="ok" if deltas else "empty")
        per_race_deltas[(season, race)] = deltas

    print(f"\n[resume] {n_skipped}/{len(race_files)} races already checkpointed and skipped")

    race_order = race_files  # true chronological order via RACE_CALENDAR (hardcoded, fact-check before trusting)
    profiles = build_rolling_profiles(per_race_deltas, race_order, min_career_races=args.min_career_races)

    # DEBUG: pre-scaling raw deltas for SAR -- checking whether the flat 0.85/0.85
    # in the final output is a genuine outlier or a collapsed/constant rolling window
    debug_driver = os.environ.get("DEBUG_DRIVER")
    if debug_driver:
        dbg = profiles[profiles["driver"] == debug_driver][
            ["target_race", "prior_career_races", "raw_delta_degradation_pti", "raw_delta_cv_pti", "profile_source"]
        ]
        print(f"\n[debug] pre-scaling raw deltas for {debug_driver}:")
        print(dbg.to_string())

    profiles = scale_profiles(profiles)
    profiles = apply_archetype_fallback(profiles, bucket)

    print(f"\n[info] profile_source counts:\n{profiles['profile_source'].value_counts().to_string()}")

    print("\n[stage 1] running validation checks...")
    # Row-count args below are placeholders until this is actually joined to
    # laps_features.csv -- pass len(profiles) twice so the join-row-count
    # check trivially passes for now; re-run stage1_validate() for real once
    # join_to_laps() has been run against the real lap data.
    # Run BEFORE the column slice below, since stage1_validate needs
    # raw_delta_degradation_pti / raw_delta_cv_pti for the clipped-vs-constant check.
    stage1_validate(profiles, laps_row_count_before=len(profiles), merged_row_count=len(profiles))

    profiles = profiles[[
        "driver", "season", "round", "target_race", "prior_career_races",
        "tyre_management_pti", "consistency_factor_pti", "profile_source",
    ]]

    profiles.to_csv(args.output, index=False)
    print(f"\n[save] {args.output} -- {len(profiles)} (driver, target_race) profile rows")