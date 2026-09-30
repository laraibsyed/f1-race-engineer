#!/usr/bin/env python3
"""
HERMES Master Orchestrator
============================================================================
ONE file that joins the Gate Tree (Tier 1 -> Tier 2 -> Tier 3), the
Execution Tree, the tyre-life projection feed, the SC/VSC gamble evaluator
and the weather crossover modules into a single per-lap, per-driver
pit/pace recommendation. Built directly against HERMES_MASTER_BLUEPRINT.md
and HERMES_CODE_BUNDLE.md (2026-09-26 snapshot of the repo) - every design
choice below traces back to a specific section of that blueprint, quoted
in comments as "BP §x.y".

WHAT THIS DOES NOT DO (BP §11 open questions, decided here so this file is
actually runnable rather than blocked on you):
  1. One entry file that LOADS the existing tree/module files (BP §11 Q1
     recommendation) - none of your tree logic is duplicated or rewritten
     here, it's imported via a hyphen-safe loader and called as-is.
  2. Replay-only (BP §11 Q2) - this reads real historical laps_features.csv
     rows and evaluates the trees against them, then reports what HERMES
     would have recommended alongside what actually happened (is_pit_in).
     It does NOT simulate a counterfactual race forward - "live-style"
     streaming is a thin follow-up (see LiveSource stub at the bottom).
  3. D1/D2 are whatever two driver codes you pass on the CLI (BP §11 Q3)
     - no hardcoded team.
  4. 2026 mandatory-compound rule: BP §7-B9 flags a contradiction between
     f1_tyre_constraints.json (0 mandatory compounds in 2026) and
     gate-tier-2.py (always requires 2, era-independent). This file uses
     gate-tier-2.py AS WRITTEN (trusts the code, not the JSON) because
     that's the one module actually wired into the tree - flagged loudly
     in the --selftest output and in every 2026 lap's data_quality_notes
     so you see it, not so it's silently "resolved".
  5. Definitions for can_delay_one_lap / overcut_opportunity /
     rival_undercut_threat / unsafe_weather / tyre_structurally_damaged
     (BP §11 Q5): implemented exactly per the proposals in BP §8.3, each
     marked with "# ASSUMPTION (BP §8.3)" at its definition. Nothing here
     was invented outside what the blueprint already proposed.
  6. expected_stint_length (BP §11 Q6): historical median n_laps_true from
     cliff_detection_stints.csv for (compound, circuit, era) if available,
     falling back to (compound, era), falling back to a rough Pirelli
     lap_survival table (flagged approximate - the exact C3/C4/C5->
     compound-name mapping isn't in the blueprint, see PIRELLI_FALLBACK).
  7. PIT_FLEXIBLE vs PIT_NOW (BP §11 Q7): kept as the tree files already
     treat it - PIT_FLEXIBLE is fed into the Execution Tree exactly like a
     Tier-1 PIT_NOW (BP §4.1 quirk (c), NOT changed here, since changing it
     means changing your execution-tree.py, out of scope for a join file).
  8. SC-gamble pit loss (BP §11 Q8): uses the real per-circuit empirical
     constant from archive_per_race_analysis.csv when available, else the
     22.0/11.0 assumptions already in sc-gamble.py.
  9. Second-Driver priority (BP §11 Q9): v1 as the blueprint recommends -
     priority_driver_id is always None (Execution Tree's own default:
     lower gate-tree tier, then "Driver 1 Priority"). The reward-based v2
     comparison (compute_team_reward + standings + risk) is a real,
     separate piece of work and is deliberately NOT bolted on here half
     finished - see `resolve_second_driver_priority_v1` for the extension
     point if you build it later.

Known, inherited limitations this file does NOT try to paper over (each
one is a BUILD/ADAPT item from BP §5 the blueprint itself says has no real
producer yet):
  - driver_stress_signal (BP §7-B6, updated): the mapping pipeline
    (`estimate_lap_start_times`, `match_messages_to_nearest_lap` from
    notebooks/03-nlp-classifier.py) IS lifted into this file now - see
    `load_radios_for_race`/`estimate_lap_start_times`/
    `build_stress_lookup_for_driver` below - rather than left as a stub.
    It is still an approximation, for the same reason the original notebook
    flags it as one: LapStartDate is null for most seasons, so lap starts
    are estimated from the earliest radio message timestamp of the race
    plus cumulative LapTime, which drifts increasingly late through any
    red-flag race (stoppage time isn't in LapTime). The "nearest lap can be
    the next lap" leakage risk the blueprint calls out is handled by only
    ever looking at matched laps <= the lap currently being evaluated
    (STRESS_LOOKBACK_LAPS's window is `(lap - lookback, lap]`, never
    forward). If classified_radios.csv isn't present, or has no rows for
    this race, this honestly returns False for every lap - never fabricated.
  - tyre_structurally_damaged has no live sensor feed; defaults False
    unless you pass a per-lap override.
  - undercut/overcut/rival-threat use `gap_to_car_ahead`/Position from
    laps_features.csv (a lap-boundary proxy, BP §2.3/§7-B14), NOT the raw
    telemetry archive - deliberately, since the telemetry JSON (~200k
    files) isn't something a join file should require just to run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
# 0. CONFIG (BP §8.1 step 0)
# ============================================================================
HORIZON_LAPS = 5                       # matches cliff_horizon_laps, SC prior HORIZON_LAPS,
                                        # sc-gamble n_laps_horizon convention (BP §9)
CLOSE_FOLLOWING_SECONDS = 1.0          # ADAPT (BP §8.3 in_dirty_air)
UNDERCUT_MIN_TYRE_AGE_GAP = 3          # ADAPT (BP §8.3 live_undercut)
SC_GAMBLE_MODE = os.environ.get("HERMES_SC_GAMBLE", "analytic")
                                        # "analytic" = validated closed-form reference (DEFAULT).
                                        # "mc" = Monte Carlo uncertainty propagation around that SAME
                                        # cost model (sc-gamble.py: evaluate_sc_gamble_mc). Same inputs,
                                        # same return keys (MC adds distribution fields), same
                                        # WAIT / NO_ADVANTAGE_TO_WAITING / INSUFFICIENT_DATA vocabulary.
                                        # Override with env HERMES_SC_GAMBLE or `replay --sc-gamble`.
SC_GAMBLE_MODES = ("analytic", "mc")
DEFAULT_PIT_LOSS_SECONDS = 22.0        # sc-gamble.py's own NORMAL_PIT_LOSS_SECONDS default
DELAY_MARGIN_SECONDS = 2.0             # ASSUMPTION (BP §8.3 can_delay_one_lap)
STRESS_LOOKBACK_LAPS = 3               # BP §8.3 stress_trigger default (sweep used 1/3/5/7)
RADIOS_CSV_RELATIVE = Path("data") / "external" / "team-radios" / "classified_radios.csv"

DRY_COMPOUNDS = {"HARD", "MEDIUM", "SOFT", "HYPERSOFT", "SUPERSOFT", "ULTRASOFT"}
WET_COMPOUNDS = {"WET", "INTERMEDIATE"}

# Rough fallback only - the exact C3/C4/C5 -> HARD/MEDIUM/SOFT mapping is
# race-weekend-relative (BP §4.5 taxonomy note 3) and NOT resolvable in
# general, so this is a flagged approximation of last resort, used only
# when cliff_detection_stints.csv has no data for (compound, circuit, era)
# OR (compound, era). See expected_stint_length().
PIRELLI_LAP_SURVIVAL_FALLBACK = {
    "SOFT": 25, "HYPERSOFT": 22, "SUPERSOFT": 23, "ULTRASOFT": 24,
    "MEDIUM": 35, "HARD": 45, "INTERMEDIATE": 45, "WET": 60,
}

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", ".")).resolve()
TREES_DIR = REPO_ROOT / "system" / "HERMES" / "trees"
SC_DIR = REPO_ROOT / "system" / "HERMES" / "safety-car"
WEATHER_DIR = REPO_ROOT / "system" / "HERMES" / "weather"
RIVAL_DIR = REPO_ROOT / "system" / "HERMES" / "rival-awareness"
CACHE_DIR = Path(os.environ.get("GCS_CACHE_DIR", REPO_ROOT / "gcs_cache"))


def note(msg: str) -> None:
    print(f"[hermes] {msg}", file=sys.stderr)


# ============================================================================
# 1. hyphen-safe loader (BP §3 point 2 / §7-B3)
# ============================================================================
def load_module(path: Path, name: str, required: bool = True):
    """importlib.util loader for files that can't be `import`ed because of
    hyphens in the filename. The tree files' `if __name__ == "__main__":`
    blocks are guarded, so this is side-effect free for all of them (BP
    §3 point 2), EXCEPT sc-vsc-probability-model.py, which imports
    google.cloud.storage / dotenv at module level - that one is
    deliberately NOT loaded here; we read its validated output CSV
    directly instead (BP §4.3 "avoid the GCS import")."""
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"HERMES orchestrator needs '{path.name}' at {path} and it "
                f"isn't there. Set HERMES_REPO_ROOT or check your tree "
                f"directory layout matches system/HERMES/... ."
            )
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate1 = load_module(TREES_DIR / "gate-tier-1.py", "gate_tier_1")
gate2 = load_module(TREES_DIR / "gate-tier-2.py", "gate_tier_2")
gate3 = load_module(TREES_DIR / "gate-tier-3.py", "gate_tier_3")
exec_tree = load_module(TREES_DIR / "execution-tree.py", "execution_tree")
tyre_proj = load_module(TREES_DIR / "tyre_life_projection.py", "tyre_life_projection")
sc_gamble = load_module(SC_DIR / "sc-gamble.py", "sc_gamble")


def set_sc_gamble_mode(mode: str) -> None:
    """Validates and sets which SC-gamble evaluator run_sc_gamble() uses.
    Fails loudly on a typo or a sc-gamble.py that predates the MC evaluator,
    rather than silently falling back to the other one."""
    global SC_GAMBLE_MODE
    if mode not in SC_GAMBLE_MODES:
        raise ValueError(f"SC gamble mode must be one of {SC_GAMBLE_MODES}, got {mode!r}")
    if mode == "mc" and not hasattr(sc_gamble, "evaluate_sc_gamble_mc"):
        raise RuntimeError("SC gamble mode 'mc' requested but sc-gamble.py has no evaluate_sc_gamble_mc")
    SC_GAMBLE_MODE = mode


def run_sc_gamble(inputs):
    """The ONE call site for the SC gamble. Both evaluators take the same
    SCGambleInputs and return a dict with the same core keys; the tree only
    reads ["recommendation"] from it."""
    if SC_GAMBLE_MODE == "mc":
        return sc_gamble.evaluate_sc_gamble_mc(inputs)
    return sc_gamble.evaluate_sc_gamble(inputs)


set_sc_gamble_mode(SC_GAMBLE_MODE)   # validate the env/default value at import time

# Weather + risk modules are soft constraints only - the system must still
# run (with unsafe_weather always False, no risk-based priority) if they
# aren't present, so these are optional loads.
crossover = load_module(WEATHER_DIR / "crossover.py", "crossover", required=False)
drying_line = load_module(WEATHER_DIR / "drying_line.py", "drying_line", required=False)
risk_mod = load_module(RIVAL_DIR / "risk.py", "risk", required=False)

if crossover is None:
    note("weather/crossover.py not found - unsafe_weather will only ever "
         "consider 'raining now while on slicks', never the forecast state.")
if risk_mod is None:
    note("rival-awareness/risk.py not found - Second-Driver risk appetite unavailable (v1 doesn't need it).")


# ============================================================================
# 2. B1 FIX - the confirmed silent bug in tyre_life_projection.py
# ============================================================================
# BP §7-B1: the pickle's temp_dummy_columns are built with prefix="temp"
# (e.g. "temp_hot"), but the shipped _regression_design_row/_cox_design_row
# compare col == f"track_temp_bucket_{bucket}", which never matches -> all
# temp dummies are always 0 -> cool/warm/hot/extreme give IDENTICAL output.
# Fix: match on "temp_{bucket}" instead. regulation_era_ matching is
# already correct and is left untouched.
#
# This is done by monkeypatching the two module-level helper functions
# AFTER loading the file (rather than hand-editing your source), so this
# orchestrator is the single place the fix lives and re-running --selftest
# proves it's actually applied every time this file runs.
def _fixed_regression_design_row(tyre_age, fuel_load_estimate, stint_number,
                                  track_temp_bucket, temp_dummy_columns):
    row = {"tyre_age": tyre_age, "fuel_load_estimate": fuel_load_estimate,
           "stint_number": stint_number}
    for col in temp_dummy_columns:
        row[col] = 1.0 if col == f"temp_{track_temp_bucket}" else 0.0
    return pd.DataFrame([row])[["tyre_age", "fuel_load_estimate", "stint_number"] + temp_dummy_columns]


def _fixed_cox_design_row(fuel_load_estimate, stint_number, circuit_degredation_ordinal,
                           track_temp_bucket, regulation_era, temp_dummy_columns, era_dummy_columns):
    row = {"fuel_load_estimate": fuel_load_estimate, "stint_number": stint_number,
           "circuit_degredation_ordinal": circuit_degredation_ordinal}
    for col in temp_dummy_columns:
        row[col] = 1.0 if col == f"temp_{track_temp_bucket}" else 0.0
    for col in era_dummy_columns:
        row[col] = 1.0 if col == f"regulation_era_{regulation_era}" else 0.0
    return pd.DataFrame([row])


tyre_proj._regression_design_row = _fixed_regression_design_row
tyre_proj._cox_design_row = _fixed_cox_design_row

# BP §7-B13: duplicated constants must be asserted equal at startup, not
# just trusted to stay in sync by hand.
assert gate3.CLIFF_PROBABILITY_THRESHOLD == exec_tree.CLIFF_PROBABILITY_THRESHOLD, (
    f"CLIFF_PROBABILITY_THRESHOLD drift: gate-tier-3.py={gate3.CLIFF_PROBABILITY_THRESHOLD} "
    f"!= execution-tree.py={exec_tree.CLIFF_PROBABILITY_THRESHOLD}. Fix the source files, "
    f"not this assert."
)

# ---------------------------------------------------------------------------
# WALK-FORWARD FOLD OVERRIDE (added for hermes_evaluate.py --fold-mode)
# ---------------------------------------------------------------------------
# HERMES_FOLD_DIR points at a folder produced by fit_fold.py. When set:
#   * the tyre pickle, cliff_detection_stints.csv, SC prior and pit-loss table
#     are read from that folder FIRST (see _data_path below), and
#   * the three calibrated Tier-3 thresholds come from its thresholds.json
#     instead of the all-data constants baked into gate-tier-3.py.
# When unset, behaviour is byte-for-byte what it was before. A missing or NaN
# threshold in the fold json is an ERROR, never a silent fallback to the
# all-data value (that would reintroduce the leak this exists to remove).
FOLD_DIR = Path(os.environ["HERMES_FOLD_DIR"]).resolve() if os.environ.get("HERMES_FOLD_DIR") else None
FOLD_THRESHOLDS = None
if FOLD_DIR is not None:
    _tj = FOLD_DIR / "thresholds.json"
    if not _tj.exists():
        raise SystemExit(f"HERMES_FOLD_DIR={FOLD_DIR} set but thresholds.json missing")
    FOLD_THRESHOLDS = json.load(open(_tj))
    for _k in ("CLIFF_PROBABILITY_THRESHOLD", "PACE_LOSS_THRESHOLD_SECONDS", "TYRE_AGE_TRIGGER_RATIO"):
        _v = FOLD_THRESHOLDS.get(_k)
        if _v is None or (isinstance(_v, float) and _v != _v):
            raise SystemExit(f"fold threshold {_k} is missing/NaN in {_tj}; refusing to fall back "
                             f"to the all-data value (that would leak). Fix fit_fold.py output.")
        setattr(gate3, _k, float(_v))
    exec_tree.CLIFF_PROBABILITY_THRESHOLD = gate3.CLIFF_PROBABILITY_THRESHOLD   # keep the two in sync
    assert gate3.CLIFF_PROBABILITY_THRESHOLD == exec_tree.CLIFF_PROBABILITY_THRESHOLD
    note(f"FOLD MODE ({FOLD_DIR.name}): thresholds overridden -> {FOLD_THRESHOLDS}")


def _data_path(repo_root: Path, *parts) -> Path:
    """Fold folder first (if set and the file exists there), else the real repo root."""
    if FOLD_DIR is not None:
        cand = FOLD_DIR.joinpath(*parts)
        if cand.exists():
            return cand
        # In fold mode a MISSING fold file must not silently fall back to the all-data file
        # for the leaky artefacts. Those four names are the ones fit_fold.py writes.
        if parts[-1] in ("tyre_life_models.pkl", "cliff_detection_stints.csv",
                         "sc_vsc_circuit_level_prior.csv"):
            raise SystemExit(f"FOLD MODE: {cand} not found; refusing to use the all-data {parts[-1]}")
    return repo_root.joinpath(*parts)
# Note re: BP §4.1 quirk (a), the duplicate get_driving_instruction stub in
# execution-tree.py - harmless at runtime (Python's later `def` simply
# overwrites the stub when the file executes top-to-bottom), so no
# orchestrator-side fix is needed. Still worth deleting the stub for
# cleanliness; not required for correctness.


# ============================================================================
# 3. Static loaders (cached once, all optional except the tyre pickle)
# ============================================================================
def load_tyre_models(path: Path) -> Optional[dict]:
    if not path.exists():
        note(f"tyre_life_models.pkl not found at {path} - every projection will be None "
             f"and Tier 1's cliff check / Tier 3's tyre triggers will never fire.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sc_prior(path: Path) -> Optional[pd.DataFrame]:
    """BP §4.3: read the VALIDATED circuit-level prior CSV directly rather
    than importing sc-vsc-probability-model.py (which pulls in
    google.cloud.storage / dotenv at module level)."""
    if not path.exists():
        note(f"sc_vsc_circuit_level_prior.csv not found at {path} - SC gamble will "
             f"always return INSUFFICIENT_DATA.")
        return None
    return pd.read_csv(path)


def get_sc_probability(circuit_prior: Optional[pd.DataFrame], circuit: str) -> Optional[float]:
    """Re-implements sc-vsc-probability-model.get_sc_probability() against
    the CSV directly (lap_number has no effect, as validated - BP §4.3)."""
    if circuit_prior is None:
        return None
    row = circuit_prior[circuit_prior["circuit"] == circuit]
    if row.empty:
        return None
    return float(row["p_window_horizon"].iloc[0])


def load_cliff_stints(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        note(f"cliff_detection_stints.csv not found at {path} - expected_stint_length "
             f"will fall back to the Pirelli approximation for every stint.")
        return None
    return pd.read_csv(path)


def load_circuit_taxonomy(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        note(f"circuit_taxonomy.xlsx not found at {path} - circuit_degredation_ordinal "
             f"will default to 1 (medium) for every circuit.")
        return None
    try:
        return pd.read_excel(path)
    except ImportError:
        note("openpyxl not installed - can't read circuit_taxonomy.xlsx, defaulting ordinal to 1.")
        return None


def load_pit_loss_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 - genuinely best-effort, never fatal
        note(f"couldn't read archive_per_race_analysis.csv ({e}) - using flat pit-loss assumption.")
        return None


DEGR_ORDINAL_MAP = {"low": 0, "medium": 1, "high": 2}


def circuit_degredation_ordinal_for(taxonomy: Optional[pd.DataFrame], race_folder_name: str) -> tuple:
    """ADAPT: taxonomy.xlsx keys on circuit_name, not the Race folder name
    (BP §2.2/§4.5). The exact CIRCUIT_ID_TO_RACE_NAMES alias table lives in
    model-fit.py, which isn't part of the code bundle handed to this
    orchestrator - so this does a best-effort normalised text match instead
    of guessing the alias table, and says so in the note it returns."""
    if taxonomy is None or "circuit_degredation" not in taxonomy.columns:
        return 1, "no taxonomy loaded - defaulted to medium (1)"
    needle = race_folder_name.replace("_", " ").replace("Grand Prix", "").strip().lower()
    for _, row in taxonomy.iterrows():
        hay = str(row.get("circuit_name", "")).lower()
        if needle and (needle in hay or hay in needle):
            band = str(row.get("circuit_degredation", "medium")).lower()
            return DEGR_ORDINAL_MAP.get(band, 1), f"matched taxonomy row '{row.get('circuit_name')}'"
    return 1, f"no taxonomy match for '{race_folder_name}' - defaulted to medium (1)"


def pit_loss_for_circuit(pit_loss_table: Optional[pd.DataFrame], race_folder_name: str) -> tuple:
    """BP §11 Q8 / §4.5: use the real per-circuit empirical constant if the
    archive file is present and has a recognisable column; else the
    sc-gamble.py assumption. Column names for archive_per_race_analysis.csv
    aren't pinned down in the blueprint, so this tries a few plausible ones
    defensively rather than assuming."""
    if pit_loss_table is None:
        return DEFAULT_PIT_LOSS_SECONDS, "sc-gamble.py assumption (no archive_per_race_analysis.csv)"
    candidates = [c for c in pit_loss_table.columns if "pit_loss" in c.lower()]
    circuit_col = next((c for c in pit_loss_table.columns if c.lower() in ("circuit", "race")), None)
    if not candidates or circuit_col is None:
        return DEFAULT_PIT_LOSS_SECONDS, "archive file present but columns unrecognised - using assumption"
    row = pit_loss_table[pit_loss_table[circuit_col] == race_folder_name]
    if row.empty:
        return DEFAULT_PIT_LOSS_SECONDS, f"no archive row for '{race_folder_name}' - using assumption"
    return float(row[candidates[0]].iloc[0]), f"empirical, from {candidates[0]}"


def load_radios_for_race(csv_path: Path, season: int, race_folder: str) -> Optional[pd.DataFrame]:
    """BP §7-B6 (updated): adapted from notebooks/03-nlp-classifier.py's
    `load_classified_radios`, filtered down to one race so the caller isn't
    loading all 17,619 messages per lookup. `classified_radios.csv`'s
    `race_id` is `<season>_<race name>`; the race-name half's spacing
    convention isn't pinned down in the blueprint, so this tries both an
    underscore and a space match against the Race folder name rather than
    assuming one."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["message_timestamp"] != "message_timestamp"]  # notebook's own embedded-header guard
    df["message_timestamp"] = pd.to_datetime(df["message_timestamp"], format="ISO8601",
                                              utc=True, errors="coerce")
    df["season"] = pd.to_numeric(df["race_id"].astype(str).str.split("_", n=1).str[0], errors="coerce")
    df["race"] = df["race_id"].astype(str).str.split("_", n=1).str[1]
    df["racing_number"] = pd.to_numeric(df["racing_number"], errors="coerce")
    df = df[(df["category"] == "tyre_feedback") & (df["season"] == season)]
    race_variants = {race_folder, race_folder.replace("_", " ")}
    df = df[df["race"].isin(race_variants)]
    return df if not df.empty else None


def estimate_lap_start_times(laps: pd.DataFrame, radios: pd.DataFrame) -> pd.DataFrame:
    """Adapted, near-verbatim, from notebooks/03-nlp-classifier.py's
    `estimate_lap_start_times` (BP §7-B6). FALLBACK time alignment only:
    race_start is approximated as the earliest radio message timestamp for
    this race, and each driver's lap starts are that plus their own
    cumulative LapTime - a proxy, not a real timestamp, and one that drifts
    increasingly late through any red-flag race since stoppage time isn't
    captured in LapTime. Kept as a documented limitation, not hidden."""
    laps = laps.copy()
    if "LapTime_seconds" not in laps.columns:
        laps["LapTime_seconds"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    race_start = radios["message_timestamp"].min()
    laps = laps.sort_values(["Driver", "LapNumber"])
    cum_prior = laps.groupby("Driver")["LapTime_seconds"].cumsum() - laps["LapTime_seconds"].fillna(0)
    laps["LapStartDate_estimated"] = race_start + pd.to_timedelta(cum_prior, unit="s")
    return laps


def build_stress_lookup_for_driver(radios: pd.DataFrame, laps_with_starts: pd.DataFrame,
                                    driver_number: int, driver_code: str) -> pd.DataFrame:
    """Adapted from `match_messages_to_nearest_lap` (BP §7-B6), narrowed to
    one driver and to what get_driver_stress_trigger actually needs: each
    tyre_feedback message's nearest estimated lap start. The blueprint's own
    caveat - 'nearest lap can be the NEXT lap, up to half a lap of
    look-ahead' - is why get_driver_stress_trigger's window is `(lap -
    lookback, lap]` and never looks past the lap currently being evaluated;
    this function itself doesn't filter by time, only the caller does."""
    msgs = radios[radios["racing_number"] == driver_number]
    driver_laps = laps_with_starts[laps_with_starts["Driver"] == driver_code].dropna(
        subset=["LapStartDate_estimated"])
    if msgs.empty or driver_laps.empty:
        return pd.DataFrame(columns=["LapNumber", "category", "stress_level"])
    rows = []
    for _, msg in msgs.iterrows():
        if pd.isna(msg["message_timestamp"]):
            continue
        diffs = (driver_laps["LapStartDate_estimated"] - msg["message_timestamp"]).abs()
        nearest_lap = driver_laps.loc[diffs.idxmin(), "LapNumber"]
        rows.append({"LapNumber": nearest_lap, "category": "tyre_feedback",
                     "stress_level": msg.get("stress_level")})
    return pd.DataFrame(rows)


def build_radios_by_driver_lap(repo_root: Path, laps: pd.DataFrame, ctx: "RaceContext") -> Optional[dict]:
    """Ties the three functions above together for run_replay: loads this
    race's radios (if the CSV exists), estimates lap starts once for the
    whole race, then builds the {driver_code: DataFrame} shape
    get_driver_stress_trigger expects, for D1 and D2 only (no need to match
    every driver on the grid)."""
    radios = load_radios_for_race(repo_root / RADIOS_CSV_RELATIVE, ctx.season, ctx.circuit)
    if radios is None:
        note("classified_radios.csv not found (or no tyre_feedback rows for this race) - "
             "driver_stress_signal will be False all race (BP §7-B6 fallback).")
        return None
    laps_with_starts = estimate_lap_start_times(laps, radios)
    number_map = laps[["Driver", "DriverNumber"]].drop_duplicates().set_index("Driver")["DriverNumber"]
    result = {}
    for code in (ctx.d1_code, ctx.d2_code):
        num = number_map.get(code)
        if num is None or pd.isna(num):
            continue
        result[code] = build_stress_lookup_for_driver(radios, laps_with_starts, int(num), code)
    return result or None


# ============================================================================
# 4. Adapters (BP §8.3 - each proposal implemented exactly, flagged ASSUMPTION)
# ============================================================================
def get_regulation_era(season: int) -> str:
    return gate2.get_regulation_era(season)


def track_temp_bucket_from_c(temp_c: Optional[float]) -> str:
    if temp_c is None or (isinstance(temp_c, float) and np.isnan(temp_c)):
        return "warm"  # ASSUMPTION: mid-band default when weather.csv is missing for this session
    if temp_c <= 25:
        return "cool"
    if temp_c <= 35:
        return "warm"
    if temp_c <= 45:
        return "hot"
    return "extreme"


def fuel_load_estimate(lap_number: int, total_laps: int) -> float:
    """features.py's own flat 110kg linear-burn assumption (BP §2.3) - not
    reinvented here, just reapplied per-lap."""
    if total_laps <= 0:
        return 0.0
    return max(0.0, 110.0 * (1 - (lap_number - 1) / total_laps))


def expected_stint_length(cliff_stints: Optional[pd.DataFrame], compound: str,
                           circuit: str, era: str) -> tuple:
    """BP §8.3: median n_laps_true for (compound, circuit, era), fallback
    (compound, era), fallback Pirelli. Returns (value, note-of-which-source)."""
    if cliff_stints is not None and "n_laps_true" in cliff_stints.columns:
        exact = cliff_stints[(cliff_stints["compound"] == compound)
                              & (cliff_stints["circuit"] == circuit)
                              & (cliff_stints["regulation_era"] == era)]
        if not exact.empty:
            return float(exact["n_laps_true"].median()), "historical median (compound, circuit, era)"
        coarser = cliff_stints[(cliff_stints["compound"] == compound)
                                & (cliff_stints["regulation_era"] == era)]
        if not coarser.empty:
            return float(coarser["n_laps_true"].median()), "historical median (compound, era) - no circuit match"
    fallback = PIRELLI_LAP_SURVIVAL_FALLBACK.get(compound, 30)
    return float(fallback), "Pirelli lap_survival approximation - flagged, not calibrated per this circuit"


def unsafe_weather(rain_now: bool, on_slicks: bool, rain_probability_pct: Optional[float],
                    laps_remaining: int, total_laps: int) -> bool:
    """ASSUMPTION (BP §8.3): raining now while on slicks, OR the forecast
    crossover state has already reached CONSIDER_INTERS/BOX_INTERS while
    it's raining now."""
    if rain_now and on_slicks:
        return True
    if crossover is not None and rain_now and rain_probability_pct is not None and total_laps > 0:
        result = crossover.evaluate_crossover(rain_probability_pct, max(0, laps_remaining), total_laps)
        if result.state in (crossover.CrossoverState.CONSIDER_INTERS, crossover.CrossoverState.BOX_INTERS):
            return True
    return False


def live_undercut_opportunity(gap_ahead_s: Optional[float], own_tyre_age: Optional[float],
                               ahead_tyre_age: Optional[float], pit_loss_s: float) -> bool:
    """ADAPT (BP §5 Tier-3 wiring / §7-B4): the real flag_undercut_windows
    needs hindsight (future pit lap) and can't run live. This is the
    blueprint's own proposed live substitute: gap ahead is smaller than
    the pit-loss cost AND our tyres are meaningfully older."""
    if gap_ahead_s is None or own_tyre_age is None or ahead_tyre_age is None:
        return False
    return gap_ahead_s < pit_loss_s and (own_tyre_age - ahead_tyre_age) >= UNDERCUT_MIN_TYRE_AGE_GAP


def rival_undercut_threat(gap_behind_s: Optional[float], own_tyre_age: Optional[float],
                           behind_tyre_age: Optional[float], pit_loss_s: float) -> bool:
    """BUILD (BP §5): mirror of live_undercut_opportunity using the car
    BEHIND - is it close enough and fresh enough to undercut US."""
    if gap_behind_s is None or own_tyre_age is None or behind_tyre_age is None:
        return False
    return gap_behind_s < pit_loss_s and (own_tyre_age - behind_tyre_age) >= UNDERCUT_MIN_TYRE_AGE_GAP


def overcut_opportunity(ahead_pitted_recently: bool, own_cliff_probability: Optional[float],
                         gap_ahead_s: Optional[float]) -> bool:
    """BUILD (BP §5/§8.3): rival ahead has just pitted (or is assumed to
    imminently), our own tyres are still healthy (cliff prob below the
    calibrated threshold), and we have clean air to extend."""
    cliff_ok = own_cliff_probability is not None and own_cliff_probability < gate3.CLIFF_PROBABILITY_THRESHOLD
    clean_air = gap_ahead_s is None or gap_ahead_s >= CLOSE_FOLLOWING_SECONDS
    return bool(ahead_pitted_recently) and cliff_ok and clean_air


def in_dirty_air(gap_ahead_s: Optional[float]) -> bool:
    """ADAPT (BP §8.3): DistanceToDriverAhead/speed < 1.0s is the real
    telemetry-based rule; using the lap-boundary gap_to_car_ahead proxy
    here since telemetry JSON isn't required by this orchestrator."""
    if gap_ahead_s is None:
        return False
    return gap_ahead_s < CLOSE_FOLLOWING_SECONDS


def can_delay_one_lap(gap_behind_s: Optional[float], pit_loss_s: float,
                       margin: float = DELAY_MARGIN_SECONDS) -> bool:
    """ASSUMPTION (BP §8.3): unknown gap -> assume delaying WOULD cost
    position (the safer default for a mechanical decision), not the
    reverse."""
    if gap_behind_s is None:
        return False
    return gap_behind_s > (pit_loss_s + margin)


def get_driver_stress_trigger(radios_by_driver_lap: Optional[dict], driver_code: str,
                               lap_number: int, lookback: int = STRESS_LOOKBACK_LAPS) -> bool:
    """BP §7-B6 (updated): `radios_by_driver_lap` is
    {driver_code: DataFrame[LapNumber, category, stress_level]}, built by
    `build_radios_by_driver_lap` (which adapts the mapping already written
    in notebooks/03-nlp-classifier.py, rather than reinventing it). If no
    radios data exists for this race, this honestly returns False rather
    than guessing - never a fabricated signal, same convention as the rest
    of the project. Also worth noting (BP §3): stress is validated to have
    NO significant predictive link to pit timing, so this is context for
    the Execution Tree only, never a Tier 3 vote."""
    if not radios_by_driver_lap or driver_code not in radios_by_driver_lap:
        return False
    df = radios_by_driver_lap[driver_code]
    if "LapNumber" not in df.columns:
        return False
    window = df[(df["LapNumber"] > lap_number - lookback) & (df["LapNumber"] <= lap_number)]
    tyre_feedback = window[window.get("category") == "tyre_feedback"]
    return bool(tyre_feedback["stress_level"].isin(["medium", "high"]).any())


def marginal_pace_loss(build_projection, tyre_age: float, horizon: int = HORIZON_LAPS) -> Optional[float]:
    """BP §7-B5 fix: sc-gamble.py's predicted_pace_loss_per_lap is
    documented as per-lap but the regression output is an ABSOLUTE pace
    loss at the current age - multiplying by the horizon overstates the
    waiting cost. Use the marginal slope instead: (loss(age+h)-loss(age))/h,
    floored at 0."""
    p0 = build_projection(tyre_age)["predicted_pace_loss"]
    p1 = build_projection(tyre_age + horizon)["predicted_pace_loss"]
    if p0 is None or p1 is None:
        return None
    return max(0.0, (p1 - p0) / horizon)


def resolve_second_driver_priority_v1(*_args, **_kwargs) -> Optional[str]:
    """BP §11 Q9 v1: always None -> Execution Tree's own default
    resolution (lower gate-tree tier, then "Driver 1 Priority"). The v2
    reward-based comparison (compute_team_reward + standings.py +
    risk.compute_risk_appetite) is real additional work, not a
    ten-minute bolt-on - left as an explicit extension point rather than
    a half-wired guess."""
    return None


# ============================================================================
# 5. Contracts (BP §6)
# ============================================================================
@dataclass
class RaceContext:
    season: int
    circuit: str                    # underscore Race folder name, e.g. Bahrain_Grand_Prix
    total_laps: int
    is_sprint_weekend: bool
    d1_code: str
    d2_code: str
    session: str = "R"
    regulation_era: str = field(init=False)
    circuit_degredation_ordinal: int = 1
    pit_loss_s: float = DEFAULT_PIT_LOSS_SECONDS
    p_sc_5lap: Optional[float] = None

    def __post_init__(self):
        self.regulation_era = get_regulation_era(self.season)


@dataclass
class DriverRuntimeState:
    """Accumulated, mutable per-driver state carried lap-to-lap across the
    replay (BP §5 Loop closure) - NOT a forward simulation of future laps,
    just the running totals a real system would maintain (compound
    history, current clean-stint window)."""
    driver_id: str                      # "D1" / "D2"
    code: str
    compound_history_dry: set = field(default_factory=set)
    used_wet_or_inter: bool = False
    stint_ages: list = field(default_factory=list)
    stint_laptimes: list = field(default_factory=list)
    current_stint: Optional[int] = None
    stops_made: int = 0


# ============================================================================
# 6. The per-lap engine (BP §8.2)
# ============================================================================
def _safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _row_or_none(lap_df: pd.DataFrame, code: str):
    match = lap_df[lap_df["Driver"] == code]
    return match.iloc[0] if not match.empty else None


def _rank_by_gap_to_leader(lap_df: pd.DataFrame) -> pd.DataFrame:
    """BP §7-B14 fallback: Position isn't reliably populated in every
    session/season - derive an order from gap_to_leader when it's missing."""
    if "Position" in lap_df.columns and lap_df["Position"].notna().all():
        return lap_df
    ranked = lap_df.copy()
    ranked["Position"] = ranked["gap_to_leader"].rank(method="first")
    return ranked


def build_projection_fn(tyre_models: Optional[dict], compound: str, circuit: str,
                         regulation_era: str, fuel: float, stint_number: int,
                         temp_bucket: str, degr_ordinal: int):
    """Returns a closure over everything fixed for this lap so
    marginal_pace_loss() only needs to vary tyre_age."""
    def _call(tyre_age):
        if tyre_models is None:
            return {"predicted_pace_loss": None, "cliff_probability_next_5_laps": None}
        # WALK-FORWARD ERA GUARD (fold pickles only; the production pickle has no "train_eras" key,
        # so this line is a no-op there). MDP para 125: a model fit on other regulation eras must not
        # be used for this race's era. Set HERMES_ALLOW_CROSS_ERA=1 to run the cross-era TRANSFER
        # variant instead (sensitivity analysis: how does HERMES do with last era's models?).
        _train_eras = tyre_models.get("train_eras")
        if (_train_eras is not None and regulation_era not in _train_eras
                and not os.environ.get("HERMES_ALLOW_CROSS_ERA")):
            return {"predicted_pace_loss": None, "cliff_probability_next_5_laps": None}
        try:
            return tyre_proj.build_tyre_life_projection(
                reg_models=tyre_models["reg_models"], cph=tyre_models["cph"],
                temp_dummy_columns=tyre_models["temp_dummy_columns"],
                era_dummy_columns=tyre_models["era_dummy_columns"],
                compound=compound, circuit=circuit, regulation_era=regulation_era,
                tyre_age=tyre_age, fuel_load_estimate=fuel, stint_number=stint_number,
                track_temp_bucket=temp_bucket, circuit_degredation_ordinal=degr_ordinal,
                cliff_horizon_laps=HORIZON_LAPS,
            )
        except Exception as e:  # noqa: BLE001 - BP convention: never let a projection crash the lap
            note(f"projection failed for {compound}/{circuit}/{regulation_era} "
                 f"at age {tyre_age}: {e} - treating as None/None this lap.")
            return {"predicted_pace_loss": None, "cliff_probability_next_5_laps": None}
    return _call


def find_adjacent_rows(lap_df: pd.DataFrame, own_position: Optional[float]) -> tuple:
    """BP §5 undercut/overcut/rival-threat/dirty-air inputs are about
    whichever car is physically adjacent on track, NOT necessarily the
    other tracked driver (D1/D2 are usually TEAMMATES, not rivals) - an
    earlier version of this file wrongly used the teammate's row as a
    stand-in, which also meant a leader's spurious `gap_to_car_ahead==0`
    (there's no one ahead of P1, but the CSV doesn't encode that as null)
    fed straight into in_dirty_air() and fired every single lap for
    whoever was leading. Fixed by looking up the real Position-1/Position+1
    rows in the FULL field for this lap (laps_features.csv has every
    driver, not just D1/D2 - no reason to throw that away). Returns
    (ahead_row, behind_row); either is None when there genuinely isn't one
    (the leader has no ahead_row, last place has no behind_row)."""
    if own_position is None or pd.isna(own_position):
        return None, None
    ahead = lap_df[lap_df["Position"] == own_position - 1]
    behind = lap_df[lap_df["Position"] == own_position + 1]
    return (ahead.iloc[0] if not ahead.empty else None,
            behind.iloc[0] if not behind.empty else None)


def evaluate_driver_lap(ctx: RaceContext, state: DriverRuntimeState, row, lap_df: pd.DataFrame,
                         resources: dict, radios_by_driver_lap: Optional[dict],
                         data_quality_notes: list) -> dict:
    """Runs Tier 1 -> Tier 2 -> Tier 3 (+ SC gamble) for one driver on one
    lap. Returns a dict matching the DriverDecision shape in BP §6, minus
    the Execution Tree part (added afterwards once both drivers are known)."""
    lap_number = int(row["LapNumber"])
    compound = row.get("Compound")
    tyre_age = _safe_float(row.get("TyreLife")) or 0.0
    track_status = str(row.get("TrackStatus", ""))

    # --- update accumulated race state (BP §5 Tier-2 wiring: ADAPT) ---
    if compound in DRY_COMPOUNDS:
        state.compound_history_dry.add(compound)
    elif compound in WET_COMPOUNDS:
        state.used_wet_or_inter = True

    stint = row.get("Stint")
    if state.current_stint != stint:
        state.current_stint = stint
        state.stint_ages, state.stint_laptimes = [], []
    is_clean_lap = not any([
        bool(row.get("is_pit_in")), bool(row.get("is_pit_out")),
        bool(row.get("is_sc_lap")), bool(row.get("is_vsc_lap")),
        bool(row.get("is_outlier_laptime")), bool(row.get("is_missing_laptime")),
    ])
    if bool(row.get("is_pit_in")):
        # REAL stop count, from the data - not from whatever HERMES itself
        # recommended (see the fix note on `stops_made` below). This is what
        # Tier2State.stops_made_so_far should reflect: a replay evaluates
        # HERMES against the race that actually happened, not a hypothetical
        # one where only HERMES's own advice was followed.
        state.stops_made += 1
    laptime_s = _safe_float(row.get("LapTime_seconds"))
    if is_clean_lap and laptime_s is not None:
        state.stint_ages.append(tyre_age)
        state.stint_laptimes.append(laptime_s)

    # --- derive_features (BP §8.2) ---
    fuel = fuel_load_estimate(lap_number, ctx.total_laps)
    temp_bucket = track_temp_bucket_from_c(_safe_float(row.get("track_temp_c")))
    stint_number = int(stint) if stint is not None and not pd.isna(stint) else 1

    proj_fn = build_projection_fn(resources["tyre_models"], compound, ctx.circuit,
                                   ctx.regulation_era, fuel, stint_number, temp_bucket,
                                   ctx.circuit_degredation_ordinal)
    projection = proj_fn(tyre_age)
    if projection["predicted_pace_loss"] is None and resources["tyre_models"] is not None:
        data_quality_notes.append(
            f"no tyre model for ({compound}, {ctx.circuit}, {ctx.regulation_era}) - "
            f"pace_loss/cliff_probability are None this lap (BP §7-B11 coverage gap)."
        )

    # --- rival-awareness proxies from laps_features.csv (BP §5, adapted) ---
    # Real adjacency, not the teammate: find whoever is actually one
    # position ahead/behind in the full field this lap.
    own_position = _safe_float(row.get("Position"))
    ahead_row, behind_row = find_adjacent_rows(lap_df, own_position)
    gap_ahead_s = _safe_float(row.get("gap_to_car_ahead")) if ahead_row is not None else None
    # The car behind's own gap_to_car_ahead IS the gap between it and us -
    # exact, not a proxy (we are, by definition, the car ahead of them).
    gap_behind_s = _safe_float(behind_row.get("gap_to_car_ahead")) if behind_row is not None else None
    ahead_tyre_age = _safe_float(ahead_row.get("TyreLife")) if ahead_row is not None else None
    behind_tyre_age = _safe_float(behind_row.get("TyreLife")) if behind_row is not None else None
    ahead_pitted_recently = bool(ahead_row.get("is_pit_in")) if ahead_row is not None else False

    adjacency = {
        "ahead_driver": ahead_row.get("Driver") if ahead_row is not None else None,
        "ahead_tyre_age": ahead_tyre_age,
        "behind_driver": behind_row.get("Driver") if behind_row is not None else None,
        "behind_tyre_age": behind_tyre_age,
    }
    # A tyre-age gap this large before either car has pitted is implausible -
    # far more likely find_adjacent_rows landed on a lapped/pitted car via a
    # Position tie, gap-based fallback ranking, or a pre-race compound
    # anomaly than a genuine live undercut window. Surfaced as data, not
    # silently trusted.
    for label, other_age in (("ahead", ahead_tyre_age), ("behind", behind_tyre_age)):
        if other_age is not None and state.stops_made == 0 and abs(tyre_age - other_age) >= UNDERCUT_MIN_TYRE_AGE_GAP:
            data_quality_notes.append(
                f"{label} car (tyre_age={other_age}) is {abs(tyre_age - other_age):.0f} laps "
                f"different from ours (tyre_age={tyre_age}) despite neither of us having pitted "
                f"yet this race - check whether Position/adjacency picked the right car "
                f"(adjacency={adjacency})."
            )

    # --- Tier 1 ---
    red_flag = "5" in track_status
    unsafe_wx = unsafe_weather(
        rain_now=bool(row.get("Rainfall", False)), on_slicks=compound in DRY_COMPOUNDS,
        rain_probability_pct=_safe_float(row.get("rain_probability_pct")),
        laps_remaining=ctx.total_laps - lap_number, total_laps=ctx.total_laps,
    )
    t1_state = gate1.Tier1State(
        red_flag_or_race_stopped=red_flag,
        tyre_structurally_damaged=bool(row.get("tyre_structurally_damaged", False)),
        unsafe_weather=unsafe_wx,
        tyre_age_history=state.stint_ages,
        laptime_seconds_history=state.stint_laptimes,
    )
    try:
        t1 = gate1.evaluate_tier1(t1_state)
    except np.linalg.LinAlgError:
        # ROBUSTNESS FIX (found while testing the walk-forward pipeline; reproduces the
        # "SVD did not converge in Linear Least Squares" failure that killed 2022 Austrian GP).
        # Cause: `tyre_age = _safe_float(row.get("TyreLife")) or 0.0` turns a MISSING TyreLife into
        # 0.0, so a stint can hold >=6 clean laps whose ages are all 0.0; np.polyfit then divides by a
        # zero-norm column and LAPACK fails. That crashed the WHOLE race replay.
        # Convention (blueprint): unknown -> not a cliff, never a crash. Only this previously-fatal case
        # changes behaviour; every race that already ran produces byte-identical decisions.
        data_quality_notes.append(
            "tier1 cliff scan could not run on this stint's history (degenerate tyre-age/lap-time "
            "data, e.g. missing TyreLife) - treated as 'no cliff detected' this lap.")
        t1_state = gate1.Tier1State(
            red_flag_or_race_stopped=red_flag,
            tyre_structurally_damaged=bool(row.get("tyre_structurally_damaged", False)),
            unsafe_weather=unsafe_wx,
            tyre_age_history=[], laptime_seconds_history=[],   # kept consistent for explanation_for()
        )
        t1 = gate1.evaluate_tier1(t1_state)
    result = {"driver": state.code, "driver_id": state.driver_id, "lap": lap_number,
              "compound": compound, "tyre_age": tyre_age, "gate_decision": t1,
              "tier_reached": 1, "reason": None, "triggers": {}, "projection": projection,
              "sc_gamble": None, "gate_tree_trigger_tier": None, "adjacency": adjacency,
              "t1_state": t1_state}  # kept for explanation_for()'s _tier1_plain_reason() -
                                      # explainability only, never re-evaluated or re-decided from.
    if t1 == "PIT_NOW":
        result["gate_tree_trigger_tier"] = 1
        return result

    # --- Tier 2 ---
    wet_exception = state.used_wet_or_inter  # ADAPT (BP §5): ran INTER/WET this race
    t2_state = gate2.Tier2State(
        season=ctx.season, circuit=ctx.circuit, is_sprint_weekend=ctx.is_sprint_weekend,
        compound_history_dry=state.compound_history_dry, wet_race_exception=wet_exception,
        laps_remaining_in_race=ctx.total_laps - lap_number,
        remaining_sets={}, sets_used_so_far=0, stops_made_so_far=state.stops_made,
    )
    if ctx.season >= 2026:
        data_quality_notes.append(
            "2026 season: f1_tyre_constraints.json says 0 mandatory dry compounds but "
            "gate-tier-2.py requires 2 regardless of era (BP §7-B9, unresolved) - this "
            "orchestrator trusts gate-tier-2.py as written; PIT_FLEXIBLE near the end "
            "of a 2026 race may be spurious."
        )
    t2 = gate2.evaluate_tier2(t2_state)
    result["gate_decision"] = t2
    result["tier_reached"] = 2
    if t2 == "PIT_FLEXIBLE":
        result["gate_tree_trigger_tier"] = 2
        return result

    # --- SC gamble (only worth computing once we know we're in Tier 3) ---
    p_sc = ctx.p_sc_5lap
    m_pace_loss = marginal_pace_loss(proj_fn, tyre_age)
    gamble = run_sc_gamble(sc_gamble.SCGambleInputs(
        p_sc_next_n_laps=p_sc, predicted_pace_loss_per_lap=m_pace_loss,
        cliff_probability_next_n_laps=projection["cliff_probability_next_5_laps"],
        n_laps_horizon=HORIZON_LAPS,
    ))
    result["sc_gamble"] = gamble

    # --- Tier 3 ---
    est_len, est_note = expected_stint_length(resources["cliff_stints"], compound,
                                               ctx.circuit, ctx.regulation_era)
    if "flagged" in est_note or "no circuit match" in est_note:
        data_quality_notes.append(f"expected_stint_length: {est_note}")

    safety_car_deployed = ("4" in track_status) or ("6" in track_status) or ("7" in track_status)  # ASSUMPTION: SC+VSC both count (BP §5 "decide VSC handling")
    undercut = live_undercut_opportunity(gap_ahead_s, tyre_age, ahead_tyre_age, ctx.pit_loss_s)
    overcut = overcut_opportunity(ahead_pitted_recently, projection["cliff_probability_next_5_laps"], gap_ahead_s)
    rival_threat = rival_undercut_threat(gap_behind_s, tyre_age, behind_tyre_age, ctx.pit_loss_s)
    dirty_air = in_dirty_air(gap_ahead_s)
    stress = get_driver_stress_trigger(radios_by_driver_lap, state.code, lap_number)

    # --- drying crossover (British GP investigation, Tier 3 - NOT Tier 1: this
    # is a pace/strategy signal, not a safety one - staying on wets too long
    # once the track is dry is slow, not dangerous, unlike rain-on-slicks) ---
    # Only meaningful while OUR driver is still on a wet compound - if we're
    # already dry, there's nothing to "consider switching" to.
    drying_crossover_opportunity = False
    seconds_since_rain_end = _safe_float(row.get("_seconds_since_rain_end"))
    if (drying_line is not None and compound in WET_COMPOUNDS
            and seconds_since_rain_end is not None and seconds_since_rain_end >= 0
            and lap_df is not None and not lap_df.empty):
        def _is_clean(r):
            return not any([bool(r.get("is_pit_in")), bool(r.get("is_pit_out")),
                            bool(r.get("is_sc_lap")), bool(r.get("is_vsc_lap")),
                            bool(r.get("is_outlier_laptime"))])
        field_clean = lap_df[lap_df.apply(_is_clean, axis=1)]
        # "mobile sensor": whichever car in the field has ALREADY switched to
        # a dry compound and has driven the fewest laps on it (freshest
        # switch, most representative of "just crossed over") - ASSUMPTION on
        # the tie-break, flagged, since drying_line.py's own docstring doesn't
        # specify which switched driver to use when several exist.
        switched_candidates = field_clean[field_clean["Compound"].isin(DRY_COMPOUNDS)].sort_values("TyreLife")
        reference_candidates = field_clean[field_clean["Compound"].isin(WET_COMPOUNDS)
                                            & (field_clean["Driver"] != row.get("Driver"))]
        if not switched_candidates.empty and not reference_candidates.empty:
            switched_row = switched_candidates.iloc[0]
            switched_lt = _safe_float(switched_row.get("LapTime_seconds"))
            reference_laps_list = []
            for _, r in reference_candidates.iterrows():
                lt = _safe_float(r.get("LapTime_seconds"))
                if lt is not None:
                    reference_laps_list.append(
                        drying_line.LapTimeSample(driver=r.get("Driver"), lap_time_seconds=lt,
                                                   compound=r.get("Compound")))
            if switched_lt is not None and reference_laps_list:
                switched_sample = drying_line.LapTimeSample(
                    driver=switched_row.get("Driver"), lap_time_seconds=switched_lt,
                    compound=switched_row.get("Compound"))
                drying_result = drying_line.evaluate_drying_crossover(
                    seconds_since_rain_end=seconds_since_rain_end,
                    switched_driver_lap=switched_sample, reference_laps=reference_laps_list)
                drying_crossover_opportunity = (drying_result.state == drying_line.DryingState.CONSIDER_DRIER_TYRE)

    t3_state = gate3.Tier3State(
        cliff_probability_next_5_laps=projection["cliff_probability_next_5_laps"],
        predicted_pace_loss=projection["predicted_pace_loss"],
        tyre_age=tyre_age, expected_stint_length=est_len,
        undercut_opportunity=undercut, overcut_opportunity=overcut,
        safety_car_deployed=safety_car_deployed, rival_undercut_threat=rival_threat,
        in_dirty_air=dirty_air, driver_stress_signal=stress,
        sc_gamble_recommendation=gamble.get("recommendation"),
        drying_crossover_opportunity=drying_crossover_opportunity,
    )
    t3 = gate3.evaluate_tier3(t3_state)
    result["gate_decision"] = t3["decision"]
    result["reason"] = t3["reason"]
    result["triggers"] = t3["triggers"]
    result["tier_reached"] = 3
    result["driver_stress_signal"] = stress
    result["track_position"] = row.get("Position")
    result["can_delay_one_lap_without_position_loss"] = can_delay_one_lap(gap_behind_s, ctx.pit_loss_s)
    result["t3_state"] = t3_state  # kept for explanation_for()'s "top contributing signals"
                                     # ranking - needs the raw values/thresholds, not just the
                                     # trigger booleans already in result["triggers"].
    if t3["decision"] == "PIT_NOW":
        result["gate_tree_trigger_tier"] = 3
    return result


def merge_execution(ctx: RaceContext, results: list, safety_car_active: bool) -> None:
    """BP §8.2 tail + §4.1 quirk (b): fires the Execution Tree for whoever
    triggered PIT_NOW/PIT_FLEXIBLE, and separately calls
    get_driving_instruction for anyone who didn't trigger (PIT_LATER/
    DONT_PIT), since the Execution Tree itself only outputs for triggered
    drivers."""
    triggered = [r for r in results if r["gate_tree_trigger_tier"] is not None]
    if triggered:
        contexts = [exec_tree.DriverPitContext(
            driver_id=r["driver_id"], gate_tree_trigger_tier=r["gate_tree_trigger_tier"],
            tyre_age=r["tyre_age"], track_position=int(r.get("track_position") or 99),
            can_delay_one_lap_without_position_loss=r.get("can_delay_one_lap_without_position_loss", False),
            cliff_probability_next_5_laps=r["projection"]["cliff_probability_next_5_laps"],
            tier3_reason=r["reason"], driver_stress_signal=r.get("driver_stress_signal", False),
        ) for r in triggered]
        state = exec_tree.ExecutionTreeState(
            triggered_drivers=contexts, safety_car_active=safety_car_active,
            circuit=ctx.circuit, priority_driver_id=resolve_second_driver_priority_v1(),
        )
        exec_out = exec_tree.evaluate_execution_tree(state)
        for r in triggered:
            r["execution"] = exec_out.get(r["driver_id"])

    for r in results:
        if r["gate_tree_trigger_tier"] is None:
            decision = r["gate_decision"]  # "PIT_LATER" or "DONT_PIT"
            instruction = exec_tree.get_driving_instruction(
                decision, r["projection"]["cliff_probability_next_5_laps"],
                tier3_reason=r.get("reason"), driver_stress_signal=r.get("driver_stress_signal", False),
            )
            r["execution"] = {"decision": decision, "penalty_seconds": 0.0,
                               "driving_instruction": instruction}


# ============================================================================
# 6b. Explainability layer - decision trace for explanation_for()
# ============================================================================
# This is a TRACE FORMATTER, not a model-explainability tool - gate-tier-1/2/3
# are transparent if/else rules, so there is nothing for SHAP (or any other
# attribution method) to do here. SHAP belongs on the tyre model itself
# (tyre_life_projection.py's regression + Cox output that FEEDS Tier 3), which
# is a separate, later piece of work - see the module-level note below
# SIGNAL_DESCRIPTIONS for where that plugs in.
#
# "Top contributing signals" for a Tier-3 decision means: of the triggers
# that were actually active, which ones were furthest past their own
# threshold. That's a real, non-arbitrary ranking (not dict order), built
# from the same raw values/thresholds gate-tier-3.py already computes with -
# nothing here re-derives or approximates those numbers.

# Plain-English fragments per trigger name, written from a race engineer's
# point of view - kept separate from gate-tier-3.py's own trigger names so
# that file doesn't have to carry human-readable strings for a purpose it
# was never designed for.
SIGNAL_DESCRIPTIONS = {
    "cliff_proximity": "tyre cliff risk in the next {h} laps is elevated",
    "pace_lap_delta": "predicted pace loss is above the calibrated threshold",
    "tyre_age": "tyre age relative to expected stint length is high",
    "undercut": "an undercut opportunity is open on the car ahead",
    "overcut": "an overcut opportunity is open (car ahead just pitted / about to cliff)",
    "safety_car": "a safety car or VSC is already deployed",
    "rival_undercut_threat": "the car behind is threatening an undercut",
    "dirty_air": "running in dirty air behind another car",
    "drying_crossover": "track is drying and a dry-tyre crossover point looks reached",
}

TIER_NAMES = {1: "Tier 1 (hard safety gate)", 2: "Tier 2 (regulatory gate)",
              3: "Tier 3 (soft/strategic triggers)"}


def _tier1_plain_reason(r: dict) -> str:
    """Identifies WHICH Tier 1 hard-safety trigger(s) actually fired, from
    the same Tier1State evaluate_tier1() was called with (kept on the
    result as r["t1_state"] purely for this - see evaluate_driver_lap).
    This is READ-ONLY inspection of gate-tier-1.py's own inputs/logic:
    - red_flag_or_race_stopped / tyre_structurally_damaged / unsafe_weather
      are read straight off t1_state, no re-derivation.
    - the cliff-already-hit check calls gate1.tyre_cliff_already_hit() again
      on the SAME tyre_age_history/laptime_seconds_history t1_state already
      carries - the exact function evaluate_tier1() itself uses - so this
      can never disagree with what actually fired. Nothing here changes
      Tier 1's decision, thresholds, or evaluation order; it only reports
      after the fact which of the four conditions were true.
    If multiple triggers are true at once, all of them are reported -
    gate-tier-1.py's own early-return means only the FIRST true one in its
    checked order actually caused the short-circuit, but from a race
    engineer's point of view "red flag AND damage" is more honest than
    hiding the second one, so all active triggers are surfaced rather than
    just the one gate-tier-1.py happened to check first.
    """
    t1_state = r.get("t1_state")
    if t1_state is None:
        # Should not happen for a Tier-1 PIT_NOW result (evaluate_driver_lap
        # always sets it), but degrade honestly rather than guess.
        return "a Tier 1 hard-safety trigger fired (detail unavailable)"

    active = []
    if t1_state.red_flag_or_race_stopped:
        active.append("a red flag / race suspension is active")
    if t1_state.tyre_structurally_damaged:
        active.append("the tyre shows structural damage (retirement risk if not pitted)")
    if t1_state.unsafe_weather:
        active.append("rain/wet conditions make the current tyre unsafe")
    if gate1.tyre_cliff_already_hit(t1_state.tyre_age_history, t1_state.laptime_seconds_history):
        active.append("a tyre cliff has already been detected in the current stint")

    if not active:
        # Defensive only: gate-tier-1.py returned PIT_NOW but none of the
        # four checks read back as True here - flag the mismatch rather
        # than fabricate a reason, since that would hide a real bug.
        return "a Tier 1 hard-safety trigger fired (could not identify which - check t1_state wiring)"
    if len(active) == 1:
        return active[0]
    return "; ".join(active[:-1]) + f"; and {active[-1]}"


def _tier2_plain_reason(r: dict) -> str:
    return "the mandatory dry-compound rule isn't satisfied and the deadline to do it safely is approaching"


def top_contributing_signals(r: dict, n: int = 3) -> list:
    """Ranks ACTIVE Tier-3 triggers by how far past their own threshold they
    are (a real strength measure, not dict order). Returns a list of dicts:
    {"signal": name, "value": ..., "threshold": ..., "margin_ratio": ...,
     "description": plain-English text}. Empty list if Tier 3 was never
     reached, or nothing was active there (e.g. clean DONT_PIT lap)."""
    t3_state = r.get("t3_state")
    triggers = r.get("triggers") or {}
    active_names = [k for k, v in triggers.items() if v]
    if t3_state is None or not active_names:
        return []

    ranked = []
    for name in active_names:
        value = threshold = margin_ratio = None
        if name == "cliff_proximity":
            value = t3_state.cliff_probability_next_5_laps
            threshold = gate3.CLIFF_PROBABILITY_THRESHOLD
        elif name == "pace_lap_delta":
            value = t3_state.predicted_pace_loss
            threshold = gate3.PACE_LOSS_THRESHOLD_SECONDS
        elif name == "tyre_age":
            value = t3_state.tyre_age
            threshold = gate3.TYRE_AGE_TRIGGER_RATIO * t3_state.expected_stint_length
        # External/boolean-only triggers (undercut, overcut, safety_car,
        # rival_undercut_threat, dirty_air, drying_crossover) have no
        # continuous value to rank by - they're either on or off, so they
        # get no margin_ratio and sort after every ranked numeric trigger,
        # in the fixed order gate-tier-3.py declares them, not silently
        # dropped or given a fabricated number.
        if value is not None and threshold not in (None, 0):
            margin_ratio = value / threshold
        ranked.append({
            "signal": name,
            "value": value,
            "threshold": threshold,
            "margin_ratio": margin_ratio,
            "description": SIGNAL_DESCRIPTIONS.get(name, name).format(h=HORIZON_LAPS),
        })

    ranked.sort(key=lambda d: (d["margin_ratio"] is None, -(d["margin_ratio"] or 0)))
    return ranked[:n]


def gate_path_for(r: dict) -> list:
    """Exact tier path this lap actually walked, e.g. [1, 2, 3] for a lap
    that cleared Tier 1 and Tier 2 and was decided at Tier 3, or [1] for a
    lap that stopped dead at Tier 1. Matches tier_reached, which
    evaluate_driver_lap already sets to the LAST tier evaluated - the path
    is just range(1, tier_reached + 1) since the trees are a strict
    sequential cascade with no skipping."""
    return list(range(1, r["tier_reached"] + 1))


def plain_english_for(r: dict) -> str:
    """The 'PIT_NOW because...' sentence. Built per-tier since each tier's
    reasoning shape is genuinely different (Tier 1: single hard trigger,
    Tier 2: a regulatory deadline, Tier 3: N soft triggers combining) -
    forcing one template across all three would either hide detail or
    fabricate a "top signal" for a tier that fired for one specific,
    already-known reason."""
    decision = r["gate_decision"]
    tier = r["tier_reached"]

    if tier == 1 and decision == "PIT_NOW":
        return f"PIT_NOW because {_tier1_plain_reason(r)}."

    if tier == 2 and decision == "PIT_FLEXIBLE":
        return f"PIT_FLEXIBLE because {_tier2_plain_reason(r)}."

    if tier == 3:
        signals = top_contributing_signals(r)
        if decision == "DONT_PIT":
            return "DONT_PIT - no Tier 3 triggers are currently active."
        if not signals:
            # Reached here only if a boolean-only trigger set n_active>=1 but
            # ranking somehow returned nothing - defensive, shouldn't happen
            # given the active_names guard above, kept honest rather than silent.
            return f"{decision} - Tier 3 triggers active but no signal detail available."
        lead = signals[0]["description"]
        if len(signals) > 1:
            rest = "; ".join(s["description"] for s in signals[1:])
            return f"{decision} because {lead} (also: {rest})."
        return f"{decision} because {lead}."

    # Tier 2 -> MOVE_TO_TIER_3 or Tier 1 -> MOVE_TO_TIER_2 never reach here
    # (evaluate_driver_lap only stops early on PIT_NOW/PIT_FLEXIBLE), kept
    # as a fallback rather than assumed unreachable.
    return f"{decision} at Tier {tier}."


def explanation_for(r: dict) -> dict:
    """Full structured decision trace for one driver-lap result. Returns a
    dict (not a string) so callers can render it as plain text (--explain
    CLI output, unchanged in spirit), log it as JSONL alongside the rest of
    `decisions`, or feed it into a dissertation write-up table without
    re-parsing a formatted string.

    Fields:
      plain_text   - "PIT_NOW because ..." human-readable sentence
      gate_path    - e.g. [1, 2, 3], the exact tiers walked this lap
      decision     - final gate decision string (PIT_NOW/PIT_FLEXIBLE/etc)
      triggers     - active Tier 3 trigger names (empty list if tier<3)
      top_signals  - ranked list from top_contributing_signals()
      instruction  - Execution Tree's driving_instruction, if computed yet
    """
    active_triggers = [k for k, v in (r.get("triggers") or {}).items() if v]
    exe = r.get("execution") or {}
    return {
        "plain_text": plain_english_for(r),
        "gate_path": gate_path_for(r),
        "decision": r["gate_decision"],
        "triggers": active_triggers,
        "top_signals": top_contributing_signals(r),
        "instruction": exe.get("driving_instruction"),
    }


def explanation_text_for(r: dict) -> str:
    """Single-line summary for places that just want text (the --explain
    CLI printout) - built from explanation_for()'s structured output so the
    two never drift out of sync with each other."""
    e = explanation_for(r)
    path_str = "->".join(f"Tier {t}" for t in e["gate_path"])
    bits = [path_str, e["plain_text"]]
    if e["triggers"]:
        bits.append("triggers=" + ",".join(e["triggers"]))
    if e["top_signals"]:
        top_str = ", ".join(
            f"{s['signal']}" + (f" ({s['margin_ratio']:.2f}x threshold)" if s["margin_ratio"] else "")
            for s in e["top_signals"]
        )
        bits.append(f"top_signals=[{top_str}]")
    if e["instruction"]:
        bits.append(f"instruction={e['instruction']}")
    return " | ".join(bits)


# ============================================================================
# 7. ReplaySource (BP §8.1 step 5)
# ============================================================================
def find_laps_features(season: int, race: str, session: str) -> Optional[Path]:
    p = CACHE_DIR / "clean" / "features" / str(season) / race / session / "laps_features.csv"
    return p if p.exists() else None


def load_weather(season: int, race: str, session: str) -> Optional[pd.DataFrame]:
    for candidate in (
        CACHE_DIR / "clean" / "fastf1" / str(season) / race / session / "weather_cleaned.csv",
        CACHE_DIR / "raw" / "fastf1" / str(season) / race / session / "weather.csv",
    ):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def _attach_weather_columns(laps: pd.DataFrame, weather_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """PART 1/2 integration point (British GP investigation): attaches REAL
    historical Rainfall, plus a derived seconds-since-last-rain-end column,
    onto `laps` - once per race, not per lap/driver, not a second file load
    (load_weather() is still called exactly once, in main()). After this,
    row.get("Rainfall", False) at the existing Tier 1 call site needs ZERO
    changes - the interface is preserved exactly, it just finally receives
    real data instead of always missing the column.

    NO LOOKAHEAD: uses pd.merge_asof with direction="backward" - a lap can
    only ever be matched to a weather sample AT OR BEFORE its own session
    time, never a later one. This is a hard guarantee from merge_asof's own
    semantics, not a convention this function has to enforce by hand.

    Does NOT touch rain_probability_pct (stays real-columns-absent -> None,
    exactly as before) and does NOT feed anything into crossover.py -
    crossover.py's forecast-probability path remains exactly as inert as it
    was, for the same real reason as before (no forecast data exists in
    replay), not because of a wiring bug."""
    if weather_df is None:
        note("No weather data available for this race - Rainfall will be False for every "
             "lap (Tier 1's weather gate stays silent, same as before this integration) "
             "and drying_crossover will never fire.")
        laps = laps.copy()
        laps["Rainfall"] = False
        laps["_seconds_since_rain_end"] = None
        return laps

    laps_time_col = "Time" if "Time" in laps.columns else None
    weather_time_col = "time_seconds" if "time_seconds" in weather_df.columns else (
        "Time" if "Time" in weather_df.columns else None)
    if laps_time_col is None or weather_time_col is None:
        note(f"Could not find a usable time column to align weather data (laps has "
             f"{'a Time column' if laps_time_col else 'no Time column'}, weather has "
             f"{weather_time_col or 'no usable time column'}) - Rainfall will be False for "
             f"every lap this race, same as if no weather data existed at all.")
        laps = laps.copy()
        laps["Rainfall"] = False
        laps["_seconds_since_rain_end"] = None
        return laps

    def to_seconds(series):
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        return pd.to_timedelta(series, errors="coerce").dt.total_seconds()

    laps = laps.reset_index(drop=True).copy()
    laps["_lap_time_s"] = to_seconds(laps[laps_time_col])

    weather = weather_df.copy()
    weather["_weather_time_s"] = to_seconds(weather[weather_time_col])
    weather = weather.dropna(subset=["_weather_time_s"]).sort_values("_weather_time_s")

    if "is_rain_end" in weather.columns:
        weather["_last_rain_end_time_s"] = weather["_weather_time_s"].where(
            weather["is_rain_end"].fillna(False).astype(bool))
        weather["_last_rain_end_time_s"] = weather["_last_rain_end_time_s"].ffill()
    else:
        weather["_last_rain_end_time_s"] = pd.NA
        note("weather data has no 'is_rain_end' column - drying_crossover will never fire "
             "this race (it needs a real rain-end event to measure time since).")

    weather_cols = ["_weather_time_s", "_last_rain_end_time_s"]
    if "Rainfall" in weather.columns:
        weather_cols.insert(1, "Rainfall")
    else:
        note("weather data has no 'Rainfall' column - Tier 1's weather gate stays silent "
             "this race, same as before this integration.")

    # Never silently drop a lap row just because its own Time failed to
    # parse - split, merge only the valid-time rows, then recombine so
    # every original row survives (with weather columns as NaN/False if its
    # own time couldn't be resolved).
    laps["_orig_order"] = range(len(laps))
    valid = laps[laps["_lap_time_s"].notna()].sort_values("_lap_time_s")
    invalid = laps[laps["_lap_time_s"].isna()].copy()
    for col in weather_cols:
        if col not in invalid.columns:
            invalid[col] = pd.NA

    merged_valid = pd.merge_asof(valid, weather[weather_cols], left_on="_lap_time_s",
                                  right_on="_weather_time_s", direction="backward")
    combined = pd.concat([merged_valid, invalid], ignore_index=True).sort_values("_orig_order")
    combined = combined.drop(columns=["_orig_order"]).reset_index(drop=True)

    if "Rainfall" in combined.columns:
        combined["Rainfall"] = combined["Rainfall"].fillna(False).astype(bool)
    else:
        combined["Rainfall"] = False
    combined["_seconds_since_rain_end"] = combined["_lap_time_s"] - combined["_last_rain_end_time_s"]
    return combined


def run_replay(ctx: RaceContext, resources: dict, lap_range: range,
                radios_by_driver_lap: Optional[dict] = None, explain: bool = False,
                weather_df: Optional[pd.DataFrame] = None) -> list:
    laps_path = find_laps_features(ctx.season, ctx.circuit, ctx.session)
    if laps_path is None:
        raise FileNotFoundError(
            f"laps_features.csv not found under {CACHE_DIR}/clean/features/{ctx.season}/"
            f"{ctx.circuit}/{ctx.session}/ - make sure gcs_cache/ is populated for this race "
            f"(BP §8.5 runtime facts)."
        )
    laps = pd.read_csv(laps_path, dtype={"TrackStatus": str})
    if "LapTime_seconds" not in laps.columns and "LapTime" in laps.columns:
        laps["LapTime_seconds"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    laps = _attach_weather_columns(laps, weather_df)

    if radios_by_driver_lap is None:
        radios_by_driver_lap = build_radios_by_driver_lap(REPO_ROOT, laps, ctx)

    states = {ctx.d1_code: DriverRuntimeState("D1", ctx.d1_code),
              ctx.d2_code: DriverRuntimeState("D2", ctx.d2_code)}

    all_decisions = []
    for lap_number in lap_range:
        lap_df = laps[laps["LapNumber"] == lap_number]
        if lap_df.empty:
            continue
        lap_df = _rank_by_gap_to_leader(lap_df)
        row_d1 = _row_or_none(lap_df, ctx.d1_code)
        row_d2 = _row_or_none(lap_df, ctx.d2_code)
        if row_d1 is None and row_d2 is None:
            continue

        safety_car_active = "4" in str(lap_df["TrackStatus"].iloc[0]) if not lap_df.empty else False
        lap_results, dq_notes = [], []
        if row_d1 is not None:
            lap_results.append(evaluate_driver_lap(ctx, states[ctx.d1_code], row_d1, lap_df,
                                                     resources, radios_by_driver_lap, dq_notes))
        if row_d2 is not None:
            lap_results.append(evaluate_driver_lap(ctx, states[ctx.d2_code], row_d2, lap_df,
                                                     resources, radios_by_driver_lap, dq_notes))

        merge_execution(ctx, lap_results, safety_car_active)

        for r in lap_results:
            r["data_quality_notes"] = list(dq_notes)
            r["explanation"] = explanation_for(r)          # structured trace (dict)
            r["explanation_text"] = explanation_text_for(r)  # single-line rendering of the same trace
            actual_pit = None
            actual_row = row_d1 if r["driver"] == ctx.d1_code else row_d2
            if actual_row is not None:
                actual_pit = bool(actual_row.get("is_pit_in"))
            r["actual_is_pit_in_lap"] = actual_pit
            all_decisions.append(r)
            if explain:
                flag = " <-- real pit lap" if actual_pit else ""
                print(f"L{lap_number:>3} {r['driver']}: {r['explanation_text']}{flag}")
                if dq_notes and r is lap_results[-1]:
                    for note_text in dq_notes:
                        print(f"      ! {note_text}")

            # NOTE: stint-history reset already happens inside evaluate_driver_lap
            # from the real Stint/is_pit_in columns (not from what HERMES itself
            # recommended) - state.stops_made is likewise now driven by the real
            # is_pit_in flag there, so there's nothing left to do here. An
            # earlier version incremented stops_made when HERMES's own
            # driving_instruction said PIT_LAP, which silently drifted out of
            # sync with reality whenever HERMES disagreed with the actual pit
            # lap (i.e. most of the time) - that's what caused stale
            # "neither of us has pitted yet" diagnostic notes late in a race.

    return all_decisions


class LiveSource:
    """Stub matching ReplaySource's interface (BP §8.1 step 5), so a
    later live feed only needs to implement `next_lap_snapshot()` -
    everything downstream (evaluate_driver_lap, merge_execution) already
    takes a plain row-like object and doesn't care where it came from."""

    def __init__(self, ctx: RaceContext):
        self.ctx = ctx

    def next_lap_snapshot(self, driver_code: str) -> dict:
        raise NotImplementedError(
            "LiveSource is a stub (BP §8.1: 'LiveSource stub (same interface)'). "
            "Wire this to your live timing feed; it must return a dict/row with "
            "the same keys evaluate_driver_lap() reads from laps_features.csv."
        )


# ============================================================================
# 8. Self-checks (BP §8.4)
# ============================================================================
def selftest() -> bool:
    ok = True

    def check(label, condition):
        nonlocal ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            ok = False
        print(f"[selftest] {status}: {label}")

    # 1. Constant-sync assert already ran at import time (would have raised).
    check("CLIFF_PROBABILITY_THRESHOLD in sync (tier3 vs execution-tree)", True)

    # 2. Re-run each tree's own documented scenarios through the shim.
    s1 = gate1.Tier1State(False, False, False, [1, 2, 3], [90.1, 90.0, 89.9])
    check("gate-tier-1 clean short stint -> MOVE_TO_TIER_2", gate1.evaluate_tier1(s1) == "MOVE_TO_TIER_2")
    s2 = gate1.Tier1State(True, False, False, [1, 2, 3], [90.1, 90.0, 89.9])
    check("gate-tier-1 red flag -> PIT_NOW", gate1.evaluate_tier1(s2) == "PIT_NOW")

    t2 = gate2.Tier2State(2023, "Bahrain_Grand_Prix", False, {"SOFT"}, False, 3,
                           {"SOFT": 2, "MEDIUM": 3, "HARD": 4}, 1, 0)
    check("gate-tier-2 non-compliant, deadline close -> PIT_FLEXIBLE", gate2.evaluate_tier2(t2) == "PIT_FLEXIBLE")

    t3_none = gate3.Tier3State(None, None, 10, 25, False, False, False, False, False)
    check("gate-tier-3 all-None tyre model -> DONT_PIT", gate3.evaluate_tier3(t3_none)["decision"] == "DONT_PIT")
    t3_three = gate3.Tier3State(0.05, 0.2, 22, 25, False, False, True, False, False)
    check("gate-tier-3 three triggers -> PIT_NOW", gate3.evaluate_tier3(t3_three)["decision"] == "PIT_NOW")

    d1_only = exec_tree.ExecutionTreeState(
        triggered_drivers=[exec_tree.DriverPitContext("D1", 1, 22, 3, False, cliff_probability_next_5_laps=0.20)],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    out = exec_tree.evaluate_execution_tree(d1_only)
    check("execution-tree single driver -> PIT_LAP", out["D1"]["decision"] == "PIT_NOW"
          and out["D1"]["driving_instruction"] == "PIT_LAP")

    # 3. B1 regression test - synthetic models with a genuine temp
    #    coefficient must give >=2 distinct outputs across the 4 buckets.
    try:
        from sklearn.linear_model import LinearRegression
        from lifelines import CoxPHFitter
        rng = np.random.default_rng(0)
        n = 300
        temp_cols = ["temp_extreme", "temp_hot", "temp_warm"]  # matches the REAL pickle's prefix
        era_cols = ["regulation_era_2022-2025"]
        X = pd.DataFrame({
            "tyre_age": rng.integers(1, 30, n),
            "fuel_load_estimate": rng.uniform(0, 110, n),
            "stint_number": rng.integers(1, 4, n),
            "temp_extreme": rng.integers(0, 2, n),
            "temp_hot": rng.integers(0, 2, n),
            "temp_warm": rng.integers(0, 2, n),
        })
        y = (0.03 * X["tyre_age"] + 0.5 * X["temp_hot"] + 0.8 * X["temp_extreme"]
             + rng.normal(0, 0.05, n))
        reg_models = {("MEDIUM", "Test_Circuit", "2022-2025"): LinearRegression().fit(X, y)}
        cox_df = pd.DataFrame({
            "duration": rng.integers(3, 40, n), "event": rng.integers(0, 2, n),
            "compound": "MEDIUM", "fuel_load_estimate": rng.uniform(0, 110, n),
            "stint_number": rng.integers(1, 4, n), "circuit_degredation_ordinal": rng.integers(0, 3, n),
            "temp_extreme": rng.integers(0, 2, n), "temp_hot": rng.integers(0, 2, n),
            "temp_warm": rng.integers(0, 2, n), "regulation_era_2022-2025": rng.integers(0, 2, n),
        })
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col="duration", event_col="event", strata=["compound"])

        outputs = set()
        for bucket in ("cool", "warm", "hot", "extreme"):
            proj = tyre_proj.build_tyre_life_projection(
                reg_models=reg_models, cph=cph, temp_dummy_columns=temp_cols, era_dummy_columns=era_cols,
                compound="MEDIUM", circuit="Test_Circuit", regulation_era="2022-2025",
                tyre_age=15, fuel_load_estimate=60, stint_number=2, track_temp_bucket=bucket,
                circuit_degredation_ordinal=1,
            )
            outputs.add(round(proj["predicted_pace_loss"], 6))
        check("B1 fix: 4 temp buckets give >=2 distinct pace_loss outputs", len(outputs) >= 2)
    except ImportError as e:
        print(f"[selftest] SKIP: B1 regression test needs sklearn/lifelines ({e})")

    return ok


# ============================================================================
# 9. CLI
# ============================================================================
def main():
    global REPO_ROOT, CACHE_DIR
    parser = argparse.ArgumentParser(description="HERMES master orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("replay", help="evaluate the trees against a historical race")
    sp.add_argument("--repo-root", default=str(REPO_ROOT),
                     help="Only affects where data files (pickle, CSVs) are read from. "
                          "The tree/module .py files themselves are loaded at import time "
                          "from HERMES_REPO_ROOT (env var) - set that instead if your trees "
                          "live somewhere other than the default.")
    sp.add_argument("--season", type=int, required=True)
    sp.add_argument("--race", required=True, help="underscore Race folder name, e.g. Bahrain_Grand_Prix")
    sp.add_argument("--session", default="R")
    sp.add_argument("--d1", required=True, help="3-letter driver code, priority driver")
    sp.add_argument("--d2", required=True, help="3-letter driver code, second car")
    sp.add_argument("--sprint-weekend", action="store_true")
    sp.add_argument("--laps", default=None, help="e.g. 1-57; default is the whole race")
    sp.add_argument("--out", default=None, help="write per-driver-per-lap decisions as JSONL")
    sp.add_argument("--explain", action="store_true")
    sp.add_argument("--sc-gamble", choices=SC_GAMBLE_MODES, default=None,
                     help="SC-gamble evaluator: 'analytic' (default, validated reference) or 'mc' "
                          "(Monte Carlo uncertainty propagation around the same cost model). "
                          "Overrides env HERMES_SC_GAMBLE.")

    sub.add_parser("selftest", help="run the wiring/regression self-checks (BP §8.4)")

    args = parser.parse_args()

    if args.command == "replay":
        if args.sc_gamble:
            set_sc_gamble_mode(args.sc_gamble)
        note(f"SC gamble evaluator: {SC_GAMBLE_MODE}")
        REPO_ROOT = Path(args.repo_root).resolve()
        CACHE_DIR = Path(os.environ.get("GCS_CACHE_DIR", REPO_ROOT / "gcs_cache"))

        tyre_models = load_tyre_models(_data_path(REPO_ROOT, "tyre_life_models.pkl"))
        sc_prior = load_sc_prior(_data_path(REPO_ROOT, "sc_vsc_circuit_level_prior.csv"))
        cliff_stints = load_cliff_stints(_data_path(REPO_ROOT, "cliff_detection_stints.csv"))
        taxonomy = load_circuit_taxonomy(REPO_ROOT / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
        pit_loss_table = load_pit_loss_table(
            _data_path(REPO_ROOT, "checkpoints", "rival_knowledge", "archive_per_race_analysis.csv"))

        degr_ordinal, degr_note = circuit_degredation_ordinal_for(taxonomy, args.race)
        pit_loss_s, pit_loss_note = pit_loss_for_circuit(pit_loss_table, args.race)
        note(f"circuit_degredation_ordinal: {degr_ordinal} ({degr_note})")
        note(f"pit_loss_s: {pit_loss_s} ({pit_loss_note})")

        weather_df = load_weather(args.season, args.race, args.session)
        total_laps = 0
        laps_path = find_laps_features(args.season, args.race, args.session)
        if laps_path is not None:
            total_laps = int(pd.read_csv(laps_path, usecols=["LapNumber"])["LapNumber"].max())
        else:
            note("can't pre-read total_laps - laps_features.csv missing; will fail in run_replay.")

        ctx = RaceContext(
            season=args.season, circuit=args.race, total_laps=total_laps,
            is_sprint_weekend=args.sprint_weekend, d1_code=args.d1, d2_code=args.d2,
            session=args.session, circuit_degredation_ordinal=degr_ordinal,
            pit_loss_s=pit_loss_s, p_sc_5lap=get_sc_probability(sc_prior, args.race),
        )
        resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}

        if args.laps:
            start, end = (int(x) for x in args.laps.split("-"))
            lap_range = range(start, end + 1)
        else:
            lap_range = range(1, total_laps + 1) if total_laps else range(1, 1)

        decisions = run_replay(ctx, resources, lap_range, explain=args.explain, weather_df=weather_df)

        if args.out:
            with open(args.out, "w") as f:
                for d in decisions:
                    # t3_state is a dataclass kept on the result purely for
                    # explanation_for()'s internal ranking (see evaluate_driver_lap) -
                    # everything a reader needs from it is already surfaced in
                    # d["explanation"], so drop the raw object rather than
                    # json.dumps(default=str)-ing an unreadable repr into the file.
                    d_out = {k: v for k, v in d.items() if k not in ("t1_state", "t3_state")}
                    f.write(json.dumps(d_out, default=str) + "\n")
            note(f"wrote {len(decisions)} decisions to {args.out}")

        n_agree = sum(1 for d in decisions if d.get("actual_is_pit_in_lap")
                      and d.get("execution", {}).get("driving_instruction") == "PIT_LAP")
        n_actual_pits = sum(1 for d in decisions if d.get("actual_is_pit_in_lap"))
        print(f"\n[summary] {len(decisions)} driver-lap decisions evaluated. "
              f"{n_actual_pits} real pit-in laps in this window, HERMES also said "
              f"PIT_LAP on {n_agree} of them (BP §8.4 point 4: report agreement, don't claim accuracy).")

    elif args.command == "selftest":
        ok = selftest()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()