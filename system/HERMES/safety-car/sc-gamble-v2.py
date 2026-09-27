#!/usr/bin/env python3
"""
SC/VSC Gamble Evaluator - v2 (expected-utility rewrite)
============================================================================
Deliberately kept OUT of hermes_master.py and out of system/HERMES/safety-car/
sc-gamble.py, per the explicit decision to treat this as a separate
calibration/model-design question, not an orchestrator integration bug.
This file is self-contained: it re-implements the ORIGINAL v1 cost model
verbatim (for honest side-by-side comparison) and a v2 model that fixes the
structural issue diagnosed in conversation, then runs the sweep to prove
the decision boundary actually moves before anything goes near HERMES.

THE DIAGNOSED PROBLEM (v1, sc-gamble.py as shipped)
----------------------------------------------------
    cost_pit_now = NORMAL_PIT_LOSS                      # 22.0, fixed
    cost_if_sc   = SC_PIT_LOSS + COLD_TYRE + degr/2      # 11.0 + 1.5 + degr/2
    cost_if_no_sc= NORMAL_PIT_LOSS + degr + cliff_penalty
    cost_wait    = p_sc*cost_if_sc + (1-p_sc)*cost_if_no_sc

Even at degr=0 and cliff=0: cost_wait = 22 - 9.5*p_sc, which is LESS than
cost_pit_now (22) for any p_sc > 0. "Wait" is a strictly free improvement
before tyre state enters the calculation at all - the SC branch's discount
is baked in structurally, not earned by the state of the race. The `degr/2`
term (arbitrary, no stated reason for exactly one half) also silently drops
cliff risk from the SC branch entirely, and the no-SC branch always re-pays
the SAME normal_pit_loss regardless of how long you waited, so there is no
genuine "cost of committing to wait and being wrong" beyond degradation.

THE FIX (v2)
------------
1. Both branches are evaluated under BOTH futures (SC / no SC), matching the
   explicit expected-value structure requested:
       EV_WAIT   = (1-p_sc)*cost_if_no_sc + p_sc*cost_if_sc   (wait, gamble on SC)
       cost_pit_now stays a fixed, deterministic baseline (paying now
       forecloses the SC question entirely - correct, unchanged from v1).
2. The `degr/2` multiplier is replaced by an expected-LAPS-DRIVEN split, not
   an expected-COST split - and it's derived from an assumption ALREADY
   validated elsewhere in this project (sc-vsc-probability-model.py's
   build_circuit_level_prior: an SC, if it occurs, is placed uniformly at
   random within the window). Under that same assumption, expected laps
   driven before a cheap SC-stop = n_laps_horizon / 2. That number then
   flows through to BOTH degradation AND cliff exposure symmetrically -
   previously cliff risk was silently absent from the SC branch altogether.
3. cost_if_no_sc's baseline is still normal_pit_loss (you still have to pit
   eventually if the SC never comes) - unchanged, this part of v1 was
   correct. The asymmetry was entirely in the SC branch's missing cliff term
   and the ungrounded 0.5.
All cost constants (pit losses, cold-tyre penalty, cliff penalty scale) are
still ASSUMPTIONS, unchanged in this file - v2 fixes the STRUCTURE of the
comparison, not the constants themselves. A constants sweep is section 3
below, separate from the structural fix in section 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import itertools

import numpy as np
import pandas as pd

# ============================================================================
# 0. Shared input contract (identical shape to sc-gamble.py's SCGambleInputs,
#    so either version is a drop-in for the other in hermes_master.py later)
# ============================================================================
@dataclass
class SCGambleInputs:
    p_sc_next_n_laps: Optional[float]
    predicted_pace_loss_per_lap: Optional[float]
    cliff_probability_next_n_laps: Optional[float]
    n_laps_horizon: int


# ============================================================================
# 1. v1 - VERBATIM from system/HERMES/safety-car/sc-gamble.py (unchanged),
#    kept here only so the sweep can compare old vs new honestly.
# ============================================================================
V1_NORMAL_PIT_LOSS_SECONDS = 22.0
V1_SC_PIT_LOSS_SECONDS = 11.0
V1_RESTART_COLD_TYRE_PENALTY_SECONDS = 1.5


def evaluate_sc_gamble_v1(inputs: SCGambleInputs) -> dict:
    if inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None:
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None}
    p_sc = inputs.p_sc_next_n_laps
    degradation_cost_of_waiting = inputs.predicted_pace_loss_per_lap * inputs.n_laps_horizon
    cliff_risk = inputs.cliff_probability_next_n_laps or 0.0
    cliff_penalty_seconds = 5.0 * cliff_risk

    cost_pit_now = V1_NORMAL_PIT_LOSS_SECONDS
    cost_if_sc_comes = V1_SC_PIT_LOSS_SECONDS + V1_RESTART_COLD_TYRE_PENALTY_SECONDS + (degradation_cost_of_waiting / 2)
    cost_if_sc_doesnt_come = V1_NORMAL_PIT_LOSS_SECONDS + degradation_cost_of_waiting + cliff_penalty_seconds
    cost_wait = p_sc * cost_if_sc_comes + (1 - p_sc) * cost_if_sc_doesnt_come

    recommendation = "WAIT" if cost_wait < cost_pit_now else "NO_ADVANTAGE_TO_WAITING"
    return {"recommendation": recommendation, "cost_pit_now": cost_pit_now, "cost_wait": cost_wait,
            "expected_saving_if_wait": cost_pit_now - cost_wait}


# ============================================================================
# 2. v2 - expected-utility rewrite
# ============================================================================
def evaluate_sc_gamble_v2(inputs: SCGambleInputs, *, normal_pit_loss_s: float = 22.0,
                           sc_pit_loss_s: float = 11.0, cold_tyre_penalty_s: float = 1.5,
                           cliff_penalty_scale: float = 5.0) -> dict:
    """Same call signature and same recommendation vocabulary as v1
    ("WAIT" | "NO_ADVANTAGE_TO_WAITING" | "INSUFFICIENT_DATA") so gate-tier-3's
    `sc_gamble_recommendation != "WAIT"` suppression check needs no changes
    if this is ever swapped in - "WAIT" still means "gamble on the SC",
    matching the existing wiring."""
    if inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None:
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None}

    p_sc = inputs.p_sc_next_n_laps
    n = inputs.n_laps_horizon
    pace = inputs.predicted_pace_loss_per_lap
    cliff = inputs.cliff_probability_next_n_laps or 0.0

    # Expected laps actually driven before pitting, in each future - derived
    # from the SAME uniform-placement assumption already validated in
    # sc-vsc-probability-model.py, not a fresh guess.
    laps_if_sc = n / 2.0       # SC lands on average halfway through the window
    laps_if_no_sc = float(n)  # you drive the whole window before pitting

    degradation_if_sc = pace * laps_if_sc
    degradation_if_no_sc = pace * laps_if_no_sc

    # Cliff exposure now scales the SAME way degradation does - a cliff you
    # never reach, because you pitted early under a cheap SC stop, can't
    # cost you. v1 dropped this term from the SC branch entirely.
    cliff_penalty_if_sc = cliff_penalty_scale * cliff * (laps_if_sc / n)
    cliff_penalty_if_no_sc = cliff_penalty_scale * cliff * (laps_if_no_sc / n)

    cost_if_sc = sc_pit_loss_s + cold_tyre_penalty_s + degradation_if_sc + cliff_penalty_if_sc
    cost_if_no_sc = normal_pit_loss_s + degradation_if_no_sc + cliff_penalty_if_no_sc

    cost_wait = p_sc * cost_if_sc + (1 - p_sc) * cost_if_no_sc
    cost_pit_now = normal_pit_loss_s

    recommendation = "WAIT" if cost_wait < cost_pit_now else "NO_ADVANTAGE_TO_WAITING"
    return {"recommendation": recommendation, "cost_pit_now": cost_pit_now, "cost_wait": cost_wait,
            "expected_saving_if_wait": cost_pit_now - cost_wait,
            "cost_if_sc": cost_if_sc, "cost_if_no_sc": cost_if_no_sc}


# ============================================================================
# 3. The sweep
# ============================================================================
def run_sweep():
    p_sc_values = np.linspace(0.0, 1.0, 11)
    cliff_values = np.linspace(0.0, 1.0, 11)
    # Per-lap pace loss: 0 to a plausible max. sc-gamble.py's own demos and
    # the real Bahrain run both stayed under ~0.3s/lap marginal, but the
    # blueprint's TL;DR asks for "0 -> plausible maximum", so this goes
    # further out (1.0s/lap) to see whether the boundary keeps behaving
    # sensibly outside the range actually observed so far.
    pace_values = np.linspace(0.0, 1.0, 11)
    n_horizon = 5

    def count_wait(fn, **kwargs):
        n_wait, n_total = 0, 0
        for p_sc, cliff, pace in itertools.product(p_sc_values, cliff_values, pace_values):
            inputs = SCGambleInputs(p_sc, pace, cliff, n_horizon)
            result = fn(inputs, **kwargs) if kwargs else fn(inputs)
            n_total += 1
            if result["recommendation"] == "WAIT":
                n_wait += 1
        return n_wait, n_total

    v1_wait, v1_total = count_wait(evaluate_sc_gamble_v1)
    v2_wait, v2_total = count_wait(evaluate_sc_gamble_v2)

    print("=" * 78)
    print("SWEEP 1: full grid, p_sc x cliff x pace_loss (11 x 11 x 11 = 1331 cells)")
    print("=" * 78)
    print(f"v1 (original):  WAIT in {v1_wait}/{v1_total} cells ({100*v1_wait/v1_total:.1f}%)")
    print(f"v2 (rewrite):   WAIT in {v2_wait}/{v2_total} cells ({100*v2_wait/v2_total:.1f}%)")

    # Decision-surface slices: p_sc (rows) x cliff (cols), at two fixed pace
    # levels, showing v1's flatness vs v2's actual boundary movement.
    def render_slice(fn, pace_fixed, **kwargs):
        header = "      " + "".join(f"c={c:.1f} " for c in cliff_values)
        lines = [header]
        for p_sc in p_sc_values:
            row = f"p={p_sc:.1f} "
            for cliff in cliff_values:
                inputs = SCGambleInputs(p_sc, pace_fixed, cliff, n_horizon)
                result = fn(inputs, **kwargs) if kwargs else fn(inputs)
                row += (" W    " if result["recommendation"] == "WAIT" else " .    ")
            lines.append(row)
        return "\n".join(lines)

    for pace_fixed in (0.1, 0.5):
        print(f"\n--- v1, pace_loss/lap={pace_fixed}s (W=WAIT, .=NO_ADVANTAGE) ---")
        print(render_slice(evaluate_sc_gamble_v1, pace_fixed))
        print(f"\n--- v2, pace_loss/lap={pace_fixed}s (W=WAIT, .=NO_ADVANTAGE) ---")
        print(render_slice(evaluate_sc_gamble_v2, pace_fixed))

    # Sweep 2: does the DEFAULT constants sweep matter, at the structural
    # level? (pit-loss cost, SC pit-loss cost) - per point 5 of the plan.
    print("\n" + "=" * 78)
    print("SWEEP 2: v2 WAIT-fraction under alternative pit-loss constants")
    print("=" * 78)
    for normal_pit, sc_pit in [(22.0, 11.0), (22.0, 16.0), (18.0, 11.0), (25.0, 20.0)]:
        n_wait, n_total = count_wait(evaluate_sc_gamble_v2, normal_pit_loss_s=normal_pit, sc_pit_loss_s=sc_pit)
        print(f"  normal_pit_loss_s={normal_pit:<5} sc_pit_loss_s={sc_pit:<5} "
              f"-> WAIT in {100*n_wait/n_total:.1f}% of cells")

    # Sweep 3: the REALISTIC range - circuit-level p_sc priors in this
    # project run ~2-15% (BP §9's own examples: Baku 0.098, Abu Dhabi 0.044),
    # cliff_probability_next_5_laps is calibrated against a 0.017 threshold
    # (BP §9), and the real Bahrain run's marginal pace loss stayed under
    # ~0.15s/lap. The 0->1 sweep above answers "is the formula sane at the
    # extremes"; THIS sweep answers "does the boundary move within the range
    # HERMES will actually see."
    print("\n" + "=" * 78)
    print("SWEEP 3: REALISTIC parameter ranges only (p_sc 0-0.15, cliff 0-0.1, pace 0-0.3)")
    print("=" * 78)
    p_sc_real = np.linspace(0.0, 0.15, 8)
    cliff_real = np.linspace(0.0, 0.10, 6)
    pace_real = np.linspace(0.0, 0.30, 7)

    def count_wait_grid(fn, p_vals, c_vals, pa_vals, **kwargs):
        n_wait, n_total = 0, 0
        for p_sc, cliff, pace in itertools.product(p_vals, c_vals, pa_vals):
            inputs = SCGambleInputs(p_sc, pace, cliff, n_horizon)
            result = fn(inputs, **kwargs) if kwargs else fn(inputs)
            n_total += 1
            if result["recommendation"] == "WAIT":
                n_wait += 1
        return n_wait, n_total

    v1_r, tot_r = count_wait_grid(evaluate_sc_gamble_v1, p_sc_real, cliff_real, pace_real)
    v2_r, _ = count_wait_grid(evaluate_sc_gamble_v2, p_sc_real, cliff_real, pace_real)
    print(f"v1 in realistic range: WAIT in {v1_r}/{tot_r} ({100*v1_r/tot_r:.1f}%)")
    print(f"v2 in realistic range: WAIT in {v2_r}/{tot_r} ({100*v2_r/tot_r:.1f}%)")
    print("(if these two numbers are close, the fix is directionally correct")
    print(" but small in this project's actual operating range - the dominant")
    print(" lever is the pit-loss GAP itself, see Sweep 2, not the missing")
    print(" cliff term this fix adds.)")

    # Tests structural sensitivity correctly: HOLD p_sc constant, vary the
    # cost of waiting (degradation + cliff risk). Higher degradation/cliff
    # should push the decision TOWARD NO_ADVANTAGE_TO_WAITING, not toward
    # WAIT - an earlier version of this comment/example had that backwards
    # (varied all three together and asserted "high everything -> GAMBLE",
    # which conflates "high p_sc" (genuinely favors waiting) with "high
    # degradation/cliff" (should discourage it) into one misleading case).
    print("\n" + "=" * 78)
    print("Worked examples: holding p_sc constant, does raising the cost of waiting")
    print("(degradation + cliff) push the decision toward NO_ADVANTAGE_TO_WAITING?")
    print("=" * 78)
    low_risk = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.0,
                               cliff_probability_next_n_laps=0.0, n_laps_horizon=5)
    high_risk = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.30,
                                cliff_probability_next_n_laps=0.10, n_laps_horizon=5)
    for label, inputs in (("same p_sc=0.15, zero degradation/cliff", low_risk),
                          ("same p_sc=0.15, high degradation/cliff", high_risk)):
        r1, r2 = evaluate_sc_gamble_v1(inputs), evaluate_sc_gamble_v2(inputs)
        print(f"  {label}: v1={r1['recommendation']:<22} v2={r2['recommendation']}")
    print("(expect: zero-risk case can favor WAIT since there's nothing to lose by waiting; "
          "the high-risk case should flip to NO_ADVANTAGE_TO_WAITING - if it doesn't, that's a "
          "real utility-formulation issue, not just a labeling one.)")


# ============================================================================
# 4. EMPIRICAL CALIBRATION - real per-circuit pit-loss constants
# ============================================================================
# Run this file's own repo needs (not hermes_master.py's, not sc-gamble.py's -
# this stays an isolated experiment per the explicit instruction to keep
# sc-gamble.py and hermes_master.py both unchanged).
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "."))
ARCHIVE_PER_RACE_PATH = REPO_ROOT / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv"
ARCHIVE_EVENTS_PATH = REPO_ROOT / "checkpoints" / "rival_knowledge" / "archive_event_summary_enriched.csv"
CACHE_DIR = Path(os.environ.get("GCS_CACHE_DIR", REPO_ROOT / "gcs_cache"))

# ASSUMPTION, flagged explicitly (matches this project's own convention,
# e.g. MIN_CLEAN_LAPS_FOR_BASELINE, MIN_ROWS_PER_GROUP): a per-circuit
# SC-window pit-loss estimate needs at least this many real caution-window
# pit events before it's trusted; below that, fall back to the global
# pooled estimate (still real data, just less granular) rather than guess.
MIN_EVENTS_FOR_CIRCUIT_SC_ESTIMATE = 5
MAX_PLAUSIBLE_PIT_STOP_S = 120.0  # same physical cap as knowledge.py's compute_pit_lane_loss -
                                    # no real pit stop takes 2+ minutes


def compute_real_pit_losses_from_laps(cache_dir: Path) -> pd.DataFrame:
    """STEP 2b: bypasses pit_loss_constant_s ENTIRELY (confirmed above to be
    a duplicated per-(circuit, year) constant, not a per-stop measurement -
    no amount of correct SC/VSC lap-matching can extract a real split from
    it). Instead, computes genuinely independent per-stop pit-lane
    durations directly from each race's own laps_features.csv, using the
    same PitOutTime-PitInTime logic as knowledge.py's compute_pit_lane_loss
    (matching a pit-in lap to the following pit-out lap, real elapsed
    time), then classifies EACH stop's SC/green status individually - the
    thing Step 2 needed but the archive files can't supply. This scans
    every real Race-session laps_features.csv already in the cache (no
    dependency on the archive files' race-name list at all), so it also
    covers any race the archive happens to be missing."""
    laps_files = sorted(cache_dir.glob("clean/features/*/*/R/laps_features.csv"))
    print(f"[calibration] STEP 2b: found {len(laps_files)} real Race-session laps_features.csv "
          f"files to compute genuine per-stop pit-loss durations from.")
    rows = []
    n_files_used = 0
    for laps_path in laps_files:
        year_str, race_folder = laps_path.parts[-4], laps_path.parts[-3]
        cols = pd.read_csv(laps_path, nrows=0).columns
        needed = {"Driver", "LapNumber", "PitInTime", "PitOutTime", "is_pit_in", "is_pit_out"}
        if not needed <= set(cols):
            continue  # can't compute a real duration without these - skip, don't guess
        n_files_used += 1
        usecols = list(needed | ({"is_sc_lap", "is_vsc_lap"} & set(cols)))
        laps = pd.read_csv(laps_path, usecols=usecols)
        laps_sorted = laps.sort_values(["Driver", "LapNumber"])
        indexed = laps_sorted.set_index(["Driver", "LapNumber"])
        pit_in_rows = laps_sorted[laps_sorted["is_pit_in"].fillna(False).astype(bool)]
        for _, in_row in pit_in_rows.iterrows():
            driver, in_lap = in_row["Driver"], in_row["LapNumber"]
            out_lap = in_lap + 1
            if (driver, out_lap) not in indexed.index:
                continue
            out_row = indexed.loc[(driver, out_lap)]
            if isinstance(out_row, pd.DataFrame):
                out_row = out_row.iloc[0]
            if not bool(out_row.get("is_pit_out", False)):
                continue
            try:
                duration = (pd.to_timedelta(out_row["PitOutTime"]) - pd.to_timedelta(in_row["PitInTime"])).total_seconds()
            except Exception:
                continue
            if pd.isna(duration) or duration <= 0 or duration > MAX_PLAUSIBLE_PIT_STOP_S:
                continue
            under_caution = bool(in_row.get("is_sc_lap", False)) or bool(in_row.get("is_vsc_lap", False)) or \
                            bool(out_row.get("is_sc_lap", False)) or bool(out_row.get("is_vsc_lap", False))
            rows.append({"race": race_folder.replace("_", " "), "year": int(year_str),
                         "duration_s": duration, "under_caution": under_caution})
    result = pd.DataFrame(rows)
    print(f"[calibration] STEP 2b: used {n_files_used}/{len(laps_files)} files (rest missing "
          f"PitInTime/PitOutTime/is_pit_in/is_pit_out columns), computed {len(result)} REAL "
          f"pit-stop durations directly from raw timestamps.")
    return result


def derive_real_pit_loss_constants(real_durations: pd.DataFrame) -> tuple:
    """Same shape as derive_pit_loss_constants, but built on STEP 2b's
    genuinely independent durations - this is the version that should
    actually drive Step 4, since it's the only one where an SC/green split
    is a real split of real per-stop values.

    TERMINOLOGY CAVEAT (kept as sc_pit_loss_s for backward compatibility
    with evaluate_sc_gamble_v2's existing parameter name and Step 4's
    generic table-consuming code, but be precise about what it measures
    when writing this up): `under_caution` flags whether the pit-in OR
    pit-out lap overlapped an is_sc_lap/is_vsc_lap window - it measures
    'this stop's elapsed pit-lane time, for stops whose entry/exit lap
    overlapped a caution period', NOT 'the driver chose to pit because the
    SC made it cheaper', and NOT a validated causal SC-specific discount.
    A car that entered under green and happened to exit as the SC ended
    counts as under_caution=True here. Report this as an empirical
    CAUTION-WINDOW pit-stop duration, not as 'the SC pit loss' - the
    distinction matters because the two could diverge in either direction
    depending on how much of the actual stop happened under reduced pit
    lane speed versus how much happened under a still-green in-lap/out-lap."""
    rows = []
    for race, g in real_durations.groupby("race"):
        green = g[~g["under_caution"]]["duration_s"]
        sc = g[g["under_caution"]]["duration_s"]
        rows.append({
            "race": race,
            "normal_pit_loss_s": float(green.median()) if len(green) > 0 else None,
            "n_green_events": len(green),
            "sc_pit_loss_s": (float(sc.median()) if len(sc) >= MIN_EVENTS_FOR_CIRCUIT_SC_ESTIMATE else None),
            "n_sc_events": len(sc),
        })
    per_circuit = pd.DataFrame(rows)

    global_green = real_durations[~real_durations["under_caution"]]["duration_s"]
    global_sc = real_durations[real_durations["under_caution"]]["duration_s"]
    pooled = {
        "normal_pit_loss_s": float(global_green.median()) if len(global_green) > 0 else None,
        "n_green_events": len(global_green),
        "sc_pit_loss_s": float(global_sc.median()) if len(global_sc) > 0 else None,
        "n_sc_events": len(global_sc),
    }
    return per_circuit, pooled



def load_archive_per_race(path: Path):
    """Step 1: real per-circuit normal_pit_loss_s, if the aggregated file
    exists. Column names aren't pinned down anywhere in the project's own
    documentation, so this prints what's actually there (BP's own "never
    guess a schema, inspect real data first" convention) rather than
    assuming - same defensive approach hermes_master.py already uses."""
    if not path.exists():
        print(f"[calibration] {path} not found - normal_pit_loss_s will be "
              f"derived from archive_event_summary_enriched.csv's raw events instead "
              f"(step 1 falls through to the same source step 2 already needs).")
        return None
    df = pd.read_csv(path)
    print(f"[calibration] {path.name} columns: {list(df.columns)}")
    return df


def load_archive_events(path: Path):
    """Step 2 depends on this file existing - it's the only place a REAL,
    per-event pit-loss constant with a known lap number lives (BP §4.5:
    archive_event_summary_enriched.csv has pit_loss_constant_s, year, race,
    session, actual_pit_lap)."""
    if not path.exists():
        print(f"[calibration] {path} not found - cannot derive a real "
              f"sc_pit_loss_s at all without it. Stopping calibration here; "
              f"nothing downstream can be trusted without this file.")
        return None
    df = pd.read_csv(path)
    print(f"[calibration] {path.name} columns: {list(df.columns)}")
    required = {"year", "race", "session", "actual_pit_lap", "pit_loss_constant_s"}
    missing = required - set(df.columns)
    if missing:
        print(f"[calibration] WARNING: expected columns {missing} not found - "
              f"the real column names differ from what BP §2.2 documents; "
              f"update this loader to match before trusting anything below.")
    return df


def race_to_folder_name(race: str) -> str:
    """archive_event_summary_enriched.csv / archive_per_race_analysis.csv use
    SPACE-separated race names (the raw tracinginsights convention, RACE_TI
    in knowledge.py's own terms), but gcs_cache/clean/features/ folders use
    UNDERSCORE names (the FastF1 convention, RACE_FASTF1). Confirmed by the
    diagnostic: real folder '2018/Abu_Dhabi_Grand_Prix/R/...' vs the archive's
    own 'Abu Dhabi Grand Prix'. Convert ONLY for path lookups - the archive's
    own 'race' field is left untouched everywhere else so it still joins
    cleanly against archive_per_race_analysis.csv, which uses the same
    space-separated convention."""
    return str(race).replace(" ", "_")


def classify_events_by_caution(events_df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """The actual empirical test (step 2): for each real historical pit
    event, was TrackStatus showing SC/VSC on that EXACT lap number? This
    measures the thing we actually need directly - 'do real pit stops taken
    under caution really cost less' - rather than inferring it from SC
    duration (BP's own caution: SC duration in laps tells you how long a
    caution period lasts, not what a pit stop taken during one costs).
    is_sc_lap/is_vsc_lap are track-wide flags, so any driver's row for that
    LapNumber in that race answers the question - we don't need to match
    the specific driver who took the stop.

    PERFORMANCE FIX: an earlier version called pd.read_csv once PER EVENT
    ROW - with ~5900 events sharing only ~170 distinct (year, race, session)
    files, that's ~5900 full-file reads instead of ~170, which is slow
    enough on Windows disk I/O to look like a hang. Each laps_features.csv
    is now read exactly once, cached as a set of caution lap numbers keyed
    by its path, and every event referencing that race does an O(1) set
    lookup against the cached result instead of re-reading and re-filtering
    the whole file."""
    caution_laps_cache: dict = {}   # laps_path -> set of LapNumbers under SC or VSC
    full_sc_laps_cache: dict = {}   # laps_path -> set of LapNumbers under full SC (not VSC)
    rows = []
    missing_laps_files = 0
    skipped_incomplete = 0
    n_events = len(events_df)
    n_files_read = 0

    for i, (_, ev) in enumerate(events_df.iterrows()):
        if n_events > 200 and i % 500 == 0:
            print(f"[calibration] classifying event {i}/{n_events} "
                  f"({n_files_read} distinct laps_features.csv read so far)...")
        year, race, session = ev.get("year"), ev.get("race"), ev.get("session")
        lap_number, pit_loss = ev.get("actual_pit_lap"), ev.get("pit_loss_constant_s")
        if pd.isna(year) or pd.isna(race) or pd.isna(session) or pd.isna(lap_number) or pd.isna(pit_loss):
            skipped_incomplete += 1
            continue
        laps_path = cache_dir / "clean" / "features" / str(int(year)) / race_to_folder_name(race) / str(session) / "laps_features.csv"

        if laps_path not in caution_laps_cache:
            if not laps_path.exists():
                caution_laps_cache[laps_path] = None  # sentinel: file missing, don't retry
                full_sc_laps_cache[laps_path] = None
            else:
                n_files_read += 1
                cols = pd.read_csv(laps_path, nrows=0).columns
                usecols = [c for c in ("LapNumber", "is_sc_lap", "is_vsc_lap") if c in cols]
                laps = pd.read_csv(laps_path, usecols=usecols)
                sc = laps.get("is_sc_lap")
                vsc = laps.get("is_vsc_lap")
                caution_mask = (sc.fillna(False) if sc is not None else False) | \
                                (vsc.fillna(False) if vsc is not None else False)
                caution_laps_cache[laps_path] = set(laps.loc[caution_mask, "LapNumber"]) if isinstance(caution_mask, pd.Series) else set()
                full_sc_mask = sc.fillna(False) if sc is not None else pd.Series([False] * len(laps))
                full_sc_laps_cache[laps_path] = set(laps.loc[full_sc_mask, "LapNumber"])

        if caution_laps_cache[laps_path] is None:
            missing_laps_files += 1
            continue

        under_caution = int(lap_number) in caution_laps_cache[laps_path]
        under_full_sc = int(lap_number) in full_sc_laps_cache[laps_path]
        rows.append({"race": race, "pit_loss_constant_s": float(pit_loss),
                      "under_caution": under_caution, "under_full_sc": under_full_sc})

    if skipped_incomplete:
        print(f"[calibration] {skipped_incomplete} events skipped - missing year/race/session/lap/pit_loss field.")
    if missing_laps_files:
        print(f"[calibration] {missing_laps_files} events skipped - matching laps_features.csv not in {cache_dir}.")
    print(f"[calibration] read {n_files_read} distinct laps_features.csv files "
          f"(instead of one per event) to classify {n_events} events.")
    result = pd.DataFrame(rows)
    print(f"[calibration] {len(result)} real pit events successfully classified "
          f"({int(result['under_caution'].sum()) if not result.empty else 0} under SC/VSC, "
          f"{int((~result['under_caution']).sum()) if not result.empty else 0} green-flag).")
    return result


def diagnose_value_duplication(classified: pd.DataFrame):
    """A direct test for a suspicion, not an assumption: if sc_pit_loss_s
    keeps coming out numerically identical (or near-identical) to
    normal_pit_loss_s for the same circuit, that's the signature of
    pit_loss_constant_s being a shared value duplicated across many event
    rows (e.g. per race, or per driver-per-race) rather than a genuinely
    independent per-stop measurement - in which case NO split of the
    events (by SC/green, by anything else) can produce a real answer,
    regardless of how correct the SC/VSC lap-matching itself is. This
    checks it directly: how many DISTINCT pit_loss_constant_s values
    actually exist per circuit, versus how many events claim to have one."""
    print("\n" + "=" * 78)
    print("DIAGNOSTIC: is pit_loss_constant_s a genuinely per-event value, or a "
          "duplicated per-race/per-driver constant? (checked directly, not assumed)")
    print("=" * 78)
    summary = classified.groupby("race")["pit_loss_constant_s"].agg(
        n_events="count", n_distinct_values="nunique").reset_index()
    summary["duplication_ratio"] = 1 - (summary["n_distinct_values"] / summary["n_events"])
    summary = summary.sort_values("duplication_ratio", ascending=False)
    print(summary.to_string(index=False))
    high_dup = summary[summary["duplication_ratio"] > 0.5]
    print(f"\n{len(high_dup)}/{len(summary)} circuits have >50% of events sharing a "
          f"pit_loss_constant_s value with at least one other event in that circuit.")
    if len(high_dup) > len(summary) * 0.3:
        print("CONCLUSION: pit_loss_constant_s is NOT reliably an independent per-event "
              "measurement across this archive - it repeats too often within a circuit for "
              "that to be coincidental. A median split by SC/green status of this column "
              "cannot produce a genuine SC-specific estimate no matter how correct the SC/VSC "
              "lap-matching is, because the input values themselves aren't independent per "
              "stop. This is itself the calibration's real, reportable finding for circuits "
              "where duplication is high: the available data does not support a clean "
              "SC-specific pit-loss estimate, for a data-quality reason distinct from small "
              "sample size (BP's own stopping condition: 'if the historical data cannot "
              "support a clean SC-specific estimate, that is also a valid result').")
    print()


def derive_pit_loss_constants(classified: pd.DataFrame) -> tuple:
    """Per-circuit constants where the sample supports them, plus a global
    pooled fallback (still real data, just less granular) for circuits that
    don't have enough real caution-window events - explicitly None, never
    fabricated, exactly per the 'if the data can't support a clean estimate,
    that's a valid result' instruction."""
    per_circuit_rows = []
    for race, g in classified.groupby("race"):
        green = g[~g["under_caution"]]["pit_loss_constant_s"]
        sc = g[g["under_caution"]]["pit_loss_constant_s"]
        per_circuit_rows.append({
            "race": race,
            "normal_pit_loss_s": float(green.median()) if len(green) > 0 else None,
            "n_green_events": len(green),
            "sc_pit_loss_s": (float(sc.median()) if len(sc) >= MIN_EVENTS_FOR_CIRCUIT_SC_ESTIMATE else None),
            "n_sc_events": len(sc),
        })
    per_circuit = pd.DataFrame(per_circuit_rows)

    global_green = classified[~classified["under_caution"]]["pit_loss_constant_s"]
    global_sc = classified[classified["under_caution"]]["pit_loss_constant_s"]
    pooled = {
        "normal_pit_loss_s": float(global_green.median()) if len(global_green) > 0 else None,
        "n_green_events": len(global_green),
        "sc_pit_loss_s": float(global_sc.median()) if len(global_sc) > 0 else None,
        "n_sc_events": len(global_sc),
    }
    return per_circuit, pooled


def normal_pit_loss_from_per_race(per_race_df: pd.DataFrame) -> pd.DataFrame:
    """STEP 1, done properly: archive_per_race_analysis.csv already IS the
    real per-circuit normal_pit_loss_s (BP §2.2: 'per-circuit empirical
    pit-loss constants') - an earlier version of this script loaded the
    file, printed its columns, and then never actually used it, re-deriving
    the same number from the events file instead. Fixed: use it directly.
    Filters on whatever reliability columns the real file turns out to
    have (printed below so nothing is assumed) - `known_backfilled_limitation`
    in particular sounds like exactly the kind of flag that should exclude a
    row from a trusted estimate.

    IMPORTANT (found by inspecting the real file, not assumed): despite the
    blueprint calling this 'per-circuit', the real file has ~174 rows for
    ~25-30 unique circuit names - i.e. one row per (circuit, year), not one
    row per circuit. This aggregates across years into a single estimate
    per circuit (median pit_loss_constant_s, summed n_events for sample-size
    transparency) - a deliberate, stated simplification: pit-lane geometry
    is mostly stable year to year for the same circuit, but this does pool
    across regulation eras and any real pit-lane changes, which a more
    careful version could stratify by era if the per-era numbers turn out
    to differ meaningfully."""
    print(f"[calibration] status value counts:\n{per_race_df['status'].value_counts().to_string()}")
    print(f"[calibration] known_backfilled_limitation value counts:\n"
          f"{per_race_df['known_backfilled_limitation'].value_counts(dropna=False).to_string()}")
    reliable = per_race_df.copy()
    uniq = set(reliable["known_backfilled_limitation"].dropna().unique())
    if uniq <= {True, False}:
        before = len(reliable)
        reliable = reliable[reliable["known_backfilled_limitation"] != True]  # noqa: E712
        print(f"[calibration] dropped {before - len(reliable)} rows flagged known_backfilled_limitation=True")

    n_rows_before = len(reliable)
    n_unique_races = reliable["race"].nunique()
    if n_rows_before != n_unique_races:
        print(f"[calibration] {n_rows_before} rows but only {n_unique_races} unique circuit names - "
              f"this file is (circuit, year), not one row per circuit as documented. Aggregating "
              f"across years per circuit (median pit_loss_constant_s, summed n_events).")

    aggregated = reliable.dropna(subset=["pit_loss_constant_s"]).groupby("race").agg(
        normal_pit_loss_s=("pit_loss_constant_s", "median"),
        n_events=("n_events", "sum"),
        n_years=("pit_loss_constant_s", "count"),
    ).reset_index()
    return aggregated


def diagnose_laps_features_paths(events_df: pd.DataFrame, cache_dir: Path):
    """When classification finds ZERO matches across thousands of events,
    that's too total to be an ordinary coverage gap - almost certainly a
    wrong path assumption on this script's part, not missing data. Rather
    than guess again, this prints exactly what it's checking so the actual
    mismatch (a race-name convention, a session code, a different cache
    root) is visible directly, per this project's own 'inspect real data
    before assuming' rule."""
    print(f"\n[diagnose] CACHE_DIR resolves to: {cache_dir.resolve()}")
    print(f"[diagnose] CACHE_DIR exists: {cache_dir.exists()}")
    features_root = cache_dir / "clean" / "features"
    print(f"[diagnose] {features_root} exists: {features_root.exists()}")
    actual_laps_files = []
    if features_root.exists():
        actual_laps_files = list(features_root.glob("*/*/*/laps_features.csv"))
        print(f"[diagnose] {len(actual_laps_files)} real laps_features.csv files found under {features_root}")
        for p in actual_laps_files[:5]:
            print(f"[diagnose]   example real path: {p.relative_to(features_root)}")

    sample = events_df[["year", "race", "session"]].drop_duplicates().head(5)
    print(f"[diagnose] first {len(sample)} distinct (year, race, session) combos from the events file, "
          f"and the exact path this script tried for each:")
    for _, r in sample.iterrows():
        tried = cache_dir / "clean" / "features" / str(int(r["year"])) / race_to_folder_name(r["race"]) / str(r["session"]) / "laps_features.csv"
        print(f"[diagnose]   year={r['year']!r} race={r['race']!r} session={r['session']!r} "
              f"-> tried: {tried}  (exists: {tried.exists()})")
    if actual_laps_files:
        print(f"[diagnose] compare the 'tried' paths above against the real example paths - "
              f"a mismatched year/race/session format there is almost certainly the cause.")


def run_calibration():
    print("=" * 78)
    print("STEP 1+2: loading real archive files")
    print("=" * 78)
    per_race_df = load_archive_per_race(ARCHIVE_PER_RACE_PATH)
    events_df = load_archive_events(ARCHIVE_EVENTS_PATH)

    normal_from_archive = None
    if per_race_df is not None:
        print("\n" + "=" * 78)
        print("STEP 1 (from archive_per_race_analysis.csv directly, not re-derived)")
        print("=" * 78)
        normal_from_archive = normal_pit_loss_from_per_race(per_race_df)
        print(normal_from_archive.to_string(index=False))

    classified = pd.DataFrame()
    if events_df is not None:
        classified = classify_events_by_caution(events_df, CACHE_DIR)
        if not classified.empty:
            diagnose_value_duplication(classified)
            print("\n" + "=" * 78)
            print(f"STEP 2 results (from archive pit_loss_constant_s - see the duplication "
                  f"diagnostic above for why these are NOT trusted for Step 4)")
            print("=" * 78)
            per_circuit_archive, pooled_archive = derive_pit_loss_constants(classified)
            print(per_circuit_archive.sort_values("n_sc_events", ascending=False).to_string(index=False))
        else:
            diagnose_laps_features_paths(events_df, CACHE_DIR)
            print("\nNo events could be classified against a real is_sc_lap flag.")

    # --- STEP 2b: the version that actually matters - real per-stop
    # durations computed directly from raw PitInTime/PitOutTime, bypassing
    # the archive's compromised pit_loss_constant_s column entirely.
    print("\n" + "=" * 78)
    print("STEP 2b: REAL per-stop pit-loss durations, computed directly from raw timestamps")
    print("(the 'sc_pit_loss_s' column below is a CAUTION-WINDOW pit duration - elapsed")
    print(" PitIn->PitOut time for stops whose entry/exit lap overlapped is_sc_lap/is_vsc_lap -")
    print(" NOT a validated claim that the driver pitted BECAUSE the SC made it cheaper.")
    print(" Report it that way: 'empirical caution-window pit-stop duration', not 'the SC")
    print(" pit-loss', since those could genuinely diverge depending on how much of the")
    print(" specific stop happened under reduced pit-lane speed vs. a still-green in/out lap.)")
    print("=" * 78)
    real_durations = compute_real_pit_losses_from_laps(CACHE_DIR)
    per_circuit_real, pooled_real = (pd.DataFrame(), {"normal_pit_loss_s": None, "sc_pit_loss_s": None,
                                                        "n_green_events": 0, "n_sc_events": 0})
    if not real_durations.empty:
        per_circuit_real, pooled_real = derive_real_pit_loss_constants(real_durations)
        print(per_circuit_real.rename(columns={"sc_pit_loss_s": "caution_window_pit_duration_s"})
              .sort_values("n_sc_events", ascending=False).to_string(index=False))
        n_with_estimate = per_circuit_real["sc_pit_loss_s"].notna().sum()
        print(f"\n{n_with_estimate}/{len(per_circuit_real)} circuits have >= "
              f"{MIN_EVENTS_FOR_CIRCUIT_SC_ESTIMATE} REAL caution-window stops for their OWN "
              f"caution_window_pit_duration_s estimate.")
        print(f"Global pooled (REAL): normal_pit_loss_s={pooled_real['normal_pit_loss_s']} "
              f"(n={pooled_real['n_green_events']}), "
              f"caution_window_pit_duration_s={pooled_real['sc_pit_loss_s']} (n={pooled_real['n_sc_events']})")
        diff = None
        if pooled_real["normal_pit_loss_s"] is not None and pooled_real["sc_pit_loss_s"] is not None:
            diff = pooled_real["normal_pit_loss_s"] - pooled_real["sc_pit_loss_s"]
            print(f"Observed difference: {diff:.2f}s - {'a large, real SC-window discount' if diff > 5 else 'nowhere near the 11s/50pct discount the original assumption used; the data does not support a large reduction'}.")
        if per_race_df is not None and normal_from_archive is not None:
            cross_check = normal_from_archive.merge(
                per_circuit_real[["race", "normal_pit_loss_s"]], on="race", suffixes=("_archive", "_real"))
            cross_check["diff_s"] = (cross_check["normal_pit_loss_s_archive"] - cross_check["normal_pit_loss_s_real"]).abs()
            print(f"\nCross-check: archive-derived vs directly-computed normal_pit_loss_s agree to "
                  f"within {cross_check['diff_s'].mean():.2f}s on average across "
                  f"{len(cross_check)} circuits present in both - {'a good sanity check' if cross_check['diff_s'].mean() < 2 else 'a notable discrepancy worth investigating'}.")

    else:
        print("\nNo real durations could be computed (raw PitInTime/PitOutTime/is_pit_in/"
              "is_pit_out columns not found in the cached laps_features.csv files) - this IS a "
              "valid stopping result: neither the archive files nor the raw cache can support a "
              "genuine SC-specific pit-loss estimate with what's currently available.")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC (not a pit-loss estimate): compute_historical_sc_duration gives mean SC "
          "duration in LAPS per circuit - kept separate from the pit-loss numbers above since SC "
          "duration in laps is not the same quantity as SC pit-loss cost in seconds.")
    print("=" * 78)

    # --- STEP 4: re-run Sweep 1 + Sweep 2 with REAL constants ---
    print("\n" + "=" * 78)
    print("STEP 4: Sweep 1 + Sweep 2, re-run with REAL circuit constants")
    print("=" * 78)
    n_horizon = 5
    p_sc_real = np.linspace(0.0, 0.15, 8)
    cliff_real = np.linspace(0.0, 0.10, 6)
    pace_real = np.linspace(0.0, 0.30, 7)
    ASSUMED_SC_PIT_LOSS_S = 11.0  # v1's original, unvalidated guess - absolute last resort only

    def wait_fraction(normal_pit, sc_pit):
        n_wait, n_total = 0, 0
        for p_sc, cliff, pace in itertools.product(p_sc_real, cliff_real, pace_real):
            inputs = SCGambleInputs(p_sc, pace, cliff, n_horizon)
            result = evaluate_sc_gamble_v2(inputs, normal_pit_loss_s=normal_pit, sc_pit_loss_s=sc_pit)
            n_total += 1
            if result["recommendation"] == "WAIT":
                n_wait += 1
        return n_wait, n_total

    def wait_cells(normal_pit, sc_pit):
        """Which exact (p_sc, cliff, pace) cells actually produce WAIT for
        this circuit's real constants - the diagnostic that tells us
        whether a nonzero WAIT% is coming from a sensible corner of the
        parameter space (high p_sc, low cliff/degradation) or from
        something that would indicate a real formulation problem."""
        cells = []
        for p_sc, cliff, pace in itertools.product(p_sc_real, cliff_real, pace_real):
            inputs = SCGambleInputs(p_sc, pace, cliff, n_horizon)
            result = evaluate_sc_gamble_v2(inputs, normal_pit_loss_s=normal_pit, sc_pit_loss_s=sc_pit)
            if result["recommendation"] == "WAIT":
                cells.append({"p_sc": p_sc, "cliff": cliff, "pace_loss": pace,
                              "saving": result["expected_saving_if_wait"]})
        return pd.DataFrame(cells)

    # Prefer STEP 2b (real per-stop durations) - it's the only source where
    # sc_pit_loss_s is a genuine split, not a duplicated constant.
    if not per_circuit_real.empty:
        sweep_table = per_circuit_real
        sc_source_label = "STEP 2b (real per-stop durations)"
        pooled_normal, pooled_sc = pooled_real["normal_pit_loss_s"], pooled_real["sc_pit_loss_s"]
        pooled_n_sc = pooled_real["n_sc_events"]
    else:
        sweep_table = normal_from_archive.copy() if normal_from_archive is not None else pd.DataFrame()
        if not sweep_table.empty:
            sweep_table["sc_pit_loss_s"] = np.nan
            sweep_table["n_sc_events"] = 0
        sc_source_label = f"assumed ({ASSUMED_SC_PIT_LOSS_S}s, no real SC-specific data available)"
        pooled_normal, pooled_sc, pooled_n_sc = None, None, 0

    if sweep_table.empty:
        print("\n[calibration] No empirical data available from any source - cannot run Step 4.")
        return

    print(f"(sc_pit_loss_s source: {sc_source_label})\n")
    print(f"{'circuit':<30} {'normal':>8} {'sc':>8} {'n_sc':>6} {'WAIT %':>8}  source")
    # FIX: both fallbacks must come from the SAME aggregation method - an
    # earlier version paired the per-circuit MEDIAN of normal_pit_loss_s
    # with the POOLED (raw-event-weighted) sc_pit_loss_s, silently mixing
    # two different aggregations under one "GLOBAL" row. Both now come from
    # the same pooled_real dict (raw-event-weighted, matching what's
    # already reported as "Global pooled (REAL)" above), falling back to
    # the per-circuit median ONLY when pooled data isn't available at all
    # (the archive-only path, where there's no pooled dict to draw from).
    global_sc_fallback = pooled_sc if pooled_sc is not None else ASSUMED_SC_PIT_LOSS_S
    global_normal_fallback = pooled_normal if pooled_normal is not None else sweep_table["normal_pit_loss_s"].dropna().median()
    n_wait, n_total = wait_fraction(global_normal_fallback, global_sc_fallback)
    print(f"{'GLOBAL':<30} {global_normal_fallback:>8.2f} "
          f"{global_sc_fallback:>8.2f} {pooled_n_sc:>6} {100*n_wait/n_total:>7.1f}%")

    best_wait_pct, best_row = -1.0, None
    for _, row in sweep_table.sort_values("n_sc_events", ascending=False, na_position="last").head(15).iterrows():
        normal = row["normal_pit_loss_s"]
        if pd.isna(normal):
            continue
        sc, n_sc = row.get("sc_pit_loss_s"), row.get("n_sc_events")
        if pd.isna(sc):
            sc = global_sc_fallback
            source = "assumed" if pooled_sc is None else "pooled"
        else:
            source = "own data"
        n_sc = 0 if pd.isna(n_sc) else int(n_sc)
        n_wait, n_total = wait_fraction(normal, sc)
        wait_pct = 100 * n_wait / n_total
        print(f"{str(row['race'])[:30]:<30} {normal:>8.2f} {sc:>8.2f} {n_sc:>6} "
              f"{wait_pct:>7.1f}%  sc from {source}")
        is_sao_paulo = "s" in str(row["race"]).lower() and "paulo" in str(row["race"]).lower()
        if is_sao_paulo or (wait_pct > best_wait_pct and wait_pct > 0):
            best_wait_pct, best_row = wait_pct, (row["race"], normal, sc)
        if is_sao_paulo:
            break  # named circuit takes priority - stop looking for a "best" alternative

    if best_row is not None:
        race_name, normal, sc = best_row
        print(f"\n{'=' * 78}")
        print(f"WAIT cells for {race_name} (WAIT%={best_wait_pct:.1f}) - which exact "
              f"(p_sc, cliff, pace_loss) combinations actually produce WAIT:")
        print("=" * 78)
        cells = wait_cells(normal, sc)
        # Precise expectation, derived algebraically (verified against the
        # actual code to machine precision, not asserted): at pace=cliff=0,
        # expected_saving_if_wait = p_sc * D, where
        # D = normal_pit_loss_s - sc_pit_loss_s - cold_tyre_penalty_s.
        # d(saving)/d(p_sc) = D + 0.5*(pace*n_horizon + cliff_penalty_scale*cliff),
        # which is >= D always. So:
        #  - if D > 0 (this circuit's caution discount exceeds the cold-tyre
        #    penalty): WAIT cells should appear across the WHOLE p_sc range
        #    tested (even p_sc near 0), shrinking in tolerated
        #    cliff/pace_loss as p_sc falls - not "only near high p_sc".
        #  - if D < 0 (the discount is smaller than the cold-tyre penalty,
        #    as in the GLOBAL pooled case): WAIT should be rare or absent
        #    everywhere in this realistic range, and MORE p_sc can
        #    legitimately make WAIT LESS attractive (a higher chance of
        #    landing in a net-unfavorable SC branch is worse, not better) -
        #    that is correct expected-value mixing, not a bug.
        # Genuinely pathological behaviour would be: WAIT appearing at
        # higher pace_loss/cliff for a FIXED p_sc than at lower pace/cliff
        # (degradation or cliff risk must never make waiting MORE
        # attractive, regardless of D's sign) - that check is unconditional.
        D = normal - sc - 1.5  # cold_tyre_penalty_s default
        print(f"D = normal - caution - cold_tyre = {D:.3f} -> "
              f"expect WAIT across the {'full' if D > 0 else 'empty/near-empty'} p_sc range "
              f"at low cliff/pace_loss, {'shrinking' if D > 0 else 'never appearing'} as "
              f"cliff/pace_loss rise (unconditional check, holds regardless of D's sign).")
        if cells.empty:
            print("(no WAIT cells found for this circuit's real constants - 0.0% is exact, not rounding)")
        else:
            print(cells.to_string(index=False))
            # Unconditional check, actually executed (not just asserted):
            # for every pair of grid points at the SAME p_sc, the one with
            # higher-or-equal pace AND cliff must never be WAIT while the
            # lower one is NO_ADVANTAGE_TO_WAITING - i.e. WAIT-ness can only
            # shrink as pace/cliff rise, never grow, regardless of D's sign.
            grid_results = {}
            for p_val, cliff_val, pace_val in itertools.product(p_sc_real, cliff_real, pace_real):
                inputs = SCGambleInputs(p_val, pace_val, cliff_val, n_horizon)
                grid_results[(p_val, cliff_val, pace_val)] = evaluate_sc_gamble_v2(
                    inputs, normal_pit_loss_s=normal, sc_pit_loss_s=sc)["recommendation"] == "WAIT"
            violations = 0
            for p_val in p_sc_real:
                pts = [(c, pa) for (pp, c, pa) in grid_results if pp == p_val]
                for c1, pa1 in pts:
                    for c2, pa2 in pts:
                        if c2 >= c1 and pa2 >= pa1 and (c2 > c1 or pa2 > pa1):
                            if grid_results[(p_val, c2, pa2)] and not grid_results[(p_val, c1, pa1)]:
                                violations += 1
            print(f"\nUnconditional monotonicity check (higher pace/cliff at fixed p_sc must never "
                  f"newly produce WAIT): {'PASS - 0 violations' if violations == 0 else f'FAIL - {violations} violations found'}.")




    print("\nIf the WAIT% column varies meaningfully across circuits above, the decision "
          "boundary is genuinely responding to real circuit-specific pit-loss economics - "
          "that's the result this calibration exists to produce.")


if __name__ == "__main__":
    run_sweep()
    print("\n\n")
    run_calibration()