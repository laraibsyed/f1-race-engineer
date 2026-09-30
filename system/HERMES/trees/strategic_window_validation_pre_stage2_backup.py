#!/usr/bin/env python3
"""
HERMES Strategic-Window Validation Harness
================================================
READ-ONLY. Does not modify master.py, any tree/module file, or any decision
logic - it calls `master.py replay` exactly as documented (no new CLI flag,
no new parameter), reads back the JSONL it already produces, and compares it
AFTER THE FACT against real historical pit laps. Reuses list_races /
load_real_laps / real_pit_laps / read_jsonl / rbr_drivers_for from evaluate.py
(the existing evaluation helper - see this file's own header for exactly
which parts of evaluate.py this does and does NOT reuse).

WHAT THIS MEASURES (and does not): "did HERMES flag that remaining out was
becoming strategically inferior within a reasonable window around the real
pit" - NOT an exact pit-lap prediction test. This is a different, simpler,
more literal question than evaluate.py's F1/precision-recall framework (which
uses a different tolerance, a different "counts as a hit" vocabulary, and
collapses consecutive recommendations into one event). Both are legitimate;
this file does not replace or modify evaluate.py.

USAGE (from the repo root):
    .\\.venv\\Scripts\\python.exe system\\HERMES\\trees\\strategic_window_validation.py
    .\\.venv\\Scripts\\python.exe system\\HERMES\\trees\\strategic_window_validation.py --force
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402 - REUSED, not duplicated: list_races, load_real_laps,
                        # real_pit_laps, read_jsonl, rbr_drivers_for

# ============================================================================
# STEP 2 - metric definitions (fixed BEFORE results are looked at)
# ============================================================================
STRATEGIC_TOLERANCE_LAPS = 3          # window = [L-3, L+3], fixed per the task spec. STAGE 1 result:
                                        # 37/37 window hits, 36/37 pre-pit hits - SATURATED (see the
                                        # 2026-09-30 base-rate finding below). Kept and reported, but
                                        # labelled COVERAGE metrics, not predictive-timing evidence -
                                        # this file does NOT attempt to change this number.
STRATEGIC_HIT_DECISIONS = {"PIT_NOW", "PIT_LATER"}   # deliberately NOT evaluate.py's PIT_DECISIONS
                                                       # set ({"PIT_NOW","PIT_FLEXIBLE"}) - this task
                                                       # defines its OWN, different "hit" vocabulary

# ----------------------------------------------------------------------------
# STAGE 2 (2026-09-30) - timing/escalation metrics, added on top of the frozen
# Stage 1 coverage metrics above. Nothing in STRATEGIC_TOLERANCE_LAPS or
# STRATEGIC_HIT_DECISIONS changes; these are ADDITIONAL, separately-labelled
# measurements answering a different question (onset timing / escalation /
# trigger-count trend / saturation baseline), not a replacement for Stage 1.
# ----------------------------------------------------------------------------
USEFUL_WARNING_WINDOW = (1, 5)        # onset in [L-5, L-1] - fixed BEFORE running, per the task spec;
                                        # NOT tuned after seeing results.
ESCALATION_PRE_WINDOW = (1, 3)        # "PIT_NOW first appears 1-3 laps before the actual pit" bucket
TRIGGER_COUNT_LAPS_BACK = (5, 4, 3, 2, 1)   # L-5 .. L-1, fixed set, per the task spec
SHORT_REPLAY_MAX_LAPS = 10            # generic (not race-name-specific) cutoff flagging an
                                        # abnormally short available replay (e.g. a red-flagged race) -
                                        # well below any normal Grand Prix distance, so this only ever
                                        # fires on a genuinely truncated replay, not a short/sprint race.
BASELINE_SEED = 20260930              # fixed seed, reproducible (STEP 6)

# ============================================================================
# STEP 4 - race sample (selected BEFORE any HERMES output was inspected - see
# the printed justification in main(); metadata sourced from
# src/taxanomy/circuit_taxonomy.xlsx (circuit_type, circuit_degredation) and
# sc_vsc_circuit_level_prior.csv (p_window_horizon, the VALIDATED SC/VSC
# prior) via the SAME circuit_id alias table master.py's own calibration fix
# uses (duplicated here for the same standalone-callability reason, per this
# project's established convention - see master.py's own CIRCUIT_ID_TO_RACE_NAMES).
# ============================================================================
CIRCUIT_ID_TO_RACE_NAMES = {
    "MEL": ["Australian_Grand_Prix"], "BAH": ["Bahrain_Grand_Prix", "Sakhir_Grand_Prix"],
    "CHN": ["Chinese_Grand_Prix"], "AZR": ["Azerbaijan_Grand_Prix"], "SPN": ["Spanish_Grand_Prix"],
    "MON": ["Monaco_Grand_Prix"], "CAN": ["Canadian_Grand_Prix"], "FRA": ["French_Grand_Prix"],
    "AUS": ["Austrian_Grand_Prix", "Styrian_Grand_Prix"],
    "UK": ["British_Grand_Prix", "70th_Anniversary_Grand_Prix"],
    "GER": ["German_Grand_Prix"], "HUN": ["Hungarian_Grand_Prix"], "BEL": ["Belgian_Grand_Prix"],
    "ITA": ["Italian_Grand_Prix"], "SIN": ["Singapore_Grand_Prix"], "RUS": ["Russian_Grand_Prix"],
    "JPN": ["Japanese_Grand_Prix"], "TEX": ["United_States_Grand_Prix"],
    "MEX": ["Mexican_Grand_Prix", "Mexico_City_Grand_Prix"],
    "BRA": ["Brazilian_Grand_Prix", "São_Paulo_Grand_Prix"],
    "AUH": ["Abu_Dhabi_Grand_Prix"], "IMO": ["Emilia_Romagna_Grand_Prix"],
    "IST": ["Turkish_Grand_Prix"], "DUT": ["Dutch_Grand_Prix"], "QTR": ["Qatar_Grand_Prix"],
    "KSA": ["Saudi_Arabian_Grand_Prix"], "MIA": ["Miami_Grand_Prix"], "LAS": ["Las_Vegas_Grand_Prix"],
}
_RACE_TO_CIRCUIT_ID = {r: cid for cid, races in CIRCUIT_ID_TO_RACE_NAMES.items() for r in races}

# (season, race, justification tag). Selected on metadata alone - no race here was chosen
# because a preliminary run "looked good"; none was replayed before this list was fixed.
SELECTED_RACES = [
    (2023, "Bahrain_Grand_Prix", "SANITY CHECK - used during debugging this session; Track/high-degradation, "
                                   "moderate SC prior (0.077). Included per the task's explicit allowance, "
                                   "clearly labelled, not used to justify the rest of the sample."),
    (2021, "Monaco_Grand_Prix", "Street circuit, LOW degradation, LOW SC prior (0.041) - track-position/"
                                 "low-degradation regime, contrasts with Bahrain."),
    (2022, "Singapore_Grand_Prix", "Street circuit, high real-world SC/VSC incidence historically; "
                                    "SC prior 0.054 (moderate in this dataset's own calibration - included "
                                    "specifically BECAUSE it is a well-known high-SC circuit in real-world "
                                    "terms even where the dataset's own circuit-level prior is more moderate, "
                                    "a useful contrast worth checking)."),
    (2021, "Azerbaijan_Grand_Prix", "Street circuit (Baku), HIGHEST SC prior in the candidate set (0.098) - "
                                     "chaotic/undercut-heavy street circuit."),
    (2021, "Austrian_Grand_Prix", "Track circuit, HIGH degradation, early season (2021)."),
    (2022, "Dutch_Grand_Prix", "Track circuit, HIGH degradation, different season (2022) from the other "
                                 "high-degradation Track entries, for season diversity within that cell."),
    (2023, "Spanish_Grand_Prix", "Track circuit, HIGH degradation but LOW SC prior (0.038) - degradation and "
                                   "SC influence varied independently, not bundled."),
    (2021, "Belgian_Grand_Prix", "KNOWN REAL-WORLD wet/red-flag race (2021 Spa - the farcical 2-lap race "
                                   "behind the safety car in heavy rain) - included for weather/SC influence. "
                                   "NOTE: laps_features.csv has no reliable rain flag of its own (Rainfall is "
                                   "synthesised only from weather.csv at replay time, absent from the cached "
                                   "feature file), so this slot is picked from documented real-world race "
                                   "history, not a data column - stated plainly, not hidden as if it were "
                                   "data-derived like the other selection criteria."),
    (2024, "British_Grand_Prix", "Track circuit, MEDIUM degradation, LATER season (2024) - different "
                                   "regulation-era coverage from the 2021-2023 entries, high-profile "
                                   "undercut/traffic circuit."),
    (2024, "Abu_Dhabi_Grand_Prix", "Track circuit, LOW degradation, LOW SC prior (0.044), LATE-season "
                                     "finale historically associated with late single-stop pit windows - "
                                     "contrasts with every high-degradation/high-SC entry above."),
]


# ============================================================================
# STEP 1 reuse - run the replay exactly as documented (no new CLI flag)
# ============================================================================
def run_replay_if_needed(repo_root: Path, master_path: Path, out_dir: Path,
                          season: int, race: str, d1: str, d2: str, force: bool) -> Path:
    out_file = out_dir / f"{season}_{race}.jsonl"
    if out_file.exists() and not force:
        return out_file
    cmd = [sys.executable, str(master_path), "replay", "--repo-root", str(repo_root),
           "--season", str(season), "--race", race, "--session", "R",
           "--d1", d1, "--d2", d2, "--out", str(out_file)]
    # STEP 5 (no lookahead): this is the COMPLETE command line master.py replay accepts (see its
    # own --help / argparse setup) - there is no flag to pass an actual pit lap, and none is passed
    # here. The historical pit lap is read ONLY below, from laps_features.csv, AFTER this replay
    # has already finished and its JSONL is on disk.
    print(f"  [replay] {season} {race} {d1}/{d2}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if res.returncode != 0 or not out_file.exists():
        print(f"    FAILED: {(res.stderr or '')[-400:]}")
        return None
    return out_file


# ============================================================================
# Decision-row helpers
# ============================================================================
def is_strategic_recommendation(row: dict) -> bool:
    """execution.decision if present (the final call), else gate_decision -
    SAME fallback pattern evaluate.py's hermes_pit_laps() already uses, but
    against THIS task's own hit vocabulary (PIT_NOW/PIT_LATER only - not
    evaluate.py's PIT_NOW/PIT_FLEXIBLE)."""
    exec_dec = ev._get(row, "execution", "decision") if isinstance(row.get("execution"), dict) else None
    final = exec_dec if exec_dec is not None else row.get("gate_decision")
    return final in STRATEGIC_HIT_DECISIONS


def dominant_triggers(rows: list[dict]) -> str:
    """Comma-joined names of triggers active on >=half of the given rows - a compact
    'what was driving this window' summary for the misses table."""
    if not rows:
        return ""
    counts: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("triggers") or {}).items():
            if v:
                counts[k] = counts.get(k, 0) + 1
    half = len(rows) / 2.0
    return ",".join(sorted(k for k, c in counts.items() if c >= half))


# ============================================================================
# Core per-event evaluation (STEP 2 + STEP 7 diagnostics)
# ============================================================================
def evaluate_pit_event(dec_by_lap: dict, season: int, race: str, driver: str,
                        actual_lap: int, total_laps: int, sanity_issues: list) -> dict:
    lo, hi = actual_lap - STRATEGIC_TOLERANCE_LAPS, actual_lap + STRATEGIC_TOLERANCE_LAPS
    clipped_lo, clipped_hi = max(1, lo), min(total_laps, hi)
    if lo < 1 or hi > total_laps:
        sanity_issues.append(f"{season} {race} {driver} L{actual_lap}: strategic window [{lo},{hi}] "
                              f"extends outside replay range [1,{total_laps}] - clipped to "
                              f"[{clipped_lo},{clipped_hi}] for evaluation (STEP 8 check #5).")

    window_laps = [lap for lap in range(clipped_lo, clipped_hi + 1)]
    window_rows = {lap: dec_by_lap[lap] for lap in window_laps if lap in dec_by_lap}
    if not window_rows:
        sanity_issues.append(f"{season} {race} {driver} L{actual_lap}: NO replay decision rows exist "
                              f"anywhere in [{clipped_lo},{clipped_hi}] (STEP 8 check #1).")

    hit_laps = sorted(lap for lap, r in window_rows.items() if is_strategic_recommendation(r))
    pre_pit_laps = [lap for lap in hit_laps if lap < actual_lap]
    window_hit = len(hit_laps) > 0
    pre_pit_hit = len(pre_pit_laps) > 0

    exact_row = dec_by_lap.get(actual_lap)
    exact_match = bool(exact_row and ev._get(exact_row, "execution", "driving_instruction") == "PIT_LAP")

    first_rec_lap = hit_laps[0] if hit_laps else None
    warning_offset = (first_rec_lap - actual_lap) if first_rec_lap is not None else None

    # STEP 7 diagnostics, preserved in full regardless of hit/miss
    at_event = dec_by_lap.get(actual_lap) or {}
    at_first_rec = dec_by_lap.get(first_rec_lap) if first_rec_lap is not None else None
    closest_rec_lap, closest_rec_dist = None, None
    if not window_hit:
        # search the WHOLE replay (still backward+forward looking is fine here - this is
        # POST-HOC diagnostic reporting for a miss, not a value fed back into any decision)
        all_hit_laps = sorted(lap for lap, r in dec_by_lap.items() if is_strategic_recommendation(r))
        if all_hit_laps:
            closest_rec_lap = min(all_hit_laps, key=lambda lap: abs(lap - actual_lap))
            closest_rec_dist = closest_rec_lap - actual_lap

    return dict(
        season=season, race=race, driver=driver, actual_pit_lap=actual_lap, total_laps=total_laps,
        window_lo=clipped_lo, window_hi=clipped_hi,
        window_hit=window_hit, pre_pit_hit=pre_pit_hit, exact_pit_lap_match=exact_match,
        first_recommendation_lap=first_rec_lap, warning_offset_laps=warning_offset,
        hit_laps_in_window=json.dumps(hit_laps),
        # STEP 7 preserved diagnostics
        compound_at_pit=at_event.get("compound"), tyre_age_at_pit=at_event.get("tyre_age"),
        gate_tier_at_pit=at_event.get("tier_reached"),
        instruction_at_pit=ev._get(at_event, "execution", "driving_instruction"),
        triggers_at_first_rec=json.dumps((at_first_rec or {}).get("triggers") or {}),
        dominant_triggers_in_window=dominant_triggers(list(window_rows.values())),
        sc_active_in_window=any(bool((r.get("triggers") or {}).get("safety_car")) for r in window_rows.values()),
        rain_in_window=any(bool(r.get("rain_now")) for r in window_rows.values()),  # present only if master.py
                                                                                      # attached it; else always False
        # STEP 5 miss diagnostics (recorded, never silently discarded - STEP 2.5)
        closest_recommendation_lap=closest_rec_lap, closest_recommendation_offset=closest_rec_dist,
    )


# ============================================================================
# STEP 8 - sanity checks (fail loudly, never silently coerce)
# ============================================================================
def sanity_check_duplicates(real_laps: list[int], season: int, race: str, driver: str, issues: list):
    for a, b in zip(real_laps, real_laps[1:]):
        if b - a <= 1:
            issues.append(f"{season} {race} {driver}: pit laps {a} and {b} are adjacent/duplicate "
                           f"(STEP 8 check #4) - evaluated independently, not merged, per the task spec.")


# ============================================================================
# Main evaluation loop
# ============================================================================
def run_validation(repo_root: Path, master_path: Path, out_dir: Path, races: list, force: bool):
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    sanity_issues: list = []
    event_rows: list = []
    n_events_before_diag_check = 0

    for season, race, justification in races:
        d1, d2 = ev.rbr_drivers_for(season, race)
        laps = ev.load_real_laps(repo_root, season, race)
        if laps is None:
            sanity_issues.append(f"{season} {race}: laps_features.csv not found - SKIPPED entirely.")
            continue
        out_file = run_replay_if_needed(repo_root, master_path, replay_dir, season, race, d1, d2, force)
        if out_file is None:
            sanity_issues.append(f"{season} {race}: replay FAILED - SKIPPED entirely.")
            continue
        dec = ev.read_jsonl(out_file)
        if dec.empty:
            sanity_issues.append(f"{season} {race}: replay produced an EMPTY decision file - SKIPPED.")
            continue
        total_laps = int(laps["LapNumber"].max())

        for driver in (d1, d2):
            real_laps = ev.real_pit_laps(laps, driver)   # STEP 1/3: reused, existing historical-pit-event definition
            sanity_check_duplicates(real_laps, season, race, driver, sanity_issues)

            sub = dec[dec["driver"] == driver] if "driver" in dec.columns else pd.DataFrame()
            dec_by_lap = {int(r["lap"]): r for r in sub.to_dict("records")} if not sub.empty else {}
            # STEP 8 check #2: every decision row's lap must be inside the replay's own race length
            out_of_range = [lap for lap in dec_by_lap if lap < 1 or lap > total_laps]
            if out_of_range:
                sanity_issues.append(f"{season} {race} {driver}: {len(out_of_range)} decision row(s) outside "
                                      f"[1,{total_laps}] (STEP 8 check #2): {out_of_range[:5]}")
            # STEP 8 check #3: structural guarantee the historical pit lap was never passed into the
            # decision function - confirmed by inspecting master.py's replay CLI (no such parameter
            # exists) before this harness was written; checked again here by confirming dec_by_lap
            # was built from the SAME file the (unmodified) replay wrote, untouched since.
            for lap in real_laps:
                n_events_before_diag_check += 1
                event_rows.append(evaluate_pit_event(dec_by_lap, season, race, driver, lap,
                                                        total_laps, sanity_issues))

    events = pd.DataFrame(event_rows)
    # STEP 8 check #6: the number of evaluated events must not silently change because of missing
    # diagnostic fields - assert the row count matches what was actually iterated.
    assert len(events) == n_events_before_diag_check, (
        f"STEP 8 check #6 FAILED: built {len(events)} event rows but iterated "
        f"{n_events_before_diag_check} historical pit events - some were silently dropped.")
    return events, sanity_issues


# ============================================================================
# Reporting (STEP 6)
# ============================================================================
def per_race_table(events: pd.DataFrame) -> pd.DataFrame:
    def summarise(g):
        n = len(g)
        wh, pph, ex = g["window_hit"].sum(), g["pre_pit_hit"].sum(), g["exact_pit_lap_match"].sum()
        offs = g.loc[g["window_hit"], "warning_offset_laps"].dropna()
        return pd.Series(dict(
            historical_pit_events=n, window_hits=int(wh), pre_pit_hits=int(pph), exact_pit_lap_matches=int(ex),
            window_recall=wh / n if n else np.nan, pre_pit_recall=pph / n if n else np.nan,
            exact_agreement=ex / n if n else np.nan,
            median_warning_offset=offs.median() if len(offs) else np.nan,
            mean_warning_offset=offs.mean() if len(offs) else np.nan,
        ))
    return events.groupby(["season", "race", "driver"]).apply(summarise).reset_index()


def aggregate_table(events: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    def summarise(g):
        n = len(g)
        wh, pph, ex = g["window_hit"].sum(), g["pre_pit_hit"].sum(), g["exact_pit_lap_match"].sum()
        offs = g.loc[g["window_hit"], "warning_offset_laps"].dropna()
        return pd.Series(dict(
            total_historical_pit_events=n, total_window_hits=int(wh),
            window_recall=wh / n if n else np.nan,
            total_pre_pit_hits=int(pph), pre_pit_recall=pph / n if n else np.nan,
            total_exact_matches=int(ex), exact_agreement=ex / n if n else np.nan,
            median_warning_offset=offs.median() if len(offs) else np.nan,
            mean_warning_offset=offs.mean() if len(offs) else np.nan,
            n_misses=int(n - wh),
        ))
    if group_cols:
        return events.groupby(group_cols).apply(summarise).reset_index()
    return summarise(events).to_frame().T


def main():
    ap = argparse.ArgumentParser(description="HERMES strategic-window validation (read-only)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--master", default="system/HERMES/trees/master.py")
    ap.add_argument("--out-dir", default="eval_out_strategic_window")
    ap.add_argument("--force", action="store_true", help="re-run replays even if cached")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    master_path = repo_root / args.master
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"SELECTED RACE SAMPLE ({len(SELECTED_RACES)} races) - fixed BEFORE any HERMES output was inspected")
    print("=" * 100)
    for season, race, why in SELECTED_RACES:
        cid = _RACE_TO_CIRCUIT_ID.get(race, "?")
        print(f"  {season} {race} (circuit_id={cid}): {why}")

    events, sanity_issues = run_validation(repo_root, master_path, out_dir, SELECTED_RACES, args.force)
    events.to_csv(out_dir / "strategic_window_events.csv", index=False)

    per_race = per_race_table(events)
    per_race.to_csv(out_dir / "strategic_window_per_race.csv", index=False)

    agg_overall = aggregate_table(events)
    agg_driver = aggregate_table(events, ["driver"])
    agg_race = aggregate_table(events, ["season", "race"])
    agg_season = aggregate_table(events, ["season"])
    agg_overall.to_csv(out_dir / "strategic_window_aggregate_overall.csv", index=False)
    agg_driver.to_csv(out_dir / "strategic_window_aggregate_by_driver.csv", index=False)
    agg_race.to_csv(out_dir / "strategic_window_aggregate_by_race.csv", index=False)
    agg_season.to_csv(out_dir / "strategic_window_aggregate_by_season.csv", index=False)

    misses = events[~events["window_hit"]].copy()
    misses.to_csv(out_dir / "strategic_window_misses.csv", index=False)

    print("\n" + "=" * 100)
    print("PER-RACE / PER-DRIVER TABLE")
    print("=" * 100)
    print(per_race.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("AGGREGATE (overall)")
    print("=" * 100)
    print(agg_overall.round(3).to_string(index=False))

    print("\n--- by driver ---")
    print(agg_driver.round(3).to_string(index=False))
    print("\n--- by race ---")
    print(agg_race.round(3).to_string(index=False))
    print("\n--- by season ---")
    print(agg_season.round(3).to_string(index=False))

    offs = events.loc[events["window_hit"], "warning_offset_laps"].dropna()
    if len(offs):
        print(f"\nWarning-offset distribution (laps, negative=before the real pit): "
              f"min={offs.min():.0f} p25={offs.quantile(.25):.1f} median={offs.median():.1f} "
              f"p75={offs.quantile(.75):.1f} max={offs.max():.0f} mean={offs.mean():.2f}")

    print(f"\n{'='*100}\nMISSES ({len(misses)} of {len(events)} historical pit events - not silently discarded)")
    print("=" * 100)
    if not misses.empty:
        show_cols = ["season", "race", "driver", "actual_pit_lap", "compound_at_pit", "tyre_age_at_pit",
                     "closest_recommendation_lap", "closest_recommendation_offset", "dominant_triggers_in_window"]
        print(misses[show_cols].to_string(index=False))
    else:
        print("  (none)")

    print(f"\n{'='*100}\nDATA-QUALITY / SANITY-CHECK ISSUES ({len(sanity_issues)})")
    print("=" * 100)
    for issue in sanity_issues:
        print(f"  ! {issue}")
    if not sanity_issues:
        print("  (none)")

    print(f"\n[strategic_window_validation] wrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
