
""
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
import evaluate as ev

STRATEGIC_TOLERANCE_LAPS = 3

STRATEGIC_HIT_DECISIONS = {"PIT_NOW", "PIT_LATER"}

USEFUL_WARNING_WINDOW = (1, 5)

ESCALATION_PRE_WINDOW = (1, 3)
TRIGGER_COUNT_LAPS_BACK = (5, 4, 3, 2, 1)
SHORT_REPLAY_MAX_LAPS = 10

BASELINE_SEED = 20260930

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

def run_replay_if_needed(repo_root: Path, master_path: Path, out_dir: Path,
                          season: int, race: str, d1: str, d2: str, force: bool) -> Path:
    out_file = out_dir / f"{season}_{race}.jsonl"
    if out_file.exists() and not force:
        return out_file
    cmd = [sys.executable, str(master_path), "replay", "--repo-root", str(repo_root),
           "--season", str(season), "--race", race, "--session", "R",
           "--d1", d1, "--d2", d2, "--out", str(out_file)]

    print(f"  [replay] {season} {race} {d1}/{d2}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if res.returncode != 0 or not out_file.exists():
        print(f"    FAILED: {(res.stderr or '')[-400:]}")
        return None
    return out_file

def is_strategic_recommendation(row: dict) -> bool:
    ""
    exec_dec = ev._get(row, "execution", "decision") if isinstance(row.get("execution"), dict) else None
    final = exec_dec if exec_dec is not None else row.get("gate_decision")
    return final in STRATEGIC_HIT_DECISIONS

def dominant_triggers(rows: list[dict]) -> str:
    ""
    if not rows:
        return ""
    counts: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("triggers") or {}).items():
            if v:
                counts[k] = counts.get(k, 0) + 1
    half = len(rows) / 2.0
    return ",".join(sorted(k for k, c in counts.items() if c >= half))

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

    at_event = dec_by_lap.get(actual_lap) or {}
    at_first_rec = dec_by_lap.get(first_rec_lap) if first_rec_lap is not None else None
    closest_rec_lap, closest_rec_dist = None, None
    if not window_hit:

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

        compound_at_pit=at_event.get("compound"), tyre_age_at_pit=at_event.get("tyre_age"),
        gate_tier_at_pit=at_event.get("tier_reached"),
        instruction_at_pit=ev._get(at_event, "execution", "driving_instruction"),
        triggers_at_first_rec=json.dumps((at_first_rec or {}).get("triggers") or {}),
        dominant_triggers_in_window=dominant_triggers(list(window_rows.values())),
        sc_active_in_window=any(bool((r.get("triggers") or {}).get("safety_car")) for r in window_rows.values()),
        rain_in_window=any(bool(r.get("rain_now")) for r in window_rows.values()),

        closest_recommendation_lap=closest_rec_lap, closest_recommendation_offset=closest_rec_dist,
    )

def sanity_check_duplicates(real_laps: list[int], season: int, race: str, driver: str, issues: list):
    for a, b in zip(real_laps, real_laps[1:]):
        if b - a <= 1:
            issues.append(f"{season} {race} {driver}: pit laps {a} and {b} are adjacent/duplicate "
                           f"(STEP 8 check #4) - evaluated independently, not merged, per the task spec.")

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
            real_laps = ev.real_pit_laps(laps, driver)
            sanity_check_duplicates(real_laps, season, race, driver, sanity_issues)

            sub = dec[dec["driver"] == driver] if "driver" in dec.columns else pd.DataFrame()
            dec_by_lap = {int(r["lap"]): r for r in sub.to_dict("records")} if not sub.empty else {}

            out_of_range = [lap for lap in dec_by_lap if lap < 1 or lap > total_laps]
            if out_of_range:
                sanity_issues.append(f"{season} {race} {driver}: {len(out_of_range)} decision row(s) outside "
                                      f"[1,{total_laps}] (STEP 8 check #2): {out_of_range[:5]}")

            for lap in real_laps:
                n_events_before_diag_check += 1
                event_rows.append(evaluate_pit_event(dec_by_lap, season, race, driver, lap,
                                                        total_laps, sanity_issues))

    events = pd.DataFrame(event_rows)

    assert len(events) == n_events_before_diag_check, (
        f"STEP 8 check #6 FAILED: built {len(events)} event rows but iterated "
        f"{n_events_before_diag_check} historical pit events - some were silently dropped.")
    return events, sanity_issues

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
