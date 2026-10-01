
""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
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

def final_decision(row: dict):
    ""
    exec_dec = ev._get(row, "execution", "decision") if isinstance(row.get("execution"), dict) else None
    return exec_dec if exec_dec is not None else row.get("gate_decision")

def is_strategic_recommendation(row: dict) -> bool:
    ""
    return final_decision(row) in STRATEGIC_HIT_DECISIONS

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

def count_active_triggers(row: dict) -> int:
    ""
    return sum(1 for v in (row.get("triggers") or {}).values() if v)

def window_metrics(hit_lookup: dict, actual_lap: int, total_laps: int,
                    tolerance: int = STRATEGIC_TOLERANCE_LAPS) -> dict:
    ""
    lo, hi = actual_lap - tolerance, actual_lap + tolerance
    clipped_lo, clipped_hi = max(1, lo), min(total_laps, hi)
    window_laps = [l for l in range(clipped_lo, clipped_hi + 1) if l in hit_lookup]
    hit_laps = sorted(l for l in window_laps if hit_lookup[l])
    pre_pit_laps = [l for l in hit_laps if l < actual_lap]
    return dict(window_lo=clipped_lo, window_hi=clipped_hi,
                window_hit=len(hit_laps) > 0, pre_pit_hit=len(pre_pit_laps) > 0,
                hit_laps=hit_laps, clipped=(lo < 1 or hi > total_laps))

def stint_start_lap(real_laps: list[int], actual_lap: int, available_laps: list[int]) -> int:
    ""
    prior_pits = [l for l in real_laps if l < actual_lap]
    if prior_pits:
        return prior_pits[-1] + 1
    return min(available_laps) if available_laps else 1

def onset_and_lead(hit_lookup: dict, stint_start: int, actual_lap: int) -> dict:
    ""
    pre_laps = [l for l in range(stint_start, actual_lap) if l in hit_lookup]
    onset_lap = next((l for l in pre_laps if hit_lookup[l]), None)
    warning_lead = (actual_lap - onset_lap) if onset_lap is not None else None
    useful = bool(onset_lap is not None and
                  USEFUL_WARNING_WINDOW[0] <= warning_lead <= USEFUL_WARNING_WINDOW[1])
    return dict(recommendation_onset_lap=onset_lap, warning_lead_laps=warning_lead,
                useful_warning=useful, pre_pit_laps_available=len(pre_laps))

def escalation_info(dec_by_lap: dict, stint_start: int, actual_lap: int) -> dict:
    ""
    pre_now_laps = sorted(l for l in range(stint_start, actual_lap)
                           if l in dec_by_lap and final_decision(dec_by_lap[l]) == "PIT_NOW")
    first_pit_now_lap = pre_now_laps[0] if pre_now_laps else None
    if first_pit_now_lap is None:
        bucket = "PIT_NOW_NEVER_BEFORE_PIT"
    else:
        dist = actual_lap - first_pit_now_lap
        bucket = ("PIT_NOW_1_3_LAPS_BEFORE" if ESCALATION_PRE_WINDOW[0] <= dist <= ESCALATION_PRE_WINDOW[1]
                  else "PIT_NOW_MORE_THAN_3_LAPS_BEFORE")

    last_pre_lap = actual_lap - 1
    last_pre_row = dec_by_lap.get(last_pre_lap)
    last_pre_pit_state = final_decision(last_pre_row) if last_pre_row is not None else None
    last_pre_pit_triggers = json.dumps({k: v for k, v in (last_pre_row.get("triggers") or {}).items() if v}) \
        if last_pre_row is not None else None

    trigger_counts = {}
    for k in TRIGGER_COUNT_LAPS_BACK:
        lap = actual_lap - k
        row = dec_by_lap.get(lap)
        trigger_counts[k] = count_active_triggers(row) if row is not None else None

    known = [(k, v) for k, v in trigger_counts.items() if v is not None]
    delta_trigger_count = None
    if TRIGGER_COUNT_LAPS_BACK[0] in trigger_counts and TRIGGER_COUNT_LAPS_BACK[-1] in trigger_counts \
            and trigger_counts[TRIGGER_COUNT_LAPS_BACK[0]] is not None \
            and trigger_counts[TRIGGER_COUNT_LAPS_BACK[-1]] is not None:
        delta_trigger_count = trigger_counts[TRIGGER_COUNT_LAPS_BACK[-1]] - trigger_counts[TRIGGER_COUNT_LAPS_BACK[0]]

    return dict(
        first_pit_now_lap=first_pit_now_lap,
        pit_now_before_pit=first_pit_now_lap is not None,
        escalation_bucket=bucket,
        last_pre_pit_state=last_pre_pit_state,
        last_pre_pit_triggers=last_pre_pit_triggers,
        trigger_count_L5=trigger_counts[5], trigger_count_L4=trigger_counts[4],
        trigger_count_L3=trigger_counts[3], trigger_count_L2=trigger_counts[2],
        trigger_count_L1=trigger_counts[1],
        trigger_count_delta=delta_trigger_count,
        trigger_count_laps_with_data=len(known),
    )

def evaluate_pit_event(dec_by_lap: dict, real_laps: list, season: int, race: str, driver: str,
                        actual_lap: int, total_laps: int, sanity_issues: list) -> dict:
    hit_lookup = {lap: is_strategic_recommendation(r) for lap, r in dec_by_lap.items()}
    wm = window_metrics(hit_lookup, actual_lap, total_laps)
    clipped_lo, clipped_hi = wm["window_lo"], wm["window_hi"]
    if wm["clipped"]:
        sanity_issues.append(f"{season} {race} {driver} L{actual_lap}: strategic window "
                              f"[{actual_lap - STRATEGIC_TOLERANCE_LAPS},{actual_lap + STRATEGIC_TOLERANCE_LAPS}] "
                              f"extends outside replay range [1,{total_laps}] - clipped to "
                              f"[{clipped_lo},{clipped_hi}] for evaluation (STEP 8 check #5).")

    window_rows = {lap: dec_by_lap[lap] for lap in range(clipped_lo, clipped_hi + 1) if lap in dec_by_lap}
    if not window_rows:
        sanity_issues.append(f"{season} {race} {driver} L{actual_lap}: NO replay decision rows exist "
                              f"anywhere in [{clipped_lo},{clipped_hi}] (STEP 8 check #1).")

    hit_laps = wm["hit_laps"]
    window_hit, pre_pit_hit = wm["window_hit"], wm["pre_pit_hit"]

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

    available_laps = sorted(dec_by_lap.keys())
    stint_start = stint_start_lap(real_laps, actual_lap, available_laps)
    onset = onset_and_lead(hit_lookup, stint_start, actual_lap)
    esc = escalation_info(dec_by_lap, stint_start, actual_lap)
    short_replay = total_laps < SHORT_REPLAY_MAX_LAPS
    if short_replay:
        sanity_issues.append(f"{season} {race} {driver} L{actual_lap}: total_laps={total_laps} < "
                              f"SHORT_REPLAY_MAX_LAPS={SHORT_REPLAY_MAX_LAPS} - flagged short_replay=True, "
                              f"included in event set but reportable separately (STAGE 2 point 8).")

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

        stint_start_lap=stint_start, short_replay=short_replay, **onset, **esc,
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
    baseline_rows: list = []
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
                event_rows.append(evaluate_pit_event(dec_by_lap, real_laps, season, race, driver, lap,
                                                        total_laps, sanity_issues))

            if dec_by_lap:
                real_hit_rate = float(np.mean([is_strategic_recommendation(r) for r in dec_by_lap.values()]))
                seed = (BASELINE_SEED + zlib.crc32(f"{season}|{race}|{driver}".encode())) % (2**32)
                rng = np.random.default_rng(seed)
                available_laps = sorted(dec_by_lap.keys())
                baseline_hit_lookup = {lap: bool(rng.random() < real_hit_rate) for lap in available_laps}
                for lap in real_laps:
                    wm = window_metrics(baseline_hit_lookup, lap, total_laps)
                    stint_start = stint_start_lap(real_laps, lap, available_laps)
                    onset = onset_and_lead(baseline_hit_lookup, stint_start, lap)
                    baseline_rows.append(dict(
                        season=season, race=race, driver=driver, actual_pit_lap=lap, total_laps=total_laps,
                        baseline_hit_rate=real_hit_rate, window_hit=wm["window_hit"],
                        pre_pit_hit=wm["pre_pit_hit"], short_replay=total_laps < SHORT_REPLAY_MAX_LAPS,
                        **onset,
                    ))

    events = pd.DataFrame(event_rows)
    baseline_events = pd.DataFrame(baseline_rows)

    assert len(events) == n_events_before_diag_check, (
        f"STEP 8 check #6 FAILED: built {len(events)} event rows but iterated "
        f"{n_events_before_diag_check} historical pit events - some were silently dropped.")
    return events, baseline_events, sanity_issues

def _coverage_metrics(g: pd.DataFrame) -> dict:
    ""
    n = len(g)
    wh, pph, ex = g["window_hit"].sum(), g["pre_pit_hit"].sum(), g["exact_pit_lap_match"].sum()
    offs = g.loc[g["window_hit"], "warning_offset_laps"].dropna()
    return dict(
        coverage_window_hits=int(wh), coverage_window_recall=wh / n if n else np.nan,
        coverage_pre_pit_hits=int(pph), coverage_pre_pit_recall=pph / n if n else np.nan,
        coverage_exact_matches=int(ex), coverage_exact_agreement=ex / n if n else np.nan,
        coverage_median_warning_offset=offs.median() if len(offs) else np.nan,
        coverage_mean_warning_offset=offs.mean() if len(offs) else np.nan,
    )

def _timing_metrics(g: pd.DataFrame) -> dict:
    ""
    n = len(g)
    lead = g["warning_lead_laps"].dropna()
    n_onset = int(g["recommendation_onset_lap"].notna().sum())
    useful_recall = g["useful_warning"].sum() / n if n else np.nan

    esc_counts = g["escalation_bucket"].value_counts()
    esc_total = len(g)
    esc = {f"escalation_pct_{k}": (esc_counts.get(k, 0) / esc_total if esc_total else np.nan)
           for k in ("PIT_NOW_1_3_LAPS_BEFORE", "PIT_NOW_MORE_THAN_3_LAPS_BEFORE", "PIT_NOW_NEVER_BEFORE_PIT")}
    pit_now_before_rate = g["pit_now_before_pit"].sum() / n if n else np.nan

    state_counts = g["last_pre_pit_state"].fillna("NO_DATA").value_counts()
    state_total = len(g)
    states = {f"last_pre_pit_state_pct_{k}": (state_counts.get(k, 0) / state_total if state_total else np.nan)
              for k in ("DONT_PIT", "PIT_LATER", "PIT_NOW", "NO_DATA")}

    delta = g["trigger_count_delta"].dropna()
    n_delta = len(delta)
    delta_dir = dict(
        trigger_delta_pct_positive=(delta > 0).sum() / n_delta if n_delta else np.nan,
        trigger_delta_pct_zero=(delta == 0).sum() / n_delta if n_delta else np.nan,
        trigger_delta_pct_negative=(delta < 0).sum() / n_delta if n_delta else np.nan,
        trigger_delta_n_with_data=n_delta,
    )

    return dict(
        n_with_onset=n_onset, n_without_onset=n - n_onset,
        median_warning_lead=lead.median() if len(lead) else np.nan,
        mean_warning_lead=lead.mean() if len(lead) else np.nan,
        p25_warning_lead=lead.quantile(.25) if len(lead) else np.nan,
        p75_warning_lead=lead.quantile(.75) if len(lead) else np.nan,
        min_warning_lead=lead.min() if len(lead) else np.nan,
        max_warning_lead=lead.max() if len(lead) else np.nan,
        useful_warning_recall=useful_recall,
        pit_now_before_pit_rate=pit_now_before_rate,
        **esc, **states, **delta_dir,
    )

def per_race_table(events: pd.DataFrame) -> pd.DataFrame:
    def summarise(g):
        n = len(g)
        d = dict(historical_pit_events=n)
        d.update(_coverage_metrics(g))
        d.update(_timing_metrics(g))
        return pd.Series(d)
    return events.groupby(["season", "race", "driver"]).apply(summarise).reset_index()

def aggregate_table(events: pd.DataFrame, group_cols=None, exclude_short_replay: bool = False) -> pd.DataFrame:
    ""
    ev_use = events[~events["short_replay"]] if exclude_short_replay else events

    def summarise(g):
        n = len(g)
        d = dict(total_historical_pit_events=n, n_misses=int(n - g["window_hit"].sum()))
        d.update(_coverage_metrics(g))
        d.update(_timing_metrics(g))
        return pd.Series(d)

    if group_cols:
        if ev_use.empty:
            return pd.DataFrame(columns=group_cols)
        return ev_use.groupby(group_cols).apply(summarise).reset_index()
    if ev_use.empty:
        return pd.DataFrame()
    return summarise(ev_use).to_frame().T

def baseline_aggregate_table(baseline_events: pd.DataFrame, exclude_short_replay: bool = False) -> pd.DataFrame:
    ""
    b = baseline_events[~baseline_events["short_replay"]] if exclude_short_replay else baseline_events
    if b.empty:
        return pd.DataFrame()
    n = len(b)
    lead = b["warning_lead_laps"].dropna()
    return pd.DataFrame([dict(
        total_historical_pit_events=n,
        mean_baseline_hit_rate=b["baseline_hit_rate"].mean(),
        baseline_window_recall=b["window_hit"].sum() / n if n else np.nan,
        baseline_pre_pit_recall=b["pre_pit_hit"].sum() / n if n else np.nan,
        baseline_useful_warning_recall=b["useful_warning"].sum() / n if n else np.nan,
        baseline_median_warning_lead=lead.median() if len(lead) else np.nan,
    )])

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

    events, baseline_events, sanity_issues = run_validation(
        repo_root, master_path, out_dir, SELECTED_RACES, args.force)
    events.to_csv(out_dir / "strategic_window_events.csv", index=False)
    baseline_events.to_csv(out_dir / "strategic_window_baseline_events.csv", index=False)

    per_race = per_race_table(events)
    per_race.to_csv(out_dir / "strategic_window_per_race.csv", index=False)

    agg_overall = aggregate_table(events)
    agg_driver = aggregate_table(events, ["driver"])
    agg_race = aggregate_table(events, ["season", "race"])
    agg_season = aggregate_table(events, ["season"])
    agg_overall_excl = aggregate_table(events, exclude_short_replay=True)
    agg_overall.to_csv(out_dir / "strategic_window_aggregate_overall.csv", index=False)
    agg_driver.to_csv(out_dir / "strategic_window_aggregate_by_driver.csv", index=False)
    agg_race.to_csv(out_dir / "strategic_window_aggregate_by_race.csv", index=False)
    agg_season.to_csv(out_dir / "strategic_window_aggregate_by_season.csv", index=False)
    agg_overall_excl.to_csv(out_dir / "strategic_window_aggregate_overall_excl_short_replay.csv", index=False)

    baseline_overall = baseline_aggregate_table(baseline_events)
    baseline_overall_excl = baseline_aggregate_table(baseline_events, exclude_short_replay=True)
    baseline_overall.to_csv(out_dir / "strategic_window_baseline_aggregate_overall.csv", index=False)
    baseline_overall_excl.to_csv(out_dir / "strategic_window_baseline_aggregate_overall_excl_short_replay.csv",
                                  index=False)

    misses = events[~events["window_hit"]].copy()
    misses.to_csv(out_dir / "strategic_window_misses.csv", index=False)

    diag_cols = dict(
        season="season", race="race", driver="driver", actual_pit_lap="actual_pit_lap",
        recommendation_onset_lap="recommendation_onset_lap", warning_lead_laps="warning_lead",
        useful_warning="useful_warning", first_pit_now_lap="first_pit_now_lap",
        pit_now_before_pit="pit_now_before_pit", escalation_bucket="escalation_bucket",
        last_pre_pit_state="last_pre_pit_state", last_pre_pit_triggers="last_pre_pit_triggers",
        trigger_count_L5="trigger_count_L5", trigger_count_L4="trigger_count_L4",
        trigger_count_L3="trigger_count_L3", trigger_count_L2="trigger_count_L2",
        trigger_count_L1="trigger_count_L1", trigger_count_delta="trigger_count_delta",
        pre_pit_laps_available="pre_pit_laps_available", short_replay="short_replay",
        dominant_triggers_in_window="dominant_triggers_in_window",
    )
    per_event_diag = events[list(diag_cols.keys())].rename(columns=diag_cols)
    per_event_diag.to_csv(out_dir / "strategic_window_per_event_diagnostics.csv", index=False)

    print("\n" + "=" * 100)
    print("PER-RACE / PER-DRIVER TABLE (coverage_* = STAGE 1 metrics, relabelled; rest = STAGE 2)")
    print("=" * 100)
    print(per_race.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("AGGREGATE (overall) - including all events")
    print("=" * 100)
    print(agg_overall.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("AGGREGATE (overall) - EXCLUDING short_replay events (total_laps < "
          f"{SHORT_REPLAY_MAX_LAPS})")
    print("=" * 100)
    if not agg_overall_excl.empty:
        print(agg_overall_excl.round(3).to_string(index=False))
    else:
        print("  (empty - no events survive short_replay exclusion)")

    print("\n--- by driver ---")
    print(agg_driver.round(3).to_string(index=False))
    print("\n--- by race ---")
    print(agg_race.round(3).to_string(index=False))
    print("\n--- by season ---")
    print(agg_season.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("SATURATION-AWARE BASELINE (same driver-race prevalence as HERMES, no trigger information)")
    print("=" * 100)
    print("Including all events:")
    print(baseline_overall.round(3).to_string(index=False) if not baseline_overall.empty else "  (empty)")
    print("Excluding short_replay events:")
    print(baseline_overall_excl.round(3).to_string(index=False) if not baseline_overall_excl.empty else "  (empty)")

    offs = events.loc[events["window_hit"], "warning_offset_laps"].dropna()
    if len(offs):
        print(f"\nCoverage warning-offset distribution (laps, negative=before the real pit): "
              f"min={offs.min():.0f} p25={offs.quantile(.25):.1f} median={offs.median():.1f} "
              f"p75={offs.quantile(.75):.1f} max={offs.max():.0f} mean={offs.mean():.2f}")
    lead = events["warning_lead_laps"].dropna()
    if len(lead):
        print(f"Recommendation-onset warning-lead distribution (laps before the real pit, stint-bounded): "
              f"min={lead.min():.0f} p25={lead.quantile(.25):.1f} median={lead.median():.1f} "
              f"p75={lead.quantile(.75):.1f} max={lead.max():.0f} mean={lead.mean():.2f} "
              f"(n_without_onset={int(events['recommendation_onset_lap'].isna().sum())} of {len(events)})")

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

    n_events = len(events)
    cov_recall = agg_overall["coverage_window_recall"].iloc[0] if n_events else np.nan
    base_recall = baseline_overall["baseline_window_recall"].iloc[0] if not baseline_overall.empty else np.nan
    mean_base_rate = (baseline_events.groupby(["season", "race", "driver"])["baseline_hit_rate"]
                       .first().mean()) if not baseline_events.empty else np.nan
    useful_recall = agg_overall["useful_warning_recall"].iloc[0] if n_events else np.nan
    base_useful_recall = (baseline_overall["baseline_useful_warning_recall"].iloc[0]
                           if not baseline_overall.empty else np.nan)
    median_lead = agg_overall["median_warning_lead"].iloc[0] if n_events else np.nan
    esc_1_3 = agg_overall.get("escalation_pct_PIT_NOW_1_3_LAPS_BEFORE", pd.Series([np.nan])).iloc[0]
    esc_earlier = agg_overall.get("escalation_pct_PIT_NOW_MORE_THAN_3_LAPS_BEFORE", pd.Series([np.nan])).iloc[0]
    esc_never = agg_overall.get("escalation_pct_PIT_NOW_NEVER_BEFORE_PIT", pd.Series([np.nan])).iloc[0]
    n_short = int(events["short_replay"].sum())

    print(f"\n{'='*100}\nFINAL REPORT - 5 QUESTIONS (STAGE 2)")
    print("=" * 100)
    print(f"1. Is the original +/-{STRATEGIC_TOLERANCE_LAPS}-lap window metric saturated on this sample?\n"
          f"   Mean per-driver-race base rate of PIT_NOW/PIT_LATER across ALL decision laps = "
          f"{mean_base_rate:.3f} ({n_events} historical pit events). Coverage window recall = "
          f"{cov_recall:.3f}. A same-prevalence baseline with no trigger information reaches a "
          f"window recall of {base_recall:.3f} on the same events.")
    print(f"2. Does the onset-timing measurement (recommendation_onset_lap, stint-bounded) differ "
          f"from the same-prevalence baseline?\n"
          f"   HERMES useful-warning recall (onset within [L-{USEFUL_WARNING_WINDOW[1]},"
          f"L-{USEFUL_WARNING_WINDOW[0]}]) = {useful_recall:.3f}, median warning lead = "
          f"{median_lead if pd.isna(median_lead) else f'{median_lead:.1f}'} laps. Baseline useful-warning "
          f"recall on the same events = {base_useful_recall:.3f}.")
    print(f"3. Does the trigger/decision state escalate toward PIT_NOW in the laps preceding a "
          f"historical pit?\n"
          f"   Of {n_events} events: {esc_1_3:.3f} had PIT_NOW first appear 1-{ESCALATION_PRE_WINDOW[1]} "
          f"laps before the pit, {esc_earlier:.3f} had it appear more than {ESCALATION_PRE_WINDOW[1]} laps "
          f"before, {esc_never:.3f} had no PIT_NOW row before the pit within the stint-bounded search. "
          f"See escalation_pct_* columns and the trigger_count_delta distribution in the aggregate tables "
          f"for the count-level detail behind this split.")
    print(f"4. How much of the original window recall does the same-prevalence, no-trigger-information "
          f"baseline reproduce?\n"
          f"   Baseline window recall / coverage window recall = "
          f"{(base_recall / cov_recall) if cov_recall else np.nan:.3f} "
          f"({base_recall:.3f} of {cov_recall:.3f}).")
    print(f"5. Remaining evaluation-design issues in this sample:\n"
          f"   {len(sanity_issues)} sanity-check issue(s) logged above (adjacent/duplicate real pit laps, "
          f"decision rows outside the replay range, windows clipped at the replay boundary). "
          f"{n_short} of {n_events} events are flagged short_replay (total_laps < {SHORT_REPLAY_MAX_LAPS}) "
          f"and are reported both included and excluded above. The escalation and last-pre-pit-state "
          f"metrics only look backward from the historical pit lap within the current stint, by the "
          f"stint-bounding design decision documented in stint_start_lap() - a different bounding choice "
          f"(e.g. whole-race-so-far) was not measured here and would need to be run separately if wanted.")

    print(f"\n[strategic_window_validation] wrote CSVs to {out_dir}")

if __name__ == "__main__":
    main()
