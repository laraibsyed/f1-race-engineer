
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import itertools

import numpy as np
import pandas as pd

@dataclass
class SCGambleInputs:
    p_sc_next_n_laps: Optional[float]
    predicted_pace_loss_per_lap: Optional[float]
    cliff_probability_next_n_laps: Optional[float]
    n_laps_horizon: int

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

def evaluate_sc_gamble_v2(inputs: SCGambleInputs, *, normal_pit_loss_s: float = 22.0,
                           sc_pit_loss_s: float = 11.0, cold_tyre_penalty_s: float = 1.5,
                           cliff_penalty_scale: float = 5.0) -> dict:
    ""
    if inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None:
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None}

    p_sc = inputs.p_sc_next_n_laps
    n = inputs.n_laps_horizon
    pace = inputs.predicted_pace_loss_per_lap
    cliff = inputs.cliff_probability_next_n_laps or 0.0

    laps_if_sc = n / 2.0
    laps_if_no_sc = float(n)

    degradation_if_sc = pace * laps_if_sc
    degradation_if_no_sc = pace * laps_if_no_sc

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

def run_sweep():
    p_sc_values = np.linspace(0.0, 1.0, 11)
    cliff_values = np.linspace(0.0, 1.0, 11)

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

    print("\n" + "=" * 78)
    print("SWEEP 2: v2 WAIT-fraction under alternative pit-loss constants")
    print("=" * 78)
    for normal_pit, sc_pit in [(22.0, 11.0), (22.0, 16.0), (18.0, 11.0), (25.0, 20.0)]:
        n_wait, n_total = count_wait(evaluate_sc_gamble_v2, normal_pit_loss_s=normal_pit, sc_pit_loss_s=sc_pit)
        print(f"  normal_pit_loss_s={normal_pit:<5} sc_pit_loss_s={sc_pit:<5} "
              f"-> WAIT in {100*n_wait/n_total:.1f}% of cells")

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

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "."))
ARCHIVE_PER_RACE_PATH = REPO_ROOT / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv"
ARCHIVE_EVENTS_PATH = REPO_ROOT / "checkpoints" / "rival_knowledge" / "archive_event_summary_enriched.csv"
CACHE_DIR = Path(os.environ.get("GCS_CACHE_DIR", REPO_ROOT / "gcs_cache"))

MIN_EVENTS_FOR_CIRCUIT_SC_ESTIMATE = 5
MAX_PLAUSIBLE_PIT_STOP_S = 120.0

def compute_real_pit_losses_from_laps(cache_dir: Path) -> pd.DataFrame:
    ""
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
            continue
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
    ""
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
    ""
    if not path.exists():
        print(f"[calibration] {path} not found - normal_pit_loss_s will be "
              f"derived from archive_event_summary_enriched.csv's raw events instead "
              f"(step 1 falls through to the same source step 2 already needs).")
        return None
    df = pd.read_csv(path)
    print(f"[calibration] {path.name} columns: {list(df.columns)}")
    return df

def load_archive_events(path: Path):
    ""
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
    ""
    return str(race).replace(" ", "_")

def classify_events_by_caution(events_df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    ""
    caution_laps_cache: dict = {}
    full_sc_laps_cache: dict = {}
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
                caution_laps_cache[laps_path] = None
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
    ""
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
    ""
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
    ""
    print(f"[calibration] status value counts:\n{per_race_df['status'].value_counts().to_string()}")
    print(f"[calibration] known_backfilled_limitation value counts:\n"
          f"{per_race_df['known_backfilled_limitation'].value_counts(dropna=False).to_string()}")
    reliable = per_race_df.copy()
    uniq = set(reliable["known_backfilled_limitation"].dropna().unique())
    if uniq <= {True, False}:
        before = len(reliable)
        reliable = reliable[reliable["known_backfilled_limitation"] != True]
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
    ""
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

    print("\n" + "=" * 78)
    print("STEP 4: Sweep 1 + Sweep 2, re-run with REAL circuit constants")
    print("=" * 78)
    n_horizon = 5
    p_sc_real = np.linspace(0.0, 0.15, 8)
    cliff_real = np.linspace(0.0, 0.10, 6)
    pace_real = np.linspace(0.0, 0.30, 7)
    ASSUMED_SC_PIT_LOSS_S = 11.0

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
        ""
        cells = []
        for p_sc, cliff, pace in itertools.product(p_sc_real, cliff_real, pace_real):
            inputs = SCGambleInputs(p_sc, pace, cliff, n_horizon)
            result = evaluate_sc_gamble_v2(inputs, normal_pit_loss_s=normal_pit, sc_pit_loss_s=sc_pit)
            if result["recommendation"] == "WAIT":
                cells.append({"p_sc": p_sc, "cliff": cliff, "pace_loss": pace,
                              "saving": result["expected_saving_if_wait"]})
        return pd.DataFrame(cells)

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
            break

    if best_row is not None:
        race_name, normal, sc = best_row
        print(f"\n{'=' * 78}")
        print(f"WAIT cells for {race_name} (WAIT%={best_wait_pct:.1f}) - which exact "
              f"(p_sc, cliff, pace_loss) combinations actually produce WAIT:")
        print("=" * 78)
        cells = wait_cells(normal, sc)

        D = normal - sc - 1.5
        print(f"D = normal - caution - cold_tyre = {D:.3f} -> "
              f"expect WAIT across the {'full' if D > 0 else 'empty/near-empty'} p_sc range "
              f"at low cliff/pace_loss, {'shrinking' if D > 0 else 'never appearing'} as "
              f"cliff/pace_loss rise (unconditional check, holds regardless of D's sign).")
        if cells.empty:
            print("(no WAIT cells found for this circuit's real constants - 0.0% is exact, not rounding)")
        else:
            print(cells.to_string(index=False))

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
