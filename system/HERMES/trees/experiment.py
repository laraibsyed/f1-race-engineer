
""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", ".")).resolve()
RACES = [

    (2023, "Bahrain_Grand_Prix", "R", "VER", "PER"),

    (2022, "Monaco_Grand_Prix", "R", "VER", "PER"),

    (2023, "Australian_Grand_Prix", "R", "VER", "PER"),

    (2021, "Saudi_Arabian_Grand_Prix", "R", "VER", "PER"),

    (2024, "British_Grand_Prix", "R", "VER", "PER"),

]
FLOORS = [None, 4, 5, 6]

HERMES_MASTER_PATH = Path(os.environ.get("HERMES_MASTER_PATH", "system/HERMES/trees/master.py"))

def die(msg):
    print(msg)
    sys.exit(1)

if not HERMES_MASTER_PATH.exists():
    die(f"Could not find {HERMES_MASTER_PATH.resolve()} - set HERMES_MASTER_PATH "
        f"to wherever your orchestrator file actually lives (you named yours "
        f"system/HERMES/trees/master.py in earlier runs - adjust if it moved).")

spec = importlib.util.spec_from_file_location("hermes_master_under_test", HERMES_MASTER_PATH)
hm = importlib.util.module_from_spec(spec)
sys.modules["hermes_master_under_test"] = hm
spec.loader.exec_module(hm)

gate3 = hm.gate3
_real_evaluate_tier3 = gate3.evaluate_tier3

def make_floored_evaluate_tier3(floor_age):
    if floor_age is None:
        return _real_evaluate_tier3

    def _floored(state):
        if state.tyre_age is not None and state.tyre_age < floor_age:
            state.cliff_probability_next_5_laps = None
        return _real_evaluate_tier3(state)
    return _floored

def run_one(season, race, session, d1, d2, floor_age):
    gate3.evaluate_tier3 = make_floored_evaluate_tier3(floor_age)

    tyre_models = hm.load_tyre_models(REPO_ROOT / "tyre_life_models.pkl")
    sc_prior = hm.load_sc_prior(REPO_ROOT / "sc_vsc_circuit_level_prior.csv")
    cliff_stints = hm.load_cliff_stints(REPO_ROOT / "cliff_detection_stints.csv")
    taxonomy = hm.load_circuit_taxonomy(REPO_ROOT / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
    pit_loss_table = hm.load_pit_loss_table(
        REPO_ROOT / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv")

    degr_ordinal, _ = hm.circuit_degredation_ordinal_for(taxonomy, race)
    pit_loss_s, _ = hm.pit_loss_for_circuit(pit_loss_table, race)

    laps_path = hm.find_laps_features(season, race, session)
    if laps_path is None:
        print(f"    SKIP {season} {race} {session} - laps_features.csv not found under gcs_cache")
        return []
    total_laps = int(pd.read_csv(laps_path, usecols=["LapNumber"])["LapNumber"].max())

    ctx = hm.RaceContext(
        season=season, circuit=race, total_laps=total_laps, is_sprint_weekend=False,
        d1_code=d1, d2_code=d2, session=session, circuit_degredation_ordinal=degr_ordinal,
        pit_loss_s=pit_loss_s, p_sc_5lap=hm.get_sc_probability(sc_prior, race),
    )
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}

    decisions = hm.run_replay(ctx, resources, range(1, total_laps + 1), explain=False)
    for d in decisions:
        d["_race"] = f"{season}_{race}"
        d["_floor"] = floor_age
    return decisions

def summarize(decisions, floor_label, quiet=False):
    if not decisions:
        if not quiet:
            print(f"\n[{floor_label}] no decisions collected.")
        return None
    df = pd.DataFrame(decisions)

    df["cliff_active"] = df["triggers"].apply(lambda t: bool((t or {}).get("cliff_proximity")))
    pit_now = df[df["gate_decision"] == "PIT_NOW"]
    early5 = pit_now[pit_now["tyre_age"] < 5]
    early6 = pit_now[pit_now["tyre_age"] < 6]

    n_actual_pits = int(df["actual_is_pit_in_lap"].fillna(False).sum())
    agreed_mask = df["actual_is_pit_in_lap"].fillna(False) & df["execution"].apply(
        lambda e: (e or {}).get("driving_instruction") == "PIT_LAP")
    n_agreed = int(agreed_mask.sum())
    n_missed = n_actual_pits - n_agreed

    false_early = early5[~early5["actual_is_pit_in_lap"].fillna(False)]

    cliff_share_pct = (pit_now["cliff_active"].mean() * 100) if len(pit_now) else float("nan")

    if not quiet:
        print(f"\n[{floor_label}]  (n_decisions={len(df)})")
        print(f"  total PIT_NOW:                       {len(pit_now)}")
        print(f"  PIT_NOW at tyre_age<5:                {len(early5)}")
        print(f"  PIT_NOW at tyre_age<6:                {len(early6)}")
        print(f"  cliff_proximity active in PIT_NOW:    {cliff_share_pct:.1f}% of PIT_NOW rows")
        print(f"  real pit-in laps in window:           {n_actual_pits}")
        print(f"  HERMES agreed (PIT_LAP) on:           {n_agreed}")
        print(f"  real pits HERMES's PIT_LAP MISSED:    {n_missed}")
        print(f"  'false/early' PIT_NOW<age5 (rough proxy, no real stop that lap): {len(false_early)}")

    return {"floor": floor_label, "total_pit_now": len(pit_now), "pit_now_age_lt5": len(early5),
            "pit_now_age_lt6": len(early6), "cliff_share_pct": round(cliff_share_pct, 1),
            "real_pits": n_actual_pits, "agreed": n_agreed, "missed": n_missed,
            "false_early_lt5": len(false_early)}

def main():
    all_rows = []
    per_race_rows = []
    for floor_age in FLOORS:
        floor_label = "no floor" if floor_age is None else f">= age {floor_age}"
        print("\n" + "=" * 78)
        print(f"CONDITION: {floor_label}")
        print("=" * 78)
        all_decisions = []
        for season, race, session, d1, d2 in RACES:
            print(f"  running {season} {race} {session} ({d1}/{d2})...")
            race_decisions = run_one(season, race, session, d1, d2, floor_age)
            all_decisions.extend(race_decisions)
            if race_decisions:
                race_row = summarize(race_decisions, f"{floor_label} | {season} {race}", quiet=True)
                if race_row:
                    race_row["race"] = f"{season}_{race}"
                    per_race_rows.append(race_row)
        row = summarize(all_decisions, floor_label)
        if row:
            all_rows.append(row)

    gate3.evaluate_tier3 = _real_evaluate_tier3

    print("\n" + "=" * 78)
    print("PER-RACE BREAKDOWN - this is the table that actually answers 'does it")
    print("generalize', not just the combined one below. Look for whether EVERY")
    print("race shows early PIT_NOW shrinking with the floor while 'missed' stays")
    print("flat - if one race breaks that pattern, the floor is not a general fix.")
    print("=" * 78)
    if per_race_rows:
        per_race_df = pd.DataFrame(per_race_rows)[
            ["race", "floor", "total_pit_now", "pit_now_age_lt5", "pit_now_age_lt6",
             "real_pits", "agreed", "missed", "false_early_lt5"]]
        print(per_race_df.sort_values(["race", "floor"]).to_string(index=False))

    print("\n" + "=" * 78)
    print("SUMMARY ACROSS ALL CONDITIONS (all configured races COMBINED - can mask")
    print("a race that disagrees with the others; check the per-race table above first)")
    print("=" * 78)
    if all_rows:
        print(pd.DataFrame(all_rows).to_string(index=False))
        print("\nAdopt a floor ONLY if, in the per-race table above, EVERY race shows:")
        print("  early PIT_NOW (age<5/age<6) decreasing as the floor rises, AND")
        print("  'missed' staying flat (not increasing), AND")
        print("  the same direction of effect (not one race improving while another")
        print("  worsens, even if the combined row above looks fine on net).")
        print("If any race breaks that pattern, don't adopt the floor - the combined")
        print("row alone is not sufficient evidence, by design.")
    else:
        print("No results collected - check the RACES list and HERMES_REPO_ROOT/GCS_CACHE_DIR.")

if __name__ == "__main__":
    main()
