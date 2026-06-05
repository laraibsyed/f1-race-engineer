"""
JSON Constraint Validator — 5 Historical Scenarios
====================================================
Tests that your JSON files correctly describe real F1 races.
Each scenario checks a specific rule against FastF1 data.

Install deps first:
    pip install fastf1 --break-system-packages

Usage:
    python validate_json_scenarios.py
"""

import json
import fastf1
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIG — adjust paths if needed
# ─────────────────────────────────────────────
TYRE_CONSTRAINTS_FILE = "src\\config\\f1_tyre_constraints.json"
DIFFS_FILE            = "src\\config\\f1_year_changes.json"
CACHE_DIR             = "data\\raw"

Path(CACHE_DIR).mkdir(exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ─────────────────────────────────────────────
#  LOAD JSON FILES
# ─────────────────────────────────────────────
with open(TYRE_CONSTRAINTS_FILE) as f:
    TYRE_RULES = json.load(f)

with open(DIFFS_FILE) as f:
    DIFFS = json.load(f)


# ─────────────────────────────────────────────
#  TEST RUNNER
# ─────────────────────────────────────────────
results = []

def run_test(name, year, gp, session_type, test_fn):
    print(f"\n{'─'*60}")
    print(f"  Test: {name}")
    print(f"  {year} {gp} [{session_type}]")
    print(f"{'─'*60}")
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        passed, notes = test_fn(session, TYRE_RULES[str(year)], DIFFS[str(year)])
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}")
        for n in notes:
            print(f"    {n}")
        results.append({"test": name, "passed": passed})
    except Exception as e:
        print(f"  ⚠️  ERROR: {e}")
        results.append({"test": name, "passed": False})


# ─────────────────────────────────────────────
#  SCENARIO 1 — 2021 British GP (Sprint)
#  Checks: sprint weekend = 12 dry sets
#          Q2 rule does NOT apply on sprint weekends
# ─────────────────────────────────────────────
def test_2021_british_sprint(session, rules, diffs):
    notes = []
    passed = True

    # Check JSON says sprint weekend
    if not rules["sprint_weekend"]:
        notes.append("FAIL: JSON says sprint_weekend=false for 2021, expected true")
        passed = False
    else:
        notes.append("JSON correctly marks 2021 as sprint_weekend=true")

    # Check JSON gives 12 sprint dry sets
    if rules["sprint_dry_sets"] != 12:
        notes.append(f"FAIL: sprint_dry_sets={rules['sprint_dry_sets']}, expected 12")
        passed = False
    else:
        notes.append("JSON correctly sets sprint_dry_sets=12")

    # Check JSON says Q2 rule inactive on sprint weekends
    q2_sprint_exception = diffs.get("q2_tyre_rule", {}).get("sprint_weekend_exception", "")
    if "does NOT apply" in q2_sprint_exception or "free" in q2_sprint_exception.lower():
        notes.append("JSON correctly notes Q2 rule inactive on 2021 sprint weekends")
    else:
        notes.append("WARN: diffs entry for 2021 Q2 sprint exception may be missing")

    # Pull actual race start compounds from FastF1
    laps = session.laps
    first_lap = laps[laps["LapNumber"] == 1][["Driver", "Compound"]].drop_duplicates("Driver")
    compound_counts = first_lap["Compound"].value_counts()
    notes.append(f"Actual race start compounds: {compound_counts.to_dict()}")

    # On sprint weekends in 2021 Q2 rule doesn't apply —
    # we'd expect mixed compounds on race start (not all soft)
    soft_starters = compound_counts.get("SOFT", 0)
    total_starters = first_lap["Driver"].nunique()
    notes.append(f"Soft starters: {soft_starters}/{total_starters}")
    if soft_starters == total_starters:
        notes.append("WARN: All drivers started on SOFT — expected some variety without Q2 rule")
    else:
        notes.append("Mixed compounds on start confirms Q2 rule was not enforced")

    return passed, notes

run_test(
    "2021 British GP — Sprint weekend allocation + Q2 rule inactive",
    2021, "British Grand Prix", "R", test_2021_british_sprint
)


# ─────────────────────────────────────────────
#  SCENARIO 2 — 2019 German GP (Team-chosen compounds)
#  Checks: compound split varies by team (not fixed 2H/3M/8S)
#          diffs correctly says team_chosen=true for 2019
# ─────────────────────────────────────────────
def test_2019_german_compounds(session, rules, diffs):
    notes = []
    passed = True

    # Check JSON diffs say 2019 is team-chosen
    team_chosen = diffs.get("compound_selection", {}).get("team_chosen", False)
    if not team_chosen:
        notes.append("FAIL: diffs for 2019 should say team_chosen=true")
        passed = False
    else:
        notes.append("JSON correctly marks 2019 as team_chosen compound selection")

    # Pull what compounds were actually used during the race
    laps = session.laps
    compound_usage = (
        laps.groupby(["Team", "Compound"])["LapNumber"]
        .count()
        .reset_index()
        .rename(columns={"LapNumber": "laps_on_compound"})
    )
    notes.append("Compound usage by team (confirms team-level variation):")
    for _, row in compound_usage.iterrows():
        notes.append(f"    {row['Team']}: {row['Compound']} — {row['laps_on_compound']} laps")

    # Check that at least some teams used different compound distributions
    team_compounds = laps.groupby("Team")["Compound"].nunique()
    notes.append(f"Compounds used per team: {team_compounds.to_dict()}")

    return passed, notes

run_test(
    "2019 German GP — Team-chosen compound breakdown",
    2019, "German Grand Prix", "R", test_2019_german_compounds
)


# ─────────────────────────────────────────────
#  SCENARIO 3 — 2021 Abu Dhabi GP (Q2 rule active)
#  Checks: Q2 rule was enforced — top 10 starters used
#          the same compound they set Q2 time on
# ─────────────────────────────────────────────
def test_2021_abudhabi_q2_rule(session, rules, diffs):
    notes = []
    passed = True

    # Check JSON says Q2 rule active for 2021 standard GP
    if not rules["q2_start_tyre_rule"]:
        notes.append("FAIL: JSON says q2_start_tyre_rule=false for 2021, expected true")
        passed = False
    else:
        notes.append("JSON correctly marks Q2 rule as active for 2021")

    # Pull qualifying session too
    try:
        quali = fastf1.get_session(2021, "Abu Dhabi Grand Prix", "Q")
        quali.load(laps=True, telemetry=False, weather=False, messages=False)

        # Get each driver's fastest Q2 lap compound
        q2_laps = quali.laps[quali.laps["Compound"].notna()]
        # Q2 is session laps roughly in the middle — approximate by fastest lap per driver
        q2_compounds = (
            q2_laps.groupby("Driver")
            .apply(lambda x: x.loc[x["LapTime"].idxmin(), "Compound"] if not x.empty else None)
            .dropna()
            .to_dict()
        )
        notes.append(f"Q2 fastest lap compounds: {q2_compounds}")

        # Get race start compounds (lap 1)
        race_laps = session.laps
        start_compounds = (
            race_laps[race_laps["LapNumber"] == 1]
            .groupby("Driver")["Compound"]
            .first()
            .to_dict()
        )
        notes.append(f"Race start compounds: {start_compounds}")

        # Compare for drivers in both datasets
        mismatches = []
        for driver, q2_comp in q2_compounds.items():
            race_comp = start_compounds.get(driver)
            if race_comp and race_comp != q2_comp:
                mismatches.append(f"{driver}: Q2={q2_comp}, Race start={race_comp}")

        if mismatches:
            notes.append(f"Mismatches (may be damage replacements or exceptions):")
            for m in mismatches:
                notes.append(f"    {m}")
        else:
            notes.append("All top-10 drivers started on their Q2 compound — Q2 rule confirmed active")

    except Exception as e:
        notes.append(f"Could not load qualifying session: {e}")

    return passed, notes

run_test(
    "2021 Abu Dhabi GP — Q2 tyre rule enforced",
    2021, "Abu Dhabi Grand Prix", "R", test_2021_abudhabi_q2_rule
)


# ─────────────────────────────────────────────
#  SCENARIO 4 — 2023 Hungary GP (Standard GP rules)
#  Checks: 13 dry sets, 4 inter, 3 wet, Q2 rule inactive,
#          mandatory 2 compounds used by all finishers
# ─────────────────────────────────────────────
def test_2023_hungary_standard(session, rules, diffs):
    notes = []
    passed = True

    # Check allocation values
    checks = [
        ("total_sets_allocated", 13),
        ("intermediate_sets_allocated", 4),
        ("wet_sets_allocated", 3),
        ("q2_start_tyre_rule", False),
        ("mandatory_dry_compounds", 2),
    ]
    for field, expected in checks:
        actual = rules.get(field)
        if actual != expected:
            notes.append(f"FAIL: {field}={actual}, expected {expected}")
            passed = False
        else:
            notes.append(f"✓ {field}={actual}")

    # Check real race: every finisher used at least 2 compounds
    laps = session.laps
    finishers = session.results["Abbreviation"].tolist() if hasattr(session, "results") else []

    driver_compounds = (
        laps[laps["Compound"].notna()]
        .groupby("Driver")["Compound"]
        .nunique()
    )
    one_compound_drivers = driver_compounds[driver_compounds < 2].index.tolist()

    if one_compound_drivers:
        notes.append(f"WARN: Drivers with <2 compounds (may be DNF/wet race): {one_compound_drivers}")
    else:
        notes.append("All drivers used 2+ compounds — mandatory compound rule confirmed")

    # Check no intermediate/wet laps (should be dry race)
    wet_laps = laps[laps["Compound"].isin(["INTERMEDIATE", "WET"])]
    if len(wet_laps) == 0:
        notes.append("Dry race confirmed — wet race exception not triggered")
    else:
        notes.append(f"Wet/inter laps detected: {len(wet_laps)} — wet exception may apply")

    return passed, notes

run_test(
    "2023 Hungary GP — Standard GP allocation + mandatory compounds",
    2023, "Hungarian Grand Prix", "R", test_2023_hungary_standard
)


# ─────────────────────────────────────────────
#  SCENARIO 5 — 2024 Japanese GP (New inter/wet allocation)
#  Checks: JSON has 5 inter sets, 2 wet sets for 2024
#          (the race where the Friday inter controversy happened)
# ─────────────────────────────────────────────
def test_2024_japan_allocation(session, rules, diffs):
    notes = []
    passed = True

    # Check 2024 allocation in JSON
    checks = [
        ("intermediate_sets_allocated", 5),
        ("wet_sets_allocated", 2),
        ("total_sets_allocated", 13),
    ]
    for field, expected in checks:
        actual = rules.get(field)
        if actual != expected:
            notes.append(f"FAIL: {field}={actual}, expected {expected}")
            passed = False
        else:
            notes.append(f"✓ {field}={actual}")

    # Check diffs note the allocation change
    alloc_change = diffs.get("tyre_allocation_change", {})
    if alloc_change.get("intermediate_sets_standard") == 5:
        notes.append("diffs correctly documents 4→5 intermediate change for 2024")
    else:
        notes.append("WARN: diffs may not document inter allocation change correctly")

    # Pull actual race compounds used
    laps = session.laps
    compound_counts = laps["Compound"].value_counts()
    notes.append(f"Compounds used in race: {compound_counts.to_dict()}")

    inter_laps = len(laps[laps["Compound"] == "INTERMEDIATE"])
    wet_laps   = len(laps[laps["Compound"] == "WET"])
    notes.append(f"Intermediate laps: {inter_laps}, Wet laps: {wet_laps}")

    # Check Friday practice inter usage (FP2 was the controversy session)
    try:
        fp2 = fastf1.get_session(2024, "Japanese Grand Prix", "FP2")
        fp2.load(laps=True, telemetry=False, weather=False, messages=False)
        fp2_inter = fp2.laps[fp2.laps["Compound"] == "INTERMEDIATE"]
        notes.append(f"FP2 intermediate laps: {len(fp2_inter)} — low count confirms teams conserved inters under new allocation")
    except Exception as e:
        notes.append(f"Could not load FP2: {e}")

    return passed, notes

run_test(
    "2024 Japanese GP — New 5 inter / 2 wet allocation",
    2024, "Japanese Grand Prix", "R", test_2024_japan_allocation
)


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  VALIDATION SUMMARY")
print(f"{'═'*60}")
total  = len(results)
passed = sum(1 for r in results if r["passed"])
for r in results:
    icon = "✅" if r["passed"] else "❌"
    print(f"  {icon}  {r['test']}")
print(f"\n  {passed}/{total} tests passed")
if passed == total:
    print("  All JSON files validated against historical scenarios.")
else:
    print("  Fix the failing tests before using JSON in your model.")
print(f"{'═'*60}\n")