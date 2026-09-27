#!/usr/bin/env python3
"""
Pull the real pit-lap decision rows, side-by-side - British GP investigation,
step 1 (evidence only, no fix, no hypothesis baked in).

Standalone, READ-ONLY. Does not modify hermes_master.py, gate-tier-3.py,
execution-tree.py, or sc-gamble.py - runs the real replay unmodified and
cross-references laps_features.csv directly for the handful of fields the
decision output doesn't currently surface (actual next compound, raw
TrackStatus, raw Rainfall).

Field coverage against the requested list:
  driver, lap, compound, tyre_age              -> from the decision row directly
  cliff probability / cliff_proximity          -> projection.cliff_probability_next_5_laps
                                                   + triggers.cliff_proximity
  all Tier-3 triggers                          -> the full triggers dict, printed whole
  Tier-3 decision + instruction                -> gate_decision / execution.driving_instruction
  Tier 1 / Tier 2 result if applicable         -> tier_reached tells you which tier produced
                                                   gate_decision (1 or 2 means Tier 1/2 fired and
                                                   Tier 3 was never reached that lap)
  rival undercut / overcut signal              -> triggers.undercut / overcut / rival_undercut_threat
  dirty-air signal                             -> triggers.dirty_air
  SC/VSC state                                 -> triggers.safety_car (Tier-3's OWN reading) AND
                                                   the raw TrackStatus string (ground truth) side by
                                                   side, since these can legitimately differ (VSC
                                                   codes, string concatenation across a lap - BP §2.3)
  weather state                                -> raw Rainfall flag (rain_probability_pct isn't a
                                                   documented real column - if it's absent, that's
                                                   itself worth knowing, not silently guessed)
  actual next tyre                             -> looked up directly: next lap's Compound for the
                                                   same driver
  competing reason to PIT_LATER/STAY OUT       -> the full triggers dict + reason already shows
                                                   this; no separate field needed

Run from the repo root (same place you run master.py from):
    python pull_real_pit_decisions.py
Edit SEASON/RACE/SESSION/D1/D2 below if you want to point this at a
different race for the same kind of side-by-side pull later.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", ".")).resolve()
HERMES_MASTER_PATH = Path(os.environ.get("HERMES_MASTER_PATH", "system/HERMES/trees/master.py"))
SEASON = 2024
RACE = "British_Grand_Prix"
SESSION = "R"
D1, D2 = "VER", "PER"


def die(msg):
    print(msg)
    sys.exit(1)


if not HERMES_MASTER_PATH.exists():
    die(f"Could not find {HERMES_MASTER_PATH.resolve()} - set HERMES_MASTER_PATH.")

spec = importlib.util.spec_from_file_location("hermes_master_under_test", HERMES_MASTER_PATH)
hm = importlib.util.module_from_spec(spec)
sys.modules["hermes_master_under_test"] = hm
spec.loader.exec_module(hm)

tyre_models = hm.load_tyre_models(REPO_ROOT / "tyre_life_models.pkl")
sc_prior = hm.load_sc_prior(REPO_ROOT / "sc_vsc_circuit_level_prior.csv")
cliff_stints = hm.load_cliff_stints(REPO_ROOT / "cliff_detection_stints.csv")
taxonomy = hm.load_circuit_taxonomy(REPO_ROOT / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
pit_loss_table = hm.load_pit_loss_table(
    REPO_ROOT / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv")
degr_ordinal, _ = hm.circuit_degredation_ordinal_for(taxonomy, RACE)
pit_loss_s, _ = hm.pit_loss_for_circuit(pit_loss_table, RACE)

laps_path = hm.find_laps_features(SEASON, RACE, SESSION)
if laps_path is None:
    die(f"laps_features.csv not found for {SEASON} {RACE} {SESSION} under gcs_cache.")
raw_laps = pd.read_csv(laps_path, dtype={"TrackStatus": str})
total_laps = int(raw_laps["LapNumber"].max())
has_rain_prob_col = "rain_probability_pct" in raw_laps.columns
print(f"[note] 'rain_probability_pct' column present in laps_features.csv: {has_rain_prob_col} "
      f"(if False, unsafe_weather could only ever use the raw Rainfall flag this whole race - "
      f"itself worth knowing, not assumed).")

ctx = hm.RaceContext(
    season=SEASON, circuit=RACE, total_laps=total_laps, is_sprint_weekend=False,
    d1_code=D1, d2_code=D2, session=SESSION, circuit_degredation_ordinal=degr_ordinal,
    pit_loss_s=pit_loss_s, p_sc_5lap=hm.get_sc_probability(sc_prior, RACE),
)
resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
decisions = hm.run_replay(ctx, resources, range(1, total_laps + 1), explain=False)

real_pit_rows = [d for d in decisions if d.get("actual_is_pit_in_lap")]
print(f"\nFound {len(real_pit_rows)} real pit-in-lap decisions for {D1}/{D2} "
      f"in {SEASON} {RACE} {SESSION}.\n")


def lookup_raw(driver, lap):
    match = raw_laps[(raw_laps["Driver"] == driver) & (raw_laps["LapNumber"] == lap)]
    return match.iloc[0] if not match.empty else None


def next_compound(driver, lap):
    nxt = raw_laps[(raw_laps["Driver"] == driver) & (raw_laps["LapNumber"] == lap + 1)]
    return nxt.iloc[0]["Compound"] if not nxt.empty else None


rows_out = []
for d in real_pit_rows:
    raw_row = lookup_raw(d["driver"], d["lap"])
    track_status_raw = str(raw_row.get("TrackStatus", "")) if raw_row is not None else None
    rainfall_raw = bool(raw_row.get("Rainfall", False)) if raw_row is not None else None
    rain_prob = raw_row.get("rain_probability_pct") if (raw_row is not None and has_rain_prob_col) else None
    next_tyre = next_compound(d["driver"], d["lap"])
    trig = d.get("triggers") or {}
    exe = d.get("execution") or {}
    proj = d.get("projection") or {}
    sc_gamble = d.get("sc_gamble") or {}

    rows_out.append({
        "driver": d["driver"], "lap": d["lap"], "compound": d["compound"], "tyre_age": d["tyre_age"],
        "cliff_probability": proj.get("cliff_probability_next_5_laps"),
        "predicted_pace_loss": proj.get("predicted_pace_loss"),
        "trig_cliff_proximity": trig.get("cliff_proximity"),
        "trig_pace_lap_delta": trig.get("pace_lap_delta"),
        "trig_tyre_age": trig.get("tyre_age"),
        "trig_undercut": trig.get("undercut"),
        "trig_overcut": trig.get("overcut"),
        "trig_rival_undercut_threat": trig.get("rival_undercut_threat"),
        "trig_safety_car": trig.get("safety_car"),
        "trig_dirty_air": trig.get("dirty_air"),
        "driver_stress_signal": d.get("driver_stress_signal"),
        "sc_gamble_recommendation": sc_gamble.get("recommendation"),
        "tier_reached": d["tier_reached"],
        "gate_decision": d["gate_decision"],
        "reason": d.get("reason"),
        "execution_decision": exe.get("decision"),
        "driving_instruction": exe.get("driving_instruction"),
        "track_status_RAW": track_status_raw,
        "rainfall_RAW": rainfall_raw,
        "rain_probability_pct_RAW": rain_prob,
        "actual_next_compound": next_tyre,
        "adjacency": d.get("adjacency"),
        "data_quality_notes": d.get("data_quality_notes"),
    })

df = pd.DataFrame(rows_out)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)

# Print in two blocks (core decision fields, then context/raw fields) since
# the full row is too wide for any terminal to show usefully in one table.
core_cols = ["driver", "lap", "compound", "tyre_age", "cliff_probability", "predicted_pace_loss",
             "trig_cliff_proximity", "trig_pace_lap_delta", "trig_tyre_age", "trig_undercut",
             "trig_overcut", "trig_rival_undercut_threat", "trig_safety_car", "trig_dirty_air",
             "tier_reached", "gate_decision", "reason", "execution_decision", "driving_instruction"]
context_cols = ["driver", "lap", "sc_gamble_recommendation", "driver_stress_signal",
                "track_status_RAW", "rainfall_RAW", "rain_probability_pct_RAW",
                "actual_next_compound", "data_quality_notes"]

print("=" * 100)
print("CORE DECISION FIELDS (per real pit-in lap)")
print("=" * 100)
print(df[core_cols].to_string(index=False))

print("\n" + "=" * 100)
print("CONTEXT / RAW FIELDS")
print("=" * 100)
print(df[context_cols].to_string(index=False))

out_path = REPO_ROOT / "british_gp_real_pit_decisions.csv"
df.to_csv(out_path, index=False)
print(f"\nAlso wrote {out_path} for easier side-by-side review (e.g. in Excel).")
print("\nNo classification or hypothesis applied here - this is the raw evidence pull only.")