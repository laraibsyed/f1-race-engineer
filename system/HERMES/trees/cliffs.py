#!/usr/bin/env python3
"""
Diagnose the SOFT / Bahrain_Grand_Prix / 2022-2025 cliff-probability stratum
==============================================================================
Standalone, READ-ONLY diagnostic. Does not modify hermes_master.py,
sc-gamble.py, gate-tier-3.py, or tyre_life_projection.py - just inspects
tyre_life_models.pkl and cliff_detection_stints.csv directly.

Answers, in order (per the agreed decision tree - do NOT touch 0.017,
cliff_proximity, or the Tier-3 gate until this comes back):

    thin stratum (few events)  -> investigate fallback/confidence handling
    adequate data              -> inspect the Cox fit + baseline hazard
    baseline hazard itself
    non-monotonic in age       -> that's the root cause, found directly

Run from the repo root (same place you run master.py from):
    python diagnose_cliff_stratum.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(".")   # edit if you run this from somewhere else
COMPOUND = "SOFT"
CIRCUIT = "Bahrain_Grand_Prix"
ERA = "2022-2025"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# STEP 1: load the pickle, inspect its structure defensively
# ============================================================================
section("STEP 1: load tyre_life_models.pkl")
pkl_path = REPO_ROOT / "tyre_life_models.pkl"
if not pkl_path.exists():
    print(f"NOT FOUND: {pkl_path.resolve()}")
    print("Run this from the repo root, or edit REPO_ROOT at the top of this file.")
    sys.exit(1)

with open(pkl_path, "rb") as f:
    models = pickle.load(f)

print(f"type(models) = {type(models)}")
if not hasattr(models, "keys"):
    print("models has no .keys() - printing repr directly (truncated):")
    print(repr(models)[:2000])
    sys.exit(0)
print(f"top-level keys: {list(models.keys())}")

reg_models = models.get("reg_models")
cph = models.get("cph")
temp_dummy_columns = models.get("temp_dummy_columns")
era_dummy_columns = models.get("era_dummy_columns")

print(f"\nreg_models: type={type(reg_models)}, n_keys={len(reg_models) if reg_models else 0}")
print(f"cph: type={type(cph)}")
print(f"temp_dummy_columns: {temp_dummy_columns}")
print(f"era_dummy_columns: {era_dummy_columns}")


# ============================================================================
# STEP 2: find the exact regression stratum key
# ============================================================================
section(f"STEP 2: find the regression key ({COMPOUND!r}, {CIRCUIT!r}, {ERA!r})")
target_key = (COMPOUND, CIRCUIT, ERA)
if reg_models is None:
    print("reg_models is None/missing.")
elif target_key in reg_models:
    model = reg_models[target_key]
    print(f"FOUND exact key: {target_key}")
    print(f"  model: {model}")
    if hasattr(model, "coef_"):
        print(f"  coef_: {model.coef_}")
    if hasattr(model, "intercept_"):
        print(f"  intercept_: {model.intercept_}")
else:
    print(f"NOT FOUND exact key {target_key}.")
    print(f"  keys containing '{CIRCUIT}': {[k for k in reg_models if CIRCUIT in str(k)]}")
    print(f"  keys with compound={COMPOUND}: {[k for k in reg_models if k[0] == COMPOUND][:10]}")


# ============================================================================
# STEP 3: inspect the Cox model itself
# ============================================================================
section("STEP 3: inspect the Cox model (cph)")
if cph is None:
    print("cph is None/missing.")
else:
    print(f"type(cph): {type(cph)}")
    for attr in ("strata", "params_", "baseline_hazard_", "baseline_cumulative_hazard_",
                 "baseline_survival_", "event_col", "duration_col"):
        if hasattr(cph, attr):
            val = getattr(cph, attr)
            if isinstance(val, pd.DataFrame):
                print(f"\n  {attr}: DataFrame shape={val.shape}")
                print(val.head(15).to_string())
            else:
                print(f"  {attr}: {val}")
    try:
        print("\n  --- cph.print_summary() ---")
        cph.print_summary()
    except Exception as e:  # noqa: BLE001
        print(f"  print_summary() failed: {e}")


# ============================================================================
# STEP 4: real sample size for this exact stratum, from cliff_detection_stints.csv
# ============================================================================
section("STEP 4: cliff_detection_stints.csv - real sample size for this stratum")
csv_path = REPO_ROOT / "cliff_detection_stints.csv"
if not csv_path.exists():
    print(f"NOT FOUND: {csv_path.resolve()}")
else:
    df = pd.read_csv(csv_path)
    print(f"columns: {df.columns.tolist()}")
    print(f"total rows: {len(df)}")

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        lower_map = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    # Documented schema first (season,race,session,driver,stint,team,is_rbr,
    # compound,circuit,n_laps_cleaned,n_laps_true,duration,event,
    # cliff_tyre_age,slope_ratio,improvement,track_temp_bucket,
    # fuel_load_estimate,stint_number,regulation_era) - falls back to a
    # case-insensitive / alias search if these aren't actually present.
    compound_col = find_col(["compound"])
    circuit_col = find_col(["circuit"])
    era_col = find_col(["regulation_era", "era"])
    event_col = find_col(["event", "cliff_event"])
    duration_col = find_col(["duration"])
    age_col = find_col(["cliff_tyre_age", "tyre_age"])
    n_laps_col = find_col(["n_laps_true"])

    print(f"\nresolved columns: compound={compound_col}, circuit={circuit_col}, "
          f"era={era_col}, event={event_col}, duration={duration_col}, "
          f"cliff_age={age_col}, n_laps_true={n_laps_col}")

    if None in (compound_col, circuit_col, era_col):
        print("\nCould not resolve compound/circuit/era columns automatically.")
        print("Paste the columns list above plus this head() so it can be fixed:")
        print(df.head(3).to_string())
    else:
        mask = (
            (df[compound_col].astype(str).str.upper() == COMPOUND)
            & (df[circuit_col].astype(str) == CIRCUIT)
            & (df[era_col].astype(str) == ERA)
        )
        stratum = df[mask]
        print(f"\nRows matching ({COMPOUND}, {CIRCUIT}, {ERA}): n = {len(stratum)}")
        if len(stratum) > 0:
            if event_col:
                print(f"  n_events (sum of '{event_col}'): {stratum[event_col].sum()}")
            if duration_col:
                print(f"  '{duration_col}' range: {stratum[duration_col].min()} - {stratum[duration_col].max()}")
            if age_col and event_col:
                cliff_rows = stratum[stratum[event_col] == 1]
                print(f"  n cliff-event rows: {len(cliff_rows)}")
                if len(cliff_rows) > 0:
                    print(f"  '{age_col}' at cliff events - min/median/max: "
                          f"{cliff_rows[age_col].min()}, {cliff_rows[age_col].median()}, "
                          f"{cliff_rows[age_col].max()}")
            if n_laps_col:
                print(f"  '{n_laps_col}' range: {stratum[n_laps_col].min()} - {stratum[n_laps_col].max()}")
            print("\n  Full stratum rows:")
            show_cols = [c for c in [compound_col, circuit_col, era_col, duration_col, event_col,
                                       age_col, n_laps_col, "season", "driver"] if c and c in df.columns]
            print(stratum[show_cols].to_string(index=False))
        else:
            print("  ZERO rows matched exactly - checking circuit name / era spelling:")
            near = df[(df[compound_col].astype(str).str.upper() == COMPOUND)
                      & (df[circuit_col].astype(str).str.contains("Bahrain", case=False, na=False))]
            print(f"  rows with compound=SOFT and circuit containing 'Bahrain' (any era): n={len(near)}")
            if len(near) > 0:
                print(near[[compound_col, circuit_col, era_col]].drop_duplicates().to_string(index=False))


# ============================================================================
# STEP 5: does the non-monotonicity live in S(t) itself, or in the
#         1 - S(t+5)/S(t) conditional-probability step?
# ============================================================================
section("STEP 5: S(t) monotonicity vs. the 5-lap conditional probability")
if reg_models is None or cph is None or not temp_dummy_columns or not era_dummy_columns:
    print("Missing reg_models/cph/temp_dummy_columns/era_dummy_columns - can't run this step.")
else:
    def cox_design_row(fuel_load_estimate, stint_number, circuit_degredation_ordinal,
                        track_temp_bucket, regulation_era):
        """Same B1-fixed column matching hermes_master.py actually uses -
        this reflects the model as HERMES calls it today, not the pre-fix bug."""
        row = {"fuel_load_estimate": fuel_load_estimate, "stint_number": stint_number,
               "circuit_degredation_ordinal": circuit_degredation_ordinal}
        for col in temp_dummy_columns:
            row[col] = 1.0 if col == f"temp_{track_temp_bucket}" else 0.0
        for col in era_dummy_columns:
            row[col] = 1.0 if col == f"regulation_era_{regulation_era}" else 0.0
        row["compound"] = COMPOUND
        return pd.DataFrame([row])

    # Fixed inputs (fuel/stint/temp/degr aren't in the pasted out.jsonl, so
    # these are reasonable stand-ins) - only tyre_age varies below, isolating
    # the age term specifically, matching what count_active_triggers() sees.
    FUEL = 90.0          # ASSUMPTION for this diagnostic only
    STINT_NUMBER = 1
    TEMP_BUCKET = "hot"  # ASSUMPTION - doesn't change the SHAPE of the age curve
    DEGR_ORDINAL = 1

    row = cox_design_row(FUEL, STINT_NUMBER, DEGR_ORDINAL, TEMP_BUCKET, ERA)

    ages = list(range(1, 30))
    try:
        times = sorted(set(ages) | {a + 5 for a in ages})
        surv_fn = cph.predict_survival_function(row, times=times)
        survival_at_age = {}
        print(f"{'age t':>6} {'S(t)':>10} {'S(t+5)':>10} {'S(t+5)/S(t)':>14} {'cliff_p=1-ratio':>18}")
        for t in ages:
            if t not in surv_fn.index or (t + 5) not in surv_fn.index:
                continue
            s_t = float(surv_fn.loc[t].iloc[0])
            s_t5 = float(surv_fn.loc[t + 5].iloc[0])
            ratio = (s_t5 / s_t) if s_t > 0 else None
            cliff_p = (1 - ratio) if ratio is not None else None
            print(f"{t:>6} {s_t:>10.5f} {s_t5:>10.5f} "
                  f"{'n/a' if ratio is None else round(ratio, 5):>14} "
                  f"{'n/a' if cliff_p is None else round(cliff_p, 5):>18}")
            survival_at_age[t] = s_t

        ages_sorted = sorted(survival_at_age)
        violations = [(a1, a2) for a1, a2 in zip(ages_sorted, ages_sorted[1:])
                      if survival_at_age[a2] > survival_at_age[a1] + 1e-9]
        print(f"\nS(t) monotonicity (should be non-increasing in t): "
              f"{'PASS' if not violations else f'FAIL - {len(violations)} violations, e.g. {violations[:5]}'}")
        print("\nIMPORTANT MATHEMATICAL CAVEAT: a Cox model's survival function is")
        print("guaranteed non-increasing by construction (the Breslow cumulative-hazard")
        print("estimator can't decrease) - so 'S(t) FAIL' above is the UNLIKELY outcome,")
        print("not the expected failure mode. S(t) passing does NOT mean the model is")
        print("healthy; it just means the maths worked as designed.")
        print("\nThe actually informative signal is the HAZARD RATE shape (below), not")
        print("S(t)'s monotonicity: cumulative hazard H(t) can rise steeply then flatten")
        print("(a 'higher risk early, declining risk once a tyre survives its first few")
        print("laps' shape - a real, named pattern in survival analysis, sometimes called")
        print("infant-mortality-shaped) - and that alone, with NO bug anywhere, would")
        print("exactly reproduce cliff_p rising then falling with age. Whether that shape")
        print("reflects real tyre physics or a small-sample artifact in THIS stratum is")
        print("exactly what Step 4's n/n_events answers - a real, data-driven hump needs")
        print("real n; a hump built from a handful of events is much more likely noise.")

        if hasattr(cph, "baseline_cumulative_hazard_"):
            bch = cph.baseline_cumulative_hazard_
            print(f"\nbaseline_cumulative_hazard_ columns: {list(bch.columns)}")
            compound_col_bch = next((c for c in bch.columns if COMPOUND in str(c)), bch.columns[0])
            print(f"using column: {compound_col_bch!r}")
            h = bch[compound_col_bch]
            h_at_ages = h.reindex(h.index.union(ages)).sort_index().ffill().reindex(ages)
            hazard_rate = h_at_ages.diff()  # approximate instantaneous hazard per lap of age
            print(f"\n{'age t':>6} {'H(t) cum.hazard':>18} {'approx hazard rate':>20}")
            for t in ages:
                if t in h_at_ages.index and not pd.isna(h_at_ages[t]):
                    rate = hazard_rate.get(t)
                    print(f"{t:>6} {h_at_ages[t]:>18.5f} "
                          f"{'n/a' if pd.isna(rate) else round(rate, 5):>20}")
            print("\nIf 'approx hazard rate' rises then falls with age (peaks early, declines")
            print("later), that is the direct mechanical explanation for the observed")
            print("cliff_probability_next_5_laps shape - go to Step 4's n/n_events next to")
            print("judge whether this specific stratum's hazard shape is trustworthy.")
    except Exception as e:  # noqa: BLE001
        print(f"predict_survival_function failed: {e}")

section("Paste this entire output back for the next diagnosis step.")