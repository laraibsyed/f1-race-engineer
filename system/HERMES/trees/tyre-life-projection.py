"""
Tyre Life Projection Feed
============================
The interface the Gate Tree (Tier 1's cliff-check node, all of Tier 3) and the
Execution Tree actually need at runtime - one function call that returns both
numbers named in the Gate Tree diagram's trigger list ("pace, Tyre age, lap
delta, cliff proximity"):

    predicted_pace_loss              <- Regression V2
    cliff_probability_next_5_laps    <- Cox (survival V2)

Everything before this point in the project has been FITTING these models.
This is the first place they get called TOGETHER, live, for a single tyre
state - which is what was actually missing (the "Produce tyre life
projection feed" to-do item).

IMPORTANT: fit_all_models() re-runs the full fitting pipeline from
tyre_regression_v2.py and survival_model_v2.py. That's fine for testing this
interface, but NOT how a live system should call this every lap - refitting
on every call would be extremely slow. In production, fit once (e.g. once per
race weekend), pickle the fitted objects, and have the tree load the pickle
instead of calling fit_all_models() directly. Flagged here, not solved here -
a real next step once this interface is confirmed correct.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from lifelines import CoxPHFitter
from lifelines.exceptions import StatError


# ---------------------------------------------------------------------------
# Shared column-construction helpers - MUST match tyre_regression_v2.py and
# survival_model_v2.py exactly, or predictions will silently use the wrong
# columns. Duplicated here deliberately (not imported) so this file is a
# single, self-contained interface a tree node can call without needing the
# rest of the fitting pipeline importable.
# ---------------------------------------------------------------------------
def _regression_design_row(tyre_age, fuel_load_estimate, stint_number,
                            track_temp_bucket, temp_dummy_columns):
    row = {"tyre_age": tyre_age, "fuel_load_estimate": fuel_load_estimate,
           "stint_number": stint_number}
    for col in temp_dummy_columns:
        row[col] = 1.0 if col == f"track_temp_bucket_{track_temp_bucket}" else 0.0
    return pd.DataFrame([row])[["tyre_age", "fuel_load_estimate", "stint_number"] + temp_dummy_columns]


def _cox_design_row(fuel_load_estimate, stint_number, circuit_degredation_ordinal,
                     track_temp_bucket, regulation_era, temp_dummy_columns, era_dummy_columns):
    row = {"fuel_load_estimate": fuel_load_estimate, "stint_number": stint_number,
           "circuit_degredation_ordinal": circuit_degredation_ordinal}
    for col in temp_dummy_columns:
        row[col] = 1.0 if col == f"track_temp_bucket_{track_temp_bucket}" else 0.0
    for col in era_dummy_columns:
        row[col] = 1.0 if col == f"regulation_era_{regulation_era}" else 0.0
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# The actual feed function
# ---------------------------------------------------------------------------
def build_tyre_life_projection(
    *,
    reg_models: dict,              # {(compound, circuit, era): fitted LinearRegression}
    cph: CoxPHFitter,              # fitted, stratified by compound
    temp_dummy_columns: list,
    era_dummy_columns: list,
    compound: str,
    circuit: str,
    regulation_era: str,
    tyre_age: float,
    fuel_load_estimate: float,
    stint_number: int,
    track_temp_bucket: str,
    circuit_degredation_ordinal: int,
    cliff_horizon_laps: int = 5,
) -> dict:
    """
    Single call for a tree node: given the current tyre state, returns both
    numbers the Gate Tree/Execution Tree need. Returns None for either value
    if no fitted model exists for this (compound, circuit, era) or compound
    stratum - a tree node MUST handle that case (e.g. fall back to a coarser
    lookup, or treat as "unknown, escalate to Tier 1") rather than crash.
    """
    result = {"predicted_pace_loss": None, "cliff_probability_next_5_laps": None}

    # --- Regression V2: predicted_pace_loss ---
    reg_key = (compound, circuit, regulation_era)
    model = reg_models.get(reg_key)
    if model is not None:
        X = _regression_design_row(tyre_age, fuel_load_estimate, stint_number,
                                    track_temp_bucket, temp_dummy_columns)
        result["predicted_pace_loss"] = float(model.predict(X)[0])

    # --- Cox: cliff_probability_next_5_laps ---
    # A compound with no fitted stratum (WET, or anything excluded via
    # MIN_EVENTS_PER_COMPOUND in survival_model_v2.py) genuinely can't have
    # cliff probability estimated from this model - None is the honest
    # answer here, not a guess.
    try:
        row = _cox_design_row(fuel_load_estimate, stint_number, circuit_degredation_ordinal,
                               track_temp_bucket, regulation_era,
                               temp_dummy_columns, era_dummy_columns)
        row["compound"] = compound
        t0, t1 = tyre_age, tyre_age + cliff_horizon_laps
        survival_fn = cph.predict_survival_function(row, times=[t0, t1])
        s0, s1 = survival_fn.iloc[0, 0], survival_fn.iloc[1, 0]
        result["cliff_probability_next_5_laps"] = float(1 - (s1 / s0)) if s0 > 0 else None
    except (KeyError, StatError):
        pass  # compound has no fitted stratum - leave as None

    return result


# ---------------------------------------------------------------------------
# Self-test with lightweight fitted models (NOT the real pipeline - just
# enough to prove the interface itself is wired correctly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[self-test] fitting lightweight synthetic models to verify the interface, "
          "NOT the real tyre data - run against real fitted models before trusting output")

    np.random.seed(0)
    n = 300
    temp_dummy_columns = ["track_temp_bucket_hot", "track_temp_bucket_warm"]
    era_dummy_columns = ["regulation_era_2022-2025"]

    # --- fake regression model for one (compound, circuit, era) key ---
    X_reg = pd.DataFrame({
        "tyre_age": np.random.randint(1, 30, n),
        "fuel_load_estimate": np.random.uniform(0, 110, n),
        "stint_number": np.random.randint(1, 4, n),
        "track_temp_bucket_hot": np.random.randint(0, 2, n),
        "track_temp_bucket_warm": np.random.randint(0, 2, n),
    })
    y_reg = 0.03 * X_reg["tyre_age"] + 0.01 * X_reg["fuel_load_estimate"] + np.random.normal(0, 0.1, n)
    reg_models = {("MEDIUM", "Bahrain_Grand_Prix", "2022-2025"): LinearRegression().fit(X_reg, y_reg)}

    # --- fake Cox model, stratified by compound ---
    cox_df = pd.DataFrame({
        "duration": np.random.randint(3, 40, n),
        "event": np.random.randint(0, 2, n),
        "compound": np.random.choice(["MEDIUM", "HARD"], n),
        "fuel_load_estimate": np.random.uniform(0, 110, n),
        "stint_number": np.random.randint(1, 4, n),
        "circuit_degredation_ordinal": np.random.randint(0, 3, n),
        "track_temp_bucket_hot": np.random.randint(0, 2, n),
        "track_temp_bucket_warm": np.random.randint(0, 2, n),
        "regulation_era_2022-2025": np.random.randint(0, 2, n),
    })
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="duration", event_col="event", strata=["compound"])

    projection = build_tyre_life_projection(
        reg_models=reg_models, cph=cph,
        temp_dummy_columns=temp_dummy_columns, era_dummy_columns=era_dummy_columns,
        compound="MEDIUM", circuit="Bahrain_Grand_Prix", regulation_era="2022-2025",
        tyre_age=15, fuel_load_estimate=60, stint_number=2,
        track_temp_bucket="hot", circuit_degredation_ordinal=1,
    )
    print("\n[self-test] projection for a known (compound, circuit, era):", projection)

    projection_unknown = build_tyre_life_projection(
        reg_models=reg_models, cph=cph,
        temp_dummy_columns=temp_dummy_columns, era_dummy_columns=era_dummy_columns,
        compound="SOFT", circuit="Monaco_Grand_Prix", regulation_era="2018-2021",
        tyre_age=15, fuel_load_estimate=60, stint_number=2,
        track_temp_bucket="hot", circuit_degredation_ordinal=1,
    )
    print("[self-test] projection for an UNKNOWN (compound, circuit, era) - "
          "predicted_pace_loss should be None:", projection_unknown)