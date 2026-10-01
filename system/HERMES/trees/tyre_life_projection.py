""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from lifelines import CoxPHFitter
from lifelines.exceptions import StatError

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

def build_tyre_life_projection(
    *,
    reg_models: dict,
    cph: CoxPHFitter,
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
    ""
    result = {"predicted_pace_loss": None, "cliff_probability_next_5_laps": None}

    reg_key = (compound, circuit, regulation_era)
    model = reg_models.get(reg_key)
    if model is not None:
        X = _regression_design_row(tyre_age, fuel_load_estimate, stint_number,
                                    track_temp_bucket, temp_dummy_columns)
        result["predicted_pace_loss"] = float(model.predict(X)[0])

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
        pass

    return result

if __name__ == "__main__":
    print("[self-test] fitting lightweight synthetic models to verify the interface, "
          "NOT the real tyre data - run against real fitted models before trusting output")

    np.random.seed(0)
    n = 300
    temp_dummy_columns = ["track_temp_bucket_hot", "track_temp_bucket_warm"]
    era_dummy_columns = ["regulation_era_2022-2025"]

    X_reg = pd.DataFrame({
        "tyre_age": np.random.randint(1, 30, n),
        "fuel_load_estimate": np.random.uniform(0, 110, n),
        "stint_number": np.random.randint(1, 4, n),
        "track_temp_bucket_hot": np.random.randint(0, 2, n),
        "track_temp_bucket_warm": np.random.randint(0, 2, n),
    })
    y_reg = 0.03 * X_reg["tyre_age"] + 0.01 * X_reg["fuel_load_estimate"] + np.random.normal(0, 0.1, n)
    reg_models = {("MEDIUM", "Bahrain_Grand_Prix", "2022-2025"): LinearRegression().fit(X_reg, y_reg)}

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
