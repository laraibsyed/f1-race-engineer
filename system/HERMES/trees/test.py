import pickle
from tyre_life_projection import build_tyre_life_projection

with open("tyre_life_models.pkl", "rb") as f:
    models = pickle.load(f)

# A real, known-common combo -- MEDIUM at Bahrain in the modern era
projection = build_tyre_life_projection(
    reg_models=models["reg_models"], cph=models["cph"],
    temp_dummy_columns=models["temp_dummy_columns"],
    era_dummy_columns=models["era_dummy_columns"],
    compound="MEDIUM", circuit="Bahrain_Grand_Prix", regulation_era="2022-2025",
    tyre_age=15, fuel_load_estimate=60, stint_number=2,
    track_temp_bucket="hot", circuit_degredation_ordinal=1,
)
print("MEDIUM, Bahrain, 2022-2025, tyre_age=15:", projection)

# WET was excluded from both models (dry-compound-only regression target,
# insufficient events for Cox) -- both fields should legitimately be None,
# not a bug.
projection_none = build_tyre_life_projection(
    reg_models=models["reg_models"], cph=models["cph"],
    temp_dummy_columns=models["temp_dummy_columns"],
    era_dummy_columns=models["era_dummy_columns"],
    compound="WET", circuit="Monaco_Grand_Prix", regulation_era="2022-2025",
    tyre_age=5, fuel_load_estimate=60, stint_number=1,
    track_temp_bucket="warm", circuit_degredation_ordinal=1,
)
print("WET (both fields should be None):", projection_none)