""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

STINTS_CSV = "cliff_detection_stints.csv"
TAXONOMY_PATH = r"src\taxanomy\circuit_taxonomy.xlsx"

MIN_EVENTS_PER_COMPOUND = 15
EXCLUDED_COMPOUNDS = ["WET"]

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

def load_and_prepare() -> pd.DataFrame:
    stints = pd.read_csv(STINTS_CSV)
    print(f"[load] {len(stints)} stints from {STINTS_CSV}")

    stints = stints[stints["session"] == "R"]
    print(f"[filter] {len(stints)} stints after excluding Sprint sessions")

    stints = stints[~stints["compound"].isin(EXCLUDED_COMPOUNDS)]
    print(f"[filter] {len(stints)} stints after excluding {EXCLUDED_COMPOUNDS} (insufficient events)")

    event_counts = stints.groupby("compound")["event"].sum().sort_values(ascending=False)
    too_few = event_counts[event_counts < MIN_EVENTS_PER_COMPOUND].index.tolist()
    if too_few:
        print(f"[filter] also excluding {too_few} - fewer than {MIN_EVENTS_PER_COMPOUND} events, "
              f"can't meaningfully estimate a stratum from this few")
        stints = stints[~stints["compound"].isin(too_few)]

    print(f"[info] event counts by compound (kept strata):\n{event_counts[~event_counts.index.isin(too_few)].to_string()}")

    race_to_id = {r: cid for cid, races in CIRCUIT_ID_TO_RACE_NAMES.items() for r in races}
    stints["circuit_id"] = stints["race"].map(race_to_id)

    taxonomy = pd.read_excel(TAXONOMY_PATH)[["circuit_id", "circuit_degredation"]]
    stints = stints.merge(taxonomy, on="circuit_id", how="left")

    unmatched = stints[stints["circuit_degredation"].isna()]["race"].unique()
    if len(unmatched) > 0:
        print(f"[warn] races with no circuit_degredation match, dropped from Cox: {unmatched.tolist()}")
    stints = stints.dropna(subset=["circuit_degredation"])

    severity_map = {"low": 0, "medium": 1, "high": 2}
    stints["circuit_degredation_ordinal"] = stints["circuit_degredation"].map(severity_map)

    return stints

def build_cox_design(stints: pd.DataFrame, include_is_rbr: bool = False) -> pd.DataFrame:
    required = ["duration", "event", "compound", "track_temp_bucket", "fuel_load_estimate",
                "stint_number", "regulation_era", "circuit_degredation_ordinal"]
    df = stints.dropna(subset=required).copy()

    design = pd.get_dummies(
        df[["duration", "event", "compound", "track_temp_bucket", "regulation_era"]],
        columns=["track_temp_bucket", "regulation_era"], drop_first=True,
    )
    design["fuel_load_estimate"] = df["fuel_load_estimate"].values
    design["stint_number"] = df["stint_number"].values
    design["circuit_degredation_ordinal"] = df["circuit_degredation_ordinal"].values
    if include_is_rbr:
        design["is_rbr"] = df["is_rbr"].astype(int).values

    design["compound"] = df["compound"].values
    return design

if __name__ == "__main__":
    stints = load_and_prepare()

    design = build_cox_design(stints, include_is_rbr=False)
    print(f"\n[cox] fitting on {len(design)} stints, stratified by compound: "
          f"{design['compound'].unique().tolist()}")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(design, duration_col="duration", event_col="event", strata=["compound"])

    print("\n=== COX MODEL SUMMARY (main model, no team/driver) ===")
    cph.print_summary()
    print(f"\n[cox] concordance index: {cph.concordance_index_:.3f}")

    fuel_coef = cph.params_["fuel_load_estimate"]
    print(f"\n[precision] fuel_load_estimate exact coefficient: {fuel_coef:.6f}")
    print(f"[precision] hazard ratio for a 50kg fuel difference: {np.exp(fuel_coef * 50):.3f}")
    print(f"[precision] hazard ratio for full fuel range (0-110kg): {np.exp(fuel_coef * 110):.3f}")

    design_rbr = build_cox_design(stints, include_is_rbr=True)
    cph_rbr = CoxPHFitter(penalizer=0.1)
    cph_rbr.fit(design_rbr, duration_col="duration", event_col="event", strata=["compound"])

    print("\n=== ROBUSTNESS CHECK: same model + is_rbr ===")
    print("(if is_rbr is NOT significant here, that supports training on all teams; "
          "if it IS significant, that's a real limitation to report)")
    cph_rbr.print_summary()
