""

from dataclasses import dataclass
from typing import Optional

def get_regulation_era(season: int) -> str:
    ""
    if season <= 2021:
        return "2018-2021"
    elif season <= 2025:
        return "2022-2025"
    else:
        return "2026+"

def check_degradation_model_compatible(model_era: str, race_season: int) -> bool:
    ""
    return model_era == get_regulation_era(race_season)

STANDARD_WEEKEND_ALLOCATION = 13
SPRINT_WEEKEND_ALLOCATION = 12
YEARS_REQUIRING_FASTF1_ALLOCATION_LOOKUP = {2018, 2019}

SPRINT_SHOOTOUT_CONSUMPTION_YEAR_START = 2024
SPRINT_SHOOTOUT_CONSUMED_SETS = {"MEDIUM": 1, "SOFT": 1}

def get_dry_set_allocation(season: int, is_sprint_weekend: bool) -> Optional[int]:
    ""
    if season in YEARS_REQUIRING_FASTF1_ALLOCATION_LOOKUP:
        return None
    return SPRINT_WEEKEND_ALLOCATION if is_sprint_weekend else STANDARD_WEEKEND_ALLOCATION

def apply_sprint_shootout_consumption(season: int, remaining_sets: dict) -> dict:
    ""
    if season < SPRINT_SHOOTOUT_CONSUMPTION_YEAR_START:
        return remaining_sets
    adjusted = dict(remaining_sets)
    for compound, n in SPRINT_SHOOTOUT_CONSUMED_SETS.items():
        adjusted[compound] = max(0, adjusted.get(compound, 0) - n)
    return adjusted

def compound_available(remaining_sets: dict, compound: str) -> bool:
    ""
    return remaining_sets.get(compound, 0) > 0

Q2_RULE_ACTIVE_YEARS = {2018, 2019, 2020, 2021}

def q2_rule_applies(season: int, is_sprint_weekend: bool) -> bool:
    ""
    if season not in Q2_RULE_ACTIVE_YEARS:
        return False
    if season == 2021 and is_sprint_weekend:
        return False
    return True

def get_mandatory_start_compound(season: int, is_sprint_weekend: bool,
                                  progressed_through_q2: bool,
                                  q2_fastest_lap_compound: Optional[str]) -> Optional[str]:
    ""
    if not q2_rule_applies(season, is_sprint_weekend):
        return None
    if not progressed_through_q2:
        return None
    return q2_fastest_lap_compound

MONACO_TWO_STOP_YEARS = {2025}

def monaco_two_stop_required(season: int, circuit: str) -> bool:
    return season in MONACO_TWO_STOP_YEARS and circuit == "Monaco_Grand_Prix"

def monaco_two_stop_satisfied(season: int, circuit: str, sets_used: int, stops_made: int) -> bool:
    ""
    if not monaco_two_stop_required(season, circuit):
        return True
    return sets_used >= 3 and stops_made >= 2

def mandatory_compound_done(compound_history_dry: set, wet_race_exception: bool) -> bool:
    ""
    if wet_race_exception:
        return True
    return len(compound_history_dry) >= 2

def mandatory_compound_reset_on_red_flag() -> bool:
    ""
    return False

DEADLINE_BUFFER_LAPS = 5

def deadline_approaching(laps_remaining_in_race: int, mandatory_done_flag: bool,
                          buffer_laps: int = DEADLINE_BUFFER_LAPS) -> bool:
    ""
    if mandatory_done_flag:
        return False
    return laps_remaining_in_race <= buffer_laps

@dataclass
class Tier2State:
    season: int
    circuit: str
    is_sprint_weekend: bool
    compound_history_dry: set
    wet_race_exception: bool
    laps_remaining_in_race: int
    remaining_sets: dict
    sets_used_so_far: int
    stops_made_so_far: int

def evaluate_tier2(state: Tier2State) -> str:
    ""
    mand_done = mandatory_compound_done(state.compound_history_dry, state.wet_race_exception)

    if mand_done:
        return "MOVE_TO_TIER_3"

    if deadline_approaching(state.laps_remaining_in_race, mand_done):
        return "PIT_FLEXIBLE"

    return "MOVE_TO_TIER_3"

if __name__ == "__main__":
    print("=== Gate Tree Tier 2 decision sequence ===")
    scenarios = {
        "Already compliant, plenty of laps left": Tier2State(
            2023, "Bahrain_Grand_Prix", False, {"SOFT", "MEDIUM"}, False, 20,
            {"SOFT": 2, "MEDIUM": 3, "HARD": 4}, 2, 1),
        "Non-compliant, deadline close (3 laps left)": Tier2State(
            2023, "Bahrain_Grand_Prix", False, {"SOFT"}, False, 3,
            {"SOFT": 2, "MEDIUM": 3, "HARD": 4}, 1, 0),
        "Non-compliant, but plenty of time left": Tier2State(
            2023, "Bahrain_Grand_Prix", False, {"SOFT"}, False, 30,
            {"SOFT": 2, "MEDIUM": 3, "HARD": 4}, 1, 0),
        "Wet race exception, single compound used": Tier2State(
            2022, "Monaco_Grand_Prix", False, {"SOFT"}, True, 10,
            {"SOFT": 2, "MEDIUM": 3, "HARD": 4}, 1, 0),
    }
    for name, state in scenarios.items():
        print(f"  {name}: {evaluate_tier2(state)}")

    print("\n=== Supporting rule checks (used elsewhere in the constraint system) ===")
    print("Q2 rule, 2020 standard weekend:", q2_rule_applies(2020, False))
    print("Q2 rule, 2021 sprint weekend (should be False):", q2_rule_applies(2021, True))
    print("Q2 rule, 2022 (lifted):", q2_rule_applies(2022, False))
    print("Dry set allocation, 2023 standard:", get_dry_set_allocation(2023, False))
    print("Dry set allocation, 2019 (should be None - needs FastF1 lookup):",
          get_dry_set_allocation(2019, False))
    print("Monaco two-stop required, 2025:", monaco_two_stop_required(2025, "Monaco_Grand_Prix"))
    print("Monaco two-stop required, 2026 (should be False - dropped):",
          monaco_two_stop_required(2026, "Monaco_Grand_Prix"))
    print("Regulation era, 2021 / 2023 / 2026:",
          get_regulation_era(2021), "/", get_regulation_era(2023), "/", get_regulation_era(2026))
