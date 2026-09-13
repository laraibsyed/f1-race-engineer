"""
Gate Tree — Tier 2: Regulatory Constraints (Feasibility Gates)
==================================================================
Implements the MDP's Tier 2 constraint system (Constraints section, MDP doc
paragraphs 117-126). These make actions LEGALLY INFEASIBLE - filtered before
the reward function ever sees them, not penalised or weighed against context
(that's Tier 3's job).

The Gate Tree diagram only draws two Tier 2 decision nodes ("Mandatory
Compound Change Done?" / "Deadline Approaching This Stint?"), but the MDP doc
names SIX real constraints. This module implements all six - the two diagram
nodes are the visible decision path (evaluate_tier2, at the bottom), but they
depend on the full rule set to actually evaluate correctly.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Regulation Year Indexing (MDP para 125)
# ---------------------------------------------------------------------------
def get_regulation_era(season: int) -> str:
    """
    2022 and 2026 are reset years - pre-2022 degradation regression outputs
    must not be used for 2022+ simulations, pre-2026 outputs must not be used
    for 2026+. Matches the regulation_era stratification already built into
    tyre_regression_v2.py / cliff_detection.py / survival_model_v2.py exactly -
    this function is the single source of truth those should all agree with.
    """
    if season <= 2021:
        return "2018-2021"
    elif season <= 2025:
        return "2022-2025"
    else:
        return "2026+"


def check_degradation_model_compatible(model_era: str, race_season: int) -> bool:
    """Enforces MDP para 125's rule isn't accidentally violated elsewhere."""
    return model_era == get_regulation_era(race_season)


# ---------------------------------------------------------------------------
# Tyre Set Allocation (MDP para 121, 123)
# ---------------------------------------------------------------------------
# 2020+: fixed by Pirelli, identical for all teams. 2018/2019: teams chose
# their own breakdown - NOT hardcoded here, must be loaded from actual FastF1
# session data. get_dry_set_allocation() returns None for those years
# deliberately, rather than guessing a plausible-looking number.
STANDARD_WEEKEND_ALLOCATION = 13   # 2020+ only, MDP para 121/123
SPRINT_WEEKEND_ALLOCATION = 12     # 2020+ only, MDP para 123
YEARS_REQUIRING_FASTF1_ALLOCATION_LOOKUP = {2018, 2019}

# From 2024: sprint shootout consumes sets before race allocation is set -
# one new MEDIUM in SQ1/SQ2, one new SOFT in SQ3 (MDP para 123)
SPRINT_SHOOTOUT_CONSUMPTION_YEAR_START = 2024
SPRINT_SHOOTOUT_CONSUMED_SETS = {"MEDIUM": 1, "SOFT": 1}


def get_dry_set_allocation(season: int, is_sprint_weekend: bool) -> Optional[int]:
    """Returns None for 2018/2019 - caller MUST pull the real number from
    FastF1 session data for those years, not assume one."""
    if season in YEARS_REQUIRING_FASTF1_ALLOCATION_LOOKUP:
        return None
    return SPRINT_WEEKEND_ALLOCATION if is_sprint_weekend else STANDARD_WEEKEND_ALLOCATION


def apply_sprint_shootout_consumption(season: int, remaining_sets: dict) -> dict:
    """MDP para 123 - deduct SQ-session-consumed sets before race strategy evaluation."""
    if season < SPRINT_SHOOTOUT_CONSUMPTION_YEAR_START:
        return remaining_sets
    adjusted = dict(remaining_sets)
    for compound, n in SPRINT_SHOOTOUT_CONSUMED_SETS.items():
        adjusted[compound] = max(0, adjusted.get(compound, 0) - n)
    return adjusted


def compound_available(remaining_sets: dict, compound: str) -> bool:
    """MDP para 121: model cannot recommend a compound with no sets remaining."""
    return remaining_sets.get(compound, 0) > 0


# ---------------------------------------------------------------------------
# Q2 Tyre Rule (MDP para 122)
# ---------------------------------------------------------------------------
Q2_RULE_ACTIVE_YEARS = {2018, 2019, 2020, 2021}


def q2_rule_applies(season: int, is_sprint_weekend: bool) -> bool:
    """Active 2018-2021, standard weekends only - lifted from 2022. Does NOT
    apply on sprint weekends in 2021 specifically (MDP para 122)."""
    if season not in Q2_RULE_ACTIVE_YEARS:
        return False
    if season == 2021 and is_sprint_weekend:
        return False
    return True


def get_mandatory_start_compound(season: int, is_sprint_weekend: bool,
                                  progressed_through_q2: bool,
                                  q2_fastest_lap_compound: Optional[str]) -> Optional[str]:
    """Returns the compound the driver MUST start the race on, or None if unconstrained."""
    if not q2_rule_applies(season, is_sprint_weekend):
        return None
    if not progressed_through_q2:
        return None  # eliminated in Q1 - free compound choice
    return q2_fastest_lap_compound


# ---------------------------------------------------------------------------
# Monaco Two-Stop Rule (MDP para 124) - 2025 ONLY
# ---------------------------------------------------------------------------
MONACO_TWO_STOP_YEARS = {2025}


def monaco_two_stop_required(season: int, circuit: str) -> bool:
    return season in MONACO_TWO_STOP_YEARS and circuit == "Monaco_Grand_Prix"


def monaco_two_stop_satisfied(season: int, circuit: str, sets_used: int, stops_made: int) -> bool:
    """MDP para 124: minimum 3 sets used AND minimum 2 pit stops, 2025 only."""
    if not monaco_two_stop_required(season, circuit):
        return True  # not applicable, trivially satisfied
    return sets_used >= 3 and stops_made >= 2


# ---------------------------------------------------------------------------
# Mandatory Compound Rule (MDP para 120) - Gate Tree decision node #1
# ---------------------------------------------------------------------------
def mandatory_compound_done(compound_history_dry: set, wet_race_exception: bool) -> bool:
    """
    At least 2 different DRY compounds used, unless wet exception applies.
    compound_history_dry must contain ONLY dry compounds already used this
    race - exclude INTERMEDIATE/WET before calling this. Applies across all
    years 2018-2026 (unlike allocation/Q2/Monaco rules, this one is constant).
    """
    if wet_race_exception:
        return True
    return len(compound_history_dry) >= 2


def mandatory_compound_reset_on_red_flag() -> bool:
    """
    MDP para 129: under red flag, gaps reset and a free tyre change is
    permitted, but compound history is PRESERVED - the mandatory obligation
    is NOT reset. Always returns False - exists so nothing downstream
    accidentally resets compound_history_dry on a red flag event.
    """
    return False


# ---------------------------------------------------------------------------
# "Deadline Approaching This Stint?" - Gate Tree decision node #2.
#
# NOT given a numeric threshold anywhere in the MDP doc - this is an
# ASSUMPTION, flagged explicitly rather than silently guessed. Interpretation
# used here: if the mandatory compound rule isn't yet satisfied and there
# aren't enough laps left to safely execute a further stint on a different
# compound, the team must act now rather than wait for a Tier 3 opportunity.
# ---------------------------------------------------------------------------
DEADLINE_BUFFER_LAPS = 5  # ASSUMPTION - confirm/tune this against your own spec


def deadline_approaching(laps_remaining_in_race: int, mandatory_done_flag: bool,
                          buffer_laps: int = DEADLINE_BUFFER_LAPS) -> bool:
    """
    True only if the mandatory compound swap hasn't happened AND fewer than
    `buffer_laps` remain to execute it safely. If the rule is already
    satisfied, there's no deadline to approach - always False in that case.
    """
    if mandatory_done_flag:
        return False
    return laps_remaining_in_race <= buffer_laps


# ---------------------------------------------------------------------------
# Gate Tree Tier 2 - the actual decision sequence drawn in the diagram
# ---------------------------------------------------------------------------
@dataclass
class Tier2State:
    season: int
    circuit: str
    is_sprint_weekend: bool
    compound_history_dry: set          # dry compounds already used this race
    wet_race_exception: bool
    laps_remaining_in_race: int
    remaining_sets: dict                # {"SOFT": n, "MEDIUM": n, "HARD": n, ...}
    sets_used_so_far: int
    stops_made_so_far: int


def evaluate_tier2(state: Tier2State) -> str:
    """
    Returns "MOVE_TO_TIER_3" or "PIT_FLEXIBLE" - matches the Gate Tree
    diagram's two Tier 2 outputs exactly.
    """
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