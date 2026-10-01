""

from dataclasses import dataclass
from typing import Optional

PIT_LANE_SPEED_LIMIT_KMH = {"default": 80, "Monaco_Grand_Prix": 60}
DOUBLE_STACK_PENALTY_SECONDS = 5.0

def get_pit_lane_speed_limit(circuit: str) -> int:
    return PIT_LANE_SPEED_LIMIT_KMH.get(circuit, PIT_LANE_SPEED_LIMIT_KMH["default"])

CLIFF_PROBABILITY_THRESHOLD = 0.017

INSTRUCTION_NEUTRAL_BAND_RATIO = 0.5

def get_driving_instruction(decision: str, cliff_probability_next_5_laps: Optional[float]) -> str:
    ""
def get_driving_instruction(decision: str, cliff_probability_next_5_laps: Optional[float],
                             tier3_reason: Optional[str] = None,
                             driver_stress_signal: bool = False) -> str:
    ""
    if decision == "PIT_NOW":
        return "PIT_LAP"

    if tier3_reason == "SC_GAMBLE":
        return "MANAGE"

    if driver_stress_signal:
        return "CONSERVE"

    if cliff_probability_next_5_laps is None:
        return "MANAGE"

    ratio = cliff_probability_next_5_laps / CLIFF_PROBABILITY_THRESHOLD
    if ratio >= 1.0:
        return "MANAGE"
    elif ratio >= INSTRUCTION_NEUTRAL_BAND_RATIO:
        return "NEUTRAL"
    else:
        return "PUSH"

@dataclass
class DriverPitContext:
    driver_id: str
    gate_tree_trigger_tier: int
    tyre_age: float
    track_position: int
    can_delay_one_lap_without_position_loss: bool
    cliff_probability_next_5_laps: Optional[float] = None
    tier3_reason: Optional[str] = None

    driver_stress_signal: bool = False

@dataclass
class ExecutionTreeState:
    triggered_drivers: list
    safety_car_active: bool
    circuit: str
    priority_driver_id: Optional[str] = None

def which_driver_triggered(state: ExecutionTreeState) -> str:
    n = len(state.triggered_drivers)
    if n == 1:
        return state.triggered_drivers[0].driver_id
    if n == 2:
        return "BOTH"
    raise ValueError("ExecutionTreeState must have 1 or 2 triggered drivers")

def resolve_priority(state: ExecutionTreeState) -> tuple:
    ""
    d1, d2 = state.triggered_drivers[0], state.triggered_drivers[1]

    if state.priority_driver_id is not None:
        priority = d1 if d1.driver_id == state.priority_driver_id else d2
        other = d2 if priority is d1 else d1
        return priority, other

    if d1.gate_tree_trigger_tier != d2.gate_tree_trigger_tier:
        priority, other = (d1, d2) if d1.gate_tree_trigger_tier < d2.gate_tree_trigger_tier else (d2, d1)
        return priority, other

    return d1, d2

def can_avoid_stacking(other_driver: DriverPitContext) -> bool:
    ""
    return other_driver.can_delay_one_lap_without_position_loss

def evaluate_double_stack(driver_a: DriverPitContext, driver_b: DriverPitContext,
                           safety_car_active: bool) -> dict:
    ""
    if safety_car_active:
        first, second = ((driver_a, driver_b) if driver_a.track_position > driver_b.track_position
                          else (driver_b, driver_a))
        return {"mode": "DOUBLE_STACK_BY_POSITION", "pits_first": first.driver_id,
                "pits_second": second.driver_id,
                "second_car_penalty_seconds": DOUBLE_STACK_PENALTY_SECONDS}
    else:

        first, second = (driver_a, driver_b) if driver_a.gate_tree_trigger_tier <= driver_b.gate_tree_trigger_tier else (driver_b, driver_a)
        return {"mode": "AVOID_DOUBLE_STACKING_STAGGER_1_LAP", "pits_first": first.driver_id,
                "pits_second_next_lap": second.driver_id, "second_car_penalty_seconds": 0.0}

def evaluate_execution_tree(state: ExecutionTreeState) -> dict:
    ""
    driver_lookup = {d.driver_id: d for d in state.triggered_drivers}

    def _attach_instruction(driver_id: str, decision_dict: dict) -> dict:
        driver_ctx = driver_lookup[driver_id]
        decision_dict["driving_instruction"] = get_driving_instruction(
            decision_dict["decision"], driver_ctx.cliff_probability_next_5_laps,
            tier3_reason=driver_ctx.tier3_reason,
            driver_stress_signal=driver_ctx.driver_stress_signal)
        return decision_dict

    trigger = which_driver_triggered(state)

    if trigger != "BOTH":
        d = state.triggered_drivers[0]
        result = {d.driver_id: {"decision": "PIT_NOW", "penalty_seconds": 0.0}}
        _attach_instruction(d.driver_id, result[d.driver_id])
        return result

    priority, other = resolve_priority(state)

    if can_avoid_stacking(other):
        result = {
            priority.driver_id: {"decision": "PIT_NOW", "penalty_seconds": 0.0},
            other.driver_id: {"decision": "PIT_LATER", "penalty_seconds": 0.0,
                               "reason": "delayed one lap, no position loss expected"},
        }
    else:
        outcome = evaluate_double_stack(priority, other, state.safety_car_active)
        if outcome["mode"] == "DOUBLE_STACK_BY_POSITION":
            result = {
                outcome["pits_first"]: {"decision": "PIT_NOW", "penalty_seconds": 0.0},
                outcome["pits_second"]: {"decision": "PIT_NOW",
                                           "penalty_seconds": outcome["second_car_penalty_seconds"]},
            }
        else:
            result = {
                outcome["pits_first"]: {"decision": "PIT_NOW", "penalty_seconds": 0.0},
                outcome["pits_second_next_lap"]: {"decision": "PIT_LATER", "penalty_seconds": 0.0,
                                                    "reason": "staggered by 1 lap to avoid same-lap double stack cost"},
            }

    for driver_id in result:
        _attach_instruction(driver_id, result[driver_id])
    return result

if __name__ == "__main__":
    print("=== Execution Tree scenarios (now with driving instructions) ===")

    d1_only = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 1, 22, 3, False, cliff_probability_next_5_laps=0.20)],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    print("Only D1 triggered (pitting -> PIT_LAP regardless of probability):",
          evaluate_execution_tree(d1_only))

    both_can_delay = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 2, 18, 2, False, cliff_probability_next_5_laps=0.01),
                            DriverPitContext("D2", 3, 15, 5, True, cliff_probability_next_5_laps=0.005)],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    print("Both triggered, D2 delayed with comfortable margin (-> PUSH):", evaluate_execution_tree(both_can_delay))

    both_cannot_delay_no_sc = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 1, 22, 2, False, cliff_probability_next_5_laps=0.10),
                            DriverPitContext("D2", 1, 21, 4, False, cliff_probability_next_5_laps=0.012)],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    print("Both triggered, D2 staggered right at the NEUTRAL band (-> NEUTRAL):",
          evaluate_execution_tree(both_cannot_delay_no_sc))

    both_cannot_delay_sc = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 2, 20, 2, False, cliff_probability_next_5_laps=0.03),
                            DriverPitContext("D2", 2, 19, 6, False, cliff_probability_next_5_laps=0.03)],
        safety_car_active=True, circuit="Monaco_Grand_Prix")
    print("Both triggered, SC active, both over threshold (-> MANAGE where delayed):",
          evaluate_execution_tree(both_cannot_delay_sc))

    unknown_risk = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 2, 18, 2, False, cliff_probability_next_5_laps=None),
                            DriverPitContext("D2", 3, 15, 5, True, cliff_probability_next_5_laps=None)],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    print("Unknown cliff probability (-> defaults to MANAGE, not a guess):", evaluate_execution_tree(unknown_risk))

    print("\nPit lane speed limit, Bahrain:", get_pit_lane_speed_limit("Bahrain_Grand_Prix"))
    print("Pit lane speed limit, Monaco:", get_pit_lane_speed_limit("Monaco_Grand_Prix"))

    sc_gamble_context = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 2, 18, 2, False, cliff_probability_next_5_laps=0.005),
                            DriverPitContext("D2", 3, 15, 5, True, cliff_probability_next_5_laps=0.001,
                                              tier3_reason="SC_GAMBLE")],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    print("\nD2 delayed specifically for an SC gamble (low cliff risk, but reason overrides -> MANAGE):",
          evaluate_execution_tree(sc_gamble_context))

    driver_stress_context = ExecutionTreeState(
        triggered_drivers=[DriverPitContext("D1", 2, 18, 2, False, cliff_probability_next_5_laps=0.005),
                            DriverPitContext("D2", 3, 15, 5, True, cliff_probability_next_5_laps=0.001,
                                              driver_stress_signal=True)],
        safety_car_active=False, circuit="Bahrain_Grand_Prix")
    print("D2 delayed with low cliff risk, but reported driver stress (-> CONSERVE):",
          evaluate_execution_tree(driver_stress_context))

    print("\n=== Closing the loop: next-lap projection call (illustrative only) ===")
    print("A driver told PIT_LATER this lap has tyre_age incremented by 1 next lap, then:")
    print("  tyre_life_projection.build_tyre_life_projection(..., tyre_age=tyre_age + 1, ...)")
    print("  -> feeds fresh predicted_pace_loss / cliff_probability_next_5_laps back into")
    print("     Gate Tree Tier 1 (cliff check) and Tier 3 (soft triggers) for the next lap's decision.")
