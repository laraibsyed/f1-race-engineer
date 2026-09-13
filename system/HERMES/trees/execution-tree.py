"""
Execution Tree
==================
Fires when the Gate Tree returns PIT_NOW or PIT_FLEXIBLE. Handles the
MECHANICAL side of pit execution - who physically pits when, double-stack
math, pit lane speed limits - not deep strategic questions (that's Second
Driver Logic's job, not yet built; this tree accepts an optional priority
override from it and falls back to a documented default otherwise).

Grounded in MDP doc paragraph 130 (Pit Execution Constraints):
  - Only one pit box per team
  - Double-stack penalty ≈5 seconds, applied to the SECOND car's pit cost
  - Unsafe releases are a penalty trigger (transition probability layer,
    not modelled here - this tree handles the decision, not the outcome
    probability of a bad stop)
  - Pit lane speed limit: 80km/h standard, 60km/h at Monaco

Action space alignment (MDP paragraphs 55-63): "Driver 1 Priority" is
explicitly named as the DEFAULT team-related state - used here as the
tie-break when no other signal distinguishes urgency. "Pit Both" / "Pit One"
/ "Stay Out" (Race Interruption Related actions) map to this tree's outputs.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Pit lane speed limits (MDP para 130) - used for pit delta calculations
# ---------------------------------------------------------------------------
PIT_LANE_SPEED_LIMIT_KMH = {"default": 80, "Monaco_Grand_Prix": 60}
DOUBLE_STACK_PENALTY_SECONDS = 5.0  # MDP para 130: "approximately five seconds"


def get_pit_lane_speed_limit(circuit: str) -> int:
    return PIT_LANE_SPEED_LIMIT_KMH.get(circuit, PIT_LANE_SPEED_LIMIT_KMH["default"])


# ---------------------------------------------------------------------------
# Driving Instruction - Push/Manage/Pit Lap
# ---------------------------------------------------------------------------
CLIFF_PROBABILITY_THRESHOLD = 0.017  # MUST stay in sync with gate_tree_tier3.py's calibrated
                                       # value - duplicated here deliberately (same pattern as
                                       # the rest of this project) so this file is callable
                                       # standalone without importing the whole Gate Tree.

# Diagram box only names two non-pitting options ("Push"/"Manage"), but the MDP's full pace
# action space (para 64-68) has four levels (Push/Neutral/Conserve/Manage Degradation). Using
# three graduated bands here for a genuinely graduated response, as requested - "Manage" in the
# diagram maps to both NEUTRAL and MANAGE in this finer split. The 0.5x split point is an
# ASSUMPTION, flagged, not derived from data - a candidate for its own sensitivity check later.
INSTRUCTION_NEUTRAL_BAND_RATIO = 0.5  # ASSUMPTION - below this fraction of threshold = PUSH


def get_driving_instruction(decision: str, cliff_probability_next_5_laps: Optional[float]) -> str:
    """
    Graduated response based on how close cliff_probability_next_5_laps is to
    the calibrated CLIFF_PROBABILITY_THRESHOLD - the closer, the more
    conservative, exactly as requested rather than a flat rule.
    """
    if decision == "PIT_NOW":
        return "PIT_LAP"  # boxing this lap - no further pace guidance needed

    if cliff_probability_next_5_laps is None:
        return "MANAGE"  # unknown risk - default to conservative, don't push blind

    ratio = cliff_probability_next_5_laps / CLIFF_PROBABILITY_THRESHOLD
    if ratio >= 1.0:
        return "MANAGE"       # at or beyond the calibrated risk threshold
    elif ratio >= INSTRUCTION_NEUTRAL_BAND_RATIO:
        return "NEUTRAL"      # approaching - moderate caution, MDP's own default pace state
    else:
        return "PUSH"         # comfortable margin below threshold


# ---------------------------------------------------------------------------
# Per-driver pit context - what the Execution Tree needs to know about each
# driver who triggered a pit recommendation this lap
# ---------------------------------------------------------------------------
@dataclass
class DriverPitContext:
    driver_id: str                                  # "D1" or "D2"
    gate_tree_trigger_tier: int                      # 1, 2, or 3 - which Gate Tree tier fired (lower = more urgent)
    tyre_age: float
    track_position: int                              # 1 = leading; higher number = further back
    can_delay_one_lap_without_position_loss: bool     # EXTERNAL - from gap/rival analysis, not computed here
    cliff_probability_next_5_laps: Optional[float] = None  # from tyre_life_projection feed - drives driving instruction


@dataclass
class ExecutionTreeState:
    triggered_drivers: list          # 1 or 2 DriverPitContext objects
    safety_car_active: bool
    circuit: str
    priority_driver_id: Optional[str] = None   # override from Second Driver Logic; None = use default resolution


# ---------------------------------------------------------------------------
# Step 1: Which driver(s) triggered?
# ---------------------------------------------------------------------------
def which_driver_triggered(state: ExecutionTreeState) -> str:
    n = len(state.triggered_drivers)
    if n == 1:
        return state.triggered_drivers[0].driver_id
    if n == 2:
        return "BOTH"
    raise ValueError("ExecutionTreeState must have 1 or 2 triggered drivers")


# ---------------------------------------------------------------------------
# Step 2 (only when BOTH triggered): resolve priority
# ---------------------------------------------------------------------------
def resolve_priority(state: ExecutionTreeState) -> tuple:
    """
    Returns (priority_driver, other_driver) - priority pits first / on
    schedule, other is the candidate for delay.

    Resolution order:
      1. Explicit override from Second Driver Logic, if provided.
      2. Gate Tree tier urgency - a driver who triggered Tier 1 (hard safety
         gate) takes priority over one who only triggered Tier 2/3.
      3. "Driver 1 Priority" - the MDP's own stated default state (para 56) -
         used as the final tie-break.
    """
    d1, d2 = state.triggered_drivers[0], state.triggered_drivers[1]

    if state.priority_driver_id is not None:
        priority = d1 if d1.driver_id == state.priority_driver_id else d2
        other = d2 if priority is d1 else d1
        return priority, other

    if d1.gate_tree_trigger_tier != d2.gate_tree_trigger_tier:
        priority, other = (d1, d2) if d1.gate_tree_trigger_tier < d2.gate_tree_trigger_tier else (d2, d1)
        return priority, other

    # Tied urgency - fall back to MDP's stated default (Driver 1 Priority)
    return d1, d2


# ---------------------------------------------------------------------------
# Step 3 (only when BOTH triggered): can stacking be avoided?
# ---------------------------------------------------------------------------
def can_avoid_stacking(other_driver: DriverPitContext) -> bool:
    """True if the non-priority driver can delay one lap without losing
    position - an EXTERNAL signal (gap/rival analysis), not computed here."""
    return other_driver.can_delay_one_lap_without_position_loss


# ---------------------------------------------------------------------------
# Step 4 (only when stacking can't be avoided): SC-conditioned outcome
# ---------------------------------------------------------------------------
def evaluate_double_stack(driver_a: DriverPitContext, driver_b: DriverPitContext,
                           safety_car_active: bool) -> dict:
    """
    Returns which driver pits first and the pit-cost penalty applied to the
    second, under the two documented outcomes:

      - SC active: pit cost is already reduced circuit-wide (MDP para 129),
        so a genuine double stack is more affordable. ASSUMPTION, flagged:
        the driver in the WORSE track position (further back) pits first,
        protecting the lead car's track position - a defensible convention,
        not the only possible one.
      - No SC: stacking is costly, so laps are staggered by the minimum
        possible gap (1 lap) rather than a genuine simultaneous double stack,
        even though "can't wait" was already established - this reduces but
        doesn't eliminate the cost, since a full lap's delay is still
        cheaper than a same-lap double stack outside SC conditions.
    """
    if safety_car_active:
        first, second = ((driver_a, driver_b) if driver_a.track_position > driver_b.track_position
                          else (driver_b, driver_a))
        return {"mode": "DOUBLE_STACK_BY_POSITION", "pits_first": first.driver_id,
                "pits_second": second.driver_id,
                "second_car_penalty_seconds": DOUBLE_STACK_PENALTY_SECONDS}
    else:
        # Stagger by 1 lap rather than genuinely double-stack - the driver
        # with the more urgent tier still pits THIS lap, the other next lap
        first, second = (driver_a, driver_b) if driver_a.gate_tree_trigger_tier <= driver_b.gate_tree_trigger_tier else (driver_b, driver_a)
        return {"mode": "AVOID_DOUBLE_STACKING_STAGGER_1_LAP", "pits_first": first.driver_id,
                "pits_second_next_lap": second.driver_id, "second_car_penalty_seconds": 0.0}


# ---------------------------------------------------------------------------
# Full Execution Tree evaluation
# ---------------------------------------------------------------------------
def evaluate_execution_tree(state: ExecutionTreeState) -> dict:
    """
    Returns a per-driver decision dict, matching the diagram's outputs:
    each driver gets a decision (PIT_NOW/PIT_LATER), a pit-cost penalty if
    double-stacked, and a driving_instruction (PIT_LAP/MANAGE/NEUTRAL/PUSH)
    based on their own cliff_probability_next_5_laps.
    """
    driver_lookup = {d.driver_id: d for d in state.triggered_drivers}

    def _attach_instruction(driver_id: str, decision_dict: dict) -> dict:
        cliff_prob = driver_lookup[driver_id].cliff_probability_next_5_laps
        decision_dict["driving_instruction"] = get_driving_instruction(
            decision_dict["decision"], cliff_prob)
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

    # --- Closing the loop: "Tyre Projection Updated - Fed to Gate Tree" ---
    # This is a structural demonstration, not new modelling - it shows that
    # after this lap's decision, tyre_age increments and the SAME projection
    # feed from tyre_life_projection.py gets called again next lap, closing
    # the loop the diagram draws. A real system calls this every lap; this
    # just proves the data flow connects correctly.
    print("\n=== Closing the loop: next-lap projection call (illustrative only) ===")
    print("A driver told PIT_LATER this lap has tyre_age incremented by 1 next lap, then:")
    print("  tyre_life_projection.build_tyre_life_projection(..., tyre_age=tyre_age + 1, ...)")
    print("  -> feeds fresh predicted_pace_loss / cliff_probability_next_5_laps back into")
    print("     Gate Tree Tier 1 (cliff check) and Tier 3 (soft triggers) for the next lap's decision.")