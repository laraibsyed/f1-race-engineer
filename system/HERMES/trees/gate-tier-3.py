""

from dataclasses import dataclass
from typing import Optional

CLIFF_PROBABILITY_THRESHOLD = 0.017

CLIFF_PROXIMITY_MIN_TYRE_AGE = 5

PACE_LOSS_THRESHOLD_SECONDS = 1.805

TYRE_AGE_TRIGGER_RATIO = 0.421

@dataclass
class Tier3State:

    cliff_probability_next_5_laps: Optional[float]
    predicted_pace_loss: Optional[float]
    tyre_age: float
    expected_stint_length: float

    undercut_opportunity: bool
    overcut_opportunity: bool
    safety_car_deployed: bool

    rival_undercut_threat: bool
    in_dirty_air: bool
    driver_stress_signal: bool = False

    sc_gamble_recommendation: Optional[str] = None

    drying_crossover_opportunity: bool = False

TYRE_ONLY_TRIGGER_NAMES = {"cliff_proximity", "pace_lap_delta", "tyre_age"}

def _apply_sc_gamble_suppression(triggers: dict, state: Tier3State) -> dict:
    ""
    if state.sc_gamble_recommendation != "WAIT":
        return triggers
    suppressed = dict(triggers)
    for name in TYRE_ONLY_TRIGGER_NAMES:
        if suppressed.get(name):
            suppressed[name] = False
    return suppressed

def _cliff_proximity_trigger(state: Tier3State) -> bool:
    ""
    if state.cliff_probability_next_5_laps is None:
        return False
    if state.tyre_age < CLIFF_PROXIMITY_MIN_TYRE_AGE:
        return False
    return state.cliff_probability_next_5_laps >= CLIFF_PROBABILITY_THRESHOLD

def _pace_loss_trigger(state: Tier3State) -> bool:
    if state.predicted_pace_loss is None:
        return False
    return state.predicted_pace_loss >= PACE_LOSS_THRESHOLD_SECONDS

def _tyre_age_trigger(state: Tier3State) -> bool:
    if state.expected_stint_length <= 0:
        return False
    return state.tyre_age >= (TYRE_AGE_TRIGGER_RATIO * state.expected_stint_length)

def count_active_triggers(state: Tier3State) -> dict:
    ""
    raw_triggers = {
        "cliff_proximity": _cliff_proximity_trigger(state),
        "pace_lap_delta": _pace_loss_trigger(state),
        "tyre_age": _tyre_age_trigger(state),
        "undercut": state.undercut_opportunity,
        "overcut": state.overcut_opportunity,
        "safety_car": state.safety_car_deployed,
        "rival_undercut_threat": state.rival_undercut_threat,
        "dirty_air": state.in_dirty_air,
        "drying_crossover": state.drying_crossover_opportunity,

    }
    return _apply_sc_gamble_suppression(raw_triggers, state)

def _tyre_triggers_were_suppressed(state: Tier3State) -> bool:
    ""
    if state.sc_gamble_recommendation != "WAIT":
        return False
    return _cliff_proximity_trigger(state) or _pace_loss_trigger(state) or _tyre_age_trigger(state)

def evaluate_tier3(state: Tier3State) -> dict:
    ""
    triggers = count_active_triggers(state)
    n_active = sum(triggers.values())

    if n_active >= 3:
        decision = "PIT_NOW"
    elif n_active >= 1:
        decision = "PIT_LATER"
    else:
        decision = "DONT_PIT"

    if _tyre_triggers_were_suppressed(state):
        reason = "SC_GAMBLE"
    elif decision == "PIT_LATER":
        reason = "SOFT_TRIGGERS"
    else:
        reason = None

    return {
        "decision": decision,
        "reason": reason,
        "triggers": triggers,
        "driver_stress_signal": state.driver_stress_signal,
    }

if __name__ == "__main__":
    print("=== Gate Tree Tier 3 decision sequence ===")
    print("(evaluate_tier3 now returns a dict: decision/reason/triggers/driver_stress_signal)")

    s1 = Tier3State(cliff_probability_next_5_laps=0.005, predicted_pace_loss=0.2,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("\nNothing active:", evaluate_tier3(s1))

    s2 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("One trigger (cliff proximity):", evaluate_tier3(s2))

    s3 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=True, rival_undercut_threat=False, in_dirty_air=False)
    print("Three triggers:", evaluate_tier3(s3))

    s4 = Tier3State(cliff_probability_next_5_laps=None, predicted_pace_loss=None,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("Unknown tyre-model output (None):", evaluate_tier3(s4))

    s5 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=True, rival_undercut_threat=False, in_dirty_air=False,
                     sc_gamble_recommendation="WAIT")
    print("Same as Scenario 3, but SC gamble says WAIT:", evaluate_tier3(s5))
    print("  (decision changed from PIT_NOW to PIT_LATER, AND reason=SC_GAMBLE is now visible to Execution Tree)")

    s6 = Tier3State(cliff_probability_next_5_laps=0.005, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False,
                     driver_stress_signal=True)
    result6 = evaluate_tier3(s6)
    print("\nDriver stress + one tyre trigger:", result6)
    print("  (OLD design would have counted stress as a 2nd vote -> PIT_NOW. FIXED: stress isn't")
    print("   a vote -> PIT_LATER, but driver_stress_signal=True is still passed through as context)")

    s7_before_floor_would_trigger = Tier3State(
        cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
        tyre_age=3, expected_stint_length=25,
        undercut_opportunity=True, overcut_opportunity=False,
        safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=True)
    result7 = evaluate_tier3(s7_before_floor_would_trigger)
    print(f"\ntyre_age=3, cliff_probability=0.05 (>= {CLIFF_PROBABILITY_THRESHOLD}), "
          f"+2 external triggers already active:", result7)
    print(f"  (cliff_proximity is False despite cliff_probability clearing the threshold, because "
          f"tyre_age={3} < CLIFF_PROXIMITY_MIN_TYRE_AGE={CLIFF_PROXIMITY_MIN_TYRE_AGE} - "
          f"only 2 triggers active -> PIT_LATER, not PIT_NOW. Before this update, this exact "
          f"case would have been PIT_NOW.)")
