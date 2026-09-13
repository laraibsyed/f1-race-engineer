"""
Gate Tree — Tier 3: Soft Constraints (Reward Modifiers and Action Filters)
==============================================================================
MDP doc paragraphs 127-128: these don't make actions illegal (that's Tier 2)
or force an action (that's Tier 1) - they make actions costly, lower
priority, or contextually inappropriate.

The diagram's sticky note names nine soft triggers: undercut, overcut, SC,
rival, dirty air, pace, tyre age, lap delta, cliff proximity. Of these, only
THREE are things a tyre model can actually produce:

  - cliff proximity   <- cliff_probability_next_5_laps, from the projection
                         feed (tyre_life_projection.py, which wraps Cox)
  - pace / lap delta   <- predicted_pace_loss, from the same feed (wraps
                         Regression V2)
  - tyre age           <- a simple threshold on tyre_age itself

The remaining six (undercut, overcut, SC, rival, dirty air) are RACE-CONTEXT
signals - gaps to other cars, safety car state, DRS/dirty-air detection.
Nothing in the tyre model produces these; they're external inputs here,
same treatment as Tier 1's red-flag/damage/weather signals.

ALL THREE THRESHOLDS BELOW WERE ORIGINALLY GUESSED PLACEHOLDERS - now calibrated
against real historical data via tier3_threshold_calibration.py (same treatment
DEADLINE_BUFFER_LAPS got). See that script and the comments below each constant
for exactly how each number was derived and what it changed from.
"""

from dataclasses import dataclass
from typing import Optional

CLIFF_PROBABILITY_THRESHOLD = 0.017  # CALIBRATED via ROC validation against 533 real cliff
                                       # events (tier3_threshold_calibration.py Part 3).
                                       # AUC=0.683, Youden-optimal point (TPR=0.650, FPR=0.395).
                                       # Deliberately kept at this permissive value rather than
                                       # a stricter point on the ROC curve - the 40% false-alarm
                                       # rate is acceptable here BECAUSE this is one of several
                                       # triggers that must combine (3+) to force PIT_NOW, not a
                                       # standalone hard gate. Was 0.30 (guessed) - the real value
                                       # is ~17x lower because cliffs are rare overall, so even
                                       # genuinely elevated risk rarely produces a "large" absolute
                                       # probability from this model.
PACE_LOSS_THRESHOLD_SECONDS = 1.805   # CALIBRATED: 90th percentile of real pace_loss_seconds
                                       # across 148,041 cleaned dry-compound laps (same cleaning
                                       # pipeline as Regression V2). Was 1.0 (guessed) - close in
                                       # magnitude, this one didn't need much correction.
TYRE_AGE_TRIGGER_RATIO = 0.538        # CALIBRATED: 10th percentile of cliff_tyre_age/n_laps
                                       # across 611 real cliff events - catches 90% of historical
                                       # cliffs with this much warning. Was 0.80 (guessed).
                                       # CAVEAT: n_laps undercounts true stint length (some real
                                       # stints showed ratios >1, which is impossible if n_laps
                                       # were the true denominator) - this number is directionally
                                       # right but not fully precise until cliff_detection.py
                                       # tracks true stint length separately from post-cleaning
                                       # row count.


@dataclass
class Tier3State:
    # --- tyre-model-derived (from tyre_life_projection.build_tyre_life_projection) ---
    cliff_probability_next_5_laps: Optional[float]
    predicted_pace_loss: Optional[float]
    tyre_age: float
    expected_stint_length: float   # from strategy plan, not the tyre model itself

    # --- EXTERNAL race-context signals - NOT tyre-model-derived ---
    undercut_opportunity: bool
    overcut_opportunity: bool
    safety_car_deployed: bool
    rival_undercut_threat: bool
    in_dirty_air: bool


def _cliff_proximity_trigger(state: Tier3State) -> bool:
    if state.cliff_probability_next_5_laps is None:
        return False  # unknown compound/circuit/era - can't evaluate, don't false-trigger
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
    """Returns each trigger's boolean state individually - useful for logging
    which specific triggers fired, not just the count."""
    return {
        "cliff_proximity": _cliff_proximity_trigger(state),
        "pace_lap_delta": _pace_loss_trigger(state),
        "tyre_age": _tyre_age_trigger(state),
        "undercut": state.undercut_opportunity,
        "overcut": state.overcut_opportunity,
        "safety_car": state.safety_car_deployed,
        "rival_undercut_threat": state.rival_undercut_threat,
        "dirty_air": state.in_dirty_air,
    }


def evaluate_tier3(state: Tier3State) -> str:
    """
    Returns "PIT_NOW" (Multiple triggers) or "MONITOR_STAY_OUT" (One/Two) -
    matches the Gate Tree diagram's Tier 3 branch exactly. "Multiple" is
    interpreted as 3+, since the diagram explicitly separates "One / Two"
    from "Multiple" as distinct branches.
    """
    triggers = count_active_triggers(state)
    n_active = sum(triggers.values())
    return "PIT_NOW" if n_active >= 3 else "MONITOR_STAY_OUT"


if __name__ == "__main__":
    print("=== Gate Tree Tier 3 decision sequence ===")

    # Scenario 1: nothing active (probabilities genuinely below the calibrated
    # threshold - note this threshold is now much smaller than before, so
    # "safe" values need to be smaller too)
    s1 = Tier3State(cliff_probability_next_5_laps=0.005, predicted_pace_loss=0.2,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("Nothing active:", evaluate_tier3(s1), count_active_triggers(s1))

    # Scenario 2: one tyre-model trigger only (cliff probability above the calibrated threshold)
    s2 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("One trigger (cliff proximity):", evaluate_tier3(s2), count_active_triggers(s2))

    # Scenario 3: three triggers active (cliff proximity + tyre age + SC) -> PIT_NOW
    s3 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=True, rival_undercut_threat=False, in_dirty_air=False)
    print("Three triggers:", evaluate_tier3(s3), count_active_triggers(s3))

    # Scenario 4: unknown tyre-model output (None) shouldn't false-trigger
    s4 = Tier3State(cliff_probability_next_5_laps=None, predicted_pace_loss=None,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("Unknown tyre-model output (None):", evaluate_tier3(s4), count_active_triggers(s4))