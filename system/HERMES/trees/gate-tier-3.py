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
TYRE_AGE_TRIGGER_RATIO = 0.421        # CALIBRATED (corrected): 10th percentile of
                                       # cliff_tyre_age/n_laps_true across 611 real cliff
                                       # events - catches 90% of historical cliffs with this
                                       # much warning. Was 0.80 (guessed), then 0.538
                                       # (calibrated but biased - see below), now 0.421 (final).
                                       # The n_laps undercounting bug (max ratio was previously
                                       # an impossible 1.857) has been FIXED in cliff_detection.py
                                       # - true stint length is now computed from raw tyre_age
                                       # before any cleaning filters run, confirmed by max ratio
                                       # dropping to a valid 0.927 after the fix.


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
    safety_car_deployed: bool          # SC ALREADY ACTIVE now - different from sc_gamble below,
                                        # which is about a PREDICTED future SC
    rival_undercut_threat: bool
    in_dirty_air: bool
    driver_stress_signal: bool = False   # from nlp_driver_stress.get_driver_stress_trigger() -
                                          # medium/high stress tyre_feedback radio message within
                                          # the lookback window. Default False so existing callers
                                          # that don't yet wire this in keep working unchanged.
    sc_gamble_recommendation: Optional[str] = None  # from sc_gamble_evaluator.evaluate_sc_gamble()
                                                     # - "WAIT", "NO_ADVANTAGE_TO_WAITING", or None/
                                                     # "INSUFFICIENT_DATA". NOT a trigger itself -
                                                     # see _apply_sc_gamble_suppression() below.


TYRE_ONLY_TRIGGER_NAMES = {"cliff_proximity", "pace_lap_delta", "tyre_age"}


def _apply_sc_gamble_suppression(triggers: dict, state: Tier3State) -> dict:
    """
    If the expected-cost analysis recommends WAITING for a potential cheap SC
    pit, suppress the TYRE-ONLY triggers specifically (cliff_proximity,
    pace_lap_delta, tyre_age) - these are exactly the triggers that would
    otherwise push toward pitting purely for tyre reasons, which is the exact
    decision the gamble evaluator already weighed tyre cost INTO and found
    outweighed by the SC opportunity. External/urgent triggers (undercut, an
    SC that's ALREADY active, rival threat, dirty air, driver stress) are
    untouched - they aren't about "should I wait for a hypothetical SC" and
    the gamble evaluator never considered them.
    """
    if state.sc_gamble_recommendation != "WAIT":
        return triggers
    suppressed = dict(triggers)
    for name in TYRE_ONLY_TRIGGER_NAMES:
        if suppressed.get(name):
            suppressed[name] = False
    return suppressed


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
    """
    Returns each STRATEGIC trigger's boolean state - used to decide WHETHER
    to pit. driver_stress_signal is deliberately NOT included here anymore -
    architectural correction: stress is context for HOW the Execution Tree
    should shape driving instructions afterward, not a vote on WHETHER to
    pit. This also lines up with the NLP validation finding (no significant
    correlation between stress level and pit timing) - it was always shaky
    to vote with a signal that doesn't actually predict pit timing.
    Tyre-only triggers are suppressed here if the SC gamble evaluator
    recommends waiting - see _apply_sc_gamble_suppression().
    """
    raw_triggers = {
        "cliff_proximity": _cliff_proximity_trigger(state),
        "pace_lap_delta": _pace_loss_trigger(state),
        "tyre_age": _tyre_age_trigger(state),
        "undercut": state.undercut_opportunity,
        "overcut": state.overcut_opportunity,
        "safety_car": state.safety_car_deployed,
        "rival_undercut_threat": state.rival_undercut_threat,
        "dirty_air": state.in_dirty_air,
    }
    return _apply_sc_gamble_suppression(raw_triggers, state)


def _tyre_triggers_were_suppressed(state: Tier3State) -> bool:
    """True only if the SC gamble recommendation actually changed the
    outcome - i.e. at least one tyre-only trigger was genuinely active
    before suppression. Distinguishes "we're waiting BECAUSE of the gamble"
    from "the gamble said wait but there was nothing to suppress anyway"."""
    if state.sc_gamble_recommendation != "WAIT":
        return False
    return _cliff_proximity_trigger(state) or _pace_loss_trigger(state) or _tyre_age_trigger(state)


def evaluate_tier3(state: Tier3State) -> dict:
    """
    Returns a STRUCTURED result, not just a decision string - the Execution
    Tree needs the REASON as context (was this an SC gamble?) plus the raw
    driver_stress_signal passed through untouched (context, not a vote - see
    count_active_triggers docstring).

    decision: "PIT_NOW" (3+ triggers) / "PIT_LATER" (1-2) / "DONT_PIT" (0) -
    matches the Gate Tree diagram's tiered branching, now split into three
    outcomes instead of two so "genuinely nothing pending" (DONT_PIT) is
    distinguishable from "watching something, not urgent yet" (PIT_LATER).
    reason: "SC_GAMBLE" if waiting is specifically to catch a cheap SC pit,
    "SOFT_TRIGGERS" if PIT_LATER for ordinary accumulating reasons, None
    for PIT_NOW or DONT_PIT.

    ARCHITECTURAL PRINCIPLE: Tier 3 decides WHAT to do, using only objective
    model/race signals. The Execution Tree decides HOW to behave while doing
    it, using the reason and driver_stress_signal as context - it does NOT
    independently re-evaluate the SC probability or stress signal itself.
    """
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
        "driver_stress_signal": state.driver_stress_signal,  # CONTEXT for Execution Tree, NOT a vote
    }


if __name__ == "__main__":
    print("=== Gate Tree Tier 3 decision sequence ===")
    print("(evaluate_tier3 now returns a dict: decision/reason/triggers/driver_stress_signal)")

    # Scenario 1: nothing active (probabilities genuinely below the calibrated
    # threshold - note this threshold is now much smaller than before, so
    # "safe" values need to be smaller too)
    s1 = Tier3State(cliff_probability_next_5_laps=0.005, predicted_pace_loss=0.2,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("\nNothing active:", evaluate_tier3(s1))

    # Scenario 2: one tyre-model trigger only -> now PIT_LATER, not PIT_NOW
    # (three-way split: 0=DONT_PIT, 1-2=PIT_LATER, 3+=PIT_NOW)
    s2 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("One trigger (cliff proximity):", evaluate_tier3(s2))

    # Scenario 3: three triggers active (cliff proximity + tyre age + SC) -> PIT_NOW
    s3 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=True, rival_undercut_threat=False, in_dirty_air=False)
    print("Three triggers:", evaluate_tier3(s3))

    # Scenario 4: unknown tyre-model output (None) shouldn't false-trigger
    s4 = Tier3State(cliff_probability_next_5_laps=None, predicted_pace_loss=None,
                     tyre_age=10, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False)
    print("Unknown tyre-model output (None):", evaluate_tier3(s4))

    # Scenario 5: SAME tyre state as scenario 3, but SC gamble recommends WAIT.
    # cliff_proximity and tyre_age get suppressed, safety_car (already active,
    # untouched by the gamble) remains -> only 1 trigger -> PIT_LATER, with
    # reason="SC_GAMBLE" explicitly flagged for the Execution Tree to consume.
    s5 = Tier3State(cliff_probability_next_5_laps=0.05, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=True, rival_undercut_threat=False, in_dirty_air=False,
                     sc_gamble_recommendation="WAIT")
    print("Same as Scenario 3, but SC gamble says WAIT:", evaluate_tier3(s5))
    print("  (decision changed from PIT_NOW to PIT_LATER, AND reason=SC_GAMBLE is now visible to Execution Tree)")

    # Scenario 6: ARCHITECTURAL FIX DEMONSTRATION. Same inputs as the old
    # "driver stress" scenario - tyre_age trigger active + driver_stress_signal
    # True. Under the OLD (wrong) design this reached "Multiple" (2 tyre+stress
    # triggers counted together) -> PIT_NOW. Under the FIXED design, stress is
    # NOT a vote - only tyre_age counts (1 trigger) -> PIT_LATER, with stress
    # passed through as context instead, for the Execution Tree to use.
    s6 = Tier3State(cliff_probability_next_5_laps=0.005, predicted_pace_loss=0.2,
                     tyre_age=22, expected_stint_length=25,
                     undercut_opportunity=False, overcut_opportunity=False,
                     safety_car_deployed=False, rival_undercut_threat=False, in_dirty_air=False,
                     driver_stress_signal=True)
    result6 = evaluate_tier3(s6)
    print("\nDriver stress + one tyre trigger:", result6)
    print("  (OLD design would have counted stress as a 2nd vote -> PIT_NOW. FIXED: stress isn't")
    print("   a vote -> PIT_LATER, but driver_stress_signal=True is still passed through as context)")