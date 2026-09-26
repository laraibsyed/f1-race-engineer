"""
rain_crossover.py
-------------------
Weather module's rain-crossover decision policy for HERMES.

DESIGN PRINCIPLE (deliberate): this module does NOT decide whether to pit.
It outputs a strategic STATE based on rain probability + race stage. HERMES'
existing decision architecture combines this state with everything else it
already knows (current track condition, tyre degradation, pit-loss model,
gap to cars around you) to produce the actual pit call.

This separation matters because rain probability alone can never justify a
box call -- e.g. even at 90% rain chance with 3 laps left, boxing might be
wrong if the pit-loss exceeds the laps remaining. That's HERMES' job to weigh,
not this module's.

States (increasing urgency):
    STAY_SLICKS      -- rain probability below the stage threshold, no action
    MONITOR          -- approaching the threshold, flag for closer attention
    CONSIDER_INTERS  -- threshold crossed, hand off to HERMES for a pit decision
    BOX_INTERS       -- reserved for a future version once real-time rain
                        (FastF1 Rainfall flag) is wired in alongside forecast
                        probability -- i.e. it's actually raining now, not just
                        forecast to. NOT triggered by forecast probability alone.

Thresholds below are DELIBERATELY PROVISIONAL -- see perturbation testing in
main() / test_thresholds(). They are initial rule parameters to be stress-tested
against historical wet races (Spa 2021, Brazil 2016, Monaco 2022), not values
claimed to be optimal. The three-stage structure itself (early/mid/late) is the
hypothesis under test, not just the numbers -- see docstring on classify_stage().
"""

from dataclasses import dataclass
from enum import Enum


class RaceStage(Enum):
    EARLY = "early"   # >60% of race remaining
    MID = "mid"       # 20-60% of race remaining
    LATE = "late"     # <20% of race remaining


class CrossoverState(Enum):
    STAY_SLICKS = "STAY_SLICKS"
    MONITOR = "MONITOR"
    CONSIDER_INTERS = "CONSIDER_INTERS"
    BOX_INTERS = "BOX_INTERS"  # not reachable from forecast probability alone yet


# Default thresholds -- provisional, see module docstring.
# Structure: {stage: (monitor_threshold_pct, consider_threshold_pct)}
# monitor_threshold: below this, STAY_SLICKS. Between monitor and consider: MONITOR.
# consider_threshold: at/above this, CONSIDER_INTERS.
DEFAULT_THRESHOLDS = {
    RaceStage.EARLY: (45, 60),
    RaceStage.MID: (30, 45),
    RaceStage.LATE: (20, 35),
}

# Perturbation sets for robustness testing (Step 2-4 in the plan: does the
# POLICY behave sensibly across a plausible range, not just at one hand-picked
# set of numbers). Each is (monitor, consider) per stage, same structure as above.
PERTURBATION_SETS = {
    "baseline": {
        RaceStage.EARLY: (45, 60),
        RaceStage.MID: (30, 45),
        RaceStage.LATE: (20, 35),
    },
    "stricter": {  # requires higher confidence before flagging at every stage
        RaceStage.EARLY: (55, 70),
        RaceStage.MID: (40, 55),
        RaceStage.LATE: (30, 45),
    },
    "looser": {  # flags earlier/more readily at every stage
        RaceStage.EARLY: (35, 50),
        RaceStage.MID: (20, 35),
        RaceStage.LATE: (10, 25),
    },
}


def classify_stage(laps_remaining: int, total_laps: int) -> RaceStage:
    """Race stage by % of race remaining. This 3-way split (60% / 20% cutoffs)
    is itself part of the hypothesis under test -- not just the thresholds
    within each stage. If backtesting shows the boundaries are in the wrong
    place (e.g. 'late' should really kick in at 30% not 20%), that's a
    legitimate finding, distinct from "the threshold number was wrong"."""
    if total_laps <= 0:
        raise ValueError("total_laps must be positive")
    pct_remaining = laps_remaining / total_laps
    if pct_remaining > 0.60:
        return RaceStage.EARLY
    elif pct_remaining > 0.20:
        return RaceStage.MID
    else:
        return RaceStage.LATE


@dataclass
class CrossoverResult:
    state: CrossoverState
    stage: RaceStage
    rain_probability: float
    monitor_threshold: float
    consider_threshold: float
    laps_remaining: int
    total_laps: int

    def __repr__(self):
        return (f"CrossoverResult(state={self.state.value}, stage={self.stage.value}, "
                f"rain_prob={self.rain_probability}%, "
                f"thresholds=({self.monitor_threshold}/{self.consider_threshold}), "
                f"laps={self.laps_remaining}/{self.total_laps})")


def evaluate_crossover(
    rain_probability: float,
    laps_remaining: int,
    total_laps: int,
    thresholds: dict = None,
) -> CrossoverResult:
    """
    Core decision function. Returns a CrossoverResult with a strategic STATE
    (not a pit call) for HERMES to combine with other signals.

    rain_probability: 0-100, from Open-Meteo's forecast precipitation_probability
                       (see fetch_openmeteo_weather.py --mode forecast). NOT the
                       historical/reanalysis endpoint -- that has no probability field.
    thresholds: defaults to DEFAULT_THRESHOLDS; pass an entry from PERTURBATION_SETS
                to stress-test a different rule set against the same race data.
    """
    if not (0 <= rain_probability <= 100):
        raise ValueError(f"rain_probability must be 0-100, got {rain_probability}")

    thresholds = thresholds or DEFAULT_THRESHOLDS
    stage = classify_stage(laps_remaining, total_laps)
    monitor_t, consider_t = thresholds[stage]

    if rain_probability >= consider_t:
        state = CrossoverState.CONSIDER_INTERS
    elif rain_probability >= monitor_t:
        state = CrossoverState.MONITOR
    else:
        state = CrossoverState.STAY_SLICKS

    return CrossoverResult(
        state=state,
        stage=stage,
        rain_probability=rain_probability,
        monitor_threshold=monitor_t,
        consider_threshold=consider_t,
        laps_remaining=laps_remaining,
        total_laps=total_laps,
    )


def run_perturbation_test(rain_probability: float, laps_remaining: int, total_laps: int):
    """Run the same scenario across all perturbation sets side by side.
    This is Step 2-4 of the testing plan: check whether the POLICY's behaviour
    is stable across a plausible range of thresholds, not just report one number."""
    print(f"\nScenario: {rain_probability}% rain chance, {laps_remaining}/{total_laps} laps remaining")
    print(f"{'Threshold set':<12} {'Stage':<8} {'State':<18} {'Thresholds'}")
    for name, thresholds in PERTURBATION_SETS.items():
        result = evaluate_crossover(rain_probability, laps_remaining, total_laps, thresholds)
        print(f"{name:<12} {result.stage.value:<8} {result.state.value:<18} "
              f"({result.monitor_threshold}/{result.consider_threshold})")


if __name__ == "__main__":
    # Sanity check scenarios -- not the real backtest (that needs actual race data
    # joined with weather.csv per lap), just confirming the policy logic behaves
    # sensibly before wiring it into a real backtest against Spa 2021 / Brazil 2016 /
    # Monaco 2022.
    print("=== Rain Crossover Policy: sanity check scenarios ===")

    scenarios = [
        (65, 55, 70),   # early race, high rain chance
        (50, 30, 70),   # mid race, moderate rain chance
        (40, 5, 70),    # late race, moderate rain chance -- the "3-5 laps left" case
        (20, 68, 70),   # early race, low rain chance
    ]

    for rain_prob, laps_rem, total in scenarios:
        result = evaluate_crossover(rain_prob, laps_rem, total)
        print(result)

    print("\n=== Perturbation robustness check ===")
    for rain_prob, laps_rem, total in scenarios:
        run_perturbation_test(rain_prob, laps_rem, total)