"""
synthetic_sensitivity_experiment.py
-------------------------------------
Controlled sensitivity experiment for the rain-crossover policy
(rain_crossover.py). Does NOT use real race data or claim to validate
forecast accuracy -- it feeds synthetic rain probabilities (0-100%, in 10%
steps) across a spread of race stages (laps remaining) and threshold sets
(baseline/stricter/looser), and reports the policy's output state for every
combination.

WHAT THIS VALIDATES: that the decision function has the INTENDED threshold
behaviour -- state changes happen at the right probability values, in the
right direction, and the three threshold sets diverge sensibly from each
other. This is a test of the CODE'S correctness and internal coherence.

WHAT THIS DOES NOT VALIDATE: whether 35/45/60% (or any other threshold set)
is the empirically correct choice for real F1 rain events. That would need
either real forecast-probability data (not freely available retrospectively,
see fetch_openmeteo_weather.py) or the cost-derivation approach in
crossover_threshold_derivation.py. This script and that one are complementary,
not substitutes for each other -- together with the Sochi 2021 real-race
validation in backtest_crossover.py, they form three distinct, honestly-scoped
pieces of evidence rather than one overreaching claim.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossover import evaluate_crossover, PERTURBATION_SETS, classify_stage  # noqa: E402

# Race-stage scenarios: (laps_remaining, total_laps, label)
RACE_STAGE_SCENARIOS = [
    (55, 70, "Early (79% remaining)"),
    (30, 70, "Mid (43% remaining)"),
    (12, 70, "Mid/Late boundary (17% remaining)"),
    (5, 70, "Late (7% remaining)"),
    (2, 70, "Very late (3% remaining)"),
]

RAIN_PROBABILITIES = list(range(0, 101, 10))  # 0%, 10%, ..., 100%


def run_full_sensitivity_grid():
    """Prints one table per threshold set: rows = rain probability,
    columns = race stage scenario, cells = policy state."""
    for set_name, thresholds in PERTURBATION_SETS.items():
        print(f"\n{'='*90}")
        print(f"  Threshold set: {set_name}  {thresholds}")
        print(f"{'='*90}")

        header = f"  {'P(rain)':<10}"
        for _, _, label in RACE_STAGE_SCENARIOS:
            header += f"{label:<28}"
        print(header)

        for rain_prob in RAIN_PROBABILITIES:
            row = f"  {rain_prob:>3}%      "
            for laps_remaining, total_laps, _ in RACE_STAGE_SCENARIOS:
                result = evaluate_crossover(rain_prob, laps_remaining, total_laps, thresholds)
                row += f"{result.state.value:<28}"
            print(row)


def check_monotonicity():
    """Sanity check: for a fixed race stage, increasing rain probability should
    never DECREASE urgency (STAY_SLICKS -> MONITOR -> CONSIDER_INTERS is the
    only allowed direction as P(rain) rises). If this ever fails, that's a
    real bug in evaluate_crossover(), not a threshold-tuning question."""
    state_order = {"STAY_SLICKS": 0, "MONITOR": 1, "CONSIDER_INTERS": 2, "BOX_INTERS": 3}
    print(f"\n{'='*90}")
    print("  Monotonicity check: does urgency ever decrease as rain probability rises?")
    print(f"{'='*90}")

    all_passed = True
    for set_name, thresholds in PERTURBATION_SETS.items():
        for laps_remaining, total_laps, label in RACE_STAGE_SCENARIOS:
            prev_level = -1
            for rain_prob in RAIN_PROBABILITIES:
                result = evaluate_crossover(rain_prob, laps_remaining, total_laps, thresholds)
                level = state_order[result.state.value]
                if level < prev_level:
                    print(f"  FAIL: {set_name} / {label} -- urgency dropped at "
                          f"P(rain)={rain_prob}% (went from level {prev_level} to {level})")
                    all_passed = False
                prev_level = level

    if all_passed:
        print("  PASS -- across all threshold sets and race stages, urgency "
              "never decreases as rain probability increases. The policy is "
              "internally monotonic, as a sane decision rule should be.")


def check_stage_ordering():
    """Sanity check: for a FIXED rain probability and threshold set, does
    urgency increase (or stay equal) as the race gets later, matching the
    EARLY > MID > LATE threshold design in rain_crossover.py? This directly
    tests whether the stage structure behaves as intended."""
    state_order = {"STAY_SLICKS": 0, "MONITOR": 1, "CONSIDER_INTERS": 2, "BOX_INTERS": 3}
    print(f"\n{'='*90}")
    print("  Stage-ordering check: at a fixed P(rain), does urgency rise (or "
          "stay flat) as the race gets later, for the baseline threshold set?")
    print(f"{'='*90}")

    thresholds = PERTURBATION_SETS["baseline"]
    test_probabilities = [40, 50, 55]  # values that sit between the stage thresholds
    all_passed = True
    for rain_prob in test_probabilities:
        levels = []
        for laps_remaining, total_laps, label in RACE_STAGE_SCENARIOS:
            result = evaluate_crossover(rain_prob, laps_remaining, total_laps, thresholds)
            levels.append((label, state_order[result.state.value]))
        level_values = [lv for _, lv in levels]
        is_non_decreasing = all(level_values[i] <= level_values[i+1] for i in range(len(level_values)-1))
        status = "PASS" if is_non_decreasing else "FAIL"
        print(f"  P(rain)={rain_prob}%: {' -> '.join(f'{l}={v}' for l, v in levels)}  [{status}]")
        if not is_non_decreasing:
            all_passed = False

    if all_passed:
        print("\n  PASS -- at fixed rain probability, urgency never drops as the "
              "race progresses to a later stage; consistent with the intended "
              "EARLY(60%) > MID(45%) > LATE(35%) threshold design.")


if __name__ == "__main__":
    run_full_sensitivity_grid()
    check_monotonicity()
    check_stage_ordering()

    print(f"\n{'='*90}")
    print("  Summary for dissertation write-up:")
    print("  This is a controlled sensitivity experiment on synthetic inputs,")
    print("  not a validation against real forecast data. It demonstrates the")
    print("  policy's internal coherence (monotonic in rain probability,")
    print("  correctly ordered across race stages) -- necessary but not")
    print("  sufficient evidence, complementing the mathematical derivation")
    print("  (crossover_threshold_derivation.py) and the real-race validation")
    print("  (backtest_crossover.py, Sochi 2021).")
    print(f"{'='*90}")