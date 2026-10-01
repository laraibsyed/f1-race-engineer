""

from dataclasses import dataclass
from enum import Enum

class RaceStage(Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"

class CrossoverState(Enum):
    STAY_SLICKS = "STAY_SLICKS"
    MONITOR = "MONITOR"
    CONSIDER_INTERS = "CONSIDER_INTERS"
    BOX_INTERS = "BOX_INTERS"

DEFAULT_THRESHOLDS = {
    RaceStage.EARLY: (45, 60),
    RaceStage.MID: (30, 45),
    RaceStage.LATE: (20, 35),
}

PERTURBATION_SETS = {
    "baseline": {
        RaceStage.EARLY: (45, 60),
        RaceStage.MID: (30, 45),
        RaceStage.LATE: (20, 35),
    },
    "stricter": {
        RaceStage.EARLY: (55, 70),
        RaceStage.MID: (40, 55),
        RaceStage.LATE: (30, 45),
    },
    "looser": {
        RaceStage.EARLY: (35, 50),
        RaceStage.MID: (20, 35),
        RaceStage.LATE: (10, 25),
    },
}

def classify_stage(laps_remaining: int, total_laps: int) -> RaceStage:
    ""
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
    ""
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
    ""
    print(f"\nScenario: {rain_probability}% rain chance, {laps_remaining}/{total_laps} laps remaining")
    print(f"{'Threshold set':<12} {'Stage':<8} {'State':<18} {'Thresholds'}")
    for name, thresholds in PERTURBATION_SETS.items():
        result = evaluate_crossover(rain_probability, laps_remaining, total_laps, thresholds)
        print(f"{name:<12} {result.stage.value:<8} {result.state.value:<18} "
              f"({result.monitor_threshold}/{result.consider_threshold})")

if __name__ == "__main__":

    print("=== Rain Crossover Policy: sanity check scenarios ===")

    scenarios = [
        (65, 55, 70),
        (50, 30, 70),
        (40, 5, 70),
        (20, 68, 70),
    ]

    for rain_prob, laps_rem, total in scenarios:
        result = evaluate_crossover(rain_prob, laps_rem, total)
        print(result)

    print("\n=== Perturbation robustness check ===")
    for rain_prob, laps_rem, total in scenarios:
        run_perturbation_test(rain_prob, laps_rem, total)
