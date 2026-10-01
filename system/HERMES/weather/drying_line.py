""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DryingState(Enum):
    STAY_WET = "STAY_WET"

    GATE_OPEN_NO_EVIDENCE = "GATE_OPEN_NO_EVIDENCE"

    CONSIDER_DRIER_TYRE = "CONSIDER_DRIER_TYRE"

MIN_SECONDS_SINCE_RAIN_STOPPED = 300

def eligibility_gate(seconds_since_rain_end: Optional[float],
                      min_seconds: float = MIN_SECONDS_SINCE_RAIN_STOPPED) -> bool:
    ""
    if seconds_since_rain_end is None:
        return False
    return seconds_since_rain_end >= min_seconds

@dataclass
class LapTimeSample:
    driver: str
    lap_time_seconds: float
    compound: str

def relative_performance_delta(
    switched_driver_lap: LapTimeSample,
    reference_laps: list,
) -> Optional[float]:
    ""
    reference_times = [lap.lap_time_seconds for lap in reference_laps
                        if lap.compound != switched_driver_lap.compound]
    if not reference_times:
        return None
    reference_times.sort()
    n = len(reference_times)
    median_reference = (reference_times[n // 2] if n % 2 == 1
                         else (reference_times[n // 2 - 1] + reference_times[n // 2]) / 2)
    return switched_driver_lap.lap_time_seconds - median_reference

PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS = -1.0

def performance_evidence(delta_seconds: Optional[float],
                          threshold: float = PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS) -> bool:
    ""
    if delta_seconds is None:
        return False
    return delta_seconds <= threshold

@dataclass
class DryingCrossoverResult:
    state: DryingState
    gate_open: bool
    seconds_since_rain_end: Optional[float]
    performance_delta_seconds: Optional[float]
    evidence_found: bool

    def __repr__(self):
        return (f"DryingCrossoverResult(state={self.state.value}, "
                f"gate_open={self.gate_open}, "
                f"seconds_since_rain_end={self.seconds_since_rain_end}, "
                f"delta={self.performance_delta_seconds}, "
                f"evidence={self.evidence_found})")

def evaluate_drying_crossover(
    seconds_since_rain_end: Optional[float],
    switched_driver_lap: Optional[LapTimeSample] = None,
    reference_laps: Optional[list] = None,
    min_seconds: float = MIN_SECONDS_SINCE_RAIN_STOPPED,
    performance_threshold: float = PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS,
) -> DryingCrossoverResult:
    ""
    gate_open = eligibility_gate(seconds_since_rain_end, min_seconds)

    delta = None
    evidence = False
    if gate_open and switched_driver_lap is not None and reference_laps:
        delta = relative_performance_delta(switched_driver_lap, reference_laps)
        evidence = performance_evidence(delta, performance_threshold)

    if not gate_open:
        state = DryingState.STAY_WET
    elif evidence:
        state = DryingState.CONSIDER_DRIER_TYRE
    else:
        state = DryingState.GATE_OPEN_NO_EVIDENCE

    return DryingCrossoverResult(
        state=state,
        gate_open=gate_open,
        seconds_since_rain_end=seconds_since_rain_end,
        performance_delta_seconds=delta,
        evidence_found=evidence,
    )

if __name__ == "__main__":
    print("=== Drying-track crossover: sanity check scenarios ===\n")

    r1 = evaluate_drying_crossover(seconds_since_rain_end=60)
    print("Rain stopped 1 min ago, no switch data yet:")
    print(f"  {r1}\n")

    r2 = evaluate_drying_crossover(seconds_since_rain_end=400)
    print("Rain stopped 6.7 min ago, no switch data yet:")
    print(f"  {r2}\n")

    switched = LapTimeSample(driver="GAS", lap_time_seconds=95.0, compound="INTERMEDIATE")
    reference = [
        LapTimeSample(driver="HAM", lap_time_seconds=94.0, compound="WET"),
        LapTimeSample(driver="VER", lap_time_seconds=94.5, compound="WET"),
        LapTimeSample(driver="LEC", lap_time_seconds=93.8, compound="WET"),
    ]
    r3 = evaluate_drying_crossover(
        seconds_since_rain_end=400, switched_driver_lap=switched, reference_laps=reference
    )
    print("Gate open, GAS switched to inters but is SLOWER than wet-tyre field:")
    print(f"  {r3}\n")

    switched_fast = LapTimeSample(driver="GAS", lap_time_seconds=91.5, compound="INTERMEDIATE")
    r4 = evaluate_drying_crossover(
        seconds_since_rain_end=400, switched_driver_lap=switched_fast, reference_laps=reference
    )
    print("Gate open, GAS switched to inters and is now clearly FASTER than wet-tyre field:")
    print(f"  {r4}\n")

    r5 = evaluate_drying_crossover(seconds_since_rain_end=None)
    print("Rain has not stopped (or never started):")
    print(f"  {r5}")
