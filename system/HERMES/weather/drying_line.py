"""
drying_track_crossover.py
---------------------------
The REVERSE transition to rain_crossover.py: wets/inters -> slicks (or
wets -> inters) as the track dries, rather than slicks -> inters as rain
worsens.

WHY THIS IS A SEPARATE MODULE, NOT A MIRRORED THRESHOLD SET
--------------------------------------------------------------
rain_crossover.py answers "is it about to rain enough to justify switching
onto a wetter tyre" -- a WEATHER question, answered by a rain probability.

This module answers "is the track dry enough to switch onto a drier tyre" --
a TRACK-CONDITION question. Weather data (has it stopped raining) cannot
answer this alone, because a track does not dry uniformly or instantly:
Monaco 2022 showed cars queuing up to switch to inters at very different
laps (Gasly lap 4, Vettel/Tsunoda soon after, Hamilton lap 16, Perez lap 16,
Leclerc/Verstappen lap 18) purely because the track dried unevenly and each
driver/team judged "is it ready" differently -- there was no single instant
where a weather signal alone would have given the right answer for everyone.

ARCHITECTURE (two-stage, not one signal)
------------------------------------------
Stage 1 -- ELIGIBILITY GATE (time-since-rain-stopped, from weather data):
    Answers "COULD the track plausibly have dried enough by now?" Uses
    is_rain_end from weather_cleaned.csv (already built) plus a minimum
    elapsed-time threshold. This does NOT trigger a switch by itself --
    it only opens the door to Stage 2. Rain stopping does not mean the
    track is ready (see module reasoning above).

Stage 2 -- PERFORMANCE EVIDENCE (relative lap-time delta):
    Answers "IS there actual on-track evidence the drier tyre is now
    competitive?" Compares a driver who has already switched to the drier
    compound against a reference group still on the wetter compound, using
    a lap-time delta. Cars that have already switched act as "mobile
    sensors" for real track conditions -- this is the actual crossover
    signal, not the elapsed time.

Only when BOTH stages pass does this module report a state suggesting the
switch is justified. Matches the same discipline as rain_crossover.py: this
module outputs a STATE for HERMES to combine with other signals (pit-loss,
track position, tyre life), not a pit command.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DryingState(Enum):
    STAY_WET = "STAY_WET"                      # rain hasn't stopped long enough
                                                 # to even consider switching
    GATE_OPEN_NO_EVIDENCE = "GATE_OPEN_NO_EVIDENCE"  # enough time has passed,
                                                 # but no performance evidence yet
                                                 # that the drier tyre pays off
    CONSIDER_DRIER_TYRE = "CONSIDER_DRIER_TYRE"  # gate open AND performance
                                                 # evidence supports switching --
                                                 # hand off to HERMES for the
                                                 # actual pit decision


# --- Stage 1: eligibility gate ---

# Minimum time (seconds) since rain was last detected before even considering
# a switch. Provisional, same status as rain_crossover.py's thresholds --
# not empirically calibrated, a starting point for sensitivity testing.
# Rationale: Monaco 2022's fastest driver to switch (Gasly) did so within
# minutes of conditions improving, but this was an aggressive/early call
# specifically rewarded by clean air and low risk given his track position --
# a conservative default should require more elapsed time than the single
# fastest real example.
MIN_SECONDS_SINCE_RAIN_STOPPED = 300  # 5 minutes, provisional


def eligibility_gate(seconds_since_rain_end: Optional[float],
                      min_seconds: float = MIN_SECONDS_SINCE_RAIN_STOPPED) -> bool:
    """Stage 1: could the track plausibly have dried enough by now?
    seconds_since_rain_end: seconds elapsed since weather.csv's is_rain_end
    fired (rain last confirmed to have stopped). None means rain hasn't
    stopped yet (or never started), so the gate cannot be open."""
    if seconds_since_rain_end is None:
        return False
    return seconds_since_rain_end >= min_seconds


# --- Stage 2: performance evidence ---

@dataclass
class LapTimeSample:
    driver: str
    lap_time_seconds: float
    compound: str  # e.g. "WET", "INTERMEDIATE", "SOFT"


def relative_performance_delta(
    switched_driver_lap: LapTimeSample,
    reference_laps: list,
) -> Optional[float]:
    """
    Compares a driver who has switched to a drier compound against a
    reference group of drivers still on the wetter compound, using the
    MEDIAN of the reference group's lap times (robust to one slow/blocked
    reference car skewing the comparison, per the "matched reference cars"
    principle -- ideally the caller has already filtered reference_laps to
    exclude in/out laps, laps behind Safety Car/VSC, and laps affected by
    traffic, since this function does not do that filtering itself).

    Returns delta in seconds: negative means the switched driver is FASTER
    than the reference (evidence the drier tyre now pays off). Returns None
    if there's no valid reference group to compare against.
    """
    reference_times = [lap.lap_time_seconds for lap in reference_laps
                        if lap.compound != switched_driver_lap.compound]
    if not reference_times:
        return None
    reference_times.sort()
    n = len(reference_times)
    median_reference = (reference_times[n // 2] if n % 2 == 1
                         else (reference_times[n // 2 - 1] + reference_times[n // 2]) / 2)
    return switched_driver_lap.lap_time_seconds - median_reference


# Performance threshold: how much faster (in seconds) the switched driver
# needs to be, relative to the reference median, before this counts as
# genuine evidence rather than noise. Provisional -- lap time variance from
# traffic/driver skill alone can easily be 0.5-1s, so this should be set
# comfortably above normal lap-to-lap variance to avoid false positives.
PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS = -1.0  # switched driver must be at
# least 1.0s/lap faster than the reference median to count as evidence


def performance_evidence(delta_seconds: Optional[float],
                          threshold: float = PERFORMANCE_EVIDENCE_THRESHOLD_SECONDS) -> bool:
    """Stage 2: is there real evidence the drier tyre is now competitive?
    delta_seconds more negative than threshold => switched driver is
    meaningfully faster than the wet-tyre reference group."""
    if delta_seconds is None:
        return False
    return delta_seconds <= threshold


# --- Combined decision ---

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
    """
    Full two-stage evaluation. If no switched_driver_lap/reference_laps are
    provided (e.g. no one has switched yet -- this is exactly the situation
    early in a drying window, before any "mobile sensor" data exists), the
    gate result alone determines the state: GATE_OPEN_NO_EVIDENCE rather than
    CONSIDER_DRIER_TYRE, since there's no on-track evidence yet either way.
    This correctly models the real situation Monaco showed: someone has to
    be first, and until they are, there's no performance signal to act on --
    only the weather-derived gate.
    """
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

    # Scenario 1: rain just stopped, no one has switched yet
    r1 = evaluate_drying_crossover(seconds_since_rain_end=60)
    print("Rain stopped 1 min ago, no switch data yet:")
    print(f"  {r1}\n")

    # Scenario 2: gate open (5+ min since rain stopped), but no one switched
    r2 = evaluate_drying_crossover(seconds_since_rain_end=400)
    print("Rain stopped 6.7 min ago, no switch data yet:")
    print(f"  {r2}\n")

    # Scenario 3: gate open, someone switched but isn't faster yet (too early)
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

    # Scenario 4: gate open, switched driver now clearly faster -- real evidence
    switched_fast = LapTimeSample(driver="GAS", lap_time_seconds=91.5, compound="INTERMEDIATE")
    r4 = evaluate_drying_crossover(
        seconds_since_rain_end=400, switched_driver_lap=switched_fast, reference_laps=reference
    )
    print("Gate open, GAS switched to inters and is now clearly FASTER than wet-tyre field:")
    print(f"  {r4}\n")

    # Scenario 5: rain hasn't stopped at all
    r5 = evaluate_drying_crossover(seconds_since_rain_end=None)
    print("Rain has not stopped (or never started):")
    print(f"  {r5}")