""

from dataclasses import dataclass
from typing import Optional
import numpy as np

MIN_SEGMENT_LENGTH = 3
MIN_IMPROVEMENT = 0.20
MIN_SLOPE_RATIO = 2.0
MIN_STEP_SECONDS = 0.3

def _fit_line_sse(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    preds = slope * x + intercept
    return float(np.sum((y - preds) ** 2)), float(slope)

def detect_cliff(x: np.ndarray, y: np.ndarray):
    ""
    n = len(x)
    if n < 2 * MIN_SEGMENT_LENGTH:
        return None, False, {"reason": "stint too short to scan yet"}

    whole_sse, _ = _fit_line_sse(x, y)
    best = {"total_sse": np.inf, "split": None, "pre_slope": None, "post_slope": None}
    for i in range(MIN_SEGMENT_LENGTH, n - MIN_SEGMENT_LENGTH):
        pre_sse, pre_slope = _fit_line_sse(x[:i], y[:i])
        post_sse, post_slope = _fit_line_sse(x[i:], y[i:])
        total_sse = pre_sse + post_sse
        if total_sse < best["total_sse"]:
            best.update(total_sse=total_sse, split=i, pre_slope=pre_slope, post_slope=post_slope)

    if best["split"] is None:
        return None, False, {"reason": "no valid split found"}

    i = best["split"]
    improvement = (whole_sse - best["total_sse"]) / whole_sse if whole_sse > 0 else 0.0
    slope_ratio = (best["post_slope"] / best["pre_slope"]) if best["pre_slope"] > 0 else np.inf

    pre_tail_median = float(np.median(y[max(0, i - MIN_SEGMENT_LENGTH):i]))
    post_head_median = float(np.median(y[i:i + MIN_SEGMENT_LENGTH]))
    local_step = post_head_median - pre_tail_median

    is_cliff = (
        improvement >= MIN_IMPROVEMENT
        and best["post_slope"] > 0
        and slope_ratio >= MIN_SLOPE_RATIO
        and local_step >= MIN_STEP_SECONDS
    )
    cliff_tyre_age = x[i] if is_cliff else None
    return cliff_tyre_age, is_cliff, {"improvement": improvement, "slope_ratio": slope_ratio,
                                        "local_step_seconds": local_step}

def tyre_cliff_already_hit(tyre_age_history: list, laptime_seconds_history: list) -> bool:
    ""
    x = np.array(tyre_age_history, dtype=float)
    y = np.array(laptime_seconds_history, dtype=float)
    _, is_cliff, _ = detect_cliff(x, y)
    return is_cliff

@dataclass
class Tier1State:
    red_flag_or_race_stopped: bool
    tyre_structurally_damaged: bool
    unsafe_weather: bool
    tyre_age_history: list
    laptime_seconds_history: list

def evaluate_tier1(state: Tier1State) -> str:
    ""
    if state.red_flag_or_race_stopped:
        return "PIT_NOW"
    if state.tyre_structurally_damaged:
        return "PIT_NOW"
    if tyre_cliff_already_hit(state.tyre_age_history, state.laptime_seconds_history):
        return "PIT_NOW"
    if state.unsafe_weather:
        return "PIT_NOW"
    return "MOVE_TO_TIER_2"

if __name__ == "__main__":
    print("=== Gate Tree Tier 1 decision sequence ===")

    s1 = Tier1State(False, False, False, [1, 2, 3], [90.1, 90.0, 89.9])
    print("Clean state, short stint:", evaluate_tier1(s1))

    s2 = Tier1State(True, False, False, [1, 2, 3], [90.1, 90.0, 89.9])
    print("Red flag active:", evaluate_tier1(s2))

    ages = list(range(1, 15))
    laptimes = [90 + 0.02 * a for a in ages[:9]] + [90 + 0.02 * 9 + 0.6 * (a - 9) for a in ages[9:]]
    s3 = Tier1State(False, False, False, ages, laptimes)
    print("Live cliff developing in current stint:", evaluate_tier1(s3))

    s4 = Tier1State(False, False, False, list(range(1, 15)),
                     [90 + 0.02 * a + (0.3 if a == 7 else 0) for a in range(1, 15)])
    print("Single noisy lap, no real cliff:", evaluate_tier1(s4))
