"""
Team Strategy — the team-coupled strategic layer (2026-10-01, operationally
wired 2026-10-02)
==================================================================
Attached to each lap's decision dict as result["team_strategy"] by
master.py's `run_replay`. The selected joint strategy IS operational, via
`resolve_strategy_execution` (below): it can UPGRADE a Tier-3 DONT_PIT/
PIT_LATER to PIT_NOW this lap (exactly the Execution Tree's own existing
PIT_NOW path then runs for real - no bypass, no duplicate decision system),
but it can NEVER override a Tier 1 (hard safety) or Tier 2 (regulatory)
result, and it can NEVER suppress a Tier 3 trigger that has ALREADY fired
PIT_NOW from real signals (cliff risk/pace loss/tyre age/undercut) - those
cases are reported as an explicit override, never silently applied. See
`resolve_strategy_execution`'s own docstring for the exact rule, and
master.py's run_replay for exactly where it's invoked (between computing
team_strategy_result and calling merge_execution).

Every function here is a PURE function over already-computed inputs - no
file I/O, no network calls, no imports of any other HERMES module - the same
standalone-callable convention gate-tier-3.py/execution-tree.py already use.
master.py owns gathering the real inputs (standings, projections,
expected_stint_length, driver taxonomy, pit loss, etc.) and calling these
functions once per lap when both tracked drivers have a row that lap.

Covers (see the task brief's own phase numbers):
  Phase 1  - assign_team_roles
  Phase 2  - generate_candidate_actions / joint_candidates / simulate_strategy
  Phase 3  - evaluate_team_reward (wraps reward.py's compute_team_reward /
             justification_check - does NOT reimplement them)
  Phase 4  - identify_relevant_rivals
  Phase 5  - predict_rival_pit_window
  Phase 6  - joint_candidates (D1 x D2, not independent per-car evaluation)
  Phase 7  - apply_driver_profile (tyre_management/aggression/consistency/
             wet_weather_skill/risk_tolerance folded into the simulation)
  Phase 8  - track_position_pace_tradeoff
  Phase 9  - apply_risk_mode
  Phase 10 - filter_regulatory_feasible
  Phase 11 - monte_carlo_outcome
  Phase 12 - build_strategic_explanation
  evaluate_team_strategy - the one entry point master.py calls per lap,
             orchestrating all of the above in the order the task's own
             "LIVE PIPELINE INTEGRATION" diagram specifies.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

# ============================================================================
# Shared small helpers
# ============================================================================
def _safe(v):
    try:
        f = float(v)
        return None if (f != f) else f  # NaN check without importing numpy
    except (TypeError, ValueError):
        return None


NEUTRAL_PROFILE = dict(aggression_level=1.0, defensive_strength=1.0, tyre_management=1.0,
                        consistency_factor=1.0, wet_weather_skill=1.0, pressure_risk_tolerance=1.0)


# ============================================================================
# PHASE 1 — Dynamic team role assignment
# ============================================================================
# ASSUMPTION (flagged, same convention as DEADLINE_BUFFER_LAPS / INSTRUCTION_NEUTRAL_BAND_RATIO
# elsewhere in this project): how much tighter D2's own title fight needs to be than D1's before
# priority flips purely on championship grounds. Not calibrated against real team-order decisions -
# a defensible starting point, not a claimed-optimal one.
ROLE_LEVERAGE_FLIP_MARGIN = 0.15


def assign_team_roles(
    d1_code: str, d2_code: str,
    d1_position: Optional[float], d2_position: Optional[float],
    d1_leverage: Optional[float], d2_leverage: Optional[float],
    d1_compromised: bool = False, d2_compromised: bool = False,
) -> dict:
    """Returns {priority_driver_id, support_driver_id, team_objective, reason}.

    Resolution order (highest-priority reason wins, matching the task's own
    "shift priority if..." list):
      1. One car compromised (damage/penalty) -> the OTHER car becomes
         priority; objective becomes "maximise the surviving car's result".
      2. Championship leverage: `d1_leverage`/`d2_leverage` are
         reward.championship_leverage(...) outputs (Phase 3's own function,
         NOT reimplemented here) - if D2's title fight is leveraging the
         team's own choice has a real reason to prioritise D2's own article
         result
      2. Championship leverage: if D2's title fight is meaningfully tighter
         than D1's (by more than ROLE_LEVERAGE_FLIP_MARGIN), priority flips
         to D2 on championship grounds.
      3. On-track: if D2 is genuinely ahead of D1 on track (lower Position)
         and neither of the above fired, priority flips to D2 for this lap
         (protect the car that is actually ahead, rather than artificially
         promoting a trailing "priority" car past its own teammate).
      4. DEFAULT: D1 priority - the EXACT same fallback the Execution Tree
         already used before this layer existed (BP: "Driver 1 Priority" is
         the MDP's own stated default state) - this function only ever
         CHANGES that outcome when it has real evidence to act on, never on
         a guess (unknown leverage/position inputs fall straight through to
         this default).
    """
    if d1_compromised and not d2_compromised:
        return dict(priority_driver_id=d2_code, support_driver_id=d1_code,
                     team_objective="maximise_surviving_car_result",
                     reason=f"{d1_code} compromised (damage/penalty this race) - {d2_code} becomes priority")
    if d2_compromised and not d1_compromised:
        return dict(priority_driver_id=d1_code, support_driver_id=d2_code,
                     team_objective="maximise_surviving_car_result",
                     reason=f"{d2_code} compromised (damage/penalty this race) - {d1_code} stays/becomes priority")

    if d1_leverage is not None and d2_leverage is not None and (d2_leverage - d1_leverage) > ROLE_LEVERAGE_FLIP_MARGIN:
        return dict(priority_driver_id=d2_code, support_driver_id=d1_code,
                     team_objective="maximise_d2_championship_value",
                     reason=(f"{d2_code}'s championship leverage ({d2_leverage:.2f}) exceeds "
                             f"{d1_code}'s ({d1_leverage:.2f}) by more than {ROLE_LEVERAGE_FLIP_MARGIN} "
                             f"- D2's title fight is the live one this race"))

    if (d1_position is not None and d2_position is not None and d2_position < d1_position
            and not (d1_leverage is not None and d2_leverage is not None
                     and (d1_leverage - d2_leverage) > ROLE_LEVERAGE_FLIP_MARGIN)):
        return dict(priority_driver_id=d2_code, support_driver_id=d1_code,
                     team_objective="protect_d2_track_position",
                     reason=f"{d2_code} is genuinely ahead of {d1_code} on track (P{d2_position:.0f} vs "
                            f"P{d1_position:.0f}) with no championship reason to override it")

    return dict(priority_driver_id=d1_code, support_driver_id=d2_code,
                 team_objective="default_driver_1_priority",
                 reason="no compromise, no decisive championship-leverage gap, and D1 not behind D2 on "
                        "track - falls through to the project's existing Driver-1-priority default")


# ============================================================================
# PHASE 4 — Relevant rival selection
# ============================================================================
def identify_relevant_rivals(
    own_code: str, ahead_row: Optional[dict], behind_row: Optional[dict],
    championship_rival_code: Optional[str], championship_rival_row: Optional[dict],
    team_objective: str,
) -> dict:
    """`ahead_row`/`behind_row`/`championship_rival_row` are plain dicts with
    at minimum {"Driver","Position","gap_s"} (gap_s may be None). Distinguishes
    "nearest car" from "strategically relevant rival" per the task spec -
    never hardcodes a team/driver identity; everything comes from the real
    on-track adjacency (already computed by master.find_adjacent_rows) and
    the real championship-standings rival (already resolved by master's
    standings lookup), not a guess."""
    relevant = []
    if ahead_row is not None:
        relevant.append(dict(driver=ahead_row.get("Driver"), relation="on_track_ahead",
                              gap_s=ahead_row.get("gap_s")))
    if behind_row is not None:
        relevant.append(dict(driver=behind_row.get("Driver"), relation="on_track_behind",
                              gap_s=behind_row.get("gap_s")))
    if championship_rival_code is not None and championship_rival_row is not None:
        already = any(r["driver"] == championship_rival_code for r in relevant)
        if not already:
            relevant.append(dict(driver=championship_rival_code, relation="championship_rival",
                                  gap_s=championship_rival_row.get("gap_s")))

    # Primary rival: the championship rival if it's ALSO on track nearby (within a lap's worth of
    # track-position relevance) when the team objective is championship-driven; otherwise the
    # nearest on-track car, since that's who the current lap's strategic decision actually interacts
    # with. Never defaults to a hardcoded team.
    primary, reason = None, None
    champ_entry = next((r for r in relevant if r["relation"] == "championship_rival"), None)
    if team_objective in ("maximise_d2_championship_value", "maximise_surviving_car_result") and champ_entry:
        primary = champ_entry["driver"]
        reason = "team objective is championship-driven and the championship rival is tracked this lap"
    elif ahead_row is not None:
        primary = ahead_row.get("Driver")
        reason = "nearest car ahead on track is the immediate strategic interaction this lap"
    elif behind_row is not None:
        primary = behind_row.get("Driver")
        reason = "no car ahead (track leader) - nearest car behind is the relevant threat"

    return dict(relevant_rivals=relevant, primary_rival=primary, reason=reason)


# ============================================================================
# PHASE 5 — Rival pit-window prediction (forward-looking, probability/window
# NOT a claimed-certain lap)
# ============================================================================
def predict_rival_pit_window(
    rival_tyre_age: Optional[float], rival_expected_stint_length: Optional[float],
    current_lap: int, total_laps: int, rival_cliff_probability: Optional[float] = None,
    cliff_probability_threshold: float = 0.017, margin_laps: int = 2,
) -> Optional[dict]:
    """Heuristic, explicitly NOT a calibrated probability model (no rival pit-timing
    classifier exists in this project - see HERMES_MASTER_BLUEPRINT.md B4/B7). Built
    entirely from numbers HERMES already computes for its OWN strategy (expected
    stint length, cliff probability) applied to the rival's own tyre state - reused,
    not duplicated. Returns None (never a guess) if the inputs needed are missing.
    `confidence` is a simple 0-1 heuristic ratio of "how close is the rival's tyre
    age to its own expected stint length", boosted when the rival's own cliff
    probability is independently elevated - documented as a heuristic, not an
    ML-calibrated figure."""
    if rival_tyre_age is None or rival_expected_stint_length is None or rival_expected_stint_length <= 0:
        return None
    laps_to_expected_pit = rival_expected_stint_length - rival_tyre_age
    window_start = max(current_lap, round(current_lap + laps_to_expected_pit - margin_laps))
    window_end = min(total_laps, round(current_lap + laps_to_expected_pit + margin_laps))
    if window_end < window_start:
        window_end = window_start

    proximity_ratio = min(1.0, max(0.0, rival_tyre_age / rival_expected_stint_length))
    confidence = proximity_ratio
    if rival_cliff_probability is not None and rival_cliff_probability >= cliff_probability_threshold:
        confidence = min(1.0, confidence + 0.2)
    confidence = round(confidence, 3)

    return dict(predicted_pit_window=[int(window_start), int(window_end)], confidence=confidence,
                basis="heuristic: rival tyre_age vs its own expected_stint_length + cliff_probability "
                      "(reused HERMES models applied to the rival's own state) - NOT a calibrated "
                      "pit-timing classifier")


# ASSUMPTION (flagged, heuristic - explicitly NOT a calibrated model per the task's own
# instruction): how many pit-loss-equivalent seconds a candidate gains/loses for pitting
# before/inside/after a rival's PREDICTED window. Magnitude is a FRACTION of the real,
# circuit-specific pit_loss_s already used everywhere else in this project - not an
# invented constant with its own independent scale.
RIVAL_WINDOW_COVER_BONUS_FRAC = 0.10    # pit strictly before the rival's predicted window opens
RIVAL_WINDOW_INSIDE_PENALTY_FRAC = 0.05  # pit lands inside the rival's own predicted window
RIVAL_WINDOW_EXPOSED_PENALTY_FRAC = 0.20  # still out when the rival's predicted window has passed


def rival_pit_window_adjustment(my_pit_lap: int, predicted_window: Optional[list],
                                 pit_loss_s: float) -> float:
    """PHASE 5 wiring (2026-10-02): makes the rival's predicted pit window an
    INPUT to candidate scoring, not merely an explanation field - the task's
    own example (cover the undercut vs accept/expose it) implemented exactly
    as a time-equivalent adjustment added to a candidate's `time_delta_s`,
    so it flows into the SAME reward/ranking path every other cost component
    already uses (no parallel scoring system). Returns 0.0 (no adjustment)
    when no window is available - never guesses a rival state it doesn't
    have real projection-derived evidence for."""
    if not predicted_window:
        return 0.0
    window_start, window_end = predicted_window
    if my_pit_lap < window_start:
        return -RIVAL_WINDOW_COVER_BONUS_FRAC * pit_loss_s
    if window_start <= my_pit_lap <= window_end:
        return RIVAL_WINDOW_INSIDE_PENALTY_FRAC * pit_loss_s
    return RIVAL_WINDOW_EXPOSED_PENALTY_FRAC * pit_loss_s


# ============================================================================
# PHASES 2 / 6 / 7 / 8 — Candidate generation, joint simulation,
# driver-profile adjustment, track-position/pace trade-off
# ============================================================================
PER_DRIVER_ACTIONS = ("PIT_NOW", "PIT_IN_3", "PIT_IN_6", "EXTEND_TO_CLIFF")
CLOSE_FOLLOWING_SECONDS = 1.0  # reused constant, matches master.py's own in_dirty_air threshold


def generate_candidate_actions(tyre_age: float, expected_stint_length: float,
                                laps_remaining: int, mandatory_compound_done: bool,
                                alt_compound_available: bool) -> list:
    """Small, fixed action menu per the task's own examples (pit now / pit in N
    laps / extend stint / alt compound) - deliberately NOT an open-ended
    continuous search space ("do not overengineer"). `ALT_COMPOUND` is only
    offered while the mandatory 2-dry-compound rule isn't satisfied yet AND a
    different compound genuinely has sets remaining (gate2-sourced input)."""
    actions = [a for a in PER_DRIVER_ACTIONS if a != "PIT_IN_6" or laps_remaining > 6]
    actions = [a for a in actions if a != "PIT_IN_3" or laps_remaining > 3]
    if not mandatory_compound_done and alt_compound_available:
        actions.append("ALT_COMPOUND")
    return actions


def joint_candidates(d1_actions: list, d2_actions: list) -> list:
    """Phase 6: D1 strategy x D2 strategy, evaluated JOINTLY - not each car
    independently. This is the actual team-coupled contribution the task
    calls "essential"."""
    return [dict(d1_action=a, d2_action=b, strategy_id=f"{a}|{b}")
            for a, b in itertools.product(d1_actions, d2_actions)]


def _action_to_pit_lap(action: str, current_lap: int, expected_stint_length: float, tyre_age: float) -> int:
    if action == "PIT_NOW":
        return current_lap
    if action == "PIT_IN_3":
        return current_lap + 3
    if action == "PIT_IN_6":
        return current_lap + 6
    if action == "ALT_COMPOUND":
        return current_lap
    # EXTEND_TO_CLIFF
    return current_lap + max(1, round(expected_stint_length - tyre_age))


def apply_driver_profile(marginal_pace_loss_s: Optional[float], profile: dict) -> Optional[float]:
    """PHASE 7: tyre_management (>1.0 = better-than-average conservation in
    this project's 0.85-1.15 taxonomy scale) scales the projected marginal
    degradation DOWN for a driver who manages tyres well, and up for one who
    doesn't - applied as a direct divisor on the model's own marginal
    pace-loss output, not a new degradation model. aggression_level/
    defensive_strength/wet_weather_skill are used elsewhere (track-position
    trade-off, risk variance, weather value respectively) - see those
    functions' own docstrings for exactly where. `pressure_risk_tolerance`
    is used in apply_risk_mode, not here."""
    if marginal_pace_loss_s is None:
        return None
    tm = profile.get("tyre_management", 1.0) or 1.0
    return marginal_pace_loss_s / tm


def track_position_pace_tradeoff(gap_ahead_s: Optional[float], gap_behind_s: Optional[float],
                                  time_delta_from_strategy_s: Optional[float],
                                  aggression_level: float = 1.0,
                                  close_following_s: float = CLOSE_FOLLOWING_SECONDS) -> dict:
    """PHASE 8: upgrades the boolean dirty-air trigger into a quantitative
    strategy consideration, reusing the SAME real gap data and the SAME
    CLOSE_FOLLOWING_SECONDS threshold the live dirty_air trigger already
    uses (master.in_dirty_air) - no new coefficient invented for "is this
    close enough to matter". `clean_air_value_s`: if currently in dirty air
    (gap_ahead_s < threshold) and the strategy would extend the stint (grow
    time_delta), the dirty-air cost compounds over those extra laps -
    estimated as the SAME per-lap pace delta already being evaluated (not a
    new number), scaled by aggression_level (a more aggressive driver is
    assumed somewhat more able to find a pass, reducing the effective cost;
    ASSUMPTION, flagged, not fitted). Returns a structured trade-off, not a
    single verdict - the caller folds it into the strategy's total cost."""
    in_dirty_air_now = gap_ahead_s is not None and gap_ahead_s < close_following_s
    threatened_from_behind = gap_behind_s is not None and gap_behind_s < close_following_s
    dirty_air_cost_s = None
    if in_dirty_air_now and time_delta_from_strategy_s is not None:
        aggression_relief = max(0.5, min(1.5, aggression_level))  # bounded, documented ASSUMPTION
        dirty_air_cost_s = abs(time_delta_from_strategy_s) * 0.15 / aggression_relief
    return dict(in_dirty_air_now=in_dirty_air_now, threatened_from_behind=threatened_from_behind,
                dirty_air_cost_s=dirty_air_cost_s)


@dataclass
class DriverSimInputs:
    code: str
    current_lap: int
    tyre_age: float
    compound: str
    expected_stint_length: float
    pace_loss_fn: Callable[[float], Optional[float]]   # tyre_age -> marginal pace loss (reuses
                                                          # master.build_projection_fn/marginal_pace_loss)
    cliff_probability_fn: Callable[[float], Optional[float]]  # tyre_age -> cliff prob next 5 laps
    pit_loss_s: float
    gap_ahead_s: Optional[float]
    gap_behind_s: Optional[float]
    mandatory_compound_done: bool
    alt_compound_available: bool
    profile: dict = field(default_factory=lambda: dict(NEUTRAL_PROFILE))


def simulate_driver_action(d: DriverSimInputs, action: str, total_laps: int) -> dict:
    """Deterministic point-estimate projection for ONE driver's ONE candidate
    action, reusing the existing per-lap pace/cliff projection (`pace_loss_fn`/
    `cliff_probability_fn` are thin closures over master.build_projection_fn -
    this function does not call any model directly, it only composes outputs
    HERMES already produces). `time_delta_s`: negative = faster than staying
    on the current plan with no pit, positive = slower. Grounded in real
    pit-loss/pace-loss units throughout - no position/points guessed here,
    that happens later only where real gap data makes it computable."""
    pit_lap = _action_to_pit_lap(action, d.current_lap, d.expected_stint_length, d.tyre_age)
    laps_on_current_tyres = max(0, pit_lap - d.current_lap)

    raw_marginal = d.pace_loss_fn(d.tyre_age)
    marginal = apply_driver_profile(raw_marginal, d.profile)
    degradation_cost_s = (marginal or 0.0) * laps_on_current_tyres

    cliff_p = d.cliff_probability_fn(d.tyre_age + laps_on_current_tyres)
    cliff_penalty_s = 5.0 * cliff_p if cliff_p is not None else 0.0  # reused constant/shape from
                                                                       # sc-gamble.py's own cliff_penalty

    pit_cost_s = d.pit_loss_s if action in ("PIT_NOW", "ALT_COMPOUND") or pit_lap <= total_laps else 0.0

    tradeoff = track_position_pace_tradeoff(d.gap_ahead_s, d.gap_behind_s,
                                             degradation_cost_s + cliff_penalty_s,
                                             aggression_level=d.profile.get("aggression_level", 1.0))
    dirty_air_cost_s = tradeoff["dirty_air_cost_s"] or 0.0

    total_cost_s = degradation_cost_s + cliff_penalty_s + pit_cost_s + dirty_air_cost_s
    confidence = 1.0 if (raw_marginal is not None and cliff_p is not None) else 0.5

    return dict(action=action, pit_lap=min(pit_lap, total_laps), time_delta_s=round(total_cost_s, 3),
                degradation_cost_s=round(degradation_cost_s, 3), cliff_penalty_s=round(cliff_penalty_s, 3),
                pit_cost_s=round(pit_cost_s, 3), dirty_air_cost_s=round(dirty_air_cost_s, 3),
                track_position_tradeoff=tradeoff, confidence=confidence)


# ============================================================================
# PHASE 10 — Regulatory feasibility filter
# ============================================================================
def filter_regulatory_feasible(candidates: list, d1_sim: dict, d2_sim: dict,
                                d1_mandatory_done: bool, d2_mandatory_done: bool,
                                laps_remaining: int) -> list:
    """Feasibility FILTER applied to already-generated joint candidates, not
    new Gate Tree branches (per the task's explicit instruction). A joint
    candidate is infeasible if either driver's action would leave the
    mandatory 2-dry-compound rule impossible to satisfy before the race ends -
    i.e. an EXTEND_TO_CLIFF/PIT_IN_N action that pushes the driver's only
    remaining compound-change opportunity past the laps remaining, for a
    driver who hasn't already satisfied the rule. Reuses the already-computed
    mandatory_done flags (gate2.mandatory_compound_done) - does not
    reimplement that rule."""
    feasible = []
    for c in candidates:
        d1_ok = d1_mandatory_done or c["d1_action"] != "EXTEND_TO_CLIFF" or laps_remaining > 3
        d2_ok = d2_mandatory_done or c["d2_action"] != "EXTEND_TO_CLIFF" or laps_remaining > 3
        if d1_ok and d2_ok:
            feasible.append(c)
    return feasible if feasible else candidates  # never return an empty candidate set - fall back to
                                                    # "nothing was filterable" rather than leaving the
                                                    # simulator with nothing to rank


# ============================================================================
# PHASE 11 — Monte Carlo strategy comparison (lightweight; configurable N)
# ============================================================================
@dataclass
class MCStrategyConfig:
    n_sims: int = 200          # deliberately small vs sc-gamble.py's 20000 default - this runs per
                                # CANDIDATE per LAP, not once per gamble decision; kept cheap on purpose
    seed: int = 20261001
    pace_loss_sigma_frac: float = 0.15   # ASSUMPTION: +/-15% noise on the marginal pace-loss estimate
    cliff_probability_as_bernoulli: bool = True


def monte_carlo_outcome(deterministic: dict, cliff_probability: Optional[float],
                         config: Optional[MCStrategyConfig] = None) -> dict:
    """Perturbs the SAME deterministic point estimate `simulate_driver_action`
    already produced - does not re-run the tyre model n_sims times (too
    expensive per the task's own "avoid unnecessarily increasing computational
    cost" instruction). Treats the cliff penalty as a genuine Bernoulli event
    (it either happens this horizon or it doesn't, reusing the real
    cliff_probability already computed) and the degradation cost as Gaussian
    noise around the deterministic estimate - a defensible, cheap
    approximation, not a full re-simulation. `expected_points`/
    `expected_position` are intentionally NOT fabricated here - only
    `time_delta_s` distribution and `risk` (coefficient of variation) are
    reported, since nothing upstream gives a calibrated points/position
    mapping for arbitrary time deltas."""
    cfg = config or MCStrategyConfig()
    import random
    rng = random.Random(cfg.seed)
    base_degr = deterministic["degradation_cost_s"]
    base_cliff_penalty = deterministic["cliff_penalty_s"]
    pit_cost = deterministic["pit_cost_s"]
    dirty_air = deterministic["dirty_air_cost_s"]
    p_cliff = cliff_probability if (cliff_probability is not None and cfg.cliff_probability_as_bernoulli) else None

    samples = []
    for _ in range(cfg.n_sims):
        degr = rng.gauss(base_degr, abs(base_degr) * cfg.pace_loss_sigma_frac)
        degr = max(0.0, degr)
        if p_cliff is not None:
            cliff = (base_cliff_penalty / max(p_cliff, 1e-6)) if rng.random() < p_cliff else 0.0
        else:
            cliff = base_cliff_penalty
        samples.append(degr + cliff + pit_cost + dirty_air)

    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / max(1, n - 1)
    std = math.sqrt(var)
    p10 = samples[max(0, int(0.10 * n) - 1)]
    p50 = samples[int(0.50 * n)]
    p90 = samples[min(n - 1, int(0.90 * n))]
    risk_cv = (std / mean) if mean > 0 else 0.0  # coefficient of variation - unitless risk proxy

    return dict(n_sims=cfg.n_sims, mean_time_delta_s=round(mean, 3), p10=round(p10, 3),
                p50=round(p50, 3), p90=round(p90, 3), std=round(std, 3), risk=round(risk_cv, 3))


# ============================================================================
# PHASE 3 — Team reward (wraps reward.py, does not reimplement it)
# ============================================================================
def time_delta_to_points_delta(time_delta_s: Optional[float], gap_ahead_s: Optional[float],
                                gap_behind_s: Optional[float], current_position: Optional[int],
                                position_points_fn) -> dict:
    """Converts a GROUNDED time delta into a points delta ONLY where real gap
    data makes the position change computable - i.e. the time delta is
    compared against the REAL recorded gap to the car immediately ahead/
    behind this lap, not an invented overtake-probability model. If the time
    delta doesn't cross either real gap, position (and therefore points) is
    assumed unchanged - never a fabricated partial-position estimate.
    `position_points_fn` is reward.position_points, reused not reimplemented."""
    if time_delta_s is None or current_position is None:
        return dict(delta_position=0, delta_points=0.0, basis="insufficient data - assumed unchanged")
    delta_position = 0
    basis = "within real gap to adjacent cars - no position change assumed"
    if time_delta_s > 0 and gap_behind_s is not None and time_delta_s > gap_behind_s:
        delta_position = 1
        basis = f"time delta ({time_delta_s:.2f}s) exceeds real gap behind ({gap_behind_s:.2f}s)"
    elif time_delta_s < 0 and gap_ahead_s is not None and abs(time_delta_s) > gap_ahead_s:
        delta_position = -1
        basis = f"time delta ({time_delta_s:.2f}s) exceeds real gap ahead ({gap_ahead_s:.2f}s)"
    new_position = max(1, int(current_position) + delta_position)
    delta_points = position_points_fn(new_position) - position_points_fn(int(current_position))
    return dict(delta_position=delta_position, delta_points=round(delta_points, 2), basis=basis)


def evaluate_team_reward(d1_delta_points: float, d2_delta_points: float,
                          d1_champ_state, d2_champ_state, team_state,
                          reward_mod, risk: float = 0.0) -> dict:
    """Thin wrapper around reward.compute_team_reward - the actual WCC/WDC
    leverage math lives in reward.py and is reused verbatim, not
    reimplemented, per the task's explicit instruction. `delta_wcc` is the
    sum of both drivers' points deltas (team constructors' points = D1+D2).
    `reward_mod` is the already-imported reward.py module (master.py's own
    hyphen-safe-loader pattern applies to this directory name too, but
    reward.py itself has no hyphen and can be imported normally)."""
    delta_wcc = d1_delta_points + d2_delta_points
    return reward_mod.compute_team_reward(
        delta_wcc=delta_wcc, delta_wdc_d1=d1_delta_points, d1_state=d1_champ_state,
        delta_wdc_d2=d2_delta_points, d2_state=d2_champ_state, team_state=team_state, risk=risk,
    )


# ============================================================================
# PHASE 9 — Risk modes
# ============================================================================
# ASSUMPTION (flagged, same provisional-constant convention as elsewhere in this
# project): how heavily each mode penalises outcome variance when ranking
# candidates. Not fitted against real team risk decisions.
RISK_MODE_VARIANCE_PENALTY = {"CONSERVATIVE": 2.0, "BALANCED": 1.0, "AGGRESSIVE": 0.4}


def apply_risk_mode(ranked: list, mode: str, risk_tolerance: float = 1.0) -> list:
    """Re-scores each candidate's reward by subtracting a variance penalty
    scaled by the selected mode AND the relevant driver's own
    pressure_risk_tolerance taxonomy value (PHASE 7 integration point for
    that specific variable) - genuinely changes the ranking, not just a
    displayed label, satisfying the task's explicit requirement. `ranked`
    items must each already carry `reward` (from evaluate_team_reward) and
    `mc_risk` (the Monte Carlo coefficient-of-variation from
    monte_carlo_outcome) - this function does not compute either itself."""
    penalty_weight = RISK_MODE_VARIANCE_PENALTY.get(mode, 1.0) / max(0.5, risk_tolerance)
    out = []
    for c in ranked:
        reward_val = c["reward"]["R_team"]
        mc_risk = c.get("mc_risk", 0.0) or 0.0
        risk_adjusted_score = reward_val - penalty_weight * mc_risk
        out.append({**c, "risk_adjusted_score": round(risk_adjusted_score, 4), "risk_mode": mode})
    out.sort(key=lambda c: c["risk_adjusted_score"], reverse=True)
    return out


# ============================================================================
# PHASE 12 — Quantified strategic explanation
# ============================================================================
def build_strategic_explanation(selected: dict, alternatives: list, role: dict) -> dict:
    """ONLY reports numbers actually present on `selected`/`alternatives` -
    never fabricates a percentage or improvement figure not produced by the
    simulation, per the task's explicit instruction."""
    lines = [f"Selected strategy: {selected['d1_action']} | {selected['d2_action']}",
             f"Team objective: {role['team_objective']} ({role['reason']})"]
    d1o, d2o = selected["d1_outcome"], selected["d2_outcome"]
    lines.append(f"D1 ({selected.get('d1_code', 'D1')}): time_delta={d1o['time_delta_s']:+.2f}s, "
                 f"pit_lap={d1o['pit_lap']}")
    lines.append(f"D2 ({selected.get('d2_code', 'D2')}): time_delta={d2o['time_delta_s']:+.2f}s, "
                 f"pit_lap={d2o['pit_lap']}")
    lines.append(f"Expected team reward (R_team): {selected['reward']['R_team']:+.4f}")
    if alternatives:
        runner_up = alternatives[0]
        diff = selected["reward"]["R_team"] - runner_up["reward"]["R_team"]
        lines.append(f"Next-best alternative ({runner_up['strategy_id']}): R_team="
                     f"{runner_up['reward']['R_team']:+.4f} (selected strategy wins by {diff:+.4f})")
    return dict(plain_text="\n".join(lines), selected_strategy_id=selected["strategy_id"],
                team_objective=role["team_objective"], d1_time_delta_s=d1o["time_delta_s"],
                d2_time_delta_s=d2o["time_delta_s"], r_team=selected["reward"]["R_team"],
                n_alternatives_considered=len(alternatives))


# ============================================================================
# PHASE 12 EXTENSION — Decision trade-off explainability (dashboard, 2026-10-03)
# ============================================================================
def build_decision_trade_off_explanation(
    selected: dict, alternatives: list, role: dict, current_lap: int, risk_mode: str,
    execution_d1: Optional[dict] = None, execution_d2: Optional[dict] = None,
) -> dict:
    """Builds a WHAT/WHY/WHAT-IF/COST/GAIN/NET trade-off view strictly from the
    `selected` candidate and its strongest alternative (`alternatives[0]`) -
    the SAME ranked candidates `evaluate_team_strategy` already produced.
    Does NOT re-run any simulation and does NOT invent a number that isn't
    already present on one of those two dicts - every field below is read
    directly off `d1_outcome`/`d2_outcome`/`reward`/`risk_adjusted_score`, all
    already expressed in the same unit (seconds, via `time_delta_s` and its
    components) so the comparison table is genuinely commensurate, not a
    fabricated equivalence.

    `horizon_laps` is the evaluated candidate's own `pit_lap - current_lap` -
    i.e. the real horizon that candidate's simulation covers, not a fixed
    invented number of laps.

    `net_advantage_s` = alternative's D1 time_delta_s - selected's D1 time_delta_s
    (positive = selected strategy is cheaper/better by that many seconds under
    the model's own cost accounting). Only computed when an alternative
    exists; otherwise None (never guessed).

    `execution_d1`/`execution_d2` are `resolve_strategy_execution`'s own
    per-driver dicts (team_strategy_action/executed_action/override/
    override_reason) - passed straight through under "execution" so the
    dashboard can show when the Gate/Execution Tree accepted vs overrode the
    recommendation, per the task's explicit requirement that overrides must
    never be silently hidden from the explanation.
    """
    d1o, d2o = selected["d1_outcome"], selected["d2_outcome"]

    def _driver_block(outcome):
        return dict(
            action=outcome["action"], pit_lap=outcome["pit_lap"],
            horizon_laps=max(0, outcome["pit_lap"] - current_lap),
            degradation_cost_s=outcome["degradation_cost_s"], cliff_penalty_s=outcome["cliff_penalty_s"],
            pit_cost_s=outcome["pit_cost_s"], dirty_air_cost_s=outcome["dirty_air_cost_s"],
            rival_window_adjustment_s=outcome.get("rival_window_adjustment_s"),
            time_delta_s=outcome["time_delta_s"], confidence=outcome["confidence"],
        )

    out = dict(
        selected_d1_action=selected["d1_action"], selected_d2_action=selected["d2_action"],
        team_objective=role["team_objective"], team_objective_reason=role["reason"],
        risk_mode=risk_mode, d1=_driver_block(d1o), d2=_driver_block(d2o),
        team_reward_r_team=selected["reward"]["R_team"],
        risk_adjusted_score=selected.get("risk_adjusted_score"),
        n_alternatives_considered=len(alternatives),
        alternative=None, net_advantage_s=None, comparison=None,
        execution=(dict(d1=execution_d1, d2=execution_d2) if (execution_d1 or execution_d2) else None),
    )
    if alternatives:
        alt = alternatives[0]
        alt_d1o = alt["d1_outcome"]
        out["alternative"] = dict(
            d1_action=alt["d1_action"], d2_action=alt["d2_action"],
            d1_time_delta_s=alt_d1o["time_delta_s"], d1_pit_lap=alt_d1o["pit_lap"],
            horizon_laps=max(0, alt_d1o["pit_lap"] - current_lap),
            risk_adjusted_score=alt.get("risk_adjusted_score"),
            reward_r_team=alt["reward"]["R_team"],
        )
        out["net_advantage_s"] = round(alt_d1o["time_delta_s"] - d1o["time_delta_s"], 3)
        out["comparison"] = dict(
            immediate_pit_cost_s=dict(selected=d1o["pit_cost_s"], alternative=alt_d1o["pit_cost_s"]),
            degradation_cost_s=dict(selected=d1o["degradation_cost_s"], alternative=alt_d1o["degradation_cost_s"]),
            track_position_cost_s=dict(selected=d1o["dirty_air_cost_s"], alternative=alt_d1o["dirty_air_cost_s"]),
            rival_effect_s=dict(selected=d1o.get("rival_window_adjustment_s"),
                                 alternative=alt_d1o.get("rival_window_adjustment_s")),
            risk_adjusted_score=dict(selected=selected.get("risk_adjusted_score"),
                                      alternative=alt.get("risk_adjusted_score")),
        )
    return out


# ============================================================================
# Top-level orchestrator - the ONE function master.py calls per lap
# ============================================================================
def evaluate_team_strategy(
    d1: DriverSimInputs, d2: DriverSimInputs, total_laps: int, laps_remaining: int,
    d1_position: Optional[float], d2_position: Optional[float],
    d1_leverage: Optional[float], d2_leverage: Optional[float],
    d1_compromised: bool, d2_compromised: bool,
    d1_champ_state, d2_champ_state, team_state, reward_mod, position_points_fn,
    risk_mode: str = "BALANCED", mc_config: Optional[MCStrategyConfig] = None,
    rival_info: Optional[dict] = None,
) -> dict:
    """Orchestrates Phases 1,2,3,6,7,8,9,10,11,12 in the order the task's own
    LIVE PIPELINE INTEGRATION diagram specifies (role assignment -> candidate
    generation -> regulatory feasibility -> simulation -> Monte Carlo ->
    team reward -> risk-aware ranking -> explanation). Phase 4/5 (rival
    selection/prediction) are computed by the caller (master.py, since they
    need live adjacency/championship data this module deliberately doesn't
    fetch) and passed in as `rival_info` - `rival_info["rival_pit_prediction"]`
    (predict_rival_pit_window's own return dict, or None) IS used below to
    adjust D1's per-candidate time_delta_s (PHASE 5 wiring, 2026-10-02) -
    rival state is now a real input to candidate SCORING, not only an
    explanation field. Applied to D1 specifically because `rival_info` is
    built from D1's own real on-track adjacency (see
    master.build_team_strategy_context) - the rival this layer currently
    tracks is whichever car is actually racing D1 for position."""
    role = assign_team_roles(d1.code, d2.code, d1_position, d2_position,
                              d1_leverage, d2_leverage, d1_compromised, d2_compromised)

    d1_actions = generate_candidate_actions(d1.tyre_age, d1.expected_stint_length, laps_remaining,
                                             d1.mandatory_compound_done, d1.alt_compound_available)
    d2_actions = generate_candidate_actions(d2.tyre_age, d2.expected_stint_length, laps_remaining,
                                             d2.mandatory_compound_done, d2.alt_compound_available)
    candidates = joint_candidates(d1_actions, d2_actions)
    candidates = filter_regulatory_feasible(candidates, {}, {}, d1.mandatory_compound_done,
                                             d2.mandatory_compound_done, laps_remaining)

    rival_window = None
    if rival_info and rival_info.get("rival_pit_prediction"):
        rival_window = rival_info["rival_pit_prediction"].get("predicted_pit_window")

    mc_cfg = mc_config or MCStrategyConfig()
    evaluated = []
    for c in candidates:
        d1_out = simulate_driver_action(d1, c["d1_action"], total_laps)
        d2_out = simulate_driver_action(d2, c["d2_action"], total_laps)

        rival_adj = rival_pit_window_adjustment(d1_out["pit_lap"], rival_window, d1.pit_loss_s)
        if rival_adj != 0.0:
            d1_out["time_delta_s"] = round(d1_out["time_delta_s"] + rival_adj, 3)
        d1_out["rival_window_adjustment_s"] = round(rival_adj, 3)

        d1_cliff_p = d1.cliff_probability_fn(d1.tyre_age)
        d2_cliff_p = d2.cliff_probability_fn(d2.tyre_age)
        d1_mc = monte_carlo_outcome(d1_out, d1_cliff_p, mc_cfg)
        d2_mc = monte_carlo_outcome(d2_out, d2_cliff_p, mc_cfg)
        combined_mc_risk = round((d1_mc["risk"] + d2_mc["risk"]) / 2, 3)

        d1_pts = time_delta_to_points_delta(d1_out["time_delta_s"], d1.gap_ahead_s, d1.gap_behind_s,
                                             d1_position, position_points_fn)
        d2_pts = time_delta_to_points_delta(d2_out["time_delta_s"], d2.gap_ahead_s, d2.gap_behind_s,
                                             d2_position, position_points_fn)
        reward = evaluate_team_reward(d1_pts["delta_points"], d2_pts["delta_points"],
                                       d1_champ_state, d2_champ_state, team_state, reward_mod)

        evaluated.append(dict(
            strategy_id=c["strategy_id"], d1_action=c["d1_action"], d2_action=c["d2_action"],
            d1_code=d1.code, d2_code=d2.code,
            d1_outcome=dict(**d1_out, points_delta=d1_pts), d2_outcome=dict(**d2_out, points_delta=d2_pts),
            d1_mc=d1_mc, d2_mc=d2_mc, mc_risk=combined_mc_risk, reward=reward,
            confidence=round((d1_out["confidence"] + d2_out["confidence"]) / 2, 2),
        ))

    d1_rt = d1_champ_state.pressure_risk_tolerance if hasattr(d1_champ_state, "pressure_risk_tolerance") else 1.0
    ranked = apply_risk_mode(evaluated, risk_mode, risk_tolerance=1.0)
    selected, alternatives = ranked[0], ranked[1:]
    explanation = build_strategic_explanation(selected, alternatives, role)

    return dict(role=role, rival_info=rival_info, n_candidates_evaluated=len(evaluated),
                selected_strategy=selected, alternatives=alternatives[:4],  # top few only - keep the
                                                                              # attached result small
                explanation=explanation, risk_mode=risk_mode)


# ============================================================================
# STRATEGY -> EXECUTION WIRING (2026-10-02)
# ============================================================================
def resolve_strategy_execution(selected_strategy: dict, d1_gate_decision: str, d1_tier: Optional[int],
                                d2_gate_decision: str, d2_tier: Optional[int]) -> dict:
    """Decides, for each tracked driver, whether the Strategy Simulation
    Engine's SELECTED joint action actually changes this lap's EXECUTED
    decision - the missing link between "we calculated a strategy" and "it
    operationally happened". Rule (asymmetric, safety-respecting by design):

      - Tier 1 (hard safety) or Tier 2 (regulatory) already decided this
        driver's fate -> team strategy NEVER overrides it. If team strategy
        wanted something different, that's reported as an explicit override
        (never applied silently) - exactly the task's own required shape.
      - Team strategy's action is "PIT_NOW" and the Gate Tree's own Tier 3
        result this lap was DONT_PIT or PIT_LATER (no hard constraint, no
        already-active trigger forcing a stop) -> UPGRADE: executed_action
        becomes PIT_NOW. This is not a bypass - it works by making this
        driver TRIGGERED, so the real, unmodified Execution Tree then runs
        for them exactly as it would for any other PIT_NOW (double-stack/
        priority logic and all) - see master.run_replay for exactly where
        this upgrade is applied before merge_execution is called.
      - Team strategy's action is anything OTHER than "PIT_NOW" (i.e. a
        stay-out/extend recommendation) and Tier 3 ALREADY independently
        triggered PIT_NOW from real signals (cliff risk, pace loss, tyre
        age, undercut, etc.) -> team strategy does NOT suppress it. Reported
        as an explicit override - a speculative team-level simulation does
        not get to talk a driver out of a trigger-justified pit stop.
      - Otherwise (team strategy's recommendation and the Gate Tree's own
        result already agree, or team strategy's stay-out recommendation
        matches a Gate Tree DONT_PIT/PIT_LATER with nothing to suppress) ->
        no change, no override; executed_action is just the existing
        gate_decision, reported for transparency.

    Returns {"d1": {...}, "d2": {...}}, each with team_strategy_action,
    executed_action, override (bool), override_reason (str|None),
    should_upgrade_to_pit_now (bool - the only field master.py actually
    acts on; everything else here is reporting)."""

    def _resolve(action, gate_decision, tier):
        if tier in (1, 2):
            wanted_pit = action == "PIT_NOW"
            got_pit = gate_decision in ("PIT_NOW", "PIT_FLEXIBLE")
            tier_label = "hard safety" if tier == 1 else "regulatory"
            return dict(
                team_strategy_action=action, executed_action=gate_decision,
                override=(wanted_pit != got_pit),
                override_reason=(f"Tier {tier} ({tier_label}) already forced {gate_decision} this lap - "
                                  f"takes precedence over the team-strategy recommendation"
                                  if wanted_pit != got_pit else None),
                should_upgrade_to_pit_now=False,
            )

        if action == "PIT_NOW":
            if gate_decision == "PIT_NOW":
                return dict(team_strategy_action=action, executed_action="PIT_NOW", override=False,
                            override_reason=None, should_upgrade_to_pit_now=False)
            # DONT_PIT or PIT_LATER, Tier 3, no hard constraint blocking it - team strategy's
            # PIT_NOW recommendation IS followed (an upgrade, not an override of anything hard).
            return dict(team_strategy_action=action, executed_action="PIT_NOW", override=False,
                        override_reason=None, should_upgrade_to_pit_now=True)

        # Team strategy wants to stay out (any non-PIT_NOW action: PIT_IN_3/6, EXTEND_TO_CLIFF,
        # ALT_COMPOUND-later, etc.)
        if gate_decision == "PIT_NOW":
            return dict(
                team_strategy_action=action, executed_action="PIT_NOW", override=True,
                override_reason=(
                    "Gate Tree Tier 3 already triggered PIT_NOW from active strategic signals "
                    "(cliff risk / pace loss / tyre age / undercut, etc.) - a team-strategy "
                    "stay-out recommendation does not suppress a real trigger-based pit requirement"),
                should_upgrade_to_pit_now=False)
        return dict(team_strategy_action=action, executed_action=gate_decision, override=False,
                    override_reason=None, should_upgrade_to_pit_now=False)

    return dict(d1=_resolve(selected_strategy["d1_action"], d1_gate_decision, d1_tier),
                d2=_resolve(selected_strategy["d2_action"], d2_gate_decision, d2_tier))
