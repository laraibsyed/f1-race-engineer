""

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

def _safe(v):
    try:
        f = float(v)
        return None if (f != f) else f
    except (TypeError, ValueError):
        return None

NEUTRAL_PROFILE = dict(aggression_level=1.0, defensive_strength=1.0, tyre_management=1.0,
                        consistency_factor=1.0, wet_weather_skill=1.0, pressure_risk_tolerance=1.0)

ROLE_LEVERAGE_FLIP_MARGIN = 0.15

def assign_team_roles(
    d1_code: str, d2_code: str,
    d1_position: Optional[float], d2_position: Optional[float],
    d1_leverage: Optional[float], d2_leverage: Optional[float],
    d1_compromised: bool = False, d2_compromised: bool = False,
) -> dict:
    ""
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

def identify_relevant_rivals(
    own_code: str, ahead_row: Optional[dict], behind_row: Optional[dict],
    championship_rival_code: Optional[str], championship_rival_row: Optional[dict],
    team_objective: str,
) -> dict:
    ""
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

def predict_rival_pit_window(
    rival_tyre_age: Optional[float], rival_expected_stint_length: Optional[float],
    current_lap: int, total_laps: int, rival_cliff_probability: Optional[float] = None,
    cliff_probability_threshold: float = 0.017, margin_laps: int = 2,
) -> Optional[dict]:
    ""
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

RIVAL_WINDOW_COVER_BONUS_FRAC = 0.10
RIVAL_WINDOW_INSIDE_PENALTY_FRAC = 0.05
RIVAL_WINDOW_EXPOSED_PENALTY_FRAC = 0.20

def rival_pit_window_adjustment(my_pit_lap: int, predicted_window: Optional[list],
                                 pit_loss_s: float) -> float:
    ""
    if not predicted_window:
        return 0.0
    window_start, window_end = predicted_window
    if my_pit_lap < window_start:
        return -RIVAL_WINDOW_COVER_BONUS_FRAC * pit_loss_s
    if window_start <= my_pit_lap <= window_end:
        return RIVAL_WINDOW_INSIDE_PENALTY_FRAC * pit_loss_s
    return RIVAL_WINDOW_EXPOSED_PENALTY_FRAC * pit_loss_s

PER_DRIVER_ACTIONS = ("PIT_NOW", "PIT_IN_3", "PIT_IN_6", "EXTEND_TO_CLIFF")
CLOSE_FOLLOWING_SECONDS = 1.0

def generate_candidate_actions(tyre_age: float, expected_stint_length: float,
                                laps_remaining: int, mandatory_compound_done: bool,
                                alt_compound_available: bool) -> list:
    ""
    actions = [a for a in PER_DRIVER_ACTIONS if a != "PIT_IN_6" or laps_remaining > 6]
    actions = [a for a in actions if a != "PIT_IN_3" or laps_remaining > 3]
    if not mandatory_compound_done and alt_compound_available:
        actions.append("ALT_COMPOUND")
    return actions

def joint_candidates(d1_actions: list, d2_actions: list) -> list:
    ""
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

    return current_lap + max(1, round(expected_stint_length - tyre_age))

def apply_driver_profile(marginal_pace_loss_s: Optional[float], profile: dict) -> Optional[float]:
    ""
    if marginal_pace_loss_s is None:
        return None
    tm = profile.get("tyre_management", 1.0) or 1.0
    return marginal_pace_loss_s / tm

def track_position_pace_tradeoff(gap_ahead_s: Optional[float], gap_behind_s: Optional[float],
                                  time_delta_from_strategy_s: Optional[float],
                                  aggression_level: float = 1.0,
                                  close_following_s: float = CLOSE_FOLLOWING_SECONDS) -> dict:
    ""
    in_dirty_air_now = gap_ahead_s is not None and gap_ahead_s < close_following_s
    threatened_from_behind = gap_behind_s is not None and gap_behind_s < close_following_s
    dirty_air_cost_s = None
    if in_dirty_air_now and time_delta_from_strategy_s is not None:
        aggression_relief = max(0.5, min(1.5, aggression_level))
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
    pace_loss_fn: Callable[[float], Optional[float]]

    cliff_probability_fn: Callable[[float], Optional[float]]
    pit_loss_s: float
    gap_ahead_s: Optional[float]
    gap_behind_s: Optional[float]
    mandatory_compound_done: bool
    alt_compound_available: bool
    profile: dict = field(default_factory=lambda: dict(NEUTRAL_PROFILE))

def simulate_driver_action(d: DriverSimInputs, action: str, total_laps: int) -> dict:
    ""
    pit_lap = _action_to_pit_lap(action, d.current_lap, d.expected_stint_length, d.tyre_age)
    laps_on_current_tyres = max(0, pit_lap - d.current_lap)

    raw_marginal = d.pace_loss_fn(d.tyre_age)
    marginal = apply_driver_profile(raw_marginal, d.profile)
    degradation_cost_s = (marginal or 0.0) * laps_on_current_tyres

    cliff_p = d.cliff_probability_fn(d.tyre_age + laps_on_current_tyres)
    cliff_penalty_s = 5.0 * cliff_p if cliff_p is not None else 0.0

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

def filter_regulatory_feasible(candidates: list, d1_sim: dict, d2_sim: dict,
                                d1_mandatory_done: bool, d2_mandatory_done: bool,
                                laps_remaining: int) -> list:
    ""
    feasible = []
    for c in candidates:
        d1_ok = d1_mandatory_done or c["d1_action"] != "EXTEND_TO_CLIFF" or laps_remaining > 3
        d2_ok = d2_mandatory_done or c["d2_action"] != "EXTEND_TO_CLIFF" or laps_remaining > 3
        if d1_ok and d2_ok:
            feasible.append(c)
    return feasible if feasible else candidates

@dataclass
class MCStrategyConfig:
    n_sims: int = 200

    seed: int = 20261001
    pace_loss_sigma_frac: float = 0.15
    cliff_probability_as_bernoulli: bool = True

def monte_carlo_outcome(deterministic: dict, cliff_probability: Optional[float],
                         config: Optional[MCStrategyConfig] = None) -> dict:
    ""
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
    risk_cv = (std / mean) if mean > 0 else 0.0

    return dict(n_sims=cfg.n_sims, mean_time_delta_s=round(mean, 3), p10=round(p10, 3),
                p50=round(p50, 3), p90=round(p90, 3), std=round(std, 3), risk=round(risk_cv, 3))

def time_delta_to_points_delta(time_delta_s: Optional[float], gap_ahead_s: Optional[float],
                                gap_behind_s: Optional[float], current_position: Optional[int],
                                position_points_fn) -> dict:
    ""
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
    ""
    delta_wcc = d1_delta_points + d2_delta_points
    return reward_mod.compute_team_reward(
        delta_wcc=delta_wcc, delta_wdc_d1=d1_delta_points, d1_state=d1_champ_state,
        delta_wdc_d2=d2_delta_points, d2_state=d2_champ_state, team_state=team_state, risk=risk,
    )

RISK_MODE_VARIANCE_PENALTY = {"CONSERVATIVE": 2.0, "BALANCED": 1.0, "AGGRESSIVE": 0.4}

def apply_risk_mode(ranked: list, mode: str, risk_tolerance: float = 1.0) -> list:
    ""
    penalty_weight = RISK_MODE_VARIANCE_PENALTY.get(mode, 1.0) / max(0.5, risk_tolerance)
    out = []
    for c in ranked:
        reward_val = c["reward"]["R_team"]
        mc_risk = c.get("mc_risk", 0.0) or 0.0
        risk_adjusted_score = reward_val - penalty_weight * mc_risk
        out.append({**c, "risk_adjusted_score": round(risk_adjusted_score, 4), "risk_mode": mode})
    out.sort(key=lambda c: c["risk_adjusted_score"], reverse=True)
    return out

def build_strategic_explanation(selected: dict, alternatives: list, role: dict) -> dict:
    ""
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

def evaluate_team_strategy(
    d1: DriverSimInputs, d2: DriverSimInputs, total_laps: int, laps_remaining: int,
    d1_position: Optional[float], d2_position: Optional[float],
    d1_leverage: Optional[float], d2_leverage: Optional[float],
    d1_compromised: bool, d2_compromised: bool,
    d1_champ_state, d2_champ_state, team_state, reward_mod, position_points_fn,
    risk_mode: str = "BALANCED", mc_config: Optional[MCStrategyConfig] = None,
    rival_info: Optional[dict] = None,
) -> dict:
    ""
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
                selected_strategy=selected, alternatives=alternatives[:4],

                explanation=explanation, risk_mode=risk_mode)

def resolve_strategy_execution(selected_strategy: dict, d1_gate_decision: str, d1_tier: Optional[int],
                                d2_gate_decision: str, d2_tier: Optional[int]) -> dict:
    ""

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

            return dict(team_strategy_action=action, executed_action="PIT_NOW", override=False,
                        override_reason=None, should_upgrade_to_pit_now=True)

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
