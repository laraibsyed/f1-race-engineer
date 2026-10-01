""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_agent import ACTIONS, TabularQAgent
import rl_env as _rl_env

master = _rl_env.master

def rl_recommend_for_state(agent: TabularQAgent, state: tuple, valid_actions: list) -> dict:
    ""
    qv = agent.q_values(state)
    legal_qv = {a: float(qv[ACTIONS.index(a)]) for a in valid_actions}
    ranked = sorted(legal_qv.items(), key=lambda kv: kv[1], reverse=True)
    best_action, best_q = ranked[0]
    second_q = ranked[1][1] if len(ranked) > 1 else best_q
    confidence = float(best_q - second_q)
    return {
        "rl_strategy_action": best_action,
        "rl_q_value": best_q,
        "rl_confidence": confidence,
        "rl_q_values": legal_qv,
        "rl_state": state,
        "rl_valid_actions": list(valid_actions),
    }

def _hermes_action_from_result(result: dict) -> Optional[str]:
    execution = result.get("execution") or {}
    instruction = execution.get("driving_instruction")
    if instruction is None:
        return None
    if instruction == "PIT_LAP":
        family = _rl_env.COMPOUND_FAMILY.get(result.get("compound"), "MEDIUM")
        return {"SOFT": "PIT_SOFT", "MEDIUM": "PIT_MEDIUM", "HARD": "PIT_HARD"}[family]
    return "STAY_OUT"

def discretize_hermes_result(result: dict, ctx) -> tuple:
    ""
    compound = result.get("compound")
    family = _rl_env.COMPOUND_FAMILY.get(compound, "MEDIUM")
    tyre_age = float(result.get("tyre_age") or 0.0)

    if tyre_age < 5:
        age_bucket = "0-4"
    elif tyre_age < 10:
        age_bucket = "5-9"
    elif tyre_age < 15:
        age_bucket = "10-14"
    else:
        age_bucket = "15+"

    lap_number = int(result.get("lap") or 0)
    laps_remaining = max(0, ctx.total_laps - lap_number)
    phase = (master.crossover.classify_stage(laps_remaining, ctx.total_laps).value
             if master.crossover is not None and ctx.total_laps > 0 else "mid")

    projection = result.get("projection") or {}
    cliff_p = projection.get("cliff_probability_next_5_laps")
    threshold = master.gate3.CLIFF_PROBABILITY_THRESHOLD
    if cliff_p is None:
        cliff_bucket = "low"
    elif cliff_p < 0.5 * threshold:
        cliff_bucket = "low"
    elif cliff_p < threshold:
        cliff_bucket = "medium"
    else:
        cliff_bucket = "high"

    triggers = result.get("triggers") or {}
    sc_active = bool(triggers.get("safety_car", False))
    rain_now = False

    threat_behind = bool(triggers.get("rival_undercut_threat", False))
    opportunity_ahead = bool(triggers.get("undercut", False))
    gap_state = "THREAT_BEHIND" if threat_behind else ("OPPORTUNITY_AHEAD" if opportunity_ahead else "NEUTRAL")

    sets_bucket = "OK"

    return (family, age_bucket, phase, cliff_bucket, sc_active, rain_now, gap_state, sets_bucket)

def rl_recommend_for_hermes_result(agent: TabularQAgent, result: dict, ctx) -> dict:
    state = discretize_hermes_result(result, ctx)
    valid_actions = list(ACTIONS)
    return rl_recommend_for_state(agent, state, valid_actions)

def combine_with_gate_tree(rl_out: dict, hermes_result: dict) -> dict:
    ""
    hermes_action = _hermes_action_from_result(hermes_result)
    agree = (hermes_action is not None) and (hermes_action == rl_out["rl_strategy_action"])
    gate_tier = hermes_result.get("gate_tree_trigger_tier")
    if gate_tier == 1:
        override_reason = ("Gate Tree Tier 1 (hard safety gate) forced PIT_NOW independently of RL - "
                            "Tier 1 is non-negotiable and never consults the RL recommendation.")
    elif gate_tier == 2:
        override_reason = ("Gate Tree Tier 2 (regulatory gate) forced PIT_FLEXIBLE independently of RL - "
                            "a legal/feasibility rule, not a strategic choice, so RL is not consulted.")
    elif not agree:
        override_reason = (f"Gate/Execution Tree's own Tier-3 evaluation reached "
                            f"'{hermes_action}', which differs from RL's advisory "
                            f"'{rl_out['rl_strategy_action']}' - the Gate/Execution Tree result is what "
                            f"actually executes; RL is informational only.")
    else:
        override_reason = None
    return {
        "rl_recommendation": rl_out["rl_strategy_action"],
        "rl_q_value": rl_out["rl_q_value"],
        "rl_confidence": rl_out["rl_confidence"],
        "rl_q_values": rl_out["rl_q_values"],
        "hermes_final_action": hermes_action,
        "agrees_with_hermes": agree,
        "override_reason": override_reason,
    }

def explain_rl_decision(rl_out: dict) -> str:
    ""
    lines = [f"RL strategic recommendation: {rl_out['rl_strategy_action']}",
             f"Reason: {rl_out['rl_strategy_action']} had the highest learned Q-value "
             f"({rl_out['rl_q_value']:+.2f}) among the legal actions for state {rl_out['rl_state']}.",
             "Q-values (legal actions only):"]
    for action, q in sorted(rl_out["rl_q_values"].items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {action:<12} = {q:+.3f}")
    lines.append(f"Confidence (best - second-best Q-value): {rl_out['rl_confidence']:.3f}s")
    return "\n".join(lines)
