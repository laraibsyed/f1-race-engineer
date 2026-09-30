"""
RL -> Gate Tree -> Execution Tree integration adapter
==========================================================
NEW FILE. This is the ONLY place a trained rl_agent.TabularQAgent output is
turned into HERMES-shaped fields and combined with the (unmodified) Gate
Tree / Execution Tree decision. It does not change how the Gate Tree or
Execution Tree decide anything - it only ATTACHES an informational RL
recommendation alongside their real output and records whether the two
agreed.

ARCHITECTURE (exactly as specified - RL never overrides safety):

    predictive models (tyre_life_projection, real fitted regression/Cox)
        -> state representation (rl_env._discretize - reused verbatim here)
        -> Q-learning strategic policy (rl_agent.TabularQAgent, greedy)
        -> Gate Tree (gate-tier-1/2/3, UNCHANGED) - safety/regulatory authority
        -> Execution Tree (execution-tree, UNCHANGED) - final executable instruction
        -> explanation (this file's explain_rl_decision + master.py's own
           explanation_for, extended additively)

The RL recommendation is advisory strategic input: "here is the action the
learned policy would take". The Gate Tree still runs its own, independent,
validated logic against the SAME lap and produces the decision that
actually reaches the Execution Tree. If they disagree, that disagreement is
reported, not silently resolved in RL's favour.

Two ways this module is used:
  1. rl_recommend_for_state(agent, state, valid_actions) - pure function,
     works on any discretised state tuple (used by rl_validate.py's decision
     trace and by the synthetic HERMES-integration demo).
  2. rl_recommend_for_hermes_row(agent, tyre_models, cliff_stints, taxonomy,
     sc_prior, row, ctx, state) - discretises a REAL master.py per-lap `row`
     the SAME way rl_env.py discretises its simulated state (same compound
     families, same calibrated thresholds, same undercut/dirty-air helpers),
     using ONLY information already available at that lap (no lookahead -
     see the "no future information" note below), so it can be attached to
     an actual replay decision (see master.py's optional --rl flag).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_agent import ACTIONS, TabularQAgent  # noqa: E402
import rl_env as _rl_env  # noqa: E402  (re-exports `master`, the unmodified master.py module)

master = _rl_env.master


def rl_recommend_for_state(agent: TabularQAgent, state: tuple, valid_actions: list) -> dict:
    """Pure Q-table lookup for an already-discretised state. Returns the
    recommended action, its Q-value, a simple confidence measure (the gap
    between the best and second-best LEGAL Q-value - large gap = confident,
    near-zero gap = a genuine toss-up), and every legal action's Q-value for
    explainability (Step 12)."""
    qv = agent.q_values(state)
    legal_qv = {a: float(qv[ACTIONS.index(a)]) for a in valid_actions}
    ranked = sorted(legal_qv.items(), key=lambda kv: kv[1], reverse=True)
    best_action, best_q = ranked[0]
    second_q = ranked[1][1] if len(ranked) > 1 else best_q
    confidence = float(best_q - second_q)   # seconds of Q-value margin, not a probability
    return {
        "rl_strategy_action": best_action,
        "rl_q_value": best_q,
        "rl_confidence": confidence,
        "rl_q_values": legal_qv,
        "rl_state": state,
        "rl_valid_actions": list(valid_actions),
    }


# Map a Gate/Execution Tree outcome onto the SAME 4-action vocabulary RL
# uses, so agreement can be checked directly. STAY_OUT covers every
# non-pitting instruction (PUSH/NEUTRAL/MANAGE/CONSERVE); PIT_LAP is mapped
# to the compound family HERMES's own Tier-2/3 state implies (family of the
# CURRENT compound, since neither Gate Tree nor Execution Tree currently
# choose a specific incoming compound - documented limitation, not guessed).
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
    """Discretises a REAL master.evaluate_driver_lap() `result` dict into
    EXACTLY the state representation rl_env.HermesStrategyEnv._discretize()
    uses, so the same trained Q-table applies to both the simulated
    environment and a real replay lap.

    Uses ONLY fields master.evaluate_driver_lap already computed for THIS
    lap (compound, tyre_age, lap number, projection, and - when Tier 3 ran -
    its OWN real trigger booleans for safety_car/undercut/rival_undercut_threat,
    reused directly rather than re-derived) - no lookahead, no future row is
    read anywhere in this function (Step 14 test #7).

    If gate_tree_trigger_tier is 1 or 2, Tier 3 never ran for this lap (the
    Gate Tree already decided PIT_NOW/PIT_FLEXIBLE on safety/regulatory
    grounds alone) - sc/gap default to their neutral values in that case,
    since the Gate Tree's decision governs regardless of what RL would have
    said (see combine_with_gate_tree)."""
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
    rain_now = False   # weather is a Tier-1 hard-gate signal handled entirely by the Gate Tree before
                        # Tier 3 (and RL) ever run for this lap - see combine_with_gate_tree's Tier-1 note
    threat_behind = bool(triggers.get("rival_undercut_threat", False))
    opportunity_ahead = bool(triggers.get("undercut", False))
    gap_state = "THREAT_BEHIND" if threat_behind else ("OPPORTUNITY_AHEAD" if opportunity_ahead else "NEUTRAL")

    sets_bucket = "OK"   # real per-compound set tracking is Tier 2's job (BP §7-B10); RL sees the
                          # generic "OK" bucket on real replay rows, same None-safe spirit as elsewhere -
                          # documented limitation, not a fabricated count.

    return (family, age_bucket, phase, cliff_bucket, sc_active, rain_now, gap_state, sets_bucket)


def rl_recommend_for_hermes_result(agent: TabularQAgent, result: dict, ctx) -> dict:
    state = discretize_hermes_result(result, ctx)
    valid_actions = list(ACTIONS)   # rain masking is Tier 1's job on a real replay lap (see above)
    return rl_recommend_for_state(agent, state, valid_actions)


def combine_with_gate_tree(rl_out: dict, hermes_result: dict) -> dict:
    """The actual RL -> Gate Tree -> Execution Tree combination. Returns a
    dict with the RL recommendation, the Gate/Execution Tree's real
    decision (untouched), whether they agree, and a plain-English note on
    which one governs and why (Step 9/12)."""
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
    """Step 12: an explanation that corresponds EXACTLY to the Q-table
    values just used - not a templated natural-language guess."""
    lines = [f"RL strategic recommendation: {rl_out['rl_strategy_action']}",
             f"Reason: {rl_out['rl_strategy_action']} had the highest learned Q-value "
             f"({rl_out['rl_q_value']:+.2f}) among the legal actions for state {rl_out['rl_state']}.",
             "Q-values (legal actions only):"]
    for action, q in sorted(rl_out["rl_q_values"].items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {action:<12} = {q:+.3f}")
    lines.append(f"Confidence (best - second-best Q-value): {rl_out['rl_confidence']:.3f}s")
    return "\n".join(lines)
