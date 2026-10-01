
""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "eval_out_team_strategy"
REPLAYS_DIR = OUT_DIR / "replays"

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def race_tags() -> list[str]:
    run_log = pd.read_csv(OUT_DIR / "run_log.csv")
    return sorted(set(f"{s}_{r}" for s, r in zip(run_log["season"], run_log["race"])))

def load_condition(tag: str, condition_suffix: str) -> list[dict]:
    p = REPLAYS_DIR / f"{tag}__{condition_suffix}.jsonl"
    return load_jsonl(p) if p.exists() else []

def strategy_selection_and_explainability(tags: list[str]):
    rows = []
    for tag in tags:
        rows += load_condition(tag, "ts_balanced_rival_on")

    d1_actions, d2_actions, joint_pairs = [], [], []
    objectives, priority_is_d1, priority_is_d2 = [], 0, 0
    n_with_ts = 0
    n_with_tradeoff = n_with_alt = n_with_valid_horizon = n_with_net_adv = 0

    for r in rows:
        ts = r.get("team_strategy")
        if not ts:
            continue
        n_with_ts += 1
        sel = ts["selected_strategy"]
        d1_actions.append(sel["d1_action"])
        d2_actions.append(sel["d2_action"])
        joint_pairs.append(f"{sel['d1_action']}|{sel['d2_action']}")
        role = ts["role"]
        objectives.append(role["team_objective"])
        d1_code = ts["selected_strategy"].get("d1_code")
        if role["priority_driver_id"] == d1_code:
            priority_is_d1 += 1
        else:
            priority_is_d2 += 1

        to = ts.get("trade_off_explanation")
        if to:
            n_with_tradeoff += 1
            if to.get("alternative") is not None:
                n_with_alt += 1
            if to.get("d1", {}).get("horizon_laps", None) is not None and to["d1"]["horizon_laps"] >= 0:
                n_with_valid_horizon += 1
            if to.get("net_advantage_s") is not None:
                n_with_net_adv += 1

    sel_df = pd.DataFrame([
        dict(metric="n_team_strategy_rows", value=n_with_ts),
        *[dict(metric=f"d1_action={k}", value=v) for k, v in pd.Series(d1_actions).value_counts().items()],
        *[dict(metric=f"d2_action={k}", value=v) for k, v in pd.Series(d2_actions).value_counts().items()],
        *[dict(metric=f"joint_strategy={k}", value=v) for k, v in pd.Series(joint_pairs).value_counts().items()],
        *[dict(metric=f"team_objective={k}", value=v) for k, v in pd.Series(objectives).value_counts().items()],
        dict(metric="priority_driver_is_d1", value=priority_is_d1),
        dict(metric="priority_driver_is_d2", value=priority_is_d2),
    ])
    sel_df.to_csv(OUT_DIR / "metrics_strategy_selection.csv", index=False)

    expl_df = pd.DataFrame([
        dict(metric="n_team_strategy_rows", value=n_with_ts),
        dict(metric="pct_with_trade_off_explanation",
             value=round(100 * n_with_tradeoff / n_with_ts, 1) if n_with_ts else None),
        dict(metric="pct_with_alternative", value=round(100 * n_with_alt / n_with_ts, 1) if n_with_ts else None),
        dict(metric="pct_with_valid_horizon",
             value=round(100 * n_with_valid_horizon / n_with_ts, 1) if n_with_ts else None),
        dict(metric="pct_with_net_advantage",
             value=round(100 * n_with_net_adv / n_with_ts, 1) if n_with_ts else None),
    ])
    expl_df.to_csv(OUT_DIR / "metrics_explainability.csv", index=False)
    return sel_df, expl_df, n_with_ts

def risk_sensitivity(tags: list[str]):
    score_rows, flip_cb = [], 0
    flip_ba, flip_ca = 0, 0
    for tag in tags:
        by_mode = {}
        for mode in ("conservative", "balanced", "aggressive"):
            keyed = {}
            for r in load_condition(tag, f"ts_{mode}_rival_on"):
                ts = r.get("team_strategy")
                if ts:
                    keyed[(r["driver"], r["lap"])] = ts
            by_mode[mode] = keyed

        common_keys = set(by_mode["conservative"]) & set(by_mode["balanced"]) & set(by_mode["aggressive"])
        for key in common_keys:
            c, b, a = by_mode["conservative"][key], by_mode["balanced"][key], by_mode["aggressive"][key]
            score_rows.append(dict(race=tag, driver=key[0], lap=key[1],
                                    conservative_score=c["selected_strategy"].get("risk_adjusted_score"),
                                    balanced_score=b["selected_strategy"].get("risk_adjusted_score"),
                                    aggressive_score=a["selected_strategy"].get("risk_adjusted_score"),
                                    conservative_d1=c["selected_strategy"]["d1_action"],
                                    balanced_d1=b["selected_strategy"]["d1_action"],
                                    aggressive_d1=a["selected_strategy"]["d1_action"]))
            if c["selected_strategy"]["d1_action"] != b["selected_strategy"]["d1_action"]:
                flip_cb += 1
            if b["selected_strategy"]["d1_action"] != a["selected_strategy"]["d1_action"]:
                flip_ba += 1
            if c["selected_strategy"]["d1_action"] != a["selected_strategy"]["d1_action"]:
                flip_ca += 1

    df = pd.DataFrame(score_rows)
    n = len(df)
    summary = pd.DataFrame([
        dict(metric="n_rows_with_all_3_modes", value=n),
        dict(metric="mean_conservative_score", value=round(df["conservative_score"].mean(), 4) if n else None),
        dict(metric="mean_balanced_score", value=round(df["balanced_score"].mean(), 4) if n else None),
        dict(metric="mean_aggressive_score", value=round(df["aggressive_score"].mean(), 4) if n else None),
        dict(metric="pct_rows_d1_action_differs_conservative_vs_balanced",
             value=round(100 * flip_cb / n, 1) if n else None),
        dict(metric="pct_rows_d1_action_differs_balanced_vs_aggressive",
             value=round(100 * flip_ba / n, 1) if n else None),
        dict(metric="pct_rows_d1_action_differs_conservative_vs_aggressive",
             value=round(100 * flip_ca / n, 1) if n else None),
    ])
    summary.to_csv(OUT_DIR / "metrics_risk_sensitivity.csv", index=False)
    df.to_csv(OUT_DIR / "risk_sensitivity_per_row.csv", index=False)
    return summary, n

def rival_sensitivity(tags: list[str]):
    n_rival_adjustment_nonzero_on = 0
    n_rival_adjustment_nonzero_off = 0
    adjustments_on = []
    n_action_flips = 0
    n = 0
    for tag in tags:
        on_keyed = {(r["driver"], r["lap"]): r.get("team_strategy")
                    for r in load_condition(tag, "ts_balanced_rival_on") if r.get("team_strategy")}
        off_keyed = {(r["driver"], r["lap"]): r.get("team_strategy")
                     for r in load_condition(tag, "ts_balanced_rival_off") if r.get("team_strategy")}
        common = set(on_keyed) & set(off_keyed)
        n += len(common)
        for key in common:
            ts_on, ts_off = on_keyed[key], off_keyed[key]
            adj_on = ts_on["selected_strategy"]["d1_outcome"].get("rival_window_adjustment_s") or 0.0
            adj_off = ts_off["selected_strategy"]["d1_outcome"].get("rival_window_adjustment_s") or 0.0
            if adj_on != 0.0:
                n_rival_adjustment_nonzero_on += 1
                adjustments_on.append(adj_on)
            if adj_off != 0.0:
                n_rival_adjustment_nonzero_off += 1
            if ts_on["selected_strategy"]["d1_action"] != ts_off["selected_strategy"]["d1_action"]:
                n_action_flips += 1

    adj_series = pd.Series(adjustments_on)
    summary = pd.DataFrame([
        dict(metric="n_rows_with_both_rival_on_and_off", value=n),
        dict(metric="n_rows_rival_adjustment_nonzero_ON",
             value=n_rival_adjustment_nonzero_on),
        dict(metric="pct_rows_rival_adjustment_nonzero_ON",
             value=round(100 * n_rival_adjustment_nonzero_on / n, 1) if n else None),
        dict(metric="n_rows_rival_adjustment_nonzero_OFF (sanity check, must be 0)",
             value=n_rival_adjustment_nonzero_off),
        dict(metric="mean_abs_rival_adjustment_s_when_nonzero_ON",
             value=round(adj_series.abs().mean(), 3) if len(adj_series) else None),
        dict(metric="pct_rows_d1_action_differs_rival_on_vs_off",
             value=round(100 * n_action_flips / n, 1) if n else None),
    ])
    summary.to_csv(OUT_DIR / "metrics_rival_sensitivity.csv", index=False)
    return summary, n

def team_behaviour(tags: list[str]):
    by_lap = {}
    for tag in tags:
        for r in load_condition(tag, "ts_balanced_rival_on"):
            if r.get("team_strategy"):
                by_lap.setdefault((tag, r["lap"]), {})[r["driver"]] = r

    n_lap_rows = 0
    n_nondefault_objective = 0
    n_both_selected_pit_now = 0
    n_both_selected_pit_now_staggered = 0
    objective_counts = {}
    for (tag, lap), drivers in by_lap.items():
        if len(drivers) != 2:
            continue
        n_lap_rows += 1
        any_row = next(iter(drivers.values()))
        ts = any_row["team_strategy"]
        obj = ts["role"]["team_objective"]
        objective_counts[obj] = objective_counts.get(obj, 0) + 1
        if obj != "default_driver_1_priority":
            n_nondefault_objective += 1
        sel = ts["selected_strategy"]
        if sel["d1_action"] == "PIT_NOW" and sel["d2_action"] == "PIT_NOW":
            n_both_selected_pit_now += 1
            instrs = {r["driver"]: (r.get("execution") or {}).get("driving_instruction") for r in drivers.values()}
            if len(set(instrs.values())) > 1:
                n_both_selected_pit_now_staggered += 1

    summary = pd.DataFrame([
        dict(metric="n_lap_rows_both_drivers_tracked", value=n_lap_rows),
        *[dict(metric=f"team_objective={k}", value=v) for k, v in objective_counts.items()],
        dict(metric="pct_lap_rows_nondefault_role", value=round(100 * n_nondefault_objective / n_lap_rows, 1)
             if n_lap_rows else None),
        dict(metric="n_lap_rows_both_selected_joint_pit_now", value=n_both_selected_pit_now),
        dict(metric="n_of_those_staggered_by_execution", value=n_both_selected_pit_now_staggered),
        dict(metric="pct_joint_pit_now_staggered_by_execution",
             value=round(100 * n_both_selected_pit_now_staggered / n_both_selected_pit_now, 1)
             if n_both_selected_pit_now else None),
    ])
    summary.to_csv(OUT_DIR / "metrics_team_behaviour.csv", index=False)
    return summary, n_lap_rows

def execution_metrics(tags: list[str]):
    rows = []
    for tag in tags:
        rows += load_condition(tag, "ts_balanced_rival_on")

    n_with_exec = 0
    n_override = 0
    n_upgrade = 0
    reason_counts = {}
    for r in rows:
        exec_info = r.get("team_strategy_execution")
        if not exec_info:
            continue
        n_with_exec += 1
        if exec_info.get("override"):
            n_override += 1
            reason = exec_info.get("override_reason") or ""
            bucket = ("tier1_or_2_hard_constraint" if "regulatory" in reason or "hard safety" in reason
                      else "tier3_active_trigger_not_suppressed" if "Tier 3" in reason
                      else "other")
            reason_counts[bucket] = reason_counts.get(bucket, 0) + 1
        if exec_info.get("should_upgrade_to_pit_now"):
            n_upgrade += 1

    summary = pd.DataFrame([
        dict(metric="n_rows_with_team_strategy_execution", value=n_with_exec),
        dict(metric="n_override", value=n_override),
        dict(metric="pct_override", value=round(100 * n_override / n_with_exec, 1) if n_with_exec else None),
        dict(metric="n_upgrade_to_pit_now", value=n_upgrade),
        dict(metric="pct_upgrade_to_pit_now", value=round(100 * n_upgrade / n_with_exec, 1) if n_with_exec else None),
        *[dict(metric=f"override_reason={k}", value=v) for k, v in reason_counts.items()],
    ])
    summary.to_csv(OUT_DIR / "metrics_execution.csv", index=False)
    return summary, n_with_exec

def baseline_divergence(tags: list[str]):
    n_rows, n_gate_diff, n_instr_diff = 0, 0, 0
    for tag in tags:
        base = {(r["driver"], r["lap"]): r for r in load_condition(tag, "baseline")}
        ts = {(r["driver"], r["lap"]): r for r in load_condition(tag, "ts_balanced_rival_on")}
        common = set(base) & set(ts)
        for key in common:
            n_rows += 1
            b, t = base[key], ts[key]
            if b.get("gate_decision") != t.get("gate_decision"):
                n_gate_diff += 1
            b_instr = (b.get("execution") or {}).get("driving_instruction")
            t_instr = (t.get("execution") or {}).get("driving_instruction")
            if b_instr != t_instr:
                n_instr_diff += 1

    summary = pd.DataFrame([
        dict(metric="n_matched_driver_lap_rows", value=n_rows),
        dict(metric="n_gate_decision_differs", value=n_gate_diff),
        dict(metric="pct_gate_decision_differs", value=round(100 * n_gate_diff / n_rows, 1) if n_rows else None),
        dict(metric="n_driving_instruction_differs", value=n_instr_diff),
        dict(metric="pct_driving_instruction_differs",
             value=round(100 * n_instr_diff / n_rows, 1) if n_rows else None),
    ])
    summary.to_csv(OUT_DIR / "metrics_baseline_divergence.csv", index=False)
    return summary, n_rows

def main():
    tags = race_tags()
    print(f"[summary] {len(tags)} races found in run_log.csv")

    div_summary, n_div = baseline_divergence(tags)
    sel_df, expl_df, n_ts = strategy_selection_and_explainability(tags)
    risk_summary, n_risk = risk_sensitivity(tags)
    rival_summary, n_rival = rival_sensitivity(tags)
    behaviour_summary, n_behaviour = team_behaviour(tags)
    exec_summary, n_exec = execution_metrics(tags)

    md = [
        "# Team Strategy Layer - Evaluation Summary\n",
        f"\nRaces: {len(tags)} ({', '.join(tags)})\n",
        f"\nTeam-strategy decision rows (both drivers tracked, BALANCED/rival-ON): **{n_ts}**\n",
        "\n## 0. Baseline vs Team Strategy - decision divergence\n",
        div_summary.to_markdown(index=False), "\n",
        "\n## 1. Strategy selection\n",
        sel_df.to_markdown(index=False), "\n",
        "\n## 2. Risk sensitivity (CONSERVATIVE / BALANCED / AGGRESSIVE)\n",
        risk_summary.to_markdown(index=False), "\n",
        "\n## 3. Rival sensitivity (rival intelligence ON vs OFF)\n",
        rival_summary.to_markdown(index=False), "\n",
        "\n## 4. Team behaviour (role assignment, double-stack avoidance)\n",
        behaviour_summary.to_markdown(index=False), "\n",
        "\n## 5. Execution (team recommendation vs executed action)\n",
        exec_summary.to_markdown(index=False), "\n",
        "\n## 6. Explainability\n",
        expl_df.to_markdown(index=False), "\n",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[summary] wrote {OUT_DIR / 'SUMMARY.md'}")

if __name__ == "__main__":
    main()
