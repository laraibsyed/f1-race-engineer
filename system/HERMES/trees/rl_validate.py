""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_agent import ACTIONS, QLearningConfig, TabularQAgent, epsilon_at
from rl_env import HermesStrategyEnv
import rl_bridge

ARTIFACT_DIR = Path(__file__).resolve().parent / "rl_artifacts"
N_EVAL_EPISODES = 200
EVAL_SEED_BASE = 9000

def run_policy(policy_fn, n_episodes: int, seed_base: int, repo_root=None):
    ""
    returns, actions_taken, compliance_flags = [], Counter(), []
    for i in range(n_episodes):
        env = HermesStrategyEnv(repo_root=repo_root, seed=seed_base + i)
        rng = np.random.default_rng(seed_base + i + 500000)
        state = env.reset()
        done, ep_return, last_info = False, 0.0, None
        while not done:
            valid = env.valid_actions()
            action = policy_fn(state, valid, rng)
            actions_taken[action] += 1
            state, reward, done, info = env.step(action)
            ep_return += reward
            last_info = info
        returns.append(ep_return)
        compliance_flags.append(bool(last_info["mandatory_compliant"]))
    return returns, actions_taken, compliance_flags

def random_policy(state, valid, rng):
    return valid[rng.integers(len(valid))]

def always_stay_out_policy(state, valid, rng):
    ""
    return "STAY_OUT"

def make_untrained_policy():
    ""
    agent = TabularQAgent(QLearningConfig())
    def _policy(state, valid, rng):
        return agent.greedy_action(state, valid)
    return _policy

def make_trained_policy(agent: TabularQAgent):
    def _policy(state, valid, rng):
        return agent.greedy_action(state, valid)
    return _policy

def report_policy_comparison(agent: TabularQAgent) -> dict:
    print("\n" + "=" * 78)
    print(f"POLICY COMPARISON - {N_EVAL_EPISODES} matched simulated episodes per policy, "
          f"same per-episode seeds ({EVAL_SEED_BASE}..{EVAL_SEED_BASE + N_EVAL_EPISODES - 1})")
    print("(2026-09-30 reward redesign: terminal reward now includes a mandatory-two-compound")
    print(" compliance penalty - see rl_env.py - so 'always STAY_OUT' is evaluated here as an")
    print(" explicit baseline POLICY, not just a description of the untrained agent's behaviour.)")
    print("=" * 78)
    results = {}
    for name, policy_fn in (("random", random_policy),
                             ("untrained (all-zero Q)", make_untrained_policy()),
                             ("always STAY_OUT", always_stay_out_policy),
                             ("trained Q-learning", make_trained_policy(agent))):
        returns, actions, compliance = run_policy(policy_fn, N_EVAL_EPISODES, EVAL_SEED_BASE)
        arr = np.array(returns)
        compliance_rate = float(np.mean(compliance))
        results[name] = {"returns": returns, "actions": actions, "mean": float(arr.mean()),
                          "median": float(np.median(arr)), "std": float(arr.std(ddof=1)),
                          "compliance_rate": compliance_rate}
        total_actions = sum(actions.values())
        dist = {a: f"{100 * actions.get(a, 0) / total_actions:.1f}%" for a in ACTIONS}
        print(f"\n  {name}:")
        print(f"    mean reward   = {arr.mean():+9.2f}s")
        print(f"    median reward = {np.median(arr):+9.2f}s")
        print(f"    std dev       = {arr.std(ddof=1):9.2f}s")
        print(f"    action distribution: {dist}")
        print(f"    mandatory two-compound compliance: {compliance_rate:.0%} of episodes")

    diff = results["trained Q-learning"]["mean"] - results["random"]["mean"]
    print(f"\n  trained - random mean reward difference: {diff:+.2f}s "
          f"({'trained is better (less negative time cost)' if diff > 0 else 'trained is WORSE'})")
    diff_u = results["trained Q-learning"]["mean"] - results["untrained (all-zero Q)"]["mean"]
    print(f"  trained - untrained mean reward difference: {diff_u:+.2f}s")
    diff_s = results["trained Q-learning"]["mean"] - results["always STAY_OUT"]["mean"]
    print(f"  trained - always-STAY_OUT mean reward difference: {diff_s:+.2f}s "
          f"({'trained is better' if diff_s > 0 else 'trained is WORSE'})")

    returns_a, _, _ = run_policy(make_trained_policy(agent), 20, EVAL_SEED_BASE)
    returns_b, _, _ = run_policy(make_trained_policy(agent), 20, EVAL_SEED_BASE)
    reproducible = returns_a == returns_b
    print(f"\n  trained policy reproducible across two identical-seed eval runs: {reproducible}")
    results["reproducible"] = reproducible
    results["diff_vs_random"] = diff
    results["diff_vs_untrained"] = diff_u
    results["diff_vs_always_stay_out"] = diff_s
    return results

def print_decision_trace(agent: TabularQAgent, seed: int = 777, max_steps: int = 12, repo_root=None):
    print("\n" + "=" * 78)
    print(f"EXAMPLE SIMULATED DECISION TRACE (trained policy, greedy, env seed={seed})")
    print("=" * 78)
    env = HermesStrategyEnv(repo_root=repo_root, seed=seed)
    state = env.reset()
    print(f"  world: circuit={env.world.circuit}  era={env.world.era}  total_laps={env.world.total_laps}  "
          f"start_compound={env.world.start_compound}  temp={env.world.temp_bucket}  rain={env.world.rain_now}")
    for step in range(max_steps):
        valid = env.valid_actions()
        rl_out = rl_bridge.rl_recommend_for_state(agent, state, valid)
        action = rl_out["rl_strategy_action"]
        next_state, reward, done, info = env.step(action)
        print(f"  lap {info['lap']:>2}: state={state}")
        print(f"          -> action={action:<12} (Q={rl_out['rl_q_value']:+.2f}, "
              f"confidence={rl_out['rl_confidence']:.2f})  reward={reward:+.2f}s")
        state = next_state
        if done:
            print("  (episode ended)")
            break

def run_hermes_integration_demo(agent: TabularQAgent, repo_root=None):
    print("\n" + "=" * 78)
    print("HERMES INTEGRATION DEMO: RL recommendation -> Gate Tree -> Execution Tree")
    print("=" * 78)
    master = rl_bridge.master
    repo_root = Path(repo_root) if repo_root else master.REPO_ROOT
    try:
        laps_path = master.find_laps_features(2023, "Bahrain_Grand_Prix", "R")
    except Exception:
        laps_path = None

    if laps_path is None:
        print("  (2023 Bahrain Grand Prix cached data not found - using a clearly labelled "
              "SYNTHETIC scenario instead)")
        _run_synthetic_integration_examples(agent)
        return

    print(f"  Using REAL replayed data: 2023 Bahrain Grand Prix, drivers VER/PER, laps 1-20")
    tyre_models = master.load_tyre_models(master._data_path(repo_root, "tyre_life_models.pkl"))
    sc_prior = master.load_sc_prior(master._data_path(repo_root, "sc_vsc_circuit_level_prior.csv"))
    cliff_stints = master.load_cliff_stints(master._data_path(repo_root, "cliff_detection_stints.csv"))
    taxonomy = master.load_circuit_taxonomy(repo_root / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
    degr_ordinal, _ = master.circuit_degredation_ordinal_for(taxonomy, "Bahrain_Grand_Prix")
    pit_loss_table = master.load_pit_loss_table(
        master._data_path(repo_root, "checkpoints", "rival_knowledge", "archive_per_race_analysis.csv"))
    pit_loss_s, _ = master.pit_loss_for_circuit(pit_loss_table, "Bahrain_Grand_Prix")
    ctx = master.RaceContext(season=2023, circuit="Bahrain_Grand_Prix", total_laps=57,
                              is_sprint_weekend=False, d1_code="VER", d2_code="PER",
                              circuit_degredation_ordinal=degr_ordinal, pit_loss_s=pit_loss_s,
                              p_sc_5lap=master.get_sc_probability(sc_prior, "Bahrain_Grand_Prix"))
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    decisions = master.run_replay(ctx, resources, range(1, 21))

    shown = 0
    for r in decisions:
        rl_out = rl_bridge.rl_recommend_for_hermes_result(agent, r, ctx)
        combined = rl_bridge.combine_with_gate_tree(rl_out, r)
        interesting = (not combined["agrees_with_hermes"]) or (r["gate_tree_trigger_tier"] is not None)
        if not interesting and shown >= 3:
            continue
        print(f"\n  L{r['lap']:>2} {r['driver']}: compound={r['compound']} tyre_age={r['tyre_age']:.0f} "
              f"tier_reached={r['tier_reached']} gate_decision={r['gate_decision']}")
        print(f"     RL recommendation : {combined['rl_recommendation']} "
              f"(Q={combined['rl_q_value']:+.2f}, confidence={combined['rl_confidence']:.2f})")
        print(f"     HERMES final      : {combined['hermes_final_action']} "
              f"(instruction={r['execution']['driving_instruction']})")
        print(f"     agree             : {combined['agrees_with_hermes']}")
        if combined["override_reason"]:
            print(f"     override reason   : {combined['override_reason']}")
        shown += 1
        if shown >= 8:
            break

def _run_synthetic_integration_examples(agent: TabularQAgent):
    ""
    master = rl_bridge.master

    print("\n  [SYNTHETIC] Scenario A - RL wants to stay out, Tier 1 forces PIT_NOW (unsafe tyre damage):")
    state_a = ("MEDIUM", "10-14", "mid", "medium", False, False, "NEUTRAL", "OK")
    rl_a = rl_bridge.rl_recommend_for_state(agent, state_a, list(ACTIONS))
    hermes_a = {"compound": "MEDIUM", "tyre_age": 12, "lap": 30, "gate_tree_trigger_tier": 1,
                "gate_decision": "PIT_NOW", "triggers": {}, "projection": {},
                "execution": {"driving_instruction": "PIT_LAP"}}
    combined_a = rl_bridge.combine_with_gate_tree(rl_a, hermes_a)
    print(f"    RL recommendation : {combined_a['rl_recommendation']} (Q={combined_a['rl_q_value']:+.2f})")
    print(f"    HERMES final      : {combined_a['hermes_final_action']}  (Tier 1: structural tyre damage)")
    print(f"    agree             : {combined_a['agrees_with_hermes']}")
    print(f"    override reason   : {combined_a['override_reason']}")

    print("\n  [SYNTHETIC] Scenario B - no hard trigger; Tier 3 DONT_PIT, RL also says STAY_OUT:")
    state_b = ("SOFT", "0-4", "early", "low", False, False, "NEUTRAL", "OK")
    rl_b = rl_bridge.rl_recommend_for_state(agent, state_b, list(ACTIONS))
    hermes_b = {"compound": "SOFT", "tyre_age": 3, "lap": 8, "gate_tree_trigger_tier": None,
                "gate_decision": "DONT_PIT", "triggers": {"safety_car": False, "undercut": False,
                                                            "rival_undercut_threat": False},
                "projection": {"cliff_probability_next_5_laps": 0.001},
                "execution": {"driving_instruction": "PUSH"}}
    combined_b = rl_bridge.combine_with_gate_tree(rl_b, hermes_b)
    print(f"    RL recommendation : {combined_b['rl_recommendation']} (Q={combined_b['rl_q_value']:+.2f})")
    print(f"    HERMES final      : {combined_b['hermes_final_action']}  (instruction: "
          f"{hermes_b['execution']['driving_instruction']})")
    print(f"    agree             : {combined_b['agrees_with_hermes']}")

def run_tests(agent: TabularQAgent, repo_root=None) -> bool:
    print("\n" + "=" * 78)
    print("VALIDATION / SANITY CHECKS (Step 14)")
    print("=" * 78)
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    test_agent = TabularQAgent(QLearningConfig(alpha=0.1, gamma=0.9))
    s0, s1 = ("SOFT", "0-4", "early", "low", False, False, "NEUTRAL", "OK"), \
             ("SOFT", "0-4", "early", "low", False, False, "NEUTRAL", "OK")
    from rl_agent import _state_key
    test_agent.q[_state_key(s1)] = np.array([0.0, 5.0, -2.0, 1.0])
    test_agent.update(s0, "STAY_OUT", reward=-3.0, next_state=s1, done=False, next_valid_actions=list(ACTIONS))
    expected = 0.0 + 0.1 * ((-3.0) + 0.9 * 5.0 - 0.0)
    actual = float(test_agent.q[_state_key(s0)][0])
    check("Q-learning update matches Q(s,a) += alpha*(r + gamma*max Q(s',.) - Q(s,a))",
          abs(actual - expected) < 1e-9, f"expected {expected:.5f}, got {actual:.5f}")

    env1, env2 = HermesStrategyEnv(repo_root=repo_root, seed=123), HermesStrategyEnv(repo_root=repo_root, seed=123)
    s_a, s_b = env1.reset(), env2.reset()
    trace_a = [agent.greedy_action(s_a, env1.valid_actions())]
    trace_b = [agent.greedy_action(s_b, env2.valid_actions())]
    for _ in range(5):
        a_a = agent.greedy_action(s_a, env1.valid_actions())
        s_a, _, d_a, _ = env1.step(a_a)
        a_b = agent.greedy_action(s_b, env2.valid_actions())
        s_b, _, d_b, _ = env2.step(a_b)
        trace_a.append(a_a); trace_b.append(a_b)
    check("deterministic inference: identical seed -> identical greedy action trace", trace_a == trace_b)

    env = HermesStrategyEnv(repo_root=repo_root, seed=55)
    state = env.reset()
    n_checked = 0
    violations = 0
    for _ in range(30):
        valid = env.valid_actions()
        for eps in (0.0, 0.3, 1.0):
            a = agent.select_action(state, valid, epsilon=eps)
            n_checked += 1
            if a not in valid:
                violations += 1
        state, _, done, _ = env.step(agent.greedy_action(state, valid))
        if done:
            state = env.reset()
    check("valid action selection: agent never selects outside valid_actions()",
          violations == 0, f"{violations}/{n_checked} violations")

    env = HermesStrategyEnv(repo_root=repo_root, seed=7)
    env.reset()
    env.world.rain_now = True
    valid_wet = env.valid_actions()
    check("wet-weather masking: only STAY_OUT is legal while rain_now=True",
          valid_wet == ["STAY_OUT"], f"got {valid_wet}")
    ns, r, d, info = env.step("PIT_SOFT")
    check("env hard-safety-net: an illegal PIT_SOFT request while raining executes as STAY_OUT, not a pit",
          info["action_taken"] == "STAY_OUT" and info["invalid_attempted"] and info["pit_duration_s"] is None)

    env2b = HermesStrategyEnv(repo_root=repo_root, seed=8)
    env2b.reset()
    env2b.sets_left = 0
    check("zero-sets masking: only STAY_OUT is legal once sets_left == 0",
          env2b.valid_actions() == ["STAY_OUT"], f"got {env2b.valid_actions()}")

    tmp_path = ARTIFACT_DIR / "_persistence_test_qtable.json"
    agent.save(tmp_path)
    reloaded = TabularQAgent.load(tmp_path)
    same_config = reloaded.config == agent.config
    same_table = (set(reloaded.q.keys()) == set(agent.q.keys())
                  and all(np.allclose(reloaded.q[k], agent.q[k]) for k in agent.q))
    check("Q-table persistence/reload: config and table survive a save/load round trip",
          same_config and same_table)
    tmp_path.unlink(missing_ok=True)

    hermes_unsafe = {"compound": "SOFT", "tyre_age": 5, "lap": 10, "gate_tree_trigger_tier": 1,
                      "gate_decision": "PIT_NOW", "triggers": {}, "projection": {},
                      "execution": {"driving_instruction": "PIT_LAP"}}
    rl_wants_stay = {"rl_strategy_action": "STAY_OUT", "rl_q_value": 5.0, "rl_confidence": 1.0,
                      "rl_q_values": {}, "rl_state": (), "rl_valid_actions": list(ACTIONS)}
    combined = rl_bridge.combine_with_gate_tree(rl_wants_stay, hermes_unsafe)
    check("Gate Tree Tier 1 overrides an RL recommendation that disagrees with a safety PIT_NOW",
          combined["hermes_final_action"] == "PIT_SOFT" and not combined["agrees_with_hermes"]
          and combined["override_reason"] is not None)

    import inspect
    src = inspect.getsource(rl_bridge.discretize_hermes_result)
    forbidden = ["actual_is_pit_in_lap", "explanation", "data_quality_notes"]
    check("no-lookahead: discretize_hermes_result() reads none of the post-hoc/explanation fields",
          not any(f in src for f in forbidden))

    import subprocess
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "master.py"), "selftest"],
                             capture_output=True, text=True)
    check("existing HERMES selftest (master.py selftest) still passes", result.returncode == 0)

    check("rl_env.py imports master.py without modifying it (module identity)",
          rl_bridge.master.__name__ == "hermes_master")

    n_nonzero_states = sum(1 for v in agent.q.values() if np.any(v != 0))
    check("training produced non-identical Q-values from the initial all-zero table",
          n_nonzero_states > 0 and agent.n_visited_states() > 0,
          f"{n_nonzero_states} states with a non-zero Q-value, {agent.n_visited_states()} visited")

    from rl_env import COMPOUND_FAMILY

    env = HermesStrategyEnv(repo_root=repo_root, seed=200)
    env.reset()
    env.world.rain_now = False
    valid = env.valid_actions()
    pit_action = next((a for a in valid if a != "STAY_OUT"), None)
    if pit_action:
        _, r, _, info = env.step(pit_action)
        check("a PIT_x action has an immediate, non-zero real pit-stop cost",
              info["pit_duration_s"] is not None and info["pit_duration_s"] > 0 and r < 0,
              f"pit_duration_s={info['pit_duration_s']}")
    else:
        check("a PIT_x action has an immediate, non-zero real pit-stop cost", False, "no legal pit action sampled")

    env2 = HermesStrategyEnv(repo_root=repo_root, seed=201)
    positive_slope_found = False
    for compound, circuit, era in env2._reg_keys[:60]:
        mod = env2.tyre_models["reg_models"][(compound, circuit, era)]
        if mod.coef_[0] > 1e-6:
            positive_slope_found = True
            break
    check("at least one real fitted model has a positive tyre_age coefficient (searchable sample)",
          positive_slope_found)
    if positive_slope_found:

        from rl_env import master as _m
        proj_now = _m.build_projection_fn(env2.tyre_models, compound, circuit, era, 60.0, 2, "warm", 1)
        loss_fresh = proj_now(0)["predicted_pace_loss"]
        loss_aged = proj_now(25)["predicted_pace_loss"]
        check("fresh tyre (age 0) predicts lower pace loss than an aged tyre (age 25) for this model",
              loss_fresh is not None and loss_aged is not None and loss_fresh < loss_aged,
              f"age0={loss_fresh}, age25={loss_aged}")

    env3 = HermesStrategyEnv(repo_root=repo_root, seed=202)
    env3.reset()
    env3.world.rain_now = False
    first_lap_cost = None
    cumulative = 0.0
    for i in range(15):
        _, r, done, info = env3.step("STAY_OUT")
        cumulative += -r
        if i == 0:
            first_lap_cost = -r
        if done:
            break
    check("cumulative STAY_OUT cost over 15 laps exceeds a single lap's cost (degradation accumulates)",
          first_lap_cost is not None and cumulative > first_lap_cost,
          f"lap1={first_lap_cost:.3f}s, cumulative15={cumulative:.3f}s" if first_lap_cost is not None else "")

    env4 = HermesStrategyEnv(repo_root=repo_root, seed=203)
    env4.reset()
    env4.world.rain_now = False
    for _ in range(5):
        env4.step("STAY_OUT")
    age_before = env4.tyre_age
    valid4 = env4.valid_actions()
    pit_action4 = next((a for a in valid4 if a != "STAY_OUT"), None)
    if pit_action4:
        env4.step(pit_action4)
        check("a PIT_x action resets tyre_age to 0", age_before > 0 and env4.tyre_age == 0,
              f"age_before={age_before}, age_after={env4.tyre_age}")
    else:
        check("a PIT_x action resets tyre_age to 0", False, "no legal pit action sampled at this state")

    from rl_env import ACTION_TO_FAMILY
    env5 = HermesStrategyEnv(repo_root=repo_root, seed=204)
    env5.reset()
    env5.world.rain_now = False
    done = False
    last_info = None
    while not done:
        _, _, done, last_info = env5.step("STAY_OUT")
    check("never-pit (dry episode) is NOT mandatory-compliant, and is charged the real penalty",
          last_info["mandatory_compliant"] is False)

    env6 = HermesStrategyEnv(repo_root=repo_root, seed=204)
    env6.reset()
    env6.world.rain_now = False
    start_family = COMPOUND_FAMILY.get(env6.compound)
    other_action = next(a for a, f in ACTION_TO_FAMILY.items() if f != start_family)
    done, last_info = False, None
    step_i = 0
    while not done:
        step_i += 1
        valid6 = env6.valid_actions()
        a = other_action if (step_i == 5 and other_action in valid6) else "STAY_OUT"
        _, _, done, last_info = env6.step(a)
    check("pitting once to a DIFFERENT compound family satisfies mandatory compliance",
          last_info["mandatory_compliant"] is True,
          f"families_used={last_info['compound_families_used']}")

    env7 = HermesStrategyEnv(repo_root=repo_root, seed=205)
    env7.reset()
    env7.world.rain_now = True
    done, last_info = False, None
    while not done:
        _, _, done, last_info = env7.step("STAY_OUT")
    check("a rain episode (wet_race_exception) is compliant even without ever pitting",
          last_info["mandatory_compliant"] is True)

    return ok

def run_alpha_sensitivity(repo_root=None, n_episodes: int = 300):
    print("\n" + "=" * 78)
    print(f"SENSITIVITY CHECK: learning rate alpha (n_episodes={n_episodes}, gamma=0.97 fixed, seed=42)")
    print("=" * 78)
    from rl_train import train
    for alpha in (0.05, 0.15, 0.30):
        cfg = QLearningConfig(n_episodes=n_episodes, alpha=alpha, gamma=0.97, seed=42)
        trained, history = train(cfg, repo_root=repo_root, verbose=False)
        last_n = max(1, n_episodes // 20)
        last_mean = float(np.mean([h["return"] for h in history[-last_n:]]))
        print(f"  alpha={alpha:.2f}: mean return (last {last_n} episodes) = {last_mean:+.2f}s, "
              f"visited states = {trained.n_visited_states()}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", default=str(ARTIFACT_DIR / "qtable.json"))
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    qtable_path = Path(args.qtable)
    if not qtable_path.exists():
        print(f"[rl_validate] {qtable_path} not found - run rl_train.py first.")
        sys.exit(1)
    agent = TabularQAgent.load(qtable_path)
    print(f"[rl_validate] loaded {qtable_path}: {agent.n_visited_states()} visited states, "
          f"config={agent.config}")

    comparison = report_policy_comparison(agent)
    print_decision_trace(agent)
    run_hermes_integration_demo(agent)
    ok = run_tests(agent)
    if not args.skip_sensitivity:
        run_alpha_sensitivity()

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
