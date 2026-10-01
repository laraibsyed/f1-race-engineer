""
import sys
from pathlib import Path

import pytest

TREES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TREES_DIR))

import importlib.util as _ilu

def _load(name, filename):
    spec = _ilu.spec_from_file_location(name, TREES_DIR / filename)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ts = _load("team_strategy_under_test", "team-strategy.py")

def test_role_default_d1_priority_when_nothing_known():
    role = ts.assign_team_roles("VER", "PER", None, None, None, None)
    assert role["priority_driver_id"] == "VER"
    assert role["team_objective"] == "default_driver_1_priority"

def test_role_flips_when_d1_compromised():
    role = ts.assign_team_roles("VER", "PER", 5, 6, None, None, d1_compromised=True)
    assert role["priority_driver_id"] == "PER"
    assert "compromised" in role["reason"]

def test_role_flips_on_championship_leverage_gap():

    role = ts.assign_team_roles("VER", "PER", 3, 4, 0.2, 0.9)
    assert role["priority_driver_id"] == "PER"
    assert role["team_objective"] == "maximise_d2_championship_value"

def test_role_flips_when_d2_genuinely_ahead_on_track_with_no_championship_reason():
    role = ts.assign_team_roles("VER", "PER", 5, 3, None, None)
    assert role["priority_driver_id"] == "PER"
    assert role["team_objective"] == "protect_d2_track_position"

def test_role_championship_leverage_overrides_mere_track_position():

    role = ts.assign_team_roles("VER", "PER", 5, 3, 0.9, 0.2)
    assert role["priority_driver_id"] == "VER"

def test_identify_relevant_rivals_nearest_on_track_when_no_championship_objective():
    out = ts.identify_relevant_rivals(
        "VER", dict(Driver="LEC", gap_s=1.2), dict(Driver="SAI", gap_s=0.8),
        None, None, "default_driver_1_priority")
    assert out["primary_rival"] == "LEC"
    codes = {r["driver"] for r in out["relevant_rivals"]}
    assert codes == {"LEC", "SAI"}

def test_identify_relevant_rivals_championship_rival_when_objective_is_championship_driven():
    out = ts.identify_relevant_rivals(
        "PER", dict(Driver="LEC", gap_s=1.2), None,
        "HAM", dict(gap_s=None), "maximise_d2_championship_value")
    assert out["primary_rival"] == "HAM"

def test_identify_relevant_rivals_never_hardcodes_a_team():
    out = ts.identify_relevant_rivals("VER", None, None, None, None, "default_driver_1_priority")
    assert out["relevant_rivals"] == []
    assert out["primary_rival"] is None

def test_predict_rival_pit_window_none_when_inputs_missing():
    assert ts.predict_rival_pit_window(None, 30, 10, 57) is None

def test_predict_rival_pit_window_returns_a_window_not_a_certain_lap():
    out = ts.predict_rival_pit_window(rival_tyre_age=25, rival_expected_stint_length=30,
                                       current_lap=10, total_laps=57)
    assert out is not None
    lo, hi = out["predicted_pit_window"]
    assert lo <= hi
    assert 0.0 <= out["confidence"] <= 1.0

def test_predict_rival_pit_window_confidence_rises_with_cliff_probability():
    low = ts.predict_rival_pit_window(25, 30, 10, 57, rival_cliff_probability=0.001)
    high = ts.predict_rival_pit_window(25, 30, 10, 57, rival_cliff_probability=0.05)
    assert high["confidence"] >= low["confidence"]

def test_generate_candidate_actions_small_fixed_menu():
    actions = ts.generate_candidate_actions(tyre_age=10, expected_stint_length=30,
                                             laps_remaining=50, mandatory_compound_done=True,
                                             alt_compound_available=True)
    assert set(actions) == {"PIT_NOW", "PIT_IN_3", "PIT_IN_6", "EXTEND_TO_CLIFF"}

def test_generate_candidate_actions_offers_alt_compound_only_when_needed_and_available():
    actions = ts.generate_candidate_actions(10, 30, 50, mandatory_compound_done=False,
                                             alt_compound_available=True)
    assert "ALT_COMPOUND" in actions
    actions2 = ts.generate_candidate_actions(10, 30, 50, mandatory_compound_done=False,
                                              alt_compound_available=False)
    assert "ALT_COMPOUND" not in actions2

def test_joint_candidates_is_the_full_cartesian_product_not_independent_per_car():
    d1 = ["PIT_NOW", "EXTEND_TO_CLIFF"]
    d2 = ["PIT_NOW", "PIT_IN_3", "EXTEND_TO_CLIFF"]
    out = ts.joint_candidates(d1, d2)
    assert len(out) == len(d1) * len(d2) == 6
    assert any(c["d1_action"] == "PIT_NOW" and c["d2_action"] == "EXTEND_TO_CLIFF" for c in out)

def _fake_driver(code="VER", tyre_age=10.0, gap_ahead=None, gap_behind=None, profile=None):
    return ts.DriverSimInputs(
        code=code, current_lap=10, tyre_age=tyre_age, compound="MEDIUM",
        expected_stint_length=30.0,
        pace_loss_fn=lambda age: 0.05 * age,
        cliff_probability_fn=lambda age: min(1.0, 0.001 * age),
        pit_loss_s=22.0, gap_ahead_s=gap_ahead, gap_behind_s=gap_behind,
        mandatory_compound_done=True, alt_compound_available=False,
        profile=profile or dict(ts.NEUTRAL_PROFILE),
    )

def test_simulate_driver_action_pit_now_costs_pit_loss_not_degradation():
    d = _fake_driver(tyre_age=10.0)
    out = ts.simulate_driver_action(d, "PIT_NOW", total_laps=57)
    assert out["pit_lap"] == 10
    assert out["pit_cost_s"] == pytest.approx(22.0, abs=0.01)
    assert out["degradation_cost_s"] == pytest.approx(0.0, abs=0.01)

def test_simulate_driver_action_extend_accrues_real_degradation():
    d = _fake_driver(tyre_age=10.0)
    out = ts.simulate_driver_action(d, "EXTEND_TO_CLIFF", total_laps=57)
    assert out["pit_lap"] > 10
    assert out["degradation_cost_s"] > 0.0

def test_apply_driver_profile_better_tyre_management_lowers_degradation():
    base = 1.0
    good = ts.apply_driver_profile(base, dict(tyre_management=1.15))
    poor = ts.apply_driver_profile(base, dict(tyre_management=0.85))
    assert good < base < poor

def test_track_position_pace_tradeoff_only_charges_dirty_air_when_actually_in_it():
    clean = ts.track_position_pace_tradeoff(gap_ahead_s=5.0, gap_behind_s=5.0, time_delta_from_strategy_s=10.0)
    dirty = ts.track_position_pace_tradeoff(gap_ahead_s=0.3, gap_behind_s=5.0, time_delta_from_strategy_s=10.0)
    assert clean["dirty_air_cost_s"] is None
    assert dirty["dirty_air_cost_s"] is not None and dirty["dirty_air_cost_s"] > 0

def test_filter_regulatory_feasible_drops_infeasible_extend_when_mandatory_not_done_and_laps_short():
    candidates = ts.joint_candidates(["EXTEND_TO_CLIFF", "PIT_NOW"], ["PIT_NOW"])
    feasible = ts.filter_regulatory_feasible(candidates, {}, {}, d1_mandatory_done=False,
                                              d2_mandatory_done=True, laps_remaining=2)
    assert all(c["d1_action"] != "EXTEND_TO_CLIFF" for c in feasible)

def test_filter_regulatory_feasible_never_returns_empty():
    candidates = ts.joint_candidates(["EXTEND_TO_CLIFF"], ["EXTEND_TO_CLIFF"])
    feasible = ts.filter_regulatory_feasible(candidates, {}, {}, d1_mandatory_done=False,
                                              d2_mandatory_done=False, laps_remaining=1)
    assert len(feasible) > 0

def test_monte_carlo_outcome_is_configurable_and_reproducible():
    det = dict(degradation_cost_s=5.0, cliff_penalty_s=1.0, pit_cost_s=0.0, dirty_air_cost_s=0.0)
    cfg = ts.MCStrategyConfig(n_sims=100, seed=42)
    a = ts.monte_carlo_outcome(det, cliff_probability=0.1, config=cfg)
    b = ts.monte_carlo_outcome(det, cliff_probability=0.1, config=cfg)
    assert a == b
    assert a["n_sims"] == 100
    assert a["p10"] <= a["p50"] <= a["p90"]

@pytest.fixture(scope="module")
def reward_mod():
    sacrifice_dir = TREES_DIR.parent / "strategic-sacrifice"
    spec = _ilu.spec_from_file_location("reward_under_test", sacrifice_dir / "reward.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_evaluate_team_reward_calls_real_reward_module(reward_mod):
    d1_state = reward_mod.DriverChampionshipState("VER", wdc_gap=10, role_weight=0.65, title_secured=False)
    d2_state = reward_mod.DriverChampionshipState("PER", wdc_gap=50, role_weight=0.35, title_secured=False)
    team_state = reward_mod.TeamChampionshipState(wcc_gap=0.0, races_remaining=5, alpha=0.5)
    out = ts.evaluate_team_reward(d1_delta_points=7.0, d2_delta_points=-7.0,
                                   d1_champ_state=d1_state, d2_champ_state=d2_state,
                                   team_state=team_state, reward_mod=reward_mod)
    assert "R_team" in out and "L1" in out and "L2" in out

def test_risk_modes_can_change_which_strategy_ranks_first():

    high_reward_high_risk = dict(strategy_id="A", reward=dict(R_team=1.0), mc_risk=0.9)
    low_reward_low_risk = dict(strategy_id="B", reward=dict(R_team=0.5), mc_risk=0.0)
    ranked_conservative = ts.apply_risk_mode([high_reward_high_risk, low_reward_low_risk], "CONSERVATIVE")
    ranked_aggressive = ts.apply_risk_mode([high_reward_high_risk, low_reward_low_risk], "AGGRESSIVE")
    assert ranked_conservative[0]["strategy_id"] == "B"
    assert ranked_aggressive[0]["strategy_id"] == "A"

def test_build_strategic_explanation_only_uses_real_numbers_present_on_inputs():
    selected = dict(strategy_id="A|B", d1_action="PIT_NOW", d2_action="EXTEND_TO_CLIFF",
                     d1_code="VER", d2_code="PER",
                     d1_outcome=dict(time_delta_s=5.0, pit_lap=10),
                     d2_outcome=dict(time_delta_s=8.0, pit_lap=15),
                     reward=dict(R_team=0.42))
    role = dict(team_objective="default_driver_1_priority", reason="test")
    out = ts.build_strategic_explanation(selected, [], role)
    assert out["r_team"] == 0.42
    assert out["d1_time_delta_s"] == 5.0
    assert "0.4200" in out["plain_text"] or "0.42" in out["plain_text"]
    assert "%" not in out["plain_text"]

@pytest.fixture(scope="module")
def master_mod():

    import sys as _sys
    spec = _ilu.spec_from_file_location("master_under_test", TREES_DIR / "master.py")
    mod = _ilu.module_from_spec(spec)
    _sys.modules["master_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod

def test_end_to_end_team_strategy_wired_into_real_gate_and_execution_tree(master_mod):
    if master_mod.team_strategy is None or master_mod.reward_mod is None:
        pytest.skip("team-strategy.py or reward.py not loadable in this environment")

    repo_root = TREES_DIR.parent.parent.parent
    tyre_models = master_mod.load_tyre_models(master_mod._data_path(repo_root, "tyre_life_models.pkl"))
    sc_prior = master_mod.load_sc_prior(master_mod._data_path(repo_root, "sc_vsc_circuit_level_prior.csv"))
    cliff_stints = master_mod.load_cliff_stints(master_mod._data_path(repo_root, "cliff_detection_stints.csv"))
    taxonomy = master_mod.load_circuit_taxonomy(repo_root / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
    pit_loss_table = master_mod.load_pit_loss_table(
        master_mod._data_path(repo_root, "checkpoints", "rival_knowledge", "archive_per_race_analysis.csv"))
    degr_ordinal, _ = master_mod.circuit_degredation_ordinal_for(taxonomy, "Bahrain_Grand_Prix")
    pit_loss_s, _ = master_mod.pit_loss_for_circuit(pit_loss_table, "Bahrain_Grand_Prix", season=2023)

    laps_path = master_mod.find_laps_features(2023, "Bahrain_Grand_Prix", "R")
    if laps_path is None:
        pytest.skip("Bahrain 2023 laps_features.csv not cached in this environment")
    import pandas as pd
    total_laps = int(pd.read_csv(laps_path, usecols=["LapNumber"])["LapNumber"].max())

    ctx = master_mod.RaceContext(
        season=2023, circuit="Bahrain_Grand_Prix", total_laps=total_laps, is_sprint_weekend=False,
        d1_code="VER", d2_code="PER", session="R", circuit_degredation_ordinal=degr_ordinal,
        pit_loss_s=pit_loss_s, p_sc_5lap=master_mod.get_sc_probability(sc_prior, "Bahrain_Grand_Prix"),
    )
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}

    driver_taxonomy = master_mod.load_driver_taxonomy(
        master_mod._data_path(repo_root, "driver_taxonomy_master_final.csv"))

    champ_ctx = dict(available=True, d1_points=50.0, d2_points=45.0, round_num=1, races_remaining=20,
                      nearest_rival_code="HAM", d1_signed_gap_to_rival=10.0, d2_signed_gap_to_rival=5.0)
    team_strategy_fn = master_mod.build_team_strategy_context(
        ctx, resources, driver_taxonomy, champ_ctx, risk_mode="BALANCED", n_sims=20)

    decisions = master_mod.run_replay(ctx, resources, range(1, 6), team_strategy_fn=team_strategy_fn)

    assert len(decisions) > 0
    with_strategy = [d for d in decisions if d.get("team_strategy")]
    assert len(with_strategy) > 0, "team_strategy was never attached to a single decision - not genuinely wired"

    sample = with_strategy[0]["team_strategy"]

    assert sample["role"]["priority_driver_id"] in ("VER", "PER")
    assert sample["n_candidates_evaluated"] > 0
    assert "selected_strategy" in sample and "reward" in sample["selected_strategy"]
    assert "R_team" in sample["selected_strategy"]["reward"]
    assert "explanation" in sample and sample["explanation"]["plain_text"]

    for d in decisions:
        assert d["gate_decision"] in ("PIT_NOW", "PIT_FLEXIBLE", "PIT_LATER", "DONT_PIT")
        assert d["execution"]["driving_instruction"] in ("PIT_LAP", "MANAGE", "NEUTRAL", "PUSH", "CONSERVE")

def test_default_path_unaffected_when_team_strategy_flag_is_off(master_mod):
    ""
    repo_root = TREES_DIR.parent.parent.parent
    laps_path = master_mod.find_laps_features(2023, "Bahrain_Grand_Prix", "R")
    if laps_path is None:
        pytest.skip("Bahrain 2023 laps_features.csv not cached in this environment")
    tyre_models = master_mod.load_tyre_models(master_mod._data_path(repo_root, "tyre_life_models.pkl"))
    cliff_stints = master_mod.load_cliff_stints(master_mod._data_path(repo_root, "cliff_detection_stints.csv"))
    import pandas as pd
    total_laps = int(pd.read_csv(laps_path, usecols=["LapNumber"])["LapNumber"].max())
    ctx = master_mod.RaceContext(season=2023, circuit="Bahrain_Grand_Prix", total_laps=total_laps,
                                  is_sprint_weekend=False, d1_code="VER", d2_code="PER", session="R")
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    decisions = master_mod.run_replay(ctx, resources, range(1, 4))
    assert all("team_strategy" not in d for d in decisions)
    assert all("team_strategy_execution" not in d for d in decisions)

def test_resolve_strategy_execution_upgrades_pit_later_to_pit_now():
    ""
    out = ts.resolve_strategy_execution(
        dict(d1_action="PIT_NOW", d2_action="EXTEND_TO_CLIFF"),
        d1_gate_decision="PIT_LATER", d1_tier=None, d2_gate_decision="DONT_PIT", d2_tier=None)
    assert out["d1"]["should_upgrade_to_pit_now"] is True
    assert out["d1"]["executed_action"] == "PIT_NOW"
    assert out["d1"]["override"] is False
    assert out["d2"]["should_upgrade_to_pit_now"] is False
    assert out["d2"]["executed_action"] == "DONT_PIT"

def test_resolve_strategy_execution_does_not_suppress_an_active_tier3_pit_now():
    ""
    out = ts.resolve_strategy_execution(
        dict(d1_action="EXTEND_TO_CLIFF", d2_action="PIT_IN_3"),
        d1_gate_decision="DONT_PIT", d1_tier=None, d2_gate_decision="PIT_NOW", d2_tier=None)
    assert out["d2"]["override"] is True
    assert out["d2"]["executed_action"] == "PIT_NOW"
    assert "does not suppress" in out["d2"]["override_reason"]
    assert out["d2"]["should_upgrade_to_pit_now"] is False

def test_resolve_strategy_execution_never_touches_tier1_hard_safety():
    ""
    out = ts.resolve_strategy_execution(
        dict(d1_action="EXTEND_TO_CLIFF", d2_action="PIT_NOW"),
        d1_gate_decision="PIT_NOW", d1_tier=1, d2_gate_decision="PIT_NOW", d2_tier=1)
    assert out["d1"]["override"] is True
    assert "Tier 1 (hard safety)" in out["d1"]["override_reason"]
    assert out["d1"]["should_upgrade_to_pit_now"] is False

    assert out["d2"]["override"] is False

def test_resolve_strategy_execution_never_touches_tier2_regulatory():
    out = ts.resolve_strategy_execution(
        dict(d1_action="PIT_NOW", d2_action="PIT_NOW"),
        d1_gate_decision="PIT_FLEXIBLE", d1_tier=2, d2_gate_decision="PIT_FLEXIBLE", d2_tier=2)
    assert out["d1"]["should_upgrade_to_pit_now"] is False
    assert out["d1"]["override"] is False

def test_resolve_strategy_execution_no_op_when_already_aligned():
    out = ts.resolve_strategy_execution(
        dict(d1_action="EXTEND_TO_CLIFF", d2_action="EXTEND_TO_CLIFF"),
        d1_gate_decision="PIT_LATER", d1_tier=None, d2_gate_decision="DONT_PIT", d2_tier=None)
    assert out["d1"]["override"] is False and out["d1"]["should_upgrade_to_pit_now"] is False
    assert out["d2"]["override"] is False and out["d2"]["should_upgrade_to_pit_now"] is False

def test_rival_pit_window_adjustment_rewards_covering_and_penalises_exposure():
    cover = ts.rival_pit_window_adjustment(my_pit_lap=20, predicted_window=[24, 27], pit_loss_s=25.0)
    inside = ts.rival_pit_window_adjustment(my_pit_lap=25, predicted_window=[24, 27], pit_loss_s=25.0)
    exposed = ts.rival_pit_window_adjustment(my_pit_lap=30, predicted_window=[24, 27], pit_loss_s=25.0)
    assert cover < 0 < inside < exposed

def test_rival_window_changes_which_joint_candidate_ranks_first():
    ""
    d1 = ts.DriverSimInputs(
        code="VER", current_lap=20, tyre_age=22.0, compound="MEDIUM", expected_stint_length=30.0,
        pace_loss_fn=lambda age: 0.02 * age, cliff_probability_fn=lambda age: 0.001 * age,
        pit_loss_s=25.0, gap_ahead_s=5.0, gap_behind_s=5.0,
        mandatory_compound_done=True, alt_compound_available=False, profile=dict(ts.NEUTRAL_PROFILE))

    out = ts.simulate_driver_action(d1, "EXTEND_TO_CLIFF", total_laps=57)
    assert out["pit_lap"] == 28

    adj_soon = ts.rival_pit_window_adjustment(out["pit_lap"], [21, 23], d1.pit_loss_s)
    adj_late = ts.rival_pit_window_adjustment(out["pit_lap"], [45, 47], d1.pit_loss_s)
    assert adj_soon > 0 > adj_late
    assert adj_soon != adj_late

    cost_soon = round(out["time_delta_s"] + adj_soon, 3)
    cost_late = round(out["time_delta_s"] + adj_late, 3)
    assert cost_soon != cost_late

def test_end_to_end_selected_strategy_reaches_real_execution_tree(master_mod):
    ""
    if master_mod.team_strategy is None or master_mod.reward_mod is None:
        pytest.skip("team-strategy.py or reward.py not loadable in this environment")

    repo_root = TREES_DIR.parent.parent.parent
    tyre_models = master_mod.load_tyre_models(master_mod._data_path(repo_root, "tyre_life_models.pkl"))
    sc_prior = master_mod.load_sc_prior(master_mod._data_path(repo_root, "sc_vsc_circuit_level_prior.csv"))
    cliff_stints = master_mod.load_cliff_stints(master_mod._data_path(repo_root, "cliff_detection_stints.csv"))
    taxonomy = master_mod.load_circuit_taxonomy(repo_root / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
    pit_loss_table = master_mod.load_pit_loss_table(
        master_mod._data_path(repo_root, "checkpoints", "rival_knowledge", "archive_per_race_analysis.csv"))
    degr_ordinal, _ = master_mod.circuit_degredation_ordinal_for(taxonomy, "Bahrain_Grand_Prix")
    pit_loss_s, _ = master_mod.pit_loss_for_circuit(pit_loss_table, "Bahrain_Grand_Prix", season=2023)

    laps_path = master_mod.find_laps_features(2023, "Bahrain_Grand_Prix", "R")
    if laps_path is None:
        pytest.skip("Bahrain 2023 laps_features.csv not cached in this environment")
    import pandas as pd
    total_laps = int(pd.read_csv(laps_path, usecols=["LapNumber"])["LapNumber"].max())

    ctx = master_mod.RaceContext(
        season=2023, circuit="Bahrain_Grand_Prix", total_laps=total_laps, is_sprint_weekend=False,
        d1_code="VER", d2_code="PER", session="R", circuit_degredation_ordinal=degr_ordinal,
        pit_loss_s=pit_loss_s, p_sc_5lap=master_mod.get_sc_probability(sc_prior, "Bahrain_Grand_Prix"),
    )
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    driver_taxonomy = master_mod.load_driver_taxonomy(
        master_mod._data_path(repo_root, "driver_taxonomy_master_final.csv"))
    champ_ctx = dict(available=True, d1_points=50.0, d2_points=45.0, round_num=1, races_remaining=20,
                      nearest_rival_code="HAM", d1_signed_gap_to_rival=10.0, d2_signed_gap_to_rival=5.0)
    team_strategy_fn = master_mod.build_team_strategy_context(
        ctx, resources, driver_taxonomy, champ_ctx, risk_mode="BALANCED", n_sims=20)

    decisions = master_mod.run_replay(ctx, resources, range(1, 16), team_strategy_fn=team_strategy_fn)

    tse_rows = [d for d in decisions if d.get("team_strategy_execution")]
    assert len(tse_rows) > 0, "team_strategy_execution was never attached - not genuinely wired"

    with_strategy = [d for d in decisions if d.get("team_strategy")]
    sample_selected = with_strategy[0]["team_strategy"]["selected_strategy"]
    assert "d1_action" in sample_selected and "d2_action" in sample_selected

    upgraded = [d for d in tse_rows if d["team_strategy_execution"]["should_upgrade_to_pit_now"]]
    assert len(upgraded) > 0, "no upgrade case occurred in this window - cannot prove the connection"
    for d in upgraded:
        assert d["team_strategy_execution"]["override"] is False
        assert d["gate_decision"] == "PIT_NOW"

        assert d["execution"]["driving_instruction"] != "PUSH" or d["gate_decision"] != "PIT_NOW"
        assert d["gate_tree_trigger_tier"] == 3

    real_pit_now_rows = [d for d in decisions if d["tier_reached"] == 3 and d["gate_decision"] == "PIT_NOW"]
    assert len(real_pit_now_rows) > 0, "no real Tier-3 PIT_NOW occurred in this window to test against"
    real_row = real_pit_now_rows[0]
    forced = master_mod.team_strategy.resolve_strategy_execution(
        dict(d1_action="EXTEND_TO_CLIFF", d2_action="EXTEND_TO_CLIFF"),
        real_row["gate_decision"], real_row["gate_tree_trigger_tier"], "DONT_PIT", None,
    )["d1" if real_row["driver"] == ctx.d1_code else "d2"]
    assert forced["override"] is True
    assert forced["executed_action"] == "PIT_NOW"
    assert forced["override_reason"]

    overridden = [d for d in tse_rows if d["team_strategy_execution"]["override"]]
    if overridden:
        for d in overridden:
            assert d["team_strategy_execution"]["override_reason"]
            assert d["gate_decision"] == "PIT_NOW"

def _fixture_candidate(strategy_id, d1_action, d2_action, d1_pit_lap, d1_time_delta,
                        d1_degr=1.0, d1_pit_cost=0.0, d1_cliff=0.0, d1_dirty=0.0, d1_rival_adj=0.0,
                        r_team=0.5, risk_adjusted_score=0.5):
    d1o = dict(action=d1_action, pit_lap=d1_pit_lap, time_delta_s=d1_time_delta,
               degradation_cost_s=d1_degr, cliff_penalty_s=d1_cliff, pit_cost_s=d1_pit_cost,
               dirty_air_cost_s=d1_dirty, rival_window_adjustment_s=d1_rival_adj, confidence=1.0)
    d2o = dict(action=d2_action, pit_lap=d1_pit_lap, time_delta_s=d1_time_delta,
               degradation_cost_s=0.0, cliff_penalty_s=0.0, pit_cost_s=0.0,
               dirty_air_cost_s=0.0, confidence=1.0)
    return dict(strategy_id=strategy_id, d1_action=d1_action, d2_action=d2_action,
                d1_outcome=d1o, d2_outcome=d2o, reward=dict(R_team=r_team),
                risk_adjusted_score=risk_adjusted_score)

def test_trade_off_explanation_reports_what_and_why_from_selected_candidate():
    selected = _fixture_candidate("A", "PIT_NOW", "EXTEND_TO_CLIFF", d1_pit_lap=20, d1_time_delta=18.7,
                                   d1_pit_cost=18.7, r_team=0.6, risk_adjusted_score=0.55)
    role = dict(team_objective="default_driver_1_priority", reason="test")
    out = ts.build_decision_trade_off_explanation(selected, [], role, current_lap=20, risk_mode="BALANCED")
    assert out["selected_d1_action"] == "PIT_NOW"
    assert out["d1"]["pit_cost_s"] == 18.7
    assert out["d1"]["horizon_laps"] == 0
    assert out["risk_mode"] == "BALANCED"
    assert out["alternative"] is None
    assert out["net_advantage_s"] is None

def test_trade_off_explanation_what_if_alternative_and_net_advantage_are_commensurate():
    selected = _fixture_candidate("A", "PIT_NOW", "EXTEND_TO_CLIFF", d1_pit_lap=20, d1_time_delta=18.7,
                                   d1_pit_cost=18.7)
    alternative = _fixture_candidate("B", "EXTEND_TO_CLIFF", "EXTEND_TO_CLIFF", d1_pit_lap=25, d1_time_delta=21.1,
                                      d1_degr=21.1)
    role = dict(team_objective="default_driver_1_priority", reason="test")
    out = ts.build_decision_trade_off_explanation(selected, [alternative], role, current_lap=20, risk_mode="BALANCED")
    assert out["alternative"]["d1_action"] == "EXTEND_TO_CLIFF"
    assert out["alternative"]["horizon_laps"] == 5

    assert out["net_advantage_s"] == round(21.1 - 18.7, 3)
    assert out["comparison"]["immediate_pit_cost_s"]["selected"] == 18.7
    assert out["comparison"]["immediate_pit_cost_s"]["alternative"] == 0.0

def test_trade_off_explanation_rival_effect_surfaces_when_present():
    selected = _fixture_candidate("A", "PIT_NOW", "EXTEND_TO_CLIFF", d1_pit_lap=20, d1_time_delta=15.0,
                                   d1_pit_cost=18.7, d1_rival_adj=-2.5)
    role = dict(team_objective="default_driver_1_priority", reason="test")
    out = ts.build_decision_trade_off_explanation(selected, [], role, current_lap=20, risk_mode="AGGRESSIVE")
    assert out["d1"]["rival_window_adjustment_s"] == -2.5

def test_trade_off_explanation_carries_execution_override_status():
    selected = _fixture_candidate("A", "EXTEND_TO_CLIFF", "EXTEND_TO_CLIFF", d1_pit_lap=25, d1_time_delta=5.0)
    role = dict(team_objective="default_driver_1_priority", reason="test")
    exec_d1 = dict(team_strategy_action="EXTEND_TO_CLIFF", executed_action="PIT_NOW", override=True,
                    override_reason="Tier 3 already triggered PIT_NOW", should_upgrade_to_pit_now=False)
    out = ts.build_decision_trade_off_explanation(selected, [], role, current_lap=20, risk_mode="BALANCED",
                                                    execution_d1=exec_d1, execution_d2=None)
    assert out["execution"]["d1"]["override"] is True
    assert out["execution"]["d1"]["executed_action"] == "PIT_NOW"

def test_trade_off_explanation_never_fabricates_a_field_not_present_on_inputs():
    ""
    selected = _fixture_candidate("A", "PIT_NOW", "EXTEND_TO_CLIFF", d1_pit_lap=20, d1_time_delta=18.7,
                                   d1_pit_cost=18.7, d1_degr=0.0)
    alternative = _fixture_candidate("B", "EXTEND_TO_CLIFF", "EXTEND_TO_CLIFF", d1_pit_lap=23, d1_time_delta=9.4,
                                      d1_degr=9.4)
    role = dict(team_objective="default_driver_1_priority", reason="test")
    out = ts.build_decision_trade_off_explanation(selected, [alternative], role, current_lap=20, risk_mode="BALANCED")
    assert out["comparison"]["degradation_cost_s"]["selected"] == selected["d1_outcome"]["degradation_cost_s"]
    assert out["comparison"]["degradation_cost_s"]["alternative"] == alternative["d1_outcome"]["degradation_cost_s"]
    assert out["team_reward_r_team"] == selected["reward"]["R_team"]

def test_build_team_strategy_context_risk_mode_override_changes_result_risk_mode(master_mod):
    ""
    if master_mod.team_strategy is None or master_mod.reward_mod is None:
        pytest.skip("team-strategy.py or reward.py not loadable in this environment")

    repo_root = TREES_DIR.parent.parent.parent
    tyre_models = master_mod.load_tyre_models(master_mod._data_path(repo_root, "tyre_life_models.pkl"))
    cliff_stints = master_mod.load_cliff_stints(master_mod._data_path(repo_root, "cliff_detection_stints.csv"))
    laps_path = master_mod.find_laps_features(2023, "Bahrain_Grand_Prix", "R")
    if laps_path is None:
        pytest.skip("Bahrain 2023 laps_features.csv not cached in this environment")
    import pandas as pd
    laps = pd.read_csv(laps_path, dtype={"TrackStatus": str})
    total_laps = int(laps["LapNumber"].max())

    ctx = master_mod.RaceContext(season=2023, circuit="Bahrain_Grand_Prix", total_laps=total_laps,
                                  is_sprint_weekend=False, d1_code="VER", d2_code="PER", session="R",
                                  circuit_degredation_ordinal=2, pit_loss_s=25.0, p_sc_5lap=None)
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    champ_ctx = dict(available=True, d1_points=50.0, d2_points=45.0, round_num=1, races_remaining=20,
                      nearest_rival_code="HAM", d1_signed_gap_to_rival=10.0, d2_signed_gap_to_rival=5.0)

    team_strategy_fn = master_mod.build_team_strategy_context(
        ctx, resources, None, champ_ctx, risk_mode="BALANCED", n_sims=20)

    decisions = master_mod.run_replay(ctx, resources, range(10, 12))
    r_d1 = next(d for d in decisions if d["driver"] == "VER" and d["lap"] == 10)
    r_d2 = next(d for d in decisions if d["driver"] == "PER" and d["lap"] == 10)
    lap_df = laps[laps["LapNumber"] == 10]
    lap_df = master_mod._rank_by_gap_to_leader(lap_df.copy())

    result_default = team_strategy_fn(r_d1, r_d2, lap_df, laps, 10)
    result_conservative = team_strategy_fn(r_d1, r_d2, lap_df, laps, 10, risk_mode_override="CONSERVATIVE")
    result_aggressive = team_strategy_fn(r_d1, r_d2, lap_df, laps, 10, risk_mode_override="AGGRESSIVE")

    assert result_default["risk_mode"] == "BALANCED"
    assert result_conservative["risk_mode"] == "CONSERVATIVE"
    assert result_aggressive["risk_mode"] == "AGGRESSIVE"

    assert result_conservative["selected_strategy"]["risk_mode"] == "CONSERVATIVE"
    assert result_aggressive["selected_strategy"]["risk_mode"] == "AGGRESSIVE"
