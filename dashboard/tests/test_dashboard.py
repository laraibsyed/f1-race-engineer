""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

import hermes_adapter as ha

SEASON, RACE, SESSION = 2023, "Bahrain_Grand_Prix", "R"
D1, D2 = "VER", "PER"

@pytest.fixture(scope="module")
def bundle():
    return ha.load_race_bundle(SEASON, RACE, SESSION)

@pytest.fixture()
def dpc(bundle):
    return ha.build_driver_pair_context(bundle, D1, D2)

def test_race_loading(bundle):
    assert bundle.total_laps > 0
    assert not bundle.laps.empty
    assert {"Driver", "LapNumber", "Compound", "TyreLife"}.issubset(bundle.laps.columns)

def test_driver_listing_full_grid(bundle):
    drivers = ha.list_drivers(bundle)
    codes = [d["code"] for d in drivers]
    assert len(codes) >= 15
    assert D1 in codes and D2 in codes

def test_lap_navigation_forward_and_back(bundle, dpc):
    cache = ha.ReplayCache(bundle, dpc)
    d10 = cache.get(10)
    d5 = cache.get(5)
    d10_again = cache.get(10)
    assert d10[D1]["gate_decision"] == d10_again[D1]["gate_decision"]
    assert d5[D1]["lap"] == 5
    assert d10[D1]["lap"] == 10

def test_replay_cache_reproducible(bundle, dpc):
    ""
    dpc2 = ha.build_driver_pair_context(bundle, D1, D2)
    cache1 = ha.ReplayCache(bundle, dpc)
    cache2 = ha.ReplayCache(bundle, dpc2)
    for lap in (3, 12, 25):
        r1, r2 = cache1.get(lap), cache2.get(lap)
        assert r1[D1]["gate_decision"] == r2[D1]["gate_decision"]
        assert r1[D1]["tier_reached"] == r2[D1]["tier_reached"]

def test_sc_scenario_does_not_mutate_source_laps(bundle, dpc):
    before = bundle.laps.copy(deep=True)
    cache = ha.ReplayCache(bundle, dpc)
    scen = ha.ScenarioReplayCache(cache, 15, dict(TrackStatus=ha.TRACK_STATUS_SC, is_sc_lap=True), 3)
    scen.get(17)
    pd.testing.assert_frame_equal(before, bundle.laps)

def test_weather_scenario_does_not_mutate_source_laps(bundle, dpc):
    before = bundle.laps.copy(deep=True)
    cache = ha.ReplayCache(bundle, dpc)
    scen = ha.ScenarioReplayCache(cache, 20, ha.WEATHER_PRESETS["HEAVY_RAIN"], None)
    scen.get(22)
    pd.testing.assert_frame_equal(before, bundle.laps)

def test_sc_scenario_changes_decision(bundle, dpc):
    ""
    cache = ha.ReplayCache(bundle, dpc)
    injection_lap = 15
    hist = cache.get(injection_lap)
    scen = ha.ScenarioReplayCache(cache, injection_lap,
                                   dict(TrackStatus=ha.TRACK_STATUS_SC, is_sc_lap=True), 3)
    scenario_result = scen.get(injection_lap)
    changed = any(
        scenario_result[code]["gate_decision"] != hist[code]["gate_decision"]
        or bool((scenario_result[code].get("triggers") or {}).get("safety_car"))
        != bool((hist[code].get("triggers") or {}).get("safety_car"))
        for code in (D1, D2) if code in scenario_result and code in hist
    )
    assert changed, "SC scenario produced no observable change in either driver's decision/triggers"

def test_scenario_reset_does_not_affect_historical_cache(bundle, dpc):
    cache = ha.ReplayCache(bundle, dpc)
    hist_before = cache.get(15)
    scen = ha.ScenarioReplayCache(cache, 15, dict(TrackStatus=ha.TRACK_STATUS_SC, is_sc_lap=True), 3)
    scen.get(16)
    hist_after = cache.get(15)
    assert hist_before[D1]["gate_decision"] == hist_after[D1]["gate_decision"]
    assert hist_before[D1]["tyre_age"] == hist_after[D1]["tyre_age"]

def test_reset_returns_identical_state(bundle):
    dpc_a = ha.build_driver_pair_context(bundle, D1, D2)
    cache_a = ha.ReplayCache(bundle, dpc_a)
    first_pass = {lap: cache_a.get(lap) for lap in range(1, 21)}

    dpc_b = ha.build_driver_pair_context(bundle, D1, D2)
    cache_b = ha.ReplayCache(bundle, dpc_b)
    second_pass = {lap: cache_b.get(lap) for lap in range(1, 21)}

    for lap in range(1, 21):
        assert first_pass[lap][D1]["gate_decision"] == second_pass[lap][D1]["gate_decision"]
        assert first_pass[lap][D1]["compound"] == second_pass[lap][D1]["compound"]

def test_no_lookahead_full_grid(bundle):
    ""
    for lap in (1, 10, 30):
        grid = ha.full_grid_for_lap(bundle, lap)
        if not grid.empty:
            assert set(grid["LapNumber"].unique()) == {lap}

def test_track_map_genuine_telemetry_or_honest_unavailable(bundle):
    ""
    sys.path.insert(0, str(REPO_ROOT / "dashboard"))
    import track_map as tmap
    telemetry = tmap.load_telemetry(bundle.season, bundle.race, bundle.session)
    if telemetry is None:
        pytest.skip("No FastF1 telemetry available in this environment for this race - "
                    "honest None return, not a failure of the fallback contract.")
    assert len(telemetry["outline_x"]) >= 10
    assert len(telemetry["outline_x"]) == len(telemetry["outline_y"])
    assert "np_tracks" in telemetry and len(telemetry["np_tracks"]) > 0
    assert D1 in telemetry["np_tracks"] and D2 in telemetry["np_tracks"]
    for code, (t, x, y) in telemetry["np_tracks"].items():
        assert len(t) == len(x) == len(y) > 0
    window = tmap.lap_window_for(telemetry, 10, [D1, D2])
    assert window is not None and window[1] > window[0]

def test_scenario_injection_at_lap_1_does_not_crash(bundle, dpc):
    ""
    cache = ha.ReplayCache(bundle, dpc)
    scen = ha.ScenarioReplayCache(cache, 1, dict(TrackStatus=ha.TRACK_STATUS_SC, is_sc_lap=True), 3)
    result = scen.get(1)
    assert D1 in result and D2 in result

def test_scenario_duration_actually_ends(bundle, dpc):
    ""
    cache = ha.ReplayCache(bundle, dpc)
    injection_lap, duration = 10, 2
    scen = ha.ScenarioReplayCache(cache, injection_lap, dict(TrackStatus=ha.TRACK_STATUS_SC, is_sc_lap=True), duration)
    end_lap = injection_lap + duration - 1
    assert scen.end_lap == end_lap
    after_lap = end_lap + 1
    assert scen._overrides_for(after_lap) is None, "override must not still apply past its own duration"

    scen_dec = scen.get(after_lap)
    hist_dec = cache.get(after_lap)
    assert scen_dec[D1]["gate_decision"] == hist_dec[D1]["gate_decision"]
    assert (scen_dec[D1].get("triggers") or {}).get("safety_car") == (hist_dec[D1].get("triggers") or {}).get("safety_car")

def _offline_champ_ctx(*_args, **_kwargs):
    ""
    return dict(available=True, d1_points=50.0, d2_points=45.0, round_num=1, races_remaining=20,
                nearest_rival_code="HAM", d1_signed_gap_to_rival=10.0, d2_signed_gap_to_rival=5.0)

@pytest.fixture()
def dpc_with_team_strategy(bundle, monkeypatch):
    ""
    import system.HERMES.trees.master as master
    if master.team_strategy is None or master.reward_mod is None:
        pytest.skip("team-strategy.py or reward.py not loadable in this environment")
    monkeypatch.setattr(master, "fetch_championship_context", _offline_champ_ctx)
    ha._TEAM_STRATEGY_CACHE.clear()
    dpc = ha.build_driver_pair_context(bundle, D1, D2)
    assert dpc.team_strategy_fn is not None, f"team-strategy unavailable: {dpc.team_strategy_unavailable_reason}"
    return dpc

def test_team_strategy_attached_to_decision_when_available(bundle, dpc_with_team_strategy):
    cache = ha.ReplayCache(bundle, dpc_with_team_strategy)
    dec = cache.get(10)
    assert dec[D1].get("team_strategy") is not None
    selected = dec[D1]["team_strategy"]["selected_strategy"]
    assert "d1_action" in selected and "d2_action" in selected

def test_team_strategy_execution_present_and_consistent_with_gate_decision(bundle, dpc_with_team_strategy):
    cache = ha.ReplayCache(bundle, dpc_with_team_strategy)
    dec = cache.get(10)
    exec_info = dec[D1].get("team_strategy_execution")
    assert exec_info is not None
    assert exec_info["executed_action"] == dec[D1]["gate_decision"]

def test_risk_mode_control_genuinely_changes_reported_risk_mode(bundle, dpc_with_team_strategy):
    ""
    cache = ha.ReplayCache(bundle, dpc_with_team_strategy)
    cache.set_risk_mode("CONSERVATIVE")
    dec_conservative = cache.get(10)
    assert dec_conservative[D1]["team_strategy"]["risk_mode"] == "CONSERVATIVE"

    cache.set_risk_mode("AGGRESSIVE")
    dec_aggressive = cache.get(10)
    assert dec_aggressive[D1]["team_strategy"]["risk_mode"] == "AGGRESSIVE"

def test_risk_mode_change_only_clears_decision_cache_not_state(bundle, dpc_with_team_strategy):
    ""
    cache = ha.ReplayCache(bundle, dpc_with_team_strategy)
    cache.get(10)
    assert 10 in cache._state_after
    cache.set_risk_mode("AGGRESSIVE")
    assert 10 in cache._state_after
    assert 10 not in cache._decisions

def test_trade_off_explanation_available_and_commensurate(bundle, dpc_with_team_strategy):
    cache = ha.ReplayCache(bundle, dpc_with_team_strategy)
    dec = cache.get(10)
    trade_off = dec[D1]["team_strategy"].get("trade_off_explanation")
    assert trade_off is not None
    assert trade_off["selected_d1_action"] == dec[D1]["team_strategy"]["selected_strategy"]["d1_action"]
    assert trade_off["d1"]["horizon_laps"] >= 0

def test_team_strategy_disabled_when_module_unavailable_degrades_gracefully(bundle):
    ""
    dpc_no_ts = ha.build_driver_pair_context(bundle, D1, D2)
    dpc_no_ts.team_strategy_fn = None
    cache = ha.ReplayCache(bundle, dpc_no_ts)
    dec = cache.get(10)
    assert dec[D1].get("team_strategy") is None
    assert dec[D1].get("team_strategy_execution") is None
    assert "gate_decision" in dec[D1]

def test_red_bull_pair_resolution(bundle):
    ""
    d1, d2 = ha.resolve_red_bull_pair(SEASON, RACE)
    assert (d1, d2) == (D1, D2)
    teams = bundle.laps.loc[bundle.laps["Driver"].isin([d1, d2]), "Team"].unique()
    assert all(t == "Red Bull Racing" for t in teams)
