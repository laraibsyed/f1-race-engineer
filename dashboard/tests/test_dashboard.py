"""
HERMES Pit Wall - dashboard tests (spec item 26).
Exercises the adapter/data layer directly (no Streamlit server needed) against
a real, small, fast-to-load race. Covers the feasible subset of the spec's
12-item testing list:
  1. Race loading                          -> test_race_loading
  2. Lap navigation                         -> test_lap_navigation_forward_and_back
  3. Driver selection                       -> test_driver_listing_full_grid
  4. Track coordinate rendering             -> test_track_map_renders_with_fallback_label
  5. Historical state reproduction          -> test_replay_cache_reproducible
  6. Scenario reset                         -> test_scenario_reset_does_not_affect_historical_cache
  7. SC scenario does not mutate historical data -> test_sc_scenario_does_not_mutate_source_laps
  8. Weather scenario does not mutate historical data -> test_weather_scenario_does_not_mutate_source_laps
  9. Scenario changes are actually passed to HERMES -> test_sc_scenario_changes_decision
  10. Reset returns to identical historical state -> test_reset_returns_identical_state
  11. No-lookahead at selected lap           -> test_no_lookahead_full_grid
  12. Existing HERMES selftest still passes  -> run manually: `python system/HERMES/trees/master.py selftest`
      (not re-invoked here to avoid duplicating master.py's own ~2000-line regression suite in a
      dashboard test file; run it as its own step, see README note in app.py docstring).

Run:  .venv\\Scripts\\python.exe -m pytest dashboard/tests/test_dashboard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

import hermes_adapter as ha  # noqa: E402

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
    assert len(codes) >= 15          # full grid, not just the tracked pair
    assert D1 in codes and D2 in codes


def test_lap_navigation_forward_and_back(bundle, dpc):
    cache = ha.ReplayCache(bundle, dpc)
    d10 = cache.get(10)
    d5 = cache.get(5)   # jump backward after having computed forward
    d10_again = cache.get(10)
    assert d10[D1]["gate_decision"] == d10_again[D1]["gate_decision"]
    assert d5[D1]["lap"] == 5
    assert d10[D1]["lap"] == 10


def test_replay_cache_reproducible(bundle, dpc):
    """Same lap evaluated via two independently-built caches from the same
    bundle/driver pair must agree - HERMES's own decision is deterministic."""
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
    """The scenario must actually be passed into the real HERMES pipeline -
    verified structurally: with SC injected, at least one driver's `triggers`
    dict must report safety_car True where the historical run did not, on
    some lap in the injected window (this will legitimately fail only if the
    real Gate Tree logic itself would never let SC matter here, which is not
    the case for a mid-race injection on a green-flag lap)."""
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
    scen.get(16)   # advance the scenario branch
    hist_after = cache.get(15)   # historical cache must be untouched by the scenario branch existing
    assert hist_before[D1]["gate_decision"] == hist_after[D1]["gate_decision"]
    assert hist_before[D1]["tyre_age"] == hist_after[D1]["tyre_age"]


def test_reset_returns_identical_state(bundle):
    dpc_a = ha.build_driver_pair_context(bundle, D1, D2)
    cache_a = ha.ReplayCache(bundle, dpc_a)
    first_pass = {lap: cache_a.get(lap) for lap in range(1, 21)}

    # simulate "RESET RACE": build a brand-new pair context + cache
    dpc_b = ha.build_driver_pair_context(bundle, D1, D2)
    cache_b = ha.ReplayCache(bundle, dpc_b)
    second_pass = {lap: cache_b.get(lap) for lap in range(1, 21)}

    for lap in range(1, 21):
        assert first_pass[lap][D1]["gate_decision"] == second_pass[lap][D1]["gate_decision"]
        assert first_pass[lap][D1]["compound"] == second_pass[lap][D1]["compound"]


def test_no_lookahead_full_grid(bundle):
    """full_grid_for_lap must only ever return rows for the requested lap
    number - never a future lap's rows alongside it."""
    for lap in (1, 10, 30):
        grid = ha.full_grid_for_lap(bundle, lap)
        if not grid.empty:
            assert set(grid["LapNumber"].unique()) == {lap}


def test_track_map_renders_with_fallback_label(bundle):
    sys.path.insert(0, str(REPO_ROOT / "dashboard"))
    import track_map as tmap
    grid = ha.full_grid_for_lap(bundle, 10)
    fig, is_real = tmap.build_outline(bundle.season, bundle.race)[2], None
    outline_x, outline_y, is_real = tmap.build_outline(bundle.season, bundle.race)
    assert len(outline_x) >= 3
    # Bahrain 2023 is not in the 2018-only corners dataset -> must honestly fall back
    assert is_real is False
