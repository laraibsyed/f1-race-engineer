"""
HERMES Pit Wall - adapter layer.
=================================
READ-ONLY with respect to HERMES's own source. This file never edits, monkeypatches,
or redesigns any tree/module file under system/HERMES/ - it only imports master.py
(and evaluate.py, for its two small race/driver-listing helpers) exactly as they
already exist, and calls their already-public functions:
    master.evaluate_driver_lap, master.merge_execution, master.explanation_for,
    master.explanation_text_for, master._rank_by_gap_to_leader, master._row_or_none,
    master.load_tyre_models, master.load_sc_prior, master.load_cliff_stints,
    master.load_circuit_taxonomy, master.load_pit_loss_table,
    master.circuit_degredation_ordinal_for, master.pit_loss_for_circuit,
    master.get_sc_probability, master.load_weather, master._attach_weather_columns,
    master.find_laps_features, master.build_radios_by_driver_lap,
    master.RaceContext, master.DriverRuntimeState.

SCENARIO INJECTION (safety car / VSC / weather): implemented exactly the way the
pre-build inventory of this codebase confirmed is already supported - by handing
evaluate_driver_lap a COPY of the affected lap's row(s) with specific columns
overridden (TrackStatus, is_sc_lap, is_vsc_lap, Rainfall, _seconds_since_rain_end).
Nothing here ever mutates RaceBundle.laps (the historical dataframe loaded once per
race) - every override happens on a `.copy()` taken fresh for that one evaluation.
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TREES_DIR = REPO_ROOT / "system" / "HERMES" / "trees"
if str(TREES_DIR) not in sys.path:
    sys.path.insert(0, str(TREES_DIR))

# master.py resolves its OWN repo root from HERMES_REPO_ROOT (env var, default ".") to find its
# hyphenated sibling tree/module files (gate-tier-1.py etc, loaded via its own load_module()) - a
# Streamlit process's working directory is not guaranteed to be this repo's root, so this must be
# set BEFORE `import master` runs its module-level load_module() calls. Read-only: does not touch
# any HERMES source file, just tells master.py where itself already lives on disk.
os.environ.setdefault("HERMES_REPO_ROOT", str(REPO_ROOT))

import master  # noqa: E402 - the existing, unmodified HERMES orchestrator
import evaluate as hermes_eval  # noqa: E402 - reused: list_races (race listing only)

# F1 TrackStatus codes used by laps_features.csv / master.py (see master.py line 1128:
# safety_car_deployed = "4" in track_status or "6" in track_status or "7" in track_status).
# 1=green, 2=yellow, 4=SC, 5=red, 6=VSC deployed, 7=VSC ending.
TRACK_STATUS_GREEN = "1"
TRACK_STATUS_SC = "4"
TRACK_STATUS_VSC = "6"

WEATHER_PRESETS = {
    "NO_RAIN": dict(Rainfall=False, rain_probability_pct=0.0),
    "LIGHT_RAIN": dict(Rainfall=True, rain_probability_pct=35.0),
    "MODERATE_RAIN": dict(Rainfall=True, rain_probability_pct=65.0),
    "HEAVY_RAIN": dict(Rainfall=True, rain_probability_pct=90.0),
    "DRYING_TRACK": dict(Rainfall=False, rain_probability_pct=10.0, _seconds_since_rain_end=300.0),
}


# ============================================================================
# Race-level data bundle - built ONCE per "Load Race", cached by the caller
# (data_loader.py wraps this in st.cache_resource so re-selecting the same
# race/season does not reload laps_features.csv / weather / model pickles).
# ============================================================================
@dataclass
class RaceBundle:
    season: int
    race: str
    session: str
    laps: pd.DataFrame            # FULL grid (every driver), historical, weather columns attached
    weather_df: Optional[pd.DataFrame]
    total_laps: int
    resources: dict               # {"tyre_models": ..., "cliff_stints": ...}
    degr_ordinal: int
    pit_loss_s: float
    p_sc_5lap: Optional[float]
    fallback_notes: list          # data-quality / fallback notes surfaced to the UI


def list_seasons(repo_root: Path = REPO_ROOT) -> list[int]:
    base = repo_root / "gcs_cache" / "clean" / "features"
    if not base.exists():
        return []
    return sorted((int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()), reverse=True)


def list_races(repo_root: Path, season: int) -> list[str]:
    """Reused verbatim from evaluate.py - do NOT hardcode a race list."""
    return hermes_eval.list_races(repo_root, season)


def list_drivers(bundle: RaceBundle) -> list[dict]:
    """Every driver actually in this race (not just a hardcoded two-driver pair),
    used to populate the D1/D2 dropdowns and the timing tower. Code + team,
    sorted by median race position (best first) so the dropdown is meaningfully
    ordered, falling back to alphabetical if Position is unusable."""
    laps = bundle.laps
    if "Position" in laps.columns and laps["Position"].notna().any():
        order = laps.groupby("Driver")["Position"].median().sort_values()
        codes = list(order.index)
    else:
        codes = sorted(laps["Driver"].dropna().unique())
    out = []
    for code in codes:
        team = laps.loc[laps["Driver"] == code, "Team"].dropna()
        out.append(dict(code=code, team=team.iloc[0] if len(team) else "?"))
    return out


def load_race_bundle(season: int, race: str, session: str = "R", repo_root: Path = REPO_ROOT) -> RaceBundle:
    """Mirrors exactly what master.py's own `replay` CLI does in main() before
    calling run_replay() - same functions, same order (master.py lines
    1952-1970) - so the dashboard's inputs to HERMES are identical to the
    validated CLI/evaluation path. Called ONCE per race; the dashboard then
    drives its own lap-by-lap loop via ReplayCache below instead of
    master.py's batch `for lap_number in lap_range`."""
    notes: list = []

    tyre_models = master.load_tyre_models(master._data_path(repo_root, "tyre_life_models.pkl"))
    sc_prior = master.load_sc_prior(master._data_path(repo_root, "sc_vsc_circuit_level_prior.csv"))
    cliff_stints = master.load_cliff_stints(master._data_path(repo_root, "cliff_detection_stints.csv"))
    taxonomy = master.load_circuit_taxonomy(repo_root / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
    pit_loss_table = master.load_pit_loss_table(
        master._data_path(repo_root, "checkpoints", "rival_knowledge", "archive_per_race_analysis.csv"))

    if tyre_models is None:
        notes.append("tyre_life_models.pkl not found - tyre-life projection/cliff probability will "
                      "be unavailable this session (HERMES degrades gracefully, not a crash).")

    degr_ordinal, degr_note = master.circuit_degredation_ordinal_for(taxonomy, race)
    pit_loss_s, pit_loss_note = master.pit_loss_for_circuit(pit_loss_table, race, season=season)
    p_sc_5lap = master.get_sc_probability(sc_prior, race)
    notes.append(f"circuit_degredation_ordinal={degr_ordinal} ({degr_note})")
    notes.append(f"pit_loss_s={pit_loss_s} ({pit_loss_note})")
    if p_sc_5lap is None:
        notes.append("No circuit-level SC/VSC prior for this race - SC gamble will report "
                      "INSUFFICIENT_DATA unless a scenario override supplies a hypothetical figure.")

    weather_df = master.load_weather(season, race, session)
    laps_path = master.find_laps_features(season, race, session)
    if laps_path is None:
        raise FileNotFoundError(
            f"No laps_features.csv for season={season} race={race} session={session} under "
            f"{repo_root}/gcs_cache/clean/features/ - this race is not available to replay.")
    laps = pd.read_csv(laps_path, dtype={"TrackStatus": str})
    if "LapTime_seconds" not in laps.columns and "LapTime" in laps.columns:
        laps["LapTime_seconds"] = pd.to_timedelta(laps["LapTime"], errors="coerce").dt.total_seconds()
    laps = master._attach_weather_columns(laps, weather_df)   # HISTORICAL only; demo overrides never touch this frame
    total_laps = int(laps["LapNumber"].max())

    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    return RaceBundle(season, race, session, laps, weather_df, total_laps, resources,
                       degr_ordinal, pit_loss_s, p_sc_5lap, notes)


# ============================================================================
# Per-driver-pair evaluation context + single-lap evaluation
# ============================================================================
@dataclass
class DriverPairContext:
    ctx: "master.RaceContext"
    state_d1: "master.DriverRuntimeState"
    state_d2: "master.DriverRuntimeState"
    radios_by_driver_lap: Optional[dict]


def build_driver_pair_context(bundle: RaceBundle, d1: str, d2: str,
                               is_sprint_weekend: bool = False) -> DriverPairContext:
    ctx = master.RaceContext(
        season=bundle.season, circuit=bundle.race, total_laps=bundle.total_laps,
        is_sprint_weekend=is_sprint_weekend, d1_code=d1, d2_code=d2, session=bundle.session,
        circuit_degredation_ordinal=bundle.degr_ordinal, pit_loss_s=bundle.pit_loss_s,
        p_sc_5lap=bundle.p_sc_5lap,
    )
    radios = master.build_radios_by_driver_lap(REPO_ROOT, bundle.laps, ctx)
    return DriverPairContext(ctx, master.DriverRuntimeState("D1", d1),
                              master.DriverRuntimeState("D2", d2), radios)


def _snapshot_states(dpc: DriverPairContext):
    return copy.deepcopy(dpc.state_d1), copy.deepcopy(dpc.state_d2)


def _restore_states(dpc: DriverPairContext, snapshot) -> None:
    dpc.state_d1, dpc.state_d2 = copy.deepcopy(snapshot[0]), copy.deepcopy(snapshot[1])


def evaluate_lap(bundle: RaceBundle, dpc: DriverPairContext, lap_number: int,
                  lap_overrides: Optional[dict] = None) -> dict:
    """Evaluate ONE lap for the tracked D1/D2 pair, mutating dpc's
    DriverRuntimeState objects forward - the exact same functions, in the
    exact same order, that master.run_replay()'s own loop uses (evaluate_driver_lap
    for d1, then d2, then merge_execution). MUST be called in strictly
    increasing lap order for a given `dpc` object, because DriverRuntimeState
    is lap-to-lap accumulating state (same contract master.py already has) -
    ReplayCache below is what makes arbitrary UI navigation (jump back, jump
    forward, play/pause) safe on top of that constraint.

    `lap_overrides`: the DEMO/SIMULATED SCENARIO injection point. A dict of
    {column_name: value} applied to a **copy** of this lap's row(s) for BOTH
    drivers before evaluation - e.g. {"TrackStatus": "4", "is_sc_lap": True}
    for a Safety Car scenario. Never mutates `bundle.laps`. Returns
    {driver_code: decision_dict} using HERMES's own, unmodified decision-row
    schema (gate_decision, tier_reached, triggers, sc_gamble, execution,
    explanation, explanation_text, ...).
    """
    laps = bundle.laps
    lap_df = laps[laps["LapNumber"] == lap_number]
    if lap_df.empty:
        return {}
    lap_df = master._rank_by_gap_to_leader(lap_df)

    is_scenario = bool(lap_overrides)
    if is_scenario:
        lap_df = lap_df.copy()
        for col, val in lap_overrides.items():
            lap_df[col] = val

    row_d1 = master._row_or_none(lap_df, dpc.ctx.d1_code)
    row_d2 = master._row_or_none(lap_df, dpc.ctx.d2_code)

    track_status = str(lap_df["TrackStatus"].iloc[0]) if not lap_df.empty else ""
    safety_car_active = any(code in track_status for code in ("4", "6", "7"))

    lap_results, dq_notes = [], []
    if row_d1 is not None:
        lap_results.append(master.evaluate_driver_lap(
            dpc.ctx, dpc.state_d1, row_d1, lap_df, bundle.resources,
            dpc.radios_by_driver_lap, dq_notes, laps))
    if row_d2 is not None:
        lap_results.append(master.evaluate_driver_lap(
            dpc.ctx, dpc.state_d2, row_d2, lap_df, bundle.resources,
            dpc.radios_by_driver_lap, dq_notes, laps))

    master.merge_execution(dpc.ctx, lap_results, safety_car_active)

    out = {}
    for r in lap_results:
        r["data_quality_notes"] = list(dq_notes)
        r["explanation"] = master.explanation_for(r)
        r["explanation_text"] = master.explanation_text_for(r)
        actual_row = row_d1 if r["driver"] == dpc.ctx.d1_code else row_d2
        r["actual_is_pit_in_lap"] = bool(actual_row.get("is_pit_in")) if actual_row is not None else None
        r["is_scenario"] = is_scenario
        out[r["driver"]] = r
    return out


# ============================================================================
# Replay cache - avoids re-walking the whole race from lap 1 on every UI
# interaction (spec item 22: "cache replay decisions"). Historical laps are
# cached forward-only, exactly as they were computed; a scenario branch is
# cached SEPARATELY, starting from a snapshot of driver state immediately
# before the injection lap, and is discarded on "RESET SCENARIO" without
# touching the historical cache at all.
# ============================================================================
class ReplayCache:
    def __init__(self, bundle: RaceBundle, dpc: DriverPairContext):
        self.bundle = bundle
        self.dpc = dpc
        self._decisions: dict[int, dict] = {}
        self._state_after: dict[int, tuple] = {0: _snapshot_states(dpc)}

    def get(self, lap_number: int) -> dict:
        if lap_number in self._decisions:
            return self._decisions[lap_number]
        last_cached = max((l for l in self._state_after if l < lap_number), default=0)
        _restore_states(self.dpc, self._state_after[last_cached])
        for lap in range(last_cached + 1, lap_number + 1):
            dec = evaluate_lap(self.bundle, self.dpc, lap)
            self._decisions[lap] = dec
            self._state_after[lap] = _snapshot_states(self.dpc)
        return self._decisions[lap_number]

    def get_range(self, lo: int, hi: int) -> dict[int, dict]:
        return {lap: self.get(lap) for lap in range(lo, hi + 1)}


class ScenarioReplayCache:
    """Forks off a ReplayCache at `fork_from_lap` (the last lap the scenario
    should still be HISTORICAL for) and applies `lap_overrides` to every lap
    from `injection_lap` onward, up to `duration_laps` (None = rest of race).
    Laps before `injection_lap` are read straight from the historical cache -
    not recomputed, not duplicated."""

    def __init__(self, historical: ReplayCache, injection_lap: int,
                 lap_overrides: dict, duration_laps: Optional[int] = None):
        self.historical = historical
        self.injection_lap = injection_lap
        self.lap_overrides = lap_overrides
        self.end_lap = (injection_lap + duration_laps - 1) if duration_laps else None
        fork_from = injection_lap - 1
        historical.get(max(fork_from, 0))  # ensure the fork point is computed
        self.bundle = historical.bundle
        self.dpc = copy.deepcopy(historical.dpc) if fork_from == 0 else None
        if fork_from > 0:
            snap = historical._state_after[fork_from]
            # build a fresh dpc sharing ctx, cloned states from the fork point
            self.dpc = DriverPairContext(historical.dpc.ctx, copy.deepcopy(snap[0]),
                                          copy.deepcopy(snap[1]), historical.dpc.radios_by_driver_lap)
        self._decisions: dict[int, dict] = {}
        self._state_after: dict[int, tuple] = {fork_from: (copy.deepcopy(self.dpc.state_d1),
                                                             copy.deepcopy(self.dpc.state_d2))}
        self._fork_from = fork_from

    def _overrides_for(self, lap: int) -> Optional[dict]:
        if lap < self.injection_lap:
            return None
        if self.end_lap is not None and lap > self.end_lap:
            return None
        return self.lap_overrides

    def get(self, lap_number: int) -> dict:
        if lap_number < self.injection_lap:
            return self.historical.get(lap_number)
        if lap_number in self._decisions:
            return self._decisions[lap_number]
        last_cached = max((l for l in self._state_after if l < lap_number), default=self._fork_from)
        self.dpc.state_d1, self.dpc.state_d2 = copy.deepcopy(self._state_after[last_cached][0]), \
            copy.deepcopy(self._state_after[last_cached][1])
        for lap in range(last_cached + 1, lap_number + 1):
            dec = evaluate_lap(self.bundle, self.dpc, lap, lap_overrides=self._overrides_for(lap))
            self._decisions[lap] = dec
            self._state_after[lap] = (copy.deepcopy(self.dpc.state_d1), copy.deepcopy(self.dpc.state_d2))
        return self._decisions[lap_number]


# ============================================================================
# Full-grid helpers (timing tower / track map - every car, not just D1/D2)
# ============================================================================
def full_grid_for_lap(bundle: RaceBundle, lap_number: int) -> pd.DataFrame:
    """Rows for every driver at this lap, ranked exactly as HERMES's own
    adjacency logic ranks them (master._rank_by_gap_to_leader), with no
    columns beyond lap_number ever consulted (NO LOOKAHEAD)."""
    lap_df = bundle.laps[bundle.laps["LapNumber"] == lap_number]
    if lap_df.empty:
        return lap_df
    return master._rank_by_gap_to_leader(lap_df)


def weather_for_lap(bundle: RaceBundle, lap_number: int) -> dict:
    lap_df = bundle.laps[bundle.laps["LapNumber"] == lap_number]
    if lap_df.empty:
        return {}
    row = lap_df.iloc[0]
    return dict(
        air_temp=row.get("AirTemp"), track_temp=row.get("TrackTemp"),
        rainfall=bool(row.get("Rainfall", False)), humidity=row.get("Humidity"),
        wind_speed=row.get("WindSpeed"),
    )
