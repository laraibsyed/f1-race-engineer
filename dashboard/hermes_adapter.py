""
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

os.environ.setdefault("HERMES_REPO_ROOT", str(REPO_ROOT))

import system.HERMES.trees.master as master
import system.HERMES.trees.evaluate as hermes_eval

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

@dataclass
class RaceBundle:
    season: int
    race: str
    session: str
    laps: pd.DataFrame
    weather_df: Optional[pd.DataFrame]
    total_laps: int
    resources: dict
    degr_ordinal: int
    pit_loss_s: float
    p_sc_5lap: Optional[float]
    fallback_notes: list

    laps_by_lap: dict = field(default_factory=dict)
    weather_sorted: Optional[pd.DataFrame] = None
    grid_cache: dict = field(default_factory=dict)

def _prepare_weather(weather_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    ""
    if weather_df is None:
        return None
    w = weather_df.copy()
    time_col = "Time" if "Time" in w.columns else ("time_seconds" if "time_seconds" in w.columns else None)
    if time_col is None:
        return None
    if pd.api.types.is_numeric_dtype(w[time_col]):
        w["_t"] = pd.to_timedelta(w[time_col], unit="s")
    else:
        w["_t"] = pd.to_timedelta(w[time_col], errors="coerce")
    return w.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=True)

def _lap_frame(bundle: "RaceBundle", lap_number: int) -> Optional[pd.DataFrame]:
    ""
    if bundle.laps_by_lap:
        return bundle.laps_by_lap.get(int(lap_number))
    f = bundle.laps[bundle.laps["LapNumber"] == lap_number]
    return None if f.empty else f

def list_seasons(repo_root: Path = REPO_ROOT) -> list[int]:
    base = repo_root / "gcs_cache" / "clean" / "features"
    if not base.exists():
        return []
    return sorted((int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()), reverse=True)

def list_races(repo_root: Path, season: int) -> list[str]:
    ""
    return hermes_eval.list_races(repo_root, season)

def resolve_red_bull_pair(season: int, race: str) -> tuple[str, str]:
    ""
    return hermes_eval.rbr_drivers_for(season, race)

def list_drivers(bundle: RaceBundle) -> list[dict]:
    ""
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
    ""
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
    laps = master._attach_weather_columns(laps, weather_df)
    total_laps = int(laps["LapNumber"].max())

    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    laps_by_lap = {int(n): g for n, g in laps.groupby("LapNumber")}
    return RaceBundle(season, race, session, laps, weather_df, total_laps, resources,
                       degr_ordinal, pit_loss_s, p_sc_5lap, notes,
                       laps_by_lap=laps_by_lap, weather_sorted=_prepare_weather(weather_df))

RISK_MODES = ("CONSERVATIVE", "BALANCED", "AGGRESSIVE")
DEFAULT_RISK_MODE = "BALANCED"
TEAM_STRATEGY_N_SIMS = 200

@dataclass
class DriverPairContext:
    ctx: "master.RaceContext"
    state_d1: "master.DriverRuntimeState"
    state_d2: "master.DriverRuntimeState"
    radios_by_driver_lap: Optional[dict]
    team_strategy_fn: Optional[object] = None
    team_strategy_unavailable_reason: Optional[str] = None

_TEAM_STRATEGY_CACHE: dict = {}

def _build_team_strategy_fn(bundle: RaceBundle, ctx: "master.RaceContext") -> tuple:
    ""
    key = (ctx.season, ctx.circuit, ctx.d1_code, ctx.d2_code)
    if key in _TEAM_STRATEGY_CACHE:
        return _TEAM_STRATEGY_CACHE[key]
    if master.team_strategy is None or master.reward_mod is None:
        result = (None, "trees/team-strategy.py or strategic-sacrifice/reward.py could not be loaded.")
        _TEAM_STRATEGY_CACHE[key] = result
        return result
    try:
        driver_taxonomy = master.load_driver_taxonomy(master._data_path(REPO_ROOT, "driver_taxonomy_master_final.csv"))
        champ_ctx = _fetch_championship_context_with_timeout(ctx.season, ctx.circuit, ctx.d1_code, ctx.d2_code)
        fn = master.build_team_strategy_context(
            ctx, bundle.resources, driver_taxonomy, champ_ctx,
            risk_mode=DEFAULT_RISK_MODE, n_sims=TEAM_STRATEGY_N_SIMS)
        result = (fn, None)
    except Exception as e:
        result = (None, f"team-strategy context build failed ({e!r}).")
    _TEAM_STRATEGY_CACHE[key] = result
    return result

TEAM_STRATEGY_CHAMP_CTX_TIMEOUT_S = 8.0

def _fetch_championship_context_with_timeout(season, race_folder, d1_code, d2_code) -> dict:
    import threading
    empty = dict(available=False, d1_points=None, d2_points=None, round_num=None, races_remaining=None,
                 nearest_rival_code=None, d1_signed_gap_to_rival=None, d2_signed_gap_to_rival=None)

    result_box: list = []
    t = threading.Thread(
        target=lambda: result_box.append(master.fetch_championship_context(season, race_folder, d1_code, d2_code)),
        daemon=True)
    t.start()
    t.join(timeout=TEAM_STRATEGY_CHAMP_CTX_TIMEOUT_S)
    return result_box[0] if result_box else empty

def build_driver_pair_context(bundle: RaceBundle, d1: str, d2: str,
                               is_sprint_weekend: bool = False) -> DriverPairContext:
    ctx = master.RaceContext(
        season=bundle.season, circuit=bundle.race, total_laps=bundle.total_laps,
        is_sprint_weekend=is_sprint_weekend, d1_code=d1, d2_code=d2, session=bundle.session,
        circuit_degredation_ordinal=bundle.degr_ordinal, pit_loss_s=bundle.pit_loss_s,
        p_sc_5lap=bundle.p_sc_5lap,
    )
    radios = master.build_radios_by_driver_lap(REPO_ROOT, bundle.laps, ctx)
    team_strategy_fn, unavailable_reason = _build_team_strategy_fn(bundle, ctx)
    return DriverPairContext(ctx, master.DriverRuntimeState("D1", d1),
                              master.DriverRuntimeState("D2", d2), radios,
                              team_strategy_fn=team_strategy_fn,
                              team_strategy_unavailable_reason=unavailable_reason)

def _snapshot_states(dpc: DriverPairContext):
    return copy.deepcopy(dpc.state_d1), copy.deepcopy(dpc.state_d2)

def _restore_states(dpc: DriverPairContext, snapshot) -> None:
    dpc.state_d1, dpc.state_d2 = copy.deepcopy(snapshot[0]), copy.deepcopy(snapshot[1])

def evaluate_lap(bundle: RaceBundle, dpc: DriverPairContext, lap_number: int,
                  lap_overrides: Optional[dict] = None, risk_mode: Optional[str] = None) -> dict:
    ""
    laps = bundle.laps
    lap_df = _lap_frame(bundle, lap_number)
    if lap_df is None or lap_df.empty:
        return {}
    lap_df = master._rank_by_gap_to_leader(lap_df.copy())

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

    team_strategy_result, priority_driver_id = master.apply_team_strategy_to_lap(
        dpc.ctx, lap_results, row_d1, row_d2, lap_df, laps, lap_number,
        dpc.team_strategy_fn, dq_notes, risk_mode_override=risk_mode)

    master.merge_execution(dpc.ctx, lap_results, safety_car_active, priority_driver_id=priority_driver_id)

    out = {}
    for r in lap_results:
        r["data_quality_notes"] = list(dq_notes)
        if team_strategy_result is not None:
            r["team_strategy"] = team_strategy_result
        r["explanation"] = master.explanation_for(r)
        r["explanation_text"] = master.explanation_text_for(r)
        actual_row = row_d1 if r["driver"] == dpc.ctx.d1_code else row_d2
        r["actual_is_pit_in_lap"] = bool(actual_row.get("is_pit_in")) if actual_row is not None else None
        r["is_scenario"] = is_scenario
        out[r["driver"]] = r
    return out

class ReplayCache:
    def __init__(self, bundle: RaceBundle, dpc: DriverPairContext, risk_mode: str = DEFAULT_RISK_MODE):
        self.bundle = bundle
        self.dpc = dpc
        self.risk_mode = risk_mode
        self._decisions: dict[int, dict] = {}
        self._state_after: dict[int, tuple] = {0: _snapshot_states(dpc)}

    def set_risk_mode(self, risk_mode: str) -> None:
        ""
        if risk_mode != self.risk_mode:
            self.risk_mode = risk_mode
            self._decisions.clear()

    def get(self, lap_number: int) -> dict:
        if lap_number in self._decisions:
            return self._decisions[lap_number]
        last_cached = max((l for l in self._state_after if l < lap_number), default=0)
        _restore_states(self.dpc, self._state_after[last_cached])
        for lap in range(last_cached + 1, lap_number + 1):
            dec = evaluate_lap(self.bundle, self.dpc, lap, risk_mode=self.risk_mode)
            self._decisions[lap] = dec
            self._state_after[lap] = _snapshot_states(self.dpc)
        return self._decisions[lap_number]

    def get_range(self, lo: int, hi: int) -> dict[int, dict]:
        return {lap: self.get(lap) for lap in range(lo, hi + 1)}

class ScenarioReplayCache:
    ""

    def __init__(self, historical: ReplayCache, injection_lap: int,
                 lap_overrides: dict, duration_laps: Optional[int] = None):
        self.historical = historical
        self.injection_lap = injection_lap
        self.lap_overrides = lap_overrides
        self.end_lap = (injection_lap + duration_laps - 1) if duration_laps else None
        fork_from = injection_lap - 1
        if fork_from > 0:
            historical.get(fork_from)

        self.bundle = historical.bundle
        self.risk_mode = historical.risk_mode

        snap = historical._state_after[fork_from]
        self.dpc = DriverPairContext(historical.dpc.ctx, copy.deepcopy(snap[0]), copy.deepcopy(snap[1]),
                                      historical.dpc.radios_by_driver_lap,
                                      team_strategy_fn=historical.dpc.team_strategy_fn,
                                      team_strategy_unavailable_reason=historical.dpc.team_strategy_unavailable_reason)
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

    def set_risk_mode(self, risk_mode: str) -> None:
        if risk_mode != self.risk_mode:
            self.risk_mode = risk_mode
            self._decisions.clear()

    def get(self, lap_number: int) -> dict:
        if lap_number < self.injection_lap:
            return self.historical.get(lap_number)
        if lap_number in self._decisions:
            return self._decisions[lap_number]
        last_cached = max((l for l in self._state_after if l < lap_number), default=self._fork_from)
        self.dpc.state_d1, self.dpc.state_d2 = copy.deepcopy(self._state_after[last_cached][0]), \
            copy.deepcopy(self._state_after[last_cached][1])
        for lap in range(last_cached + 1, lap_number + 1):
            dec = evaluate_lap(self.bundle, self.dpc, lap, lap_overrides=self._overrides_for(lap),
                                risk_mode=self.risk_mode)
            self._decisions[lap] = dec
            self._state_after[lap] = (copy.deepcopy(self.dpc.state_d1), copy.deepcopy(self.dpc.state_d2))
        return self._decisions[lap_number]

def full_grid_for_lap(bundle: RaceBundle, lap_number: int) -> pd.DataFrame:
    ""
    hit = bundle.grid_cache.get(lap_number)
    if hit is not None:
        return hit
    lap_df = _lap_frame(bundle, lap_number)
    if lap_df is None or lap_df.empty:
        return pd.DataFrame()
    out = master._rank_by_gap_to_leader(lap_df.copy())
    bundle.grid_cache[lap_number] = out
    return out

def weather_for_lap(bundle: RaceBundle, lap_number: int) -> dict:
    ""
    lap_df = _lap_frame(bundle, lap_number)
    if lap_df is None or lap_df.empty:
        return {}
    row = lap_df.iloc[0]
    out = dict(rainfall=bool(row.get("Rainfall", False)), air_temp=None, track_temp=None,
               humidity=None, wind_speed=None)
    w = bundle.weather_sorted
    if w is None and bundle.weather_df is not None:
        w = bundle.weather_sorted = _prepare_weather(bundle.weather_df)
    if w is None or w.empty or "Time" not in lap_df.columns:
        return out
    lap_time = row.get("Time")
    if not isinstance(lap_time, pd.Timedelta):
        lap_time = pd.to_timedelta(lap_time, errors="coerce")
    if pd.isna(lap_time):
        return out
    idx = int(w["_t"].searchsorted(lap_time, side="right")) - 1
    if idx < 0 or idx >= len(w):
        return out
    wrow = w.iloc[idx]
    out.update(air_temp=wrow.get("AirTemp"), track_temp=wrow.get("TrackTemp"),
               humidity=wrow.get("Humidity"), wind_speed=wrow.get("WindSpeed"))
    return out
