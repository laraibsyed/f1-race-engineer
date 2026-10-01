""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_TREES_DIR = Path(__file__).resolve().parent
if str(_TREES_DIR) not in sys.path:
    sys.path.insert(0, str(_TREES_DIR))

_spec = importlib.util.spec_from_file_location("hermes_master", _TREES_DIR / "master.py")
master = importlib.util.module_from_spec(_spec)
sys.modules["hermes_master"] = master
_spec.loader.exec_module(master)

REPO_ROOT = master.REPO_ROOT

ACTIONS = ["STAY_OUT", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"]
ACTION_TO_FAMILY = {"PIT_SOFT": "SOFT", "PIT_MEDIUM": "MEDIUM", "PIT_HARD": "HARD"}
N_ACTIONS = len(ACTIONS)

COMPOUND_FAMILY = {
    "HYPERSOFT": "SOFT", "SUPERSOFT": "SOFT", "ULTRASOFT": "SOFT", "SOFT": "SOFT",
    "MEDIUM": "MEDIUM", "HARD": "HARD",
}
FAMILY_TO_COMPOUNDS = {"SOFT": {"HYPERSOFT", "SUPERSOFT", "ULTRASOFT", "SOFT"},
                        "MEDIUM": {"MEDIUM"}, "HARD": {"HARD"}}

TEMP_BUCKETS = ("cool", "warm", "hot", "extreme")
RAIN_EPISODE_PROB = 0.05

MIN_TOTAL_LAPS, MAX_TOTAL_LAPS = 40, 72
GAP_WALK_SIGMA_FRACTION = 0.35

SETS_BUDGET = master.gate2.STANDARD_WEEKEND_ALLOCATION
SETS_LOW_THRESHOLD = 2

STAY_OUT_LOOKAHEAD_LAPS = 3
MANDATORY_COMPOUND_PENALTY_SECONDS = master.sc_gamble.NORMAL_PIT_LOSS_SECONDS

@dataclass
class EpisodeWorld:
    ""
    circuit: str
    era: str
    start_compound: str
    degr_ordinal: int
    p_sc_window: float
    total_laps: int
    temp_bucket: str
    rain_now: bool

class HermesStrategyEnv:
    ""

    def __init__(self, repo_root: Optional[Path] = None, seed: int = 0):
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.rng = np.random.default_rng(seed)

        self.tyre_models = master.load_tyre_models(master._data_path(self.repo_root, "tyre_life_models.pkl"))
        self.cliff_stints = master.load_cliff_stints(master._data_path(self.repo_root, "cliff_detection_stints.csv"))
        self.sc_prior = master.load_sc_prior(master._data_path(self.repo_root, "sc_vsc_circuit_level_prior.csv"))
        self.taxonomy = master.load_circuit_taxonomy(self.repo_root / "src" / "taxanomy" / "circuit_taxonomy.xlsx")

        if self.tyre_models is None or not self.tyre_models.get("reg_models"):
            raise RuntimeError(
                "rl_env requires tyre_life_models.pkl with at least one fitted regression model - "
                "run system/HERMES/trees/model-fit.py first (see HERMES_MASTER_BLUEPRINT.md)."
            )
        self._reg_keys = list(self.tyre_models["reg_models"].keys())

        pools = master.sc_gamble.load_pit_durations()
        if pools is None:
            raise RuntimeError(
                "rl_env requires pit_stop_durations.csv (sc-gamble.py's circuit-centred pools) - "
                "see system/HERMES/safety-car/sc-gamble-v2.py's compute_real_pit_losses_from_laps()."
            )
        self.green_pool, self.caution_pool = pools

        self._pooled_p_sc = (float(self.sc_prior["p_window_horizon"].mean())
                              if self.sc_prior is not None and not self.sc_prior.empty else 0.05)
        self._pooled_total_laps = (float(self.sc_prior["mean_total_laps"].mean())
                                    if self.sc_prior is not None and not self.sc_prior.empty else 56.0)

        self.world: Optional[EpisodeWorld] = None
        self.lap = 0
        self.compound = None
        self.tyre_age = 0
        self.stint_number = 1
        self.sc_active = False
        self.sc_laps_left = 0
        self.gap_ahead_s = master.DEFAULT_PIT_LOSS_SECONDS
        self.gap_behind_s = master.DEFAULT_PIT_LOSS_SECONDS
        self.ahead_tyre_age = 0.0
        self.behind_tyre_age = 0.0
        self.ahead_pitted_recently = False
        self.sets_left = SETS_BUDGET
        self.compound_families_used: set = set()
        self._last_info: dict = {}
        self._proj_cache: dict = {}

    def reset(self):
        compound, circuit, era = self._reg_keys[self.rng.integers(len(self._reg_keys))]
        degr_ordinal, _ = master.circuit_degredation_ordinal_for(self.taxonomy, circuit)
        p_sc = master.get_sc_probability(self.sc_prior, circuit)
        if p_sc is None:
            p_sc = self._pooled_p_sc
        row = self.sc_prior[self.sc_prior["circuit"] == circuit] if self.sc_prior is not None else None
        mean_laps = (float(row["mean_total_laps"].iloc[0]) if row is not None and not row.empty
                     else self._pooled_total_laps)
        total_laps = int(np.clip(round(mean_laps + self.rng.integers(-3, 4)), MIN_TOTAL_LAPS, MAX_TOTAL_LAPS))
        temp_bucket = TEMP_BUCKETS[self.rng.integers(len(TEMP_BUCKETS))]
        rain_now = bool(self.rng.random() < RAIN_EPISODE_PROB)

        self.world = EpisodeWorld(circuit=circuit, era=era, start_compound=compound,
                                   degr_ordinal=degr_ordinal, p_sc_window=float(p_sc),
                                   total_laps=total_laps, temp_bucket=temp_bucket, rain_now=rain_now)

        self._fixed_fuel = master.fuel_load_estimate(max(1, total_laps // 2), total_laps)
        self.lap = 1
        self.compound = compound
        self.tyre_age = 0
        self.stint_number = 1
        self.sc_active = False
        self.sc_laps_left = 0
        self.gap_ahead_s = float(self.rng.normal(master.DEFAULT_PIT_LOSS_SECONDS, 5.0))
        self.gap_behind_s = float(self.rng.normal(master.DEFAULT_PIT_LOSS_SECONDS, 5.0))
        self.ahead_tyre_age = float(self.rng.integers(0, 15))
        self.behind_tyre_age = float(self.rng.integers(0, 15))
        self.ahead_pitted_recently = False
        self.sets_left = SETS_BUDGET

        self.compound_families_used = {COMPOUND_FAMILY.get(compound, "MEDIUM")}
        return self._discretize()

    def _proj_fn(self, compound: str):
        return master.build_projection_fn(self.tyre_models, compound, self.world.circuit,
                                           self.world.era, self._fixed_fuel, self.stint_number,
                                           self.world.temp_bucket, self.world.degr_ordinal)

    def _get_projection(self, compound: str, tyre_age: float) -> dict:
        ""
        key = (compound, self.world.circuit, self.world.era, self.stint_number,
               self.world.temp_bucket, self.world.degr_ordinal, round(self._fixed_fuel),
               round(tyre_age, 1))

        cached = self._proj_cache.get(key)
        if cached is None:
            cached = self._proj_fn(compound)(tyre_age)
            self._proj_cache[key] = cached
        return cached

    def _current_projection(self):
        return self._get_projection(self.compound, self.tyre_age)

    def valid_actions(self) -> list:
        ""
        if self.world.rain_now:
            return ["STAY_OUT"]
        valid = ["STAY_OUT"]
        if self.sets_left > 0:
            for action, family in ACTION_TO_FAMILY.items():
                if any((c, self.world.circuit, self.world.era) in self.tyre_models["reg_models"]
                       for c in FAMILY_TO_COMPOUNDS[family]):
                    valid.append(action)
        return valid

    def _concrete_compound_for_family(self, family: str) -> Optional[str]:
        candidates = [c for c in FAMILY_TO_COMPOUNDS[family]
                      if (c, self.world.circuit, self.world.era) in self.tyre_models["reg_models"]]
        if not candidates:
            return None

        return family if family in candidates else candidates[0]

    def _step_gap_walk(self):
        sigma = GAP_WALK_SIGMA_FRACTION * master.DEFAULT_PIT_LOSS_SECONDS
        self.gap_ahead_s = float(np.clip(self.gap_ahead_s + self.rng.normal(0, sigma), 0.0, 60.0))
        self.gap_behind_s = float(np.clip(self.gap_behind_s + self.rng.normal(0, sigma), 0.0, 60.0))
        self.ahead_tyre_age = max(0.0, self.ahead_tyre_age + 1 - (3 if self.rng.random() < 0.06 else 0))
        self.behind_tyre_age = max(0.0, self.behind_tyre_age + 1 - (3 if self.rng.random() < 0.06 else 0))
        self.ahead_pitted_recently = bool(self.rng.random() < 0.06)

    def _step_sc(self):
        if self.sc_active:
            self.sc_laps_left -= 1
            if self.sc_laps_left <= 0:
                self.sc_active = False
        elif self.rng.random() < (self.world.p_sc_window / master.HORIZON_LAPS):
            self.sc_active = True
            self.sc_laps_left = master.HORIZON_LAPS

    def step(self, action: str):
        if self.world is None:
            raise RuntimeError("call reset() before step()")
        valid = self.valid_actions()
        if action not in valid:

            action = "STAY_OUT"
            invalid_attempted = True
        else:
            invalid_attempted = False

        if action == "STAY_OUT":
            self.tyre_age += 1

            future_losses = [self._get_projection(self.compound, self.tyre_age + k)["predicted_pace_loss"]
                              for k in range(STAY_OUT_LOOKAHEAD_LAPS)]
            future_losses = [v for v in future_losses if v is not None]
            pace_loss = (sum(future_losses) / len(future_losses)) if future_losses else 0.0
            proj_now = self._current_projection()
            cliff_p = proj_now["cliff_probability_next_5_laps"] or 0.0
            cliff_penalty = master.sc_gamble.CLIFF_PENALTY_SCALE * cliff_p
            time_cost = pace_loss + cliff_penalty
            pit_duration = None
        else:
            family = ACTION_TO_FAMILY[action]
            concrete = self._concrete_compound_for_family(family)
            pool = self.caution_pool if self.sc_active else self.green_pool
            pit_duration = float(self.rng.choice(pool))
            time_cost = pit_duration
            self.compound = concrete if concrete is not None else self.compound
            self.tyre_age = 0
            self.stint_number += 1
            self.sets_left -= 1
            self.compound_families_used.add(family)
            pace_loss = cliff_p = cliff_penalty = 0.0

        self._step_sc()
        self._step_gap_walk()
        self.lap += 1
        reward = -time_cost
        done = (self.lap > self.world.total_laps)

        mandatory_compliant = None
        if done:
            mandatory_compliant = master.gate2.mandatory_compound_done(
                self.compound_families_used, wet_race_exception=self.world.rain_now)
            if not mandatory_compliant:
                reward -= MANDATORY_COMPOUND_PENALTY_SECONDS

        info = {
            "action_taken": action, "invalid_attempted": invalid_attempted,
            "lap": self.lap - 1, "compound": self.compound, "tyre_age": self.tyre_age,
            "pace_loss_s": pace_loss, "cliff_probability": cliff_p, "cliff_penalty_s": cliff_penalty,
            "pit_duration_s": pit_duration, "sc_active": self.sc_active, "sets_left": self.sets_left,
            "circuit": self.world.circuit, "era": self.world.era,
            "compound_families_used": sorted(self.compound_families_used),
            "mandatory_compliant": mandatory_compliant,
        }
        self._last_info = info
        return self._discretize(), reward, done, info

    def _discretize(self) -> tuple:
        family = COMPOUND_FAMILY.get(self.compound, "MEDIUM")

        if self.tyre_age < 5:
            age_bucket = "0-4"
        elif self.tyre_age < 10:
            age_bucket = "5-9"
        elif self.tyre_age < 15:
            age_bucket = "10-14"
        else:
            age_bucket = "15+"

        laps_remaining = max(0, self.world.total_laps - self.lap + 1)
        phase = master.crossover.classify_stage(laps_remaining, self.world.total_laps).value if master.crossover \
            else ("early" if laps_remaining / self.world.total_laps > 0.6 else
                  "mid" if laps_remaining / self.world.total_laps > 0.2 else "late")

        proj = self._current_projection()
        cliff_p = proj["cliff_probability_next_5_laps"]
        threshold = master.gate3.CLIFF_PROBABILITY_THRESHOLD
        if cliff_p is None:
            cliff_bucket = "low"
        elif cliff_p < 0.5 * threshold:
            cliff_bucket = "low"
        elif cliff_p < threshold:
            cliff_bucket = "medium"
        else:
            cliff_bucket = "high"

        threat_behind = master.rival_undercut_threat(self.gap_behind_s, self.tyre_age,
                                                       self.behind_tyre_age, master.DEFAULT_PIT_LOSS_SECONDS)
        opportunity_ahead = master.live_undercut_opportunity(self.gap_ahead_s, self.tyre_age,
                                                               self.ahead_tyre_age, master.DEFAULT_PIT_LOSS_SECONDS)
        gap_state = "THREAT_BEHIND" if threat_behind else ("OPPORTUNITY_AHEAD" if opportunity_ahead else "NEUTRAL")

        sets_bucket = "LOW" if self.sets_left <= SETS_LOW_THRESHOLD else "OK"

        return (family, age_bucket, phase, cliff_bucket, self.sc_active, self.world.rain_now,
                gap_state, sets_bucket)

    def q_values_explained(self, agent) -> dict:
        ""
        state = self._discretize()
        return {"state": state, "valid_actions": self.valid_actions(),
                "q_values": agent.q_values(state)}
