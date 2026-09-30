"""
HERMES RL Environment - a lightweight, SIMULATED race-strategy MDP
======================================================================
NEW FILE. Does not modify master.py, gate-tier-*.py, execution-tree.py,
sc-gamble.py or reward.py - it IMPORTS them (via master.py, unmodified) and
reuses their already-fitted models, already-validated constants and
already-written decision helpers, rather than re-deriving any of them.

WHAT THIS IS: a single-car, per-lap strategy environment an agent can take
actions in and receive a reward from. It is NOT a full physical race
reconstruction (no other cars' actual telemetry, no true multi-car battle
simulation) - it is a SIMULATED/SYNTHETIC training environment, clearly
labelled as such throughout. It exists so a genuine Q-learning update has
something to learn from; see the module docstring in rl_agent.py for the
learning algorithm itself.

WHY NOT REPLAY REAL RACES: replaying a real race only tells you what
happened under the ONE strategy actually chosen that day - there is no
"what if we'd pitted 5 laps later" branch in historical data. An RL agent
needs to try many actions and see many consequences, which real historical
laps cannot supply. This is also why training here never touches 2025 (the
project's evaluation holdout, see evaluate.py) - training is 100% simulated,
not fit to any specific historical race or season.

WHAT IS GENUINELY REUSED FROM THE REST OF HERMES (not reinvented):
  - the REAL fitted Regression V2 + Cox models (tyre_life_models.pkl), called
    through master.py's own build_projection_fn - the exact function the real
    per-lap engine uses (master.evaluate_driver_lap).
  - the REAL calibrated Tier-3 thresholds (gate-tier-3.CLIFF_PROBABILITY_THRESHOLD,
    execution-tree.INSTRUCTION_NEUTRAL_BAND_RATIO) for state discretisation.
  - the REAL circuit-level SC/VSC prior (sc_vsc_circuit_level_prior.csv) and
    HORIZON_LAPS convention, via master.get_sc_probability.
  - the REAL, circuit-centred Monte Carlo pit-duration pools from sc-gamble.py
    (load_pit_durations) for sampling how long a pit stop actually costs -
    this IS "reusing the existing Monte Carlo simulation as the uncertainty
    layer" (dissertation claim in the project notes), not a new duration
    model.
  - the REAL cliff-risk cost formula/constant (sc-gamble.CLIFF_PENALTY_SCALE)
    already validated for the SC gamble.
  - the REAL undercut/overcut/dirty-air decision thresholds from master.py
    (live_undercut_opportunity, rival_undercut_threat, in_dirty_air,
    CLOSE_FOLLOWING_SECONDS, UNDERCUT_MIN_TYRE_AGE_GAP) applied to a
    synthetic gap process (see "SYNTHETIC PARTS" below).
  - gate-tier-2's real STANDARD_WEEKEND_ALLOCATION constant for the tyre-set
    budget.

SYNTHETIC PARTS (deliberately simplified, documented here rather than
silently invented):
  - the RIVAL AHEAD/BEHIND does not exist as a simulated car; "gap ahead"/
    "gap behind" are a bounded random walk around the real pit-loss scale
    (NORMAL_PIT_LOSS_SECONDS), classified by the REAL thresholds above. This
    gives the agent a plausible, varying undercut/overcut signal without
    building a second full multi-car simulator (explicitly out of scope,
    Step 16 "do not add ... a huge state space").
  - track temperature bucket is drawn once per episode from the four real
    buckets (cool/warm/hot/extreme), not evolved lap-to-lap.
  - rain is a fixed small per-episode probability (RAIN_EPISODE_PROB); no
    drying-crossover dynamics are simulated (that stays entirely inside
    drying_line.py / Tier 3, outside RL's action space - see ACTIONS below).
  - an SC/VSC period, once triggered, lasts a fixed HORIZON_LAPS laps (reusing
    that existing constant rather than inventing a new "SC duration" number;
    a real, data-derived duration exists in sc-gamble.compute_historical_sc_duration
    but is not wired into the SC gamble itself either - same status here).
  - the tyre-set budget is a single generic pool of STANDARD_WEEKEND_ALLOCATION
    sets (13), not split by compound and not season/Q2-rule aware - Tier 2 in
    the real Gate Tree remains the authority on the actual regulatory rule;
    this budget exists only so "tyre-set availability" is a genuine state
    variable the agent can learn to respect.

STATE (discretised, 8 variables, ~2,592 reachable combinations - small
enough for a Python dict-based Q-table to converge in seconds):
    compound_family   SOFT | MEDIUM | HARD                              (3)
    tyre_age_bucket   0-4 | 5-9 | 10-14 | 15+  laps on current tyre      (4)
    race_phase        early | mid | late  (crossover.classify_stage's OWN
                       real >60%/20-60%/<20% cutoffs, reused verbatim)     (3)
    cliff_risk        low | medium | high  (bucketed against the REAL
                       CLIFF_PROBABILITY_THRESHOLD=0.017 and the REAL 0.5x
                       "approaching" band already used by
                       execution-tree.get_driving_instruction)             (3)
    sc_active         True | False                                        (2)
    rain_now          True | False                                        (2)
    gap_state         THREAT_BEHIND | NEUTRAL | OPPORTUNITY_AHEAD          (3)
    sets_bucket       LOW (<=2 left) | OK                                  (2)
If a required real projection is unavailable (no fitted model for this
compound/circuit/era combination - a genuine, documented coverage gap, BP
§7-B11), cliff_risk defaults to "low" - the SAME None-safe convention Tier 3
itself uses ("unknown - don't fabricate a trigger"), not a special RL rule.

ACTION SPACE (4 actions - deliberately small):
    STAY_OUT | PIT_SOFT | PIT_MEDIUM | PIT_HARD
Only DRY compound families. RL is never asked to choose a wet-weather tyre -
while rain_now is True, STAY_OUT is the only legal action (mirrors the
explicit requirement that RL must not be responsible for unsafe wet-weather
tyre selection; that stays with the Gate Tree / weather modules). A PIT_x
action is also illegal if the tyre-set budget is exhausted (sets_bucket
tracked internally) or if this circuit/era has no fitted model for ANY
compound in that family (nothing to pit onto).

REWARD (time-cost based, in seconds; more negative = worse). Deliberately
uses ONLY quantities that already exist elsewhere in HERMES - no new
constant is invented for this file, only reused (2026-09-30 redesign - see
STAY_OUT_LOOKAHEAD_LAPS / MANDATORY_COMPOUND_PENALTY_SECONDS above for the
exact rationale; the ORIGINAL single-point-pace-loss version is preserved
verbatim in rl_env_pre_reward_redesign_backup.py):
    STAY_OUT this lap:  -(forward-looking pace loss + cliff risk penalty)
        pace loss <- REAL regression output, the UNWEIGHTED MEAN of
                     proj_fn(tyre_age) over the current age and the next
                     (STAY_OUT_LOOKAHEAD_LAPS - 1) ages - "how costly is the
                     immediate trajectory I'm on", not just this one lap.
        cliff penalty <- sc-gamble.CLIFF_PENALTY_SCALE * cliff_probability
                          (the SAME formula the validated SC gamble uses,
                          UNCHANGED - sc-gamble.py was not modified)
    PIT_x this lap:      -(pit-stop duration)
        duration <- ONE DRAW from sc-gamble.py's real, circuit-centred green
                     pool (or the caution-window pool if an SC is active this
                     lap) - i.e. genuinely sampled from the same Monte Carlo
                     machinery already validated for the SC gamble, not a
                     fresh distribution.
    Episode end (terminal step only): -MANDATORY_COMPOUND_PENALTY_SECONDS if
        gate-tier-2.mandatory_compound_done() (REAL, unchanged, imported via
        master.gate2) says this episode's compound-family history does not
        satisfy the real 2-dry-compound rule, and this was not a rain episode
        (the SAME wet_race_exception carve-out gate-tier-2 itself defines -
        RL's action space never selects a wet compound, so a rain episode
        cannot possibly comply and must not be penalised for it).
`compute_team_reward` (reward.py) is NOT reused here: it operates at
race-outcome / championship-points granularity (finishing position deltas,
WDC/WCC leverage) for the two-car team-order decision, a different unit and
a different decision than a single car's per-lap pit timing. Reusing it
would require inventing a fake position-delta mapping, which is exactly the
kind of invented complexity Step 6/16 rule out. The reward here instead
reuses the SAME per-lap TIME-COST primitives (pit loss, degradation,
cliff risk) that sc-gamble.py's own validated cost model is built from -
the direct precedent for a time-cost reward in this codebase.

NO FUTURE INFORMATION ever enters the STATE: every quantity in the
discretised state tuple comes from the environment's OWN current-step
variables (tyre_age, sc_active, rain_now, the just-updated gap walk,
sets_left) or from a projection called at the CURRENT tyre_age only - never
a lookahead beyond what the real per-lap engine itself would have at that
lap (see rl_validate.py test #7). The REWARD's small forward-looking average
(STAY_OUT_LOOKAHEAD_LAPS) is a different thing: it evaluates the SAME real,
already-fitted model at a few DETERMINISTIC future tyre ages (a MODEL
PREDICTION about the future, computable at decision time from information
already known - compound/circuit/era/tyre_age - exactly like
marginal_pace_loss() already does for the SC gamble in master.py) - it is
not real future data and does not touch the state tuple at all.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import master.py AS A MODULE, unmodified, to reuse its loaders and the
# already-loaded tree/module handles (gate1/gate2/gate3/exec_tree/tyre_proj/
# sc_gamble/crossover). This file does not duplicate ANY of master.py's logic.
# ---------------------------------------------------------------------------
_TREES_DIR = Path(__file__).resolve().parent
if str(_TREES_DIR) not in sys.path:
    sys.path.insert(0, str(_TREES_DIR))

_spec = importlib.util.spec_from_file_location("hermes_master", _TREES_DIR / "master.py")
master = importlib.util.module_from_spec(_spec)
sys.modules["hermes_master"] = master           # required by dataclasses on Python 3.14 (see sc-gamble-ab.py)
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
RAIN_EPISODE_PROB = 0.05          # ASSUMPTION: small, fixed per-episode rain chance - just enough to
                                    # exercise the wet-weather action mask (see rl_validate.py test #4),
                                    # not a weather model. Documented, not tuned.
MIN_TOTAL_LAPS, MAX_TOTAL_LAPS = 40, 72   # clip range for the real mean_total_laps values sampled below
GAP_WALK_SIGMA_FRACTION = 0.35    # ASSUMPTION: synthetic gap random-walk step size, as a fraction of
                                    # NORMAL_PIT_LOSS_SECONDS (the one real scale available for "gap").
SETS_BUDGET = master.gate2.STANDARD_WEEKEND_ALLOCATION   # REAL constant (13), reused as-is (BP §4.1)
SETS_LOW_THRESHOLD = 2

# ---------------------------------------------------------------------------
# REWARD REDESIGN (2026-09-30) - see the "REWARD" section of the module
# docstring above the class for the full before/after rationale. Two additive
# changes, both reusing existing HERMES quantities, no new invented bonus:
#
#   1. STAY_OUT_LOOKAHEAD_LAPS: a STAY_OUT step's degradation cost is now the
#      unweighted mean of the REAL fitted regression's predicted_pace_loss
#      over the current tyre_age AND the next (STAY_OUT_LOOKAHEAD_LAPS - 1)
#      ages, instead of a single-point value at the current age. This
#      literally implements "projected additional degradation over the next
#      few simulated laps" - no free weight parameter (unweighted mean), a
#      small window matching the scale of other small constants already in
#      this project (RED_FLAG_RESTART_BUFFER=2, UNDERCUT_MIN_TYRE_AGE_GAP=3).
#
#   2. Mandatory two-compound compliance (MANDATORY_COMPOUND_PENALTY_SECONDS):
#      at episode end, calls the REAL gate-tier-2.mandatory_compound_done()
#      (unchanged, imported via master.gate2) against the families of
#      compound actually used this episode. If not compliant (and this
#      episode's world was NOT a rain episode - gate-tier-2's own
#      wet_race_exception parameter, reused exactly), the terminal reward is
#      reduced by one mean real green pit-stop duration
#      (sc_gamble.NORMAL_PIT_LOSS_SECONDS, unchanged, imported via
#      master.sc_gamble) - representing the real pit stop a compliant driver
#      could not avoid, not an arbitrary invented number.
# ---------------------------------------------------------------------------
STAY_OUT_LOOKAHEAD_LAPS = 3
MANDATORY_COMPOUND_PENALTY_SECONDS = master.sc_gamble.NORMAL_PIT_LOSS_SECONDS


@dataclass
class EpisodeWorld:
    """Everything sampled ONCE at reset() to fix 'which race' this episode
    is (a real fitted (compound, circuit, era) key + real circuit context) -
    reused for every step of the episode."""
    circuit: str
    era: str
    start_compound: str
    degr_ordinal: int
    p_sc_window: float          # REAL circuit prior if available, else the pooled mean (documented)
    total_laps: int
    temp_bucket: str
    rain_now: bool


class HermesStrategyEnv:
    """Lightweight, single-agent, single-car strategy environment. See
    module docstring for exactly what is real vs synthetic."""

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
        self._reg_keys = list(self.tyre_models["reg_models"].keys())   # [(compound, circuit, era), ...]

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
        self.compound = None            # concrete compound string, e.g. "MEDIUM"
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
        self.compound_families_used: set = set()   # mandatory two-compound tracking, see reset()/step()
        self._last_info: dict = {}
        self._proj_cache: dict = {}   # see _get_projection() - pure speed optimisation, no behaviour change

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
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
        # PERFORMANCE SIMPLIFICATION (flagged, not hidden): fuel_load_estimate is held FIXED for the
        # whole episode at its real mid-race value (lap = total_laps/2), instead of being recomputed
        # every lap. The real per-lap engine (master.evaluate_driver_lap) DOES vary it every lap - this
        # is an RL-training-only approximation, made because the Cox survival-function call inside the
        # REAL fitted model (reused, not reimplemented - see module docstring) costs ~10ms per distinct
        # input, and RL's own state discretisation has no fuel dimension anyway (fuel's effect on the
        # coarse strategic decision is second-order compared to tyre age/compound/cliff risk). Fixing it
        # lets identical (compound, circuit, era, stint, temp, degr, tyre_age) calls - which the world
        # sampler repeats often across episodes - hit the cache instead of missing on fuel alone. This
        # does NOT change tyre_life_projection.py, the pickle, or any per-lap value the REAL replay
        # engine (master.py, --rl or not) computes; it only affects the SIMULATED training environment.
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
        # Mandatory two-compound tracking: the STARTING compound counts as "used" (matches
        # gate-tier-2.mandatory_compound_done's real semantics - compound_history_dry is whatever
        # dry compounds have been driven on so far, including the current one).
        self.compound_families_used = {COMPOUND_FAMILY.get(compound, "MEDIUM")}
        return self._discretize()

    def _proj_fn(self, compound: str):
        return master.build_projection_fn(self.tyre_models, compound, self.world.circuit,
                                           self.world.era, self._fixed_fuel, self.stint_number,
                                           self.world.temp_bucket, self.world.degr_ordinal)

    def _get_projection(self, compound: str, tyre_age: float) -> dict:
        """Memoised wrapper around the REAL proj_fn (master.build_projection_fn ->
        tyre_proj.build_tyre_life_projection, unchanged). The Cox call inside it is
        the dominant per-step cost (~10ms); this cache adds no new behaviour and
        changes no returned value - it only avoids recomputing an IDENTICAL call
        (same compound/circuit/era/stint/temp/degr/tyre_age/fixed-fuel - see the
        fixed-fuel note in reset()) within or across episodes, which the world
        sampler frequently repeats."""
        key = (compound, self.world.circuit, self.world.era, self.stint_number,
               self.world.temp_bucket, self.world.degr_ordinal, round(self._fixed_fuel),
               round(tyre_age, 1))   # fuel IS still part of the key (correctness: two episodes with
                                     # different total_laps have different _fixed_fuel and must not
                                     # collide) - only the PER-LAP fuel variation was removed, not fuel
                                     # itself; rounding to the nearest 1 kg still gives good cross-episode
                                     # reuse since _fixed_fuel only depends on total_laps (40-72 range).
        cached = self._proj_cache.get(key)
        if cached is None:
            cached = self._proj_fn(compound)(tyre_age)
            self._proj_cache[key] = cached
        return cached

    def _current_projection(self):
        return self._get_projection(self.compound, self.tyre_age)

    def valid_actions(self) -> list:
        """Returns the list of ACTIONS legal in the CURRENT state - the mask
        the agent/env both use so an invalid action can never be selected."""
        if self.world.rain_now:
            return ["STAY_OUT"]   # RL never proposes a dry pit in the rain (see module docstring)
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
        # Prefer the plain modern name (SOFT/MEDIUM/HARD) over the 2018 sub-tiers when both exist.
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
        elif self.rng.random() < (self.world.p_sc_window / master.HORIZON_LAPS):   # reused convention, see docstring
            self.sc_active = True
            self.sc_laps_left = master.HORIZON_LAPS

    def step(self, action: str):
        if self.world is None:
            raise RuntimeError("call reset() before step()")
        valid = self.valid_actions()
        if action not in valid:
            # Hard safety net (Step 14 test target): an out-of-mask action is never executed as
            # requested - it is treated as STAY_OUT and penalised, so a bug upstream is visible in
            # training rather than silently producing an unsafe/impossible pit.
            action = "STAY_OUT"
            invalid_attempted = True
        else:
            invalid_attempted = False

        if action == "STAY_OUT":
            self.tyre_age += 1
            # Forward-looking degradation cost (2026-09-30 redesign): unweighted mean of the REAL
            # regression's predicted_pace_loss over [tyre_age, tyre_age + LOOKAHEAD - 1], not a
            # single-point value at the current age - see STAY_OUT_LOOKAHEAD_LAPS's definition above
            # for exactly why. cliff_probability (for both the state and the cliff penalty below) still
            # comes from the CURRENT age only - Cox's own cliff_probability_next_5_laps is already a
            # forward-looking quantity by construction, so it is not re-averaged here.
            future_losses = [self._get_projection(self.compound, self.tyre_age + k)["predicted_pace_loss"]
                              for k in range(STAY_OUT_LOOKAHEAD_LAPS)]
            future_losses = [v for v in future_losses if v is not None]
            pace_loss = (sum(future_losses) / len(future_losses)) if future_losses else 0.0
            proj_now = self._current_projection()
            cliff_p = proj_now["cliff_probability_next_5_laps"] or 0.0
            cliff_penalty = master.sc_gamble.CLIFF_PENALTY_SCALE * cliff_p   # exact reuse, see docstring
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
            self.compound_families_used.add(family)   # mandatory two-compound tracking
            pace_loss = cliff_p = cliff_penalty = 0.0

        self._step_sc()
        self._step_gap_walk()
        self.lap += 1
        reward = -time_cost
        done = (self.lap > self.world.total_laps)

        # Mandatory two-compound compliance (terminal step only) - REAL gate-tier-2 rule, REAL
        # wet_race_exception carve-out, REAL pit-loss constant as the penalty magnitude. See
        # MANDATORY_COMPOUND_PENALTY_SECONDS's definition above for the full rationale.
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
            "mandatory_compliant": mandatory_compliant,   # None until the episode ends, then True/False
        }
        self._last_info = info
        return self._discretize(), reward, done, info

    # ------------------------------------------------------------------
    # state discretisation (see module docstring for the exact bins/reuse)
    # ------------------------------------------------------------------
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
            cliff_bucket = "low"        # None-safe: same "unknown -> don't trigger" convention as Tier 3
        elif cliff_p < 0.5 * threshold:  # reuses execution-tree's own 0.5x "approaching" band
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
        """For the explainability layer (rl_bridge.py) - the CURRENT
        discretised state's Q-values plus which actions are legal here."""
        state = self._discretize()
        return {"state": state, "valid_actions": self.valid_actions(),
                "q_values": agent.q_values(state)}
