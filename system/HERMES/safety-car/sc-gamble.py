"""
SC/VSC Gamble Evaluator
============================
Answers: given current tyre state and elevated P(SC), is it cheaper in
expectation to pit now, or wait N laps and hope for a cheap SC-window pit?

Deliberately NOT a boolean trigger like Tier 3's other soft triggers - this
is a genuine expected-cost comparison, since "should I wait for a possible
SC" depends on several things pulling in different directions at once
(P(SC), how much waiting costs in tyre degradation, how much cheaper an SC
pit actually is, and the cold-tyre cost of restarting after one).

============================================================================
2026 CALIBRATION UPDATE - read this before touching the constants again
============================================================================
The cost model below WAS:
    NORMAL_PIT_LOSS_SECONDS = 22.0   # ASSUMPTION
    SC_PIT_LOSS_SECONDS = 11.0       # ASSUMPTION - "roughly half"
with cost_if_sc_comes using HALF the degradation cost and NO cliff term at
all, while cost_if_sc_doesnt_come used the FULL degradation and cliff cost.

Both the constants and that asymmetric formula have since been validated
against real data and found wanting, via a standalone calibration
(system/HERMES/safety-car/sc-gamble-v2.py - kept as the record of that
work; this file ports the validated result into production, it does not
duplicate the experiment):

  1. The archive's own `pit_loss_constant_s` column
     (checkpoints/rival_knowledge/archive_event_summary_enriched.csv) is
     NOT a per-event measurement - it's a duplicated per-(circuit, year)
     constant (confirmed directly: n_distinct_values matched n_years
     exactly across all 36 circuits checked). No split of that column can
     ever produce a real SC-vs-green estimate.
  2. Bypassing it, 5,744 REAL pit-stop durations were computed directly
     from raw PitInTime/PitOutTime timestamps (the same approach
     rival-awareness/knowledge.py's compute_pit_lane_loss uses), split by
     whether the stop's entry/exit lap overlapped a real is_sc_lap/
     is_vsc_lap window. Cross-checked against the archive's own aggregate
     to within 0.10s - a good sanity check that the raw-timestamp method
     itself is sound.
  3. Result: pooled normal (green-flag) pit duration ~23.685s, pooled
     CAUTION-WINDOW pit duration ~22.902s - a real but small ~0.78s
     difference, nothing like the assumed ~11s ("roughly half") gap.
  4. TERMINOLOGY, kept precise deliberately: what's measured is elapsed
     PitIn->PitOut time for stops whose entry/exit lap overlapped a
     caution period - NOT a validated claim that the driver pitted
     BECAUSE the SC made it cheaper (a car could enter under green and
     happen to exit as the SC ended, for instance). Call it an empirical
     CAUTION-WINDOW pit-stop duration, not "the SC pit-loss" - that's why
     the constant below is named accordingly even though the dataclass
     field feeding it (p_sc_next_n_laps etc.) keeps its original name for
     interface stability.
  5. The FORMULA was also asymmetric, not just the constants: the old
     cost_if_sc_comes branch halved degradation with no stated reason and
     dropped cliff risk entirely, while cost_if_sc_doesnt_come charged the
     full amount of both. That half was actually a *laps-driven* split in
     disguise, so it's now made explicit and applied symmetrically to
     BOTH degradation and cliff exposure: an SC, if it comes, is assumed
     (matching the SAME uniform-placement assumption already validated in
     safety-car/sc-vsc-probability-model.py's build_circuit_level_prior)
     to land on average halfway through the horizon, so only half the
     window's degradation/cliff risk is actually incurred before the
     cheaper stop - not an arbitrary discount on the cost, a principled
     discount on the LAPS actually driven.
  6. At zero degradation and zero cliff risk, this reduces to the
     interpretable rule:
         WAIT iff p_sc * (NORMAL_PIT_LOSS_SECONDS
                           - CAUTION_WINDOW_PIT_DURATION_SECONDS
                           - RESTART_COLD_TYRE_PENALTY_SECONDS) > 0
     With the real numbers above (23.685 - 22.902 - 1.5 = -0.717), this is
     NEGATIVE - i.e. waiting purely to catch a cheaper SC stop is NOT,
     on its own, worth the cold-tyre penalty at the pooled global level.
     WAIT still emerges at circuits with a genuinely larger discount (São
     Paulo's real numbers, 23.64 vs 20.30, give a positive net benefit and
     produce real WAIT recommendations across a meaningful slice of the
     realistic parameter range) - the point is that the decision now
     tracks real, circuit-level economics instead of an assumed universal
     50% discount.
  7. Do NOT "fix" a disappointing WAIT rate by re-inflating the caution
     discount, re-halving degradation, or dropping the cliff term again.
     If a genuinely richer SC-vs-green split becomes available later
     (e.g. per-circuit caution-window durations wired all the way through
     instead of a flat pooled constant), that's a real enhancement -
     revisit sc-gamble-v2.py's per-circuit table for that when it's worth
     doing, not a reason to touch this file's formula again on suspicion
     alone.

Every other design choice in this file (compute_historical_sc_duration,
the SCGambleInputs contract, the "WAIT"/"NO_ADVANTAGE_TO_WAITING"/
"INSUFFICIENT_DATA" vocabulary gate-tier-3.py's suppression check depends
on) is UNCHANGED - this is a formula/constant port, not a rewrite.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# --- validated (2026 calibration) - see the block above before changing these ---
NORMAL_PIT_LOSS_SECONDS = 23.685              # was 22.0 (ASSUMPTION) - now the real pooled
                                                # green-flag median from 4,224 raw PitIn->PitOut
                                                # stops (sc-gamble-v2.py Step 2b)
CAUTION_WINDOW_PIT_DURATION_SECONDS = 22.902  # was SC_PIT_LOSS_SECONDS = 11.0 (ASSUMPTION,
                                                # "roughly half") - now the real pooled median
                                                # from 1,520 stops whose entry/exit lap
                                                # overlapped is_sc_lap/is_vsc_lap. NOT a claim
                                                # the stop happened BECAUSE of the caution - see
                                                # the terminology note above.
RESTART_COLD_TYRE_PENALTY_SECONDS = 1.5       # ASSUMPTION, unchanged - the calibration didn't
                                                # touch this; still flagged as such
CLIFF_PENALTY_SCALE = 5.0                     # ASSUMPTION, unchanged (was the bare literal
                                                # `5.0 *` in the old cliff_penalty_seconds line)


def compute_historical_sc_duration(laps: pd.DataFrame) -> pd.DataFrame:
    """
    REAL, data-derived (not guessed): average length of an SC period in laps,
    per circuit, from consecutive is_sc_lap=True runs. Requires is_sc_lap
    (already engineered from race control messages) - reused infrastructure.

    UNCHANGED by the 2026 calibration and STILL not consumed by
    evaluate_sc_gamble - it answers "how long does a caution period last in
    laps", not "what does a pit stop taken during one cost in seconds",
    which is a genuinely different quantity (see the calibration note
    above and sc-gamble-v2.py's own diagnostic that kept these separate).
    Kept here as a real, useful, independent statistic.
    """
    rows = []
    for (season, race), g in laps.groupby(["Season", "Race"]):
        g = g.drop_duplicates("LapNumber").sort_values("LapNumber")
        sc_flags = g["is_sc_lap"].astype(bool).values
        run_length = 0
        for flag in sc_flags:
            if flag:
                run_length += 1
            elif run_length > 0:
                rows.append({"circuit": race, "season": season, "sc_duration_laps": run_length})
                run_length = 0
        if run_length > 0:
            rows.append({"circuit": race, "season": season, "sc_duration_laps": run_length})

    durations = pd.DataFrame(rows)
    if durations.empty:
        return pd.DataFrame(columns=["circuit", "mean_sc_duration_laps", "n_sc_periods"])
    return durations.groupby("circuit").agg(
        mean_sc_duration_laps=("sc_duration_laps", "mean"),
        n_sc_periods=("sc_duration_laps", "count"),
    ).reset_index()


@dataclass
class SCGambleInputs:
    p_sc_next_n_laps: Optional[float]        # from sc_vsc_probability_model.get_sc_probability() -
                                              # NOTE: as of the model's leave-one-race-out validation,
                                              # this is a CIRCUIT-LEVEL constant (same value regardless
                                              # of current lap), not a genuinely lap-specific forecast -
                                              # lap-level resolution was tested and found to carry no
                                              # real signal (AUC 0.503). "Current circuit propensity,"
                                              # not "risk right now specifically."
    predicted_pace_loss_per_lap: Optional[float]  # from tyre_life_projection - regression V2 output.
                                              # Pass the MARGINAL slope here (loss(age+h)-loss(age))/h,
                                              # not the absolute pace loss at the current age - see
                                              # hermes_master.py's marginal_pace_loss() for why (an
                                              # absolute value here overstates the waiting cost).
    cliff_probability_next_n_laps: Optional[float]  # from tyre_life_projection - Cox output
    n_laps_horizon: int                       # how many laps ahead this decision considers


def evaluate_sc_gamble(inputs: SCGambleInputs) -> dict:
    """
    Returns the expected cost of each option plus a recommendation. Returns
    "INSUFFICIENT_DATA" (not a guess) if P(SC) or pace-loss inputs are
    unavailable - same None-safety pattern as every other tyre-model consumer.

    Interface is UNCHANGED from before the 2026 calibration: same inputs
    dataclass, same return keys, same recommendation vocabulary ("WAIT" |
    "NO_ADVANTAGE_TO_WAITING" | "INSUFFICIENT_DATA") - gate-tier-3.py's
    `sc_gamble_recommendation != "WAIT"` suppression check, and
    hermes_master.py's call site, both need zero changes for this update.
    Only the internal cost model changed - see the module docstring for
    exactly what and why.
    """
    if inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None:
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None}

    p_sc = inputs.p_sc_next_n_laps
    n = inputs.n_laps_horizon
    pace = inputs.predicted_pace_loss_per_lap
    cliff_risk = inputs.cliff_probability_next_n_laps or 0.0

    # Expected laps actually driven before pitting, in each future - an SC,
    # if it comes, is assumed to land on average halfway through the
    # horizon (the SAME uniform-placement assumption already validated in
    # sc-vsc-probability-model.py's build_circuit_level_prior), so only
    # half the window's degradation/cliff exposure is incurred before the
    # cheaper stop. This replaces the old formula's un-derived "/2" on
    # degradation alone with a laps-driven split applied symmetrically to
    # BOTH degradation and cliff risk - the old formula dropped cliff risk
    # from the SC branch entirely, which understated its cost.
    laps_if_sc_comes = n / 2.0
    laps_if_no_sc = float(n)

    degradation_if_sc_comes = pace * laps_if_sc_comes
    degradation_if_no_sc = pace * laps_if_no_sc

    cliff_penalty_if_sc_comes = CLIFF_PENALTY_SCALE * cliff_risk * (laps_if_sc_comes / n)
    cliff_penalty_if_no_sc = CLIFF_PENALTY_SCALE * cliff_risk * (laps_if_no_sc / n)

    cost_pit_now = NORMAL_PIT_LOSS_SECONDS

    cost_if_sc_comes = (CAUTION_WINDOW_PIT_DURATION_SECONDS + RESTART_COLD_TYRE_PENALTY_SECONDS
                         + degradation_if_sc_comes + cliff_penalty_if_sc_comes)
    cost_if_sc_doesnt_come = NORMAL_PIT_LOSS_SECONDS + degradation_if_no_sc + cliff_penalty_if_no_sc
    cost_wait = p_sc * cost_if_sc_comes + (1 - p_sc) * cost_if_sc_doesnt_come

    recommendation = "WAIT" if cost_wait < cost_pit_now else "NO_ADVANTAGE_TO_WAITING"

    return {
        "recommendation": recommendation,
        "cost_pit_now": cost_pit_now,
        "cost_wait": cost_wait,
        "expected_saving_if_wait": cost_pit_now - cost_wait,
    }


# ============================================================================
# MONTE CARLO VERSION - ADDITIVE. evaluate_sc_gamble() above is untouched and
# remains the default; this is a second, opt-in evaluator with the SAME inputs
# dataclass, the SAME return keys and the SAME recommendation vocabulary, so
# it can be swapped in at hermes_master's call site with no other change.
#
# WHY: the analytic version collapses every uncertain quantity to its mean.
# But the gamble's payoff is highly skewed - it wins big in the few futures
# where an SC lands and loses a little in all the others - so the MEAN alone
# hides how often waiting actually pays. This version simulates many futures
# and reports the distribution: P(waiting beats pitting now), spread, tails.
#
# What is sampled (per simulated future):
#   * does an SC/VSC land inside the horizon?  Bernoulli(p_sc)
#   * when?  Uniform over the horizon - the SAME uniform-placement assumption
#     the validated circuit prior is built on (so E[laps driven before an SC
#     stop] = n/2, exactly the analytic model's laps_if_sc_comes)
#   * the pit stop itself: RESAMPLED from the real PitIn->PitOut durations
#     behind the 2026 calibration (pit_stop_durations.csv; green stops for
#     pit-now and for the no-SC wait branch, caution-window stops for the SC
#     branch), as DEVIATIONS from each stop's circuit baseline, winsorised
#     per circuit at P10/P90 (knowledge.py's per-race convention), then placed
#     on the calibrated constants. Full method + assumptions: see
#     load_pit_durations() and the MODELLING ASSUMPTIONS block below. Falls
#     back to a normal centred on the calibrated constants if the CSV is missing.
#   * degradation noise: pace * laps + N(0, PACE_LOSS_RMSE * sqrt(laps))
#   * does a cliff hit BEFORE the pit stop? Bernoulli(cliff_p) at a uniform
#     time; an early SC stop avoids a later cliff (analytic model can't see
#     this interaction - it only scales the penalty by laps/n).
#
# Deliberately NOT changed: the calibrated constants, the cost structure, the
# cold-tyre penalty and cliff penalty (both still flagged ASSUMPTIONS), and the
# advice in the docstring above not to re-inflate the caution discount. The MC
# is a REPORTING / UNCERTAINTY layer around the analytic model, not a second
# decision mechanism: what it adds is the spread, not a different answer.
#
# ---------------------------------------------------------------------------
# MODELLING ASSUMPTIONS of the Monte Carlo layer (assumptions, NOT thresholds
# proven optimal - none was tuned against outcomes; all fixed before looking at
# the results they produce):
#   1. Circuit baseline = median of that circuit's green stops (all years
#      pooled); the circuit effect on a stop's duration is additive.
#   2. MIN_STOPS_FOR_CIRCUIT_BASELINE = 30 green stops for a circuit to have a
#      baseline. Thinner circuits are left out of the pools.
#   3. MIN_STOPS_FOR_CIRCUIT_WINSOR = 10 stops for a (circuit, group) cell to get
#      its own P10/P90 winsorisation band. Thinner cells are left out.
#   4. Caution-window stops are measured against the same circuit's GREEN
#      baseline (keeps circuit-to-circuit variation in the discount as
#      uncertainty, because the pooled model does not know the circuit).
#   5. Winsorisation is P10/P90 within each (circuit, group). This choice - not
#      the centring - sets how much slow-stop tail risk is kept: the raw
#      within-circuit green-vs-green spread is ~5.9 s, ~2.6 s after this
#      winsorisation. It changes reported tails and win probabilities, not the
#      decisions.
#   6. Deviations are pooled across circuits whose spread differs a lot
#      (per-circuit P10-P90 width 1.5 - 17.5 s), i.e. the pooled residual shape
#      is a mixture; the specific circuit's own scale is not used.
#   7. Pools are shifted so their MEDIANS equal the calibrated constants. The
#      empirical green deviations are RIGHT-SKEWED (mean 24.13 s vs median
#      23.685 s), while the analytic evaluator uses median-calibrated
#      constants. The simulation averages a skewed distribution, so its mean
#      discount is 0.97 s vs the analytic 0.78 s, giving a small POSITIVE
#      MC-minus-analytic mean saving of about p_sc x 0.19 s (realistic range:
#      +0.008 s on average, max 0.025 s). Understood and deliberately NOT
#      corrected - the MC mean is not forced to match the analytic one.
#
# What the 2.6 s simulated spread does and does NOT show: the simulated
# green-vs-green spread (2.61 s) agrees with the empirical within-circuit
# spread computed under the SAME per-circuit P10/P90 convention (2.59 s). That
# agreement is a consistency check that the centring was implemented
# correctly - it is NOT independent validation, because the winsorisation
# convention contributes to both numbers. The independent evidence is
# decision-level: see sc-gamble-ab.py (decision agreement with the analytic
# evaluator on sc-gamble-v2.py's own grids, mean-saving fidelity, seed
# stability, schema/semantics parity) and the master.py replay comparison.
#
# The MC outcome probabilities (P(waiting wins), p05/p50/p95) are conditional
# on this empirical pit-duration distribution and on the assumptions above;
# they are not calibrated forecasts of real race outcomes.
# ---------------------------------------------------------------------------
# ============================================================================
PIT_DURATIONS_CSV = Path(__file__).resolve().parents[3] / "pit_stop_durations.csv"
                                        # columns: race, year, duration_s, under_caution. Produced by
                                        # sc-gamble-v2.py's compute_real_pit_losses_from_laps() (4,224
                                        # green + 1,520 caution-window stops from 178 races).
PACE_LOSS_RMSE_SECONDS = 0.057          # DATA-DERIVED: mean RMSE of the pace-loss regression at the
                                        # production outlier cut (z=4.0) - z_threshold_sensitivity.csv.
                                        # Treated as independent per lap (ASSUMPTION - real model error is
                                        # probably persistent, which would widen the spread).
MIN_STOPS_FOR_CIRCUIT_BASELINE = 30     # ASSUMPTION: a circuit needs >= this many green stops for its median to serve
                                        # as that circuit's baseline; stops from thinner circuits are left out of the
                                        # pools (not pooled uncentred, which would re-import between-circuit noise).
MIN_STOPS_FOR_CIRCUIT_WINSOR = 10       # ASSUMPTION: a (circuit, group) cell needs >= this many stops for its own P10/P90
                                        # to be a usable winsorisation band; thinner cells are dropped (21 caution stops).
FALLBACK_GREEN_SD_SECONDS = 1.80        # DATA-DERIVED: sd of the circuit-centred, per-circuit-winsorised(10/90) green pool
FALLBACK_CAUTION_SD_SECONDS = 4.66      #   and caution-window pool that load_pit_durations() builds (33 circuits,
                                        #   4,172 / 1,438 stops). Were 2.98 / 4.40 before circuit-centring. Used only
                                        #   if pit_stop_durations.csv is missing; recompute if the CSV is regenerated.


@dataclass
class MCConfig:
    n_sims: int = 20000
    seed: int = 0                        # fixed for reproducible recommendations (same inputs -> same answer)
    winsorize: tuple = (0.10, 0.90)
    pace_loss_rmse_s: float = PACE_LOSS_RMSE_SECONDS
    min_win_prob: float = 0.0            # 0.0 = pure expected-value rule (matches the analytic evaluator).
                                          # Raise it (e.g. 0.3) for a risk-averse "only gamble if it pays in
                                          # at least this share of futures" rule. Not a validated value.


_PIT_POOL_CACHE: dict = {}
_PIT_POOL_META: dict = {}


def load_pit_durations(path: Optional[Path] = None, winsorize: tuple = (0.10, 0.90)):
    """Returns (green, caution) arrays of pit-stop DEVIATIONS placed on the
    calibrated constants, or None if the CSV isn't available (caller falls
    back to a normal around the calibrated constants - never a silent guess:
    the result's `pit_duration_source` says which was used).

    WHY CIRCUIT-CENTRED. ~61% of the variance in raw green pit durations is
    BETWEEN circuits (circuit medians run 21-30 s). The gamble compares
    "pit now" with "wait, then pit later" at the SAME circuit, so that shared
    circuit baseline cancels in reality. Resampling raw durations treated it as
    random noise (two independent draws from a cross-circuit pool), inflating
    the spread of the simulated saving ~1.6x. Fix, in three steps:
      1. baseline_c = median green stop at circuit c (needs >=
         MIN_STOPS_FOR_CIRCUIT_BASELINE green stops; other circuits are dropped);
      2. deviation = duration - baseline_c, for BOTH green and caution-window
         stops (caution stops are measured against the same circuit's GREEN
         baseline, so the per-circuit caution discount stays in as genuine
         uncertainty about which discount applies - the pooled model doesn't
         know the circuit), then winsorised at the same P10/P90;
      3. each pool is shifted so its median equals the calibrated constant
         (NORMAL_PIT_LOSS_SECONDS / CAUTION_WINDOW_PIT_DURATION_SECONDS), so the
         calibrated analytic constants are preserved exactly and only the SHAPE
         (spread) comes from data.
    Nothing here changes the cost formula or the decision rule."""
    path = Path(path) if path else PIT_DURATIONS_CSV
    key = (str(path), winsorize)
    if key in _PIT_POOL_CACHE:
        return _PIT_POOL_CACHE[key]
    pools, meta = None, None
    if path.exists():
        d = pd.read_csv(path)
        if {"duration_s", "under_caution", "race"} <= set(d.columns):
            d = d.dropna(subset=["duration_s", "race"]).copy()
            d["under_caution"] = d["under_caution"].astype(bool)
            green = d[~d["under_caution"]].groupby("race")["duration_s"].agg(["median", "size"])
            usable = green[green["size"] >= MIN_STOPS_FOR_CIRCUIT_BASELINE]
            n_before = {False: int((~d["under_caution"]).sum()), True: int(d["under_caution"].sum())}
            d = d[d["race"].isin(usable.index)]
            d["deviation"] = d["duration_s"] - d["race"].map(usable["median"])
            pools = []
            for flag, constant in ((False, NORMAL_PIT_LOSS_SECONDS), (True, CAUTION_WINDOW_PIT_DURATION_SECONDS)):
                x = d.loc[d["under_caution"] == flag, ["race", "deviation"]].copy()
                x = x[x.groupby("race")["deviation"].transform("size") >= MIN_STOPS_FOR_CIRCUIT_WINSOR]
                if len(x) < 30:
                    pools = None
                    break
                # winsorise WITHIN each circuit (and group), like knowledge.py does per race - the data are
                # homogeneous there. Clipping at the POOLED P10/P90 instead (first attempt) chopped the genuine
                # slow-stop tail off every circuit (paired spread 1.71 s vs 2.55 s per-circuit).
                dev = x.groupby("race")["deviation"].transform(
                    lambda s: s.clip(s.quantile(winsorize[0]), s.quantile(winsorize[1]))).astype(float)
                pools.append((dev - dev.median() + constant).to_numpy())
            if pools:
                pools = tuple(pools)
                meta = {"n_circuits": int(len(usable)), "n_green": len(pools[0]), "n_caution": len(pools[1]),
                        "dropped_green": n_before[False] - len(pools[0]), "dropped_caution": n_before[True] - len(pools[1])}
    _PIT_POOL_CACHE[key] = pools
    _PIT_POOL_META[key] = meta
    return pools


def evaluate_sc_gamble_mc(inputs: SCGambleInputs, config: Optional[MCConfig] = None,
                          pit_durations=None) -> dict:
    """
    Monte Carlo counterpart of evaluate_sc_gamble(). Same inputs, same core
    return keys ("recommendation", "cost_pit_now", "cost_wait",
    "expected_saving_if_wait") plus distribution fields:

      p_wait_better          share of simulated futures where waiting was cheaper
      saving_p05/p50/p95     percentiles of (cost_pit_now - cost_wait); positive = waiting saved time
      saving_std / saving_se spread of one outcome / standard error of the mean saving
      decision_confident     |mean saving| > 2 standard errors (else the sign is simulation noise -
                             raise n_sims or treat as a genuine toss-up)
      analytic_expected_saving   the closed-form answer, for side-by-side comparison
      pit_duration_source    "empirical, circuit-centred (...)" or "normal fallback"

    "INSUFFICIENT_DATA" (never a guess) under the same conditions as the
    analytic evaluator, and for a non-positive horizon.
    """
    if (inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None
            or inputs.n_laps_horizon is None or inputs.n_laps_horizon <= 0):
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None,
                "method": "monte_carlo"}

    cfg = config or MCConfig()
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.n_sims)
    H = float(inputs.n_laps_horizon)
    p_sc = float(np.clip(inputs.p_sc_next_n_laps, 0.0, 1.0))
    pace = float(inputs.predicted_pace_loss_per_lap)
    cliff_p = float(np.clip(inputs.cliff_probability_next_n_laps or 0.0, 0.0, 1.0))

    pools = pit_durations if pit_durations is not None else load_pit_durations(winsorize=cfg.winsorize)
    if pools is not None:
        green_pool, caution_pool = pools
        draw_green = lambda: rng.choice(green_pool, n)
        draw_caution = lambda: rng.choice(caution_pool, n)
        meta = _PIT_POOL_META.get((str(PIT_DURATIONS_CSV), cfg.winsorize)) if pit_durations is None else None
        source = (f"empirical, circuit-centred ({meta['n_green']} green / {meta['n_caution']} caution-window stops "
                  f"from {meta['n_circuits']} circuits)" if meta else
                  f"empirical ({len(green_pool)} green / {len(caution_pool)} caution-window stops, caller-supplied pools)")
    else:
        draw_green = lambda: rng.normal(NORMAL_PIT_LOSS_SECONDS, FALLBACK_GREEN_SD_SECONDS, n)
        draw_caution = lambda: rng.normal(CAUTION_WINDOW_PIT_DURATION_SECONDS, FALLBACK_CAUTION_SD_SECONDS, n)
        source = "normal fallback around calibrated constants (pit_stop_durations.csv not found)"

    # --- the futures -------------------------------------------------------
    sc_lands = rng.random(n) < p_sc
    sc_time = rng.uniform(0.0, H, n)                       # laps driven before the SC stop, if one comes
    laps_waited = np.where(sc_lands, sc_time, H)           # otherwise wait the whole horizon, then pit

    degradation = pace * laps_waited + rng.standard_normal(n) * cfg.pace_loss_rmse_s * np.sqrt(laps_waited)

    cliff_occurs = rng.random(n) < cliff_p
    cliff_time = rng.uniform(0.0, H, n)
    cliff_hit_before_stop = cliff_occurs & (cliff_time <= laps_waited)

    stop_if_sc = draw_caution() + RESTART_COLD_TYRE_PENALTY_SECONDS
    stop_if_no_sc = draw_green()
    cost_wait_each = (np.where(sc_lands, stop_if_sc, stop_if_no_sc)
                      + degradation + CLIFF_PENALTY_SCALE * cliff_hit_before_stop)
    cost_now_each = draw_green()
    saving = cost_now_each - cost_wait_each

    mean_saving = float(saving.mean())
    p_better = float((saving > 0).mean())
    se = float(saving.std(ddof=1) / np.sqrt(n))
    recommend_wait = mean_saving > 0 and p_better >= cfg.min_win_prob

    analytic = evaluate_sc_gamble(inputs)
    return {
        "recommendation": "WAIT" if recommend_wait else "NO_ADVANTAGE_TO_WAITING",
        "cost_pit_now": float(cost_now_each.mean()),
        "cost_wait": float(cost_wait_each.mean()),
        "expected_saving_if_wait": mean_saving,
        "method": "monte_carlo",
        "n_sims": n,
        "p_wait_better": p_better,
        "saving_p05": float(np.percentile(saving, 5)),
        "saving_p50": float(np.percentile(saving, 50)),
        "saving_p95": float(np.percentile(saving, 95)),
        "saving_std": float(saving.std(ddof=1)),
        "saving_se": se,
        "decision_confident": bool(abs(mean_saving) > 2 * se),
        "analytic_expected_saving": analytic.get("expected_saving_if_wait"),
        "pit_duration_source": source,
    }


if __name__ == "__main__":
    print("=== SC Gamble Evaluator scenarios (2026 calibration: validated pit-loss constants) ===")
    print(f"NORMAL_PIT_LOSS_SECONDS = {NORMAL_PIT_LOSS_SECONDS} (was 22.0, ASSUMPTION)")
    print(f"CAUTION_WINDOW_PIT_DURATION_SECONDS = {CAUTION_WINDOW_PIT_DURATION_SECONDS} "
          f"(was SC_PIT_LOSS_SECONDS=11.0, ASSUMPTION)\n")

    # Scenario 1: high P(SC), low degradation cost, low cliff risk. Even at
    # p_sc=0.5 this now correctly gives NO_ADVANTAGE_TO_WAITING with the
    # real constants, because D = normal - caution - cold_tyre = -0.717 is
    # negative and the (small but nonzero) degradation/cliff here only
    # push further toward NO_ADVANTAGE (see the module docstring's rule 6)
    # - under the OLD 22/11 assumption this scenario used to say WAIT.
    # That flip IS the calibration's point, not a regression to chase away.
    s1 = SCGambleInputs(p_sc_next_n_laps=0.5, predicted_pace_loss_per_lap=0.1,
                         cliff_probability_next_n_laps=0.01, n_laps_horizon=3)
    print("High P(SC), low degradation risk:", evaluate_sc_gamble(s1))

    # Scenario 2: low P(SC), high degradation cost - pit now should win,
    # same as before.
    s2 = SCGambleInputs(p_sc_next_n_laps=0.02, predicted_pace_loss_per_lap=1.5,
                         cliff_probability_next_n_laps=0.3, n_laps_horizon=3)
    print("Low P(SC), high degradation risk:", evaluate_sc_gamble(s2))

    # Scenario 3: moderate case - genuinely marginal, and now actually
    # sensitive to the real (smaller) caution discount rather than the
    # inflated old one.
    s3 = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.4,
                         cliff_probability_next_n_laps=0.05, n_laps_horizon=3)
    print("Moderate case:", evaluate_sc_gamble(s3))

    # Scenario 4: unknown P(SC) - should not guess, unchanged.
    s4 = SCGambleInputs(p_sc_next_n_laps=None, predicted_pace_loss_per_lap=0.4,
                         cliff_probability_next_n_laps=0.05, n_laps_horizon=3)
    print("Unknown P(SC):", evaluate_sc_gamble(s4))

    # Scenario 5: zero waiting cost at a realistic p_sc - demonstrates the
    # interpretable rule directly. With the pooled global constants this
    # should be NO_ADVANTAGE_TO_WAITING (the discount doesn't clear the
    # cold-tyre penalty) - not a bug, the calibration's real finding.
    s5 = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.0,
                         cliff_probability_next_n_laps=0.0, n_laps_horizon=5)
    r5 = evaluate_sc_gamble(s5)
    print("Zero degradation/cliff, p_sc=0.15 (pooled constants):", r5)
    print("  (expect NO_ADVANTAGE_TO_WAITING with the pooled global constants above - "
          "the ~0.78s discount doesn't clear the 1.5s cold-tyre penalty; this is the "
          "validated result, not a regression.)")
    # ------------------------------------------------------------------
    # Monte Carlo vs analytic, side by side (same inputs as the scenarios above)
    # ------------------------------------------------------------------
    print("\n=== Monte Carlo evaluator vs analytic (mean saving in seconds; + = waiting saves time) ===")
    print(f"{'scenario':<34}{'analytic':>9}{'MC mean':>9}{'+-2se':>7}{'P(wait wins)':>14}{'p05':>8}{'p95':>8}  MC rec")
    for name, s in [("1 high P(SC), low degr", s1), ("2 low P(SC), high degr", s2),
                    ("3 moderate", s3), ("5 zero degr, p=0.15", s5)]:
        r = evaluate_sc_gamble_mc(s)
        print(f"{name:<34}{r['analytic_expected_saving']:>9.2f}{r['expected_saving_if_wait']:>9.2f}"
              f"{2 * r['saving_se']:>7.2f}{r['p_wait_better']:>14.1%}{r['saving_p05']:>8.1f}{r['saving_p95']:>8.1f}"
              f"  {r['recommendation']}")
    print("  (source:", evaluate_sc_gamble_mc(s1)["pit_duration_source"], ")")
    print("  Unknown P(SC):", evaluate_sc_gamble_mc(s4))

    print("\n=== Break-even sweep: zero degradation/cliff, 5-lap horizon - how does P(SC) change the gamble? ===")
    print(f"{'p_sc':>6}{'analytic':>10}{'MC mean':>9}{'P(wait wins)':>14}{'MC rec':>26}")
    for p in (0.05, 0.10, 0.25, 0.50, 0.75, 1.00):
        si = SCGambleInputs(p_sc_next_n_laps=p, predicted_pace_loss_per_lap=0.0,
                             cliff_probability_next_n_laps=0.0, n_laps_horizon=5)
        r = evaluate_sc_gamble_mc(si)
        print(f"{p:>6.2f}{r['analytic_expected_saving']:>10.2f}{r['expected_saving_if_wait']:>9.2f}"
              f"{r['p_wait_better']:>14.1%}{r['recommendation']:>26}")
