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
from typing import Optional
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