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

COST MODEL:
  cost_pit_now = NORMAL_PIT_LOSS_SECONDS

  cost_wait = P_sc * (SC_PIT_LOSS_SECONDS + RESTART_COLD_TYRE_PENALTY_SECONDS
                       + partial_degradation_cost_before_sc)
            + (1 - P_sc) * (NORMAL_PIT_LOSS_SECONDS + full_degradation_cost_of_waiting)

  Recommend WAIT if cost_wait < cost_pit_now, else NO_ADVANTAGE_TO_WAITING.

ALL COST CONSTANTS BELOW ARE ASSUMPTIONS, not calibrated against real data -
same treatment every other Gate Tree threshold got before being trusted.
SC_DURATION_LAPS_BY_CIRCUIT is the one exception - genuinely computable from
existing is_sc_lap flags, built here as a real historical statistic rather
than guessed. Everything else (pit loss deltas, restart penalty) should get
the same calibration treatment as CLIFF_PROBABILITY_THRESHOLD etc. before
this evaluator is trusted for a live decision.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

NORMAL_PIT_LOSS_SECONDS = 22.0   # ASSUMPTION - typical F1 pit lane loss, circuit-independent for now
SC_PIT_LOSS_SECONDS = 11.0       # ASSUMPTION - roughly half, under SC bunching/reduced speed delta
RESTART_COLD_TYRE_PENALTY_SECONDS = 1.5  # ASSUMPTION - extra cost on the out-lap after an SC restart


def compute_historical_sc_duration(laps: pd.DataFrame) -> pd.DataFrame:
    """
    REAL, data-derived (not guessed): average length of an SC period in laps,
    per circuit, from consecutive is_sc_lap=True runs. Requires is_sc_lap
    (already engineered from race control messages) - reused infrastructure.
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
    predicted_pace_loss_per_lap: Optional[float]  # from tyre_life_projection - regression V2 output
    cliff_probability_next_n_laps: Optional[float]  # from tyre_life_projection - Cox output
    n_laps_horizon: int                       # how many laps ahead this decision considers


def evaluate_sc_gamble(inputs: SCGambleInputs) -> dict:
    """
    Returns the expected cost of each option plus a recommendation. Returns
    "INSUFFICIENT_DATA" (not a guess) if P(SC) or pace-loss inputs are
    unavailable - same None-safety pattern as every other tyre-model consumer.
    """
    if inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None:
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None}

    p_sc = inputs.p_sc_next_n_laps
    degradation_cost_of_waiting = inputs.predicted_pace_loss_per_lap * inputs.n_laps_horizon

    # If a cliff is likely within the horizon, waiting risks a MUCH larger cost
    # than smooth degradation alone - add this as an extra expected penalty,
    # scaled by how likely the cliff actually is within this same horizon.
    cliff_risk = inputs.cliff_probability_next_n_laps or 0.0
    cliff_penalty_seconds = 5.0 * cliff_risk  # ASSUMPTION - flat penalty scaled by probability

    cost_pit_now = NORMAL_PIT_LOSS_SECONDS

    cost_if_sc_comes = SC_PIT_LOSS_SECONDS + RESTART_COLD_TYRE_PENALTY_SECONDS + (degradation_cost_of_waiting / 2)
    cost_if_sc_doesnt_come = NORMAL_PIT_LOSS_SECONDS + degradation_cost_of_waiting + cliff_penalty_seconds
    cost_wait = p_sc * cost_if_sc_comes + (1 - p_sc) * cost_if_sc_doesnt_come

    recommendation = "WAIT" if cost_wait < cost_pit_now else "NO_ADVANTAGE_TO_WAITING"

    return {
        "recommendation": recommendation,
        "cost_pit_now": cost_pit_now,
        "cost_wait": cost_wait,
        "expected_saving_if_wait": cost_pit_now - cost_wait,
    }


if __name__ == "__main__":
    print("=== SC Gamble Evaluator scenarios ===")

    # Scenario 1: high P(SC), low degradation cost, no cliff risk -> WAIT should win
    s1 = SCGambleInputs(p_sc_next_n_laps=0.5, predicted_pace_loss_per_lap=0.1,
                         cliff_probability_next_n_laps=0.01, n_laps_horizon=3)
    print("High P(SC), low degradation risk:", evaluate_sc_gamble(s1))

    # Scenario 2: low P(SC), high degradation cost -> pit now should win
    s2 = SCGambleInputs(p_sc_next_n_laps=0.02, predicted_pace_loss_per_lap=1.5,
                         cliff_probability_next_n_laps=0.3, n_laps_horizon=3)
    print("Low P(SC), high degradation risk:", evaluate_sc_gamble(s2))

    # Scenario 3: moderate P(SC), moderate everything - genuinely marginal case
    s3 = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.4,
                         cliff_probability_next_n_laps=0.05, n_laps_horizon=3)
    print("Moderate case:", evaluate_sc_gamble(s3))

    # Scenario 4: unknown P(SC) - should not guess
    s4 = SCGambleInputs(p_sc_next_n_laps=None, predicted_pace_loss_per_lap=0.4,
                         cliff_probability_next_n_laps=0.05, n_laps_horizon=3)
    print("Unknown P(SC):", evaluate_sc_gamble(s4))