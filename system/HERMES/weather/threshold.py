"""
crossover_threshold_derivation.py
-----------------------------------
Derives the rain-probability crossover threshold from expected utility, rather
than treating 35/45/60% as arbitrary hand-picked numbers. This turns the
provisional thresholds in rain_crossover.py into worked examples of what the
theoretical crossover point looks like under different race-state assumptions,
rather than unexplained constants.

THE MODEL
---------
At any point in the race, the driver/strategist chooses between two tyre
states: SLICKS (stay out) or INTERS (pit now). Let P(R) be the probability
it's raining hard enough to matter for the remainder of the relevant window.

Expected utility of staying on slicks:
    EU(SLICKS) = (1 - P(R)) * 0  -  P(R) * C_wet_on_slicks
  (if it stays dry, no penalty; if it rains, a heavy penalty for being on the
   wrong tyre -- lost time, spins, possible DNF risk)

Expected utility of switching to inters:
    EU(INTERS) = P(R) * G_wet  -  (1 - P(R)) * C_dry_on_inters  -  C_pit
  (if it does rain, you gain by already being on the right tyre; if it stays
   dry, you pay a time penalty for being on inters on a dry track; either way
   you pay the pit-stop time loss)

Switch to inters when EU(INTERS) > EU(SLICKS). Solving for P(R) gives the
crossover probability P*:

    P* = (C_dry_on_inters + C_pit) / (G_wet + C_wet_on_slicks + C_dry_on_inters)

This is the SAME asymmetric-risk logic already argued qualitatively earlier
in this project (being wrong on the side of caution costs less than being
wrong on the side of staying out too long) -- this module just makes it
quantitative and shows where 35/45/60% could plausibly come from, rather than
presenting them as arbitrarily chosen.

WHAT THIS IS AND ISN'T
-----------------------
This IS a legitimate derivation of a decision threshold from a cost model.
This is NOT a claim that the specific cost values below (in seconds) are
empirically measured for any real circuit or car -- they are illustrative,
sourced from general knowledge of F1 pit-stop and tyre-crossover behaviour
(pit loss ~20-25s typical; wet-tyre-on-dry-track penalty vs dry-tyre-on-wet-
track penalty are asymmetric because aquaplaning risk is a discontinuous,
not just gradual, cost). The derived P* values do NOT numerically match the
35/45/60% used in rain_crossover.py, and this script does not force them to --
see the closing note in main() for why that gap is itself an honest, reportable
finding rather than a discrepancy to hide. The contribution is the derivation
structure and its correct qualitative ordering (EARLY requires higher
confidence than LATE), not a claim of numerical calibration.
"""

from dataclasses import dataclass


@dataclass
class CrossoverCostModel:
    """All costs in seconds (or a comparable single unit) lost, so P* is
    dimensionless and comparable across scenarios."""
    pit_loss_seconds: float          # C_pit: time lost in the pit stop itself
    dry_on_inters_penalty: float     # C_dry_on_inters: time lost per relevant
                                      # lap running inters on a track that stays dry
    wet_on_slicks_penalty: float     # C_wet_on_slicks: time/risk cost of staying
                                      # on slicks if it does rain (this is where
                                      # the asymmetry lives -- includes a risk
                                      # premium for aquaplaning/spin/DNF, not just
                                      # lap time)
    wet_on_inters_gain: float        # G_wet: time gained per relevant lap by
                                      # already being on inters if it does rain

    def crossover_probability(self) -> float:
        """P*: the rain probability at which EU(INTERS) == EU(SLICKS).
        Above P*, switching is the higher-expected-utility choice."""
        numerator = self.dry_on_inters_penalty + self.pit_loss_seconds
        denominator = (self.wet_on_inters_gain
                       + self.wet_on_slicks_penalty
                       + self.dry_on_inters_penalty)
        if denominator <= 0:
            raise ValueError("Denominator must be positive -- check cost model inputs.")
        return numerator / denominator


def scenario_early_race() -> CrossoverCostModel:
    """Many laps remaining: a wrong early switch to inters costs many laps of
    dry-on-inters penalty before the stop even pays for itself, so the model
    should require a HIGHER P(R) to justify switching. This is what makes the
    EARLY stage's threshold higher than LATE's in rain_crossover.py -- not an
    assumption bolted on afterward, but a consequence of more remaining laps
    inflating dry_on_inters_penalty's total impact.

    dry_on_inters_penalty here represents the TOTAL accumulated cost of
    running inters on a dry track for the remaining stint (not per-lap) --
    scaled up for early race because there are more laps for that penalty
    to accumulate over before the tyres would need changing again anyway."""
    return CrossoverCostModel(
        pit_loss_seconds=22.0,        # typical modern F1 pit loss (illustrative)
        dry_on_inters_penalty=30.0,   # total accumulated cost if wrong, many laps left
        wet_on_slicks_penalty=18.0,   # risk cost if it rains and you're caught out
        wet_on_inters_gain=12.0,      # benefit if it rains and you're already switched
    )


def scenario_late_race() -> CrossoverCostModel:
    """Few laps remaining: the dry-on-inters penalty has little time left to
    accumulate (few laps remain regardless of tyre choice), while the
    wet-on-slicks risk stays high or increases slightly -- less time to
    recover position after a mistake this late (Norris at Sochi 2021 is the
    illustrative case: a late-race slide on slicks cost him the win with no
    laps left to recover)."""
    return CrossoverCostModel(
        pit_loss_seconds=22.0,
        dry_on_inters_penalty=5.0,    # much less impactful with few laps left
        wet_on_slicks_penalty=20.0,   # higher stakes: less time to recover position
        wet_on_inters_gain=12.0,
    )


def scenario_mid_race() -> CrossoverCostModel:
    """Interpolates between early and late -- not a new assumption, just the
    same cost structure with moderate values."""
    return CrossoverCostModel(
        pit_loss_seconds=22.0,
        dry_on_inters_penalty=15.0,
        wet_on_slicks_penalty=19.0,
        wet_on_inters_gain=12.0,
    )


if __name__ == "__main__":
    scenarios = {
        "EARLY (>60% laps remaining)": scenario_early_race(),
        "MID (20-60% laps remaining)": scenario_mid_race(),
        "LATE (<20% laps remaining)": scenario_late_race(),
    }

    print("=== Crossover threshold derived from expected utility ===\n")
    print(f"{'Stage':<32}{'P* (derived)':<15}{'rain_crossover.py threshold'}")
    rain_crossover_thresholds = {
        "EARLY (>60% laps remaining)": 60,
        "MID (20-60% laps remaining)": 45,
        "LATE (<20% laps remaining)": 35,
    }
    for stage_name, model in scenarios.items():
        p_star = model.crossover_probability()
        used_threshold = rain_crossover_thresholds[stage_name]
        print(f"{stage_name:<32}{p_star*100:>6.1f}%       {used_threshold}% (used in code)")

    print("\nNote on scale: these illustrative costs were chosen to demonstrate "
          "the DIRECTION and STRUCTURE of the derivation (why EARLY should "
          "require higher confidence than LATE, and why that gap follows from "
          "an asymmetric cost model rather than intuition alone). The derived "
          "P* values land higher than the 35/45/60% used in rain_crossover.py "
          "because the illustrative wet_on_slicks_penalty here is deliberately "
          "conservative (modelling a driver who copes reasonably well in light "
          "rain) rather than tuned to match the code's thresholds exactly. This "
          "gap is itself worth reporting honestly: it shows the provisional "
          "35/45/60% would correspond to a strategist who weighs the risk of "
          "being caught out in the wet more heavily than these illustrative "
          "costs do -- i.e. errs toward switching early, consistent with the "
          "asymmetric-risk argument (a late switch risking a Norris-at-Sochi-style "
          "off is worse than an early switch losing a few seconds). The dissertation "
          "point is that the MODEL STRUCTURE explains and can reproduce the "
          "existing threshold ordering; exact numerical alignment would require "
          "real cost data this project doesn't have access to (see limitation "
          "in module docstring), and forcing it via  cost tuning would be curve-"
          "fitting the derivation to a conclusion rather than deriving it.")