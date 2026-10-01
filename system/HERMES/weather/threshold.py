""

from dataclasses import dataclass

@dataclass
class CrossoverCostModel:
    ""
    pit_loss_seconds: float
    dry_on_inters_penalty: float

    wet_on_slicks_penalty: float

    wet_on_inters_gain: float

    def crossover_probability(self) -> float:
        ""
        numerator = self.dry_on_inters_penalty + self.pit_loss_seconds
        denominator = (self.wet_on_inters_gain
                       + self.wet_on_slicks_penalty
                       + self.dry_on_inters_penalty)
        if denominator <= 0:
            raise ValueError("Denominator must be positive -- check cost model inputs.")
        return numerator / denominator

def scenario_early_race() -> CrossoverCostModel:
    ""
    return CrossoverCostModel(
        pit_loss_seconds=22.0,
        dry_on_inters_penalty=30.0,
        wet_on_slicks_penalty=18.0,
        wet_on_inters_gain=12.0,
    )

def scenario_late_race() -> CrossoverCostModel:
    ""
    return CrossoverCostModel(
        pit_loss_seconds=22.0,
        dry_on_inters_penalty=5.0,
        wet_on_slicks_penalty=20.0,
        wet_on_inters_gain=12.0,
    )

def scenario_mid_race() -> CrossoverCostModel:
    ""
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
