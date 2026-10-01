
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

NORMAL_PIT_LOSS_SECONDS = 23.685

CAUTION_WINDOW_PIT_DURATION_SECONDS = 22.902

RESTART_COLD_TYRE_PENALTY_SECONDS = 1.5

CLIFF_PENALTY_SCALE = 5.0

def compute_historical_sc_duration(laps: pd.DataFrame) -> pd.DataFrame:
    ""
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
    p_sc_next_n_laps: Optional[float]

    predicted_pace_loss_per_lap: Optional[float]

    cliff_probability_next_n_laps: Optional[float]
    n_laps_horizon: int

def evaluate_sc_gamble(inputs: SCGambleInputs) -> dict:
    ""
    if inputs.p_sc_next_n_laps is None or inputs.predicted_pace_loss_per_lap is None:
        return {"recommendation": "INSUFFICIENT_DATA", "cost_pit_now": None, "cost_wait": None}

    p_sc = inputs.p_sc_next_n_laps
    n = inputs.n_laps_horizon
    pace = inputs.predicted_pace_loss_per_lap
    cliff_risk = inputs.cliff_probability_next_n_laps or 0.0

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

PIT_DURATIONS_CSV = Path(__file__).resolve().parents[3] / "pit_stop_durations.csv"

PACE_LOSS_RMSE_SECONDS = 0.057

MIN_STOPS_FOR_CIRCUIT_BASELINE = 30

MIN_STOPS_FOR_CIRCUIT_WINSOR = 10

FALLBACK_GREEN_SD_SECONDS = 1.80
FALLBACK_CAUTION_SD_SECONDS = 4.66

@dataclass
class MCConfig:
    n_sims: int = 20000
    seed: int = 0
    winsorize: tuple = (0.10, 0.90)
    pace_loss_rmse_s: float = PACE_LOSS_RMSE_SECONDS
    min_win_prob: float = 0.0

_PIT_POOL_CACHE: dict = {}
_PIT_POOL_META: dict = {}

def load_pit_durations(path: Optional[Path] = None, winsorize: tuple = (0.10, 0.90)):
    ""
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
    ""
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

    sc_lands = rng.random(n) < p_sc
    sc_time = rng.uniform(0.0, H, n)
    laps_waited = np.where(sc_lands, sc_time, H)

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

    s1 = SCGambleInputs(p_sc_next_n_laps=0.5, predicted_pace_loss_per_lap=0.1,
                         cliff_probability_next_n_laps=0.01, n_laps_horizon=3)
    print("High P(SC), low degradation risk:", evaluate_sc_gamble(s1))

    s2 = SCGambleInputs(p_sc_next_n_laps=0.02, predicted_pace_loss_per_lap=1.5,
                         cliff_probability_next_n_laps=0.3, n_laps_horizon=3)
    print("Low P(SC), high degradation risk:", evaluate_sc_gamble(s2))

    s3 = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.4,
                         cliff_probability_next_n_laps=0.05, n_laps_horizon=3)
    print("Moderate case:", evaluate_sc_gamble(s3))

    s4 = SCGambleInputs(p_sc_next_n_laps=None, predicted_pace_loss_per_lap=0.4,
                         cliff_probability_next_n_laps=0.05, n_laps_horizon=3)
    print("Unknown P(SC):", evaluate_sc_gamble(s4))

    s5 = SCGambleInputs(p_sc_next_n_laps=0.15, predicted_pace_loss_per_lap=0.0,
                         cliff_probability_next_n_laps=0.0, n_laps_horizon=5)
    r5 = evaluate_sc_gamble(s5)
    print("Zero degradation/cliff, p_sc=0.15 (pooled constants):", r5)
    print("  (expect NO_ADVANTAGE_TO_WAITING with the pooled global constants above - "
          "the ~0.78s discount doesn't clear the 1.5s cold-tyre penalty; this is the "
          "validated result, not a regression.)")

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
