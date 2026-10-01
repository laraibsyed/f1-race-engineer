
""
from __future__ import annotations

import importlib.util as _ilu
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

TREES_DIR = Path(__file__).resolve().parent
REPO_ROOT = TREES_DIR.parent.parent.parent
OUT_DIR = REPO_ROOT / "eval_out_team_strategy"
REPLAYS_DIR = OUT_DIR / "replays"

def _load(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

print("[setup] loading master.py ...")
master = _load("master_eval", TREES_DIR / "master.py")
evaluate_mod = _load("evaluate_eval", TREES_DIR / "evaluate.py")
team_strategy = master.team_strategy
if team_strategy is None or master.reward_mod is None:
    raise SystemExit("team-strategy.py or strategic-sacrifice/reward.py could not be loaded - "
                      "see the [hermes] notes master.py printed above for which one.")

RACES = [
    (2021, "Bahrain_Grand_Prix"), (2021, "British_Grand_Prix"),
    (2022, "Bahrain_Grand_Prix"), (2022, "Italian_Grand_Prix"),
    (2023, "Bahrain_Grand_Prix"), (2023, "Monaco_Grand_Prix"),
    (2024, "Bahrain_Grand_Prix"), (2024, "British_Grand_Prix"),
    (2025, "Bahrain_Grand_Prix"), (2025, "Australian_Grand_Prix"),
]
RISK_MODES = ("CONSERVATIVE", "BALANCED", "AGGRESSIVE")
N_SIMS = 200

tyre_models = master.load_tyre_models(master._data_path(REPO_ROOT, "tyre_life_models.pkl"))
sc_prior = master.load_sc_prior(master._data_path(REPO_ROOT, "sc_vsc_circuit_level_prior.csv"))
cliff_stints = master.load_cliff_stints(master._data_path(REPO_ROOT, "cliff_detection_stints.csv"))
taxonomy = master.load_circuit_taxonomy(REPO_ROOT / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
pit_loss_table = master.load_pit_loss_table(
    master._data_path(REPO_ROOT, "checkpoints", "rival_knowledge", "archive_per_race_analysis.csv"))
driver_taxonomy = master.load_driver_taxonomy(master._data_path(REPO_ROOT, "driver_taxonomy_master_final.csv"))

def build_ctx(season: int, race: str, d1: str, d2: str):
    degr_ordinal, _ = master.circuit_degredation_ordinal_for(taxonomy, race)
    pit_loss_s, _ = master.pit_loss_for_circuit(pit_loss_table, race, season=season)
    weather_df = master.load_weather(season, race, "R")
    laps_path = master.find_laps_features(season, race, "R")
    total_laps = int(pd.read_csv(laps_path, usecols=["LapNumber"])["LapNumber"].max())
    ctx = master.RaceContext(
        season=season, circuit=race, total_laps=total_laps, is_sprint_weekend=False,
        d1_code=d1, d2_code=d2, session="R", circuit_degredation_ordinal=degr_ordinal,
        pit_loss_s=pit_loss_s, p_sc_5lap=master.get_sc_probability(sc_prior, race),
    )
    resources = {"tyre_models": tyre_models, "cliff_stints": cliff_stints}
    return ctx, resources, weather_df, range(1, total_laps + 1)

def strip_internal(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("t1_state", "t3_state") and not k.startswith("_")}

def write_jsonl(path: Path, decisions: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(strip_internal(d), default=str) + "\n")

def main():
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    run_log = []
    t_start = time.time()

    for season, race in RACES:
        d1, d2 = evaluate_mod.rbr_drivers_for(season, race)
        print(f"\n=== {season} {race}  (D1={d1} D2={d2}) ===")
        ctx, resources, weather_df, lap_range = build_ctx(season, race, d1, d2)
        tag = f"{season}_{race}"

        t0 = time.time()
        dec_baseline = master.run_replay(ctx, resources, lap_range, weather_df=weather_df)
        write_jsonl(REPLAYS_DIR / f"{tag}__baseline.jsonl", dec_baseline)
        run_log.append(dict(season=season, race=race, condition="baseline",
                             n_decisions=len(dec_baseline), seconds=round(time.time() - t0, 2)))
        print(f"  baseline: {len(dec_baseline)} decisions in {time.time() - t0:.1f}s")

        champ_ctx = master.fetch_championship_context(season, race, d1, d2)
        print(f"  championship context: {'live' if champ_ctx['available'] else 'UNAVAILABLE'} "
              f"(round={champ_ctx.get('round_num')}, nearest_rival={champ_ctx.get('nearest_rival_code')})")

        for mode in RISK_MODES:
            t0 = time.time()
            ts_fn = master.build_team_strategy_context(ctx, resources, driver_taxonomy, champ_ctx,
                                                        risk_mode=mode, n_sims=N_SIMS)
            dec = master.run_replay(ctx, resources, lap_range, weather_df=weather_df, team_strategy_fn=ts_fn)
            write_jsonl(REPLAYS_DIR / f"{tag}__ts_{mode.lower()}_rival_on.jsonl", dec)
            run_log.append(dict(season=season, race=race, condition=f"team_strategy_{mode.lower()}_rival_on",
                                 n_decisions=len(dec), seconds=round(time.time() - t0, 2)))
            print(f"  team_strategy[{mode}] rival=ON: {len(dec)} decisions in {time.time() - t0:.1f}s")

        original_predict = team_strategy.predict_rival_pit_window
        team_strategy.predict_rival_pit_window = lambda *a, **kw: None
        try:
            t0 = time.time()
            ts_fn_off = master.build_team_strategy_context(ctx, resources, driver_taxonomy, champ_ctx,
                                                             risk_mode="BALANCED", n_sims=N_SIMS)
            dec_off = master.run_replay(ctx, resources, lap_range, weather_df=weather_df,
                                         team_strategy_fn=ts_fn_off)
        finally:
            team_strategy.predict_rival_pit_window = original_predict
        write_jsonl(REPLAYS_DIR / f"{tag}__ts_balanced_rival_off.jsonl", dec_off)
        run_log.append(dict(season=season, race=race, condition="team_strategy_balanced_rival_off",
                             n_decisions=len(dec_off), seconds=round(time.time() - t0, 2)))
        print(f"  team_strategy[BALANCED] rival=OFF: {len(dec_off)} decisions in {time.time() - t0:.1f}s")

    pd.DataFrame(run_log).to_csv(OUT_DIR / "run_log.csv", index=False)
    print(f"\n[done] {len(RACES)} races x 5 conditions in {time.time() - t_start:.1f}s total. "
          f"Raw JSONL under {REPLAYS_DIR}, run_log.csv under {OUT_DIR}")

if __name__ == "__main__":
    main()
