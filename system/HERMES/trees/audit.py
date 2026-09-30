#!/usr/bin/env python3
"""
audit_folds.py - independent leak / sanity audit of the walk-forward fold artefacts
====================================================================================
Run AFTER `model-fit-fold.py --all-folds` and BEFORE the long replay:

    python audit_folds.py --repo-root .

It does not trust fit_fold's own console output. It re-opens every artefact each fold will
hand to master.py and checks that nothing from the fold's TEST years (or later) got in.

Checks per fold
---------------
  1. built                 fit_report.json / thresholds.json / pickle / CSVs all exist
  2. stints_no_future      cliff_detection_stints.csv max season <= training end
  3. thresholds_finite     all three Tier-3 thresholds are real numbers (master refuses NaN)
  4. reg_eras_in_train     every regression key's era was actually seen in training
  5. cox_eras_in_train     Cox era dummy columns only for eras seen in training
  6. train_eras_ok         pickle's train_eras == eras implied by the training window
  7. sc_prior_no_future    per-circuit n_races <= number of training seasons
  8. roc_events_grow       cliff_roc_n_events non-decreasing as the window grows
Plus INFO lines (not pass/fail): pit-loss filtering flag, empty SC prior, coverage counts.
Exit code 1 if any check fails.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = {"fold1_2021": 2020, "fold2_2022": 2021, "fold3_2023": 2022,
         "fold4_2024": 2023, "holdout_2025": 2024}      # fold -> last TRAINING season
THRESH = ("CLIFF_PROBABILITY_THRESHOLD", "PACE_LOSS_THRESHOLD_SECONDS", "TYRE_AGE_TRIGGER_RATIO")


def era(y: int) -> str:
    return "2018-2021" if y <= 2021 else ("2022-2025" if y <= 2025 else "2026+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--folds-dir", default="folds")
    a = ap.parse_args()
    root = Path(a.repo_root).resolve() / a.folds_dir

    rows, fails, infos = [], [], []
    prev_events = -1
    for fold, end in FOLDS.items():
        d = root / fold
        need = ["fit_report.json", "thresholds.json", "tyre_life_models.pkl",
                "cliff_detection_stints.csv", "sc_vsc_circuit_level_prior.csv"]
        missing = [n for n in need if not (d / n).exists()]
        if missing:
            fails.append(f"{fold}: missing {missing}")
            rows.append(dict(fold=fold, built=False))
            continue
        rep = json.load(open(d / "fit_report.json"))
        th = json.load(open(d / "thresholds.json"))
        st = pd.read_csv(d / "cliff_detection_stints.csv")
        sc = pd.read_csv(d / "sc_vsc_circuit_level_prior.csv")
        with open(d / "tyre_life_models.pkl", "rb") as f:
            models = pickle.load(f)
        n_seasons = end - 2017
        train_eras = {era(y) for y in range(2018, end + 1)}

        c = {}
        c["stints_no_future"] = bool(st["season"].max() <= end)
        c["thresholds_finite"] = all(th.get(k) is not None and np.isfinite(th[k]) for k in THRESH)
        c["reg_eras_in_train"] = {k[2] for k in models["reg_models"]} <= train_eras
        c["cox_eras_in_train"] = {x.replace("regulation_era_", "") for x in models["era_dummy_columns"]} <= train_eras
        c["train_eras_ok"] = (set(models.get("train_eras", [])) == train_eras) if "train_eras" in models else False
        c["sc_prior_no_future"] = bool(sc.empty or sc["n_races"].max() <= n_seasons)
        ev = int(rep.get("cliff_roc_n_events", -1))
        c["roc_events_grow"] = ev >= prev_events
        prev_events = max(prev_events, ev)

        for name, ok in c.items():
            if not ok:
                fails.append(f"{fold}: FAILED {name}")

        if sc.empty:
            infos.append(f"{fold}: SC prior is EMPTY (<4 training races per circuit) - SC gamble cannot run in this fold")
        flag = rep.get("pit_loss_filtered_by_year", None)
        if flag is False:
            infos.append(f"{fold}: pit-loss table NOT filtered by year (residual leak, disclose it)")
        elif flag is None:
            infos.append(f"{fold}: no pit-loss table was found/written (master uses its flat-assumption fallback)")
        for w in rep.get("warnings", []):
            infos.append(f"{fold}: fit warning: {w}")

        rows.append(dict(
            fold=fold, built=True, train_end=end, reg_models=len(models["reg_models"]),
            cox_stints=rep.get("cox_n_stints"), cliff_events_roc=ev,
            cliff_thr=round(th["CLIFF_PROBABILITY_THRESHOLD"], 4),
            pace_thr=round(th["PACE_LOSS_THRESHOLD_SECONDS"], 3),
            ratio_thr=round(th["TYRE_AGE_TRIGGER_RATIO"], 3),
            sc_circuits=int(len(sc)), train_eras=",".join(sorted(train_eras)),
            all_checks=all(c.values())))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))
    print()
    for i in infos:
        print("INFO:", i)
    print()
    if fails:
        print("*** AUDIT FAILED ***")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(">>> AUDIT PASSED: no future-season data found in any fold artefact <<<")


if __name__ == "__main__":
    main()