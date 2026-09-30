#!/usr/bin/env python3
"""
fit_fold.py - build ONE walk-forward fold's artefacts from TRAINING YEARS ONLY
============================================================================
Why this exists
---------------
The production tyre pickle and the three Tier-3 thresholds were fit on 100% of
the data (model-fit.py says so itself). Replaying 2021-2025 with them means the
"test" years were seen during fitting, so every fold in the first evaluation is
optimistic. This script closes that leak.

What gets refit per fold (everything that saw outcome data)
-----------------------------------------------------------
  1. Regression V2 models   (compound, circuit, era) LinearRegression   <- model-fit.py logic
  2. Cox V2 model           stratified by compound                      <- model-fit.py logic
  3. cliff_detection_stints.csv filtered to train seasons               (feeds Cox AND expected_stint_length)
  4. Three Tier-3 thresholds recomputed on train years:
        PACE_LOSS_THRESHOLD_SECONDS = P90 of pace_loss over clean dry laps
        TYRE_AGE_TRIGGER_RATIO      = P10 of cliff_tyre_age / n_laps_true over cliff stints
        CLIFF_PROBABILITY_THRESHOLD = Youden-optimal point of the ROC of Cox cliff-probability
                                      vs "cliff occurs within 5 laps"
  5. SC circuit prior       rebuilt from train-year races only
  6. Per-circuit pit-loss   archive_per_race_analysis.csv filtered to train years (if it has a year col)

Output layout (one folder per fold, used as --repo-root overlay by evaluate)
---------------------------------------------------------------------------
  <out>/<fold_name>/tyre_life_models.pkl
  <out>/<fold_name>/cliff_detection_stints.csv
  <out>/<fold_name>/sc_vsc_circuit_level_prior.csv
  <out>/<fold_name>/thresholds.json
  <out>/<fold_name>/fit_report.json        (coverage counts, what fell back, warnings)
The fold folder is meant to be handed to master.py via HERMES_FOLD_DIR (see
master_patch.md). Files that master.py reads from --repo-root are looked up in
the fold folder FIRST, then fall back to the real repo root.

ASSUMPTIONS (flagged, because I could not see the original calibration scripts)
-------------------------------------------------------------------------------
  A1. (RESOLVED) All three thresholds now follow threshold-calibration.py exactly. The cliff
      threshold copies Part 3 line for line (cliff stints only; 5 positives at duration-1..-5;
      1 negative at duration-10; Youden's J). CAVEAT: gate-tier-3.py says 0.017 came from 533
      cliff events, but the tyre-age ratio was later computed on 611 (after the n_laps fix), so
      today's CSV may give a slightly different Youden point than 0.017. --check-reproduces
      therefore ALSO runs your original script on today's CSV; if both print the same number,
      the re-implementation is verified and any gap to 0.017 is staleness in the constant.
  A2. SC prior formula from the blueprint: p_window = p_race_incidence * (5 / mean_total_laps),
      min 4 races per circuit. Incidence = fraction of races with any SC/VSC lap.
  A3. Sprint sessions excluded from the Cox fit (matches model-fit.py: session == "R").

Usage
-----
  python fit_fold.py --repo-root . --fold fold2_2022            # one fold
  python fit_fold.py --repo-root . --all-folds                  # folds 1-4 + holdout
  python fit_fold.py --repo-root . --fold all_data --check-reproduces
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# Fold definitions (must match hermes_evaluate.py)
# ----------------------------------------------------------------------------
FOLD_TRAIN_YEARS = {
    "fold1_2021": (2018, 2020),
    "fold2_2022": (2018, 2021),
    "fold3_2023": (2018, 2022),
    "fold4_2024": (2018, 2023),
    "holdout_2025": (2018, 2024),
    "all_data": (2018, 2100),      # reproduces production, used ONLY for --check-reproduces
}
CLIFF_HORIZON = 5
NEGATIVE_CHECKPOINT_BUFFER = 10   # same as threshold-calibration.py Part 3
PRODUCTION_REFERENCE = {          # what your gate-tier-3.py currently says
    "CLIFF_PROBABILITY_THRESHOLD": 0.017,
    "PACE_LOSS_THRESHOLD_SECONDS": 1.805,
    "TYRE_AGE_TRIGGER_RATIO": 0.421,
}
MIN_CIRCUIT_RACES_FOR_SC_PRIOR = 4


# ----------------------------------------------------------------------------
# Load model-fit.py as a module (hyphenated filename, so importlib)
# ----------------------------------------------------------------------------
def load_model_fit(repo_root: Path):
    p = repo_root / "system" / "HERMES" / "trees" / "model-fit.py"
    if not p.exists():
        raise FileNotFoundError(f"model-fit.py not found at {p}")
    spec = importlib.util.spec_from_file_location("model_fit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)       # main block is guarded, so this is side-effect free
    return mod


# ----------------------------------------------------------------------------
# Data loading from the LOCAL cache only (no GCS calls, no credentials needed)
# ----------------------------------------------------------------------------
class LocalCache:
    """Duck-types model-fit.CachedBucket but only ever reads ./gcs_cache, never GCS.
    Raises loudly if a file is missing rather than silently downloading."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def read_csv(self, blob_path, **kwargs):
        p = self.cache_dir / blob_path
        if not p.exists():
            raise FileNotFoundError(f"{p} not in local cache (fit_fold never calls GCS)")
        return pd.read_csv(p, **kwargs)

    def list_blob_names(self, prefix):
        base = self.cache_dir / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.cache_dir)).replace("\\", "/")
                      for p in base.rglob("laps_features.csv"))


def load_laps_for_years(mf, cache: LocalCache, y0: int, y1: int) -> pd.DataFrame:
    """Same loading as model-fit.load_all_teams_laps but restricted to seasons in [y0, y1]."""
    paths = [p for p in cache.list_blob_names("clean/features/")
             if p.endswith("laps_features.csv") and ("/R/" in p or "/S/" in p)]
    frames = []
    for p in paths:
        season = int(p.split("/")[2])
        if not (y0 <= season <= y1):
            continue
        df = cache.read_csv(p, dtype={"TrackStatus": str})
        parts = p.split("/")
        df["Season"], df["Race"], df["Session"] = season, parts[3], parts[4]
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No laps_features.csv found for seasons {y0}-{y1} under {cache.cache_dir}")
    full = pd.concat(frames, ignore_index=True)
    full["is_rbr"] = full["Team"].isin(mf.RBR_ALIASES)
    return full


# ----------------------------------------------------------------------------
# 1+2. Models
# ----------------------------------------------------------------------------
def fit_models(mf, repo_root: Path, laps: pd.DataFrame, stints_train: pd.DataFrame, report: dict):
    clean = mf.filter_valid_laps(laps)
    clean = mf.filter_global_degradation_outliers(clean)
    modeling_df = mf.add_pace_loss_and_era(clean)
    report["regression_rows"] = int(len(modeling_df))

    reg_temp_cols = sorted(pd.get_dummies(
        modeling_df["track_temp_bucket"], prefix="temp", drop_first=True).columns.tolist())
    reg_models = mf.fit_regression_v2_models(modeling_df, reg_temp_cols)
    report["n_regression_models"] = len(reg_models)
    report["regression_eras_present"] = sorted({k[2] for k in reg_models})

    # Cox. model-fit's prepare_cox_data reads TAXONOMY_PATH relative to cwd, so chdir to repo root.
    import os
    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        stints = mf.prepare_cox_data(stints_train.copy(), bucket=None)
    finally:
        os.chdir(cwd)
    design, cox_temp_cols, era_cols = mf.build_cox_design_matrix(stints)
    cph = mf.fit_cox_v2_model(design)
    report["cox_concordance"] = float(cph.concordance_index_)
    report["cox_n_stints"] = int(len(design))
    report["cox_eras_present"] = [c.replace("regulation_era_", "") for c in era_cols]

    if set(reg_temp_cols) != set(cox_temp_cols):
        report.setdefault("warnings", []).append(
            f"temp dummy mismatch reg={reg_temp_cols} cox={cox_temp_cols}; using regression list")
    return {"reg_models": reg_models, "cph": cph,
            "temp_dummy_columns": reg_temp_cols, "era_dummy_columns": era_cols}, modeling_df, stints


# ----------------------------------------------------------------------------
# 4. Thresholds
# ----------------------------------------------------------------------------
def threshold_pace_loss(modeling_df: pd.DataFrame) -> float:
    """P90 of pace_loss_seconds over clean dry laps (same population Regression V2 trains on)."""
    v = modeling_df["pace_loss_seconds"].dropna()
    return float(v.quantile(0.90))


def threshold_tyre_age_ratio(stints_train: pd.DataFrame) -> float:
    """P10 of cliff_tyre_age / n_laps_true across stints where a cliff was detected (event==1)."""
    s = stints_train
    if "event" in s.columns:
        s = s[s["event"] == 1]
    s = s.dropna(subset=["cliff_tyre_age", "n_laps_true"])
    s = s[s["n_laps_true"] > 0]
    if s.empty:
        return float("nan")
    return float((s["cliff_tyre_age"] / s["n_laps_true"]).quantile(0.10))


def threshold_cliff_probability(models: dict, stints_cox: pd.DataFrame, report: dict):
    """
    FAITHFUL COPY of threshold-calibration.py, Part 3 (calibrate_cliff_probability_threshold).
    Only the data window (train years) and the Cox model (this fold's) differ.

      * Only stints with event == 1 (a real detected cliff) are used. Censored stints are
        deliberately EXCLUDED (their true label near the end of observation is unknowable).
      * POSITIVE checkpoints: 5 per cliff stint, at tyre age  duration-1 ... duration-5   (label 1)
      * NEGATIVE checkpoint : 1 per cliff stint, at tyre age  duration-10                 (label 0)
      * score  = 1 - S(t+5)/S(t) straight from the Cox model (same formula as
                 tyre_life_projection.py)
      * threshold = Youden's J (argmax of TPR - FPR) on the ROC curve.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    cph = models["cph"]
    cols = list(cph.params_.index)                       # exact covariates this Cox model was fit with
    required = ["duration", "event", "compound", "track_temp_bucket", "fuel_load_estimate",
                "stint_number", "regulation_era", "circuit_degredation_ordinal"]
    events = stints_cox.dropna(subset=required)
    events = events[events["event"] == 1]
    report["cliff_roc_n_events"] = int(len(events))

    rows = []
    for s in events.to_dict("records"):
        base = {c: 0.0 for c in cols}
        for name, val in (("fuel_load_estimate", s["fuel_load_estimate"]),
                          ("stint_number", s["stint_number"]),
                          ("circuit_degredation_ordinal", s["circuit_degredation_ordinal"])):
            if name in base:
                base[name] = float(val)
        for dummy in (f"temp_{s['track_temp_bucket']}", f"regulation_era_{s['regulation_era']}"):
            if dummy in base:                            # reference level (e.g. 'cool') has no column
                base[dummy] = 1.0
        base["compound"] = s["compound"]
        duration = s["duration"]
        for offset in range(1, CLIFF_HORIZON + 1):
            age = duration - offset
            if age > 0:
                rows.append({**base, "_age": age, "_y": 1})
        age = duration - NEGATIVE_CHECKPOINT_BUFFER
        if age > 0:
            rows.append({**base, "_age": age, "_y": 0})

    labels, scores, n_fail, first_err = [], [], 0, None
    for r in rows:
        t0 = r["_age"]
        t1 = t0 + CLIFF_HORIZON
        row = pd.DataFrame([{k: v for k, v in r.items() if k not in ("_age", "_y")}])
        try:
            sf = cph.predict_survival_function(row, times=[t0, t1])
            s0, s1 = sf.iloc[0, 0], sf.iloc[1, 0]
            p = (1 - s1 / s0) if s0 > 0 else np.nan
        except Exception as e:                           # noqa: BLE001 - counted and reported, never hidden
            p, n_fail, first_err = np.nan, n_fail + 1, (first_err or e)
        if not (p is None or np.isnan(p)):
            labels.append(r["_y"])
            scores.append(p)

    labels, scores = np.array(labels), np.array(scores)
    report["cliff_roc_n_checkpoints"] = int(len(labels))
    report["cliff_roc_n_positive"] = int(labels.sum())
    report["cliff_roc_n_negative"] = int((labels == 0).sum())
    report["cliff_roc_n_predict_failures"] = int(n_fail)
    if first_err is not None:
        report.setdefault("warnings", []).append(
            f"{n_fail} ROC checkpoints failed to predict; first error: {type(first_err).__name__}: {first_err}")
    if labels.sum() < 20 or (labels == 0).sum() < 20:
        report.setdefault("warnings", []).append("too few ROC checkpoints for a stable threshold")
        return float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    j = int(np.argmax(tpr - fpr))
    report["cliff_roc_auc"] = float(roc_auc_score(labels, scores))
    report["cliff_roc_tpr_at_threshold"] = float(tpr[j])
    report["cliff_roc_fpr_at_threshold"] = float(fpr[j])
    return float(thr[j])


def run_original_part3(repo_root: Path, calib_path: Path | None, report: dict):
    """Runs YOUR ORIGINAL threshold-calibration.py Part 3 on today's cliff_detection_stints.csv
    (plot saving disabled so cliff_probability_roc.png is not overwritten). Purely a cross-check:
    if it prints the same number as threshold_cliff_probability on 'all_data', the re-implementation
    is verified even if that number differs from the 0.017 baked into gate-tier-3.py."""
    p = calib_path or (repo_root / "system" / "HERMES" / "trees" / "threshold-calibration.py")
    if not p.exists():
        report.setdefault("warnings", []).append(f"original calibration script not found at {p}; cross-check skipped")
        return None
    import os
    spec = importlib.util.spec_from_file_location("threshold_calibration_orig", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["threshold_calibration_orig"] = mod
    spec.loader.exec_module(mod)                          # main block is guarded
    mod.STINTS_CSV = str(repo_root / "cliff_detection_stints.csv")
    mod.TAXONOMY_PATH = str(repo_root / "src" / "taxanomy" / "circuit_taxonomy.xlsx")
    real_savefig = mod.plt.savefig
    mod.plt.savefig = lambda *a, **k: None
    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        print("\n----- ORIGINAL threshold-calibration.py Part 3, run on today's CSV (cross-check) -----")
        val = mod.calibrate_cliff_probability_threshold()
        print("----- end of original -----\n")
    finally:
        os.chdir(cwd)
        mod.plt.savefig = real_savefig
    return None if val is None else float(val)


# ----------------------------------------------------------------------------
# 5. SC prior (train years only)
# ----------------------------------------------------------------------------
def build_sc_prior(laps: pd.DataFrame, horizon: int = CLIFF_HORIZON) -> pd.DataFrame:
    """circuit, n_races, p_race_incidence, mean_total_laps, p_window_horizon  (blueprint 4.3)."""
    r = laps[laps["Session"] == "R"].copy()
    ts_sc = r["is_sc_lap"].fillna(False).astype(bool) | r["is_vsc_lap"].fillna(False).astype(bool)
    r["_sc"] = ts_sc
    per_race = r.groupby(["Race", "Season"]).agg(
        had_sc=("_sc", "any"), total_laps=("LapNumber", "max")).reset_index()
    g = per_race.groupby("Race").agg(
        n_races=("Season", "nunique"), p_race_incidence=("had_sc", "mean"),
        mean_total_laps=("total_laps", "mean")).reset_index().rename(columns={"Race": "circuit"})
    g = g[g["n_races"] >= MIN_CIRCUIT_RACES_FOR_SC_PRIOR].copy()
    g["p_window_horizon"] = g["p_race_incidence"] * (horizon / g["mean_total_laps"])
    return g


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def build_fold(repo_root: Path, fold: str, out_root: Path, check_reproduces: bool,
               calib_path: Path | None = None):
    y0, y1 = FOLD_TRAIN_YEARS[fold]
    fold_dir = out_root / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    report = {"fold": fold, "train_years": [y0, min(y1, 2026)], "warnings": []}
    print(f"\n=== {fold}: training on {y0}-{min(y1, 2026)} ===")

    mf = load_model_fit(repo_root)
    cache = LocalCache(repo_root / "gcs_cache")
    laps = load_laps_for_years(mf, cache, y0, y1)
    report["train_races"] = int(laps.groupby(["Season", "Race", "Session"]).ngroups)

    stints_all = pd.read_csv(repo_root / "cliff_detection_stints.csv")
    stints_train = stints_all[(stints_all["season"] >= y0) & (stints_all["season"] <= y1)].copy()
    report["train_stints"] = int(len(stints_train))
    if stints_train.empty:
        raise RuntimeError("no stints in training window")

    models, modeling_df, stints_cox = fit_models(mf, repo_root, laps, stints_train, report)

    thresholds = {
        "PACE_LOSS_THRESHOLD_SECONDS": threshold_pace_loss(modeling_df),
        "TYRE_AGE_TRIGGER_RATIO": threshold_tyre_age_ratio(stints_train),
        "CLIFF_PROBABILITY_THRESHOLD": threshold_cliff_probability(models, stints_cox, report),
    }
    report["thresholds"] = thresholds
    for k, v in thresholds.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            report["warnings"].append(f"{k} could not be computed on this fold; evaluate must NOT "
                                      f"silently reuse the all-data value")
    if check_reproduces:
        orig_val = run_original_part3(repo_root, calib_path, report)
        report["original_script_cliff_threshold_on_current_csv"] = orig_val
        if orig_val is not None and not np.isnan(thresholds["CLIFF_PROBABILITY_THRESHOLD"]):
            report["reimplementation_matches_original_script"] = bool(
                abs(orig_val - thresholds["CLIFF_PROBABILITY_THRESHOLD"]) < 1e-9)
        report["reproduction_vs_production"] = {
            k: {"refit": thresholds[k], "production": PRODUCTION_REFERENCE[k],
                "rel_diff": (abs(thresholds[k] - PRODUCTION_REFERENCE[k]) / PRODUCTION_REFERENCE[k])
                if not np.isnan(thresholds[k]) else None}
            for k in PRODUCTION_REFERENCE}

    # Fold-specific CSV artefacts
    stints_train.to_csv(fold_dir / "cliff_detection_stints.csv", index=False)
    build_sc_prior(laps).to_csv(fold_dir / "sc_vsc_circuit_level_prior.csv", index=False)
    pit_loss_path = repo_root / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv"
    if pit_loss_path.exists():
        pl = pd.read_csv(pit_loss_path)
        ycol = next((c for c in pl.columns if c.lower() in ("year", "season")), None)
        if ycol:
            pl = pl[(pl[ycol] >= y0) & (pl[ycol] <= y1)]
            report["pit_loss_filtered_by_year"] = True
        else:
            report["warnings"].append("archive_per_race_analysis.csv has no year/season column: "
                                      "per-circuit pit loss NOT filtered (small residual leak, state it)")
            report["pit_loss_filtered_by_year"] = False
        (fold_dir / "checkpoints" / "rival_knowledge").mkdir(parents=True, exist_ok=True)
        pl.to_csv(fold_dir / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv", index=False)

    # Which regulation eras this fold actually saw. master.py (fold mode) refuses to use these models
    # for a race in any OTHER era (MDP para 125: pre-2022 outputs must not be used for 2022+, and
    # the Cox model would otherwise silently treat an unseen era as the reference/old era).
    def _era(y):
        return "2018-2021" if y <= 2021 else ("2022-2025" if y <= 2025 else "2026+")
    models["train_eras"] = sorted({_era(int(y)) for y in laps["Season"].unique()})
    report["train_eras"] = models["train_eras"]
    with open(fold_dir / "tyre_life_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open(fold_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(fold_dir / "fit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("n_regression_models", "cox_concordance",
                                              "thresholds", "warnings") if k in report}, indent=2, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--fold", choices=list(FOLD_TRAIN_YEARS), default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--out", default="folds")
    ap.add_argument("--original-calibration", default=None,
                    help="path to your threshold-calibration.py (default: system/HERMES/trees/threshold-calibration.py)")
    ap.add_argument("--check-reproduces", action="store_true",
                    help="only meaningful with --fold all_data: compares refit thresholds to production")
    a = ap.parse_args()
    repo_root = Path(a.repo_root).resolve()
    out_root = repo_root / a.out
    if a.all_folds:
        for f in ["fold1_2021", "fold2_2022", "fold3_2023", "fold4_2024", "holdout_2025"]:
            if (out_root / f / "thresholds.json").exists():
                print(f"[skip] {f} already built")
                continue
            build_fold(repo_root, f, out_root, False)
    elif a.fold:
        build_fold(repo_root, a.fold, out_root, a.check_reproduces,
                   Path(a.original_calibration) if a.original_calibration else None)
    else:
        ap.error("give --fold NAME or --all-folds")


if __name__ == "__main__":
    main()