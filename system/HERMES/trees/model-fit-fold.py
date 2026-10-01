
""
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

FOLD_TRAIN_YEARS = {
    "fold1_2021": (2018, 2020),
    "fold2_2022": (2018, 2021),
    "fold3_2023": (2018, 2022),
    "fold4_2024": (2018, 2023),
    "holdout_2025": (2018, 2024),
    "all_data": (2018, 2100),
}
CLIFF_HORIZON = 5
NEGATIVE_CHECKPOINT_BUFFER = 10
PRODUCTION_REFERENCE = {
    "CLIFF_PROBABILITY_THRESHOLD": 0.017,
    "PACE_LOSS_THRESHOLD_SECONDS": 1.805,
    "TYRE_AGE_TRIGGER_RATIO": 0.421,
}
MIN_CIRCUIT_RACES_FOR_SC_PRIOR = 4

def load_model_fit(repo_root: Path):
    p = repo_root / "system" / "HERMES" / "trees" / "model-fit.py"
    if not p.exists():
        raise FileNotFoundError(f"model-fit.py not found at {p}")
    spec = importlib.util.spec_from_file_location("model_fit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class LocalCache:
    ""

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
    ""
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

def threshold_pace_loss(modeling_df: pd.DataFrame) -> float:
    ""
    v = modeling_df["pace_loss_seconds"].dropna()
    return float(v.quantile(0.90))

def threshold_tyre_age_ratio(stints_train: pd.DataFrame) -> float:
    ""
    s = stints_train
    if "event" in s.columns:
        s = s[s["event"] == 1]
    s = s.dropna(subset=["cliff_tyre_age", "n_laps_true"])
    s = s[s["n_laps_true"] > 0]
    if s.empty:
        return float("nan")
    return float((s["cliff_tyre_age"] / s["n_laps_true"]).quantile(0.10))

def threshold_cliff_probability(models: dict, stints_cox: pd.DataFrame, report: dict):
    ""
    from sklearn.metrics import roc_curve, roc_auc_score
    cph = models["cph"]
    cols = list(cph.params_.index)
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
            if dummy in base:
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
        except Exception as e:
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
    ""
    p = calib_path or (repo_root / "system" / "HERMES" / "trees" / "threshold-calibration.py")
    if not p.exists():
        report.setdefault("warnings", []).append(f"original calibration script not found at {p}; cross-check skipped")
        return None
    import os
    spec = importlib.util.spec_from_file_location("threshold_calibration_orig", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["threshold_calibration_orig"] = mod
    spec.loader.exec_module(mod)
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

def build_sc_prior(laps: pd.DataFrame, horizon: int = CLIFF_HORIZON) -> pd.DataFrame:
    ""
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
