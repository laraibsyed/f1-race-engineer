"""
Regression V2 Ablation — driver_archetype covariates
========================================================
Tests whether tyre_management_pti / consistency_factor_pti (leakage-safe,
point-in-time driver profiles from rolling_driver_profile.py, joined via
join_profiles_to_laps.py) improve on Regression V2's existing covariates.

Four models, same structure, same data, same splits:
    V2 baseline          -- existing covariates only (unchanged from tyre-regression-v2.py)
    V2 + TM               -- baseline + tyre_management_pti
    V2 + Consistency       -- baseline + consistency_factor_pti
    V2 + Both              -- baseline + both

This answers a clean question: do driver characteristics, calculated only
from information available before the race, improve tyre-degradation
modelling beyond the existing covariates -- not whether the full taxonomy
is useful in general.

Design decisions carried over unchanged from tyre-regression-v2.py:
  - Target: pace_loss_seconds (not degradation_rate) -- avoids the tyre_age
    circularity flagged in V1.
  - WET/INTERMEDIATE excluded entirely.
  - Stratified by (Compound, Race, regulation_era).
  - Same structural cleaning (short-stint filter, red-flag+restart buffer,
    global MAD outlier filter) -- reused unmodified.
  - Same three split methods: random, fixed 2024 cutoff, expanding window.

New for this ablation:
  - Input is laps_features_with_driver_profiles.csv (the join output), not
    a fresh GCS pull -- this ablation is a downstream consumer of that file,
    not a re-run of the whole pipeline.
  - Rows with no driver profile (profile_source is null -- the 265 pairs/
    7,109 rows documented in rolling_driver_profiles_notes.md) are dropped
    ONLY for the three driver-covariate models, via dropna on the relevant
    column. The baseline model keeps every row baseline already had, so the
    four models are not silently evaluated on different baseline coverage
    just because a driver column is added -- baseline uses its own historical
    row count, exactly as tyre-regression-v2.py already did.
  - VIF re-run per model that adds a driver covariate, since a new predictor
    changes the multicollinearity picture -- not assumed clean because the
    baseline VIFs were.
  - Original Regression V2 results are not overwritten -- reported here as
    a fixed reference (RMSE by split method), copied from the project's own
    session notes, not recomputed, so the "keep the original as reference"
    requirement is met without re-running unrelated code.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

load_dotenv()

# ---------------------------------------------------------------------------
# Reference: original Regression V2 results (NOT recomputed here -- copied
# from tyre_modelling session notes, kept as the baseline to compare against)
# ---------------------------------------------------------------------------
ORIGINAL_V2_REFERENCE = {
    "2018-2021_rmse": 0.609,
    "2022-2025_rmse": 0.618,
    "2026+_rmse": 0.467,  # not yet trustworthy -- partial season, 13 groups
}

RBR_ALIASES = {
    "Red Bull Racing", "Red Bull Racing Honda", "Red Bull Racing RBPT",
    "Oracle Red Bull Racing", "Red Bull",
}

DRY_COMPOUNDS = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD"]
MIN_ROWS_PER_GROUP = 40


# ---------------------------------------------------------------------------
# 1. Load the join output directly -- no fresh GCS pull, this is a downstream
#    consumer of join_profiles_to_laps.py's output.
# ---------------------------------------------------------------------------
def load_joined_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"TrackStatus": str}, low_memory=False)
    if "is_rbr" not in df.columns:
        df["is_rbr"] = df["Team"].isin(RBR_ALIASES)
    if "Session" not in df.columns:
        # join_profiles_to_laps.py only reads .../R/laps_features.csv, so every
        # row here is already Race-session -- it never tagged Session because
        # it was always constant. Add it back since the structural filters
        # below (copied from tyre-regression-v2.py) group by it.
        df["Session"] = "R"
    return df


# ---------------------------------------------------------------------------
# 2. Structural filters -- IDENTICAL to tyre-regression-v2.py, reused not
#    reimplemented, since the underlying anomalous-lap causes don't change
#    based on which covariates are being tested.
# ---------------------------------------------------------------------------
MIN_STINT_LENGTH = 5
RED_FLAG_RESTART_BUFFER = 2
LAPTIME_OUTLIER_Z_THRESH = 4.0


def get_red_flag_affected_laps(df: pd.DataFrame, buffer: int = RED_FLAG_RESTART_BUFFER) -> set:
    red_flag_mask = df["TrackStatus"].astype(str).str.contains("5", na=False)
    red_flag_laps = df.loc[red_flag_mask, ["Season", "Race", "Session", "LapNumber"]].drop_duplicates()
    affected = set()
    for _, row in red_flag_laps.iterrows():
        for offset in range(buffer + 1):
            affected.add((row["Season"], row["Race"], row["Session"], row["LapNumber"] + offset))
    return affected


def filter_valid_laps(df: pd.DataFrame, min_stint_length: int = MIN_STINT_LENGTH) -> pd.DataFrame:
    red_flag_affected = get_red_flag_affected_laps(df)
    lap_keys = list(zip(df["Season"], df["Race"], df["Session"], df["LapNumber"]))
    is_red_flag_affected = pd.Series(lap_keys, index=df.index).isin(red_flag_affected)
    mask = (
        (~df["is_pit_in"].astype(bool)) & (~df["is_pit_out"].astype(bool))
        & (~df["is_out_lap"].astype(bool)) & (~df["is_in_lap"].astype(bool))
        & (~df["is_missing_laptime"].astype(bool)) & (~df["is_outlier_laptime"].astype(bool))
        & (~df["is_sc_lap"].astype(bool)) & (~df["is_vsc_lap"].astype(bool))
        & (~is_red_flag_affected)
    )
    clean = df[mask].dropna(subset=["tyre_age", "degradation_rate", "Compound"])
    stint_lengths = clean.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapNumber"].transform("count")
    return clean[stint_lengths >= min_stint_length]


def filter_global_degradation_outliers(df: pd.DataFrame, z_thresh: float = LAPTIME_OUTLIER_Z_THRESH) -> pd.DataFrame:
    grp = df.groupby(["Compound", "Race"])["degradation_rate"]
    med = grp.transform("median")
    mad = grp.transform(lambda x: (x - x.median()).abs().median())
    mad_safe = mad.replace(0, np.nan)
    robust_z = 0.6745 * (df["degradation_rate"] - med) / mad_safe
    is_outlier = robust_z.abs().gt(z_thresh).fillna(False)
    return df[~is_outlier]


def add_pace_loss_and_era(clean_laps: pd.DataFrame) -> pd.DataFrame:
    df = clean_laps[clean_laps["Compound"].isin(DRY_COMPOUNDS)].copy()
    df["LapTime_seconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    baseline = df.groupby(["Season", "Race", "Session", "Driver", "Stint"])["LapTime_seconds"].transform("min")
    df["pace_loss_seconds"] = df["LapTime_seconds"] - baseline
    df["regulation_era"] = pd.cut(
        df["Season"], bins=[-np.inf, 2021, 2025, np.inf],
        labels=["2018-2021", "2022-2025", "2026+"],
    )
    return df


# ---------------------------------------------------------------------------
# 3. Multicollinearity check -- re-run per model spec, not assumed clean
# ---------------------------------------------------------------------------
def check_multicollinearity(df: pd.DataFrame, driver_cols: list, label: str) -> None:
    base_cols = ["tyre_age", "fuel_load_estimate", "stint_number"] + driver_cols
    numeric_predictors = df[base_cols].dropna()
    temp_dummies = pd.get_dummies(df.loc[numeric_predictors.index, "track_temp_bucket"],
                                   prefix="temp", drop_first=True).astype(float)
    X = pd.concat([numeric_predictors, temp_dummies], axis=1)

    print(f"\n=== VIF check: {label} ===")
    for col in X.columns:
        others = X.drop(columns=[col])
        model = LinearRegression().fit(others, X[col])
        r2 = model.score(others, X[col])
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        flag = "  <-- concerning (VIF>5)" if vif > 5 else ""
        print(f"  {col:>28}: VIF = {vif:.2f}{flag}")


# ---------------------------------------------------------------------------
# 4. Model fitting -- same per-(Compound, Race, regulation_era) grouping as V2
# ---------------------------------------------------------------------------
def build_design_matrix(df: pd.DataFrame, temp_dummy_columns: list, driver_cols: list) -> pd.DataFrame:
    temp_dummies = pd.get_dummies(df["track_temp_bucket"], prefix="temp", drop_first=True)
    temp_dummies = temp_dummies.reindex(columns=temp_dummy_columns, fill_value=0).astype(float)
    cols = ["tyre_age", "fuel_load_estimate", "stint_number"] + driver_cols
    X = pd.concat([df[cols].reset_index(drop=True),
                   temp_dummies.reset_index(drop=True)], axis=1)
    return X


def fit_group_models(train_df: pd.DataFrame, temp_dummy_columns: list, driver_cols: list) -> dict:
    required = ["tyre_age", "fuel_load_estimate", "stint_number",
                "track_temp_bucket", "pace_loss_seconds"] + driver_cols
    models = {}
    for (compound, circuit, era), g in train_df.groupby(["Compound", "Race", "regulation_era"], observed=True):
        g = g.dropna(subset=required)
        if len(g) < MIN_ROWS_PER_GROUP:
            continue
        X = build_design_matrix(g, temp_dummy_columns, driver_cols)
        y = g["pace_loss_seconds"].reset_index(drop=True)
        models[(compound, circuit, era)] = LinearRegression().fit(X, y)
    return models


def evaluate(models: dict, test_df: pd.DataFrame, temp_dummy_columns: list,
             driver_cols: list, rbr_only: bool = True) -> pd.DataFrame:
    if rbr_only:
        test_df = test_df[test_df["is_rbr"]]

    required = ["tyre_age", "fuel_load_estimate", "stint_number",
                "track_temp_bucket", "pace_loss_seconds"] + driver_cols
    rows = []
    for (compound, circuit, era), model in models.items():
        g = test_df[(test_df["Compound"] == compound) & (test_df["Race"] == circuit)
                     & (test_df["regulation_era"] == era)].dropna(subset=required)
        if g.empty:
            continue
        X = build_design_matrix(g, temp_dummy_columns, driver_cols)
        preds = model.predict(X)
        rmse = root_mean_squared_error(g["pace_loss_seconds"], preds)
        rows.append({"compound": compound, "circuit": circuit, "era": era,
                      "n_test_rows": len(g), "rmse": rmse})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Three train/test splits, identical structure to V2, parameterised by
#    driver_cols so the same runner serves all four models
# ---------------------------------------------------------------------------
def run_random_split(df, temp_dummy_columns, driver_cols, test_frac=0.2, seed=42) -> pd.DataFrame:
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    models = fit_group_models(train_df, temp_dummy_columns, driver_cols)
    results = evaluate(models, test_df, temp_dummy_columns, driver_cols)
    results["split_method"] = "random"
    return results


def run_fixed_split(df, temp_dummy_columns, driver_cols, cutoff=2024) -> pd.DataFrame:
    train_df = df[df["Season"] < cutoff]
    test_df = df[df["Season"] >= cutoff]
    models = fit_group_models(train_df, temp_dummy_columns, driver_cols)
    results = evaluate(models, test_df, temp_dummy_columns, driver_cols)
    results["split_method"] = "fixed_2018_2023_train"
    return results


def run_expanding_window(df, temp_dummy_columns, driver_cols, last_complete_season=2025) -> pd.DataFrame:
    all_results = []
    for test_year in range(2021, last_complete_season + 1):
        train_df = df[df["Season"] < test_year]
        test_df = df[df["Season"] == test_year]
        models = fit_group_models(train_df, temp_dummy_columns, driver_cols)
        results = evaluate(models, test_df, temp_dummy_columns, driver_cols)
        results["split_method"] = f"expanding_window_test_{test_year}"
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


def run_all_splits(df: pd.DataFrame, temp_dummy_columns: list, driver_cols: list, label: str) -> pd.DataFrame:
    print(f"\n{'='*70}\nMODEL: {label}  (driver covariates: {driver_cols or 'none'})\n{'='*70}")

    if driver_cols:
        before = len(df)
        model_df = df.dropna(subset=driver_cols)
        after = len(model_df)
        print(f"[filter] {before - after} rows dropped for missing driver covariate "
              f"({after} rows remain) -- expected, matches the documented unmatched-join gap")
    else:
        model_df = df

    random_results = run_random_split(model_df, temp_dummy_columns, driver_cols)
    fixed_results = run_fixed_split(model_df, temp_dummy_columns, driver_cols)
    expanding_results = run_expanding_window(model_df, temp_dummy_columns, driver_cols)
    all_results = pd.concat([random_results, fixed_results, expanding_results], ignore_index=True)
    all_results["model"] = label

    summary = (
        all_results.groupby("split_method")["rmse"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_rmse", "median": "median_rmse",
                          "std": "std_rmse", "count": "n_groups_evaluated"})
        .sort_values("mean_rmse")
    )
    print(f"\n--- {label}: RMSE (seconds) by split method ---")
    print(summary.to_string())

    return all_results


# ---------------------------------------------------------------------------
# 6b. Fair, paired comparison on an IDENTICAL evaluation population
# ---------------------------------------------------------------------------
def restrict_to_common_rows(df: pd.DataFrame, driver_cols_all: list) -> pd.DataFrame:
    """
    The baseline model sees every row; each driver-covariate model drops its
    own missing rows. That makes the raw RMSE comparison from run_all_splits
    unfair -- different models are scored on different data. This rebuilds
    ALL FOUR models (including baseline) on exactly the same row set: every
    row that has a value for every driver covariate used anywhere in this
    ablation. This is a stricter population than any individual model
    needed, by design, so the comparison is apples-to-apples across all four.
    """
    all_driver_cols = sorted({c for cols in driver_cols_all for c in cols})
    before = len(df)
    common_df = df.dropna(subset=all_driver_cols)
    after = len(common_df)
    print(f"\n[common population] {before - after} rows dropped so ALL FOUR models share the "
          f"same evaluation rows ({after} rows remain, {after/before:.1%} of the input to this step)")
    return common_df


def fair_compare_all_models(common_df: pd.DataFrame, temp_dummy_columns: list,
                             model_specs: list) -> tuple:
    """
    Refits and evaluates every model on the SAME common_df, using the SAME
    split logic per split method (identical train/test row membership across
    models within a split, since common_df itself is now shared). Returns:
      - long-format results (one row per model x group x split)
      - a wide, per-group PAIRED table: one row per (split_method, compound,
        circuit, era, n_test_rows), one rmse column per model, plus a delta
        column for each non-baseline model. Built only from groups that
        EVERY model actually produced an RMSE for in that split -- a group a
        model skipped (e.g. dropped below MIN_ROWS_PER_GROUP after this
        stricter common-population filter) is excluded from ALL models for
        that (split, group), not just the one that dropped it, so the paired
        comparison stays genuinely paired.
    """
    per_model_results = {}
    for label, driver_cols in model_specs:
        random_results = run_random_split(common_df, temp_dummy_columns, driver_cols)
        fixed_results = run_fixed_split(common_df, temp_dummy_columns, driver_cols)
        expanding_results = run_expanding_window(common_df, temp_dummy_columns, driver_cols)
        results = pd.concat([random_results, fixed_results, expanding_results], ignore_index=True)
        results["model"] = label
        per_model_results[label] = results

    key_cols = ["split_method", "compound", "circuit", "era"]

    # Inner-join every model's group-level results together on the group key,
    # so only groups where ALL models produced a result survive -- this is
    # what makes the comparison genuinely paired rather than just "same input
    # rows, possibly different surviving groups per model".
    merged = None
    for label, _ in model_specs:
        renamed = per_model_results[label][key_cols + ["rmse", "n_test_rows"]].rename(
            columns={"rmse": f"rmse_{label}", "n_test_rows": f"n_{label}"})
        merged = renamed if merged is None else merged.merge(renamed, on=key_cols, how="inner")

    before_pairing = {label: len(per_model_results[label]) for label, _ in model_specs}
    print(f"\n[paired groups] group counts before pairing (per model): {before_pairing}")
    print(f"[paired groups] groups surviving in ALL FOUR models: {len(merged)}")

    baseline_label = model_specs[0][0]
    for label, _ in model_specs[1:]:
        merged[f"delta_{label}"] = merged[f"rmse_{label}"] - merged[f"rmse_{baseline_label}"]

    long_results = pd.concat(per_model_results.values(), ignore_index=True)
    return long_results, merged


def paired_group_bootstrap(paired_df: pd.DataFrame, model_specs: list, n_boot: int = 5000,
                            seed: int = 42) -> pd.DataFrame:
    """
    Group-level bootstrap on the paired table: resample GROUPS (not rows)
    with replacement, since the unit of comparison here is one RMSE per
    (split_method, compound, circuit, era) group, not one lap. This respects
    the same non-independence the original evaluate() function already
    implies (RMSE is computed per group, not per row) -- bootstrapping rows
    instead would badly understate uncertainty by ignoring within-group
    correlation entirely.

    For each non-baseline model, reports the bootstrap distribution of the
    mean paired delta (that model's RMSE minus baseline's RMSE, averaged
    across resampled groups), a 95% percentile CI, and the fraction of
    bootstrap resamples where the delta is negative (i.e. that model beats
    baseline) -- a simple, honest one-number readout of how consistently the
    improvement holds up under resampling.
    """
    rng = np.random.default_rng(seed)
    n_groups = len(paired_df)
    baseline_label = model_specs[0][0]

    rows = []
    for label, _ in model_specs[1:]:
        deltas = paired_df[f"delta_{label}"].to_numpy()
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n_groups, size=n_groups)
            boot_means[b] = deltas[idx].mean()

        observed_mean = deltas.mean()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        frac_improves = (boot_means < 0).mean()

        rows.append({
            "model": label,
            "observed_mean_delta": observed_mean,
            "boot_mean_delta": boot_means.mean(),
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "frac_bootstrap_resamples_improving": frac_improves,
            "crosses_zero": (ci_lo < 0 < ci_hi),
        })

    result = pd.DataFrame(rows).set_index("model")
    return result


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="laps_features_with_driver_profiles.csv")
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    print(f"[load] reading {args.input} ...")
    raw = load_joined_data(args.input)
    print(f"[load] {len(raw)} rows, {raw['is_rbr'].sum()} RBR")

    clean = filter_valid_laps(raw)
    clean = filter_global_degradation_outliers(clean)
    print(f"[filter] {len(clean)} rows after structural + outlier cleaning")

    modeling_df = add_pace_loss_and_era(clean)
    print(f"[filter] {len(modeling_df)} rows after excluding WET/INTERMEDIATE")

    # Consistent temp dummy columns across every model/fold, built once
    temp_dummy_columns = sorted(pd.get_dummies(
        modeling_df["track_temp_bucket"], prefix="temp", drop_first=True).columns.tolist())
    print(f"[info] temp dummy columns: {temp_dummy_columns}")

    MODEL_SPECS = [
        ("V2_baseline", []),
        ("V2_plus_TM", ["tyre_management_pti"]),
        ("V2_plus_Consistency", ["consistency_factor_pti"]),
        ("V2_plus_Both", ["tyre_management_pti", "consistency_factor_pti"]),
    ]

    all_model_results = []
    for label, driver_cols in MODEL_SPECS:
        if driver_cols:
            check_multicollinearity(modeling_df, driver_cols, label)
        results = run_all_splits(modeling_df, temp_dummy_columns, driver_cols, label)
        all_model_results.append(results)

    combined = pd.concat(all_model_results, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"\n[save] {args.output}")

    # --- Head-to-head summary: does adding either/both covariate help? ---
    print(f"\n{'='*70}\nHEAD-TO-HEAD: mean RMSE by model, across all split methods\n{'='*70}")
    comparison = combined.groupby("model")["rmse"].agg(["mean", "median", "count"]).round(4)
    comparison = comparison.reindex(["V2_baseline", "V2_plus_TM", "V2_plus_Consistency", "V2_plus_Both"])
    print(comparison.to_string())

    baseline_mean = comparison.loc["V2_baseline", "mean"]
    print(f"\n[reference] V2_baseline mean RMSE: {baseline_mean:.4f}s")
    for label in ["V2_plus_TM", "V2_plus_Consistency", "V2_plus_Both"]:
        delta = comparison.loc[label, "mean"] - baseline_mean
        direction = "IMPROVES" if delta < 0 else "WORSENS"
        print(f"[compare] {label}: {delta:+.4f}s vs baseline ({direction} on this metric alone -- "
              f"check per-group counts too, since RMSE means aren't directly comparable across "
              f"models with different row/group coverage from the driver-covariate dropna)")

    print(f"\n[reference] Original Regression V2 (no driver covariates, static baseline): "
          f"{ORIGINAL_V2_REFERENCE}")
    print("[note] Original V2 numbers are NOT recomputed here -- copied from the project's own "
          "session notes as a fixed point of comparison, per the requirement to keep the original "
          "result as reference rather than overwrite it.")

    # -----------------------------------------------------------------
    # FAIR, PAIRED COMPARISON on an identical evaluation population
    # -----------------------------------------------------------------
    print(f"\n\n{'#'*70}\nFAIR COMPARISON: all four models on an IDENTICAL row/group population\n{'#'*70}")

    all_driver_cols = [cols for _, cols in MODEL_SPECS]
    common_df = restrict_to_common_rows(modeling_df, all_driver_cols)

    long_results, paired = fair_compare_all_models(common_df, temp_dummy_columns, MODEL_SPECS)
    paired.to_csv("ablation_paired_results.csv", index=False)
    print(f"[save] ablation_paired_results.csv -- {len(paired)} paired groups")

    print(f"\n--- Paired mean RMSE per model (identical groups, n={len(paired)}) ---")
    for label, _ in MODEL_SPECS:
        print(f"  {label:>22}: mean RMSE = {paired[f'rmse_{label}'].mean():.4f}s")

    baseline_label = MODEL_SPECS[0][0]
    print(f"\n--- Paired mean delta vs {baseline_label} (negative = improvement) ---")
    for label, _ in MODEL_SPECS[1:]:
        delta_col = paired[f"delta_{label}"]
        n_improve = (delta_col < 0).sum()
        print(f"  {label:>22}: mean delta = {delta_col.mean():+.4f}s, "
              f"median = {delta_col.median():+.4f}s, "
              f"improves in {n_improve}/{len(paired)} groups ({n_improve/len(paired):.1%})")

    print(f"\n--- Group-level bootstrap (5000 resamples, resampling GROUPS not rows) ---")
    boot_results = paired_group_bootstrap(paired, MODEL_SPECS)
    print(boot_results.to_string())
    print("\n[interpretation] crosses_zero=True means the 95% CI on the mean paired delta "
          "includes zero -- i.e. this ablation cannot rule out 'no real difference from "
          "baseline' for that model at conventional confidence. frac_bootstrap_resamples_improving "
          "close to 0.5 means the direction of the effect is essentially a coin flip under "
          "resampling, regardless of what the single observed mean delta says.")