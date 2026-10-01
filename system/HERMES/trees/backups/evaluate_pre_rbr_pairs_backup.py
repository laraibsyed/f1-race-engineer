#!/usr/bin/env python3
"""
HERMES full-system evaluation (Red Bull, VER + teammate)
============================================================================
Run from the repo root (same place you run master.py from):

    python hermes_evaluate.py --repo-root . --stage all

Stages (each is resumable, results are cached in eval_out/):
    replay     run master.py replay for every Red Bull race 2021-2025 -> eval_out/replays/*.jsonl
    baseline   tyre-life-only baseline (Tier 3 tyre triggers only) built from the same replays
    evaluate   walk-forward folds + 2025 holdout + counterfactual + metrics -> eval_out/*.csv
    report     markdown summary + figures -> eval_out/evaluation_report.md
    all        everything above, in order

DESIGN NOTES (read these, they matter for the write-up)
----------------------------------------------------------------------------
1. HERMES has no learned parameters per fold in the strict sense (the tyre
   pickle was fit on 100% of data, see blueprint 4.1). So "walk-forward" here
   is done honestly in two parts:
     (a) FOLD-WISE PERFORMANCE: performance reported per test year so you can
         see stability across time (2021, 2022, 2023, 2024) plus the 2025 holdout.
     (b) LEAKAGE FLAG: because production models were fit on ALL data, the
         script marks every fold as LEAKY unless you pass --fold-models with
         a pickle refit on train years only. This is stated in the report, not
         hidden. If you can refit per fold with model-fit.py, do it, it is the
         single biggest thing an examiner will ask about.
2. "Decision accuracy" needs a ground-truth definition. A team's real pit is
   NOT automatically the right answer (bad strategies happen). So we report
   THREE separate things:
       - pit_lap_agreement     : HERMES PIT_NOW within +-K laps of a real pit
       - false_alarm_rate      : HERMES PIT_NOW windows with no real pit nearby
       - miss_rate             : real pits with no HERMES PIT_NOW within +-K
   and the counterfactual (below) is what judges whether disagreement was good.
3. COUNTERFACTUAL is a SIMPLIFIED, ASSUMPTION-HEAVY estimate (flagged). It uses
   per-lap pace loss and the empirical pit-loss constant to estimate the time
   delta of pitting on HERMES's lap instead of the real lap. It is NOT a full
   race simulation and cannot model traffic or rival reactions. Say so.
4. Position gain/loss is the REAL final-position change vs grid, reported as
   context, plus an ESTIMATED time delta from the counterfactual. We do not
   claim HERMES would have gained N positions.
5. Nothing here fabricates a number: missing data -> NaN, and counts of
   skipped races/drivers are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Walk-forward design agreed in project chats:
#   Train 2018-2020 -> test 2021 ; 2018-2021 -> 2022 ; 2018-2022 -> 2023 ; 2018-2023 -> 2024
#   2025 = untouched final holdout ; 2026 = separate regulation stress test (excluded here)
FOLDS = {
    "fold1_2021": {"train": (2018, 2020), "test": 2021},
    "fold2_2022": {"train": (2018, 2021), "test": 2022},
    "fold3_2023": {"train": (2018, 2022), "test": 2023},
    "fold4_2024": {"train": (2018, 2023), "test": 2024},
}
HOLDOUT_YEAR = 2025

# Red Bull driver pairs per season (D1, D2). ASSUMPTION: verify against your data.
# The script skips a race automatically if a driver code is absent from laps_features.
RBR_PAIRS = {
    2021: ("VER", "PER"),
    2022: ("VER", "PER"),
    2023: ("VER", "PER"),
    2024: ("VER", "PER"),
    # 2025: PER was replaced by LAW (rounds 1-2) then TSU. Handled per race below.
    2025: ("VER", "TSU"),
}
# 2025 exceptions (ASSUMPTION, verify): LAW drove the first two rounds, TSU after.
RBR_2025_EARLY_LAW_RACES = {"Australian_Grand_Prix", "Chinese_Grand_Prix"}

EMPTY_REPLAYS: list = []          # races that produced no usable decisions (reported, never hidden)
# Which fold's artefacts (built by fit_fold.py) are used to replay each test season.
# Only used in --fold-mode. A season with no entry is REFUSED, never run with all-data models.
SEASON_TO_FOLD = {2021: "fold1_2021", 2022: "fold2_2022", 2023: "fold3_2023",
                  2024: "fold4_2024", 2025: "holdout_2025"}
PIT_MATCH_TOLERANCE_LAPS = 2      # +-K laps for "agreement" (also swept below)
TOLERANCE_SWEEP = [0, 1, 2, 3, 5]
DEFAULT_PIT_LOSS_S = 22.0         # ASSUMPTION, used only if no empirical value found

# Decision strings HERMES emits that count as "pit now"
PIT_DECISIONS = {"PIT_NOW", "PIT_FLEXIBLE"}
# Tyre-only Tier 3 triggers that make up the BASELINE ("tyre life only")
BASELINE_TRIGGERS = ["cliff_proximity", "pace_lap_delta", "tyre_age"]
# Baseline variants (all tyre-life-only, no rival/SC/radio/weather logic):
#   baseline_strict : 3 of 3 tyre triggers  (very conservative, almost never fires - kept for transparency)
#   baseline_2of3   : 2 of 3 tyre triggers  (FAIR main baseline)
#   baseline_any    : 1 of 3 tyre triggers  (most permissive)
BASELINE_VARIANTS = {"baseline_strict": 3, "baseline_2of3": 2, "baseline_any": 1}
MAIN_BASELINE = "baseline_2of3"   # the one used for the headline full-vs-baseline comparison


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def rbr_drivers_for(season: int, race: str):
    if season == 2025 and race in RBR_2025_EARLY_LAW_RACES:
        return ("VER", "LAW")
    return RBR_PAIRS[season]


def list_races(repo_root: Path, season: int) -> list[str]:
    base = repo_root / "gcs_cache" / "clean" / "features" / str(season)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "R" / "laps_features.csv").exists())


def load_real_laps(repo_root: Path, season: int, race: str) -> pd.DataFrame | None:
    p = repo_root / "gcs_cache" / "clean" / "features" / str(season) / race / "R" / "laps_features.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"TrackStatus": str}, low_memory=False)


def real_pit_laps(laps: pd.DataFrame, driver: str) -> list[int]:
    d = laps[laps["Driver"] == driver]
    if d.empty or "is_pit_in" not in d.columns:
        return []
    m = d["is_pit_in"].fillna(False).astype(bool)
    return sorted(int(x) for x in d.loc[m, "LapNumber"].dropna().unique())


def final_and_grid_position(laps: pd.DataFrame, driver: str):
    """Last classified lap Position as final; first lap Position as grid proxy.
    Returns (grid, final) or (nan, nan). Position can be null in some races (blueprint B14)."""
    d = laps[laps["Driver"] == driver].sort_values("LapNumber")
    if d.empty or "Position" not in d.columns or d["Position"].notna().sum() < 2:
        return (np.nan, np.nan)
    pos = d["Position"].dropna()
    return (float(pos.iloc[0]), float(pos.iloc[-1]))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if path.stat().st_size == 0:
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _get(d, *keys, default=None):
    """Safe nested get, since row schema was not visible to this script."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def first_present(row: dict, candidates: list[str]):
    for c in candidates:
        if c in row and row[c] is not None:
            return row[c]
    return None


# ----------------------------------------------------------------------------
# STAGE 1: REPLAY (shells out to your own master.py, no internals guessed)
# ----------------------------------------------------------------------------
def stage_replay(repo_root: Path, out_dir: Path, master_path: Path, seasons: list[int], force: bool,
                 fold_mode: bool = False, folds_dir: Path | None = None,
                 allow_cross_era: bool = False):
    rep_dir = out_dir / "replays"
    rep_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for season in seasons:
        for race in list_races(repo_root, season):
            d1, d2 = rbr_drivers_for(season, race)
            laps = load_real_laps(repo_root, season, race)
            if laps is None or not {d1, d2}.issubset(set(laps["Driver"].unique())):
                log_rows.append(dict(season=season, race=race, status="SKIP_driver_not_in_race"))
                continue
            out_file = rep_dir / f"{season}_{race}.jsonl"
            if out_file.exists() and not force:
                log_rows.append(dict(season=season, race=race, status="CACHED"))
                continue
            env = os.environ.copy()
            if allow_cross_era:
                env["HERMES_ALLOW_CROSS_ERA"] = "1"       # master_fold_aware.py: cross-era TRANSFER variant
            else:
                env.pop("HERMES_ALLOW_CROSS_ERA", None)
            if fold_mode:
                fold_name = SEASON_TO_FOLD.get(season)
                fold_path = (folds_dir / fold_name) if (folds_dir and fold_name) else None
                if fold_path is None or not (fold_path / "thresholds.json").exists():
                    log_rows.append(dict(season=season, race=race, status="SKIP_no_fold_artefacts",
                                         err=f"missing {fold_path}. Run fit_fold.py first."))
                    print(f"   SKIP {season} {race}: no fold artefacts at {fold_path}")
                    continue
                env["HERMES_FOLD_DIR"] = str(fold_path)      # master_patched.py reads this
            else:
                env.pop("HERMES_FOLD_DIR", None)            # development run must NOT pick up a stale fold
            cmd = [sys.executable, str(master_path), "replay", "--repo-root", str(repo_root),
                   "--season", str(season), "--race", race, "--session", "R",
                   "--d1", d1, "--d2", d2, "--out", str(out_file)]
            print(f"[replay] {season} {race} {d1}/{d2}")
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), env=env)
            if res.returncode != 0 or not out_file.exists():
                log_rows.append(dict(season=season, race=race, status="FAIL",
                                     err=(res.stderr or "")[-300:]))
                print(f"   FAILED: {(res.stderr or '')[-200:]}")
            else:
                log_rows.append(dict(season=season, race=race, status="OK"))
    pd.DataFrame(log_rows).to_csv(out_dir / "replay_log.csv", index=False)
    print(f"[replay] log -> {out_dir / 'replay_log.csv'}")


# ----------------------------------------------------------------------------
# DECISION EXTRACTION
# ----------------------------------------------------------------------------
def hermes_pit_laps(dec: pd.DataFrame, driver: str, mode: str = "full") -> list[int]:
    """
    Laps where HERMES says pit for `driver`.
      mode='full'     : full system. Uses execution decision if present, else gate decision.
      mode='baseline' : tyre-life-only baseline = all three tyre triggers true.
    Column names are resolved defensively because the row schema is spec'd in the blueprint
    (section 6) but the exact JSON keys were not visible when this script was written.
    """
    if dec.empty:
        return []
    dcol = next((c for c in ("driver", "Driver") if c in dec.columns), None)
    lcol = next((c for c in ("lap", "LapNumber", "lap_number") if c in dec.columns), None)
    if dcol is None or lcol is None:
        raise KeyError(f"Cannot find driver/lap columns. Got: {list(dec.columns)}")
    sub = dec[dec[dcol] == driver].copy()
    laps = []
    for _, r in sub.iterrows():
        row = r.to_dict()
        if mode == "full":
            exec_dec = _get(row, "execution", "decision") if isinstance(row.get("execution"), dict) else row.get("execution_decision")
            gate_dec = first_present(row, ["gate_decision", "decision"])
            # Execution PIT_NOW is the final call; if execution absent fall back to gate.
            final = exec_dec if exec_dec is not None else gate_dec
            is_pit = final in PIT_DECISIONS
        else:
            trig = row.get("triggers") or {}
            n_on = sum(bool(trig.get(t, False)) for t in BASELINE_TRIGGERS)
            is_pit = n_on >= BASELINE_VARIANTS[mode]
        if is_pit:
            laps.append(int(row[lcol]))
    return sorted(set(laps))


def collapse_to_events(laps: list[int], gap: int = 2) -> list[int]:
    """A signal that stays 'on' for 8 straight laps is ONE recommendation, not 8.
    (Blueprint/chat finding: VER weather window was open 8 laps.) Collapse runs to their FIRST lap."""
    if not laps:
        return []
    events = [laps[0]]
    for prev, cur in zip(laps, laps[1:]):
        if cur - prev > gap:
            events.append(cur)
    return events


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
def match_events(pred: list[int], real: list[int], tol: int):
    """Greedy one-to-one matching of predicted events to real pits within +-tol laps.
    Returns (matched_pairs, unmatched_pred, unmatched_real)."""
    real_left = list(real)
    matched, unmatched_pred = [], []
    for p in pred:
        best, bd = None, None
        for r in real_left:
            d = abs(p - r)
            if d <= tol and (bd is None or d < bd):
                best, bd = r, d
        if best is None:
            unmatched_pred.append(p)
        else:
            matched.append((p, best))
            real_left.remove(best)
    return matched, unmatched_pred, real_left


def counterfactual_time_delta(laps_df: pd.DataFrame, driver: str, hermes_lap: int, real_lap: int,
                              pit_loss_s: float) -> float:
    """
    ESTIMATED seconds gained (+) / lost (-) by pitting on hermes_lap instead of real_lap.
    ASSUMPTIONS (flagged, this is a first-order estimate, not a race simulation):
      - Pit-loss is identical on either lap (so it cancels) -> only tyre pace difference matters.
      - Pitting EARLY (hermes_lap < real_lap): we avoid the worn-tyre laps in between but start the
        fresh stint earlier. Gain = sum of (worn lap time - fresh-tyre reference) over the gap laps.
      - Pitting LATE (hermes_lap > real_lap): the reverse, we extend the worn stint. Loss = same sum.
      - Traffic, undercut reactions and SC timing are NOT modelled.
    Fresh-tyre reference = median of the driver's first 3 clean laps of the NEXT stint.
    Returns NaN if it cannot be computed (never a guess).
    """
    if hermes_lap == real_lap:
        return 0.0   # identical timing = zero estimated difference (a real result, not "unknown")
    d = laps_df[laps_df["Driver"] == driver].copy()
    if d.empty or "LapTime" not in d.columns:
        return np.nan
    d["lt"] = pd.to_timedelta(d["LapTime"], errors="coerce").dt.total_seconds()
    clean = ~(d.get("is_pit_in", False).fillna(False).astype(bool)
              | d.get("is_pit_out", False).fillna(False).astype(bool)
              | d.get("is_sc_lap", False).fillna(False).astype(bool)
              | d.get("is_vsc_lap", False).fillna(False).astype(bool)
              | d.get("is_outlier_laptime", False).fillna(False).astype(bool))
    d = d[clean & d["lt"].notna()]
    lo, hi = sorted((hermes_lap, real_lap))
    gap = d[(d["LapNumber"] > lo) & (d["LapNumber"] <= hi)]
    after = d[d["LapNumber"] > real_lap].sort_values("LapNumber").head(3)
    if gap.empty or after.empty:
        return np.nan
    fresh_ref = after["lt"].median()
    delta = float((gap["lt"] - fresh_ref).clip(lower=0).sum())
    return delta if hermes_lap < real_lap else -delta


# ----------------------------------------------------------------------------
# STAGE 3: EVALUATE
# ----------------------------------------------------------------------------
def evaluate_one_race(repo_root: Path, out_dir: Path, season: int, race: str,
                      pit_loss_lookup: dict, tol: int):
    jf = out_dir / "replays" / f"{season}_{race}.jsonl"
    if not jf.exists():
        return None, []
    dec = read_jsonl(jf)
    laps = load_real_laps(repo_root, season, race)
    if laps is None or dec.empty:
        EMPTY_REPLAYS.append(dict(season=season, race=race,
                                  reason="empty_replay" if dec.empty else "no_laps_features"))
        return None, []
    d1, d2 = rbr_drivers_for(season, race)
    pit_loss = pit_loss_lookup.get(race, DEFAULT_PIT_LOSS_S)
    rows, cf_rows = [], []
    for drv in (d1, d2):
        real = real_pit_laps(laps, drv)
        grid, final = final_and_grid_position(laps, drv)
        for system in ("full", *BASELINE_VARIANTS):
            raw = hermes_pit_laps(dec, drv, mode=system)
            ev = collapse_to_events(raw)
            m, fa, miss = match_events(ev, real, tol)
            rows.append(dict(
                season=season, race=race, driver=drv, system=system, tol=tol,
                n_real_pits=len(real), n_events=len(ev), n_matched=len(m),
                n_false_alarm=len(fa), n_missed=len(miss),
                mean_abs_lap_error=(np.mean([abs(p - r) for p, r in m]) if m else np.nan),
                mean_signed_lap_error=(np.mean([p - r for p, r in m]) if m else np.nan),
                grid_pos=grid, final_pos=final,
                pos_change=(grid - final) if not (np.isnan(grid) or np.isnan(final)) else np.nan,
                pit_loss_used_s=pit_loss,
                event_laps=json.dumps([int(x) for x in ev]),
                real_laps=json.dumps([int(x) for x in real]),
                total_laps=int(laps["LapNumber"].max()),
            ))
            if system == "full":
                for p, r in m:
                    cf_rows.append(dict(season=season, race=race, driver=drv,
                                        hermes_lap=p, real_lap=r, lap_diff=p - r,
                                        est_time_delta_s=counterfactual_time_delta(laps, drv, p, r, pit_loss)))
    return pd.DataFrame(rows), cf_rows


def load_pit_loss_lookup(repo_root: Path) -> dict:
    """Empirical per-circuit pit-loss from archive_per_race_analysis.csv (blueprint 2.2). Best effort."""
    p = repo_root / "checkpoints" / "rival_knowledge" / "archive_per_race_analysis.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    race_col = next((c for c in df.columns if c.lower() in ("race", "circuit")), None)
    loss_col = next((c for c in df.columns if "pit_loss" in c.lower()), None)
    if not race_col or not loss_col:
        return {}
    g = df.groupby(race_col)[loss_col].median()
    return {str(k): float(v) for k, v in g.items() if 8 <= v <= 35}   # plausible band per blueprint 4.5



def compute_coverage(out_dir: Path) -> pd.DataFrame:
    """How often was each mechanism actually AVAILABLE, per split? A low score in a fold is only
    interpretable next to this: e.g. fold 2 (test 2022) has no 2022-2025 tyre models (para 125),
    and fold 1 (test 2021) has an empty SC prior (<4 races per circuit in 2018-2020)."""
    rows = []
    for f in sorted((out_dir / "replays").glob("*.jsonl")):
        season = int(f.name.split("_")[0])
        dec = read_jsonl(f)
        if dec.empty:
            continue
        split = "holdout_2025" if season == HOLDOUT_YEAR else f"test_{season}"
        proj = dec["projection"] if "projection" in dec.columns else pd.Series([{}] * len(dec))
        scg = dec["sc_gamble"] if "sc_gamble" in dec.columns else pd.Series([{}] * len(dec))
        get = lambda d, k: (d or {}).get(k) if isinstance(d, dict) else None
        rows.append(dict(
            split=split, race=f.stem, n=len(dec),
            pace_loss_none=sum(get(p, "predicted_pace_loss") is None for p in proj),
            cliff_none=sum(get(p, "cliff_probability_next_5_laps") is None for p in proj),
            # sc_gamble is None when Tier 1/2 decided first, so the gamble was never invoked
            sc_not_reached=sum(g is None for g in scg),
            # invoked but returned INSUFFICIENT_DATA (no SC prior, or no pace-loss/cliff input)
            sc_insufficient=sum(get(g, "recommendation") == "INSUFFICIENT_DATA" for g in scg)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby("split")[["n", "pace_loss_none", "cliff_none", "sc_not_reached", "sc_insufficient"]].sum().reset_index()
    for c in ("pace_loss_none", "cliff_none", "sc_not_reached", "sc_insufficient"):
        g[f"pct_{c}"] = (100 * g[c] / g["n"]).round(1)
    return g[["split", "n", "pct_pace_loss_none", "pct_cliff_none", "pct_sc_not_reached", "pct_sc_insufficient"]]


def bootstrap_ci(values, n=5000, seed=0):
    v = np.asarray([x for x in values if not pd.isna(x)], dtype=float)
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))



# ----------------------------------------------------------------------------
# NO-SKILL CONTROLS
# ----------------------------------------------------------------------------
# Why: F1 at +-2 laps has a NON-ZERO floor. Pit stops cluster mid-race, so a system with no real
# skill still "agrees" with some real stops by luck. Without a null, "F1 = 0.27" is uninterpretable.
#   Null A  UNIFORM TIMING   HERMES's events (same COUNT) placed at random laps of the race; real pits as they were.
#                            Weakest null (real stops cluster mid-race, uniform does not).
#   Null B  SAME CIRCUIT,    HERMES's events for this race scored against the real pits of the SAME circuit in a
#           OTHER SEASON     DIFFERENT season. Keeps circuit-typical timing (e.g. Monaco vs Bahrain), so beating it
#                            means HERMES carries race-specific information beyond "stops at this track usually
#                            happen around lap N". THIS is the demanding control.
#   Ref     COPY OTHER YEAR  F1 of predicting this race's stops by copying another season's real stops at the same
#                            circuit. A naive-strategist reference. Uses future seasons, so it is a REFERENCE, not a
#                            legitimate forecasting competitor.
# p = (1 + #null replicates with F1 >= actual) / (1 + replicates); one-sided.
CONTROL_SYSTEMS = ["full", "baseline_2of3", "baseline_any"]
CONTROL_REPS = 300
CONTROL_REPS_RUNTIME = CONTROL_REPS   # overridden by --control-reps


def _pooled_f1(preds: list, reals: list, tol: int) -> float:
    tp = n_pred = n_real = 0
    for p, r in zip(preds, reals):
        m, _, _ = match_events(list(p), list(r), tol)
        tp += len(m)
        n_pred += len(p)
        n_real += len(r)
    d = 2 * tp + (n_pred - tp) + (n_real - tp)
    return 2 * tp / d if d else float("nan")


def stage_controls(per_race: pd.DataFrame, out_dir: Path, tol: int = PIT_MATCH_TOLERANCE_LAPS,
                   reps: int = CONTROL_REPS, seed: int = 0):
    need = {"event_laps", "real_laps", "total_laps"}
    if not need.issubset(per_race.columns):
        print("[controls] per_race_driver_metrics.csv lacks event_laps/real_laps/total_laps. "
              "Re-run --stage evaluate (cached replays, takes seconds) with this version first.")
        return None
    main = per_race[per_race["tol"] == tol].copy()
    main["ev"] = main["event_laps"].apply(json.loads)
    main["real"] = main["real_laps"].apply(json.loads)
    rng = np.random.default_rng(seed)

    real_by = {(r.season, r.race, r.driver): r.real for r in main[main.system == "full"].itertuples()}
    by_race: dict = {}
    for (season, race, drv), real in real_by.items():
        by_race.setdefault(race, []).append((season, real))

    groups = [("ALL", main)] + [(sp, g) for sp, g in main.groupby("split")]
    out = []
    for sysname in CONTROL_SYSTEMS:
        for gname, g in groups:
            g = g[g["system"] == sysname]
            if g.empty:
                continue
            units = [(r.season, r.race, r.ev, r.real, int(r.total_laps)) for r in g.itertuples()]
            actual = _pooled_f1([u[2] for u in units], [u[3] for u in units], tol)

            # ---- Null A: uniform random timing, same event count, real pits as they were
            null_a = []
            for _ in range(reps):
                preds = [sorted(rng.choice(np.arange(1, L + 1), size=min(len(ev), L), replace=False).tolist())
                         if len(ev) else [] for (_, _, ev, _, L) in units]
                null_a.append(_pooled_f1(preds, [u[3] for u in units], tol))

            # ---- Null B + Ref: same circuit, different season (units with no other season are excluded from BOTH)
            usable, cands = [], []
            for u in units:
                cs = [real for (s2, real) in by_race.get(u[1], []) if s2 != u[0]]
                if cs:
                    usable.append(u)
                    cands.append(cs)
            null_b, ref = [], []
            actual_usable = _pooled_f1([u[2] for u in usable], [u[3] for u in usable], tol) if usable else float("nan")
            for _ in range(reps if usable else 0):
                pick = [c[rng.integers(len(c))] for c in cands]
                null_b.append(_pooled_f1([u[2] for u in usable], pick, tol))
                ref.append(_pooled_f1(pick, [u[3] for u in usable], tol))

            def summ(x):
                x = np.array(x, dtype=float)
                return (np.nanmean(x), np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)) if len(x) else (np.nan,) * 3

            a, b, r = summ(null_a), summ(null_b), summ(ref)
            out.append(dict(
                system=sysname, group=gname, n_units=len(units), actual_f1=actual,
                nullA_mean=a[0], nullA_lo=a[1], nullA_hi=a[2],
                pA=(1 + sum(x >= actual for x in null_a)) / (1 + len(null_a)),
                n_units_B=len(usable), actual_f1_B=actual_usable,
                nullB_mean=b[0], nullB_lo=b[1], nullB_hi=b[2],
                pB=((1 + sum(x >= actual_usable for x in null_b)) / (1 + len(null_b))) if null_b else np.nan,
                copy_other_year_f1=r[0], copy_other_year_lo=r[1], copy_other_year_hi=r[2]))
    df = pd.DataFrame(out)
    df.to_csv(out_dir / "controls.csv", index=False)
    show = df[["system", "group", "n_units", "actual_f1", "nullA_mean", "pA", "actual_f1_B", "nullB_mean", "pB",
               "copy_other_year_f1"]].round(3)
    print("\n=== NO-SKILL CONTROLS (tol=+-%d laps, %d replicates) ===" % (tol, reps))
    print(show.to_string(index=False))
    return df


def stage_evaluate(repo_root: Path, out_dir: Path, fold_models_used: bool):
    pit_loss_lookup = load_pit_loss_lookup(repo_root)
    all_rows, cf_all = [], []
    seasons = [f["test"] for f in FOLDS.values()] + [HOLDOUT_YEAR]

    for tol in TOLERANCE_SWEEP:
        for season in seasons:
            for race in list_races(repo_root, season):
                res, cf = evaluate_one_race(repo_root, out_dir, season, race, pit_loss_lookup, tol)
                if res is not None:
                    all_rows.append(res)
                    if tol == PIT_MATCH_TOLERANCE_LAPS:
                        cf_all.extend(cf)
    if EMPTY_REPLAYS:
        er = pd.DataFrame(EMPTY_REPLAYS).drop_duplicates()
        er.to_csv(out_dir / "excluded_races.csv", index=False)
        print(f"[evaluate] WARNING: {len(er)} race(s) excluded, see excluded_races.csv")
    if not all_rows:
        print("[evaluate] No replay outputs found. Run --stage replay first.")
        return
    per_race = pd.concat(all_rows, ignore_index=True)
    per_race["split"] = per_race["season"].apply(
        lambda y: "holdout_2025" if y == HOLDOUT_YEAR else f"test_{y}")
    per_race["leaky_models"] = not fold_models_used
    per_race.to_csv(out_dir / "per_race_driver_metrics.csv", index=False)

    main = per_race[per_race["tol"] == PIT_MATCH_TOLERANCE_LAPS]

    def summarise(g):
        tp, fa, ms = g["n_matched"].sum(), g["n_false_alarm"].sum(), g["n_missed"].sum()
        prec = tp / (tp + fa) if (tp + fa) else np.nan
        rec = tp / (tp + ms) if (tp + ms) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec)) else np.nan
        return pd.Series(dict(
            n_driver_races=len(g), real_pits=g["n_real_pits"].sum(), hermes_events=g["n_events"].sum(),
            matched=tp, false_alarms=fa, missed=ms,
            precision=prec, recall=rec, f1=f1,
            mean_abs_lap_error=g["mean_abs_lap_error"].mean(),
        ))

    by_split = main.groupby(["split", "system"]).apply(summarise).reset_index()
    by_split.to_csv(out_dir / "metrics_by_fold.csv", index=False)

    overall = main.groupby("system").apply(summarise).reset_index()
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)

    # Tolerance sweep (how strict is "agreement"?)
    sweep = per_race.groupby(["tol", "system"]).apply(summarise).reset_index()
    sweep.to_csv(out_dir / "tolerance_sweep.csv", index=False)
    cov = compute_coverage(out_dir)
    if not cov.empty:
        cov.to_csv(out_dir / "coverage_by_split.csv", index=False)
        print("\n=== MECHANISM COVERAGE (% of decisions where the input was UNAVAILABLE) ===")
        print(cov.to_string(index=False))

    # Paired full-vs-baseline bootstrap on per-driver-race F1 style recall (same races both systems)
    piv = main.pivot_table(index=["season", "race", "driver"], columns="system",
                           values=["n_matched", "n_false_alarm", "n_missed"], aggfunc="sum")
    paired = pd.DataFrame(index=piv.index)
    for s in ("full", MAIN_BASELINE):
        tp, fa, ms = piv[("n_matched", s)], piv[("n_false_alarm", s)], piv[("n_missed", s)]
        paired[f"f1_{s}"] = (2 * tp) / (2 * tp + fa + ms).replace(0, np.nan)
    paired["f1_diff"] = paired["f1_full"] - paired[f"f1_{MAIN_BASELINE}"]
    paired = paired.reset_index()
    paired.to_csv(out_dir / "paired_full_vs_baseline.csv", index=False)
    lo, hi = bootstrap_ci(paired["f1_diff"].tolist())
    pd.DataFrame([dict(metric="mean_f1_diff_full_minus_baseline",
                       mean=paired["f1_diff"].mean(), ci_lo=lo, ci_hi=hi,
                       n=paired["f1_diff"].notna().sum())]).to_csv(out_dir / "paired_bootstrap.csv", index=False)

    # Counterfactual
    cf = pd.DataFrame(cf_all)
    cf.to_csv(out_dir / "counterfactual_events.csv", index=False)
    if not cf.empty:
        cf_valid = cf.dropna(subset=["est_time_delta_s"])
        cflo, cfhi = bootstrap_ci(cf_valid["est_time_delta_s"].tolist())
        pd.DataFrame([dict(n_matched_events=len(cf), n_with_estimate=len(cf_valid),
                           mean_est_time_delta_s=cf_valid["est_time_delta_s"].mean(),
                           median_est_time_delta_s=cf_valid["est_time_delta_s"].median(),
                           ci_lo=cflo, ci_hi=cfhi,
                           frac_hermes_better=(cf_valid["est_time_delta_s"] > 0).mean())]
                     ).to_csv(out_dir / "counterfactual_summary.csv", index=False)

    # Position context (real, NOT a claim about HERMES)
    pos = main[main["system"] == "full"].groupby("split")["pos_change"].agg(["mean", "count"]).reset_index()
    pos.to_csv(out_dir / "real_position_change_context.csv", index=False)

    print("\n=== OVERALL (tol=+-%d laps) ===" % PIT_MATCH_TOLERANCE_LAPS)
    print(overall.round(3).to_string(index=False))
    print("\n=== BY FOLD ===")
    print(by_split.round(3).to_string(index=False))
    stage_controls(per_race, out_dir, reps=CONTROL_REPS_RUNTIME)
    print(f"\n[evaluate] wrote CSVs to {out_dir}")


# ----------------------------------------------------------------------------
# STAGE 4: REPORT
# ----------------------------------------------------------------------------
def stage_report(out_dir: Path, fold_models_used: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rd(name):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    by_fold, overall = rd("metrics_by_fold.csv"), rd("metrics_overall.csv")
    sweep, cfs = rd("tolerance_sweep.csv"), rd("counterfactual_summary.csv")
    boot, rlog = rd("paired_bootstrap.csv"), rd("replay_log.csv")
    coverage = rd("coverage_by_split.csv")
    controls = rd("controls.csv")

    if not by_fold.empty:
        piv = by_fold.pivot(index="split", columns="system", values="f1")
        ax = piv.plot(kind="bar", figsize=(8, 4.5))
        ax.set_ylabel("Pit-timing F1 (+-2 laps)")
        ax.set_title("HERMES full system vs tyre-only baseline, by test year")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_f1_by_fold.png", dpi=150)
        plt.close()
    if not sweep.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for s, g in sweep.groupby("system"):
            ax.plot(g["tol"], g["recall"], marker="o", label=f"{s} recall")
            ax.plot(g["tol"], g["precision"], marker="s", ls="--", label=f"{s} precision")
        ax.set_xlabel("Match tolerance (+- laps)")
        ax.set_ylabel("Score")
        ax.set_title("Sensitivity of agreement to tolerance")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "fig_tolerance_sweep.png", dpi=150)
        plt.close()

    def md(df):
        return df.round(3).to_markdown(index=False) if not df.empty else "_no data_"

    n_ok = int((rlog["status"].isin(["OK", "CACHED"])).sum()) if not rlog.empty else 0
    n_skip = int((rlog["status"].str.startswith("SKIP")).sum()) if not rlog.empty else 0
    n_fail = int((rlog["status"] == "FAIL").sum()) if not rlog.empty else 0

    leak_note = ("**Models, thresholds, stint priors and SC prior refit on TRAINING YEARS ONLY per fold (clean walk-forward; 2025 trained on 2018-2024).** Pit-loss table filtering: see fit_report.json." if fold_models_used else
                 "**DEVELOPMENT / CONSISTENCY EVALUATION ONLY: production tyre models and thresholds were fit on "
                 "100% of data (2018-2026), so every fold below is optimistic (leaky).** Read as a stability "
                 "check across years, NOT as an out-of-sample generalisation result. See eval_out_clean for that.")

    report = f"""# HERMES Full-System Evaluation (Red Bull)

## Setup
- Races replayed OK/cached: {n_ok} | skipped: {n_skip} | failed: {n_fail} (see replay_log.csv)
- Walk-forward test years: 2021, 2022, 2023, 2024. Untouched holdout: 2025.
- Baselines (tyre-life-only, no rival/SC/radio/weather): main = 2 of 3 tyre triggers; also reported: strict (3 of 3) and any (1 of 3). Strict almost never fires, so it is NOT the headline comparison.
- Agreement tolerance: +-{PIT_MATCH_TOLERANCE_LAPS} laps (sweep in tolerance_sweep.csv).
- Consecutive PIT_NOW laps are collapsed into ONE recommendation event.

{leak_note}

## Overall
{md(overall)}

## By fold / holdout
{md(by_fold)}

## Mechanism coverage (% of decisions where the input was UNAVAILABLE)
{md(coverage)}
Read every fold score next to this table. `pct_pace_loss_none` / `pct_cliff_none` high = the tyre model had no
stratum for that race's era/circuit (regulation reset, MDP para 125). `pct_sc_not_reached` = Tier 1/2 decided first,
so the SC gamble was never invoked. `pct_sc_insufficient` = it WAS invoked but had no data (no SC prior, which needs
>=4 training races per circuit, or no tyre projection).

## No-skill controls (read BEFORE the tables above)
{md(controls[["system","group","n_units","actual_f1","nullA_mean","pA","actual_f1_B","nullB_mean","pB","copy_other_year_f1"]]) if not controls.empty else "_run --stage evaluate with the controls version_"}
`nullA` = events at random laps. `nullB` = same events scored against the same circuit's real stops in a DIFFERENT
season (keeps circuit-typical timing; the demanding control). `pA`/`pB` = one-sided permutation p-values that the
actual F1 beats the null. `copy_other_year_f1` = F1 of just copying another season's stops (a naive reference, uses
future seasons). F1 at +-2 laps has a non-zero floor because stops cluster mid-race.

## Paired full-vs-baseline (per driver-race F1 difference, bootstrap 95% CI)
{md(boot)}

## Counterfactual (ESTIMATED, first-order, see assumptions)
{md(cfs)}
Positive = HERMES timing estimated faster than the real pit lap. Assumes equal pit loss, ignores traffic and rival reactions.

## Honest interpretation guardrails
- A real team pit is NOT ground truth. High agreement means HERMES resembles teams, not that it is better.
- Low precision may be correct behaviour if HERMES flags earlier windows than teams act on.
- The counterfactual is an estimate, not a race simulation.
- Driver-priority tie-break (D1 always wins) suppresses D2 instructions when both trigger (known, see British GP finding).
"""
    (out_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"[report] {out_dir / 'evaluation_report.md'}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HERMES full-system evaluation")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--master", default="system/HERMES/trees/master.py",
                    help="path to your orchestrator, relative to repo root")
    ap.add_argument("--out-dir", default="eval_out")
    ap.add_argument("--stage", choices=["replay", "evaluate", "report", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="re-run replays even if cached")
    ap.add_argument("--fold-models", action="store_true",
                    help="(legacy) mark results as fold-trained. Prefer --fold-mode, which sets this for you.")
    ap.add_argument("--fold-mode", action="store_true",
                    help="CLEAN walk-forward run: each test year is replayed with artefacts fit on "
                         "TRAINING years only (from fit_fold.py). Writes to eval_out_clean by default.")
    ap.add_argument("--folds-dir", default="folds", help="where fit_fold.py wrote fold artefacts")
    ap.add_argument("--control-reps", type=int, default=CONTROL_REPS, help="permutation replicates for the controls")
    ap.add_argument("--allow-cross-era", action="store_true",
                    help="SENSITIVITY variant of --fold-mode: let a fold's models be used for a race in an era "
                         "they were not trained on (e.g. 2018-2021 models on 2022). Writes to eval_out_clean_crossera.")
    args = ap.parse_args()

    global CONTROL_REPS_RUNTIME
    CONTROL_REPS_RUNTIME = args.control_reps
    repo_root = Path(args.repo_root).resolve()
    if args.allow_cross_era and not args.fold_mode:
        ap.error("--allow-cross-era only makes sense together with --fold-mode")
    if args.fold_mode and args.out_dir == "eval_out":
        args.out_dir = "eval_out_clean_crossera" if args.allow_cross_era else "eval_out_clean"  # never overwrite dev results
    if args.fold_mode:
        args.fold_models = True                # so the report stops saying "leaky"
    out_dir = (repo_root / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master = repo_root / args.master
    seasons = [f["test"] for f in FOLDS.values()] + [HOLDOUT_YEAR]

    if args.stage in ("replay", "all"):
        stage_replay(repo_root, out_dir, master, seasons, args.force,
                     fold_mode=args.fold_mode, folds_dir=(repo_root / args.folds_dir),
                     allow_cross_era=args.allow_cross_era)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(repo_root, out_dir, args.fold_models)
    if args.stage in ("report", "all"):
        stage_report(out_dir, args.fold_models)


if __name__ == "__main__":
    main()