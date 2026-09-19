"""
Rival Knowledge — Archive Analysis
====================================
Analyzes the full-archive undercut-event results (archive_event_summary.csv +
archive_run_log.csv) produced by rival-archive.py. Run this AFTER the full archive run
has completed and the log has been reconciled (25 known-backfilled sessions + 2021
Belgian GP excluded, per the earlier validation conversation).

Reads from LOCAL checkpoints by default (./checkpoints/rival_knowledge/), since that's
what's immediately available without extra bucket round-trips. If you've since run
individual sessions with --force and only the bucket copy is current, pull those down
first with knowledge.py's CachedBucket before running this.

Produces:
  1. Headline stats: races with data, total events, events-per-race distribution.
  2. Per-year trend: does undercut activity change across regulation eras
     (2018-2021 vs 2022-2025 ground-effect vs 2026)?
  3. Per-circuit ranking: which circuits produce the most/fewest candidate events —
     a rough proxy for "how much does pit strategy matter here" (expect low-overtake
     circuits like Monaco/Singapore to still show meaningful activity, per the Bahrain/
     Monaco/Abu Dhabi validation findings earlier).
  4. Event window-length distribution: single-lap ("traffic-forced" pattern, per the
     Gasly/Bahrain finding) vs multi-lap ("building" pattern, per the Ocon/Bahrain
     confirmed undercut) — is one more common than the other archive-wide?
  5. Pit-loss-constant outlier flags: the archive showed a huge spread (3.4s to 51.7s)
     — flags any race whose constant falls outside a plausible physical range for
     manual sanity-checking, rather than silently trusting every value.

Usage:
    python analyze_archive.py
    python analyze_archive.py --plot     # also save PNG charts (needs matplotlib)
"""

import argparse
import pandas as pd
import numpy as np

SUMMARY_PATH = "./checkpoints/rival_knowledge/archive_event_summary.csv"
LOG_PATH = "./checkpoints/rival_knowledge/archive_run_log.csv"

# Regulation eras — a natural grouping for a per-era comparison, since the underlying
# aerodynamics (and therefore overtaking/undercut dynamics) changed meaningfully here.
def era_for_year(year):
    if year <= 2021:
        return "2018-2021 (pre ground-effect)"
    elif year <= 2025:
        return "2022-2025 (ground-effect)"
    else:
        return "2026+ (new regs)"


# Some Grand Prix have been renamed across the archive's timespan while running at the
# SAME physical circuit — left unnormalized, the per-circuit ranking silently splits
# one circuit's real sample size across two labels. Confirmed pairs (verify against
# the full per-race CSV before assuming this list is exhaustive — there may be others
# not caught here, e.g. further renames later in the archive).
CIRCUIT_NAME_ALIASES = {
    "Brazilian Grand Prix": "São Paulo Grand Prix",       # both Interlagos
    "Mexican Grand Prix": "Mexico City Grand Prix",       # both Autódromo Hermanos Rodríguez
}


def normalize_circuit_name(name):
    return CIRCUIT_NAME_ALIASES.get(name, name)


# Plausibility bounds for a pit-lane time loss constant — genuinely varies a lot by
# circuit (a long pit lane like Bahrain's vs a short one), but anything far outside
# this range is worth eyeballing rather than trusting blind. Not a hard rule, just a
# flag threshold.
PIT_LOSS_PLAUSIBLE_MIN = 8.0
PIT_LOSS_PLAUSIBLE_MAX = 35.0


def load_data():
    summary = pd.read_csv(SUMMARY_PATH)
    log = pd.read_csv(LOG_PATH)
    return summary, log


def build_per_race_table(summary, log):
    """One row per race with usable data: n_events, pit_loss_constant_s, year, era.
    Includes races with ZERO events (which don't appear in archive_event_summary at
    all) by starting from the run log and left-joining event counts onto it.

    Accepts both "ok" (ran fresh this session) and "skipped_existing" (already in the
    bucket from a prior run) — run_archive.py now reconstructs n_events/pit_loss_
    constant_s for skipped races from their cached events file, so both statuses carry
    the same usable fields."""
    if "n_events" not in log.columns:
        raise SystemExit(
            "run_log.csv has no 'n_events' column at all — this means it was produced "
            "by an older version of run_archive.py's skip-branch (before the fix that "
            "backfills metrics for skipped races). Re-run rival-archive.py with the "
            "updated script (no --force needed, it'll just skip-and-relog everything) "
            "before running this analysis."
        )

    usable = log[
        (log["status"].isin(["ok", "skipped_existing"]))
        & (~log.get("known_backfilled_limitation", pd.Series(dtype=bool)).fillna(False).infer_objects(copy=False))
    ].copy()

    # Parse "2023 Bahrain Grand Prix (R)" back into year/race/session
    parsed = usable["race_label"].str.extract(r"^(\d{4}) (.+) \((R|S)\)$")
    usable["year"] = parsed[0].astype(int)
    usable["race"] = parsed[1].apply(normalize_circuit_name)
    usable["session"] = parsed[2]
    usable["era"] = usable["year"].apply(era_for_year)

    # n_events and pit_loss_constant_s should already be columns on the log itself
    # (run_one_race's "ok" branch includes them) — use those directly rather than
    # re-deriving from the summary, which only has per-event rows and would need a
    # groupby to reconstruct races with events, plus a separate zero-fill for races
    # with none.
    usable["n_events"] = usable["n_events"].fillna(0).astype(int)
    return usable


def headline_stats(per_race):
    print("=" * 70)
    print("HEADLINE STATS")
    print("=" * 70)
    print(f"Races with usable data: {len(per_race)}")
    print(f"Total candidate events across archive: {per_race['n_events'].sum()}")
    print(f"Events per race — mean: {per_race['n_events'].mean():.1f}, "
          f"median: {per_race['n_events'].median():.1f}, "
          f"std: {per_race['n_events'].std():.1f}, "
          f"min: {per_race['n_events'].min()}, max: {per_race['n_events'].max()}")
    zero_event_races = per_race[per_race["n_events"] == 0]
    print(f"Races with ZERO candidate events: {len(zero_event_races)} "
          f"({', '.join(zero_event_races['race_label'].tolist()[:10])}"
          f"{' ...' if len(zero_event_races) > 10 else ''})")
    print()


def per_era_comparison(per_race):
    print("=" * 70)
    print("PER-ERA COMPARISON")
    print("=" * 70)
    era_stats = per_race.groupby("era")["n_events"].agg(["count", "mean", "median", "std"]).round(2)
    era_stats.columns = ["n_races", "mean_events", "median_events", "std_events"]
    print(era_stats.to_string())
    print()
    print("NOTE: read this as a descriptive pattern, not a causal claim — regulation era")
    print("correlates with many things (car design, tyre supplier changes, calendar mix)")
    print("besides just overtaking difficulty. Useful for your discussion section, not")
    print("for asserting the regs alone caused any difference seen here.")
    print()


def per_circuit_ranking(per_race, top_n=10):
    print("=" * 70)
    print(f"PER-CIRCUIT RANKING (mean events per race, min 2 races)")
    print("=" * 70)
    circuit_stats = per_race.groupby("race")["n_events"].agg(["count", "mean"]).round(2)
    circuit_stats.columns = ["n_races", "mean_events"]
    circuit_stats = circuit_stats[circuit_stats["n_races"] >= 2].sort_values("mean_events", ascending=False)

    print(f"\nTop {top_n} — most candidate events on average:")
    print(circuit_stats.head(top_n).to_string())
    print(f"\nBottom {top_n} — fewest candidate events on average:")
    print(circuit_stats.tail(top_n).to_string())

    small_sample = circuit_stats[circuit_stats["n_races"] < 4]
    if not small_sample.empty:
        print(f"\nWARNING — {len(small_sample)} circuit(s) in this table rest on fewer than "
              f"4 races (project rule: small samples produce misleading extremes). Treat "
              f"their ranking position as provisional, not a stable circuit characteristic:")
        print(small_sample.to_string())
    print()
    print("NOTE: a circuit near the bottom (e.g. Monaco, if it lands there) is worth a")
    print("qualitative sanity check against what you already found in the single-race")
    print("validation — Monaco actually showed MEANINGFUL undercut activity there (12")
    print("events in the one race checked), not near-zero, because pit strategy matters")
    print("MORE where overtaking is hard, not less. If the archive-wide ranking disagrees")
    print("with that single-race finding, that's worth understanding, not just reporting.")
    print()


def event_window_length_analysis():
    """Reads each race's individual events CSV to get window lengths — the combined
    archive summary doesn't carry window_start/end, so this re-reads per-race files.

    FIXED: originally globbed ./checkpoints/rival_knowledge/*_undercut_events.csv, the
    OLD local-only naming convention from before the archive switched to bucket-backed
    storage. That pattern only ever matched the 3 races run individually early on
    (Bahrain/Monaco/Abu Dhabi POC validation) — 44 events total, not the real ~2800+
    archive-wide count. The real per-race files now live under the CachedBucket local
    mirror path: ./gcs_cache/derived/rival_knowledge/{year}/{race}/{session}/
    undercut_events.csv (or ./gcs_cache/derived/rival_knowledge/... under whatever
    GCS_CACHE_DIR is set to). Checks both locations so results from any run stage
    still count."""
    import glob

    patterns = [
        "./checkpoints/rival_knowledge/*_undercut_events.csv",          # old POC-stage files
        "./gcs_cache/derived/rival_knowledge/**/undercut_events.csv",   # real archive-wide files
    ]
    event_files = []
    for pattern in patterns:
        event_files.extend(glob.glob(pattern, recursive=True))
    event_files = sorted(set(event_files))

    all_lengths = []
    races_read = 0
    for f in event_files:
        try:
            df = pd.read_csv(f)
            if "window_start_lap" in df.columns and "window_end_lap" in df.columns and not df.empty:
                lengths = (df["window_end_lap"] - df["window_start_lap"] + 1).tolist()
                all_lengths.extend(lengths)
                races_read += 1
        except Exception:
            continue

    if not all_lengths:
        print("[window analysis] No per-race event files with window columns found — "
              "check that ./gcs_cache/ (or your GCS_CACHE_DIR) actually has the "
              "downloaded events files, not just run_log.csv/archive_event_summary.csv.")
        return

    print(f"[window analysis] Read window lengths from {races_read} race files "
          f"({len(all_lengths)} total events).")

    lengths_series = pd.Series(all_lengths)
    print("=" * 70)
    print("EVENT WINDOW LENGTH DISTRIBUTION (archive-wide)")
    print("=" * 70)
    print(lengths_series.value_counts().sort_index().to_string())
    single_lap_pct = (lengths_series == 1).mean() * 100
    print(f"\n{single_lap_pct:.0f}% of events are single-lap (the 'traffic-forced' pattern seen")
    print("with Gasly at Bahrain) vs multi-lap ('building' pattern seen with Ocon's")
    print("confirmed undercut). Worth reporting this split explicitly — it's evidence")
    print("for the point made earlier: the detector doesn't distinguish intent, and a")
    print("large single-lap share would reinforce treating it as a correlation detector.")
    print()


def pit_loss_outliers(per_race):
    print("=" * 70)
    print(f"PIT-LOSS CONSTANT SANITY CHECK (plausible range: {PIT_LOSS_PLAUSIBLE_MIN}-{PIT_LOSS_PLAUSIBLE_MAX}s)")
    print("=" * 70)
    outliers = per_race[
        (per_race["pit_loss_constant_s"] < PIT_LOSS_PLAUSIBLE_MIN)
        | (per_race["pit_loss_constant_s"] > PIT_LOSS_PLAUSIBLE_MAX)
    ].sort_values("pit_loss_constant_s")
    if outliers.empty:
        print("No outliers outside the plausible range — good sign.")
    else:
        print(f"{len(outliers)} race(s) outside the plausible range — check these against")
        print("real pit-lane data before trusting their undercut flags, since the small-")
        print("sample rule (MIN_CLEAN_LAPS_FOR_BASELINE) can still let a handful of pit")
        print("laps produce a skewed constant:\n")
        print(outliers[["race_label", "pit_loss_constant_s", "n_events"]].to_string(index=False))
    print()


def maybe_plot(per_race):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — skipping charts. pip install matplotlib to enable.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    yearly = per_race.groupby("year")["n_events"].mean()
    axes[0].plot(yearly.index, yearly.values, marker="o")
    axes[0].set_title("Mean candidate events per race, by year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Mean events")

    axes[1].hist(per_race["n_events"], bins=range(0, per_race["n_events"].max() + 2))
    axes[1].set_title("Distribution of events per race")
    axes[1].set_xlabel("Events in race")
    axes[1].set_ylabel("Number of races")

    plt.tight_layout()
    out_path = "./checkpoints/rival_knowledge/archive_analysis_charts.png"
    plt.savefig(out_path, dpi=150)
    print(f"[plot] Saved charts to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Also save PNG charts (needs matplotlib).")
    args = parser.parse_args()

    summary, log = load_data()
    per_race = build_per_race_table(summary, log)

    headline_stats(per_race)
    per_era_comparison(per_race)
    per_circuit_ranking(per_race)
    event_window_length_analysis()
    pit_loss_outliers(per_race)

    per_race.to_csv("./checkpoints/rival_knowledge/archive_per_race_analysis.csv", index=False)
    print(f"Per-race analysis table saved to ./checkpoints/rival_knowledge/archive_per_race_analysis.csv")

    if args.plot:
        maybe_plot(per_race)


if __name__ == "__main__":
    main()