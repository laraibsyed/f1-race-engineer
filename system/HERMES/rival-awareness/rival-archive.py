"""
Rival Knowledge — Full Archive Runner
======================================
Runs the already-validated single-race pipeline (knowledge.py) across every Race/Sprint
session in the bucket, instead of one hardcoded race at a time.

MUST live in the same folder as knowledge.py — imports its functions directly rather
than duplicating them, so any future fix to the core logic (opponent extraction, pit
loss, undercut flag) only ever needs to happen in one place.

Validated on 3 races before this run (2023 Bahrain, 2019 Monaco, 2018 Abu Dhabi) — see
knowledge.py's own module docstring for exactly what that validation found and didn't
find. Known limitation carried forward: this detects real, correctly-timed pit-lane
stops correlated with a gap/tyre-age condition — it does NOT distinguish undercut from
overcut, or genuine strategic intent from a traffic-forced or fastest-lap-gamble stop
that happened to match the same numbers. Treat archive-wide counts as "candidate
strategic pit-timing events," not confirmed undercuts, in anything you write up from
this output.

Per project rule: full-archive raw-telemetry scans are expensive — this is the same
kind of scan the original overtake-defense work needed for the driver taxonomy. This
script is race-first and checkpointed at two levels (per-race, inside knowledge.py's
own build_opponent_table; and per-race again here, at the archive level) — safe to
Ctrl+C and resume without losing completed races.

KNOWN UNVERIFIED ASSUMPTION: Sprint session name translation (fastf1 "S" -> tracing-
insights "Sprint") is a guess. The data-cleaning handoff notes some 2022 sprints used
lowercase "sprint" instead — this script can't verify that from here. The per-race
coverage check below will warn loudly if a race's opponent table covers under 50% of
its laps, which is the symptom a wrong session-name translation would produce — inspect
any race that triggers this warning before trusting its numbers.

Usage:
    python run_archive.py --dry-run          # list every discovered race, run nothing
    python run_archive.py --limit 5          # smoke-test on the first 5 races only
    python run_archive.py                    # run everything
    python run_archive.py --force            # re-run races even if already completed
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

import knowledge as rk   # <-- this file must sit in the same folder as knowledge.py


ARCHIVE_SUMMARY_PATH = "./checkpoints/rival_knowledge/archive_event_summary.csv"
ARCHIVE_LOG_PATH = "./checkpoints/rival_knowledge/archive_run_log.csv"


def discover_races(cb):
    """Lists clean/features/ to find every (year, race_fastf1, session_fastf1) combo
    that has engineered features. Race and Sprint sessions only — gap_to_leader/
    gap_to_car_ahead/fuel_load_estimate are explicitly nulled for FP/Q sessions in
    engineer_features.py (per the data-cleaning handoff), so this module has nothing
    meaningful to do there.

    NOTE: clean/features/ has NO "fastf1" subfolder — that naming only applies under
    raw/fastf1/ and clean/tracinginsights/. Confirmed against the bucket-setup notes
    and against load_features() in knowledge.py, which already uses this exact path.
    An earlier version of this function wrongly assumed clean/features/fastf1/, which
    matched zero files."""
    names = cb.list_blob_names("clean/features/")
    pattern = re.compile(r"clean/features/(\d{4})/([^/]+)/(R|S)/laps_features\.csv")
    races = set()
    for name in names:
        m = pattern.match(name)
        if m:
            year, race_fastf1, session = m.groups()
            races.add((int(year), race_fastf1, session))
    return sorted(races)


def race_ti_and_session_ti(race_fastf1, session_fastf1):
    """Translates fastf1-style naming to tracinginsights-style naming.
    CONFIRMED for Race sessions (space-separated race name, session "Race").
    Sprint translation ("Sprint") is an educated guess, not verified — see module
    docstring. The coverage check in run_one_race() will flag it if wrong for a given
    race rather than silently producing an empty, misleading result."""
    race_ti = race_fastf1.replace("_", " ")
    session_ti = "Race" if session_fastf1 == "R" else "Sprint"
    return race_ti, session_ti


# Sessions confirmed (via this project's own data-cleaning notes) to have been
# backfilled from FastF1's own API rather than tracinginsights' per-lap JSON files.
# The backfilled data lands in clean/tracinginsights/telemetry_by_lap.csv, which the
# original handoff already confirmed LACKS DriverAhead/DistanceToDriverAhead entirely
# — so this module's precise-telemetry route genuinely cannot produce opponent data
# for these, permanently, regardless of any naming fix. Every Sprint session qualifies
# (tracinginsights never collected Sprint telemetry at all); the two known backfilled
# Race sessions are listed explicitly.
KNOWN_BACKFILLED_RACE_SESSIONS = {
    (2018, "Bahrain_Grand_Prix", "R"),       # confirmed unrecoverable — no telemetry in any source
    (2018, "Australian_Grand_Prix", "R"),    # confirmed backfilled from FastF1's API
}


def is_known_backfilled_session(year, race_fastf1, session_fastf1):
    if session_fastf1 == "S":
        return True
    return (year, race_fastf1, session_fastf1) in KNOWN_BACKFILLED_RACE_SESSIONS


# A different kind of known limitation: real telemetry exists, but there's no
# meaningful pit strategy to detect. 2021 Belgian GP (Spa) was red-flagged after ~2
# laps behind the Safety Car in torrential rain — already documented in this project's
# data-cleaning notes as "~2 laps of real racing". compute_pit_lane_loss correctly
# finds no valid pit-in/out laps to build a constant from; that's the race, not a bug.
KNOWN_NO_PIT_STRATEGY_SESSIONS = {
    (2021, "Belgian_Grand_Prix", "R"),
    # 2023 Australian GP: three separate red-flag periods fragmented the timing data so
    # badly that only 2 genuinely clean pit-stop samples survive out of ~65 pit-flagged
    # laps (confirmed via diagnose_pit_loss.py) — most pit-in/pit-out laps have no
    # computable LapTime at all, not just a contaminated one. n=2 is an order of
    # magnitude below the project's own small-sample threshold; no amount of refining
    # the duration calculation fixes a race that lacks the underlying clean data.
    (2023, "Australian_Grand_Prix", "R"),
}


def is_known_limitation_session(year, race_fastf1, session_fastf1):
    return (is_known_backfilled_session(year, race_fastf1, session_fastf1)
            or (year, race_fastf1, session_fastf1) in KNOWN_NO_PIT_STRATEGY_SESSIONS)


def run_one_race(cb, year, race_fastf1, session_fastf1, force=False):
    race_ti, session_ti = race_ti_and_session_ti(race_fastf1, session_fastf1)
    race_label = f"{year} {race_ti} ({session_fastf1})"

    # GCS paths — the real source of truth for "is this race done", not local disk.
    # Backing these up as the archive progresses is what lets the run survive a crash
    # or a move to a different machine (e.g. a VM) without starting over.
    base = f"derived/rival_knowledge/{year}/{race_fastf1}/{session_fastf1}"
    events_blob = f"{base}/undercut_events.csv"
    windows_blob = f"{base}/undercut_windows.csv"

    if cb.exists(events_blob) and not force:
        print(f"[archive] Skipping {race_label} — already in bucket ({events_blob})")
        events = cb.read_csv(events_blob)
        # Reconstruct the same fields the "ok" branch below would log, so the run log
        # is equally analyzable whether a race ran fresh this time or was skipped —
        # otherwise a fully-skipped rerun (like this one) produces a log with no
        # n_events/pit_loss_constant_s columns at all, since nothing ever set them.
        known = is_known_limitation_session(year, race_fastf1, session_fastf1)
        pit_loss = (events["pit_loss_constant_s"].iloc[0]
                    if not events.empty and "pit_loss_constant_s" in events.columns else np.nan)
        return events, {"race_label": race_label, "status": "skipped_existing",
                         "coverage": np.nan,  # genuinely unrecoverable without re-running — not backed up separately
                         "pit_loss_constant_s": pit_loss, "n_events": len(events),
                         "known_backfilled_limitation": known}

    # Set the shared module's config globals before calling its functions — they read
    # these as module-level names, so this is enough to redirect every function without
    # needing to touch knowledge.py's own code.
    rk.YEAR = year
    rk.RACE_FASTF1 = race_fastf1
    rk.RACE_TI = race_ti
    rk.SESSION_FASTF1 = session_fastf1
    rk.SESSION_TI = session_ti

    try:
        print(f"[archive] ({race_label}) Loading laps_features.csv ...")
        features_df = rk.load_features(cb)
        features_df = rk.add_cumulative_time(features_df)
        number_to_code = rk.build_number_to_code(features_df)

        print(f"[archive] ({race_label}) Building per-lap opponent table ...")
        # NOTE: this step's own per-driver checkpoint (inside knowledge.py) is still
        # LOCAL DISK ONLY — if the run dies mid-race (not between races) and you then
        # move to a fresh VM, that partial progress within this one race is lost and
        # it restarts from driver 1. Only whole-race completion is backed to the bucket
        # here. Say the word if you want that per-driver checkpoint backed up too.
        opponent_df = rk.build_opponent_table(cb, features_df, number_to_code)

        # Coverage check — never guess a schema/naming translation silently (project
        # rule). Known-backfilled sessions (all Sprints + 2018 Bahrain/Australian GP)
        # are EXPECTED to show ~0% coverage — that's a documented data ceiling, not a
        # naming bug — so they get a clear status instead of the generic warning below,
        # which is reserved for genuinely unexpected low coverage worth investigating.
        total_laps = len(features_df)
        coverage = len(opponent_df) / max(total_laps, 1)
        if is_known_backfilled_session(year, race_fastf1, session_fastf1):
            print(f"[archive] ({race_label}) Known FastF1-backfilled session — no raw "
                  f"per-lap telemetry exists (documented limitation, not a bug). "
                  f"Coverage {coverage:.0%} is expected.")
        elif coverage < 0.5:
            print(f"[archive] WARNING: {race_label} — opponent table covers only "
                  f"{coverage:.0%} of laps_features.csv rows. Likely a session/race "
                  f"naming mismatch — inspect this race's raw path before trusting it.")

        print(f"[archive] ({race_label}) Computing empirical pit-lane time loss ...")
        loss_df, circuit_constant_s = rk.compute_pit_lane_loss(features_df)
        if circuit_constant_s is None:
            known = is_known_limitation_session(year, race_fastf1, session_fastf1)
            note = " (documented known limitation — see KNOWN_NO_PIT_STRATEGY_SESSIONS / KNOWN_BACKFILLED_RACE_SESSIONS)" if known else " — UNEXPECTED, worth investigating"
            print(f"[archive] ({race_label}) No pit-lane loss constant available{note} — skipping undercut flagging.")
            return pd.DataFrame(), {"race_label": race_label, "status": "no_pit_loss_constant",
                                     "coverage": coverage, "known_backfilled_limitation": known}

        flagged = rk.flag_undercut_windows(opponent_df, features_df, circuit_constant_s)
        cb.write_csv(windows_blob, flagged)

        events = rk.summarize_undercut_events(flagged)
        events["year"] = year
        events["race"] = race_ti
        events["session"] = session_fastf1
        events["pit_loss_constant_s"] = circuit_constant_s
        cb.write_csv(events_blob, events)

        print(f"[archive] ({race_label}) Done — {len(events)} candidate events, coverage {coverage:.0%}. Backed up to {events_blob}")
        return events, {"race_label": race_label, "status": "ok", "coverage": coverage,
                         "pit_loss_constant_s": circuit_constant_s, "n_events": len(events),
                         "known_backfilled_limitation": is_known_limitation_session(year, race_fastf1, session_fastf1)}

    except Exception as e:
        # Per-race try/except so one bad session doesn't take down the whole archive
        # run — project rule, applied consistently everywhere else in this pipeline.
        print(f"[archive] ERROR on {race_label}: {e!r} — skipping, continuing with next race.")
        return pd.DataFrame(), {"race_label": race_label, "status": f"error: {e!r}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List discovered races and exit; run nothing.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N discovered races (for smoke-testing).")
    parser.add_argument("--force", action="store_true", help="Re-run races even if their events CSV already exists.")
    args = parser.parse_args()

    cb = rk.CachedBucket()

    print("[archive] Discovering Race/Sprint sessions with engineered features ...")
    races = discover_races(cb)
    print(f"[archive] Found {len(races)} Race/Sprint sessions.")

    if args.limit:
        races = races[: args.limit]
        print(f"[archive] --limit set: running only the first {len(races)}.")

    if args.dry_run:
        for year, race_fastf1, session in races:
            print(f"  {year} {race_fastf1} ({session})")
        return

    all_events = []
    run_log = []
    for i, (year, race_fastf1, session) in enumerate(races):
        print(f"\n[archive] === Race {i + 1}/{len(races)} ===")
        events, log_entry = run_one_race(cb, year, race_fastf1, session, force=args.force)
        if not events.empty:
            all_events.append(events)
        run_log.append(log_entry)

        # Checkpoint the running combined outputs after every race, not just at the
        # end — both locally AND to the bucket, so the summary itself survives a crash
        # or a move to a different machine, not just the individual per-race results.
        if all_events:
            combined = pd.concat(all_events, ignore_index=True)
            combined.to_csv(ARCHIVE_SUMMARY_PATH, index=False)
            cb.write_csv("derived/rival_knowledge/_archive/event_summary.csv", combined)
        log_df = pd.DataFrame(run_log)
        log_df.to_csv(ARCHIVE_LOG_PATH, index=False)
        cb.write_csv("derived/rival_knowledge/_archive/run_log.csv", log_df)

    print(f"\n[archive] Done. Combined event summary: {ARCHIVE_SUMMARY_PATH}")
    print(f"[archive] Per-race run log (status, coverage, errors): {ARCHIVE_LOG_PATH}")

    log_df = pd.DataFrame(run_log)
    n_ok = log_df["status"].isin(["ok", "skipped_existing"]).sum() if "status" in log_df else 0
    n_known_limitation = log_df.get("known_backfilled_limitation", pd.Series(dtype=bool)).fillna(False).sum()
    n_unexpected_low_coverage = ((log_df.get("coverage", pd.Series(dtype=float)) < 0.5)
                                  & ~log_df.get("known_backfilled_limitation", pd.Series(dtype=bool)).fillna(False)).sum()
    n_errors = log_df["status"].astype(str).str.startswith("error").sum() if "status" in log_df else 0
    print(f"[archive] Summary: {n_ok} races completed ({n_known_limitation} known-limitation, "
          f"0 events expected), {n_unexpected_low_coverage} flagged with UNEXPECTED low coverage "
          f"(worth investigating), {n_errors} errored — see {ARCHIVE_LOG_PATH} for details.")


if __name__ == "__main__":
    main()