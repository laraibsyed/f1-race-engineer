
import os
import re
import argparse
import numpy as np
import pandas as pd

import knowledge as rk

ARCHIVE_SUMMARY_PATH = "./checkpoints/rival_knowledge/archive_event_summary.csv"
ARCHIVE_LOG_PATH = "./checkpoints/rival_knowledge/archive_run_log.csv"

def discover_races(cb):
    ""
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
    ""
    race_ti = race_fastf1.replace("_", " ")
    session_ti = "Race" if session_fastf1 == "R" else "Sprint"
    return race_ti, session_ti

KNOWN_BACKFILLED_RACE_SESSIONS = {
    (2018, "Bahrain_Grand_Prix", "R"),
    (2018, "Australian_Grand_Prix", "R"),
}

def is_known_backfilled_session(year, race_fastf1, session_fastf1):
    if session_fastf1 == "S":
        return True
    return (year, race_fastf1, session_fastf1) in KNOWN_BACKFILLED_RACE_SESSIONS

KNOWN_NO_PIT_STRATEGY_SESSIONS = {
    (2021, "Belgian_Grand_Prix", "R"),

    (2023, "Australian_Grand_Prix", "R"),
}

def is_known_limitation_session(year, race_fastf1, session_fastf1):
    return (is_known_backfilled_session(year, race_fastf1, session_fastf1)
            or (year, race_fastf1, session_fastf1) in KNOWN_NO_PIT_STRATEGY_SESSIONS)

def run_one_race(cb, year, race_fastf1, session_fastf1, force=False):
    race_ti, session_ti = race_ti_and_session_ti(race_fastf1, session_fastf1)
    race_label = f"{year} {race_ti} ({session_fastf1})"

    base = f"derived/rival_knowledge/{year}/{race_fastf1}/{session_fastf1}"
    events_blob = f"{base}/undercut_events.csv"
    windows_blob = f"{base}/undercut_windows.csv"

    if cb.exists(events_blob) and not force:
        print(f"[archive] Skipping {race_label} — already in bucket ({events_blob})")
        events = cb.read_csv(events_blob)

        known = is_known_limitation_session(year, race_fastf1, session_fastf1)
        pit_loss = (events["pit_loss_constant_s"].iloc[0]
                    if not events.empty and "pit_loss_constant_s" in events.columns else np.nan)
        return events, {"race_label": race_label, "status": "skipped_existing",
                         "coverage": np.nan,
                         "pit_loss_constant_s": pit_loss, "n_events": len(events),
                         "known_backfilled_limitation": known}

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

        opponent_df = rk.build_opponent_table(cb, features_df, number_to_code)

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
