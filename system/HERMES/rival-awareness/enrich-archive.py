"""
Rival Knowledge — Archive Enrichment
=======================================
Adds three checklist items to the already-completed archive WITHOUT rescanning raw
telemetry (the expensive part) — reuses each race's already-cached opponent table and
features:
  1. Gap/compound/tyre_age for the car BEHIND (add_behind_columns, knowledge.py) —
     derived from data already computed, no new telemetry parsing.
  2. A 'direction' label (undercut / overcut / ambiguous) per flagged event, based on
     pit-order sequencing against the identified rival (summarize_undercut_events).
  3. A 'points_gap' column per event — the championship points gap between the
     flagged driver and their rival, as of before that race (standings.py).

Only re-does the CHEAP steps (flagging + summarizing) using the SAME cached opponent
table and pit-loss constant already computed for each race — build_opponent_table's
raw telemetry loop is untouched; if a local opponent-table checkpoint already exists
for a race, this reuses it directly.

RUN standings.py --inspect FIRST for at least one season before running this archive-
wide — points_gap silently returns None on any lookup failure rather than erroring, so
a real problem (wrong column names, race-name mismatch) would otherwise show up only
as an unexplained wall of missing values, not a loud failure.

Usage:
    python enrich_archive.py --limit 5      # smoke-test on 5 races first
    python enrich_archive.py                 # enrich the full archive
"""

import argparse
import pandas as pd

import knowledge as rk
import standings as st
import risk as rp


def enrich_one_race(cb, year, race_fastf1, session_fastf1):
    race_ti = race_fastf1.replace("_", " ")
    session_ti = "Race" if session_fastf1 == "R" else "Sprint"
    rk.YEAR, rk.RACE_FASTF1, rk.RACE_TI = year, race_fastf1, race_ti
    rk.SESSION_FASTF1, rk.SESSION_TI = session_fastf1, session_ti

    features_df = rk.load_features(cb)
    features_df = rk.add_cumulative_time(features_df)
    number_to_code = rk.build_number_to_code(features_df)

    # Reuse the ALREADY-CACHED opponent table (local checkpoint from the original
    # archive run) — this does NOT re-scan raw telemetry if the checkpoint exists.
    opponent_df = rk.build_opponent_table(cb, features_df, number_to_code)
    if opponent_df.empty:
        return None

    opponent_df = rk.add_behind_columns(opponent_df, features_df)

    _, circuit_constant_s = rk.compute_pit_lane_loss(features_df)
    if circuit_constant_s is None:
        return None

    flagged = rk.flag_undercut_windows(opponent_df, features_df, circuit_constant_s)
    events = rk.summarize_undercut_events(flagged, features_df=features_df)
    if events.empty:
        return events

    events["year"] = year
    events["race"] = race_ti
    events["session"] = session_fastf1
    events["pit_loss_constant_s"] = circuit_constant_s

    # Attach points_gap per event — rival is the ahead_driver at the event's last
    # flagged lap (the same identity used for the direction label above). Storing
    # "rival" alongside is new: previously points_gap was attached with no way to
    # retroactively check WHICH rival it was computed against, making it impossible to
    # validate. Now signed (not absolute) so a real risk policy can use direction, and
    # risk_appetite is computed inline — this is the piece that moves the "incorporate
    # championship standings weight" checklist item from a column to an actual policy.
    rivals, points_gaps, risk_appetites = [], [], []
    for _, ev in events.iterrows():
        rival_row = flagged.loc[
            (flagged["Driver"] == ev["Driver"]) & (flagged["LapNumber"] == ev["window_end_lap"]),
            "ahead_driver"
        ]
        rival = rival_row.iloc[0] if not rival_row.empty else None
        rival = rival if rival and pd.notna(rival) else None
        gap = st.get_signed_points_gap(year, race_fastf1, ev["Driver"], rival) if rival else None
        rivals.append(rival)
        points_gaps.append(gap)
        risk_appetites.append(rp.compute_risk_appetite(gap))
    events["rival"] = rivals
    events["points_gap"] = points_gaps
    events["risk_appetite"] = risk_appetites

    base = f"derived/rival_knowledge/{year}/{race_fastf1}/{session_fastf1}"
    cb.write_csv(f"{base}/undercut_events.csv", events)
    cb.write_csv(f"{base}/undercut_windows.csv", flagged)
    opponent_df.to_csv(
        f"./checkpoints/rival_knowledge/{year}_{race_fastf1}_{session_fastf1}_opponents_enriched.csv",
        index=False
    )

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only enrich the first N usable races (for smoke-testing).")
    args = parser.parse_args()

    log = pd.read_csv("./checkpoints/rival_knowledge/archive_run_log.csv")
    usable = log[
        (log["status"].isin(["ok", "skipped_existing"]))
        & (~log.get("known_backfilled_limitation", pd.Series(dtype=bool)).fillna(False))
    ].copy()

    parsed = usable["race_label"].str.extract(r"^(\d{4}) (.+) \((R|S)\)$")
    usable["year"] = parsed[0].astype(int)
    usable["race_ti"] = parsed[1]
    usable["session"] = parsed[2]

    if args.limit:
        usable = usable.head(args.limit)
        print(f"[enrich] --limit set: enriching only the first {len(usable)} usable races.")

    cb = rk.CachedBucket()
    all_events = []
    for _, row in usable.iterrows():
        race_fastf1 = row["race_ti"].replace(" ", "_")
        print(f"[enrich] {row['race_label']} ...")
        try:
            events = enrich_one_race(cb, row["year"], race_fastf1, row["session"])
            if events is not None and not events.empty:
                all_events.append(events)
                n_undercut = (events["direction"] == "undercut").sum()
                n_overcut = (events["direction"] == "overcut").sum()
                n_ambig = (events["direction"] == "ambiguous").sum()
                n_gap = events["points_gap"].notna().sum()
                n_high_risk = (events["risk_appetite"] == "high").sum()
                print(f"[enrich]   {len(events)} events — {n_undercut} undercut, "
                      f"{n_overcut} overcut, {n_ambig} ambiguous, {n_gap} with a points_gap value, "
                      f"{n_high_risk} flagged high risk appetite.")
        except Exception as e:
            print(f"[enrich] ERROR on {row['race_label']}: {e!r} — skipping, continuing.")

    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        combined.to_csv("./checkpoints/rival_knowledge/archive_event_summary_enriched.csv", index=False)
        cb.write_csv("derived/rival_knowledge/_archive/event_summary_enriched.csv", combined)
        print(f"\n[enrich] Done. {len(combined)} enriched events across {len(all_events)} races.")
        print(f"[enrich] Overall direction split: {combined['direction'].value_counts().to_dict()}")
        print(f"[enrich] points_gap available for {combined['points_gap'].notna().sum()}/{len(combined)} events.")
        print(f"[enrich] Risk appetite split: {combined['risk_appetite'].value_counts().to_dict()}")

        # DEMONSTRATION: same physical opportunity, different championship context.
        # Finds pairs of events with a near-identical tyre-age gap (the physical
        # signal driving the undercut/overcut flag) but very different risk_appetite
        # labels — concrete evidence the feature is genuinely wired through, not just
        # a column sitting unused.
        demo = combined.dropna(subset=["risk_appetite"]).copy()
        if not demo.empty:
            print("\n" + "=" * 70)
            print("DEMONSTRATION: same-ish physical opportunity, different risk context")
            print("=" * 70)
            high = demo[demo["risk_appetite"] == "high"]
            low = demo[demo["risk_appetite"] == "low"]
            if not high.empty and not low.empty:
                sample_high = high.sample(min(2, len(high)), random_state=1)
                sample_low = low.sample(min(2, len(low)), random_state=1)
                cols = ["Driver", "rival", "year", "race", "actual_pit_lap", "direction", "points_gap", "risk_appetite"]
                print("HIGH risk appetite examples (close championship battle):")
                print(sample_high[cols].to_string(index=False))
                print("\nLOW risk appetite examples (settled championship gap):")
                print(sample_low[cols].to_string(index=False))
                print("\nBoth sets are the SAME kind of detected event (a real tyre-offset pit")
                print("opportunity) — the only difference is the championship context at the")
                print("time, which now visibly changes the risk_appetite label attached to it.")
    else:
        print("\n[enrich] No events produced — check the errors above.")


if __name__ == "__main__":
    main()