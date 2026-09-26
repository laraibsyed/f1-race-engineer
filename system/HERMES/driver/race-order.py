"""
Race order lookup: (season, race_name) -> true chronological rank.

WHY: laps.csv / results.csv carry no usable date (LapStartDate is 100% null
in every checked file; results.csv's Time column is race duration/gap, not
a calendar date). No date or round number exists anywhere in the bucket's
own files. This has been checked directly against real files, not assumed.

FIX: fastf1.get_event_schedule(year) is not a new dependency -- fastf1 is
already imported elsewhere in this project (backfill_telemetry_fastf1.py).
Its schedule carries RoundNumber and EventDate per event, in true calendar
order, for every season. This reuses that existing source rather than
inventing a calendar or requiring a new package.

Race names from the schedule (spaces, e.g. "Abu Dhabi Grand Prix") are
converted to this project's underscore convention (e.g. "Abu_Dhabi_Grand_Prix")
to match clean/features/ folder naming -- same translation style already used
elsewhere in the project (to_telemetry_race_name in align_telemetry.py, for
the reverse direction).
"""

import fastf1
import pandas as pd


def build_race_order_lookup(seasons: list) -> dict:
    """
    Returns {(season, race_name_underscored): true_round_rank} where
    true_round_rank is a globally increasing integer across all given
    seasons, ordered by season then RoundNumber (FastF1's own calendar
    order within a season).
    """
    rows = []
    for year in seasons:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            race_name = event["EventName"].replace(" ", "_")
            rows.append({
                "season": year,
                "race_name": race_name,
                "round_number": int(event["RoundNumber"]),
            })

    order_df = pd.DataFrame(rows).sort_values(["season", "round_number"]).reset_index(drop=True)
    order_df["global_rank"] = range(len(order_df))

    lookup = {
        (row["season"], row["race_name"]): row["global_rank"]
        for _, row in order_df.iterrows()
    }
    return lookup


if __name__ == "__main__":
    # Quick sanity check: confirm RoundNumber actually produces calendar
    # order, not alphabetical -- e.g. Abu Dhabi should NOT be race 1 of a
    # season just because it starts with "A".
    lookup = build_race_order_lookup([2019])
    ordered = sorted(((v, k) for k, v in lookup.items() if k[0] == 2019))
    print("[check] 2019 season order (first 5 by RoundNumber):")
    for rank, (season, race) in ordered[:5]:
        print(f"  rank={rank}  {season} {race}")
    print("[check] Abu Dhabi's rank (should be near the END of the season, not rank 0):")
    for rank, (season, race) in ordered:
        if race == "Abu_Dhabi_Grand_Prix":
            print(f"  Abu_Dhabi_Grand_Prix rank={rank} of {len(ordered)}")