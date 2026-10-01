
import fastf1
import pandas as pd

def build_race_order_lookup(seasons: list) -> dict:
    ""
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

    lookup = build_race_order_lookup([2019])
    ordered = sorted(((v, k) for k, v in lookup.items() if k[0] == 2019))
    print("[check] 2019 season order (first 5 by RoundNumber):")
    for rank, (season, race) in ordered[:5]:
        print(f"  rank={rank}  {season} {race}")
    print("[check] Abu Dhabi's rank (should be near the END of the season, not rank 0):")
    for rank, (season, race) in ordered:
        if race == "Abu_Dhabi_Grand_Prix":
            print(f"  Abu_Dhabi_Grand_Prix rank={rank} of {len(ordered)}")
