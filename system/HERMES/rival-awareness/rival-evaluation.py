
import pandas as pd

df = pd.read_csv("./checkpoints/rival_knowledge/archive_event_summary_enriched.csv")
df["window_length"] = df["window_end_lap"] - df["window_start_lap"] + 1

overcuts = df[(df["direction"] == "overcut") & (df["window_length"] >= 3)].copy()
overcuts = overcuts.sort_values(["year", "race"])

print("=" * 70)
print("CANDIDATE MULTI-LAP OVERCUT EVENTS (window >= 3 laps), by year")
print("=" * 70)
print(overcuts[["Driver", "year", "race", "window_start_lap", "window_end_lap",
                "actual_pit_lap", "window_length", "points_gap"]].to_string(index=False))

years_available = sorted(overcuts["year"].unique())
if len(years_available) >= 2:
    early_year = years_available[0]
    late_year = years_available[-1]
    print(f"\nSuggested pick 1 (earliest available year, {early_year}):")
    print(overcuts[overcuts["year"] == early_year].head(3).to_string(index=False))
    print(f"\nSuggested pick 2 (latest available year, {late_year}):")
    print(overcuts[overcuts["year"] == late_year].head(3).to_string(index=False))

print("\n" + "=" * 70)
print("CANDIDATE MULTI-LAP AMBIGUOUS EVENTS (window >= 3 laps)")
print("=" * 70)
ambiguous = df[(df["direction"] == "ambiguous") & (df["window_length"] >= 3)].copy()
ambiguous = ambiguous.sort_values("window_length", ascending=False)
print(ambiguous[["Driver", "year", "race", "window_start_lap", "window_end_lap",
                  "actual_pit_lap", "window_length", "points_gap"]].head(10).to_string(index=False))
