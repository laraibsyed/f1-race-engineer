
import knowledge as rk
import pandas as pd

rk.YEAR = 2023
rk.RACE_FASTF1 = "Australian_Grand_Prix"
rk.RACE_TI = "Australian Grand Prix"
rk.SESSION_FASTF1 = "R"
rk.SESSION_TI = "Race"

cb = rk.CachedBucket()

print("Loading laps_features.csv for 2023 British GP ...")
features_df = rk.load_features(cb)
features_df = rk.add_cumulative_time(features_df)

print(f"\nColumns available: {list(features_df.columns)}")
print(f"\nis_sc_lap value counts:\n{features_df['is_sc_lap'].value_counts(dropna=False)}")
print(f"\nis_vsc_lap value counts:\n{features_df['is_vsc_lap'].value_counts(dropna=False)}")
print(f"\nis_pit_in value counts:\n{features_df['is_pit_in'].value_counts(dropna=False)}")
print(f"\nis_pit_out value counts:\n{features_df['is_pit_out'].value_counts(dropna=False)}")

for (driver, stint), g in features_df.groupby(["Driver", "Stint"]):
    pit_in_rows = g[g["is_pit_in"].fillna(False).astype(bool)]
    pit_out_rows = g[g["is_pit_out"].fillna(False).astype(bool)]
    if pit_in_rows.empty and pit_out_rows.empty:
        continue

    clean_mask = (
        (~g["is_out_lap"].astype(bool))
        & (~g["is_in_lap"].astype(bool))
        & (~g["is_pit_in"].fillna(False).astype(bool))
        & (~g["is_pit_out"].fillna(False).astype(bool))
        & (~g["is_outlier_laptime"].astype(bool))
        & (~g["is_sc_lap"].fillna(False).astype(bool))
        & (~g["is_vsc_lap"].fillna(False).astype(bool))
    )
    clean = g[clean_mask]

    print(f"\n--- {driver}, Stint {stint} ---")
    print(f"  Clean laps available for baseline: {len(clean)} (need >= {rk.MIN_CLEAN_LAPS_FOR_BASELINE})")
    if len(clean) >= rk.MIN_CLEAN_LAPS_FOR_BASELINE:
        baseline = clean["LapTime_s"].median()
        print(f"  Baseline (median clean LapTime_s): {baseline:.3f}")
        print(f"  Clean lap times used: {sorted(clean['LapTime_s'].dropna().round(2).tolist())}")
    else:
        print(f"  SKIPPED — not enough clean laps, this stint contributes nothing.")
        continue

    not_sc_vsc = (~g["is_sc_lap"].fillna(False).astype(bool)) & (~g["is_vsc_lap"].fillna(False).astype(bool))
    for _, row in pit_in_rows.iterrows():
        under_sc = row.get("is_sc_lap", False) or row.get("is_vsc_lap", False)
        print(f"  PIT_IN  lap {row['LapNumber']}: LapTime_s={row['LapTime_s']:.3f}, "
              f"under_SC/VSC={bool(under_sc)}, loss_if_counted={row['LapTime_s'] - baseline:.3f}")
    for _, row in pit_out_rows.iterrows():
        under_sc = row.get("is_sc_lap", False) or row.get("is_vsc_lap", False)
        print(f"  PIT_OUT lap {row['LapNumber']}: LapTime_s={row['LapTime_s']:.3f}, "
              f"under_SC/VSC={bool(under_sc)}, loss_if_counted={row['LapTime_s'] - baseline:.3f}")

print("\n" + "=" * 70)
print("VERIFICATION: real PitOutTime-PitInTime duration vs out-lap-based proxy")
print("=" * 70)
print("PitInTime/PitOutTime are session-elapsed timestamps recorded on the in-lap and")
print("out-lap rows respectively — PitOutTime minus PitInTime, for the SAME physical")
print("stop, is the actual time spent in the pit lane (entry to exit), independent of")
print("any lap-time-based proxy. Comparing this against the out-lap-based estimate")
print("tells us whether that estimate is a good stand-in for the real cost or just a")
print("correlated proxy that happens to move in the same direction.\n")

comparisons = []
sorted_df = features_df.sort_values(["Driver", "LapNumber"])
indexed = sorted_df.set_index(["Driver", "LapNumber"])

for _, in_row in sorted_df[sorted_df["is_pit_in"].fillna(False).astype(bool)].iterrows():
    driver = in_row["Driver"]
    in_lap = in_row["LapNumber"]
    out_lap = in_lap + 1
    if (driver, out_lap) not in indexed.index:
        continue
    out_row = indexed.loc[(driver, out_lap)]
    if isinstance(out_row, pd.DataFrame):
        out_row = out_row.iloc[0]
    if not bool(out_row.get("is_pit_out", False)):
        continue

    real_duration = None
    try:
        real_duration = (pd.to_timedelta(out_row["PitOutTime"]) - pd.to_timedelta(in_row["PitInTime"])).total_seconds()
    except Exception:
        pass

    stint_mask = (features_df["Driver"] == driver) & (features_df["Stint"] == out_row["Stint"])
    stint_laps = features_df[stint_mask]
    clean_mask = (
        (~stint_laps["is_out_lap"].astype(bool)) & (~stint_laps["is_in_lap"].astype(bool))
        & (~stint_laps["is_pit_in"].fillna(False).astype(bool)) & (~stint_laps["is_pit_out"].fillna(False).astype(bool))
        & (~stint_laps["is_outlier_laptime"].astype(bool))
        & (~stint_laps["is_sc_lap"].fillna(False).astype(bool)) & (~stint_laps["is_vsc_lap"].fillna(False).astype(bool))
    )
    clean = stint_laps[clean_mask]
    if len(clean) < 3 or pd.isna(out_row["LapTime_s"]):
        continue
    baseline = clean["LapTime_s"].median()
    proxy_loss = out_row["LapTime_s"] - baseline

    under_sc = bool(in_row.get("is_sc_lap", False) or in_row.get("is_vsc_lap", False)
                     or out_row.get("is_sc_lap", False) or out_row.get("is_vsc_lap", False))

    comparisons.append({
        "driver": driver, "in_lap": int(in_lap), "out_lap": int(out_lap),
        "real_duration_s": real_duration, "out_lap_proxy_s": proxy_loss, "under_sc_vsc": under_sc,
    })
    print(f"  {driver} lap {int(in_lap)}->{int(out_lap)}: real={real_duration}, "
          f"out-lap proxy={proxy_loss:.3f}, under_SC/VSC={under_sc}")

comp_df = pd.DataFrame(comparisons)
clean_comp = comp_df[~comp_df["under_sc_vsc"]].dropna(subset=["real_duration_s", "out_lap_proxy_s"])
if not clean_comp.empty:
    print(f"\nGreen-flag stops only (n={len(clean_comp)}):")
    print(f"  Real PitOutTime-PitInTime  — median: {clean_comp['real_duration_s'].median():.2f}s, "
          f"mean: {clean_comp['real_duration_s'].mean():.2f}s")
    print(f"  Out-lap-based proxy        — median: {clean_comp['out_lap_proxy_s'].median():.2f}s, "
          f"mean: {clean_comp['out_lap_proxy_s'].mean():.2f}s")
    diff = (clean_comp["out_lap_proxy_s"] - clean_comp["real_duration_s"])
    print(f"  Proxy minus real           — median: {diff.median():.2f}s, mean: {diff.mean():.2f}s")
    corr = clean_comp["real_duration_s"].corr(clean_comp["out_lap_proxy_s"])
    print(f"  Correlation (real vs proxy): {corr:.3f}")
    print("\nIf the proxy consistently runs higher than the real duration by a fairly")
    print("stable amount, that gap is likely just the extra track distance the driver")
    print("covers AFTER exiting the pits but before crossing the finish line — i.e. the")
    print("proxy = real pit-lane duration + partial out-lap racing distance, not the")
    print("physical pit-lane cost itself. In that case, name it 'out_lap_time_loss' or")
    print("similar in the codebase/write-up rather than 'pit_lane_loss', and consider")
    print("using the real PitOutTime-PitInTime duration as the undercut threshold")
    print("instead, since that's the number actually comparable to a rival's real gap.")
else:
    print("\nNo clean (non-SC/VSC) comparison pairs with both values available — can't verify.")
