""
from __future__ import annotations

import pandas as pd
import streamlit as st

import hermes_adapter as ha

@st.cache_data(show_spinner=False)
def cached_list_seasons() -> list[int]:
    return ha.list_seasons()

@st.cache_data(show_spinner=False)
def cached_list_races(season: int) -> list[str]:
    return ha.list_races(ha.REPO_ROOT, season)

@st.cache_resource(show_spinner="Loading race data...")
def cached_race_bundle(season: int, race: str, session: str = "R") -> ha.RaceBundle:
    return ha.load_race_bundle(season, race, session)

@st.cache_data(show_spinner=False)
def cached_driver_list(season: int, race: str, session: str) -> list[dict]:
    bundle = cached_race_bundle(season, race, session)
    return ha.list_drivers(bundle)

def prepare_pair_view(bundle: ha.RaceBundle, d1: str, d2: str) -> dict:
    ""
    laps = bundle.laps
    driver_laps = {code: laps[laps["Driver"] == code].sort_values("LapNumber").reset_index(drop=True)
                   for code in (d1, d2)}
    ranges: dict = {"lap_time": None, "gap": None}

    if "LapTime_seconds" in laps.columns:
        s = pd.concat([d["LapTime_seconds"] for d in driver_laps.values()]).dropna()
        if len(s) >= 5:
            lo, hi = float(s.quantile(0.03)), float(s.quantile(0.90))
            pad = max((hi - lo) * 0.10, 0.3)
            ranges["lap_time"] = [lo - pad, hi + pad]

    if "gap_to_leader" in laps.columns:
        g = pd.concat([d["gap_to_leader"] for d in driver_laps.values()]).dropna()
        if len(g):
            ranges["gap"] = [min(0.0, float(g.min())), float(g.max()) * 1.05 + 1.0]

    return dict(driver_laps=driver_laps, ranges=ranges)
