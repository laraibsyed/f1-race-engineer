"""
HERMES Pit Wall - cached data-loading layer (spec item 21/22: RaceDataLoader).
Thin Streamlit-caching wrappers around hermes_adapter.py so switching laps
never reloads laps_features.csv / weather / model pickles, and only a "Load
Race" click (new season/race/session) re-reads from disk.
"""
from __future__ import annotations

from pathlib import Path

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
