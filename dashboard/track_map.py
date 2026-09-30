"""
HERMES Pit Wall - track map rendering (spec item 6/7).

HONEST FALLBACK CHAIN (documented per the spec's "do NOT fabricate coordinates"
instruction - checked by a read-only repo inventory before writing this file):

1. Real corner X/Y markers from data/tracinginsights/2018/<Race Name>/Race/corners.json
   - genuine scraped data, but ONLY exists for the 2018 season, and is a set of
   ~15-20 corner markers, not a dense per-metre outline. Used to draw a real
   (not invented) polygon approximation of that circuit's shape for 2018 races.
2. Otherwise: NO real per-race track-shape data exists anywhere in this repo
   (confirmed by inventory - the only other candidate, gcs_cache's
   telemetry_by_lap.csv, is lap-aggregated with no X/Y/Distance columns; FastF1's
   local http-cache only covers a handful of specific races/sessions and would
   need a live API call otherwise). In that case a SCHEMATIC circular layout is
   drawn instead, clearly labelled "SCHEMATIC - not to scale", never presented as
   a real track outline.

In BOTH cases, individual car MARKER POSITIONS are placed by current race
POSITION (evenly spaced around the outline/circle), not by real per-car X/Y -
there is no broad per-lap car-position telemetry in this repo to place them by.
This is stated in the UI caption every time the map is drawn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACINGINSIGHTS_2018 = REPO_ROOT / "data" / "tracinginsights" / "2018"

# race folder name (HERMES convention, underscore) -> tracinginsights display name (spaces)
_UNDERSCORE_TO_SPACED = {}


def _spaced_name(race_folder_name: str) -> str:
    return race_folder_name.replace("_", " ")


@st.cache_data(show_spinner=False)
def load_2018_corners(race_folder_name: str) -> Optional[pd.DataFrame]:
    """Real corner markers for a 2018 race, if present. Returns None (never a
    fabricated shape) if this race/season isn't covered."""
    spaced = _spaced_name(race_folder_name)
    for session_name in ("Race", "Qualifying", "Practice 3", "Practice 2", "Practice 1"):
        p = TRACINGINSIGHTS_2018 / spaced / session_name / "corners.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                df = pd.DataFrame(data)
                if {"X", "Y"}.issubset(df.columns) and len(df) >= 3:
                    return df
            except Exception:
                continue
    return None


def _schematic_circle_points(n: int, radius: float = 100.0):
    return [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def build_outline(season: int, race_folder_name: str):
    """Returns (x_list, y_list, is_real: bool)."""
    if season == 2018:
        corners = load_2018_corners(race_folder_name)
        if corners is not None:
            xs = list(corners["X"]) + [corners["X"].iloc[0]]
            ys = list(corners["Y"]) + [corners["Y"].iloc[0]]
            return xs, ys, True
    pts = _schematic_circle_points(48)
    pts.append(pts[0])
    return [p[0] for p in pts], [p[1] for p in pts], False


def _position_to_outline_point(outline_x, outline_y, position: int, n_cars: int, is_real: bool):
    """Places a marker around the outline by RACE POSITION, evenly spaced -
    not a claim of real spatial location. The leader starts at outline index 0
    and each subsequent position is offset further around the loop, spaced out
    so cars don't overlap even when gaps are tiny."""
    n_points = max(len(outline_x) - 1, 1)
    frac = ((position - 1) / max(n_cars, 1)) % 1.0
    idx = int(frac * n_points) % n_points
    return outline_x[idx], outline_y[idx]


def render_track_map(season: int, race_folder_name: str, grid_df: pd.DataFrame,
                      d1_code: Optional[str], d2_code: Optional[str]) -> go.Figure:
    """`grid_df` = hermes_adapter.full_grid_for_lap(...) output for the CURRENT
    lap only (no lookahead - caller is responsible for that). Must contain
    Driver, Position (or has already been ranked by master._rank_by_gap_to_leader)."""
    outline_x, outline_y, is_real = build_outline(season, race_folder_name)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=outline_x, y=outline_y, mode="lines",
                              line=dict(color="#3a3f4b", width=6), hoverinfo="skip",
                              showlegend=False))

    n_cars = len(grid_df) if not grid_df.empty else 20
    if not grid_df.empty:
        grid_df = grid_df.reset_index(drop=True)
        for i, row in grid_df.iterrows():
            pos = int(row.get("Position", i + 1)) if pd.notna(row.get("Position", None)) else i + 1
            code = row.get("Driver", "?")
            x, y = _position_to_outline_point(outline_x, outline_y, pos, n_cars, is_real)
            is_tracked = code in (d1_code, d2_code)
            color = "#ff2e2e" if code == d1_code else ("#2e8bff" if code == d2_code else "#8a8f98")
            size = 22 if is_tracked else 14
            hover = (f"{code}  P{pos}<br>Compound: {row.get('Compound', '?')}<br>"
                     f"Tyre age: {row.get('tyre_age', row.get('TyreLife', '?'))}<br>"
                     f"Gap to leader: {row.get('gap_to_leader', '?')}")
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=size, color=color, line=dict(width=2, color="#0e1117")),
                text=[code], textposition="top center",
                textfont=dict(color="#e8e8e8", size=11 if is_tracked else 9),
                hovertext=[hover], hoverinfo="text", showlegend=False,
            ))

    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10), height=460,
    )
    return fig, is_real
