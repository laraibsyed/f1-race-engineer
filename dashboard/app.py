"""
HERMES Pit Wall - Streamlit demo dashboard.
=============================================
Orchestration/UI layer only. All strategy decisions come from the REAL,
unmodified HERMES Gate Tree / Execution Tree via dashboard/hermes_adapter.py
(which itself only calls master.py's existing public functions). No decision
logic lives in this file.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hermes_adapter as ha
import data_loader as dl
import track_map as tmap

st.set_page_config(page_title="HERMES Pit Wall", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling - dark pit-wall look
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.pitwall-card {
    background: #171b24; border: 1px solid #2a2f3a; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
}
.pitwall-big { font-size: 2.0rem; font-weight: 800; letter-spacing: 0.02em; }
.pitwall-label { color: #8a8f98; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
.badge-historical { color: #3ddc84; font-weight: 700; }
.badge-scenario { color: #ffb020; font-weight: 700; }
.decision-PIT_NOW { color: #ff3b3b; }
.decision-PIT_FLEXIBLE, .decision-PIT_LATER { color: #ffb020; }
.decision-DONT_PIT { color: #3ddc84; }
hr { border-color: #2a2f3a; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session state init
# ============================================================================
def _init_state():
    defaults = dict(
        race_loaded=False, season=None, race=None, session="R", d1=None, d2=None,
        current_lap=1, playing=False, replay_cache=None, dpc=None, bundle=None,
        scenario_active=False, scenario_kind=None, scenario_lap=None,
        scenario_duration=None, scenario_cache=None, weather_scenario_active=False,
        weather_kind=None, weather_lap=None, weather_cache=None,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ============================================================================
# Sidebar - race/driver selection + scenario controls
# ============================================================================
with st.sidebar:
    st.markdown("### HERMES PIT WALL")
    st.caption("Race & driver selection")

    seasons = dl.cached_list_seasons()
    season = st.selectbox("Season", seasons, index=0 if seasons else None)
    races = dl.cached_list_races(season) if season else []
    race = st.selectbox("Race", races)
    session = st.selectbox("Session", ["R"], index=0, help="Race session only (this demo does not cover practice/qualifying)")

    drivers = dl.cached_driver_list(season, race, session) if race else []
    driver_codes = [d["code"] for d in drivers]
    d1 = st.selectbox("Driver 1", driver_codes, index=0 if driver_codes else None)
    d2_opts = [c for c in driver_codes if c != d1]
    d2 = st.selectbox("Driver 2", d2_opts, index=0 if d2_opts else None)

    load_clicked = st.button("LOAD RACE", use_container_width=True, type="primary")

    if load_clicked and season and race and d1 and d2:
        bundle = dl.cached_race_bundle(season, race, session)
        dpc = ha.build_driver_pair_context(bundle, d1, d2)
        st.session_state.update(
            race_loaded=True, season=season, race=race, session=session, d1=d1, d2=d2,
            bundle=bundle, dpc=dpc, replay_cache=ha.ReplayCache(bundle, dpc),
            current_lap=1, playing=False, scenario_active=False, weather_scenario_active=False,
        )
        st.rerun()

    if st.session_state.race_loaded:
        st.markdown("---")
        st.caption("DEMO SCENARIO CONTROLS")

        with st.expander("Safety Car / VSC", expanded=False):
            sc_kind = st.radio("Type", ["OFF", "SC", "VSC"], horizontal=True, key="sc_kind_radio")
            sc_duration = st.select_slider("Duration (laps)", options=[1, 2, 3, 5], value=2, key="sc_dur")
            if st.button("APPLY SC/VSC SCENARIO", use_container_width=True):
                if sc_kind == "OFF":
                    st.session_state.scenario_active = False
                    st.session_state.scenario_cache = None
                else:
                    overrides = dict(TrackStatus=ha.TRACK_STATUS_SC if sc_kind == "SC" else ha.TRACK_STATUS_VSC,
                                      is_sc_lap=(sc_kind == "SC"), is_vsc_lap=(sc_kind == "VSC"))
                    st.session_state.scenario_active = True
                    st.session_state.scenario_kind = sc_kind
                    st.session_state.scenario_lap = st.session_state.current_lap
                    st.session_state.scenario_duration = sc_duration
                    st.session_state.scenario_cache = ha.ScenarioReplayCache(
                        st.session_state.replay_cache, st.session_state.current_lap, overrides, sc_duration)
                st.rerun()

        with st.expander("Weather", expanded=False):
            w_kind = st.selectbox("Override", ["NONE"] + list(ha.WEATHER_PRESETS.keys()), key="w_kind_select")
            if st.button("APPLY WEATHER SCENARIO", use_container_width=True):
                if w_kind == "NONE":
                    st.session_state.weather_scenario_active = False
                    st.session_state.weather_cache = None
                else:
                    overrides = ha.WEATHER_PRESETS[w_kind]
                    st.session_state.weather_scenario_active = True
                    st.session_state.weather_kind = w_kind
                    st.session_state.weather_lap = st.session_state.current_lap
                    st.session_state.weather_cache = ha.ScenarioReplayCache(
                        st.session_state.replay_cache, st.session_state.current_lap, overrides, None)
                st.rerun()

        col_a, col_b = st.columns(2)
        if col_a.button("RESET SCENARIO", use_container_width=True):
            st.session_state.scenario_active = False
            st.session_state.scenario_cache = None
            st.session_state.weather_scenario_active = False
            st.session_state.weather_cache = None
            st.rerun()
        if col_b.button("RESET RACE", use_container_width=True):
            bundle = st.session_state.bundle
            dpc = ha.build_driver_pair_context(bundle, st.session_state.d1, st.session_state.d2)
            st.session_state.update(dpc=dpc, replay_cache=ha.ReplayCache(bundle, dpc), current_lap=1,
                                     playing=False, scenario_active=False, scenario_cache=None,
                                     weather_scenario_active=False, weather_cache=None)
            st.rerun()

        with st.expander("Demo Mode presets", expanded=False):
            st.caption("Quick presets - propagate through the real HERMES pipeline exactly like the manual controls above.")
            demo = st.radio("Preset", ["NORMAL RACE", "SAFETY CAR", "VSC", "RAIN", "DRYING TRACK"], key="demo_preset")
            if st.button("RUN DEMO PRESET", use_container_width=True):
                lap = st.session_state.current_lap
                if demo == "NORMAL RACE":
                    st.session_state.scenario_active = False
                    st.session_state.weather_scenario_active = False
                elif demo in ("SAFETY CAR", "VSC"):
                    overrides = dict(TrackStatus=ha.TRACK_STATUS_SC if demo == "SAFETY CAR" else ha.TRACK_STATUS_VSC,
                                      is_sc_lap=(demo == "SAFETY CAR"), is_vsc_lap=(demo == "VSC"))
                    st.session_state.scenario_active = True
                    st.session_state.scenario_kind = "SC" if demo == "SAFETY CAR" else "VSC"
                    st.session_state.scenario_lap = lap
                    st.session_state.scenario_duration = 3
                    st.session_state.scenario_cache = ha.ScenarioReplayCache(
                        st.session_state.replay_cache, lap, overrides, 3)
                elif demo == "RAIN":
                    st.session_state.weather_scenario_active = True
                    st.session_state.weather_kind = "MODERATE_RAIN"
                    st.session_state.weather_lap = lap
                    st.session_state.weather_cache = ha.ScenarioReplayCache(
                        st.session_state.replay_cache, lap, ha.WEATHER_PRESETS["MODERATE_RAIN"], None)
                elif demo == "DRYING TRACK":
                    st.session_state.weather_scenario_active = True
                    st.session_state.weather_kind = "DRYING_TRACK"
                    st.session_state.weather_lap = lap
                    st.session_state.weather_cache = ha.ScenarioReplayCache(
                        st.session_state.replay_cache, lap, ha.WEATHER_PRESETS["DRYING_TRACK"], None)
                st.rerun()


# ============================================================================
# Main pane
# ============================================================================
if not st.session_state.race_loaded:
    st.markdown("## HERMES <span style='color:#8a8f98'>AI RACE ENGINEER</span>", unsafe_allow_html=True)
    st.markdown("### PIT WALL")
    st.info("Select a season, race and two drivers in the sidebar, then click **LOAD RACE**.")
    st.stop()

bundle = st.session_state.bundle
dpc = st.session_state.dpc
replay_cache: ha.ReplayCache = st.session_state.replay_cache
total_laps = bundle.total_laps
current_lap = st.session_state.current_lap

any_scenario = st.session_state.scenario_active or st.session_state.weather_scenario_active


def _active_cache():
    """Whichever cache should drive the CURRENT lap's displayed decision -
    a scenario cache if one is active and its injection lap has been reached,
    else the plain historical cache."""
    if st.session_state.scenario_active and current_lap >= st.session_state.scenario_lap:
        return st.session_state.scenario_cache
    if st.session_state.weather_scenario_active and current_lap >= st.session_state.weather_lap:
        return st.session_state.weather_cache
    return replay_cache


active_cache = _active_cache()
decisions = active_cache.get(current_lap)
hist_decisions = replay_cache.get(current_lap)  # always computed for "WHAT CHANGED?" comparison

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(f"## HERMES <span style='color:#8a8f98;font-weight:400'>AI RACE ENGINEER — PIT WALL</span>",
                unsafe_allow_html=True)
    st.markdown(f"**{bundle.season} • {bundle.race.replace('_', ' ')}**  &nbsp;&nbsp; LAP {current_lap} / {total_laps}")
with h2:
    if any_scenario:
        kind = st.session_state.scenario_kind if st.session_state.scenario_active else st.session_state.weather_kind
        st.markdown(f"<div class='badge-scenario'>&#9888; DEMO SCENARIO ACTIVE<br><small>{kind}</small></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-historical'>&#9679; HISTORICAL REPLAY</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# Lap control bar
# ----------------------------------------------------------------------------
c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 3])
if c1.button("◀ Prev", use_container_width=True) and current_lap > 1:
    st.session_state.current_lap -= 1
    st.rerun()
if c2.button("Next ▶", use_container_width=True) and current_lap < total_laps:
    st.session_state.current_lap += 1
    st.rerun()
if c3.button("+5", use_container_width=True):
    st.session_state.current_lap = min(total_laps, current_lap + 5)
    st.rerun()
if c4.button("+10", use_container_width=True):
    st.session_state.current_lap = min(total_laps, current_lap + 10)
    st.rerun()
if c5.button("⏮ Reset", use_container_width=True):
    st.session_state.current_lap = 1
    st.session_state.playing = False
    st.rerun()
play_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
if c6.button(play_label, use_container_width=True):
    st.session_state.playing = not st.session_state.playing
    st.rerun()

new_lap = c7.slider("Lap", 1, total_laps, current_lap, label_visibility="collapsed")
if new_lap != current_lap:
    st.session_state.current_lap = new_lap
    st.rerun()

# ----------------------------------------------------------------------------
# Track map (left) + strategy panel (right)
# ----------------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.markdown("#### TRACK MAP")
    grid_df = ha.full_grid_for_lap(bundle, current_lap)
    fig, is_real_outline = tmap.render_track_map(bundle.season, bundle.race, grid_df,
                                                   st.session_state.d1, st.session_state.d2)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if is_real_outline:
        st.caption("Track outline: real corner markers (2018 season data). Car positions placed by "
                   "current race POSITION around that outline, not real per-car telemetry (none available).")
    else:
        st.caption("SCHEMATIC layout - not to scale, no real track-shape data available for this "
                   "season/race. Car positions placed by current race POSITION only.")

    st.markdown("#### TIMING TOWER")
    if not grid_df.empty:
        tt = grid_df.copy()
        tt["LapTime"] = tt.get("LapTime", pd.NA)
        cols = ["Position", "Driver", "gap_to_leader", "LapTime", "Compound", "tyre_age"]
        cols = [c for c in cols if c in tt.columns]
        tt_show = tt[cols].rename(columns={
            "Position": "POS", "Driver": "DRIVER", "gap_to_leader": "GAP",
            "LapTime": "LAST LAP", "Compound": "TYRE", "tyre_age": "AGE"})
        st.dataframe(tt_show, use_container_width=True, hide_index=True, height=360)
    else:
        st.warning(f"No data rows for lap {current_lap}.")

with right:
    st.markdown("#### HERMES STRATEGY")
    for code, label in ((st.session_state.d1, "DRIVER 1"), (st.session_state.d2, "DRIVER 2")):
        dec = decisions.get(code)
        hist = hist_decisions.get(code)
        with st.container(border=True):
            st.markdown(f"**{label} — {code}**")
            if not dec:
                st.caption("No decision row for this driver at this lap (pit stop / out of session).")
                continue
            exec_dec = (dec.get("execution") or {}).get("decision", dec.get("gate_decision", "?"))
            instr = (dec.get("execution") or {}).get("driving_instruction", "?")
            css = f"decision-{exec_dec}" if exec_dec else ""
            st.markdown(f"<span class='pitwall-big {css}'>{exec_dec}</span>", unsafe_allow_html=True)
            st.caption(f"Instruction: {instr}")

            plain = (dec.get("explanation") or {}).get("plain_text") or dec.get("reason") or "—"
            st.markdown(f"**PRIMARY REASON**  \n{plain}")

            triggers = dec.get("triggers") or {}
            active_triggers = [k for k, v in triggers.items() if v]
            if active_triggers:
                st.markdown("**ACTIVE SIGNALS**  \n" + "  \n".join(f"● {t}" for t in active_triggers))

            sc = dec.get("sc_gamble")
            if sc:
                st.markdown(f"**SC GAMBLE**  \n{sc.get('recommendation', '?')}")

            if any_scenario and hist and exec_dec != (hist.get("execution") or {}).get("decision", hist.get("gate_decision")):
                with st.expander("WHAT CHANGED?", expanded=True):
                    h_dec = (hist.get("execution") or {}).get("decision", hist.get("gate_decision"))
                    st.markdown(f"HISTORICAL: **{h_dec}** &nbsp;→&nbsp; SCENARIO: **{exec_dec}**")
                    h_trig = set(k for k, v in (hist.get("triggers") or {}).items() if v)
                    s_trig = set(active_triggers)
                    added = s_trig - h_trig
                    removed = h_trig - s_trig
                    unchanged = s_trig & h_trig
                    if added:
                        st.markdown("Changed signals (new): " + ", ".join(f"+{t}" for t in added))
                    if removed:
                        st.markdown("Changed signals (removed): " + ", ".join(f"-{t}" for t in removed))
                    if unchanged:
                        st.caption("Unchanged: " + ", ".join(unchanged))
                    st.markdown(f"Reason (scenario): {plain}")

    st.markdown("#### WEATHER")
    w = ha.weather_for_lap(bundle, current_lap)
    wc1, wc2 = st.columns(2)
    wc1.metric("Track temp", f"{w.get('track_temp', '—')}")
    wc2.metric("Air temp", f"{w.get('air_temp', '—')}")
    if w.get("rainfall"):
        st.markdown("Rain: **YES** (historical)")
    else:
        st.markdown("Rain: NO (historical)")
    if st.session_state.weather_scenario_active and current_lap >= st.session_state.weather_lap:
        st.markdown(f"<span class='badge-scenario'>&#9888; SIMULATED WEATHER: {st.session_state.weather_kind}</span>",
                    unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# Strategy timeline (spec item 11)
# ----------------------------------------------------------------------------
st.markdown("#### STRATEGY TIMELINE (up to current lap — no lookahead)")
laps_upto = bundle.laps[bundle.laps["LapNumber"] <= current_lap]
tl_cols = st.columns(2)
for col, code in zip(tl_cols, (st.session_state.d1, st.session_state.d2)):
    with col:
        d_laps = laps_upto[laps_upto["Driver"] == code].sort_values("LapNumber")
        if d_laps.empty:
            col.caption(f"{code}: no data yet")
            continue
        import plotly.graph_objects as go
        fig = go.Figure()
        compound_colors = {"SOFT": "#ff3b3b", "MEDIUM": "#ffd23b", "HARD": "#e8e8e8",
                            "INTERMEDIATE": "#3ddc84", "WET": "#2e8bff"}
        for compound, grp in d_laps.groupby("Compound"):
            fig.add_trace(go.Scatter(x=grp["LapNumber"], y=[code] * len(grp), mode="markers",
                                      marker=dict(size=8, color=compound_colors.get(compound, "#8a8f98")),
                                      name=str(compound)))
        pit_laps = d_laps[d_laps.get("is_pit_in", False) == True]["LapNumber"].tolist()
        for pl in pit_laps:
            fig.add_vline(x=pl, line_color="#ff3b3b", line_dash="dash")
        fig.update_layout(height=120, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                           font=dict(color="#e8e8e8", size=10),
                           xaxis=dict(range=[1, total_laps], gridcolor="#2a2f3a"),
                           yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# Tyre / pace analysis (spec items 12/13 - NO LOOKAHEAD: filtered to <= current_lap)
# ----------------------------------------------------------------------------
st.markdown("#### TYRE / PACE ANALYSIS (laps <= current lap only)")
pace_cols = st.columns(2)
for col, code in zip(pace_cols, (st.session_state.d1, st.session_state.d2)):
    with col:
        d_laps = laps_upto[laps_upto["Driver"] == code].sort_values("LapNumber").copy()
        if d_laps.empty or "LapTime_seconds" not in d_laps.columns:
            d_laps["LapTime_seconds"] = pd.to_timedelta(d_laps.get("LapTime"), errors="coerce").dt.total_seconds() \
                if "LapTime" in d_laps.columns else pd.NA
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_laps["LapNumber"], y=d_laps["LapTime_seconds"], mode="lines+markers",
                                  line=dict(color="#2e8bff"), name=f"{code} pace"))
        if len(d_laps) >= 3:
            fig.add_trace(go.Scatter(x=d_laps["LapNumber"], y=d_laps["LapTime_seconds"].rolling(3).mean(),
                                      mode="lines", line=dict(color="#ffb020", dash="dot"), name="3-lap rolling"))
        fig.add_vline(x=current_lap, line_color="#3ddc84", line_dash="dash")
        fig.update_layout(height=220, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           margin=dict(l=10, r=10, t=30, b=10), title=f"{code} pace (s)",
                           font=dict(color="#e8e8e8", size=10),
                           xaxis=dict(gridcolor="#2a2f3a"), yaxis=dict(gridcolor="#2a2f3a"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")
with st.expander("Data-quality notes (this race)"):
    for n in bundle.fallback_notes:
        st.caption(n)

# ----------------------------------------------------------------------------
# Auto-play
# ----------------------------------------------------------------------------
if st.session_state.playing:
    if current_lap >= total_laps:
        st.session_state.playing = False
    else:
        time.sleep(1.0)
        st.session_state.current_lap += 1
        st.rerun()
