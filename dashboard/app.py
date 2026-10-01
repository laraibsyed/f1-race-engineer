""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hermes_adapter as ha
import data_loader as dl
import track_map as tmap
import ui_components as ui

def _stretch(fn):
    ""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}

BTN_W, CHART_W = _stretch(st.button), _stretch(st.plotly_chart)

st.set_page_config(page_title="HERMES Pit Wall", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');
:root{
  --bg:#05070a; --panel:#0b0e13; --panel2:#10141b; --line:#1c212b; --text:#e8ecf1; --dim:#6b7280;
  --red:#ff1e1e; --blue:#1e90ff; --yellow:#ffd23b; --cyan:#00e5ff; --green:#19d97a; --amber:#ffb020;
  --ui:"Titillium Web","Bahnschrift","Segoe UI",sans-serif;
  --num:"Consolas","SF Mono","Courier New",monospace;
  --col-top: 200px;        /* where the 3 main columns start (header + controls have FIXED heights below) */
  --safe-bottom: 64px;     /* free space kept at the bottom. In browser FULLSCREEN the Windows taskbar sits on top of
                              the page while 100vh still counts the hidden part - raise if cropped, 0px if windowed. */
  --col-h: calc(100vh - var(--col-top) - var(--safe-bottom));       /* height of the 3 main columns */
  --tt-h: clamp(15px, calc((100vh - var(--col-top) - 170px - var(--safe-bottom)) / 20), 25px);  /* tower row height */
}
/* ---- LOCK THE PAGE to the window: header, tower, track + charts never scroll away ---- */
html, body{ overflow:hidden !important; height:100vh; }
[data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main{
  overflow:clip !important; height:100vh; }
[data-testid="stMainBlockContainer"], .block-container{
  box-sizing:border-box !important; width:100% !important; max-width:100% !important; min-width:0 !important; }
[data-testid="stHorizontalBlock"]{ min-width:0 !important; }
/* the track+charts iframe fills its column */
iframe, div:has(> iframe){ height:var(--col-h) !important; min-height:0 !important; }
/* ONLY the strategy column scrolls */
div[data-testid="stColumn"]:has(.strat-anchor), div[data-testid="column"]:has(.strat-anchor){
  height:var(--col-h); overflow-y:auto; overflow-x:hidden; padding-right:8px; }
div[data-testid="stColumn"]:has(.strat-anchor)::-webkit-scrollbar{ width:6px; }
div[data-testid="stColumn"]:has(.strat-anchor)::-webkit-scrollbar-thumb{ background:#2a3040; border-radius:3px; }
div[data-testid="stColumn"]:has(.strat-anchor)::-webkit-scrollbar-thumb:hover{ background:var(--cyan); }
.strat-anchor{ height:0; }
/* ---- anti-flicker: Streamlit greys out + fades every element while a rerun is in flight ---- */
[data-stale="true"], .stale-element, .element-container[data-stale="true"]{ opacity:1 !important; }
[data-testid="stStatusWidget"]{ display:none !important; }
*{ transition:none !important; }

.stApp{ background:var(--bg) !important; }
html, body, [class*="css"], .stMarkdown, button, input{ font-family:var(--ui) !important; }
/* keep the header element (it holds the sidebar expand arrow) but make it slim */
header[data-testid="stHeader"]{ background:var(--bg) !important; height:2.2rem; min-height:2.2rem; }
header[data-testid="stHeader"] *{ color:var(--text) !important; }
#MainMenu, footer, [data-testid="stDecoration"]{ visibility:hidden; height:0; }
.block-container{ padding:2.6rem 1rem 0.6rem 1rem !important; max-width:100% !important; }
/* hide ONLY Deploy / menu / host actions - NOT [data-testid="stToolbar"] itself: the sidebar's
   open-arrow lives inside it, hiding the toolbar makes a collapsed sidebar impossible to reopen */
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stMainMenu"],
[data-testid="stToolbarActions"]{ display:none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"]{
  display:flex !important; visibility:visible !important; opacity:1 !important; }
[data-testid="stExpandSidebarButton"] *{ color:var(--cyan) !important; }
div[data-testid="stVerticalBlock"]{ gap:0.3rem !important; }
.stPlotlyChart{ margin:0 !important; }
iframe{ border:0 !important; display:block; }
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }

/* ---- panels / type ---- */
.pw-panel{ background:var(--panel); border:1px solid var(--line); padding:8px 12px; }
.pw-label{ color:var(--dim); font-size:0.62rem; letter-spacing:0.09em; text-transform:uppercase; font-weight:600; }
.pw-title{ color:var(--text); font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700;
           border-bottom:1px solid var(--line); padding-bottom:3px; margin-bottom:4px; }
.pw-big{ font-size:1.5rem; font-weight:900; letter-spacing:0.01em; line-height:1.1; }
.pw-num{ font-family:var(--num) !important; }
.pw-dim{ color:var(--dim); font-size:0.7rem; }
.pw-driver-d1{ color:var(--red); font-weight:900; }
.pw-driver-d2{ color:var(--blue); font-weight:900; }
.badge-hist{ color:var(--green); font-weight:700; font-size:0.8rem; }
.badge-scenario{ color:var(--amber); font-weight:800; font-size:0.82rem; animation:pulse 1.6s infinite; }
@keyframes pulse{ 0%{opacity:1;} 50%{opacity:0.55;} 100%{opacity:1;} }
.dec-PIT_NOW, .dec-PIT_LAP{ color:var(--red); }
.dec-PIT_FLEXIBLE, .dec-PIT_LATER{ color:var(--amber); }
.dec-DONT_PIT, .dec-STAY_OUT{ color:var(--green); }

/* ---- top bar ---- */
.pw-top{ display:flex; align-items:stretch; height:80px; box-sizing:border-box; overflow:hidden;
         background:var(--panel); border:1px solid var(--line); margin-bottom:10px; }
.pw-cell{ padding:6px 20px; border-right:1px solid var(--line); min-width:96px;
           display:flex; flex-direction:column; justify-content:center; }
.pw-cell:last-child{ border-right:0; }
.pw-grow{ flex:1; }
.pw-mode{ min-width:150px; }
.pw-val{ font-size:1.3rem; font-weight:900; line-height:1.15; letter-spacing:0.01em; }
.pw-race{ font-size:1.45rem; }
.pw-label{ margin-bottom:2px; }

/* ---- timing tower ---- */
.tt{ background:var(--panel); border:1px solid var(--line); }
.tt-row{ display:grid; grid-template-columns:24px 4px 38px 1fr 62px 22px 26px; align-items:center; gap:6px;
         padding:0 8px 0 4px; height:var(--tt-h); border-bottom:1px solid #12161d; font-size:clamp(0.62rem, 1.75vh, 0.8rem); }
.tt-head span{ white-space:nowrap; overflow:visible; }
.tt-head{ height:20px; color:var(--dim); font-size:0.58rem; letter-spacing:0.09em; font-weight:700; background:var(--panel2); }
.tt-pos{ text-align:right; color:var(--dim); font-family:var(--num); }
.tt-bar{ width:4px; height:65%; }
.tt-drv{ font-weight:900; letter-spacing:0.03em; }
.tt-gap, .tt-last, .tt-age{ font-family:var(--num); text-align:right; font-size:0.74rem; }
.tt-last{ color:#aeb6c2; }
.tt-age{ color:var(--dim); }
.tt-tyre{ width:17px; height:17px; border:2px solid; border-radius:50%; font-size:0.58rem; font-weight:900;
          display:flex; align-items:center; justify-content:center; box-sizing:border-box; }
.tt-d1{ background:rgba(255,30,30,0.14); } .tt-d1 .tt-drv{ color:var(--red); }
.tt-d2{ background:rgba(30,144,255,0.14); } .tt-d2 .tt-drv{ color:var(--blue); }

/* ---- HERMES strategy cards ---- */
.sc-card{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--line);
          padding:8px 12px; margin-bottom:4px; }
.sc-card.d1{ border-left-color:var(--red); } .sc-card.d2{ border-left-color:var(--blue); }

/* ---- controls ---- */
.stButton button{ background:#11151c; border:1px solid #2a3040; color:var(--text); font-size:0.74rem;
                  padding:2px 6px; border-radius:2px; min-height:2rem; }
.stButton button:hover{ border-color:var(--cyan); color:var(--cyan); }
.stSlider{ padding-top:2px; }
div[data-testid="stSlider"]{ padding-top:16px; }
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"]{ display:none; }
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSlider"]){ height:52px; max-height:52px; align-items:flex-start; }
div[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:0; background:var(--panel); }

/* ---- sidebar: compact + properly spaced ---- */
[data-testid="stSidebarUserContent"], [data-testid="stSidebarContent"]{ padding-top:0.6rem; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{ gap:0.7rem !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{
  font-size:0.66rem; letter-spacing:0.09em; text-transform:uppercase; color:var(--dim); font-weight:600; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{ min-height:34px; font-size:0.86rem;
  background:#11151c; border:1px solid #2a3040; border-radius:2px; }
section[data-testid="stSidebar"] .pw-title{ margin-bottom:6px; padding-bottom:6px; }
section[data-testid="stSidebar"] hr{ margin:0.2rem 0; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{ letter-spacing:0.09em; font-size:0.62rem; }
.pw-pair{ display:flex; align-items:baseline; gap:10px; padding:3px 0; font-size:0.95rem; font-weight:900; letter-spacing:0.03em; }
.pw-pair .pw-tag{ font-size:0.7rem; min-width:20px; }
div[data-testid="stSpinner"]{ font-size:0.72rem; color:var(--dim); }
</style>
""", unsafe_allow_html=True)

def _init_state():
    defaults = dict(
        race_loaded=False, season=None, race=None, session="R", d1=None, d2=None,
        current_lap=1, playing=False, replay_cache=None, dpc=None, bundle=None, telemetry=None,
        pair_view=None, _last_adv=0.0,
        scenario_active=False, scenario_kind=None, scenario_lap=None,
        scenario_duration=None, scenario_cache=None, weather_scenario_active=False,
        weather_kind=None, weather_lap=None, weather_cache=None,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init_state()

RED, BLUE = "#ff1e1e", "#1e90ff"
PLOT_CFG = {"displayModeBar": False, "staticPlot": True}

with st.sidebar:
    st.markdown("<div class='pw-title'>RED BULL PIT WALL</div>", unsafe_allow_html=True)
    seasons = dl.cached_list_seasons()
    _default_season_idx = seasons.index(2023) if 2023 in seasons else (0 if seasons else None)
    season = st.selectbox("Season", seasons, index=_default_season_idx)
    races = dl.cached_list_races(season) if season else []
    race = st.selectbox("Race", races, format_func=lambda r: r.replace("_", " "))
    session_code = "R"

    rb_pair, rb_error = None, None
    if race:
        try:
            rb_pair = ha.resolve_red_bull_pair(season, race)
        except Exception as e:
            rb_error = str(e)

    if rb_pair:
        st.markdown(
            "<div class='pw-panel' style='padding:8px 12px'>"
            f"<div class='pw-label' style='margin-bottom:4px'>Red Bull pairing &middot; {season}</div>"
            f"<div class='pw-pair'><span class='pw-tag pw-driver-d1'>D1</span><span>{rb_pair[0]}</span></div>"
            f"<div class='pw-pair'><span class='pw-tag pw-driver-d2'>D2</span><span>{rb_pair[1]}</span></div>"
            "</div>", unsafe_allow_html=True)
    elif rb_error:
        st.error(f"No Red Bull pairing recorded for {season}: {rb_error}")

    load_clicked = st.button("LOAD RACE", **BTN_W, type="primary", disabled=not rb_pair)

    if load_clicked and rb_pair:
        d1, d2 = rb_pair
        bundle = dl.cached_race_bundle(season, race, session_code)
        dpc = ha.build_driver_pair_context(bundle, d1, d2)
        telemetry = tmap.load_telemetry(season, race, session_code)
        st.session_state.update(
            race_loaded=True, season=season, race=race, session=session_code, d1=d1, d2=d2,
            bundle=bundle, dpc=dpc, replay_cache=ha.ReplayCache(bundle, dpc), telemetry=telemetry,
            pair_view=dl.prepare_pair_view(bundle, d1, d2),
            current_lap=1, playing=False, scenario_active=False, weather_scenario_active=False,
        )
        st.rerun()

    if st.session_state.race_loaded:
        st.select_slider("Playback: seconds per lap", options=[2, 3, 4, 6, 8], value=4, key="lap_secs")
        st.markdown("---")
        st.caption("DEMO SCENARIO CONTROLS")
        with st.expander("Safety Car / VSC"):
            sc_kind = st.radio("Type", ["OFF", "SC", "VSC"], horizontal=True, key="sc_kind_radio")
            sc_duration = st.select_slider("Duration (laps)", options=[1, 2, 3, 5], value=2, key="sc_dur")
            if st.button("APPLY SC/VSC", **BTN_W):
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
        with st.expander("Weather"):
            w_kind = st.selectbox("Override", ["NONE"] + list(ha.WEATHER_PRESETS.keys()), key="w_kind_select")
            w_delay = st.select_slider("Starts in (laps)", options=[0, 1, 2, 3, 5], value=0, key="w_delay")
            if st.button("APPLY WEATHER", **BTN_W):
                if w_kind == "NONE":
                    st.session_state.weather_scenario_active = False
                    st.session_state.weather_cache = None
                else:
                    overrides = ha.WEATHER_PRESETS[w_kind]
                    lap_at = st.session_state.current_lap + w_delay
                    st.session_state.weather_scenario_active = True
                    st.session_state.weather_kind = w_kind
                    st.session_state.weather_lap = lap_at
                    st.session_state.weather_cache = ha.ScenarioReplayCache(
                        st.session_state.replay_cache, lap_at, overrides, None)
                st.rerun()
        col_a, col_b = st.columns(2)
        if col_a.button("RESET SCENARIO", **BTN_W):
            st.session_state.scenario_active = False
            st.session_state.scenario_cache = None
            st.session_state.weather_scenario_active = False
            st.session_state.weather_cache = None
            st.rerun()
        if col_b.button("RESET RACE", **BTN_W):
            bundle = st.session_state.bundle
            dpc = ha.build_driver_pair_context(bundle, st.session_state.d1, st.session_state.d2)
            st.session_state.update(dpc=dpc, replay_cache=ha.ReplayCache(bundle, dpc), current_lap=1,
                                     playing=False, scenario_active=False, scenario_cache=None,
                                     weather_scenario_active=False, weather_cache=None)
            st.rerun()
        with st.expander("Demo presets"):
            demo = st.radio("Preset", ["NORMAL", "SAFETY CAR", "VSC", "RAIN", "DRYING TRACK"], key="demo_preset")
            if st.button("RUN PRESET", **BTN_W):
                lap = st.session_state.current_lap
                if demo == "NORMAL":
                    st.session_state.scenario_active = False
                    st.session_state.weather_scenario_active = False
                elif demo in ("SAFETY CAR", "VSC"):
                    overrides = dict(TrackStatus=ha.TRACK_STATUS_SC if demo == "SAFETY CAR" else ha.TRACK_STATUS_VSC,
                                      is_sc_lap=(demo == "SAFETY CAR"), is_vsc_lap=(demo == "VSC"))
                    st.session_state.scenario_active = True
                    st.session_state.scenario_kind = "SC" if demo == "SAFETY CAR" else "VSC"
                    st.session_state.scenario_lap = lap
                    st.session_state.scenario_duration = 3
                    st.session_state.scenario_cache = ha.ScenarioReplayCache(st.session_state.replay_cache, lap, overrides, 3)
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

if not st.session_state.race_loaded:
    st.markdown("<div class='pw-title' style='font-size:1.1rem'>HERMES PIT WALL</div>", unsafe_allow_html=True)
    st.info("Open the sidebar, pick a season and race, then LOAD RACE. D1/D2 are always the Red Bull pairing.")
    st.stop()

LAP_SECS = float(st.session_state.get("lap_secs", 4))
_DEFINED_PLAYING = bool(st.session_state.playing)

def _set_lap(lap: int):
    ss = st.session_state
    ss.current_lap = int(min(max(1, lap), ss.bundle.total_laps))
    ss._last_adv = time.time()

def _step(delta: int):
    _set_lap(st.session_state.current_lap + delta)

def _on_slider():
    _set_lap(st.session_state.lap_slider)

def _toggle_play():
    ss = st.session_state
    ss.playing = not ss.playing
    ss._last_adv = time.time()

def _reset_lap():
    st.session_state.playing = False
    _set_lap(1)

def _chart_payload(pv, d1, d2, current_lap, total_laps):
    ""
    def series(col, with_dots):
        out = []
        for code, color in ((d1, RED), (d2, BLUE)):
            d = pv["driver_laps"].get(code)
            if d is None or col not in d.columns:
                continue
            d = d[d["LapNumber"] <= current_lap].dropna(subset=[col])
            if d.empty:
                continue
            item = dict(c=color, pts=[[int(a), round(float(b), 3)] for a, b in zip(d["LapNumber"], d[col])])
            if with_dots and "Compound" in d.columns:
                item["dots"] = [ui.compound_color(x) for x in d["Compound"]]
            out.append(item)
        return out
    return dict(total=int(total_laps), lap=int(current_lap),
                tyre=dict(range=pv["ranges"].get("lap_time"), series=series("LapTime_seconds", True)),
                gap=dict(range=pv["ranges"].get("gap"), series=series("gap_to_leader", False)))

def _now_line(fig, lap):
    fig.add_shape(type="line", x0=lap, x1=lap, y0=0, y1=1, yref="paper",
                  line=dict(color="#19d97a", width=1, dash="dot"))

@st.fragment(run_every=LAP_SECS if _DEFINED_PLAYING else None)
def pit_wall():
    ss = st.session_state
    bundle = ss.bundle
    total_laps = bundle.total_laps
    d1, d2 = ss.d1, ss.d2
    replay_cache = ss.replay_cache

    if ss.playing and time.time() - ss._last_adv >= LAP_SECS * 0.9:
        if ss.current_lap < total_laps:
            ss.current_lap += 1
            ss._last_adv = time.time()
        else:
            ss.playing = False

    if ss.playing != _DEFINED_PLAYING:
        st.rerun(scope="app")

    current_lap = ss.current_lap

    sc_window_active = (
        ss.scenario_active and ss.scenario_lap <= current_lap
        and (ss.scenario_duration is None or current_lap <= ss.scenario_lap + ss.scenario_duration - 1))
    weather_window_active = ss.weather_scenario_active and current_lap >= ss.weather_lap
    any_scenario = sc_window_active or weather_window_active

    if ss.scenario_active and current_lap >= ss.scenario_lap:
        active_cache = ss.scenario_cache
    elif ss.weather_scenario_active and current_lap >= ss.weather_lap:
        active_cache = ss.weather_cache
    else:
        active_cache = replay_cache

    decisions = active_cache.get(current_lap) or {}
    hist_decisions = replay_cache.get(current_lap) or {}
    grid_df = ha.full_grid_for_lap(bundle, current_lap)
    weather = ha.weather_for_lap(bundle, current_lap)

    track_status = ""
    if not grid_df.empty and "TrackStatus" in grid_df.columns:
        track_status = str(grid_df["TrackStatus"].iloc[0])
    sc_state = "SC" if "4" in track_status else ("VSC" if any(c in track_status for c in ("6", "7")) else "OFF")
    if sc_window_active:
        sc_state = ss.scenario_kind

    scen_kind = None
    if any_scenario:
        scen_kind = ss.scenario_kind if ss.scenario_active else ss.weather_kind

    st.markdown(ui.top_bar_html(bundle.season, bundle.race, current_lap, total_laps, sc_state,
                                bool(weather.get("rainfall")), weather.get("air_temp"),
                                weather.get("track_temp"), scen_kind), unsafe_allow_html=True)

    ss.lap_slider = current_lap
    c = st.columns([0.6, 0.6, 0.6, 0.6, 0.6, 0.9, 0.9, 4.0])
    c[0].button("◀", key="b_prev", **BTN_W, on_click=_step, args=(-1,))
    c[1].button("▶", key="b_next", **BTN_W, on_click=_step, args=(1,))
    c[2].button("-5", key="b_m5", **BTN_W, on_click=_step, args=(-5,))
    c[3].button("+5", key="b_p5", **BTN_W, on_click=_step, args=(5,))
    c[4].button("+10", key="b_p10", **BTN_W, on_click=_step, args=(10,))
    c[5].button("⏮ RESET", key="b_reset", **BTN_W, on_click=_reset_lap)
    c[6].button("⏸ PAUSE" if ss.playing else "▶ PLAY", key="b_play", **BTN_W,
                on_click=_toggle_play)
    c[7].slider("Lap", 1, total_laps, key="lap_slider", label_visibility="collapsed", on_change=_on_slider)

    col_tower, col_mid, col_right = st.columns([0.95, 2.2, 1.15])

    with col_tower:
        st.markdown("<div class='pw-title'>TIMING TOWER</div>", unsafe_allow_html=True)
        st.markdown(ui.timing_tower_html(grid_df, d1, d2), unsafe_allow_html=True)
        st.markdown("<div class='pw-title' style='margin-top:6px'>WEATHER</div>", unsafe_allow_html=True)
        st.markdown(ui.weather_html(weather, ss.weather_kind if ss.weather_scenario_active else None,
                                    ss.weather_lap, current_lap), unsafe_allow_html=True)

    pv = ss.pair_view or {"driver_laps": {}, "ranges": {}}
    with col_mid:
        telemetry = ss.telemetry
        window = tmap.lap_window_for(telemetry, current_lap, [d1, d2]) if telemetry is not None else None
        msg = None
        if telemetry is None:
            msg = "TRACK GEOMETRY UNAVAILABLE - no genuine FastF1 telemetry for this race/session"
        elif window is None:
            msg = "NO RECORDED TIMING WINDOW FOR THIS LAP"
        codes = grid_df["Driver"].tolist() if not grid_df.empty else [d1, d2]
        tmap.render_track_panel(telemetry, window, codes, ui.team_colors_for(grid_df), d1, d2,
                                playing=ss.playing, lap_secs=LAP_SECS,
                                charts=_chart_payload(pv, d1, d2, current_lap, total_laps), msg=msg)

    with col_right:
        st.markdown("<div class='strat-anchor'></div><div class='pw-title'>HERMES STRATEGY</div>", unsafe_allow_html=True)
        teams = {}
        if not grid_df.empty and "Team" in grid_df.columns:
            teams = dict(zip(grid_df["Driver"], grid_df["Team"]))
        for code, label, role in ((d1, "D1", "d1"), (d2, "D2", "d2")):
            team = teams.get(code, "Red Bull Racing")
            dec = decisions.get(code)
            if not dec:
                st.markdown(ui.no_decision_card_html(label, role, code, team), unsafe_allow_html=True)
                continue
            card, exec_dec, active_triggers = ui.strategy_card_html(label, role, code, team, dec)
            st.markdown(card, unsafe_allow_html=True)

            hist = hist_decisions.get(code)
            if any_scenario and hist:
                h_dec = (hist.get("execution") or {}).get("decision", hist.get("gate_decision"))
                if exec_dec != h_dec:
                    h_trig = {k for k, v in (hist.get("triggers") or {}).items() if v}
                    added, removed = set(active_triggers) - h_trig, h_trig - set(active_triggers)
                    with st.expander("WHAT CHANGED?", expanded=True):
                        st.markdown(f"HISTORICAL **{h_dec}** &rarr; SCENARIO **{exec_dec}**")
                        if added:
                            st.markdown("+ " + ", ".join(sorted(added)))
                        if removed:
                            st.markdown("- " + ", ".join(sorted(removed)))

        st.markdown("<div class='pw-title' style='margin-top:6px'>STRATEGY TIMELINE</div>", unsafe_allow_html=True)
        tl = go.Figure()
        for i, code in enumerate((d1, d2)):
            d = pv["driver_laps"].get(code)
            if d is None:
                continue
            d = d[d["LapNumber"] <= current_lap]
            if d.empty:
                continue
            cols = ([ui.compound_color(x) for x in d["Compound"]] if "Compound" in d.columns
                    else ["#5a6170"] * len(d))
            tl.add_trace(go.Scatter(x=d["LapNumber"], y=[i] * len(d), mode="markers",
                                    marker=dict(size=8, color=cols), showlegend=False))
            if "is_pit_in" in d.columns:
                pits = d.loc[d["is_pit_in"] == True, "LapNumber"]
                if len(pits):
                    tl.add_trace(go.Scatter(x=pits, y=[i] * len(pits), mode="markers", showlegend=False,
                                            marker=dict(symbol="triangle-down", size=12, color=RED)))
        tl.update_layout(height=120, plot_bgcolor="#05070a", paper_bgcolor="#05070a",
                         margin=dict(l=36, r=10, t=8, b=20), font=dict(color="#e8ecf1", size=10),
                         xaxis=dict(range=[1, total_laps], gridcolor="#1c212b", title="LAP"),
                         yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=[d1, d2],
                                    range=[-0.6, 1.6], gridcolor="#1c212b"))
        _now_line(tl, current_lap)
        st.plotly_chart(tl, **CHART_W, config=PLOT_CFG, key="chart_timeline")

pit_wall()
