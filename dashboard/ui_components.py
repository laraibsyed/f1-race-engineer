"""
HERMES Pit Wall - HTML builders for the dense, pit-wall-style panels.
Each panel is ONE html string (one st.markdown call) instead of many Streamlit
widgets - far fewer DOM elements per rerun, no inter-widget gaps, and a look
Streamlit's own widgets can't give (team-colour bars, tyre rings, timing grid).
All text that comes from data / HERMES is html-escaped.
"""
from __future__ import annotations

import html
import math
import re
from typing import Optional

import pandas as pd

esc = html.escape

COMPOUND_COLOR = {"SOFT": "#ff1e1e", "MEDIUM": "#ffd23b", "HARD": "#e8ecf1",
                  "INTERMEDIATE": "#19d97a", "WET": "#1e90ff"}
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}

_TEAM_COLORS = [
    ("red bull", "#3671C6"), ("mercedes", "#27F4D2"), ("ferrari", "#E8002D"), ("mclaren", "#FF8000"),
    ("aston", "#229971"), ("alpine", "#FF87BC"), ("williams", "#64C4FF"), ("racing bulls", "#6692FF"),
    ("visa cash", "#6692FF"), ("alphatauri", "#6692FF"), ("toro rosso", "#6692FF"),
    ("sauber", "#52E252"), ("alfa romeo", "#C92D4B"), ("haas", "#B6BABD"), ("renault", "#FFF500"),
    ("racing point", "#F596C8"), ("force india", "#F596C8"),
]


def team_color(team) -> str:
    t = str(team or "").strip().lower()
    if t == "rb":
        return "#6692FF"
    for key, col in _TEAM_COLORS:
        if key in t:
            return col
    return "#5a6170"


def team_colors_for(grid_df: pd.DataFrame) -> dict:
    if grid_df is None or grid_df.empty or "Driver" not in grid_df.columns or "Team" not in grid_df.columns:
        return {}
    return {r["Driver"]: team_color(r["Team"]) for r in grid_df[["Driver", "Team"]].to_dict("records")}


def compound_color(c) -> str:
    return COMPOUND_COLOR.get(str(c).upper(), "#5a6170")


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def fmt_laptime(sec) -> str:
    s = _num(sec)
    if s is None:
        return "-"
    m, r = divmod(s, 60)
    return f"{int(m)}:{r:06.3f}"


def _lap_seconds(rec: dict) -> Optional[float]:
    s = _num(rec.get("LapTime_seconds"))
    if s is not None:
        return s
    lt = rec.get("LapTime")
    if lt is None:
        return None
    td = pd.to_timedelta(lt, errors="coerce")
    return None if pd.isna(td) else td.total_seconds()


# ----------------------------------------------------------------------------
def top_bar_html(season, race: str, lap: int, total: int, sc_state: str, wet: bool,
                 air, track_temp, scenario_kind: Optional[str]) -> str:
    air_v, trk_v = _num(air), _num(track_temp)
    sc_col = "var(--red)" if sc_state != "OFF" else "var(--green)"
    mode = (f"<div class='badge-scenario'>&#9670; SCENARIO ACTIVE</div><div class='pw-dim'>{esc(str(scenario_kind))}</div>"
            if scenario_kind else "<div class='badge-hist'>&#9679; HISTORICAL DATA</div>")
    temps = (f"AIR {air_v:.0f}&deg; &nbsp; TRK {trk_v:.0f}&deg;" if air_v is not None and trk_v is not None
             else (f"AIR {air_v:.0f}&deg;" if air_v is not None else "-"))
    return (
        "<div class='pw-top'>"
        "<div class='pw-cell pw-grow'><div class='pw-label'>HERMES PIT WALL</div>"
        f"<div class='pw-val pw-race'>{esc(str(season))} {esc(race.replace('_', ' ').upper())}</div></div>"
        f"<div class='pw-cell'><div class='pw-label'>LAP</div><div class='pw-val pw-num'>{lap}<span class='pw-dim'> / {total}</span></div></div>"
        f"<div class='pw-cell'><div class='pw-label'>SC/VSC</div><div class='pw-val' style='color:{sc_col}'>{esc(sc_state)}</div></div>"
        f"<div class='pw-cell'><div class='pw-label'>TRACK</div><div class='pw-val'>{'WET' if wet else 'DRY'}</div>"
        f"<div class='pw-dim pw-num' style='font-size:0.66rem'>{temps}</div></div>"
        f"<div class='pw-cell pw-mode'>{mode}</div>"
        "</div>")


def timing_tower_html(grid_df: pd.DataFrame, d1: str, d2: str) -> str:
    if grid_df is None or grid_df.empty:
        return "<div class='pw-panel pw-dim'>No rows for this lap.</div>"
    rows = ["<div class='tt'><div class='tt-row tt-head'><span>POS</span><span></span><span>DRV</span>"
            "<span>GAP</span><span>LAST</span><span>TY</span><span>AGE</span></div>"]
    # Display order = race position (then gap as tiebreak). The grid frame arrives in whatever order the
    # data is stored in, so it must be sorted here - and "LEADER" is only ever the car that IS P1/gap 0.
    def _order(r):
        p_, g_ = _num(r.get("Position")), _num(r.get("gap_to_leader"))
        return (p_ if p_ is not None else 1e9, g_ if g_ is not None else 1e9)

    for i, r in enumerate(sorted(grid_df.to_dict("records"), key=_order)):
        pos = _num(r.get("Position"))
        pos_txt = str(int(pos)) if pos else str(i + 1)
        gap = _num(r.get("gap_to_leader"))
        gap_txt = ("LEADER" if (gap == 0 or pos == 1) else (f"+{gap:.1f}" if gap is not None else "-"))
        comp = str(r.get("Compound", "?")).upper()
        age = _num(r.get("tyre_age"))
        code = str(r.get("Driver", "?"))
        cls = " tt-d1" if code == d1 else (" tt-d2" if code == d2 else "")
        rows.append(
            f"<div class='tt-row{cls}'><span class='tt-pos'>{pos_txt}</span>"
            f"<span class='tt-bar' style='background:{team_color(r.get('Team'))}'></span>"
            f"<span class='tt-drv'>{esc(code)}</span><span class='tt-gap'>{gap_txt}</span>"
            f"<span class='tt-last'>{fmt_laptime(_lap_seconds(r))}</span>"
            f"<span class='tt-tyre' style='border-color:{compound_color(comp)}'>{COMPOUND_LETTER.get(comp, '?')}</span>"
            f"<span class='tt-age'>{int(age) if age is not None else '-'}</span></div>")
    rows.append("</div>")
    return "".join(rows)


def weather_html(weather: dict, scenario_kind: Optional[str], scenario_lap: Optional[int], current_lap: int) -> str:
    wet = bool(weather.get("rainfall"))
    air, trk = _num(weather.get("air_temp")), _num(weather.get("track_temp"))
    out = ("<div class='pw-panel'><div class='pw-label'>HISTORICAL</div>"
           f"<div class='pw-big' style='font-size:1.1rem'>{'WET' if wet else 'DRY'}</div>"
           f"<div class='pw-num' style='font-size:0.78rem'>AIR {f'{air:.0f}' if air is not None else '-'}&deg;C &nbsp; "
           f"TRACK {f'{trk:.0f}' if trk is not None else '-'}&deg;C</div></div>")
    if scenario_kind and scenario_lap is not None:
        label = "ACTIVE NOW" if current_lap >= scenario_lap else f"STARTS IN {scenario_lap - current_lap} LAP(S)"
        out += ("<div class='pw-panel' style='border-color:var(--amber);margin-top:4px'>"
                "<div class='badge-scenario' style='font-size:0.68rem'>&#9670; SCENARIO WEATHER</div>"
                f"<div class='pw-num' style='font-size:0.78rem'>{esc(str(scenario_kind))} &mdash; {label}</div></div>")
    return out


# ----------------------------------------------------------------------------
def no_decision_card_html(label: str, role: str, code: str, team: str) -> str:
    return (f"<div class='sc-card {role}'><div><span class='pw-driver-{role}'>{label} {esc(code)}</span> "
            f"<span class='pw-dim'>{esc(str(team))}</span></div>"
            "<div class='pw-dim' style='margin-top:6px'>No decision row this lap.</div></div>")


def strategy_card_html(label: str, role: str, code: str, team: str, dec: dict):
    """Returns (html, exec_decision, active_trigger_names)."""
    ex = dec.get("execution") or {}
    exec_dec = ex.get("decision", dec.get("gate_decision", "?"))
    instr = ex.get("driving_instruction", "?")
    plain = (dec.get("explanation") or {}).get("plain_text") or dec.get("reason") or "-"
    css = "dec-" + re.sub(r"\W", "", str(exec_dec))

    # master.explanation_for() explains the GATE TREE's decision; the Execution Tree can still
    # adjust the headline (e.g. driver-priority tie-break: D1 gets PIT_LAP, D2 held to PIT_LATER).
    # This note appears only when they diverge - real HERMES behaviour, not a dashboard bug.
    gate_dec = dec.get("gate_decision")
    gate_note = ""
    if gate_dec and gate_dec != exec_dec:
        gate_note = (f"<div class='pw-dim' style='font-size:0.65rem;margin-top:4px'>Gate-tree call was "
                     f"{esc(str(gate_dec))}; Execution Tree adjusted to {esc(str(exec_dec))} "
                     "(driver-priority / execution rules).</div>")

    active = [k for k, v in (dec.get("triggers") or {}).items() if v]
    trig = ("<div class='pw-num' style='font-size:0.68rem;color:var(--cyan);margin-top:6px'>"
            + " &nbsp;".join(f"&#9679;{esc(str(t))}" for t in active) + "</div>") if active else ""

    sc = dec.get("sc_gamble")
    sc_html = (f"<div class='pw-dim' style='font-size:0.68rem;margin-top:6px'>SC GAMBLE: "
               f"{esc(str(sc.get('recommendation', '?')))}</div>") if sc else ""

    card = (f"<div class='sc-card {role}'>"
            f"<div><span class='pw-driver-{role}'>{label} {esc(code)}</span> <span class='pw-dim'>{esc(str(team))}</span></div>"
            f"<div style='margin-top:6px'><span class='pw-big {css}'>{esc(str(exec_dec))}</span> "
            f"<span class='pw-dim pw-num'>{esc(str(instr))}</span></div>"
            "<div class='pw-label' style='margin-top:10px'>WHY</div>"
            f"<div style='font-size:0.8rem;line-height:1.35;margin-top:2px'>{esc(str(plain))}</div>"
            f"{gate_note}{trig}{sc_html}</div>")
    return card, exec_dec, active