""
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

def no_decision_card_html(label: str, role: str, code: str, team: str) -> str:
    return (f"<div class='sc-card {role}'><div><span class='pw-driver-{role}'>{label} {esc(code)}</span> "
            f"<span class='pw-dim'>{esc(str(team))}</span></div>"
            "<div class='pw-dim' style='margin-top:6px'>No decision row this lap.</div></div>")

def strategy_card_html(label: str, role: str, code: str, team: str, dec: dict):
    ""
    ex = dec.get("execution") or {}
    exec_dec = ex.get("decision", dec.get("gate_decision", "?"))
    instr = ex.get("driving_instruction", "?")
    plain = (dec.get("explanation") or {}).get("plain_text") or dec.get("reason") or "-"
    css = "dec-" + re.sub(r"\W", "", str(exec_dec))

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

    override_html = ""
    ts_exec = dec.get("team_strategy_execution")
    if ts_exec:
        is_override = bool(ts_exec.get("override"))
        status_col = "var(--amber)" if is_override else "var(--green)"
        status_txt = "OVERRIDE" if is_override else "FOLLOWED"
        override_html = (
            "<div class='pw-panel' style='margin-top:8px;padding:6px 8px'>"
            "<div class='pw-label'>TEAM STRATEGY vs EXECUTION</div>"
            f"<div style='font-size:0.78rem;margin-top:3px'>TEAM STRATEGY: <b>{esc(str(ts_exec.get('team_strategy_action','?')))}</b>"
            f" &nbsp; EXECUTION: <b>{esc(str(ts_exec.get('executed_action','?')))}</b></div>"
            f"<div class='pw-num' style='font-size:0.68rem;margin-top:3px;color:{status_col};font-weight:700'>"
            f"STATUS: {status_txt}</div>")
        if is_override and ts_exec.get("override_reason"):
            override_html += (f"<div class='pw-dim' style='font-size:0.66rem;margin-top:3px'>"
                              f"REASON: {esc(str(ts_exec['override_reason']))}</div>")
        override_html += "</div>"

    card = (f"<div class='sc-card {role}'>"
            f"<div><span class='pw-driver-{role}'>{label} {esc(code)}</span> <span class='pw-dim'>{esc(str(team))}</span></div>"
            f"<div style='margin-top:6px'><span class='pw-big {css}'>{esc(str(exec_dec))}</span> "
            f"<span class='pw-dim pw-num'>{esc(str(instr))}</span></div>"
            "<div class='pw-label' style='margin-top:10px'>WHY</div>"
            f"<div style='font-size:0.8rem;line-height:1.35;margin-top:2px'>{esc(str(plain))}</div>"
            f"{gate_note}{trig}{sc_html}{override_html}</div>")
    return card, exec_dec, active

RISK_MODE_DESCRIPTIONS = {
    "CONSERVATIVE": "Lower tolerance for strategic downside / variance.",
    "BALANCED": "Default trade-off between expected gain and downside risk.",
    "AGGRESSIVE": "Greater tolerance for strategic variance in exchange for potentially larger gains.",
}

def risk_mode_context_html(risk_mode: str) -> str:
    desc = RISK_MODE_DESCRIPTIONS.get(risk_mode, "")
    return (f"<div class='pw-panel' style='padding:6px 8px;margin-top:4px'>"
            f"<div class='pw-label'>RISK MODE &middot; STRATEGIC, NOT DRIVING STYLE</div>"
            f"<div class='pw-big' style='font-size:0.95rem;margin-top:2px'>{esc(risk_mode)}</div>"
            f"<div class='pw-dim' style='font-size:0.68rem;margin-top:2px'>{esc(desc)}</div></div>")

def team_role_html(team_strategy_result: Optional[dict], d1: str, d2: str) -> str:
    if not team_strategy_result:
        return ("<div class='pw-panel' style='padding:6px 8px;margin-top:4px'>"
                "<div class='pw-label'>TEAM STRATEGY</div>"
                "<div class='pw-dim' style='font-size:0.72rem;margin-top:2px'>Unavailable this session.</div></div>")
    role = team_strategy_result.get("role") or {}
    priority = role.get("priority_driver_id")
    objective = role.get("team_objective", "?")
    reason = role.get("reason", "")
    rows = []
    for code, label, css in ((d1, "D1", "pw-driver-d1"), (d2, "D2", "pw-driver-d2")):
        tag = " &middot; <span style='color:var(--cyan)'>CHAMPIONSHIP / TEAM PRIORITY</span>" if code == priority else ""
        rows.append(f"<div class='pw-pair'><span class='{css}'>{label}</span><span>{esc(code)}{tag}</span></div>")
    return ("<div class='pw-panel' style='padding:6px 8px;margin-top:4px'>"
            "<div class='pw-label'>TEAM ROLE</div>" + "".join(rows) +
            f"<div class='pw-dim' style='font-size:0.64rem;margin-top:4px'>{esc(str(objective))} &mdash; {esc(str(reason))}</div></div>")

def _fs(v, suffix="s", sign=True) -> str:
    n = _num(v)
    if n is None:
        return "-"
    return (f"{n:+.1f}{suffix}" if sign else f"{n:.1f}{suffix}")

def trade_off_card_html(label: str, role_key: str, code: str, trade_off: Optional[dict]) -> str:
    if not trade_off:
        return ""
    d = trade_off.get(role_key) or {}
    alt = trade_off.get("alternative")
    net = trade_off.get("net_advantage_s")
    comp = trade_off.get("comparison")

    rows = [
        f"<div class='pw-label' style='margin-top:8px'>WHY THIS DECISION? &middot; {esc(label)} {esc(code)}</div>",
        f"<div style='font-size:0.82rem;margin-top:2px'><b>{esc(str(d.get('action','?')))}</b>"
        f"<span class='pw-dim'> &middot; horizon {esc(str(d.get('horizon_laps','?')))} lap(s)"
        f" &middot; pit lap {esc(str(d.get('pit_lap','?')))}</span></div>",
    ]
    cost_bits = []
    if d.get("pit_cost_s") is not None and d["pit_cost_s"] > 0:
        cost_bits.append(f"Immediate pit cost {_fs(d['pit_cost_s'], sign=False)}")
    if d.get("degradation_cost_s") is not None:
        cost_bits.append(f"Degradation {_fs(d['degradation_cost_s'], sign=False)}")
    if d.get("cliff_penalty_s"):
        cost_bits.append(f"Cliff risk {_fs(d['cliff_penalty_s'], sign=False)}")
    if d.get("dirty_air_cost_s"):
        cost_bits.append(f"Track position {_fs(d['dirty_air_cost_s'], sign=False)}")
    if d.get("rival_window_adjustment_s"):
        rv = d["rival_window_adjustment_s"]
        cost_bits.append(("Covers predicted rival window " if rv < 0 else "Rival window exposure ") + _fs(rv, sign=False))
    if cost_bits:
        rows.append("<div class='pw-num' style='font-size:0.7rem;color:var(--cyan);margin-top:4px'>"
                     + " &nbsp;&middot;&nbsp; ".join(cost_bits) + "</div>")

    if alt:
        rows.append(f"<div class='pw-dim' style='font-size:0.68rem;margin-top:6px'>ALTERNATIVE: "
                     f"{esc(str(alt.get('d1_action','?')))} / {esc(str(alt.get('d2_action','?')))} "
                     f"&middot; projected {_fs(alt.get('d1_time_delta_s'))} over {esc(str(alt.get('horizon_laps','?')))} lap(s)</div>")
    if net is not None:
        better = "SELECTED" if net >= 0 else "ALTERNATIVE"
        rows.append(f"<div style='font-size:0.78rem;margin-top:4px'><b>NET TRADE-OFF:</b> "
                     f"{_fs(net)} &nbsp;<span class='pw-dim'>({esc(better)} is cheaper under the model's own cost accounting)</span></div>")
    if comp:
        def _pair(lbl, key):
            sel, altv = comp[key]["selected"], comp[key]["alternative"]
            if sel is None and altv is None:
                return ""
            return (f"<tr><td class='pw-dim' style='font-size:0.64rem;padding-right:8px'>{esc(lbl)}</td>"
                    f"<td class='pw-num' style='font-size:0.64rem;text-align:right;padding-right:10px'>{_fs(sel, sign=False) if sel is not None else '-'}</td>"
                    f"<td class='pw-num' style='font-size:0.64rem;text-align:right'>{_fs(altv, sign=False) if altv is not None else '-'}</td></tr>")
        table_rows = (_pair("Immediate pit cost", "immediate_pit_cost_s") + _pair("Degradation cost", "degradation_cost_s")
                      + _pair("Track-position cost", "track_position_cost_s") + _pair("Rival effect", "rival_effect_s"))
        if table_rows:
            rows.append("<table style='margin-top:4px;border-collapse:collapse;width:100%'>"
                         "<tr><td></td><td class='pw-dim' style='font-size:0.6rem;text-align:right'>SELECTED</td>"
                         "<td class='pw-dim' style='font-size:0.6rem;text-align:right'>ALT.</td></tr>"
                         + table_rows + "</table>")
    r_team_txt = f"{trade_off.get('team_reward_r_team', 0):+.3f}"
    rows.append(f"<div class='pw-dim' style='font-size:0.62rem;margin-top:6px'>Team reward (R_team): "
                f"{esc(r_team_txt)} &middot; risk mode {esc(str(trade_off.get('risk_mode','?')))}"
                f" &middot; {esc(str(trade_off.get('n_alternatives_considered', 0)))} alternative(s) considered</div>")
    return "".join(rows)
