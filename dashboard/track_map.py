"""
HERMES Pit Wall - track map, GENUINE TELEMETRY ONLY (v2: browser-side animation).

What changed vs v1 and why:
  v1 re-built a full Plotly figure in Python every 0.15 s and pushed it over the
  websocket - that is what made Play slow and glitchy. v2 sends ONE lap of real
  FastF1 position samples to the browser as JSON; a <canvas> animates it with
  requestAnimationFrame (60 fps, zero Python reruns while a lap plays).

Data rules are unchanged: no invented circuit outline, no invented positions.
  * Outline  = one clean, representative lap's recorded X/Y trace (FastF1 pos_data).
  * Cars     = each driver's own recorded X/Y samples inside the lap window.
  * Display-only: translate/scale to fit the panel; the browser draws a straight line
    between two CONSECUTIVE RECORDED samples (~4-5 Hz) so motion is smooth instead of
    jumping sample-to-sample. State this in the methodology if you quote the map.

If telemetry can't be obtained, load_telemetry() returns None and the app shows
"TRACK GEOMETRY UNAVAILABLE" - never a placeholder shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
FASTF1_CACHE_DIR = REPO_ROOT / "fastf1_cache"

try:
    import fastf1
    FASTF1_CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    _FASTF1_OK = True
except Exception:
    _FASTF1_OK = False


def _gp_query(race_folder_name: str) -> str:
    """'Bahrain_Grand_Prix' -> 'Bahrain' (FastF1 fuzzy-matches GP names)."""
    return race_folder_name.replace("_Grand_Prix", "").replace("_", " ").strip()


@st.cache_resource(show_spinner="Fetching track telemetry (~30s first time, cached after)...")
def load_telemetry(season: int, race_folder_name: str, session_code: str = "R") -> Optional[dict]:
    """None -> TRACK GEOMETRY UNAVAILABLE. Otherwise:
    {outline_x, outline_y, np_tracks{code:(t_s, x, y)}, lap_windows{code:{lap:(start_s,end_s)}},
     reference_lap{driver, lap}} - all times are session seconds (floats)."""
    if not _FASTF1_OK:
        return None
    try:
        session = fastf1.get_session(season, _gp_query(race_folder_name), session_code)
        session.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception:
        return None

    laps = session.laps
    if laps is None or laps.empty or not getattr(session, "pos_data", None):
        return None

    # --- Outline: one genuine clean lap's full recorded trace ---
    clean = laps[(laps["PitInTime"].isna()) & (laps["PitOutTime"].isna()) & laps["LapTime"].notna()]
    if clean.empty:
        clean = laps[laps["LapTime"].notna()]
    if clean.empty:
        return None
    ref = clean.iloc[(clean["LapTime"] - clean["LapTime"].median()).abs().argsort().iloc[0]]
    ref_num = str(ref["DriverNumber"])
    if ref_num not in session.pos_data:
        return None
    pos = session.pos_data[ref_num]
    trace = pos[(pos["Time"] >= ref["LapStartTime"]) & (pos["Time"] <= ref["Time"])]
    trace = trace[(trace["X"] != 0) | (trace["Y"] != 0)]
    if len(trace) < 10:
        return None
    outline_x, outline_y = list(trace["X"]), list(trace["Y"])
    if outline_x[0] != outline_x[-1] or outline_y[0] != outline_y[-1]:
        outline_x.append(outline_x[0])
        outline_y.append(outline_y[0])

    # --- Dense per-driver recorded tracks (numpy, session seconds) + lap windows ---
    code_by_num = {str(r["DriverNumber"]): r["Driver"]
                   for _, r in laps[["Driver", "DriverNumber"]].drop_duplicates().iterrows()}
    np_tracks, lap_windows = {}, {}
    for num, code in code_by_num.items():
        if num not in session.pos_data:
            continue
        p = session.pos_data[num][["Time", "X", "Y"]].dropna().sort_values("Time")
        p = p[(p["X"] != 0) | (p["Y"] != 0)]
        if len(p) >= 2:
            np_tracks[code] = (p["Time"].dt.total_seconds().to_numpy(dtype=float),
                               p["X"].to_numpy(dtype=float), p["Y"].to_numpy(dtype=float))
        dl = laps[laps["DriverNumber"] == num][["LapNumber", "LapStartTime", "Time"]].dropna()
        if not dl.empty:
            lap_windows[code] = {int(r.LapNumber): (r.LapStartTime.total_seconds(), r.Time.total_seconds())
                                 for r in dl.itertuples(index=False)}
    if not np_tracks:
        return None

    return dict(outline_x=[float(v) for v in outline_x], outline_y=[float(v) for v in outline_y],
                np_tracks=np_tracks, lap_windows=lap_windows,
                reference_lap=dict(driver=code_by_num.get(ref_num, ref_num), lap=int(ref["LapNumber"])))


def lap_window_for(telemetry: dict, lap_number: int, preferred_codes) -> Optional[tuple]:
    """Real (start_s, end_s) of this lap for the first preferred driver that has one."""
    for code in preferred_codes:
        w = telemetry.get("lap_windows", {}).get(code, {}).get(int(lap_number))
        if w is not None and w[1] > w[0]:
            return w
    return None


# ----------------------------------------------------------------------------
# Browser-side canvas renderer (no Python involved while a lap plays)
# ----------------------------------------------------------------------------
_HTML = """<!doctype html><html><body style="margin:0;background:#05070a;overflow:hidden">
<canvas id="c" style="width:100%;height:100vh;display:block"></canvas>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const dpr = window.devicePixelRatio || 1, off = document.createElement('canvas');
let W, H, S, ox, oy;
const hint = {};

function fit(){
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = W*dpr; cv.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const x0=Math.min(...D.ox), x1=Math.max(...D.ox), y0=Math.min(...D.oy), y1=Math.max(...D.oy);
  S = 0.88*Math.min(W/((x1-x0)||1), H/((y1-y0)||1));
  ox = W/2 - S*(x0+x1)/2;  oy = H/2 + S*(y0+y1)/2;      // y flipped: telemetry Y is "up"
  buildOutline();
}
const px = x => ox + S*x, py = y => oy - S*y;

function trace(o){
  o.beginPath();
  D.ox.forEach((x,i) => i ? o.lineTo(px(x),py(D.oy[i])) : o.moveTo(px(x),py(D.oy[i])));
}
function buildOutline(){                       // static layer, drawn once per resize
  off.width = cv.width; off.height = cv.height;
  const o = off.getContext('2d'); o.setTransform(dpr,0,0,dpr,0,0);
  o.lineJoin = 'round'; o.lineCap = 'round';
  trace(o); o.strokeStyle='rgba(0,229,255,.16)'; o.lineWidth=10; o.stroke();   // glow
  trace(o); o.strokeStyle='rgba(0,229,255,.85)'; o.lineWidth=2;  o.stroke();   // line
  const k = Math.min(4, D.ox.length-1);                                         // start/finish tick
  const ax=px(D.ox[0]), ay=py(D.oy[0]), dx=px(D.ox[k])-ax, dy=py(D.oy[k])-ay, n=Math.hypot(dx,dy)||1;
  o.beginPath(); o.moveTo(ax-dy/n*9, ay+dx/n*9); o.lineTo(ax+dy/n*9, ay-dx/n*9);
  o.strokeStyle='#e8ecf1'; o.lineWidth=3; o.stroke();
}

function pos(k, t){                            // between two consecutive RECORDED samples
  const ts=D.cars[k][0], xs=D.cars[k][1], ys=D.cars[k][2];
  let i = hint[k] || 1;
  if (ts[i-1] > t) i = 1;
  while (i < ts.length-1 && ts[i] < t) i++;
  hint[k] = i;
  const f = Math.min(1, Math.max(0, (t-ts[i-1]) / ((ts[i]-ts[i-1]) || 1)));
  return [xs[i-1] + (xs[i]-xs[i-1])*f, ys[i-1] + (ys[i]-ys[i-1])*f];
}

function draw(t){
  ctx.clearRect(0,0,W,H);
  ctx.drawImage(off, 0, 0, W, H);
  Object.keys(D.cars).sort((a,b) => D.meta[a].r - D.meta[b].r).forEach(k => {
    const p = pos(k,t), m = D.meta[k], x = px(p[0]), y = py(p[1]);
    ctx.beginPath(); ctx.arc(x, y, m.r, 0, 6.2832);
    ctx.fillStyle = m.c; ctx.globalAlpha = m.lbl ? 1 : 0.85; ctx.fill(); ctx.globalAlpha = 1;
    ctx.lineWidth = m.lbl ? 2 : 1.2; ctx.strokeStyle = '#05070a'; ctx.stroke();
    if (m.lbl){
      ctx.font = '700 11px Consolas, monospace'; ctx.textAlign = 'center';
      ctx.fillStyle = '#e8ecf1'; ctx.fillText(k, x, y - m.r - 5);
    }
  });
}

const t0 = performance.now();
function frame(now){
  const t = Math.min(D.dur, (now - t0)/1000 * D.dur / D.lap_secs);
  draw(t);
  if (t < D.dur) requestAnimationFrame(frame);
}
fit();
addEventListener('resize', () => { fit(); draw(D.playing ? 0 : D.dur/2); });
if (D.playing) requestAnimationFrame(frame); else draw(D.dur/2);
</script></body></html>"""


def _embed_html(html: str, height: int) -> None:
    """st.iframe on new Streamlit (components.v1.html is deprecated there); fall back on old."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def render_live_track(telemetry: dict, window: tuple, codes, colors: dict, d1: str, d2: str,
                      playing: bool, lap_secs: float = 4.0, height: int = 340) -> None:
    """Embed one lap of genuine telemetry. While `playing`, the browser animates the lap in
    `lap_secs` wall-clock seconds; paused, it shows the lap's midpoint. The page HTML only
    changes when the lap (or play state) changes, so Streamlit does NOT reload the frame on
    unrelated reruns."""
    start, end = window
    cars, meta = {}, {}
    for code in codes:
        arr = telemetry["np_tracks"].get(code)
        if arr is None:
            continue
        t, x, y = arr
        i0, i1 = np.searchsorted(t, [start, end])
        i0, i1 = max(int(i0) - 1, 0), min(int(i1) + 1, len(t))
        if i1 - i0 < 2:
            continue
        cars[code] = [np.round(t[i0:i1] - start, 2).tolist(),
                      np.rint(x[i0:i1]).astype(int).tolist(),
                      np.rint(y[i0:i1]).astype(int).tolist()]
        hero = code in (d1, d2)
        meta[code] = dict(c="#ff1e1e" if code == d1 else "#1e90ff" if code == d2
                          else colors.get(code, "#5a6170"),
                          r=8 if hero else 5, lbl=hero)
    data = dict(ox=[int(v) for v in telemetry["outline_x"]], oy=[int(v) for v in telemetry["outline_y"]],
                cars=cars, meta=meta, dur=float(end - start), lap_secs=float(lap_secs), playing=bool(playing))
    _embed_html(_HTML.replace("__DATA__", json.dumps(data)), height)