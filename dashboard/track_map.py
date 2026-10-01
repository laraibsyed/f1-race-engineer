""
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
    ""
    return race_folder_name.replace("_Grand_Prix", "").replace("_", " ").strip()

@st.cache_resource(show_spinner="Fetching track telemetry (~30s first time, cached after)...")
def load_telemetry(season: int, race_folder_name: str, session_code: str = "R") -> Optional[dict]:
    ""
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
    ""
    for code in preferred_codes:
        w = telemetry.get("lap_windows", {}).get(code, {}).get(int(lap_number))
        if w is not None and w[1] > w[0]:
            return w
    return None

_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@600;700&display=swap');
html,body{margin:0;height:100%;background:#05070a;overflow:hidden}
#root{height:100vh;display:flex;flex-direction:column;gap:6px}
.ttl{color:#e8ecf1;font:700 12.5px 'Titillium Web','Bahnschrift','Segoe UI',sans-serif;letter-spacing:.08em;
     text-transform:uppercase;border-bottom:1px solid #1c212b;padding-bottom:3px;white-space:nowrap}
.box{position:relative;flex:1 1 0;min-height:0}
canvas{position:absolute;left:0;top:0;width:100%;height:100%}
.row{display:flex;gap:14px;flex:0 0 33%;min-height:0}
.col{flex:1;display:flex;flex-direction:column;min-width:0}
</style></head><body>
<div id="root">
  <div class="ttl">Track</div>
  <div class="box"><canvas id="track"></canvas></div>
  <div class="row">
    <div class="col"><div class="ttl">Tyre / degradation &middot; lap time, dots = compound</div><div class="box"><canvas id="c1"></canvas></div></div>
    <div class="col"><div class="ttl">Pace / gap to leader</div><div class="box"><canvas id="c2"></canvas></div></div>
  </div>
</div>
<script>
const D = __DATA__;
const dpr = window.devicePixelRatio || 1;
const FONT = "Consolas, 'SF Mono', monospace";

function setup(cv){
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = Math.max(1, Math.round(W*dpr)); cv.height = Math.max(1, Math.round(H*dpr));
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, W, H};
}

/* ---------------- track ---------------- */
const tcv = document.getElementById('track'), off = document.createElement('canvas');
let tctx, TW, TH, S = 1, ox = 0, oy = 0; const hint = {};
const px = x => ox + S*x, py = y => oy - S*y;

function trace(o){
  o.beginPath();
  D.ox.forEach((x,i) => i ? o.lineTo(px(x),py(D.oy[i])) : o.moveTo(px(x),py(D.oy[i])));
}
function fitTrack(){
  const g = setup(tcv); tctx = g.ctx; TW = g.W; TH = g.H;
  for (const k in hint) delete hint[k];
  if (D.msg || !D.ox.length) return;
  const x0=Math.min(...D.ox), x1=Math.max(...D.ox), y0=Math.min(...D.oy), y1=Math.max(...D.oy);
  S = 0.9*Math.min(TW/((x1-x0)||1), TH/((y1-y0)||1));
  ox = TW/2 - S*(x0+x1)/2;  oy = TH/2 + S*(y0+y1)/2;            // y flipped: telemetry Y is "up"
  off.width = tcv.width; off.height = tcv.height;               // static layer, once per resize
  const o = off.getContext('2d'); o.setTransform(dpr,0,0,dpr,0,0);
  o.lineJoin = 'round'; o.lineCap = 'round';
  trace(o); o.strokeStyle='rgba(0,229,255,.16)'; o.lineWidth=10; o.stroke();
  trace(o); o.strokeStyle='rgba(0,229,255,.85)'; o.lineWidth=2;  o.stroke();
  const k = Math.min(4, D.ox.length-1);                          // start/finish tick
  const ax=px(D.ox[0]), ay=py(D.oy[0]), dx=px(D.ox[k])-ax, dy=py(D.oy[k])-ay, n=Math.hypot(dx,dy)||1;
  o.beginPath(); o.moveTo(ax-dy/n*9, ay+dx/n*9); o.lineTo(ax+dy/n*9, ay-dx/n*9);
  o.strokeStyle='#e8ecf1'; o.lineWidth=3; o.stroke();
}
function pos(k, t){                                              // between two consecutive RECORDED samples
  const ts=D.cars[k][0], xs=D.cars[k][1], ys=D.cars[k][2];
  let i = hint[k] || 1;
  if (ts[i-1] > t) i = 1;
  while (i < ts.length-1 && ts[i] < t) i++;
  hint[k] = i;
  const f = Math.min(1, Math.max(0, (t-ts[i-1]) / ((ts[i]-ts[i-1]) || 1)));
  return [xs[i-1] + (xs[i]-xs[i-1])*f, ys[i-1] + (ys[i]-ys[i-1])*f];
}
function drawTrack(t){
  tctx.clearRect(0,0,TW,TH);
  if (D.msg || !D.ox.length){
    tctx.fillStyle='#6b7280'; tctx.font="600 12px 'Titillium Web', sans-serif"; tctx.textAlign='center';
    tctx.fillText(D.msg || 'TRACK GEOMETRY UNAVAILABLE', TW/2, TH/2); return;
  }
  tctx.drawImage(off, 0, 0, TW, TH);
  Object.keys(D.cars).sort((a,b) => D.meta[a].r - D.meta[b].r).forEach(k => {
    const p = pos(k,t), m = D.meta[k], x = px(p[0]), y = py(p[1]);
    tctx.beginPath(); tctx.arc(x, y, m.r, 0, 6.2832);
    tctx.fillStyle = m.c; tctx.globalAlpha = m.lbl ? 1 : 0.85; tctx.fill(); tctx.globalAlpha = 1;
    tctx.lineWidth = m.lbl ? 2 : 1.2; tctx.strokeStyle = '#05070a'; tctx.stroke();
    if (m.lbl){ tctx.font = '700 11px '+FONT; tctx.textAlign = 'center'; tctx.fillStyle = '#e8ecf1'; tctx.fillText(k, x, y - m.r - 5); }
  });
}

/* ---------------- charts (static) ---------------- */
function niceStep(r){ const p=Math.pow(10,Math.floor(Math.log10(r||1))), m=r/p; return (m<1.5?1:m<3?2:m<7?5:10)*p; }
function autoRange(series){
  let lo=Infinity, hi=-Infinity;
  series.forEach(s => s.pts.forEach(p => { lo=Math.min(lo,p[1]); hi=Math.max(hi,p[1]); }));
  if (!isFinite(lo)) return [0,1];
  const pad = (hi-lo)*0.08 || 1; return [lo-pad, hi+pad];
}
function drawChart(cv, spec){
  const g = setup(cv), ctx = g.ctx, W = g.W, H = g.H;
  ctx.clearRect(0,0,W,H);
  const L=40, R=8, T=6, B=18, pw=W-L-R, ph=H-T-B;
  if (pw < 20 || ph < 20) return;
  const lo_hi = spec.range || autoRange(spec.series), lo = lo_hi[0], hi = lo_hi[1], total = D.charts.total;
  const X = v => L + (v-1)/Math.max(1,total-1)*pw, Y = v => T + (1-(v-lo)/((hi-lo)||1))*ph;
  ctx.font = '10px '+FONT; ctx.fillStyle = '#8b93a1'; ctx.strokeStyle = '#1c212b'; ctx.lineWidth = 1;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const st = niceStep((hi-lo)/4);
  for (let v = Math.ceil(lo/st)*st; v <= hi+1e-9; v += st){
    const y = Math.round(Y(v)) + .5;
    ctx.beginPath(); ctx.moveTo(L,y); ctx.lineTo(L+pw,y); ctx.stroke();
    ctx.fillText(v.toFixed(st < 1 ? 1 : 0), L-5, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const xs = total > 40 ? 10 : 5;
  for (let v = xs; v <= total; v += xs) ctx.fillText(String(v), X(v), T+ph+4);
  ctx.save(); ctx.beginPath(); ctx.rect(L,T,pw,ph); ctx.clip();          // out-of-range outlier laps are clipped
  spec.series.forEach(s => {
    ctx.beginPath(); s.pts.forEach((p,i) => i ? ctx.lineTo(X(p[0]),Y(p[1])) : ctx.moveTo(X(p[0]),Y(p[1])));
    ctx.strokeStyle = s.c; ctx.lineWidth = 1.5; ctx.stroke();
    if (s.dots) s.pts.forEach((p,i) => {
      ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, 6.2832);
      ctx.fillStyle = s.dots[i]; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = s.c; ctx.stroke();
    });
  });
  ctx.setLineDash([3,3]); ctx.strokeStyle = '#19d97a'; ctx.lineWidth = 1;   // "now" marker
  ctx.beginPath(); ctx.moveTo(X(D.charts.lap), T); ctx.lineTo(X(D.charts.lap), T+ph); ctx.stroke();
  ctx.restore();
}
function drawCharts(){ drawChart(document.getElementById('c1'), D.charts.tyre); drawChart(document.getElementById('c2'), D.charts.gap); }

/* ---------------- run ---------------- */
const t0 = performance.now(); let playing = D.playing;
function frame(now){
  const t = Math.min(D.dur, (now - t0)/1000 * D.dur / D.lap_secs);
  drawTrack(t);
  if (t < D.dur) requestAnimationFrame(frame); else playing = false;
}
function layout(){ fitTrack(); drawCharts(); if (!playing) drawTrack(D.dur/2); }
layout();
if (typeof ResizeObserver !== 'undefined') new ResizeObserver(layout).observe(document.getElementById('root'));
else addEventListener('resize', layout);
if (D.playing && !D.msg && D.ox.length) requestAnimationFrame(frame);
</script></body></html>"""

def _embed_html(html: str, height: int) -> None:
    ""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)

def render_track_panel(telemetry: Optional[dict], window: Optional[tuple], codes, colors: dict, d1: str, d2: str,
                       playing: bool, lap_secs: float, charts: dict, msg: Optional[str] = None,
                       height: int = 440) -> None:
    ""
    cars, meta, ox, oy, dur = {}, {}, [], [], 1.0
    if telemetry is not None and window is not None and msg is None:
        start, end = window
        dur = float(end - start)
        ox = [int(v) for v in telemetry["outline_x"]]
        oy = [int(v) for v in telemetry["outline_y"]]
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
                              else colors.get(code, "#5a6170"), r=8 if hero else 5, lbl=hero)
    data = dict(ox=ox, oy=oy, cars=cars, meta=meta, dur=dur, lap_secs=float(lap_secs),
                playing=bool(playing), charts=charts, msg=msg)
    _embed_html(_HTML.replace("__DATA__", json.dumps(data)), height)
