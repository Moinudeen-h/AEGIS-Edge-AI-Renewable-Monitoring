# ============================================================
# AEGIS Dashboard — aegis_dashboard.py
# AEGIS: Autonomous Energy Grid Intelligence System
# UWE Bristol Final Year Project — 24040034
# Dashboard for monitoring Node 1 (Wind/Temp) and
# Node 2 (Solar/Light) ESP32 edge AI nodes
# Run with: streamlit run aegis_dashboard.py
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
NODE1_CSV      = "data/logs/node1_log.csv"
NODE2_CSV      = "data/logs/node2_log.csv"
THRESHOLD_N1   = 4.1968927
THRESHOLD_N2   = 1.1032283
OFFLINE_SECS   = 30
TAIL_ROWS      = 60
REFRESH_SECS   = 5

NODE1_COLS = ["timestamp_ms", "temp_C", "bus_V", "current_mA_abs", "power_mW", "mse_scaled", "decision"]
NODE2_COLS = ["timestamp_ms", "lux",    "bus_V", "current_mA_abs", "power_mW", "mse_scaled", "decision"]

# ── Colour palette ────────────────────────────
C_BG         = "#080c14"
C_SURFACE    = "#0d1220"
C_BORDER     = "#1a2540"
C_BORDER_LIT = "#243050"
C_TEXT       = "#c8d4e8"
C_TEXT_DIM   = "#4a5a78"
C_TEXT_BRT   = "#e8f0ff"

C_NORMAL     = "#00e676"
C_ANOMALY    = "#ff3d57"
C_WARN       = "#ffb300"

C_N1         = "#2979ff"
C_N1_FILL    = "rgba(41,121,255,0.08)"
C_N2         = "#ff9100"
C_N2_FILL    = "rgba(255,145,0,0.08)"

GRID_CLR     = "#1a2540"
TICK_CLR     = "#4a5a78"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS · Edge AI Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "view" not in st.session_state:
    st.session_state.view = "main"

# ─────────────────────────────────────────────
# GESTURE COMPONENT HTML
# Gestures (hold 0.9s to trigger, swipe is instant):
#   🖐 Open palm  → NODE 01
#   ☝ 1 finger   → NODE 01
#   ✌ Peace/2    → NODE 02
#   → Right swipe → NODE 02
#   ✊ Fist       → MAIN VIEW
# ─────────────────────────────────────────────
GESTURE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#080c14;font-family:'JetBrains Mono',monospace;padding:3px}
#title{font-size:7.5px;letter-spacing:.15em;color:#4a5a78;text-transform:uppercase;
       text-align:center;padding:2px 0 3px;border-bottom:1px solid #1a2540;margin-bottom:3px}
#vid-wrap{position:relative;width:100%;line-height:0}
#videoEl{width:100%;border-radius:3px;border:1px solid #1a2540;
         transform:scaleX(-1);display:block}
#canvas{position:absolute;top:0;left:0;width:100%;height:100%;
        transform:scaleX(-1);pointer-events:none}
#glabel{font-size:8.5px;letter-spacing:.1em;text-align:center;
        padding:3px 5px;margin-top:2px;background:#0d1220;
        border:1px solid #1a2540;border-radius:2px;color:#4a5a78;
        text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#pwrap{height:2px;background:#1a2540;border-radius:1px;margin-top:2px}
#pfill{height:2px;background:#2979ff;border-radius:1px;width:0;transition:width .08s linear}
#legend{display:flex;justify-content:space-between;font-size:7px;
        color:#4a5a78;margin-top:2px;letter-spacing:.06em}
</style>
</head>
<body>
<div id="title">✋ GESTURE NAV</div>
<div id="vid-wrap">
  <video id="videoEl" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
</div>
<div id="glabel">LOADING...</div>
<div id="pwrap"><div id="pfill"></div></div>
<div id="legend"><span>🖐/☝ N01</span><span>✌/→ N02</span><span>✊ MAIN</span></div>

<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
<script>
const video  = document.getElementById("videoEl");
const canvas = document.getElementById("canvas");
const ctx    = canvas.getContext("2d");
const glabel = document.getElementById("glabel");
const pfill  = document.getElementById("pfill");

const HOLD_MS        = 900;
const COOL_MS        = 1800;
const SWIPE_WINDOW   = 420;
const SWIPE_MIN_DX   = 0.07;
const SWIPE_MAX_DY   = 0.20;

let lastG        = null;
let holdStart    = null;
let triggered    = false;
let lastTrig     = 0;
let wristHistory = [];

// ── Button text must exactly match sidebar button labels ──
const BTN   = { PALM:"N01", ONE:"N01", PEACE:"N02", SWIPE:"N02", FIST:"MAIN" };
const LABEL = { PALM:"🖐 PALM → NODE 01", ONE:"☝ ONE → NODE 01",
                PEACE:"✌ PEACE → NODE 02", SWIPE:"→ SWIPE → NODE 02",
                FIST:"✊ FIST → MAIN" };
const CLR   = { PALM:"#2979ff", ONE:"#2979ff",
                PEACE:"#ff9100", SWIPE:"#ff9100",
                FIST:"#00e676" };

// Count extended fingers (index, middle, ring, pinky only — not thumb)
function countExtended(lms) {
  const tips = [8,12,16,20], pips = [6,10,14,18];
  let n = 0;
  for (let i=0;i<4;i++) if (lms[tips[i]].y < lms[pips[i]].y) n++;
  return n;
}

// Index only extended AND middle curled = one finger
function isOneFingerPoint(lms) {
  return lms[8].y < lms[6].y &&   // index extended
         lms[12].y > lms[10].y;   // middle curled
}

// Peace: index + middle extended, ring + pinky curled
function isPeace(lms) {
  return lms[8].y  < lms[6].y  &&
         lms[12].y < lms[10].y &&
         lms[16].y > lms[14].y &&
         lms[20].y > lms[18].y;
}

function detectSwipe(lms) {
  const now = Date.now(), w = lms[0];
  wristHistory.push({t:now, x:w.x, y:w.y});
  wristHistory = wristHistory.filter(p => now-p.t <= SWIPE_WINDOW);
  if (wristHistory.length < 2) return false;
  const f = wristHistory[0], l = wristHistory[wristHistory.length-1];
  return (l.x - f.x) > SWIPE_MIN_DX && Math.abs(l.y - f.y) < SWIPE_MAX_DY;
}

function detectGesture(lms) {
  if (detectSwipe(lms))    return "SWIPE";
  const ext = countExtended(lms);
  if (ext === 0)           return "FIST";
  if (isOneFingerPoint(lms)) return "ONE";
  if (isPeace(lms))        return "PEACE";
  if (ext >= 3)            return "PALM";
  return null;
}

// Click sidebar button whose exact text matches partial
function clickBtn(label) {
  try {
    const btns = window.parent.document.querySelectorAll("button");
    for (const b of btns) {
      if (b.innerText && b.innerText.trim() === label) { b.click(); return true; }
    }
  } catch(e) {}
  return false;
}

function resetHold() {
  lastG=null; holdStart=null; triggered=false; pfill.style.width="0%";
}

function onResults(res) {
  canvas.width  = video.videoWidth  || 160;
  canvas.height = video.videoHeight || 120;
  ctx.clearRect(0,0,canvas.width,canvas.height);

  if (!res.multiHandLandmarks || !res.multiHandLandmarks.length) {
    glabel.textContent = "NO HAND";
    glabel.style.color = "#4a5a78";
    wristHistory=[];
    resetHold();
    return;
  }

  const lms = res.multiHandLandmarks[0];

  // Draw landmarks
  ctx.fillStyle = "rgba(41,121,255,0.75)";
  for (const p of lms) {
    ctx.beginPath();
    ctx.arc(p.x*canvas.width, p.y*canvas.height, 2.5, 0, 2*Math.PI);
    ctx.fill();
  }
  ctx.strokeStyle="rgba(41,121,255,0.3)"; ctx.lineWidth=1;
  [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
   [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
   [0,17],[17,18],[18,19],[19,20]].forEach(([a,b])=>{
    ctx.beginPath();
    ctx.moveTo(lms[a].x*canvas.width, lms[a].y*canvas.height);
    ctx.lineTo(lms[b].x*canvas.width, lms[b].y*canvas.height);
    ctx.stroke();
  });

  const g   = detectGesture(lms);
  const now = Date.now();

  if (!g) {
    glabel.textContent = "SHOW GESTURE";
    glabel.style.color = "#4a5a78";
    resetHold();
    return;
  }

  // Swipe = instant trigger
  if (g === "SWIPE") {
    pfill.style.width      = "100%";
    pfill.style.background = CLR[g];
    glabel.textContent     = LABEL[g];
    glabel.style.color     = CLR[g];
    if (now - lastTrig > COOL_MS) {
      lastTrig = now;
      clickBtn(BTN[g]);
      wristHistory = [];
      resetHold();
    }
    return;
  }

  // Hold gestures
  if (g !== lastG) { lastG=g; holdStart=now; triggered=false; }
  const elapsed = now - holdStart;
  const pct     = Math.min(100, (elapsed/HOLD_MS)*100);

  pfill.style.width      = pct + "%";
  pfill.style.background = CLR[g];
  glabel.textContent     = LABEL[g];
  glabel.style.color     = CLR[g];

  if (elapsed >= HOLD_MS && !triggered && now-lastTrig > COOL_MS) {
    triggered = true;
    lastTrig  = now;
    const ok  = clickBtn(BTN[g]);
    glabel.textContent = ok ? "✓ " + BTN[g] : "ACTIVATED";
    setTimeout(()=>resetHold(), 900);
  }
}

const hands = new Hands({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
});
hands.setOptions({
  maxNumHands:1, modelComplexity:0,
  minDetectionConfidence:0.72, minTrackingConfidence:0.55
});
hands.onResults(onResults);

navigator.mediaDevices.getUserMedia({video:{width:160,height:120,facingMode:"user"}})
  .then(stream=>{
    video.srcObject = stream;
    return video.play();
  })
  .then(()=>{
    glabel.textContent = "READY";
    glabel.style.color = "#00e676";
    async function loop() {
      if (video.readyState >= 2) await hands.send({image:video});
      requestAnimationFrame(loop);
    }
    loop();
  })
  .catch(()=>{
    glabel.textContent = "CAMERA ERROR";
    glabel.style.color = "#ff3d57";
  });
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&family=Barlow:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background-color:{C_BG} !important;
    color:{C_TEXT} !important;
    font-family:'Barlow',sans-serif;
}}
[data-testid="stSidebar"] {{
    background-color:{C_SURFACE} !important;
    border-right:1px solid {C_BORDER} !important;
}}
[data-testid="block-container"] {{
    padding:1.2rem 2rem 2rem 2rem !important;
    max-width:1800px !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top:1.5rem; }}

.node-label {{
    font-family:'Barlow Condensed',sans-serif;
    font-size:1.1rem; font-weight:600;
    letter-spacing:0.14em; text-transform:uppercase;
    color:{C_TEXT_BRT};
}}
.node-id {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.62rem; color:{C_TEXT_DIM};
    letter-spacing:0.08em;
}}
.status-pill {{
    display:inline-flex; align-items:center; gap:6px;
    font-family:'JetBrains Mono',monospace;
    font-size:0.65rem; font-weight:500;
    letter-spacing:0.12em; text-transform:uppercase;
    padding:4px 12px 4px 8px;
    border-radius:2px; border:1px solid;
}}
.status-online  {{ background:rgba(0,230,118,0.08); border-color:rgba(0,230,118,0.4); color:{C_NORMAL}; }}
.status-offline {{ background:rgba(255,61,87,0.08);  border-color:rgba(255,61,87,0.4);  color:{C_ANOMALY}; }}
.status-waiting {{ background:rgba(255,179,0,0.08);  border-color:rgba(255,179,0,0.3);  color:{C_WARN}; }}
.dot {{ width:6px; height:6px; border-radius:50%; display:inline-block; }}
.dot-online  {{ background:{C_NORMAL};  box-shadow:0 0 6px {C_NORMAL}; }}
.dot-offline {{ background:{C_ANOMALY}; box-shadow:0 0 6px {C_ANOMALY}; }}
.dot-waiting {{ background:{C_WARN}; }}

.metric-grid {{
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:8px; margin:0.8rem 0;
}}
.metric-tile {{
    background:{C_BG}; border:1px solid {C_BORDER};
    border-radius:3px; padding:10px 12px 8px;
    position:relative; overflow:hidden;
}}
.metric-tile::before {{
    content:''; position:absolute;
    top:0; left:0; right:0; height:1px;
}}
.tile-n1::before {{ background:linear-gradient(90deg,{C_N1},transparent); }}
.tile-n2::before {{ background:linear-gradient(90deg,{C_N2},transparent); }}
.metric-label {{
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.65rem; font-weight:500;
    letter-spacing:0.14em; text-transform:uppercase;
    color:{C_TEXT_DIM}; margin-bottom:4px;
}}
.metric-value {{
    font-family:'JetBrains Mono',monospace;
    font-size:1.3rem; font-weight:400;
    color:{C_TEXT_BRT}; line-height:1.1;
    letter-spacing:-0.02em;
}}
.metric-unit {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.62rem; color:{C_TEXT_DIM}; margin-top:2px;
}}

.decision-normal {{
    display:flex; align-items:center; gap:10px;
    background:rgba(0,230,118,0.06);
    border:1px solid rgba(0,230,118,0.25);
    border-left:3px solid {C_NORMAL};
    border-radius:3px; padding:9px 14px; margin:0.6rem 0;
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.95rem; font-weight:600;
    letter-spacing:0.14em; text-transform:uppercase;
    color:{C_NORMAL};
}}
.decision-anomaly {{
    display:flex; align-items:center; gap:10px;
    background:rgba(255,61,87,0.08);
    border:1px solid rgba(255,61,87,0.35);
    border-left:3px solid {C_ANOMALY};
    border-radius:3px; padding:9px 14px; margin:0.6rem 0;
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.95rem; font-weight:600;
    letter-spacing:0.14em; text-transform:uppercase;
    color:{C_ANOMALY};
}}
.decision-waiting {{
    background:rgba(74,90,120,0.10); border:1px solid {C_BORDER};
    border-left:3px solid {C_TEXT_DIM};
    border-radius:3px; padding:9px 14px; margin:0.6rem 0;
    font-family:'JetBrains Mono',monospace;
    font-size:0.72rem; color:{C_TEXT_DIM};
}}
.mse-badge {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.75rem; opacity:0.7; margin-left:auto;
}}
.section-label {{
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.65rem; font-weight:600;
    letter-spacing:0.18em; text-transform:uppercase;
    color:{C_TEXT_DIM};
    border-bottom:1px solid {C_BORDER};
    padding-bottom:4px; margin:1rem 0 0.5rem;
}}
.count-box {{
    background:{C_SURFACE}; border:1px solid {C_BORDER};
    border-radius:3px; padding:14px 18px; text-align:center;
}}
.count-label {{
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.65rem; letter-spacing:0.14em;
    text-transform:uppercase; color:{C_TEXT_DIM}; margin-bottom:4px;
}}
.count-num       {{ font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:300; color:{C_ANOMALY}; }}
.count-num-zero  {{ font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:300; color:{C_NORMAL}; }}
.sidebar-section {{
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.62rem; font-weight:600;
    letter-spacing:0.18em; text-transform:uppercase;
    color:{C_TEXT_DIM}; border-bottom:1px solid {C_BORDER};
    padding-bottom:4px; margin:1.1rem 0 0.5rem;
}}
.sidebar-kv {{
    display:flex; justify-content:space-between;
    font-family:'JetBrains Mono',monospace;
    font-size:0.65rem; color:{C_TEXT};
    padding:3px 0; border-bottom:1px solid rgba(26,37,64,0.5);
}}
.sidebar-key {{ color:{C_TEXT_DIM}; }}
.sidebar-val {{ color:{C_TEXT_BRT}; }}
.stButton > button {{
    background:{C_BG} !important; border:1px solid {C_BORDER_LIT} !important;
    color:{C_TEXT} !important; font-family:'Barlow Condensed',sans-serif !important;
    letter-spacing:0.1em !important; font-size:0.78rem !important;
    text-transform:uppercase !important; border-radius:2px !important;
}}
.stButton > button:hover {{
    border-color:{C_N1} !important; color:{C_TEXT_BRT} !important;
}}
#MainMenu, footer, header {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_csv(path, columns):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, names=columns, header=0, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    if df.empty:
        return df
    for col in [c for c in columns if c != "decision"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["decision"] = df["decision"].astype(str).str.strip().str.upper()
    df.dropna(subset=["timestamp_ms"], inplace=True)
    df.sort_values("timestamp_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def node_status(df):
    if df is None or df.empty:
        return "WAITING"
    return "ONLINE" if (time.time()*1000 - df["timestamp_ms"].iloc[-1]) < OFFLINE_SECS*1000 else "OFFLINE"


def latest_row(df):
    return df.iloc[-1] if df is not None and not df.empty else None


def tail_df(df, n=TAIL_ROWS):
    if df is None or df.empty:
        return pd.DataFrame()
    return df.tail(n).reset_index(drop=True)


def fmt(val, d=4):
    try:
        return f"{float(val):.{d}f}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

# Base layout WITHOUT showlegend — caller adds it explicitly
_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", size=10, color=TICK_CLR),
    margin=dict(l=46, r=16, t=36, b=28),
    hovermode="x unified",
    xaxis=dict(
        showgrid=True, gridcolor=GRID_CLR, gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=9), tickcolor=TICK_CLR, title=None,
    ),
    yaxis=dict(
        showgrid=True, gridcolor=GRID_CLR, gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=9), tickcolor=TICK_CLR,
    ),
)

TITLE_FONT = dict(family="Barlow Condensed, sans-serif", size=12, color="#6a80a8")


def base_title(text):
    return dict(text=text, font=TITLE_FONT, x=0, xanchor="left", pad=dict(l=0, t=2))


def anomaly_mask(df_t):
    if df_t.empty or "decision" not in df_t.columns:
        return pd.Series(False, index=df_t.index)
    return df_t["decision"].str.upper() == "ANOMALY"


def add_anomaly_markers(fig, df_t, col, row=None, col_num=None):
    mask = anomaly_mask(df_t)
    if mask.any() and col in df_t.columns:
        kw = dict(row=row, col=col_num) if row is not None else {}
        fig.add_trace(go.Scatter(
            x=df_t.index[mask], y=df_t.loc[mask, col],
            mode="markers",
            marker=dict(color=C_ANOMALY, size=7, symbol="x",
                        line=dict(width=1.5, color=C_ANOMALY)),
            showlegend=False,
            hovertemplate="⚠ ANOMALY<extra></extra>",
        ), **kw)


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

def mse_chart(df_t, threshold, node_colour, fill_colour):
    fig = go.Figure()
    if not df_t.empty and "mse_scaled" in df_t.columns:
        # Red shading above threshold
        fig.add_trace(go.Scatter(
            x=df_t.index, y=df_t["mse_scaled"].clip(lower=threshold),
            fill="tozeroy", fillcolor="rgba(255,61,87,0.05)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df_t.index, y=df_t["mse_scaled"],
            mode="lines",
            line=dict(color=node_colour, width=1.8),
            fill="tozeroy", fillcolor=fill_colour,
            showlegend=False,
            hovertemplate="%{y:.6f}<extra>MSE</extra>",
        ))
        fig.add_hline(
            y=threshold, line_dash="dot",
            line_color=C_ANOMALY, line_width=1.2,
            annotation_text=f"THR {threshold:.7f}",
            annotation_position="top right",
            annotation_font=dict(size=9, color=C_ANOMALY,
                                 family="JetBrains Mono"),
        )
        add_anomaly_markers(fig, df_t, "mse_scaled")
    fig.update_layout(
        **_BASE_LAYOUT,
        showlegend=False,
        height=215,
        title=base_title("MSE  ·  ANOMALY SCORE"),
    )
    return fig


def _subplot_trace(fig, df_t, r, c, col, clr, fill):
    """Add a line+fill trace and anomaly markers to a subplot cell."""
    if not df_t.empty and col in df_t.columns:
        fig.add_trace(go.Scatter(
            x=df_t.index, y=df_t[col],
            mode="lines", line=dict(color=clr, width=1.6),
            fill="tozeroy", fillcolor=fill,
            showlegend=False,
            hovertemplate=f"%{{y:.4f}}<extra>{col}</extra>",
        ), row=r, col=c)
        add_anomaly_markers(fig, df_t, col, row=r, col_num=c)


def _build_multi_chart(df_t, configs, node_label):
    """Generic 2×2 subplot chart for any node's 4 signals."""
    subtitles = tuple(cfg[0] for cfg in configs)
    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.20,
        subplot_titles=subtitles,
    )
    for title, r, c, col, clr, fill in configs:
        _subplot_trace(fig, df_t, r, c, col, clr, fill)

    # Style subplot title annotations
    for ann in fig.layout.annotations:
        ann.font = dict(family="Barlow Condensed, sans-serif",
                        size=10, color="#6a80a8")

    # Build layout without duplicate keys
    layout = dict(**_BASE_LAYOUT)
    layout.update(
        showlegend=False,
        height=390,
        title=base_title("LIVE SIGNAL OVERVIEW"),
    )
    # Subplot axes need per-axis grid settings
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR,
                     tickfont=dict(size=8), zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR,
                     tickfont=dict(size=8), zeroline=False)
    return fig


def multi_chart_n1(df_t):
    configs = [
        ("TEMPERATURE  °C", 1, 1, "temp_C",         C_N1,      C_N1_FILL),
        ("BUS VOLTAGE  V",  1, 2, "bus_V",           "#4dd0e1", "rgba(77,208,225,0.07)"),
        ("CURRENT  mA",     2, 1, "current_mA_abs",  "#ab47bc", "rgba(171,71,188,0.07)"),
        ("POWER  mW",       2, 2, "power_mW",        "#66bb6a", "rgba(102,187,106,0.07)"),
    ]
    return _build_multi_chart(df_t, configs, "NODE 01")


def multi_chart_n2(df_t):
    configs = [
        ("LIGHT LEVEL  lux", 1, 1, "lux",            C_N2,      C_N2_FILL),
        ("BUS VOLTAGE  V",   1, 2, "bus_V",           "#4dd0e1", "rgba(77,208,225,0.07)"),
        ("CURRENT  mA",      2, 1, "current_mA_abs",  "#ab47bc", "rgba(171,71,188,0.07)"),
        ("POWER  mW",        2, 2, "power_mW",        "#66bb6a", "rgba(102,187,106,0.07)"),
    ]
    return _build_multi_chart(df_t, configs, "NODE 02")


# ─────────────────────────────────────────────
# HTML COMPONENT HELPERS
# ─────────────────────────────────────────────

def status_pill_html(status):
    cls = {"ONLINE": "status-online", "OFFLINE": "status-offline",
           "WAITING": "status-waiting"}[status]
    dot = {"ONLINE": "dot-online", "OFFLINE": "dot-offline",
           "WAITING": "dot-waiting"}[status]
    return (f'<span class="status-pill {cls}">'
            f'<span class="dot {dot}"></span>{status}</span>')


def metric_tile_html(label, value, unit, node="n1"):
    return (f'<div class="metric-tile tile-{node}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'</div>')


def decision_banner_html(decision, mse_val):
    mse_str = (f'<span class="mse-badge">MSE {fmt(mse_val, 6)}</span>'
               if mse_val != "—" else "")
    if decision == "NORMAL":
        return (f'<div class="decision-normal">'
                f'<span>▶ NORMAL — All signals within threshold</span>'
                f'{mse_str}</div>')
    elif decision == "ANOMALY":
        return (f'<div class="decision-anomaly">'
                f'<span>⚠ ANOMALY DETECTED — MSE exceeds threshold</span>'
                f'{mse_str}</div>')
    return '<div class="decision-waiting">Awaiting inference data...</div>'


# ─────────────────────────────────────────────
# NODE PANEL RENDERER
# ─────────────────────────────────────────────

def render_node(node_id, df, threshold, node_colour, fill_colour,
                sensor_col, sensor_label, sensor_unit):
    nd     = f"n{node_id}"
    status = node_status(df)
    row    = latest_row(df)
    df_t   = tail_df(df)

    n_label = ("NODE 01  ·  WIND / TEMPERATURE"
               if node_id == 1 else "NODE 02  ·  SOLAR / LIGHT")
    hw      = ("ESP32 + DS18B20 + INA219"
               if node_id == 1 else "ESP32 + BH1750 + INA219")

    # Heading row
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.8rem;">
      <div>
        <div class="node-label" style="color:{'#2979ff' if node_id==1 else '#ff9100'};">
          {n_label}
        </div>
        <div class="node-id">{hw}</div>
      </div>
      {status_pill_html(status)}
    </div>""", unsafe_allow_html=True)

    if df is None or df.empty:
        st.markdown(
            '<div class="decision-waiting">'
            '⏳ Waiting for data — start the serial logger for this node'
            '</div>',
            unsafe_allow_html=True)
        return

    # Metric tiles
    sv = fmt(row.get(sensor_col),        2) if row is not None else "—"
    bv = fmt(row.get("bus_V"),           4) if row is not None else "—"
    ca = fmt(row.get("current_mA_abs"),  4) if row is not None else "—"
    pw = fmt(row.get("power_mW"),        4) if row is not None else "—"

    tiles = (
        '<div class="metric-grid">'
        + metric_tile_html(sensor_label, sv, sensor_unit, nd)
        + metric_tile_html("Bus Voltage",   bv,  "V",  nd)
        + metric_tile_html("Current",       ca,  "mA", nd)
        + metric_tile_html("Power",         pw,  "mW", nd)
        + '</div>'
    )
    st.markdown(tiles, unsafe_allow_html=True)

    # Decision banner
    decision = str(row.get("decision", "—")).upper() if row is not None else "—"
    mse_val  = row.get("mse_scaled", "—")             if row is not None else "—"
    st.markdown(decision_banner_html(decision, mse_val), unsafe_allow_html=True)

    # MSE chart
    st.markdown('<div class="section-label">Anomaly Score — MSE</div>',
                unsafe_allow_html=True)
    st.plotly_chart(mse_chart(df_t, threshold, node_colour, fill_colour),
                    width="stretch", key=f"mse_{nd}")

    # Multi-signal chart
    st.markdown('<div class="section-label">Live Sensor Signals</div>',
                unsafe_allow_html=True)
    chart_fn = multi_chart_n1 if node_id == 1 else multi_chart_n2
    st.plotly_chart(chart_fn(df_t), width="stretch", key=f"multi_{nd}")

    # Stats strip
    if not df_t.empty and "mse_scaled" in df_t.columns:
        ms = df_t["mse_scaled"].dropna()
        n_an = int((df_t["decision"] == "ANOMALY").sum()) \
               if "decision" in df_t.columns else 0
        an_clr = C_ANOMALY if n_an > 0 else C_NORMAL
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin-top:4px;
                    font-family:'JetBrains Mono',monospace;
                    font-size:0.65rem;color:{C_TEXT_DIM};">
          <span>MIN <span style="color:{C_TEXT}">{ms.min():.6f}</span></span>
          <span style="color:{C_BORDER_LIT}">|</span>
          <span>MAX <span style="color:{C_TEXT}">{ms.max():.6f}</span></span>
          <span style="color:{C_BORDER_LIT}">|</span>
          <span>AVG <span style="color:{C_TEXT}">{ms.mean():.6f}</span></span>
          <span style="color:{C_BORDER_LIT}">|</span>
          <span>ANOMALIES (last {TAIL_ROWS})
            <span style="color:{an_clr}">{n_an}</span>
          </span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

def render_header(df1, df2):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S  UTC")
    s1, s2  = node_status(df1), node_status(df2)

    def dot_span(s):
        clr = C_NORMAL if s == "ONLINE" else (C_ANOMALY if s == "OFFLINE" else C_WARN)
        return (f'<span style="width:7px;height:7px;border-radius:50%;'
                f'background:{clr};display:inline-block;margin-right:5px;'
                f'box-shadow:0 0 5px {clr};"></span>')

    st.markdown(f"""
    <div style="border-bottom:1px solid {C_BORDER};
                padding-bottom:1rem; margin-bottom:1.4rem;
                display:flex; align-items:flex-end; gap:1rem;">
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;
                    font-size:2rem; font-weight:700;
                    letter-spacing:0.16em; color:{C_TEXT_BRT};
                    line-height:1; text-transform:uppercase;">
          ⚡ AEGIS
        </div>
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:0.62rem; color:{C_TEXT_DIM};
                    letter-spacing:0.14em; text-transform:uppercase;
                    margin-top:4px;">
          Autonomous Energy Grid Intelligence System
        </div>
      </div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:1.5rem;">
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:0.62rem; color:{C_TEXT_DIM};">
          {dot_span(s1)}N01 {s1} &nbsp;&nbsp; {dot_span(s2)}N02 {s2}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:0.62rem; color:{C_TEXT_DIM};
                    border:1px solid {C_BORDER};
                    padding:4px 10px; border-radius:2px;
                    letter-spacing:0.08em;">
          {now_str}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(df1, df2):
    def kv(k, v):
        return (f'<div class="sidebar-kv">'
                f'<span class="sidebar-key">{k}</span>'
                f'<span class="sidebar-val">{v}</span></div>')

    st.sidebar.markdown(f"""
    <div style="font-family:'Barlow Condensed',sans-serif;
                font-size:1.05rem; font-weight:700;
                letter-spacing:0.18em; text-transform:uppercase;
                color:{C_TEXT_BRT};
                border-bottom:1px solid {C_BORDER};
                padding-bottom:8px; margin-bottom:0.8rem;">
      ⚡ AEGIS
    </div>
    <div style="font-family:'JetBrains Mono',monospace;
                font-size:0.58rem; color:{C_TEXT_DIM};
                letter-spacing:0.1em; margin-bottom:0.8rem;">
      UWE Bristol · 24040034
    </div>""", unsafe_allow_html=True)

    s1 = node_status(df1)
    s2 = node_status(df2)
    r1 = len(df1) if df1 is not None else 0
    r2 = len(df2) if df2 is not None else 0

    st.sidebar.markdown(
        f'<div class="sidebar-section">Node Status</div>'
        + kv("Node 01", s1) + kv("Node 02", s2)
        + kv("N01 rows", str(r1)) + kv("N02 rows", str(r2)),
        unsafe_allow_html=True)

    st.sidebar.markdown(
        f'<div class="sidebar-section">Thresholds</div>'
        + kv("N01 MSE", f"{THRESHOLD_N1:.7f}")
        + kv("N02 MSE", f"{THRESHOLD_N2:.7f}"),
        unsafe_allow_html=True)

    def path_row(path):
        ok   = os.path.exists(path)
        clr  = C_NORMAL if ok else C_ANOMALY
        mark = "✔" if ok else "✘"
        return (f'<div style="font-family:JetBrains Mono,monospace;'
                f'font-size:0.6rem;padding:3px 0;color:{C_TEXT_DIM};">'
                f'<span style="color:{clr};">{mark}</span> '
                f'{os.path.basename(path)}</div>')

    st.sidebar.markdown(
        f'<div class="sidebar-section">Data Sources</div>'
        + path_row(NODE1_CSV) + path_row(NODE2_CSV),
        unsafe_allow_html=True)

    st.sidebar.markdown(
        f'<div class="sidebar-section">Config</div>'
        + kv("Refresh interval", f"{REFRESH_SECS} s")
        + kv("Chart tail", f"{TAIL_ROWS} rows")
        + kv("Offline after", f"{OFFLINE_SECS} s"),
        unsafe_allow_html=True)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("⬡  Clear Anomaly History", use_container_width=True):
        st.sidebar.info("Implement data deletion in your logger if needed.")

    # ── Gesture Control ──────────────────────────
    st.sidebar.markdown(
        f'<div class="sidebar-section">Gesture Control</div>',
        unsafe_allow_html=True)

    # Fallback manual buttons (also targeted by gesture JS)
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("MAIN", key="view_main_btn", use_container_width=True):
            st.session_state.view = "main"
            st.rerun()
    with c2:
        if st.button("N01", key="view_n01_btn", use_container_width=True):
            st.session_state.view = "node1"
            st.rerun()
    with c3:
        if st.button("N02", key="view_n02_btn", use_container_width=True):
            st.session_state.view = "node2"
            st.rerun()

    with st.sidebar:
        components.html(GESTURE_HTML, height=215, scrolling=False)

    st.sidebar.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;
                font-size:0.55rem; color:{C_TEXT_DIM};
                margin-top:2rem; letter-spacing:0.08em;
                border-top:1px solid {C_BORDER};
                padding-top:0.8rem; line-height:1.8;">
      AEGIS · AUTONOMOUS ENERGY GRID<br>
      INTELLIGENCE SYSTEM<br>
      UWE Bristol · 2025
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SYSTEM SUMMARY
# ─────────────────────────────────────────────

def render_summary(df1, df2):
    st.markdown(f"""
    <div style="border-top:1px solid {C_BORDER};
                margin-top:1.5rem; padding-top:1.2rem;">
      <div style="font-family:'Barlow Condensed',sans-serif;
                  font-size:0.85rem; font-weight:600;
                  letter-spacing:0.16em; text-transform:uppercase;
                  color:{C_TEXT_DIM}; border-bottom:1px solid {C_BORDER};
                  padding-bottom:6px; margin-bottom:1rem;">
        ⬡ System Event Log · Anomaly History
      </div>
    </div>""", unsafe_allow_html=True)

    anomaly_rows = []

    def extract(df, label):
        if df is None or df.empty:
            return
        adf = df[df["decision"].str.upper() == "ANOMALY"].copy()
        adf["NODE"] = label
        anomaly_rows.extend(adf.to_dict("records"))

    extract(df1, "NODE 01")
    extract(df2, "NODE 02")

    n1_c  = sum(1 for r in anomaly_rows if r.get("NODE") == "NODE 01")
    n2_c  = sum(1 for r in anomaly_rows if r.get("NODE") == "NODE 02")
    total = n1_c + n2_c

    def count_box(label, n):
        cls = "count-num-zero" if n == 0 else "count-num"
        return (f'<div class="count-box">'
                f'<div class="count-label">{label}</div>'
                f'<div class="{cls}">{n}</div>'
                f'</div>')

    ca, cb, cc, _ = st.columns([1, 1, 1, 2])
    ca.markdown(count_box("Total Anomalies", total),  unsafe_allow_html=True)
    cb.markdown(count_box("Node 01",         n1_c),   unsafe_allow_html=True)
    cc.markdown(count_box("Node 02",         n2_c),   unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if anomaly_rows:
        combined = pd.DataFrame(anomaly_rows)
        combined.sort_values("timestamp_ms", ascending=False, inplace=True)
        base   = ["NODE", "timestamp_ms", "mse_scaled", "decision"]
        extras = [c for c in ["temp_C", "lux", "bus_V", "current_mA_abs", "power_mW"]
                  if c in combined.columns]
        st.dataframe(
            combined[base + extras].head(10).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )
    else:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;'
            f'font-size:0.72rem;color:{C_NORMAL};'
            f'padding:10px 0;letter-spacing:0.08em;">'
            f'▶  No anomaly events recorded — '
            f'all nodes operating within normal parameters.</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    df1 = load_csv(NODE1_CSV, NODE1_COLS)
    df2 = load_csv(NODE2_CSV, NODE2_COLS)

    render_header(df1, df2)
    render_sidebar(df1, df2)

    view = st.session_state.get("view", "main")

    # ── NODE 01 full-width ──
    if view == "node1":
        st.markdown(
            f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};'
            f'border-top:2px solid {C_N1};border-radius:4px;'
            f'padding:1.2rem 1.4rem 1rem;">',
            unsafe_allow_html=True)
        render_node(
            node_id=1, df=df1,
            threshold=THRESHOLD_N1,
            node_colour=C_N1,
            fill_colour=C_N1_FILL,
            sensor_col="temp_C",
            sensor_label="Temperature",
            sensor_unit="°C",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── NODE 02 full-width ──
    elif view == "node2":
        st.markdown(
            f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};'
            f'border-top:2px solid {C_N2};border-radius:4px;'
            f'padding:1.2rem 1.4rem 1rem;">',
            unsafe_allow_html=True)
        render_node(
            node_id=2, df=df2,
            threshold=THRESHOLD_N2,
            node_colour=C_N2,
            fill_colour=C_N2_FILL,
            sensor_col="lux",
            sensor_label="Light",
            sensor_unit="lux",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Main dual-panel ──
    else:
        left, _, right = st.columns([1, 0.03, 1])

        with left:
            st.markdown(
                f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};'
                f'border-top:2px solid {C_N1};border-radius:4px;'
                f'padding:1.2rem 1.4rem 1rem;">',
                unsafe_allow_html=True)
            render_node(
                node_id=1, df=df1,
                threshold=THRESHOLD_N1,
                node_colour=C_N1,
                fill_colour=C_N1_FILL,
                sensor_col="temp_C",
                sensor_label="Temperature",
                sensor_unit="°C",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown(
                f'<div style="background:{C_SURFACE};border:1px solid {C_BORDER};'
                f'border-top:2px solid {C_N2};border-radius:4px;'
                f'padding:1.2rem 1.4rem 1rem;">',
                unsafe_allow_html=True)
            render_node(
                node_id=2, df=df2,
                threshold=THRESHOLD_N2,
                node_colour=C_N2,
                fill_colour=C_N2_FILL,
                sensor_col="lux",
                sensor_label="Light",
                sensor_unit="lux",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        render_summary(df1, df2)

    time.sleep(REFRESH_SECS)
    st.rerun()


if __name__ == "__main__":
    main()