"""
AI-Generated Code Detector

"""

from __future__ import annotations

import os
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from streamlit_ace import st_ace
    HAS_ACE = True
except Exception:
    HAS_ACE = False

API_BASE = os.environ.get("AI_DETECTOR_API", "http://127.0.0.1:8000").rstrip("/")

# ── Palette ──────────────────────────────────────────────────
BG      = "#0F172A"
SURFACE = "#111827"
BLUE    = "#00D4FF"
VIOLET  = "#7C3AED"
MUTED   = "#9CA3AF"
ACCEPT  = "#00FFA3"
REVIEW  = "#F59E0B"
HOLD    = "#FF2E88"
GRAD    = "linear-gradient(135deg, #00D4FF 0%, #7C3AED 50%, #FF2E88 100%)"
_PAPER  = "#111827"
_TEXT   = "#9CA3AF"

LANG_OPTIONS = {
    "Python":     "python",
    "C":          "c",
    "C++":        "cpp",
    "C#":         "csharp",
    "Java":       "java",
    "JavaScript": "javascript",
}
ACE_LANG = {
    "python":     "python",
    "c":          "c_cpp",
    "cpp":        "c_cpp",
    "csharp":     "csharp",
    "java":       "java",
    "javascript": "javascript",
}

SAMPLES = {
    "python": (
        "def solve():\n"
        "    n = int(input())\n"
        "    arr = list(map(int, input().split()))\n"
        "    arr.sort()\n"
        "    print(sum(x * (i + 1) for i, x in enumerate(arr)))\n\n"
        'if __name__ == "__main__":\n'
        "    solve()\n"
    ),
    "c": (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n\n"
        "int cmp(const void *a, const void *b) {\n"
        "    return (*(long long*)a > *(long long*)b) - (*(long long*)a < *(long long*)b);\n"
        "}\n\n"
        "int main() {\n"
        "    int n; scanf(\"%d\", &n);\n"
        "    long long a[n];\n"
        "    for (int i = 0; i < n; i++) scanf(\"%lld\", &a[i]);\n"
        "    qsort(a, n, sizeof(long long), cmp);\n"
        "    long long ans = 0;\n"
        "    for (int i = 0; i < n; i++) ans += a[i] * (i + 1);\n"
        "    printf(\"%lld\\n\", ans);\n"
        "    return 0;\n"
        "}\n"
    ),
    "cpp": (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n"
        "int main(){\n"
        "    int n; cin >> n;\n"
        "    vector<long long> a(n);\n"
        "    for(auto& x : a) cin >> x;\n"
        "    sort(a.begin(), a.end());\n"
        "    long long ans = 0;\n"
        "    for(int i = 0; i < n; ++i) ans += a[i] * (long long)(i + 1);\n"
        "    cout << ans << endl;\n"
        "}\n"
    ),
    "csharp": (
        "using System;\n"
        "using System.Linq;\n\n"
        "class Program {\n"
        "    static void Main() {\n"
        "        int n = int.Parse(Console.ReadLine());\n"
        "        long[] a = Console.ReadLine().Split().Select(long.Parse).OrderBy(x => x).ToArray();\n"
        "        long ans = 0;\n"
        "        for (int i = 0; i < n; i++) ans += a[i] * (i + 1);\n"
        "        Console.WriteLine(ans);\n"
        "    }\n"
        "}\n"
    ),
    "java": (
        "import java.util.*;\n"
        "public class Main{\n"
        "    public static void main(String[] args){\n"
        "        Scanner sc = new Scanner(System.in);\n"
        "        int n = sc.nextInt();\n"
        "        long[] a = new long[n];\n"
        "        for(int i = 0; i < n; i++) a[i] = sc.nextLong();\n"
        "        Arrays.sort(a);\n"
        "        long ans = 0;\n"
        "        for(int i = 0; i < n; i++) ans += a[i] * (i + 1);\n"
        "        System.out.println(ans);\n"
        "    }\n"
        "}\n"
    ),
    "javascript": (
        "const lines = require('fs').readFileSync('/dev/stdin','utf8').trim().split('\\n');\n"
        "const n = parseInt(lines[0]);\n"
        "const a = lines[1].split(' ').map(Number).sort((x, y) => x - y);\n"
        "let ans = 0n;\n"
        "for (let i = 0; i < n; i++) ans += BigInt(a[i]) * BigInt(i + 1);\n"
        "console.log(ans.toString());\n"
    ),
}

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Generated Code Detector — Epoka",
    page_icon="◐",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700;800&display=swap');

      html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, sans-serif !important;
        background-color: {BG} !important;
        color: #E5E7EB !important;
        font-size: 16px !important;
      }}
      .stApp {{ background: {BG} !important; }}
      .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
      }}

      /* ── Page title header ── */
      .page-header {{
        background: {SURFACE};
        border-bottom: 1px solid rgba(0,212,255,0.12);
        padding: 0 32px;
        position: relative;
        margin: 0 -1rem 24px;
      }}
      .page-header::after {{
        content: '';
        display: block;
        height: 2px;
        background: {GRAD};
      }}
      .page-header__inner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 64px;
      }}
      .page-header__title {{
        display: flex; align-items: center; gap: 14px;
      }}
      .page-header__icon {{
        width: 36px; height: 36px; border-radius: 10px;
        background: {GRAD};
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; color: #fff; box-shadow: 0 0 18px rgba(0,212,255,0.30);
      }}
      .page-header h1 {{
        font-size: 22px !important; font-weight: 800 !important;
        letter-spacing: -0.02em !important; margin: 0 !important;
        background: {GRAD};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }}
      .page-header__sub {{
        font-size: 12px; color: {MUTED};
      }}
      .page-header__right {{
        display: flex; align-items: center; gap: 12px;
      }}
      .api-chip {{
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        color: {BLUE}; background: rgba(0,212,255,0.08);
        border: 1px solid rgba(0,212,255,0.20); border-radius: 6px;
        padding: 4px 10px;
      }}
      .status-chip {{
        font-size: 11px; font-weight: 700; padding: 4px 10px;
        border-radius: 999px; border: 1px solid;
        font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
      }}
      .status-chip.ok  {{ color: {ACCEPT}; border-color: rgba(0,255,163,0.35); background: rgba(0,255,163,0.10); }}
      .status-chip.err {{ color: {HOLD};   border-color: rgba(255,46,136,0.35); background: rgba(255,46,136,0.10); }}

      /* ── Section labels ── */
      .section-label {{
        font-size: 12px; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: {MUTED};
        display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
      }}
      .section-label::before {{
        content: ''; display: inline-block; width: 6px; height: 6px;
        border-radius: 50%; background: {BLUE};
        box-shadow: 0 0 8px {BLUE};
        animation: blink 2s ease-in-out infinite;
      }}
      @keyframes blink {{
        0%,100% {{ opacity:1; }} 50% {{ opacity:0.35; }}
      }}
      .section-label::after {{
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(0,212,255,0.25), transparent);
      }}

      /* ── Decision badge ── */
      .badge {{
        display: inline-block; padding: 6px 18px; border-radius: 999px;
        font-size: 13px; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; border: 1px solid transparent;
        font-family: 'JetBrains Mono', monospace;
      }}
      .badge--idle   {{ color:{MUTED};  border-color:rgba(255,255,255,0.12); background:rgba(255,255,255,0.05); }}
      .badge--accept {{ color:{ACCEPT}; border-color:rgba(0,255,163,0.40);   background:rgba(0,255,163,0.10);  box-shadow:0 0 14px rgba(0,255,163,0.20); }}
      .badge--review {{ color:{REVIEW}; border-color:rgba(245,158,11,0.40);  background:rgba(245,158,11,0.10); }}
      .badge--hold   {{ color:{HOLD};   border-color:rgba(255,46,136,0.40);  background:rgba(255,46,136,0.10); box-shadow:0 0 14px rgba(255,46,136,0.25); }}

      /* ── Signal cards ── */
      .signal-card {{
        border-left: 2px solid {BLUE}; background: rgba(0,212,255,0.06);
        color: #E5E7EB; padding: 11px 15px; border-radius: 6px;
        margin-bottom: 8px; font-size: 15px; line-height: 1.6;
      }}
      .signal-card.warn {{ border-left-color:{REVIEW}; background:rgba(245,158,11,0.08); color:#fcd34d; }}
      .signal-card.bad  {{ border-left-color:{HOLD};   background:rgba(255,46,136,0.10); color:#fda4c4; }}

      /* ── Widget overrides ── */
      .stSelectbox > div > div, .stTextInput > div > div > input {{
        background: #05070D !important; border-radius: 8px !important;
        border-color: rgba(0,212,255,0.20) !important; color: #E5E7EB !important;
      }}
      .stTextArea textarea {{
        background: #05070D !important; color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important; line-height: 1.65 !important;
        border-color: rgba(0,212,255,0.15) !important;
        caret-color: {BLUE} !important;
        min-height: 480px !important;
      }}
      .stButton > button {{
        border-radius: 8px !important; font-weight: 700 !important;
        border: 1px solid rgba(0,212,255,0.40) !important;
        background: transparent !important; color: {BLUE} !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 0 16px rgba(0,212,255,0.15) !important;
        transition: all .2s ease !important;
      }}
      .stButton > button:hover {{
        box-shadow: 0 0 28px rgba(0,212,255,0.35) !important;
        background: rgba(0,212,255,0.10) !important;
      }}
      div[data-testid="metric-container"] {{
        background: {SURFACE}; border-radius: 10px; padding: 12px 14px;
        border: 1px solid rgba(0,212,255,0.10);
      }}
      div[data-testid="stMetricLabel"] {{ color:{MUTED} !important; font-size:11px !important; }}
      div[data-testid="stMetricValue"] {{
        color:{BLUE} !important;
        font-family:'JetBrains Mono',monospace !important;
        font-size:18px !important;
      }}
      .stSidebar {{
        background: {SURFACE} !important;
        border-right: 1px solid rgba(0,212,255,0.10) !important;
      }}
      h1,h2,h3,h4,h5 {{ color:#E5E7EB !important; }}
      label {{ color:{MUTED} !important; font-size:12px !important; }}
      .stCaption {{ color:{MUTED} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper: API calls ─────────────────────────────────────────
def call_health() -> tuple[bool, dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:
        return False, {"error": str(exc)}


def call_analyze(code: str, language: str, problem_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "language": language}
    if problem_id:
        payload["problem_id"] = problem_id
    r = requests.post(f"{API_BASE}/analyze", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


# ── Helper: colors ────────────────────────────────────────────
def color_for_score(s: float) -> str:
    if s < 0.4: return ACCEPT
    if s < 0.7: return REVIEW
    return HOLD


# ── Plotly charts ─────────────────────────────────────────────
def gauge_figure(score: float) -> go.Figure:
    col = color_for_score(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 3),
        number={"font": {"size": 52, "color": col, "family": "JetBrains Mono, monospace"}},
        gauge={
            "axis": {
                "range": [0, 1], "tickwidth": 1,
                "tickcolor": _TEXT, "tickfont": {"color": _TEXT, "size": 10},
            },
            "bar": {"color": col, "thickness": 0.22},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0.0, 0.4], "color": "rgba(0,255,163,0.10)"},
                {"range": [0.4, 0.7], "color": "rgba(245,158,11,0.10)"},
                {"range": [0.7, 1.0], "color": "rgba(255,46,136,0.10)"},
            ],
            "threshold": {
                "line": {"color": col, "width": 3},
                "thickness": 0.75, "value": score,
            },
        },
    ))
    fig.update_layout(
        height=210, margin={"l": 8, "r": 8, "t": 8, "b": 0},
        paper_bgcolor=_PAPER, font_color=_TEXT,
    )
    return fig


def radar_figure(components: dict[str, float | None]) -> go.Figure:
    keys   = ["statistical", "codebert", "llm_judge"]
    labels = ["Stylometric", "CodeBERT", "LLM-judge"]
    values = [float(components.get(k) or 0.0) for k in keys]
    fig = go.Figure(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1],
        fill="toself",
        line={"color": BLUE, "width": 2},
        fillcolor="rgba(0,212,255,0.10)",
        marker={"color": BLUE, "size": 5},
    ))
    grid = "rgba(255,255,255,0.07)"
    fig.update_layout(
        polar={
            "bgcolor": _PAPER,
            "radialaxis": {
                "visible": True, "range": [0, 1],
                "tickfont": {"size": 9, "color": _TEXT},
                "gridcolor": grid, "linecolor": grid,
            },
            "angularaxis": {
                "tickfont": {"size": 11, "color": "#E5E7EB"},
                "gridcolor": grid, "linecolor": grid,
            },
        },
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        height=230, paper_bgcolor=_PAPER,
    )
    return fig


# ── Render helpers ────────────────────────────────────────────
def render_signal(text: str, decision: str) -> None:
    cls = {"hold": "bad", "review": "warn"}.get(decision, "")
    extra = f" {cls}" if cls else ""
    st.markdown(f'<div class="signal-card{extra}">{text}</div>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
health_ok, health = call_health()
loaded = sum(bool(v) for v in (health.get("models_loaded") or {}).values()) if health_ok else 0
status_cls  = "ok" if health_ok else "err"
status_text = f"API · online · {loaded} model{'s' if loaded != 1 else ''} loaded" if health_ok else "API · offline"

st.markdown(
    f"""
    <div class="page-header">
      <div class="page-header__inner">
        <div class="page-header__title">
          <div class="page-header__icon">◐</div>
          <div>
            <h1>AI-Generated Code Detector</h1>
            <div class="page-header__sub"></div>
          </div>
        </div>
        <div class="page-header__right">
          <span class="api-chip">{API_BASE}</span>
          <span class="status-chip {status_cls}">{status_text}</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 75 / 25 split ─────────────────────────────────────────────
# Streamlit columns take ratios: 3 = 75%, 1 = 25%
left_col, right_col = st.columns([3, 1], gap="medium")

# ════════════════════════════════════════
#  LEFT — Code editor (75%)
# ════════════════════════════════════════
with left_col:
    st.markdown('<div class="section-label">Submission</div>', unsafe_allow_html=True)

    ctrl_a, ctrl_b, ctrl_c = st.columns([1.2, 1.6, 1.2])
    with ctrl_a:
        lang_label = st.selectbox("Language", list(LANG_OPTIONS.keys()), index=0, label_visibility="collapsed")
        language   = LANG_OPTIONS[lang_label]
    with ctrl_b:
        problem_id = st.text_input("Problem ID", placeholder="Problem ID (optional)", label_visibility="collapsed")
    with ctrl_c:
        s1, s2 = st.columns(2)
        with s1:
            if st.button("⌗ Sample", use_container_width=True):
                st.session_state["code_text"] = SAMPLES[language]
        with s2:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state["code_text"] = ""

    if "code_text" not in st.session_state:
        st.session_state["code_text"] = ""

    if HAS_ACE:
        code = st_ace(
            value=st.session_state["code_text"],
            language=ACE_LANG.get(language, "python"),
            theme="dracula",
            keybinding="vscode",
            tab_size=4,
            wrap=True,
            min_lines=22,
            max_lines=36,
            font_size=15,
            key=f"ace_{language}",
        )
    else:
        code = st.text_area(
            "Source code",
            value=st.session_state["code_text"],
            height=500,
            key=f"ta_{language}",
            label_visibility="collapsed",
            placeholder="// paste source code here…",
        )

    if code is not None:
        st.session_state["code_text"] = code

    lines = len(code.splitlines()) if code else 0
    chars = len(code) if code else 0
    c1, c2 = st.columns([5, 1])
    with c1:
        st.caption(f"`{lines}` lines · `{chars}` chars · {lang_label}   —   Ctrl+Enter to re-run after editing")
    with c2:
        analyze = st.button("⚡ Analyze", type="primary", use_container_width=True)

# Run analysis (outside column so result persists across reruns)
if analyze:
    if not (st.session_state.get("code_text") or "").strip():
        st.warning("Paste some code first.")
        st.stop()
    with st.spinner("Scanning submission…"):
        try:
            result = call_analyze(
                st.session_state["code_text"], language, problem_id or None
            )
            st.session_state["last_result"] = result
        except Exception as exc:
            st.session_state["last_result"] = {"error": str(exc)}

result = st.session_state.get("last_result")

# ════════════════════════════════════════
#  RIGHT — Score + Explanation (25%)
# ════════════════════════════════════════
with right_col:

    # ── Score section ──────────────────
    st.markdown('<div class="section-label">Score</div>', unsafe_allow_html=True)

    if not result:
        st.markdown(
            f'<div class="signal-card" style="color:{MUTED};border-left-color:{MUTED};">'
            "Run an analysis to see the risk score."
            "</div>",
            unsafe_allow_html=True,
        )
    elif "error" in result:
        st.error(f"API error: {result['error']}")
    else:
        score    = float(result.get("risk_score", 0.0))
        decision = str(result.get("decision", ""))

        badge_map = {
            "accept": ("✓  Accept",          "badge--accept"),
            "review": ("⚑  Flag for review", "badge--review"),
            "hold":   ("✕  Hold — AI",        "badge--hold"),
        }
        label, badge_cls = badge_map.get(decision, (decision or "—", "badge--idle"))
        st.markdown(
            f'<div style="margin-bottom:8px">'
            f'<span class="badge {badge_cls}">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(gauge_figure(score), use_container_width=True)

        # Legend
        st.markdown(
            f"""
            <div style="display:flex;gap:10px;font-size:11px;color:{MUTED};
                        justify-content:center;margin:-8px 0 4px;flex-wrap:wrap;">
              <span><span style="color:{ACCEPT}">●</span> 0–0.4 accept</span>
              <span><span style="color:{REVIEW}">●</span> 0.4–0.7 review</span>
              <span><span style="color:{HOLD}">●</span> 0.7–1 hold</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(radar_figure(result.get("component_scores") or {}), use_container_width=True)

        # ── Explanation section ────────
        st.markdown('<div class="section-label">Explanation</div>', unsafe_allow_html=True)

        components = result.get("component_scores") or {}
        labels_keys = [
            ("Stylometric", "statistical"),
            ("CodeBERT",    "codebert"),
            ("LLM-judge",   "llm_judge"),
        ]
        m1, m2, m3 = st.columns(3)
        cols = [m1, m2, m3]
        for i, (lbl, key) in enumerate(labels_keys):
            v = components.get(key)
            cols[i].metric(lbl, "—" if v is None else f"{float(v):.2f}")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        signals = result.get("signals") or []
        if not signals:
            render_signal(
                "No suspicious signals detected." if decision == "accept"
                else "No signals — models may not be loaded yet.",
                "accept",
            )
        else:
            for s in signals:
                render_signal(s, decision)
