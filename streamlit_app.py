"""
streamlit_app.py
────────────────
PERCALL — PERplexity-based Regulatory Region CALLer
Premium Scientific Platform · White Theme · 5-Page Horizontal Navbar
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.motifs import MOTIF_LABELS, count_motifs, scan_motifs
from core.perplexity import compute_perplexity, local_residual
from core.plotting import (
    plot_genome_browser,
    plot_motif_distribution,
    plot_motif_enrichment,
    plot_region_distribution,
    plot_sequence_domain_map,
)
from core.region_caller import find_regions

# ============================================================================
# Page configuration — sidebar fully collapsed / hidden
# ============================================================================

st.set_page_config(
    page_title="PERCALL — Scientific Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# Premium White Design Framework — CSS
# ============================================================================

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900&display=swap');

/* ── Streamlit chrome removal ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
div[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }

/* ── App background: pure white ── */
.stApp { background: #FFFFFF !important; color: #0F172A !important; }

/* ── Typography ── */
*, h1, h2, h3, h4, h5, h6, p, span, label, div {
    font-family: 'Inter', 'IBM Plex Sans', system-ui, -apple-system, sans-serif !important;
}
h1 { color: #0F172A !important; font-weight: 800 !important; letter-spacing: -0.025em; }
h2 { color: #0F172A !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h3 { color: #1E293B !important; font-weight: 600 !important; }

/* ── Main container ── */
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

/* ═══════════════════════════════════════════════════════════
   TOP NAVBAR — st.tabs styled as horizontal navigation bar
   ═══════════════════════════════════════════════════════════ */
.stTabs { margin-top: 0 !important; }

[data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-bottom: 2px solid #E2E8F0 !important;
    padding: 0 2.5rem !important;
    gap: 0 !important;
    box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06) !important;
    position: sticky;
    top: 0;
    z-index: 200;
}

[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 0 !important;
    color: #64748B !important;
    font-weight: 500 !important;
    font-size: 0.77rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 1rem 1.35rem !important;
    height: auto !important;
    min-height: 3.2rem !important;
    margin: 0 !important;
    border: none !important;
    transition: color 0.18s ease, background 0.18s ease !important;
}
[data-baseweb="tab"]:hover {
    background: #F8FAFC !important;
    color: #2563EB !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #2563EB !important;
    background: transparent !important;
    font-weight: 700 !important;
}
[data-baseweb="tab-highlight"] {
    background-color: #2563EB !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}
[data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── Cards ── */
.sci-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin: 0.65rem 0;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    animation: fadeRise 0.32s ease-out;
}
.hl-card {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1px solid #BFDBFE;
    border-left: 4px solid #2563EB;
    border-radius: 0 14px 14px 0;
    padding: 1.2rem 1.5rem;
    margin: 0.65rem 0;
}
.hl-card h4 {
    color: #1E40AF !important;
    margin: 0 0 0.3rem 0 !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
}
.hl-card p { color: #1E3A8A; margin: 0; font-size: 0.87rem; line-height: 1.6; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.28rem 0.8rem;
    border-radius: 999px;
    font-size: 0.69rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 0.22rem;
}
.badge-blue  { background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; }
.badge-cyan  { background: #ECFEFF; color: #0E7490; border: 1px solid #67E8F9; }
.badge-green { background: #D1FAE5; color: #065F46; border: 1px solid #6EE7B7; }
.badge-amber { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }

/* ── Hero section ── */
.percall-hero {
    text-align: center;
    padding: 3.8rem 1rem 1.8rem 1rem;
}
.percall-logo {
    font-size: 5.2rem;
    font-weight: 900;
    letter-spacing: 0.24em;
    line-height: 1;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 35%, #0EA5E9 70%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.percall-full-name {
    color: #64748B;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-size: 0.81rem;
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.percall-tagline {
    color: #0F172A;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    max-width: 700px;
    margin: 0 auto 1.4rem auto;
    line-height: 1.35;
}
.percall-badges {
    display: flex;
    justify-content: center;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin: 0 auto 1.8rem auto;
    max-width: 920px;
}

/* ── Brand bar (above tabs) ── */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.65rem 2.5rem;
    background: #FFFFFF;
    border-bottom: 1px solid #F1F5F9;
}
.brand-logo-text {
    font-size: 1.28rem;
    font-weight: 900;
    letter-spacing: 0.16em;
    background: linear-gradient(135deg, #1D4ED8, #2563EB, #0EA5E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-sep  { color: #CBD5E1; font-size: 1.1rem; }
.brand-desc { font-size: 0.7rem; color: #94A3B8; letter-spacing: 0.11em; text-transform: uppercase; font-weight: 500; }

/* ── Workflow bar ── */
.workflow-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0;
    margin: 1.2rem 0 1.8rem 0;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    overflow: hidden;
}
.workflow-step {
    background: #FFFFFF;
    padding: 1.2rem 0.9rem;
    text-align: center;
    border-right: 1px solid #E2E8F0;
    transition: background 0.18s;
}
.workflow-step:last-child { border-right: none; }
.workflow-step:hover { background: #F8FAFC; }
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #2563EB;
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.workflow-step strong { display: block; color: #0F172A; font-size: 0.84rem; font-weight: 600; margin-bottom: 0.25rem; }
.workflow-step p { margin: 0; color: #64748B; font-size: 0.78rem; line-height: 1.45; }

/* ── Feature cards grid ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem;
    margin: 0.75rem 0 1.5rem 0;
}
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    transition: all 0.22s;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.feature-card:hover {
    border-color: #93C5FD;
    box-shadow: 0 6px 22px rgba(37, 99, 235, 0.1);
    transform: translateY(-3px);
}
.feature-card .fc-icon { font-size: 1.9rem; margin-bottom: 0.7rem; display: block; }
.feature-card h4 {
    color: #0F172A !important;
    font-size: 0.97rem !important;
    font-weight: 600 !important;
    margin: 0 0 0.4rem 0 !important;
}
.feature-card p { color: #64748B; font-size: 0.85rem; line-height: 1.6; margin: 0; }

/* ── Metric strip ── */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0 1.25rem 0;
}
.metric-item {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-item:hover { border-color: #93C5FD; }
.metric-val { display: block; color: #2563EB; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em; }
.metric-lbl { color: #64748B; font-size: 0.69rem; text-transform: uppercase;
    letter-spacing: 0.11em; font-weight: 600; margin-top: 0.15rem; }

/* ── Section header ── */
.section-hdr {
    font-size: 0.69rem;
    text-transform: uppercase;
    letter-spacing: 0.17em;
    font-weight: 700;
    color: #2563EB;
    margin: 1.6rem 0 0.85rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #DBEAFE;
    display: block;
}

/* ── Page header ── */
.page-kicker {
    font-size: 0.69rem;
    text-transform: uppercase;
    letter-spacing: 0.19em;
    font-weight: 700;
    color: #2563EB;
    margin-bottom: 0.25rem;
    display: block;
}
.page-subtitle {
    color: #64748B;
    font-size: 0.97rem;
    line-height: 1.7;
    max-width: 920px;
    margin-bottom: 1.5rem;
}

/* ── Info pills ── */
.info-pill {
    display: inline-block;
    margin: 0.2rem;
    padding: 0.38rem 0.75rem;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    color: #0F172A;
    font-size: 0.81rem;
    font-weight: 500;
}

/* ── Empty state ── */
.empty-state {
    background: #F8FAFC;
    border: 2px dashed #E2E8F0;
    border-radius: 16px;
    padding: 2.6rem 2rem;
    text-align: center;
    margin: 1rem 0 1.5rem 0;
}
.empty-state strong { display: block; font-size: 1.05rem; color: #0F172A; font-weight: 600; margin-bottom: 0.45rem; }
.empty-state span   { color: #64748B; font-size: 0.9rem; line-height: 1.7; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    transition: all 0.18s !important;
    min-height: 2.6rem !important;
}
.stButton > button:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.14) !important;
}
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
}
.stDownloadButton > button {
    border-radius: 10px !important;
    border: 1px solid #BFDBFE !important;
    background: #EFF6FF !important;
    color: #2563EB !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover { background: #DBEAFE !important; border-color: #93C5FD !important; }

/* ── Form inputs ── */
.stTextInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div,
.stFileUploader > div, .stNumberInput input {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    outline: none !important;
}

/* ── Metrics widget ── */
div[data-testid="metric-container"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 0.9rem 1rem !important;
}
div[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563EB, #0EA5E9) !important;
    border-radius: 999px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #E2E8F0 !important;
}

/* ── DNA animation ── */
.dna-s1 { stroke-dasharray: 12 6; animation: driftFwd 3.2s linear infinite; }
.dna-s2 { stroke-dasharray: 12 6; animation: driftBack 3.2s linear infinite; }

/* ── Keyframes ── */
@keyframes fadeRise {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes driftFwd  { from { stroke-dashoffset: 0; } to { stroke-dashoffset: -80; } }
@keyframes driftBack { from { stroke-dashoffset: 0; } to { stroke-dashoffset: 80;  } }
@keyframes logoShine {
    0%, 100% { filter: brightness(1); }
    50%       { filter: brightness(1.08); }
}

/* ── Responsive ── */
@media (max-width: 900px) {
    .percall-logo { font-size: 3.5rem; letter-spacing: 0.14em; }
    .workflow-grid { grid-template-columns: 1fr 1fr 1fr; }
    .feature-grid  { grid-template-columns: 1fr; }
    [data-baseweb="tab"] { padding: 0.75rem 0.7rem !important; font-size: 0.72rem !important; }
    .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .brand-bar { padding: 0.65rem 1rem; }
}
</style>"""

st.markdown(_CSS, unsafe_allow_html=True)

# ============================================================================
# Static HTML fragments
# ============================================================================

_BRAND_BAR = """
<div class="brand-bar">
  <span class="brand-logo-text">PERCALL</span>
  <span class="brand-sep">·</span>
  <span class="brand-desc">PERplexity-based Regulatory Region CALLer</span>
</div>"""

_DNA_SVG = """
<div style="text-align:center;padding:.6rem 0 1.4rem 0">
<svg viewBox="0 0 640 56" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:700px;height:56px;overflow:visible">
  <path class="dna-s1"
    d="M0,28 Q40,6 80,28 Q120,50 160,28 Q200,6 240,28 Q280,50 320,28
       Q360,6 400,28 Q440,50 480,28 Q520,6 560,28 Q600,50 640,28"
    fill="none" stroke="#2563EB" stroke-width="2.4" opacity="0.72"/>
  <path class="dna-s2"
    d="M0,28 Q40,50 80,28 Q120,6 160,28 Q200,50 240,28 Q280,6 320,28
       Q360,50 400,28 Q440,6 480,28 Q520,50 560,28 Q600,6 640,28"
    fill="none" stroke="#0EA5E9" stroke-width="2.4" opacity="0.72"/>
  <line x1="80"  y1="22" x2="80"  y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
  <line x1="160" y1="22" x2="160" y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
  <line x1="240" y1="22" x2="240" y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
  <line x1="320" y1="22" x2="320" y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
  <line x1="400" y1="22" x2="400" y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
  <line x1="480" y1="22" x2="480" y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
  <line x1="560" y1="22" x2="560" y2="34" stroke="#CBD5E1" stroke-width="1.6"/>
</svg>
</div>"""

# ============================================================================
# Light-theme plot palette & layout defaults
# ============================================================================

_LL: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#F8FAFC",
    font=dict(color="#0F172A", family="Inter, system-ui, sans-serif", size=11),
    hoverlabel=dict(bgcolor="#FFFFFF", font_color="#0F172A", bordercolor="#E2E8F0"),
    margin=dict(l=55, r=20, t=60, b=45),
)
_LX: dict = dict(
    gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
    color="#64748B", linecolor="#E2E8F0",
)
_LY: dict = dict(
    gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
    color="#64748B", linecolor="#E2E8F0",
)
_PC: dict = {
    "perp":  "#2563EB",
    "base":  "#94A3B8",
    "res":   "#F59E0B",
    "gc":    "#10B981",
    "trough": "#EF4444",
    "rfill": "rgba(37,99,235,0.07)",
    "rline": "rgba(37,99,235,0.55)",
}

# ============================================================================
# Utility helpers
# ============================================================================


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    """Parse FASTA text into (header, sequence) pairs."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = re.sub(r"[^ACGTN]", "N", "".join(parts).upper())
                records.append((header, seq))
            header = line[1:]
            parts = []
        else:
            parts.append(line)
    if header is not None:
        seq = re.sub(r"[^ACGTN]", "N", "".join(parts).upper())
        records.append((header, seq))
    return records


def gc_pct(seq: str) -> float:
    if not seq:
        return 0.0
    return round(100.0 * (seq.count("G") + seq.count("C")) / len(seq), 2)


def _example_files() -> List[str]:
    ex_dir = os.path.join(_ROOT, "example_data")
    if not os.path.isdir(ex_dir):
        return []
    return sorted(
        name for name in os.listdir(ex_dir)
        if name.lower().endswith((".fasta", ".fa", ".fna", ".txt"))
    )


def _valid_results(results: List[dict]) -> List[dict]:
    return [r for r in results if not r["skipped"] and r["perp"].size > 0]


def _region_seq(seq: str, region: dict, window: int) -> str:
    return seq[region["start"]: min(region["end"] + window, len(seq))]


def process_sequence(
    header: str,
    seq: str,
    *,
    window: int,
    baseline_win: int,
    min_len: int,
    max_len: int,
    top_k: int,
    score_cutoff: float,
    active_motifs: set,
) -> dict:
    """Run the full PERCALL pipeline on a single sequence."""
    perp = compute_perplexity(seq, window=window)
    if perp.size == 0 or np.all(np.isnan(perp)):
        return {
            "header": header, "seq": seq, "perp": perp,
            "baseline": perp.copy(), "residual": perp.copy(),
            "regions": [], "skipped": True,
        }
    res = local_residual(perp, baseline_win=baseline_win)
    baseline = (
        pd.Series(perp)
        .rolling(baseline_win, center=True, min_periods=baseline_win)
        .mean()
        .values
    )
    regions = find_regions(
        res, seq,
        min_len=min_len, max_len=max_len, top_k=top_k,
        score_cutoff=score_cutoff, window=window, active_motifs=active_motifs,
    )
    return {
        "header": header, "seq": seq, "perp": perp,
        "baseline": baseline, "residual": res,
        "regions": regions, "skipped": False,
    }


def _df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def _results_to_json(results: List[dict], params: dict) -> bytes:
    safe_params = {
        k: v for k, v in params.items()
        if k not in ("fasta_text", "run", "active_motifs")
    }
    safe_params["active_motifs"] = sorted(params.get("active_motifs", []))
    session = {
        "percall_version": "2025.2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": safe_params,
        "sequences": [
            {
                "header": r["header"],
                "length": len(r["seq"]),
                "skipped": r["skipped"],
                "gc_pct": gc_pct(r["seq"]),
                "n_regions": len(r["regions"]),
                "regions": r["regions"],
            }
            for r in results
        ],
    }
    return json.dumps(session, indent=2).encode()


def _regions_df(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for reg in r["regions"]:
            rows.append({
                "Sequence": r["header"][:60],
                "Rank": reg["rank"],
                "Start": reg["start"],
                "End": reg["end"],
                "Width": reg["width"],
                "Trough": reg["trough"],
                "Mean Residual": reg["mean_residual"],
                "GC%": reg["gc_pct"],
                "Motifs": reg["motifs"],
            })
    return pd.DataFrame(rows)


def _summary_df(results: List[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Header": r["header"][:70],
            "Length (bp)": len(r["seq"]),
            "GC%": gc_pct(r["seq"]),
            "Regions": len(r["regions"]),
            "Status": "Skipped" if r["skipped"] else "OK",
        }
        for r in results
    ])


# ============================================================================
# UI component helpers
# ============================================================================


def _metric_strip(items: List[Tuple[str, str]]) -> None:
    html = ['<div class="metric-strip">']
    for label, value in items:
        html.append(
            f'<div class="metric-item">'
            f'<span class="metric-val">{value}</span>'
            f'<span class="metric-lbl">{label}</span>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _section_header(text: str) -> None:
    st.markdown(f'<span class="section-hdr">{text}</span>', unsafe_allow_html=True)


def _page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div style="padding:1.6rem 0 0.1rem 0">'
        f'<span class="page-kicker">{kicker}</span>'
        f'<h1 style="margin:0 0 0.35rem;font-size:1.9rem">{title}</h1>'
        f'<p class="page-subtitle">{subtitle}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _empty_state(title: str, message: str) -> None:
    st.markdown(
        f'<div class="empty-state">'
        f'<strong>{title}</strong>'
        f'<span>{message}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _select_result(
    results: List[dict], key: str, label: str = "Select sequence"
) -> Optional[dict]:
    valid = _valid_results(results)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    labels = [r["header"][:80] for r in valid]
    chosen = st.selectbox(label, labels, key=key)
    return valid[labels.index(chosen)]


def _ltheme(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the light-theme layout to a core/plotting figure."""
    fig.update_layout(**_LL, height=height)
    fig.update_xaxes(**_LX)
    fig.update_yaxes(**_LY)
    return fig


# ============================================================================
# Light-theme plot functions
# ============================================================================


def _plot_perplexity(
    perp: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    regions: List[dict],
    title: str = "Perplexity Profile",
) -> go.Figure:
    pos = np.arange(len(perp))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Perplexity & Baseline", "Residual Signal"),
        row_heights=[0.55, 0.45],
    )
    fig.add_trace(
        go.Scatter(x=pos, y=perp, mode="lines", name="Perplexity",
                   line=dict(color=_PC["perp"], width=1.5),
                   hovertemplate="Pos %{x}<br>Perplexity %{y:.3f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=pos, y=baseline, mode="lines", name="Baseline",
                   line=dict(color=_PC["base"], width=1.3, dash="dot"),
                   hovertemplate="Pos %{x}<br>Baseline %{y:.3f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=pos, y=residual, mode="lines", name="Residual",
                   line=dict(color=_PC["res"], width=1.3),
                   hovertemplate="Pos %{x}<br>Residual %{y:.4f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="#CBD5E1", line_dash="dash",
                  line_width=0.9, row=2, col=1)
    for r in regions:
        for row in (1, 2):
            fig.add_vrect(
                x0=r["start"], x1=r["end"],
                fillcolor=_PC["rfill"], line_color=_PC["rline"], line_width=1.2,
                annotation_text=f"R{r['rank']}" if row == 1 else "",
                annotation_position="top left",
                annotation_font_color="#2563EB", annotation_font_size=9,
                row=row, col=1,
            )
        fig.add_vline(x=r["trough"], line_color=_PC["trough"],
                      line_dash="dot", line_width=1.0, row=1, col=1)
    fig.update_layout(
        **_LL,
        title=dict(text=title, font=dict(size=13, color="#0F172A")),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Position (bp)", row=2, col=1, **_LX)
    fig.update_yaxes(title_text="Perplexity", row=1, col=1, **_LY)
    fig.update_yaxes(title_text="Residual", row=2, col=1, **_LY)
    fig.update_annotations(font=dict(color="#64748B"))
    return fig


def _plot_composition(seq: str) -> go.Figure:
    bases = ["A", "C", "G", "T", "N"]
    counts = [seq.count(b) for b in bases]
    colors = ["#2563EB", "#0EA5E9", "#10B981", "#F59E0B", "#94A3B8"]
    fig = go.Figure(go.Bar(
        x=bases, y=counts, marker_color=colors,
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        **_LL,
        title=dict(text="Nucleotide Composition", font=dict(color="#0F172A")),
        height=280, showlegend=False,
    )
    fig.update_xaxes(**_LX)
    fig.update_yaxes(title="Count", **_LY)
    return fig


def _plot_gc_heatmap(seq: str, bins: int = 40) -> go.Figure:
    if not seq:
        values = np.array([[0.0]])
    else:
        chunk = max(1, len(seq) // bins)
        vals = [gc_pct(seq[i: i + chunk]) for i in range(0, len(seq), chunk)]
        values = np.array([vals])
    fig = go.Figure(go.Heatmap(
        z=values,
        colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
        hovertemplate="GC%: %{z:.1f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="GC%", tickfont=dict(color="#64748B")),
    ))
    fig.update_layout(
        **_LL,
        title=dict(text="GC Heatmap", font=dict(color="#0F172A")),
        height=180,
        xaxis=dict(showgrid=False, title="Sequence bins", **_LX),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def _plot_gc_profile(seq: str, regions: List[dict], window: int = 50) -> go.Figure:
    n = len(seq)
    if n < window:
        return go.Figure()
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    is_gc = ((arr == ord("G")) | (arr == ord("C"))).astype(np.float32)
    cs = np.concatenate([[0.0], np.cumsum(is_gc)])
    gc = (cs[window:] - cs[:-window]) / window * 100.0
    pos = np.arange(window // 2, window // 2 + len(gc))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pos, y=gc, mode="lines", name="GC%",
        line=dict(color=_PC["gc"], width=1.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="Position %{x}<br>GC% %{y:.1f}<extra></extra>",
    ))
    for r in regions:
        fig.add_vrect(
            x0=r["start"], x1=r["end"],
            fillcolor=_PC["rfill"], line_color=_PC["rline"], line_width=1.0,
        )
    fig.update_layout(
        **_LL,
        title=dict(text="GC Content Profile", font=dict(color="#0F172A")),
        height=280, hovermode="x unified",
    )
    fig.update_xaxes(title="Position (bp)", **_LX)
    fig.update_yaxes(title="GC%", **_LY)
    return fig


def _plot_perp_density(perp: np.ndarray) -> go.Figure:
    vals = pd.Series(perp).dropna()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Density Distribution", "Summary"),
        column_widths=[0.6, 0.4],
    )
    fig.add_trace(
        go.Histogram(x=vals, nbinsx=40, marker_color="#2563EB", opacity=0.82,
                     hovertemplate="Perplexity %{x:.3f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=vals, marker_color="#0EA5E9", boxmean=True,
               name="Perplexity",
               hovertemplate="Perplexity %{y:.3f}<extra></extra>"),
        row=1, col=2,
    )
    fig.update_layout(**_LL, height=300, showlegend=False)
    fig.update_xaxes(**_LX)
    fig.update_yaxes(**_LY)
    return fig


def _plot_comparative(valid: List[dict]) -> go.Figure:
    if not valid:
        return go.Figure()
    labels = [r["header"][:28] for r in valid]
    means = [float(np.nanmean(r["perp"])) for r in valid]
    mins = [float(np.nanmin(r["perp"])) for r in valid]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=means, name="Mean perplexity",
        marker_color="#2563EB",
        hovertemplate="%{x}<br>Mean %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=mins, mode="lines+markers",
        name="Minimum perplexity",
        line=dict(color="#F59E0B", width=2), marker=dict(size=8),
        hovertemplate="%{x}<br>Min %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_LL,
        title=dict(text="Comparative Perplexity Across Sequences", font=dict(color="#0F172A")),
        height=320, hovermode="x unified",
    )
    fig.update_xaxes(tickangle=-25, **_LX)
    fig.update_yaxes(title="Perplexity", **_LY)
    return fig


def _plot_signal_triptych(rv: dict) -> go.Figure:
    x = np.arange(len(rv["perp"]))
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Perplexity", "Rolling Baseline", "Residual Signal"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=rv["perp"], mode="lines", name="Perplexity",
                   line=dict(color=_PC["perp"], width=1.3)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rv["baseline"], mode="lines", name="Baseline",
                   line=dict(color=_PC["base"], width=1.3)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rv["residual"], mode="lines", name="Residual",
                   line=dict(color=_PC["res"], width=1.3)),
        row=3, col=1,
    )
    for r in rv["regions"]:
        for row in (1, 2, 3):
            fig.add_vrect(
                x0=r["start"], x1=r["end"],
                fillcolor=_PC["rfill"], line_color=_PC["rline"], line_width=1.0,
                row=row, col=1,
            )
    fig.add_hline(y=0, line_color="#CBD5E1", line_dash="dash", row=3, col=1)
    fig.update_layout(
        **_LL, height=560, showlegend=False, hovermode="x unified",
    )
    fig.update_xaxes(title_text="Position (bp)", row=3, col=1, **_LX)
    fig.update_yaxes(**_LY)
    fig.update_annotations(font=dict(color="#64748B"))
    return fig


def _plot_region_ranking(rv: dict) -> go.Figure:
    regions = rv["regions"]
    fig = go.Figure()
    if regions:
        fig.add_trace(go.Bar(
            x=[f"R{r['rank']}" for r in regions],
            y=[abs(r["mean_residual"]) for r in regions],
            marker_color="#2563EB",
            hovertemplate="Region %{x}<br>Depth %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        **_LL,
        title=dict(text="Region Depth Ranking", font=dict(color="#0F172A")),
        height=260, showlegend=False,
    )
    fig.update_xaxes(**_LX)
    fig.update_yaxes(title="Absolute mean residual", **_LY)
    return fig


def _plot_region_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df["Width"],
            y=df["Mean Residual"],
            mode="markers",
            marker=dict(
                size=11,
                color=df["GC%"] if "GC%" in df.columns else df["Width"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="GC%", tickfont=dict(color="#64748B")),
                line=dict(width=1, color="#FFFFFF"),
            ),
            text=df["Sequence"],
            hovertemplate="%{text}<br>Width %{x} bp<br>Mean residual %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        **_LL,
        title=dict(text="Region Width vs Depth", font=dict(color="#0F172A")),
        height=340,
    )
    fig.update_xaxes(title="Width (bp)", **_LX)
    fig.update_yaxes(title="Mean Residual", **_LY)
    return fig


def _plot_motif_heatmap(
    results: List[dict], active_motifs: set, window: int
) -> go.Figure:
    valid = _valid_results(results)
    motifs = [m for m in MOTIF_LABELS if m in active_motifs]
    z: List[list] = []
    y_labels: List[str] = []
    for r in valid:
        y_labels.append(r["header"][:26])
        row = []
        for motif in motifs:
            total = 0
            for reg in r["regions"]:
                total += count_motifs(
                    _region_seq(r["seq"], reg, window), {motif}
                ).get(motif, 0)
            row.append(total)
        z.append(row)
    fig = go.Figure()
    if z and motifs:
        fig.add_trace(go.Heatmap(
            z=z,
            x=[MOTIF_LABELS.get(m, m) for m in motifs],
            y=y_labels,
            colorscale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1D4ED8"]],
            hovertemplate="%{y}<br>%{x}: %{z} hits<extra></extra>",
        ))
    fig.update_layout(
        **_LL,
        title=dict(text="Motif Signal Across Called Regions", font=dict(color="#0F172A")),
        height=max(240, 120 + 36 * max(len(y_labels), 1)),
        margin=dict(l=120, r=20, t=60, b=40),
    )
    return fig


# ============================================================================
# Page 1 — Home
# ============================================================================


def _page_home(results: List[dict]) -> None:
    valid = _valid_results(results)
    total_regions = sum(len(r["regions"]) for r in valid)
    total_bp = sum(len(r["seq"]) for r in results)

    # Hero
    st.markdown(
        '<div class="percall-hero">'
        '<div class="percall-logo">PERCALL</div>'
        '<div class="percall-full-name">PERplexity-based Regulatory Region CALLer</div>'
        '<p class="percall-tagline">'
        'Information-Theoretic Regulatory Domain Calling from DNA Sequence'
        '</p>'
        '<div class="percall-badges">'
        '<span class="badge badge-blue">Information Theoretic</span>'
        '<span class="badge badge-cyan">Bounded Min-Mean Kadane</span>'
        '<span class="badge badge-green">Regulatory Domain Calling</span>'
        '<span class="badge badge-blue">Non-B DNA Annotation</span>'
        '<span class="badge badge-amber">Publication Grade</span>'
        '<span class="badge badge-green">Open Science</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(_DNA_SVG, unsafe_allow_html=True)

    # Live session metrics
    _metric_strip([
        ("Algorithmic core", "PERCALL"),
        ("Sequences loaded", str(len(results)) if results else "—"),
        ("Regions called", str(total_regions) if total_regions else "—"),
        ("Base pairs analysed", f"{total_bp:,}" if total_bp else "—"),
    ])

    # Quick-start hint
    st.markdown(
        '<div class="hl-card">'
        '<h4>🚀 Quick Start</h4>'
        '<p>Upload or paste a DNA sequence in the '
        '<strong>Sequence &amp; Perplexity</strong> tab, configure the analysis '
        'parameters, and click <strong>Run Analysis</strong>. '
        'Results populate all downstream tabs automatically.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Scientific workflow
    _section_header("Scientific Workflow")
    st.markdown(
        '<div class="workflow-grid">'
        '<div class="workflow-step">'
        '<div class="step-num">1</div>'
        '<strong>DNA Sequence</strong>'
        '<p>FASTA upload, pasted sequence, or curated examples</p>'
        '</div>'
        '<div class="workflow-step">'
        '<div class="step-num">2</div>'
        '<strong>Perplexity Calculation</strong>'
        '<p>Dinucleotide entropy → positional perplexity signal</p>'
        '</div>'
        '<div class="workflow-step">'
        '<div class="step-num">3</div>'
        '<strong>Residual Signal</strong>'
        '<p>Rolling baseline compensation isolates local troughs</p>'
        '</div>'
        '<div class="workflow-step">'
        '<div class="step-num">4</div>'
        '<strong>Bounded Min-Mean Kadane</strong>'
        '<p>O(n) optimal region calling under length constraints</p>'
        '</div>'
        '<div class="workflow-step">'
        '<div class="step-num">5</div>'
        '<strong>Domain Calling</strong>'
        '<p>Ranked low-perplexity regulatory domains reported</p>'
        '</div>'
        '<div class="workflow-step">'
        '<div class="step-num">6</div>'
        '<strong>Non-B DNA Annotation</strong>'
        '<p>G4, i-Motif, Z-DNA, eGZ-DNA, STR, Triplex overlay</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Feature cards
    _section_header("Platform Highlights")
    st.markdown(
        '<div class="feature-grid">'
        '<div class="feature-card">'
        '<span class="fc-icon">🧬</span>'
        '<h4>Sequence Analysis</h4>'
        '<p>FASTA upload, composition statistics, GC profiling, '
        'and multi-sequence comparative analytics.</p>'
        '</div>'
        '<div class="feature-card">'
        '<span class="fc-icon">📈</span>'
        '<h4>Perplexity Explorer</h4>'
        '<p>Interactive positional perplexity and residual signal '
        'with zoom, hover, and density distribution views.</p>'
        '</div>'
        '<div class="feature-card">'
        '<span class="fc-icon">📍</span>'
        '<h4>Region Caller</h4>'
        '<p>Flagship bounded minimum-mean Kadane optimisation. '
        'Region architecture maps, ranking, and trough visualisation.</p>'
        '</div>'
        '<div class="feature-card">'
        '<span class="fc-icon">🔬</span>'
        '<h4>Non-B DNA Structures</h4>'
        '<p>G4, i-Motif, Z-DNA, eGZ-DNA, STR, PolyA/T, Direct Repeat, '
        'Triplex enrichment and positional tracks.</p>'
        '</div>'
        '<div class="feature-card">'
        '<span class="fc-icon">📊</span>'
        '<h4>Reports &amp; Exports</h4>'
        '<p>One-click CSV, Excel, JSON, PNG, SVG, and PDF '
        'publication-ready outputs.</p>'
        '</div>'
        '<div class="feature-card">'
        '<span class="fc-icon">⚗️</span>'
        '<h4>Algorithmic Transparency</h4>'
        '<p>The original PERCALL algorithm is preserved unchanged. '
        'No alternative models or replacement callers are introduced.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Algorithm overview
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        _section_header("Core Algorithm")
        st.markdown(
            '<div class="sci-card">'
            '<div class="info-pill">Dinucleotide perplexity  H = −Σ p·log₂p → 2ᴴ</div>'
            '<div class="info-pill">Rolling baseline compensation</div>'
            '<div class="info-pill">Residual trough detection</div>'
            '<div class="info-pill">Bounded minimum-mean region optimisation</div>'
            '<div class="info-pill">Non-B DNA structural annotation</div>'
            '<p style="margin-top:.85rem;color:#64748B;font-size:.88rem;line-height:1.65">'
            'The platform preserves the supplied algorithmic pathway exactly. '
            'No machine learning model, alternative region caller, or replacement predictor '
            'is introduced at any stage.'
            '</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        _section_header("Algorithm Reference")
        st.latex(r"H = -\sum_i p_i \log_2 p_i")
        st.latex(r"\mathrm{Perplexity} = 2^H")
        st.latex(r"\mathrm{Residual}(x) = \mathrm{Perplexity}(x) - \mathrm{Baseline}(x)")

    # Live session preview
    if valid:
        _section_header("Live Session Preview")
        preview = valid[0]
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            st.plotly_chart(
                _plot_perplexity(
                    preview["perp"], preview["baseline"], preview["residual"],
                    preview["regions"],
                    title=f"Preview — {preview['header'][:55]}",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _ltheme(
                    plot_sequence_domain_map(
                        len(preview["seq"]), preview["regions"],
                        title="Region Architecture",
                    )
                ),
                use_container_width=True,
            )
    else:
        _empty_state(
            "Ready for interactive analysis",
            "Navigate to the Sequence & Perplexity tab, load a FASTA file "
            "or paste a DNA sequence, and click Run Analysis to begin.",
        )


# ============================================================================
# Page 2 — Sequence & Perplexity Analysis
# ============================================================================


def _page_sequence_perplexity() -> None:
    _page_header(
        "Page 02",
        "Sequence & Perplexity Analysis",
        "Upload or paste a DNA sequence, configure analysis parameters, and explore "
        "the full perplexity signal, rolling baseline, and residual architecture.",
    )

    # ── Sequence input ──────────────────────────────────────────────────────
    _section_header("A · Sequence Input")
    col_up, col_paste = st.columns([1, 1])
    with col_up:
        upload = st.file_uploader(
            "Upload FASTA / Multi-FASTA",
            type=["fasta", "fa", "fna", "txt"],
            help="Accepts single or multi-FASTA files.",
        )
    with col_paste:
        pasted = st.text_area(
            "Paste Sequence (FASTA format)",
            height=120,
            placeholder=">sequence_id\nATGCATGCATGC...",
        )

    examples = _example_files()
    example_choice = "None"
    if examples:
        example_choice = st.selectbox(
            "Or load an example dataset",
            ["None"] + examples,
            help="Example datasets bundled with PERCALL.",
        )

    fasta_text: Optional[str] = None
    if upload is not None:
        fasta_text = upload.read().decode("utf-8", errors="replace")
    elif pasted.strip():
        fasta_text = pasted.strip()
        if not fasta_text.startswith(">"):
            fasta_text = f">pasted_sequence\n{fasta_text}"
    elif example_choice != "None":
        example_path = os.path.join(_ROOT, "example_data", example_choice)
        if os.path.isfile(example_path):
            with open(example_path, encoding="utf-8", errors="replace") as fh:
                fasta_text = fh.read()

    # Preview loaded records
    if fasta_text:
        preview_records = parse_fasta(fasta_text)
        if preview_records:
            st.caption(
                f"✅ {len(preview_records)} sequence(s) loaded — "
                f"total {sum(len(s) for _, s in preview_records):,} bp"
            )

    # ── Analysis parameters ─────────────────────────────────────────────────
    # Defaults (always defined; overridden by widgets when expander is open)
    window = 10
    min_len = 50
    max_len = 300
    baseline_win = 200
    top_k = 5
    score_cutoff = -0.05
    active_motifs: set = set(MOTIF_LABELS.keys())

    with st.expander("⚙️  Analysis Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            window = st.slider("Perplexity Window (bp)", 4, 30, 10, 1)
            min_len = st.slider("Min Region Length (bp)", 10, 200, 50, 5)
        with c2:
            max_len = st.slider("Max Region Length (bp)", 50, 1000, 300, 10)
            baseline_win = st.slider("Baseline Window (bp)", 50, 500, 200, 10)
        with c3:
            top_k = st.slider("Max Ranked Regions", 1, 20, 5, 1)
            score_cutoff = st.slider(
                "Residual Score Threshold", -1.0, 0.0, -0.05, 0.01,
                format="%.2f",
            )
        st.markdown("**Non-B DNA Motif Selection**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        motif_sel: Dict[str, bool] = {}
        motif_keys = list(MOTIF_LABELS.keys())
        per_col = max(1, (len(motif_keys) + 3) // 4)
        for idx, key in enumerate(motif_keys):
            col_idx = idx // per_col
            col = [mc1, mc2, mc3, mc4][min(col_idx, 3)]
            motif_sel[key] = col.checkbox(
                MOTIF_LABELS[key], value=True, key=f"m2_{key}"
            )
        active_motifs = {k for k, v in motif_sel.items() if v}

    # ── Run button ──────────────────────────────────────────────────────────
    run_btn = st.button(
        "▶  Run PERCALL Analysis", type="primary", use_container_width=True
    )

    current_params = {
        "window": window, "min_len": min_len, "max_len": max_len,
        "baseline_win": baseline_win, "top_k": top_k,
        "score_cutoff": score_cutoff, "active_motifs": active_motifs,
        "fasta_text": fasta_text,
    }

    if run_btn:
        if not fasta_text:
            st.warning("Please upload, paste, or select a sequence first.")
        else:
            records = parse_fasta(fasta_text)
            if not records:
                st.error("No valid sequences found. Check the FASTA format.")
            else:
                progress = st.progress(0, text="Analysing sequences…")
                computed: List[dict] = []
                for idx, (hdr, seq) in enumerate(records):
                    if len(seq) < min_len + window:
                        computed.append({
                            "header": hdr, "seq": seq,
                            "perp": np.array([]), "baseline": np.array([]),
                            "residual": np.array([]),
                            "regions": [], "skipped": True,
                        })
                    else:
                        computed.append(process_sequence(
                            hdr, seq,
                            window=window, baseline_win=baseline_win,
                            min_len=min_len, max_len=max_len,
                            top_k=top_k, score_cutoff=score_cutoff,
                            active_motifs=active_motifs,
                        ))
                    progress.progress(
                        (idx + 1) / len(records),
                        text=f"Processed {idx + 1}/{len(records)} sequences",
                    )
                progress.empty()
                st.session_state["percall_results"] = computed
                st.session_state["percall_params"] = current_params
                st.success(
                    f"Analysis complete — {len(computed)} sequence(s) processed."
                )

    # ── Results display ─────────────────────────────────────────────────────
    results: List[dict] = st.session_state.get("percall_results", [])
    stored_params: dict = st.session_state.get("percall_params", current_params)
    valid = _valid_results(results)

    if not results:
        return

    # B. Sequence statistics
    _section_header("B · Sequence Statistics")
    total_bp = sum(len(r["seq"]) for r in results)
    mean_gc = float(np.mean([gc_pct(r["seq"]) for r in results])) if results else 0.0
    total_n = sum(r["seq"].count("N") for r in results)
    total_reg = sum(len(r["regions"]) for r in valid)
    _metric_strip([
        ("Sequences", str(len(results))),
        ("Total bp", f"{total_bp:,}"),
        ("Mean GC%", f"{mean_gc:.1f}%"),
        ("Ambiguous N", f"{total_n:,}"),
        ("Regions called", str(total_reg)),
    ])

    seq_labels = [r["header"][:80] for r in results]
    sel_label = st.selectbox(
        "Inspect sequence", seq_labels, key="seq_stat_sel"
    )
    sel_rec = results[seq_labels.index(sel_label)]
    sel_seq = sel_rec["seq"]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_plot_composition(sel_seq), use_container_width=True)
    with c2:
        st.plotly_chart(_plot_gc_heatmap(sel_seq), use_container_width=True)

    # GC profile if analysis was run
    if not sel_rec["skipped"] and sel_rec["perp"].size > 0:
        st.plotly_chart(
            _plot_gc_profile(
                sel_seq, sel_rec["regions"],
                window=max(stored_params.get("window", 10) * 5, 50),
            ),
            use_container_width=True,
        )

    # C. Perplexity Explorer
    if not valid:
        _empty_state(
            "Perplexity data not available",
            "All sequences were skipped (too short). "
            "Reduce Min Region Length or use longer sequences.",
        )
        return

    _section_header("C · Perplexity Explorer")
    rv = _select_result(results, "perp_seq_sel", "Sequence for perplexity analysis")
    if rv is None:
        return

    st.plotly_chart(
        _plot_perplexity(
            rv["perp"], rv["baseline"], rv["residual"], rv["regions"],
            title=f"Perplexity Profile — {rv['header'][:55]}",
        ),
        use_container_width=True,
    )
    c1, c2 = st.columns([1.3, 0.7])
    with c1:
        _metric_strip([
            ("Mean perplexity", f"{np.nanmean(rv['perp']):.3f}"),
            ("Min perplexity", f"{np.nanmin(rv['perp']):.3f}"),
            ("Signal length", f"{len(rv['perp']):,}"),
            ("Regions called", str(len(rv["regions"]))),
        ])
    with c2:
        st.plotly_chart(_plot_perp_density(rv["perp"]), use_container_width=True)

    if len(valid) > 1:
        st.plotly_chart(_plot_comparative(valid), use_container_width=True)

    # D. Residual Analysis
    _section_header("D · Residual Analysis")
    st.plotly_chart(_plot_signal_triptych(rv), use_container_width=True)
    _metric_strip([
        ("Baseline mean", f"{np.nanmean(rv['baseline']):.3f}"),
        ("Residual minimum", f"{np.nanmin(rv['residual']):.4f}"),
        ("Residual median", f"{np.nanmedian(rv['residual']):.4f}"),
        ("Trough count", str(len(rv["regions"]))),
    ])


# ============================================================================
# Page 3 — PERCALL Region Caller
# ============================================================================


def _page_region_caller() -> None:
    _page_header(
        "Page 03",
        "PERCALL Region Caller",
        "Flagship bounded minimum-mean Kadane region-calling environment. "
        "Explore region architecture maps, ranked domains, trough positions, "
        "and interactive region inspection.",
    )

    results: List[dict] = st.session_state.get("percall_results", [])
    valid = _valid_results(results)

    if not valid:
        _empty_state(
            "Region caller awaiting analysis",
            "Run PERCALL from the Sequence & Perplexity tab to populate "
            "ranked regulatory regions and their optimisation-derived architecture.",
        )
        return

    # Highlight innovation
    st.markdown(
        '<div class="hl-card">'
        '<h4>PERplexity-based Regulatory Region CALLer — Core Innovation</h4>'
        '<p>PERCALL applies a bounded minimum-mean contiguous-subarray algorithm '
        '(Kadane family, O(n)) to the perplexity-residual signal, identifying the '
        'statistically optimal low-perplexity regions under user-defined length constraints. '
        'Called domains are iteratively ranked by trough depth, with each successive '
        'region found after masking the previous.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    rv = _select_result(results, "caller_seq_sel", "Sequence for region caller")
    if rv is None:
        return

    # Overview metrics
    _metric_strip([
        ("Sequence length", f"{len(rv['seq']):,} bp"),
        ("Regions called", str(len(rv["regions"]))),
        ("Deepest trough", f"{min((r['mean_residual'] for r in rv['regions']), default=0):.4f}"),
        ("GC% overall", f"{gc_pct(rv['seq']):.1f}%"),
    ])

    # Main perplexity profile
    _section_header("Region Signal Profile")
    st.plotly_chart(
        _plot_perplexity(
            rv["perp"], rv["baseline"], rv["residual"], rv["regions"],
            title=f"PERCALL Region Signal — {rv['header'][:55]}",
        ),
        use_container_width=True,
    )

    # Region architecture
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        _section_header("Region Architecture Map")
        st.plotly_chart(
            _ltheme(
                plot_sequence_domain_map(
                    len(rv["seq"]), rv["regions"],
                    title="Region Architecture Map",
                )
            ),
            use_container_width=True,
        )
    with c2:
        _section_header("Region Depth Ranking")
        st.plotly_chart(_plot_region_ranking(rv), use_container_width=True)

    # Region explorer
    if rv["regions"]:
        _section_header("Region Explorer")
        opts = [
            f"R{r['rank']} · {r['start']}–{r['end']} bp · {r['width']} bp"
            for r in rv["regions"]
        ]
        chosen = st.selectbox("Inspect region", opts, key="region_inspector")
        reg = rv["regions"][opts.index(chosen)]
        _metric_strip([
            ("Start", str(reg["start"])),
            ("End", str(reg["end"])),
            ("Width", f"{reg['width']} bp"),
            ("Trough position", str(reg["trough"])),
            ("Mean residual", f"{reg['mean_residual']:.4f}"),
            ("GC%", f"{reg['gc_pct']}%"),
        ])
        st.markdown(
            '<div class="sci-card">'
            f'<div class="info-pill">Rank R{reg["rank"]}</div>'
            f'<div class="info-pill">Trough @ {reg["trough"]}</div>'
            f'<div class="info-pill">GC% {reg["gc_pct"]}</div>'
            f'<div class="info-pill">Motifs: {reg["motifs"] or "None detected"}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Full regions table
    df = _regions_df(results)
    if not df.empty:
        _section_header("Regulatory Region Ranking — All Sequences")
        st.dataframe(
            df.style.background_gradient(subset=["Mean Residual"], cmap="Blues_r"),
            use_container_width=True,
            hide_index=True,
        )

        # Search / filter
        _section_header("Region Filter")
        cf1, cf2, cf3 = st.columns([1.1, 0.9, 0.8])
        with cf1:
            search = st.text_input(
                "Search sequence label", placeholder="Fragment of sequence name",
                key="rc_search",
            )
        with cf2:
            seqs = ["All"] + sorted(df["Sequence"].unique().tolist())
            sf = st.selectbox("Filter by sequence", seqs, key="rc_seq_filter")
        with cf3:
            mf = st.text_input(
                "Filter by motif", placeholder="G4, ZDNA, …", key="rc_motif_filter"
            )

        filtered = df.copy()
        if search:
            filtered = filtered[
                filtered["Sequence"].str.contains(search, case=False, na=False)
            ]
        if sf != "All":
            filtered = filtered[filtered["Sequence"] == sf]
        if mf:
            filtered = filtered[
                filtered["Motifs"].str.contains(mf, case=False, na=False)
            ]

        st.caption(f"Showing **{len(filtered)}** of **{len(df)}** regions")
        if not filtered.empty:
            st.dataframe(
                filtered.style.background_gradient(
                    subset=["Mean Residual"], cmap="Blues_r"
                ),
                use_container_width=True,
                hide_index=True,
            )

        # Scatter
        _section_header("Region Width vs Depth")
        st.plotly_chart(_plot_region_scatter(df), use_container_width=True)

        # Distribution
        st.plotly_chart(
            _ltheme(plot_region_distribution(df), height=340),
            use_container_width=True,
        )

    # Algorithmic explanation
    _section_header("Algorithm Transparency")
    cols = st.columns(4)
    steps = [
        ("Phase 1", "Residual Segmentation",
         "The residual signal is segmented at NaN boundaries; each finite run is "
         "searched independently."),
        ("Phase 2", "Bounded Optimisation",
         "Minimum-mean subarray search under min_len ≤ length ≤ max_len "
         "via a monotonic deque — provably O(n)."),
        ("Phase 3", "Iterative Ranking",
         "The best-scoring span is recorded, masked to NaN, and the scan "
         "repeats for up to top_k regions."),
        ("Phase 4", "Structural Context",
         "Each region is annotated with GC% and screened against the "
         "supplied Non-B DNA motif layer."),
    ]
    for col, (phase, title, body) in zip(cols, steps):
        col.markdown(
            f'<div class="sci-card">'
            f'<span class="page-kicker">{phase}</span>'
            f'<strong style="display:block;color:#0F172A;margin:.2rem 0 .35rem">'
            f'{title}</strong>'
            f'<p style="color:#64748B;font-size:.84rem;line-height:1.55;margin:0">'
            f'{body}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# Page 4 — Non-B DNA Structure Analysis
# ============================================================================


def _page_nonb_dna() -> None:
    _page_header(
        "Page 04",
        "Non-B DNA Structure Analysis",
        "Structural genomics interpretation for PERCALL-called domains. "
        "Motif enrichment, positional distribution, region overlap maps, "
        "and interactive non-B DNA explorer.",
    )

    results: List[dict] = st.session_state.get("percall_results", [])
    stored_params: dict = st.session_state.get("percall_params", {})
    valid = _valid_results(results)
    default_active = set(MOTIF_LABELS.keys())
    active_motifs: set = stored_params.get("active_motifs", default_active)
    window: int = stored_params.get("window", 10)

    if not valid:
        _empty_state(
            "Structural analysis awaiting data",
            "Run PERCALL from the Sequence & Perplexity tab to populate "
            "enrichment plots, motif tracks, and regional motif summaries.",
        )
        return

    # Motif selector
    _section_header("Motif Selection")
    with st.expander("Configure motifs for this page", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        page_motif_sel: Dict[str, bool] = {}
        motif_keys = list(MOTIF_LABELS.keys())
        per_col = max(1, (len(motif_keys) + 3) // 4)
        for idx, key in enumerate(motif_keys):
            col_idx = idx // per_col
            col = [mc1, mc2, mc3, mc4][min(col_idx, 3)]
            page_motif_sel[key] = col.checkbox(
                MOTIF_LABELS[key],
                value=(key in active_motifs),
                key=f"m4_{key}",
            )
        active_motifs = {k for k, v in page_motif_sel.items() if v}

    ordered = [m for m in MOTIF_LABELS if m in active_motifs]

    if not ordered:
        st.warning("Enable at least one motif above to view structural analysis.")
        return

    # Compute counts
    total_seq_bp = sum(len(r["seq"]) for r in valid)
    total_region_bp = sum(
        reg["width"] for r in valid for reg in r["regions"]
    )
    bg_counts: Dict[str, int] = {m: 0 for m in ordered}
    reg_counts: Dict[str, int] = {m: 0 for m in ordered}
    motif_rows: List[dict] = []

    for r in valid:
        bg = count_motifs(r["seq"], active_motifs=set(ordered))
        for motif in ordered:
            bg_counts[motif] += bg.get(motif, 0)
        for reg in r["regions"]:
            seq_piece = _region_seq(r["seq"], reg, window)
            rc = count_motifs(seq_piece, active_motifs=set(ordered))
            for motif in ordered:
                reg_counts[motif] += rc.get(motif, 0)
                if rc.get(motif, 0) > 0:
                    motif_rows.append({
                        "Sequence": r["header"][:50],
                        "Region Rank": reg["rank"],
                        "Motif": MOTIF_LABELS.get(motif, motif),
                        "Count in Region": rc[motif],
                    })

    # Supported motifs info
    st.markdown(
        '<div class="sci-card">'
        '<div class="info-pill">🔴 G-Quadruplex (G4)</div>'
        '<div class="info-pill">🟢 i-Motif</div>'
        '<div class="info-pill">🔵 Z-DNA</div>'
        '<div class="info-pill">🟠 eGZ-DNA</div>'
        '<div class="info-pill">🟣 Triplex</div>'
        '<div class="info-pill">🩵 STR</div>'
        '<div class="info-pill">🩷 Direct Repeat</div>'
        '<div class="info-pill">🟡 PolyA/T</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Enrichment plots
    _section_header("Motif Enrichment — Regions vs Background")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _ltheme(
                plot_motif_enrichment(
                    reg_counts, bg_counts,
                    total_region_bp, total_seq_bp, ordered,
                ),
                height=380,
            ),
            use_container_width=True,
        )
    with c2:
        r0 = valid[0]
        st.plotly_chart(
            _ltheme(
                plot_motif_distribution(
                    scan_motifs(r0["seq"], active_motifs=set(ordered)),
                    len(r0["seq"]), ordered,
                ),
                height=380,
            ),
            use_container_width=True,
        )

    # Motif-region heatmap
    _section_header("Motif Signal Across Called Regions")
    st.plotly_chart(
        _plot_motif_heatmap(results, active_motifs, window),
        use_container_width=True,
    )

    # Genome browser
    _section_header("Interactive Genome Viewer")
    rv = _select_result(results, "nonb_seq_sel", "Sequence for genome viewer")
    if rv is not None:
        motif_positions = scan_motifs(rv["seq"], active_motifs=set(ordered))
        st.plotly_chart(
            _ltheme(
                plot_genome_browser(
                    seq_len=len(rv["seq"]),
                    regions=rv["regions"],
                    motif_positions=motif_positions,
                    active_motifs=ordered,
                    seq_label=rv["header"][:40],
                ),
                height=420,
            ),
            use_container_width=True,
        )

    # Summary table
    if motif_rows:
        _section_header("Motif Hits Within Called Regions")
        st.dataframe(
            pd.DataFrame(motif_rows), use_container_width=True, hide_index=True
        )

    # Structural motif overview
    _section_header("Structural Motif Definitions")
    defs = [
        ("G-Quadruplex (G4)",
         "Four G-tracts of ≥3 Gs with short loops (1–7 nt). "
         "Balasubramanian lab consensus pattern."),
        ("i-Motif",
         "Four C-tracts of ≥3 Cs with short loops. "
         "Complement of G4; forms in acidic conditions."),
        ("Z-DNA",
         "Alternating purine-pyrimidine repeats CG/GC/CA/TG ≥4 units. "
         "Rich et al. (1984) approximation."),
        ("eGZ-DNA",
         "Expanded G/Z-DNA within CGG/GGC trinucleotide repeats ≥4 units. "
         "Fakharzadeh et al. (2022)."),
        ("Triplex",
         "Purine (AG) mirror repeats ≥10 bp; triplex-forming oligonucleotides."),
        ("STR",
         "Short Tandem Repeats — 1–6 bp repeat unit with ≥4 copies."),
        ("Direct Repeat",
         "Tandem direct repeat pairs — 4–10 bp unit with ≤10 bp gap."),
        ("PolyA/T",
         "Homopolymeric A or T runs of ≥7 consecutive bases."),
    ]
    cols = st.columns(4)
    for idx, (name, description) in enumerate(defs):
        col = cols[idx % 4]
        col.markdown(
            f'<div class="sci-card" style="min-height:100px">'
            f'<strong style="display:block;color:#0F172A;margin-bottom:.3rem;font-size:.9rem">'
            f'{name}</strong>'
            f'<p style="color:#64748B;font-size:.82rem;line-height:1.5;margin:0">'
            f'{description}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# Page 5 — Reports & Exports
# ============================================================================


def _page_reports() -> None:
    _page_header(
        "Page 05",
        "Reports & Exports",
        "Publication-quality figure generation, structured data exports, "
        "and one-click PDF report for the current PERCALL session.",
    )

    results: List[dict] = st.session_state.get("percall_results", [])
    stored_params: dict = st.session_state.get("percall_params", {})
    valid = _valid_results(results)

    if not valid:
        _empty_state(
            "Reports & Exports awaiting analysis",
            "Run PERCALL from the Sequence & Perplexity tab to enable figure export, "
            "data tables, session packages, and PDF report generation.",
        )
        return

    rv = _select_result(results, "reports_seq_sel", "Select sequence for figures")
    if rv is None:
        return

    summary_df = _summary_df(results)
    regions_df = _regions_df(results)

    # Publication figure
    _section_header("Publication-Ready Figure")
    pub_fig = _plot_perplexity(
        rv["perp"], rv["baseline"], rv["residual"], rv["regions"],
        title=f"PERCALL — {rv['header'][:55]}",
    )
    st.plotly_chart(pub_fig, use_container_width=True)

    # Region architecture figure
    arch_fig = _ltheme(
        plot_sequence_domain_map(
            len(rv["seq"]), rv["regions"],
            title=f"Region Architecture — {rv['header'][:40]}",
        )
    )
    st.plotly_chart(arch_fig, use_container_width=True)

    # Data tables
    _section_header("Data Tables")
    t1, t2 = st.tabs(["Sequence Summary", "Region Details"])
    with t1:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with t2:
        if not regions_df.empty:
            st.dataframe(
                regions_df.style.background_gradient(
                    subset=["Mean Residual"], cmap="Blues_r"
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No regions to display.")

    # Data exports
    _section_header("Data Exports")
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    with ec1:
        st.download_button(
            "📄 CSV Summary",
            _df_to_csv(summary_df), "percall_summary.csv", "text/csv",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "📊 Excel Summary",
            _df_to_excel(summary_df), "percall_summary.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with ec3:
        st.download_button(
            "📄 CSV Regions",
            _df_to_csv(regions_df), "percall_regions.csv", "text/csv",
            use_container_width=True,
        )
    with ec4:
        st.download_button(
            "📊 Excel Regions",
            _df_to_excel(regions_df), "percall_regions.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with ec5:
        st.download_button(
            "🗂 Session JSON",
            _results_to_json(results, stored_params),
            "percall_session.json", "application/json",
            use_container_width=True,
        )

    # Image exports
    _section_header("Figure Image Exports")
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(pub_fig, format="png", width=1400, height=620, scale=2)
        svg_bytes = pio.to_image(pub_fig, format="svg", width=1400, height=620, scale=2)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.download_button(
                "🖼 High-Resolution PNG",
                png_bytes, "percall_profile.png", "image/png",
                use_container_width=True,
            )
        with ic2:
            st.download_button(
                "🎨 Vector SVG",
                svg_bytes, "percall_profile.svg", "image/svg+xml",
                use_container_width=True,
            )
    except Exception:
        st.caption(
            "Install kaleido (pip install kaleido) to enable PNG/SVG export."
        )

    # PDF report
    _section_header("PDF Publication Report")
    st.markdown(
        '<div class="hl-card">'
        '<h4>One-Click PDF Generation</h4>'
        '<p>Generates a publication-ready PDF containing the perplexity profile, '
        'region table, motif enrichment, statistical summary, and full analysis '
        'metadata for the current PERCALL session.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("📄  Generate Publication PDF", type="primary"):
        try:
            from core.report import generate_pdf

            ordered = [m for m in MOTIF_LABELS
                       if m in stored_params.get("active_motifs", set())]
            window = stored_params.get("window", 10)
            motif_counts: Dict[str, int] = {m: 0 for m in ordered}
            bg_counts: Dict[str, int] = {m: 0 for m in ordered}
            reg_counts: Dict[str, int] = {m: 0 for m in ordered}
            total_region_bp = 0
            total_seq_bp = 0

            for r in valid:
                mc = count_motifs(r["seq"], active_motifs=set(ordered))
                total_seq_bp += len(r["seq"])
                for motif in ordered:
                    motif_counts[motif] += mc.get(motif, 0)
                    bg_counts[motif] += mc.get(motif, 0)
                for reg in r["regions"]:
                    total_region_bp += reg["width"]
                    rc = count_motifs(
                        _region_seq(r["seq"], reg, window),
                        active_motifs=set(ordered),
                    )
                    for motif in ordered:
                        reg_counts[motif] += rc.get(motif, 0)

            pdf_bytes = generate_pdf(
                params=stored_params,
                seq_stats=[
                    {
                        "header": r["header"][:60],
                        "length": len(r["seq"]),
                        "gc_pct": gc_pct(r["seq"]),
                        "n_regions": len(r["regions"]),
                    }
                    for r in results
                ],
                regions_df_records=regions_df.to_dict("records"),
                motif_counts=motif_counts,
                perp=rv["perp"],
                baseline=rv["baseline"],
                residual=rv["residual"],
                regions=rv["regions"],
                region_counts=reg_counts,
                background_counts=bg_counts,
                total_region_bp=total_region_bp,
                total_seq_bp=total_seq_bp,
                active_motifs=ordered,
                motif_labels=MOTIF_LABELS,
                seq_label=rv["header"][:60],
            )
            st.success("PDF report generated successfully.")
            st.download_button(
                "⬇ Download PERCALL_Report.pdf",
                data=pdf_bytes,
                file_name="PERCALL_Report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    # Statistical summary
    _section_header("Statistical Summary")
    if not regions_df.empty:
        numeric_cols = ["Width", "Mean Residual", "GC%"]
        stat_cols = [c for c in numeric_cols if c in regions_df.columns]
        if stat_cols:
            st.dataframe(
                regions_df[stat_cols].describe().round(4),
                use_container_width=True,
            )

    # Citation
    _section_header("Citation")
    st.code(
        textwrap.dedent("""\
            Yella VR (2025). PERCALL: PERplexity-based Regulatory Region CALLer.
            An information-theoretic framework for identifying low-perplexity regulatory
            DNA regions using bounded minimum-mean Kadane optimisation.
            GitHub: https://github.com/VRYella/PerCALL
        """),
        language=None,
    )


# ============================================================================
# Application entry point
# ============================================================================


def main() -> None:
    # Brand bar (above tabs)
    st.markdown(_BRAND_BAR, unsafe_allow_html=True)

    # Five-page horizontal navigation via st.tabs
    (
        tab_home,
        tab_seq,
        tab_region,
        tab_nonb,
        tab_reports,
    ) = st.tabs([
        "🏠  Home",
        "🧬  Sequence & Perplexity",
        "📍  Region Caller",
        "🔬  Non-B DNA",
        "📊  Reports & Exports",
    ])

    results: List[dict] = st.session_state.get("percall_results", [])

    with tab_home:
        _page_home(results)

    with tab_seq:
        _page_sequence_perplexity()

    with tab_region:
        _page_region_caller()

    with tab_nonb:
        _page_nonb_dna()

    with tab_reports:
        _page_reports()


if __name__ == "__main__":
    main()
