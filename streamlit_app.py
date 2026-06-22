"""
streamlit_app.py
────────────────
PERCALL — PERplexity-based Regulatory Region CALLer
===================================================

A premium scientific Streamlit platform for exploring perplexity-derived
regulatory regions without altering the original PERCALL computational core.
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.motifs import MOTIF_LABELS, count_motifs, scan_motifs
from core.perplexity import compute_perplexity, local_residual
from core.plotting import (
    plot_gc_profile,
    plot_genome_browser,
    plot_motif_distribution,
    plot_motif_enrichment,
    plot_perplexity_profile,
    plot_region_distribution,
    plot_sequence_domain_map,
)
from core.region_caller import find_regions

# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="PERCALL — Scientific Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Design system
# ============================================================================

_DARK_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent !important;}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(0, 212, 255, 0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(0, 255, 157, 0.10), transparent 24%),
        linear-gradient(180deg, #050a14 0%, #091223 42%, #060b18 100%);
    color: #e6edf7;
}

section[data-testid="stSidebar"] {
    background: rgba(4, 10, 22, 0.92) !important;
    border-right: 1px solid rgba(86, 186, 255, 0.18) !important;
    backdrop-filter: blur(18px);
}

.main .block-container {
    max-width: 1420px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

@keyframes fadeRise {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(0, 212, 255, 0.0); }
    50% { box-shadow: 0 0 24px rgba(0, 212, 255, 0.22); }
}

@keyframes driftFwd {
    from { stroke-dashoffset: 0; }
    to { stroke-dashoffset: -76; }
}

@keyframes driftBack {
    from { stroke-dashoffset: 0; }
    to { stroke-dashoffset: 76; }
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    font-family: Inter, "IBM Plex Sans", system-ui, sans-serif !important;
}

h1 {
    color: #f5fbff !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem !important;
}

h2, h3 {
    color: #dff7ff !important;
}

.page-kicker {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #6bcfff;
    margin-bottom: 0.35rem;
    font-weight: 700;
}

.page-subtitle {
    color: #90a7c2;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 980px;
    margin-bottom: 1.2rem;
}

.nav-status {
    margin: 0.8rem 0 1.2rem 0;
    padding: 1rem 1.15rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(107, 207, 255, 0.18);
    border-radius: 18px;
    backdrop-filter: blur(18px);
    animation: fadeRise 0.35s ease-out;
}

.nav-status strong {
    color: #f4fbff;
    font-size: 1.02rem;
}

.nav-status span {
    display: block;
    color: #87a0bc;
    margin-top: 0.15rem;
    font-size: 0.92rem;
}

.glass-card,
.story-card,
.workflow-step,
.feature-card,
.empty-state {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
    border: 1px solid rgba(95, 189, 255, 0.16);
    border-radius: 20px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.22);
    animation: fadeRise 0.35s ease-out;
}

.glass-card {
    padding: 1.25rem 1.35rem;
    margin: 0.55rem 0;
}

.story-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 1.3rem 0;
}

.story-card, .feature-card {
    padding: 1rem 1.05rem;
}

.story-card h4,
.feature-card h4 {
    margin: 0 0 0.45rem 0 !important;
    color: #f4fbff !important;
    font-size: 1rem !important;
}

.story-card p,
.feature-card p {
    margin: 0;
    color: #91a7c2;
    font-size: 0.9rem;
    line-height: 1.6;
}

.feature-meta {
    display: inline-block;
    margin-bottom: 0.55rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #6bcfff;
    font-weight: 700;
}

.workflow-ribbon {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0 1.4rem 0;
}

.workflow-step {
    padding: 1rem 0.95rem;
    min-height: 118px;
    position: relative;
}

.workflow-step::after {
    content: "→";
    position: absolute;
    right: -0.55rem;
    top: calc(50% - 0.8rem);
    color: rgba(107, 207, 255, 0.55);
    font-size: 1.2rem;
}

.workflow-ribbon .workflow-step:last-child::after {
    display: none;
}

.workflow-step small {
    color: #6bcfff;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.68rem;
    font-weight: 700;
}

.workflow-step strong {
    display: block;
    color: #f4fbff;
    font-size: 1rem;
    margin: 0.35rem 0 0.45rem 0;
}

.workflow-step p {
    margin: 0;
    color: #90a7c2;
    font-size: 0.88rem;
    line-height: 1.55;
}

.percall-hero {
    text-align: center;
    padding: 1.3rem 0 0.5rem 0;
}

.percall-logo {
    font-size: 4.4rem;
    font-weight: 900;
    letter-spacing: 0.22em;
    line-height: 1;
    margin-bottom: 0.55rem;
    background: linear-gradient(135deg, #6ee7ff 0%, #00d4ff 45%, #00ff9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 38px rgba(0, 212, 255, 0.18);
}

.percall-full-name {
    color: #9ab3ce;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}

.percall-badges {
    display: flex;
    justify-content: center;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.3rem 0;
}

.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.36rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.badge-blue {
    background: rgba(0, 212, 255, 0.14);
    border: 1px solid rgba(0, 212, 255, 0.26);
    color: #6ee7ff;
}

.badge-emerald {
    background: rgba(0, 255, 157, 0.12);
    border: 1px solid rgba(0, 255, 157, 0.24);
    color: #86ffcd;
}

.badge-amber {
    background: rgba(227, 179, 65, 0.12);
    border: 1px solid rgba(227, 179, 65, 0.22);
    color: #ffd98b;
}

.section-header {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-weight: 800;
    color: #6bcfff;
    margin: 1.3rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(107, 207, 255, 0.22);
}

.info-pill {
    display: inline-block;
    margin: 0.18rem;
    padding: 0.5rem 0.78rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(107, 207, 255, 0.14);
    border-radius: 12px;
    color: #d8e7f8;
    font-size: 0.86rem;
}

.empty-state {
    padding: 1.3rem 1.35rem;
    margin: 0.8rem 0 1.2rem 0;
}

.empty-state strong {
    display: block;
    font-size: 1rem;
    color: #eff8ff;
    margin-bottom: 0.35rem;
}

.empty-state span {
    color: #8ea5bf;
    line-height: 1.6;
    font-size: 0.92rem;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin: 0.9rem 0 1.1rem 0;
}

.metric-strip .feature-card {
    padding: 0.95rem 1rem;
}

.metric-strip .value {
    display: block;
    color: #f4fbff;
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 0.18rem;
}

.metric-strip .label {
    color: #7fa1c1;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 700;
}

.dna-s1 {
    stroke-dasharray: 10 5;
    animation: driftFwd 2.6s linear infinite;
}

.dna-s2 {
    stroke-dasharray: 10 5;
    animation: driftBack 2.6s linear infinite;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(95, 189, 255, 0.16) !important;
    border-radius: 16px !important;
    padding: 1rem 0.95rem !important;
    animation: glowPulse 3.8s ease-in-out infinite;
}

div[data-testid="metric-container"] label {
    color: #85a0be !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem !important;
}

div[data-testid="metric-container"] div {
    color: #f4fbff !important;
}

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    border-radius: 14px !important;
    border: 1px solid rgba(107, 207, 255, 0.20) !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03)) !important;
    color: #edf7ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    min-height: 2.8rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(107, 207, 255, 0.42) !important;
    box-shadow: 0 0 18px rgba(0, 212, 255, 0.14) !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff 0%, #00ff9d 100%) !important;
    color: #04101d !important;
    border: none !important;
}

.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stTextArea textarea,
.stTextInput input,
.stFileUploader > div,
.stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    color: #edf7ff !important;
    border: 1px solid rgba(107, 207, 255, 0.16) !important;
    border-radius: 14px !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.16);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 0.3rem;
    gap: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 14px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.18), rgba(0,255,157,0.12)) !important;
    color: #eff9ff !important;
}

@media (max-width: 900px) {
    .percall-logo {
        font-size: 3rem;
        letter-spacing: 0.12em;
    }

    .workflow-step::after {
        display: none;
    }
}
</style>
"""
st.markdown(_DARK_CSS, unsafe_allow_html=True)

# ============================================================================
# DNA animation
# ============================================================================

_DNA_ANIMATION = """
<div style="text-align:center;padding:0.2rem 0 1rem 0">
<svg viewBox="0 0 640 62" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:660px;height:62px;overflow:visible">
  <path class="dna-s1"
    d="M0,31 Q40,6 80,31 Q120,56 160,31 Q200,6 240,31 Q280,56 320,31
       Q360,6 400,31 Q440,56 480,31 Q520,6 560,31 Q600,56 640,31"
    fill="none" stroke="#00d4ff" stroke-width="2.8" opacity="0.88"/>
  <path class="dna-s2"
    d="M0,31 Q40,56 80,31 Q120,6 160,31 Q200,56 240,31 Q280,6 320,31
       Q360,56 400,31 Q440,6 480,31 Q520,56 560,31 Q600,6 640,31"
    fill="none" stroke="#00ff9d" stroke-width="2.8" opacity="0.88"/>
  <line x1="80" y1="27" x2="80" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
  <line x1="160" y1="27" x2="160" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
  <line x1="240" y1="27" x2="240" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
  <line x1="320" y1="27" x2="320" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
  <line x1="400" y1="27" x2="400" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
  <line x1="480" y1="27" x2="480" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
  <line x1="560" y1="27" x2="560" y2="35" stroke="rgba(255,255,255,0.28)" stroke-width="1.6"/>
</svg>
</div>
"""

PAGE_CONFIG = [
    ("Home", "🏠 Home"),
    ("Sequence Workbench", "🧬 Sequence Workbench"),
    ("Perplexity Explorer", "📈 Perplexity Explorer"),
    ("Baseline & Residual Analysis", "📉 Baseline & Residual"),
    ("PERCALL Region Caller", "🗺 PERCALL Region Caller"),
    ("Regulatory Domain Explorer", "🧭 Regulatory Domains"),
    ("Non-B DNA Structure Center", "🧪 Non-B DNA Center"),
    ("Interactive Genome Viewer", "🖥 Interactive Genome Viewer"),
    ("Statistics & Discovery Center", "📊 Discovery Center"),
    ("Publication Studio", "📑 Publication Studio"),
    ("Methods & Algorithm", "∑ Methods & Algorithm"),
    ("About PERCALL", "ℹ️ About PERCALL"),
]

PAGE_DESCRIPTIONS = {
    "Home": "Scientific overview, design narrative, and launch pathway.",
    "Sequence Workbench": "Sequence ingestion, quality control, composition, and architecture.",
    "Perplexity Explorer": "Interactive perplexity profiling and comparative sequence analytics.",
    "Baseline & Residual Analysis": "Signal construction, baseline compensation, and residual inspection.",
    "PERCALL Region Caller": "Flagship bounded minimum-mean Kadane region-calling environment.",
    "Regulatory Domain Explorer": "Filter, search, and inspect detected regulatory domains.",
    "Non-B DNA Structure Center": "Motif enrichment, positional mapping, and structure-aware interpretation.",
    "Interactive Genome Viewer": "Track-based visual inspection of sequence, regions, and motifs.",
    "Statistics & Discovery Center": "Aggregate analytics, correlations, and discovery summaries.",
    "Publication Studio": "Export publication-grade figures, tables, and reports.",
    "Methods & Algorithm": "Mathematical transparency for the supplied PERCALL algorithm.",
    "About PERCALL": "Citation, software identity, credits, and repository context.",
}


# ============================================================================
# Utility helpers
# ============================================================================


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    """Parse FASTA-format text into [(header, sequence)] pairs."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = "".join(parts).upper()
                seq = re.sub(r"[^ACGTN]", "N", seq)
                records.append((header, seq))
            header = line[1:]
            parts = []
        else:
            parts.append(line)
    if header is not None:
        seq = "".join(parts).upper()
        seq = re.sub(r"[^ACGTN]", "N", seq)
        records.append((header, seq))
    return records


def gc_pct(seq: str) -> float:
    if not seq:
        return 0.0
    return round(100.0 * (seq.count("G") + seq.count("C")) / len(seq), 2)


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
    """Run the full PERCALL pipeline on one sequence."""
    perp = compute_perplexity(seq, window=window)
    if perp.size == 0 or np.all(np.isnan(perp)):
        return {
            "header": header,
            "seq": seq,
            "perp": perp,
            "baseline": perp.copy(),
            "residual": perp.copy(),
            "regions": [],
            "skipped": True,
        }

    res = local_residual(perp, baseline_win=baseline_win)
    baseline = (
        pd.Series(perp)
        .rolling(baseline_win, center=True, min_periods=baseline_win)
        .mean()
        .values
    )

    regions = find_regions(
        res,
        seq,
        min_len=min_len,
        max_len=max_len,
        top_k=top_k,
        score_cutoff=score_cutoff,
        window=window,
        active_motifs=active_motifs,
    )
    return {
        "header": header,
        "seq": seq,
        "perp": perp,
        "baseline": baseline,
        "residual": res,
        "regions": regions,
        "skipped": False,
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
            rows.append(
                {
                    "Sequence": r["header"][:60],
                    "Rank": reg["rank"],
                    "Start": reg["start"],
                    "End": reg["end"],
                    "Width": reg["width"],
                    "Trough": reg["trough"],
                    "Mean Residual": reg["mean_residual"],
                    "GC%": reg["gc_pct"],
                    "Motifs": reg["motifs"],
                }
            )
    return pd.DataFrame(rows)


def _summary_df(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Header": r["header"][:70],
                "Length (bp)": len(r["seq"]),
                "GC%": gc_pct(r["seq"]),
                "Regions": len(r["regions"]),
                "Status": "Skipped" if r["skipped"] else "OK",
            }
        )
    return pd.DataFrame(rows)


def _valid_results(results: List[dict]) -> List[dict]:
    return [r for r in results if not r["skipped"] and r["perp"].size > 0]


def _example_files() -> List[str]:
    ex_dir = os.path.join(_ROOT, "example_data")
    if not os.path.isdir(ex_dir):
        return []
    return sorted(
        [
            name for name in os.listdir(ex_dir)
            if name.lower().endswith((".fasta", ".fa", ".fna", ".txt"))
        ]
    )


def _preview_records(fasta_text: Optional[str]) -> List[Tuple[str, str]]:
    if not fasta_text:
        return []
    try:
        return parse_fasta(fasta_text)
    except Exception:
        return []


def _select_result(results: List[dict], key: str, label: str = "Sequence") -> Optional[dict]:
    valid = _valid_results(results)
    if not valid:
        return None
    labels = [r["header"][:80] for r in valid]
    if len(valid) == 1:
        return valid[0]
    chosen = st.selectbox(label, labels, key=key)
    return valid[labels.index(chosen)]


def _empty_state(title: str, message: str) -> None:
    st.markdown(
        f'<div class="empty-state"><strong>{title}</strong><span>{message}</span></div>',
        unsafe_allow_html=True,
    )


def _page_intro(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _hero_metric_strip(items: List[Tuple[str, str]]) -> None:
    html = ['<div class="metric-strip">']
    for label, value in items:
        html.append(
            '<div class="feature-card">'
            f'<span class="label">{label}</span>'
            f'<span class="value">{value}</span>'
            '</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _story_cards(cards: List[Tuple[str, str, str]]) -> None:
    html = ['<div class="story-grid">']
    for meta, title, body in cards:
        html.append(
            '<div class="story-card">'
            f'<span class="feature-meta">{meta}</span>'
            f'<h4>{title}</h4>'
            f'<p>{body}</p>'
            '</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _workflow_cards(cards: List[Tuple[str, str, str]]) -> None:
    html = ['<div class="workflow-ribbon">']
    for step, title, body in cards:
        html.append(
            '<div class="workflow-step">'
            f'<small>{step}</small>'
            f'<strong>{title}</strong>'
            f'<p>{body}</p>'
            '</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _region_rank_options(rv: dict) -> List[str]:
    return [
        f"R{reg['rank']} · {reg['start']}-{reg['end']} · {reg['width']} bp"
        for reg in rv["regions"]
    ]


def _plot_nucleotide_composition(seq: str) -> go.Figure:
    counts = {base: seq.count(base) for base in ["A", "C", "G", "T", "N"]}
    colors = ["#00d4ff", "#7dd3fc", "#00ff9d", "#e3b341", "#64748b"]
    fig = go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=colors,
            hovertemplate="%{x}: %{y} bases<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Nucleotide Composition", font=dict(color="#dce9f8")),
        margin=dict(l=50, r=20, t=60, b=40),
        height=280,
        showlegend=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


def _plot_gc_heatmap(seq: str, bins: int = 36) -> go.Figure:
    if not seq:
        values = np.array([[0.0]])
    else:
        chunk = max(1, len(seq) // bins)
        vals = []
        for i in range(0, len(seq), chunk):
            piece = seq[i: i + chunk]
            vals.append(gc_pct(piece))
        values = np.array([vals])
    fig = go.Figure(
        go.Heatmap(
            z=values,
            colorscale=[
                [0.0, "#0f172a"],
                [0.35, "#155e75"],
                [0.7, "#00d4ff"],
                [1.0, "#00ff9d"],
            ],
            hovertemplate="GC%: %{z:.1f}<extra></extra>",
            showscale=True,
            colorbar=dict(title="GC%"),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="GC Heatmap", font=dict(color="#dce9f8")),
        margin=dict(l=30, r=20, t=60, b=30),
        height=180,
        xaxis=dict(showgrid=False, title="Sequence bins"),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def _plot_perplexity_focus(rv: dict) -> go.Figure:
    positions = np.arange(len(rv["perp"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=rv["perp"],
            mode="lines",
            line=dict(color="#00d4ff", width=1.35),
            name="Perplexity",
            hovertemplate="Position %{x}<br>Perplexity %{y:.3f}<extra></extra>",
        )
    )
    for reg in rv["regions"]:
        fig.add_vrect(
            x0=reg["start"],
            x1=reg["end"],
            fillcolor="rgba(0,212,255,0.10)",
            line_color="rgba(0,212,255,0.55)",
            line_width=1.0,
            annotation_text=f"R{reg['rank']}",
            annotation_position="top left",
            annotation_font_color="#6ee7ff",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Perplexity Profile", font=dict(color="#dce9f8")),
        margin=dict(l=50, r=20, t=60, b=45),
        height=360,
        xaxis=dict(title="Position (bp)", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Perplexity", gridcolor="rgba(255,255,255,0.07)"),
        hovermode="x unified",
    )
    return fig


def _plot_perplexity_distribution(perp: np.ndarray) -> go.Figure:
    values = pd.Series(perp).dropna()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Density Distribution", "Position-wise Summary"),
        column_widths=[0.58, 0.42],
    )
    fig.add_trace(
        go.Histogram(
            x=values,
            marker_color="#00d4ff",
            opacity=0.8,
            nbinsx=40,
            hovertemplate="Perplexity %{x:.3f}<br>Count %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(
            y=values,
            marker_color="#00ff9d",
            boxmean=True,
            hovertemplate="Perplexity %{y:.3f}<extra></extra>",
            name="Perplexity",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", row=1, col=1)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", row=1, col=1)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", row=1, col=2)
    return fig


def _plot_comparative_perplexity(valid: List[dict]) -> go.Figure:
    if not valid:
        return go.Figure()
    labels = [r["header"][:28] for r in valid]
    means = [float(np.nanmean(r["perp"])) for r in valid]
    minima = [float(np.nanmin(r["perp"])) for r in valid]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=means,
            name="Mean perplexity",
            marker_color="#00d4ff",
            hovertemplate="%{x}<br>Mean %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=minima,
            mode="lines+markers",
            name="Minimum perplexity",
            line=dict(color="#e3b341", width=2),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Min %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Comparative Perplexity Across Sequences", font=dict(color="#dce9f8")),
        height=330,
        margin=dict(l=50, r=20, t=60, b=80),
        xaxis=dict(tickangle=-25, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Perplexity", gridcolor="rgba(255,255,255,0.07)"),
        hovermode="x unified",
    )
    return fig


def _plot_signal_triptych(rv: dict) -> go.Figure:
    x = np.arange(len(rv["perp"]))
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Perplexity", "Rolling Baseline", "Residual Signal"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=rv["perp"], mode="lines", line=dict(color="#00d4ff", width=1.2), name="Perplexity"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rv["baseline"], mode="lines", line=dict(color="#9baec8", width=1.2), name="Baseline"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rv["residual"], mode="lines", line=dict(color="#e3b341", width=1.2), name="Residual"),
        row=3,
        col=1,
    )
    for reg in rv["regions"]:
        for row in (1, 2, 3):
            fig.add_vrect(
                x0=reg["start"],
                x1=reg["end"],
                fillcolor="rgba(0,212,255,0.08)",
                line_color="rgba(0,212,255,0.50)",
                line_width=1.0,
                row=row,
                col=1,
            )
    fig.add_hline(y=0, line_color="#6b7280", line_dash="dash", row=3, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        height=560,
        margin=dict(l=55, r=20, t=75, b=40),
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(title="Position (bp)", gridcolor="rgba(255,255,255,0.07)", row=3, col=1)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def _plot_region_depth_width(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df["Width"],
                y=df["Mean Residual"],
                mode="markers",
                marker=dict(
                    size=11,
                    color=df["GC%"] if "GC%" in df.columns else df["Width"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="GC%"),
                    line=dict(width=1, color="rgba(255,255,255,0.28)"),
                ),
                text=df["Sequence"],
                hovertemplate=(
                    "%{text}<br>Width %{x} bp"
                    "<br>Mean residual %{y:.4f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Region Width vs Depth", font=dict(color="#dce9f8")),
        height=340,
        margin=dict(l=50, r=20, t=60, b=45),
        xaxis=dict(title="Width (bp)", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Mean Residual", gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


def _plot_region_rank_profile(rv: dict) -> go.Figure:
    regions = rv["regions"]
    fig = go.Figure()
    if regions:
        fig.add_trace(
            go.Bar(
                x=[f"R{r['rank']}" for r in regions],
                y=[abs(r["mean_residual"]) for r in regions],
                marker_color="#00d4ff",
                hovertemplate="Region %{x}<br>Depth %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Region Depth Ranking", font=dict(color="#dce9f8")),
        height=260,
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Absolute mean residual", gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    return fig


def _plot_metric_heatmap(df: pd.DataFrame) -> go.Figure:
    metric_cols = [col for col in ["Width", "Mean Residual", "GC%"] if col in df.columns]
    corr = df[metric_cols].corr(numeric_only=True) if metric_cols else pd.DataFrame()
    fig = go.Figure()
    if not corr.empty:
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                zmin=-1,
                zmax=1,
                colorscale="Tealgrn",
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Correlation Heatmap", font=dict(color="#dce9f8")),
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _plot_motif_sequence_heatmap(results: List[dict], active_motifs: set, window: int) -> go.Figure:
    valid = _valid_results(results)
    motifs = [m for m in MOTIF_LABELS if m in active_motifs]
    z = []
    y_labels = []
    for r in valid:
        y_labels.append(r["header"][:26])
        row = []
        for motif in motifs:
            total = 0
            for reg in r["regions"]:
                total += count_motifs(_region_seq(r["seq"], reg, window), {motif}).get(motif, 0)
            row.append(total)
        z.append(row)
    fig = go.Figure()
    if z and motifs:
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[MOTIF_LABELS.get(m, m) for m in motifs],
                y=y_labels,
                colorscale="Bluyl",
                hovertemplate="%{y}<br>%{x}: %{z} hits<extra></extra>",
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Motif Signal Across Called Regions", font=dict(color="#dce9f8")),
        height=max(240, 120 + 36 * max(len(y_labels), 1)),
        margin=dict(l=100, r=20, t=60, b=40),
    )
    return fig


def _render_navigation() -> str:
    if "percall_page" not in st.session_state:
        st.session_state["percall_page"] = PAGE_CONFIG[0][0]

    st.markdown(
        '<div class="section-header">Navigation Matrix</div>',
        unsafe_allow_html=True,
    )
    for start in range(0, len(PAGE_CONFIG), 4):
        row = PAGE_CONFIG[start: start + 4]
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            if idx >= len(row):
                continue
            page_key, button_label = row[idx]
            if col.button(button_label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state["percall_page"] = page_key

    current = st.session_state["percall_page"]
    st.markdown(
        '<div class="nav-status">'
        f'<strong>{current}</strong>'
        f'<span>{PAGE_DESCRIPTIONS.get(current, "")}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    return current


# ============================================================================
# Sidebar controls
# ============================================================================


def _sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<p style="font-size:1.75rem;font-weight:900;background:linear-gradient('
            '135deg,#6ee7ff,#00ff9d);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;letter-spacing:0.18em;'
            'margin-bottom:0.1rem">PERCALL</p>'
            '<p style="font-size:0.76rem;color:#93a8c3;letter-spacing:0.14em;'
            'text-transform:uppercase;margin-top:0">Scientific Command Deck</p>',
            unsafe_allow_html=True,
        )
        st.markdown(_DNA_ANIMATION, unsafe_allow_html=True)
        st.divider()

        st.markdown("### 🧬 Sequence Intake")
        upload = st.file_uploader(
            "Upload FASTA / Multi-FASTA",
            type=["fasta", "fa", "fna", "txt"],
            help="PERCALL accepts uploaded FASTA files, pasted sequence text, and curated examples.",
        )
        pasted = st.text_area(
            "Paste Sequence",
            height=110,
            placeholder=">sequence_id\nATGCATGCATGC...",
        )
        examples = ["None"] + _example_files()
        example_choice = st.selectbox(
            "Example Dataset",
            examples,
            help="Loaded only when no upload or pasted sequence is provided.",
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
                with open(example_path, encoding="utf-8", errors="replace") as handle:
                    fasta_text = handle.read()

        st.divider()
        st.markdown("### ⚙️ Analysis Controls")
        window = st.slider("Perplexity Window (bp)", 4, 30, 10, 1)
        min_len = st.slider("Min Region Length (bp)", 10, 200, 50, 5)
        max_len = st.slider("Max Region Length (bp)", 50, 1000, 300, 10)
        baseline_win = st.slider("Baseline Window (bp)", 50, 500, 200, 10)
        top_k = st.slider("Max Ranked Regions", 1, 20, 5, 1)
        score_cutoff = st.slider(
            "Residual Score Threshold",
            -1.0,
            0.0,
            -0.05,
            0.01,
            format="%.2f",
        )

        st.markdown("### 🧪 Structure Motifs")
        motif_sel: Dict[str, bool] = {}
        for key, label in MOTIF_LABELS.items():
            motif_sel[key] = st.checkbox(label, value=True, key=f"m_{key}")
        active_motifs = {k for k, v in motif_sel.items() if v}

        st.divider()
        run = st.button("▶ Launch Scientific Analysis", type="primary", use_container_width=True)

    return {
        "fasta_text": fasta_text,
        "window": window,
        "min_len": min_len,
        "max_len": max_len,
        "baseline_win": baseline_win,
        "top_k": top_k,
        "score_cutoff": score_cutoff,
        "active_motifs": active_motifs,
        "run": run,
        "example_choice": example_choice,
    }


# ============================================================================
# Pages
# ============================================================================


def _page_home(results: List[dict], params: dict) -> None:
    valid = _valid_results(results)
    total_regions = sum(len(r["regions"]) for r in valid)
    total_bp = sum(len(r["seq"]) for r in results)

    st.markdown(
        '<div class="percall-hero">'
        '<div class="percall-logo">PERCALL</div>'
        '<div class="percall-full-name">PERplexity-based Regulatory Region CALLer</div>'
        '<div class="percall-badges">'
        '<span class="badge badge-blue">Information Theoretic</span>'
        '<span class="badge badge-emerald">Bounded Minimum-Mean Kadane</span>'
        '<span class="badge badge-amber">Regulatory Domain Calling</span>'
        '<span class="badge badge-blue">Non-B DNA Annotation</span>'
        '<span class="badge badge-emerald">Publication Grade</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_DNA_ANIMATION, unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle" style="text-align:center;margin:0 auto 1.1rem auto;">'
        'PERCALL is presented here as a next-generation scientific environment: '
        'perplexity signal, residual derivation, region optimisation, structural annotation, '
        'and publication export unified in one immersive platform.'
        '</div>',
        unsafe_allow_html=True,
    )

    _hero_metric_strip(
        [
            ("Algorithmic core", "Original PERCALL"),
            ("Sequences in session", str(len(results))),
            ("Called regions", f"{total_regions}"),
            ("Analysed base pairs", f"{total_bp:,}" if total_bp else "—"),
        ]
    )

    if st.button("⚡ Quick Start in Sequence Workbench", type="primary"):
        st.session_state["percall_page"] = "Sequence Workbench"

    st.markdown('<div class="section-header">Scientific Workflow</div>', unsafe_allow_html=True)
    _workflow_cards(
        [
            ("Step 01", "Input Sequence", "FASTA upload, multi-sequence ingestion, and curated examples."),
            ("Step 02", "Perplexity Profile", "Dinucleotide entropy is transformed into a positional perplexity signal."),
            ("Step 03", "Residual Signal", "Rolling baseline compensation isolates locally unusual sequence behavior."),
            ("Step 04", "Kadane Optimisation", "Bounded minimum-mean scanning ranks optimal low-perplexity regions."),
            ("Step 05", "Domain & Motif Interpretation", "Regulatory domains are contextualised with Non-B DNA annotations."),
        ]
    )

    st.markdown('<div class="section-header">Platform Highlights</div>', unsafe_allow_html=True)
    _story_cards(
        [
            ("Platform", "Scientific Storytelling", "Each analysis stage is separated into focused modules so users can move from sequence intake to publication-grade outputs without losing context."),
            ("Signal", "Residual-First Reasoning", "PERCALL explains why a region is called by exposing raw perplexity, rolling baseline, and the derived residual signal in tandem."),
            ("Optimization", "Flagship Region Calling", "The supplied bounded minimum-mean Kadane formulation remains intact and visually central to the experience."),
            ("Structure", "Non-B DNA Context", "Detected regions can be interpreted alongside motif-centric evidence from the supplied PERCALL motif layer."),
            ("Export", "Publication Studio", "Figures, tables, session exports, and PDF reporting are assembled for manuscript-ready communication."),
            ("Identity", "Premium Scientific UI", "Glassmorphism, dark laboratory aesthetics, responsive layouts, and custom navigation replace default Streamlit styling."),
        ]
    )

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown('<div class="section-header">Core Algorithm Overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-card">'
            '<div class="info-pill">Dinucleotide perplexity</div>'
            '<div class="info-pill">Rolling baseline compensation</div>'
            '<div class="info-pill">Residual trough detection</div>'
            '<div class="info-pill">Bounded minimum-mean region optimisation</div>'
            '<div class="info-pill">Non-B DNA structural annotation</div>'
            '<p style="margin-top:0.85rem;color:#90a7c2">'
            'The platform intentionally preserves the supplied algorithmic pathway. '
            'No alternative prediction engine, machine learning model, or replacement region caller is introduced.'
            '</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="section-header">Publication Highlights</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-card">'
            '<div class="info-pill">Premium dark scientific presentation</div>'
            '<div class="info-pill">Interactive sequence-to-region storytelling</div>'
            '<div class="info-pill">Exportable Plotly figures</div>'
            '<div class="info-pill">Session JSON, CSV, Excel, PNG, SVG, PDF pathways</div>'
            '<div class="info-pill">Methods transparency with equations</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if valid:
        st.markdown('<div class="section-header">Live Session Preview</div>', unsafe_allow_html=True)
        preview = valid[0]
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            st.plotly_chart(
                plot_perplexity_profile(
                    preview["perp"],
                    preview["baseline"],
                    preview["residual"],
                    preview["regions"],
                    title=f"Preview — {preview['header'][:60]}",
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plot_sequence_domain_map(
                    len(preview["seq"]),
                    preview["regions"],
                    title="Region Architecture Snapshot",
                ),
                use_container_width=True,
            )
    else:
        _empty_state(
            "Ready for interactive analysis",
            "Load or paste a sequence in the Scientific Command Deck, launch the analysis, and the remaining modules will populate with live PERCALL outputs.",
        )


def _page_sequence_workbench(results: List[dict], params: dict) -> None:
    _page_intro(
        "Page 02",
        "Sequence Workbench",
        "Ingest, validate, and characterise input sequence collections before moving into signal-level analysis.",
    )
    preview = _preview_records(params.get("fasta_text"))
    records = preview or [(r["header"], r["seq"]) for r in results]
    if not records:
        _empty_state(
            "No sequence loaded",
            "Upload FASTA, paste a sequence, or choose an example dataset from the command deck to populate the workbench.",
        )
        return

    total_bp = sum(len(seq) for _, seq in records)
    mean_gc = np.mean([gc_pct(seq) for _, seq in records]) if records else 0
    total_n = sum(seq.count("N") for _, seq in records)
    _hero_metric_strip(
        [
            ("Sequences", str(len(records))),
            ("Total bp", f"{total_bp:,}"),
            ("Mean GC%", f"{mean_gc:.1f}%"),
            ("Ambiguous bases", f"{total_n:,}"),
        ]
    )

    df_preview = pd.DataFrame(
        [
            {
                "Header": header[:70],
                "Length (bp)": len(seq),
                "GC%": gc_pct(seq),
                "Ambiguous N": seq.count("N"),
                "Validation": "Ready",
            }
            for header, seq in records
        ]
    )
    st.dataframe(df_preview, use_container_width=True, hide_index=True)

    labels = [header[:80] for header, _ in records]
    selected = st.selectbox("Sequence", labels, key="workbench_seq")
    seq = records[labels.index(selected)][1]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_plot_nucleotide_composition(seq), use_container_width=True)
    with col2:
        st.plotly_chart(_plot_gc_heatmap(seq), use_container_width=True)

    if results:
        rv = next((r for r in _valid_results(results) if r["header"][:80] == selected), None)
        if rv is not None:
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(
                    plot_gc_profile(
                        rv["seq"],
                        regions=rv["regions"],
                        window=max(params["window"] * 5, 50),
                        title="GC Content with Called Domains",
                    ),
                    use_container_width=True,
                )
            with col4:
                st.plotly_chart(
                    plot_sequence_domain_map(
                        len(rv["seq"]),
                        rv["regions"],
                        title="Sequence Architecture Display",
                    ),
                    use_container_width=True,
                )


def _page_perplexity_explorer(results: List[dict]) -> None:
    _page_intro(
        "Page 03",
        "Perplexity Explorer",
        "Interact with the raw dinucleotide perplexity landscape, compare sequences, and inspect where low-complexity signal begins to emerge.",
    )
    valid = _valid_results(results)
    if not valid:
        _empty_state(
            "Perplexity explorer awaiting data",
            "Run PERCALL once to unlock positional perplexity plots, density summaries, and comparative sequence analytics.",
        )
        return

    rv = _select_result(results, "perplexity_seq_sel")
    if rv is None:
        return

    col1, col2 = st.columns([1.3, 0.7])
    with col1:
        st.plotly_chart(_plot_perplexity_focus(rv), use_container_width=True)
    with col2:
        st.plotly_chart(_plot_perplexity_distribution(rv["perp"]), use_container_width=True)

    _hero_metric_strip(
        [
            ("Mean perplexity", f"{np.nanmean(rv['perp']):.3f}"),
            ("Minimum perplexity", f"{np.nanmin(rv['perp']):.3f}"),
            ("Signal length", f"{len(rv['perp']):,}"),
            ("Called regions", str(len(rv["regions"]))),
        ]
    )
    st.plotly_chart(_plot_comparative_perplexity(valid), use_container_width=True)


def _page_baseline_residual(results: List[dict]) -> None:
    _page_intro(
        "Page 04",
        "Baseline & Residual Analysis",
        "See exactly how PERCALL constructs its signal: raw perplexity, rolling baseline, and residual trough architecture in one synchronized analytical view.",
    )
    rv = _select_result(results, "baseline_seq_sel")
    if rv is None:
        _empty_state(
            "Residual analysis awaiting data",
            "Launch an analysis to render synchronized perplexity, baseline, and residual plots.",
        )
        return

    st.plotly_chart(_plot_signal_triptych(rv), use_container_width=True)
    _hero_metric_strip(
        [
            ("Baseline mean", f"{np.nanmean(rv['baseline']):.3f}"),
            ("Residual minimum", f"{np.nanmin(rv['residual']):.4f}"),
            ("Residual median", f"{np.nanmedian(rv['residual']):.4f}"),
            ("Trough count", str(len(rv["regions"]))),
        ]
    )
    _story_cards(
        [
            ("Interpretation", "Rolling Compensation", "The local baseline suppresses broad compositional drift so that regional departures become interpretable."),
            ("Interpretation", "Residual Troughs", "Negative residual segments indicate windows whose perplexity falls below local expectation."),
            ("Interpretation", "Linked Visuals", "Any parameter update in the command deck immediately recalculates all three signal layers."),
        ]
    )


def _page_region_caller(results: List[dict], params: dict) -> None:
    _page_intro(
        "Page 05",
        "PERCALL Region Caller",
        "The flagship module: bounded minimum-mean Kadane optimisation, ranked regulatory domains, and direct visual explanation of why each region was called.",
    )
    rv = _select_result(results, "caller_seq_sel")
    if rv is None:
        _empty_state(
            "Region caller awaiting analysis",
            "Run PERCALL to populate ranked regulatory regions and their optimisation-derived architecture.",
        )
        return

    _workflow_cards(
        [
            ("Phase 01", "Residual Scan", "Residual signal is segmented while NaN intervals remain hard boundaries."),
            ("Phase 02", "Bounded Optimisation", "The minimum-mean contiguous subarray is found under user-specified length limits."),
            ("Phase 03", "Iterative Ranking", "Detected spans are masked and the search repeats for non-overlapping ranked regions."),
            ("Phase 04", "Structural Context", "Each ranked region is annotated with GC% and motif evidence from the supplied motif layer."),
        ]
    )

    st.plotly_chart(
        plot_perplexity_profile(
            rv["perp"],
            rv["baseline"],
            rv["residual"],
            rv["regions"],
            title=f"PERCALL Region Signal — {rv['header'][:60]}",
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.plotly_chart(
            plot_sequence_domain_map(
                len(rv["seq"]),
                rv["regions"],
                title="Region Architecture Map",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(_plot_region_rank_profile(rv), use_container_width=True)

    if rv["regions"]:
        opts = _region_rank_options(rv)
        chosen = st.selectbox("Inspect called region", opts, key="region_browser")
        reg = rv["regions"][opts.index(chosen)]
        _hero_metric_strip(
            [
                ("Region start", str(reg["start"])),
                ("Region end", str(reg["end"])),
                ("Width", f"{reg['width']} bp"),
                ("Depth", f"{reg['mean_residual']:.4f}"),
            ]
        )
        st.markdown(
            '<div class="glass-card">'
            f'<div class="info-pill">Rank R{reg["rank"]}</div>'
            f'<div class="info-pill">Trough {reg["trough"]}</div>'
            f'<div class="info-pill">GC% {reg["gc_pct"]}</div>'
            f'<div class="info-pill">Motifs {reg["motifs"] or "None detected"}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    df = _regions_df(results)
    if not df.empty:
        st.markdown('<div class="section-header">Regulatory Region Ranking</div>', unsafe_allow_html=True)
        st.dataframe(
            df.style.background_gradient(subset=["Mean Residual"], cmap="Blues_r"),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(_plot_region_depth_width(df), use_container_width=True)


def _page_domain_explorer(results: List[dict]) -> None:
    _page_intro(
        "Page 06",
        "Regulatory Domain Explorer",
        "Search, filter, and inspect detected regulatory domains as a dedicated browser for region-level interpretation.",
    )
    df = _regions_df(results)
    if df.empty:
        _empty_state(
            "No regulatory domains yet",
            "Run PERCALL and return here to browse detected regions across all sequences.",
        )
        return

    col1, col2, col3 = st.columns([1.1, 0.9, 0.8])
    with col1:
        search = st.text_input("Search sequence label", placeholder="Sequence name fragment")
    with col2:
        sequences = ["All"] + sorted(df["Sequence"].unique().tolist())
        seq_filter = st.selectbox("Filter sequence", sequences, key="domain_seq_filter")
    with col3:
        motif_filter = st.text_input("Filter motif text", placeholder="G4, ZDNA, ...")

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["Sequence"].str.contains(search, case=False, na=False)]
    if seq_filter != "All":
        filtered = filtered[filtered["Sequence"] == seq_filter]
    if motif_filter:
        filtered = filtered[filtered["Motifs"].str.contains(motif_filter, case=False, na=False)]

    st.markdown(f"Showing **{len(filtered)}** domains from **{len(df)}** total calls.")
    st.dataframe(
        filtered.style.background_gradient(subset=["Mean Residual"], cmap="Blues_r"),
        use_container_width=True,
        hide_index=True,
    )

    if filtered.empty:
        return

    options = [
        f"{row.Sequence} · R{row.Rank} · {row.Start}-{row.End}"
        for row in filtered.itertuples(index=False)
    ]
    chosen = st.selectbox("Click a domain to inspect", options, key="domain_detail_sel")
    row = filtered.iloc[options.index(chosen)]
    _hero_metric_strip(
        [
            ("Sequence", row["Sequence"]),
            ("Region rank", f"R{row['Rank']}"),
            ("Width", f"{row['Width']} bp"),
            ("GC%", f"{row['GC%']}"),
        ]
    )
    st.markdown(
        '<div class="glass-card">'
        f'<div class="info-pill">Start {row["Start"]}</div>'
        f'<div class="info-pill">End {row["End"]}</div>'
        f'<div class="info-pill">Mean Residual {row["Mean Residual"]:.4f}</div>'
        f'<div class="info-pill">Trough {row["Trough"]}</div>'
        f'<div class="info-pill">Motifs {row["Motifs"] or "None detected"}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _page_nonb_center(results: List[dict], params: dict) -> None:
    _page_intro(
        "Page 07",
        "Non-B DNA Structure Center",
        "Structural genomics interpretation for PERCALL-called domains using the supplied motif definitions and region-centered enrichment analytics.",
    )
    valid = _valid_results(results)
    ordered = [m for m in MOTIF_LABELS if m in params["active_motifs"]]
    if not valid or not ordered:
        _empty_state(
            "Structural center awaiting motif-enabled analysis",
            "Run PERCALL with at least one motif enabled to populate enrichment, distribution, and regional motif summaries.",
        )
        return

    total_seq_bp = sum(len(r["seq"]) for r in valid)
    total_region_bp = sum(reg["width"] for r in valid for reg in r["regions"])
    bg_counts: Dict[str, int] = {m: 0 for m in ordered}
    reg_counts: Dict[str, int] = {m: 0 for m in ordered}
    motif_rows = []

    for r in valid:
        bg = count_motifs(r["seq"], active_motifs=set(ordered))
        for motif in ordered:
            bg_counts[motif] += bg.get(motif, 0)
        for reg in r["regions"]:
            seq_piece = _region_seq(r["seq"], reg, params["window"])
            rc = count_motifs(seq_piece, active_motifs=set(ordered))
            for motif in ordered:
                reg_counts[motif] += rc.get(motif, 0)
                if rc.get(motif, 0) > 0:
                    motif_rows.append(
                        {
                            "Sequence": r["header"][:50],
                            "Region Rank": reg["rank"],
                            "Motif": MOTIF_LABELS.get(motif, motif),
                            "Count in Region": rc[motif],
                        }
                    )

    _story_cards(
        [
            ("Supported", "G4", "G-quadruplex signatures derived from the supplied motif set."),
            ("Supported", "i-Motif", "C-rich structural motif support from the supplied PERCALL layer."),
            ("Supported", "Z / eGZ-DNA", "Alternating sequence and expanded repeat structures."),
            ("Supported", "Triplex / STR / Direct Repeat / PolyA-T", "Contextual structural annotations shown without replacing the supplied algorithm."),
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_motif_enrichment(reg_counts, bg_counts, total_region_bp, total_seq_bp, ordered),
            use_container_width=True,
        )
    with col2:
        r0 = valid[0]
        st.plotly_chart(
            plot_motif_distribution(
                scan_motifs(r0["seq"], active_motifs=set(ordered)),
                len(r0["seq"]),
                ordered,
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        _plot_motif_sequence_heatmap(results, params["active_motifs"], params["window"]),
        use_container_width=True,
    )

    if motif_rows:
        st.dataframe(pd.DataFrame(motif_rows), use_container_width=True, hide_index=True)


def _page_genome_viewer(results: List[dict], params: dict) -> None:
    _page_intro(
        "Page 08",
        "Interactive Genome Viewer",
        "Inspect sequence-level architecture as coordinated tracks for sequence span, called regions, motif positions, and linked signal context.",
    )
    rv = _select_result(results, "genome_seq_sel")
    if rv is None:
        _empty_state(
            "Genome viewer awaiting data",
            "Launch an analysis to inspect zoomable region and motif tracks for each sequence.",
        )
        return

    ordered = [m for m in MOTIF_LABELS if m in params["active_motifs"]]
    motif_positions = scan_motifs(rv["seq"], active_motifs=set(ordered))

    st.plotly_chart(
        plot_genome_browser(
            seq_len=len(rv["seq"]),
            regions=rv["regions"],
            motif_positions=motif_positions,
            active_motifs=ordered,
            seq_label=rv["header"][:40],
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns([1.25, 0.75])
    with col1:
        st.plotly_chart(
            plot_perplexity_profile(
                rv["perp"],
                rv["baseline"],
                rv["residual"],
                rv["regions"],
                title="Linked Signal View",
            ),
            use_container_width=True,
        )
    with col2:
        if rv["regions"]:
            st.markdown('<div class="section-header">Region Coordinates</div>', unsafe_allow_html=True)
            for reg in rv["regions"]:
                st.markdown(
                    '<div class="glass-card">'
                    f'<div class="info-pill">R{reg["rank"]}</div>'
                    f'<div class="info-pill">{reg["start"]}–{reg["end"]} bp</div>'
                    f'<div class="info-pill">Width {reg["width"]} bp</div>'
                    f'<div class="info-pill">Score {reg["mean_residual"]:.4f}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )


def _page_statistics_center(results: List[dict], params: dict) -> None:
    _page_intro(
        "Page 09",
        "Statistics & Discovery Center",
        "Aggregate analytics laboratory for region distributions, GC structure, motif activity, and cross-metric relationships.",
    )
    valid = _valid_results(results)
    if not valid:
        _empty_state(
            "Discovery center awaiting analysis",
            "Run PERCALL to populate session-level summary statistics and structural discovery charts.",
        )
        return

    total_bp = sum(len(r["seq"]) for r in results)
    mean_len = total_bp / len(results) if results else 0
    total_regions = sum(len(r["regions"]) for r in valid)
    motif_hits = sum(1 for r in valid for reg in r["regions"] if reg["motifs"])
    overall_gc = sum(gc_pct(r["seq"]) * len(r["seq"]) for r in results) / total_bp if total_bp else 0
    _hero_metric_strip(
        [
            ("Sequences", str(len(results))),
            ("Total bp", f"{total_bp:,}"),
            ("Mean length", f"{mean_len:,.0f}"),
            ("Overall GC%", f"{overall_gc:.1f}%"),
            ("Regions called", str(total_regions)),
            ("Regions with motifs", str(motif_hits)),
        ]
    )

    labels = [r["header"][:40] for r in valid]
    counts = [len(r["regions"]) for r in valid]
    fig_bar = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker_color="#00d4ff",
            opacity=0.82,
            hovertemplate="%{x}<br>Regions %{y}<extra></extra>",
        )
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dce9f8"),
        title=dict(text="Region Count per Sequence", font=dict(color="#dce9f8")),
        height=300,
        margin=dict(l=45, r=20, t=60, b=75),
        xaxis=dict(tickangle=-25, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Regions", gridcolor="rgba(255,255,255,0.07)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    df = _regions_df(results)
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_region_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(_plot_metric_heatmap(df), use_container_width=True)

    st.plotly_chart(
        _plot_motif_sequence_heatmap(results, params["active_motifs"], params["window"]),
        use_container_width=True,
    )


def _page_publication_studio(results: List[dict], params: dict) -> None:
    _page_intro(
        "Page 10",
        "Publication Studio",
        "Generate manuscript-ready outputs from the current PERCALL session: tables, structured exports, figures, and a PDF-ready report pathway.",
    )
    valid = _valid_results(results)
    if not valid:
        _empty_state(
            "Publication studio awaiting analysis",
            "Run PERCALL to enable figure export, data tables, session packages, and PDF report generation.",
        )
        return

    rv = _select_result(results, "publication_seq_sel")
    if rv is None:
        return

    summary_df = _summary_df(results)
    regions_df = _regions_df(results)
    profile_fig = plot_perplexity_profile(
        rv["perp"],
        rv["baseline"],
        rv["residual"],
        rv["regions"],
        title=f"Publication Figure — {rv['header'][:60]}",
    )
    st.plotly_chart(profile_fig, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.download_button("CSV Summary", _df_to_csv(summary_df), "percall_summary.csv", "text/csv")
    with c2:
        st.download_button(
            "Excel Summary",
            _df_to_excel(summary_df),
            "percall_summary.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c3:
        st.download_button("CSV Regions", _df_to_csv(regions_df), "percall_regions.csv", "text/csv")
    with c4:
        st.download_button(
            "Excel Regions",
            _df_to_excel(regions_df),
            "percall_regions.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c5:
        st.download_button("Session JSON", _results_to_json(results, params), "percall_session.json", "application/json")

    try:
        import plotly.io as pio

        png_bytes = pio.to_image(profile_fig, format="png", width=1300, height=600, scale=2)
        svg_bytes = pio.to_image(profile_fig, format="svg", width=1300, height=600, scale=2)
        x1, x2 = st.columns(2)
        with x1:
            st.download_button("⬇ High-Resolution PNG", png_bytes, "percall_profile.png", "image/png")
        with x2:
            st.download_button("⬇ Vector SVG", svg_bytes, "percall_profile.svg", "image/svg+xml")
    except Exception:
        st.caption("Install kaleido for PNG/SVG export if these options are unavailable.")

    st.markdown('<div class="section-header">PDF Report Generator</div>', unsafe_allow_html=True)
    if st.button("📄 Generate Publication PDF", type="primary"):
        try:
            from core.report import generate_pdf

            ordered = [m for m in MOTIF_LABELS if m in params["active_motifs"]]
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
                    rc = count_motifs(_region_seq(r["seq"], reg, params["window"]), active_motifs=set(ordered))
                    for motif in ordered:
                        reg_counts[motif] += rc.get(motif, 0)

            pdf_bytes = generate_pdf(
                params=params,
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
            st.success("PDF report generated.")
            st.download_button(
                "⬇ Download PERCALL_Report.pdf",
                data=pdf_bytes,
                file_name="PERCALL_Report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")


def _page_methods() -> None:
    _page_intro(
        "Page 11",
        "Methods & Algorithm",
        "Mathematical transparency for the supplied PERCALL method, including perplexity formulation, residual construction, and bounded minimum-mean optimisation.",
    )
    _workflow_cards(
        [
            ("Method", "Perplexity Formulation", "Sliding windows tally dinucleotide frequencies and transform Shannon entropy to perplexity units."),
            ("Method", "Residual Construction", "A centred rolling baseline estimates local expectation and is subtracted from the raw signal."),
            ("Method", "Kadane Family Optimisation", "A bounded minimum-mean scan identifies the optimal region under user-defined length limits."),
            ("Method", "Structural Annotation", "Called regions are screened against the supplied motif layer for interpretive context."),
        ]
    )

    st.latex(r"H = -\sum_i p_i \log_2 p_i")
    st.latex(r"\mathrm{Perplexity} = 2^H")
    st.latex(r"\mathrm{Residual}(x) = \mathrm{Perplexity}(x) - \mathrm{Baseline}(x)")
    st.latex(
        r"(s^\*, e^\*) = \arg\min_{L_{\min} \le e-s+1 \le L_{\max}} "
        r"\frac{1}{e-s+1}\sum_{i=s}^{e}\mathrm{Residual}_i"
    )

    _story_cards(
        [
            ("Transparency", "Dinucleotide Signal", "Perplexity is computed directly from the supplied sequence windows; windows with ambiguous bases are masked."),
            ("Transparency", "Local Baseline", "The centred rolling mean compensates for background drift while preserving local trough structure."),
            ("Transparency", "No Replacement Algorithms", "The bounded minimum-mean Kadane-family region caller is preserved exactly as the core optimisation strategy."),
            ("Transparency", "Motif Definitions", "Structural context is restricted to motif classes already present in the supplied PERCALL codebase."),
        ]
    )


def _page_about() -> None:
    _page_intro(
        "Page 12",
        "About PERCALL",
        "Tool identity, scientific motivation, citation guidance, versioning, and repository context for the PERCALL platform.",
    )
    st.markdown(
        '<div class="glass-card">'
        '<div class="info-pill">Scientific focus: regulatory region calling from perplexity signal</div>'
        '<div class="info-pill">Core innovation: bounded minimum-mean Kadane optimisation</div>'
        '<div class="info-pill">Structural context: supplied Non-B DNA motif layer</div>'
        '<div class="info-pill">Interface: premium Streamlit scientific platform</div>'
        '<div class="info-pill">Version: 2025.2</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.code(
        textwrap.dedent(
            """\
            Yella VR (2025). PERCALL: PERplexity-based Regulatory Region CALLer.
            An information-theoretic framework for identifying low-perplexity regulatory
            DNA regions using bounded minimum-mean Kadane optimisation.
            GitHub: https://github.com/VRYella/PerCALL
            """
        ),
        language=None,
    )
    _story_cards(
        [
            ("Credits", "Scientific Motivation", "PERCALL explores how locally unusual low-perplexity sequence architecture can reveal regulatory DNA regions."),
            ("Credits", "Documentation", "Methods, exports, and visuals are aligned around the supplied PERCALL manuscript-driven workflow."),
            ("Credits", "Repository", "The platform runs directly from this repository’s supplied algorithms without introducing replacement predictors."),
        ]
    )


# ============================================================================
# Application entry point
# ============================================================================


def main() -> None:
    params = _sidebar()

    results: List[dict]
    if params["run"] and params["fasta_text"]:
        records = parse_fasta(params["fasta_text"])
        if not records:
            st.error("No valid sequences found. Please check the FASTA formatting.")
            return

        progress = st.progress(0, text="Analysing sequences…")
        results = []
        for idx, (header, seq) in enumerate(records):
            if len(seq) < params["min_len"] + params["window"]:
                results.append(
                    {
                        "header": header,
                        "seq": seq,
                        "perp": np.array([]),
                        "baseline": np.array([]),
                        "residual": np.array([]),
                        "regions": [],
                        "skipped": True,
                    }
                )
            else:
                results.append(
                    process_sequence(
                        header,
                        seq,
                        window=params["window"],
                        baseline_win=params["baseline_win"],
                        min_len=params["min_len"],
                        max_len=params["max_len"],
                        top_k=params["top_k"],
                        score_cutoff=params["score_cutoff"],
                        active_motifs=params["active_motifs"],
                    )
                )
            progress.progress((idx + 1) / len(records), text=f"Processed {idx + 1}/{len(records)} sequences")
        progress.empty()
        st.session_state["percall_results"] = results
        st.session_state["percall_params"] = params
    elif "percall_results" in st.session_state:
        results = st.session_state["percall_results"]
        stored_params = st.session_state.get("percall_params", {})
        params = {**stored_params, **params}
    else:
        results = []

    current_page = _render_navigation()

    if current_page == "Home":
        _page_home(results, params)
    elif current_page == "Sequence Workbench":
        _page_sequence_workbench(results, params)
    elif current_page == "Perplexity Explorer":
        _page_perplexity_explorer(results)
    elif current_page == "Baseline & Residual Analysis":
        _page_baseline_residual(results)
    elif current_page == "PERCALL Region Caller":
        _page_region_caller(results, params)
    elif current_page == "Regulatory Domain Explorer":
        _page_domain_explorer(results)
    elif current_page == "Non-B DNA Structure Center":
        _page_nonb_center(results, params)
    elif current_page == "Interactive Genome Viewer":
        _page_genome_viewer(results, params)
    elif current_page == "Statistics & Discovery Center":
        _page_statistics_center(results, params)
    elif current_page == "Publication Studio":
        _page_publication_studio(results, params)
    elif current_page == "Methods & Algorithm":
        _page_methods()
    elif current_page == "About PERCALL":
        _page_about()


if __name__ == "__main__":
    main()
