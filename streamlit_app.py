"""
streamlit_app.py
────────────────
PERCALL — PERplexity-based Regulatory Region CALLer
=====================================================

A visually stunning, scientifically rigorous bioinformatics platform that
identifies low-perplexity regulatory DNA regions using information-theoretic
sequence analysis and a bounded minimum-mean Kadane optimisation algorithm.

Author  : Dr. Venkata Rajesh Yella
Version : 2025.1
Licence : MIT
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.perplexity import compute_perplexity, local_residual
from core.region_caller import find_regions
from core.motifs import (
    MOTIF_LABELS,
    scan_motifs,
    count_motifs,
)
from core.plotting import (
    plot_perplexity_profile,
    plot_gc_profile,
    plot_region_distribution,
    plot_motif_enrichment,
    plot_motif_distribution,
    plot_genome_browser,
    plot_sequence_domain_map,
)

# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="PERCALL — Regulatory Region Caller",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Dark scientific theme  (injected once at startup)
# ============================================================================

_DARK_CSS = """
<style>
/* ── BASE LAYER ────────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(150deg, #060b18 0%, #0d1b2a 60%, #060b18 100%);
    background-attachment: fixed;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(6, 11, 24, 0.97) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.18) !important;
}

/* Main content area */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }

/* ── METRIC CARDS ───────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: rgba(0, 212, 255, 0.07) !important;
    border: 1px solid rgba(0, 212, 255, 0.22) !important;
    border-radius: 14px !important;
    padding: 18px 16px !important;
    animation: glowPulse 4s ease-in-out infinite;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 212, 255, 0.55) !important;
}
div[data-testid="metric-container"] > label {
    color: #8b9bb4 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="metric-container"] > div {
    color: #00d4ff !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}

@keyframes glowPulse {
    0%,100% { box-shadow: 0 0 8px rgba(0,212,255,0.12); }
    50%      { box-shadow: 0 0 22px rgba(0,212,255,0.38); }
}

/* ── TABS ───────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    color: #8b9bb4 !important;
    font-weight: 500 !important;
    letter-spacing: 0.025em;
    padding: 8px 16px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00d4ff !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-border"] {
    background: rgba(0,212,255,0.25) !important;
}
div[data-baseweb="tab-highlight"] {
    background: #00d4ff !important;
}

/* ── BUTTONS ────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
    color: #060b18 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 9px !important;
    letter-spacing: 0.03em;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00eeff 0%, #00bfff 100%) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.45) !important;
}
.stDownloadButton > button {
    background: rgba(0,255,157,0.12) !important;
    color: #00ff9d !important;
    border: 1px solid rgba(0,255,157,0.35) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    background: rgba(0,255,157,0.2) !important;
    border-color: rgba(0,255,157,0.6) !important;
}

/* ── INPUT WIDGETS ──────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stTextArea textarea,
.stFileUploader > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(0,212,255,0.25) !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* ── DATAFRAME ──────────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── CUSTOM CLASSES ─────────────────────────────────────────────────────── */
.percall-hero {
    text-align: center;
    padding: 32px 0 16px 0;
}
.percall-logo {
    font-size: 3.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #00ff9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 6px;
    margin-bottom: 6px;
    line-height: 1.1;
}
.percall-full-name {
    font-size: 0.95rem;
    color: #8b9bb4;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.percall-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 14px 0 28px 0;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-blue {
    background: rgba(0,212,255,0.12);
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.28);
}
.badge-emerald {
    background: rgba(0,255,157,0.12);
    color: #00ff9d;
    border: 1px solid rgba(0,255,157,0.28);
}
.badge-amber {
    background: rgba(227,179,65,0.12);
    color: #e3b341;
    border: 1px solid rgba(227,179,65,0.28);
}
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0,212,255,0.14);
    border-radius: 16px;
    padding: 22px 24px;
    margin: 10px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.28);
}
.section-header {
    font-size: 0.78rem;
    font-weight: 700;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid rgba(0,212,255,0.28);
    padding-bottom: 7px;
    margin: 22px 0 14px 0;
}
.info-pill {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.85rem;
    color: #c9d1d9;
    margin: 4px 2px;
}
.emerald-text { color: #00ff9d; font-weight: 700; }
.blue-text    { color: #00d4ff; font-weight: 700; }
.amber-text   { color: #e3b341; font-weight: 700; }

/* DNA animation strand */
.dna-s1 {
    stroke-dasharray: 10 5;
    animation: driftFwd 2.5s linear infinite;
}
.dna-s2 {
    stroke-dasharray: 10 5;
    animation: driftBack 2.5s linear infinite;
}
@keyframes driftFwd  { from { stroke-dashoffset: 0; } to { stroke-dashoffset: -75; } }
@keyframes driftBack { from { stroke-dashoffset: 0; } to { stroke-dashoffset:  75; } }
</style>
"""
st.markdown(_DARK_CSS, unsafe_allow_html=True)

# ============================================================================
# DNA animation
# ============================================================================

_DNA_ANIMATION = """
<div style="text-align:center;padding:6px 0 18px 0">
<svg viewBox="0 0 640 62" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:640px;height:62px;overflow:visible">
  <path class="dna-s1"
    d="M0,31 Q40,6 80,31 Q120,56 160,31 Q200,6 240,31 Q280,56 320,31
       Q360,6 400,31 Q440,56 480,31 Q520,6 560,31 Q600,56 640,31"
    fill="none" stroke="#00d4ff" stroke-width="2.6" opacity="0.88"/>
  <path class="dna-s2"
    d="M0,31 Q40,56 80,31 Q120,6 160,31 Q200,56 240,31 Q280,6 320,31
       Q360,56 400,31 Q440,6 480,31 Q520,56 560,31 Q600,6 640,31"
    fill="none" stroke="#00ff9d" stroke-width="2.6" opacity="0.88"/>
  <line x1="80"  y1="27" x2="80"  y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
  <line x1="160" y1="27" x2="160" y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
  <line x1="240" y1="27" x2="240" y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
  <line x1="320" y1="27" x2="320" y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
  <line x1="400" y1="27" x2="400" y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
  <line x1="480" y1="27" x2="480" y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
  <line x1="560" y1="27" x2="560" y2="35" stroke="rgba(255,255,255,0.32)" stroke-width="1.6"/>
</svg>
</div>
"""

# ============================================================================
# Lightweight FASTA parser
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


# ============================================================================
# Processing helpers
# ============================================================================


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
            "header": header, "seq": seq,
            "perp": perp, "baseline": perp.copy(),
            "residual": perp.copy(), "regions": [], "skipped": True,
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
        score_cutoff=score_cutoff, window=window,
        active_motifs=active_motifs,
    )
    return {
        "header": header, "seq": seq,
        "perp": perp, "baseline": baseline,
        "residual": res, "regions": regions, "skipped": False,
    }


# ============================================================================
# Export helpers
# ============================================================================


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
        "percall_version": "2025.1",
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
    rows = []
    for r in results:
        rows.append({
            "Header": r["header"][:70],
            "Length (bp)": len(r["seq"]),
            "GC%": gc_pct(r["seq"]),
            "Regions": len(r["regions"]),
            "Status": "Skipped" if r["skipped"] else "OK",
        })
    return pd.DataFrame(rows)


# ============================================================================
# Sidebar
# ============================================================================


def _sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<p style="font-size:1.6rem;font-weight:900;background:linear-gradient('
            '135deg,#00d4ff,#00ff9d);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;letter-spacing:4px;'
            'margin-bottom:0">PERCALL</p>'
            '<p style="font-size:0.7rem;color:#8b9bb4;letter-spacing:0.1em;'
            'text-transform:uppercase;margin-top:0">Regulatory Region Caller</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("### 📂 Input")
        upload = st.file_uploader(
            "Upload FASTA",
            type=["fasta", "fa", "fna", "txt"],
            help="Multi-FASTA files are supported.",
        )
        st.markdown("**— or —**")
        pasted = st.text_area(
            "Paste Sequence",
            height=90,
            placeholder=">seq_name\nATGCATGCATGC...",
        )
        use_example = st.checkbox("Use built-in example", value=False)

        fasta_text: Optional[str] = None
        if upload is not None:
            fasta_text = upload.read().decode("utf-8", errors="replace")
        elif pasted.strip():
            fasta_text = pasted.strip()
            if not fasta_text.startswith(">"):
                fasta_text = f">pasted_sequence\n{fasta_text}"
        elif use_example:
            # Primary path: example_data/ (new canonical location)
            # Fallback: assets/ (legacy location kept for backward compatibility)
            _ex = os.path.join(_ROOT, "example_data", "example.fasta")
            if not os.path.isfile(_ex):
                _ex = os.path.join(_ROOT, "assets", "example.fasta")
            if os.path.isfile(_ex):
                with open(_ex) as _fh:
                    fasta_text = _fh.read()

        st.divider()
        st.markdown("### ⚙️ Parameters")

        st.markdown("**Perplexity**")
        window = st.slider("Window Size (bp)", 4, 30, 10, 1,
                           help="Dinucleotide perplexity window.")

        st.markdown("**Region Detection**")
        min_len = st.slider("Min Region Length (bp)", 10, 200, 50, 5)
        max_len = st.slider("Max Region Length (bp)", 50, 1000, 300, 10)
        baseline_win = st.slider("Baseline Window (bp)", 50, 500, 200, 10,
                                 help="Rolling window for GC-drift compensation.")
        top_k = st.slider("Max Regions", 1, 20, 5, 1)
        score_cutoff = st.slider(
            "Score Threshold", -1.0, 0.0, -0.05, 0.01, format="%.2f",
            help="Regions with mean residual ≥ this are excluded.",
        )

        st.markdown("**Non-B DNA Motifs**")
        motif_sel: Dict[str, bool] = {}
        for key, label in MOTIF_LABELS.items():
            motif_sel[key] = st.checkbox(label, value=True, key=f"m_{key}")
        active_motifs = {k for k, v in motif_sel.items() if v}

        st.divider()
        run = st.button("▶  Run PERCALL", type="primary", use_container_width=True)

    return dict(
        fasta_text=fasta_text, window=window, min_len=min_len,
        max_len=max_len, baseline_win=baseline_win, top_k=top_k,
        score_cutoff=score_cutoff, active_motifs=active_motifs, run=run,
    )


# ============================================================================
# Landing / hero
# ============================================================================


def _landing() -> None:
    st.markdown(
        '<div class="percall-hero">'
        '<div class="percall-logo">PERCALL</div>'
        '<div class="percall-full-name">PERplexity-based Regulatory Region CALLer</div>'
        '<div class="percall-badges">'
        '<span class="badge badge-blue">Information-Theoretic</span>'
        '<span class="badge badge-emerald">Kadane Optimisation</span>'
        '<span class="badge badge-amber">Non-B DNA Annotation</span>'
        '<span class="badge badge-blue">O(n) Algorithm</span>'
        '<span class="badge badge-emerald">Publication-Ready</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_DNA_ANIMATION, unsafe_allow_html=True)

    with st.expander("📖 About PERCALL", expanded=False):
        st.markdown(
            textwrap.dedent("""
            **PERCALL** reframes regulatory-region detection as a rigorous
            **optimisation problem**: find the contiguous stretch of the
            sequence perplexity-residual signal with the **minimum mean value**,
            subject only to user-set length bounds.

            **Core algorithm steps**

            1. **Dinucleotide Perplexity** — sliding-window Shannon entropy
               exponentiated to perplexity units (2ᴴ).
            2. **Local Baseline** — centred rolling mean compensates for
               GC-content drift along the sequence.
            3. **Residual Signal** — perplexity minus baseline isolates
               locally unusual low-complexity segments.
            4. **Bounded Minimum-Mean Kadane** — O(n) monotonic-deque scan
               finds the provably optimal region within [min_len, max_len].
            5. **Non-B DNA Annotation** — each region is screened for
               G4, i-Motif, Z-DNA, eGZ-DNA, Triplex, STR,
               DirectRepeat, and PolyA/T motifs.

            ---
            **How to use:** upload / paste a FASTA → adjust parameters
            (defaults suit most promoter sequences) → click **▶ Run PERCALL**
            → explore the seven result tabs.
            """)
        )


# ============================================================================
# Tab 1 – Dashboard
# ============================================================================


def _tab_dashboard(results: List[dict]) -> None:
    st.markdown('<p class="section-header">📊 Analysis Dashboard</p>',
                unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"]]
    total_bp = sum(len(r["seq"]) for r in results)
    mean_len = total_bp / len(results) if results else 0
    total_regions = sum(len(r["regions"]) for r in valid)
    motif_hits = sum(
        1 for r in valid for reg in r["regions"] if reg["motifs"]
    )
    overall_gc = (
        sum(gc_pct(r["seq"]) * len(r["seq"]) for r in results) / total_bp
        if total_bp else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sequences", len(results))
    c2.metric("Total bp", f"{total_bp:,}")
    c3.metric("Mean Length", f"{mean_len:,.0f} bp")
    c4.metric("Overall GC%", f"{overall_gc:.1f}%")
    c5.metric("Regions Called", total_regions)

    if total_regions:
        st.markdown('<p class="section-header">Region Overview</p>',
                    unsafe_allow_html=True)
        # Bar chart: regions per sequence
        labels = [r["header"][:40] for r in valid]
        counts = [len(r["regions"]) for r in valid]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=counts,
            name="Regions detected",
            marker_color="#00d4ff",
            opacity=0.85,
            hovertemplate="%{x}<br>Regions: %{y}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#c9d1d9"),
            height=240,
            margin=dict(l=50, r=20, t=30, b=80),
            xaxis=dict(tickangle=-30,
                       gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="# Regions",
                       gridcolor="rgba(255,255,255,0.07)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="glass-card">'
                '<div class="section-header">Top Regions by Score</div>'
                + "".join(
                    f'<div class="info-pill"><span class="blue-text">R{reg["rank"]}</span> '
                    f'{r["header"][:30]}… &nbsp;'
                    f'<span class="amber-text">{reg["mean_residual"]:.4f}</span>'
                    f'&nbsp; {reg["width"]} bp</div>'
                    for r in sorted(valid, key=lambda x: min(
                        (reg["mean_residual"] for reg in x["regions"]), default=0))[:3]
                    for reg in sorted(r["regions"], key=lambda x: x["mean_residual"])[:2]
                )
                + '</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="glass-card">'
                '<div class="section-header">Motif Summary</div>'
                + f'<div class="info-pill">Regions with motifs: '
                f'<span class="emerald-text">{motif_hits}</span> / {total_regions}</div>'
                + "".join(
                    f'<div class="info-pill">'
                    f'<span class="blue-text">{MOTIF_LABELS.get(m, m)}</span></div>'
                    for m in MOTIF_LABELS
                    if any(m in reg["motifs"] for r in valid for reg in r["regions"])
                )
                + '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No regions detected. Adjust parameters and re-run.")


# ============================================================================
# Tab 2 – Sequence Analysis
# ============================================================================


def _tab_sequence_analysis(results: List[dict], params: dict) -> None:
    st.markdown('<p class="section-header">🔬 Sequence Analysis</p>',
                unsafe_allow_html=True)

    df = _summary_df(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download summary.csv", _df_to_csv(df),
            file_name="percall_summary.csv", mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇ Download summary.xlsx", _df_to_excel(df),
            file_name="percall_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    valid = [r for r in results if not r["skipped"] and r["perp"].size > 0]
    if not valid:
        return

    st.markdown('<p class="section-header">GC Content Profile</p>',
                unsafe_allow_html=True)
    seq_labels = [r["header"][:80] for r in valid]
    chosen_idx = 0
    if len(valid) > 1:
        chosen = st.selectbox("Sequence:", seq_labels, key="gc_seq_sel")
        chosen_idx = seq_labels.index(chosen)
    rv = valid[chosen_idx]

    fig_gc = plot_gc_profile(
        rv["seq"], regions=rv["regions"],
        window=max(params["window"] * 5, 50),
        title=f"GC Content — {rv['header'][:60]}",
    )
    st.plotly_chart(fig_gc, use_container_width=True)

    # Sequence length distribution
    st.markdown('<p class="section-header">Sequence Length Distribution</p>',
                unsafe_allow_html=True)
    lengths = [len(r["seq"]) for r in results]
    fig_len = go.Figure(go.Histogram(
        x=lengths, nbinsx=min(20, len(lengths)),
        marker_color="#00d4ff", opacity=0.8,
        hovertemplate="Length: %{x} bp<br>Count: %{y}<extra></extra>",
    ))
    fig_len.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#c9d1d9"),
        height=200,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(title="Length (bp)", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.07)"),
        showlegend=False,
    )
    st.plotly_chart(fig_len, use_container_width=True)


# ============================================================================
# Tab 3 – Region Calling
# ============================================================================


def _tab_region_calling(results: List[dict], params: dict) -> pd.DataFrame:
    st.markdown('<p class="section-header">🗺 Region Calling</p>',
                unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"] and r["perp"].size > 0]
    if not valid:
        st.info("No valid sequences to display.")
        return pd.DataFrame()

    seq_labels = [r["header"][:80] for r in valid]
    chosen_idx = 0
    if len(valid) > 1:
        chosen = st.selectbox("Sequence:", seq_labels, key="rc_seq_sel")
        chosen_idx = seq_labels.index(chosen)
    rv = valid[chosen_idx]

    # Perplexity profile
    fig_pp = plot_perplexity_profile(
        rv["perp"], rv["baseline"], rv["residual"], rv["regions"],
        title=f"PERCALL — {rv['header'][:60]}",
    )
    st.plotly_chart(fig_pp, use_container_width=True)

    try:
        import plotly.io as pio
        png_b = pio.to_image(fig_pp, format="png", width=1200, height=520, scale=2)
        st.download_button(
            "⬇ Export Profile PNG", png_b,
            file_name="perplexity_profile.png", mime="image/png",
        )
    except Exception:
        st.caption("Install kaleido (`pip install kaleido`) for PNG export.")

    # Region architecture map
    if rv["regions"]:
        fig_map = plot_sequence_domain_map(
            len(rv["seq"]), rv["regions"],
            title="Region Architecture",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Regions table (all sequences)
    st.markdown('<p class="section-header">Regulatory Regions Table</p>',
                unsafe_allow_html=True)
    df = _regions_df(results)
    if df.empty:
        st.info(
            "No regulatory regions detected.  "
            "Try lowering the Score Threshold or decreasing Minimum Region Length."
        )
        return df

    st.dataframe(
        df.style.background_gradient(subset=["Mean Residual"], cmap="Blues_r"),
        use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download regions.csv", _df_to_csv(df),
            file_name="percall_regions.csv", mime="text/csv",
        )
    with col2:
        st.download_button(
            "⬇ Download regions.xlsx", _df_to_excel(df),
            file_name="percall_regions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    return df


# ============================================================================
# Tab 4 – Non-B DNA Analysis
# ============================================================================


def _tab_nonb_analysis(
    results: List[dict],
    active_motifs: set,
    window: int = 10,
) -> None:
    st.markdown('<p class="section-header">🔬 Non-B DNA Motif Analysis</p>',
                unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"]]
    if not valid:
        st.info("No valid sequences.")
        return

    ordered = [m for m in MOTIF_LABELS if m in active_motifs]
    if not ordered:
        st.info("Enable at least one motif in the sidebar.")
        return

    total_seq_bp = sum(len(r["seq"]) for r in valid)
    total_region_bp = sum(reg["width"] for r in valid for reg in r["regions"])
    bg_counts: Dict[str, int] = {m: 0 for m in ordered}
    reg_counts: Dict[str, int] = {m: 0 for m in ordered}
    motif_rows = []

    for r in valid:
        bg = count_motifs(r["seq"], active_motifs=set(ordered))
        for m in ordered:
            bg_counts[m] += bg.get(m, 0)
        for reg in r["regions"]:
            rs = _region_seq(r["seq"], reg, window)
            rc = count_motifs(rs, active_motifs=set(ordered))
            for m in ordered:
                reg_counts[m] += rc.get(m, 0)
            for m in ordered:
                if rc.get(m, 0) > 0:
                    motif_rows.append({
                        "Sequence": r["header"][:50],
                        "Region Rank": reg["rank"],
                        "Motif": MOTIF_LABELS.get(m, m),
                        "Count in Region": rc[m],
                    })

    col1, col2 = st.columns(2)
    with col1:
        fig_en = plot_motif_enrichment(
            reg_counts, bg_counts, total_region_bp, total_seq_bp, ordered,
        )
        st.plotly_chart(fig_en, use_container_width=True)
        try:
            import plotly.io as pio
            png_b = pio.to_image(fig_en, format="png", width=900, height=380, scale=2)
            st.download_button(
                "⬇ Export Enrichment PNG", png_b,
                file_name="motif_enrichment.png", mime="image/png",
            )
        except Exception:
            pass

    with col2:
        r0 = valid[0]
        mp = scan_motifs(r0["seq"], active_motifs=set(ordered))
        fig_dist = plot_motif_distribution(mp, len(r0["seq"]), ordered)
        st.plotly_chart(fig_dist, use_container_width=True)

    if motif_rows:
        st.markdown('<p class="section-header">Motifs within Called Regions</p>',
                    unsafe_allow_html=True)
        df_m = pd.DataFrame(motif_rows)
        st.dataframe(df_m, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download motifs.csv", _df_to_csv(df_m),
            file_name="percall_motifs.csv", mime="text/csv",
        )
    else:
        st.info("No motif hits found inside called regions.")


# ============================================================================
# Tab 5 – Interactive Genome View
# ============================================================================


def _tab_genome_view(results: List[dict], active_motifs: set) -> None:
    st.markdown('<p class="section-header">🖥 Interactive Genome View</p>',
                unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"]]
    if not valid:
        st.info("No valid sequences.")
        return

    ordered = [m for m in MOTIF_LABELS if m in active_motifs]
    seq_labels = [r["header"][:80] for r in valid]
    chosen_idx = 0
    if len(valid) > 1:
        chosen = st.selectbox("Sequence:", seq_labels, key="gv_seq_sel")
        chosen_idx = seq_labels.index(chosen)

    rv = valid[chosen_idx]
    mp = scan_motifs(rv["seq"], active_motifs=set(ordered))

    fig = plot_genome_browser(
        seq_len=len(rv["seq"]), regions=rv["regions"],
        motif_positions=mp, active_motifs=ordered,
        seq_label=rv["header"][:40],
    )
    st.plotly_chart(fig, use_container_width=True)

    if rv["regions"]:
        st.markdown('<p class="section-header">Region Coordinates</p>',
                    unsafe_allow_html=True)
        cols = st.columns(min(len(rv["regions"]), 5))
        for i, reg in enumerate(rv["regions"]):
            with cols[i % len(cols)]:
                st.markdown(
                    f'<div class="glass-card" style="padding:14px">'
                    f'<span class="blue-text">Region {reg["rank"]}</span><br>'
                    f'<span style="color:#c9d1d9;font-size:0.85rem">'
                    f'{reg["start"]}–{reg["end"]} bp &nbsp; '
                    f'Width: {reg["width"]} bp<br>'
                    f'Score: <span class="amber-text">{reg["mean_residual"]:.4f}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# Tab 6 – Results Explorer
# ============================================================================


def _tab_results_explorer(results: List[dict], params: dict) -> None:
    st.markdown('<p class="section-header">📂 Results Explorer</p>',
                unsafe_allow_html=True)

    df = _regions_df(results)

    if df.empty:
        st.info("No regions detected to explore.")
        return

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        min_w = int(df["Width"].min())
        max_w = int(df["Width"].max())
        w_range = st.slider("Filter by Width (bp)", min_w, max_w, (min_w, max_w))
    with col2:
        min_s = float(df["Mean Residual"].min())
        max_s = float(df["Mean Residual"].max())
        # Ensure a non-degenerate range so the slider renders correctly
        if min_s == max_s:
            min_s -= 0.001
        s_range = st.slider(
            "Filter by Mean Residual", min_s, max_s,
            (min_s, max_s), step=0.001, format="%.3f",
        )

    mask = (
        df["Width"].between(w_range[0], w_range[1])
        & df["Mean Residual"].between(s_range[0], s_range[1])
    )
    df_f = df[mask]
    st.markdown(f"Showing **{len(df_f)}** / {len(df)} regions")
    st.dataframe(
        df_f.style.background_gradient(subset=["Mean Residual"], cmap="Blues_r"),
        use_container_width=True, hide_index=True,
    )

    # Distribution charts
    if len(df_f) >= 2:
        fig_dist = plot_region_distribution(df_f)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('<p class="section-header">Export</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇ regions.csv", _df_to_csv(df_f),
            file_name="percall_regions_filtered.csv", mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇ regions.xlsx", _df_to_excel(df_f),
            file_name="percall_regions_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c3:
        json_bytes = _results_to_json(results, params)
        st.download_button(
            "⬇ session.json", json_bytes,
            file_name="percall_session.json", mime="application/json",
        )


# ============================================================================
# Tab 7 – Publication Report
# ============================================================================


def _tab_publication_report(
    results: List[dict],
    params: dict,
    active_motifs: set,
    regions_df: Optional[pd.DataFrame] = None,
) -> None:
    st.markdown('<p class="section-header">📑 Publication Report</p>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="glass-card">'
        '<div class="section-header">Report Contents</div>'
        '<div class="info-pill">✔ Run parameters</div>'
        '<div class="info-pill">✔ Methods summary</div>'
        '<div class="info-pill">✔ Sequence statistics table</div>'
        '<div class="info-pill">✔ Regulatory regions table</div>'
        '<div class="info-pill">✔ Non-B DNA motif summary</div>'
        '<div class="info-pill">✔ Perplexity profile figure</div>'
        '<div class="info-pill">✔ Motif enrichment figure</div>'
        '<div class="info-pill">✔ Tool citation</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("📄 Generate PDF Report", type="primary"):
        try:
            from core.report import generate_pdf
            from core.motifs import MOTIF_LABELS as _LABELS

            valid = [r for r in results if not r["skipped"]]
            seq_stats = [
                {
                    "header": r["header"][:60],
                    "length": len(r["seq"]),
                    "gc_pct": gc_pct(r["seq"]),
                    "n_regions": len(r["regions"]),
                }
                for r in results
            ]
            reg_records = (
                regions_df.to_dict("records")
                if regions_df is not None and not regions_df.empty
                else []
            )
            ordered = [m for m in MOTIF_LABELS if m in active_motifs]
            motif_counts: Dict[str, int] = {m: 0 for m in ordered}
            bg_counts: Dict[str, int] = {m: 0 for m in ordered}
            reg_counts: Dict[str, int] = {m: 0 for m in ordered}
            total_region_bp = 0
            total_seq_bp = 0
            for r in valid:
                mc = count_motifs(r["seq"], active_motifs=set(ordered))
                for m in ordered:
                    motif_counts[m] = motif_counts.get(m, 0) + mc.get(m, 0)
                    bg_counts[m] = bg_counts.get(m, 0) + mc.get(m, 0)
                total_seq_bp += len(r["seq"])
                for reg in r["regions"]:
                    total_region_bp += reg["width"]
                    rs = _region_seq(r["seq"], reg, params.get("window", 10))
                    rc = count_motifs(rs, active_motifs=set(ordered))
                    for m in ordered:
                        reg_counts[m] = reg_counts.get(m, 0) + rc.get(m, 0)

            r0 = valid[0] if valid else None
            with st.spinner("Generating PDF report…"):
                pdf_bytes = generate_pdf(
                    params=params,
                    seq_stats=seq_stats,
                    regions_df_records=reg_records,
                    motif_counts=motif_counts,
                    perp=r0["perp"] if r0 else None,
                    baseline=r0["baseline"] if r0 else None,
                    residual=r0["residual"] if r0 else None,
                    regions=r0["regions"] if r0 else None,
                    region_counts=reg_counts,
                    background_counts=bg_counts,
                    total_region_bp=total_region_bp,
                    total_seq_bp=total_seq_bp,
                    active_motifs=ordered,
                    motif_labels=_LABELS,
                    seq_label=r0["header"][:60] if r0 else "",
                )
            st.success("✔ PDF report generated!")
            st.download_button(
                "⬇ Download PERCALL_Report.pdf",
                data=pdf_bytes,
                file_name="PERCALL_Report.pdf",
                mime="application/pdf",
            )
        except ImportError as exc:
            st.error(
                f"PDF generation requires **fpdf2**.  "
                f"Install with: `pip install fpdf2`\n\n{exc}"
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    # Citation block
    st.markdown('<p class="section-header">Citation</p>',
                unsafe_allow_html=True)
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
# Main application
# ============================================================================


def main() -> None:
    params = _sidebar()
    _landing()

    fasta_text = params["fasta_text"]

    if not params["run"] and "percall_results" not in st.session_state:
        st.markdown("---")
        st.info(
            "👈 Upload a FASTA file or paste a sequence in the sidebar, "
            "then click **▶ Run PERCALL**."
        )
        return

    # Run analysis
    if params["run"] and fasta_text:
        records = parse_fasta(fasta_text)
        if not records:
            st.error("No valid sequences found.  Please check the FASTA format.")
            return

        progress = st.progress(0, text="Analysing sequences…")
        results = []
        for i, (header, seq) in enumerate(records):
            if len(seq) < params["min_len"] + params["window"]:
                results.append({
                    "header": header, "seq": seq,
                    "perp": np.array([]), "baseline": np.array([]),
                    "residual": np.array([]), "regions": [], "skipped": True,
                })
            else:
                results.append(process_sequence(
                    header, seq,
                    window=params["window"],
                    baseline_win=params["baseline_win"],
                    min_len=params["min_len"],
                    max_len=params["max_len"],
                    top_k=params["top_k"],
                    score_cutoff=params["score_cutoff"],
                    active_motifs=params["active_motifs"],
                ))
            progress.progress(
                (i + 1) / len(records),
                text=f"Processed {i + 1}/{len(records)} sequences",
            )
        progress.empty()
        st.session_state["percall_results"] = results
        st.session_state["percall_params"] = params

    elif "percall_results" in st.session_state:
        results = st.session_state["percall_results"]
        params = st.session_state.get("percall_params", params)
    else:
        if params["run"]:
            st.warning("Please provide a FASTA file or paste a sequence.")
        return

    # ── Seven result tabs ────────────────────────────────────────────────────
    st.markdown("---")
    (tab1, tab2, tab3, tab4, tab5, tab6, tab7) = st.tabs([
        "📊 Dashboard",
        "🔬 Sequence Analysis",
        "🗺 Region Calling",
        "🧬 Non-B DNA",
        "🖥 Genome View",
        "📂 Results Explorer",
        "📑 Publication Report",
    ])

    regions_df: pd.DataFrame = pd.DataFrame()

    with tab1:
        _tab_dashboard(results)

    with tab2:
        _tab_sequence_analysis(results, params)

    with tab3:
        regions_df = _tab_region_calling(results, params)

    with tab4:
        _tab_nonb_analysis(results, params["active_motifs"], window=params["window"])

    with tab5:
        _tab_genome_view(results, params["active_motifs"])

    with tab6:
        _tab_results_explorer(results, params)

    with tab7:
        _tab_publication_report(
            results, params, params["active_motifs"], regions_df=regions_df,
        )


main()
