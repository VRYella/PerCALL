"""
streamlit_app.py
────────────────
PERCALL — PERplexity-based Regulatory Region CALLer
=====================================================

Web-based bioinformatics platform that identifies low-perplexity regulatory
DNA regions using information-theoretic sequence analysis and a bounded
minimum-mean Kadane optimisation algorithm.

Author  : Dr. Venkata Rajesh Yella
Version : 2025.1
Licence : MIT
"""

from __future__ import annotations

import io
import sys
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.perplexity import compute_perplexity, local_residual
from core.region_caller import find_regions
from core.motifs import (
    MOTIF_PATTERNS,
    MOTIF_LABELS,
    MOTIF_COLORS,
    scan_motifs,
    count_motifs,
)
from core.plotting import (
    plot_perplexity_profile,
    plot_motif_enrichment,
    plot_motif_distribution,
    plot_genome_browser,
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

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Main title */
    .percall-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1565c0;
        margin-bottom: 0.2rem;
    }
    .percall-subtitle {
        font-size: 1.1rem;
        color: #455a64;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #e8f0fe;
        border-radius: 8px;
        padding: 10px 14px;
    }
    /* Download buttons */
    .stDownloadButton button {
        background-color: #1565c0;
        color: white;
        border-radius: 6px;
    }
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }
    /* Section divider */
    .percall-section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1565c0;
        border-bottom: 2px solid #1565c0;
        padding-bottom: 4px;
        margin: 18px 0 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# FASTA parser (lightweight, no BioPython dependency)
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


def _region_seq(seq: str, region: dict, window: int) -> str:
    """Extract the sequence for a region, extending by *window* for motif context."""
    return seq[region["start"] : min(region["end"] + window, len(seq))]


def gc_pct(seq: str) -> float:
    if not seq:
        return 0.0
    return round(100.0 * (seq.count("G") + seq.count("C")) / len(seq), 2)


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
        return {"header": header, "seq": seq, "perp": perp,
                "baseline": perp.copy(), "residual": perp.copy(),
                "regions": [], "skipped": True}

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


# ============================================================================
# Sidebar
# ============================================================================


def sidebar() -> dict:
    """Render sidebar and return parameter dict + raw FASTA text."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/000000/dna-helix.png",
            width=60,
        )
        st.title("PERCALL")
        st.caption("PERplexity-based Regulatory Region CALLer")
        st.divider()

        # ── Input options ──────────────────────────────────────────────────
        st.markdown("### 📂 Input Options")
        upload = st.file_uploader(
            "Upload FASTA File",
            type=["fasta", "fa", "fna", "txt"],
            help="Multi-FASTA files are supported.",
        )
        st.markdown("**OR**")
        pasted = st.text_area(
            "Paste Sequence",
            height=100,
            placeholder=">my_sequence\nATGCATGCATGC...",
            help="Single or multi-FASTA format accepted.",
        )
        use_example = st.checkbox("Use example FASTA", value=False)

        fasta_text: Optional[str] = None
        if upload is not None:
            fasta_text = upload.read().decode("utf-8", errors="replace")
        elif pasted.strip():
            fasta_text = pasted.strip()
            if not fasta_text.startswith(">"):
                fasta_text = f">pasted_sequence\n{fasta_text}"
        elif use_example:
            example_path = os.path.join(_ROOT, "assets", "example.fasta")
            if os.path.isfile(example_path):
                with open(example_path) as fh:
                    fasta_text = fh.read()

        st.divider()

        # ── Perplexity settings ────────────────────────────────────────────
        st.markdown("### ⚙️ Parameters")
        st.markdown("**Perplexity Settings**")
        window = st.slider(
            "Window Size",
            min_value=4,
            max_value=30,
            value=10,
            step=1,
            help="Sliding window size (bp) for dinucleotide perplexity calculation.",
        )

        # ── Region detection ───────────────────────────────────────────────
        st.markdown("**Region Detection**")
        min_len = st.slider(
            "Minimum Region Length (bp)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
        )
        max_len = st.slider(
            "Maximum Region Length (bp)",
            min_value=50,
            max_value=1000,
            value=300,
            step=10,
        )
        baseline_win = st.slider(
            "Baseline Window (bp)",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            help="Rolling window for local GC-drift compensation.",
        )
        top_k = st.slider(
            "Number of Regions",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )
        score_cutoff = st.slider(
            "Score Threshold",
            min_value=-1.0,
            max_value=0.0,
            value=-0.05,
            step=0.01,
            format="%.2f",
            help="Regions with mean residual ≥ this are not reported.",
        )

        # ── Motif detection ────────────────────────────────────────────────
        st.markdown("**Motif Detection**")
        motif_selection: Dict[str, bool] = {}
        for key, label in MOTIF_LABELS.items():
            motif_selection[key] = st.checkbox(label, value=True, key=f"motif_{key}")

        active_motifs = {k for k, v in motif_selection.items() if v}

        st.divider()
        run = st.button("▶  Run PERCALL", type="primary", use_container_width=True)

    return dict(
        fasta_text=fasta_text,
        window=window,
        min_len=min_len,
        max_len=max_len,
        baseline_win=baseline_win,
        top_k=top_k,
        score_cutoff=score_cutoff,
        active_motifs=active_motifs,
        run=run,
    )


# ============================================================================
# Landing page (always shown)
# ============================================================================


def landing_page() -> None:
    st.markdown(
        '<p class="percall-title">🧬 PERCALL</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="percall-subtitle">PERplexity-based Regulatory Region CALLer</p>',
        unsafe_allow_html=True,
    )

    with st.expander("📖 About PERCALL", expanded=True):
        st.markdown(
            textwrap.dedent(
                """
                **PERCALL** (*PERplexity-based Regulatory Region CALLer*) is an
                information-theoretic framework for identifying low-perplexity regulatory DNA
                regions.

                The method computes **dinucleotide sequence perplexity profiles** and applies
                an **optimal bounded minimum-mean subarray algorithm** (Kadane family, O(n)) to
                detect candidate regulatory domains.

                Detected regions are further characterised using integrated **non-B DNA motif
                annotation**, including G-quadruplexes, i-motifs, Z-DNA, eGZ-DNA,
                triplex-forming sequences, and repetitive DNA elements.

                ---
                **How to use**
                1. Upload a FASTA file or paste a sequence in the left sidebar.
                2. Adjust analysis parameters if needed (defaults work well for most promoters).
                3. Click **▶ Run PERCALL**.
                4. Explore the six result tabs below.
                """
            )
        )


# ============================================================================
# Tab 1 – Sequence Statistics
# ============================================================================


def tab_sequence_stats(results: List[dict]) -> None:
    st.markdown('<p class="percall-section-header">Sequence Statistics</p>', unsafe_allow_html=True)

    total_seqs = len(results)
    valid = [r for r in results if not r["skipped"]]
    total_bp = sum(len(r["seq"]) for r in results)
    mean_len = total_bp / total_seqs if total_seqs else 0
    overall_gc = (
        sum(gc_pct(r["seq"]) * len(r["seq"]) for r in results) / total_bp
        if total_bp
        else 0
    )
    total_regions = sum(len(r["regions"]) for r in valid)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total sequences", total_seqs)
    c2.metric("Total bp", f"{total_bp:,}")
    c3.metric("Mean length (bp)", f"{mean_len:,.0f}")
    c4.metric("Overall GC%", f"{overall_gc:.1f}%")
    c5.metric("Regions detected", total_regions)

    st.markdown("---")
    rows = []
    for r in results:
        rows.append(
            {
                "Header": r["header"][:60],
                "Length (bp)": len(r["seq"]),
                "GC%": gc_pct(r["seq"]),
                "Regions": len(r["regions"]),
                "Status": "Skipped" if r["skipped"] else "OK",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download summary CSV
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download summary.csv",
        data=csv_bytes,
        file_name="summary.csv",
        mime="text/csv",
    )


# ============================================================================
# Tab 2 – Perplexity Profile
# ============================================================================


def tab_perplexity_profile(results: List[dict]) -> None:
    st.markdown('<p class="percall-section-header">Perplexity Profile</p>', unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"] and r["perp"].size > 0]
    if not valid:
        st.info("No valid sequences to display.")
        return

    # Sequence selector for multi-FASTA
    seq_labels = [r["header"][:80] for r in valid]
    chosen_idx = 0
    if len(valid) > 1:
        chosen_label = st.selectbox("Select sequence to display:", seq_labels)
        chosen_idx = seq_labels.index(chosen_label)

    r = valid[chosen_idx]
    fig = plot_perplexity_profile(
        r["perp"],
        r["baseline"],
        r["residual"],
        r["regions"],
        title=f"PERCALL — {r['header'][:60]}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Export PNG via Plotly
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", width=1200, height=520, scale=2)
        st.download_button(
            "⬇ Export as PNG",
            data=png_bytes,
            file_name="perplexity_profile.png",
            mime="image/png",
        )
    except Exception:
        st.caption("Install kaleido (`pip install kaleido`) to enable PNG export.")


# ============================================================================
# Tab 3 – Regulatory Regions
# ============================================================================


def tab_regulatory_regions(results: List[dict]) -> pd.DataFrame:
    st.markdown('<p class="percall-section-header">Regulatory Regions</p>', unsafe_allow_html=True)

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

    if not rows:
        st.info("No regulatory regions detected with the current parameters. "
                "Try lowering the Score Threshold or decreasing Minimum Region Length.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["Mean Residual"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download regions.csv",
        data=csv_bytes,
        file_name="regions.csv",
        mime="text/csv",
    )
    return df


# ============================================================================
# Tab 4 – Non-B DNA Motifs
# ============================================================================


def tab_motif_analysis(
    results: List[dict],
    active_motifs: set,
    window: int = 10,
) -> None:
    st.markdown('<p class="percall-section-header">Non-B DNA Motifs</p>', unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"]]
    if not valid:
        st.info("No valid sequences.")
        return

    ordered_motifs = [m for m in MOTIF_LABELS if m in active_motifs]
    if not ordered_motifs:
        st.info("No motifs selected.  Enable at least one motif in the sidebar.")
        return

    # Aggregate counts across all sequences
    total_seq_bp = sum(len(r["seq"]) for r in valid)
    total_region_bp = sum(reg["width"] for r in valid for reg in r["regions"])

    background_counts: Dict[str, int] = {m: 0 for m in ordered_motifs}
    region_counts: Dict[str, int] = {m: 0 for m in ordered_motifs}
    motif_rows = []

    for r in valid:
        bg = count_motifs(r["seq"], active_motifs=set(ordered_motifs))
        for m in ordered_motifs:
            background_counts[m] += bg.get(m, 0)

        for reg in r["regions"]:
            region_seq = _region_seq(r["seq"], reg, window)
            rc = count_motifs(region_seq, active_motifs=set(ordered_motifs))
            for m in ordered_motifs:
                region_counts[m] += rc.get(m, 0)
            for m in ordered_motifs:
                if rc.get(m, 0) > 0:
                    motif_rows.append(
                        {
                            "Sequence": r["header"][:50],
                            "Region Rank": reg["rank"],
                            "Motif": MOTIF_LABELS.get(m, m),
                            "Count in Region": rc[m],
                        }
                    )

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        fig_enrich = plot_motif_enrichment(
            region_counts,
            background_counts,
            total_region_bp,
            total_seq_bp,
            ordered_motifs,
        )
        st.plotly_chart(fig_enrich, use_container_width=True)
        try:
            import plotly.io as pio
            png_bytes = pio.to_image(fig_enrich, format="png", width=900, height=420, scale=2)
            st.download_button(
                "⬇ Export enrichment PNG",
                data=png_bytes,
                file_name="motif_enrichment.png",
                mime="image/png",
            )
        except Exception:
            pass

    with col2:
        # Choose first valid sequence for distribution plot
        r0 = valid[0]
        mp = scan_motifs(r0["seq"], active_motifs=set(ordered_motifs))
        fig_dist = plot_motif_distribution(mp, len(r0["seq"]), ordered_motifs)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Motif detail table
    if motif_rows:
        st.markdown("**Motifs found within called regions**")
        df_m = pd.DataFrame(motif_rows)
        st.dataframe(df_m, use_container_width=True, hide_index=True)
        csv_bytes = df_m.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download motifs.csv",
            data=csv_bytes,
            file_name="motifs.csv",
            mime="text/csv",
        )
    else:
        st.info("No motif hits found inside called regions.")

    return background_counts, region_counts, total_region_bp, total_seq_bp


# ============================================================================
# Tab 5 – Genome Browser View
# ============================================================================


def tab_genome_browser(results: List[dict], active_motifs: set) -> None:
    st.markdown('<p class="percall-section-header">Genome Browser View</p>', unsafe_allow_html=True)

    valid = [r for r in results if not r["skipped"]]
    if not valid:
        st.info("No valid sequences.")
        return

    ordered_motifs = [m for m in MOTIF_LABELS if m in active_motifs]
    seq_labels = [r["header"][:80] for r in valid]
    chosen_idx = 0
    if len(valid) > 1:
        chosen_label = st.selectbox(
            "Select sequence:", seq_labels, key="browser_seq_select"
        )
        chosen_idx = seq_labels.index(chosen_label)

    r = valid[chosen_idx]
    mp = scan_motifs(r["seq"], active_motifs=set(ordered_motifs))

    fig = plot_genome_browser(
        seq_len=len(r["seq"]),
        regions=r["regions"],
        motif_positions=mp,
        active_motifs=ordered_motifs,
        seq_label=r["header"][:40],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Textual map legend
    if r["regions"]:
        st.markdown("**Region coordinates:**")
        cols = st.columns(min(len(r["regions"]), 5))
        for i, reg in enumerate(r["regions"]):
            with cols[i % len(cols)]:
                st.markdown(
                    f"**Region {reg['rank']}**  \n"
                    f"{reg['start']}–{reg['end']} bp  \n"
                    f"Width: {reg['width']} bp  \n"
                    f"Trough: {reg['trough']} bp"
                )


# ============================================================================
# Tab 6 – Publication Report
# ============================================================================


def tab_publication_report(
    results: List[dict],
    params: dict,
    active_motifs: set,
    regions_df: Optional[pd.DataFrame] = None,
) -> None:
    st.markdown('<p class="percall-section-header">Publication Report</p>', unsafe_allow_html=True)

    st.markdown(
        """
        Generate a publication-quality PDF report containing:
        - Run parameters
        - Sequence statistics table
        - Regulatory regions table
        - Motif summary
        - Perplexity profile figure
        - Motif enrichment figure
        """
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

            reg_records = []
            if regions_df is not None and not regions_df.empty:
                reg_records = regions_df.to_dict("records")

            # Motif counts (first valid sequence)
            ordered_motifs = [m for m in MOTIF_LABELS if m in active_motifs]
            motif_counts: Dict[str, int] = {m: 0 for m in ordered_motifs}
            bg_counts: Dict[str, int] = {m: 0 for m in ordered_motifs}
            reg_counts: Dict[str, int] = {m: 0 for m in ordered_motifs}
            total_region_bp = 0
            total_seq_bp = 0

            for r in valid:
                mc = count_motifs(r["seq"], active_motifs=set(ordered_motifs))
                for m in ordered_motifs:
                    motif_counts[m] = motif_counts.get(m, 0) + mc.get(m, 0)
                    bg_counts[m] = bg_counts.get(m, 0) + mc.get(m, 0)
                total_seq_bp += len(r["seq"])
                for reg in r["regions"]:
                    total_region_bp += reg["width"]
                    rs = _region_seq(r["seq"], reg, params.get("window", 10))
                    rc = count_motifs(rs, active_motifs=set(ordered_motifs))
                    for m in ordered_motifs:
                        reg_counts[m] = reg_counts.get(m, 0) + rc.get(m, 0)

            # Use first valid sequence for the profile figure
            r0 = valid[0] if valid else None

            with st.spinner("Generating PDF…"):
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
                    active_motifs=ordered_motifs,
                    motif_labels=_LABELS,
                    seq_label=r0["header"][:60] if r0 else "",
                )

            st.success("PDF report generated!")
            st.download_button(
                "⬇ Download PERCALL_Report.pdf",
                data=pdf_bytes,
                file_name="PERCALL_Report.pdf",
                mime="application/pdf",
            )
        except ImportError as exc:
            st.error(
                f"PDF generation requires the **fpdf2** package.  "
                f"Install it with: `pip install fpdf2`\n\n{exc}"
            )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")


# ============================================================================
# Main application
# ============================================================================


def main() -> None:
    params = sidebar()

    landing_page()

    fasta_text = params["fasta_text"]

    if not params["run"] and "percall_results" not in st.session_state:
        st.markdown("---")
        st.info(
            "👈  Upload a FASTA file or paste a sequence in the sidebar, then click **▶ Run PERCALL**."
        )
        return

    # Run analysis
    if params["run"] and fasta_text:
        records = parse_fasta(fasta_text)
        if not records:
            st.error("No valid sequences found in the input.  Please check FASTA format.")
            return

        progress = st.progress(0, text="Analysing sequences…")
        results = []
        for i, (header, seq) in enumerate(records):
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
            progress.progress((i + 1) / len(records), text=f"Processed {i+1}/{len(records)} sequences")

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

    # ── Results tabs ────────────────────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Sequence Statistics",
            "📈 Perplexity Profile",
            "🗺 Regulatory Regions",
            "🔬 Non-B DNA Motifs",
            "🖥 Genome Browser",
            "📑 Publication Report",
        ]
    )

    regions_df: pd.DataFrame = pd.DataFrame()

    with tab1:
        tab_sequence_stats(results)

    with tab2:
        tab_perplexity_profile(results)

    with tab3:
        regions_df = tab_regulatory_regions(results)

    with tab4:
        tab_motif_analysis(results, params["active_motifs"], window=params["window"])

    with tab5:
        tab_genome_browser(results, params["active_motifs"])

    with tab6:
        tab_publication_report(
            results,
            params,
            params["active_motifs"],
            regions_df=regions_df,
        )


main()
