"""REGPLEX v13 – Streamlit Application

A training-free framework for identifying extended genomic perplexity valleys.
"""

from __future__ import annotations

import io
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import regplex_core as rc
import visualization as viz
from motif_engine import compile_motifs, annotate_domains

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="REGPLEX v13",
    page_icon=":dna:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimal custom CSS – scientific, clean
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1 { font-size: 32px !important; font-weight: 700 !important; color: #1E3A8A !important; }
    h2 { font-size: 24px !important; font-weight: 700 !important; color: #1f2937 !important; }
    h3 { font-size: 20px !important; font-weight: 600 !important; color: #1f2937 !important; }
    p, li, label, .stMarkdown { font-size: 18px !important; color: #1f2937 !important; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state["results"] = []  # list of AnalysisResult
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

LOGO_CANDIDATES = [
    Path(__file__).parent / "assets" / "regplex_logo.png",
    Path(__file__).parent / "regplex_logo.png",
]
_logo_path = next((p for p in LOGO_CANDIDATES if p.exists()), None)

col_logo, col_title = st.columns([1, 5])
with col_logo:
    if _logo_path:
        st.image(str(_logo_path), width=110)
with col_title:
    st.markdown("# REGPLEX v13")
    st.markdown(
        "<p style='color:#475569;font-size:16px!important;margin-top:-8px;'>"
        "A training-free framework for identifying extended genomic perplexity valleys — "
        "dinucleotide complexity · local contrast · Kadane detection"
        "</p>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    tab_input,
    tab_results,
    tab_plots,
    tab_motifs,
    tab_downloads,
    tab_about,
) = st.tabs([
    "📂  Input & Parameters",
    "📊  Valley Table",
    "🔬  Plots",
    "🧬  Motifs",
    "⬇️  Downloads",
    "ℹ️  About",
])

# ---------------------------------------------------------------------------
# TAB 1 – Input & Parameters
# ---------------------------------------------------------------------------

with tab_input:
    st.markdown("## Sequence Input")

    input_method = st.radio(
        "Input method",
        ["Paste FASTA", "Upload FASTA file"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_method == "Paste FASTA":
        raw_fasta = st.text_area(
            "Paste FASTA sequence(s)",
            height=200,
            placeholder=">sequence_1\nACGTACGTACGT...",
        )
    else:
        uploaded = st.file_uploader("Upload FASTA", type=["fa", "fasta", "fna", "txt"])
        raw_fasta = uploaded.read().decode("utf-8") if uploaded else ""

    st.markdown("## Algorithm Parameters")
    with st.expander("Expand parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Perplexity**")
            p_window = st.number_input("Perplexity window (nt)", 5, 51, 17, step=2)

            st.markdown("**Smoothing**")
            sg_window = st.number_input("SG window (nt, odd)", 5, 101, 21, step=2)
            sg_order = st.number_input("SG polynomial order", 2, 5, 3)

        with c2:
            st.markdown("**PDS windows**")
            flank_size = st.number_input("Flank size (bp)", 30, 500, 100)
            spacer_size = st.number_input("Spacer size (bp)", 0, 200, 50)
            min_cand = st.number_input("Min candidate (bp)", 20, 500, 50)
            max_cand = st.number_input("Max candidate (bp)", 100, 5000, 1000)

        with c3:
            st.markdown("**Valley detection**")
            min_valley = st.number_input("Min valley length (bp)", 50, 1000, 100, step=10)
            max_valley = st.number_input("Max valley length (bp)", 200, 10000, 1000, step=100)
            merge_gap = st.number_input("Merge gap (bp)", 0, 500, 100, step=10)

    analyze = st.button("Run REGPLEX v13", type="primary", use_container_width=True)

    if analyze:
        if not raw_fasta.strip():
            st.error("Please provide a FASTA sequence.")
        else:
            records = rc.parse_fasta(raw_fasta)
            if not records:
                st.error("Could not parse any sequences from the input.")
            else:
                results: list[rc.AnalysisResult] = []
                progress = st.progress(0, "Analysing sequences…")
                for i, (header, seq) in enumerate(records):
                    r = rc.analyze_sequence(
                        header,
                        seq,
                        perplexity_window=int(p_window),
                        sg_window=int(sg_window),
                        sg_order=int(sg_order),
                        flank_size=int(flank_size),
                        spacer_size=int(spacer_size),
                        min_candidate=int(min_cand),
                        max_candidate=int(max_cand),
                        min_valley_length=int(min_valley),
                        max_valley_length=int(max_valley),
                        merge_gap=int(merge_gap),
                    )
                    results.append(r)
                    progress.progress((i + 1) / len(records), f"Done: {header[:60]}")

                progress.empty()
                st.session_state["results"] = results

                # Flatten domains into a DataFrame
                rows = []
                for r in results:
                    for d in r.domains:
                        d_copy = dict(d)
                        d_copy.setdefault("Sequence_ID", r.sequence_id)
                        rows.append(d_copy)
                st.session_state["df"] = pd.DataFrame(rows) if rows else pd.DataFrame()

                n_total = sum(len(r.domains) for r in results)
                if n_total:
                    st.success(
                        f"Detected **{n_total}** perplexity valley(s) across "
                        f"{len(results)} sequence(s)."
                    )
                else:
                    st.warning(
                        "No perplexity valleys detected. "
                        "Try adjusting flank size, spacer, or valley length parameters."
                    )

# ---------------------------------------------------------------------------
# TAB 2 – Valley Table
# ---------------------------------------------------------------------------

with tab_results:
    st.markdown("## Detected Perplexity Valleys")
    df = st.session_state["df"]
    if df.empty:
        st.info("Run the analysis to see results.")
    else:
        # Columns to display (hide raw Sequence from table; keep for downloads)
        _hide = {"Sequence", "Motifs"}
        display_cols = [c for c in df.columns if c not in _hide]

        st.markdown(f"**{len(df)} valley(s) detected** across "
                    f"{df['Sequence_ID'].nunique() if 'Sequence_ID' in df.columns else '?'} "
                    f"sequence(s).")

        # Format numeric columns
        fmt = {}
        for col in df.columns:
            if df[col].dtype == float:
                fmt[col] = "{:.4f}"

        st.dataframe(
            df[display_cols].style.format(fmt, na_rep="—"),
            use_container_width=True,
            height=520,
        )

# ---------------------------------------------------------------------------
# TAB 3 – Plots
# ---------------------------------------------------------------------------

with tab_plots:
    results = st.session_state["results"]
    if not results:
        st.info("Run the analysis to see plots.")
    else:
        # Sequence selector
        seq_ids = [r.sequence_id for r in results]
        selected_id = st.selectbox("Sequence", seq_ids, label_visibility="collapsed")
        r = next((x for x in results if x.sequence_id == selected_id), results[0])

        sub_tabs = st.tabs([
            "Raw perplexity",
            "Smoothed perplexity",
            "PDS landscape",
            "Three-window",
            "Valley ranking",
            "Workflow",
        ])

        with sub_tabs[0]:
            st.plotly_chart(
                viz.plot_perplexity_landscape(r.di, r.domains),
                use_container_width=True,
            )

        with sub_tabs[1]:
            st.plotly_chart(
                viz.plot_smoothed_perplexity_landscape(r.di, r.smoothed_di, r.domains),
                use_container_width=True,
            )

        with sub_tabs[2]:
            st.plotly_chart(
                viz.plot_pds_landscape(r.pds, r.domains),
                use_container_width=True,
            )

        with sub_tabs[3]:
            if not r.domains:
                st.info("No valleys detected for this sequence.")
            else:
                valley_ids = [d["ID"] for d in r.domains]
                sel_v = st.selectbox("Select valley", valley_ids, key="three_window_sel")
                sel_domain = next((d for d in r.domains if d["ID"] == sel_v), r.domains[0])
                st.plotly_chart(
                    viz.plot_three_window(r.smoothed_di, sel_domain,
                                          r.params.get("flank_size", 100),
                                          r.params.get("spacer_size", 50)),
                    use_container_width=True,
                )

        with sub_tabs[4]:
            if not r.domains:
                st.info("No valleys detected for this sequence.")
            else:
                st.plotly_chart(
                    viz.plot_valley_ranking(r.domains),
                    use_container_width=True,
                )

        with sub_tabs[5]:
            st.plotly_chart(viz.plot_algorithm_workflow(), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 – Motifs
# ---------------------------------------------------------------------------

with tab_motifs:
    st.markdown("## Motif Annotation")
    df = st.session_state["df"]
    results = st.session_state["results"]

    if df.empty or not results:
        st.info("Run the analysis first, then annotate motifs.")
    else:
        motif_input = st.text_area(
            "Enter motifs (one per line) — IUPAC or regex",
            height=120,
            placeholder="TATAWAWR\nGGGN{1,7}GGG\n(CAG){5,}",
        )
        run_motifs = st.button("Annotate motifs", type="primary")

        if run_motifs:
            motifs_list = [m.strip() for m in motif_input.strip().splitlines() if m.strip()]
            if not motifs_list:
                st.warning("No motifs entered.")
            else:
                # Annotate each result and rebuild df
                all_rows = []
                compiled = compile_motifs("\n".join(motifs_list))
                for r in results:
                    r.domains = annotate_domains(r.domains, compiled)
                    for d in r.domains:
                        d_copy = dict(d)
                        d_copy.setdefault("Sequence_ID", r.sequence_id)
                        all_rows.append(d_copy)
                if all_rows:
                    st.session_state["df"] = pd.DataFrame(all_rows)
                    df = st.session_state["df"]
                st.success(f"Annotated {len(motifs_list)} motif(s) across {len(df)} valley(s).")

        if "MotifCount" in df.columns:
            motif_df = df[df["MotifCount"] > 0]
            if not motif_df.empty:
                st.markdown(f"**{len(motif_df)} valley(s) with motif matches:**")
                display_cols = [c for c in ["ID", "Sequence_ID", "Start", "End", "Length",
                                             "MotifCount", "Motifs", "ValleyScore", "Rank"]
                                if c in df.columns]
                st.dataframe(motif_df[display_cols], use_container_width=True)
            else:
                st.info("No motif matches found in detected valleys.")

# ---------------------------------------------------------------------------
# TAB 5 – Downloads
# ---------------------------------------------------------------------------

with tab_downloads:
    st.markdown("## Download Results")
    df = st.session_state["df"]
    results = st.session_state["results"]

    if df.empty:
        st.info("Run the analysis to enable downloads.")
    else:
        # Reference sequence dict for FASTA/GFF3
        _seq_dict = {r.sequence_id: r.sequence for r in results if hasattr(r, "sequence")}

        c1, c2 = st.columns(2)

        # --- CSV ---
        with c1:
            csv_buf = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv_buf.encode("utf-8"),
                file_name="regplex_v13_valleys.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # --- Excel ---
        with c2:
            xl_buf = io.BytesIO()
            df.to_excel(xl_buf, index=False, sheet_name="Valleys")
            st.download_button(
                "Download Excel",
                data=xl_buf.getvalue(),
                file_name="regplex_v13_valleys.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # --- BED ---
        with c1:
            bed_lines = []
            for _, row in df.iterrows():
                name = row.get("ID", ".")
                score = int(min(1000, float(row.get("ValleyScore", 0)) * 100))
                start = int(row.get("Start", 0))
                end = int(row.get("End", 0))
                seq_id = row.get("Sequence_ID", "unknown")
                bed_lines.append(f"{seq_id}\t{start}\t{end}\t{name}\t{score}\t.")
            bed_str = "\n".join(bed_lines) + "\n"
            st.download_button(
                "Download BED",
                data=bed_str.encode("utf-8"),
                file_name="regplex_v13_valleys.bed",
                mime="text/plain",
                use_container_width=True,
            )

        # --- GFF3 ---
        with c2:
            gff_lines = ["##gff-version 3"]
            for _, row in df.iterrows():
                seq_id = row.get("Sequence_ID", "unknown")
                start = int(row.get("Start", 0)) + 1
                end = int(row.get("End", 0))
                score = f"{float(row.get('ValleyScore', 0)):.4f}"
                v_id = row.get("ID", ".")
                pds = row.get("PDSMean", ".")
                gff_lines.append(
                    f"{seq_id}\tREGPLEX_v13\tperplexity_valley\t{start}\t{end}"
                    f"\t{score}\t.\t.\tID={v_id};PDSMean={pds}"
                )
            gff_str = "\n".join(gff_lines) + "\n"
            st.download_button(
                "Download GFF3",
                data=gff_str.encode("utf-8"),
                file_name="regplex_v13_valleys.gff3",
                mime="text/plain",
                use_container_width=True,
            )

        # --- FASTA ---
        with c1:
            fa_lines = []
            for _, row in df.iterrows():
                seq_str = row.get("Sequence", "")
                if seq_str:
                    v_id = row.get("ID", "seq")
                    seq_id = row.get("Sequence_ID", "")
                    fa_lines.append(f">{v_id} {seq_id} {row.get('Start',0)}-{row.get('End',0)}")
                    for chunk in textwrap.wrap(seq_str, 70):
                        fa_lines.append(chunk)
            fa_str = "\n".join(fa_lines) + "\n"
            st.download_button(
                "Download FASTA",
                data=fa_str.encode("utf-8"),
                file_name="regplex_v13_valleys.fasta",
                mime="text/plain",
                use_container_width=True,
            )

        # --- JSON ---
        with c2:
            json_rows = df.copy()
            for col in json_rows.select_dtypes(include="number").columns:
                json_rows[col] = json_rows[col].astype(float)
            json_str = json_rows.to_json(orient="records", indent=2)
            st.download_button(
                "Download JSON",
                data=json_str.encode("utf-8"),
                file_name="regplex_v13_valleys.json",
                mime="application/json",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# TAB 6 – About
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown("## REGPLEX v13")
    st.markdown(
        """
REGPLEX is a **training-free computational framework** that identifies extended genomic
perplexity valleys — regions where local sequence complexity is significantly lower than
the surrounding genomic background — using dinucleotide perplexity, local contrast analysis,
bounded Kadane optimization, and interpretable valley ranking.

### Hypothesis
> Identify extended low-perplexity genomic valleys relative to their local genomic background
> using a training-free information-theoretic framework.

### Algorithm
| Step | Description |
|------|-------------|
| 1 | Dinucleotide perplexity (17 nt sliding window) |
| 2 | Savitzky–Golay smoothing (21 bp, polynomial order 3) |
| 3 | Perplexity Depression Score — three-window contrast |
| 4 | Bounded Kadane detection (100–1000 bp) |
| 5 | Valley expansion (PDS threshold) |
| 6 | Valley merging (gap ≤ 100 bp) |
| 7 | Biological filter + valley metrics |
| 8 | ValleyScore ranking |
| 9 | Optional motif annotation |

### Output columns
`ID · Start · End · Length · MeanPerplexity · MinPerplexity · MaxPerplexity ·
UpstreamMean · CandidateMean · DownstreamMean · UpstreamDifference · DownstreamDifference ·
PDSMean · PDSMax · Prominence · Persistence · AreaUnderValley · Variance ·
GC% · MotifCount · Motifs · Sequence · ValleyScore · ValleyScoreNormalized · Rank`

### ValleyScore formula
```
ValleyScore = PDSMean × Persistence × log(Length) × (1 / (Variance + 1e-9))
```
Normalized 0–1. No arbitrary weights.

### Supported motif patterns
- IUPAC ambiguity codes (e.g. `TATAWAWR`)
- Regular expressions (e.g. `GGGN{1,7}GGG`)
- Repeats (e.g. `(CAG){5,}`)

---
REGPLEX v13 · No training · No ML · No species-specific parameters
        """
    )
