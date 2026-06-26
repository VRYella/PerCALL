from __future__ import annotations

import base64
import html
import re
import sys
import time
from pathlib import Path

# Ensure local modules (regplex_core, motif_engine, visualization) are importable
# on all deployment environments including Streamlit Cloud.
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from motif_engine import annotate_domains, compile_motifs, iupac_to_regex
from regplex_core import (
    DEFAULT_SCALES,
    ENSEMBLE_METHOD,
    FLANK_WINDOW,
    LANDSCAPE_METHOD,
    MAX_CANDIDATE,
    MAX_DOMAIN,
    MERGE_GAP,
    MIN_CANDIDATE,
    MIN_DOMAIN,
    NMS_OVERLAP_THRESHOLD,
    NORMALIZATION_METHOD,
    PERPLEXITY_WINDOW,
    PERSISTENCE_THRESHOLD,
    SG_POLY_ORDER,
    SG_WINDOW_LENGTH,
    TOP_N_DISPLAY,
    AnalysisResult,
    analyze_sequence,
    domains_dataframe,
    export_bed,
    export_fasta,
    export_gff,
    export_table,
    parse_fasta,
)
from visualization import (
    plot_algorithm_workflow,
    plot_consensus_lpc,
    plot_domain_ranking,
    plot_kadane_domains,
    plot_layer_consensus,
    plot_motif_architecture,
    plot_multiscale_landscapes,
    plot_perplexity_layers,
    plot_scale_support_heatmap,
    plot_smoothed_perplexity,
)

st.set_page_config(page_title="REGPLEX", layout="wide", initial_sidebar_state="collapsed")

_NAV_ITEMS = ["Home", "Analysis", "Results", "Motifs", "About"]
_README_URL = "https://github.com/VRYella/PerCALL#readme"

PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "filename": "regplex_v11_figure", "scale": 2},
    "modeBarButtonsToAdd": ["resetScale2d"],
}


def _render_html_block(body: str) -> None:
    rendered = body.strip()
    if hasattr(st, "html"):
        st.html(rendered)
    else:
        st.markdown(rendered, unsafe_allow_html=True)


def _load_styles() -> None:
    css_path = Path(__file__).with_name("styles.css")
    try:
        css = css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error("styles.css not found.")
        return
    _render_html_block(f"<style>{css}</style>")


def _svg_logo() -> str:
    return """
<svg width="40" height="40" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="REGPLEX icon">
<rect x="3" y="3" width="58" height="58" rx="14" fill="#FFFFFF" stroke="#1E3A8A"/>
<path d="M16 18C24 18 24 32 32 32C40 32 40 18 48 18" stroke="#1E3A8A" stroke-width="3" stroke-linecap="round"/>
<path d="M16 28C24 28 24 42 32 42C40 42 40 28 48 28" stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
<path d="M17 47C22 43 27 41 32 41C37 41 42 43 47 47" stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
</svg>
"""


def _load_hero_image_b64() -> str | None:
    img_path = Path(__file__).with_name("regplexlogo.png")
    if not img_path.exists():
        return None
    return base64.b64encode(img_path.read_bytes()).decode()


def _load_example_text() -> str:
    example_path = Path(__file__).with_name("examples") / "ecoli.fasta"
    if not example_path.exists():
        return ""
    return example_path.read_text(encoding="utf-8")


def _jump_to_nav(label: str) -> None:
    st.session_state["jump_nav"] = label
    st.rerun()


def _render_topbar() -> None:
    _render_html_block(
        f"""
        <div class="regplex-topbar">
          <div class="regplex-topbar-inner">
            <div class="brand">{_svg_logo()}<div><h1>REGPLEX</h1><span>v11 · Signal-Processing Valley Detection</span></div></div>
            <div class="top-links">
              <a href="https://github.com/VRYella/PerCALL" target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href="{_README_URL}" target="_blank" rel="noopener noreferrer">Documentation</a>
              <span>Scientific Theme</span>
            </div>
          </div>
        </div>
        """
    )


def _render_nav() -> int:
    if "jump_nav" in st.session_state:
        st.session_state["main_nav"] = st.session_state.pop("jump_nav")
    tab = st.radio("Navigation", _NAV_ITEMS, horizontal=True, label_visibility="collapsed", key="main_nav")
    return _NAV_ITEMS.index(tab)


def _show_figure(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)


def _metric_card(label: str, value: str) -> None:
    _render_html_block(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>")


def _validate_motifs(motif_text: str) -> list[dict]:
    rows: list[dict] = []
    for raw in motif_text.splitlines():
        motif = raw.strip()
        if not motif:
            continue
        try:
            compiled = compile_motifs(motif)
            regex = compiled[0][1].pattern if compiled else iupac_to_regex(motif)
            rows.append({"Motif": motif, "Regex": regex, "Status": "valid"})
        except re.error:
            rows.append({"Motif": motif, "Regex": "invalid", "Status": "invalid"})
    return rows


def _run_analysis(fasta_text: str, params: dict, motif_text: str) -> tuple[list[AnalysisResult], float]:
    started = time.perf_counter()
    records = parse_fasta(fasta_text)
    motifs = compile_motifs(motif_text)
    results: list[AnalysisResult] = []
    for header, sequence in records:
        result = analyze_sequence(header, sequence, **params)
        annotate_domains(result.domains, motifs)
        results.append(result)
    return results, time.perf_counter() - started


def _render_home() -> None:
    img_b64 = _load_hero_image_b64()
    image_html = (
        f'<img src="data:image/png;base64,{img_b64}" class="hero-image" alt="REGPLEX v10 workflow"/>'
        if img_b64
        else '<div class="hero-image-missing">Hero image unavailable</div>'
    )
    _render_html_block(
        f"""
<div class="hero-section">
  <div class="hero-left">
    <div class="hero-brand">REGPLEX</div>
    <div class="hero-subtitle">
      <span class="hero-subtitle-line">Publication-grade Perplexity Valley Discovery</span>
      <span class="hero-subtitle-line">with Signal-Processing Candidate Refinement</span>
    </div>
    <p class="hero-desc">
      REGPLEX v11 computes mononucleotide, dinucleotide and trinucleotide perplexity once,
      applies Savitzky–Golay smoothing to preserve valley shape, builds multi-scale local contrast
      profiles, generates candidate valleys, refines each with Kadane's algorithm, then filters by
      persistence ≥ 80 %, adaptive prominence, and non-maximum suppression to produce a concise
      list of high-confidence regulatory valleys.
    </p>
    <div class="hero-chips">
      <span class="hero-chip">Savitzky–Golay smoothing</span>
      <span class="hero-chip">Multi-scale consensus</span>
      <span class="hero-chip">Persistence ≥ 80 %</span>
      <span class="hero-chip">Non-maximum suppression</span>
    </div>
  </div>
  <div class="hero-right">{image_html}</div>
</div>
"""
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Run Analysis", key="home_run", width="stretch", type="primary"):
            _jump_to_nav("Analysis")
    with b2:
        if st.button("Load Example", key="home_example", width="stretch"):
            st.session_state["input_fasta_text"] = _load_example_text()
            _jump_to_nav("Analysis")
    with b3:
        st.link_button("Documentation", _README_URL, width="stretch")
    with b4:
        st.link_button("GitHub", "https://github.com/VRYella/PerCALL", width="stretch")


def _render_analysis() -> None:
    _render_html_block("<div class='card'><h3>REGPLEX v11 Analysis</h3><p>Upload FASTA and configure the signal-processing detection pipeline.</p></div>")

    default_text = st.session_state.get("input_fasta_text", "")
    upload = st.file_uploader("Upload FASTA", type=["fasta", "fa", "fna", "txt"], key="analysis_file")
    pasted = st.text_area("Or paste FASTA", value=default_text, height=180, key="analysis_paste")

    fasta_text = ""
    uploaded_name = ""
    if upload is not None:
        fasta_text = upload.read().decode("utf-8", errors="replace")
        uploaded_name = upload.name
    elif pasted.strip():
        fasta_text = pasted if pasted.startswith(">") else f">query\n{pasted}"
        uploaded_name = "pasted_sequence.fasta"

    if fasta_text:
        parsed = parse_fasta(fasta_text)
        total_len = sum(len(seq) for _, seq in parsed)
        m1, m2, m3 = st.columns(3)
        with m1:
            _metric_card("Input", uploaded_name)
        with m2:
            _metric_card("Sequences", str(len(parsed)))
        with m3:
            _metric_card("Total length", f"{total_len:,} bp")

    st.markdown("#### Perplexity & Smoothing")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        perplexity_window = st.number_input("Perplexity window", 4, 50, PERPLEXITY_WINDOW)
        sg_window_length = st.number_input("SG window length (odd)", 5, 101, SG_WINDOW_LENGTH, step=2)
    with c2:
        sg_poly_order = st.number_input("SG polynomial order", 1, 9, SG_POLY_ORDER)
        landscape_method = st.selectbox("Landscape method", ["mean", "median"], index=0 if LANDSCAPE_METHOD == "mean" else 1)
    with c3:
        scales_text = st.text_input("Scales (comma-separated)", value=",".join(str(s) for s in DEFAULT_SCALES))
        normalization_method = st.selectbox("Layer normalization", ["robust_z", "percentile"], index=0 if NORMALIZATION_METHOD == "robust_z" else 1)
    with c4:
        ensemble_method = st.selectbox("Final ensemble", ["median", "trimmed_mean"], index=0 if ENSEMBLE_METHOD == "median" else 1)

    st.markdown("#### Detection & Filtering")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        min_candidate = st.number_input("Kadane min length (bp)", 10, 500, MIN_CANDIDATE)
        max_candidate = st.number_input("Kadane max length (bp)", int(min_candidate), 10000, max(int(min_candidate), MAX_CANDIDATE))
    with d2:
        min_domain = st.number_input("Min valley length (bp)", 20, 10000, MIN_DOMAIN)
        max_domain = st.number_input("Max valley length (bp)", int(min_domain), 50000, max(int(min_domain), MAX_DOMAIN))
    with d3:
        persistence_threshold = st.slider("Persistence threshold", 0.0, 1.0, PERSISTENCE_THRESHOLD, 0.05,
                                           help="Hard filter: fraction of valley positions with ConsensusLPC > 0")
        nms_overlap = st.slider("NMS overlap threshold", 0.1, 1.0, NMS_OVERLAP_THRESHOLD, 0.05,
                                 help="Suppress overlapping valleys above this overlap fraction")
    with d4:
        merge_gap = st.number_input("Merge gap (bp)", 0, 500, MERGE_GAP,
                                     help="Merge valleys within this genomic gap after NMS")
        top_n = st.number_input("Top N valleys to display", 1, 500, TOP_N_DISPLAY,
                                 help="Default display limit; use Show All to see every valley")

    motif_text = st.text_area("Motifs (Regex/IUPAC, one per line)", height=160, key="analysis_motifs")
    motif_rows = _validate_motifs(motif_text)
    if motif_rows:
        lines = []
        for row in motif_rows:
            klass = "motif-valid" if row["Status"] == "valid" else "motif-invalid"
            lines.append(f"<div class='{klass} mono'>{html.escape(row['Motif'])} → {html.escape(row['Regex'])}</div>")
        _render_html_block("".join(lines))

    run_col, reset_col, example_col = st.columns([2, 1, 1])
    with run_col:
        run_clicked = st.button("Run REGPLEX v11", type="primary", width="stretch")
    with reset_col:
        if st.button("Reset", width="stretch"):
            for key in ["results", "domains_df", "runtime", "input_fasta_text", "motif_text"]:
                st.session_state.pop(key, None)
            st.rerun()
    with example_col:
        if st.button("Load Example", width="stretch"):
            st.session_state["input_fasta_text"] = _load_example_text()
            st.rerun()

    if run_clicked:
        if not fasta_text:
            st.warning("No FASTA provided.")
            return
        try:
            scales = [int(part.strip()) for part in scales_text.split(",") if part.strip()]
            if not scales:
                raise ValueError
        except ValueError:
            st.warning("Scales must be comma-separated integers.")
            return

        params = {
            "perplexity_window": int(perplexity_window),
            "scales": scales,
            "landscape_method": landscape_method,
            "normalization_method": normalization_method,
            "ensemble_method": ensemble_method,
            "min_candidate": int(min_candidate),
            "max_candidate": int(max_candidate),
            "min_domain": int(min_domain),
            "max_domain": int(max_domain),
            "merge_gap": int(merge_gap),
            "sg_window_length": int(sg_window_length),
            "sg_poly_order": int(sg_poly_order),
            "persistence_threshold": float(persistence_threshold),
            "nms_overlap": float(nms_overlap),
        }

        with st.status("Running v11 signal-processing valley detection...", expanded=True) as status:
            for step in [
                "Step 1 · Mono/Di/Tri perplexity (single pass)",
                "Step 2 · Savitzky–Golay smoothing",
                "Step 3 · Multi-scale landscapes",
                "Step 4 · Three-window local contrast (LPC)",
                "Step 5 · Layer consensus",
                "Step 6 · Ensemble ConsensusLPC",
                "Step 7 · Candidate valley generation",
                "Step 8 · Kadane refinement per candidate",
                "Step 9 · Persistence filter (≥ 80 %)",
                "Step 10 · Adaptive prominence filter",
                "Step 11 · Non-maximum suppression",
                "Step 12 · Valley merging + ranking",
                "Step 13 · Motif annotation",
            ]:
                st.write(step)
            try:
                results, runtime_seconds = _run_analysis(fasta_text, params, motif_text)
            except re.error as exc:
                status.update(label=f"Invalid motif regex: {exc}", state="error")
                return
            st.session_state["results"] = results
            st.session_state["domains_df"] = domains_dataframe(results)
            st.session_state["runtime"] = runtime_seconds
            st.session_state["motif_text"] = motif_text
            st.session_state["top_n"] = int(top_n)
            status.update(label="Analysis complete", state="complete")


def _render_results(results: list[AnalysisResult], df: pd.DataFrame) -> None:
    if df.empty:
        _render_html_block("<div class='empty-state'><div>No valleys detected yet. Run Analysis first.</div></div>")
        return

    selected_seq = st.selectbox("Sequence", [result.sequence_id for result in results], key="results_sequence")
    result = next(item for item in results if item.sequence_id == selected_seq)
    selected_df = df[df["Sequence_ID"] == selected_seq].copy()

    runtime_seconds = float(st.session_state.get("runtime", 0.0))
    longest = int(selected_df["Length"].max()) if not selected_df.empty else 0
    top_score = float(selected_df["ValleyScore"].max()) if not selected_df.empty else 0.0
    support = selected_df["OverallSupport"].iloc[0] if not selected_df.empty else "0/0"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _metric_card("Valleys", str(len(result.domains)))
    with m2:
        _metric_card("Longest", f"{longest:,} bp")
    with m3:
        _metric_card("Top ValleyScore", f"{top_score:.4f}")
    with m4:
        _metric_card("Runtime", f"{runtime_seconds:.2f}s")
    st.caption(f"Representative support: {support}")

    tabs = st.tabs([
        "Raw & Smoothed",
        "Layer Perplexity",
        "Layer Landscapes",
        "Layer Consensus",
        "ConsensusLPC",
        "Support",
        "Kadane & Candidates",
        "Ranking",
        "Motifs",
        "Downloads",
    ])

    scales = result.params.get("resolved_scales", [])
    with tabs[0]:
        _show_figure(
            plot_smoothed_perplexity(result.mono, result.di, result.tri,
                                     result.smoothed_mono, result.smoothed_di, result.smoothed_tri,
                                     result.domains),
            f"smooth-{selected_seq}",
        )
    with tabs[1]:
        _show_figure(plot_perplexity_layers(result.mono, result.di, result.tri), f"layers-{selected_seq}")
    with tabs[2]:
        layer_tabs = st.tabs(["Mono", "Di", "Tri"])
        with layer_tabs[0]:
            _show_figure(plot_multiscale_landscapes(result.landscapes["mono"], result.domains, "Mono"), f"land-mono-{selected_seq}")
        with layer_tabs[1]:
            _show_figure(plot_multiscale_landscapes(result.landscapes["di"], result.domains, "Di"), f"land-di-{selected_seq}")
        with layer_tabs[2]:
            _show_figure(plot_multiscale_landscapes(result.landscapes["tri"], result.domains, "Tri"), f"land-tri-{selected_seq}")
    with tabs[3]:
        _show_figure(plot_layer_consensus(result.layer_consensus, result.domains), f"layer-cons-{selected_seq}")
    with tabs[4]:
        _show_figure(plot_consensus_lpc(result.consensus_lpc, result.domains), f"cons-{selected_seq}")
    with tabs[5]:
        _show_figure(plot_scale_support_heatmap(result.lpc_profiles, result.domains, scales), f"support-{selected_seq}")
    with tabs[6]:
        _show_figure(plot_kadane_domains(result.consensus_lpc, result.domains, result.kadane_core,
                                          result.candidates), f"kadane-{selected_seq}")
    with tabs[7]:
        _show_figure(plot_domain_ranking(result.domains), f"rank-{selected_seq}")

        # v11 display columns per spec (Step 13)
        display_cols = [
            col for col in [
                "Rank", "Start", "End", "Length",
                "MeanPerplexity", "MinPerplexity", "MaxPerplexity",
                "MeanLPC", "MaxLPC", "Prominence", "Area", "Persistence",
                "ScaleSupport", "MonoSupport", "DiSupport", "TriSupport",
                "Variance", "GC%", "ValleyScore", "Motifs", "Sequence",
            ]
            if col in selected_df.columns
        ]

        # Top-N display (Step 12)
        top_n = int(st.session_state.get("top_n", TOP_N_DISPLAY))
        show_all = st.toggle(f"Show all {len(selected_df)} valleys", value=False, key="show_all_valleys")
        display_df = selected_df if show_all else selected_df.nsmallest(top_n, "Rank")
        st.caption(
            f"Displaying {'all' if show_all else f'top {min(top_n, len(selected_df))}'} of {len(selected_df)} valleys, "
            f"ranked by ValleyScore."
        )
        st.dataframe(display_df[display_cols], width="stretch", hide_index=True)
    with tabs[8]:
        _show_figure(plot_motif_architecture(selected_df.to_dict("records")), f"motifs-{selected_seq}")
    with tabs[9]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("CSV", export_table(selected_df, "csv"), "regplex_v11_valleys.csv", "text/csv", width="stretch")
            st.download_button("Excel", export_table(selected_df, "xlsx"), "regplex_v11_valleys.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
            st.download_button("BED", export_bed(selected_df), "regplex_v11_valleys.bed", "text/plain", width="stretch")
        with c2:
            st.download_button("GFF", export_gff(selected_df, gff3=False), "regplex_v11_valleys.gff", "text/plain", width="stretch")
            st.download_button("GFF3", export_gff(selected_df, gff3=True), "regplex_v11_valleys.gff3", "text/plain", width="stretch")
            st.download_button("FASTA", export_fasta(selected_df), "regplex_v11_valleys.fasta", "text/plain", width="stretch")
        with c3:
            st.download_button("JSON", export_table(selected_df, "json"), "regplex_v11_valleys.json", "application/json", width="stretch")


def _render_motifs(df: pd.DataFrame) -> None:
    _render_html_block("<div class='card'><h3>Motif Annotation</h3><p>Validate motif syntax and inspect valley-level motif support.</p></div>")
    motif_text = st.text_area("Motif editor", value=st.session_state.get("motif_text", ""), height=220)
    rows = _validate_motifs(motif_text)
    if not rows:
        _render_html_block("<div class='empty-state'><div>No motifs entered.</div></div>")
        return
    for row in rows:
        klass = "motif-valid" if row["Status"] == "valid" else "motif-invalid"
        _render_html_block(f"<div class='{klass} mono'>{html.escape(row['Motif'])} → {html.escape(row['Regex'])}</div>")
    if not df.empty:
        st.markdown("---")
        st.dataframe(df[["ID", "Sequence_ID", "MotifCount", "Motifs", "OverallSupport", "ValleyScore"]], width="stretch", hide_index=True)


def _render_about() -> None:
    _render_html_block(
        """
        <div class="card">
          <h3>REGPLEX v11 Scientific Hypothesis</h3>
          <p>Regulatory regions are low-complexity genomic intervals that remain contrastive across independent information layers (mono/di/tri) and across multiple observation scales. A biologically meaningful valley must persist across the majority of evaluated positions (persistence ≥ 80 %) and be distinctly prominent relative to its genomic context.</p>
          <h3>Algorithm Overview</h3>
          <p>Each layer computes perplexity once, then a Savitzky–Golay filter (window 21, order 3) preserves valley shape while removing local noise. Multi-scale landscapes are built from the smoothed profiles; three-window LPC profiles are derived per scale and normalized, then median-combined to a layer consensus. The final ConsensusLPC is the median/trimmed-mean across layers. Candidate valleys are all contiguous positive runs; each is refined by a bounded Kadane search. Candidates are then filtered by persistence (≥ 0.80, hard), adaptive prominence (lower quartile), and non-maximum suppression (50 % overlap). Nearby survivors are merged (gap &lt; 25 bp) and ranked by the composite ValleyScore.</p>
        </div>
        """
    )
    _show_figure(plot_algorithm_workflow(), "workflow-v11")


def main() -> None:
    _load_styles()
    _render_topbar()
    active_tab = _render_nav()

    results: list[AnalysisResult] = st.session_state.get("results", [])
    df = st.session_state.get("domains_df", pd.DataFrame())

    if active_tab == 0:
        _render_home()
    elif active_tab == 1:
        _render_analysis()
    elif active_tab == 2:
        _render_results(results, df)
    elif active_tab == 3:
        _render_motifs(df)
    elif active_tab == 4:
        _render_about()

    st.markdown("---")
    st.markdown("REGPLEX v11 · Signal-Processing Valley Detection · MIT License")


if __name__ == "__main__":
    main()
