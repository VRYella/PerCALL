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
    CORE_WINDOW_DOWNSTREAM,
    CORE_WINDOW_UPSTREAM,
    DEFAULT_MODE,
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
    plot_kadane_domains,
    plot_domain_ranking,
    plot_motif_architecture,
    plot_scale_support_heatmap,
    plot_smoothed_perplexity,
)

st.set_page_config(page_title="REGPLEX", layout="wide", initial_sidebar_state="collapsed")

_NAV_ITEMS = ["Home", "Analysis", "Results", "Motifs", "About"]
_README_URL = "https://github.com/VRYella/PerCALL#readme"

# non-B DNA motifs: A/T homopolymer tracts, G/C homopolymer tracts, G-quadruplex-like and C-quadruplex-like patterns
_DEFAULT_NON_B_DNA_MOTIFS = (
    "A{7}|T{7}",
    "G{7}|C{7}",
    "G{3,5}[ACGT]{1,7}G{3,5}[ACGT]{1,7}G{3,5}[ACGT]{1,7}G{3,5}",
    "C{3,5}[ACGT]{1,7}C{3,5}[ACGT]{1,7}C{3,5}[ACGT]{1,7}C{3,5}",
)

# Promoter motifs in IUPAC order:
# TATA-box, Initiator, TCT-element, BREu, BREd, XCPE1, DRE, MTE, DPE, Pause Button
_DEFAULT_PROMOTER_IUPAC_MOTIFS = (
    "TATAWAWR",
    "BBCABW",
    "YYCTTTYY",
    "SSRCGCC",
    "RTDKKKK",
    "DSGYGGRASM",
    "WATCGATW",
    "CSARCSSAACGS",
    "RGWCGTG",
    "KCGRWCG",
)

PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "filename": "regplex_v12_figure", "scale": 2},
    "modeBarButtonsToAdd": ["resetScale2d"],
}

_RESET_SESSION_KEYS = (
    "results",
    "domains_df",
    "runtime",
    "input_fasta_text",
    "motif_text",
    "analysis_custom_motif_text",
    "motifs_custom_motif_text",
)


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
            <div class="brand">{_svg_logo()}<div><h1>REGPLEX</h1><span>v12 · Di-centric Perplexity Valley Detector</span></div></div>
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


def _combine_motif_text(custom_text: str) -> str:
    """Return a newline-separated motif set containing built-ins plus user-added motifs."""
    default_lines = [*_DEFAULT_NON_B_DNA_MOTIFS, *_DEFAULT_PROMOTER_IUPAC_MOTIFS]
    custom_lines = [line.strip() for line in custom_text.splitlines() if line.strip()]
    return "\n".join([*default_lines, *custom_lines])


def _current_custom_motifs(primary_key: str, fallback_key: str) -> str:
    return st.session_state.get(primary_key, st.session_state.get(fallback_key, ""))


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
        f'<img src="data:image/png;base64,{img_b64}" class="hero-image" alt="REGPLEX v12 workflow"/>'
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
      REGPLEX v12 uses dinucleotide perplexity as the primary detection signal,
      applies Savitzky–Golay smoothing, builds multi-scale local contrast profiles,
      generates candidates from Di consensus LPC, refines each with Kadane's algorithm,
      then filters by persistence ≥ 80 %, prominence, non-maximum suppression, and merging.
      Mono and Tri layers remain as interpretability support only.
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
    _render_html_block(
        "<div class='card'>"
        "<h3>🔬 REGPLEX Analysis</h3>"
        "<p>Upload or paste a FASTA sequence, tune the valley-detection parameters, then run the pipeline.</p>"
        "</div>"
    )

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

    st.markdown("#### ⚙️ Detection Parameters")
    mode = st.selectbox(
        "Operating mode",
        options=["promoter", "genome", "ensemble"],
        index=["promoter", "genome", "ensemble"].index(DEFAULT_MODE),
        help="promoter: Di + positional prior (default); genome: Di only; ensemble: Mono+Di+Tri exploratory mode.",
    )

    core_window_upstream: int | None = None
    core_window_downstream: int | None = None
    reference_point: int | None = 0
    if mode == "promoter":
        st.caption("Promoter mode uses the default positional prior window [-500, +200] around reference point 0.")
        mpos1, mpos2, mpos3 = st.columns(3)
        with mpos1:
            reference_point = int(st.number_input("Reference point", value=0, step=1))
        with mpos2:
            core_window_upstream = int(st.number_input("Core window upstream (bp)", min_value=0, value=CORE_WINDOW_UPSTREAM))
        with mpos3:
            core_window_downstream = int(st.number_input("Core window downstream (bp)", min_value=0, value=CORE_WINDOW_DOWNSTREAM))
    elif mode == "genome":
        st.caption("Genome mode disables positional prior and performs whole-genome discovery.")
        reference_point = None
        core_window_upstream = None
        core_window_downstream = None
    else:
        st.warning("Ensemble mode is experimental and not recommended for primary predictions.")
        use_positional_prior = st.checkbox("Apply positional prior in ensemble mode", value=False)
        if use_positional_prior:
            mpos1, mpos2, mpos3 = st.columns(3)
            with mpos1:
                reference_point = int(st.number_input("Reference point", value=0, step=1))
            with mpos2:
                core_window_upstream = int(st.number_input("Core window upstream (bp)", min_value=0, value=CORE_WINDOW_UPSTREAM))
            with mpos3:
                core_window_downstream = int(st.number_input("Core window downstream (bp)", min_value=0, value=CORE_WINDOW_DOWNSTREAM))
        else:
            reference_point = None
            core_window_upstream = None
            core_window_downstream = None

    p1, p2, p3 = st.columns(3)
    with p1:
        min_domain = st.number_input(
            "Min valley length (bp)",
            min_value=MIN_DOMAIN, max_value=10000, value=MIN_DOMAIN,
            help="Minimum output region size in nucleotides (default 50 bp).",
        )
    with p2:
        persistence_threshold = st.slider(
            "Persistence threshold",
            0.0, 1.0, PERSISTENCE_THRESHOLD, 0.05,
            help="Fraction of valley positions with ConsensusLPC > 0 (higher = stricter)",
        )
    with p3:
        top_n = st.number_input(
            "Top N valleys to display",
            1, 500, TOP_N_DISPLAY,
            help="Display limit; toggle Show All in Results to see every valley",
        )

    with st.expander("Advanced detection parameters"):
        a1, a2, a3 = st.columns(3)
        with a1:
            max_domain = st.number_input(
                "Max valley length (bp)",
                min_value=MIN_DOMAIN, max_value=10000, value=MAX_DOMAIN,
                help="Maximum final valley length and Kadane refinement bound.",
            )
            merge_gap = st.number_input(
                "Merge gap (bp)",
                min_value=0, max_value=1000, value=MERGE_GAP,
                help="Merge post-NMS valleys when the gap is smaller than this value.",
            )
        with a2:
            sg_window_length = st.number_input(
                "Savitzky–Golay window",
                min_value=3, max_value=401, value=SG_WINDOW_LENGTH, step=2,
                help="Odd smoothing window length applied once to each perplexity profile.",
            )
            nms_overlap = st.slider(
                "NMS overlap threshold",
                0.0, 1.0, NMS_OVERLAP_THRESHOLD, 0.05,
                help="Suppress overlapping valleys when their overlap fraction exceeds this value.",
            )
        with a3:
            sg_poly_order = st.number_input(
                "Savitzky–Golay order",
                min_value=1, max_value=10, value=SG_POLY_ORDER,
                help="Polynomial order for Savitzky–Golay smoothing.",
            )
            max_candidate = st.number_input(
                "LPC candidate max length (bp)",
                min_value=MIN_CANDIDATE, max_value=10000, value=MAX_CANDIDATE,
                help="Upper bound for the scale-adapted LPC candidate window.",
            )

    st.markdown("#### 🧩 Built-in Motif Boxes (always included)")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Non-B DNA motifs (fixed)**")
        st.code("\n".join(_DEFAULT_NON_B_DNA_MOTIFS), language="text")
    with b2:
        st.markdown("**Promoter motifs (IUPAC, fixed)**")
        st.code("\n".join(_DEFAULT_PROMOTER_IUPAC_MOTIFS), language="text")

    custom_motif_text = st.text_area(
        "Add more motifs (Regex/IUPAC, one per line)",
        value=_current_custom_motifs("analysis_custom_motif_text", "motifs_custom_motif_text"),
        height=120,
        key="analysis_custom_motif_text",
    )
    motif_text = _combine_motif_text(custom_motif_text)
    motif_rows = _validate_motifs(motif_text)
    st.caption(f"Annotating with {len(motif_rows)} motif patterns (built-in + custom).")
    if motif_rows:
        lines = []
        for row in motif_rows:
            klass = "motif-valid" if row["Status"] == "valid" else "motif-invalid"
            lines.append(f"<div class='{klass} mono'>{html.escape(row['Motif'])} → {html.escape(row['Regex'])}</div>")
        _render_html_block("".join(lines))

    run_col, reset_col, example_col = st.columns([2, 1, 1])
    with run_col:
        run_clicked = st.button("▶ Run REGPLEX", type="primary", width="stretch")
    with reset_col:
        if st.button("↺ Reset", width="stretch"):
            for key in _RESET_SESSION_KEYS:
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

        max_domain = max(int(max_domain), int(min_domain))
        max_candidate = max(int(max_candidate), int(min_domain))
        params = {
            "perplexity_window": PERPLEXITY_WINDOW,
            "scales": DEFAULT_SCALES,
            "landscape_method": LANDSCAPE_METHOD,
            "normalization_method": NORMALIZATION_METHOD,
            "ensemble_method": ENSEMBLE_METHOD,
            "mode": mode,
            "core_window_upstream": core_window_upstream,
            "core_window_downstream": core_window_downstream,
            "reference_point": reference_point,
            "min_candidate": MIN_CANDIDATE,
            "max_candidate": max_candidate,
            "min_domain": int(min_domain),
            "max_domain": max_domain,
            "merge_gap": int(merge_gap),
            "sg_window_length": int(sg_window_length),
            "sg_poly_order": int(sg_poly_order),
            "persistence_threshold": float(persistence_threshold),
            "nms_overlap": float(nms_overlap),
        }

        with st.status("Running REGPLEX v12 di-centric valley detection…", expanded=False) as status:
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
            n_seqs = len(results)
            seq_word = "sequence" if n_seqs == 1 else "sequences"
            status.update(label=f"✅ Analysis complete — {n_seqs} {seq_word} processed", state="complete")


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
        "📈 Perplexity",
        "🎯 ConsensusLPC",
        "🧭 Candidate Refinement",
        "🌡️ Scale Support",
        "🏆 Valley Ranking",
        "🔎 Motifs",
        "⬇️ Downloads",
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
        _show_figure(plot_consensus_lpc(result.consensus_lpc, result.domains), f"cons-{selected_seq}")
    with tabs[2]:
        _show_figure(
            plot_kadane_domains(result.consensus_lpc, result.domains, result.kadane_core, result.candidates),
            f"candidate-refine-{selected_seq}",
        )
    with tabs[3]:
        _show_figure(plot_scale_support_heatmap(result.lpc_profiles, result.domains, scales), f"support-{selected_seq}")
    with tabs[4]:
        _show_figure(plot_domain_ranking(result.domains), f"rank-{selected_seq}")

        display_cols = [
            col for col in [
                "Rank", "Start", "End", "Length",
                "MeanPerplexity", "MinPerplexity", "MaxPerplexity",
                "MeanLPC", "MaxLPC", "Prominence", "Area", "Persistence",
                "ScaleSupport", "MonoSupport", "DiSupport", "TriSupport",
                "Variance", "GC%", "ValleyScore", "ValleyScoreNormalized", "Motifs", "Sequence",
            ]
            if col in selected_df.columns
        ]

        # Top-N display
        top_n = int(st.session_state.get("top_n", TOP_N_DISPLAY))
        show_all = st.toggle(f"Show all {len(selected_df)} valleys", value=False, key="show_all_valleys")
        display_df = selected_df if show_all else selected_df.nsmallest(top_n, "Rank")
        st.caption(
            f"Displaying {'all' if show_all else f'top {min(top_n, len(selected_df))}'} of {len(selected_df)} valleys, "
            f"ranked by ValleyScore."
        )
        st.dataframe(display_df[display_cols], width="stretch", hide_index=True)
    with tabs[5]:
        _show_figure(plot_motif_architecture(selected_df.to_dict("records")), f"motifs-{selected_seq}")
    with tabs[6]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("CSV", export_table(selected_df, "csv"), "regplex_v12_valleys.csv", "text/csv", width="stretch")
            st.download_button("Excel", export_table(selected_df, "xlsx"), "regplex_v12_valleys.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
            st.download_button("BED", export_bed(selected_df), "regplex_v12_valleys.bed", "text/plain", width="stretch")
        with c2:
            st.download_button("GFF", export_gff(selected_df, gff3=False), "regplex_v12_valleys.gff", "text/plain", width="stretch")
            st.download_button("GFF3", export_gff(selected_df, gff3=True), "regplex_v12_valleys.gff3", "text/plain", width="stretch")
            st.download_button("FASTA", export_fasta(selected_df), "regplex_v12_valleys.fasta", "text/plain", width="stretch")
        with c3:
            st.download_button("JSON", export_table(selected_df, "json"), "regplex_v12_valleys.json", "application/json", width="stretch")


def _render_motifs(df: pd.DataFrame) -> None:
    _render_html_block("<div class='card'><h3>Motif Annotation</h3><p>Validate motif syntax and inspect valley-level motif support.</p></div>")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Non-B DNA motifs (fixed)**")
        st.code("\n".join(_DEFAULT_NON_B_DNA_MOTIFS), language="text")
    with m2:
        st.markdown("**Promoter motifs (IUPAC, fixed)**")
        st.code("\n".join(_DEFAULT_PROMOTER_IUPAC_MOTIFS), language="text")
    custom_text = st.text_area(
        "Add more motifs",
        value=_current_custom_motifs("motifs_custom_motif_text", "analysis_custom_motif_text"),
        height=220,
        key="motifs_custom_motif_text",
    )
    motif_text = _combine_motif_text(custom_text)
    rows = _validate_motifs(motif_text)
    st.caption(f"Active motif patterns: {len(rows)} (built-in + custom).")
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
          <h3>🧬 Scientific Hypothesis</h3>
          <p>REGPLEX v12 is <strong>di-centric</strong>: dinucleotide perplexity is the primary detection signal because it best captures local dependency and structural organization. Mono and Tri are retained as support evidence for interpretability, not as primary decision layers.</p>
          <h3 style="margin-top:1rem">⚙️ Algorithm Overview</h3>
          <p>Dinucleotide perplexity is smoothed by <strong>Savitzky–Golay</strong>, converted to multi-scale landscapes (25, 50, 100, 200, 400), transformed to LPC, and collapsed to a Di consensus LPC. Candidate valleys are refined by Kadane, filtered by persistence, prominence, and NMS, then merged.</p>
          <p><strong>Operating modes:</strong> promoter (default, Di + positional prior), genome (Di only, no positional prior), ensemble (Mono+Di+Tri exploratory mode). Positional prior defaults to [-500,+200] around the reference point and can be disabled by setting windows to None.</p>
        </div>
        """
    )
    _show_figure(plot_algorithm_workflow(), "workflow-v12")


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
    st.markdown("**REGPLEX v12** · Di-centric Perplexity Valley Detector · MIT License")


if __name__ == "__main__":
    main()
