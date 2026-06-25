from __future__ import annotations

import html
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from motif_engine import annotate_domains, compile_motifs, iupac_to_regex
from regplex_core import (
    BASE_SCALE,
    LANDSCAPE_METHOD,
    MAX_CANDIDATE,
    MAX_DOMAIN,
    MERGE_GAP,
    MIN_CANDIDATE,
    MIN_DOMAIN,
    PERPLEXITY_WINDOW,
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
    plot_motif_architecture,
    plot_multiscale_landscapes,
    plot_p1_profile,
    plot_scale_support_heatmap,
)

st.set_page_config(page_title="REGPLEX", layout="wide", initial_sidebar_state="collapsed")

# Navigation labels used throughout.
_NAV_ITEMS = ["Home", "Analysis", "Results", "Motifs", "About"]

PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "filename": "regplex_figure", "scale": 2},
    "modeBarButtonsToAdd": ["resetScale2d"],
}

# Map nav labels to integer indices.
_NAV_INDEX = {label: i for i, label in enumerate(_NAV_ITEMS)}

# Module-level SVG icon path fragments for feature cards (avoids recreation on each render).
_ICON_PATHS_TRAINING_FREE = (
    '<path d="M11 21H31" stroke="#1E3A8A" stroke-width="2.2"/>'
    '<path d="M21 11V31" stroke="#0F766E" stroke-width="2.2"/>'
)
_ICON_PATHS_GENOME_SCALE = (
    '<path d="M10 26L18 18L24 24L32 16" stroke="#1E3A8A" stroke-width="2.2"/>'
    '<circle cx="32" cy="16" r="2.5" fill="#0F766E"/>'
)
_ICON_PATHS_EXPLAINABLE = (
    '<path d="M10 30L32 12" stroke="#1E3A8A" stroke-width="2.2"/>'
    '<path d="M10 12H32V30" stroke="#0F766E" stroke-width="2.2"/>'
)


def _svg_square_logo() -> str:
    return """
<svg width="40" height="40" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="REGPLEX icon">
<rect x="3" y="3" width="58" height="58" rx="14" fill="#FFFFFF" stroke="#1E3A8A"/>
<path d="M16 18C24 18 24 32 32 32C40 32 40 18 48 18" stroke="#1E3A8A" stroke-width="3" stroke-linecap="round"/>
<path d="M16 28C24 28 24 42 32 42C40 42 40 28 48 28" stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
<path d="M17 47C22 43 27 41 32 41C37 41 42 43 47 47" stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
</svg>
"""


def _svg_horizontal_logo() -> str:
    return """
<svg width="220" height="44" viewBox="0 0 220 44" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="REGPLEX logo">
<path d="M8 11C16 11 16 23 24 23C32 23 32 11 40 11" stroke="#1E3A8A" stroke-width="2.8" stroke-linecap="round"/>
<path d="M8 19C16 19 16 31 24 31C32 31 32 19 40 19" stroke="#0F766E" stroke-width="2.8" stroke-linecap="round"/>
<path d="M41 26C49 31 57 31 65 26" stroke="#0F766E" stroke-width="2.8" stroke-linecap="round"/>
<text x="74" y="28" fill="#1E3A8A" font-size="22" font-family="Inter, Arial, sans-serif" font-weight="700" letter-spacing="1">REGPLEX</text>
</svg>
"""


def _svg_feature(icon: str) -> str:
    return f"""
<svg width="42" height="42" viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<rect x="1" y="1" width="40" height="40" rx="12" fill="#F8FAFC" stroke="#1E3A8A"/>
{icon}
</svg>
"""


def _svg_empty() -> str:
    return """
<svg width="150" height="86" viewBox="0 0 150 86" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<path d="M10 24C24 24 24 44 38 44C52 44 52 24 66 24" stroke="#1E3A8A" stroke-width="3" stroke-linecap="round"/>
<path d="M10 36C24 36 24 56 38 56C52 56 52 36 66 36" stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
<path d="M74 56C90 43 106 43 122 56" stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
<circle cx="136" cy="56" r="4" fill="#F59E0B"/>
</svg>
"""


def _load_styles() -> None:
    css_path = Path(__file__).with_name("styles.css")
    try:
        css = css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error("styles.css not found. Please ensure styles.css exists in the same directory as app.py.")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _jump_to_nav(label: str) -> None:
    """Request a navigation jump on the next rerun."""
    st.session_state["jump_nav"] = label
    st.rerun()


def _render_nav() -> int:
    """Custom full-width navigation.

    Renders a horizontal radio widget styled by CSS to look like a professional
    tab bar.  Returns the active tab index (0 = Home … 4 = About).
    """
    # Apply any programmatic jump requested before rendering the widget.
    if "jump_nav" in st.session_state:
        st.session_state["main_nav"] = st.session_state.pop("jump_nav")

    tab = st.radio(
        "Navigation",
        _NAV_ITEMS,
        horizontal=True,
        label_visibility="collapsed",
        key="main_nav",
    )
    return _NAV_INDEX[tab]


def _show_figure(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key)


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


def _render_topbar() -> None:
    st.markdown(
        f"""
        <div class="regplex-topbar">
          <div class="regplex-topbar-inner">
            <div class="brand">
              {_svg_square_logo()}
              <div>
                <h1>REGPLEX</h1>
                <span>v9 · Multi-scale Consensus UI</span>
              </div>
            </div>
            <div class="top-links">
              <span>Workflow UI</span>
              <a href="https://github.com/VRYella/PerCALL" target="_blank">GitHub</a>
              <a href="https://github.com/VRYella/PerCALL#readme" target="_blank">Documentation</a>
              <a href="https://github.com/VRYella/PerCALL#citation" target="_blank">Citation</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _render_empty(message: str) -> None:
    st.markdown(
        f"<div class='empty-state'>{_svg_empty()}<div>{message}</div></div>",
        unsafe_allow_html=True,
    )


def _load_example_text() -> str:
    example_path = Path(__file__).with_name("examples") / "ecoli.fasta"
    if not example_path.exists():
        st.warning("Example FASTA not found at examples/ecoli.fasta (relative to app.py).")
        return ""
    return example_path.read_text(encoding="utf-8")


def _render_home() -> None:
    st.markdown(
        f"""
        <section class="hero">
          {_svg_horizontal_logo()}
          <p><strong>Regulatory Region Discovery through Perplexity Valleys</strong></p>
          <p>Training-free discovery of localized sequence uncertainty collapse.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("Start Analysis", key="home_start_analysis", use_container_width=True, type="primary"):
            _jump_to_nav("Analysis")
    with c2:
        if st.button("Load Example", key="home_load_example", use_container_width=True):
            st.session_state["input_fasta_text"] = _load_example_text()
            _jump_to_nav("Analysis")
    with c3:
        st.link_button("Documentation", "https://github.com/VRYella/PerCALL#readme", use_container_width=True)
    with c4:
        st.link_button("GitHub", "https://github.com/VRYella/PerCALL", use_container_width=True)

    _icon_tf = _svg_feature(_ICON_PATHS_TRAINING_FREE)
    _icon_gs = _svg_feature(_ICON_PATHS_GENOME_SCALE)
    _icon_ex = _svg_feature(_ICON_PATHS_EXPLAINABLE)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            f"<div class='card'>{_icon_tf}<h4>Training-free</h4>"
            "<p>No pre-trained model required. Sequence uncertainty alone drives discovery.</p></div>",
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f"<div class='card'>{_icon_gs}<h4>Genome-scale</h4>"
            "<p>Efficient profile construction for long contiguous genomic sequences.</p></div>",
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            f"<div class='card'>{_icon_ex}<h4>Explainable</h4>"
            "<p>Every valley includes interpretable metrics and explicit motif evidence.</p></div>",
            unsafe_allow_html=True,
        )


def _validate_motifs(motif_text: str) -> list[dict]:
    rows: list[dict] = []
    for raw in motif_text.splitlines():
        motif = raw.strip()
        if not motif:
            continue
        try:
            compiled = compile_motifs(motif)
            if compiled:
                regex = compiled[0][1].pattern
            else:
                regex = iupac_to_regex(motif)
            rows.append({"Motif": motif, "Regex": regex, "Status": "valid"})
        except re.error:
            rows.append({"Motif": motif, "Regex": "invalid", "Status": "invalid"})
    return rows


def _render_analysis() -> None:
    st.markdown(
        "<div class='card'><h3>Analysis Input</h3>"
        "<p>Upload FASTA or paste sequence text.</p></div>",
        unsafe_allow_html=True,
    )

    default_text = st.session_state.get("input_fasta_text", "")
    upload = st.file_uploader(
        "Drag-and-drop FASTA",
        type=["fasta", "fa", "fna", "txt"],
        help="Supported formats: FASTA, FA, FNA, TXT",
        key="analysis_file_uploader",
    )
    pasted = st.text_area(
        "Or paste FASTA",
        value=default_text,
        height=180,
        placeholder=">sequence\nACTG...",
        help="Input is parsed as FASTA records and sanitized to DNA/IUPAC-compatible sequence characters.",
        key="analysis_paste_fasta",
    )

    fasta_text = ""
    uploaded_name = ""
    pasted_text = pasted.strip()
    if upload is not None:
        fasta_text = upload.read().decode("utf-8", errors="replace")
        uploaded_name = upload.name
    elif pasted_text:
        fasta_text = pasted_text if pasted_text.startswith(">") else f">query\n{pasted_text}"
        uploaded_name = "pasted_sequence.fasta"

    if fasta_text:
        parsed = parse_fasta(fasta_text)
        total_len = sum(len(seq) for _, seq in parsed)
        m1, m2, m3 = st.columns(3)
        with m1:
            _metric_card("Filename", uploaded_name)
        with m2:
            _metric_card("Sequence Count", str(len(parsed)))
        with m3:
            _metric_card("Genome Length", f"{total_len:,} bp")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        perplexity_window = st.number_input(
            "P1 window", 4, 50, PERPLEXITY_WINDOW,
            help="k-mer window for dinucleotide perplexity",
            key="param_perplexity_window",
        )
        landscape_method = st.selectbox(
            "Landscape method", ["mean", "median"],
            index=0 if LANDSCAPE_METHOD == "mean" else 1,
            key="param_landscape_method",
        )
    with p2:
        base_scale = st.number_input(
            "Base scale (bp)", 20, 2000, BASE_SCALE,
            help=(
                "Centre of 5 auto-generated observation scales: "
                "base÷4, base÷2, base, base×2, base×4"
            ),
            key="param_base_scale",
        )
        custom_scales_text = st.text_input(
            "Custom scales (comma-separated, overrides base scale)",
            value="",
            placeholder="e.g. 25,50,100,200,400",
            key="param_custom_scales",
        )
        scales_list = None
        if custom_scales_text.strip():
            try:
                parsed_scales = [
                    int(s.strip())
                    for s in custom_scales_text.split(",")
                    if s.strip()
                ]
                invalid = [s for s in parsed_scales if s <= 0]
                if invalid:
                    st.warning(
                        f"Custom scales must be positive integers. "
                        f"Invalid values: {invalid}"
                    )
                else:
                    scales_list = parsed_scales
            except ValueError:
                st.warning("Custom scales must be comma-separated integers.")
    with p3:
        min_candidate = st.number_input(
            "Min candidate (bp)", 10, 500, MIN_CANDIDATE,
            help="Minimum candidate window size per scale (≥ 50 bp enforced)",
            key="param_min_candidate",
        )
        max_candidate = st.number_input(
            "Max candidate (bp)", int(min_candidate), 5000,
            max(int(min_candidate), MAX_CANDIDATE),
            key="param_max_candidate",
        )
        min_domain = st.number_input(
            "Min valley (bp)", 20, 10000, MIN_DOMAIN,
            key="param_min_domain",
        )
        max_domain = st.number_input(
            "Max valley (bp)", int(min_domain), 20000,
            max(int(min_domain), MAX_DOMAIN),
            key="param_max_domain",
        )
        merge_gap = st.number_input(
            "Merge gap (bp)", 0, 500, MERGE_GAP,
            help="Merge valleys closer than this distance",
            key="param_merge_gap",
        )
    with p4:
        motif_text = st.text_area(
            "Motifs (Regex/IUPAC, one per line)",
            height=190, key="param_motif_text",
        )
        motif_rows = _validate_motifs(motif_text)
        if motif_rows:
            status_lines = [
                f"<div class='motif-valid mono'>{html.escape(row['Motif'])}"
                f" → {html.escape(row['Regex'])}</div>"
                if row["Status"] == "valid"
                else f"<div class='motif-invalid mono'>{html.escape(row['Motif'])}"
                f" → invalid regex</div>"
                for row in motif_rows
            ]
            st.markdown("".join(status_lines), unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1.5, 1, 1])
    with b1:
        run_clicked = st.button(
            "Run Analysis", key="analysis_run",
            type="primary", use_container_width=True,
        )
    with b2:
        if st.button("Reset", key="analysis_reset", use_container_width=True):
            for key in [
                "results", "domains_df", "runtime",
                "input_fasta_text", "motif_text",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
    with b3:
        if st.button(
            "Load Example", key="analysis_load_example", use_container_width=True
        ):
            st.session_state["input_fasta_text"] = _load_example_text()
            st.rerun()

    if run_clicked:
        if not fasta_text:
            st.warning("No genome loaded.")
            return

        params = {
            "perplexity_window": int(perplexity_window),
            "base_scale": int(base_scale),
            "scales": scales_list,
            "landscape_method": str(landscape_method),
            "min_candidate": int(min_candidate),
            "max_candidate": int(max_candidate),
            "min_domain": int(min_domain),
            "max_domain": int(max_domain),
            "merge_gap": int(merge_gap),
        }

        with st.status("Running REGPLEX v9 pipeline...", expanded=True) as status:
            steps = [
                "Step 1 · Computing P1 Perplexity",
                "Step 2 · Building Multi-scale Landscapes",
                "Step 3 · Computing Per-scale LPC Profiles",
                "Step 4 · Building Consensus LPC",
                "Step 5 · Kadane + Valley Expansion + Merging",
                "Step 6 · Annotating Motifs",
            ]
            for step in steps:
                st.write(step)
            try:
                results, runtime_seconds = _run_analysis(
                    fasta_text, params, motif_text
                )
            except re.error as exc:
                status.update(
                    label=f"Failed: invalid motif pattern ({exc})",
                    state="error",
                )
                return
            st.session_state["results"] = results
            st.session_state["domains_df"] = domains_dataframe(results)
            st.session_state["runtime"] = runtime_seconds
            st.session_state["motif_text"] = motif_text
            status.update(label="Analysis complete", state="complete")


def _render_results(results: list[AnalysisResult], df: pd.DataFrame) -> None:
    if df.empty:
        _render_empty("No genome loaded.")
        return

    runtime_seconds = float(st.session_state.get("runtime", 0.0))
    selected_seq = st.selectbox("Sequence", [result.sequence_id for result in results], key="results_selected_seq")
    result = next(item for item in results if item.sequence_id == selected_seq)
    selected_df = df[df["Sequence_ID"] == selected_seq].copy()

    longest = int(selected_df["Length"].max()) if not selected_df.empty else 0
    top_score = float(selected_df["ValleyScore"].max()) if not selected_df.empty else 0.0
    gc = float(selected_df["GC%"].mean()) if not selected_df.empty else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        _metric_card("Detected Valleys", str(len(result.domains)))
    with m2:
        _metric_card("Longest Valley", f"{longest:,} bp")
    with m3:
        _metric_card("Highest Score", f"{top_score:.4f}")
    with m4:
        _metric_card("Runtime", f"{runtime_seconds:.2f}s")
    with m5:
        _metric_card("GC", f"{gc:.2f}%")

    inner_tabs = st.tabs([
        "Perplexity", "Landscapes", "Scale Support",
        "Consensus LPC", "Kadane", "Ranking", "Motifs", "Downloads",
    ])
    scales = result.params.get("resolved_scales", [])

    with inner_tabs[0]:
        _show_figure(plot_p1_profile(result.p1), f"p1-{selected_seq}")
    with inner_tabs[1]:
        _show_figure(
            plot_multiscale_landscapes(result.landscapes, result.domains),
            f"landscapes-{selected_seq}",
        )
    with inner_tabs[2]:
        _show_figure(
            plot_scale_support_heatmap(
                result.lpc_profiles, result.domains, scales
            ),
            f"support-{selected_seq}",
        )
    with inner_tabs[3]:
        _show_figure(
            plot_consensus_lpc(result.consensus_lpc, result.domains),
            f"consensus-{selected_seq}",
        )
    with inner_tabs[4]:
        _show_figure(
            plot_kadane_domains(result.consensus_lpc, result.domains),
            f"kadane-{selected_seq}",
        )
    with inner_tabs[5]:
        _show_figure(
            plot_domain_ranking(result.domains), f"ranking-{selected_seq}"
        )

        st.subheader("Valley Table")
        display_cols = [
            "ID", "Start", "End", "Length",
            "ScaleSupport", "Persistence",
            "ValleyScore", "GC%", "MotifCount", "Motifs",
        ]
        search = st.text_input(
            "Search valleys", placeholder="Filter by ID or motif",
            key="results_valley_search",
        )
        filtered = selected_df
        if search:
            q = search.lower()
            filtered = selected_df[
                selected_df["ID"].astype(str).str.lower().str.contains(q)
                | selected_df["Motifs"].astype(str).str.lower().str.contains(q)
            ]

        row_choice = None
        try:
            event = st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="results_valley_table",
            )
            picked = event.selection.get("rows", []) if event else []
            if picked:
                row_choice = filtered.iloc[picked[0]]
        except TypeError:
            st.info(
                "Interactive row selection is unavailable "
                "(requires a Streamlit runtime with selection-enabled dataframe events)."
            )
            st.dataframe(
                filtered[display_cols], use_container_width=True,
                hide_index=True, key="results_valley_table_static",
            )

        if row_choice is not None:
            st.info(
                f"Highlighted valley: {row_choice['ID']} "
                f"({int(row_choice['Start'])}-{int(row_choice['End'])})"
            )

    with inner_tabs[6]:
        _show_figure(
            plot_motif_architecture(selected_df.to_dict("records")),
            f"motif-{selected_seq}",
        )

    with inner_tabs[7]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "CSV", export_table(selected_df, "csv"),
                "regplex_valleys.csv", "text/csv",
                use_container_width=True, key="dl_csv",
            )
            st.download_button(
                "Excel", export_table(selected_df, "xlsx"),
                "regplex_valleys.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="dl_xlsx",
            )
            st.download_button(
                "BED", export_bed(selected_df),
                "regplex_valleys.bed", "text/plain",
                use_container_width=True, key="dl_bed",
            )
        with c2:
            st.download_button(
                "GFF", export_gff(selected_df, gff3=False),
                "regplex_valleys.gff", "text/plain",
                use_container_width=True, key="dl_gff",
            )
            st.download_button(
                "GFF3", export_gff(selected_df, gff3=True),
                "regplex_valleys.gff3", "text/plain",
                use_container_width=True, key="dl_gff3",
            )
            st.download_button(
                "FASTA", export_fasta(selected_df),
                "regplex_valleys.fasta", "text/plain",
                use_container_width=True, key="dl_fasta",
            )
        with c3:
            st.download_button(
                "JSON", export_table(selected_df, "json"),
                "regplex_valleys.json", "application/json",
                use_container_width=True, key="dl_json",
            )


def _render_motifs(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='card'><h3>Motif Editor</h3>"
        "<p>Enter one motif per line (Regex or IUPAC).</p></div>",
        unsafe_allow_html=True,
    )
    motif_text = st.text_area(
        "Motif editor",
        value=st.session_state.get("motif_text", ""),
        height=240,
        key="motifs_editor",
    )
    rows = _validate_motifs(motif_text)
    if not rows:
        _render_empty("No motifs entered.")
        return

    for row in rows:
        klass = "motif-valid" if row["Status"] == "valid" else "motif-invalid"
        label = row["Regex"] if row["Status"] == "valid" else "invalid regex"
        safe_motif = html.escape(row["Motif"])
        safe_label = html.escape(label)
        st.markdown(f"<div class='{klass} mono'>{safe_motif} → {safe_label}</div>", unsafe_allow_html=True)

    if not df.empty:
        st.markdown("---")
        st.subheader("Detected motifs on genome")
        st.dataframe(df[["ID", "Sequence_ID", "MotifCount", "Motifs"]], use_container_width=True, hide_index=True)


def _render_about() -> None:
    st.markdown(
        """
        <div class="card">
          <h3>Scientific Workflow</h3>
          <p>DNA → P1 (once) → Multi-scale Landscapes → Per-scale LPC →
          Consensus LPC → Kadane + Expansion → Merging → Valleys → Motifs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _show_figure(plot_algorithm_workflow(), "workflow")

    st.markdown(
        """
        <div class="card">
          <h3>Algorithm — REGPLEX v9 Multi-scale Consensus</h3>
          <p>
            REGPLEX v9 computes dinucleotide perplexity (P1) exactly once, then
            builds rolling-mean/median landscapes at five independent observation
            scales. At each scale a three-window Local Perplexity Contrast (LPC)
            profile is computed with upstream = downstream = scale and spacer =
            scale ÷ 2. Each LPC profile is independently normalised by robust
            z-score and the profiles are combined into a single Consensus LPC via
            the positional nanmedian. Bounded Kadane locates valley cores in the
            Consensus LPC, which are expanded to natural boundaries (where
            ConsensusLPC &gt; 0) and merged if within the merge-gap. Every
            returned valley is a single continuous genomic interval supported by
            evidence from multiple observation scales.
          </p>
          <h3>Scientific Hypothesis</h3>
          <p>
            A true regulatory region exhibits lower sequence uncertainty than its
            surrounding DNA <em>across multiple observation scales</em>. Consensus
            evidence across scales — not any single-window comparison — is what
            defines a Perplexity Valley.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
          <h3>Parameter Reference</h3>
          <div class="help-grid">
            <div class="help-tile"><strong>P1 window</strong><br/>
              k-mer size for dinucleotide perplexity (default 10 bp).</div>
            <div class="help-tile"><strong>Base scale</strong><br/>
              Generates 5 scales: base÷4, base÷2, base, base×2, base×4.</div>
            <div class="help-tile"><strong>Custom scales</strong><br/>
              Override auto-scales with any comma-separated bp values.</div>
            <div class="help-tile"><strong>Landscape method</strong><br/>
              Rolling mean (fast) or rolling median (robust).</div>
            <div class="help-tile"><strong>Min/Max candidate</strong><br/>
              Bounds for the candidate window size at each scale (≥ 50 bp).</div>
            <div class="help-tile"><strong>Min/Max valley</strong><br/>
              Minimum and maximum reported valley length (bp).</div>
            <div class="help-tile"><strong>Merge gap</strong><br/>
              Valleys closer than this are merged into one domain.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
          <h3>Citation</h3>
          <div class="mono" style="background:var(--surface-2);padding:0.75rem;border-radius:10px;">
            REGPLEX v9: Multi-scale Perplexity Valley Framework
          </div>
          <h3 style="margin-top:1rem;">References</h3>
          <ul>
            <li><a href="https://genome.ucsc.edu" target="_blank">UCSC Genome Browser</a></li>
            <li><a href="https://www.ensembl.org" target="_blank">Ensembl Genome Browser</a></li>
            <li><a href="https://alphafold.ebi.ac.uk" target="_blank">AlphaFold DB</a></li>
            <li><a href="https://igv.org/web/release/latest/webapp" target="_blank">IGV-Web</a></li>
            <li><a href="https://www.nature.com/nmeth/" target="_blank">Nature Methods</a></li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.markdown(
        "REGPLEX v9 · [Citation](https://github.com/VRYella/PerCALL#citation) · "
        "[GitHub](https://github.com/VRYella/PerCALL) · MIT License · "
        "[Contact](https://github.com/VRYella)"
    )


if __name__ == "__main__":
    main()
