from __future__ import annotations

import base64
import html
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from motif_engine import annotate_domains, compile_motifs, iupac_to_regex
from regplex_core import (
    FLANK_SIZE,
    MAX_CANDIDATE,
    MAX_VALLEY_LENGTH,
    MERGE_GAP,
    MIN_CANDIDATE,
    MIN_VALLEY_LENGTH,
    PERPLEXITY_WINDOW,
    SG_POLY_ORDER,
    SG_WINDOW_LENGTH,
    SPACER_SIZE,
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
    plot_motif_architecture,
    plot_pds_landscape,
    plot_perplexity_landscape,
    plot_smoothed_perplexity,
    plot_three_window,
    plot_valley_ranking,
)

st.set_page_config(page_title="REGPLEX v13", layout="wide", initial_sidebar_state="collapsed")

_NAV_ITEMS = ["Home", "Analysis", "Results", "Motifs", "About"]
_README_URL = "https://github.com/VRYella/PerCALL#readme"

# ─── Default motif sets ──────────────────────────────────────────────────────
_DEFAULT_NON_B_DNA_MOTIFS = (
    "A{7}|T{7}",
    "G{7}|C{7}",
    "G{3,5}[ACGT]{1,7}G{3,5}[ACGT]{1,7}G{3,5}[ACGT]{1,7}G{3,5}",
    "C{3,5}[ACGT]{1,7}C{3,5}[ACGT]{1,7}C{3,5}[ACGT]{1,7}C{3,5}",
)

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
    "toImageButtonOptions": {"format": "svg", "filename": "regplex_v13_figure", "scale": 2},
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

# ─── Display columns (no support columns) ───────────────────────────────────
_DISPLAY_COLS = [
    "Rank", "ID", "Start", "End", "Length",
    "MeanPerplexity", "MinPerplexity",
    "UpstreamMean", "CandidateMean", "DownstreamMean",
    "UpstreamDifference", "DownstreamDifference",
    "PDSMean", "PDSMax",
    "Prominence", "Persistence", "AreaUnderValley", "Variance",
    "GC%", "MotifCount", "ValleyScore", "ValleyScoreNormalized",
]


# ─── UI helpers ──────────────────────────────────────────────────────────────

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
<svg width="40" height="40" viewBox="0 0 64 64" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-label="REGPLEX icon">
  <rect x="3" y="3" width="58" height="58" rx="14" fill="#FFFFFF" stroke="#1E3A8A"/>
  <path d="M16 18C24 18 24 32 32 32C40 32 40 18 48 18"
        stroke="#1E3A8A" stroke-width="3" stroke-linecap="round"/>
  <path d="M16 28C24 28 24 42 32 42C40 42 40 28 48 28"
        stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
  <path d="M17 47C22 43 27 41 32 41C37 41 42 43 47 47"
        stroke="#0F766E" stroke-width="3" stroke-linecap="round"/>
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
            <div class="brand">
              {_svg_logo()}
              <div>
                <h1>REGPLEX</h1>
                <span>v13 · Perplexity Valley Detector</span>
              </div>
            </div>
            <div class="top-links">
              <a href="https://github.com/VRYella/PerCALL" target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href="{_README_URL}" target="_blank" rel="noopener noreferrer">Documentation</a>
              <span class="top-version-badge">v13</span>
            </div>
          </div>
        </div>
        """
    )


def _render_nav() -> int:
    if "jump_nav" in st.session_state:
        st.session_state["main_nav"] = st.session_state.pop("jump_nav")
    tab = st.radio("Navigation", _NAV_ITEMS, horizontal=True,
                   label_visibility="collapsed", key="main_nav")
    return _NAV_ITEMS.index(tab)


def _show_figure(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key)


def _metric_card(label: str, value: str, icon: str = "") -> None:
    _render_html_block(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{icon} {label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>"
    )


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
    default_lines = [*_DEFAULT_NON_B_DNA_MOTIFS, *_DEFAULT_PROMOTER_IUPAC_MOTIFS]
    custom_lines  = [line.strip() for line in custom_text.splitlines() if line.strip()]
    return "\n".join([*default_lines, *custom_lines])


def _current_custom_motifs(primary_key: str, fallback_key: str) -> str:
    return st.session_state.get(primary_key, st.session_state.get(fallback_key, ""))


def _run_analysis(
    fasta_text: str,
    params: dict,
    motif_text: str,
) -> tuple[list[AnalysisResult], float]:
    started = time.perf_counter()
    records = parse_fasta(fasta_text)
    motifs  = compile_motifs(motif_text)
    results: list[AnalysisResult] = []
    for header, sequence in records:
        result = analyze_sequence(header, sequence, **params)
        annotate_domains(result.domains, motifs)
        results.append(result)
    return results, time.perf_counter() - started


# ─── Home ────────────────────────────────────────────────────────────────────

def _render_home() -> None:
    img_b64 = _load_hero_image_b64()
    image_html = (
        f'<img src="data:image/png;base64,{img_b64}" class="hero-image" alt="REGPLEX v13 workflow"/>'
        if img_b64
        else '<div class="hero-image-missing">Logo unavailable</div>'
    )
    _render_html_block(
        f"""
<div class="hero-section">
  <div class="hero-logo">{image_html}</div>
  <div class="hero-center">
    <div class="hero-brand">REGPLEX</div>
    <div class="hero-subtitle">Perplexity Valley Detector</div>
    <div class="hero-chips">
      <span class="hero-chip">Training-free</span>
      <span class="hero-chip">Species-independent</span>
      <span class="hero-chip">Dinucleotide Perplexity</span>
      <span class="hero-chip">Motif Annotation</span>
    </div>
  </div>
  <div class="hero-metrics">
    <div class="metric-card"><div class="metric-label">Method</div><div class="metric-value" style="font-size:18px">Info-Theoretic</div></div>
    <div class="metric-card"><div class="metric-label">Window</div><div class="metric-value" style="font-size:18px">17 nt</div></div>
    <div class="metric-card"><div class="metric-label">Version</div><div class="metric-value" style="font-size:18px">v13</div></div>
  </div>
</div>
"""
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("▶ Run Analysis", key="home_run", use_container_width=True, type="primary"):
            _jump_to_nav("Analysis")
    with b2:
        if st.button("📂 Load Example", key="home_example", use_container_width=True):
            st.session_state["input_fasta_text"] = _load_example_text()
            _jump_to_nav("Analysis")
    with b3:
        st.link_button("📖 Documentation", _README_URL, use_container_width=True)
    with b4:
        st.link_button("🐙 GitHub", "https://github.com/VRYella/PerCALL", use_container_width=True)


# ─── Analysis ────────────────────────────────────────────────────────────────

def _render_analysis() -> None:
    _render_html_block(
        "<div class='card'>"
        "<h3>🔬 Analysis</h3>"
        "<p style='margin:0;color:var(--muted)'>Upload or paste a FASTA sequence, tune parameters, then run.</p>"
        "</div>"
    )

    default_text = st.session_state.get("input_fasta_text", "")
    upload = st.file_uploader("Upload FASTA", type=["fasta", "fa", "fna", "txt"],
                               key="analysis_file")
    pasted = st.text_area("Or paste FASTA", value=default_text, height=180, key="analysis_paste")

    fasta_text    = ""
    uploaded_name = ""
    if upload is not None:
        fasta_text    = upload.read().decode("utf-8", errors="replace")
        uploaded_name = upload.name
    elif pasted.strip():
        fasta_text    = pasted if pasted.startswith(">") else f">query\n{pasted}"
        uploaded_name = "pasted_sequence.fasta"

    if fasta_text:
        parsed    = parse_fasta(fasta_text)
        total_len = sum(len(seq) for _, seq in parsed)
        m1, m2, m3 = st.columns(3)
        with m1:
            _metric_card("Input",      uploaded_name, "📄")
        with m2:
            _metric_card("Sequences",  str(len(parsed)), "🧬")
        with m3:
            _metric_card("Total length", f"{total_len:,} bp", "📏")

    st.markdown("#### ⚙️ Detection Parameters")
    p1, p2, p3 = st.columns(3)
    with p1:
        min_valley_length = st.number_input(
            "Min valley length (bp)", min_value=50, max_value=10000,
            value=MIN_VALLEY_LENGTH,
            help="Valleys shorter than this are discarded. Default 100 bp.",
        )
    with p2:
        max_valley_length = st.number_input(
            "Max valley length (bp)", min_value=100, max_value=10000,
            value=MAX_VALLEY_LENGTH,
            help="Kadane core upper bound. Default 1000 bp.",
        )
    with p3:
        merge_gap = st.number_input(
            "Merge gap (bp)", min_value=0, max_value=2000,
            value=MERGE_GAP,
            help="Merge adjacent valleys whose gap ≤ this value. Default 100 bp.",
        )

    with st.expander("Advanced parameters"):
        a1, a2, a3 = st.columns(3)
        with a1:
            sg_window_length = st.number_input(
                "Savitzky–Golay window", min_value=3, max_value=401,
                value=SG_WINDOW_LENGTH, step=2,
                help="Odd smoothing window applied once. Default 21.",
            )
            flank_size = st.number_input(
                "PDS flank size (bp)", min_value=20, max_value=1000,
                value=FLANK_SIZE,
                help="Upstream/downstream reference window for PDS. Default 100 bp.",
            )
        with a2:
            sg_poly_order = st.number_input(
                "Savitzky–Golay order", min_value=1, max_value=10,
                value=SG_POLY_ORDER,
                help="Polynomial order for SG smoothing. Default 3.",
            )
            spacer_size = st.number_input(
                "PDS spacer size (bp)", min_value=0, max_value=500,
                value=SPACER_SIZE,
                help="Gap between flank and candidate window. Default 50 bp.",
            )
        with a3:
            min_candidate = st.number_input(
                "PDS min candidate (bp)", min_value=10, max_value=5000,
                value=MIN_CANDIDATE,
                help="Lower bound for adaptive PDS candidate window. Default 50 bp.",
            )
            max_candidate = st.number_input(
                "PDS max candidate (bp)", min_value=50, max_value=10000,
                value=MAX_CANDIDATE,
                help="Upper bound for adaptive PDS candidate window. Default 1000 bp.",
            )

    st.markdown("#### 🧩 Motif Annotation")
    b1_col, b2_col = st.columns(2)
    with b1_col:
        st.markdown("**Non-B DNA motifs (fixed)**")
        st.code("\n".join(_DEFAULT_NON_B_DNA_MOTIFS), language="text")
    with b2_col:
        st.markdown("**Promoter motifs – IUPAC (fixed)**")
        st.code("\n".join(_DEFAULT_PROMOTER_IUPAC_MOTIFS), language="text")

    custom_motif_text = st.text_area(
        "Add custom motifs (Regex / IUPAC, one per line)",
        value=_current_custom_motifs("analysis_custom_motif_text", "motifs_custom_motif_text"),
        height=100,
        key="analysis_custom_motif_text",
    )
    motif_text = _combine_motif_text(custom_motif_text)
    motif_rows = _validate_motifs(motif_text)
    st.caption(f"Annotating with **{len(motif_rows)}** motif patterns (built-in + custom).")
    if motif_rows:
        lines = []
        for row in motif_rows:
            klass = "motif-valid" if row["Status"] == "valid" else "motif-invalid"
            lines.append(
                f"<div class='{klass} mono'>"
                f"{html.escape(row['Motif'])} → {html.escape(row['Regex'])}"
                f"</div>"
            )
        _render_html_block("".join(lines))

    run_col, reset_col, example_col = st.columns([2, 1, 1])
    with run_col:
        run_clicked = st.button("▶ Run REGPLEX v13", type="primary", use_container_width=True)
    with reset_col:
        if st.button("↺ Reset", use_container_width=True):
            for key in _RESET_SESSION_KEYS:
                st.session_state.pop(key, None)
            st.rerun()
    with example_col:
        if st.button("📂 Load Example", use_container_width=True):
            st.session_state["input_fasta_text"] = _load_example_text()
            st.rerun()

    if run_clicked:
        if not fasta_text:
            st.warning("No FASTA input provided.")
            return

        params = {
            "perplexity_window": PERPLEXITY_WINDOW,
            "sg_window_length":  int(sg_window_length),
            "sg_poly_order":     int(sg_poly_order),
            "flank_size":        int(flank_size),
            "spacer_size":       int(spacer_size),
            "min_candidate":     int(min_candidate),
            "max_candidate":     int(max_candidate),
            "min_valley_length": int(min_valley_length),
            "max_valley_length": int(max_valley_length),
            "merge_gap":         int(merge_gap),
        }

        with st.status("Running REGPLEX v13 …", expanded=False) as status:
            try:
                results, runtime_seconds = _run_analysis(fasta_text, params, motif_text)
            except re.error as exc:
                status.update(label=f"Invalid motif regex: {exc}", state="error")
                return
            st.session_state["results"]    = results
            st.session_state["domains_df"] = domains_dataframe(results)
            st.session_state["runtime"]    = runtime_seconds
            st.session_state["motif_text"] = motif_text
            n_seqs = len(results)
            seq_word = "sequence" if n_seqs == 1 else "sequences"
            status.update(
                label=f"✅ Complete — {n_seqs} {seq_word} processed in {runtime_seconds:.2f}s",
                state="complete",
            )
        _jump_to_nav("Results")


# ─── Results ─────────────────────────────────────────────────────────────────

def _render_results(results: list[AnalysisResult], df: pd.DataFrame) -> None:
    if df.empty:
        _render_html_block(
            "<div class='empty-state'>"
            "<div>No valleys detected yet. Run Analysis first.</div>"
            "</div>"
        )
        return

    selected_seq = st.selectbox(
        "Sequence", [r.sequence_id for r in results], key="results_sequence"
    )
    result      = next(r for r in results if r.sequence_id == selected_seq)
    selected_df = df[df["Sequence_ID"] == selected_seq].copy()

    runtime_seconds = float(st.session_state.get("runtime", 0.0))
    longest   = int(selected_df["Length"].max())    if not selected_df.empty else 0
    top_score = float(selected_df["ValleyScore"].max()) if not selected_df.empty else 0.0
    mean_pds  = float(selected_df["PDSMean"].mean())   if not selected_df.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _metric_card("Valleys detected", str(len(result.domains)), "🏔️")
    with m2:
        _metric_card("Longest valley",   f"{longest:,} bp", "📏")
    with m3:
        _metric_card("Top ValleyScore",  f"{top_score:.4f}", "🏆")
    with m4:
        _metric_card("Runtime",          f"{runtime_seconds:.2f}s", "⏱️")

    # ── Interactive valley table — ALWAYS VISIBLE ──────────────────────────
    _render_html_block(
        "<div class='section-header'>"
        "<span class='section-icon'>📋</span>"
        "<span class='section-title'>Valley Predictions</span>"
        "<span class='section-subtitle'>Ranked by ValleyScore (highest = best)</span>"
        "</div>"
    )

    top_n    = TOP_N_DISPLAY
    show_all = st.toggle(
        f"Show all {len(selected_df)} valleys",
        value=False,
        key="show_all_valleys",
    )
    display_df = selected_df if show_all else selected_df.nsmallest(top_n, "Rank")
    display_cols = [c for c in _DISPLAY_COLS if c in selected_df.columns]
    st.caption(
        f"Displaying **{'all' if show_all else min(top_n, len(selected_df))}** "
        f"of **{len(selected_df)}** valleys."
    )
    st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Visualisation tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Raw Perplexity",
        "🔵 Smoothed Perplexity",
        "🪟 Three-Window",
        "🌊 PDS Landscape",
        "🏆 Valley Ranking",
        "🔎 Motifs",
        "⬇️ Downloads",
    ])

    with tabs[0]:
        _show_figure(
            plot_perplexity_landscape(result.di, result.domains),
            f"raw-perplexity-{selected_seq}",
        )

    with tabs[1]:
        _show_figure(
            plot_smoothed_perplexity(result.di, result.smoothed_di, result.domains),
            f"smoothed-{selected_seq}",
        )

    with tabs[2]:
        if result.domains:
            valley_ids  = [d.get("ID", f"PV_{i+1:06d}") for i, d in enumerate(result.domains)]
            selected_id = st.selectbox("Select valley", valley_ids, key="three_window_select")
            sel_domain  = next(d for d in result.domains if d.get("ID") == selected_id)
            _show_figure(
                plot_three_window(
                    result.smoothed_di,
                    result.pds,
                    sel_domain,
                    flank_size=result.params.get("flank_size", 100),
                    spacer_size=result.params.get("spacer_size", 50),
                ),
                f"three-window-{selected_seq}-{selected_id}",
            )
        else:
            st.info("No valleys to display.")

    with tabs[3]:
        _show_figure(
            plot_pds_landscape(result.pds, result.domains),
            f"pds-{selected_seq}",
        )

    with tabs[4]:
        _show_figure(
            plot_valley_ranking(result.domains),
            f"rank-{selected_seq}",
        )

    with tabs[5]:
        _show_figure(
            plot_motif_architecture(selected_df.to_dict("records")),
            f"motifs-{selected_seq}",
        )
        if not selected_df.empty and "Motifs" in selected_df.columns:
            motif_df = selected_df[["ID", "MotifCount", "Motifs"]].copy()
            motif_df = motif_df[motif_df["MotifCount"] > 0]
            if not motif_df.empty:
                st.dataframe(motif_df, use_container_width=True, hide_index=True)

    with tabs[6]:
        _render_html_block("<div class='card'><h3>⬇️ Download Results</h3></div>")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("CSV",
                export_table(selected_df, "csv"), "regplex_v13_valleys.csv", "text/csv",
                use_container_width=True)
            st.download_button("Excel",
                export_table(selected_df, "xlsx"), "regplex_v13_valleys.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
            st.download_button("BED",
                export_bed(selected_df), "regplex_v13_valleys.bed", "text/plain",
                use_container_width=True)
        with c2:
            st.download_button("GFF",
                export_gff(selected_df, gff3=False), "regplex_v13_valleys.gff", "text/plain",
                use_container_width=True)
            st.download_button("GFF3",
                export_gff(selected_df, gff3=True), "regplex_v13_valleys.gff3", "text/plain",
                use_container_width=True)
            st.download_button("FASTA",
                export_fasta(selected_df), "regplex_v13_valleys.fasta", "text/plain",
                use_container_width=True)
        with c3:
            st.download_button("JSON",
                export_table(selected_df, "json"), "regplex_v13_valleys.json", "application/json",
                use_container_width=True)


# ─── Motifs ──────────────────────────────────────────────────────────────────

def _render_motifs(df: pd.DataFrame) -> None:
    _render_html_block(
        "<div class='card'>"
        "<h3>🧩 Motif Annotation</h3>"
        "<p style='margin:0;color:var(--muted)'>Validate motif syntax and inspect valley-level counts. "
        "Motifs do not affect valley detection.</p>"
        "</div>"
    )
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Non-B DNA motifs (fixed)**")
        st.code("\n".join(_DEFAULT_NON_B_DNA_MOTIFS), language="text")
    with m2:
        st.markdown("**Promoter motifs – IUPAC (fixed)**")
        st.code("\n".join(_DEFAULT_PROMOTER_IUPAC_MOTIFS), language="text")

    custom_text = st.text_area(
        "Add custom motifs",
        value=_current_custom_motifs("motifs_custom_motif_text", "analysis_custom_motif_text"),
        height=200,
        key="motifs_custom_motif_text",
    )
    motif_text = _combine_motif_text(custom_text)
    rows = _validate_motifs(motif_text)
    st.caption(f"Active motif patterns: **{len(rows)}** (built-in + custom).")
    if not rows:
        _render_html_block("<div class='empty-state'><div>No motifs entered.</div></div>")
        return
    for row in rows:
        klass = "motif-valid" if row["Status"] == "valid" else "motif-invalid"
        _render_html_block(
            f"<div class='{klass} mono'>"
            f"{html.escape(row['Motif'])} → {html.escape(row['Regex'])}"
            f"</div>"
        )
    if not df.empty and "MotifCount" in df.columns:
        st.markdown("---")
        show_cols = [c for c in ["ID", "Sequence_ID", "Length", "MotifCount", "Motifs", "ValleyScore"]
                     if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


# ─── About ───────────────────────────────────────────────────────────────────

def _render_about() -> None:
    _render_html_block(
        """
        <div class="card">
          <h3>🧬 Scientific Basis</h3>
          <blockquote class="hypothesis-quote">
            Extended low-perplexity genomic valleys are identified relative to local background
            using a training-free information-theoretic framework. No species-specific parameters required.
          </blockquote>
        </div>
        <div class="card">
          <h3>⚙️ Algorithm Pipeline</h3>
          <ol class="algo-list">
            <li><strong>Dinucleotide Perplexity</strong> — window = 17 nt; 16 dinucleotide transitions.</li>
            <li><strong>Savitzky–Golay Smoothing</strong> — window=21, order=3; applied once.</li>
            <li><strong>PDS (three-window contrast)</strong> — <code>PDS = (UpMean + DnMean) / 2 − CandMean</code>; flanks must exceed candidate.</li>
            <li><strong>Bounded Kadane</strong> — all positive-PDS valleys, 100–1000 bp.</li>
            <li><strong>Expansion &amp; Merging</strong> — grow while PDS &gt; 0 or &gt;20% peak; merge gap ≤ 100 bp.</li>
            <li><strong>ValleyScore</strong> = PDSMean × Persistence × log(Length) × Stability.</li>
            <li><strong>Optional Motif Annotation</strong> — IUPAC / regex; scanned within valleys only.</li>
          </ol>
        </div>
        <div class="card">
          <h3>🚫 Out of Scope</h3>
          <p style="margin:0;color:var(--muted)">Not a promoter predictor · Not a classifier · No training data ·
          No ML · No ensemble voting · No species-specific parameters.</p>
        </div>
        """
    )
    _show_figure(plot_algorithm_workflow(), "workflow-v13")


# ─── Main entry point ────────────────────────────────────────────────────────

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
        "**REGPLEX v13** · Training-Free Perplexity Valley Detector · MIT License"
    )


if __name__ == "__main__":
    main()
