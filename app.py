from __future__ import annotations

import pandas as pd
import streamlit as st

from motif_engine import NON_B_MOTIFS, annotate_domains, compile_motifs
from regplex_core import (
    DOWNSTREAM_WINDOW,
    DOMAIN_WINDOW,
    MAX_DOMAIN,
    MIN_DOMAIN,
    PERPLEXITY_WINDOW,
    SPACER,
    TOP_DOMAINS,
    UPSTREAM_WINDOW,
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
    plot_algorithm_illustration,
    plot_domain_map,
    plot_domain_ranking,
    plot_domain_statistics,
    plot_motif_architecture,
    plot_p1_profile,
    plot_pdi_profile,
)


st.set_page_config(page_title="REGPLEX", layout="wide")

SCIENTIFIC_THEME_CSS = """
<style>
/* ── Design tokens ─────────────────────────────────────────────── */
:root {
    --bg:        #071126;
    --bg2:       #0a1830;
    --bg3:       #0d1f3a;
    --text:      #e7edf7;
    --muted:     #7a9bbf;
    --line:      rgba(120,157,201,0.2);
    --cyan:      #35d0ff;
    --purple:    #8a7dff;
    --gold:      #ffc857;
    --green:     #36f7b0;
    --pink:      #ff6b9d;
    --glow-c:    rgba(53,208,255,0.35);
    --glow-p:    rgba(138,125,255,0.25);
}

/* ── Base ──────────────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at 10% 5%,  rgba(53,208,255,0.13) 0%, transparent 40%),
        radial-gradient(ellipse at 90% 8%,  rgba(138,125,255,0.10) 0%, transparent 38%),
        radial-gradient(ellipse at 50% 95%, rgba(54,247,176,0.07) 0%, transparent 40%),
        linear-gradient(175deg, #071126 0%, #050c1d 100%);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.stApp :is(h1,h2,h3,h4,p,li,label,span),
.stMarkdown, .stCaption { color: var(--text) !important; }

/* ── Hero ──────────────────────────────────────────────────────── */
.sci-hero {
    position: relative;
    padding: 2rem 2.2rem 1.6rem;
    border: 1px solid rgba(53,208,255,0.3);
    border-radius: 18px;
    background:
        linear-gradient(130deg, rgba(13,31,58,0.95), rgba(7,17,38,0.9));
    box-shadow:
        0 0 40px rgba(53,208,255,0.10),
        inset 0 0 60px rgba(53,208,255,0.03);
    margin-bottom: 1.2rem;
    overflow: hidden;
}
.sci-hero::before {
    content: '';
    position: absolute;
    top: -40%; left: -10%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(53,208,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.sci-hero-title {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(110deg, var(--cyan) 0%, var(--purple) 55%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}
.sci-hero-sub {
    margin: 0.5rem 0 0;
    font-size: 0.95rem;
    color: var(--muted) !important;
    letter-spacing: 0.01em;
}
.sci-badge {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 0.4rem;
    margin-top: 0.75rem;
}
.badge-cyan  { background: rgba(53,208,255,0.15);  color: var(--cyan);   border: 1px solid rgba(53,208,255,0.35); }
.badge-purple{ background: rgba(138,125,255,0.15); color: var(--purple); border: 1px solid rgba(138,125,255,0.35); }
.badge-gold  { background: rgba(255,200,87,0.15);  color: var(--gold);   border: 1px solid rgba(255,200,87,0.35); }
.badge-green { background: rgba(54,247,176,0.15);  color: var(--green);  border: 1px solid rgba(54,247,176,0.35); }

/* ── Cards ─────────────────────────────────────────────────────── */
.sci-card {
    padding: 1.1rem 1.3rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background:
        linear-gradient(135deg, rgba(13,31,58,0.75), rgba(7,17,38,0.7));
    backdrop-filter: blur(6px);
    margin-bottom: 0.85rem;
    transition: border-color .25s, box-shadow .25s;
}
.sci-card:hover {
    border-color: rgba(53,208,255,0.4);
    box-shadow: 0 0 20px rgba(53,208,255,0.10);
}
.sci-card strong { color: var(--cyan) !important; }

.sci-card-accent {
    border-left: 3px solid var(--cyan);
}
.sci-card-purple {
    border-left: 3px solid var(--purple);
}

/* ── Pipeline flow label ───────────────────────────────────────── */
.pipeline-flow {
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 2;
    letter-spacing: 0.01em;
}
.pipeline-flow span { color: var(--cyan); font-weight: 600; }
.pipeline-arrow { color: var(--purple); margin: 0 0.3rem; }

/* ── Tabs ───────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    gap: 0.35rem !important;
    background: transparent !important;
    border-bottom: 1px solid var(--line) !important;
    padding-bottom: 0 !important;
}
[data-baseweb="tab"] {
    border: 1px solid var(--line) !important;
    border-bottom: none !important;
    border-radius: 10px 10px 0 0 !important;
    background: rgba(8,21,44,0.85) !important;
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: color .2s, background .2s, border-color .2s !important;
    padding: 0.45rem 1rem !important;
}
[data-baseweb="tab"]:hover {
    background: rgba(53,208,255,0.08) !important;
    border-color: rgba(53,208,255,0.35) !important;
    color: var(--cyan) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(180deg, rgba(53,208,255,0.14), rgba(8,21,44,0.9)) !important;
    border-color: rgba(53,208,255,0.45) !important;
    color: var(--cyan) !important;
}
[data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
    height: 2px !important;
}
[data-baseweb="tab-panel"] {
    padding-top: 1.2rem !important;
}

/* ── Metrics ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.7rem 1rem !important;
    background: linear-gradient(135deg, rgba(10,24,48,0.75), rgba(7,17,38,0.65));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    transition: border-color .25s, box-shadow .25s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(53,208,255,0.4);
    box-shadow: 0 0 16px rgba(53,208,255,0.08);
}
[data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Buttons ────────────────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(53,208,255,0.9), rgba(138,125,255,0.85)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #07112a !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.8rem !important;
    box-shadow: 0 4px 18px rgba(53,208,255,0.30) !important;
    transition: box-shadow .25s, transform .15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(53,208,255,0.45) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(10,24,48,0.8) !important;
    border: 1px solid var(--line) !important;
    border-radius: 9px !important;
    color: var(--cyan) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    transition: border-color .2s, background .2s, box-shadow .2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: rgba(53,208,255,0.5) !important;
    background: rgba(53,208,255,0.08) !important;
    box-shadow: 0 0 12px rgba(53,208,255,0.12) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="textarea"],
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(9,23,46,0.75) !important;
    border: 1px solid var(--line) !important;
    border-radius: 9px !important;
    color: var(--text) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-baseweb="input"]:focus-within,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(53,208,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(53,208,255,0.10) !important;
}

/* ── File uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(53,208,255,0.3) !important;
    border-radius: 12px !important;
    background: rgba(9,23,46,0.5) !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(53,208,255,0.55) !important;
}

/* ── Selectbox ──────────────────────────────────────────────────── */
[data-baseweb="select"] > div {
    background: rgba(9,23,46,0.75) !important;
    border: 1px solid var(--line) !important;
    border-radius: 9px !important;
    color: var(--text) !important;
}

/* ── Dataframe ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Alerts / info ──────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--line) !important;
    background: rgba(9,23,46,0.6) !important;
}

/* ── Section divider ────────────────────────────────────────────── */
.sci-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.25;
    margin: 1.2rem 0;
}

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb {
    background: rgba(53,208,255,0.25);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(53,208,255,0.45); }
</style>
"""


def apply_custom_css_theme() -> None:
    """Apply a custom scientific theme to the Streamlit interface."""
    st.markdown(SCIENTIFIC_THEME_CSS, unsafe_allow_html=True)


def run_analysis(fasta_text: str, params: dict, motif_text: str) -> list[AnalysisResult]:
    records = parse_fasta(fasta_text)
    results = []
    motifs = compile_motifs(motif_text)
    for header, seq in records:
        res = analyze_sequence(header, seq, **params)
        annotate_domains(res.domains, motifs)
        results.append(res)
    return results


def main() -> None:
    apply_custom_css_theme()
    st.markdown(
        """
        <div class="sci-hero">
            <h1 class="sci-hero-title">REGPLEX</h1>
            <p class="sci-hero-sub">
                Regulatory Architecture Discovery Through Perplexity Depression
            </p>
            <div>
                <span class="sci-badge badge-cyan">Statistical</span>
                <span class="sci-badge badge-purple">Annotation-free</span>
                <span class="sci-badge badge-gold">Non-B DNA</span>
                <span class="sci-badge badge-green">Open Source</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Home", "Analysis", "Results", "Motifs", "Downloads"])

    with tabs[0]:
        st.markdown(
            "### Scientific Motivation\n"
            "REGPLEX quantifies local uncertainty-collapse signatures in DNA "
            "using a reproducible, annotation-free statistical framework."
        )
        st.markdown(
            """
            <div class="sci-card sci-card-accent">
                <strong>Pipeline</strong>
                <div class="pipeline-flow" style="margin-top:0.5rem;">
                    <span>DNA</span>
                    <span class="pipeline-arrow">→</span>
                    <span>10-mer Perplexity (P1)</span>
                    <span class="pipeline-arrow">→</span>
                    <span>Perplexity Depression Index (PDI)</span>
                    <span class="pipeline-arrow">→</span>
                    <span>Bounded Min-Mean Kadane</span>
                    <span class="pipeline-arrow">→</span>
                    <span>Ranked Regulatory Domains</span>
                    <span class="pipeline-arrow">→</span>
                    <span>Motif Architecture</span>
                </div>
            </div>
            <div class="sci-card sci-card-purple">
                <strong>Interpretation Principle</strong><br>
                <span style="color:var(--muted); font-size:0.93rem;">
                Higher domain RCS and sustained PDI elevation indicate stronger evidence
                of putative regulatory architecture.
                </span>
            </div>
            <div class="sci-divider"></div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(plot_algorithm_illustration(), use_container_width=True)

    with tabs[1]:
        st.markdown(
            """
            <div class="sci-card sci-card-accent">
                Configure statistically controlled windows, then run a full
                uncertainty-collapse scan across all FASTA records.
            </div>
            """,
            unsafe_allow_html=True,
        )
        upload = st.file_uploader("Upload FASTA", type=["fasta", "fa", "fna", "txt"])
        pasted = st.text_area("Or paste FASTA", height=180)
        c1, c2, c3 = st.columns(3)
        with c1:
            perplexity_window = st.number_input("Perplexity Window", 4, 50, PERPLEXITY_WINDOW)
            domain_window = st.number_input("Domain Window", 20, 300, DOMAIN_WINDOW)
            upstream_window = st.number_input("Upstream Context", 20, 500, UPSTREAM_WINDOW)
        with c2:
            downstream_window = st.number_input("Downstream Context", 20, 500, DOWNSTREAM_WINDOW)
            spacer = st.number_input("Spacer", 0, 500, SPACER)
            min_domain = st.number_input("Minimum Domain Length", 20, 2000, MIN_DOMAIN)
        with c3:
            max_default = min(max(int(min_domain), MAX_DOMAIN), 5000)
            max_domain = st.number_input(
                "Maximum Domain Length",
                int(min_domain),
                5000,
                max_default,
            )
            top_domains = st.number_input("Top Domains", 1, 5000, TOP_DOMAINS)
        motif_text = st.text_area("Additional motifs (one per line; IUPAC or regex)", height=120)

        if st.button("Run Analysis", type="primary"):
            pasted_text = pasted.strip()
            fasta_text = ""
            if upload is not None:
                fasta_text = upload.read().decode("utf-8", errors="replace")
            elif pasted_text:
                fasta_text = pasted_text if pasted_text.startswith(">") else f">query\n{pasted_text}"
            if not fasta_text:
                st.warning("Please upload or paste FASTA.")
            else:
                params = {
                    "perplexity_window": int(perplexity_window),
                    "domain_window": int(domain_window),
                    "upstream_window": int(upstream_window),
                    "downstream_window": int(downstream_window),
                    "spacer": int(spacer),
                    "min_domain": int(min_domain),
                    "max_domain": int(max_domain),
                    "top_domains": int(top_domains),
                }
                st.session_state["results"] = run_analysis(fasta_text, params, motif_text)
                st.session_state["motif_text"] = motif_text
                st.success("Analysis complete.")

    results: list[AnalysisResult] = st.session_state.get("results", [])
    df = domains_dataframe(results) if results else pd.DataFrame()

    with tabs[2]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            selected = st.selectbox("Sequence", [r.sequence_id for r in results])
            res = next(r for r in results if r.sequence_id == selected)
            selected_df = df[df["Sequence_ID"] == selected]
            if selected_df.empty:
                mean_rcs_display = "N/A"
                mean_gc_display = "N/A"
            else:
                mean_rcs = selected_df["RCS"].mean()
                mean_gc = selected_df["GC_Content"].mean()
                mean_rcs_display = f"{mean_rcs:.4f}" if pd.notna(mean_rcs) else "N/A"
                mean_gc_display = f"{mean_gc * 100:.2f}%" if pd.notna(mean_gc) else "N/A"
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Detected Domains", len(res.domains))
            with m2:
                st.metric("Mean RCS", mean_rcs_display)
            with m3:
                st.metric("Mean GC Content", mean_gc_display)
            st.plotly_chart(plot_p1_profile(res.p1), use_container_width=True)
            st.plotly_chart(plot_pdi_profile(res.pdi, res.domains), use_container_width=True)
            st.plotly_chart(plot_domain_map(res.length, res.domains), use_container_width=True)
            st.plotly_chart(plot_domain_ranking(res.domains), use_container_width=True)
            st.plotly_chart(plot_domain_statistics(res.domains), use_container_width=True)
            st.dataframe(selected_df, use_container_width=True)

    with tabs[3]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            st.subheader("Built-in Non-B DNA Motifs (always applied)")
            st.table(pd.DataFrame(list(NON_B_MOTIFS.items()), columns=["Name", "Pattern"]))
            user_motifs = st.session_state.get("motif_text", "").strip()
            st.subheader("User-provided Motifs")
            st.code(user_motifs or "None")
            st.plotly_chart(plot_motif_architecture(df.to_dict("records")), use_container_width=True)

    with tabs[4]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            st.download_button("CSV", export_table(df, "csv"), "regplex_domains.csv", "text/csv")
            st.download_button("TSV", export_table(df, "tsv"), "regplex_domains.tsv", "text/tab-separated-values")
            st.download_button(
                "Excel",
                export_table(df, "xlsx"),
                "regplex_domains.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button("BED", export_bed(df), "regplex_domains.bed", "text/plain")
            st.download_button("GFF", export_gff(df, gff3=False), "regplex_domains.gff", "text/plain")
            st.download_button("GFF3", export_gff(df, gff3=True), "regplex_domains.gff3", "text/plain")
            st.download_button("FASTA", export_fasta(df), "regplex_domains.fasta", "text/plain")
            st.download_button("JSON", export_table(df, "json"), "regplex_domains.json", "application/json")


if __name__ == "__main__":
    main()
