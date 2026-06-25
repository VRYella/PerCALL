from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from motif_engine import annotate_domains, compile_motifs
from regplex_core import (
    ADAPTIVE_MAX_WINDOW,
    ADAPTIVE_MIN_WINDOW,
    DOWNSTREAM_WINDOW,
    LANDSCAPE_METHOD,
    LANDSCAPE_WINDOW,
    MAX_DOMAIN,
    MIN_DOMAIN,
    PERPLEXITY_WINDOW,
    SPACER,
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
    plot_algorithm_workflow,
    plot_domain_ranking,
    plot_kadane_domains,
    plot_motif_architecture,
    plot_p1_profile,
    plot_p2_landscape,
    plot_pvs_profile,
    plot_three_window_illustration,
)

st.set_page_config(page_title="REGPLEX", layout="wide")

_THEME = """
<style>
    .stApp { background: #f8fafc; color: #0f172a; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(37,99,235,0.12);
        border-radius: 18px;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 0.25rem 0; color: #0f172a; }
    .hero p { margin: 0; color: #475569; }
    .card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        background: #ffffff;
        margin-bottom: 0.9rem;
    }
</style>
"""

PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "filename": "regplex_figure", "scale": 2},
}


def run_analysis(fasta_text: str, params: dict, motif_text: str) -> list[AnalysisResult]:
    records = parse_fasta(fasta_text)
    motifs = compile_motifs(motif_text)
    results: list[AnalysisResult] = []
    for header, sequence in records:
        result = analyze_sequence(header, sequence, **params)
        annotate_domains(result.domains, motifs)
        results.append(result)
    return results


def _show_figure(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key)


def main() -> None:
    st.markdown(_THEME, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>REGPLEX</h1>
            <p>
                De novo discovery of <strong>Perplexity Valleys</strong> via Local Perplexity Contrast.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Home", "Analysis", "Results", "Motifs", "Downloads"])

    with tabs[0]:
        st.markdown(
            """
            <div class="card">
                <strong>Scientific positioning</strong><br>
                REGPLEX is a training-free, species-independent framework that identifies sustained local
                reductions in sequence uncertainty relative to immediate upstream/downstream genomic context.
            </div>
            """,
            unsafe_allow_html=True,
        )
        _show_figure(plot_algorithm_workflow(), "workflow")

    with tabs[1]:
        upload = st.file_uploader("Upload FASTA", type=["fasta", "fa", "fna", "txt"])
        pasted = st.text_area("Or paste FASTA", height=180)
        left, middle, right = st.columns(3)
        with left:
            perplexity_window = st.number_input("P1 window", 4, 50, PERPLEXITY_WINDOW)
            landscape_window = st.number_input("Landscape window", 20, 5000, LANDSCAPE_WINDOW)
            landscape_method = st.selectbox("Landscape method", ["mean", "median"], index=0 if LANDSCAPE_METHOD == "mean" else 1)
            upstream_window = st.number_input("Upstream window", 20, 2000, UPSTREAM_WINDOW)
            downstream_window = st.number_input("Downstream window", 20, 2000, DOWNSTREAM_WINDOW)
        with middle:
            spacer = st.number_input("Spacer", 0, 1000, SPACER)
            candidate_mode = st.selectbox("Candidate mode", ["adaptive", "fixed"], format_func=lambda value: value.title())
            if candidate_mode == "fixed":
                candidate_window = st.number_input("Candidate window", 20, 1500, ADAPTIVE_MIN_WINDOW)
                adaptive_min_window = ADAPTIVE_MIN_WINDOW
                adaptive_max_window = ADAPTIVE_MAX_WINDOW
            else:
                adaptive_min_window = st.number_input("Adaptive minimum window", 20, 2000, ADAPTIVE_MIN_WINDOW)
                adaptive_max_window = st.number_input(
                    "Adaptive maximum window",
                    int(adaptive_min_window),
                    3000,
                    max(int(adaptive_min_window), ADAPTIVE_MAX_WINDOW),
                )
                candidate_window = ADAPTIVE_MIN_WINDOW
        with right:
            min_domain = st.number_input("Minimum valley length", 20, 10000, MIN_DOMAIN)
            max_domain = st.number_input("Maximum valley length", int(min_domain), 20000, max(int(min_domain), MAX_DOMAIN))
            motif_text = st.text_area("Optional motifs (one per line; IUPAC or regex)", height=190)

        if st.button("Run Analysis", type="primary"):
            fasta_text = ""
            pasted_text = pasted.strip()
            if upload is not None:
                fasta_text = upload.read().decode("utf-8", errors="replace")
            elif pasted_text:
                fasta_text = pasted_text if pasted_text.startswith(">") else f">query\n{pasted_text}"

            if not fasta_text:
                st.warning("Please upload or paste FASTA.")
            else:
                params = {
                    "perplexity_window": int(perplexity_window),
                    "landscape_window": int(landscape_window),
                    "landscape_method": str(landscape_method),
                    "candidate_mode": candidate_mode,
                    "candidate_window": int(candidate_window),
                    "upstream_window": int(upstream_window),
                    "downstream_window": int(downstream_window),
                    "spacer": int(spacer),
                    "adaptive_min_window": int(adaptive_min_window),
                    "adaptive_max_window": int(adaptive_max_window),
                    "min_domain": int(min_domain),
                    "max_domain": int(max_domain),
                }
                try:
                    st.session_state["results"] = run_analysis(fasta_text, params, motif_text)
                    st.session_state["motif_text"] = motif_text
                    st.success("Perplexity valley analysis complete.")
                except re.error as exc:
                    st.error(f"Invalid motif pattern: {str(exc)}. Please check the regex syntax or use IUPAC symbols.")

    results: list[AnalysisResult] = st.session_state.get("results", [])
    df = domains_dataframe(results) if results else pd.DataFrame()

    with tabs[2]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            sequence_id = st.selectbox("Sequence", [result.sequence_id for result in results])
            result = next(item for item in results if item.sequence_id == sequence_id)
            selected_df = df[df["Sequence_ID"] == sequence_id]
            m1, m2, m3 = st.columns(3)
            m1.metric("Detected valleys", len(result.domains))
            m2.metric("Mean ValleyScore", f"{selected_df['ValleyScore'].mean():.4f}" if not selected_df.empty else "N/A")
            m3.metric("Top ValleyScore", f"{selected_df['ValleyScore'].max():.4f}" if not selected_df.empty else "N/A")

            _show_figure(plot_p1_profile(result.p1), f"p1-{sequence_id}")
            _show_figure(plot_p2_landscape(result.landscape), f"landscape-{sequence_id}")
            _show_figure(
                plot_three_window_illustration(
                    result.landscape,
                    result.lpc,
                    result.params,
                    result.domains,
                    result.candidate_window,
                ),
                f"illustration-{sequence_id}",
            )
            _show_figure(plot_pvs_profile(result.lpc, result.domains), f"lpc-{sequence_id}")
            _show_figure(plot_kadane_domains(result.lpc, result.domains), f"kadane-{sequence_id}")
            _show_figure(plot_domain_ranking(result.domains), f"ranking-{sequence_id}")
            st.dataframe(selected_df, use_container_width=True)

    with tabs[3]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            st.text_area("Submitted motifs", st.session_state.get("motif_text", "") or "None", height=140, disabled=True)
            _show_figure(plot_motif_architecture(df.to_dict("records")), "motifs")

    with tabs[4]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            st.download_button("CSV", export_table(df, "csv"), "regplex_valleys.csv", "text/csv")
            st.download_button("Excel", export_table(df, "xlsx"), "regplex_valleys.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("BED", export_bed(df), "regplex_valleys.bed", "text/plain")
            st.download_button("GFF", export_gff(df, gff3=False), "regplex_valleys.gff", "text/plain")
            st.download_button("GFF3", export_gff(df, gff3=True), "regplex_valleys.gff3", "text/plain")
            st.download_button("FASTA", export_fasta(df), "regplex_valleys.fasta", "text/plain")
            st.download_button("JSON", export_table(df, "json"), "regplex_valleys.json", "application/json")


if __name__ == "__main__":
    main()
