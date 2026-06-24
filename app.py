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
st.title("REGPLEX")
st.caption("Regulatory Architecture Discovery Through Perplexity Depression")


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
    tabs = st.tabs(["Home", "Analysis", "Results", "Motifs", "Downloads"])

    with tabs[0]:
        st.markdown("""
        ### Scientific Motivation
        Functional DNA regions can exhibit sustained local collapse in sequence uncertainty.

        **Pipeline**: DNA → 10-mer Perplexity → PDI → Bounded Minimum-Mean Kadane → Domains → Optional Motif Annotation
        """)
        st.plotly_chart(plot_algorithm_illustration(), use_container_width=True)

    with tabs[1]:
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
            st.plotly_chart(plot_p1_profile(res.p1), use_container_width=True)
            st.plotly_chart(plot_pdi_profile(res.pdi, res.domains), use_container_width=True)
            st.plotly_chart(plot_domain_map(res.length, res.domains), use_container_width=True)
            st.plotly_chart(plot_domain_ranking(res.domains), use_container_width=True)
            st.plotly_chart(plot_domain_statistics(res.domains), use_container_width=True)
            st.dataframe(df[df["Sequence_ID"] == selected], use_container_width=True)

    with tabs[3]:
        if df.empty:
            st.info("Run analysis first.")
        else:
            st.subheader("Built-in Non-B DNA Motifs (always applied)")
            st.table(
                [{"Name": name, "Pattern": pattern} for name, pattern in NON_B_MOTIFS.items()]
            )
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
