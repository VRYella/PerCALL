from __future__ import annotations

import streamlit as st

from src.output.bed import export_bed
from src.output.csv import export_csv, results_dataframe
from src.output.fasta import export_region_fasta
from src.output.gff import export_gff


def render_download_page() -> None:
    st.subheader("Download")
    results = st.session_state.get("analysis_results", [])
    records = st.session_state.get("analysis_records", [])
    if not results:
        st.info("Run Analyze first.")
        return

    seq_ids = [result.sequence_id for result in results]
    sequence_id = st.selectbox("Sequence", seq_ids, key="download_sequence")
    result = next(r for r in results if r.sequence_id == sequence_id)

    record_map = {header: seq for header, seq in records}
    source_sequence = record_map.get(sequence_id)
    if source_sequence is None:
        st.error("Source sequence is unavailable for this result. Re-run analysis before downloading FASTA.")
        return

    df = results_dataframe(result)

    st.markdown('<div class="section-subtitle">Export REGPLEX calls with motif annotations for downstream genomics workflows</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("Download CSV", export_csv(df), file_name="regplex_regions.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("Download BED", export_bed(df), file_name="regplex_regions.bed", mime="text/plain", use_container_width=True)
    with d3:
        st.download_button("Download GFF3", export_gff(df), file_name="regplex_regions.gff3", mime="text/plain", use_container_width=True)
    with d4:
        st.download_button(
            "Download FASTA",
            export_region_fasta(df, source_sequence),
            file_name="regplex_regions.fasta",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("#### Preview")
    st.dataframe(df, hide_index=True, use_container_width=True)
