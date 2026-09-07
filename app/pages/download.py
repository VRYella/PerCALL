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
    source_sequence = next(seq for header, seq in records if header == sequence_id)
    df = results_dataframe(result)

    st.download_button("CSV", export_csv(df), file_name="percall_regions.csv", mime="text/csv")
    st.download_button("BED", export_bed(df), file_name="percall_regions.bed", mime="text/plain")
    st.download_button("GFF3", export_gff(df), file_name="percall_regions.gff3", mime="text/plain")
    st.download_button("FASTA", export_region_fasta(df, source_sequence), file_name="percall_regions.fasta", mime="text/plain")
