from __future__ import annotations

import streamlit as st
import numpy as np

from src.output.csv import results_dataframe
from src.visualization.regions import region_summary_text


def render_results_page() -> None:
    st.subheader("Results")
    results = st.session_state.get("analysis_results", [])
    if not results:
        st.info("Run Analyze first.")
        return

    seq_ids = [result.sequence_id for result in results]
    sequence_id = st.selectbox("Sequence", seq_ids, key="results_sequence")
    result = next(r for r in results if r.sequence_id == sequence_id)
    df = results_dataframe(result)
    source_label = st.session_state.get("analysis_source_label", "Unknown source")
    motif_count = int(st.session_state.get("analysis_motif_count", 0))

    total_regions = len(result.candidate_regions)
    top_pds = max((region.max_pds for region in result.candidate_regions), default=0.0)
    mean_length = int(np.mean([region.length for region in result.candidate_regions])) if result.candidate_regions else 0
    top_motif_hits = max((region.motif_count for region in result.candidate_regions), default=0)

    st.caption(f"Input source: {source_label} · Active motif patterns: {motif_count}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted regions", total_regions)
    with c2:
        st.metric("Top max PDS", f"{top_pds:.2f}")
    with c3:
        st.metric("Average region length", f"{mean_length} bp")
    with c4:
        st.metric("Top motif hits", top_motif_hits)

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("#### Why these regions were predicted")
    for region in result.candidate_regions:
        with st.expander(f"Region {sequence_id}:{region.start}-{region.end}", expanded=False):
            st.text(region_summary_text(region))
