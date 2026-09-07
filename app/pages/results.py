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

    total_regions = len(result.candidate_regions)
    top_pds = max((region.max_pds for region in result.candidate_regions), default=0.0)
    mean_length = int(np.mean([region.length for region in result.candidate_regions])) if result.candidate_regions else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted regions", total_regions)
    with c2:
        st.metric("Top max PDS", f"{top_pds:.2f}")
    with c3:
        st.metric("Average region length", f"{mean_length} bp")

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("#### Why these regions were predicted")
    for region in result.candidate_regions:
        with st.expander(f"Region {sequence_id}:{region.start}-{region.end}", expanded=False):
            st.text(region_summary_text(region))
