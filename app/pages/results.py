from __future__ import annotations

import streamlit as st

from src.output.csv import results_dataframe
from src.visualization.regions import region_summary_text


def render_results_page() -> None:
    st.subheader("Results")
    results = st.session_state.get("analysis_results", [])
    if not results:
        st.info("Run Analyze first.")
        return

    seq_ids = [result.sequence_id for result in results]
    sequence_id = st.selectbox("Sequence", seq_ids)
    result = next(r for r in results if r.sequence_id == sequence_id)
    df = results_dataframe(result)

    st.metric("Predicted regions", len(result.candidate_regions))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Why these regions were predicted")
    for region in result.candidate_regions:
        st.markdown(f"**Region:** {sequence_id}:{region.start}-{region.end}")
        st.text(region_summary_text(region))
