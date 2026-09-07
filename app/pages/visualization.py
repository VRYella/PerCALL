from __future__ import annotations

import streamlit as st

from src.visualization.profiles import plot_perplexity_background_pds
from src.visualization.regions import add_region_highlights


def render_visualization_page() -> None:
    st.subheader("Visualization")
    results = st.session_state.get("analysis_results", [])
    if not results:
        st.info("Run Analyze first.")
        return

    seq_ids = [result.sequence_id for result in results]
    sequence_id = st.selectbox("Sequence", seq_ids, key="viz_sequence")
    result = next(r for r in results if r.sequence_id == sequence_id)

    fig = plot_perplexity_background_pds(
        positions=result.profile.positions,
        perplexity=result.profile.smoothed_perplexity,
        background=result.background,
        pds=result.pds,
    )
    fig = add_region_highlights(fig, result.candidate_regions)
    st.plotly_chart(fig, use_container_width=True)
