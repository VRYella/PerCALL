from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

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

    st.markdown('<div class="section-subtitle">Interactive profile, region overlays, and signal diagnostics</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sequence length", f"{result.sequence_length} bp")
    with c2:
        st.metric("Candidate regions", len(result.candidate_regions))
    with c3:
        peak_pds = max((float(region.max_pds) for region in result.candidate_regions), default=0.0)
        st.metric("Peak region PDS", f"{peak_pds:.2f}")

    fig = plot_perplexity_background_pds(
        positions=result.profile.positions,
        perplexity=result.profile.smoothed_perplexity,
        background=result.background,
        pds=result.pds,
    )
    fig = add_region_highlights(fig, result.candidate_regions)
    fig.update_layout(height=520, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))

    tabs = st.tabs(["Profile & Regions", "PDS Distribution", "Region Quality"])
    with tabs[0]:
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        pds_frame = pd.DataFrame({"PDS": result.pds})
        pds_frame = pds_frame[pds_frame["PDS"].notna()]
        pds_hist = px.histogram(pds_frame, x="PDS", nbins=50, color_discrete_sequence=["#1E3A8A"])
        pds_hist.update_layout(template="plotly_white", height=460, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(pds_hist, use_container_width=True)
    with tabs[2]:
        quality = pd.DataFrame(
            [
                {"Region": f"{region.start}-{region.end}", "Mean PDS": region.mean_pds, "Max PDS": region.max_pds}
                for region in result.candidate_regions
            ]
        )
        if quality.empty:
            st.info("No regions detected for this sequence.")
        else:
            scatter = px.scatter(
                quality,
                x="Mean PDS",
                y="Max PDS",
                hover_name="Region",
                color="Max PDS",
                color_continuous_scale="Tealgrn",
            )
            scatter.update_layout(template="plotly_white", height=460, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(scatter, use_container_width=True)
            st.dataframe(quality, hide_index=True, use_container_width=True)
