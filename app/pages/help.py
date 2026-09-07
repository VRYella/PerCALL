from __future__ import annotations

import streamlit as st


def render_help_page() -> None:
    st.subheader("Help")
    st.markdown('<div class="section-subtitle">Model interpretation and usage guidance</div>', unsafe_allow_html=True)
    st.markdown(
        """
PerCALL predicts **candidate regulatory regions** from low local DNA perplexity relative to flanking background.

Interpretation:
- Positive PDS means local sequence is less perplexing than nearby sequence.
- Regions are retained only when depression is persistent and length-constrained.
- Predictions are hypothesis-generating and should be biologically validated.
"""
    )
    with st.expander("Recommended usage workflow"):
        st.markdown(
            """
1. Paste one or more FASTA entries in **Analyze**.
2. Run analysis with default parameters first.
3. Explore overlays and diagnostics in **Visualization**.
4. Review ranked intervals in **Results**.
5. Export files in **Download** for downstream pipelines.
"""
        )
