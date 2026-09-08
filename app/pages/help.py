from __future__ import annotations

import streamlit as st


def render_help_page() -> None:
    st.subheader("Help")
    st.markdown('<div class="section-subtitle">REGPLEX interpretation and usage guidance</div>', unsafe_allow_html=True)
    st.markdown(
        """
REGPLEX predicts **candidate regulatory regions** from low local DNA perplexity relative to flanking background and can annotate them with motif-library hits.

Interpretation:
- Positive PDS means local sequence is less perplexing than nearby sequence.
- Regions are retained only when depression is persistent and length-constrained.
- Predictions are hypothesis-generating and should be biologically validated.
"""
    )
    with st.expander("Recommended usage workflow"):
        st.markdown(
            """
1. Choose pasted text, uploaded FASTA, an indexed local file, or an NCBI accession in **Analyze**.
2. Run analysis with default parameters first.
3. Enable the bundled motif library or add custom motif text when needed.
4. Explore overlays and diagnostics in **Visualization**.
5. Review ranked intervals in **Results**.
6. Export files in **Download** for downstream pipelines.
"""
        )
