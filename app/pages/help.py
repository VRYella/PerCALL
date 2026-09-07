from __future__ import annotations

import streamlit as st


def render_help_page() -> None:
    st.subheader("Help")
    st.markdown(
        """
PerCALL predicts **candidate regulatory regions** from low local DNA perplexity relative to flanking background.

Interpretation:
- Positive PDS means local sequence is less perplexing than nearby sequence.
- Regions are retained only when depression is persistent and length-constrained.
- Predictions are hypothesis-generating and should be biologically validated.
"""
    )
