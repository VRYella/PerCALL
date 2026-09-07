from __future__ import annotations

import streamlit as st


def render_home_page() -> None:
    st.subheader("Explainable DNA-perplexity predictor")
    st.markdown(
        """
PerCALL predicts **candidate regulatory regions** by finding local depressions in DNA perplexity.

Workflow:
1. DNA sequence
2. Dinucleotide probabilities
3. Shannon entropy and perplexity (PPL = 2^H)
4. Local background perplexity
5. Perplexity depression (PDS)
6. Persistent candidate region detection
"""
    )
