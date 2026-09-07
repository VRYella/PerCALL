from __future__ import annotations

import streamlit as st


def render_home_page() -> None:
    st.subheader("Explainable DNA-perplexity predictor")
    st.markdown('<div class="section-subtitle">High-level workflow and interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card">
PerCALL predicts <strong>candidate regulatory regions</strong> by finding persistent local depressions in DNA perplexity relative to nearby background context.
</div>

<div class="section-header">
  <span class="section-title">Workflow</span>
  <span class="section-subtitle">From sequence to ranked candidate regions</span>
</div>

<ol class="algo-list">
  <li>DNA sequence ingestion and cleaning</li>
  <li>Dinucleotide probability estimation</li>
  <li>Shannon entropy and perplexity profile (PPL = 2<sup>H</sup>)</li>
  <li>Local flank-based background estimation</li>
  <li>Perplexity depression score (PDS) computation</li>
  <li>Persistent candidate region detection and ranking</li>
</ol>
""",
        unsafe_allow_html=True,
    )
