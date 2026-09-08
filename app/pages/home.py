from __future__ import annotations

import streamlit as st


def render_home_page() -> None:
    st.subheader("REGPLEX regulatory region explorer")
    st.markdown('<div class="section-subtitle">Sequence ingestion, motif annotation, and explainable perplexity analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="card">
REGPLEX predicts <strong>candidate regulatory regions</strong> by combining local DNA perplexity depressions with optional motif-library annotation.
</div>

<div class="section-header">
  <span class="section-title">Workflow</span>
  <span class="section-subtitle">From sequence to ranked candidate regions</span>
</div>

<ol class="algo-list">
  <li>DNA sequence ingestion from pasted text, disk, upload, or NCBI accession</li>
  <li>Dinucleotide probability estimation</li>
  <li>Shannon entropy and perplexity profile (PPL = 2<sup>H</sup>)</li>
  <li>Local flank-based background estimation</li>
  <li>Perplexity depression score (PDS) computation</li>
  <li>Persistent candidate region detection, motif annotation, and ranking</li>
</ol>
""",
        unsafe_allow_html=True,
    )
