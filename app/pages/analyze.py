from __future__ import annotations

import streamlit as st

from src.models.dataclasses import PerplexityConfig
from src.prediction.regulatory_predictor import predict_regulatory_regions
from src.preprocessing.fasta import parse_fasta
from src.preprocessing.validation import SequenceValidationError


def render_analyze_page() -> None:
    st.subheader("Analyze")
    fasta_text = st.text_area("FASTA input", height=220)

    st.markdown("#### Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        perplexity_window = st.number_input("Perplexity window", min_value=5, max_value=200, value=17)
    with c2:
        step_size = st.number_input("Step size", min_value=1, max_value=20, value=1)
    with c3:
        min_region = st.number_input("Min region length", min_value=20, max_value=5000, value=100)
    with c4:
        max_region = st.number_input("Max region length", min_value=50, max_value=10000, value=1000)

    st.markdown("#### Detection")
    d1, d2, d3 = st.columns(3)
    with d1:
        min_pds = st.number_input("Minimum PDS", min_value=0.0, max_value=10.0, value=0.25, step=0.05)
    with d2:
        persistence = st.number_input("Persistence (bp)", min_value=10, max_value=5000, value=80)
    with d3:
        merge_distance = st.number_input("Merge distance", min_value=0, max_value=5000, value=100)

    with st.expander("Advanced", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            smoothing_window = st.number_input("Smoothing window", min_value=3, max_value=401, value=21, step=2)
        with a2:
            smoothing_order = st.number_input("Smoothing polynomial order", min_value=1, max_value=10, value=3)
        with a3:
            flank_size = st.number_input("Flank size", min_value=5, max_value=2000, value=100)

    if st.button("Run analysis", type="primary"):
        try:
            records = parse_fasta(fasta_text)
        except SequenceValidationError as exc:
            st.error(str(exc))
            return

        if not records:
            st.error("No valid sequence records found. Provide FASTA or a plain DNA sequence.")
            return

        config = PerplexityConfig(
            perplexity_window=int(perplexity_window),
            step_size=int(step_size),
            smoothing_window=int(smoothing_window),
            smoothing_poly_order=int(smoothing_order),
            flank_size=int(flank_size),
            min_region_length=int(min_region),
            max_region_length=int(max_region),
            min_perplexity_depression=float(min_pds),
            min_persistence_bp=int(persistence),
            merge_distance=int(merge_distance),
        )
        results = [predict_regulatory_regions(sequence_id=h, sequence=s, config=config) for h, s in records]
        st.session_state["analysis_results"] = results
        st.session_state["analysis_records"] = records
        st.success(f"Processed {len(results)} sequence(s).")
