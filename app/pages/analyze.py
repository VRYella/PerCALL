from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.motifs import compile_motifs, load_motifs_from_path
from src.models.dataclasses import PerplexityConfig
from src.prediction.regulatory_predictor import predict_regulatory_regions
from src.preprocessing.input_sources import InputSourceError, load_sequence_records


def render_analyze_page() -> None:
    st.subheader("Analyze")
    st.markdown(
        '<div class="section-subtitle">Load sequence text, files, or NCBI accessions and tune REGPLEX controls</div>',
        unsafe_allow_html=True,
    )

    default_motif_path = Path(__file__).resolve().parents[2] / "regulatory_motifs.txt"

    with st.form("analysis_form", clear_on_submit=False):
        input_source = st.radio("Sequence source", ["Paste sequence", "Upload file", "Disk file", "NCBI accession"], horizontal=True)
        fasta_text = ""
        uploaded_file = None
        disk_path = ""
        accession = ""
        if input_source == "Paste sequence":
            fasta_text = st.text_area("FASTA input", height=240, placeholder=">seq_1\nATGCGT...\n>seq_2\n...")
        elif input_source == "Upload file":
            uploaded_file = st.file_uploader("Upload FASTA/text", type=["fa", "fasta", "fna", "txt"])
        elif input_source == "Disk file":
            disk_path = st.text_input("Absolute file path", placeholder="/home/runner/work/REGPLEX/REGPLEX/examples/ecoli.fasta")
        else:
            accession = st.text_input("NCBI nucleotide accession", placeholder="NC_000913.3")

        st.markdown("#### Regulatory motif library")
        use_default_library = st.checkbox("Use bundled regulatory_motifs.txt library", value=True)
        custom_motifs = st.text_area(
            "Additional motif lines",
            height=120,
            placeholder="TATA_box\tTATA[AT]A[AT][AG]\nG_quadruplex\tG{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}",
        )
        uploaded_motif_file = st.file_uploader("Optional motif file", type=["txt"], key="motif_upload")

        st.markdown("#### Analysis geometry")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            perplexity_window = st.number_input("Perplexity window", min_value=5, max_value=200, value=17)
        with c2:
            step_size = st.number_input("Step size", min_value=1, max_value=20, value=1)
        with c3:
            min_region = st.number_input("Min region length", min_value=20, max_value=5000, value=100)
        with c4:
            max_region = st.number_input("Max region length", min_value=50, max_value=10000, value=1000)

        st.markdown("#### Detection controls")
        d1, d2, d3 = st.columns(3)
        with d1:
            min_pds = st.number_input("Minimum PDS", min_value=0.0, max_value=10.0, value=0.25, step=0.05)
        with d2:
            persistence = st.number_input("Persistence (bp)", min_value=10, max_value=5000, value=80)
        with d3:
            merge_distance = st.number_input("Merge distance", min_value=0, max_value=5000, value=100)

        with st.expander("Advanced smoothing and background parameters", expanded=False):
            a1, a2, a3 = st.columns(3)
            with a1:
                smoothing_window = st.number_input("Smoothing window", min_value=3, max_value=401, value=21, step=2)
            with a2:
                smoothing_order = st.number_input("Smoothing polynomial order", min_value=1, max_value=10, value=3)
            with a3:
                flank_size = st.number_input("Flank size", min_value=5, max_value=2000, value=100)

        run_analysis = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

    if run_analysis:
        try:
            records, source_label = load_sequence_records(
                pasted_text=fasta_text,
                file_path=disk_path,
                accession=accession,
                uploaded_bytes=uploaded_file.getvalue() if uploaded_file is not None else None,
            )
        except InputSourceError as exc:
            st.error(str(exc))
            return

        compiled_motifs = []
        if use_default_library:
            compiled_motifs.extend(load_motifs_from_path(default_motif_path))
        if uploaded_motif_file is not None:
            compiled_motifs.extend(compile_motifs(uploaded_motif_file.getvalue().decode("utf-8")))
        if custom_motifs.strip():
            compiled_motifs.extend(compile_motifs(custom_motifs))

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
        results = [
            predict_regulatory_regions(sequence_id=h, sequence=s, config=config, compiled_motifs=compiled_motifs or None)
            for h, s in records
        ]
        st.session_state["analysis_results"] = results
        st.session_state["analysis_records"] = records
        st.session_state["analysis_source_label"] = source_label
        st.session_state["analysis_motif_count"] = len(compiled_motifs)
        st.success(f"Processed {len(results)} sequence(s) from {source_label}.")
