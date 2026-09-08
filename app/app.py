from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.pages.analyze import render_analyze_page
from app.pages.download import render_download_page
from app.pages.help import render_help_page
from app.pages.home import render_home_page
from app.pages.results import render_results_page
from app.pages.visualization import render_visualization_page

PAGES = ["Home", "Analyze", "Results", "Visualization", "Download", "Help"]


def _inject_styles() -> None:
    css_path = Path(__file__).resolve().parents[1] / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_shell() -> None:
    st.markdown(
        """
<div class="regplex-topbar">
  <div class="regplex-topbar-inner">
    <div class="brand">
      <h1>REGPLEX</h1>
      <span>Regulatory Genomics via Perplexity and Motif Exploration</span>
    </div>
    <div class="top-links">
      <a href="#home">Home</a>
      <a href="#analyze">Analyze</a>
      <a href="#visualization">Visualization</a>
      <a href="#download">Download</a>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="REGPLEX", page_icon="🧬", layout="wide")
    _inject_styles()
    _render_shell()

    st.markdown(
        """
<div class="hero-center">
  <div class="hero-brand">REGPLEX</div>
  <div class="hero-subtitle">Regulatory motif and perplexity-guided candidate region discovery</div>
  <div class="hero-chips">
    <span class="hero-chip">Explainable</span>
    <span class="hero-chip">Training-Free</span>
    <span class="hero-chip">Interactive Visual Analytics</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    page = st.radio("Page", PAGES, horizontal=True, label_visibility="collapsed")

    if page == "Home":
        render_home_page()
    elif page == "Analyze":
        render_analyze_page()
    elif page == "Results":
        render_results_page()
    elif page == "Visualization":
        render_visualization_page()
    elif page == "Download":
        render_download_page()
    else:
        render_help_page()


if __name__ == "__main__":
    main()
