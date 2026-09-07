from __future__ import annotations

import streamlit as st

from app.pages.analyze import render_analyze_page
from app.pages.download import render_download_page
from app.pages.help import render_help_page
from app.pages.home import render_home_page
from app.pages.results import render_results_page
from app.pages.visualization import render_visualization_page

PAGES = ["Home", "Analyze", "Results", "Visualization", "Download", "Help"]


def main() -> None:
    st.set_page_config(page_title="PerCALL", layout="wide")
    st.title("PerCALL")
    st.caption("Candidate regulatory region prediction from local DNA perplexity depressions")

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
