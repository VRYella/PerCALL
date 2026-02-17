"""
UI Layout Module - Page Structure and Layout Components
=======================================================

This module provides layout components for structuring the Streamlit UI,
including containers, columns, and page layout utilities.

Key Components:
    - Page configuration
    - Column layouts
    - Containers and sections
    - Tab management
    - Header and footer components

Module Integration:
    - Used by app.py for page structure
    - Provides consistent layout patterns
    - Manages responsive design
"""

import streamlit as st
from typing import List, Optional, Dict, Any

__all__ = [
    'configure_page',
    'create_header',
    'create_columns',
    'create_tabs',
    'create_container',
    'create_expander',
]


def configure_page(
    title: str = "Application",
    icon: str = "🧬",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded"
) -> None:
    """
    Configure Streamlit page settings.
    
    Args:
        title: Page title
        icon: Page icon (emoji or URL)
        layout: Layout mode ('centered' or 'wide')
        initial_sidebar_state: Initial sidebar state ('expanded' or 'collapsed')
    
    Example:
        >>> configure_page(
        ...     title="NonBDNA Finder",
        ...     icon="🧬",
        ...     layout="wide"
        ... )
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state
    )


def create_header(
    title: str,
    subtitle: Optional[str] = None,
    divider: bool = True
) -> None:
    """
    Create a page header with optional subtitle.
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
        divider: Show divider after header if True
    
    Example:
        >>> create_header(
        ...     "Non-B DNA Motif Detection",
        ...     subtitle="Professional Analysis Suite"
        ... )
    """
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    if divider:
        st.divider()


def create_columns(
    num_columns: int = 2,
    gap: str = "medium",
    spec: Optional[List[int]] = None
) -> List[st.delta_generator.DeltaGenerator]:
    """
    Create a row of columns with specified widths.
    
    Args:
        num_columns: Number of columns
        gap: Gap between columns ('small', 'medium', 'large')
        spec: Optional list of column widths (e.g., [2, 1] for 2:1 ratio)
    
    Returns:
        List of column objects
    
    Example:
        >>> cols = create_columns(3, gap="large")
        >>> with cols[0]:
        ...     st.write("Column 1")
        >>> with cols[1]:
        ...     st.write("Column 2")
    """
    if spec is not None:
        return st.columns(spec, gap=gap)
    else:
        return st.columns(num_columns, gap=gap)


def create_tabs(tab_names: List[str]) -> Dict[str, st.delta_generator.DeltaGenerator]:
    """
    Create tabs and return a dictionary mapping names to tab objects.
    
    Args:
        tab_names: List of tab names
    
    Returns:
        Dictionary mapping tab names to tab objects
    
    Example:
        >>> tabs = create_tabs(["Home", "Analysis", "Results"])
        >>> with tabs["Home"]:
        ...     st.write("Home content")
        >>> with tabs["Analysis"]:
        ...     st.write("Analysis content")
    """
    tab_objects = st.tabs(tab_names)
    return dict(zip(tab_names, tab_objects))


def create_container(border: bool = False) -> st.delta_generator.DeltaGenerator:
    """
    Create a container for grouping elements.
    
    Args:
        border: Show border around container if True
    
    Returns:
        Container object
    
    Example:
        >>> container = create_container(border=True)
        >>> with container:
        ...     st.write("Content in container")
    """
    return st.container(border=border)


def create_expander(
    label: str,
    expanded: bool = False
) -> st.delta_generator.DeltaGenerator:
    """
    Create an expandable section.
    
    Args:
        label: Label for the expander
        expanded: Initially expanded if True
    
    Returns:
        Expander object
    
    Example:
        >>> with create_expander("Advanced Options", expanded=False):
        ...     st.write("Advanced settings here")
    """
    return st.expander(label, expanded=expanded)


def create_sidebar_header(title: str, icon: str = "") -> None:
    """
    Create a sidebar header.
    
    Args:
        title: Header title
        icon: Optional icon/emoji
    
    Example:
        >>> with st.sidebar:
        ...     create_sidebar_header("Settings", "⚙️")
    """
    if icon:
        st.sidebar.title(f"{icon} {title}")
    else:
        st.sidebar.title(title)


def create_two_column_layout(
    left_content_fn,
    right_content_fn,
    left_width: int = 1,
    right_width: int = 1,
    gap: str = "large"
) -> None:
    """
    Create a two-column layout and execute content functions.
    
    Args:
        left_content_fn: Function to execute in left column
        right_content_fn: Function to execute in right column
        left_width: Relative width of left column
        right_width: Relative width of right column
        gap: Gap between columns
    
    Example:
        >>> def left():
        ...     st.write("Left content")
        >>> def right():
        ...     st.write("Right content")
        >>> create_two_column_layout(left, right)
    """
    cols = st.columns([left_width, right_width], gap=gap)
    
    with cols[0]:
        left_content_fn()
    
    with cols[1]:
        right_content_fn()


def create_section(
    title: str,
    content_fn,
    border: bool = False,
    divider_above: bool = False,
    divider_below: bool = False
) -> None:
    """
    Create a content section with optional borders and dividers.
    
    Args:
        title: Section title
        content_fn: Function to execute for section content
        border: Show border around section if True
        divider_above: Show divider above section if True
        divider_below: Show divider below section if True
    
    Example:
        >>> def content():
        ...     st.write("Section content")
        >>> create_section("My Section", content, border=True)
    """
    if divider_above:
        st.divider()
    
    container = st.container(border=border)
    
    with container:
        if title:
            st.subheader(title)
        content_fn()
    
    if divider_below:
        st.divider()


def add_vertical_space(lines: int = 1) -> None:
    """
    Add vertical spacing.
    
    Args:
        lines: Number of blank lines to add
    
    Example:
        >>> add_vertical_space(3)  # Add 3 blank lines
    """
    for _ in range(lines):
        st.write("")


def create_centered_content(content_fn, max_width: int = 800) -> None:
    """
    Create centered content with maximum width.
    
    Args:
        content_fn: Function to execute for content
        max_width: Maximum width in pixels
    
    Example:
        >>> def content():
        ...     st.write("Centered content")
        >>> create_centered_content(content, max_width=600)
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        content_fn()
