"""
Summary Renderer Module
=======================

This module provides safe and robust HTML rendering utilities for the NonBDNAFinder
Streamlit application. It ensures proper HTML display without escaping issues and
provides guaranteed-visible progress bars.

Key Functions:
- render_summary_block(): Safe HTML injection for summary sections
- render_progress(): Guaranteed-visible progress bar wrapper

Author: Dr. Venkata Rajesh Yella
Version: 2025.1
License: MIT
"""

import streamlit as st
import re
from typing import Optional


def render_summary_block(html: str, container: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
    """
    Render HTML summary block with proper formatting and no escaping.
    
    This function ensures that HTML content is displayed correctly without being
    escaped by Streamlit. It strips leading whitespace, validates HTML structure,
    and prevents automatic wrapping that can cause display issues.
    
    Features:
    - No HTML escaping (renders actual HTML tags)
    - Automatic whitespace stripping
    - Basic HTML validation
    - Prevents Streamlit auto-wrapping
    - Compatible with CSS overrides
    
    Args:
        html: HTML string to render. Can contain <div>, <p>, <span>, <b>, <i>, etc.
        container: Optional Streamlit container to render into. If None, uses default.
    
    Example:
        >>> render_summary_block('''
        ...     <div style="background: #f0f9ff; padding: 1rem;">
        ...         <b>Processing time:</b> 00:03:42
        ...         <br>
        ...         <b>Motifs detected:</b> 33
        ...     </div>
        ... ''')
    
    Notes:
        - HTML must be well-formed (matching opening/closing tags)
        - Inline styles are recommended for consistent appearance
        - Use with st.markdown(..., unsafe_allow_html=True) under the hood
    """
    # Strip leading/trailing whitespace and normalize line breaks
    cleaned_html = html.strip()
    
    # Remove excessive leading whitespace from each line (common with multiline strings)
    lines = cleaned_html.split('\n')
    # Find minimum indentation (ignoring empty lines)
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
        lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
    cleaned_html = '\n'.join(lines)
    
    # Basic validation: check for balanced tags (simple check)
    # Count opening and closing div tags as a basic sanity check
    open_divs = len(re.findall(r'<div[^>]*>', cleaned_html))
    close_divs = len(re.findall(r'</div>', cleaned_html))
    
    if open_divs != close_divs:
        st.warning(f"⚠️ HTML validation warning: Unmatched div tags ({open_divs} open, {close_divs} close)")
    
    # Render the HTML using Streamlit's markdown with unsafe_allow_html
    if container is None:
        st.markdown(cleaned_html, unsafe_allow_html=True)
    else:
        container.markdown(cleaned_html, unsafe_allow_html=True)


def render_progress(value: float, label: Optional[str] = None, 
                   container: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
    """
    Render a guaranteed-visible progress bar with proper isolation.
    
    This function ensures progress bars are always visible by using native Streamlit
    widgets and proper container isolation. It prevents HTML interference and CSS
    conflicts that can hide progress indicators.
    
    Features:
    - Native Streamlit widget (not HTML-based)
    - Isolated container prevents CSS conflicts
    - Visible against bright themes
    - Supports optional labeling
    - Compatible with columns, tabs, and expanders
    
    Args:
        value: Progress value between 0.0 and 1.0 (e.g., 0.5 for 50%)
        label: Optional text label to display above the progress bar
        container: Optional Streamlit container to render into. If None, uses default.
    
    Example:
        >>> render_progress(0.75, label="Analysis Progress")
        >>> # Renders: "Analysis Progress"
        >>> #           [████████████████████▁▁▁▁▁] 75%
        
        >>> # In a specific container
        >>> col1, col2 = st.columns(2)
        >>> render_progress(0.5, label="Processing", container=col1)
    
    Notes:
        - Value is automatically clamped to [0.0, 1.0] range
        - Progress bar uses Streamlit's native styling
        - Works inside st.columns(), st.tabs(), st.expander()
        - Not affected by custom CSS that might hide HTML-based progress
    """
    # Clamp value to valid range
    value = max(0.0, min(1.0, value))
    
    # Use a container to isolate the progress bar
    if container is None:
        # Create a container for isolation
        with st.container():
            if label:
                st.caption(label)
            st.progress(value)
    else:
        # Use provided container
        with container:
            if label:
                st.caption(label)
            st.progress(value)


def render_metric_card(label: str, value: str, delta: Optional[str] = None,
                      color: str = "#10b981", 
                      container: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
    """
    Render a styled metric card with optional delta indicator.
    
    Creates a visually appealing metric card with customizable colors and optional
    delta (change) indicator. Useful for displaying summary statistics.
    
    Args:
        label: Metric label (e.g., "Total Motifs")
        value: Metric value (e.g., "1,234")
        delta: Optional delta text (e.g., "+15%" or "↑ 20")
        color: Primary color for the card (hex format)
        container: Optional Streamlit container
    
    Example:
        >>> render_metric_card("Total Motifs", "1,234", delta="+15%")
        >>> render_metric_card("Processing Time", "03:42", color="#2563EB")
    """
    delta_html = ""
    if delta:
        delta_html = f"<div style='color: {color}; font-size: 0.9rem; margin-top: 0.3rem;'>{delta}</div>"
    
    html = f"""
    <div style='background: white; padding: 1rem; border-radius: 12px; 
                border-left: 4px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                margin-bottom: 1rem;'>
        <div style='color: #6b7280; font-size: 0.85rem; font-weight: 500; 
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
            {label}
        </div>
        <div style='color: #111827; font-size: 1.8rem; font-weight: 700; line-height: 1;'>
            {value}
        </div>
        {delta_html}
    </div>
    """
    
    render_summary_block(html, container=container)


def render_info_box(title: str, content: str, box_type: str = "info",
                   container: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
    """
    Render an information box with icon and styling.
    
    Creates a styled information box for warnings, tips, errors, or general info.
    
    Args:
        title: Box title
        content: Box content (supports HTML)
        box_type: Type of box - "info", "warning", "error", "success"
        container: Optional Streamlit container
    
    Example:
        >>> render_info_box("Important", "Please validate your results", box_type="warning")
    """
    # Define colors and icons for each box type
    styles = {
        "info": {"color": "#0ea5e9", "bg": "#f0f9ff", "icon": "ℹ️"},
        "warning": {"color": "#f59e0b", "bg": "#fef3c7", "icon": "⚠️"},
        "error": {"color": "#ef4444", "bg": "#fef2f2", "icon": "❌"},
        "success": {"color": "#10b981", "bg": "#f0fdf4", "icon": "✅"},
    }
    
    style = styles.get(box_type, styles["info"])
    
    html = f"""
    <div style='background: {style["bg"]}; padding: 1rem; border-radius: 8px; 
                margin-bottom: 1rem; border-left: 4px solid {style["color"]};'>
        <div style='color: {style["color"]}; font-weight: 600; margin-bottom: 0.5rem;'>
            {style["icon"]} {title}
        </div>
        <div style='color: #374151; line-height: 1.6;'>
            {content}
        </div>
    </div>
    """
    
    render_summary_block(html, container=container)


# Export public API
__all__ = [
    'render_summary_block',
    'render_progress',
    'render_metric_card',
    'render_info_box',
]
