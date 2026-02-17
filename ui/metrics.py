"""
UI Metrics Module - Metric Display Components
==============================================

This module provides components for displaying metrics, statistics, and
summary information in the Streamlit UI.

Key Components:
    - Metric cards for displaying key statistics
    - Summary panels for analysis results
    - Performance metrics displays
    - Statistical summaries

Module Integration:
    - Used by app.py for results display
    - Formats and presents analysis statistics
    - Creates visual metric displays
"""

import streamlit as st
from typing import Dict, Any, List, Optional

__all__ = [
    'display_metric_card',
    'display_metrics_row',
    'display_summary_stats',
    'create_metric_columns',
]


def display_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
    column: Optional[st.delta_generator.DeltaGenerator] = None
) -> None:
    """
    Display a single metric card with optional delta and help text.
    
    Args:
        label: Label for the metric
        value: Value to display (will be converted to string)
        delta: Optional delta/change indicator
        help_text: Optional help text for the metric
        column: Optional Streamlit column to place metric in
    
    Example:
        >>> display_metric_card("Total Motifs", 42, "+5", "Number of motifs detected")
    """
    if column is not None:
        with column:
            st.metric(label=label, value=value, delta=delta, help=help_text)
    else:
        st.metric(label=label, value=value, delta=delta, help=help_text)


def display_metrics_row(metrics: List[Dict[str, Any]], num_columns: Optional[int] = None) -> None:
    """
    Display multiple metrics in a row of columns.
    
    Args:
        metrics: List of metric dictionaries with keys: 'label', 'value', 'delta' (optional), 'help' (optional)
        num_columns: Number of columns (defaults to len(metrics))
    
    Example:
        >>> metrics = [
        ...     {'label': 'Sequences', 'value': 10},
        ...     {'label': 'Motifs', 'value': 150, 'delta': '+20'},
        ...     {'label': 'Classes', 'value': 8}
        ... ]
        >>> display_metrics_row(metrics)
    """
    if not metrics:
        return
    
    n_cols = num_columns if num_columns is not None else len(metrics)
    cols = st.columns(n_cols)
    
    for i, metric in enumerate(metrics):
        if i < len(cols):
            with cols[i]:
                st.metric(
                    label=metric.get('label', ''),
                    value=metric.get('value', ''),
                    delta=metric.get('delta'),
                    help=metric.get('help')
                )


def display_summary_stats(
    stats: Dict[str, Any],
    title: str = "Summary Statistics",
    show_header: bool = True
) -> None:
    """
    Display summary statistics in a formatted layout.
    
    Args:
        stats: Dictionary of statistics to display
        title: Title for the statistics section
        show_header: Whether to show the section header
    
    Example:
        >>> stats = {
        ...     'total_sequences': 10,
        ...     'total_motifs': 150,
        ...     'unique_classes': 8,
        ...     'avg_motifs_per_seq': 15.0
        ... }
        >>> display_summary_stats(stats, "Analysis Summary")
    """
    if show_header:
        st.subheader(title)
    
    # Create metrics from stats dictionary
    metrics = []
    for key, value in stats.items():
        # Format key as label (replace underscores with spaces, capitalize)
        label = key.replace('_', ' ').title()
        
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)
        
        metrics.append({'label': label, 'value': formatted_value})
    
    # Display in rows of up to 4 columns
    chunk_size = 4
    for i in range(0, len(metrics), chunk_size):
        chunk = metrics[i:i+chunk_size]
        display_metrics_row(chunk)


def create_metric_columns(num_columns: int = 4) -> List[st.delta_generator.DeltaGenerator]:
    """
    Create a row of columns for displaying metrics.
    
    Args:
        num_columns: Number of columns to create
    
    Returns:
        List of Streamlit column objects
    
    Example:
        >>> cols = create_metric_columns(3)
        >>> with cols[0]:
        ...     st.metric("Metric 1", "100")
        >>> with cols[1]:
        ...     st.metric("Metric 2", "200")
    """
    return st.columns(num_columns)


def display_performance_metrics(
    metrics: Dict[str, Any],
    title: str = "Performance Metrics"
) -> None:
    """
    Display performance metrics in a formatted layout.
    
    Args:
        metrics: Dictionary containing performance data
        title: Title for the metrics section
    
    Example:
        >>> metrics = {
        ...     'total_time': 45.2,
        ...     'throughput': 5432.1,
        ...     'sequences_processed': 10,
        ...     'avg_time_per_seq': 4.52
        ... }
        >>> display_performance_metrics(metrics)
    """
    st.subheader(title)
    
    cols = st.columns(4)
    
    # Total Time
    if 'total_time' in metrics:
        with cols[0]:
            st.metric("Total Time", f"{metrics['total_time']:.1f}s")
    
    # Throughput
    if 'throughput' in metrics:
        with cols[1]:
            throughput = metrics['throughput']
            st.metric("Throughput", f"{throughput:,.0f} bp/s")
    
    # Sequences Processed
    if 'sequences_processed' in metrics:
        with cols[2]:
            st.metric("Sequences", f"{metrics['sequences_processed']:,}")
    
    # Average Time per Sequence
    if 'avg_time_per_seq' in metrics:
        with cols[3]:
            st.metric("Avg Time/Seq", f"{metrics['avg_time_per_seq']:.2f}s")


def create_info_box(
    title: str,
    content: str,
    box_type: str = "info"
) -> None:
    """
    Create an information box with title and content.
    
    Args:
        title: Title for the info box
        content: Content to display
        box_type: Type of box ('info', 'success', 'warning', 'error')
    
    Example:
        >>> create_info_box("Notice", "Analysis completed successfully!", "success")
    """
    if box_type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif box_type == "success":
        st.success(f"**{title}**\n\n{content}")
    elif box_type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif box_type == "error":
        st.error(f"**{title}**\n\n{content}")
    else:
        st.info(f"**{title}**\n\n{content}")
