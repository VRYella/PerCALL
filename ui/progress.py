"""
UI Progress Module - Progress Indicators and Status Display
===========================================================

This module provides progress bar and status indicator components for the
Streamlit UI, including real-time analysis progress tracking.

Key Components:
    - Progress bars for sequence analysis
    - Status indicators and metrics
    - Estimated time remaining calculations
    - Memory usage displays

Module Integration:
    - Used by app.py for displaying analysis progress
    - Integrates with engine.detection for progress callbacks
    - Provides real-time feedback during long-running operations
"""

import streamlit as st
import time
from typing import Dict, Any, Optional

__all__ = [
    'display_progress_bar',
    'display_analysis_metrics',
    'format_time_display',
    'create_progress_container',
]


def format_time_display(seconds: float) -> str:
    """
    Format elapsed time in simple MM:SS format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "02:35")
    
    Example:
        >>> format_time_display(155.5)
        '02:35'
    """
    if seconds < 0:
        return "00:00"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    return f"{minutes:02d}:{secs:02d}"


def format_time_compact(seconds: float) -> str:
    """
    Format elapsed time in compact MM:SS format for displays.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    
    Example:
        >>> format_time_compact(125.7)
        '02:05'
    """
    return format_time_display(seconds)


def display_progress_bar(
    progress: float,
    text: str = "",
    placeholder: Optional[st.delta_generator.DeltaGenerator] = None
) -> None:
    """
    Display a progress bar with optional text.
    
    Args:
        progress: Progress value between 0.0 and 1.0
        text: Optional text to display with progress bar
        placeholder: Optional Streamlit placeholder to update
    
    Example:
        >>> placeholder = st.empty()
        >>> display_progress_bar(0.5, "Processing...", placeholder)
    """
    if placeholder is not None:
        placeholder.progress(progress, text=text)
    else:
        st.progress(progress, text=text)


def display_analysis_metrics(
    elapsed: float,
    estimated_remaining: float,
    progress_percent: float,
    memory_mb: Optional[float] = None
) -> None:
    """
    Display analysis metrics in a column layout.
    
    Args:
        elapsed: Elapsed time in seconds
        estimated_remaining: Estimated remaining time in seconds
        progress_percent: Progress percentage (0-100)
        memory_mb: Optional memory usage in MB
    
    Example:
        >>> display_analysis_metrics(30.5, 15.2, 67.5, 150.3)
    """
    cols = st.columns(4 if memory_mb is not None else 3)
    
    with cols[0]:
        st.metric("Elapsed", format_time_display(elapsed))
    
    with cols[1]:
        st.metric("Remaining", format_time_display(estimated_remaining))
    
    with cols[2]:
        st.metric("Progress", f"{progress_percent:.1f}%")
    
    if memory_mb is not None and len(cols) > 3:
        with cols[3]:
            st.metric("Memory", f"{memory_mb:.0f} MB")


def create_progress_container(title: str = "Analysis Progress") -> Dict[str, Any]:
    """
    Create a container with placeholders for progress display.
    
    Args:
        title: Title for the progress container
    
    Returns:
        Dictionary with 'container', 'progress_bar', 'status', and 'metrics' placeholders
    
    Example:
        >>> progress = create_progress_container("Running Analysis")
        >>> progress['status'].write("Processing sequence 1...")
        >>> progress['progress_bar'].progress(0.5)
    """
    container = st.container()
    
    with container:
        st.subheader(title)
        progress_bar = st.empty()
        status = st.empty()
        metrics = st.empty()
    
    return {
        'container': container,
        'progress_bar': progress_bar,
        'status': status,
        'metrics': metrics
    }


def update_progress(
    progress_dict: Dict[str, Any],
    progress: float,
    status_text: str = "",
    elapsed: Optional[float] = None,
    remaining: Optional[float] = None,
    memory_mb: Optional[float] = None
) -> None:
    """
    Update a progress container with current values.
    
    Args:
        progress_dict: Dictionary returned from create_progress_container()
        progress: Progress value (0.0 to 1.0)
        status_text: Status message to display
        elapsed: Optional elapsed time in seconds
        remaining: Optional remaining time in seconds
        memory_mb: Optional memory usage in MB
    
    Example:
        >>> progress = create_progress_container()
        >>> update_progress(progress, 0.75, "Almost done...", 45.2, 15.1, 200.5)
    """
    # Update progress bar
    progress_dict['progress_bar'].progress(progress, text=status_text)
    
    # Update status
    if status_text:
        progress_dict['status'].write(status_text)
    
    # Update metrics if provided
    if elapsed is not None and remaining is not None:
        with progress_dict['metrics']:
            display_analysis_metrics(
                elapsed=elapsed,
                estimated_remaining=remaining,
                progress_percent=progress * 100,
                memory_mb=memory_mb
            )


def create_simple_progress(text: str = "Processing...") -> st.delta_generator.DeltaGenerator:
    """
    Create a simple progress bar with text.
    
    Args:
        text: Text to display with progress bar
    
    Returns:
        Streamlit empty placeholder for updating progress
    
    Example:
        >>> progress = create_simple_progress("Analyzing sequences...")
        >>> progress.progress(0.5)
    """
    return st.progress(0, text=text)
