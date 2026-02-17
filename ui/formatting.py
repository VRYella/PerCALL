"""UI Formatting Module
====================

Text formatting and display helper functions.
Provides consistent formatting across the UI.

Extracted from app.py for modular architecture.
"""

import matplotlib.pyplot as plt
import numpy as np
import os  # Added for image path checking
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

def format_time_scientific(seconds: float) -> str:
    """
    Format elapsed time in simple MM:SS format.
    
    This format provides:
    - Human-readable minutes and seconds
    - No hours or microseconds (simplified display)
    - Consistent display across all workflows
    
    Args:
        seconds: Elapsed time in seconds (float)
        
    Returns:
        Formatted time string (e.g., "02:15" or "125:32")
        
    Examples:
        >>> format_time_scientific(0.234)
        "00:00"
        >>> format_time_scientific(135.678)
        "02:15"
        >>> format_time_scientific(5432.123)
        "90:32"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    return f"{minutes:02d}:{secs:02d}"


def format_time_compact(seconds: float) -> str:
    """
    Format elapsed time in MM:SS format for compact displays.
    
    Simple minutes:seconds format for all durations.
    
    Args:
        seconds: Elapsed time in seconds (float)
        
    Returns:
        Formatted time string (e.g., "02:15" or "125:32")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ---------- CACHING FUNCTIONS (Memory-Efficient) ----------
@st.cache_resource(show_spinner=False)
def cache_genome_as_numpy(sequence: str) -> np.ndarray:
    """
    Cache genome sequence as NumPy byte array for memory efficiency.
    
    This prevents reloading large genomes and reduces memory footprint
    when using Streamlit on free tier (1GB limit).
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        NumPy array of sequence bytes
    """
    return np.frombuffer(sequence.encode('utf-8'), dtype=np.uint8)


@st.cache_resource(show_spinner=False)
def cache_hyperscan_database(_patterns: list = None):
    """
    Cache compiled Hyperscan database for pattern matching.
    
    The underscore prefix on _patterns parameter is used by Streamlit
    to indicate the parameter should not be hashed for caching purposes.
    This prevents Streamlit from attempting to hash complex pattern objects.
    
    Args:
        _patterns: List of (pattern, pattern_id) tuples to compile
        
    Returns:
        Compiled Hyperscan database or None
    """
    if not HYPERSCAN_AVAILABLE or _patterns is None or len(_patterns) == 0:
        logger.debug("Hyperscan not available or no patterns provided")
        return None
    
    try:
        # Compile patterns into Hyperscan database
        logger.debug(f"Compiling Hyperscan database with {len(_patterns)} patterns...")
        
        expressions = []
        ids = []
        flags = []
        
        for pattern, pattern_id in _patterns:
            # Encode pattern with error handling
            # Note: DNA patterns should normally be ASCII (ATGC), but we provide
            # UTF-8 fallback for robustness in case patterns contain metadata or
            # special characters. A warning is logged to help identify data quality issues.
            try:
                pattern_bytes = pattern.encode('ascii')
            except UnicodeEncodeError:
                # Fall back to UTF-8 if ASCII fails
                pattern_bytes = pattern.encode('utf-8')
                logger.warning(f"Pattern {pattern_id} contains non-ASCII characters (expected ATGC). Using UTF-8 encoding.")
            
            expressions.append(pattern_bytes)
            ids.append(pattern_id)
            # Use CASELESS and DOTALL flags for DNA matching
            flags.append(hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_DOTALL)
        
        db = hyperscan.Database()
        db.compile(
            expressions=expressions,
            ids=ids,
            elements=len(expressions),
            flags=flags
        )
        
        logger.info(f"Successfully compiled Hyperscan database with {len(expressions)} patterns")
        return db
        
    except Exception as e:
        logger.error(f"Hyperscan database compilation failed: {e}")
        st.warning(f"Hyperscan database compilation failed: {e}. Falling back to regex matching.")
        return None


@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def cache_analysis_results(sequence_hash: str, sequence: str, name: str):
    """
    Cache analysis results for sequences to avoid re-computation.
    Uses sequence hash for efficient cache key lookup.
    
    Args:
        sequence_hash: Hash of sequence for cache key
        sequence: DNA sequence string
        name: Sequence name
        
    Returns:
        List of detected motifs
    """
    return analyze_sequence(sequence, name)


@st.cache_data(show_spinner=False, max_entries=20, ttl=3600)
def get_cached_stats(sequence: str, motifs_json: str):
    """
    Cache statistics calculation for sequences.
    
    Args:
        sequence: DNA sequence
        motifs_json: JSON string of motifs (for cache key)
        
    Returns:
        Dictionary of sequence statistics
    """
    import json
    motifs = json.loads(motifs_json) if motifs_json else []
    return get_basic_stats(sequence, motifs)


def get_system_info():
    """
    Get current system memory and resource information.
    Uses psutil if available, otherwise provides basic info.
    
    Returns:
        Dictionary with memory and system info
    """
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        
        return {
            'memory_total_mb': memory.total / (1024 * 1024),
            'memory_used_mb': memory.used / (1024 * 1024),
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_percent': memory.percent,
            'cpu_count': psutil.cpu_count(),
            'available': True
        }
    except (ImportError, AttributeError):
        # psutil not available or doesn't support this platform
        return {
            'memory_total_mb': 0,
            'memory_used_mb': 0,
            'memory_available_mb': 0,
            'memory_percent': 0,
            'cpu_count': 1,
            'available': False
        }


# ---------- PAGE CONFIG ----------
# Note: Page configuration should be done in the main app, not at module level
# st.set_page_config(
#     page_title=f"{UI_TEXT['app_title']} - Non-B DNA Motif Finder",
#     layout=LAYOUT_CONFIG['layout_mode'],
#     page_icon=None,
#     menu_items={'About': f"NBDScanner | Developed by {UI_TEXT['author']}"}
# )


def format_sequence_limit():
    """Format the sequence limit for display - now shows 'unlimited' since limit is removed"""
    return "unlimited (chunked processing enabled)"

# Get motif classification info
# Note: Should be obtained when needed, not at module level
# CLASSIFICATION_INFO = get_motif_classification_info()

# ---------- PATCH: Ensure every motif has Subclass ----------
def ensure_subclass(motif):
    """Guarantee every motif has a string 'Subclass'"""
    if isinstance(motif, dict):
        if 'Subclass' not in motif or motif['Subclass'] is None:
            motif['Subclass'] = motif.get('Subtype', 'Other')
        return motif
    else:
        # Handle non-dict motifs gracefully
        return {'Subclass': 'Other', 'Motif': motif}


# ---------- HELPER: Format time for display ----------
def format_time(seconds):
    """Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds (float or int)
        
    Returns:
        Formatted string (e.g., "45.3s", "12m 30s", "2h 15m")
    
    Examples:
        >>> format_time(45.3)
        '45.3s'
        >>> format_time(750)
        '12m 30s'
        >>> format_time(7800)
        '2h 10m'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

