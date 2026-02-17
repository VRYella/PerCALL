"""
Utils Caching Module - Caching Mechanisms for Performance
==========================================================

This module provides caching utilities for expensive operations including:
    - Sequence caching (NumPy arrays for memory efficiency)
    - Analysis results caching
    - Statistics caching
    - Scanner instance caching

Module Integration:
    - Used by app.py for Streamlit caching
    - Used by nonbscanner.py for scanner instance caching
    - Provides thread-safe singleton patterns

Performance Benefits:
    - Avoids re-computation of expensive analyses
    - Reduces memory footprint for large sequences
    - Prevents re-initialization of detector instances
"""

import threading
from typing import Optional

__all__ = [
    'get_cached_scanner',
    'clear_scanner_cache',
]

# Module-level cache for scanner instance
_CACHED_SCANNER = None
_SCANNER_LOCK = threading.Lock()


def get_cached_scanner():
    """
    Get or create cached NonBScanner instance with pre-initialized detectors.
    
    Thread-safe singleton pattern using double-checked locking.
    This avoids re-initializing all detectors on repeated calls, which
    significantly improves performance when analyzing multiple sequences.
    
    Returns:
        Cached NonBScanner instance
    
    Example:
        >>> from utils.caching import get_cached_scanner
        >>> scanner = get_cached_scanner()
        >>> motifs = scanner.analyze_sequence("ATGC...", "seq1")
    """
    global _CACHED_SCANNER
    
    # First check without lock (fast path)
    if _CACHED_SCANNER is None:
        with _SCANNER_LOCK:
            # Double-check inside lock (slow path)
            if _CACHED_SCANNER is None:
                # Import here to avoid circular dependency
                from engine.detection import NonBScanner
                _CACHED_SCANNER = NonBScanner(enable_all_detectors=True)
    
    return _CACHED_SCANNER


def clear_scanner_cache():
    """
    Clear the cached scanner instance.
    
    This forces re-initialization of all detectors on the next call
    to get_cached_scanner(). Useful for testing or when detector
    configuration needs to be updated.
    
    Example:
        >>> from utils.caching import clear_scanner_cache
        >>> clear_scanner_cache()  # Force re-initialization
    """
    global _CACHED_SCANNER
    
    with _SCANNER_LOCK:
        _CACHED_SCANNER = None
