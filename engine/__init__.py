"""
Engine Module - Core Detection Logic
====================================

This module contains all backend engine components including:
- Detection orchestration
- Scoring algorithms
- Pattern processing
- Overlap merging and deduplication
- Sequence chunking
- Motif detectors
"""

__version__ = "2025.1"

# Import core engine modules
from . import scoring
from . import merging
from . import chunking
from . import sequence_ops
from . import detectors
from . import detection
from . import patterns

# Export key classes and functions for direct access
from .detection import NonBScanner, AnalysisProgress, get_cached_scanner, analyze_sequence
from .scoring import normalize_motif_scores, calculate_motif_statistics
from .merging import remove_overlaps, detect_hybrid_motifs, detect_cluster_motifs
from .chunking import process_sequence_chunks, should_chunk_sequence
from .sequence_ops import reverse_complement, gc_content

__all__ = [
    # Modules
    'scoring',
    'merging',
    'chunking',
    'sequence_ops',
    'detectors',
    'detection',
    'patterns',
    # Key functions and classes
    'NonBScanner',
    'AnalysisProgress',
    'get_cached_scanner',
    'analyze_sequence',
    'normalize_motif_scores',
    'calculate_motif_statistics',
    'remove_overlaps',
    'detect_hybrid_motifs',
    'detect_cluster_motifs',
    'process_sequence_chunks',
    'should_chunk_sequence',
    'reverse_complement',
    'gc_content',
]
