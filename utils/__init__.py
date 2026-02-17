"""
Utils Module - Shared Utilities
================================

This module contains shared utility functions including:
- Registry and pattern loading
- Caching mechanisms
- State management
- Export functionality
- Visualization and plotting
- FASTA parsing
- Sequence validation
"""

__version__ = "2025.1"

# Import utility modules
from . import fasta
from . import validation
from . import export
from . import constants
from . import registry
from . import plotting
from . import caching
from . import state

# Export key functions for direct access
from .fasta import parse_fasta, read_fasta_file, format_fasta
from .validation import validate_sequence, validate_motif, validate_fasta_format
from .export import (
    export_to_csv, export_to_bed, export_to_json, 
    export_to_excel, export_to_gff3
)
from .constants import (
    CORE_OUTPUT_COLUMNS, MOTIF_SPECIFIC_COLUMNS,
    CHUNK_THRESHOLD, DEFAULT_CHUNK_SIZE
)
from .registry import load_registry_for_class
from .caching import get_cached_scanner
from .state import initialize_session_state, get_state, set_state

# Import from plotting styles
from .plotting.styles import MOTIF_CLASS_COLORS, NATURE_MOTIF_COLORS

__all__ = [
    # Modules
    'fasta',
    'validation',
    'export',
    'constants',
    'registry',
    'plotting',
    'caching',
    'state',
    # Functions
    'parse_fasta',
    'read_fasta_file',
    'format_fasta',
    'validate_sequence',
    'validate_motif',
    'validate_fasta_format',
    'export_to_csv',
    'export_to_bed',
    'export_to_json',
    'export_to_excel',
    'export_to_gff3',
    'load_registry_for_class',
    'get_cached_scanner',
    'initialize_session_state',
    'get_state',
    'set_state',
    # Constants
    'CORE_OUTPUT_COLUMNS',
    'MOTIF_SPECIFIC_COLUMNS',
    'CHUNK_THRESHOLD',
    'DEFAULT_CHUNK_SIZE',
    'MOTIF_CLASS_COLORS',
    'NATURE_MOTIF_COLORS',
]
