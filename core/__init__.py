"""
PERCALL core modules.
"""
from .perplexity import compute_perplexity, local_residual
from .region_caller import find_regions
from .motifs import MOTIF_PATTERNS, MOTIF_LABELS, MOTIF_COLORS, scan_motifs, count_motifs
from .second_order import compute_second_order_perplexity, second_order_residual

__all__ = [
    "compute_perplexity",
    "local_residual",
    "find_regions",
    "MOTIF_PATTERNS",
    "MOTIF_LABELS",
    "MOTIF_COLORS",
    "scan_motifs",
    "count_motifs",
    "compute_second_order_perplexity",
    "second_order_residual",
]
