"""
Plotting Module
==============

Visualization functions for Non-B DNA motif analysis.

Submodules:
- distributions: Distribution plots
- coverage: Coverage maps
- density: Density analysis
- statistical: Statistical plots
- genomic: Genome-wide visualizations
- styles: Style configurations and color schemes
"""

__version__ = "2025.1"

# Import plotting modules
from . import distributions
from . import coverage
from . import density
from . import statistical
from . import genomic
from . import styles

# Export key plotting functions for direct access
from .distributions import (
    plot_motif_distribution, 
    plot_nested_pie_chart,
    plot_score_distribution,
    plot_length_distribution
)
from .coverage import plot_coverage_map, plot_density_heatmap
from .density import (
    plot_circos_motif_density,
    plot_density_comparison,
    plot_density_comparison_by_subclass,
    plot_subclass_density_heatmap,
    plot_enrichment_analysis,
    plot_enrichment_analysis_by_subclass
)
from .statistical import (
    plot_class_analysis_comprehensive,
    plot_subclass_analysis_comprehensive
)
from .genomic import (
    plot_manhattan_motif_density,
    plot_cumulative_motif_distribution,
    plot_motif_cooccurrence_matrix,
    plot_gc_content_correlation,
    plot_linear_motif_track,
    plot_cluster_size_distribution,
    plot_motif_length_kde
)
from .styles import (
    MOTIF_CLASS_COLORS,
    NATURE_MOTIF_COLORS,
    VISUALIZATION_PALETTE,
    PLOT_DPI
)

__all__ = [
    # Modules
    'distributions',
    'coverage',
    'density',
    'statistical',
    'genomic',
    'styles',
    # Distribution functions
    'plot_motif_distribution',
    'plot_nested_pie_chart',
    'plot_score_distribution',
    'plot_length_distribution',
    # Coverage functions
    'plot_coverage_map',
    'plot_density_heatmap',
    # Density functions
    'plot_circos_motif_density',
    'plot_density_comparison',
    'plot_density_comparison_by_subclass',
    'plot_subclass_density_heatmap',
    'plot_enrichment_analysis',
    'plot_enrichment_analysis_by_subclass',
    # Statistical functions
    'plot_class_analysis_comprehensive',
    'plot_subclass_analysis_comprehensive',
    # Genomic functions
    'plot_manhattan_motif_density',
    'plot_cumulative_motif_distribution',
    'plot_motif_cooccurrence_matrix',
    'plot_gc_content_correlation',
    'plot_linear_motif_track',
    'plot_cluster_size_distribution',
    'plot_motif_length_kde',
    # Styles
    'MOTIF_CLASS_COLORS',
    'NATURE_MOTIF_COLORS',
    'VISUALIZATION_PALETTE',
    'PLOT_DPI',
]
