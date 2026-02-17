"""
Utils Plotting Styles Module - Style Configurations for Visualizations
======================================================================

This module provides centralized style configurations for all plots and visualizations,
including color themes, fonts, and layout parameters.

Key Components:
    - Motif class color schemes
    - Visualization color palettes
    - Font and typography configurations
    - Publication-quality plot parameters

Module Integration:
    - Used by all plotting modules in utils.plotting
    - Provides consistent visual styling across all charts
    - Supports colorblind-friendly palettes
"""

__all__ = [
    'MOTIF_CLASS_COLORS',
    'NATURE_MOTIF_COLORS',
    'VISUALIZATION_PALETTE',
    'PLOT_DPI',
    'PLOT_FONT_SIZE',
    'PLOT_TITLE_SIZE',
]

# =============================================================================
# MOTIF CLASS COLORS
# =============================================================================

# Nature/Science-inspired color scheme for motif classes
# Colorblind-friendly palette optimized for publication
NATURE_MOTIF_COLORS = {
    'Curved_DNA': '#CC79A7',           # Pink/Rose
    'G-Quadruplex': '#0072B2',         # Blue
    'Z-DNA': '#882255',                # Magenta/Purple
    'Cruciform': '#56B4E9',            # Light Blue/Cyan
    'Triplex': '#E69F00',              # Orange/Gold
    'R-Loop': '#009E73',               # Green/Teal
    'i-Motif': '#0072B2',              # Blue (same as G4)
    'A-philic_DNA': '#CC79A7',         # Pink/Rose (same as Curved)
    'Slipped_DNA': '#E69F00',          # Orange/Gold (same as Triplex)
    'Hybrid': '#888888',               # Gray
    'Non-B_DNA_Clusters': '#666666',   # Dark Gray
}

# Alternative color scheme with higher contrast
MOTIF_CLASS_COLORS = {
    'Curved_DNA': '#FF6B9D',           # Bright Pink
    'G-Quadruplex': '#4A90E2',         # Sky Blue
    'Z-DNA': '#9B59B6',                # Purple
    'Cruciform': '#3498DB',            # Ocean Blue
    'Triplex': '#F39C12',              # Orange
    'R-Loop': '#27AE60',               # Emerald Green
    'i-Motif': '#16A085',              # Turquoise
    'A-philic_DNA': '#E74C3C',         # Red
    'Slipped_DNA': '#F1C40F',          # Yellow
    'Hybrid': '#95A5A6',               # Light Gray
    'Non-B_DNA_Clusters': '#7F8C8D',   # Medium Gray
}

# =============================================================================
# VISUALIZATION PALETTE
# =============================================================================

# Scientific color palette for charts and plots
# Ultra vibrant and accessible colorblind-friendly palette
VISUALIZATION_PALETTE = {
    'chart_1': '#FF6D00',    # Blazing Orange
    'chart_2': '#0091FF',    # Electric Blue
    'chart_3': '#00E676',    # Neon Green
    'chart_4': '#FFEA00',    # Brilliant Yellow
    'chart_5': '#0043A8',    # Deep Electric Blue
    'chart_6': '#FF1744',    # Neon Red
    'chart_7': '#FF00AA',    # Electric Pink
    'chart_8': '#76FF03',    # Vivid Lime
    'chart_9': '#D500F9',    # Electric Purple
    'chart_10': '#00E5FF',   # Neon Cyan
    'chart_11': '#FFC400',   # Gold Flash
    'chart_12': '#546E7A',   # Steel Blue
}

# Semantic colors for status and state indication
SEMANTIC_COLORS = {
    'success': '#00E676',          # Neon Green
    'warning': '#FF9100',          # Blazing Orange
    'error': '#FF1744',            # Neon Red
    'info': '#00B4FF',             # Electric Blue
    'progress': '#E040FB',         # Neon Purple
}

# =============================================================================
# PLOT PARAMETERS
# =============================================================================

# Default DPI for plots (150 for screen, 300 for publication)
PLOT_DPI = 150

# Font sizes for plots
PLOT_FONT_SIZE = 10      # Body text and labels
PLOT_TITLE_SIZE = 12     # Plot titles
PLOT_AXIS_LABEL_SIZE = 10  # Axis labels
PLOT_LEGEND_SIZE = 9     # Legend text

# Plot dimensions (in inches for matplotlib)
PLOT_WIDTH = 10
PLOT_HEIGHT = 6

# Line widths and marker sizes
PLOT_LINE_WIDTH = 2.0
PLOT_MARKER_SIZE = 6

# Alpha transparency for overlapping elements
PLOT_ALPHA = 0.7
PLOT_FILL_ALPHA = 0.3

# Grid styling
PLOT_GRID_ALPHA = 0.3
PLOT_GRID_LINEWIDTH = 0.5
PLOT_GRID_LINESTYLE = '--'

# =============================================================================
# TYPOGRAPHY
# =============================================================================

# Font families for plots
PLOT_FONT_FAMILY = 'sans-serif'
PLOT_FONTS = ['Inter', 'IBM Plex Sans', 'Segoe UI', 'Arial', 'sans-serif']

# Font weights
FONT_WEIGHT_NORMAL = 400
FONT_WEIGHT_BOLD = 700

# =============================================================================
# STYLE PRESETS
# =============================================================================

def get_matplotlib_style():
    """
    Get matplotlib style configuration dictionary.
    
    Returns:
        Dictionary of matplotlib rcParams settings
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> from utils.plotting.styles import get_matplotlib_style
        >>> plt.rcParams.update(get_matplotlib_style())
    """
    return {
        'figure.dpi': PLOT_DPI,
        'figure.figsize': (PLOT_WIDTH, PLOT_HEIGHT),
        'font.size': PLOT_FONT_SIZE,
        'font.family': PLOT_FONT_FAMILY,
        'axes.titlesize': PLOT_TITLE_SIZE,
        'axes.labelsize': PLOT_AXIS_LABEL_SIZE,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': PLOT_GRID_ALPHA,
        'grid.linewidth': PLOT_GRID_LINEWIDTH,
        'grid.linestyle': PLOT_GRID_LINESTYLE,
        'legend.fontsize': PLOT_LEGEND_SIZE,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'lines.linewidth': PLOT_LINE_WIDTH,
        'lines.markersize': PLOT_MARKER_SIZE,
        'savefig.dpi': PLOT_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    }


def get_color_for_motif(motif_class: str, use_nature_colors: bool = True) -> str:
    """
    Get color for a specific motif class.
    
    Args:
        motif_class: Name of the motif class
        use_nature_colors: If True, use NATURE_MOTIF_COLORS; otherwise use MOTIF_CLASS_COLORS
    
    Returns:
        Hex color string
    
    Example:
        >>> color = get_color_for_motif('G-Quadruplex')
        >>> print(color)
        '#0072B2'
    """
    colors = NATURE_MOTIF_COLORS if use_nature_colors else MOTIF_CLASS_COLORS
    return colors.get(motif_class, '#808080')  # Default to gray if not found
