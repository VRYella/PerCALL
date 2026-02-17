"""
UI Module - User Interface Components
=====================================

This module contains all UI-related components including:
- Layout elements and page structure
- Formatting utilities
- Metrics display components
- Progress indicators
- Input widgets
- Download buttons
"""

__version__ = "2025.1"

# Import UI modules
from . import formatting
from . import downloads
from . import layout
from . import metrics
from . import progress
from . import inputs

# Export key functions for direct access
from .formatting import format_time_scientific, format_time_compact, format_time
from .downloads import generate_excel_bytes, load_css
from .layout import (
    configure_page, create_header, create_tabs, 
    create_expander, add_vertical_space
)
from .metrics import display_metric_card, display_metrics_row, display_summary_stats
from .progress import (
    display_progress_bar, create_progress_container, 
    update_progress, create_simple_progress
)

__all__ = [
    # Modules
    'formatting',
    'downloads',
    'layout',
    'metrics',
    'progress',
    'inputs',
    # Formatting functions
    'format_time_scientific',
    'format_time_compact',
    'format_time',
    # Download functions
    'generate_excel_bytes',
    'load_css',
    # Layout functions
    'configure_page',
    'create_header',
    'create_tabs',
    'create_expander',
    'add_vertical_space',
    # Metrics functions
    'display_metric_card',
    'display_metrics_row',
    'display_summary_stats',
    # Progress functions
    'display_progress_bar',
    'create_progress_container',
    'update_progress',
    'create_simple_progress',
]
