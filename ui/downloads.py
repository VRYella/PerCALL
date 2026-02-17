"""UI Downloads Module
===================

Download and export functionality for UI.
Handles file generation and download buttons.

Extracted from app.py for modular architecture.
"""

import matplotlib.pyplot as plt
import numpy as np
import os  # Added for image path checking
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

def generate_excel_bytes(motifs, simple_format=True):
    """
    Generate Excel file as bytes for Streamlit download button.
    
    Args:
        motifs: List of motif dictionaries
        simple_format: If True, use 2-tab format (NonOverlappingConsolidated, OverlappingAll)
        
    Returns:
        bytes: Excel file data as bytes
    """
    import tempfile
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use the existing export_to_excel function to create the file
        export_to_excel(motifs, tmp_path, simple_format=simple_format)
        
        # Read the file as bytes
        with open(tmp_path, 'rb') as f:
            excel_bytes = f.read()
        
        return excel_bytes
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------- ENHANCED PROFESSIONAL CSS FOR RESEARCH-QUALITY UI ----------
# Note: !important declarations are necessary to override Streamlit's default high-specificity CSS

# Initialize theme state in session state
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'
if 'table_density' not in st.session_state:
    st.session_state.table_density = 'relaxed'
if 'color_theme' not in st.session_state:
    st.session_state.color_theme = 'scientific_blue'

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple for CSS rgba() usage."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_dna_pattern_svg(stroke_color: str) -> str:
    """Generate subtle DNA helix SVG pattern for background."""
    # Compact and URL-encoded SVG for background pattern
    svg = (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60' viewBox='0 0 60 60'%3E"
        f"%3Cg fill='none' stroke='%23{stroke_color}' stroke-width='0.6' opacity='0.12'%3E"
        "%3Cpath d='M10 30 C 18 12, 42 12, 50 30'/%3E"
        "%3Cpath d='M10 30 C 18 48, 42 48, 50 30'/%3E"
        "%3C/g%3E%3C/svg%3E"
    )
    return f"url(\"data:image/svg+xml,{svg}\")"

# Get current theme colors (using configuration from top)
# Note: These should be accessed when needed, not at module level
# current_theme = COLOR_THEMES.get(st.session_state.color_theme, COLOR_THEMES['scientific_blue'])
# is_dark_mode = st.session_state.theme_mode == 'dark'
# is_compact = st.session_state.table_density == 'compact'

# Dark mode color overrides - Soothing dark palette
# if is_dark_mode:
#     # Use the primary color from the base theme for consistency
#     base_primary = COLOR_THEMES.get(st.session_state.color_theme, COLOR_THEMES['scientific_blue'])['primary']
#     current_theme = {
#         **current_theme,
#         'bg_light': '#1A1F2E',        # Soft dark blue-gray
#         'bg_card': '#252B3B',          # Slightly lighter dark
#         'text': '#E5E7EB',             # Softer white for dark mode
#         'tab_bg': '#1F2937',           # Dark slate for tab bar
#         'tab_active': current_theme.get('primary', base_primary),
#         'shadow': 'rgba(0, 0, 0, 0.25)'
#     }
#
# # Pre-calculate RGB values for all theme colors (performance optimization)
# rgb = {key: hex_to_rgb(value) if key not in ['shadow'] else (0, 0, 0) for key, value in current_theme.items()}
#
# # Additional color calculations for new soothing theme
# tab_bg_color = current_theme.get('tab_bg', current_theme['bg_card'])
# tab_active_color = current_theme.get('tab_active', current_theme['primary'])
# shadow_color = current_theme.get('shadow', f"rgba({rgb['primary'][0]}, {rgb['primary'][1]}, {rgb['primary'][2]}, 0.15)")
#
# # Generate SVG pattern based on theme
# dna_pattern = get_dna_pattern_svg('1e3a5f' if is_dark_mode else 'bbdefb')

def load_css(theme_name=None):
    """
    Load external CSS file and inject dynamic theme variables.
    This makes the app.py file more succinct by separating styling concerns.
    If theme_name provided, use per-page theme, otherwise use session color_theme.
    """
    theme_to_use = COLOR_THEMES.get(theme_name, COLOR_THEMES.get(st.session_state.color_theme, COLOR_THEMES['scientific_blue']))
    is_dark = st.session_state.theme_mode == 'dark'
    
    # Read the external CSS file if present
    css_file_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    try:
        with open(css_file_path, 'r') as f:
            css_content = f.read()
    except Exception:
        css_content = ""
    
    # Compute some derived values
    p_rgb = hex_to_rgb(theme_to_use['primary'])
    s_rgb = hex_to_rgb(theme_to_use['secondary'])
    tab_bg_color_local = theme_to_use.get('tab_bg', theme_to_use['bg_card'])
    tab_active_color_local = theme_to_use.get('tab_active', theme_to_use['primary'])
    dna_svg_local = get_dna_pattern_svg('1e3a5f' if is_dark else 'bbdefb')
    
    # Inject centralized configuration into CSS variables
    theme_vars = f"""
    <style>
    :root {{
        /* Theme Colors */
        --primary-color: {theme_to_use['primary']};
        --secondary-color: {theme_to_use['secondary']};
        --accent-color: {theme_to_use['accent']};
        --bg-light: {theme_to_use['bg_light']};
        --bg-card: {theme_to_use['bg_card']};
        --text-color: {theme_to_use['text']};
        --tab-bg: {tab_bg_color_local};
        --tab-active: {tab_active_color_local};
        --shadow-color: {theme_to_use['shadow']};
        --primary-rgb: {p_rgb[0]}, {p_rgb[1]}, {p_rgb[2]};
        --secondary-rgb: {s_rgb[0]}, {s_rgb[1]}, {s_rgb[2]};
        
        /* Typography from FONT_CONFIG */
        --font-primary: {FONT_CONFIG['primary_font']};
        --font-monospace: {FONT_CONFIG['monospace_font']};
        --font-h1: {FONT_CONFIG['h1_size']};
        --font-h2: {FONT_CONFIG['h2_size']};
        --font-h3: {FONT_CONFIG['h3_size']};
        --font-h4: {FONT_CONFIG['h4_size']};
        --font-body: {FONT_CONFIG['body_size']};
        --font-small: {FONT_CONFIG['small_size']};
        --font-caption: {FONT_CONFIG['caption_size']};
        --font-weight-light: {FONT_CONFIG['light_weight']};
        --font-weight-normal: {FONT_CONFIG['normal_weight']};
        --font-weight-medium: {FONT_CONFIG['medium_weight']};
        --font-weight-semibold: {FONT_CONFIG['semibold_weight']};
        --font-weight-bold: {FONT_CONFIG['bold_weight']};
        --font-weight-extrabold: {FONT_CONFIG['extrabold_weight']};
        
        /* Layout from LAYOUT_CONFIG */
        --border-radius-sm: {LAYOUT_CONFIG['border_radius']['small']};
        --border-radius-md: {LAYOUT_CONFIG['border_radius']['medium']};
        --border-radius-lg: {LAYOUT_CONFIG['border_radius']['large']};
        --border-radius-pill: {LAYOUT_CONFIG['border_radius']['pill']};
        --spacing-xs: 0.25rem;
        --spacing-sm: {LAYOUT_CONFIG['padding']['small']};
        --spacing-md: {LAYOUT_CONFIG['padding']['medium']};
        --spacing-lg: {LAYOUT_CONFIG['padding']['large']};
        --spacing-xl: {LAYOUT_CONFIG['padding']['xlarge']};
        --spacing-2xl: 2.5rem;
        
        /* Transitions from ANIMATION_CONFIG */
        --transition-fast: {ANIMATION_CONFIG['transition_fast']} {ANIMATION_CONFIG['easing_smooth']};
        --transition-normal: {ANIMATION_CONFIG['transition_normal']} {ANIMATION_CONFIG['easing_smooth']};
        --transition-slow: {ANIMATION_CONFIG['transition_slow']} {ANIMATION_CONFIG['easing_smooth']};
        
        /* Theme State */
        --dark-mode: {1 if is_dark else 0};
        --dna-pattern: {dna_svg_local};
    }}
    {css_content}
    </style>
    """
    
