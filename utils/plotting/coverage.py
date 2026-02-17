"""Coverage Maps Module
===================

Visualization functions for motif coverage and spatial distribution.
Creates coverage maps and position-based visualizations.

Extracted from utilities.py for modular architecture.
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple, Union
import os

def plot_coverage_map(motifs: List[Dict[str, Any]], 
                     sequence_length: int,
                     title: Optional[str] = None,
                     figsize: Tuple[float, float] = None,
                     style: str = 'nature') -> plt.Figure:
    """
    Plot motif coverage map showing positions along sequence.
    
    Publication-quality genomic track visualization.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Length of the analyzed sequence
        title: Custom plot title
        figsize: Figure size (width, height) in inches
        style: Style preset ('nature', 'default', 'presentation')
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style(style)
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            ax.set_title(title)
        return fig
    
    # Group motifs by class
    class_motifs = defaultdict(list)
    for motif in motifs:
        class_name = motif.get('Class', 'Unknown')
        class_motifs[class_name].append(motif)
    
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    y_pos = 0
    class_positions = {}
    
    for class_name, class_motif_list in class_motifs.items():
        class_positions[class_name] = y_pos
        color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
        
        for motif in class_motif_list:
            start = motif.get('Start', 0) - 1  # Convert to 0-based
            end = motif.get('End', start + 1)
            length = end - start
            
            # Draw motif as rectangle (Nature style - clean edges)
            rect = patches.Rectangle(
                (start, y_pos - 0.35), length, 0.7,
                facecolor=color, edgecolor='black', linewidth=0.3
            )
            ax.add_patch(rect)
        
        y_pos += 1
    
    # Customize axes (Nature style)
    ax.set_xlim(0, sequence_length)
    ax.set_ylim(-0.5, len(class_motifs) - 0.5)
    ax.set_xlabel('Position (bp)')
    ax.set_ylabel('Motif Class')
    if title:
        # Replace underscores with spaces in title
        display_title = title.replace('_', ' ')
        ax.set_title(display_title, fontweight='bold', pad=10)
    
    # Set y-axis labels with underscores replaced by spaces
    ax.set_yticks(list(class_positions.values()))
    display_labels = [label.replace('_', ' ') for label in class_positions.keys()]
    ax.set_yticklabels(display_labels)
    
    # Clean x-axis ticks
    ax.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


def plot_density_heatmap(motifs: List[Dict[str, Any]], 
                        sequence_length: int,
                        window_size: int = 1000,
                        title: Optional[str] = None,
                        figsize: Tuple[float, float] = None,
                        style: str = 'nature') -> plt.Figure:
    """
    Plot motif density heatmap along sequence.
    
    Publication-quality density visualization.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Length of the analyzed sequence
        window_size: Window size for density calculation
        title: Custom plot title
        figsize: Figure size (width, height) in inches
        style: Style preset
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style(style)
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            ax.set_title(title)
        return fig
    
    # Calculate windows
    num_windows = max(1, sequence_length // window_size)
    windows = np.linspace(0, sequence_length, num_windows + 1)
    
    # Get unique classes
    classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
    
    # Calculate density matrix
    density_matrix = np.zeros((len(classes), num_windows))
    
    for i, class_name in enumerate(classes):
        class_motifs = [m for m in motifs if m.get('Class') == class_name]
        
        for j in range(num_windows):
            window_start = windows[j]
            window_end = windows[j + 1]
            
            # Count motifs in window
            count = 0
            for motif in class_motifs:
                motif_start = motif.get('Start', 0)
                motif_end = motif.get('End', 0)
                
                # Check if motif overlaps with window
                if not (motif_end <= window_start or motif_start >= window_end):
                    count += 1
            
            density_matrix[i, j] = count
    
    # Create heatmap with publication quality
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Use colorblind-friendly colormap
    im = ax.imshow(density_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Customize axes (Nature style)
    ax.set_xlabel(f'Position (kb)')
    ax.set_ylabel('Motif Class')
    if title:
        # Replace underscores with spaces in title
        display_title = title.replace('_', ' ')
        ax.set_title(display_title, fontweight='bold', pad=10)
    
    # Set ticks and labels with underscores replaced by spaces
    ax.set_yticks(range(len(classes)))
    display_classes = [cls.replace('_', ' ') for cls in classes]
    ax.set_yticklabels(display_classes)
    
    # Clean x-axis with kb units
    x_ticks = np.arange(0, num_windows, max(1, num_windows // 5))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(windows[i]/1000)}' for i in x_ticks])
    
    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    return fig


# =============================================================================
# STATISTICAL PLOTS (Nature-Level Quality)
# =============================================================================
