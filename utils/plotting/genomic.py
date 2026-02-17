"""Genomic Plots Module
====================

Genome-wide visualization functions.
Includes Manhattan plots, landscape tracks, and genomic overview visualizations.

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

def plot_manhattan_motif_density(motifs: List[Dict[str, Any]], 
                                  sequence_length: int,
                                  window_size: int = None,
                                  score_type: str = 'density',
                                  title: str = "Manhattan Plot - Motif Distribution",
                                  figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    Create Manhattan plot showing motif density or score across genomic coordinates.
    
    Ideal for highlighting hotspots, clusters, and hybrid zones in large genomes.
    Publication-quality visualization following Nature Methods guidelines.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        window_size: Window size for density calculation (auto if None)
        score_type: 'density' for motif count or 'score' for average score
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object (publication-ready at 300 DPI)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Auto-calculate window size (1% of sequence or minimum 1kb)
    if window_size is None:
        window_size = max(1000, sequence_length // 100)
    
    num_windows = max(1, sequence_length // window_size)
    
    # Get unique classes (exclude synthetic classes for cleaner visualization)
    classes = sorted(set(m.get('Class', 'Unknown') for m in motifs
                        if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES))
    
    if not classes:
        classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Calculate metric for each window and class
    positions_kb = []
    values = []
    colors = []
    
    for class_name in classes:
        class_motifs = [m for m in motifs if m.get('Class') == class_name]
        color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
        
        for i in range(num_windows):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            window_center_kb = (window_start + window_end) / 2 / 1000
            
            # Count motifs in this window
            window_motifs = []
            for motif in class_motifs:
                motif_start = motif.get('Start', 0) - 1
                motif_end = motif.get('End', 0)
                if not (motif_end <= window_start or motif_start >= window_end):
                    window_motifs.append(motif)
            
            if window_motifs:
                if score_type == 'density':
                    # Density: motifs per kb
                    value = len(window_motifs) / (window_size / 1000)
                else:  # score
                    # Average score in window
                    scores = [m.get('Score', 0) for m in window_motifs if isinstance(m.get('Score'), (int, float))]
                    value = np.mean(scores) if scores else 0
                
                positions_kb.append(window_center_kb)
                values.append(value)
                colors.append(color)
    
    # Plot points with class-specific colors
    ax.scatter(positions_kb, values, c=colors, s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
    
    # Styling
    ax.set_xlabel('Genomic Position (kb)', fontsize=12, fontweight='bold')
    y_label = 'Motif Density (motifs/kb)' if score_type == 'density' else 'Average Motif Score'
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    
    # Replace underscores with spaces in title
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    
    # Add horizontal grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Create legend with class names
    legend_elements = []
    display_classes = []
    for class_name in classes:
        if any(c == MOTIF_CLASS_COLORS.get(class_name, '#808080') for c in colors):
            legend_elements.append(plt.scatter([], [], c=MOTIF_CLASS_COLORS.get(class_name, '#808080'), 
                                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5))
            display_classes.append(class_name.replace('_', ' '))
    
    if legend_elements:
        ax.legend(legend_elements, display_classes, loc='upper right', 
                 fontsize=8, framealpha=0.9, ncol=min(3, len(legend_elements)))
    
    # Apply label suppression policy (Nature-ready: no individual labels)
    # Only annotate top 5% density hotspots if requested
    if values:
        threshold = np.percentile(values, 95)  # Top 5%
        hotspot_count = 0
        max_labels = 10  # Maximum labels per plot
        min_distance_kb = (sequence_length / 1000) / 20  # Minimum 5% sequence spacing
        
        last_label_pos = -float('inf')
        for pos, val, color in sorted(zip(positions_kb, values, colors), 
                                      key=lambda x: x[1], reverse=True):
            if val >= threshold and hotspot_count < max_labels:
                # Check distance from last label
                if pos - last_label_pos >= min_distance_kb:
                    # Subtle annotation (Nature-ready: minimal, professional)
                    ax.annotate(f'{val:.1f}', xy=(pos, val), 
                               xytext=(0, 5), textcoords='offset points',
                               fontsize=6, alpha=0.7, ha='center',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', 
                                       ec=color, lw=0.5, alpha=0.8))
                    last_label_pos = pos
                    hotspot_count += 1
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


def plot_cumulative_motif_distribution(motifs: List[Dict[str, Any]], 
                                       sequence_length: int,
                                       title: str = "Cumulative Motif Distribution",
                                       figsize: Tuple[int, int] = None,
                                       by_class: bool = True) -> plt.Figure:
    """
    Create cumulative distribution plot showing running sum of motifs over genome.
    
    Useful for comparing motifs or samples, showing how motif accumulation
    varies across the sequence. Publication-quality following Nature guidelines.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        title: Plot title
        figsize: Figure size (width, height)
        by_class: Whether to separate by motif class
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    if by_class:
        # Group by class
        classes = sorted(set(m.get('Class', 'Unknown') for m in motifs
                            if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES))
        
        if not classes:
            classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
        
        for class_name in classes:
            class_motifs = [m for m in motifs if m.get('Class') == class_name]
            
            # Sort by start position
            class_motifs_sorted = sorted(class_motifs, key=lambda m: m.get('Start', 0))
            
            # Calculate cumulative count
            positions = [0]
            cumulative = [0]
            
            for i, motif in enumerate(class_motifs_sorted, 1):
                positions.append(motif.get('Start', 0) / 1000)  # Convert to kb
                cumulative.append(i)
            
            # Add final point at sequence end
            positions.append(sequence_length / 1000)
            cumulative.append(len(class_motifs))
            
            # Plot with class color
            color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
            display_name = class_name.replace('_', ' ')
            ax.plot(positions, cumulative, color=color, linewidth=1.5, 
                   label=display_name, alpha=0.8)
    else:
        # Overall cumulative
        motifs_sorted = sorted(motifs, key=lambda m: m.get('Start', 0))
        
        positions = [0]
        cumulative = [0]
        
        for i, motif in enumerate(motifs_sorted, 1):
            positions.append(motif.get('Start', 0) / 1000)
            cumulative.append(i)
        
        positions.append(sequence_length / 1000)
        cumulative.append(len(motifs))
        
        ax.plot(positions, cumulative, color='#0072B2', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Genomic Position (kb)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Motif Count', fontsize=12, fontweight='bold')
    
    # Replace underscores in title
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    
    # Add grid
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    if by_class:
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


def plot_motif_cooccurrence_matrix(motifs: List[Dict[str, Any]], 
                                   title: str = "Motif Co-occurrence Matrix",
                                   figsize: Tuple[int, int] = None,
                                   overlap_threshold: int = 1) -> plt.Figure:
    """
    Create heatmap showing co-occurrence frequency between motif classes.
    
    Shows which motif classes tend to appear together (within overlap_threshold bp).
    Excellent for publication figures showing motif relationships.
    
    Args:
        motifs: List of motif dictionaries
        title: Plot title
        figsize: Figure size (width, height)
        overlap_threshold: Maximum distance (bp) to consider as co-occurrence
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['square']
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Get unique classes (exclude synthetic ones)
    classes = sorted(set(m.get('Class', 'Unknown') for m in motifs
                        if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES))
    
    if not classes:
        classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
    
    # Initialize co-occurrence matrix
    n_classes = len(classes)
    cooccurrence_matrix = np.zeros((n_classes, n_classes))
    
    # Calculate co-occurrences
    for i, class_i in enumerate(classes):
        motifs_i = [m for m in motifs if m.get('Class') == class_i]
        
        for j, class_j in enumerate(classes):
            motifs_j = [m for m in motifs if m.get('Class') == class_j]
            
            # Count overlaps
            count = 0
            for mi in motifs_i:
                start_i = mi.get('Start', 0)
                end_i = mi.get('End', 0)
                
                for mj in motifs_j:
                    start_j = mj.get('Start', 0)
                    end_j = mj.get('End', 0)
                    
                    # Check if they overlap or are within threshold
                    distance = max(0, max(start_i, start_j) - min(end_i, end_j))
                    
                    if distance <= overlap_threshold:
                        count += 1
            
            cooccurrence_matrix[i, j] = count
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Use colorblind-friendly colormap
    im = ax.imshow(cooccurrence_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    
    # Replace underscores with spaces in labels
    display_classes = [c.replace('_', ' ') for c in classes]
    ax.set_xticklabels(display_classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(display_classes, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Co-occurrence Count', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    # Add values to cells (if not too many)
    if n_classes <= 10:
        for i in range(n_classes):
            for j in range(n_classes):
                value = int(cooccurrence_matrix[i, j])
                if value > 0:
                    text_color = 'white' if value > cooccurrence_matrix.max() / 2 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center', 
                           color=text_color, fontsize=8, fontweight='bold')
    
    # Title and labels
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Motif Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Motif Class', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_gc_content_correlation(motifs: List[Dict[str, Any]], 
                                sequence: str,
                                window_size: int = 1000,
                                title: str = "Motif Density vs GC Content",
                                figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    Create scatter plot showing correlation between GC content and motif density.
    
    Shows GC-driven motif enrichment patterns. One dot per genomic window.
    Publication-quality visualization with regression line.
    
    Args:
        motifs: List of motif dictionaries
        sequence: DNA sequence string
        window_size: Window size for GC and density calculation
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['one_and_half']
    
    sequence_length = len(sequence)
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    num_windows = max(1, sequence_length // window_size)
    
    # Calculate GC% and motif density for each window
    gc_percentages = []
    motif_densities = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = min((i + 1) * window_size, sequence_length)
        
        # Calculate GC%
        window_seq = sequence[window_start:window_end].upper()
        gc_count = window_seq.count('G') + window_seq.count('C')
        window_length = len(window_seq)
        gc_pct = (gc_count / window_length * 100) if window_length > 0 else 0
        
        # Count motifs in window
        motif_count = 0
        for motif in motifs:
            motif_start = motif.get('Start', 0) - 1
            motif_end = motif.get('End', 0)
            if not (motif_end <= window_start or motif_start >= window_end):
                motif_count += 1
        
        # Density: motifs per kb
        density = motif_count / (window_size / 1000)
        
        gc_percentages.append(gc_pct)
        motif_densities.append(density)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    ax.scatter(gc_percentages, motif_densities, alpha=0.5, s=30, 
              color='#0072B2', edgecolors='black', linewidth=0.3)
    
    # Add regression line
    if len(gc_percentages) > 1:
        z = np.polyfit(gc_percentages, motif_densities, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(gc_percentages), max(gc_percentages), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, alpha=0.8, 
               label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(gc_percentages, motif_densities)[0, 1]
        ax.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Styling
    ax.set_xlabel('GC Content (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Motif Density (motifs/kb)', fontsize=12, fontweight='bold')
    
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    if len(gc_percentages) > 1:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


def plot_linear_motif_track(motifs: List[Dict[str, Any]], 
                            sequence_length: int,
                            region_start: int = 0,
                            region_end: int = None,
                            title: str = "Linear Motif Track",
                            figsize: Tuple[int, int] = None,
                            show_labels: bool = False) -> plt.Figure:  # Default to False per Nature-ready standards
    """
    Create horizontal graphical track with colored blocks for motifs.
    
    Best for visualizing <10kb regions. Colored blocks show motif positions
    with class-specific colors. Publication-quality linear genome browser view.
    
    NATURE-READY: Individual motif labels suppressed by default for clarity.
    Only class track labels are shown.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        region_start: Start position of region to display (0-based)
        region_end: End position of region to display (None = full sequence)
        title: Plot title
        figsize: Figure size (width, height)
        show_labels: Whether to show class labels (default False for cleaner view)
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if region_end is None:
        region_end = sequence_length
    
    # Filter motifs in region
    region_motifs = [m for m in motifs 
                     if m.get('End', 0) > region_start and m.get('Start', 0) < region_end]
    
    if not region_motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, f'No motifs in region {region_start}-{region_end}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Group by class
    class_motifs = defaultdict(list)
    for motif in region_motifs:
        class_name = motif.get('Class', 'Unknown')
        class_motifs[class_name].append(motif)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Plot each class on a separate track
    classes = sorted(class_motifs.keys())
    track_height = 0.6
    track_spacing = 1.0
    
    for i, class_name in enumerate(classes):
        y_pos = i * track_spacing
        color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
        
        for motif in class_motifs[class_name]:
            start = max(region_start, motif.get('Start', 0))
            end = min(region_end, motif.get('End', 0))
            length = end - start
            
            # Draw motif as rectangle
            rect = patches.Rectangle(
                (start, y_pos - track_height/2), length, track_height,
                facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8
            )
            ax.add_patch(rect)
            
            # NATURE-READY: No individual motif labels
            # Labels removed to prevent clutter and overlap per publication standards
        
        # Add class label on the left (always shown for track identification)
        display_name = class_name.replace('_', ' ')
        ax.text(region_start - (region_end - region_start) * 0.02, y_pos, 
               display_name, ha='right', va='center', fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_xlim(region_start, region_end)
    ax.set_ylim(-0.5, len(classes) * track_spacing - 0.5)
    
    ax.set_xlabel('Position (bp)', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    
    # Add position ruler at top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Format x-axis
    ax.ticklabel_format(style='plain', axis='x')
    
    # Apply Nature journal style (minimal spines)
    for spine in ['left', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['top'].set_visible(True)
    
    plt.tight_layout()
    return fig


def plot_cluster_size_distribution(motifs: List[Dict[str, Any]], 
                                   title: str = "Cluster Size Distribution",
                                   figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    Plot distribution of cluster sizes (number of motifs per cluster).
    
    Shows histogram and statistics of cluster composition.
    Publication-quality visualization.
    
    Args:
        motifs: List of motif dictionaries (should include cluster motifs)
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['one_and_half']
    
    # Extract cluster motifs
    cluster_motifs = [m for m in motifs if m.get('Class') == 'Non-B_DNA_Clusters']
    
    if not cluster_motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No cluster motifs found', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Extract cluster sizes
    cluster_sizes = []
    cluster_diversities = []
    
    for motif in cluster_motifs:
        size = motif.get('Motif_Count', 0)
        diversity = motif.get('Class_Diversity', 0)
        if size > 0:
            cluster_sizes.append(size)
            cluster_diversities.append(diversity)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=PUBLICATION_DPI)
    
    # 1. Cluster size histogram
    ax1.hist(cluster_sizes, bins=min(20, max(cluster_sizes) if cluster_sizes else 1), 
            edgecolor='black', linewidth=0.5, color='#4ecdc4', alpha=0.7)
    
    ax1.set_xlabel('Motifs per Cluster', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    
    # Add statistics
    if cluster_sizes:
        mean_size = np.mean(cluster_sizes)
        median_size = np.median(cluster_sizes)
        ax1.axvline(mean_size, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_size:.1f}')
        ax1.axvline(median_size, color='orange', linestyle='--', linewidth=1.5, 
                   label=f'Median: {median_size:.1f}')
        ax1.legend(fontsize=9, framealpha=0.9)
    
    ax1.grid(axis='y', alpha=0.3)
    _apply_nature_style(ax1)
    
    # 2. Class diversity histogram
    if cluster_diversities:
        ax2.hist(cluster_diversities, bins=min(10, max(cluster_diversities)), 
                edgecolor='black', linewidth=0.5, color='#95e1d3', alpha=0.7)
        
        ax2.set_xlabel('Class Diversity per Cluster', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Cluster Diversity Distribution', fontsize=12, fontweight='bold')
        
        mean_div = np.mean(cluster_diversities)
        ax2.axvline(mean_div, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {mean_div:.1f}')
        ax2.legend(fontsize=9, framealpha=0.9)
        
        ax2.grid(axis='y', alpha=0.3)
        _apply_nature_style(ax2)
    else:
        ax2.text(0.5, 0.5, 'No diversity data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.axis('off')
    
    display_title = title.replace('_', ' ')
    plt.suptitle(display_title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def plot_motif_length_kde(motifs: List[Dict[str, Any]], 
                          by_class: bool = True,
                          title: str = "Motif Length Distribution (KDE)",
                          figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    Plot kernel density estimation of motif length distributions.
    
    Shows smooth probability density curves for motif lengths,
    useful for comparing length patterns across classes.
    Publication-quality visualization.
    
    Args:
        motifs: List of motif dictionaries
        by_class: Whether to separate by motif class
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['one_and_half']
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    if by_class:
        # Group by class
        classes = sorted(set(m.get('Class', 'Unknown') for m in motifs
                            if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES))
        
        if not classes:
            classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
        
        for class_name in classes:
            class_motifs = [m for m in motifs if m.get('Class') == class_name]
            lengths = [m.get('Length', 0) for m in class_motifs if m.get('Length', 0) > 0]
            
            if len(lengths) > 1:
                color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
                display_name = class_name.replace('_', ' ')
                
                # Plot KDE
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(lengths)
                    x_range = np.linspace(min(lengths), max(lengths), 200)
                    density = kde(x_range)
                    ax.plot(x_range, density, color=color, linewidth=2, 
                           label=display_name, alpha=0.8)
                    ax.fill_between(x_range, density, alpha=0.2, color=color)
                except:
                    # Fallback to histogram if KDE fails
                    ax.hist(lengths, bins=20, alpha=0.3, color=color, 
                           label=display_name, density=True, edgecolor='black', linewidth=0.5)
    else:
        # Overall distribution
        lengths = [m.get('Length', 0) for m in motifs if m.get('Length', 0) > 0]
        
        if len(lengths) > 1:
            try:
                from scipy import stats
                kde = stats.gaussian_kde(lengths)
                x_range = np.linspace(min(lengths), max(lengths), 200)
                density = kde(x_range)
                ax.plot(x_range, density, color='#0072B2', linewidth=2.5, alpha=0.8)
                ax.fill_between(x_range, density, alpha=0.3, color='#0072B2')
            except:
                ax.hist(lengths, bins=30, alpha=0.7, color='#0072B2', 
                       density=True, edgecolor='black', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Motif Length (bp)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    if by_class:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


# =============================================================================
# FUNCTION ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

# Aliases for functions with different naming conventions
# plot_comprehensive_class_analysis = plot_class_analysis_comprehensive
# plot_comprehensive_subclass_analysis = plot_subclass_analysis_comprehensive


# =============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS FOR NOTEBOOK COMPATIBILITY
# =============================================================================

def plot_genome_landscape_track(motifs: List[Dict[str, Any]], 
                               sequence_length: int,
                               title: str = "Genome Landscape Track",
                               figsize: Tuple[int, int] = None,
                               window_size: int = None) -> plt.Figure:
    """
    Create genome landscape track visualization showing motif distribution along sequence.
    
    This is a simplified horizontal track view showing motif positions and density.
    Similar to plot_linear_motif_track but with additional density information.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        title: Plot title
        figsize: Figure size (width, height)
        window_size: Window size for density calculation (auto if None)
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Auto-calculate window size
    if window_size is None:
        window_size = max(1000, sequence_length // 50)
    
    # Create figure with two subplots: density track + motif positions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=PUBLICATION_DPI,
                                    gridspec_kw={'height_ratios': [1, 2]})
    
    # Top panel: Density line plot
    num_windows = max(1, sequence_length // window_size)
    positions = []
    densities = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        window_center = (window_start + window_end) / 2 / 1000  # In kb
        
        # Count motifs in window
        count = 0
        for motif in motifs:
            motif_start = motif.get('Start', 0) - 1
            motif_end = motif.get('End', 0)
            if not (motif_end <= window_start or motif_start >= window_end):
                count += 1
        
        density = count / (window_size / 1000)  # motifs per kb
        positions.append(window_center)
        densities.append(density)
    
    # Plot density line
    ax1.fill_between(positions, densities, alpha=0.3, color='#0072B2')
    ax1.plot(positions, densities, color='#0072B2', linewidth=1.5)
    ax1.set_ylabel('Density\n(motifs/kb)', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, sequence_length / 1000)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    _apply_nature_style(ax1)
    
    # Bottom panel: Motif track by class
    class_motifs = defaultdict(list)
    for motif in motifs:
        class_name = motif.get('Class', 'Unknown')
        if class_name not in CIRCOS_EXCLUDED_CLASSES:
            class_motifs[class_name].append(motif)
    
    if not class_motifs:
        # Include all classes if none pass the filter
        for motif in motifs:
            class_name = motif.get('Class', 'Unknown')
            class_motifs[class_name].append(motif)
    
    classes = sorted(class_motifs.keys())
    track_height = 0.5
    track_spacing = 1.0
    
    for i, class_name in enumerate(classes):
        y_pos = i * track_spacing
        color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
        
        for motif in class_motifs[class_name]:
            start_kb = motif.get('Start', 0) / 1000
            end_kb = motif.get('End', 0) / 1000
            length_kb = end_kb - start_kb
            
            # Draw motif as rectangle
            rect = patches.Rectangle(
                (start_kb, y_pos - track_height/2), length_kb, track_height,
                facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.8
            )
            ax2.add_patch(rect)
    
    # Customize bottom panel
    ax2.set_xlim(0, sequence_length / 1000)
    ax2.set_ylim(-0.5, len(classes) * track_spacing - 0.5)
    ax2.set_xlabel('Position (kb)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Motif Class', fontsize=11, fontweight='bold')
    ax2.set_yticks([i * track_spacing for i in range(len(classes))])
    display_classes = [c.replace('_', ' ') for c in classes]
    ax2.set_yticklabels(display_classes, fontsize=9)
    _apply_nature_style(ax2)
    
    # Overall title
    display_title = title.replace('_', ' ')
    fig.suptitle(display_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_sliding_window_heat_ribbon(motifs: List[Dict[str, Any]], 
                                    sequence_length: int,
                                    title: str = "Sliding Window Heat Ribbon",
                                    figsize: Tuple[int, int] = None,
                                    window_size: int = None) -> plt.Figure:
    """
    Create a 1D heatmap ribbon showing motif density along the sequence.
    
    This creates a horizontal ribbon colored by density values, with an
    accompanying line plot showing the density profile.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        title: Plot title
        figsize: Figure size (width, height)
        window_size: Window size for density calculation (auto if None)
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Auto-calculate window size
    if window_size is None:
        window_size = max(1000, sequence_length // 100)
    
    num_windows = max(1, sequence_length // window_size)
    
    # Calculate density for each window
    density_values = []
    positions = []
    
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        
        # Count motifs in window
        count = 0
        for motif in motifs:
            motif_start = motif.get('Start', 0) - 1
            motif_end = motif.get('End', 0)
            if not (motif_end <= window_start or motif_start >= window_end):
                count += 1
        
        density = count / (window_size / 1000)  # motifs per kb
        density_values.append(density)
        positions.append((window_start + window_end) / 2 / 1000)  # Center position in kb
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize, dpi=PUBLICATION_DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.3)
    
    # Top: Heat ribbon (1D heatmap)
    ax1 = fig.add_subplot(gs[0])
    
    # Create 2D array for heatmap (1 row)
    density_array = np.array(density_values).reshape(1, -1)
    
    # Plot heatmap
    im = ax1.imshow(density_array, cmap='YlOrRd', aspect='auto', 
                    extent=[0, sequence_length / 1000, 0, 1],
                    interpolation='bilinear')
    
    ax1.set_yticks([])
    ax1.set_ylabel('Density\nHeatmap', fontsize=10, fontweight='bold')
    ax1.set_xticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Motifs/kb', fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    # Bottom: Line plot
    ax2 = fig.add_subplot(gs[1])
    
    ax2.plot(positions, density_values, color='#0072B2', linewidth=2, alpha=0.8)
    ax2.fill_between(positions, density_values, alpha=0.3, color='#0072B2')
    
    # Mark peaks (top 10% density)
    if density_values:
        threshold = np.percentile(density_values, 90)
        peak_indices = [i for i, d in enumerate(density_values) if d >= threshold]
        if peak_indices:
            peak_positions = [positions[i] for i in peak_indices]
            peak_densities = [density_values[i] for i in peak_indices]
            ax2.scatter(peak_positions, peak_densities, color='red', s=50, 
                       zorder=5, alpha=0.7, edgecolors='black', linewidth=0.5,
                       label=f'High density (top 10%)')
    
    # Styling
    ax2.set_xlabel('Position (kb)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Motif Density (motifs/kb)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, sequence_length / 1000)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    if peak_indices:
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    _apply_nature_style(ax2)
    
    # Overall title
    display_title = title.replace('_', ' ')
    fig.suptitle(display_title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# JOB ID AND DOWNLOAD PACKAGE GENERATION
# =============================================================================

def generate_job_id(sequence_name: str) -> str:
    """
    Generate a unique, human-readable Job ID for a NonBDNA analysis run.
    
    Format: JOB_<sequence_name>_<short_hash>
    Example: JOB_TP53_chr17_f8a21c
    
    Args:
        sequence_name: Name of the sequence being analyzed
        
    Returns:
        Job ID string (safe for filenames)
    """
    # Clean sequence name for filename safety
    clean_name = re.sub(r'[^\w\-]', '_', sequence_name)[:MAX_SEQUENCE_NAME_LENGTH]
    clean_name = clean_name.strip('_')
    
    # Generate short hash from sequence name + timestamp for uniqueness
    hash_input = f"{sequence_name}_{time.time()}".encode('utf-8')
    short_hash = hashlib.sha256(hash_input).hexdigest()[:HASH_LENGTH]
    
    job_id = f"JOB_{clean_name}_{short_hash}"
    return job_id


def save_figures_to_pdf(figures: List[plt.Figure], output_path: str) -> None:
    """
    Save multiple matplotlib figures to a single multi-page PDF.
    
    Args:
        figures: List of matplotlib figure objects
        output_path: Path to output PDF file
    """
    with PdfPages(output_path) as pdf:
        for fig in figures:
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
                plt.close(fig)


def create_consolidated_pdf(
    motifs: List[Dict[str, Any]],
    sequence_length: int,
    sequence_name: str,
    job_id: str
) -> None:
    """
    Create a consolidated PDF report with all visualizations.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Length of the analyzed sequence
        sequence_name: Name of the sequence
        job_id: Job identifier
    """
    # Placeholder function - to be implemented
    pass
