"""Density Plots Module
===================

Visualization functions for motif density analysis.
Includes heatmaps, circos plots, and density comparisons.

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

def plot_density_comparison(genomic_density: Dict[str, float],
                            positional_density: Dict[str, float],
                            title: str = "Motif Density Analysis",
                            figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot comparison of genomic density (coverage %) and positional density (motifs/kbp).
    
    Args:
        genomic_density: Dictionary of class -> genomic density (%)
        positional_density: Dictionary of class -> positional density (motifs/unit)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    # Remove 'Overall' for class-specific comparison
    classes = [k for k in genomic_density.keys() if k != 'Overall']
    if not classes:
        classes = list(genomic_density.keys())
    
    # Sort classes alphabetically
    classes = sorted(classes)
    
    genomic_vals = [genomic_density.get(c, 0) for c in classes]
    positional_vals = [positional_density.get(c, 0) for c in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Genomic Density (Coverage %)
    colors1 = [MOTIF_CLASS_COLORS.get(c, '#808080') for c in classes]
    
    # Replace underscores with spaces in labels
    display_classes = [c.replace('_', ' ') for c in classes]
    
    bars1 = ax1.barh(display_classes, genomic_vals, color=colors1, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Genomic Density (Coverage %)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Motif Class', fontsize=11, fontweight='bold')
    ax1.set_title('A. Genomic Density (σ_G)', fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels with safe max calculation
    max_genomic = max(genomic_vals) if genomic_vals and max(genomic_vals) > 0 else 1
    for i, (bar, val) in enumerate(zip(bars1, genomic_vals)):
        if val > 0:
            ax1.text(val + max_genomic * 0.01, i, f'{val:.3f}%', 
                    va='center', fontsize=9, fontweight='bold')
    
    # Positional Density (Frequency)
    colors2 = [MOTIF_CLASS_COLORS.get(c, '#808080') for c in classes]
    bars2 = ax2.barh(display_classes, positional_vals, color=colors2, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Positional Density (motifs/kbp)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Motif Class', fontsize=11, fontweight='bold')
    ax2.set_title('B. Positional Density (λ)', fontsize=12, fontweight='bold', pad=10)
    
    # Add value labels with safe max calculation
    max_positional = max(positional_vals) if positional_vals and max(positional_vals) > 0 else 1
    for i, (bar, val) in enumerate(zip(bars2, positional_vals)):
        if val > 0:
            ax2.text(val + max_positional * 0.01, i, f'{val:.2f}', 
                    va='center', fontsize=9, fontweight='bold')
    
    # Apply Nature journal style
    _apply_nature_style(ax1)
    _apply_nature_style(ax2)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def plot_enrichment_analysis(enrichment_results: Dict[str, Dict[str, Any]],
                             title: str = "Motif Enrichment Analysis",
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot enrichment analysis results with fold enrichment and p-values.
    
    Note: Enrichment analysis has been removed for performance.
    This function is kept for backward compatibility with legacy data.
    
    Args:
        enrichment_results: Dictionary with enrichment metrics
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    # Extract data (exclude 'Overall' for class-specific view)
    classes = [k for k in enrichment_results.keys() if k != 'Overall']
    if not classes:
        classes = list(enrichment_results.keys())
    
    fold_enrichments = []
    p_values = []
    observed_densities = []
    background_means = []
    
    for cls in classes:
        result = enrichment_results[cls]
        fe = result.get('fold_enrichment', 0)
        # Handle both string 'Inf' and float infinity values robustly
        if fe == 'Inf' or (isinstance(fe, float) and np.isinf(fe)):
            fe = INFINITE_FOLD_ENRICHMENT_CAP  # Cap infinite values for visualization
        fold_enrichments.append(fe)
        p_values.append(result.get('p_value', 1.0))
        observed_densities.append(result.get('observed_density', 0))
        background_means.append(result.get('background_mean', 0))
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # 1. Fold Enrichment
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [MOTIF_CLASS_COLORS.get(c, '#808080') for c in classes]
    bars1 = ax1.barh(classes, fold_enrichments, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No enrichment (FE=1)')
    ax1.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Motif Class', fontsize=11, fontweight='bold')
    ax1.set_title('A. Fold Enrichment', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, fold_enrichments)):
        label_text = f'{val:.2f}' if val < INFINITE_FOLD_ENRICHMENT_CAP else 'Inf'
        ax1.text(val + max(fold_enrichments) * 0.01, i, label_text, 
                va='center', fontsize=9, fontweight='bold')
    
    # 2. P-values
    ax2 = fig.add_subplot(gs[0, 1])
    # Color code by significance
    p_colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
    bars2 = ax2.barh(classes, p_values, color=p_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='p=0.05')
    ax2.axvline(x=0.1, color='orange', linestyle='--', linewidth=1.5, label='p=0.1')
    ax2.set_xlabel('P-value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Motif Class', fontsize=11, fontweight='bold')
    ax2.set_title('B. Statistical Significance', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(1.0, max(p_values) * 1.1))
    ax2.legend(loc='best', fontsize=9)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, p_values)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 3. Observed vs Background Density
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(classes))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, observed_densities, width, label='Observed', 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3b = ax3.bar(x + width/2, background_means, width, label='Background (Mean)', 
                    color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Motif Class', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density (%)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Observed vs. Background Density Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    
    return fig


def plot_enrichment_summary_table(enrichment_results: Dict[str, Dict[str, Any]],
                                  title: str = "Enrichment Summary Statistics") -> plt.Figure:
    """
    Create a summary table visualization for enrichment results.
    
    Note: Enrichment analysis has been removed for performance.
    This function is kept for backward compatibility with legacy data.
    
    Args:
        enrichment_results: Dictionary with enrichment metrics
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    # Prepare data for table
    classes = [k for k in enrichment_results.keys() if k != 'Overall']
    if not classes:
        return None
    
    table_data = []
    for cls in classes:
        result = enrichment_results[cls]
        fe = result.get('fold_enrichment', 0)
        fe_str = f"{fe:.2f}" if fe != 'Inf' else 'Inf'
        
        row = [
            cls,
            result.get('observed_count', 0),
            f"{result.get('observed_density', 0):.4f}%",
            f"{result.get('background_mean', 0):.4f}%",
            fe_str,
            f"{result.get('p_value', 1.0):.4f}",
            '***' if result.get('p_value', 1.0) < 0.001 else 
            '**' if result.get('p_value', 1.0) < 0.01 else 
            '*' if result.get('p_value', 1.0) < 0.05 else 'ns'
        ]
        table_data.append(row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, len(classes) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    headers = ['Class', 'Count', 'Observed\nDensity', 'Background\nMean', 
               'Fold\nEnrichment', 'P-value', 'Sig.']
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.20, 0.10, 0.15, 0.15, 0.15, 0.12, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i, header in enumerate(headers):
        cell = table[(0, i)]
        cell.set_facecolor('#2196F3')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('white')
            
            # Highlight significant results
            if j == 6:  # Significance column
                if table_data[i-1][j] in ['***', '**', '*']:
                    cell.set_facecolor('#C8E6C9')
                    cell.set_text_props(weight='bold', color='green')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


# =============================================================================
# CIRCOS PLOT FOR NON-B DNA MOTIF DENSITY
# =============================================================================

# Motif classes to exclude from Circos visualization (dynamic classes)
CIRCOS_EXCLUDED_CLASSES = ['Hybrid', 'Non-B_DNA_Clusters']


def plot_circos_motif_density(motifs: List[Dict[str, Any]], 
                               sequence_length: int,
                               title: str = "Non-B DNA Motif Density Circos Plot",
                               figsize: Tuple[int, int] = (12, 12),
                               window_size: int = None) -> plt.Figure:
    """
    Create a circular Circos-style plot showing non-B DNA motif class density.
    
    The plot shows:
    - Outer ring: Sequence position ruler
    - Inner rings: One ring per motif class showing density
    - Center: Summary statistics
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        title: Plot title
        figsize: Figure size (width, height)
        window_size: Window size for density calculation (auto-calculated if None)
        
    Returns:
        Matplotlib figure object with Circos-style visualization
    """
    set_scientific_style()
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Auto-calculate window size if not provided
    if window_size is None:
        window_size = max(100, sequence_length // 50)
    
    # Calculate number of windows
    num_windows = max(1, sequence_length // window_size)
    
    # Get unique classes (excluding Hybrid and Clusters for cleaner visualization)
    classes = sorted(set(m.get('Class', 'Unknown') for m in motifs 
                        if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES))
    
    if not classes:
        classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
    
    # Calculate density per window per class
    class_densities = {}
    for class_name in classes:
        densities = []
        class_motifs = [m for m in motifs if m.get('Class') == class_name]
        
        for i in range(num_windows):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            
            # Count motifs in this window
            count = 0
            for motif in class_motifs:
                motif_start = motif.get('Start', 0) - 1  # 0-based
                motif_end = motif.get('End', 0)
                # Check overlap with window
                if not (motif_end <= window_start or motif_start >= window_end):
                    count += 1
            
            # Convert to density (motifs per kb)
            window_kb = window_size / 1000
            densities.append(count / window_kb if window_kb > 0 else 0)
        
        class_densities[class_name] = densities
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Calculate angles for each window
    theta = np.linspace(0, 2 * np.pi, num_windows, endpoint=False)
    
    # Width of each bar
    width = 2 * np.pi / num_windows * 0.8
    
    # Ring configuration
    ring_width = 0.12
    inner_radius = 0.3
    
    # Plot each class as a ring
    for i, class_name in enumerate(classes):
        densities = class_densities[class_name]
        
        # Normalize densities for this class (0-1 scale for bar height)
        max_density = max(densities) if max(densities) > 0 else 1
        normalized = [d / max_density for d in densities]
        
        # Calculate ring position
        ring_bottom = inner_radius + i * ring_width
        
        # Get color for this class
        color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
        
        # Plot bars for this ring
        heights = [n * ring_width * 0.9 for n in normalized]
        # Replace underscores with spaces in legend label
        display_name = class_name.replace('_', ' ')
        bars = ax.bar(theta, heights, width=width, bottom=ring_bottom,
                     color=color, alpha=0.7, edgecolor='white', linewidth=0.5,
                     label=f'{display_name} (max: {max_density:.1f}/kb)')
    
    # Add outer position ruler
    outer_radius = inner_radius + len(classes) * ring_width + 0.05
    ruler_theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ruler_labels = [f'{int(i * sequence_length / 12 / 1000)}kb' for i in range(12)]
    
    ax.set_xticks(ruler_theta)
    ax.set_xticklabels(ruler_labels, fontsize=9, fontweight='bold')
    
    # Remove radial labels
    ax.set_yticklabels([])
    
    # Set limits
    ax.set_ylim(0, outer_radius + 0.1)
    
    # Add legend
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=8, 
             framealpha=0.9, ncol=1)
    
    # Add title (replace underscores with spaces)
    display_title = title.replace('_', ' ')
    fig.suptitle(display_title, fontsize=14, fontweight='bold', y=0.98)
    
    # Add center statistics
    total_motifs = len([m for m in motifs if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES])
    center_text = f"Total: {total_motifs}\n{len(classes)} classes\n{sequence_length/1000:.1f} kb"
    ax.text(0, 0, center_text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_radial_class_density(motifs: List[Dict[str, Any]], 
                               sequence_length: int,
                               title: str = "Radial Motif Class Density",
                               figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    Create a radial bar chart showing motif density per class.
    
    A simpler alternative to full Circos plot, showing aggregate density
    per motif class in a radial/polar layout.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Calculate density per class (motifs per kb)
    sequence_kb = sequence_length / 1000
    class_counts = Counter(m.get('Class', 'Unknown') for m in motifs
                          if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES)
    
    if not class_counts:
        class_counts = Counter(m.get('Class', 'Unknown') for m in motifs)
    
    classes = list(class_counts.keys())
    densities = [class_counts[c] / sequence_kb for c in classes]
    
    # Create polar plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Calculate angles
    num_classes = len(classes)
    theta = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    width = 2 * np.pi / num_classes * 0.7
    
    # Get colors
    colors = [MOTIF_CLASS_COLORS.get(c, '#808080') for c in classes]
    
    # Plot bars
    bars = ax.bar(theta, densities, width=width, color=colors, alpha=0.8,
                 edgecolor='white', linewidth=2)
    
    # Add class labels (replace underscores with spaces)
    ax.set_xticks(theta)
    display_classes = [cls.replace('_', ' ') for cls in classes]
    ax.set_xticklabels(display_classes, fontsize=10, fontweight='bold')
    
    # Add value labels on bars
    for angle, density, bar in zip(theta, densities, bars):
        ax.text(angle, density + max(densities) * 0.05, f'{density:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Style (replace underscores with spaces in title)
    ax.set_ylabel('Density (motifs/kb)', labelpad=30, fontsize=11, fontweight='bold')
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def plot_stacked_density_track(motifs: List[Dict[str, Any]], 
                                sequence_length: int,
                                title: str = "Stacked Motif Density Track",
                                figsize: Tuple[int, int] = (14, 6),
                                window_size: int = None) -> plt.Figure:
    """
    Create a stacked area chart showing motif density along the sequence.
    
    A linear (non-circular) alternative to Circos plot showing how different
    motif classes are distributed along the sequence.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Total length of analyzed sequence in bp
        title: Plot title
        figsize: Figure size
        window_size: Window size for density calculation (auto if None)
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Auto-calculate window size
    if window_size is None:
        window_size = max(100, sequence_length // 100)
    
    num_windows = max(1, sequence_length // window_size)
    
    # Get unique classes
    classes = sorted(set(m.get('Class', 'Unknown') for m in motifs
                        if m.get('Class') not in CIRCOS_EXCLUDED_CLASSES))
    
    if not classes:
        classes = sorted(set(m.get('Class', 'Unknown') for m in motifs))
    
    # Calculate density per window per class
    positions = np.arange(num_windows) * window_size / 1000  # In kb
    class_densities = {}
    
    for class_name in classes:
        densities = []
        class_motifs = [m for m in motifs if m.get('Class') == class_name]
        
        for i in range(num_windows):
            window_start = i * window_size
            window_end = (i + 1) * window_size
            
            count = 0
            for motif in class_motifs:
                motif_start = motif.get('Start', 0) - 1
                motif_end = motif.get('End', 0)
                if not (motif_end <= window_start or motif_start >= window_end):
                    count += 1
            
            densities.append(count)
        
        class_densities[class_name] = densities
    
    # Create stacked area plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Stack the densities
    colors = [MOTIF_CLASS_COLORS.get(c, '#808080') for c in classes]
    
    # Create arrays for stacking (replace underscores with spaces in labels)
    density_arrays = [np.array(class_densities[c]) for c in classes]
    display_classes = [cls.replace('_', ' ') for cls in classes]
    
    ax.stackplot(positions, *density_arrays, labels=display_classes, colors=colors, alpha=0.8)
    
    # Styling (replace underscores with spaces in title)
    ax.set_xlabel('Position (kb)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Motif Count per Window', fontsize=12, fontweight='bold')
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SUBCLASS-LEVEL DENSITY AND ENRICHMENT VISUALIZATIONS
# =============================================================================

def plot_density_comparison_by_subclass(genomic_density: Dict[str, float],
                                        positional_density: Dict[str, float],
                                        title: str = "Motif Density Analysis (by Subclass)",
                                        figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Plot comparison of genomic density (coverage %) and positional density (motifs/kbp)
    at the subclass level.
    
    Args:
        genomic_density: Dictionary of 'Class:Subclass' -> genomic density (%)
        positional_density: Dictionary of 'Class:Subclass' -> positional density (motifs/unit)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    # Remove 'Overall' for subclass-specific comparison
    subclasses = [k for k in genomic_density.keys() if k != 'Overall' and ':' in k]
    if not subclasses:
        # Fallback to regular class-level if no subclass data
        subclasses = [k for k in genomic_density.keys() if k != 'Overall']
    
    # Sort by class, then by subclass
    subclasses.sort()
    
    genomic_vals = [genomic_density.get(c, 0) for c in subclasses]
    positional_vals = [positional_density.get(c, 0) for c in subclasses]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Genomic Density (Coverage %)
    y_pos = np.arange(len(subclasses))
    
    # Get colors based on parent class
    colors1 = []
    for subclass in subclasses:
        if ':' in subclass:
            parent_class = subclass.split(':')[0]
            colors1.append(MOTIF_CLASS_COLORS.get(parent_class, '#808080'))
        else:
            colors1.append(MOTIF_CLASS_COLORS.get(subclass, '#808080'))
    
    bars1 = ax1.barh(y_pos, genomic_vals, color=colors1, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Genomic Density (Coverage %)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Motif Subclass', fontsize=11, fontweight='bold')
    ax1.set_title('A. Genomic Density (σ_G) by Subclass', fontsize=12, fontweight='bold', pad=10)
    ax1.set_yticks(y_pos)
    
    # Replace underscores and format labels
    display_labels = [label.replace('_', ' ').replace(':', ': ') for label in subclasses]
    ax1.set_yticklabels(display_labels, fontsize=8)
    
    # Add value labels
    max_val = max(genomic_vals) if genomic_vals else 1
    for i, (bar, val) in enumerate(zip(bars1, genomic_vals)):
        if val > 0:
            ax1.text(val + max_val * 0.01, i, f'{val:.3f}%', 
                    va='center', fontsize=7, fontweight='bold')
    
    # Positional Density (Frequency)
    bars2 = ax2.barh(y_pos, positional_vals, color=colors1, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Positional Density (motifs/kbp)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Motif Subclass', fontsize=11, fontweight='bold')
    ax2.set_title('B. Positional Density (λ) by Subclass', fontsize=12, fontweight='bold', pad=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_labels, fontsize=8)
    
    # Add value labels
    max_val2 = max(positional_vals) if positional_vals else 1
    for i, (bar, val) in enumerate(zip(bars2, positional_vals)):
        if val > 0:
            ax2.text(val + max_val2 * 0.01, i, f'{val:.2f}', 
                    va='center', fontsize=7, fontweight='bold')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def plot_enrichment_analysis_by_subclass(enrichment_results: Dict[str, Dict[str, Any]],
                                         title: str = "Motif Enrichment Analysis (by Subclass)",
                                         figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Plot enrichment analysis results at subclass level with fold enrichment and p-values.
    
    Note: Enrichment analysis has been removed for performance.
    This function is kept for backward compatibility with legacy data.
    
    Args:
        enrichment_results: Dictionary with enrichment metrics
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    set_scientific_style()
    
    # Extract data (exclude 'Overall' for subclass-specific view)
    subclasses = [k for k in enrichment_results.keys() if k != 'Overall' and ':' in k]
    if not subclasses:
        # Fallback to class level
        subclasses = [k for k in enrichment_results.keys() if k != 'Overall']
    
    # Sort alphabetically
    subclasses.sort()
    
    fold_enrichments = []
    p_values = []
    observed_densities = []
    background_means = []
    
    for subclass in subclasses:
        result = enrichment_results[subclass]
        fe = result.get('fold_enrichment', 0)
        # Handle both string 'Inf' and float infinity values robustly
        if fe == 'Inf' or (isinstance(fe, float) and np.isinf(fe)):
            fe = INFINITE_FOLD_ENRICHMENT_CAP  # Cap infinite values for visualization
        fold_enrichments.append(fe)
        p_values.append(result.get('p_value', 1.0))
        observed_densities.append(result.get('observed_density', 0))
        background_means.append(result.get('background_mean', 0))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[2, 2, 1.5])
    
    y_pos = np.arange(len(subclasses))
    
    # Get colors based on parent class
    colors = []
    for subclass in subclasses:
        if ':' in subclass:
            parent_class = subclass.split(':')[0]
            colors.append(MOTIF_CLASS_COLORS.get(parent_class, '#808080'))
        else:
            colors.append(MOTIF_CLASS_COLORS.get(subclass, '#808080'))
    
    # 1. Fold Enrichment
    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.barh(y_pos, fold_enrichments, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No enrichment (FE=1)')
    ax1.set_xlabel('Fold Enrichment', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Motif Subclass', fontsize=11, fontweight='bold')
    ax1.set_title('A. Fold Enrichment by Subclass', fontsize=12, fontweight='bold')
    ax1.set_yticks(y_pos)
    
    # Format labels
    display_labels = [label.replace('_', ' ').replace(':', ': ') for label in subclasses]
    ax1.set_yticklabels(display_labels, fontsize=8)
    ax1.legend(loc='best', fontsize=9)
    
    # Add value labels
    max_fe = max(fold_enrichments) if fold_enrichments else 1
    for i, (bar, val) in enumerate(zip(bars1, fold_enrichments)):
        label_text = f'{val:.2f}' if val < INFINITE_FOLD_ENRICHMENT_CAP else 'Inf'
        ax1.text(val + max_fe * 0.01, i, label_text, 
                va='center', fontsize=7, fontweight='bold')
    
    # 2. P-values
    ax2 = fig.add_subplot(gs[1])
    # Color code by significance
    p_colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
    bars2 = ax2.barh(y_pos, p_values, color=p_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='p=0.05')
    ax2.axvline(x=0.1, color='orange', linestyle='--', linewidth=1.5, label='p=0.1')
    ax2.set_xlabel('P-value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Motif Subclass', fontsize=11, fontweight='bold')
    ax2.set_title('B. Statistical Significance by Subclass', fontsize=12, fontweight='bold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_labels, fontsize=8)
    ax2.set_xlim(0, max(1.0, max(p_values) * 1.1))
    ax2.legend(loc='best', fontsize=9)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, p_values)):
        ax2.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=7, fontweight='bold')
    
    # 3. Observed vs Background Density (grouped bar chart)
    ax3 = fig.add_subplot(gs[2])
    x = np.arange(len(subclasses))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, observed_densities, width, label='Observed', 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3b = ax3.bar(x + width/2, background_means, width, label='Background (Mean)', 
                    color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Motif Subclass', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density (%)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Observed vs. Background Density Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=7)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    return fig


def plot_subclass_density_heatmap(motifs: List[Dict[str, Any]], 
                                  sequence_length: int,
                                  window_size: int = 1000,
                                  title: str = "Subclass Density Heatmap",
                                  figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create a heatmap showing density of each subclass across the sequence.
    
    Args:
        motifs: List of motif dictionaries
        sequence_length: Length of the analyzed sequence
        window_size: Window size for density calculation
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style()
    
    if not motifs or sequence_length == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    # Calculate windows
    num_windows = max(1, sequence_length // window_size)
    windows = np.linspace(0, sequence_length, num_windows + 1)
    
    # Get unique subclasses
    subclass_groups = defaultdict(list)
    for motif in motifs:
        class_name = motif.get('Class', 'Unknown')
        subclass_name = motif.get('Subclass', 'Unknown')
        key = f"{class_name}:{subclass_name}"
        subclass_groups[key].append(motif)
    
    subclasses = sorted(subclass_groups.keys())
    
    # Calculate density matrix
    density_matrix = np.zeros((len(subclasses), num_windows))
    
    for i, subclass_key in enumerate(subclasses):
        subclass_motifs = subclass_groups[subclass_key]
        
        for j in range(num_windows):
            window_start = windows[j]
            window_end = windows[j + 1]
            
            # Count motifs in window
            count = 0
            for motif in subclass_motifs:
                motif_start = motif.get('Start', 0) - 1  # 0-based
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
    ax.set_xlabel(f'Position (kb)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Motif Subclass', fontsize=12, fontweight='bold')
    
    # Replace underscores with spaces in title
    display_title = title.replace('_', ' ')
    ax.set_title(display_title, fontsize=14, fontweight='bold', pad=10)
    
    # Set ticks and labels with underscores replaced by spaces
    ax.set_yticks(range(len(subclasses)))
    display_subclasses = [sc.replace('_', ' ').replace(':', ': ') for sc in subclasses]
    ax.set_yticklabels(display_subclasses, fontsize=8)
    
    # Clean x-axis with kb units
    x_ticks = np.arange(0, num_windows, max(1, num_windows // 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(windows[i]/1000)}' for i in x_ticks])
    
    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Motif Count', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig


# =============================================================================
# NATURE-LEVEL PUBLICATION VISUALIZATIONS (GENOME-WIDE)
# =============================================================================

