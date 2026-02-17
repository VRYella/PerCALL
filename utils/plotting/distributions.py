"""Distribution Plots Module
=========================

Visualization functions for motif distributions, scores, and lengths.
Includes bar charts, histograms, and comparative visualizations.

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

def plot_motif_distribution(motifs: List[Dict[str, Any]], 
                           by: str = 'Class',
                           title: Optional[str] = None,
                           figsize: Tuple[float, float] = None,
                           style: str = 'nature') -> plt.Figure:
    """
    Plot distribution of motifs by class or subclass.
    
    Publication-quality bar chart following Nature Methods guidelines.
    Shows all classes/subclasses even when count is 0, ensuring comprehensive
    visualization of both detected and undetected motif types.
    
    Args:
        motifs: List of motif dictionaries.
        by: Group by 'Class' or 'Subclass'.
        title: Custom plot title.
        figsize: Figure size (width, height) in inches. Uses Nature standard if None.
        style: Style preset ('nature', 'default', 'presentation').
        
    Returns:
        Matplotlib figure object (publication-ready at 300 DPI)
    """
    set_scientific_style(style)
    
    # Use Nature-appropriate figure size
    if figsize is None:
        figsize = FIGURE_SIZES['double_column'] if by == 'Subclass' else FIGURE_SIZES['one_and_half']
    
    # Define ALL expected classes and subclasses (always show these)
    ALL_CLASSES = [
        'Curved_DNA', 'Slipped_DNA', 'Cruciform', 'R-Loop', 'Triplex',
        'G-Quadruplex', 'i-Motif', 'Z-DNA', 'A-philic_DNA', 'Hybrid', 'Non-B_DNA_Clusters'
    ]
    
    ALL_SUBCLASSES = [
        'Global Curvature', 'Local Curvature',  # Curved DNA
        'Direct Repeat', 'STR',  # Slipped DNA
        'Inverted Repeats',  # Cruciform
        'R-loop formation sites', 'QmRLFS-m1', 'QmRLFS-m2',  # R-Loop
        'Triplex', 'Sticky DNA',  # Triplex
        'Telomeric G4', 'Stacked canonical G4s', 'Stacked G4s with linker',  # G-Quadruplex
        'Canonical intramolecular G4', 'Extended-loop canonical',
        'Higher-order G4 array/G4-wire', 'Intramolecular G-triplex', 'Two-tetrad weak PQS',
        'Canonical i-motif', 'Relaxed i-motif', 'AC-motif',  # i-Motif
        'Z-DNA', 'eGZ (Extruded-G) DNA',  # Z-DNA
        'A-philic DNA',  # A-philic
    ]
    
    # Count motifs by specified grouping
    counts = Counter(m.get(by, 'Unknown') for m in motifs) if motifs else Counter()
    
    # Prepare data with all categories
    if by == 'Class':
        categories = ALL_CLASSES
    else:
        categories = ALL_SUBCLASSES
    
    # Get counts (0 if not present)
    values = [counts.get(cat, 0) for cat in categories]
    
    # Get colors (colorblind-friendly)
    if by == 'Class':
        colors = [MOTIF_CLASS_COLORS.get(cat, '#808080') for cat in categories]
    else:
        # Map subclasses to their parent class colors
        subclass_to_class = {
            'Global Curvature': 'Curved_DNA', 'Local Curvature': 'Curved_DNA',
            'Direct Repeat': 'Slipped_DNA', 'STR': 'Slipped_DNA',
            'Inverted Repeats': 'Cruciform',
            'R-loop formation sites': 'R-Loop', 'QmRLFS-m1': 'R-Loop', 'QmRLFS-m2': 'R-Loop',
            'Triplex': 'Triplex', 'Sticky DNA': 'Triplex',
            'Telomeric G4': 'G-Quadruplex', 'Stacked canonical G4s': 'G-Quadruplex',
            'Stacked G4s with linker': 'G-Quadruplex', 'Canonical intramolecular G4': 'G-Quadruplex',
            'Extended-loop canonical': 'G-Quadruplex', 'Higher-order G4 array/G4-wire': 'G-Quadruplex',
            'Intramolecular G-triplex': 'G-Quadruplex', 'Two-tetrad weak PQS': 'G-Quadruplex',
            'Canonical i-motif': 'i-Motif', 'Relaxed i-motif': 'i-Motif', 'AC-motif': 'i-Motif',
            'Z-DNA': 'Z-DNA', 'eGZ (Extruded-G) DNA': 'Z-DNA',
            'A-philic DNA': 'A-philic_DNA',
        }
        colors = [MOTIF_CLASS_COLORS.get(subclass_to_class.get(cat, ''), '#808080') for cat in categories]
    
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Create bars with Nature-style aesthetics
    bars = ax.bar(range(len(categories)), values, color=colors, 
                  edgecolor='black', linewidth=0.5, width=0.8)
    
    # Customize axes (Nature style - minimal, clean)
    ax.set_xlabel(f'Motif {by}', fontweight='normal')
    ax.set_ylabel('Count', fontweight='normal')
    if title:
        # Replace underscores with spaces in title
        display_title = title.replace('_', ' ')
        ax.set_title(display_title, fontweight='bold', pad=10)
    ax.set_xticks(range(len(categories)))
    
    # Replace underscores with spaces in category labels
    # Apply 45° rotation for all categories (Nature style - consistent readability)
    display_categories = [cat.replace('_', ' ') for cat in categories]
    ax.set_xticklabels(display_categories, rotation=45, ha='right')
    
    # Add count labels on ALL bars (improved visibility)
    # Show numbers for all categories to make distribution clear
    max_val = max(values) if max(values) > 0 else 1
    for bar, count in zip(bars, values):
        height = bar.get_height()
        # Position label above bar if count > 0, at baseline if 0
        y_pos = height + max_val * 0.02 if count > 0 else 0.5
        # Use larger font (8pt) and bold for better readability
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


def plot_class_subclass_sunburst(motifs: List[Dict[str, Any]], 
                                 title: str = "Motif Class-Subclass Distribution",
                                 figsize: Tuple[float, float] = None) -> Union[plt.Figure, Any]:
    """
    Create sunburst plot showing class-subclass hierarchy.
    
    Publication-quality hierarchical visualization.
    
    Args:
        motifs: List of motif dictionaries
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Plotly figure if available, otherwise matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['square']
        
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            ax.set_title(title)
        return fig
    
    if not PLOTLY_AVAILABLE:
        # Fallback to matplotlib nested pie chart
        return plot_nested_pie_chart(motifs, title)
    
    # Build hierarchical data for sunburst
    class_subclass_counts = defaultdict(lambda: defaultdict(int))
    
    for motif in motifs:
        class_name = motif.get('Class', 'Unknown')
        subclass_name = motif.get('Subclass', 'Unknown')
        class_subclass_counts[class_name][subclass_name] += 1
    
    # Prepare data for plotly
    ids = []
    labels = []
    parents = []
    values = []
    colors = []
    
    # Add classes (inner ring)
    for class_name, subclasses in class_subclass_counts.items():
        total_class_count = sum(subclasses.values())
        ids.append(class_name)
        labels.append(f"{class_name}<br>({total_class_count})")
        parents.append("")
        values.append(total_class_count)
        colors.append(MOTIF_CLASS_COLORS.get(class_name, '#808080'))
    
    # Add subclasses (outer ring)
    for class_name, subclasses in class_subclass_counts.items():
        for subclass_name, count in subclasses.items():
            ids.append(f"{class_name}_{subclass_name}")
            labels.append(f"{subclass_name}<br>({count})")
            parents.append(class_name)
            values.append(count)
            # Lighter shade of class color for subclasses
            base_color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
            colors.append(base_color + '80')  # Add transparency
    
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1)),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
    ))
    
    # Publication-quality layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=10, family='Arial')),
        font=dict(size=8, family='Arial'),
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    return fig


def plot_nested_pie_chart(motifs: List[Dict[str, Any]], 
                         title: str = "Motif Distribution",
                         figsize: Tuple[float, float] = None) -> plt.Figure:
    """
    Create nested donut chart with improved text placement to avoid overlapping labels.
    
    Publication-quality hierarchical pie chart following Nature guidelines.
    
    Args:
        motifs: List of motif dictionaries
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style('nature')
    
    if figsize is None:
        figsize = FIGURE_SIZES['square']
    
    # Count by class and subclass
    class_counts = Counter(m.get('Class', 'Unknown') for m in motifs)
    class_subclass_counts = defaultdict(lambda: defaultdict(int))
    
    for motif in motifs:
        class_name = motif.get('Class', 'Unknown')
        subclass_name = motif.get('Subclass', 'Unknown')
        class_subclass_counts[class_name][subclass_name] += 1
    
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    # Inner donut (classes)
    class_names = list(class_counts.keys())
    class_values = list(class_counts.values())
    class_colors = [MOTIF_CLASS_COLORS.get(name, '#808080') for name in class_names]
    
    # Create inner donut with Nature-style clean design
    # Use labels=None to manually place labels later for better control
    wedges1, texts1, autotexts1 = ax.pie(
        class_values, 
        labels=None,  # We'll add labels manually
        colors=class_colors,
        radius=0.65,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',  # Show more percentage labels with 1 decimal
        pctdistance=0.80,
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2)  # Thicker edge for better clarity
    )
    
    # Manually add class labels with better positioning (replace underscores with spaces)
    for i, (wedge, class_name) in enumerate(zip(wedges1, class_names)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 0.5 * np.cos(np.radians(angle))
        y = 0.5 * np.sin(np.radians(angle))
        # Replace underscores with spaces in labels
        display_name = class_name.replace('_', ' ')
        # Add white background box for better readability
        ax.text(x, y, display_name, ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    # Outer donut (subclasses)
    all_subclass_counts = []
    all_subclass_colors = []
    all_subclass_labels = []
    
    for class_name in class_names:
        subclass_dict = class_subclass_counts[class_name]
        base_color = MOTIF_CLASS_COLORS.get(class_name, '#808080')
        
        for subclass_name, count in subclass_dict.items():
            all_subclass_counts.append(count)
            # Truncate long names for clean appearance and replace underscores with spaces
            # Use consistent truncation length (15 chars max, including ellipsis)
            display_name = subclass_name.replace('_', ' ')
            MAX_LABEL_LENGTH = 15
            label = display_name if len(display_name) <= MAX_LABEL_LENGTH else display_name[:MAX_LABEL_LENGTH-1] + '…'
            all_subclass_labels.append(label)
            all_subclass_colors.append(base_color)
    
    # Use smarter labeling strategy to avoid overlap
    # For many subclasses, hide labels and rely on legend instead
    if len(all_subclass_labels) > 10:
        # Hide outer ring labels when there are too many
        wedges2, texts2 = ax.pie(
            all_subclass_counts,
            labels=None,  # No labels for cleaner appearance
            colors=all_subclass_colors,
            radius=1.0,
            startangle=90,
            wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1.5)  # Thicker edge for clarity
        )
        
        # Add a legend for subclasses instead
        # Note: All subclasses of the same parent class share the same color by design.
        # This ensures visual grouping in the nested donut chart.
        # Track first occurrence of each unique label for the legend.
        seen_labels = {}
        legend_handles = []
        legend_labels = []
        for i, (label, color) in enumerate(zip(all_subclass_labels, all_subclass_colors)):
            if label not in seen_labels:
                seen_labels[label] = color
                legend_handles.append(plt.Rectangle((0,0),1,1, fc=color, ec='white', lw=1))
                legend_labels.append(label)
                if len(legend_labels) >= 12:  # Limit to 12 for better display
                    break
        
        ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
                 fontsize=7, frameon=True, title='Top Subclasses', title_fontsize=8,
                 framealpha=0.95, edgecolor='lightgray')
    else:
        # For fewer subclasses, show labels with improved spacing
        wedges2, texts2 = ax.pie(
            all_subclass_counts,
            labels=all_subclass_labels,
            colors=all_subclass_colors,
            radius=1.0,
            labeldistance=1.18,  # Push labels further out to avoid overlap
            startangle=90,
            wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1.5),  # Thicker edge
            textprops={'fontsize': 7, 'weight': 'medium'}  # Larger, bolder text
        )
        
        # Adjust label positions to avoid overlap
        for text in texts2:
            text.set_fontsize(7)
            text.set_weight('medium')
            # Add slight rotation for better readability
            angle = text.get_rotation()
            if 90 < angle < 270:
                text.set_rotation(angle - 180)
    
    if title:
        # Replace underscores with spaces in title
        display_title = title.replace('_', ' ')
        ax.set_title(display_title, fontweight='bold', fontsize=12, pad=15)
    
    # Style percentage labels - larger and bolder
    for autotext in autotexts1:
        autotext.set_fontsize(7)
        autotext.set_weight('bold')
        autotext.set_color('white')
    
    return fig


# =============================================================================
# COVERAGE & POSITIONAL PLOTS (Nature-Level Quality)
# =============================================================================

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

def plot_score_distribution(motifs: List[Dict[str, Any]], 
                           by_class: bool = True,
                           title: Optional[str] = None,
                           figsize: Tuple[float, float] = None,
                           style: str = 'nature') -> plt.Figure:
    """
    Plot distribution of motif scores.
    
    Publication-quality box plot visualization.
    
    Args:
        motifs: List of motif dictionaries
        by_class: Whether to separate by motif class
        title: Custom plot title
        figsize: Figure size (width, height) in inches
        style: Style preset
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style(style)
    
    if figsize is None:
        figsize = FIGURE_SIZES['one_and_half']
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            ax.set_title(title)
        return fig
    
    # Extract scores
    scores_data = []
    for motif in motifs:
        score = motif.get('Score', motif.get('Normalized_Score'))
        if isinstance(score, (int, float)):
            if by_class:
                scores_data.append({
                    'Score': score,
                    'Class': motif.get('Class', 'Unknown')
                })
            else:
                scores_data.append(score)
    
    if not scores_data:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No score data available', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            display_title = title.replace('_', ' ')
            ax.set_title(display_title)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    if by_class and isinstance(scores_data[0], dict):
        # Create DataFrame for seaborn
        df = pd.DataFrame(scores_data)
        
        # Box plot by class (Nature style - clean, minimal)
        colors = [MOTIF_CLASS_COLORS.get(cls, '#808080') for cls in df['Class'].unique()]
        sns.boxplot(data=df, x='Class', y='Score', ax=ax, palette=colors,
                   linewidth=0.8, fliersize=2)
        # Replace underscores with spaces in x-tick labels
        ax.set_xticklabels([label.get_text().replace('_', ' ') for label in ax.get_xticklabels()], 
                          rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_xlabel('Motif Class')
    else:
        # Simple histogram
        ax.hist(scores_data, bins=20, edgecolor='black', linewidth=0.5, color='#56B4E9')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
    
    if title:
        # Replace underscores with spaces in title
        display_title = title.replace('_', ' ')
        ax.set_title(display_title, fontweight='bold', pad=10)
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


def plot_length_distribution(motifs: List[Dict[str, Any]], 
                           by_class: bool = True,
                           title: Optional[str] = None,
                           figsize: Tuple[float, float] = None,
                           style: str = 'nature') -> plt.Figure:
    """
    Plot distribution of motif lengths.
    
    Publication-quality violin plot visualization.
    
    Args:
        motifs: List of motif dictionaries
        by_class: Whether to separate by motif class
        title: Custom plot title
        figsize: Figure size (width, height) in inches
        style: Style preset
        
    Returns:
        Matplotlib figure object (publication-ready)
    """
    set_scientific_style(style)
    
    if figsize is None:
        figsize = FIGURE_SIZES['one_and_half']
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No motifs to display', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            ax.set_title(title)
        return fig
    
    # Extract lengths
    length_data = []
    for motif in motifs:
        length = motif.get('Length')
        if isinstance(length, int) and length > 0:
            if by_class:
                length_data.append({
                    'Length': length,
                    'Class': motif.get('Class', 'Unknown')
                })
            else:
                length_data.append(length)
    
    if not length_data:
        fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
        ax.text(0.5, 0.5, 'No length data available', ha='center', va='center', 
                transform=ax.transAxes)
        ax.axis('off')
        if title:
            display_title = title.replace('_', ' ')
            ax.set_title(display_title)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize, dpi=PUBLICATION_DPI)
    
    if by_class and isinstance(length_data[0], dict):
        # Create DataFrame for seaborn
        df = pd.DataFrame(length_data)
        
        # Violin plot by class (Nature style - clean)
        colors = [MOTIF_CLASS_COLORS.get(cls, '#808080') for cls in df['Class'].unique()]
        sns.violinplot(data=df, x='Class', y='Length', ax=ax, palette=colors,
                      linewidth=0.8, inner='box')
        # Replace underscores with spaces in x-tick labels
        ax.set_xticklabels([label.get_text().replace('_', ' ') for label in ax.get_xticklabels()], 
                          rotation=45, ha='right')
        ax.set_ylabel('Length (bp)')
        ax.set_xlabel('Motif Class')
    else:
        # Simple histogram
        ax.hist(length_data, bins=20, edgecolor='black', linewidth=0.5, color='#009E73')
        ax.set_xlabel('Length (bp)')
        ax.set_ylabel('Frequency')
    
    if title:
        # Replace underscores with spaces in title
        display_title = title.replace('_', ' ')
        ax.set_title(display_title, fontweight='bold', pad=10)
    
    # Apply Nature journal style
    _apply_nature_style(ax)
    
    plt.tight_layout()
    return fig


# =============================================================================
# COMPARISON PLOTS (Nature-Level Quality)
# =============================================================================

