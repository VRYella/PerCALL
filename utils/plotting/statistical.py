"""Statistical Plots Module
========================

Statistical analysis and visualization functions.
Includes comprehensive class analysis and statistical comparisons.

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

def plot_class_analysis_comprehensive(motifs: List[Dict[str, Any]], 
                                     figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Comprehensive class-level analysis with multiple subplots.
    Shows distribution, statistics, and comparison of all 11 Non-B DNA classes.
    Highlights which classes were detected and which were not.
    
    Args:
        motifs: List of motif dictionaries
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object with multiple subplots
    """
    set_scientific_style()
    
    # Define all 11 Non-B DNA classes
    all_classes = [
        'Curved_DNA', 'Slipped_DNA', 'Cruciform', 'R-Loop', 'Triplex',
        'G-Quadruplex', 'i-Motif', 'Z-DNA', 'A-philic_DNA', 
        'Hybrid', 'Non-B_DNA_Clusters'
    ]
    
    # Count motifs by class
    detected_classes = Counter(m.get('Class', 'Unknown') for m in motifs)
    
    # Identify detected vs not detected
    detected = [cls for cls in all_classes if detected_classes.get(cls, 0) > 0]
    not_detected = [cls for cls in all_classes if detected_classes.get(cls, 0) == 0]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Main distribution bar chart
    ax1 = fig.add_subplot(gs[0, :])
    counts = [detected_classes.get(cls, 0) for cls in all_classes]
    colors = [MOTIF_CLASS_COLORS.get(cls, '#808080') for cls in all_classes]
    bars = ax1.bar(range(len(all_classes)), counts, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Non-B DNA Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of All 11 Non-B DNA Classes', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(all_classes)))
    ax1.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Detected vs Not Detected pie chart
    ax2 = fig.add_subplot(gs[1, 0])
    detection_counts = [len(detected), len(not_detected)]
    detection_labels = [f'Detected\n({len(detected)} classes)', 
                       f'Not Detected\n({len(not_detected)} classes)']
    colors_pie = ['#4CAF50', '#FF5722']
    ax2.pie(detection_counts, labels=detection_labels, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Class Detection Status', fontsize=12, fontweight='bold')
    
    # 3. Statistics table for detected classes
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    if detected:
        # Calculate statistics for detected classes
        class_stats = []
        for cls in detected[:5]:  # Show top 5
            cls_motifs = [m for m in motifs if m.get('Class') == cls]
            count = len(cls_motifs)
            avg_length = np.mean([m.get('Length', 0) for m in cls_motifs]) if cls_motifs else 0
            avg_score = np.mean([m.get('Score', 0) for m in cls_motifs]) if cls_motifs else 0
            class_stats.append([cls[:15], count, f'{avg_length:.1f}', f'{avg_score:.3f}'])
        
        table = ax3.table(cellText=class_stats,
                         colLabels=['Class', 'Count', 'Avg Len', 'Avg Score'],
                         cellLoc='left', loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
        
        ax3.set_title('Top Detected Classes (Statistics)', fontsize=12, fontweight='bold', pad=20)
    else:
        ax3.text(0.5, 0.5, 'No classes detected', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
    
    # 4. List of not detected classes
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    if not_detected:
        not_detected_text = 'Classes NOT Detected:\n' + ', '.join(not_detected)
        ax4.text(0.5, 0.5, not_detected_text, ha='center', va='center',
                transform=ax4.transAxes, fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8),
                wrap=True)
    else:
        ax4.text(0.5, 0.5, 'All 11 Non-B DNA classes detected! [OK]', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                fontweight='bold', color='green')
    
    plt.suptitle(f'Comprehensive Class Analysis ({len(motifs)} total motifs)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def plot_subclass_analysis_comprehensive(motifs: List[Dict[str, Any]], 
                                        figsize: Tuple[int, int] = (18, 14)) -> plt.Figure:
    """
    Comprehensive subclass-level analysis showing all detected subclasses
    organized by their parent class.
    
    Args:
        motifs: List of motif dictionaries
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object with subclass analysis
    """
    set_scientific_style()
    
    # Group motifs by class and subclass
    class_subclass_counts = defaultdict(lambda: defaultdict(int))
    for motif in motifs:
        class_name = motif.get('Class', 'Unknown')
        subclass_name = motif.get('Subclass', 'Unknown')
        class_subclass_counts[class_name][subclass_name] += 1
    
    # Prepare data for visualization
    all_subclasses = []
    all_counts = []
    all_classes = []
    
    for class_name in sorted(class_subclass_counts.keys()):
        for subclass_name, count in sorted(class_subclass_counts[class_name].items()):
            all_subclasses.append(f"{class_name}:{subclass_name}")
            all_counts.append(count)
            all_classes.append(class_name)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Subclass distribution bar chart
    colors = [MOTIF_CLASS_COLORS.get(cls, '#808080') for cls in all_classes]
    x_pos = range(len(all_subclasses))
    bars = ax1.barh(x_pos, all_counts, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(all_subclasses, fontsize=9)
    ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Class:Subclass', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of All Detected Subclasses', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, all_counts):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count}', ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 2. Subclass summary by class
    ax2.axis('off')
    
    # Create summary text
    summary_lines = ['Subclass Summary by Class:\n']
    for class_name in sorted(class_subclass_counts.keys()):
        subclasses = class_subclass_counts[class_name]
        n_subclasses = len(subclasses)
        total_count = sum(subclasses.values())
        summary_lines.append(f'{class_name}: {n_subclasses} subclass(es), {total_count} motifs')
    
    summary_text = '\n'.join(summary_lines)
    ax2.text(0.1, 0.5, summary_text, ha='left', va='center',
            transform=ax2.transAxes, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    
    plt.suptitle(f'Comprehensive Subclass Analysis ({len(motifs)} total motifs)', 
                fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    return fig


def plot_score_statistics_by_class(motifs: List[Dict[str, Any]], 
                                   figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Advanced statistical visualization of scores by class.
    Shows box plots, violin plots, and statistical annotations.
    
    Args:
        motifs: List of motif dictionaries
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object with score statistics
    """
    set_scientific_style()
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No motifs to analyze', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        return fig
    
    # Prepare data
    df_data = []
    for motif in motifs:
        df_data.append({
            'Class': motif.get('Class', 'Unknown'),
            'Score': motif.get('Score', 0),
            'Length': motif.get('Length', 0)
        })
    df = pd.DataFrame(df_data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Violin plot with box plot overlay
    classes = sorted(df['Class'].unique())
    positions = range(len(classes))
    
    # Create violin plot
    parts = ax1.violinplot([df[df['Class'] == cls]['Score'].values for cls in classes],
                          positions=positions, widths=0.7, showmeans=True, showmedians=True)
    
    # Color violins by class
    for i, pc in enumerate(parts['bodies']):
        cls = classes[i]
        color = MOTIF_CLASS_COLORS.get(cls, '#808080')
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # Overlay box plots
    bp = ax1.boxplot([df[df['Class'] == cls]['Score'].values for cls in classes],
                     positions=positions, widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor='white', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Score Distribution by Class (Violin + Box Plot)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add statistical annotations
    for i, cls in enumerate(classes):
        scores = df[df['Class'] == cls]['Score'].values
        if len(scores) > 0:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            ax1.text(i, ax1.get_ylim()[1] * 0.95, 
                    f'μ={mean_score:.2f}\nσ={std_score:.2f}',
                    ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 2. Statistical summary table
    ax2.axis('off')
    
    # Calculate statistics
    stats_data = []
    for cls in classes:
        cls_scores = df[df['Class'] == cls]['Score'].values
        if len(cls_scores) > 0:
            stats_data.append([
                cls[:15],
                len(cls_scores),
                f'{np.mean(cls_scores):.3f}',
                f'{np.median(cls_scores):.3f}',
                f'{np.std(cls_scores):.3f}',
                f'{np.min(cls_scores):.3f}',
                f'{np.max(cls_scores):.3f}'
            ])
    
    if stats_data:
        table = ax2.table(cellText=stats_data,
                         colLabels=['Class', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max'],
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.1, 0.13, 0.13, 0.13, 0.13, 0.13])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
    
    plt.suptitle('Score Statistics Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def plot_length_statistics_by_class(motifs: List[Dict[str, Any]], 
                                   figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Advanced visualization of motif length distributions by class.
    
    Args:
        motifs: List of motif dictionaries
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object with length statistics
    """
    set_scientific_style()
    
    if not motifs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No motifs to analyze', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        return fig
    
    # Prepare data
    df_data = []
    for motif in motifs:
        df_data.append({
            'Class': motif.get('Class', 'Unknown'),
            'Length': motif.get('Length', 0)
        })
    df = pd.DataFrame(df_data)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                       gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # 1. Histogram with KDE overlay for each class
    classes = sorted(df['Class'].unique())
    
    for cls in classes:
        cls_lengths = df[df['Class'] == cls]['Length'].values
        if len(cls_lengths) > 1:
            color = MOTIF_CLASS_COLORS.get(cls, '#808080')
            ax1.hist(cls_lengths, bins=20, alpha=0.4, label=cls, color=color, edgecolor='black')
    
    ax1.set_xlabel('Length (bp)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Length Distribution by Class', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Box plot comparison
    bp = ax2.boxplot([df[df['Class'] == cls]['Length'].values for cls in classes],
                     labels=classes, patch_artist=True, vert=True)
    
    # Color boxes by class
    for i, (patch, cls) in enumerate(zip(bp['boxes'], classes)):
        color = MOTIF_CLASS_COLORS.get(cls, '#808080')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Length (bp)', fontsize=12, fontweight='bold')
    ax2.set_title('Length Comparison (Box Plot)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Statistical summary table
    ax3.axis('off')
    
    stats_data = []
    for cls in classes:
        cls_lengths = df[df['Class'] == cls]['Length'].values
        if len(cls_lengths) > 0:
            stats_data.append([
                cls[:15],
                len(cls_lengths),
                f'{np.mean(cls_lengths):.1f}',
                f'{np.median(cls_lengths):.1f}',
                f'{np.std(cls_lengths):.1f}',
                f'{np.min(cls_lengths):.0f}',
                f'{np.max(cls_lengths):.0f}'
            ])
    
    if stats_data:
        table = ax3.table(cellText=stats_data,
                         colLabels=['Class', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max'],
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.1, 0.13, 0.13, 0.13, 0.13, 0.13])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
    
    plt.suptitle('Length Statistics Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


# =============================================================================
# TESTING & EXAMPLES
# =============================================================================

def test_visualizations():
    """Test visualization functions with example data"""
    print("Testing NBDScanner visualizations...")
    
    # Create example motif data
    example_motifs = [
        {'Class': 'G-Quadruplex', 'Subclass': 'Canonical intramolecular G4', 'Start': 1, 'End': 21, 'Length': 21, 'Score': 0.85},
        {'Class': 'G-Quadruplex', 'Subclass': 'Extended-loop canonical', 'Start': 45, 'End': 60, 'Length': 16, 'Score': 0.72},
        {'Class': 'Curved_DNA', 'Subclass': 'A-tract', 'Start': 80, 'End': 95, 'Length': 16, 'Score': 0.65},
        {'Class': 'Z-DNA', 'Subclass': 'CG alternating', 'Start': 120, 'End': 135, 'Length': 16, 'Score': 0.90},
        {'Class': 'i-Motif', 'Subclass': 'Canonical i-motif', 'Start': 160, 'End': 180, 'Length': 21, 'Score': 0.78}
    ]
    
    sequence_length = 200
    
    print(f"\nTesting with {len(example_motifs)} example motifs:")
    for motif in example_motifs:
        print(f"  {motif['Class']} at {motif['Start']}-{motif['End']}")
    
    # Test basic plots
    try:
        fig1 = plot_motif_distribution(example_motifs, by='Class')
        plt.close(fig1)
        print("[OK] Motif distribution plot: PASS")
    except Exception as e:
        print(f"✗ Motif distribution plot: FAIL - {e}")
    
    try:
        fig2 = plot_coverage_map(example_motifs, sequence_length)
        plt.close(fig2)
        print("[OK] Coverage map plot: PASS")
    except Exception as e:
        print(f"✗ Coverage map plot: FAIL - {e}")
    
    try:
        fig3 = plot_score_distribution(example_motifs)
        plt.close(fig3)
        print("[OK] Score distribution plot: PASS")
    except Exception as e:
        print(f"✗ Score distribution plot: FAIL - {e}")
    
    try:
        fig4 = plot_nested_pie_chart(example_motifs)
        plt.close(fig4)
        print("[OK] Nested pie chart: PASS")
    except Exception as e:
        print(f"✗ Nested pie chart: FAIL - {e}")
    
    print(f"\n[OK] Visualization testing completed")
    print(f"Plotly available: {'Yes' if PLOTLY_AVAILABLE else 'No'}")


# =============================================================================
# ENHANCED STATISTICS VISUALIZATIONS: DENSITY AND ENRICHMENT
# =============================================================================

