"""
Data Export Module
==================

Functions for exporting motif data to various formats (CSV, BED, JSON, Excel, PDF).
Extracted from utilities.py for focused export functionality following modular architecture.
"""

from typing import Dict, Any, List, Optional
import json
import csv
from io import StringIO
from collections import defaultdict

# Import pandas for Excel export
try:
    import pandas as pd
except ImportError:
    pd = None

# Constants for export (from utilities.py)
CORE_OUTPUT_COLUMNS = [
    'Sequence_Name',  # Identity: Traceability
    'Class',          # Classification: Biological interpretation
    'Subclass',       # Classification: Detailed subtype
    'Start',          # Genomics: Absolute genomic context
    'End',            # Genomics: Absolute genomic context
    'Length',         # Genomics: Feature size (bp)
    'Sequence',       # Sequence: Always visible motif sequence
    'Strand',         # Strand: DNA strand orientation (+/- indicates forward/reverse)
    'Score',          # Confidence: 0-3 normalized, cross-motif comparability
    'Method',         # Evidence: Reproducibility (Regex/k-mer/ΔG/Hyperscan)
    'Pattern_ID',     # Evidence: Pattern identifier for traceability
]

# Motif-specific columns (ONLY reported when relevant per motif class)
MOTIF_SPECIFIC_COLUMNS = {
    'G-Quadruplex': ['Num_Tracts', 'Loop_Length', 'Num_Stems', 'Stem_Length', 'Priority'],
    'Z-DNA': ['Mean_10mer_Score', 'Contributing_10mers', 'Alternating_CG_Regions'],
    'i-Motif': ['Num_C_Tracts', 'Loop_Length', 'Motif_Type'],
    'Slipped DNA': ['Repeat_Unit', 'Unit_Length', 'Repeat_Count'],
    'Cruciform': ['Arm_Length', 'Loop_Length', 'Num_Stems'],
    'Triplex': ['Mirror_Type', 'Spacer_Length', 'Arm_Length', 'Loop_Length'],
    'R-Loop': ['GC_Skew', 'RIZ_Length', 'REZ_Length'],
    'Curved DNA': ['Tract_Type', 'Tract_Length', 'Num_Tracts'],
    'A-Philic': ['Tract_Type', 'Tract_Length'],
}

# Classes that are excluded from non-overlapping consolidated outputs
EXCLUDED_FROM_CONSOLIDATED = ['Hybrid', 'Non-B_DNA_Clusters']

# Default values for missing core columns
DEFAULT_COLUMN_VALUES = {
    'Strand': '+',
    'Method': 'Pattern_detection',
    'Pattern_ID': 'Unknown',
    'Score': 0.0
}


def export_to_bed(motifs: List[Dict[str, Any]], sequence_name: str = "sequence", 
                  filename: Optional[str] = None) -> str:
    """
    Export motifs to BED format.
    
    Args:
        motifs: List of motif dictionaries
        sequence_name: Name of the sequence
        filename: Optional output filename
        
    Returns:
        BED format string
    """
    bed_lines = []
    bed_lines.append("track name=NBDScanner_motifs description=\"Non-B DNA motifs\" itemRgb=On")
    
    # Color mapping for different classes
    class_colors = {
        'Curved_DNA': '255,182,193',      # Light pink
        'Slipped_DNA': '255,218,185',     # Peach
        'Cruciform': '173,216,230',       # Light blue
        'R-Loop': '144,238,144',          # Light green
        'Triplex': '221,160,221',         # Plum
        'G-Quadruplex': '255,215,0',      # Gold
        'i-Motif': '255,165,0',           # Orange
        'Z-DNA': '138,43,226',            # Blue violet
        'A-philic_DNA': '230,230,250',    # Lavender
        'Hybrid': '192,192,192',          # Silver
        'Non-B_DNA_Clusters': '128,128,128'  # Gray
    }
    
    for motif in motifs:
        chrom = sequence_name
        start = max(0, motif.get('Start', 1) - 1)  # Convert to 0-based
        end = motif.get('End', start + 1)
        name = f"{motif.get('Class', 'Unknown')}_{motif.get('Subclass', 'Unknown')}"
        score = int(min(1000, max(0, motif.get('Score', 0) * 1000)))  # Scale to 0-1000
        strand = motif.get('Strand', '+')
        color = class_colors.get(motif.get('Class'), '128,128,128')
        
        bed_line = f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t{start}\t{end}\t{color}"
        bed_lines.append(bed_line)
    
    bed_content = '\n'.join(bed_lines)
    
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write(bed_content)
        except Exception as e:
            print(f"Error writing BED file {filename}: {e}")
    
    return bed_content


def export_to_csv(motifs: List[Dict[str, Any]], filename: Optional[str] = None, 
                 non_overlapping_only: bool = False) -> str:
    """
    Export motifs to CSV format with CORE fields only.
    
    Output tables use minimal, high-value features per publication standards:
    - Sequence_Name, Class, Subclass, Start, End, Length, Strand, Score, Method, Pattern_ID
    
    Args:
        motifs: List of motif dictionaries
        filename: Optional output filename
        non_overlapping_only: If True, exclude Hybrid and Cluster motifs (default: False)
        
    Returns:
        CSV format string
    """
    if not motifs:
        return "No motifs to export"
    
    # Filter motifs if requested
    if non_overlapping_only:
        motifs = [m for m in motifs if m.get('Class') not in EXCLUDED_FROM_CONSOLIDATED]
    
    # Use core columns constant
    core_columns = CORE_OUTPUT_COLUMNS
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=core_columns)
    writer.writeheader()
    
    for motif in motifs:
        # Create row with only core fields
        row = {}
        for col in core_columns:
            value = motif.get(col, None)
            
            # Set appropriate defaults for missing columns using constants
            if value == '' or value is None:
                value = DEFAULT_COLUMN_VALUES.get(col, 'NA')
            
            row[col] = value
        
        writer.writerow(row)
    
    csv_content = output.getvalue()
    output.close()
    
    if filename:
        try:
            with open(filename, 'w', newline='') as f:
                f.write(csv_content)
        except Exception as e:
            print(f"Error writing CSV file {filename}: {e}")
    
    return csv_content


def export_to_json(motifs: List[Dict[str, Any]], filename: Optional[str] = None, 
                   pretty: bool = True) -> str:
    """
    Export motifs to JSON format.
    
    Args:
        motifs: List of motif dictionaries
        filename: Optional output filename
        pretty: Whether to format JSON prettily
        
    Returns:
        JSON format string
    """
    json_data = {
        'version': '2024.1',
        'analysis_type': 'NBDScanner_Non-B_DNA_Analysis',
        'total_motifs': len(motifs),
        'motifs': motifs
    }
    
    if pretty:
        json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
    else:
        json_content = json.dumps(json_data, ensure_ascii=False)
    
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write(json_content)
        except Exception as e:
            print(f"Error writing JSON file {filename}: {e}")
    
    return json_content


def export_to_excel(motifs: List[Dict[str, Any]], filename: str = "nonbscanner_results.xlsx", 
                   simple_format: bool = False) -> str:
    """
    Export motifs to Excel format with publication-grade sheets:
    - Main sheet: Core columns only (minimal, publication-grade)
    - Additional sheets: Motif-specific columns per class (conditional reporting)
    
    Args:
        motifs: List of motif dictionaries
        filename: Output Excel filename (default: "nonbscanner_results.xlsx")
        simple_format: If True, use 2-tab format; if False, use class-specific format
        
    Returns:
        Success message string
    """
    if pd is None:
        raise ImportError("pandas is required for Excel export. Install with: pip install pandas openpyxl")
    
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
    
    if not motifs:
        return "No motifs to export"
    
    # Core columns for all sheets
    core_columns = CORE_OUTPUT_COLUMNS
    
    # Prepare row with core columns only
    def prepare_core_row(motif):
        row = {}
        for col in core_columns:
            value = motif.get(col, None)
            if value == '' or value is None:
                value = DEFAULT_COLUMN_VALUES.get(col, 'NA')
            row[col] = value
        return row
    
    # Prepare row with motif-specific columns
    def prepare_detailed_row(motif, motif_class):
        row = prepare_core_row(motif)
        
        # Add motif-specific columns based on class
        specific_cols = MOTIF_SPECIFIC_COLUMNS.get(motif_class, [])
        for col in specific_cols:
            value = motif.get(col, 'NA')
            if value == '' or value is None:
                value = 'NA'
            row[col] = value
        
        return row
    
    # Create Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        if simple_format:
            # Simple 2-tab format for user downloads
            # Tab 1: Core columns only (NonOverlappingConsolidated)
            consolidated_motifs = [m for m in motifs 
                                 if m.get('Class') not in EXCLUDED_FROM_CONSOLIDATED]
            
            if consolidated_motifs:
                consolidated_data = [prepare_core_row(m) for m in consolidated_motifs]
                df_consolidated = pd.DataFrame(consolidated_data, columns=core_columns)
                df_consolidated.to_excel(writer, sheet_name='NonOverlappingConsolidated', index=False)
            
            # Tab 2: Core columns only (OverlappingAll)
            all_data = [prepare_core_row(m) for m in motifs]
            df_all = pd.DataFrame(all_data, columns=core_columns)
            df_all.to_excel(writer, sheet_name='OverlappingAll', index=False)
        else:
            # Publication-grade format with motif-specific sheets
            # Sheet 1: Core columns only (all non-overlapping motifs)
            consolidated_motifs = [m for m in motifs 
                                 if m.get('Class') not in EXCLUDED_FROM_CONSOLIDATED]
            
            if consolidated_motifs:
                consolidated_data = [prepare_core_row(m) for m in consolidated_motifs]
                df_consolidated = pd.DataFrame(consolidated_data, columns=core_columns)
                df_consolidated.to_excel(writer, sheet_name='Core_Results', index=False)
            
            # Group motifs by class for detailed sheets
            class_groups = defaultdict(list)
            for motif in motifs:
                cls = motif.get('Class', 'Unknown')
                if cls not in EXCLUDED_FROM_CONSOLIDATED:  # Skip these for detailed sheets
                    class_groups[cls].append(motif)
            
            # Create class-specific sheets with motif-specific columns
            for cls, class_motifs in sorted(class_groups.items()):
                # Sanitize sheet name (Excel has 31 character limit)
                sheet_name = cls.replace('/', '_').replace(' ', '_').replace('-', '_')[:31]
                
                # Get columns for this class: core + motif-specific
                class_columns = core_columns + MOTIF_SPECIFIC_COLUMNS.get(cls, [])
                
                class_data = [prepare_detailed_row(m, cls) for m in class_motifs]
                df_class = pd.DataFrame(class_data, columns=class_columns)
                df_class.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Separate sheets for Hybrid and Cluster motifs (if present)
            hybrid_motifs = [m for m in motifs if m.get('Class') == 'Hybrid']
            if hybrid_motifs:
                hybrid_data = [prepare_core_row(m) for m in hybrid_motifs]
                df_hybrid = pd.DataFrame(hybrid_data, columns=core_columns)
                df_hybrid.to_excel(writer, sheet_name='Hybrid_Motifs', index=False)
            
            cluster_motifs = [m for m in motifs if m.get('Class') == 'Non-B_DNA_Clusters']
            if cluster_motifs:
                cluster_data = [prepare_core_row(m) for m in cluster_motifs]
                df_cluster = pd.DataFrame(cluster_data, columns=core_columns)
                df_cluster.to_excel(writer, sheet_name='Cluster_Motifs', index=False)
    
    return f"Excel file exported successfully to {filename}"


def export_to_gff3(motifs: List[Dict[str, Any]], sequence_name: str = "sequence", 
                  filename: Optional[str] = None) -> str:
    """
    Export motifs to GFF3 format.
    
    Args:
        motifs: List of motif dictionaries
        sequence_name: Name of the sequence
        filename: Optional output filename
        
    Returns:
        GFF3 format string
    """
    gff3_lines = []
    gff3_lines.append("##gff-version 3")
    gff3_lines.append(f"##sequence-region {sequence_name} 1 {max(m.get('End', 0) for m in motifs) if motifs else 0}")
    
    for motif in motifs:
        seqid = sequence_name
        source = "NBDScanner"
        feature_type = motif.get('Class', 'motif').replace(' ', '_').replace('-', '_')
        start = motif.get('Start', 1)
        end = motif.get('End', start)
        score = motif.get('Score', 0)
        strand = motif.get('Strand', '+')
        phase = '.'
        
        # Build attributes
        attributes = [
            f"ID={motif.get('ID', 'unknown')}",
            f"Name={motif.get('Class', 'Unknown')}",
            f"subclass={motif.get('Subclass', 'Unknown')}",
            f"method={motif.get('Method', 'unknown')}",
            f"pattern_id={motif.get('Pattern_ID', 'unknown')}"
        ]
        attributes_str = ';'.join(attributes)
        
        gff3_line = f"{seqid}\t{source}\t{feature_type}\t{start}\t{end}\t{score}\t{strand}\t{phase}\t{attributes_str}"
        gff3_lines.append(gff3_line)
    
    gff3_content = '\n'.join(gff3_lines)
    
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write(gff3_content)
        except Exception as e:
            print(f"Error writing GFF3 file {filename}: {e}")
    
    return gff3_content
