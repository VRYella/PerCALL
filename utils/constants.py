"""
Constants Module
================

Shared constants and configuration values used across NonBDNAFinder modules.
Centralized location for system-wide constants following modular architecture.
"""

# Application version
APP_NAME = "NonBDNAFinder"
APP_VERSION = "2025.1"

# Core output columns for motif data
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

# Chunking configuration
CHUNK_THRESHOLD = 100000  # 100KB - only chunk very large sequences
DEFAULT_CHUNK_SIZE = 500000  # 500KB chunks for optimal performance
DEFAULT_CHUNK_OVERLAP = 1000  # 1KB overlap (optimal balance)

# Hybrid and cluster detection parameters
HYBRID_MIN_OVERLAP = 0.50  # Minimum overlap ratio for hybrid detection (50%)
HYBRID_MAX_OVERLAP = 0.99  # Maximum overlap ratio for hybrid detection (99%)
CLUSTER_WINDOW_SIZE = 300  # Sliding window size in bp for cluster detection
CLUSTER_MIN_MOTIFS = 4     # Minimum number of motifs required in a cluster
CLUSTER_MIN_CLASSES = 3    # Minimum number of different classes required in a cluster

# Detector display names
DETECTOR_DISPLAY_NAMES = {
    'curved_dna': 'Curved DNA',
    'slipped_dna': 'Slipped DNA',
    'cruciform': 'Cruciform',
    'r_loop': 'R-Loop',
    'triplex': 'Triplex',
    'g_quadruplex': 'G-Quadruplex',
    'i_motif': 'i-Motif',
    'z_dna': 'Z-DNA',
    'a_philic': 'A-philic DNA'
}
