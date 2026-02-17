"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DEPRECATED: LEGACY MONOLITHIC FILE                         ║
║                    Use Modular Architecture Instead                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

⚠️  DEPRECATION NOTICE ⚠️

This monolithic file is DEPRECATED and maintained for backward compatibility only.

MIGRATION GUIDE:
Please migrate to the modular architecture located in:
  - engine/detectors/ - Individual detector classes
  - engine/detectors/__init__.py - Unified detector exports

OLD (Deprecated):
    from detectors import CurvedDNADetector, ZDNADetector

NEW (Recommended):
    from engine.detectors import CurvedDNADetector, ZDNADetector

BENEFITS OF MODULAR ARCHITECTURE:
  ✅ Better code organization and maintainability
  ✅ Easier testing and debugging
  ✅ Clearer dependency management
  ✅ Follows Python best practices
  ✅ Enables selective imports

See MODULAR_ARCHITECTURE_STATUS.md for complete migration guide.

════════════════════════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════════════════════╗
║                    CONSOLIDATED MOTIF DETECTORS MODULE                        ║
║             Non-B DNA Motif Detection Classes - All in One                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

MODULE: detectors.py
AUTHOR: Dr. Venkata Rajesh Yella
VERSION: 2024.1 - Consolidated
LICENSE: MIT

DESCRIPTION:
    Consolidated module containing all motif detector classes for Non-B DNA
    structures. Combines functionality from 10 separate detector files into
    a single, well-organized module.

DETECTOR CLASSES:
    - BaseMotifDetector: Abstract base class for all detectors
    - CurvedDNADetector: A-tract mediated DNA curvature detection
    - ZDNADetector: Z-DNA and left-handed helix detection
    - APhilicDetector: A-philic DNA tetranucleotide analysis
    - SlippedDNADetector: Direct repeats and STR detection
    - CruciformDetector: Palindromic inverted repeat detection
    - RLoopDetector: R-loop formation site detection
    - TriplexDetector: Triplex and mirror repeat detection
    - GQuadruplexDetector: G4 and G-quadruplex variants
    - IMotifDetector: i-Motif and AC-motif detection

PERFORMANCE:
    - Hyperscan-based patterns: 15,000-65,000 bp/s
    - Algorithmic detectors: 5,000-280,000 bp/s
    - Memory efficient: ~5 MB for 100K sequences

USAGE:
    from detectors import CurvedDNADetector, ZDNADetector
    
    detector = CurvedDNADetector()
    motifs = detector.detect(sequence)
"""

import warnings
warnings.warn(
    "DEPRECATED: Importing from 'detectors.py' is deprecated. "
    "Please use the modular architecture instead: "
    "from engine.detectors import CurvedDNADetector, ZDNADetector, etc. "
    "See MODULAR_ARCHITECTURE_STATUS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

# Import centralized pattern definitions
try:
    from motif_patterns import (
        CURVED_DNA_PATTERNS,
        ZDNA_PATTERNS,
        APHILIC_DNA_PATTERNS,
        SLIPPED_DNA_PATTERNS,
        CRUCIFORM_PATTERNS,
        RLOOP_PATTERNS,
        TRIPLEX_PATTERNS,
        G4_PATTERNS,
        IMOTIF_PATTERNS,
    )
except ImportError:
    # Fallback if motif_patterns module unavailable
    CURVED_DNA_PATTERNS = {}
    ZDNA_PATTERNS = {}
    APHILIC_DNA_PATTERNS = {}
    SLIPPED_DNA_PATTERNS = {}
    CRUCIFORM_PATTERNS = {}
    RLOOP_PATTERNS = {}
    TRIPLEX_PATTERNS = {}
    G4_PATTERNS = {}
    IMOTIF_PATTERNS = {}

# Import optimized scanner functions
try:
    from scanner import (
        find_direct_repeats as _find_direct_repeats_optimized,
        find_inverted_repeats as _find_inverted_repeats_optimized,
        find_mirror_repeats as _find_mirror_repeats_optimized,
        find_strs as _find_strs_optimized
    )
except ImportError:
    # Fallback if scanner module unavailable
    _find_direct_repeats_optimized = None
    _find_inverted_repeats_optimized = None
    _find_mirror_repeats_optimized = None
    _find_strs_optimized = None

"""
Base Detector Class for Modular Motif Detection
================================================

TABULAR SUMMARY:
┌──────────────────────────────────────────────────────────────────────────────┐
│ Module:        base_detector.py                                              │
│ Purpose:       Abstract base class for all motif detectors                   │
│ Performance:   O(n) regex matching per pattern                               │
│ Author:        Dr. Venkata Rajesh Yella                                      │
│ Last Updated:  2024                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ KEY FEATURES:                                                                │
│ • Abstract base class with common interface for all detectors                │
│ • Pattern compilation and caching for performance                            │
│ • Standardized motif detection and scoring methods                           │
│ • Quality threshold validation                                               │
│ • Statistics and metadata generation                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ METHODS:                                                                     │
│ • get_patterns()          → Returns detector-specific patterns               │
│ • calculate_score()       → Computes motif-specific scores                   │
│ • detect_motifs()         → Main detection method                            │
│ • passes_quality_threshold() → Validates motif quality                       │
│ • get_statistics()        → Returns detector statistics                      │
└──────────────────────────────────────────────────────────────────────────────┘
"""



class BaseMotifDetector(ABC):
    """
    Abstract base class for all Non-B DNA motif detectors.
    
    # Motif Output Structure:
    # | Field         | Type  | Description                          |
    # |---------------|-------|--------------------------------------|
    # | ID            | str   | Unique motif identifier              |
    # | Sequence_Name | str   | Source sequence name                 |
    # | Class         | str   | Motif class (e.g., 'Curved_DNA')     |
    # | Subclass      | str   | Motif subclass/variant               |
    # | Start         | int   | 1-based start position               |
    # | End           | int   | End position (inclusive)             |
    # | Length        | int   | Motif length in bp                   |
    # | Sequence      | str   | Actual DNA sequence                  |
    # | Score         | float | Detection confidence score (0-1)     |
    # | Strand        | str   | Strand orientation ('+' or '-')      |
    # | Method        | str   | Detection method identifier          |
    # | Pattern_ID    | str   | Pattern identifier used for match    |
    """
    
    def __init__(self):
        self.patterns = self.get_patterns()
        self.compiled_patterns = self._compile_patterns()
        # Detector execution audit - tracks detection pipeline
        self.audit = {
            'invoked': False,
            'windows_scanned': 0,
            'candidates_seen': 0,
            'candidates_filtered': 0,
            'reported': 0,
            'seed_hits': 0,
            'both_strands_scanned': False
        }
    
    @abstractmethod
    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Return patterns specific to this motif class.
        
        # Pattern Structure:
        # | Field       | Type  | Description                    |
        # |-------------|-------|--------------------------------|
        # | regex       | str   | Regular expression pattern     |
        # | pattern_id  | str   | Unique pattern identifier      |
        # | name        | str   | Human-readable name            |
        # | subclass    | str   | Motif subclass                 |
        # | min_length  | int   | Minimum match length           |
        # | score_type  | str   | Scoring method name            |
        # | threshold   | float | Quality threshold (0-1)        |
        # | description | str   | Pattern description            |
        # | reference   | str   | Literature citation            |
        """
    
    @abstractmethod  
    def get_motif_class_name(self) -> str:
        """Return the motif class name (e.g., 'Curved_DNA', 'G_Quadruplex')"""
        pass
    
    @abstractmethod
    def calculate_score(self, sequence: str, pattern_info: Tuple) -> float:
        """
        Calculate motif-specific confidence score.
        
        Args:
            sequence: DNA sequence string (uppercase)
            pattern_info: Pattern tuple with metadata
            
        Returns:
            Score value between 0.0 and 1.0
        """
        pass
    
    def _compile_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Compile all regex patterns once for performance.
        Uses re.IGNORECASE | re.ASCII for optimal DNA sequence matching.
        """
        compiled_patterns = {}
        
        for pattern_group, patterns in self.patterns.items():
            compiled_group = []
            for pattern_info in patterns:
                pattern, pattern_id, name, subclass = pattern_info[:4]
                try:
                    compiled_re = re.compile(pattern, re.IGNORECASE | re.ASCII)
                    compiled_group.append((compiled_re, pattern_id, name, subclass, pattern_info))
                except re.error as e:
                    print(f"Warning: Invalid pattern {pattern}: {e}")
                    continue
            compiled_patterns[pattern_group] = compiled_group
        
        return compiled_patterns
    
    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Main detection method - scans sequence for all compiled patterns.
        
        # Detection Process:
        # | Step | Action                              |
        # |------|-------------------------------------|
        # | 1    | Normalize sequence to uppercase     |
        # | 2    | Iterate through compiled patterns   |
        # | 3    | Find all regex matches              |
        # | 4    | Calculate motif-specific scores     |
        # | 5    | Apply quality thresholds            |
        # | 6    | Return list of motif dictionaries   |
        
        Args:
            sequence: DNA sequence string
            sequence_name: Identifier for the sequence
            
        Returns:
            List of motif dictionaries with standardized fields
        """
        # Reset and mark audit as invoked
        self.audit['invoked'] = True
        self.audit['windows_scanned'] = 1
        self.audit['candidates_seen'] = 0
        self.audit['candidates_filtered'] = 0
        self.audit['reported'] = 0
        self.audit['seed_hits'] = 0
        
        sequence = sequence.upper().strip()
        motifs = []
        
        for pattern_group, compiled_patterns in self.compiled_patterns.items():
            for compiled_re, pattern_id, name, subclass, full_info in compiled_patterns:
                for match in compiled_re.finditer(sequence):
                    self.audit['seed_hits'] += 1
                    self.audit['candidates_seen'] += 1
                    start, end = match.span()
                    motif_seq = sequence[start:end]
                    
                    score = self.calculate_score(motif_seq, full_info)
                    
                    if self.passes_quality_threshold(motif_seq, score, full_info):
                        motifs.append({
                            'ID': f"{sequence_name}_{pattern_id}_{start+1}",
                            'Sequence_Name': sequence_name,
                            'Class': self.get_motif_class_name(),
                            'Subclass': subclass,
                            'Start': start + 1,
                            'End': end,
                            'Length': len(motif_seq),
                            'Sequence': motif_seq,
                            'Score': round(score, 3),
                            'Strand': '+',
                            'Method': f'{self.get_motif_class_name()}_detection',
                            'Pattern_ID': pattern_id
                        })
                        self.audit['reported'] += 1
                    else:
                        self.audit['candidates_filtered'] += 1
        
        return motifs
    
    def passes_quality_threshold(self, sequence: str, score: float, pattern_info: Tuple) -> bool:
        """Apply quality thresholds - can be overridden by subclasses"""
        # Default threshold from pattern info if available
        if len(pattern_info) > 6:
            min_threshold = pattern_info[6]  # confidence/threshold from pattern
            return score >= min_threshold
        
        # Default minimum score threshold
        return score >= 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        total_patterns = sum(len(patterns) for patterns in self.patterns.values())
        return {
            'motif_class': self.get_motif_class_name(),
            'total_patterns': total_patterns,
            'pattern_groups': list(self.patterns.keys()),
            'patterns_by_group': {k: len(v) for k, v in self.patterns.items()}
        }
    
    def get_audit_info(self) -> Dict[str, Any]:
        """Get detector execution audit information"""
        return self.audit.copy()

# =============================================================================
# Curved Dna Detector
# =============================================================================
"""
CurvedDNADetector

Detects:
 - Global curvature (A-phased repeats, APRs): >= 3 A-tract centers phased ~11 bp apart
 - Local curvature: long A-tracts (>=7) or T-tracts (>=7)

Implements A-tract detection logic similar to the provided C code: within AT-rich windows,
computes the longest A/AnTn run and the longest T-only run, uses difference (maxATlen - maxTlen)
to decide bona fide A-tracts and reports the tract center.
"""

import re
from typing import List, Dict, Any, Tuple
# # from .base_detector import BaseMotifDetector


def revcomp(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]


def _generate_phased_repeat_patterns(base: str, num_tracts: int, tract_sizes: range, 
                                      id_start: int) -> List[Tuple[str, str, str, str, int, str, float, str, str]]:
    """
    Generate phased repeat patterns programmatically.
    
    # Output Pattern Structure:
    # | Field       | Type  | Description                           |
    # |-------------|-------|---------------------------------------|
    # | pattern     | str   | Regex pattern for matching            |
    # | pattern_id  | str   | Unique pattern identifier             |
    # | name        | str   | Human-readable pattern name           |
    # | subclass    | str   | Motif subclass                        |
    # | min_length  | int   | Minimum expected match length         |
    # | score_type  | str   | Scoring method identifier             |
    # | threshold   | float | Quality threshold (0-1)               |
    # | description | str   | Pattern description                   |
    # | reference   | str   | Literature reference                  |
    
    Args:
        base: Base nucleotide ('A' or 'T')
        num_tracts: Number of tracts (3, 4, or 5)
        tract_sizes: Range of tract sizes (e.g., range(3, 10))
        id_start: Starting ID number for pattern naming
    
    Returns:
        List of 9-field pattern tuples
    """
    patterns = []
    label = 'APR' if base == 'A' else 'TPR'
    scores = {3: 0.90, 4: 0.92, 5: 0.95}
    min_lens = {3: 20, 4: 25, 5: 30}
    
    pattern_id = id_start
    for size in tract_sizes:
        spacing_min = max(0, 11 - size)
        spacing_max = min(8, 11 - size + 2)
        
        # Build pattern using f-string for clarity
        repeat_unit = f'{base}{{{size}}}[ACGT]{{{spacing_min},{spacing_max}}}'
        pattern = f'(?:{repeat_unit * (num_tracts - 1)}{base}{{{size}}})'
        
        name = f'{base}{size}-{label}' + (f'-{num_tracts}' if num_tracts > 3 else '')
        patterns.append((
            pattern,
            f'CRV_{pattern_id:03d}',
            name,
            'Global Curvature',
            min_lens[num_tracts],
            'phasing_score',
            scores[num_tracts],
            f'{num_tracts}-tract {label}',
            'Koo 1986'
        ))
        pattern_id += 1
    
    return patterns


class CurvedDNADetector(BaseMotifDetector):
    """
    Detects curved DNA motifs using A-tract and T-tract phasing patterns.
    
    # Motif Structure:
    # | Field           | Type  | Description                        |
    # |-----------------|-------|------------------------------------|
    # | Class           | str   | Always 'Curved_DNA'                |
    # | Subclass        | str   | 'Local Curvature' or 'Global Curvature' |
    # | Start           | int   | 1-based start position             |
    # | End             | int   | End position (inclusive)           |
    # | Score           | float | Phasing or curvature score (0-1)   |
    # | Tract_Type      | str   | 'A-tract' or 'T-tract' (local)     |
    # | Center_Positions| list  | Tract center positions (global)    |
    """
    
    def get_motif_class_name(self) -> str:
        return "Curved_DNA"

    # Tunable parameters
    MIN_AT_TRACT = 3
    MAX_AT_WINDOW = None
    PHASING_CENTER_SPACING = 11.0
    PHASING_TOL_LOW = 9.9
    PHASING_TOL_HIGH = 11.1
    MIN_APR_TRACTS = 3
    LOCAL_LONG_TRACT = 7

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Generate curved DNA patterns programmatically (reduces ~65 lines to ~20).
        """
        patterns = {
            'local_curved': [
                (r'A{7,}', 'CRV_002', 'Long A-tract', 'Local Curvature', 7, 
                 'curvature_score', 0.95, 'A-tract curvature', 'Olson 1998'),
                (r'T{7,}', 'CRV_003', 'Long T-tract', 'Local Curvature', 7, 
                 'curvature_score', 0.95, 'T-tract curvature', 'Olson 1998'),
            ]
        }
        
        # Generate A-phased repeats programmatically (3-, 4-, and 5-tract)
        tract_range = range(3, 10)  # A3..A9
        patterns['global_curved_a_3tract'] = _generate_phased_repeat_patterns('A', 3, tract_range, 8)
        patterns['global_curved_a_4tract'] = _generate_phased_repeat_patterns('A', 4, tract_range, 15)
        patterns['global_curved_a_5tract'] = _generate_phased_repeat_patterns('A', 5, tract_range, 22)
        
        # Generate T-phased repeats programmatically
        patterns['global_curved_t_3tract'] = _generate_phased_repeat_patterns('T', 3, tract_range, 29)
        patterns['global_curved_t_4tract'] = _generate_phased_repeat_patterns('T', 4, tract_range, 36)
        patterns['global_curved_t_5tract'] = _generate_phased_repeat_patterns('T', 5, tract_range, 43)
        
        return patterns

    def _remove_overlaps(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove overlapping motifs within the same subclass.
        Allows overlaps between different subclasses (Global vs Local curvature).
        """
        if not motifs:
            return []
        
        from collections import defaultdict
        
        # Group by subclass
        groups = defaultdict(list)
        for motif in motifs:
            subclass = motif.get('Subclass', 'unknown')
            groups[subclass].append(motif)
        
        non_overlapping = []
        
        # Process each subclass separately
        for subclass, group_motifs in groups.items():
            # Sort by score (descending), then by length (descending)
            sorted_motifs = sorted(group_motifs, 
                                  key=lambda x: (-x.get('Score', 0), -x.get('Length', 0)))
            
            selected = []
            for motif in sorted_motifs:
                # Check if this motif overlaps with any already selected in this subclass
                overlaps = False
                for selected_motif in selected:
                    if not (motif['End'] <= selected_motif['Start'] or 
                           motif['Start'] >= selected_motif['End']):
                        overlaps = True
                        break
                
                if not overlaps:
                    selected.append(motif)
            
            non_overlapping.extend(selected)
        
        # Sort by start position for output
        non_overlapping.sort(key=lambda x: x['Start'])
        return non_overlapping

    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """Override base method to use sophisticated curved DNA detection with component details"""
        sequence = sequence.upper().strip()
        motifs = []
        
        # Use the sophisticated annotation method
        annotation = self.annotate_sequence(sequence)
        
        # Extract APR (A-phased repeat) motifs
        for i, apr in enumerate(annotation.get('aprs', [])):
            if apr.get('score', 0) > 0.1:  # Lower threshold for sensitivity
                start_pos = int(min(apr['center_positions'])) - 10  # Estimate start
                end_pos = int(max(apr['center_positions'])) + 10    # Estimate end
                start_pos = max(0, start_pos)
                end_pos = min(len(sequence), end_pos)
                
                motif_seq = sequence[start_pos:end_pos]
                
                # Extract A-tracts (components)
                a_tracts = re.findall(r'A{3,}', motif_seq)
                t_tracts = re.findall(r'T{3,}', motif_seq)
                
                # Calculate GC content
                gc_total = (motif_seq.count('G') + motif_seq.count('C')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
                at_content = (motif_seq.count('A') + motif_seq.count('T')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
                
                motifs.append({
                    'ID': f"{sequence_name}_CRV_APR_{start_pos+1}",
                    'Sequence_Name': sequence_name,
                    'Class': self.get_motif_class_name(),
                    'Subclass': 'Global Curvature',
                    'Start': start_pos + 1,  # 1-based coordinates
                    'End': end_pos,
                    'Length': end_pos - start_pos,
                    'Sequence': motif_seq,
                    'Score': round(apr.get('score', 0), 3),
                    'Strand': '+',
                    'Method': 'Curved_DNA_detection',
                    'Pattern_ID': f'CRV_APR_{i+1}',
                    # Component details
                    'A_Tracts': a_tracts,
                    'T_Tracts': t_tracts,
                    'Num_A_Tracts': len(a_tracts),
                    'Num_T_Tracts': len(t_tracts),
                    'A_Tract_Lengths': [len(t) for t in a_tracts],
                    'T_Tract_Lengths': [len(t) for t in t_tracts],
                    'GC_Content': round(gc_total, 2),
                    'AT_Content': round(at_content, 2),
                    'Center_Positions': apr.get('center_positions', [])
                })
        
        # Extract long tract motifs
        for i, tract in enumerate(annotation.get('long_tracts', [])):
            if tract.get('score', 0) > 0.1:  # Lower threshold for sensitivity
                start_pos = tract['start']
                end_pos = tract['end']
                motif_seq = sequence[start_pos:end_pos]
                
                # Identify tract type
                tract_type = 'A-tract' if motif_seq.count('A') > motif_seq.count('T') else 'T-tract'
                
                # Calculate GC content
                gc_total = (motif_seq.count('G') + motif_seq.count('C')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
                at_content = (motif_seq.count('A') + motif_seq.count('T')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
                
                motifs.append({
                    'ID': f"{sequence_name}_CRV_TRACT_{start_pos+1}",
                    'Sequence_Name': sequence_name,
                    'Class': self.get_motif_class_name(),
                    'Subclass': 'Local Curvature',
                    'Start': start_pos + 1,  # 1-based coordinates
                    'End': end_pos,
                    'Length': end_pos - start_pos,
                    'Sequence': motif_seq,
                    'Score': round(tract.get('score', 0), 3),
                    'Strand': '+',
                    'Method': 'Curved_DNA_detection',
                    'Pattern_ID': f'CRV_TRACT_{i+1}',
                    # Component details
                    'Tract_Type': tract_type,
                    'Tract_Length': end_pos - start_pos,
                    'GC_Content': round(gc_total, 2),
                    'AT_Content': round(at_content, 2)
                })
        
        # Remove overlaps within each subclass
        motifs = self._remove_overlaps(motifs)
        
        return motifs

    # -------------------------
    # Top-level scoring API
    # -------------------------
    def calculate_score(self, sequence: str, pattern_info: Tuple = None) -> float:
        """
        Returns a combined raw score reflecting:
          - phasing_score for APRs (sum of APR phasing scores)
          - local curvature contribution (sum of local A/T tract scores)
        The sum reflects both number and quality of hits.
        """
        seq = sequence.upper()
        ann = self.annotate_sequence(seq)
        # Sum APR scores
        apr_sum = sum(a['score'] for a in ann.get('aprs', []))
        local_sum = sum(l['score'] for l in ann.get('long_tracts', []))
        return float(apr_sum + local_sum)

    # -------------------------
    # A-tract detection (core)
    # -------------------------
    def find_a_tracts(self, sequence: str, minAT: int = None, max_window: int = None) -> List[Dict[str, Any]]:
        """
        Detect A-tract candidates across the sequence using logic adapted from your C code.

        Returns list of dicts:
          {
            'start', 'end'            : region bounds of the AT-window inspected (0-based, end-exclusive)
            'maxATlen'                : maximal A/AnTn length found inside window
            'maxTlen'                 : maximal T-only run length (not following A)
            'maxATend'                : index (0-based) of end position(where maxATlen ends) relative to full seq (inclusive end index)
            'a_center'                : float center coordinate (1-based in C code; here 0-based float)
            'call'                    : bool whether (maxATlen - maxTlen) >= minAT
            'window_len'              : length of AT window
            'window_seq'              : the AT-window substring
          }

        Implementation detail:
        - We scan for contiguous runs of A/T (AT windows). Within each, we iterate positions and
          compute Alen/Tlen/ATlen per the C algorithm to determine maxATlen and maxTlen.
        - If either forward strand or reverse complement has (maxATlen - maxTlen) >= minAT, we call it an A-tract.
        """
        seq = sequence.upper()
        len(seq)
        if minAT is None:
            minAT = self.MIN_AT_TRACT
        if max_window is None:
            max_window = self.MAX_AT_WINDOW  # None allowed

        results: List[Dict[str, Any]] = []

        # Find contiguous A/T windows (length >= minAT)
        for m in re.finditer(r'[AT]{' + str(minAT) + r',}', seq):
            wstart, wend = m.start(), m.end()  # [wstart, wend)
            window_seq = seq[wstart:wend]
            window_len = wend - wstart

            # analyze forward strand window
            maxATlen, maxATend, maxTlen = self._analyze_at_window(window_seq)
            # analyze reverse complement window (to mimic C code check on reverse)
            rc_window = revcomp(window_seq)
            maxATlen_rc, maxATend_rc, maxTlen_rc = self._analyze_at_window(rc_window)

            # compute decisions - apply same logic as C code:
            diff_forward = maxATlen - maxTlen
            diff_rc = maxATlen_rc - maxTlen_rc
            call = False
            chosen_center = None
            chosen_maxATlen = None

            if diff_forward >= minAT or diff_rc >= minAT:
                call = True
                # choose the strand giving larger difference
                if diff_forward >= diff_rc:
                    chosen_maxATlen = maxATlen
                    # compute center coordinate in full sequence (0-based float center)
                    # in C code: a_center = maxATend - ((maxATlen-1)/2) + 1  (1-based)
                    # we'll produce 0-based center = (wstart + maxATend - ((maxATlen-1)/2))
                    chosen_center = (wstart + maxATend) - ((maxATlen - 1) / 2.0)
                else:
                    chosen_maxATlen = maxATlen_rc
                    # maxATend_rc is position in RC sequence; convert to original coords:
                    # RC index i corresponds to original index: wstart + (window_len - 1 - i)
                    # maxATend_rc is index in window_rc (end position index)
                    # In C, they convert similarly; for simplicity compute center via rc mapping
                    # rc_end_original = wstart + (window_len - 1 - i_rc)
                    rc_end_original = wstart + (window_len - 1 - maxATend_rc)
                    chosen_center = rc_end_original - ((chosen_maxATlen - 1) / 2.0)

            results.append({
                'start': wstart,
                'end': wend,
                'window_len': window_len,
                'window_seq': window_seq,
                'maxATlen': int(maxATlen),
                'maxATend': int(wstart + maxATend),
                'maxTlen': int(maxTlen),
                'maxATlen_rc': int(maxATlen_rc),
                'maxATend_rc': int(wstart + (window_len - 1 - maxATend_rc)),
                'maxTlen_rc': int(maxTlen_rc),
                'diff_forward': int(diff_forward),
                'diff_rc': int(diff_rc),
                'call': bool(call),
                'a_center': float(chosen_center) if chosen_center is not None else None,
                'chosen_maxATlen': int(chosen_maxATlen) if chosen_maxATlen is not None else None
            })

        return results

    def _analyze_at_window(self, window_seq: str) -> Tuple[int,int,int]:
        """
        Analyze a contiguous A/T window and return (maxATlen, maxATend_index_in_window, maxTlen)
        Implemented following the logic in your C code:
         - iterate positions; update Alen, Tlen, ATlen, TAlen; track maxATlen, maxTlen and their end positions.
         - maxATend returned as index (0-based) *within the window* of the last position of the max AT run.
        """
        Alen = 0
        Tlen = 0
        ATlen = 0
        TAlen = 0
        maxATlen = 0
        maxTlen = 0
        maxATend = 0
        # we'll iterate from index 0..len(window_seq)-1
        L = len(window_seq)
        # to mimic C code scanning with lookbacks, we iterate straightforwardly
        for i in range(L):
            ch = window_seq[i]
            prev = window_seq[i-1] if i>0 else None
            if ch == 'A':
                Tlen = 0
                TAlen = 0
                # if previous base was T, reset A-run counters per C code
                if prev == 'T':
                    Alen = 1
                    ATlen = 1
                else:
                    Alen += 1
                    ATlen += 1
            elif ch == 'T':
                # if T follows A-run shorter than Alen, it's considered TAlen (T following A)
                if TAlen < Alen:
                    TAlen += 1
                    ATlen += 1
                else:
                    # T is starting a T-only run
                    Tlen += 1
                    TAlen = 0
                    ATlen = 0
                    Alen = 0
            else:
                # non-AT not expected inside this window (we only pass contiguous AT windows)
                Alen = 0
                Tlen = 0
                ATlen = 0
                TAlen = 0
            if ATlen > maxATlen:
                maxATlen = ATlen
                maxATend = i  # end index within window
            if Tlen > maxTlen:
                maxTlen = Tlen
        return int(maxATlen), int(maxATend), int(maxTlen)

    # -------------------------
    # APR grouping / phasing
    # -------------------------
    def find_aprs(self, sequence: str, min_tract: int = None, min_apr_tracts: int = None) -> List[Dict[str, Any]]:
        """
        Group a-tract centers into APRs (A-phased repeats).
        Criteria:
          - at least min_apr_tracts centers
          - consecutive center-to-center spacing must be within PHASING_TOL_LOW..PHASING_TOL_HIGH
            (we allow flexible grouping: we slide through centers looking for runs of centers that satisfy spacing)
        Returns list of dicts:
          { 'start_center_idx', 'end_center_idx', 'centers': [...], 'center_positions': [...], 'score': phasing_score, 'n_tracts' }
        """
        if min_tract is None:
            min_tract = self.MIN_AT_TRACT
        if min_apr_tracts is None:
            min_apr_tracts = self.MIN_APR_TRACTS

        # get a-tract calls
        a_calls = [r for r in self.find_a_tracts(sequence, minAT=min_tract) if r['call'] and r['a_center'] is not None]
        centers = [r['a_center'] for r in a_calls]
        centers_sorted = sorted(centers)
        aprs: List[Dict[str, Any]] = []

        if len(centers_sorted) < min_apr_tracts:
            return aprs

        # find runs of centers where consecutive spacing is within tolerance
        i = 0
        while i < len(centers_sorted):
            run = [centers_sorted[i]]
            j = i + 1
            while j < len(centers_sorted):
                spacing = centers_sorted[j] - centers_sorted[j-1]
                if self.PHASING_TOL_LOW <= spacing <= self.PHASING_TOL_HIGH:
                    run.append(centers_sorted[j])
                    j += 1
                else:
                    break
            # if run has enough tracts, call APR
            if len(run) >= min_apr_tracts:
                # score APR by how close spacings are to ideal spacing
                spacings = [run[k+1] - run[k] for k in range(len(run)-1)]
                # closeness = product of gaussian-like terms, but simpler: average deviation
                devs = [abs(sp - self.PHASING_CENTER_SPACING) for sp in spacings]
                # normalized closeness
                mean_dev = sum(devs) / len(devs) if devs else 0.0
                # phasing_score between 0..1: 1 when mean_dev==0, drop linearly with dev up to tolerance
                max_dev_allowed = max(abs(self.PHASING_TOL_HIGH - self.PHASING_CENTER_SPACING),
                                      abs(self.PHASING_CENTER_SPACING - self.PHASING_TOL_LOW))
                phasing_score = max(0.0, 1.0 - (mean_dev / (max_dev_allowed if max_dev_allowed>0 else 1.0)))
                aprs.append({
                    'start_center_idx': i,
                    'end_center_idx': j-1,
                    'center_positions': run,
                    'n_tracts': len(run),
                    'spacings': spacings,
                    'mean_deviation': mean_dev,
                    'score': round(phasing_score, 6)
                })
            i = j

        return aprs

    # -------------------------
    # Local long tract finder
    # -------------------------
    def find_long_tracts(self, sequence: str, min_len: int = None) -> List[Dict[str, Any]]:
        """
        Finds long A-tracts or T-tracts with length >= min_len (default LOCAL_LONG_TRACT).
        Returns list of dicts: {start,end,base,len,score} with score derived from len.
        """
        if min_len is None:
            min_len = self.LOCAL_LONG_TRACT
        seq = sequence.upper()
        results = []
        # A runs
        for m in re.finditer(r'A{' + str(min_len) + r',}', seq):
            ln = m.end() - m.start()
            # simple local score: normalized by (len/(len+6)) to saturate
            score = float(ln) / (ln + 6.0)
            results.append({'start': m.start(), 'end': m.end(), 'base': 'A', 'len': ln, 'score': round(score, 6), 'seq': seq[m.start():m.end()]})
        # T runs
        for m in re.finditer(r'T{' + str(min_len) + r',}', seq):
            ln = m.end() - m.start()
            score = float(ln) / (ln + 6.0)
            results.append({'start': m.start(), 'end': m.end(), 'base': 'T', 'len': ln, 'score': round(score, 6), 'seq': seq[m.start():m.end()]})
        # sort by start
        results.sort(key=lambda x: x['start'])
        return results

    # -------------------------
    # Scoring helpers (interpretability)
    # -------------------------
    def phasing_score(self, apr: Dict[str, Any]) -> float:
        """Return APR phasing score (already stored in apr['score'])."""
        return float(apr.get('score', 0.0))

    def local_curvature_score(self, tract: Dict[str, Any]) -> float:
        """Return local curvature score for a long tract (already stored)."""
        return float(tract.get('score', 0.0))

    # -------------------------
    # Annotate (summary)
    # -------------------------
    def annotate_sequence(self, sequence: str) -> Dict[str, Any]:
        """
        Returns comprehensive annotation:
         - a_tract_windows: raw outputs from find_a_tracts
         - aprs: list of APRs with phasing scores
         - long_tracts: list of local A/T long tracts
         - summary counts and combined score
        """
        seq = sequence.upper()
        a_windows = self.find_a_tracts(seq, minAT=self.MIN_AT_TRACT)
        # filtered called a-tract centers
        a_centers = [w for w in a_windows if w['call'] and w['a_center'] is not None]
        aprs = self.find_aprs(seq, min_tract=self.MIN_AT_TRACT, min_apr_tracts=self.MIN_APR_TRACTS)
        long_tracts = self.find_long_tracts(seq, min_len=self.LOCAL_LONG_TRACT)

        # annotate aprs with constituent windows (optional)
        for apr in aprs:
            apr['constituent_windows'] = []
            for center in apr['center_positions']:
                # find closest a_window with that center
                best = min(a_windows, key=lambda w: abs((w['a_center'] or 1e9) - center))
                apr['constituent_windows'].append(best)

        summary = {
            'n_a_windows': len(a_windows),
            'n_a_centers': len(a_centers),
            'n_aprs': len(aprs),
            'n_long_tracts': len(long_tracts),
            'apr_score_sum': sum(self.phasing_score(a) for a in aprs),
            'long_tract_score_sum': sum(self.local_curvature_score(l) for l in long_tracts),
            'combined_score': sum(self.phasing_score(a) for a in aprs) + sum(self.local_curvature_score(l) for l in long_tracts)
        }

        return {
            'a_tract_windows': a_windows,
            'a_centers': a_centers,
            'aprs': aprs,
            'long_tracts': long_tracts,
            'summary': summary
        }


# =============================================================================
# Z Dna Detector
# =============================================================================
"""
Z-DNA Motif Detector (10-mer table)
==================================

Detects Z-DNA-like 10-mer motifs using a provided motif -> score table.

MERGING GUARANTEE:
------------------
This detector ALWAYS outputs MERGED REGIONS, never individual 10-mers.
All overlapping or adjacent 10-mer matches are automatically merged into 
contiguous regions, ensuring no duplicate or split reporting.

Behavior:
 - Use Hyperscan (if available) for very fast matching of the 10-mer list.
 - Fallback to a pure-Python exact matcher if Hyperscan isn't installed.
 - **CRITICAL**: Merge overlapping/adjacent 10-mer matches into contiguous regions.
 - Redistribute each 10-mer's score evenly across its 10 bases (score/10 per base).
 - Region sum_score = sum of per-base contributions inside merged region.
 - calculate_score returns total sum_score across merged regions.
 - annotate_sequence(...) returns detailed merged region-level results.
 - detect_motifs(...) returns motif entries for merged regions meeting thresholds.

Output Structure (from detect_motifs and annotate_sequence):
- start, end: Region boundaries (0-based, end-exclusive)
- length: Region length in bp
- sequence: Full sequence of the merged region
- n_10mers: Number of contributing 10-mer matches
- score (sum_score): Sum of per-base score contributions across the region
- mean_score_per10mer: Average of the individual 10-mer scores
"""

import re
from typing import List, Dict, Any, Tuple
# # from .base_detector import BaseMotifDetector

# Optional Hyperscan for high-performance pattern matching
import logging
logger = logging.getLogger(__name__)

_HYPERSCAN_AVAILABLE = False
_HYPERSCAN_VERSION = None
_HYPERSCAN_ERROR = None

try:
    import hyperscan
    _HYPERSCAN_AVAILABLE = True
    # Try to get version, but don't fail if not available
    try:
        _HYPERSCAN_VERSION = hyperscan.__version__
    except AttributeError:
        _HYPERSCAN_VERSION = 'unknown'
    logger.info(f"Hyperscan loaded successfully (version: {_HYPERSCAN_VERSION})")
except ImportError as e:
    _HYPERSCAN_ERROR = f"Hyperscan Python bindings not installed: {e}"
    logger.info(f"Hyperscan not available - using pure Python fallback. {_HYPERSCAN_ERROR}")
except Exception as e:
    _HYPERSCAN_ERROR = f"Hyperscan initialization failed: {e}"
    logger.warning(f"Hyperscan not available - using pure Python fallback. {_HYPERSCAN_ERROR}")


class ZDNADetector(BaseMotifDetector):
    """
    Detector for Z-DNA (left-handed helix) motifs using 10-mer scoring.
    
    # Motif Structure:
    # | Field      | Type  | Description                         |
    # |------------|-------|-------------------------------------|
    # | Class      | str   | Always 'Z_DNA'                      |
    # | Subclass   | str   | 'Classic_Z_DNA' or 'eGZ'            |
    # | Start      | int   | 1-based start position              |
    # | End        | int   | End position                        |
    # | Score      | float | Z-DNA propensity score              |
    # | Length     | int   | Motif length in bp                  |
    # | GC_Content | float | GC% (Z-DNA favors high GC)          |
    """

    # eGZ-motif detection constants
    MIN_EGZ_REPEATS = 3  # Minimum number of trinucleotide repeats required
    EGZ_BASE_SCORE = 0.85  # Base score for eGZ-motif detection
    EGZ_MIN_SCORE_THRESHOLD = 0.80  # Minimum score threshold for reporting eGZ motifs

    # Full 10-mer scoring table from Ho et al. 1986
    TENMER_SCORE: Dict[str, float] = {
        "AACGCGCGCG": 50.25,
        "ATGCGCGCGC": 51.25,
        "ATCGCGCGCG": 50.0,
        "AGCGCGCGCA": 50.25,
        "AGCGCGCGCG": 56.0,
        "ACGGCGCGCG": 50.25,
        "ACGCGGCGCG": 50.25,
        "ACGCGCGGCG": 50.25,
        "ACGCGCGCGA": 50.25,
        "ACGCGCGCGT": 51.5,
        "ACGCGCGCGG": 50.25,
        "ACGCGCGCGC": 57.25,
        "ACGCGCGCCG": 50.25,
        "ACGCGCCGCG": 50.25,
        "ACGCCGCGCG": 50.25,
        "ACCGCGCGCG": 50.25,
        "TAGCGCGCGC": 50.0,
        "TACGCGCGCG": 51.25,
        "TTGCGCGCGC": 50.25,
        "TGGCGCGCGC": 50.25,
        "TGCGGCGCGC": 50.25,
        "TGCGCGGCGC": 50.25,
        "TGCGCGCGGC": 50.25,
        "TGCGCGCGCA": 51.5,
        "TGCGCGCGCT": 50.25,
        "TGCGCGCGCG": 57.25,
        "TGCGCGCGCC": 50.25,
        "TGCGCGCCGC": 50.25,
        "TGCGCCGCGC": 50.25,
        "TGCCGCGCGC": 50.25,
        "TCGCGCGCGT": 50.25,
        "TCGCGCGCGC": 56.0,
        "GACGCGCGCG": 50.25,
        "GTGCGCGCGC": 51.5,
        "GTCGCGCGCG": 50.25,
        "GGCGCGCGCA": 50.25,
        "GGCGCGCGCG": 56.0,
        "GCAGCGCGCG": 50.25,
        "GCACGCGCGC": 51.5,
        "GCTGCGCGCG": 50.25,
        "GCGACGCGCG": 50.25,
        "GCGTGCGCGC": 51.5,
        "GCGTCGCGCG": 50.25,
        "GCGGCGCGCA": 50.25,
        "GCGGCGCGCG": 56.0,
        "GCGCAGCGCG": 50.25,
        "GCGCACGCGC": 51.5,
        "GCGCTGCGCG": 50.25,
        "GCGCGACGCG": 50.25,
        "GCGCGTGCGC": 51.5,
        "GCGCGTCGCG": 50.25,
        "GCGCGGCGCA": 50.25,
        "GCGCGGCGCG": 56.0,
        "GCGCGCAGCG": 50.25,
        "GCGCGCACGC": 51.5,
        "GCGCGCTGCG": 50.25,
        "GCGCGCGACG": 50.25,
        "GCGCGCGTGC": 51.5,
        "GCGCGCGTCG": 50.25,
        "GCGCGCGGCA": 50.25,
        "GCGCGCGGCG": 56.0,
        "GCGCGCGCAA": 50.25,
        "GCGCGCGCAT": 51.25,
        "GCGCGCGCAG": 50.25,
        "GCGCGCGCAC": 51.5,
        "GCGCGCGCTA": 50.0,
        "GCGCGCGCTG": 50.25,
        "GCGCGCGCGA": 56.0,
        "GCGCGCGCGT": 57.25,
        "GCGCGCGCGG": 56.0,
        "GCGCGCGCGC": 63.0,
        "GCGCGCGCCA": 50.25,
        "GCGCGCGCCG": 56.0,
        "GCGCGCCGCA": 50.25,
        "GCGCGCCGCG": 56.0,
        "GCGCCGCGCA": 50.25,
        "GCGCCGCGCG": 56.0,
        "GCCGCGCGCA": 50.25,
        "GCCGCGCGCG": 56.0,
        "CAGCGCGCGC": 50.25,
        "CACGCGCGCG": 51.5,
        "CTGCGCGCGC": 50.25,
        "CGACGCGCGC": 50.25,
        "CGTGCGCGCG": 51.5,
        "CGTCGCGCGC": 50.25,
        "CGGCGCGCGT": 50.25,
        "CGGCGCGCGC": 56.0,
        "CGCAGCGCGC": 50.25,
        "CGCACGCGCG": 51.5,
        "CGCTGCGCGC": 50.25,
        "CGCGACGCGC": 50.25,
        "CGCGTGCGCG": 51.5,
        "CGCGTCGCGC": 50.25,
        "CGCGGCGCGT": 50.25,
        "CGCGGCGCGC": 56.0,
        "CGCGCAGCGC": 50.25,
        "CGCGCACGCG": 51.5,
        "CGCGCTGCGC": 50.25,
        "CGCGCGACGC": 50.25,
        "CGCGCGTGCG": 51.5,
        "CGCGCGTCGC": 50.25,
        "CGCGCGGCGT": 50.25,
        "CGCGCGGCGC": 56.0,
        "CGCGCGCAGC": 50.25,
        "CGCGCGCACG": 51.5,
        "CGCGCGCTGC": 50.25,
        "CGCGCGCGAT": 50.0,
        "CGCGCGCGAC": 50.25,
        "CGCGCGCGTA": 51.25,
        "CGCGCGCGTT": 50.25,
        "CGCGCGCGTG": 51.5,
        "CGCGCGCGTC": 50.25,
        "CGCGCGCGGT": 50.25,
        "CGCGCGCGGC": 56.0,
        "CGCGCGCGCA": 57.25,
        "CGCGCGCGCT": 56.0,
        "CGCGCGCGCG": 63.0,
        "CGCGCGCGCC": 56.0,
        "CGCGCGCCGT": 50.25,
        "CGCGCGCCGC": 56.0,
        "CGCGCCGCGT": 50.25,
        "CGCGCCGCGC": 56.0,
        "CGCCGCGCGT": 50.25,
        "CGCCGCGCGC": 56.0,
        "CCGCGCGCGT": 50.25,
        "CCGCGCGCGC": 56.0,
    }

    def get_motif_class_name(self) -> str:
        return "Z-DNA"

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Return patterns for Z-DNA detection.
        
        Includes:
          * 10-mer scoring table (classic Z-DNA)
          * eGZ-motif patterns: long (CGG)n, (GGC)n, (CCG)n, (GCC)n runs
        
        eGZ (Extruded-G Z-DNA) patterns search for trinucleotide repeats
        that form Z-DNA structures with guanine extrusion.
        
        Patterns are loaded from the centralized motif_patterns module.
        """
        # Load patterns from centralized module
        if ZDNA_PATTERNS:
            patterns = ZDNA_PATTERNS.copy()
            # Update EGZ_BASE_SCORE in patterns if needed
            if 'egz_motifs' in patterns:
                updated_egz = []
                for pattern in patterns['egz_motifs']:
                    # Update base score to use instance variable
                    updated = list(pattern)
                    updated[6] = self.EGZ_BASE_SCORE
                    updated_egz.append(tuple(updated))
                patterns['egz_motifs'] = updated_egz
            return patterns
        else:
            # Fallback patterns if import failed
            return {
                "z_dna_10mers": [
                    (r"", "ZDN_10MER", "Z-DNA 10-mer table", "Z-DNA", 10, "z_dna_10mer_score", 0.9, "Z-DNA 10mer motif", "user_table"),
                ],
                "egz_motifs": [
                    (r"(?:CGG){3,}", "ZDN_EGZ_CGG", "CGG repeat (eGZ)", "eGZ", 9, "egz_score", self.EGZ_BASE_SCORE, "Extruded-G Z-DNA CGG repeat", "Herbert 1997"),
                    (r"(?:GGC){3,}", "ZDN_EGZ_GGC", "GGC repeat (eGZ)", "eGZ", 9, "egz_score", self.EGZ_BASE_SCORE, "Extruded-G Z-DNA GGC repeat", "Herbert 1997"),
                    (r"(?:CCG){3,}", "ZDN_EGZ_CCG", "CCG repeat (eGZ)", "eGZ", 9, "egz_score", self.EGZ_BASE_SCORE, "Extruded-G Z-DNA CCG repeat", "Herbert 1997"),
                    (r"(?:GCC){3,}", "ZDN_EGZ_GCC", "GCC repeat (eGZ)", "eGZ", 9, "egz_score", self.EGZ_BASE_SCORE, "Extruded-G Z-DNA GCC repeat", "Herbert 1997"),
                ]
            }

    # -------------------------
    # Public API
    # -------------------------
    def calculate_score(self, sequence: str, pattern_info: Tuple) -> float:
        """
        Return total sum_score across merged Z-like regions in sequence.
        sum_score is computed by redistributing each matched 10-mer's score equally across its 10 bases
        and summing per-base contributions inside merged regions.
        """
        seq = sequence.upper()
        merged = self._find_and_merge_10mer_matches(seq)
        if not merged:
            return 0.0
        contrib = self._build_per_base_contrib(seq)
        total = 0.0
        for s, e in merged:
            total += sum(contrib[s:e])
        return float(total)

    def annotate_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Return list of merged region annotations.
        
        Detects both:
          1. Classic Z-DNA regions (from 10-mer scoring)
          2. eGZ-motif regions (from regex patterns)
        
        MERGING GUARANTEE: This method ALWAYS merges overlapping/adjacent 10-mer 
        matches into contiguous regions. No individual 10-mers are returned; only 
        merged regions covering all contributing 10-mer matches.
        
        Each dict contains:
          - start, end (0-based, end-exclusive)
          - length
          - sum_score (sum of per-base contributions for 10-mers OR repeat count score for eGZ)
          - subclass: 'Z-DNA' or 'eGZ'
          - pattern_id: identifier for the pattern
          - (for eGZ) repeat_unit, repeat_count
          - (for Z-DNA) mean_score_per10mer, n_10mers, contributing_10mers
        """
        seq = sequence.upper()
        annotations = []
        
        # Step 1: Find classic Z-DNA regions from 10-mer matches
        matches = self._find_10mer_matches(seq)
        if matches:
            # Merge overlapping/adjacent matches into regions
            merged = self._merge_matches(matches)
            
            # Build per-base contribution array for scoring
            contrib = self._build_per_base_contrib(seq)
            
            # Create annotation for each merged region
            for (s, e, region_matches) in merged:
                # Sum contributions across the merged region
                sum_score = sum(contrib[s:e])
                n10 = len(region_matches)
                mean10 = (sum(m[2] for m in region_matches) / n10) if n10 > 0 else 0.0
                
                # Build merged region annotation
                ann = {
                    "start": s,
                    "end": e,
                    "length": e - s,
                    "sum_score": round(sum_score, 6),
                    "mean_score_per10mer": round(mean10, 6),
                    "n_10mers": n10,
                    "contributing_10mers": [{"tenmer": m[1], "start": m[0], "score": m[2]} for m in region_matches],
                    "subclass": "Z-DNA",
                    "pattern_id": "ZDN_10MER"
                }
                annotations.append(ann)
        
        # Step 2: Find eGZ-motif regions using regex patterns
        egz_annotations = self._find_egz_motifs(seq)
        annotations.extend(egz_annotations)
        
        # Step 3: Sort by start position
        annotations.sort(key=lambda x: x['start'])
        
        return annotations

    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Override base method to use sophisticated Z-DNA detection with component details.
        
        Detects both:
          1. Classic Z-DNA regions (from 10-mer scoring) - labeled as 'Z-DNA' subclass
          2. eGZ-motif regions (from regex patterns) - labeled as 'eGZ' subclass
        
        IMPORTANT: This method ALWAYS outputs merged regions, not individual 10-mers.
        All overlapping/adjacent 10-mer matches are merged into contiguous regions
        via annotate_sequence(), ensuring no duplicate or split reporting.
        
        Returns:
            List of motif dictionaries, each representing a Z-DNA or eGZ region
            with start, end, length, sequence, score, and component information.
        """
        sequence = sequence.upper().strip()
        motifs = []
        
        # Use the annotation method to find Z-DNA regions.
        # This GUARANTEES that overlapping/adjacent 10-mer matches are merged.
        annotations = self.annotate_sequence(sequence)
        
        for i, region in enumerate(annotations):
            subclass = region.get('subclass', 'Z-DNA')
            
            # Handle eGZ-motif regions
            if subclass == 'eGZ':
                # Filter by minimal score threshold
                if region.get('sum_score', 0) >= self.EGZ_MIN_SCORE_THRESHOLD:
                    start_pos = region['start']
                    end_pos = region['end']
                    motif_seq = sequence[start_pos:end_pos]
                    
                    # Calculate GC content
                    gc_content = (motif_seq.count('G') + motif_seq.count('C')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
                    
                    motifs.append({
                        'ID': f"{sequence_name}_{region['pattern_id']}_{start_pos+1}",
                        'Sequence_Name': sequence_name,
                        'Class': self.get_motif_class_name(),
                        'Subclass': 'eGZ',
                        'Start': start_pos + 1,  # 1-based coordinates
                        'End': end_pos,
                        'Length': region['length'],
                        'Sequence': motif_seq,
                        'Score': round(region['sum_score'], 3),
                        'Strand': '+',
                        'Method': 'Z-DNA_detection',
                        'Pattern_ID': region['pattern_id'],
                        # eGZ-specific details
                        'Repeat_Unit': region.get('repeat_unit', ''),
                        'Repeat_Count': region.get('repeat_count', 0),
                        'GC_Content': round(gc_content, 2)
                    })
            else:
                # Handle classic Z-DNA regions (from 10-mer scoring)
                # Filter by meaningful score threshold
                if region.get('sum_score', 0) > 50.0 and region.get('n_10mers', 0) >= 1:
                    start_pos = region['start']
                    end_pos = region['end']
                    motif_seq = sequence[start_pos:end_pos]
                    
                    # Extract CG/AT dinucleotides (characteristic of Z-DNA)
                    cg_count = motif_seq.count('CG') + motif_seq.count('GC')
                    at_count = motif_seq.count('AT') + motif_seq.count('TA')
                    
                    # Calculate GC content
                    gc_content = (motif_seq.count('G') + motif_seq.count('C')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
                    
                    # Extract alternating pattern information
                    alternating_cg = len(re.findall(r'(?:CG){2,}', motif_seq)) + len(re.findall(r'(?:GC){2,}', motif_seq))
                    alternating_at = len(re.findall(r'(?:AT){2,}', motif_seq)) + len(re.findall(r'(?:TA){2,}', motif_seq))
                    
                    motifs.append({
                        'ID': f"{sequence_name}_ZDNA_{start_pos+1}",
                        'Sequence_Name': sequence_name,
                        'Class': self.get_motif_class_name(),
                        'Subclass': 'Z-DNA',
                        'Start': start_pos + 1,  # 1-based coordinates
                        'End': end_pos,
                        'Length': region['length'],
                        'Sequence': motif_seq,
                        'Score': round(region['sum_score'], 3),
                        'Strand': '+',
                        'Method': 'Z-DNA_detection',
                        'Pattern_ID': f'ZDNA_{i+1}',
                        # Component details
                        'Contributing_10mers': region.get('n_10mers', 0),
                        'Mean_10mer_Score': region.get('mean_score_per10mer', 0),
                        'CG_Dinucleotides': cg_count,
                        'AT_Dinucleotides': at_count,
                        'Alternating_CG_Regions': alternating_cg,
                        'Alternating_AT_Regions': alternating_at,
                        'GC_Content': round(gc_content, 2)
                    })
        
        return motifs

    # -------------------------
    # Core helpers
    # -------------------------
    def _find_egz_motifs(self, seq: str) -> List[Dict[str, Any]]:
        """
        Find eGZ-motif (Extruded-G Z-DNA) patterns using regex.
        
        Searches for long runs of (CGG)n, (GGC)n, (CCG)n, (GCC)n repeats.
        Minimum MIN_EGZ_REPEATS repeats (9 bp) are required for detection.
        
        Returns list of annotation dicts with:
          - start, end (0-based, end-exclusive)
          - length
          - sum_score (based on repeat count)
          - subclass: 'eGZ'
          - pattern_id: pattern identifier
          - repeat_unit: the trinucleotide being repeated
          - repeat_count: number of repeats found
        """
        patterns = self.get_patterns().get('egz_motifs', [])
        annotations = []
        
        for pattern_info in patterns:
            regex, pattern_id, name, subclass, min_len, score_type, threshold, desc, ref = pattern_info
            
            # Find all matches for this pattern
            for match in re.finditer(regex, seq, re.IGNORECASE):
                start, end = match.span()
                matched_seq = seq[start:end]
                
                # Determine repeat unit (first 3 characters of matched sequence)
                repeat_unit = matched_seq[:3].upper()
                repeat_count = len(matched_seq) // 3
                
                # Calculate score based on repeat count
                # Score increases with more repeats: base_score * (repeat_count / MIN_EGZ_REPEATS)
                # Minimum MIN_EGZ_REPEATS repeats gives score = base_score, more repeats give higher scores
                repeat_score = threshold * (repeat_count / float(self.MIN_EGZ_REPEATS))
                
                ann = {
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "sum_score": round(repeat_score, 6),
                    "subclass": "eGZ",
                    "pattern_id": pattern_id,
                    "repeat_unit": repeat_unit,
                    "repeat_count": repeat_count,
                    "description": desc,
                    "reference": ref
                }
                annotations.append(ann)
        
        return annotations
    
    def _find_10mer_matches(self, seq: str) -> List[Tuple[int, str, float]]:
        """
        Find all exact 10-mer matches in the sequence.
        
        Returns list of (start, tenmer, score) tuples for each match found.
        Uses Hyperscan if available for high performance; otherwise falls back
        to pure-Python scanning. Matches may overlap (e.g., positions 0,1,2...).
        
        Note: This method finds ALL matches, including overlapping ones.
        Merging into regions happens in _merge_matches().
        """
        if _HYPERSCAN_AVAILABLE:
            try:
                return self._hs_find_matches(seq)
            except Exception as e:
                logger.warning(f"Hyperscan matching failed for Z-DNA, falling back to pure Python: {e}")
                return self._py_find_matches(seq)
        else:
            return self._py_find_matches(seq)

    def _hs_find_matches(self, seq: str) -> List[Tuple[int, str, float]]:
        """
        Hyperscan-based matching with improved error handling.
        
        Returns list of (start, tenmer, score) sorted by start position.
        """
        try:
            expressions = []
            ids = []
            id_to_ten = {}
            id_to_score = {}
            for idx, (ten, score) in enumerate(self.TENMER_SCORE.items()):
                expressions.append(ten.encode())
                ids.append(idx)
                id_to_ten[idx] = ten
                id_to_score[idx] = float(score)
            
            db = hyperscan.Database()
            db.compile(expressions=expressions, ids=ids, elements=len(expressions))
            logger.debug(f"Hyperscan database compiled successfully for Z-DNA ({len(expressions)} patterns)")
            
            matches: List[Tuple[int, str, float]] = []

            def on_match(id, start, end, flags, context):
                # Hyperscan 'end' parameter is the offset after the match
                # Calculate actual start position: end - pattern_length
                actual_start = end - 10
                matches.append((actual_start, id_to_ten[id], id_to_score[id]))

            db.scan(seq.encode(), match_event_handler=on_match)
            matches.sort(key=lambda x: x[0])
            logger.debug(f"Hyperscan scan completed: {len(matches)} Z-DNA 10-mer matches found")
            return matches
            
        except Exception as e:
            logger.error(f"Hyperscan matching failed for Z-DNA: {e}. Falling back to pure Python.")
            raise  # Re-raise to trigger fallback in _find_10mer_matches

    def _py_find_matches(self, seq: str) -> List[Tuple[int, str, float]]:
        """Pure-Python exact search (overlapping matches allowed)."""
        n = len(seq)
        matches: List[Tuple[int, str, float]] = []
        for i in range(0, n - 10 + 1):
            ten = seq[i:i + 10]
            score = self.TENMER_SCORE.get(ten)
            if score is not None:
                matches.append((i, ten, float(score)))
        return matches

    def _merge_matches(self, matches: List[Tuple[int, str, float]],
                       merge_gap: int = 0) -> List[Tuple[int, int, List[Tuple[int, str, float]]]]:
        """
        Merge overlapping/adjacent 10-mer matches into contiguous regions.
        
        This is the CORE MERGING LOGIC that ensures no duplicate or split reporting
        of overlapping 10-mers. All 10-mers that overlap or are within merge_gap 
        bases of each other are combined into a single region.
        
        Args:
            matches: List of (start, tenmer, score) tuples, sorted by start position
            merge_gap: Maximum gap (in bases) between matches to still merge them.
                      Default 0 means only overlapping/adjacent matches are merged.
        
        Returns:
            List of (region_start, region_end, list_of_matches_in_region) tuples.
            region_end is exclusive (Python slice convention).
        
        Example:
            If 10-mers at positions 0, 1, 2 all match (overlapping by 9bp each),
            they will be merged into ONE region [0, 12) containing all 3 matches.
        """
        if not matches:
            return []
        
        merged = []
        # Initialize with first match
        cur_start = matches[0][0]
        cur_end = matches[0][0] + 10
        cur_matches = [matches[0]]
        
        # Iterate through remaining matches and merge if overlapping/adjacent
        for m in matches[1:]:
            s = m[0]
            m_end = s + 10
            
            # Check if this match overlaps or is within merge_gap of current region
            if s <= cur_end + merge_gap:
                # Extend current region and add match
                cur_end = max(cur_end, m_end)
                cur_matches.append(m)
            else:
                # Gap too large; finalize current region and start new one
                merged.append((cur_start, cur_end, cur_matches))
                cur_start, cur_end = s, m_end
                cur_matches = [m]
        
        # Don't forget to add the last region
        merged.append((cur_start, cur_end, cur_matches))
        return merged

    def _find_and_merge_10mer_matches(self, seq: str, merge_gap: int = 0) -> List[Tuple[int, int]]:
        matches = self._find_10mer_matches(seq)
        merged = self._merge_matches(matches, merge_gap=merge_gap)
        return [(s, e) for (s, e, _) in merged]

    def _build_per_base_contrib(self, seq: str) -> List[float]:
        """
        Build per-base contribution array for the sequence.
        
        For each matched 10-mer starting at position j with score S,
        we distribute S equally across its 10 bases by adding S/10 to 
        positions j, j+1, ..., j+9 in the contribution array.
        
        This redistribution allows us to compute region scores as the sum
        of per-base contributions, properly accounting for overlapping matches.
        
        Returns:
            List of floats of length len(seq), where contrib[i] is the total
            contribution from all 10-mers covering position i.
        """
        n = len(seq)
        contrib = [0.0] * n
        matches = self._find_10mer_matches(seq)
        
        # Distribute each 10-mer's score across its 10 bases
        for (start, ten, score) in matches:
            per_base = float(score) / 10.0
            for k in range(start, min(start + 10, n)):
                contrib[k] += per_base
        
        return contrib


# =============================================================================
# A Philic Detector
# =============================================================================
"""
A-philic DNA Motif Detector
===========================

Detects A-philic 10-mer motifs using a provided 10-mer -> avg_log2 table.

MERGING GUARANTEE:
------------------
This detector ALWAYS outputs MERGED REGIONS, never individual 10-mers.
All overlapping or adjacent 10-mer matches are automatically merged into 
contiguous regions, ensuring no duplicate or split reporting.

Behavior:
- Use Hyperscan (if available) for blazing-fast exact matching of the 10-mer list.
- Fallback to a pure-Python exact matcher if Hyperscan isn't installed.
- **CRITICAL**: Merge overlapping/adjacent 10-mer matches into contiguous regions.
- Redistribute each 10-mer's avg_log2 evenly across its 10 bases (L/10 per base)
  and sum per-base contributions inside merged regions to compute a region sum score.
- calculate_score(sequence, pattern_info) returns the raw sum_log2 (float) for the
  merged regions inside `sequence`.
- annotate_sequence(...) returns detailed merged region-level results.
- detect_motifs(...) returns motif entries for merged regions meeting thresholds.

Output Structure (from detect_motifs and annotate_sequence):
- start, end: Region boundaries (0-based, end-exclusive)
- length: Region length in bp
- sequence: Full sequence of the merged region
- n_10mers: Number of contributing 10-mer matches
- score (sum_log2): Sum of per-base log2 contributions across the region
- mean_log2_per10mer: Average of the individual 10-mer log2 values
"""

import re
from typing import List, Dict, Any, Tuple, Optional
# # from .base_detector import BaseMotifDetector

# Note: Hyperscan availability is checked once at module initialization (above)
# to avoid redundant imports and maintain consistent state across the module.


class APhilicDetector(BaseMotifDetector):
    """Detector for A-philic DNA motifs using a 10-mer scoring table."""

    # -------------------------
    # Full provided 10-mer -> avg_log2 table
    # -------------------------
    TENMER_LOG2: Dict[str, float] = {
        "AGGGGGGGGG": 2.702428571428571,
        "CCCCCCCCCT": 2.702428571428571,
        "AGGGGGGGGC": 2.683285714285714,
        "GCCCCCCCCT": 2.683285714285714,
        "CCCCCCCCTA": 2.5665571428571425,
        "TAGGGGGGGG": 2.5665571428571425,
        "GCCCCCCCTA": 2.5474142857142854,
        "TAGGGGGGGC": 2.5474142857142854,
        "ACCCCCCCCT": 2.4612714285714286,
        "AGGGGGGGGT": 2.4612714285714286,
        "AGGGCCCCCT": 2.487357142857143,
        "AGGGGCCCCT": 2.487357142857143,
        "AGGGGGCCCT": 2.487357142857143,
        "ACCCCCCCTA": 2.3253999999999997,
        "TAGGGGGGGT": 2.3253999999999997,
        "ACCCCCCCCC": 2.294942857142857,
        "GGGGGGGGGT": 2.294942857142857,
        "CCCCCCCCCG": 2.294942857142857,
        "CGGGGGGGGG": 2.294942857142857,
        "AGGGCCCCTA": 2.3514857142857144,
        "AGGGGCCCTA": 2.3514857142857144,
        "TAGGGCCCCT": 2.3514857142857144,
        "TAGGGGCCCT": 2.3514857142857144,
        "AGGGGGGGCC": 2.3401714285714283,
        "GGCCCCCCCT": 2.3401714285714283,
        "CGGGGGGGGC": 2.2758,
        "GCCCCCCCCG": 2.2758,
        "ACCCCCCCCA": 2.2284142857142855,
        "TGGGGGGGGT": 2.2284142857142855,
        "AGGGCCCCCC": 2.321028571428571,
        "AGGGGCCCCC": 2.321028571428571,
        "AGGGGGCCCC": 2.321028571428571,
        "AGGGGGGCCC": 2.321028571428571,
        "GGGCCCCCCT": 2.321028571428571,
        "GGGGCCCCCT": 2.321028571428571,
        "GGGGGCCCCT": 2.321028571428571,
        "GGGGGGCCCT": 2.321028571428571,
        "AGGGCCCCCA": 2.2544999999999997,
        "AGGGGCCCCA": 2.2544999999999997,
        "AGGGGGCCCA": 2.2544999999999997,
        "TGGGCCCCCT": 2.2544999999999997,
        "TGGGGCCCCT": 2.2544999999999997,
        "TGGGGGCCCT": 2.2544999999999997,
        "TAGGGCCCTA": 2.2156142857142855,
        "GGGCCCCCTA": 2.185157142857143,
        "GGGGCCCCTA": 2.185157142857143,
        "GGGGGCCCTA": 2.185157142857143,
        "TAGGGCCCCC": 2.185157142857143,
        "TAGGGGCCCC": 2.185157142857143,
        "TAGGGGGCCC": 2.185157142857143,
        "GGCCCCCCCC": 2.173842857142857,
        "GGGGGGGGCC": 2.173842857142857,
        "GGGCCCCCCC": 2.1546999999999996,
        "GGGGCCCCCC": 2.1546999999999996,
        "GGGGGCCCCC": 2.1546999999999996,
        "GGGGGGCCCC": 2.1546999999999996,
        "GGGGGGGCCC": 2.1546999999999996,
        "TAGGGCCCCA": 2.1186285714285713,
        "TAGGGGCCCA": 2.1186285714285713,
        "TGGGCCCCTA": 2.1186285714285713,
        "TGGGGCCCTA": 2.1186285714285713,
        "GGCCCCCCCA": 2.1073142857142857,
        "TGGGGGGGCC": 2.1073142857142857,
        "GGGCCCCCCA": 2.0881714285714286,
        "GGGGCCCCCA": 2.0881714285714286,
        "GGGGGCCCCA": 2.0881714285714286,
        "GGGGGGCCCA": 2.0881714285714286,
        "TGGGCCCCCC": 2.0881714285714286,
        "TGGGGCCCCC": 2.0881714285714286,
        "TGGGGGCCCC": 2.0881714285714286,
        "TGGGGGGCCC": 2.0881714285714286,
        "ACCCCCCCCG": 2.053785714285714,
        "CGGGGGGGGT": 2.053785714285714,
        "TGGGCCCCCA": 2.021642857142857,
        "TGGGGCCCCA": 2.021642857142857,
        "TGGGGGCCCA": 2.021642857142857,
        "AGGGCCCCCG": 2.0798714285714284,
        "AGGGGCCCCG": 2.0798714285714284,
        "AGGGGGCCCG": 2.0798714285714284,
        "CGGGCCCCCT": 2.0798714285714284,
        "CGGGGCCCCT": 2.0798714285714284,
        "CGGGGGCCCT": 2.0798714285714284,
        "CCCCCCCCGG": 1.9696571428571428,
        "CCGGGGGGGG": 1.9696571428571428,
        "CCGGGGGGGC": 1.9505142857142856,
        "GCCCCCCCGG": 1.9505142857142856,
        "CGGGCCCCTA": 1.9440000000000002,
        "CGGGGCCCTA": 1.9440000000000002,
        "TAGGGCCCCG": 1.9440000000000002,
        "TAGGGGCCCG": 1.9440000000000002,
        "CGGGGGGGCC": 1.9326857142857141,
        "GGCCCCCCCG": 1.9326857142857141,
        "CGGGCCCCCC": 1.9135428571428572,
        "CGGGGCCCCC": 1.9135428571428572,
        "CGGGGGCCCC": 1.9135428571428572,
        "CGGGGGGCCC": 1.9135428571428572,
        "GGGCCCCCCG": 1.9135428571428572,
        "GGGGCCCCCG": 1.9135428571428572,
        "GGGGGCCCCG": 1.9135428571428572,
        "GGGGGGCCCG": 1.9135428571428572,
        "CGGGCCCCCA": 1.8470142857142857,
        "CGGGGCCCCA": 1.8470142857142857,
        "CGGGGGCCCA": 1.8470142857142857,
        "TGGGCCCCCG": 1.8470142857142857,
        "TGGGGCCCCG": 1.8470142857142857,
        "TGGGGGCCCG": 1.8470142857142857,
        "ACCCCCCCGG": 1.7285,
        "CCGGGGGGGT": 1.7285,
        "CCCCCCCGGG": 1.7285,
        "CCCCCCGGGG": 1.7285,
        "CCCCCGGGGG": 1.7285,
        "CCCCGGGGGG": 1.7285,
        "CCCGGGGGGG": 1.7285,
        "CCCCCCGGGC": 1.7093571428571426,
        "CCCCCGGGGC": 1.7093571428571426,
        "CCCCGGGGGC": 1.7093571428571426,
        "CCCGGGGGGC": 1.7093571428571426,
        "GCCCCCCGGG": 1.7093571428571426,
        "GCCCCCGGGG": 1.7093571428571426,
        "GCCCCGGGGG": 1.7093571428571426,
        "GCCCGGGGGG": 1.7093571428571426,
        "AGGGCCCCGG": 1.7545857142857142,
        "AGGGGCCCGG": 1.7545857142857142,
        "CCGGGCCCCT": 1.7545857142857142,
        "CCGGGGCCCT": 1.7545857142857142,
        "GCCCCCGGGC": 1.6902142857142857,
        "GCCCCGGGGC": 1.6902142857142857,
        "GCCCGGGGGC": 1.6902142857142857,
        "CCGGGCCCTA": 1.6187142857142856,
        "TAGGGCCCGG": 1.6187142857142856,
        "CCGGGGGGCC": 1.6074,
        "GGCCCCCCGG": 1.6074,
        "CCGGGCCCCC": 1.5882571428571428,
        "CCGGGGCCCC": 1.5882571428571428,
        "CCGGGGGCCC": 1.5882571428571428,
        "GGGCCCCCGG": 1.5882571428571428,
        "GGGGCCCCGG": 1.5882571428571428,
        "GGGGGCCCGG": 1.5882571428571428,
        "CCGGGCCCCA": 1.5217285714285713,
        "CCGGGGCCCA": 1.5217285714285713,
        "TGGGCCCCGG": 1.5217285714285713,
        "TGGGGCCCGG": 1.5217285714285713,
        "ACCCCCCGGG": 1.4873428571428569,
        "ACCCCCGGGG": 1.4873428571428569,
        "ACCCCGGGGG": 1.4873428571428569,
        "ACCCGGGGGG": 1.4873428571428569,
        "CCCCCCGGGT": 1.4873428571428569,
        "CCCCCGGGGT": 1.4873428571428569,
        "CCCCGGGGGT": 1.4873428571428569,
        "CCCGGGGGGT": 1.4873428571428569,
        "ACCCCCGGGC": 1.4682,
        "ACCCCGGGGC": 1.4682,
        "ACCCGGGGGC": 1.4682,
        "GCCCCCGGGT": 1.4682,
        "GCCCCGGGGT": 1.4682,
        "GCCCGGGGGT": 1.4682,
        "AGGGCCCGGG": 1.5134285714285713,
        "CCCGGGCCCT": 1.5134285714285713,
        "CCCCCGGGCC": 1.366242857142857,
        "CCCCGGGGCC": 1.366242857142857,
        "CCCGGGGGCC": 1.366242857142857,
        "GGCCCCCGGG": 1.366242857142857,
        "GGCCCCGGGG": 1.366242857142857,
        "GGCCCGGGGG": 1.366242857142857,
        "CCCCGGGCCC": 1.3471,
        "CCCGGGCCCC": 1.3471,
        "CCCGGGGCCC": 1.3471,
        "CCGGGCCCCG": 1.3471,
        "CCGGGGCCCG": 1.3471,
        "CGGGCCCCGG": 1.3471,
        "CGGGGCCCGG": 1.3471,
        "GCCCCGGGCC": 1.3471,
        "GCCCGGGGCC": 1.3471,
        "GGCCCCGGGC": 1.3471,
        "GGCCCGGGGC": 1.3471,
        "GGGCCCCGGG": 1.3471,
        "GGGCCCGGGG": 1.3471,
        "GGGGCCCGGG": 1.3471,
        "GCCCGGGCCC": 1.3279571428571428,
        "GGGCCCGGGC": 1.3279571428571428,
        "ACCCCCGGGT": 1.2461857142857142,
        "ACCCCGGGGT": 1.2461857142857142,
        "ACCCGGGGGT": 1.2461857142857142,
        "CCCGGGCCCA": 1.2805714285714287,
        "TGGGCCCGGG": 1.2805714285714287,
        "ACCCCGGGCC": 1.1250857142857142,
        "ACCCGGGGCC": 1.1250857142857142,
        "GGCCCCGGGT": 1.1250857142857142,
        "GGCCCGGGGT": 1.1250857142857142,
        "ACCCGGGCCC": 1.1059428571428571,
        "GGGCCCGGGT": 1.1059428571428571,
        "CCCGGGCCCG": 1.1059428571428571,
        "CGGGCCCGGG": 1.1059428571428571,
        "CCGGGCCCGG": 1.0218142857142856,
        "GGCCCGGGCC": 1.0039857142857143,
        "CCCCCCCCCC": 2.5361,
        "GGGGGGGGGG": 2.5361,
        "GCCCCCCCCC": 2.5169571428571422,
        "GGGGGGGGGC": 2.5169571428571422,
        "CCCCCCCCCA": 2.4695714285714283,
        "TGGGGGGGGG": 2.4695714285714283,
        "GCCCCCCCCA": 2.450428571428571,
        "TGGGGGGGGC": 2.450428571428571,
        "GGCCCCCCTA": 2.2043,
        "TAGGGGGGCC": 2.2043,
        "CGGGCCCCCG": 1.6723857142857141,
        "CGGGGCCCCG": 1.6723857142857141,
        "CGGGGGCCCG": 1.6723857142857141,
    }

    def get_motif_class_name(self) -> str:
        return "A-philic_DNA"

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Keep compatibility with framework: return a single synthetic "pattern"
        that indicates the detector uses the TENMER_LOG2 table.  The detection
        itself will use hyperscan / Python matching inside calculate_score
        and annotate_sequence.
        """
        return {
            "a_philic_10mers": [
                (r"", "APH_10MER", "A-philic 10-mer table", "A-philic DNA",
                 10, "a_philic_10mer_score", 0.9, "A-philic 10mer motif", "user_table"),
            ]
        }

    # -------------------------
    # Public API: calculate_score used by framework
    # -------------------------
    def calculate_score(self, sequence: str, pattern_info: Tuple) -> float:
        """
        Calculate and return the **raw** sum of log2 odds for merged A-philic regions
        found in `sequence`. This value is the **sum_log2** obtained by redistributing
        each 10-mer's avg_log2 equally over its 10 bases and summing per-base values
        inside merged regions.
        """
        seq = sequence.upper()
        merged_regions = self._find_and_merge_10mer_matches(seq)
        if not merged_regions:
            return 0.0
        contrib = self._build_per_base_contrib(seq)
        total_sum = 0.0
        for s, e in merged_regions:
            total_sum += sum(contrib[s:e])
        return float(total_sum)

    # -------------------------
    # Helper: return detailed annotation
    # -------------------------
    def annotate_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Return list of region annotations (merged regions).
        
        MERGING GUARANTEE: This method ALWAYS merges overlapping/adjacent 10-mer 
        matches into contiguous regions. No individual 10-mers are returned; only 
        merged regions covering all contributing 10-mer matches.
        
        Each dict contains:
          - start, end (0-based, end-exclusive)
          - length (bp)
          - sum_log2 (raw sum of redistributed contributions)
          - mean_log2_per10mer (sum of tenmer avg_log2 / n_10mers)
          - n_10mers: number of matched 10-mers contributing
          - contributing_10mers: list of (tenmer, start, log2)
        """
        seq = sequence.upper()
        
        # Step 1: Find all 10-mer matches (may overlap)
        matches = self._find_10mer_matches(seq)
        if not matches:
            return []
        
        # Step 2: Merge overlapping/adjacent matches into regions
        # This is the critical step that ensures no duplicate/split reporting
        merged = self._merge_matches(matches)
        
        # Step 3: Build per-base contribution array for scoring
        contrib = self._build_per_base_contrib(seq)
        
        # Step 4: Create annotation for each merged region
        annotations = []
        for region in merged:
            s, e, region_matches = region
            # Sum contributions across the merged region
            sum_log2 = sum(contrib[s:e])
            n_10 = len(region_matches)
            mean_per10 = (sum(m[2] for m in region_matches) / n_10) if n_10 > 0 else 0.0
            
            # Build merged region annotation
            ann = {
                "start": s,
                "end": e,
                "length": e - s,
                "sum_log2": round(sum_log2, 6),
                "mean_log2_per10mer": round(mean_per10, 6),
                "n_10mers": n_10,
                "contributing_10mers": [{"tenmer": m[1], "start": m[0], "log2": m[2]} for m in region_matches]
            }
            annotations.append(ann)
        return annotations

    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Override base method to use sophisticated A-philic detection.
        
        IMPORTANT: This method ALWAYS outputs merged regions, not individual 10-mers.
        All overlapping/adjacent 10-mer matches are merged into contiguous regions
        via annotate_sequence(), ensuring no duplicate or split reporting.
        
        Returns:
            List of motif dictionaries, each representing a merged A-philic region
            with start, end, length, sequence, score, and contributing 10-mer count.
        """
        sequence = sequence.upper().strip()
        motifs = []
        
        # Use the annotation method to find A-philic regions.
        # This GUARANTEES that overlapping/adjacent 10-mer matches are merged.
        annotations = self.annotate_sequence(sequence)
        
        for i, region in enumerate(annotations):
            # Filter by meaningful score threshold - lowered for better sensitivity
            if region.get('sum_log2', 0) > 0.5 and region.get('n_10mers', 0) >= 1:
                start_pos = region['start']
                end_pos = region['end']
                
                motifs.append({
                    'ID': f"{sequence_name}_APHIL_{start_pos+1}",
                    'Sequence_Name': sequence_name,
                    'Class': self.get_motif_class_name(),
                    'Subclass': 'A-philic DNA',
                    'Start': start_pos + 1,  # 1-based coordinates
                    'End': end_pos,
                    'Length': region['length'],
                    'Sequence': sequence[start_pos:end_pos],
                    'Score': round(region['sum_log2'], 3),
                    'Strand': '+',
                    'Method': 'A-philic_detection',
                    'Pattern_ID': f'APHIL_{i+1}'
                })
        
        return motifs

    # -------------------------
    # Core match / merge / contrib helpers
    # -------------------------
    def _find_10mer_matches(self, seq: str) -> List[Tuple[int, str, float]]:
        """
        Find all exact 10-mer matches in the sequence.
        
        Returns list of (start, tenmer, avg_log2) tuples for each match found.
        Uses Hyperscan if available for high performance; otherwise falls back
        to pure-Python scanning. Matches may overlap (e.g., positions 0,1,2...).
        
        Note: This method finds ALL matches, including overlapping ones.
        Merging into regions happens in _merge_matches().
        """
        if _HYPERSCAN_AVAILABLE:
            try:
                return self._hs_find_matches(seq)
            except Exception as e:
                logger.warning(f"Hyperscan matching failed for A-philic DNA, falling back to pure Python: {e}")
                return self._py_find_matches(seq)
        else:
            return self._py_find_matches(seq)

    def _hs_find_matches(self, seq: str) -> List[Tuple[int, str, float]]:
        """
        Use Hyperscan compiled database of 10-mers to find matches quickly.
        Returns list of (start, tenmer, log2) sorted by start.
        
        Improved error handling and logging.
        """
        try:
            expressions = []
            ids = []
            id_to_ten = {}
            id_to_log2 = {}
            for idx, (ten, log2) in enumerate(self.TENMER_LOG2.items()):
                expressions.append(ten.encode())
                ids.append(idx)
                id_to_ten[idx] = ten
                id_to_log2[idx] = float(log2)
            
            db = hyperscan.Database()
            db.compile(expressions=expressions, ids=ids, elements=len(expressions))
            logger.debug(f"Hyperscan database compiled successfully for A-philic DNA ({len(expressions)} patterns)")
            
            matches: List[Tuple[int, str, float]] = []

            def on_match(id, start, end, flags, context):
                ten = id_to_ten[id]
                log2 = id_to_log2[id]
                # Hyperscan 'end' parameter is the offset after the match
                # Calculate actual start position: end - pattern_length
                actual_start = end - 10
                matches.append((actual_start, ten, log2))

            db.scan(seq.encode(), match_event_handler=on_match)
            matches.sort(key=lambda x: x[0])
            logger.debug(f"Hyperscan scan completed: {len(matches)} A-philic 10-mer matches found")
            return matches
            
        except Exception as e:
            logger.error(f"Hyperscan matching failed for A-philic DNA: {e}. Falling back to pure Python.")
            raise  # Re-raise to trigger fallback in _find_10mer_matches

    def _py_find_matches(self, seq: str) -> List[Tuple[int, str, float]]:
        """Pure-Python exact search (overlapping matches included)"""
        n = len(seq)
        matches: List[Tuple[int, str, float]] = []
        for i in range(0, n - 10 + 1):
            ten = seq[i:i + 10]
            log2 = self.TENMER_LOG2.get(ten)
            if log2 is not None:
                matches.append((i, ten, float(log2)))
        return matches

    def _merge_matches(self, matches: List[Tuple[int, str, float]],
                       merge_gap: int = 0) -> List[Tuple[int, int, List[Tuple[int, str, float]]]]:
        """
        Merge overlapping/adjacent 10-mer matches into contiguous regions.
        
        This is the CORE MERGING LOGIC that ensures no duplicate or split reporting
        of overlapping 10-mers. All 10-mers that overlap or are within merge_gap 
        bases of each other are combined into a single region.
        
        Args:
            matches: List of (start, tenmer, log2) tuples, sorted by start position
            merge_gap: Maximum gap (in bases) between matches to still merge them.
                      Default 0 means only overlapping/adjacent matches are merged.
        
        Returns:
            List of (region_start, region_end, list_of_matches_in_region) tuples.
            region_end is exclusive (Python slice convention).
        
        Example:
            If 10-mers at positions 0, 1, 2 all match (overlapping by 9bp each),
            they will be merged into ONE region [0, 12) containing all 3 matches.
        """
        if not matches:
            return []
        
        merged = []
        # Initialize with first match
        cur_start, cur_end = matches[0][0], matches[0][0] + 10
        cur_matches = [matches[0]]
        
        # Iterate through remaining matches and merge if overlapping/adjacent
        for m in matches[1:]:
            s = m[0]
            m_end = s + 10
            
            # Check if this match overlaps or is within merge_gap of current region
            if s <= cur_end + merge_gap:
                # Extend current region and add match
                cur_end = max(cur_end, m_end)
                cur_matches.append(m)
            else:
                # Gap too large; finalize current region and start new one
                merged.append((cur_start, cur_end, cur_matches))
                cur_start, cur_end = s, m_end
                cur_matches = [m]
        
        # Don't forget to add the last region
        merged.append((cur_start, cur_end, cur_matches))
        return merged

    def _find_and_merge_10mer_matches(self, seq: str, merge_gap: int = 0) -> List[Tuple[int, int]]:
        matches = self._find_10mer_matches(seq)
        merged = self._merge_matches(matches, merge_gap=merge_gap)
        return [(s, e) for (s, e, _) in merged]

    def _build_per_base_contrib(self, seq: str) -> List[float]:
        """
        Build per-base contribution array for the sequence.
        
        For each matched 10-mer starting at position j with log2 value L,
        we distribute L equally across its 10 bases by adding L/10 to 
        positions j, j+1, ..., j+9 in the contribution array.
        
        This redistribution allows us to compute region scores as the sum
        of per-base contributions, properly accounting for overlapping matches.
        
        Returns:
            List of floats of length len(seq), where contrib[i] is the total
            contribution from all 10-mers covering position i.
        """
        n = len(seq)
        contrib = [0.0] * n
        matches = self._find_10mer_matches(seq)
        
        # Distribute each 10-mer's log2 value across its 10 bases
        for (start, ten, log2) in matches:
            per_base = float(log2) / 10.0
            for k in range(start, min(start + 10, n)):
                contrib[k] += per_base
        
        return contrib


# =============================================================================
# Slipped Dna Detector
# =============================================================================
"""
Slipped DNA Motif Detector (Optimized for Performance)
------------------------------------------------------
PERFORMANCE OPTIMIZATIONS:
- Uses optimized seed-and-extend k-mer index approach from repeat_scanner
- Efficient linear scanning for STRs
- Direct repeats detection with k-mer seeding (no catastrophic backtracking!)
- O(N) complexity for most operations
- Safe-guards to avoid explosion on highly-repetitive seeds

Detects and annotates complete repeat regions, following:
- STRs: Unit size 1–9 bp, ≥20 bp in span, non-overlapping, match full region
- Direct repeats: Unit length 10-300 bp, spacer <= 10 bp

References:
Wells 2005, Schlötterer 2000, Weber 1989, Verkerk 1991
"""

import re
from typing import List, Dict, Any, Tuple

# BaseMotifDetector is defined above
# Optimized scanner functions imported at module top

class SlippedDNADetector(BaseMotifDetector):
    """
    Mechanism-Driven Slipped DNA Detector (Publication-Grade, 2024)
    ================================================================
    
    Unified detector for slippage-prone DNA motifs based on experimental
    and theoretical evidence of slipped-strand DNA formation.
    
    SCIENTIFIC RATIONALE (Sinden; Pearson; Mirkin):
    ------------------------------------------------
    Slipped DNA structures arise ONLY when direct repeats are:
    - Long enough (≥20 bp total, ≥30 bp recommended for genomes)
    - Pure enough (≥90% repeat purity, minimal interruptions)
    - Register-ambiguous (multiple alignment possibilities)
    
    UNIFIED MODEL:
    --------------
    STRs (k=1-9) and Direct Repeats (k≥10) are treated as a single
    mechanistic class representing out-of-register re-annealing during
    replication.
    
    KEY IMPROVEMENTS:
    -----------------
    1. Stringent Entry Criteria (Hard Gates):
       - Minimum tract length: ≥20 bp (configurable)
       - Repeat purity: ≥0.90 for high-confidence calls
       - Copy number: ≥3 for STRs, ≥2 for direct repeats
       
    2. Redundancy Elimination (Critical Fix):
       - Computes primitive motif (irreducible repeat unit)
       - Selects dominant representation per locus
       - Max-k dominance: retains longest effective unit
       - ONE call per genomic locus (no STR_1...STR_9 spam)
       
    3. Mechanistic Slippage Scoring (1-3 scale):
       - Total repeat length (dominant factor)
       - Repeat purity (interruptions penalized)
       - Repeat unit size (k)
       - Copy number
       - NN-based ΔG proxy (stability heuristic)
       - Normalized to [1-3] consistent with all Non-B classes
    
    OUTPUT STRUCTURE (Simplified & Publication-Ready):
    ---------------------------------------------------
    | Field                  | Type  | Description                    |
    |------------------------|-------|--------------------------------|
    | Class                  | str   | Always 'Slipped_DNA'           |
    | Subclass               | str   | Always 'Slipped_DNA'           |
    | Start                  | int   | 1-based start position         |
    | End                    | int   | End position (inclusive)       |
    | Length                 | int   | Total tract length             |
    | Sequence               | str   | Full repeat tract              |
    | Repeat_Unit            | str   | Primitive motif                |
    | Unit_Size              | int   | Length of primitive unit (k)   |
    | Copy_Number            | float | Number of repeat copies        |
    | Purity                 | float | Repeat purity (0-1)            |
    | Slippage_Energy_Score  | float | Unified score (1-3)            |
    
    REFERENCES:
    -----------
    Sinden RR (1994) DNA Structure and Function
    Pearson CE et al. (2005) Nat Rev Genet
    Mirkin SM (2007) Nature
    """
    
    # Stringent slippage criteria (experimentally validated)
    MIN_TRACT_LENGTH = 20        # Minimum total tract length (configurable, ≥30 bp recommended)
    MIN_PURITY = 0.90            # Minimum repeat purity (90%)
    MIN_COPIES_STR = 3           # Minimum copies for STRs (k=1-9)
    MIN_COPIES_DIRECT = 2        # Minimum copies for direct repeats (k≥10)
    MAX_UNIT_SIZE = 50           # Maximum unit size to consider (OPTIMIZED: reduced from 100 to 50 for 2x speedup)
    
    def get_motif_class_name(self) -> str:
        return "Slipped_DNA"
    
    @staticmethod
    def compute_primitive_motif(sequence: str) -> str:
        """
        Compute the primitive (irreducible) repeat unit of a sequence.
        
        The primitive motif is the shortest substring that, when repeated,
        generates the full sequence (or as close as possible for partial repeats).
        
        Handles both perfect repeats and partial repeats at the end.
        
        Example:
            CAGCAGCAGCAG → primitive motif = CAG (not CAGCAG)
            ATATATATAT → primitive motif = AT (not ATAT or ATATAT)
            CAGCAGCA → primitive motif = CAG (handles partial repeat)
        
        Args:
            sequence: DNA sequence (uppercase)
            
        Returns:
            Primitive repeat unit (shortest non-reducible motif)
        """
        n = len(sequence)
        if n == 0:
            return ""
        
        # Try all possible periods from 1 to n/2 (a repeat must occur at least twice)
        for period in range(1, n // 2 + 1):
            unit = sequence[:period]
            
            # Check if this period can generate the sequence (with possible partial at end)
            is_primitive = True
            for i in range(0, n, period):
                # Check each position against the unit
                check_len = min(period, n - i)
                if sequence[i:i+check_len] != unit[:check_len]:
                    is_primitive = False
                    break
            
            if is_primitive:
                return unit
        
        # If no period found, the sequence itself is primitive
        return sequence
    
    @staticmethod
    def compute_repeat_purity(sequence: str, unit: str) -> float:
        """
        Compute repeat purity: fraction of sequence matching perfect repeats.
        
        Purity = (number of bases matching perfect repeat) / total length
        
        Interruptions (mismatches, insertions, deletions) reduce purity.
        
        Args:
            sequence: DNA sequence (uppercase)
            unit: Repeat unit (uppercase)
            
        Returns:
            Purity value (0-1), where 1.0 = perfect repeat
        """
        if not unit or not sequence:
            return 0.0
        
        unit_len = len(unit)
        seq_len = len(sequence)
        
        if unit_len > seq_len:
            return 0.0
        
        # Count matching bases when aligning unit repeatedly
        matches = 0
        for i in range(seq_len):
            if sequence[i] == unit[i % unit_len]:
                matches += 1
        
        return matches / seq_len
    
    @staticmethod
    def calculate_entropy(sequence: str) -> float:
        """
        Calculate Shannon entropy of a DNA sequence.
        
        Args:
            sequence: DNA sequence string (uppercase)
            
        Returns:
            Entropy value (0-2 bits for DNA, 2 = maximum complexity)
        """
        from math import log2
        if not sequence:
            return 0.0
        
        # Calculate base frequencies in a single pass (O(n) complexity)
        freq = {}
        for base in sequence:
            if base in "ACGT":
                freq[base] = freq.get(base, 0) + 1
        
        # Normalize to probabilities
        seq_len = sum(freq.values())
        if seq_len == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = -sum((count / seq_len) * log2(count / seq_len) 
                       for count in freq.values() if count > 0)
        return entropy

    def compute_slippage_energy_score(self, sequence: str, unit: str, 
                                      copy_number: float, purity: float) -> float:
        """
        Compute mechanistic slippage energy score (1-3 scale).
        
        Integrates multiple factors reflecting slip-out formation energy:
        1. Total repeat length (dominant factor)
        2. Repeat purity (interruptions penalized)
        3. Repeat unit size (k)
        4. Copy number
        5. NN-based ΔG proxy (stability heuristic)
        
        Score interpretation:
        - 1.0-1.5: Weak/conditional slippage (short tracts, low purity)
        - 1.5-2.5: Moderate slippage (typical disease-relevant repeats)
        - 2.5-3.0: Strong/high-confidence (long, pure, expansion-prone)
        
        Args:
            sequence: Full repeat tract
            unit: Primitive repeat unit
            copy_number: Number of repeat copies
            purity: Repeat purity (0-1)
            
        Returns:
            Slippage energy score normalized to [1-3]
        """
        import math
        
        tract_length = len(sequence)
        unit_size = len(unit)
        
        # Factor 1: Total tract length (dominant, scales logarithmically)
        # Longer tracts → more stable slip-outs
        length_factor = min(1.0, math.log(max(1, tract_length), 20) / 2.0)
        
        # Factor 2: Repeat purity (interruptions penalize stability)
        # High purity required for stable slip-out
        purity_factor = purity ** 2  # Quadratic penalty for impurity
        
        # Factor 3: Repeat unit size (k)
        # Longer units → more register ambiguity
        unit_factor = min(1.0, math.log(max(1, unit_size), 2) / 3.5)
        
        # Factor 4: Copy number
        # More copies → more alignment possibilities
        copy_factor = min(1.0, math.log(max(1, copy_number), 3) / 2.0)
        
        # Factor 5: NN-based ΔG proxy (simple GC content heuristic)
        # Higher GC → more stable base stacking
        gc_count = sequence.count('G') + sequence.count('C')
        gc_fraction = gc_count / len(sequence) if len(sequence) > 0 else 0.0
        stability_factor = 0.5 + 0.5 * gc_fraction  # Range [0.5, 1.0]
        
        # Composite score (weighted sum)
        raw_score = (
            0.35 * length_factor +
            0.30 * purity_factor +
            0.15 * unit_factor +
            0.10 * copy_factor +
            0.10 * stability_factor
        )
        
        # Normalize to [1-3] scale (ΔG-inspired: 1=weak, 2=moderate, 3=strong)
        # Map [0-1] raw_score to [1-3]
        normalized_score = 1.0 + (2.0 * raw_score)
        
        return min(3.0, max(1.0, normalized_score))

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        # Detection via optimized k-mer scanner
        # Keep patterns for metadata/compatibility but don't use for regex matching
        return {
            "short_tandem_repeats": [],
            "direct_repeats": []
        }

    def find_all_tandem_repeats(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Unified tandem repeat finder: detects all k-mer repeats (k=1 to MAX_UNIT_SIZE).
        
        Uses efficient algorithmic detection (no catastrophic backtracking).
        Finds all candidate repeat regions regardless of k, then applies
        stringent entry criteria and redundancy elimination.
        
        Returns:
            List of candidate repeat dictionaries with raw detection data
        """
        seq = sequence.upper()
        n = len(seq)
        candidates = []
        
        # Scan all possible unit sizes (k=1 to MAX_UNIT_SIZE)
        # OPTIMIZED: Skip by larger steps for faster scanning
        for k in range(1, min(self.MAX_UNIT_SIZE + 1, n // 2)):
            # Skip positions by k (repeat unit size) for faster scanning
            # Since we're looking for tandem repeats, we can skip ahead
            current_pos = 0
            while current_pos < n - k:
                unit = seq[current_pos:current_pos+k]
                
                # Skip units with ambiguous bases
                if 'N' in unit:
                    current_pos += k  # OPTIMIZED: Skip by k instead of 1
                    continue
                
                # Count consecutive copies of this unit
                copies = 1
                repeat_end_pos = current_pos + k
                while repeat_end_pos + k <= n and seq[repeat_end_pos:repeat_end_pos+k] == unit:
                    copies += 1
                    repeat_end_pos += k
                
                # Check if this tract meets minimum length
                tract_length = copies * k
                if tract_length >= self.MIN_TRACT_LENGTH:
                    # Record this candidate
                    candidates.append({
                        'start': current_pos,
                        'end': repeat_end_pos,
                        'length': tract_length,
                        'unit': unit,
                        'unit_size': k,
                        'copies': copies,
                        'sequence': seq[current_pos:repeat_end_pos]
                    })
                    
                    # Skip past this repeat to avoid overlapping detections
                    current_pos = repeat_end_pos
                else:
                    # OPTIMIZED: Skip by k/2 to balance coverage and speed
                    # For small k (1-3), still check every position
                    # For larger k, can skip more aggressively
                    current_pos += max(1, k // 2) if k > 3 else 1
        
        return candidates
    
    def apply_stringent_criteria(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply stringent entry criteria to filter candidates.
        
        Hard Gates (must pass ALL):
        1. Minimum tract length ≥ MIN_TRACT_LENGTH
        2. Repeat purity ≥ MIN_PURITY
        3. Copy number ≥ MIN_COPIES_STR (k<10) or MIN_COPIES_DIRECT (k≥10)
        4. Not low-complexity (entropy check)
        
        Returns:
            Filtered list of high-confidence slipped DNA candidates
        """
        filtered = []
        
        for cand in candidates:
            sequence = cand['sequence']
            copies = cand['copies']
            unit_size = cand['unit_size']
            
            # Gate 1: Minimum tract length (already checked in detection, but reconfirm)
            if cand['length'] < self.MIN_TRACT_LENGTH:
                continue
            
            # Gate 2: Compute primitive motif from full sequence (not just unit)
            # This ensures we get the true primitive even if detected unit is composite
            primitive_unit = self.compute_primitive_motif(sequence)
            purity = self.compute_repeat_purity(sequence, primitive_unit)
            
            if purity < self.MIN_PURITY:
                continue
            
            # Gate 3: Minimum copy number (recompute based on primitive unit)
            primitive_copies = len(sequence) / len(primitive_unit) if len(primitive_unit) > 0 else 0
            min_copies = self.MIN_COPIES_STR if len(primitive_unit) < 10 else self.MIN_COPIES_DIRECT
            if primitive_copies < min_copies:
                continue
            
            # Gate 4: Entropy check (exclude low-complexity)
            entropy = self.calculate_entropy(primitive_unit)
            if entropy < 0.5:  # Very low entropy threshold for homopolymers
                continue
            
            # Passed all gates - add to filtered list with enriched data
            cand['primitive_unit'] = primitive_unit
            cand['purity'] = purity
            cand['entropy'] = entropy
            cand['primitive_copies'] = primitive_copies  # Updated copy count based on primitive
            filtered.append(cand)
        
        return filtered
    
    def eliminate_redundancy(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Eliminate redundant calls for the same genomic locus.
        
        Max-k dominance: For overlapping candidates, retain only the one
        with the longest effective repeat unit (primitive motif).
        
        This resolves the problem of a single locus producing STR_1, STR_2,
        ..., STR_9 calls simultaneously.
        
        Returns:
            Non-redundant list with one call per genomic locus
        """
        if not candidates:
            return []
        
        # Sort by start position, then by unit size (descending for max-k preference)
        sorted_cands = sorted(candidates, key=lambda c: (c['start'], -len(c['primitive_unit'])))
        
        non_redundant = []
        used_intervals = []  # Track (start, end) of accepted calls
        
        for cand in sorted_cands:
            start, end = cand['start'], cand['end']
            
            # Check if this overlaps with any already-accepted call
            overlaps = False
            for (used_start, used_end) in used_intervals:
                # Two intervals overlap if neither is completely before the other
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                non_redundant.append(cand)
                used_intervals.append((start, end))
        
        return non_redundant
    
    def annotate_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Mechanism-driven slipped DNA detection pipeline.
        
        Pipeline stages:
        1. Unified tandem repeat detection (all k-values)
        2. Apply stringent entry criteria (hard gates)
        3. Eliminate redundancy (max-k dominance)
        4. Compute mechanistic slippage scores
        
        Returns:
            List of non-redundant, high-confidence slipped DNA annotations
        """
        seq = sequence.upper()
        
        # Stage 1: Find all tandem repeat candidates
        candidates = self.find_all_tandem_repeats(seq)
        
        # Stage 2: Apply stringent criteria
        filtered = self.apply_stringent_criteria(candidates)
        
        # Stage 3: Eliminate redundancy (one call per locus)
        non_redundant = self.eliminate_redundancy(filtered)
        
        # Stage 4: Compute mechanistic slippage energy scores
        for cand in non_redundant:
            score = self.compute_slippage_energy_score(
                sequence=cand['sequence'],
                unit=cand['primitive_unit'],
                copy_number=cand['copies'],
                purity=cand['purity']
            )
            cand['slippage_score'] = score
        
        return non_redundant
    
    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Main detection method returning publication-ready slipped DNA annotations.
        
        Output format (simplified & non-redundant):
        - Class: Slipped_DNA
        - Subclass: Slipped_DNA (unified, no STR vs Direct_Repeat distinction)
        - Genomic interval (Start, End, Length)
        - Repeat_Unit: Primitive motif
        - Unit_Size: Length of primitive unit (k)
        - Copy_Number: Number of repeat copies
        - Purity: Repeat purity (0-1)
        - Slippage_Energy_Score: Mechanistic score (1-3)
        
        Guarantees:
        - ONE call per genomic locus (no redundancy)
        - NO STR_1...STR_9 spam
        - Biologically defensible (passes stringent criteria)
        """
        # Reset and mark audit as invoked
        self.audit['invoked'] = True
        self.audit['windows_scanned'] = 1
        self.audit['candidates_seen'] = 0
        self.audit['candidates_filtered'] = 0
        self.audit['reported'] = 0
        
        annotations = self.annotate_sequence(sequence)
        self.audit['candidates_seen'] = len(annotations)
        
        motifs = []
        
        for i, ann in enumerate(annotations):
            # Use primitive_copies if available, otherwise fall back to original copies
            copy_number = ann.get('primitive_copies', ann.get('copies', 0))
            
            motif = {
                'ID': f"{sequence_name}_SLIPPED_{ann['start']+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': 'Slipped_DNA',  # Unified subclass
                'Start': ann['start'] + 1,  # 1-based coordinates
                'End': ann['end'],
                'Length': ann['length'],
                'Sequence': ann['sequence'],
                'Repeat_Unit': ann['primitive_unit'],
                'Unit_Size': len(ann['primitive_unit']),
                'Copy_Number': copy_number,
                'Purity': round(ann['purity'], 3),
                'Slippage_Energy_Score': round(ann['slippage_score'], 3),
                'Score': round(ann['slippage_score'], 3),  # For compatibility
                'Strand': '+',
                'Method': 'Slipped_DNA_detection',
                'Pattern_ID': f'SLIPPED_{i+1}'
            }
            
            motifs.append(motif)
            self.audit['reported'] += 1
        
        return motifs
    
    def calculate_score(self, sequence: str, pattern_info: Tuple = None) -> float:
        """Calculate score for a sequence (mechanism-driven)."""
        # Find repeats and apply pipeline
        annotations = self.annotate_sequence(sequence)
        if annotations:
            # Return max score among annotations
            return max(ann['slippage_score'] for ann in annotations)
        return 0.0


# =============================================================================
# Cruciform Detector
# =============================================================================
"""
CruciformDetector (Optimized for Performance)
=============================================

PERFORMANCE OPTIMIZATIONS:
- Uses optimized seed-and-extend k-mer index approach from repeat_scanner
- O(n) complexity with k-mer seeding instead of O(n²) exhaustive search
- No sliding window needed - efficient on all sequence lengths
- Maintains accuracy while improving speed dramatically

Detects inverted repeats (potential cruciform-forming) with:
 - arm length >= 6 bp
 - loop (spacer) <= 100 bp
 - optional mismatch tolerance
Scoring: interpretable 0..1 score that favors long arms and small loops.
"""

import re
from typing import List, Dict, Any, Tuple
# # from .base_detector import BaseMotifDetector





class CruciformDetector(BaseMotifDetector):
    """
    Detector for cruciform (inverted repeat) DNA structures.
    
    # Motif Structure:
    # | Field       | Type  | Description                      |
    # |-------------|-------|----------------------------------|
    # | Class       | str   | Always 'Cruciform'               |
    # | Subclass    | str   | Inverted_arm_{length}            |
    # | Arm_Length  | int   | Length of palindromic arm        |
    # | Loop_Length | int   | Length of loop between arms      |
    # | Left_Arm    | str   | Left arm sequence                |
    # | Right_Arm   | str   | Right arm (RC of left)           |
    # | Score       | float | Stability score (0-1)            |
    """
    
    def get_motif_class_name(self) -> str:
        return "Cruciform"

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Pattern metadata for compatibility (actual detection uses k-mer indexing).
        Patterns are loaded from the centralized motif_patterns module.
        """
        # Load patterns from centralized module
        if CRUCIFORM_PATTERNS:
            return CRUCIFORM_PATTERNS.copy()
        else:
            # Fallback patterns if import failed
            return {
                'inverted_repeats': [
                    (r'', 'CRU_3_1', 'Potential palindrome', 'Inverted Repeats', 
                     12, 'cruciform_stability', 0.95, 'DNA secondary structure', 'Lilley 2000'),
                ]
            }

    # Configuration - Cruciform DNA: 10–100 nt arm length with reverse complement separated by spacer = 0–3 nt
    MIN_ARM = 10
    MAX_ARM = 100  # Maximum arm length constraint
    MAX_LOOP = 3
    MAX_MISMATCHES = 0

    def find_inverted_repeats(self, sequence: str, min_arm: int = None,
                              max_arm: int = None, max_loop: int = None, 
                              max_mismatches: int = None) -> List[Dict[str, Any]]:
        """
        Find inverted repeats (cruciform precursors) using optimized k-mer indexing.
        Falls back to slower exhaustive search only if mismatch tolerance needed.
        
        Parameters updated for:
        - Cruciform DNA: 10–100 nt arm length with reverse complement separated by spacer = 0–3 nt
        """
        seq = sequence.upper()
        
        if min_arm is None:
            min_arm = self.MIN_ARM
        if max_arm is None:
            max_arm = self.MAX_ARM
        if max_loop is None:
            max_loop = self.MAX_LOOP
        if max_mismatches is None:
            max_mismatches = self.MAX_MISMATCHES

        hits: List[Dict[str, Any]] = []
        
        # Use optimized scanner when available and perfect matches required
        if _find_inverted_repeats_optimized is not None and max_mismatches == 0:
            results = _find_inverted_repeats_optimized(seq, min_arm=min_arm, max_arm=max_arm, max_loop=max_loop)
            
            # Convert to internal format
            for rec in results:
                match_fraction = 1.0
                score = self._score_arm_loop(rec['Arm_Length'], rec['Loop'], match_fraction)
                hits.append({
                    'left_start': rec['Start'] - 1,
                    'left_end': rec['Start'] - 1 + rec['Arm_Length'],
                    'right_start': rec['Right_Start'] - 1,
                    'right_end': rec['Right_Start'] - 1 + rec['Arm_Length'],
                    'arm_len': rec['Arm_Length'],
                    'loop_len': rec['Loop'],
                    'left_seq': rec['Left_Arm'],
                    'right_seq': rec['Right_Arm'],
                    'right_seq_rc': rec['Right_Arm'],
                    'mismatches': 0,
                    'match_fraction': 1.0,
                    'score': round(score, 6)
                })
        else:
            # Fallback for mismatch tolerance or when optimized scanner unavailable
            hits = self._find_inverted_repeats_fallback(seq, min_arm, max_arm, max_loop, max_mismatches)
        
        # Sort by descending score, then position
        hits.sort(key=lambda h: (-h['score'], h['left_start'], -h['arm_len']))
        return hits
    
    def _find_inverted_repeats_fallback(self, seq: str, min_arm: int, max_arm: int,
                                        max_loop: int, max_mismatches: int) -> List[Dict[str, Any]]:
        """Fallback implementation for when mismatch tolerance is needed or optimized scanner unavailable"""
        def revcomp(s: str) -> str:
            trans = str.maketrans("ACGTacgt", "TGCAtgca")
            return s.translate(trans)[::-1]
        
        hits: List[Dict[str, Any]] = []
        n = len(seq)
        
        # For very long sequences, use adaptive sampling
        # Increased from 1000 to 10000 for 10x performance improvement
        MAX_SEQUENCE_LENGTH = 10000
        if n > MAX_SEQUENCE_LENGTH:
            window_size = MAX_SEQUENCE_LENGTH
            # Increased overlap for better boundary detection (was window_size // 2)
            step_size = window_size - 200  # 200bp overlap between windows
            
            for window_start in range(0, n, step_size):
                window_end = min(window_start + window_size, n)
                window_seq = seq[window_start:window_end]
                window_hits = self._find_inverted_repeats_in_window(
                    window_seq, min_arm, max_arm, max_loop, max_mismatches, revcomp
                )
                for hit in window_hits:
                    hit['left_start'] += window_start
                    hit['left_end'] += window_start
                    hit['right_start'] += window_start
                    hit['right_end'] += window_start
                    hits.append(hit)
                if window_end >= n:
                    break
            hits = self._deduplicate_hits(hits)
        else:
            hits = self._find_inverted_repeats_in_window(seq, min_arm, max_arm, max_loop, max_mismatches, revcomp)
        
        return hits
    
    def _find_inverted_repeats_in_window(self, seq: str, min_arm: int, max_arm: int,
                                         max_loop: int, max_mismatches: int, revcomp_fn=None) -> List[Dict[str, Any]]:
        """Core inverted repeat detection in a single window"""
        if revcomp_fn is None:
            revcomp_fn = revcomp
        
        hits: List[Dict[str, Any]] = []
        n = len(seq)
        
        # Adaptive max_loop based on sequence length
        if n > 500:
            max_loop = min(max_loop, 50)
        
        # Sample positions for large windows
        step = 1 if n <= 300 else 3 if n <= 600 else 5 if n <= 1000 else 10
        
        # Limit total iterations
        max_iterations = 10000
        iteration_count = 0

        for left_start in range(0, n - 2 * min_arm, step):
            # Use the max_arm parameter instead of hardcoded value
            max_possible_arm = min(max_arm, (n - left_start) // 2)
            
            # Search from larger arm lengths first (better quality)
            for arm_len in range(max_possible_arm, min_arm - 1, -1):
                left_end = left_start + arm_len
                right_start_min = left_end
                right_start_max = min(left_end + max_loop, n - arm_len)
                
                found_good_match = False
                
                # Check iteration limit
                iteration_count += 1
                if iteration_count > max_iterations:
                    return hits
                
                for right_start in range(right_start_min, right_start_max + 1):
                    loop_len = right_start - left_end
                    right_end = right_start + arm_len
                    if right_end > n:
                        break
                    left_seq = seq[left_start:left_end]
                    right_seq = seq[right_start:right_end]
                    right_rc = revcomp_fn(right_seq)
                    mismatches = sum(1 for a, b in zip(left_seq, right_rc) if a != b)
                    if mismatches <= max_mismatches:
                        match_fraction = (arm_len - mismatches) / arm_len if arm_len > 0 else 0.0
                        score = self._score_arm_loop(arm_len, loop_len, match_fraction)
                        hits.append({
                            'left_start': left_start,
                            'left_end': left_end,
                            'right_start': right_start,
                            'right_end': right_end,
                            'arm_len': arm_len,
                            'loop_len': loop_len,
                            'left_seq': left_seq,
                            'right_seq': right_seq,
                            'right_seq_rc': right_rc,
                            'mismatches': mismatches,
                            'match_fraction': round(match_fraction, 4),
                            'score': round(score, 6)
                        })
                        found_good_match = True
                        if mismatches == 0 and score > 0.5:
                            break
                
                if found_good_match and arm_len >= min_arm * 2:
                    break
                    
        return hits
    
    def _deduplicate_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate hits that may occur in overlapping windows"""
        if not hits:
            return hits
        
        # Create a unique key for each hit based on positions
        seen = {}
        unique_hits = []
        
        for hit in hits:
            key = (hit['left_start'], hit['left_end'], hit['right_start'], hit['right_end'])
            if key not in seen:
                seen[key] = True
                unique_hits.append(hit)
        
        return unique_hits

    # --------------------------
    # Scoring function (interpretable)
    # --------------------------
    def _score_arm_loop(self, arm_len: int, loop_len: int, match_fraction: float) -> float:
        """
        Compute a normalized score 0..1:
          - arm contribution: sigmoid-like increase with arm_len
          - loop penalty: linear penalty up to MAX_LOOP
          - match_fraction scales final score (1.0 = perfect match)
        Formula:
          arm_term = arm_len / (arm_len + 8)    -> approaches 1 for long arms
          loop_term = max(0.0, 1.0 - (loop_len / float(self.MAX_LOOP)))  -> 1 at loop 0, 0 at MAX_LOOP
          base = arm_term * loop_term
          final = base * match_fraction
        This yields scores near 1 for long arms + short loops + perfect match.
        """
        arm_term = float(arm_len) / (arm_len + 8.0)
        loop_term = max(0.0, 1.0 - (float(loop_len) / float(self.MAX_LOOP)))
        base = arm_term * loop_term
        final = base * float(match_fraction)
        # clamp
        return max(0.0, min(1.0, final))

    # --------------------------
    # Public API: calculate_score & annotate_sequence
    # --------------------------
    def calculate_score(self, sequence: str, pattern_info: Tuple = None) -> float:
        """
        Sum scores of detected inverted repeats in the sequence (non-overlap-resolved).
        If you prefer overlap resolution (one strongest per region), call annotate_sequence() and sum accepted.
        """
        seq = sequence.upper()
        hits = self.find_inverted_repeats(seq,
                                         min_arm=self.MIN_ARM,
                                         max_arm=self.MAX_ARM,
                                         max_loop=self.MAX_LOOP,
                                         max_mismatches=self.MAX_MISMATCHES)
        # Sum of hit scores (reflects number + quality)
        total = sum(h['score'] for h in hits)
        return float(total)

    def annotate_sequence(self, sequence: str, max_hits: int = 0) -> List[Dict[str, Any]]:
        """
        Return list of detected inverted repeats with details.
        If max_hits > 0, return at most that many top hits (by score). Otherwise return all.
        """
        seq = sequence.upper()
        hits = self.find_inverted_repeats(seq,
                                         min_arm=self.MIN_ARM,
                                         max_arm=self.MAX_ARM,
                                         max_loop=self.MAX_LOOP,
                                         max_mismatches=self.MAX_MISMATCHES)
        if max_hits and len(hits) > max_hits:
            return hits[:max_hits]
        return hits

    # --------------------------
    # Quality threshold check (keeps similar signature)
    # --------------------------
    def passes_quality_threshold(self, sequence: str, score: float, pattern_info: Tuple) -> bool:
        """
        Enhanced quality check:
          - require that at least one inverted repeat was found
          - optionally enforce minimal per-hit score threshold if pattern_info provides one
        pattern_info[6] was your previous 'score threshold' position; we accept either that or default 0.2
        """
        seq = sequence.upper()
        hits = self.find_inverted_repeats(seq,
                                         min_arm=self.MIN_ARM,
                                         max_arm=self.MAX_ARM,
                                         max_loop=self.MAX_LOOP,
                                         max_mismatches=self.MAX_MISMATCHES)
        if not hits:
            return False
        # prefer the best hit score
        best_score = hits[0]['score']
        # read threshold from pattern_info if provided (position 6 per your metadata format)
        try:
            provided_thresh = float(pattern_info[6]) if (pattern_info and len(pattern_info) > 6) else None
        except Exception:
            provided_thresh = None
        thresh = provided_thresh if provided_thresh is not None else 0.2
        return best_score >= thresh

    def _remove_overlaps(self, inverted_repeats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping inverted repeats, keeping highest scoring non-overlapping set"""
        if not inverted_repeats:
            return []
        
        # Sort by score (descending), then by length (descending)
        sorted_repeats = sorted(inverted_repeats, 
                               key=lambda x: (-x['score'], -(x['right_end'] - x['left_start'])))
        
        non_overlapping = []
        for repeat in sorted_repeats:
            # Check if this repeat overlaps with any already selected
            overlaps = False
            for selected in non_overlapping:
                # Two repeats overlap if their full regions (left_start to right_end) overlap
                if not (repeat['right_end'] <= selected['left_start'] or 
                       repeat['left_start'] >= selected['right_end']):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(repeat)
        
        # Sort by start position for output
        non_overlapping.sort(key=lambda x: x['left_start'])
        return non_overlapping

    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Override base method to use sophisticated cruciform detection with component details.
        Scans BOTH strands as cruciforms are strand-agnostic structures.
        """
        # Reset audit
        self.audit['invoked'] = True
        self.audit['windows_scanned'] = 0
        self.audit['candidates_seen'] = 0
        self.audit['candidates_filtered'] = 0
        self.audit['reported'] = 0
        self.audit['both_strands_scanned'] = True
        
        sequence = sequence.upper().strip()
        motifs = []
        
        # Scan forward strand
        self.audit['windows_scanned'] += 1
        inverted_repeats_fwd = self.find_inverted_repeats(sequence, 
                                                     min_arm=self.MIN_ARM,
                                                     max_arm=self.MAX_ARM,
                                                     max_loop=self.MAX_LOOP,
                                                     max_mismatches=self.MAX_MISMATCHES)
        self.audit['candidates_seen'] += len(inverted_repeats_fwd)
        
        # Filter by meaningful score threshold before overlap removal
        filtered_repeats_fwd = [r for r in inverted_repeats_fwd if r.get('score', 0) > 0.1]
        self.audit['candidates_filtered'] += len(inverted_repeats_fwd) - len(filtered_repeats_fwd)
        
        # Remove overlapping repeats
        non_overlapping_repeats_fwd = self._remove_overlaps(filtered_repeats_fwd)
        
        for i, repeat in enumerate(non_overlapping_repeats_fwd):
            start_pos = repeat['left_start']
            end_pos = repeat['right_end'] 
            full_length = end_pos - start_pos
            full_seq = sequence[start_pos:end_pos]
            
            # Extract components
            left_arm = repeat.get('left_seq', '')
            right_arm = repeat.get('right_seq', '')
            loop_seq = sequence[repeat['left_end']:repeat['right_start']] if repeat['right_start'] > repeat['left_end'] else ''
            
            # Calculate GC content
            gc_total = (full_seq.count('G') + full_seq.count('C')) / len(full_seq) * 100 if len(full_seq) > 0 else 0
            gc_left_arm = (left_arm.count('G') + left_arm.count('C')) / len(left_arm) * 100 if len(left_arm) > 0 else 0
            gc_right_arm = (right_arm.count('G') + right_arm.count('C')) / len(right_arm) * 100 if len(right_arm) > 0 else 0
            gc_loop = (loop_seq.count('G') + loop_seq.count('C')) / len(loop_seq) * 100 if len(loop_seq) > 0 else 0
            
            motifs.append({
                'ID': f"{sequence_name}_CRU_{start_pos+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': 'Inverted_Repeat',
                'Start': start_pos + 1,  # 1-based coordinates
                'End': end_pos,
                'Length': full_length,
                'Sequence': full_seq,
                'Score': round(repeat['score'], 3),
                'Strand': '+',
                'Method': 'Cruciform_detection',
                'Pattern_ID': f'CRU_{i+1}',
                # Component details
                'Left_Arm': left_arm,
                'Right_Arm': right_arm,
                'Loop_Seq': loop_seq,
                'Arm_Length': repeat.get('arm_len', 0),
                'Loop_Length': repeat.get('loop_len', 0),
                'Stem_Length': repeat.get('arm_len', 0),  # For cruciforms, stem length equals arm length
                'GC_Total': round(gc_total, 2),
                'GC_Left_Arm': round(gc_left_arm, 2),
                'GC_Right_Arm': round(gc_right_arm, 2),
                'GC_Loop': round(gc_loop, 2),
                'Mismatches': repeat.get('mismatches', 0),
                'Match_Fraction': repeat.get('match_fraction', 1.0)
            })
            self.audit['reported'] += 1
        
        # Scan reverse strand
        self.audit['windows_scanned'] += 1
        seq_rc = revcomp(sequence)
        inverted_repeats_rc = self.find_inverted_repeats(seq_rc, 
                                                     min_arm=self.MIN_ARM,
                                                     max_arm=self.MAX_ARM,
                                                     max_loop=self.MAX_LOOP,
                                                     max_mismatches=self.MAX_MISMATCHES)
        self.audit['candidates_seen'] += len(inverted_repeats_rc)
        
        # Filter by meaningful score threshold
        filtered_repeats_rc = [r for r in inverted_repeats_rc if r.get('score', 0) > 0.1]
        self.audit['candidates_filtered'] += len(inverted_repeats_rc) - len(filtered_repeats_rc)
        
        # Remove overlapping repeats
        non_overlapping_repeats_rc = self._remove_overlaps(filtered_repeats_rc)
        
        # Convert coordinates to forward strand
        seq_len = len(sequence)
        for i, repeat in enumerate(non_overlapping_repeats_rc):
            # Convert reverse strand coordinates to forward strand
            fwd_start = seq_len - repeat['right_end']
            fwd_end = seq_len - repeat['left_start']
            full_length = fwd_end - fwd_start
            full_seq = sequence[fwd_start:fwd_end]
            
            # Extract components from original sequence
            left_arm = repeat.get('left_seq', '')
            right_arm = repeat.get('right_seq', '')
            loop_seq = seq_rc[repeat['left_end']:repeat['right_start']] if repeat['right_start'] > repeat['left_end'] else ''
            
            # Calculate GC content
            gc_total = (full_seq.count('G') + full_seq.count('C')) / len(full_seq) * 100 if len(full_seq) > 0 else 0
            gc_left_arm = (left_arm.count('G') + left_arm.count('C')) / len(left_arm) * 100 if len(left_arm) > 0 else 0
            gc_right_arm = (right_arm.count('G') + right_arm.count('C')) / len(right_arm) * 100 if len(right_arm) > 0 else 0
            gc_loop = (loop_seq.count('G') + loop_seq.count('C')) / len(loop_seq) * 100 if len(loop_seq) > 0 else 0
            
            motifs.append({
                'ID': f"{sequence_name}_CRU_RC_{fwd_start+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': 'Inverted_Repeat',
                'Start': fwd_start + 1,  # 1-based coordinates
                'End': fwd_end,
                'Length': full_length,
                'Sequence': full_seq,
                'Score': round(repeat['score'], 3),
                'Strand': '-',
                'Method': 'Cruciform_detection',
                'Pattern_ID': f'CRU_RC_{i+1}',
                # Component details
                'Left_Arm': left_arm,
                'Right_Arm': right_arm,
                'Loop_Seq': loop_seq,
                'Arm_Length': repeat.get('arm_len', 0),
                'Loop_Length': repeat.get('loop_len', 0),
                'Stem_Length': repeat.get('arm_len', 0),
                'GC_Total': round(gc_total, 2),
                'GC_Left_Arm': round(gc_left_arm, 2),
                'GC_Right_Arm': round(gc_right_arm, 2),
                'GC_Loop': round(gc_loop, 2),
                'Mismatches': repeat.get('mismatches', 0),
                'Match_Fraction': repeat.get('match_fraction', 1.0)
            })
            self.audit['reported'] += 1
        
        return motifs


# =============================================================================
# R Loop Detector
# =============================================================================
"""
R-loop DNA Motif Detector (QmRLFS-finder v1.5-hs)
================================================

QmRLFS-finder (hyperscan-enhanced)
Based on original QmRLFS-finder v1.5 by Piroon Jenjaroenpun & Thidathip Wongsurawat
Updated: Hyperscan integration, Python 3 modernization, fallback to re when Hyperscan absent.

Detects RNA-DNA hybrid structures including:
- RIZ (R-loop Initiation Zones) with G-tract patterns
- REZ (R-loop Extension Zones) with optimized G content
- QmRLFS Model 1 (m1): G{3,} tracts
- QmRLFS Model 2 (m2): G{4,} tracts

Based on Jenjaroenpun & Wongsurawat 2016.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# Try to import hyperscan for performance
try:
    import hyperscan
    HS_AVAILABLE = True
except Exception:
    HS_AVAILABLE = False


class RLoopDetector(BaseMotifDetector):
    """
    QmRLFS-finder (hyperscan-enhanced) R-loop detector.
    
    Implements QmRLFS models for detecting R-loop forming sequences:
    - Model 1 (m1): G{3,} tract patterns  
    - Model 2 (m2): G{4,} tract patterns
    
    Uses Hyperscan when available for fast pattern matching, falls back to re otherwise.
    """
    
    # QmRLFS parameters (from Jenjaroenpun 2016)
    MIN_PERC_G_RIZ = 50      # Minimum G% in RIZ
    NUM_LINKER = 50          # Linker length for REZ search
    WINDOW_STEP = 100        # Window step for sliding window
    MAX_LENGTH_REZ = 2000    # Maximum REZ length
    MIN_PERC_G_REZ = 40      # Minimum G% in REZ
    
    def __init__(self):
        super().__init__()
        # Compile hyperscan database if available
        self.hs_db = None
        self.hs_id_to_model = {}
        if HS_AVAILABLE:
            self._compile_hyperscan_patterns()
    
    def get_motif_class_name(self) -> str:
        return "R-Loop"
    
    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Return QmRLFS model patterns.
        Patterns are loaded from the centralized motif_patterns module.
        """
        # Load patterns from centralized module
        if RLOOP_PATTERNS:
            return RLOOP_PATTERNS.copy()
        else:
            # Fallback patterns if import failed (using [ATCG] for DNA, not [ATCGU])
            # OPTIMIZED: Removed lazy quantifiers that cause catastrophic backtracking
            # Use atomic groups and possessive quantifiers for better performance
            return {
                'qmrlfs_model_1': [
                    # OLD: r'G{3,}[ATCG]{1,10}?G{3,}(?:[ATCG]{1,10}?G{3,}){1,}?'
                    # NEW: More efficient pattern without lazy quantifiers
                    (r'G{3,}[ATCG]{1,10}G{3,}(?:[ATCG]{1,10}G{3,})+', 
                     'QmRLFS_M1', 'QmRLFS Model 1', 'QmRLFS-m1', 25, 'qmrlfs_score', 
                     0.90, 'RIZ detection with 3+ G tracts', 'Jenjaroenpun 2016'),
                ],
                'qmrlfs_model_2': [
                    # OLD: r'G{4,}(?:[ATCG]{1,10}?G{4,}){1,}?'
                    # NEW: More efficient pattern without lazy quantifiers
                    (r'G{4,}(?:[ATCG]{1,10}G{4,})+', 
                     'QmRLFS_M2', 'QmRLFS Model 2', 'QmRLFS-m2', 30, 'qmrlfs_score', 
                     0.95, 'RIZ detection with 4+ G tracts', 'Jenjaroenpun 2016'),
                ]
            }
    
    def _compile_hyperscan_patterns(self):
        """Compile patterns for hyperscan if available"""
        if not HS_AVAILABLE:
            return
        
        try:
            self.hs_db = hyperscan.Database()
            expressions = []
            ids = []
            flags = []
            next_id = 1
            
            patterns = self.get_patterns()
            for model_name, pattern_list in patterns.items():
                for pattern_info in pattern_list:
                    pattern = pattern_info[0]
                    expressions.append(pattern.encode())
                    ids.append(next_id)
                    flags.append(hyperscan.HS_FLAG_DOTALL | hyperscan.HS_FLAG_UTF8)
                    self.hs_id_to_model[next_id] = model_name
                    next_id += 1
            
            if expressions:
                self.hs_db.compile(expressions=expressions, ids=ids, flags=flags)
        except Exception:
            # If hyperscan compilation fails, fallback to re
            self.hs_db = None
    
    def _riz_search_hyperscan(self, seq: str, model: str) -> List[Dict[str, Any]]:
        """Search for RIZ regions using Hyperscan (fast path)"""
        if not HS_AVAILABLE or self.hs_db is None:
            return []
        
        result_list = []
        seq_bytes = seq.encode('utf-8')
        
        def on_match(id, from_, to, flags, context):
            model_name = self.hs_id_to_model.get(id, None)
            if model_name and model_name == model:
                match_bytes = seq_bytes[from_:to]
                try:
                    match_str = match_bytes.decode('utf-8')
                    result_list.append({
                        'start': from_,
                        'end': to,
                        'sequence': match_str
                    })
                except UnicodeDecodeError:
                    pass
            return 0
        
        try:
            self.hs_db.scan(seq_bytes, match_event_handler=on_match)
        except Exception:
            return []
        
        return result_list
    
    def _riz_search_regex(self, seq: str, model: str) -> List[Dict[str, Any]]:
        """Search for RIZ regions using regex (fallback path)"""
        result_list = []
        patterns = self.get_patterns()
        
        if model not in patterns:
            return result_list
        
        for pattern_info in patterns[model]:
            pattern = pattern_info[0]
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.ASCII)
            
            for match in compiled_pattern.finditer(seq):
                result_list.append({
                    'start': match.start(),
                    'end': match.end(),
                    'sequence': match.group(0)
                })
        
        return result_list
    
    def _riz_search(self, seq: str, model: str) -> List[Dict[str, Any]]:
        """
        Search for RIZ (R-loop Initiation Zone) regions.
        Uses Hyperscan if available, otherwise falls back to regex.
        """
        # Try hyperscan first
        if HS_AVAILABLE and self.hs_db is not None:
            results = self._riz_search_hyperscan(seq, model)
            if results:
                return results
        
        # Fallback to regex
        return self._riz_search_regex(seq, model)
    
    def _percent_g(self, seq: str) -> float:
        """Calculate percentage of G in sequence"""
        if not seq:
            return 0.0
        return round((seq.count("G") / float(len(seq))) * 100.0, 2)
    
    def _find_rez(self, seq: str, riz_end: int) -> Optional[Dict[str, Any]]:
        """
        Find REZ (R-loop Extension Zone) following a RIZ.
        Searches for G-rich regions downstream of RIZ.
        """
        seq_len = len(seq)
        
        # Start search from RIZ end + linker
        search_start = riz_end + self.NUM_LINKER
        if search_start >= seq_len:
            return None
        
        # Find the best REZ within max length
        best_rez = None
        max_score = 0.0
        
        # Sliding window to find best G-rich region
        for window_start in range(search_start, min(seq_len, riz_end + self.MAX_LENGTH_REZ), self.WINDOW_STEP):
            for window_end in range(window_start + 50, min(seq_len, window_start + self.MAX_LENGTH_REZ), 50):
                window_seq = seq[window_start:window_end]
                
                perc_g = self._percent_g(window_seq)
                if perc_g >= self.MIN_PERC_G_REZ:
                    # Score based on G% and length
                    score = perc_g * (window_end - window_start) / 100.0
                    
                    if score > max_score:
                        max_score = score
                        best_rez = {
                            'start': window_start,
                            'end': window_end,
                            'sequence': window_seq,
                            'perc_g': perc_g,
                            'length': window_end - window_start,
                            'score': score
                        }
        
        return best_rez
    
    def _count_g_tracts(self, seq: str, min_g: int) -> Tuple[int, int]:
        """Count G-tracts of minimum length"""
        pattern = r'G{' + str(min_g) + r',}'
        tracts = re.findall(pattern, seq)
        total_g = sum(len(t) for t in tracts)
        return len(tracts), total_g
    
    def annotate_sequence(self, sequence: str, models: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Annotate sequence with QmRLFS predictions.
        
        Args:
            sequence: DNA sequence to analyze
            models: List of models to use ['qmrlfs_model_1', 'qmrlfs_model_2'], default both
        
        Returns:
            List of R-loop region annotations with RIZ and REZ information
        """
        seq = sequence.upper()
        
        if models is None:
            models = ['qmrlfs_model_1', 'qmrlfs_model_2']
        
        results = []
        
        for model in models:
            # Find all RIZ regions for this model
            riz_regions = self._riz_search(seq, model)
            
            for riz in riz_regions:
                riz_seq = riz['sequence']
                riz_start = riz['start']
                riz_end = riz['end']
                
                # Check if RIZ meets G% threshold
                perc_g_riz = self._percent_g(riz_seq)
                if perc_g_riz < self.MIN_PERC_G_RIZ:
                    continue
                
                # Count G-tracts in RIZ
                min_g = 3 if model == 'qmrlfs_model_1' else 4
                num_3gs, total_3gs_riz = self._count_g_tracts(riz_seq, 3)
                num_4gs, total_4gs_riz = self._count_g_tracts(riz_seq, 4)
                
                # Try to find REZ
                rez = self._find_rez(seq, riz_end)
                
                # Create result entry
                result = {
                    'model': model,
                    'riz_start': riz_start,
                    'riz_end': riz_end,
                    'riz_length': len(riz_seq),
                    'riz_sequence': riz_seq,
                    'riz_perc_g': perc_g_riz,
                    'riz_g_total': riz_seq.count('G'),
                    'riz_3gs_count': num_3gs,
                    'riz_4gs_count': num_4gs,
                    'riz_3gs_total': total_3gs_riz,
                    'riz_4gs_total': total_4gs_riz,
                }
                
                if rez:
                    num_3gs_rez, total_3gs_rez = self._count_g_tracts(rez['sequence'], 3)
                    num_4gs_rez, total_4gs_rez = self._count_g_tracts(rez['sequence'], 4)
                    
                    result.update({
                        'rez_start': rez['start'],
                        'rez_end': rez['end'],
                        'rez_length': rez['length'],
                        'rez_sequence': rez['sequence'],
                        'rez_perc_g': rez['perc_g'],
                        'rez_g_total': rez['sequence'].count('G'),
                        'rez_3gs_count': num_3gs_rez,
                        'rez_4gs_count': num_4gs_rez,
                        'rez_3gs_total': total_3gs_rez,
                        'rez_4gs_total': total_4gs_rez,
                        'rez_score': rez['score'],
                        'linker_length': rez['start'] - riz_end,
                        'total_start': riz_start,
                        'total_end': rez['end'],
                        'total_length': rez['end'] - riz_start
                    })
                else:
                    # RIZ only (no REZ found)
                    result.update({
                        'rez_start': None,
                        'rez_end': None,
                        'rez_length': 0,
                        'rez_sequence': '',
                        'rez_perc_g': 0.0,
                        'rez_g_total': 0,
                        'rez_3gs_count': 0,
                        'rez_4gs_count': 0,
                        'rez_3gs_total': 0,
                        'rez_4gs_total': 0,
                        'rez_score': 0.0,
                        'linker_length': 0,
                        'total_start': riz_start,
                        'total_end': riz_end,
                        'total_length': len(riz_seq)
                    })
                
                results.append(result)
        
        # Remove overlaps - keep higher scoring entries
        results = self._remove_overlapping_results(results)
        
        return results
    
    def _remove_overlapping_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping R-loop predictions, keeping highest scoring"""
        if not results:
            return []
        
        # Sort by score (RIZ G% + REZ score), then by length
        sorted_results = sorted(
            results,
            key=lambda x: (
                -x.get('riz_perc_g', 0) - x.get('rez_score', 0),
                -x.get('total_length', 0)
            )
        )
        
        non_overlapping = []
        for result in sorted_results:
            start = result['total_start']
            end = result['total_end']
            
            # Check for overlap with already selected results
            overlaps = False
            for selected in non_overlapping:
                sel_start = selected['total_start']
                sel_end = selected['total_end']
                
                if not (end <= sel_start or start >= sel_end):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(result)
        
        # Sort by start position
        non_overlapping.sort(key=lambda x: x['total_start'])
        return non_overlapping
    
    def calculate_score(self, sequence: str, pattern_info: Tuple) -> float:
        """Calculate R-loop formation score using QmRLFS algorithm"""
        annotations = self.annotate_sequence(sequence)
        
        if not annotations:
            return 0.0
        
        # Sum scores from all detected regions
        total_score = sum(
            (ann.get('riz_perc_g', 0) / 100.0) + 
            (ann.get('rez_score', 0) / 100.0)
            for ann in annotations
        )
        
        return min(total_score, 1.0)
    
    def passes_quality_threshold(self, sequence: str, score: float, pattern_info: Tuple) -> bool:
        """Quality threshold for R-loop detection"""
        return score >= 0.4
    
    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Detect R-loop forming sequences using QmRLFS algorithm on BOTH strands.
        
        R-loops are strand-agnostic structures, so we need to scan both forward and reverse strands.
        Returns motifs with RIZ and REZ component information.
        """
        # Reset audit
        self.audit['invoked'] = True
        self.audit['windows_scanned'] = 0
        self.audit['candidates_seen'] = 0
        self.audit['candidates_filtered'] = 0
        self.audit['reported'] = 0
        self.audit['both_strands_scanned'] = True
        
        sequence = sequence.upper().strip()
        motifs = []
        
        # Scan forward strand
        self.audit['windows_scanned'] += 1
        annotations_fwd = self.annotate_sequence(sequence)
        self.audit['candidates_seen'] += len(annotations_fwd)
        
        for i, ann in enumerate(annotations_fwd):
            # Determine model subclass
            model = ann['model']
            subclass = 'QmRLFS-m1' if model == 'qmrlfs_model_1' else 'QmRLFS-m2'
            
            # Calculate combined score
            score = (ann['riz_perc_g'] / 100.0) + (ann.get('rez_score', 0) / 100.0)
            
            # Create motif entry
            motif = {
                'ID': f"{sequence_name}_RLOOP_{ann['total_start']+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': subclass,
                'Start': ann['total_start'] + 1,  # 1-based coordinates
                'End': ann['total_end'],
                'Length': ann['total_length'],
                'Sequence': sequence[ann['total_start']:ann['total_end']],
                'Score': round(score, 3),
                'Strand': '+',
                'Method': 'QmRLFS_detection',
                'Pattern_ID': f"QmRLFS_{i+1}",
                # RIZ component details
                'RIZ_Start': ann['riz_start'] + 1,
                'RIZ_End': ann['riz_end'],
                'RIZ_Length': ann['riz_length'],
                'RIZ_Sequence': ann['riz_sequence'],
                'RIZ_G_Percent': ann['riz_perc_g'],
                'RIZ_G_Total': ann['riz_g_total'],
                'RIZ_3G_Tracts': ann['riz_3gs_count'],
                'RIZ_4G_Tracts': ann['riz_4gs_count'],
                # REZ component details (if present)
                'REZ_Start': ann.get('rez_start', None),
                'REZ_End': ann.get('rez_end', None),
                'REZ_Length': ann.get('rez_length', 0),
                'REZ_Sequence': ann.get('rez_sequence', ''),
                'REZ_G_Percent': ann.get('rez_perc_g', 0.0),
                'REZ_G_Total': ann.get('rez_g_total', 0),
                'REZ_3G_Tracts': ann.get('rez_3gs_count', 0),
                'REZ_4G_Tracts': ann.get('rez_4gs_count', 0),
                'Linker_Length': ann.get('linker_length', 0)
            }
            
            motifs.append(motif)
            self.audit['reported'] += 1
        
        # Scan reverse strand
        self.audit['windows_scanned'] += 1
        seq_rc = revcomp(sequence)
        annotations_rc = self.annotate_sequence(seq_rc)
        self.audit['candidates_seen'] += len(annotations_rc)
        
        for i, ann in enumerate(annotations_rc):
            # Determine model subclass
            model = ann['model']
            subclass = 'QmRLFS-m1' if model == 'qmrlfs_model_1' else 'QmRLFS-m2'
            
            # Calculate combined score
            score = (ann['riz_perc_g'] / 100.0) + (ann.get('rez_score', 0) / 100.0)
            
            # Convert reverse strand coordinates to forward strand coordinates
            seq_len = len(sequence)
            fwd_start = seq_len - ann['total_end']
            fwd_end = seq_len - ann['total_start']
            
            # Create motif entry for reverse strand
            motif = {
                'ID': f"{sequence_name}_RLOOP_RC_{fwd_start+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': subclass,
                'Start': fwd_start + 1,  # 1-based coordinates
                'End': fwd_end,
                'Length': ann['total_length'],
                'Sequence': sequence[fwd_start:fwd_end],
                'Score': round(score, 3),
                'Strand': '-',
                'Method': 'QmRLFS_detection',
                'Pattern_ID': f"QmRLFS_RC_{i+1}",
                # RIZ component details
                'RIZ_Start': seq_len - ann['riz_end'] + 1,
                'RIZ_End': seq_len - ann['riz_start'],
                'RIZ_Length': ann['riz_length'],
                'RIZ_Sequence': ann['riz_sequence'],
                'RIZ_G_Percent': ann['riz_perc_g'],
                'RIZ_G_Total': ann['riz_g_total'],
                'RIZ_3G_Tracts': ann['riz_3gs_count'],
                'RIZ_4G_Tracts': ann['riz_4gs_count'],
                # REZ component details (if present)
                'REZ_Start': seq_len - ann.get('rez_end', 0) + 1 if ann.get('rez_end') else None,
                'REZ_End': seq_len - ann.get('rez_start', 0) if ann.get('rez_start') else None,
                'REZ_Length': ann.get('rez_length', 0),
                'REZ_Sequence': ann.get('rez_sequence', ''),
                'REZ_G_Percent': ann.get('rez_perc_g', 0.0),
                'REZ_G_Total': ann.get('rez_g_total', 0),
                'REZ_3G_Tracts': ann.get('rez_3gs_count', 0),
                'REZ_4G_Tracts': ann.get('rez_4gs_count', 0),
                'Linker_Length': ann.get('linker_length', 0)
            }
            
            motifs.append(motif)
            self.audit['reported'] += 1
        
        return motifs

# =============================================================================
# Triplex Detector
# =============================================================================
"""
Triplex DNA Motif Detector (Mirror repeat strict, content threshold)
===================================================================
Detects three-stranded DNA structures using optimized k-mer seed-and-extend approach.

Key features from Frank-Kamenetskii 1995, Sakamoto 1999, Bacolla 2006:
- Intramolecular triplex: mirror repeats ≥10 bp per arm, loop ≤100 bp, 
  homopurine or homopyrimidine content >90% in arms
- Sticky DNA: pure (GAA)n or (TTC)n (≥12 bp)

PERFORMANCE OPTIMIZATIONS:
- Uses optimized seed-and-extend k-mer index approach from repeat_scanner
- O(n) complexity with k-mer seeding for mirror repeats
- Efficient purine/pyrimidine content filtering
"""

import re
from typing import List, Dict, Any, Tuple

# BaseMotifDetector is defined above



class TriplexDetector(BaseMotifDetector):
    """
    Detector for triplex-forming DNA: mirror repeats and sticky DNA.
    
    # Motif Structure:
    # | Field       | Type  | Description                      |
    # |-------------|-------|----------------------------------|
    # | Class       | str   | Always 'Triplex'                 |
    # | Subclass    | str   | 'Mirror_Repeat' or 'Sticky_DNA'  |
    # | Arm_Length  | int   | Length of mirror arm             |
    # | Loop_Length | int   | Length of loop (mirror only)     |
    # | Score       | float | Formation potential (0-1)        |
    
    # Configuration - Triplex DNA: 10–100 nt mirrored with spacer = 0–8 nt, 90% purine or pyrimidine
    """
    
    # Triplex DNA parameters
    MIN_ARM = 10
    MAX_ARM = 100  # Maximum arm length
    MAX_LOOP = 8   # Maximum spacer/loop length
    PURINE_PYRIMIDINE_THRESHOLD = 0.9  # 90% purine or pyrimidine content required

    def get_motif_class_name(self) -> str:
        return "Triplex"

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Return sticky DNA patterns (simple regex).
        Mirror repeats use optimized k-mer scanner.
        
        Patterns are loaded from the centralized motif_patterns module.
        """
        # Load patterns from centralized module
        if TRIPLEX_PATTERNS:
            return TRIPLEX_PATTERNS.copy()
        else:
            # Fallback patterns if import failed
            return {
                'triplex_forming_sequences': [
                    (r'(?:GAA){4,}', 'TRX_5_4', 'GAA repeat', 'Sticky_DNA', 12, 
                     'sticky_dna_score', 0.95, 'Disease-associated repeats', 'Sakamoto 1999'),
                    (r'(?:TTC){4,}', 'TRX_5_5', 'TTC repeat', 'Sticky_DNA', 12, 
                     'sticky_dna_score', 0.95, 'Disease-associated repeats', 'Sakamoto 1999'),
                ]
            }
    
    def annotate_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        seq = sequence.upper()
        results = []
        used = [False] * len(seq)
        patterns = self.get_patterns()['triplex_forming_sequences']

        # Use optimized scanner if available
        # Triplex DNA: 10–100 nt mirrored with spacer = 0–8 nt, and 90% Purine or Pyrimidine
        if _find_mirror_repeats_optimized is not None:
            mirror_results = _find_mirror_repeats_optimized(seq, min_arm=self.MIN_ARM, 
                                                            max_arm=self.MAX_ARM,
                                                            max_loop=self.MAX_LOOP, 
                                                            purine_pyrimidine_threshold=self.PURINE_PYRIMIDINE_THRESHOLD)
            
            # Only keep those that pass the triplex threshold (>90% purine or pyrimidine)
            for mr_rec in mirror_results:
                if mr_rec.get('Is_Triplex', False):
                    s = mr_rec['Start'] - 1  # Convert to 0-based
                    e = mr_rec['End']
                    
                    if any(used[s:e]):
                        continue
                    
                    for i in range(s, e):
                        used[i] = True
                    
                    # Determine if purine or pyrimidine
                    pur_frac = mr_rec['Purine_Fraction']
                    pyr_frac = mr_rec['Pyrimidine_Fraction']
                    if pur_frac >= self.PURINE_PYRIMIDINE_THRESHOLD:
                        subtype = 'Homopurine mirror repeat'
                        pid = 'TRX_MR_PU'
                    else:
                        subtype = 'Homopyrimidine mirror repeat'
                        pid = 'TRX_MR_PY'
                    
                    results.append({
                        'class_name': 'Triplex',
                        'pattern_id': pid,
                        'start': s,
                        'end': e,
                        'length': e - s,
                        'score': self._triplex_potential(seq[s:e]),
                        'matched_seq': seq[s:e],
                        'details': {
                            'type': subtype,
                            'reference': 'Frank-Kamenetskii 1995',
                            'description': 'H-DNA formation',
                            'arm_length': mr_rec['Arm_Length'],
                            'loop_length': mr_rec['Loop'],
                            'left_arm': mr_rec.get('Left_Arm', ''),
                            'right_arm': mr_rec.get('Right_Arm', ''),
                            'loop_seq': mr_rec.get('Loop_Seq', ''),
                            'purine_fraction': pur_frac,
                            'pyrimidine_fraction': pyr_frac
                        }
                    })
        else:
            # Fallback to regex-based detection for mirror repeats
            # Triplex DNA: 10–100 nt mirrored with spacer = 0–8 nt
            # Homopurine mirror repeat - match 10-100 consecutive purine nucleotides
            pat_pu = rf'([GA]{{{self.MIN_ARM},{self.MAX_ARM}}})([ATGC]{{0,{self.MAX_LOOP}}})([GA]{{{self.MIN_ARM},{self.MAX_ARM}}})'
            for m in re.finditer(pat_pu, seq):
                s, e = m.span()
                if any(used[s:e]):
                    continue
                arm1 = m.group(1)
                arm2 = m.group(3)
                loop = m.group(2)
                # Apply length constraints
                if len(arm1) < self.MIN_ARM or len(arm1) > self.MAX_ARM:
                    continue
                if len(arm2) < self.MIN_ARM or len(arm2) > self.MAX_ARM:
                    continue
                if len(loop) > self.MAX_LOOP:
                    continue
                pur_ct = sum(1 for b in arm1+arm2 if b in 'AG') / max(1, len(arm1+arm2))
                if pur_ct < self.PURINE_PYRIMIDINE_THRESHOLD:
                    continue
                for i in range(s, e):
                    used[i] = True
                results.append({
                    'class_name': 'Triplex',
                    'pattern_id': 'TRX_MR_PU',
                    'start': s,
                    'end': e,
                    'length': e-s,
                    'score': self._triplex_potential(seq[s:e]),
                    'matched_seq': seq[s:e],
                    'details': {
                        'type': 'Homopurine mirror repeat',
                        'reference': 'Frank-Kamenetskii 1995',
                        'description': 'H-DNA formation (homopurine)'
                    }
                })
            
            # Homopyrimidine mirror repeat - match 10-100 consecutive pyrimidine nucleotides
            pat_py = rf'([CT]{{{self.MIN_ARM},{self.MAX_ARM}}})([ATGC]{{0,{self.MAX_LOOP}}})([CT]{{{self.MIN_ARM},{self.MAX_ARM}}})'
            for m in re.finditer(pat_py, seq):
                s, e = m.span()
                if any(used[s:e]):
                    continue
                arm1 = m.group(1)
                arm2 = m.group(3)
                loop = m.group(2)
                # Apply length constraints
                if len(arm1) < self.MIN_ARM or len(arm1) > self.MAX_ARM:
                    continue
                if len(arm2) < self.MIN_ARM or len(arm2) > self.MAX_ARM:
                    continue
                if len(loop) > self.MAX_LOOP:
                    continue
                pyr_ct = sum(1 for b in arm1+arm2 if b in 'CT') / max(1, len(arm1+arm2))
                if pyr_ct < self.PURINE_PYRIMIDINE_THRESHOLD:
                    continue
                for i in range(s, e):
                    used[i] = True
                results.append({
                    'class_name': 'Triplex',
                    'pattern_id': 'TRX_MR_PY',
                    'start': s,
                    'end': e,
                    'length': e-s,
                    'score': self._triplex_potential(seq[s:e]),
                    'matched_seq': seq[s:e],
                    'details': {
                        'type': 'Homopyrimidine mirror repeat',
                        'reference': 'Frank-Kamenetskii 1995',
                        'description': 'H-DNA formation (homopyrimidine)'
                    }
                })

        # Sticky DNA patterns (GAA/TTC) - use regex
        for patinfo in patterns:
            pat, pid, name, cname, minlen, scoretype, cutoff, desc, ref = patinfo
            for m in re.finditer(pat, seq):
                s, e = m.span()
                if any(used[s:e]):
                    continue
                for i in range(s, e):
                    used[i] = True
                results.append({
                    'class_name': cname,
                    'pattern_id': pid,
                    'start': s,
                    'end': e,
                    'length': e-s,
                    'score': self.calculate_score(seq[s:e], patinfo),
                    'matched_seq': seq[s:e],
                    'details': {
                        'type': name,
                        'reference': ref,
                        'description': desc
                    }
                })
        
        results.sort(key=lambda r: r['start'])
        return results

    def calculate_score(self, sequence: str, pattern_info: Tuple) -> float:
        scoring_method = pattern_info[5] if len(pattern_info) > 5 else 'triplex_potential'
        if scoring_method == 'triplex_potential':
            return self._triplex_potential(sequence)
        elif scoring_method == 'sticky_dna_score':
            return self._sticky_dna_score(sequence)
        else:
            return 0.0

    def _triplex_potential(self, sequence: str) -> float:
        """Score: tract length and purine/pyrimidine content (≥90%)"""
        if len(sequence) < 20:
            return 0.0
        pur = sum(1 for b in sequence if b in "AG") / len(sequence)
        pyr = sum(1 for b in sequence if b in "CT") / len(sequence)
        score = (pur if pur > 0.9 else 0) + (pyr if pyr > 0.9 else 0)
        # tract length bonus: scale for very long arms, up to 1.0
        return min(score * len(sequence) / 150, 1.0)
        
    def _sticky_dna_score(self, sequence: str) -> float:
        """Score for sticky DNA: repeat density and length"""
        if len(sequence) < 12:
            return 0.0
        gaa_count = sequence.count("GAA")
        ttc_count = sequence.count("TTC")
        rep_total = gaa_count + ttc_count
        density = (rep_total * 3) / len(sequence)
        extras = sum(len(m.group(0)) for m in re.finditer(r'(?:GAA){2,}', sequence)) + \
                 sum(len(m.group(0)) for m in re.finditer(r'(?:TTC){2,}', sequence))
        cons_bonus = extras / len(sequence) if len(sequence) else 0
        return min(0.7 * density + 0.3 * cons_bonus, 1.0)

    def passes_quality_threshold(self, sequence: str, score: float, pattern_info: Tuple) -> bool:
        """Lower threshold for triplex detection"""
        return score >= 0.2  # Lower threshold for better sensitivity

    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Override base method to use sophisticated triplex detection with component details.
        Scans BOTH strands as mirror repeats are strand-agnostic structures.
        """
        # Reset audit
        self.audit['invoked'] = True
        self.audit['windows_scanned'] = 0
        self.audit['candidates_seen'] = 0
        self.audit['candidates_filtered'] = 0
        self.audit['reported'] = 0
        self.audit['both_strands_scanned'] = True
        
        sequence = sequence.upper().strip()
        motifs = []
        
        # Scan forward strand
        self.audit['windows_scanned'] += 1
        results_fwd = self.annotate_sequence(sequence)
        self.audit['candidates_seen'] += len(results_fwd)
        
        for i, result in enumerate(results_fwd):
            motif_dict = {
                'ID': f"{sequence_name}_{result['pattern_id']}_{result['start']+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': result['details']['type'],
                'Start': result['start'] + 1,  # 1-based coordinates
                'End': result['end'],
                'Length': result['length'],
                'Sequence': result['matched_seq'],
                'Score': round(result['score'], 3),
                'Strand': '+',
                'Method': 'Triplex_detection',
                'Pattern_ID': result['pattern_id']
            }
            
            # For mirror repeats, extract arm and loop components
            if 'mirror repeat' in result['details']['type'].lower():
                details = result['details']
                
                # Check if we have the arm/loop sequences from scanner
                if 'left_arm' in details and details['left_arm']:
                    motif_dict['Left_Arm'] = details['left_arm']
                    motif_dict['Right_Arm'] = details['right_arm']
                    motif_dict['Loop_Seq'] = details['loop_seq']
                    motif_dict['Arm_Length'] = details.get('arm_length', len(details['left_arm']))
                    motif_dict['Loop_Length'] = details.get('loop_length', len(details['loop_seq']))
                    
                    # Calculate GC content for components
                    left_arm = details['left_arm']
                    right_arm = details['right_arm']
                    loop_seq = details['loop_seq']
                    
                    gc_left = (left_arm.count('G') + left_arm.count('C')) / len(left_arm) * 100 if len(left_arm) > 0 else 0
                    gc_right = (right_arm.count('G') + right_arm.count('C')) / len(right_arm) * 100 if len(right_arm) > 0 else 0
                    gc_loop = (loop_seq.count('G') + loop_seq.count('C')) / len(loop_seq) * 100 if len(loop_seq) > 0 else 0
                    
                    motif_dict['GC_Left_Arm'] = round(gc_left, 2)
                    motif_dict['GC_Right_Arm'] = round(gc_right, 2)
                    motif_dict['GC_Loop'] = round(gc_loop, 2)
                else:
                    # Fallback: extract from matched sequence if arm_length/loop_length are present
                    arm_len = details.get('arm_length', 0)
                    loop_len = details.get('loop_length', 0)
                    
                    if arm_len > 0 and loop_len > 0:
                        matched_seq = result['matched_seq']
                        left_arm = matched_seq[:arm_len]
                        loop_seq = matched_seq[arm_len:arm_len + loop_len]
                        right_arm = matched_seq[arm_len + loop_len:arm_len + loop_len + arm_len]
                        
                        motif_dict['Left_Arm'] = left_arm
                        motif_dict['Right_Arm'] = right_arm
                        motif_dict['Loop_Seq'] = loop_seq
                        motif_dict['Arm_Length'] = arm_len
                        motif_dict['Loop_Length'] = loop_len
                        
                        # Calculate GC content for components
                        gc_left = (left_arm.count('G') + left_arm.count('C')) / len(left_arm) * 100 if len(left_arm) > 0 else 0
                        gc_right = (right_arm.count('G') + right_arm.count('C')) / len(right_arm) * 100 if len(right_arm) > 0 else 0
                        gc_loop = (loop_seq.count('G') + loop_seq.count('C')) / len(loop_seq) * 100 if len(loop_seq) > 0 else 0
                        
                        motif_dict['GC_Left_Arm'] = round(gc_left, 2)
                        motif_dict['GC_Right_Arm'] = round(gc_right, 2)
                        motif_dict['GC_Loop'] = round(gc_loop, 2)
            
            motifs.append(motif_dict)
            self.audit['reported'] += 1
        
        # Scan reverse strand
        self.audit['windows_scanned'] += 1
        seq_rc = revcomp(sequence)
        results_rc = self.annotate_sequence(seq_rc)
        self.audit['candidates_seen'] += len(results_rc)
        
        # Convert reverse strand coordinates to forward strand
        seq_len = len(sequence)
        for i, result in enumerate(results_rc):
            # Convert coordinates
            fwd_start = seq_len - result['end']
            fwd_end = seq_len - result['start']
            
            motif_dict = {
                'ID': f"{sequence_name}_{result['pattern_id']}_RC_{fwd_start+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': result['details']['type'],
                'Start': fwd_start + 1,  # 1-based coordinates
                'End': fwd_end,
                'Length': result['length'],
                'Sequence': sequence[fwd_start:fwd_end],
                'Score': round(result['score'], 3),
                'Strand': '-',
                'Method': 'Triplex_detection',
                'Pattern_ID': result['pattern_id'] + '_RC'
            }
            
            # For mirror repeats, extract arm and loop components
            if 'mirror repeat' in result['details']['type'].lower():
                details = result['details']
                
                # Check if we have the arm/loop sequences from scanner
                if 'left_arm' in details and details['left_arm']:
                    motif_dict['Left_Arm'] = details['left_arm']
                    motif_dict['Right_Arm'] = details['right_arm']
                    motif_dict['Loop_Seq'] = details['loop_seq']
                    motif_dict['Arm_Length'] = details.get('arm_length', len(details['left_arm']))
                    motif_dict['Loop_Length'] = details.get('loop_length', len(details['loop_seq']))
                    
                    # Calculate GC content for components
                    left_arm = details['left_arm']
                    right_arm = details['right_arm']
                    loop_seq = details['loop_seq']
                    
                    gc_left = (left_arm.count('G') + left_arm.count('C')) / len(left_arm) * 100 if len(left_arm) > 0 else 0
                    gc_right = (right_arm.count('G') + right_arm.count('C')) / len(right_arm) * 100 if len(right_arm) > 0 else 0
                    gc_loop = (loop_seq.count('G') + loop_seq.count('C')) / len(loop_seq) * 100 if len(loop_seq) > 0 else 0
                    
                    motif_dict['GC_Left_Arm'] = round(gc_left, 2)
                    motif_dict['GC_Right_Arm'] = round(gc_right, 2)
                    motif_dict['GC_Loop'] = round(gc_loop, 2)
                else:
                    # Fallback: extract from matched sequence if arm_length/loop_length are present
                    arm_len = details.get('arm_length', 0)
                    loop_len = details.get('loop_length', 0)
                    
                    if arm_len > 0 and loop_len > 0:
                        matched_seq = result['matched_seq']
                        left_arm = matched_seq[:arm_len]
                        loop_seq = matched_seq[arm_len:arm_len + loop_len]
                        right_arm = matched_seq[arm_len + loop_len:arm_len + loop_len + arm_len]
                        
                        motif_dict['Left_Arm'] = left_arm
                        motif_dict['Right_Arm'] = right_arm
                        motif_dict['Loop_Seq'] = loop_seq
                        motif_dict['Arm_Length'] = arm_len
                        motif_dict['Loop_Length'] = loop_len
                        
                        # Calculate GC content for components
                        gc_left = (left_arm.count('G') + left_arm.count('C')) / len(left_arm) * 100 if len(left_arm) > 0 else 0
                        gc_right = (right_arm.count('G') + right_arm.count('C')) / len(right_arm) * 100 if len(right_arm) > 0 else 0
                        gc_loop = (loop_seq.count('G') + loop_seq.count('C')) / len(loop_seq) * 100 if len(loop_seq) > 0 else 0
                        
                        motif_dict['GC_Left_Arm'] = round(gc_left, 2)
                        motif_dict['GC_Right_Arm'] = round(gc_right, 2)
                        motif_dict['GC_Loop'] = round(gc_loop, 2)
            
            motifs.append(motif_dict)
            self.audit['reported'] += 1
        
        return motifs


# =============================================================================
# G Quadruplex Detector
# =============================================================================
"""
G-Quadruplex Detector (G4Hunter-based, overlap-resolved)

Updated G-Quadruplex Definitions (2024):
=========================================
Class: G-Quadruplex
Subclasses with priorities (higher number = higher priority):

1. Telomeric G4 (Priority 95):
   - Pattern: (TTAGGG){4,}
   - Reference: Parkinson et al., Nature 2002; Phan & Mergny, NAR 2002; Neidle, Nat Rev Chem 2017
   - Overrides all other G-structures

2. Stacked canonical G4s (Priority 90):
   - Pattern: ((G{3,}[ACGT]{1,7}){3}G{3,}){2,}
   - Reference: Phan et al., NAR 2007; Neidle, Chem Soc Rev 2009; Mergny & Sen, Chem Rev 2019
   - Suppresses contained single G4s

3. Stacked G4s with linker (Priority 85):
   - Pattern: ((G{3,}[ACGT]{1,7}){3}G{3,})([ACGT]{0,20}((G{3,}[ACGT]{1,7}){3}G{3,})){1,}
   - Reference: Hänsel-Hertsch et al., Nat Genet 2016; Neidle, Nat Rev Chem 2017
   - Dominates isolated G4s and triplexes

4. Canonical intramolecular G4 (Priority 80):
   - Pattern: G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}
   - Reference: Huppert & Balasubramanian, NAR 2005; Huppert, Chem Soc Rev 2008
   - Suppresses G-triplex & weak PQS

5. Extended-loop canonical (Priority 70):
   - Pattern: G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}
   - Reference: Chambers et al., Nat Biotechnol 2015; Mergny & Sen, Chem Rev 2019
   - Loses to canonical if overlapping

6. Higher-order G4 array/G4-wire (Priority 65):
   - Pattern: (G{3,}[ACGT]{1,7}){7,}
   - Reference: Wong & Wu, Biochimie 2003; Neidle, Chem Soc Rev 2009
   - Regional annotation; may coexist

7. Intramolecular G-triplex (Priority 45):
   - Pattern: G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}
   - Reference: Lim & Phan, JACS 2013; Mergny & Sen, Chem Rev 2019
   - Suppressed by canonical/stacked G4

8. Two-tetrad weak PQS (Priority 25):
   - Pattern: G{2,}[ACGT]{1,7}G{2,}[ACGT]{1,7}G{2,}[ACGT]{1,7}G{2,}
   - Reference: Kikin et al., NAR 2006 (QGRS Mapper); Bedrat et al., NAR 2016
   - Suppressed by all canonical motifs
"""
import re
from typing import List, Dict, Any, Tuple

# BaseMotifDetector is defined above

WINDOW_SIZE_DEFAULT = 25
MIN_REGION_LEN = 8
# Priority order based on updated definitions (higher number = higher priority)
CLASS_PRIORITY = [
    "telomeric_g4",           # Priority 95
    "stacked_canonical_g4s",  # Priority 90
    "stacked_g4s_linker",     # Priority 85
    "canonical_g4",           # Priority 80
    "extended_loop_g4",       # Priority 70
    "higher_order_g4",        # Priority 65
    "g_triplex",              # Priority 45
    "weak_pqs",               # Priority 25
]

class GQuadruplexDetector(BaseMotifDetector):
    """Detector for G-quadruplex DNA motifs using G4Hunter scoring and overlap resolution."""

    def get_motif_class_name(self) -> str:
        """Returns high-level motif class name for reporting."""
        return "G-Quadruplex"

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Returns regexes and metadata for major G4 motif families.
        Patterns are loaded from the centralized motif_patterns module.
        
        Updated G-Quadruplex Patterns (2024):
          1. Telomeric G4: (TTAGGG){4,} - Priority 95 - Parkinson 2002, Neidle 2017
          2. Stacked canonical G4s: ((G{3,}[ACGT]{1,7}){3}G{3,}){2,} - Priority 90 - Phan 2007, Mergny & Sen 2019
          3. Stacked G4s with linker - Priority 85 - Hänsel-Hertsch 2016
          4. Canonical intramolecular G4 - Priority 80 - Huppert & Balasubramanian 2005
          5. Extended-loop canonical - Priority 70 - Chambers 2015
          6. Higher-order G4 array/G4-wire - Priority 65 - Wong & Wu 2003
          7. Intramolecular G-triplex - Priority 45 - Lim & Phan 2013
          8. Two-tetrad weak PQS - Priority 25 - Kikin 2006, Bedrat 2016
        
        Optimized with non-capturing groups for performance.
        """
        # Load patterns from centralized module
        if G4_PATTERNS:
            return G4_PATTERNS.copy()
        else:
            # Fallback patterns based on updated definitions
            return {
                'telomeric_g4': [
                    (r'(?:TTAGGG){4,}', 'G4_TEL', 'Telomeric G4', 'Telomeric G4', 24, 'g4hunter_score', 
                     0.95, 'Human telomeric G4 structure', 'Parkinson et al., Nature 2002'),
                ],
                'stacked_canonical_g4s': [
                    (r'(?:(?:G{3,}[ACGT]{1,7}){3}G{3,}){2,}', 'G4_STK_CAN', 'Stacked canonical G4s', 'Stacked canonical G4s', 30, 'g4hunter_score', 
                     0.90, 'Structural polymorphism & stacking', 'Phan et al., NAR 2007'),
                ],
                'stacked_g4s_linker': [
                    (r'(?:(?:G{3,}[ACGT]{1,7}){3}G{3,})(?:[ACGT]{0,20}(?:(?:G{3,}[ACGT]{1,7}){3}G{3,})){1,}', 'G4_STK_LNK', 'Stacked G4s with linker', 'Stacked G4s with linker', 30, 'g4hunter_score', 
                     0.85, 'Clustered G4s in chromatin', 'Hänsel-Hertsch et al., Nat Genet 2016'),
                ],
                'canonical_g4': [
                    (r'G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}', 'G4_CAN', 'Canonical intramolecular G4', 'Canonical intramolecular G4', 15, 'g4hunter_score', 
                     0.80, 'Canonical G4 structure', 'Huppert & Balasubramanian, NAR 2005'),
                ],
                'extended_loop_g4': [
                    (r'G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}', 'G4_EXT', 'Extended-loop canonical', 'Extended-loop canonical', 15, 'g4hunter_score', 
                     0.70, 'G4-seq reveals long-loop G4s', 'Chambers et al., Nat Biotechnol 2015'),
                ],
                'higher_order_g4': [
                    (r'(?:G{3,}[ACGT]{1,7}){7,}', 'G4_HIGH', 'Higher-order G4 array/G4-wire', 'Higher-order G4 array/G4-wire', 49, 'g4hunter_score', 
                     0.65, 'Higher-order G4s', 'Wong & Wu, Biochimie 2003'),
                ],
                'g_triplex': [
                    (r'G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}', 'G4_TRX', 'Intramolecular G-triplex', 'Intramolecular G-triplex', 12, 'g_triplex_score', 
                     0.45, 'G-triplex structures', 'Lim & Phan, JACS 2013'),
                ],
                'weak_pqs': [
                    (r'G{2,}[ACGT]{1,7}G{2,}[ACGT]{1,7}G{2,}[ACGT]{1,7}G{2,}', 'G4_WEAK', 'Two-tetrad weak PQS', 'Two-tetrad weak PQS', 11, 'g4hunter_score', 
                     0.25, 'QGRS Mapper weak PQS', 'Kikin et al., NAR 2006'),
                ]
            }

    def calculate_score(self, sequence: str, pattern_info: Tuple = None) -> float:
        """Compute total score for all accepted G4 regions after overlap resolution."""
        seq = sequence.upper()
        candidates = self._find_all_candidates(seq)
        scored = [self._score_candidate(c, seq) for c in candidates]
        accepted = self._resolve_overlaps(scored)
        total = sum(a['score'] for a in accepted)
        return float(total)

    def annotate_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Annotate all accepted motif regions after overlap resolution.
        Returns dicts: class_name, pattern_id, start, end, length, score, matched_seq, details.
        """
        seq = sequence.upper()
        candidates = self._find_all_candidates(seq)
        scored = [self._score_candidate(c, seq) for c in candidates]
        accepted = self._resolve_overlaps(scored)
        anns = []
        for a in accepted:
            ann = {
                'class_name': a['class_name'],
                'pattern_id': a['pattern_id'],
                'start': a['start'],
                'end': a['end'],
                'length': a['end'] - a['start'],
                'score': round(a['score'], 6),
                'matched_seq': seq[a['start']:a['end']],
                'details': a['details']
            }
            anns.append(ann)
        return anns

    def _find_all_candidates(self, seq: str) -> List[Dict[str, Any]]:
        """
        Find all regions matching any G4 motif.
        Returns: list of {class_name, pattern_id, start, end, match_text}.
        """
        patt_groups = self.get_patterns()
        candidates = []
        for class_name, patterns in patt_groups.items():
            for pat in patterns:
                regex = pat[0]
                pattern_id = pat[1] if len(pat) > 1 else f"{class_name}_pat"
                for m in re.finditer(regex, seq):
                    s, e = m.start(), m.end()
                    if (e - s) < MIN_REGION_LEN:
                        continue
                    candidates.append({
                        'class_name': class_name,
                        'pattern_id': pattern_id,
                        'start': s,
                        'end': e,
                        'match_text': seq[s:e]
                    })
        return candidates

    def _score_candidate(self, candidate: Dict[str, Any], seq: str, window_size: int = WINDOW_SIZE_DEFAULT) -> Dict[str, Any]:
        """
        Calculate per-region G4Hunter-derived score plus tract/GC penalties.
        Returns candidate dict plus 'score' and 'details'.
        Also extracts G-quadruplex components (stems and loops).
        """
        s = candidate['start']
        e = candidate['end']
        region = seq[s:e]
        L = len(region)
        vals = [1 if ch == 'G' else -1 if ch == 'C' else 0 for ch in region]
        ws = min(window_size, L)
        max_abs = 0
        if ws > 0:
            cur = sum(vals[0:ws])
            max_abs = abs(cur)
            for i in range(1, L - ws + 1):
                cur += vals[i + ws - 1] - vals[i - 1]
                if abs(cur) > max_abs:
                    max_abs = abs(cur)
        normalized_window = (max_abs / ws) if ws > 0 else 0.0
        g_tracts = re.findall(r'G{2,}', region)
        n_g = len(g_tracts)
        total_g_len = sum(len(t) for t in g_tracts)
        tract_bonus = 0.0
        if n_g >= 3:
            tract_bonus = min(0.5, 0.08 * (n_g - 2) * ((total_g_len / n_g) / 4.0))
        total_c = region.count('C')
        total_g = region.count('G')
        gc_balance = (total_g - total_c) / (L if L > 0 else 1)
        gc_penalty = 0.0
        if gc_balance < -0.3:
            gc_penalty = 0.2
        elif gc_balance < -0.1:
            gc_penalty = 0.1

        normalized_score = max(0.0, min(1.0, normalized_window + tract_bonus - gc_penalty))
        region_score = normalized_score * (L / float(ws)) if ws > 0 else 0.0

        # Extract G-quadruplex components (stems and loops)
        stems, loops = self._extract_g4_components(region)
        
        # Calculate GC content
        gc_total = (region.count('G') + region.count('C')) / len(region) * 100 if len(region) > 0 else 0
        gc_stems = 0
        if stems:
            all_stems = ''.join(stems)
            gc_stems = (all_stems.count('G') + all_stems.count('C')) / len(all_stems) * 100 if len(all_stems) > 0 else 0
        
        details = {
            'n_g_tracts': n_g,
            'total_g_len': total_g_len,
            'gc_balance': round(gc_balance, 4),
            'max_window_abs': float(max_abs),
            'normalized_window': round(normalized_window, 6),
            'tract_bonus': round(tract_bonus, 6),
            'gc_penalty': round(gc_penalty, 6),
            'normalized_score': round(normalized_score, 6),
            'region_score': round(region_score, 6),
            # Component information
            'stems': stems,
            'loops': loops,
            'num_stems': len(stems),
            'num_loops': len(loops),
            'stem_lengths': [len(s) for s in stems],
            'loop_lengths': [len(l) for l in loops],
            'GC_Total': round(gc_total, 2),
            'GC_Stems': round(gc_stems, 2)
        }
        out = candidate.copy()
        out['score'] = float(region_score)
        out['details'] = details
        return out
    
    def _extract_g4_components(self, sequence: str) -> Tuple[List[str], List[str]]:
        """
        Extract stems (G-tracts) and loops from a G-quadruplex sequence.
        Returns (stems_list, loops_list).
        """
        # Find all G-tracts (potential stems)
        g_tract_pattern = re.compile(r'G{2,}')
        matches = list(g_tract_pattern.finditer(sequence))
        
        stems = []
        loops = []
        
        if len(matches) >= 2:
            for i, match in enumerate(matches):
                stems.append(match.group())
                # Get loop between this stem and the next
                if i < len(matches) - 1:
                    loop_start = match.end()
                    loop_end = matches[i + 1].start()
                    if loop_end > loop_start:
                        loops.append(sequence[loop_start:loop_end])
        
        return stems, loops

    def _resolve_overlaps(self, scored_candidates: List[Dict[str, Any]], merge_gap: int = 0) -> List[Dict[str, Any]]:
        """
        Select non-overlapping candidates by score and class priority.
        Returns list of accepted regions.
        """
        if not scored_candidates:
            return []
        def class_prio_idx(class_name):
            try:
                return CLASS_PRIORITY.index(class_name)
            except ValueError:
                return len(CLASS_PRIORITY)
        scored_sorted = sorted(
            scored_candidates,
            key=lambda x: (-x['score'], class_prio_idx(x['class_name']), -(x['end'] - x['start']))
        )
        accepted = []
        occupied = []
        for cand in scored_sorted:
            s, e = cand['start'], cand['end']
            conflict = False
            for (as_, ae) in occupied:
                if not (e <= as_ - merge_gap or s >= ae + merge_gap):
                    conflict = True
                    break
            if not conflict:
                accepted.append(cand)
                occupied.append((s, e))
        accepted.sort(key=lambda x: x['start'])
        return accepted

    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Override base method to use annotate_sequence with overlap resolution.
        
        This ensures that for overlapping G4 subclass motifs, only the longest or 
        highest-scoring non-overlapping motif is reported within each subclass.
        
        Returns:
            List of motif dictionaries with complete G4 component information
        """
        sequence = sequence.upper().strip()
        motifs = []
        
        # Use annotate_sequence which includes overlap resolution
        annotations = self.annotate_sequence(sequence)
        
        for i, annotation in enumerate(annotations):
            # Map class_name to subclass display name
            class_name = annotation['class_name']
            subclass_map = {
                'telomeric_g4': 'Telomeric G4',
                'stacked_canonical_g4s': 'Stacked canonical G4s',
                'stacked_g4s_linker': 'Stacked G4s with linker',
                'canonical_g4': 'Canonical intramolecular G4',
                'extended_loop_g4': 'Extended-loop canonical',
                'higher_order_g4': 'Higher-order G4 array/G4-wire',
                'g_triplex': 'Intramolecular G-triplex',
                'weak_pqs': 'Two-tetrad weak PQS'
            }
            subclass = subclass_map.get(class_name, class_name)
            
            start_pos = annotation['start']
            end_pos = annotation['end']
            details = annotation.get('details', {})
            
            motif = {
                'ID': f"{sequence_name}_{annotation['pattern_id']}_{start_pos+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': subclass,
                'Start': start_pos + 1,  # 1-based coordinates
                'End': end_pos,
                'Length': annotation['length'],
                'Sequence': annotation['matched_seq'],
                'Score': round(annotation['score'], 3),
                'Strand': '+',
                'Method': 'G4Hunter_detection',
                'Pattern_ID': annotation['pattern_id']
            }
            
            # Add component details if available
            if details:
                motif['Stems'] = details.get('stems', [])
                motif['Loops'] = details.get('loops', [])
                motif['Num_Stems'] = details.get('num_stems', 0)
                motif['Num_Loops'] = details.get('num_loops', 0)
                motif['Stem_Lengths'] = details.get('stem_lengths', [])
                motif['Loop_Lengths'] = details.get('loop_lengths', [])
                motif['GC_Total'] = details.get('GC_Total', 0)
                motif['GC_Stems'] = details.get('GC_Stems', 0)
                
                # Add consolidated Stem_Length and Loop_Length fields
                # These represent the average or typical length for compatibility
                stem_lengths = details.get('stem_lengths', [])
                loop_lengths = details.get('loop_lengths', [])
                if stem_lengths:
                    motif['Stem_Length'] = sum(stem_lengths) / len(stem_lengths)
                if loop_lengths:
                    motif['Loop_Length'] = sum(loop_lengths) / len(loop_lengths)
            
            motifs.append(motif)
        
        return motifs


"""
Updated G-Quadruplex Detection (2024)
======================================

This detector implements the latest G-quadruplex classification system with 8 distinct
subclasses based on structural and functional characteristics. Each subclass has been
assigned a priority score for overlap resolution.

Key References and Pattern Classifications:

1. TELOMERIC G4 (Priority 95):
   Pattern: (TTAGGG){4,}
   - Parkinson GN et al., Nature 2002 - Human telomeric G4 structure determination
   - Phan AT, Mergny JL, NAR 2002 - Telomeric DNA G-quadruplexes
   - Neidle S, Nat Rev Chem 2017 - Review of telomeric G4s in biology
   - Overrides all other G-structures due to specific sequence and high biological significance

2. STACKED CANONICAL G4s (Priority 90):
   Pattern: ((G{3,}[ACGT]{1,7}){3}G{3,}){2,}
   - Phan AT et al., NAR 2007 - Structural polymorphism and stacking of G-quadruplexes
   - Neidle S, Chem Soc Rev 2009 - G-quadruplex structures and functions
   - Mergny JL, Sen D, Chem Rev 2019 - Comprehensive review of G4 structures
   - Suppresses contained single G4s

3. STACKED G4s WITH LINKER (Priority 85):
   Pattern: ((G{3,}[ACGT]{1,7}){3}G{3,})([ACGT]{0,20}((G{3,}[ACGT]{1,7}){3}G{3,})){1,}
   - Hänsel-Hertsch R et al., Nat Genet 2016 - Clustered G4s in chromatin structure
   - Neidle S, Nat Rev Chem 2017 - G4 clusters and their biological roles
   - Dominates isolated G4s and triplexes

4. CANONICAL INTRAMOLECULAR G4 (Priority 80):
   Pattern: G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}
   - Huppert JL, Balasubramanian S, NAR 2005 - Prevalence of G4s in human genome
   - Huppert JL, Chem Soc Rev 2008 - Four-stranded DNA structures
   - Suppresses G-triplex & weak PQS

5. EXTENDED-LOOP CANONICAL (Priority 70):
   Pattern: G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}
   - Chambers VS et al., Nat Biotechnol 2015 - G4-seq reveals long-loop G4s
   - Mergny JL, Sen D, Chem Rev 2019 - Structural variations in G4s
   - Loses to canonical if overlapping

6. HIGHER-ORDER G4 ARRAY/G4-WIRE (Priority 65):
   Pattern: (G{3,}[ACGT]{1,7}){7,}
   - Wong HM, Wu CF, Biochimie 2003 - Higher-order G4 structures
   - Neidle S, Chem Soc Rev 2009 - G4 arrays and biological implications
   - Regional annotation; may coexist with other classes

7. INTRAMOLECULAR G-TRIPLEX (Priority 45):
   Pattern: G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}
   - Lim KW, Phan AT, JACS 2013 - Structure of human telomeric G-triplex
   - Mergny JL, Sen D, Chem Rev 2019 - G-triplex intermediates
   - Suppressed by canonical/stacked G4

8. TWO-TETRAD WEAK PQS (Priority 25):
   Pattern: G{2,}[ACGT]{1,7}G{2,}[ACGT]{1,7}G{2,}[ACGT]{1,7}G{2,}
   - Kikin O et al., NAR 2006 - QGRS Mapper for potential quadruplex sequences
   - Bedrat A et al., NAR 2016 - Re-evaluation of weak PQS and their limitations
   - Suppressed by all canonical motifs

OVERLAP / DOMINANCE RULES:
The priority system ensures that when multiple G4 motifs overlap:
1. Higher priority patterns (e.g., Telomeric G4) override lower priority patterns
2. Patterns with similar structural features but different specificities are resolved by priority
3. Stacked and clustered G4s take precedence over isolated single G4s
4. Canonical structures suppress weaker/less specific variants

This classification system provides a hierarchical framework for G-quadruplex annotation
that reflects both structural characteristics and biological significance.
"""


# =============================================================================
# I Motif Detector
# =============================================================================
import re
from typing import List, Dict, Any, Tuple

# BaseMotifDetector is defined above

# Helper: reverse complement
def _rc(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]

VALIDATED_SEQS = [
    ("IM_VAL_001", "CCCCTCCCCTCCCCTCCCC", "Validated i-motif sequence 1", "Gehring 1993"),
    ("IM_VAL_002", "CCCCACCCCACCCCACCCC", "Validated i-motif sequence 2", "Leroy 1995"),
]

MIN_REGION_LEN = 10
CLASS_PRIORITIES = {'canonical_imotif': 1, 'hur_ac_motif': 2}

def _class_prio_idx(class_name: str) -> int:
    return CLASS_PRIORITIES.get(class_name, 999)

class IMotifDetector(BaseMotifDetector):
    """Detector for i-motif DNA structures (updated: Hur et al. 2021, Benabou 2014)[web:92][web:98][web:100]"""

    def get_motif_class_name(self) -> str:
        return "i-Motif"

    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Returns i-motif patterns including:
        - Canonical i-motif (4 C-tracts)
        - HUR AC-motifs (A-C alternating patterns from problem statement)
        
        Patterns are loaded from the centralized motif_patterns module.
        Based on problem statement specifications and Hur et al. 2021, Benabou 2014.
        """
        # Load patterns from centralized module
        if IMOTIF_PATTERNS:
            return IMOTIF_PATTERNS.copy()
        else:
            # Fallback patterns if import failed
            return {
                'canonical_imotif': [
                    (r'C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}', 'IM_0', 'Canonical i-motif', 'canonical_imotif', 15, 'imotif_score', 0.95, 'pH-dependent C-rich structure', 'Gehring 1993'),
                ],
                'hur_ac_motif': [
                    # HUR A-C motifs from problem statement (HUR_AC_PATTERNS)
                    (r'A{3}[ACGT]{4}C{3}[ACGT]{4}C{3}[ACGT]{4}C{3}', 'HUR_AC_1', 'HUR AC-motif (4bp)', 'AC-motif (HUR)', 18, 'ac_motif_score', 0.85, 'HUR AC alternating motif', 'Hur 2021'),
                    (r'C{3}[ACGT]{4}C{3}[ACGT]{4}C{3}[ACGT]{4}A{3}', 'HUR_AC_2', 'HUR CA-motif (4bp)', 'AC-motif (HUR)', 18, 'ac_motif_score', 0.85, 'HUR CA alternating motif', 'Hur 2021'),
                    (r'A{3}[ACGT]{5}C{3}[ACGT]{5}C{3}[ACGT]{5}C{3}', 'HUR_AC_3', 'HUR AC-motif (5bp)', 'AC-motif (HUR)', 21, 'ac_motif_score', 0.85, 'HUR AC alternating motif', 'Hur 2021'),
                    (r'C{3}[ACGT]{5}C{3}[ACGT]{5}C{3}[ACGT]{5}A{3}', 'HUR_AC_4', 'HUR CA-motif (5bp)', 'AC-motif (HUR)', 21, 'ac_motif_score', 0.85, 'HUR CA alternating motif', 'Hur 2021'),
                    (r'A{3}[ACGT]{6}C{3}[ACGT]{6}C{3}[ACGT]{6}C{3}', 'HUR_AC_5', 'HUR AC-motif (6bp)', 'AC-motif (HUR)', 24, 'ac_motif_score', 0.85, 'HUR AC alternating motif', 'Hur 2021'),
                    (r'C{3}[ACGT]{6}C{3}[ACGT]{6}C{3}[ACGT]{6}A{3}', 'HUR_AC_6', 'HUR CA-motif (6bp)', 'AC-motif (HUR)', 24, 'ac_motif_score', 0.85, 'HUR CA alternating motif', 'Hur 2021'),
                ]
            }

    def find_validated_matches(self, sequence: str, check_revcomp: bool = False) -> List[Dict[str, Any]]:
        seq = sequence.upper()
        out = []
        for vid, vseq, desc, cite in VALIDATED_SEQS:
            idx = seq.find(vseq)
            if idx >= 0:
                out.append({'id': vid, 'seq': vseq, 'start': idx, 'end': idx+len(vseq), 'strand': '+', 'desc': desc, 'cite': cite})
            elif check_revcomp:
                rc = _rc(vseq)
                idx2 = seq.find(rc)
                if idx2 >= 0:
                    out.append({'id': vid, 'seq': vseq, 'start': idx2, 'end': idx2+len(vseq), 'strand': '-', 'desc': desc, 'cite': cite})
        return out

    def find_hur_ac_candidates(self, sequence: str, scan_rc: bool = True) -> List[Dict[str, Any]]:
        seq = sequence.upper()
        candidates = []

        def _matches_hur_ac(target, strand):
            for nlink in (4, 5, 6):
                # A at start, or A at end
                pat1 = r"A{3}[ACGT]{%d}C{3}[ACGT]{%d}C{3}[ACGT]{%d}C{3}" % (nlink, nlink, nlink)
                pat2 = r"C{3}[ACGT]{%d}C{3}[ACGT]{%d}C{3}[ACGT]{%d}A{3}" % (nlink, nlink, nlink)
                for pat in (pat1, pat2):
                    for m in re.finditer(pat, target):
                        s, e = m.span()
                        matched = m.group(0).upper()
                        candidates.append({
                            'start': s if strand == '+' else len(seq) - e,
                            'end': e if strand == '+' else len(seq) - s,
                            'strand': strand,
                            'linker': nlink,
                            'pattern': pat,
                            'matched_seq': matched,
                            'loose_mode': True,
                            'high_confidence': (nlink == 4 or nlink == 5)
                        })

        _matches_hur_ac(seq, '+')
        if scan_rc:
            _matches_hur_ac(_rc(seq), '-')
        candidates.sort(key=lambda x: x['start'])
        return candidates

    def _find_regex_candidates(self, sequence: str) -> List[Dict[str, Any]]:
        seq = sequence.upper()
        patterns = self.get_patterns()
        out = []
        for class_name, pats in patterns.items():
            for patt in pats:
                regex = patt[0]
                pid = patt[1] if len(patt) > 1 else f"{class_name}_pat"
                # Use IGNORECASE | ASCII for better performance
                for m in re.finditer(regex, seq, flags=re.IGNORECASE | re.ASCII):
                    s, e = m.start(), m.end()
                    if (e - s) < MIN_REGION_LEN:
                        continue
                    out.append({
                        'class_name': class_name,
                        'pattern_id': pid,
                        'start': s,
                        'end': e,
                        'matched_seq': seq[s:e]
                    })
        return out

    def _score_imotif_candidate(self, matched_seq: str) -> float:
        region = matched_seq.upper()
        L = len(region)
        if L < 12:
            return 0.0
        c_tracts = [m.group(0) for m in re.finditer(r"C{2,}", region)]
        if len(c_tracts) < 3:
            return 0.0
        total_c = sum(len(t) for t in c_tracts)
        c_density = total_c / L
        tract_bonus = min(0.4, 0.12 * (len(c_tracts) - 2))
        score = max(0.0, min(1.0, c_density + tract_bonus))
        return float(score)

    def _score_hur_ac_candidate(self, matched_seq: str, linker: int, high_confidence: bool) -> float:
        r = matched_seq.upper()
        L = len(r)
        ac_count = r.count('A') + r.count('C')
        ac_frac = ac_count / L if L > 0 else 0.0
        a_tracts = [len(m.group(0)) for m in re.finditer(r"A{2,}", r)]
        c_tracts = [len(m.group(0)) for m in re.finditer(r"C{2,}", r)]
        tract_score = 0.0
        if any(x >= 3 for x in a_tracts) and sum(1 for x in c_tracts if x >= 3) >= 3:
            tract_score = 0.5
        base = min(0.6, ac_frac * 0.8)
        linker_boost = 0.25 if high_confidence else (0.12 if linker == 6 else 0.0)
        return max(0.0, min(1.0, base + tract_score + linker_boost))

    def _resolve_overlaps_greedy(self, scored: List[Dict[str, Any]], merge_gap: int = 0) -> List[Dict[str, Any]]:
        if not scored:
            return []
        scored_sorted = sorted(scored, key=lambda x: (-x['score'], _class_prio_idx(x.get('class_name','')), -(x['end']-x['start'])))
        accepted = []
        occupied = []
        for cand in scored_sorted:
            s, e = cand['start'], cand['end']
            conflict = False
            for (as_, ae) in occupied:
                if not (e <= as_ - merge_gap or s >= ae + merge_gap):
                    conflict = True
                    break
            if not conflict:
                accepted.append(cand)
                occupied.append((s, e))
        accepted.sort(key=lambda x: x['start'])
        return accepted

    def calculate_score(self, sequence: str, pattern_info: Tuple = None) -> float:
        seq = sequence.upper()
        validated = self.find_validated_matches(seq, check_revcomp=False)
        if validated:
            return 0.99
        hur_cands = self.find_hur_ac_candidates(seq, scan_rc=True)
        hur_scored = [dict(
            class_name='ac_motif_hur',
            pattern_id=h['pattern'],
            start=h['start'],
            end=h['end'],
            matched_seq=h['matched_seq'],
            linker=h['linker'],
            high_confidence=h['high_confidence'],
            score=self._score_hur_ac_candidate(h['matched_seq'], h['linker'], h['high_confidence']),
            details=h
        ) for h in hur_cands]
        regex_cands = self._find_regex_candidates(seq)
        regex_scored = [dict(
            class_name=r['class_name'],
            pattern_id=r['pattern_id'],
            start=r['start'],
            end=r['end'],
            matched_seq=r['matched_seq'],
            score=self._score_imotif_candidate(r['matched_seq']),
            details={}
        ) for r in regex_cands]
        combined = hur_scored + regex_scored
        accepted = self._resolve_overlaps_greedy(combined, merge_gap=0)
        total = float(sum(a['score'] * max(1, (a['end']-a['start'])/10.0) for a in accepted))
        return total

    def annotate_sequence(self, sequence: str) -> Dict[str, Any]:
        seq = sequence.upper()
        res = {}
        res['validated_matches'] = self.find_validated_matches(seq, check_revcomp=True)
        hur_cands = self.find_hur_ac_candidates(seq, scan_rc=True)
        for h in hur_cands:
            h['score'] = self._score_hur_ac_candidate(h['matched_seq'], h['linker'], h['high_confidence'])
        res['hur_candidates'] = hur_cands
        regex_cands = self._find_regex_candidates(seq)
        for r in regex_cands:
            r['score'] = self._score_imotif_candidate(r['matched_seq'])
        res['regex_matches'] = regex_cands
        combined = [dict(class_name='ac_motif_hur', start=h['start'], end=h['end'], score=h['score'], details=h) for h in hur_cands]
        combined += [dict(class_name=r['class_name'], start=r['start'], end=r['end'], score=r['score'], details=r) for r in regex_cands]
        res['accepted'] = self._resolve_overlaps_greedy(combined, merge_gap=0)
        return res
    
    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Detect i-motif structures with component details and overlap resolution.
        
        This ensures that for overlapping i-motif subclass motifs, only the longest or 
        highest-scoring non-overlapping motif is reported within each subclass.
        """
        seq = sequence.upper()
        motifs = []
        
        # Use annotate_sequence which includes overlap resolution
        annotation_result = self.annotate_sequence(seq)
        accepted_motifs = annotation_result.get('accepted', [])
        
        # Map class_name to subclass display name
        subclass_map = {
            'canonical_imotif': 'Canonical i-Motif',
            'hur_ac_motif': 'HUR AC-Motif',
            'ac_motif_hur': 'HUR AC-Motif'
        }
        
        # Create motif dictionaries from accepted motifs
        for i, accepted in enumerate(accepted_motifs):
            start_pos = accepted['start']
            end_pos = accepted['end']
            motif_seq = seq[start_pos:end_pos]
            class_name = accepted.get('class_name', 'canonical_imotif')
            subclass = subclass_map.get(class_name, 'i-Motif')
            score = accepted.get('score', 0)
            
            # Extract C-tracts (stems for i-motifs)
            c_tracts = re.findall(r'C{2,}', motif_seq)
            
            # Extract loops (regions between C-tracts)
            loops = []
            c_tract_matches = list(re.finditer(r'C{2,}', motif_seq))
            for j in range(len(c_tract_matches) - 1):
                loop_start = c_tract_matches[j].end()
                loop_end = c_tract_matches[j + 1].start()
                if loop_end > loop_start:
                    loops.append(motif_seq[loop_start:loop_end])
            
            # Calculate GC content
            gc_total = (motif_seq.count('G') + motif_seq.count('C')) / len(motif_seq) * 100 if len(motif_seq) > 0 else 0
            gc_stems = 0
            if c_tracts:
                all_stems = ''.join(c_tracts)
                gc_stems = (all_stems.count('G') + all_stems.count('C')) / len(all_stems) * 100 if len(all_stems) > 0 else 0
            
            motif = {
                'ID': f"{sequence_name}_IMOT_{start_pos+1}",
                'Sequence_Name': sequence_name,
                'Class': self.get_motif_class_name(),
                'Subclass': subclass,
                'Start': start_pos + 1,  # 1-based coordinates
                'End': end_pos,
                'Length': end_pos - start_pos,
                'Sequence': motif_seq,
                'Score': round(score, 3),
                'Strand': '+',
                'Method': 'i-Motif_detection',
                'Pattern_ID': f'IMOT_{i+1}',
                # Component details
                'Stems': c_tracts,
                'Loops': loops,
                'Num_Stems': len(c_tracts),
                'Num_Loops': len(loops),
                'Stem_Lengths': [len(s) for s in c_tracts],
                'Loop_Lengths': [len(l) for l in loops],
                'GC_Total': round(gc_total, 2),
                'GC_Stems': round(gc_stems, 2)
            }
            
            # Add consolidated Stem_Length and Loop_Length fields
            # These represent the average or typical length for compatibility
            if c_tracts:
                motif['Stem_Length'] = sum(len(s) for s in c_tracts) / len(c_tracts)
            if loops:
                motif['Loop_Length'] = sum(len(l) for l in loops) / len(loops)
            
            motifs.append(motif)
        
        return motifs
