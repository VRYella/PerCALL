"""
RLoopDetector Module
====================

Detector class for RLoop motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import RLOOP_PATTERNS
except ImportError:
    RLOOP_PATTERNS = {}

import logging
logger = logging.getLogger(__name__)

_HYPERSCAN_AVAILABLE = False
try:
    import hyperscan
    _HYPERSCAN_AVAILABLE = True
except ImportError:
    pass

def revcomp(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]



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



