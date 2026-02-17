"""
CruciformDetector Module
========================

Detector class for Cruciform motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import CRUCIFORM_PATTERNS
except ImportError:
    CRUCIFORM_PATTERNS = {}

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


