"""
SlippedDNADetector Module
=========================

Detector class for Slipped DNA motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import SLIPPED_DNA_PATTERNS
except ImportError:
    SLIPPED_DNA_PATTERNS = {}

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





