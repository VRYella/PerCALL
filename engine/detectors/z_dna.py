"""
ZDNADetector Module
===================

Detector class for Z DNA motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import ZDNA_PATTERNS
except ImportError:
    ZDNA_PATTERNS = {}

import logging
logger = logging.getLogger(__name__)

_HYPERSCAN_AVAILABLE = False
try:
    import hyperscan
    _HYPERSCAN_AVAILABLE = True
except ImportError:
    pass

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


