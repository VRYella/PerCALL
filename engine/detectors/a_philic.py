"""
APhilicDetector Module
======================

Detector class for APhilic motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import APHILIC_DNA_PATTERNS
except ImportError:
    APHILIC_DNA_PATTERNS = {}

import logging
logger = logging.getLogger(__name__)

_HYPERSCAN_AVAILABLE = False
try:
    import hyperscan
    _HYPERSCAN_AVAILABLE = True
except ImportError:
    pass

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

