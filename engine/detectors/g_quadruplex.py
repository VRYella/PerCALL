"""
GQuadruplexDetector Module
==========================

Detector class for GQuadruplex motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import G4_PATTERNS
except ImportError:
    G4_PATTERNS = {}

# G-Quadruplex detection constants
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

