"""
TriplexDetector Module
======================

Detector class for Triplex motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import TRIPLEX_PATTERNS
except ImportError:
    TRIPLEX_PATTERNS = {}

def revcomp(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]



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

