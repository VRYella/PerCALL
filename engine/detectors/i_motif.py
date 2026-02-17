"""
IMotifDetector Module
=====================

Detector class for IMotif motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import IMOTIF_PATTERNS
except ImportError:
    IMOTIF_PATTERNS = {}

def revcomp(seq: str) -> str:
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]



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
