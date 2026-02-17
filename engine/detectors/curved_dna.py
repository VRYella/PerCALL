"""
CurvedDNADetector Module
========================

Detector class for Curved DNA motifs.
Extracted from detectors.py for modular architecture.

Provides specialized detection algorithms and pattern matching for this motif type.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from .base import BaseMotifDetector

# Try importing patterns from motif_patterns module
try:
    from motif_patterns import CURVED_DNA_PATTERNS
except ImportError:
    CURVED_DNA_PATTERNS = {}

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


