#!/usr/bin/env python3
"""
A-philic DNA motif scanner
- Uses tetranucleotide log2 odds table
- Optionally uses Intel Hyperscan for fast pattern search
- Scores windows >=10bp based on sum of tetranuc log2 odds
- Classifies windows as High-confidence A / Moderate A / not A
"""

import math
import csv
from collections import Counter

# --- Tetranuc log2 odds table ---
TET_LOG2 = {
    "AGGG": 3.7004,
    "CCCT": 3.7004,
    "CCCC": 2.5361,
    "GGGG": 2.5361,
    "GCCC": 2.4021,
    "GGGC": 2.4021,
    "CCCA": 2.0704,
    "TGGG": 2.0704,
    "CCTA": 1.585,
    "TAGG": 1.585,
    "ACCC": 0.848,
    "CCCG": 0.848,
    "CGGG": 0.848,
    "GGGT": 0.848,
    "CCGG": 0.2591,
    "GCAC": 0.241,
    "GTGC": 0.241,
    "GGCC": 0.1343,
}

# --- Thresholds ---
STRONG_LOG2 = 2.0
HIGH_MEAN = 2.0
MOD_MEAN = 1.0
FRACTION_HIGH = 5/7
FRACTION_MOD = 4/7

# --- Hyperscan check ---
try:
    import hyperscan
    USE_HS = True
except ImportError:
    USE_HS = False


def build_contrib_array(seq, tet_log2):
    """Build contribution array for tetranucleotide scores."""
    seq = seq.upper()
    n = len(seq)
    contrib = [0.0] * n
    for i in range(n - 3):
        tet = seq[i:i + 4]
        if tet in tet_log2:
            contrib[i] = tet_log2[tet]
    return contrib


def classify_window(sum_log2, n_tets, strong_count):
    """Classify window based on log2 odds score and strong tetranucleotide count."""
    high_thresh = HIGH_MEAN * n_tets
    mod_thresh = MOD_MEAN * n_tets
    need_high = math.ceil(n_tets * FRACTION_HIGH)
    need_mod = math.ceil(n_tets * FRACTION_MOD)
    
    if sum_log2 >= high_thresh and strong_count >= need_high:
        return "A_high_confidence"
    elif sum_log2 >= mod_thresh and strong_count >= need_mod:
        return "A_moderate"
    return "not_A"


def scan_sequence(seq, min_len=10, max_len=15, step=1, out_tsv="aphilic_hits.tsv"):
    """
    Scanner for A-philic DNA motifs.
    
    Args:
        seq: DNA sequence string
        min_len: Minimum window length
        max_len: Maximum window length
        step: Step size for sliding window
        out_tsv: Output TSV file path
        
    Returns:
        Path to output TSV file
    """
    seq = seq.upper()
    n = len(seq)
    contrib = build_contrib_array(seq, TET_LOG2)

    # Prefix sums for efficient window calculations
    pref = [0.0] * (n + 1)
    strong_flag = [0] * n
    for i in range(n):
        pref[i + 1] = pref[i] + contrib[i]
        if contrib[i] >= STRONG_LOG2:
            strong_flag[i] = 1
    
    pref_strong = [0] * (n + 1)
    for i in range(n):
        pref_strong[i + 1] = pref_strong[i] + strong_flag[i]

    # Scan windows
    results = []
    for L in range(min_len, max_len + 1):
        n_tets = L - 3
        if n_tets <= 0:
            continue
        for s in range(0, n - L + 1, step):
            e = s + L
            sum_log2 = pref[e - 3 + 1] - pref[s]
            strong_c = pref_strong[e - 3 + 1] - pref_strong[s]
            label = classify_window(sum_log2, n_tets, strong_c)
            if label != "not_A":
                results.append({
                    'start': s,
                    'end': e,
                    'length': L,
                    'n_tets': n_tets,
                    'sum_log2': round(sum_log2, 3),
                    'strong_count': strong_c,
                    'label': label,
                    'seq': seq[s:e]
                })

    # Write results to TSV
    with open(out_tsv, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["start", "end", "length", "n_tets", "sum_log2", "strong_count", "label", "seq"])
        for result in results:
            w.writerow([
                result['start'], result['end'], result['length'], result['n_tets'],
                result['sum_log2'], result['strong_count'], result['label'], result['seq']
            ])
    
    return out_tsv


def detect_a_philic_motifs(sequence, min_len=10, max_len=50, step=1):
    """
    Detect A-philic DNA motifs and return results as list of dictionaries.
    
    Args:
        sequence: DNA sequence string
        min_len: Minimum window length
        max_len: Maximum window length  
        step: Step size for sliding window
        
    Returns:
        List of A-philic motif results
    """
    seq = sequence.upper()
    n = len(seq)
    contrib = build_contrib_array(seq, TET_LOG2)

    # Prefix sums for efficient window calculations
    pref = [0.0] * (n + 1)
    strong_flag = [0] * n
    for i in range(n):
        pref[i + 1] = pref[i] + contrib[i]
        if contrib[i] >= STRONG_LOG2:
            strong_flag[i] = 1
    
    pref_strong = [0] * (n + 1)
    for i in range(n):
        pref_strong[i + 1] = pref_strong[i] + strong_flag[i]

    # Scan windows
    results = []
    for L in range(min_len, max_len + 1):
        n_tets = L - 3
        if n_tets <= 0:
            continue
        for s in range(0, n - L + 1, step):
            e = s + L
            sum_log2 = pref[e - 3 + 1] - pref[s]
            strong_c = pref_strong[e - 3 + 1] - pref_strong[s]
            label = classify_window(sum_log2, n_tets, strong_c)
            if label != "not_A":
                results.append({
                    'start': s,
                    'end': e,
                    'length': L,
                    'n_tets': n_tets,
                    'sum_log2': round(sum_log2, 3),
                    'strong_count': strong_c,
                    'classification': label,
                    'sequence': seq[s:e],
                    'confidence': 'High' if label == "A_high_confidence" else 'Moderate'
                })
    
    return results


# --- Example ---
if __name__ == "__main__":
    example = "NNNNAGGGGGGGGGCCCCTGGGGGCCCAAGGGNNNN"
    out = scan_sequence(example, min_len=10, max_len=20, step=1,
                        out_tsv="aphilic_results.tsv")
    print("Results written to:", out)