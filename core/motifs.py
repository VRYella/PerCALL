"""
core/motifs.py
──────────────
Non-B DNA motif definitions and scanning utilities.

Motifs covered
──────────────
G4          – G-quadruplex forming sequences (4 runs of ≥3 Gs)
iMotif      – i-Motif forming sequences (4 runs of ≥3 Cs)
ZDNA        – Z-DNA forming alternating purine-pyrimidine repeats
eGZDNA      – Expanded G/Z-DNA within CGG/GGC trinucleotide repeats
Triplex     – Purine (AG) mirror repeats ≥10 bp (triplex-forming)
STR         – Short Tandem Repeats (1–6 bp unit, ≥4 copies)
DirectRepeat – Direct repeat pairs (4–10 bp unit, gap ≤10 bp)
PolyAT      – Homopolymeric A or T runs ≥7 bp

References
──────────
G4 / iMotif patterns follow Balasubramanian lab conventions.
Z-DNA regex approximation based on Rich et al. (1984): CG/GC/CA/TG
dinucleotide repeats ≥4 units; TA steps excluded.
eGZ-DNA: Fakharzadeh et al. (2022), CGG/GGC repeats ≥4 units.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

MOTIF_PATTERNS: Dict[str, re.Pattern] = {
    "G4": re.compile(
        r"G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}"
    ),
    "iMotif": re.compile(
        r"C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}"
    ),
    "ZDNA": re.compile(
        r"(?:CG){4,}|(?:GC){4,}|(?:CA){4,}|(?:TG){4,}"
    ),
    "eGZDNA": re.compile(
        r"(?:CGG){4,}|(?:GGC){4,}"
    ),
    "Triplex": re.compile(r"[AG]{10,}"),
    "STR": re.compile(r"([ACGT]{1,6})\1{3,}"),
    "DirectRepeat": re.compile(r"([ACGT]{4,10})[ACGT]{0,10}\1"),
    "PolyAT": re.compile(r"A{7,}|T{7,}"),
}

# ---------------------------------------------------------------------------
# Display labels and colours
# ---------------------------------------------------------------------------

MOTIF_LABELS: Dict[str, str] = {
    "G4": "G-Quadruplex (G4)",
    "iMotif": "i-Motif",
    "ZDNA": "Z-DNA",
    "eGZDNA": "eGZ-DNA",
    "Triplex": "Triplex",
    "STR": "Short Tandem Repeat",
    "DirectRepeat": "Direct Repeat",
    "PolyAT": "PolyA/T",
}

MOTIF_COLORS: Dict[str, str] = {
    "G4": "#e6194b",
    "iMotif": "#3cb44b",
    "ZDNA": "#4363d8",
    "eGZDNA": "#f58231",
    "Triplex": "#911eb4",
    "STR": "#42d4f4",
    "DirectRepeat": "#f032e6",
    "PolyAT": "#bfef45",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_motifs(
    seq: str,
    active_motifs: set[str] | None = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find all match positions for each motif in *seq*.

    Parameters
    ----------
    seq : str
        DNA sequence (uppercase).
    active_motifs : set[str] | None
        Subset of ``MOTIF_PATTERNS`` keys to scan.  None scans all.

    Returns
    -------
    dict mapping motif key → list of (start, end) tuples (0-based, half-open).
    """
    results: Dict[str, List[Tuple[int, int]]] = {}
    for name, pat in MOTIF_PATTERNS.items():
        if active_motifs is not None and name not in active_motifs:
            results[name] = []
            continue
        results[name] = [(m.start(), m.end()) for m in pat.finditer(seq)]
    return results


def count_motifs(
    seq: str,
    active_motifs: set[str] | None = None,
) -> Dict[str, int]:
    """
    Count occurrences of each motif in *seq*.

    Parameters
    ----------
    seq : str
        DNA sequence.
    active_motifs : set[str] | None
        Subset to count.  None counts all.

    Returns
    -------
    dict mapping motif key → count.
    """
    counts: Dict[str, int] = {}
    for name, pat in MOTIF_PATTERNS.items():
        if active_motifs is not None and name not in active_motifs:
            counts[name] = 0
            continue
        counts[name] = len(pat.findall(seq))
    return counts
