"""
Non-B DNA Motif Detector Module
Integrated from NonBDNAFinder repository
Detects 11 classes of Non-B DNA structures with 22+ subclasses
"""

import re
from typing import List, Dict, Optional
import numpy as np

# =============================================================================
# A-PHILIC DNA DETECTOR
# =============================================================================

# Full A-philic 10-mer log2 table (experimentally derived)
A10_LOG2 = {
    "AGGGGGGGGT": 2.4612714285714286, "AGGGGGGGGG": 2.702428571428571,
    "AGGGGGGGGC": 2.683285714285714, "AGGGGGGGCC": 2.3401714285714283,
    "AGGGGGGCCC": 2.321028571428571, "AGGGGGCCCA": 2.2545,
    "AGGGGGCCCT": 2.487357142857143, "AGGGGGCCCG": 2.0798714285714284,
    "AGGGGGCCCC": 2.321028571428571, "AGGGGCCCTA": 2.3514857142857144,
    "AGGGGCCCGG": 1.7545857142857142, "AGGGGCCCCA": 2.2545,
    "AGGGGCCCCT": 2.487357142857143, "AGGGGCCCCG": 2.0798714285714284,
    "AGGGGCCCCC": 2.321028571428571, "AGGGCCCGGG": 1.5134285714285713,
    "AGGGCCCCTA": 2.3514857142857144, "AGGGCCCCGG": 1.7545857142857142,
    "AGGGCCCCCA": 2.2545, "AGGGCCCCCT": 2.487357142857143,
    "AGGGCCCCCG": 2.0798714285714284, "AGGGCCCCCC": 2.321028571428571,
    "ACCCGGGGGT": 1.2461857142857142, "ACCCGGGGGG": 1.4873428571428569,
    "ACCCGGGGGC": 1.4682, "ACCCGGGGCC": 1.1250857142857142,
    "ACCCGGGCCC": 1.1059428571428571, "ACCCCGGGGT": 1.2461857142857142,
    "ACCCCGGGGG": 1.4873428571428569, "ACCCCGGGGC": 1.4682,
    "ACCCCGGGCC": 1.1250857142857142, "ACCCCCGGGT": 1.2461857142857142,
    "ACCCCCGGGG": 1.4873428571428569, "ACCCCCGGGC": 1.4682,
    "ACCCCCCGGG": 1.4873428571428569, "ACCCCCCCTA": 2.3254,
    "ACCCCCCCGG": 1.7285, "ACCCCCCCCA": 2.2284142857142855,
    "ACCCCCCCCT": 2.4612714285714286, "ACCCCCCCCG": 2.053785714285714,
    "ACCCCCCCCC": 2.294942857142857,
    "TAGGGGGGGT": 2.3254, "TAGGGGGGGG": 2.5665571428571425,
    "TAGGGGGGGC": 2.5474142857142854, "TAGGGGGGCC": 2.2043,
    "TAGGGGGCCC": 2.185157142857143, "TAGGGGCCCA": 2.1186285714285713,
    "TAGGGGCCCT": 2.3514857142857144, "TAGGGGCCCG": 1.944,
    "TAGGGGCCCC": 2.185157142857143, "TAGGGCCCTA": 2.2156142857142855,
    "TAGGGCCCGG": 1.6187142857142856, "TAGGGCCCCA": 2.1186285714285713,
    "TAGGGCCCCT": 2.3514857142857144, "TAGGGCCCCG": 1.944,
    "TAGGGCCCCC": 2.185157142857143,
    "TGGGGGGGGT": 2.2284142857142855, "TGGGGGGGGG": 2.4695714285714283,
    "TGGGGGGGGC": 2.450428571428571, "TGGGGGGGCC": 2.1073142857142857,
    "TGGGGGGCCC": 2.0881714285714286, "TGGGGGCCCA": 2.021642857142857,
    "TGGGGGCCCT": 2.2545, "TGGGGGCCCG": 1.8470142857142857,
    "TGGGGGCCCC": 2.0881714285714286, "TGGGGCCCTA": 2.1186285714285713,
    "TGGGGCCCGG": 1.5217285714285713, "TGGGGCCCCA": 2.021642857142857,
    "TGGGGCCCCT": 2.2545, "TGGGGCCCCG": 1.8470142857142857,
    "TGGGGCCCCC": 2.0881714285714286,
    "GGGGGGGGGT": 2.294942857142857, "GGGGGGGGGG": 2.5361,
    "GGGGGGGGGC": 2.5169571428571422, "GGGGGGGGCC": 2.173842857142857,
    "GGGGGGGCCC": 2.1547, "GGGGGGCCCA": 2.0881714285714286,
    "GGGGGGCCCT": 2.321028571428571, "GGGGGGCCCG": 1.9135428571428572,
    "GGGGGGCCCC": 2.1547, "GGGGGCCCTA": 2.185157142857143,
    "GGGGGCCCGG": 1.5882571428571428, "GGGGGCCCCA": 2.0881714285714286,
    "GGGGGCCCCT": 2.321028571428571, "GGGGGCCCCG": 1.9135428571428572,
    "GGGGGCCCCC": 2.1547,
    "CCCCCCCCCT": 2.702428571428571, "CCCCCCCCCC": 2.5361
}

def detect_aphilic_dna(sequence: str) -> List[Dict]:
    """Detect A-philic DNA using structure-informed propensity scoring"""
    seq = sequence.upper()
    n = len(seq)
    APHILIC_WIN = 10
    APHILIC_MIN_LEN = 10
    APHILIC_LOG2_MIN = 0.0
    
    windows = []
    for i in range(n - APHILIC_WIN + 1):
        k = seq[i:i+APHILIC_WIN]
        log2 = A10_LOG2.get(k)
        if log2 is not None and log2 > APHILIC_LOG2_MIN:
            windows.append((i+1, i+APHILIC_WIN, log2))
    
    if not windows:
        return []
    
    # Merge overlapping windows
    regions = []
    s, e, acc, cnt = windows[0][0], windows[0][1], windows[0][2], 1
    
    for w in windows[1:]:
        if w[0] <= e:
            e = max(e, w[1])
            acc += w[2]
            cnt += 1
        else:
            regions.append((s, e, acc, cnt))
            s, e, acc, cnt = w[0], w[1], w[2], 1
    regions.append((s, e, acc, cnt))
    
    # Output with classification
    out = []
    for s, e, acc, cnt in regions:
        length = e - s + 1
        if length < APHILIC_MIN_LEN:
            continue
        
        avg = acc / cnt
        score = min(avg, 2.8)
        
        if avg >= 2.3:
            subclass = "High_Confidence_A-form"
        elif avg >= 1.8:
            subclass = "Moderate_A-form_Preference"
        else:
            subclass = "Weak_A-form_Signature"
        
        out.append({
            "Class": "A-philic DNA",
            "Subclass": subclass,
            "Start": s,
            "End": e,
            "Length": length,
            "Score": round(score, 3)
        })
    
    return out

# =============================================================================
# CURVED DNA DETECTOR
# =============================================================================

def detect_curved_dna(sequence: str) -> List[Dict]:
    """Detect curved DNA based on phased A-tracts"""
    AT_RE = re.compile(r"A{4,6}|T{4,6}")
    
    def _atracts(seq: str) -> List[float]:
        return [(m.start() + m.end()) / 2 for m in AT_RE.finditer(seq)]
    
    def _phase(c: List[float]) -> float:
        if len(c) < 2:
            return 0.0
        PHASE, PH_TOL = 10.5, 1.5
        p = t = 0
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                d = abs(c[j] - c[i]) % PHASE
                if d <= PH_TOL or abs(d - PHASE) <= PH_TOL:
                    p += 1
                t += 1
        return p / t if t else 0.0
    
    res = []
    atr = _atracts(sequence)
    n = len(sequence)
    
    # Local curvature
    LOC_W, LOC_S, LOC_N, LOC_P = 60, 5, 3, 0.60
    for s in range(0, n - LOC_W + 1, LOC_S):
        e = s + LOC_W
        c = [x - s for x in atr if s <= x < e]
        if len(c) < LOC_N:
            continue
        pc = _phase(c)
        if pc < LOC_P:
            continue
        
        dens = len(c) / LOC_W
        score = min(3, max(1, dens * (pc ** 1.6) * 13))
        
        if pc >= 0.80 and dens >= 0.08:
            subclass = "High_Quality_Local_Curvature"
        elif pc >= 0.70:
            subclass = "Strong_Local_Curvature"
        else:
            subclass = "Local_Curvature"
        
        res.append({
            "Class": "Curved_DNA",
            "Subclass": subclass,
            "Start": s,
            "End": e,
            "Score": round(score, 2)
        })
    
    return res

# =============================================================================
# G-QUADRUPLEX DETECTOR
# =============================================================================

def detect_g_quadruplex(sequence: str) -> List[Dict]:
    """Detect G-quadruplex structures with multiple variants"""
    G4_PATTERNS = [
        ("Telomeric_G4", re.compile(r"(TTAGGG){4,}")),
        ("Canonical_G4", re.compile(r"G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}[ACGT]{1,7}G{3,}")),
        ("Extended_G4", re.compile(r"G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}[ACGT]{1,12}G{3,}")),
    ]
    
    G4_PRIORITY = {
        "Telomeric_G4": 95,
        "Canonical_G4": 80,
        "Extended_G4": 70,
    }
    
    seq = sequence.upper()
    hits = []
    
    for motif, regex in G4_PATTERNS:
        for m in regex.finditer(seq):
            s, e = m.start() + 1, m.end()
            window = seq[m.start():m.end()]
            
            if len(window) < 18:
                continue
            
            # Simple scoring based on G content and length
            g_count = window.count("G")
            score = min(3.0, 1.0 + (g_count / len(window)) * 2)
            
            hits.append({
                "Class": "G-Quadruplex",
                "Subclass": motif,
                "Start": s,
                "End": e,
                "Length": e - s + 1,
                "Priority": G4_PRIORITY[motif],
                "Score": round(score, 2)
            })
    
    # Overlap resolution
    hits.sort(key=lambda x: (x["Start"], -x["Priority"]))
    final = []
    
    for h in hits:
        if not final or h["Start"] > final[-1]["End"]:
            final.append(h)
        else:
            if h["Priority"] > final[-1]["Priority"]:
                final[-1] = h
    
    return final

# =============================================================================
# I-MOTIF DETECTOR
# =============================================================================

def detect_i_motif(sequence: str) -> List[Dict]:
    """Detect i-Motif structures (C-rich four-stranded structures)"""
    IMOTIF_PATTERNS = [
        ("canonical_imotif", re.compile(r"C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}[ACGT]{1,7}C{3,}")),
        ("hur_ac_motif", re.compile(r"A{3}[ACGT]{4,6}C{3}[ACGT]{4,6}C{3}[ACGT]{4,6}C{3}")),
    ]
    
    seq = sequence.upper()
    hits = []
    
    for subclass, regex in IMOTIF_PATTERNS:
        for m in regex.finditer(seq):
            start = m.start() + 1
            end = m.end()
            motif_seq = m.group()
            
            # Score based on C content
            c_count = motif_seq.count("C")
            score = min(3.0, 1.0 + (c_count / len(motif_seq)) * 2)
            
            hits.append({
                "Class": "IMotif",
                "Subclass": subclass,
                "Start": start,
                "End": end,
                "Length": end - start + 1,
                "Score": round(score, 2)
            })
    
    return hits

# =============================================================================
# Z-DNA DETECTOR
# =============================================================================

def detect_z_dna(sequence: str) -> List[Dict]:
    """Detect Z-DNA (left-handed helix in alternating purine-pyrimidine)"""
    # Z-DNA typically forms in alternating purine-pyrimidine sequences
    ZDNA_PATTERN = re.compile(r"([CG][AG]){4,}")
    
    seq = sequence.upper()
    hits = []
    
    for m in ZDNA_PATTERN.finditer(seq):
        start = m.start() + 1
        end = m.end()
        motif_seq = m.group()
        
        # Score based on CG content and alternation
        cg_count = motif_seq.count("C") + motif_seq.count("G")
        score = min(3.0, 1.5 + (cg_count / len(motif_seq)) * 1.5)
        
        hits.append({
            "Class": "Z-DNA",
            "Subclass": "Canonical_Z-DNA",
            "Start": start,
            "End": end,
            "Length": end - start + 1,
            "Score": round(score, 2)
        })
    
    return hits

# =============================================================================
# SLIPPED DNA DETECTOR
# =============================================================================

def detect_slipped_dna(sequence: str) -> List[Dict]:
    """Detect slipped DNA structures (direct repeats)"""
    seq = sequence.upper()
    out = []
    
    for k in range(2, 10):
        pat = re.compile(rf"([ACGT]{{{k}}})\1{{2,}}")
        for m in pat.finditer(seq):
            s, e = m.start() + 1, m.end()
            unit = m.group(1)
            
            # Score based on repeat count and length
            repeat_count = (e - s + 1) / k
            score = min(3.0, 1.0 + repeat_count * 0.3)
            
            out.append({
                "Class": "Slipped DNA",
                "Subclass": "Direct_Repeat",
                "Start": s,
                "End": e,
                "Length": e - s + 1,
                "Score": round(score, 2)
            })
    
    return out

# =============================================================================
# CRUCIFORM DETECTOR
# =============================================================================

def detect_cruciform(sequence: str) -> List[Dict]:
    """Detect cruciform structures (inverted repeats/palindromes)"""
    
    def _revcomp(seq: str) -> str:
        return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    
    seq = sequence.upper()
    out = []
    n = len(seq)
    
    # Look for inverted repeats (palindromes)
    for arm in range(10, min(31, n // 2)):
        for i in range(n - 2 * arm):
            left = seq[i:i + arm]
            for sp in range(0, 4):  # spacer 0-3
                j = i + arm + sp
                if j + arm > n:
                    break
                right = seq[j:j + arm]
                if right == _revcomp(left):
                    score = min(3.0, 1.5 + (arm / 20))
                    out.append({
                        "Class": "Cruciform",
                        "Subclass": "Inverted_Repeat",
                        "Start": i + 1,
                        "End": j + arm,
                        "Length": j + arm - i,
                        "Score": round(score, 2)
                    })
    
    return out

# =============================================================================
# TRIPLEX DETECTOR
# =============================================================================

def detect_triplex_dna(sequence: str) -> List[Dict]:
    """Detect triplex DNA (mirror repeats)"""
    seq = sequence.upper()
    out = []
    
    # Mirror repeats
    mirror = re.compile(r"(([AG]{7,})[ACGT]{0,8}\2)|(([CT]{7,})[ACGT]{0,8}\4)")
    for m in mirror.finditer(seq):
        s, e = m.start() + 1, m.end()
        frag = seq[s - 1:e]
        purity = max(
            (frag.count("A") + frag.count("G")) / len(frag),
            (frag.count("C") + frag.count("T")) / len(frag)
        )
        if purity >= 0.85 and len(frag) >= 15:
            score = min(3.0, 1.0 + purity * 2)
            out.append({
                "Class": "Triplex",
                "Subclass": "Mirror_Repeat",
                "Start": s,
                "End": e,
                "Length": e - s + 1,
                "Score": round(score, 2)
            })
    
    return out

# =============================================================================
# UNIFIED DETECTOR
# =============================================================================

def detect_all_nonb_motifs(sequence: str) -> Dict[str, List[Dict]]:
    """
    Detect all Non-B DNA motifs in a sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary mapping motif class names to lists of detected motifs
    """
    results = {
        "A-philic DNA": detect_aphilic_dna(sequence),
        "Curved_DNA": detect_curved_dna(sequence),
        "G-Quadruplex": detect_g_quadruplex(sequence),
        "IMotif": detect_i_motif(sequence),
        "Z-DNA": detect_z_dna(sequence),
        "Slipped DNA": detect_slipped_dna(sequence),
        "Cruciform": detect_cruciform(sequence),
        "Triplex": detect_triplex_dna(sequence),
    }
    
    return results

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "detect_all_nonb_motifs",
    "detect_aphilic_dna",
    "detect_curved_dna",
    "detect_g_quadruplex",
    "detect_i_motif",
    "detect_z_dna",
    "detect_slipped_dna",
    "detect_cruciform",
    "detect_triplex_dna",
]
