"""
Scoring Module
==============

Score normalization and calculation functions for Non-B DNA motifs.
Provides universal 1-3 scale normalization for cross-motif comparability.

Extracted from utilities.py for focused scoring logic following modular architecture.
"""

from typing import List, Dict, Any
import math

# Scientific normalization parameters per motif class
# Format: {'class_name': {'min_raw': min_threshold, 'max_raw': max_threshold, 'method': str}}
SCORE_NORMALIZATION_PARAMS = {
    'Curved_DNA': {
        'min_raw': 0.1,   # Minimal curvature threshold
        'max_raw': 0.95,  # Strong curvature (APR phasing)
        'method': 'linear',
        'ref': 'Koo 1986, Olson 1998'
    },
    'Slipped_DNA': {
        'min_raw': 0.3,   # Minimal repeat instability
        'max_raw': 0.98,  # High repeat instability (STRs)
        'method': 'linear',
        'ref': 'Schlötterer 2000, Weber 1989'
    },
    'Cruciform': {
        'min_raw': 0.5,   # Minimum palindrome stability
        'max_raw': 0.95,  # Strong inverted repeat
        'method': 'linear',
        'ref': 'Lilley 2000, Sinden 1994'
    },
    'R-Loop': {
        'min_raw': 0.4,   # Minimal R-loop potential
        'max_raw': 0.95,  # High R-loop formation
        'method': 'linear',
        'ref': 'Aguilera 2012, Jenjaroenpun 2016'
    },
    'Triplex': {
        'min_raw': 0.5,   # Minimal triplex potential
        'max_raw': 0.95,  # Strong H-DNA formation
        'method': 'linear',
        'ref': 'Frank-Kamenetskii 1995'
    },
    'G-Quadruplex': {
        'min_raw': 0.5,   # G4Hunter threshold ~1.2 normalized
        'max_raw': 1.0,   # Perfect G4 structure
        'method': 'g4hunter',  # Special G4Hunter-based normalization
        'ref': 'Bedrat 2016, Burge 2006'
    },
    'i-Motif': {
        'min_raw': 0.4,   # Minimal i-motif stability
        'max_raw': 0.98,  # High C-tract density
        'method': 'linear',
        'ref': 'Gehring 1993, Zeraati 2018'
    },
    'Z-DNA': {
        'min_raw': 50.0,    # Single Z-DNA 10-mer score minimum (Ho 1986)
        'max_raw': 2000.0,  # Cumulative max for long Z-DNA regions (~30+ bp)
        'method': 'zdna_cumulative',  # Uses log-linear scaling for cumulative scores
        'ref': 'Ho 1986, Rich 1984'
    },
    'A-philic_DNA': {
        'min_raw': 0.5,   # Minimal A-philic propensity
        'max_raw': 0.95,  # High A-tract density
        'method': 'linear',
        'ref': 'Gorin 1995, Vinogradov 2003'
    },
    'Hybrid': {
        'min_raw': 0.3,   # Minimal hybrid score
        'max_raw': 0.9,   # Strong hybrid overlap
        'method': 'linear',
        'ref': 'This work'
    },
    'Non-B_DNA_Clusters': {
        'min_raw': 0.3,   # Minimal cluster score
        'max_raw': 0.9,   # High-density cluster
        'method': 'linear',
        'ref': 'This work'
    }
}


def normalize_score_to_1_3(raw_score: float, motif_class: str) -> float:
    """
    Normalize raw detector score to 1-3 scale.
    
    Scientific basis:
    - 1.0 = Minimal (below typical formation threshold)
    - 1.5 = Low (marginal formation potential)
    - 2.0 = Moderate (typical formation potential)
    - 2.5 = High (strong formation potential)
    - 3.0 = Highest (exceptional formation potential)
    
    Args:
        raw_score: Raw score from detector (typically 0-1 or class-specific range)
        motif_class: Motif class name for class-specific normalization
        
    Returns:
        Normalized score between 1.0 and 3.0
        
    Example:
        >>> normalize_score_to_1_3(0.8, 'G-Quadruplex')
        2.6
    """
    # Get normalization parameters for this class
    params = SCORE_NORMALIZATION_PARAMS.get(motif_class)
    
    if params is None:
        # Default linear normalization for unknown classes
        params = {'min_raw': 0.0, 'max_raw': 1.0, 'method': 'linear'}
    
    min_raw = params['min_raw']
    max_raw = params['max_raw']
    method = params.get('method', 'linear')
    
    # Handle special normalization methods
    if method == 'g4hunter':
        # G4Hunter scores: abs value indicates formation potential
        # Typical range is 0-2+, with 1.2+ being significant
        # Always use absolute value for G4Hunter scores
        effective_score = abs(raw_score)
        if effective_score < 0.5:
            return 1.0
        elif effective_score >= 1.0:
            return min(3.0, 2.0 + (effective_score - 0.5) * 2.0)
        else:
            return 1.0 + (effective_score - 0.5) * 2.0
    
    elif method == 'zdna_cumulative':
        # Z-DNA cumulative scores (sum of multiple 10-mer matches)
        # Use log-linear scaling for better distribution
        if raw_score < min_raw:
            return 1.0
        elif raw_score >= max_raw:
            return 3.0
        else:
            # Log-scale normalization for cumulative scores
            log_score = math.log10(raw_score + 1)
            log_min = math.log10(min_raw + 1)
            log_max = math.log10(max_raw + 1)
            normalized = (log_score - log_min) / (log_max - log_min)
            return 1.0 + normalized * 2.0
    
    elif method == 'zdna_10mer':
        # Z-DNA 10-mer scores range from ~50 to 63 (deprecated, use zdna_cumulative)
        if raw_score < min_raw:
            return 1.0
        elif raw_score >= max_raw:
            return 3.0
        else:
            normalized = (raw_score - min_raw) / (max_raw - min_raw)
            return 1.0 + normalized * 2.0
    
    else:  # Linear normalization (default)
        if raw_score <= min_raw:
            return 1.0
        elif raw_score >= max_raw:
            return 3.0
        else:
            # Linear interpolation between 1 and 3
            normalized = (raw_score - min_raw) / (max_raw - min_raw)
            return 1.0 + normalized * 2.0


def normalize_motif_scores(motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize all motif scores to universal 1-3 scale for cross-motif comparability.
    
    UNIVERSAL SCORE DISCIPLINE (ΔG-inspired Scale):
    This function enforces a unified scoring framework across all motif classes,
    enabling meaningful comparison and ranking of structurally diverse Non-B DNA
    motifs. The scale is inspired by ΔG (Gibbs free energy) conventions.
    
    SCORE INTERPRETATION BANDS:
    - 1.0 - 1.7: Weak/Conditional (Low confidence, context-dependent)
    - 1.7 - 2.3: Moderate (Reasonable confidence, likely valid)
    - 2.3 - 3.0: Strong/High (High confidence, well-characterized)
    
    CROSS-MOTIF COMPARABILITY GUARANTEE:
    - G-Quadruplex score of 2.5 is comparable to Curved DNA score of 2.5
    - Higher scores always indicate stronger/more confident predictions
    - Scores are normalized AFTER detection (never influence candidate finding)
    - Normalization is deterministic and reproducible
    
    Args:
        motifs: List of motif dictionaries with 'Score' and 'Class' fields
        
    Returns:
        List of motifs with updated 'Score' field (1-3 scale)
        
    Note:
        Original raw score is preserved in 'Raw_Score' field for reference.
        Raw scores use class-specific scales and are not directly comparable.
    
    Example:
        >>> motifs = [
        ...     {'Class': 'G-Quadruplex', 'Score': 0.8},
        ...     {'Class': 'Curved_DNA', 'Score': 0.6}
        ... ]
        >>> normalized = normalize_motif_scores(motifs)
        >>> # Both scores now on comparable 1-3 scale
        >>> assert 1.0 <= normalized[0]['Score'] <= 3.0
        >>> assert 1.0 <= normalized[1]['Score'] <= 3.0
    """
    normalized_motifs = []
    
    for motif in motifs:
        m = motif.copy()
        raw_score = m.get('Score', 0.0)
        motif_class = m.get('Class', 'Unknown')
        
        # Store raw score for reference (class-specific scale)
        m['Raw_Score'] = raw_score
        
        # Normalize to universal 1-3 scale (cross-motif comparable)
        m['Score'] = round(normalize_score_to_1_3(raw_score, motif_class), 2)
        
        normalized_motifs.append(m)
    
    return normalized_motifs


def calculate_motif_statistics(motifs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for a collection of motifs.
    
    Args:
        motifs: List of motif dictionaries
        
    Returns:
        Dictionary with statistical summaries
    """
    if not motifs:
        return {
            'total_count': 0,
            'avg_score': 0.0,
            'avg_length': 0.0,
            'score_range': (0.0, 0.0),
            'length_range': (0, 0)
        }
    
    scores = [m.get('Score', 0.0) for m in motifs]
    lengths = [m.get('Length', 0) for m in motifs]
    
    return {
        'total_count': len(motifs),
        'avg_score': sum(scores) / len(scores) if scores else 0.0,
        'avg_length': sum(lengths) / len(lengths) if lengths else 0.0,
        'score_range': (min(scores) if scores else 0.0, max(scores) if scores else 0.0),
        'length_range': (min(lengths) if lengths else 0, max(lengths) if lengths else 0)
    }
