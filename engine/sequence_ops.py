"""
Sequence Operations Module
===========================

Core sequence manipulation and validation functions.
Extracted from utilities.py to provide focused sequence operations.
"""

from typing import Tuple


def reverse_complement(sequence: str) -> str:
    """
    Generate reverse complement of DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Reverse complement sequence
        
    Example:
        >>> reverse_complement("ATGC")
        'GCAT'
    """
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return sequence.translate(complement)[::-1]


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate DNA sequence contains only valid bases.
    
    Args:
        sequence: DNA sequence to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> validate_sequence("ATGC")
        (True, "")
        >>> validate_sequence("ATGCX")
        (False, "Invalid characters found: X")
    """
    if not sequence:
        return False, "Empty sequence"
    
    valid_bases = set("ATGCatgcNn")
    invalid_chars = set(sequence) - valid_bases
    
    if invalid_chars:
        return False, f"Invalid characters found: {''.join(sorted(invalid_chars))}"
    
    return True, ""


def gc_content(sequence: str) -> float:
    """
    Calculate GC content percentage of sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        GC content as percentage (0-100)
        
    Example:
        >>> gc_content("ATGC")
        50.0
    """
    if not sequence:
        return 0.0
    
    seq_upper = sequence.upper()
    gc_count = seq_upper.count('G') + seq_upper.count('C')
    
    return (gc_count / len(sequence)) * 100
