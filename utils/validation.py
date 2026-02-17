"""
Validation Module
=================

Comprehensive validation functions for DNA sequences, motifs, scores, and genomic coordinates.
Provides type checking, range validation, and format verification for all data types.

Extracted from utilities.py for focused validation logic following modular architecture.
"""

from typing import Tuple, Dict, Any, List, Optional
import re


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate DNA sequence contains only valid nucleotide characters.
    
    Accepts: A, T, G, C (case insensitive) and N for unknown bases.
    
    Args:
        sequence: DNA sequence string to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        error_message is empty string if valid
        
    Example:
        >>> validate_sequence("ATGC")
        (True, "")
        >>> validate_sequence("ATGCX")
        (False, "Invalid characters found: X")
        >>> validate_sequence("")
        (False, "Empty sequence")
    """
    if not sequence:
        return False, "Empty sequence"
    
    if not isinstance(sequence, str):
        return False, f"Sequence must be string, got {type(sequence).__name__}"
    
    # Valid DNA nucleotides (including N for unknown)
    valid_bases = set("ATGCNatgcn")
    sequence_chars = set(sequence)
    invalid_chars = sequence_chars - valid_bases
    
    if invalid_chars:
        invalid_str = ''.join(sorted(invalid_chars))
        return False, f"Invalid characters found: {invalid_str}"
    
    return True, ""


def validate_motif(motif: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate motif dictionary has required fields and valid values.
    
    Required fields: Class, Start, End, Score
    Optional fields: Subclass, Length, Sequence, Strand, Method
    
    Args:
        motif: Motif dictionary to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Example:
        >>> motif = {'Class': 'G-Quadruplex', 'Start': 100, 'End': 120, 'Score': 0.9}
        >>> validate_motif(motif)
        (True, "")
        >>> validate_motif({'Start': 100})
        (False, "Missing required field: Class")
    """
    if not isinstance(motif, dict):
        return False, f"Motif must be dictionary, got {type(motif).__name__}"
    
    # Check required fields
    required_fields = ['Class', 'Start', 'End', 'Score']
    for field in required_fields:
        if field not in motif:
            return False, f"Missing required field: {field}"
    
    # Validate coordinates
    valid_coords, coord_msg = validate_coordinates(
        motif.get('Start'), motif.get('End')
    )
    if not valid_coords:
        return False, f"Invalid coordinates: {coord_msg}"
    
    # Validate score
    valid_score, score_msg = validate_score(motif.get('Score'))
    if not valid_score:
        return False, f"Invalid score: {score_msg}"
    
    # Validate class name
    if not isinstance(motif['Class'], str) or not motif['Class'].strip():
        return False, "Class must be non-empty string"
    
    return True, ""


def validate_score(score: float, min_score: float = 0.0, max_score: float = 3.0) -> Tuple[bool, str]:
    """
    Validate motif score is within valid range.
    
    Default range is 0-3 for normalized scores, but can be customized.
    
    Args:
        score: Score value to validate
        min_score: Minimum valid score (default: 0.0)
        max_score: Maximum valid score (default: 3.0)
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Example:
        >>> validate_score(2.5)
        (True, "")
        >>> validate_score(5.0)
        (False, "Score 5.0 out of range [0.0, 3.0]")
        >>> validate_score("invalid")
        (False, "Score must be numeric, got str")
    """
    # Check type
    if not isinstance(score, (int, float)):
        return False, f"Score must be numeric, got {type(score).__name__}"
    
    # Check range
    if not (min_score <= score <= max_score):
        return False, f"Score {score} out of range [{min_score}, {max_score}]"
    
    return True, ""


def validate_coordinates(start: int, end: int, sequence_length: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate genomic coordinates are valid and properly ordered.
    
    Checks:
    - Both are positive integers
    - Start < End
    - Within sequence bounds if sequence_length provided
    
    Args:
        start: Start position (1-based)
        end: End position (inclusive)
        sequence_length: Optional sequence length for bounds checking
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Example:
        >>> validate_coordinates(100, 200)
        (True, "")
        >>> validate_coordinates(200, 100)
        (False, "Start 200 must be less than end 100")
        >>> validate_coordinates(100, 200, 150)
        (False, "End position 200 exceeds sequence length 150")
    """
    # Check types
    if not isinstance(start, int):
        return False, f"Start must be integer, got {type(start).__name__}"
    if not isinstance(end, int):
        return False, f"End must be integer, got {type(end).__name__}"
    
    # Check positive
    if start < 1:
        return False, f"Start position {start} must be >= 1"
    if end < 1:
        return False, f"End position {end} must be >= 1"
    
    # Check ordering
    if start >= end:
        return False, f"Start {start} must be less than end {end}"
    
    # Check bounds if sequence length provided
    if sequence_length is not None:
        if end > sequence_length:
            return False, f"End position {end} exceeds sequence length {sequence_length}"
    
    return True, ""


def validate_strand(strand: str) -> Tuple[bool, str]:
    """
    Validate strand is valid DNA strand indicator.
    
    Valid values: '+', '-', or '.' for unknown
    
    Args:
        strand: Strand indicator
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Example:
        >>> validate_strand('+')
        (True, "")
        >>> validate_strand('x')
        (False, "Invalid strand 'x', must be '+', '-', or '.'")
    """
    if not isinstance(strand, str):
        return False, f"Strand must be string, got {type(strand).__name__}"
    
    valid_strands = {'+', '-', '.'}
    if strand not in valid_strands:
        return False, f"Invalid strand '{strand}', must be '+', '-', or '.'"
    
    return True, ""


def validate_fasta_format(content: str) -> Tuple[bool, str]:
    """
    Validate content is properly formatted FASTA.
    
    Checks for:
    - Presence of header lines (starting with '>')
    - Non-empty sequences
    - Valid nucleotide characters
    
    Args:
        content: FASTA format string
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Example:
        >>> fasta = ">seq1\\nATGC\\n>seq2\\nGCTA"
        >>> validate_fasta_format(fasta)
        (True, "")
        >>> validate_fasta_format("ATGC")
        (False, "No FASTA headers found (lines starting with '>')")
    """
    if not content or not content.strip():
        return False, "Empty FASTA content"
    
    lines = content.strip().split('\n')
    
    # Check for at least one header
    headers = [line for line in lines if line.strip().startswith('>')]
    if not headers:
        return False, "No FASTA headers found (lines starting with '>')"
    
    # Validate sequences
    current_seq = []
    current_header = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            # Check previous sequence if exists
            if current_header and not current_seq:
                return False, f"Empty sequence for header: {current_header}"
            
            current_header = line
            current_seq = []
        elif line:
            current_seq.append(line)
            # Validate sequence characters
            valid, msg = validate_sequence(''.join(current_seq))
            if not valid:
                return False, f"Invalid sequence in {current_header}: {msg}"
    
    # Check final sequence
    if current_header and not current_seq:
        return False, f"Empty sequence for header: {current_header}"
    
    return True, ""
