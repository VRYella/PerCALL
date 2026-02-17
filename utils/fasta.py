"""
FASTA Parsing Module
====================

Functions for parsing FASTA format sequences.
Extracted from utilities.py for focused file format handling.
"""

from typing import Dict
import re


def parse_fasta(fasta_content: str) -> Dict[str, str]:
    """
    Parse FASTA format content into dictionary.
    
    Args:
        fasta_content: FASTA format string
        
    Returns:
        Dictionary mapping sequence_name -> sequence
        
    Example:
        >>> content = ">seq1\\nATGC\\n>seq2\\nGCTA"
        >>> parse_fasta(content)
        {'seq1': 'ATGC', 'seq2': 'GCTA'}
    """
    sequences = {}
    current_name = None
    current_seq = []
    
    for line in fasta_content.split('\n'):
        line = line.strip()
        
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_name is not None:
                sequences[current_name] = ''.join(current_seq)
            
            # Start new sequence
            current_name = line[1:].split()[0]  # Take first word after '>'
            current_seq = []
        elif line and current_name is not None:
            current_seq.append(line)
    
    # Save last sequence
    if current_name is not None:
        sequences[current_name] = ''.join(current_seq)
    
    return sequences


def read_fasta_file(filename: str) -> Dict[str, str]:
    """
    Read and parse FASTA file.
    
    Args:
        filename: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence_name -> sequence
        
    Example:
        >>> sequences = read_fasta_file("sequences.fasta")
        >>> len(sequences)
        2
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    return parse_fasta(content)


def format_fasta(sequences: Dict[str, str], line_width: int = 80) -> str:
    """
    Format sequences as FASTA with line wrapping.
    
    Args:
        sequences: Dictionary of name -> sequence
        line_width: Maximum characters per line
        
    Returns:
        FASTA formatted string
        
    Example:
        >>> seqs = {'seq1': 'ATGC' * 30}
        >>> fasta = format_fasta(seqs, line_width=60)
        >>> fasta.startswith('>seq1\\n')
        True
    """
    lines = []
    
    for name, seq in sequences.items():
        lines.append(f'>{name}')
        
        # Wrap sequence at line_width
        for i in range(0, len(seq), line_width):
            lines.append(seq[i:i + line_width])
    
    return '\n'.join(lines)
