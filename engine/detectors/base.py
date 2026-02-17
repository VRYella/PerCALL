"""
Base Motif Detector Module
==========================

Abstract base class for all Non-B DNA motif detectors.
Provides common detection interface, pattern compilation, and scoring framework.

Extracted from detectors.py for modular architecture.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseMotifDetector(ABC):
    """
    Abstract base class for all Non-B DNA motif detectors.
    
    Motif Output Structure:
        ID (str): Unique motif identifier
        Sequence_Name (str): Source sequence name
        Class (str): Motif class (e.g., 'Curved_DNA')
        Subclass (str): Motif subclass/variant
        Start (int): 1-based start position
        End (int): End position (inclusive)
        Length (int): Motif length in bp
        Sequence (str): Actual DNA sequence
        Score (float): Detection confidence score (0-1)
        Strand (str): Strand orientation ('+' or '-')
        Method (str): Detection method identifier
        Pattern_ID (str): Pattern identifier used for match
    """
    
    def __init__(self):
        self.patterns = self.get_patterns()
        self.compiled_patterns = self._compile_patterns()
        # Detector execution audit - tracks detection pipeline
        self.audit = {
            'invoked': False,
            'windows_scanned': 0,
            'candidates_seen': 0,
            'candidates_filtered': 0,
            'reported': 0,
            'seed_hits': 0,
            'both_strands_scanned': False
        }
    
    @abstractmethod
    def get_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Return patterns specific to this motif class.
        
        Pattern Structure:
            regex (str): Regular expression pattern
            pattern_id (str): Unique pattern identifier
            name (str): Human-readable name
            subclass (str): Motif subclass
            min_length (int): Minimum match length
            score_type (str): Scoring method name
            threshold (float): Quality threshold (0-1)
            description (str): Pattern description
            reference (str): Literature citation
        """
        pass
    
    @abstractmethod  
    def get_motif_class_name(self) -> str:
        """Return the motif class name (e.g., 'Curved_DNA', 'G_Quadruplex')"""
        pass
    
    @abstractmethod
    def calculate_score(self, sequence: str, pattern_info: Tuple) -> float:
        """
        Calculate motif-specific confidence score.
        
        Args:
            sequence: DNA sequence string (uppercase)
            pattern_info: Pattern tuple with metadata
            
        Returns:
            Score value between 0.0 and 1.0
        """
        pass
    
    def _compile_patterns(self) -> Dict[str, List[Tuple]]:
        """
        Compile all regex patterns once for performance.
        Uses re.IGNORECASE | re.ASCII for optimal DNA sequence matching.
        """
        compiled_patterns = {}
        
        for pattern_group, patterns in self.patterns.items():
            compiled_group = []
            for pattern_info in patterns:
                pattern, pattern_id, name, subclass = pattern_info[:4]
                try:
                    compiled_re = re.compile(pattern, re.IGNORECASE | re.ASCII)
                    compiled_group.append((compiled_re, pattern_id, name, subclass, pattern_info))
                except re.error as e:
                    print(f"Warning: Invalid pattern {pattern}: {e}")
                    continue
            compiled_patterns[pattern_group] = compiled_group
        
        return compiled_patterns
    
    def detect_motifs(self, sequence: str, sequence_name: str = "sequence") -> List[Dict[str, Any]]:
        """
        Main detection method - scans sequence for all compiled patterns.
        
        Detection Process:
            1. Normalize sequence to uppercase
            2. Iterate through compiled patterns
            3. Find all regex matches
            4. Calculate motif-specific scores
            5. Apply quality thresholds
            6. Return list of motif dictionaries
        
        Args:
            sequence: DNA sequence string
            sequence_name: Identifier for the sequence
            
        Returns:
            List of motif dictionaries with standardized fields
        """
        # Reset and mark audit as invoked
        self.audit['invoked'] = True
        self.audit['windows_scanned'] = 1
        self.audit['candidates_seen'] = 0
        self.audit['candidates_filtered'] = 0
        self.audit['reported'] = 0
        self.audit['seed_hits'] = 0
        
        sequence = sequence.upper().strip()
        motifs = []
        
        for pattern_group, compiled_patterns in self.compiled_patterns.items():
            for compiled_re, pattern_id, name, subclass, full_info in compiled_patterns:
                for match in compiled_re.finditer(sequence):
                    self.audit['seed_hits'] += 1
                    self.audit['candidates_seen'] += 1
                    start, end = match.span()
                    motif_seq = sequence[start:end]
                    
                    score = self.calculate_score(motif_seq, full_info)
                    
                    if self.passes_quality_threshold(motif_seq, score, full_info):
                        motifs.append({
                            'ID': f"{sequence_name}_{pattern_id}_{start+1}",
                            'Sequence_Name': sequence_name,
                            'Class': self.get_motif_class_name(),
                            'Subclass': subclass,
                            'Start': start + 1,
                            'End': end,
                            'Length': len(motif_seq),
                            'Sequence': motif_seq,
                            'Score': round(score, 3),
                            'Strand': '+',
                            'Method': f'{self.get_motif_class_name()}_detection',
                            'Pattern_ID': pattern_id
                        })
                        self.audit['reported'] += 1
                    else:
                        self.audit['candidates_filtered'] += 1
        
        return motifs
    
    def passes_quality_threshold(self, sequence: str, score: float, pattern_info: Tuple) -> bool:
        """Apply quality thresholds - can be overridden by subclasses"""
        # Default threshold from pattern info if available
        if len(pattern_info) > 6:
            min_threshold = pattern_info[6]  # confidence/threshold from pattern
            return score >= min_threshold
        
        # Default minimum score threshold
        return score >= 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        total_patterns = sum(len(patterns) for patterns in self.patterns.values())
        return {
            'motif_class': self.get_motif_class_name(),
            'total_patterns': total_patterns,
            'pattern_groups': list(self.patterns.keys()),
            'patterns_by_group': {k: len(v) for k, v in self.patterns.items()}
        }
    
    def get_audit_info(self) -> Dict[str, Any]:
        """Get detector execution audit information"""
        return self.audit.copy()
