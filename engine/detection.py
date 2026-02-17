"""
Engine Detection Module - Core NonBScanner Class
================================================

This module contains the main NonBScanner class that orchestrates all motif detection.
It implements the core detection pipeline that runs all individual detectors and
performs post-processing including overlap removal, hybrid detection, and clustering.

Key Components:
    - NonBScanner: Main scanner class coordinating detector execution
    - AnalysisProgress: Progress tracking and reporting
    - Progress callbacks: Functions for tracking analysis progress

Pipeline Architecture:
    1. DETECTION PHASE: Run all individual detectors (9 detectors)
    2. INTERPRETATION PHASE: Post-process results (overlap removal, hybrids, clusters)
    3. NORMALIZATION PHASE: Score normalization and sorting

Module Integration:
    - Uses detectors from engine.detectors package
    - Uses scoring from engine.scoring module
    - Uses merging from engine.merging module
    - Uses validation from utils.validation module
"""

import time
import warnings
import threading
import bisect
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

# Import detector classes from modular architecture
from engine.detectors import (
    CurvedDNADetector,
    SlippedDNADetector,
    CruciformDetector,
    RLoopDetector,
    TriplexDetector,
    GQuadruplexDetector,
    IMotifDetector,
    ZDNADetector,
    APhilicDetector
)

# Import utilities from modular architecture
from utils.validation import validate_sequence
from engine.scoring import normalize_motif_scores

__all__ = [
    'NonBScanner',
    'AnalysisProgress',
    'create_progress_callback',
    'create_enhanced_progress_callback',
    'get_last_detector_timings',
    'get_detector_display_names',
    'get_cached_scanner',
    'analyze_sequence',  # Convenience function
]

# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

# Human-readable display names for detectors
DETECTOR_DISPLAY_NAMES = {
    'curved_dna': 'Curved DNA',
    'slipped_dna': 'Slipped DNA',
    'cruciform': 'Cruciform',
    'r_loop': 'R-Loop',
    'triplex': 'Triplex',
    'g_quadruplex': 'G-Quadruplex',
    'i_motif': 'i-Motif',
    'z_dna': 'Z-DNA',
    'a_philic': 'A-philic DNA'
}

# Hybrid detection parameters
HYBRID_MIN_OVERLAP = 0.50  # Minimum 50% overlap
HYBRID_MAX_OVERLAP = 0.99  # Maximum 99% overlap

# Cluster detection parameters
CLUSTER_WINDOW_SIZE = 300   # 300 bp sliding window
CLUSTER_MIN_MOTIFS = 4      # Minimum 4 motifs per cluster
CLUSTER_MIN_CLASSES = 3     # Minimum 3 different classes

# Detector timing tracking (thread-safe)
_DETECTOR_TIMINGS = {}
_TIMINGS_LOCK = threading.Lock()

# Cached scanner singleton
_CACHED_SCANNER = None
_SCANNER_LOCK = threading.Lock()


# =============================================================================
# ANALYSIS PROGRESS TRACKING
# =============================================================================

class AnalysisProgress:
    """
    Enhanced progress tracking for Non-B DNA analysis.
    
    Provides real-time progress updates with visual indicators,
    timing information, and stage-by-stage tracking.
    
    Pipeline Stages:
        - INPUT: Parse and validate input sequences
        - DETECTION: Run 9 specialized motif detectors
        - PROCESSING: Overlap removal, hybrid/cluster detection
        - OUTPUT: Score normalization and result formatting
    
    Example:
        >>> progress = AnalysisProgress(sequence_length=10000)
        >>> progress.start_stage('DETECTION')
        >>> progress.update_detector('curved_dna', completed=True, motifs=5, elapsed=0.12)
        >>> summary = progress.get_summary()
    """
    
    # Pipeline stages with descriptions
    STAGES = {
        'INPUT': {
            'name': 'Input Parsing',
            'description': 'Parse FASTA and validate sequences',
            'icon': '📥'
        },
        'DETECTION': {
            'name': 'Motif Detection',
            'description': 'Run 9 specialized detectors',
            'icon': '[DETECT]'
        },
        'PROCESSING': {
            'name': 'Post-Processing',
            'description': 'Overlap removal, hybrid/cluster detection',
            'icon': '[PROCESS]'
        },
        'OUTPUT': {
            'name': 'Output Generation',
            'description': 'Score normalization and formatting',
            'icon': '[OUTPUT]'
        }
    }
    
    def __init__(self, sequence_length: int = 0, sequence_name: str = "sequence"):
        """Initialize progress tracker."""
        self.sequence_length = sequence_length
        self.sequence_name = sequence_name
        self.start_time = time.time()
        self.current_stage = None
        self.stage_times = {}
        
        # Detector tracking
        self.detector_status = {}
        for name in DETECTOR_DISPLAY_NAMES:
            self.detector_status[name] = {
                'status': 'pending',
                'elapsed': 0.0,
                'motifs': 0,
                'speed': 0.0
            }
        
        # Overall statistics
        self.total_motifs = 0
        self.total_bp_processed = 0
        self.errors = []
    
    def start_stage(self, stage: str) -> None:
        """Start a new pipeline stage."""
        if stage in self.STAGES:
            self.current_stage = stage
            self.stage_times[stage] = {'start': time.time(), 'end': None}
    
    def end_stage(self, stage: str) -> None:
        """End a pipeline stage."""
        if stage in self.stage_times:
            self.stage_times[stage]['end'] = time.time()
    
    def update_detector(self, detector_name: str, completed: bool = False, 
                       motifs: int = 0, elapsed: float = 0.0, 
                       error: Optional[str] = None) -> None:
        """Update detector status."""
        if detector_name in self.detector_status:
            if error:
                self.detector_status[detector_name]['status'] = 'error'
                self.errors.append(f"{detector_name}: {error}")
            elif completed:
                self.detector_status[detector_name]['status'] = 'complete'
                self.detector_status[detector_name]['elapsed'] = elapsed
                self.detector_status[detector_name]['motifs'] = motifs
                if elapsed > 0:
                    self.detector_status[detector_name]['speed'] = self.sequence_length / elapsed
                self.total_motifs += motifs
            else:
                self.detector_status[detector_name]['status'] = 'running'
    
    def get_progress_percentage(self) -> float:
        """Get overall progress as percentage (0-100)."""
        completed = sum(1 for d in self.detector_status.values() 
                       if d['status'] in ('complete', 'error'))
        return (completed / len(self.detector_status)) * 100
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_throughput(self) -> float:
        """Get current processing throughput in bp/s."""
        elapsed = self.get_elapsed_time()
        if elapsed > 0:
            return self.sequence_length / elapsed
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        return {
            'sequence_name': self.sequence_name,
            'sequence_length': self.sequence_length,
            'percentage': self.get_progress_percentage(),
            'elapsed': self.get_elapsed_time(),
            'throughput': self.get_throughput(),
            'current_stage': self.current_stage,
            'detectors': self.detector_status.copy(),
            'total_motifs': self.total_motifs,
            'errors': self.errors.copy()
        }


# =============================================================================
# PROGRESS CALLBACK FUNCTIONS
# =============================================================================

def create_progress_callback(progress: AnalysisProgress) -> Callable:
    """Create a detector callback that updates an AnalysisProgress instance."""
    def callback(detector_name: str, completed: int, total: int, elapsed: float, motif_count: int):
        progress.update_detector(detector_name, completed=True, 
                                motifs=motif_count, elapsed=elapsed)
    return callback


def create_enhanced_progress_callback(
    progress: AnalysisProgress,
    print_updates: bool = False,
    on_detector_complete: Optional[Callable] = None
) -> Callable:
    """Create an enhanced detector callback with optional console output."""
    def callback(detector_name: str, completed: int, total: int, elapsed: float, motif_count: int):
        progress.update_detector(detector_name, completed=True, 
                                motifs=motif_count, elapsed=elapsed)
        
        if print_updates:
            display_name = DETECTOR_DISPLAY_NAMES.get(detector_name, detector_name)
            pct = (completed / total) * 100
            print(f"  [{pct:5.1f}%] {display_name}: {elapsed:.3f}s ({motif_count} motifs)")
        
        if on_detector_complete:
            on_detector_complete(detector_name, completed, total, elapsed, motif_count)
    
    return callback


def get_last_detector_timings() -> Dict[str, float]:
    """Get timing information from the last sequence analysis."""
    with _TIMINGS_LOCK:
        return dict(_DETECTOR_TIMINGS)


def get_detector_display_names() -> Dict[str, str]:
    """Get human-readable display names for all detectors."""
    return dict(DETECTOR_DISPLAY_NAMES)


def _update_detector_timing(detector_name: str, elapsed: float) -> None:
    """Update the timing for a specific detector (internal use)."""
    global _DETECTOR_TIMINGS
    with _TIMINGS_LOCK:
        _DETECTOR_TIMINGS[detector_name] = elapsed


def _reset_detector_timings() -> None:
    """Reset all detector timings (internal use)."""
    global _DETECTOR_TIMINGS
    with _TIMINGS_LOCK:
        _DETECTOR_TIMINGS = {}


# =============================================================================
# MAIN SCANNER CLASS
# =============================================================================

class NonBScanner:
    """
    Main scanner class orchestrating all motif detectors.
    
    This class implements the core detection pipeline:
        1. Run all individual detectors (DETECTION PHASE)
        2. Remove overlaps within same class (INTERPRETATION PHASE)
        3. Detect hybrid motifs (cross-class overlaps)
        4. Detect cluster motifs (high-density regions)
        5. Normalize scores and sort results
    
    The pipeline maintains strict separation between detection (candidate
    enumeration) and interpretation (scoring, filtering, biological analysis).
    
    Example:
        >>> scanner = NonBScanner()
        >>> motifs = scanner.analyze_sequence("GGGTTAGGGTTAGGGTTAGGG", "test_seq")
        >>> print(f"Found {len(motifs)} motifs")
    """
    
    def __init__(self, enable_all_detectors: bool = True):
        """
        Initialize NonBScanner with all detector modules.
        
        Args:
            enable_all_detectors: Enable all 9 detector classes (default: True)
        """
        self.detectors = {}
        
        if enable_all_detectors:
            self.detectors = {
                'curved_dna': CurvedDNADetector(),
                'slipped_dna': SlippedDNADetector(),
                'cruciform': CruciformDetector(),
                'r_loop': RLoopDetector(),
                'triplex': TriplexDetector(),
                'g_quadruplex': GQuadruplexDetector(),
                'i_motif': IMotifDetector(),
                'z_dna': ZDNADetector(),
                'a_philic': APhilicDetector()
            }
    
    def analyze_sequence(self, sequence: str, sequence_name: str = "sequence",
                        progress_callback: Optional[Callable[[str, int, int, float, int], None]] = None
                        ) -> List[Dict[str, Any]]:
        """
        Detect all Non-B DNA motifs in a sequence.
        
        Pipeline:
            1. Validate sequence
            2. Run all detectors (DETECTION PHASE)
            3. Remove overlaps within same class
            4. Detect hybrid motifs (cross-class overlaps)
            5. Detect cluster motifs (high-density regions)
            6. Normalize scores to universal 1-3 scale
            7. Sort by genomic position
        
        Args:
            sequence: DNA sequence to analyze (ATGC characters)
            sequence_name: Identifier for the sequence
            progress_callback: Optional callback after each detector completes
                             Signature: callback(detector_name, completed, total, elapsed, motif_count)
        
        Returns:
            List of motif dictionaries sorted by genomic position
        """
        sequence = sequence.upper().strip()
        
        # Validate sequence
        is_valid, msg = validate_sequence(sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {msg}")
        
        all_motifs = []
        total_detectors = len(self.detectors)
        
        # Reset detector timings
        _reset_detector_timings()
        
        # PHASE 1: DETECTION - Run all detectors
        for idx, (detector_name, detector) in enumerate(self.detectors.items()):
            try:
                start_time = time.time()
                motifs = detector.detect_motifs(sequence, sequence_name)
                elapsed = time.time() - start_time
                motif_count = len(motifs)
                
                # Track timing
                _update_detector_timing(detector_name, elapsed)
                
                all_motifs.extend(motifs)
                
                # Call callback if provided
                if progress_callback is not None:
                    progress_callback(detector_name, idx + 1, total_detectors, elapsed, motif_count)
                    
            except Exception as e:
                warnings.warn(f"Error in {detector_name} detector: {e}")
                if progress_callback is not None:
                    progress_callback(detector_name, idx + 1, total_detectors, 0.0, 0)
                continue
        
        # PHASE 2: INTERPRETATION - Post-detection processing
        
        # Step 3: Remove overlaps within same class
        filtered_motifs = self._remove_overlaps(all_motifs)
        
        # Step 4: Detect hybrid motifs (multi-class overlaps)
        hybrid_motifs = self._detect_hybrid_motifs(filtered_motifs, sequence)
        
        # Step 5: Detect cluster motifs (high-density regions)
        cluster_motifs = self._detect_clusters(filtered_motifs, sequence)
        
        # Apply overlap removal to hybrid and cluster motifs
        hybrid_motifs = self._remove_overlaps(hybrid_motifs)
        cluster_motifs = self._remove_overlaps(cluster_motifs)
        
        final_motifs = filtered_motifs + hybrid_motifs + cluster_motifs
        
        # Step 6: Normalize scores to universal 1-3 scale
        final_motifs = normalize_motif_scores(final_motifs)
        
        # Step 7: Sort by genomic position
        final_motifs.sort(key=lambda x: x.get('Start', 0))
        
        return final_motifs
    
    def _remove_overlaps(self, motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove overlapping motifs within the same class/subclass.
        
        Uses deterministic rules: Priority × Score × Length
        - Higher Score wins
        - If scores equal, longer motif wins
        - If both equal, first in sequence wins
        """
        if not motifs:
            return motifs
        
        # Group by class/subclass
        groups = defaultdict(list)
        for motif in motifs:
            key = f"{motif.get('Class', '')}-{motif.get('Subclass', '')}"
            groups[key].append(motif)
        
        filtered_motifs = []
        
        for group_motifs in groups.values():
            if len(group_motifs) <= 1:
                filtered_motifs.extend(group_motifs)
                continue
            
            # Sort by Score (desc) and Length (desc) for deterministic ordering
            group_motifs.sort(key=lambda x: (-x.get('Score', 0), -x.get('Length', 0)))
            
            non_overlapping = []
            accepted_intervals = []
            
            for motif in group_motifs:
                start, end = motif.get('Start', 0), motif.get('End', 0)
                overlaps = False
                
                # Check overlap with accepted intervals
                for acc_start, acc_end in accepted_intervals:
                    if not (end <= acc_start or start >= acc_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    non_overlapping.append(motif)
                    bisect.insort(accepted_intervals, (start, end))
            
            filtered_motifs.extend(non_overlapping)
        
        return filtered_motifs
    
    def _calculate_overlap(self, motif1: Dict[str, Any], motif2: Dict[str, Any]) -> float:
        """Calculate overlap ratio between two motifs."""
        start1, end1 = motif1.get('Start', 0), motif1.get('End', 0)
        start2, end2 = motif2.get('Start', 0), motif2.get('End', 0)
        
        if end1 <= start2 or end2 <= start1:
            return 0.0
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_length = overlap_end - overlap_start
        
        min_length = min(end1 - start1, end2 - start2)
        return overlap_length / min_length if min_length > 0 else 0.0
    
    def _detect_hybrid_motifs(self, motifs: List[Dict[str, Any]], sequence: str) -> List[Dict[str, Any]]:
        """Detect hybrid motifs (overlapping different classes)."""
        if len(motifs) < 2:
            return []
        
        hybrid_motifs = []
        seen_hybrids = set()
        
        # Sort motifs by start position
        sorted_motifs = sorted(motifs, key=lambda x: (x.get('Start', 0), x.get('End', 0)))
        
        n = len(sorted_motifs)
        for i in range(n):
            motif1 = sorted_motifs[i]
            start1, end1 = motif1.get('Start', 0), motif1.get('End', 0)
            class1 = motif1.get('Class', '')
            
            for j in range(i + 1, n):
                motif2 = sorted_motifs[j]
                start2 = motif2.get('Start', 0)
                
                # Early termination
                if start2 >= end1:
                    break
                
                class2 = motif2.get('Class', '')
                
                # Only detect hybrids between different classes
                if class1 == class2:
                    continue
                
                end2 = motif2.get('End', 0)
                overlap = self._calculate_overlap(motif1, motif2)
                
                if HYBRID_MIN_OVERLAP < overlap < HYBRID_MAX_OVERLAP:
                    start = min(start1, start2)
                    end = max(end1, end2)
                    
                    hybrid_key = (start, end, frozenset([class1, class2]))
                    if hybrid_key in seen_hybrids:
                        continue
                    seen_hybrids.add(hybrid_key)
                    
                    avg_score = (motif1.get('Score', 0) + motif2.get('Score', 0)) / 2
                    
                    # Extract sequence
                    seq_text = 'HYBRID_REGION'
                    if 0 <= start - 1 < len(sequence) and 0 < end <= len(sequence):
                        seq_text = sequence[start-1:end]
                    
                    hybrid_motifs.append({
                        'ID': f"{motif1.get('Sequence_Name', 'seq')}_HYBRID_{start}",
                        'Sequence_Name': motif1.get('Sequence_Name', 'sequence'),
                        'Class': 'Hybrid',
                        'Subclass': f"{class1}_{class2}_Overlap",
                        'Start': start,
                        'End': end,
                        'Length': end - start,
                        'Sequence': seq_text,
                        'Score': round(avg_score, 3),
                        'Strand': '+',
                        'Method': 'Hybrid_Detection',
                        'Component_Classes': [class1, class2]
                    })
        
        return hybrid_motifs
    
    def _detect_clusters(self, motifs: List[Dict[str, Any]], sequence: str) -> List[Dict[str, Any]]:
        """Detect high-density non-B DNA clusters."""
        if len(motifs) < 3:
            return []
        
        cluster_motifs = []
        window_size = CLUSTER_WINDOW_SIZE
        min_density = CLUSTER_MIN_MOTIFS
        min_classes = CLUSTER_MIN_CLASSES
        
        # Sort motifs by start position
        sorted_motifs = sorted(motifs, key=lambda x: x.get('Start', 0))
        n = len(sorted_motifs)
        
        detected_window_starts = set()
        
        for i in range(n):
            window_start = sorted_motifs[i].get('Start', 0)
            window_end = window_start + window_size
            
            if window_start in detected_window_starts:
                continue
            
            # Find all motifs within the window
            window_motifs = []
            for j in range(i, n):
                motif_start = sorted_motifs[j].get('Start', 0)
                if motif_start <= window_end:
                    window_motifs.append(sorted_motifs[j])
                else:
                    break
            
            if len(window_motifs) >= min_density:
                classes = set(m.get('Class') for m in window_motifs)
                if len(classes) >= min_classes:
                    actual_start = min(m.get('Start', 0) for m in window_motifs)
                    actual_end = max(m.get('End', 0) for m in window_motifs)
                    
                    detected_window_starts.add(window_start)
                    
                    avg_score = sum(m.get('Score', 0) for m in window_motifs) / len(window_motifs)
                    
                    # Extract sequence
                    seq_text = 'CLUSTER_REGION'
                    if 0 <= actual_start - 1 < len(sequence) and 0 < actual_end <= len(sequence):
                        seq_text = sequence[actual_start-1:actual_end]
                    
                    cluster_motifs.append({
                        'ID': f"{sorted_motifs[i].get('Sequence_Name', 'seq')}_CLUSTER_{actual_start}",
                        'Sequence_Name': sorted_motifs[i].get('Sequence_Name', 'sequence'),
                        'Class': 'Non-B_DNA_Clusters',
                        'Subclass': f'Mixed_Cluster_{len(classes)}_classes',
                        'Start': actual_start,
                        'End': actual_end,
                        'Length': actual_end - actual_start,
                        'Sequence': seq_text,
                        'Score': round(avg_score, 3),
                        'Strand': '+',
                        'Method': 'Cluster_Detection',
                        'Motif_Count': len(window_motifs),
                        'Class_Diversity': len(classes),
                        'Component_Classes': list(classes)
                    })
        
        return cluster_motifs
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about all loaded detectors."""
        info = {
            'total_detectors': len(self.detectors),
            'detectors': {},
            'total_patterns': 0
        }
        
        for name, detector in self.detectors.items():
            stats = detector.get_statistics()
            info['detectors'][name] = stats
            info['total_patterns'] += stats['total_patterns']
        
        return info


# =============================================================================
# CACHED SCANNER SINGLETON
# =============================================================================

def get_cached_scanner() -> NonBScanner:
    """
    Get or create cached NonBScanner instance with pre-initialized detectors.
    
    Thread-safe singleton pattern using double-checked locking.
    Avoids re-initializing detectors on repeated calls.
    
    Returns:
        Cached NonBScanner instance
    """
    global _CACHED_SCANNER
    if _CACHED_SCANNER is None:
        with _SCANNER_LOCK:
            # Double-check inside lock
            if _CACHED_SCANNER is None:
                _CACHED_SCANNER = NonBScanner(enable_all_detectors=True)
    return _CACHED_SCANNER


# =============================================================================
# CONVENIENCE API FUNCTION
# =============================================================================

def analyze_sequence(sequence: str, sequence_name: str = "sequence",
                     progress_callback: Optional[Callable[[str, int, int, float, int], None]] = None
                     ) -> List[Dict[str, Any]]:
    """
    Convenience function to analyze a sequence using the cached scanner instance.
    
    This provides a simple functional API without needing to manage scanner instances.
    
    Args:
        sequence: DNA sequence to analyze (ATGC characters)
        sequence_name: Identifier for the sequence
        progress_callback: Optional callback after each detector completes
                          Signature: callback(detector_name, completed, total, elapsed, motif_count)
    
    Returns:
        List of motif dictionaries sorted by genomic position
    
    Example:
        >>> from engine.detection import analyze_sequence
        >>> motifs = analyze_sequence("GGGTTAGGGTTAGGGTTAGGG", "test_seq")
        >>> print(f"Found {len(motifs)} motifs")
    """
    scanner = get_cached_scanner()
    return scanner.analyze_sequence(sequence, sequence_name, progress_callback)
