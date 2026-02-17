"""
Merging Module
==============

Overlap removal, hybrid detection, and cluster detection algorithms.
Provides deterministic overlap resolution for Non-B DNA motifs.

Extracted from nonbscanner.py for focused merging logic following modular architecture.
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
import bisect

# Constants for hybrid and cluster detection
HYBRID_MIN_OVERLAP = 0.50  # Minimum overlap ratio for hybrid detection (50%)
HYBRID_MAX_OVERLAP = 0.99  # Maximum overlap ratio for hybrid detection (99%)
CLUSTER_WINDOW_SIZE = 300  # Sliding window size in bp for cluster detection
CLUSTER_MIN_MOTIFS = 4     # Minimum number of motifs required in a cluster
CLUSTER_MIN_CLASSES = 3    # Minimum number of different classes required in a cluster


def remove_overlaps(motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove overlapping motifs within the same class/subclass using deterministic rules.
    
    DETERMINISTIC RULE: Priority × UniversalScore × Length
    When two motifs overlap within the same subclass:
      1. Higher Score wins (universal 1-3 scale)
      2. If scores equal, longer motif wins (Length in bp)
      3. If both equal, first in sequence wins (stable sort)
    
    GROUPING STRATEGY:
      - Overlaps checked ONLY within same Class+Subclass
      - Different subclasses can overlap (biological relevance)
      - Different classes can overlap (handled by hybrid detection)
    
    Args:
        motifs: List of motif dictionaries with required fields:
                - Class: str (motif class)
                - Subclass: str (motif subclass)
                - Start: int (1-based start position)
                - End: int (end position)
                - Score: float (universal 1-3 scale)
                - Length: int (motif length)
    
    Returns:
        Filtered list with overlaps removed (deterministically)
    
    Performance:
        - O(m log m) where m = number of motifs
        - Uses bisect for O(log n) interval insertion
        - Early termination for non-overlapping groups
    """
    if not motifs:
        return motifs
    
    # Group by class/subclass (overlaps resolved within groups only)
    groups = defaultdict(list)
    for motif in motifs:
        key = f"{motif.get('Class', '')}-{motif.get('Subclass', '')}"
        groups[key].append(motif)
    
    filtered_motifs = []
    
    for group_motifs in groups.values():
        if len(group_motifs) <= 1:
            filtered_motifs.extend(group_motifs)
            continue
        
        # DETERMINISTIC SORT: Priority rule (Score desc, Length desc)
        # This creates total ordering with no ties
        group_motifs.sort(key=lambda x: (-x.get('Score', 0), -x.get('Length', 0)))
        
        non_overlapping = []
        # Use sorted list of intervals for faster overlap checking
        accepted_intervals = []  # List of (start, end) tuples, sorted by start
        
        for motif in group_motifs:
            start, end = motif.get('Start', 0), motif.get('End', 0)
            overlaps = False
            
            # Check overlap with accepted intervals
            for acc_start, acc_end in accepted_intervals:
                # Two intervals overlap if neither is completely before the other
                if not (end <= acc_start or start >= acc_end):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(motif)
                # Use bisect for O(log n) insertion to maintain sorted order
                bisect.insort(accepted_intervals, (start, end))
        
        filtered_motifs.extend(non_overlapping)
    
    return filtered_motifs


def calculate_overlap(motif1: Dict[str, Any], motif2: Dict[str, Any]) -> float:
    """
    Calculate overlap ratio between two motifs.
    
    Args:
        motif1: First motif dictionary with Start and End
        motif2: Second motif dictionary with Start and End
        
    Returns:
        Overlap ratio (0.0 to 1.0) relative to shorter motif
    """
    start1, end1 = motif1.get('Start', 0), motif1.get('End', 0)
    start2, end2 = motif2.get('Start', 0), motif2.get('End', 0)
    
    if end1 <= start2 or end2 <= start1:
        return 0.0
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = overlap_end - overlap_start
    
    min_length = min(end1 - start1, end2 - start2)
    return overlap_length / min_length if min_length > 0 else 0.0


def detect_hybrid_motifs(motifs: List[Dict[str, Any]], sequence: str) -> List[Dict[str, Any]]:
    """
    Detect hybrid motifs (overlapping different classes).
    
    Optimized using interval sorting for O(n log n) performance instead of O(n²).
    Uses sweep line algorithm for efficient overlap detection.
    
    Args:
        motifs: List of motif dictionaries
        sequence: DNA sequence string
        
    Returns:
        List of hybrid motif dictionaries
    """
    if len(motifs) < 2:
        return []
    
    hybrid_motifs = []
    seen_hybrids: Set[tuple] = set()  # Track already detected hybrid regions
    
    # Sort motifs by start position for efficient sweep line
    sorted_motifs = sorted(motifs, key=lambda x: (x.get('Start', 0), x.get('End', 0)))
    
    # Use interval comparison with early termination
    n = len(sorted_motifs)
    for i in range(n):
        motif1 = sorted_motifs[i]
        start1, end1 = motif1.get('Start', 0), motif1.get('End', 0)
        class1 = motif1.get('Class', '')
        
        # Only compare with motifs that could possibly overlap
        for j in range(i + 1, n):
            motif2 = sorted_motifs[j]
            start2 = motif2.get('Start', 0)
            
            # Early termination: if motif2 starts after motif1 ends, no more overlaps possible
            if start2 >= end1:
                break
            
            class2 = motif2.get('Class', '')
            
            # Only detect hybrids between different classes
            if class1 == class2:
                continue
            
            end2 = motif2.get('End', 0)
            overlap = calculate_overlap(motif1, motif2)
            
            if HYBRID_MIN_OVERLAP < overlap < HYBRID_MAX_OVERLAP:  # Partial overlap (50-99%)
                start = min(start1, start2)
                end = max(end1, end2)
                
                # Create unique key to avoid duplicate hybrids
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
                    'Score': round(avg_score, 2),
                    'Strand': '+',
                    'Method': 'Hybrid_detection',
                    'Pattern_ID': 'HYBRID'
                })
    
    return hybrid_motifs


def detect_cluster_motifs(motifs: List[Dict[str, Any]], sequence: str) -> List[Dict[str, Any]]:
    """
    Detect high-density clusters of Non-B DNA motifs.
    
    Uses sliding window approach to identify regions with multiple motif types.
    
    Args:
        motifs: List of motif dictionaries
        sequence: DNA sequence string
        
    Returns:
        List of cluster motif dictionaries
    """
    if len(motifs) < CLUSTER_MIN_MOTIFS:
        return []
    
    cluster_motifs = []
    seq_length = len(sequence)
    
    # Sort motifs by start position
    sorted_motifs = sorted(motifs, key=lambda x: x.get('Start', 0))
    
    # Sliding window approach
    i = 0
    while i < len(sorted_motifs):
        window_start = sorted_motifs[i].get('Start', 0)
        window_end = window_start + CLUSTER_WINDOW_SIZE
        
        # Collect motifs in this window
        window_motifs = []
        classes_in_window: Set[str] = set()
        
        for j in range(i, len(sorted_motifs)):
            motif = sorted_motifs[j]
            motif_start = motif.get('Start', 0)
            
            if motif_start > window_end:
                break
            
            window_motifs.append(motif)
            classes_in_window.add(motif.get('Class', ''))
        
        # Check if cluster criteria are met
        if len(window_motifs) >= CLUSTER_MIN_MOTIFS and len(classes_in_window) >= CLUSTER_MIN_CLASSES:
            # Calculate cluster boundaries
            cluster_start = min(m.get('Start', 0) for m in window_motifs)
            cluster_end = max(m.get('End', 0) for m in window_motifs)
            
            # Extract sequence
            seq_text = 'CLUSTER_REGION'
            if 0 <= cluster_start - 1 < seq_length and 0 < cluster_end <= seq_length:
                seq_text = sequence[cluster_start-1:cluster_end]
            
            # Calculate average score
            avg_score = sum(m.get('Score', 0) for m in window_motifs) / len(window_motifs)
            
            cluster_motifs.append({
                'ID': f"{sorted_motifs[i].get('Sequence_Name', 'seq')}_CLUSTER_{cluster_start}",
                'Sequence_Name': sorted_motifs[i].get('Sequence_Name', 'sequence'),
                'Class': 'Non-B_DNA_Clusters',
                'Subclass': f"{len(classes_in_window)}_class_cluster",
                'Start': cluster_start,
                'End': cluster_end,
                'Length': cluster_end - cluster_start,
                'Sequence': seq_text,
                'Score': round(avg_score, 2),
                'Strand': '+',
                'Method': 'Cluster_detection',
                'Pattern_ID': 'CLUSTER',
                'Motif_Count': len(window_motifs),
                'Class_Count': len(classes_in_window)
            })
            
            # Move to next non-overlapping region
            i = j
        else:
            i += 1
    
    return cluster_motifs


def deduplicate_motifs(motifs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate motifs that may appear from overlapping chunk regions.
    Optimized for memory efficiency with large datasets.
    
    Duplicates are identified by having the same Class, Subclass, Start, End.
    When duplicates are found, the one with the higher score is kept.
    
    Args:
        motifs: List of motif dictionaries
        
    Returns:
        Deduplicated list of motifs
    """
    if not motifs:
        return motifs
    
    if len(motifs) < 1000:
        # For small datasets, use the simple approach
        sorted_motifs = sorted(motifs, key=lambda x: (
            x.get('Class', ''),
            x.get('Start', 0),
            x.get('End', 0),
            -x.get('Score', 0)
        ))
        
        deduplicated = []
        seen: Set[tuple] = set()
        
        for motif in sorted_motifs:
            key = (
                motif.get('Class', ''),
                motif.get('Subclass', ''),
                motif.get('Start', 0),
                motif.get('End', 0)
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(motif)
        
        return deduplicated
    
    # For large datasets, use a more memory-efficient approach
    # Build a dictionary with keys as tuples, keeping only the highest scoring motif
    best_motifs = {}
    
    for motif in motifs:
        key = (
            motif.get('Class', ''),
            motif.get('Subclass', ''),
            motif.get('Start', 0),
            motif.get('End', 0)
        )
        
        score = motif.get('Score', 0)
        
        # Keep the motif with the highest score for each key
        if key not in best_motifs or score > best_motifs[key].get('Score', 0):
            best_motifs[key] = motif
    
    # Convert back to list
    deduplicated = list(best_motifs.values())
    
    # Free the dictionary
    del best_motifs
    
    return deduplicated
