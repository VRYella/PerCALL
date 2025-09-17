#!/usr/bin/env python3
"""
NBDFinder All Motifs Orchestrator
Parallel detection across all 11 motif classes with A-philic DNA as Class 9
"""

import concurrent.futures
import threading
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from classification_config import NBD_CLASSES, get_class_config, get_quality_filter
from motifs.a_philic_dna import detect_a_philic_motifs


class NBDMotifOrchestrator:
    """Orchestrator for parallel non-B DNA motif detection across all 11 classes."""
    
    def __init__(self, max_workers=4):
        """Initialize the orchestrator with parallel processing configuration."""
        self.max_workers = max_workers
        self.classes = NBD_CLASSES
        self.lock = threading.Lock()
        
    def detect_class_1_curved_dna(self, sequence: str) -> List[Dict]:
        """Detect Curved DNA motifs (Class 1)."""
        # Placeholder implementation - would use curvature analysis
        return []
    
    def detect_class_2_slipped_dna(self, sequence: str) -> List[Dict]:
        """Detect Slipped DNA motifs (Class 2)."""
        # Placeholder implementation - would use repeat analysis
        return []
    
    def detect_class_3_cruciform_dna(self, sequence: str) -> List[Dict]:
        """Detect Cruciform DNA motifs (Class 3)."""
        # Placeholder implementation - would use inverted repeat analysis
        return []
    
    def detect_class_4_rloop(self, sequence: str) -> List[Dict]:
        """Detect R-loop motifs (Class 4)."""
        # Placeholder implementation - would use R-loop analysis
        return []
    
    def detect_class_5_triplex(self, sequence: str) -> List[Dict]:
        """Detect Triplex DNA motifs (Class 5)."""
        # Placeholder implementation - would use triplex analysis
        return []
    
    def detect_class_6_g_quadruplex(self, sequence: str) -> List[Dict]:
        """Detect G-Quadruplex motifs (Class 6)."""
        # Placeholder implementation - would use G4 detection
        return []
    
    def detect_class_7_i_motif(self, sequence: str) -> List[Dict]:
        """Detect i-Motif structures (Class 7)."""
        # Placeholder implementation - would use i-motif analysis
        return []
    
    def detect_class_8_z_dna(self, sequence: str) -> List[Dict]:
        """Detect Z-DNA conformations (Class 8)."""
        # Placeholder implementation - would use Z-DNA analysis
        return []
    
    def detect_class_9_a_philic_dna(self, sequence: str) -> List[Dict]:
        """Detect A-philic DNA motifs (Class 9)."""
        config = get_class_config(9)
        min_len = config['s_min']
        max_len = config['s_max']
        
        # Use the A-philic DNA detector
        results = detect_a_philic_motifs(sequence, min_len=min_len, max_len=max_len)
        
        # Format results for NBDFinder framework
        formatted_results = []
        for result in results:
            formatted_results.append({
                'class_id': 9,
                'class_name': 'A-philic DNA',
                'start': result['start'],
                'end': result['end'],
                'length': result['length'],
                'sequence': result['sequence'],
                'score': result['sum_log2'],
                'confidence': result['confidence'],
                'classification': result['classification'],
                'method': 'tetranucleotide_log2_odds',
                'metadata': {
                    'n_tets': result['n_tets'],
                    'strong_count': result['strong_count'],
                    'sum_log2': result['sum_log2']
                }
            })
        
        return formatted_results
    
    def detect_class_10_hybrid(self, all_motifs: Dict) -> List[Dict]:
        """Detect Hybrid motifs (Class 10) - overlapping regions."""
        hybrid_motifs = []
        
        # Find overlapping regions between different classes
        motif_list = []
        for class_id, motifs in all_motifs.items():
            if class_id == 10:  # Skip hybrid class itself
                continue
            for motif in motifs:
                motif_list.append(motif)
        
        # Sort by start position
        motif_list.sort(key=lambda x: x['start'])
        
        # Find overlaps
        for i, motif1 in enumerate(motif_list):
            for motif2 in motif_list[i+1:]:
                if motif1['class_id'] == motif2['class_id']:
                    continue
                
                overlap_start = max(motif1['start'], motif2['start'])
                overlap_end = min(motif1['end'], motif2['end'])
                
                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    min_length = min(motif1['length'], motif2['length'])
                    overlap_fraction = overlap_length / min_length
                    
                    if overlap_fraction >= 0.5:  # Significant overlap
                        hybrid_motifs.append({
                            'class_id': 10,
                            'class_name': 'Hybrid',
                            'start': overlap_start,
                            'end': overlap_end,
                            'length': overlap_length,
                            'sequence': motif1['sequence'][overlap_start-motif1['start']:overlap_end-motif1['start']],
                            'score': (motif1['score'] + motif2['score']) / 2,
                            'confidence': 'High',
                            'classification': f"Hybrid_{motif1['class_name']}_{motif2['class_name']}",
                            'method': 'overlap_analysis',
                            'metadata': {
                                'component_classes': [motif1['class_id'], motif2['class_id']],
                                'overlap_fraction': overlap_fraction
                            }
                        })
        
        return hybrid_motifs
    
    def detect_class_11_clusters(self, all_motifs: Dict) -> List[Dict]:
        """Detect Non-B DNA Clusters (Class 11) - hotspot regions."""
        cluster_motifs = []
        
        # Collect all non-cluster motifs
        motif_list = []
        for class_id, motifs in all_motifs.items():
            if class_id in [10, 11]:  # Skip hybrid and cluster classes
                continue
            for motif in motifs:
                motif_list.append(motif)
        
        # Sort by start position
        motif_list.sort(key=lambda x: x['start'])
        
        # Find clusters
        cluster_distance = 100  # bp
        min_motifs = 3
        
        i = 0
        while i < len(motif_list):
            cluster_motifs_temp = [motif_list[i]]
            j = i + 1
            
            while j < len(motif_list) and motif_list[j]['start'] - motif_list[j-1]['end'] <= cluster_distance:
                cluster_motifs_temp.append(motif_list[j])
                j += 1
            
            if len(cluster_motifs_temp) >= min_motifs:
                cluster_start = cluster_motifs_temp[0]['start']
                cluster_end = cluster_motifs_temp[-1]['end']
                cluster_length = cluster_end - cluster_start
                
                cluster_motifs.append({
                    'class_id': 11,
                    'class_name': 'Non-B DNA Clusters',
                    'start': cluster_start,
                    'end': cluster_end,
                    'length': cluster_length,
                    'sequence': '',  # Would need full sequence to extract
                    'score': len(cluster_motifs_temp),
                    'confidence': 'High',
                    'classification': f"Cluster_{len(cluster_motifs_temp)}_motifs",
                    'method': 'cluster_analysis',
                    'metadata': {
                        'motif_count': len(cluster_motifs_temp),
                        'component_classes': [m['class_id'] for m in cluster_motifs_temp],
                        'density': len(cluster_motifs_temp) / cluster_length * 1000  # motifs per kb
                    }
                })
            
            i = j if j > i + 1 else i + 1
        
        return cluster_motifs
    
    def detect_single_class(self, sequence: str, class_id: int) -> List[Dict]:
        """Detect motifs for a single class."""
        detectors = {
            1: self.detect_class_1_curved_dna,
            2: self.detect_class_2_slipped_dna,
            3: self.detect_class_3_cruciform_dna,
            4: self.detect_class_4_rloop,
            5: self.detect_class_5_triplex,
            6: self.detect_class_6_g_quadruplex,
            7: self.detect_class_7_i_motif,
            8: self.detect_class_8_z_dna,
            9: self.detect_class_9_a_philic_dna,
        }
        
        detector = detectors.get(class_id)
        if detector:
            return detector(sequence)
        return []
    
    def detect_all_motifs(self, sequence: str, classes_to_run: Optional[List[int]] = None,
                         enable_parallel: bool = True) -> Dict[int, List[Dict]]:
        """
        Detect all NBD motifs across specified classes.
        
        Args:
            sequence: DNA sequence to analyze
            classes_to_run: List of class IDs to run (1-11), None for all
            enable_parallel: Whether to use parallel processing
            
        Returns:
            Dictionary mapping class IDs to lists of detected motifs
        """
        if classes_to_run is None:
            classes_to_run = list(range(1, 10))  # Classes 1-9 (primary detectors)
        
        all_motifs = {}
        
        # Detect primary classes (1-9)
        primary_classes = [c for c in classes_to_run if c <= 9]
        
        if enable_parallel and len(primary_classes) > 1:
            # Parallel detection
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_class = {
                    executor.submit(self.detect_single_class, sequence, class_id): class_id
                    for class_id in primary_classes
                }
                
                for future in concurrent.futures.as_completed(future_to_class):
                    class_id = future_to_class[future]
                    try:
                        results = future.result()
                        all_motifs[class_id] = results
                    except Exception as exc:
                        print(f'Class {class_id} detection generated an exception: {exc}')
                        all_motifs[class_id] = []
        else:
            # Sequential detection
            for class_id in primary_classes:
                all_motifs[class_id] = self.detect_single_class(sequence, class_id)
        
        # Detect hybrid motifs (Class 10) if requested
        if 10 in classes_to_run:
            all_motifs[10] = self.detect_class_10_hybrid(all_motifs)
        
        # Detect clusters (Class 11) if requested
        if 11 in classes_to_run:
            all_motifs[11] = self.detect_class_11_clusters(all_motifs)
        
        return all_motifs
    
    def apply_quality_filters(self, motifs: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        """Apply class-specific quality filters to detected motifs."""
        filtered_motifs = {}
        
        for class_id, motif_list in motifs.items():
            config = get_class_config(class_id)
            if not config:
                continue
            
            # Get appropriate quality filter
            filter_name = 'default'
            if class_id == 9:  # A-philic DNA
                filter_name = 'a_philic'
            elif class_id == 6:  # G-Quadruplex
                filter_name = 'g4'
            elif class_id == 3:  # Cruciform
                filter_name = 'cruciform'
            
            quality_filter = get_quality_filter(filter_name)
            
            filtered_list = []
            for motif in motif_list:
                # Apply filters
                if motif['score'] >= quality_filter['min_score'] and \
                   motif['length'] >= quality_filter['min_length']:
                    filtered_list.append(motif)
            
            filtered_motifs[class_id] = filtered_list
        
        return filtered_motifs
    
    def get_summary_stats(self, motifs: Dict[int, List[Dict]]) -> Dict:
        """Generate summary statistics for detected motifs."""
        stats = {
            'total_motifs': 0,
            'classes_detected': 0,
            'class_counts': {},
            'average_scores': {},
            'length_distributions': {}
        }
        
        for class_id, motif_list in motifs.items():
            if motif_list:
                stats['classes_detected'] += 1
                stats['class_counts'][class_id] = len(motif_list)
                stats['total_motifs'] += len(motif_list)
                
                scores = [m['score'] for m in motif_list if isinstance(m['score'], (int, float))]
                if scores:
                    stats['average_scores'][class_id] = np.mean(scores)
                
                lengths = [m['length'] for m in motif_list]
                if lengths:
                    stats['length_distributions'][class_id] = {
                        'min': min(lengths),
                        'max': max(lengths),
                        'mean': np.mean(lengths),
                        'std': np.std(lengths) if len(lengths) > 1 else 0
                    }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Test with A-philic DNA rich sequence
    test_sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGGATGCTAGCGATCGATCGTAGCGATCGTAGC"
    
    orchestrator = NBDMotifOrchestrator(max_workers=2)
    
    # Detect all motifs
    results = orchestrator.detect_all_motifs(test_sequence, classes_to_run=[9])
    
    # Apply quality filters
    filtered_results = orchestrator.apply_quality_filters(results)
    
    # Get summary
    summary = orchestrator.get_summary_stats(filtered_results)
    
    print("NBDFinder Orchestrator Test Results:")
    print(f"Total motifs detected: {summary['total_motifs']}")
    print(f"Classes with detections: {summary['classes_detected']}")
    
    for class_id, motifs in filtered_results.items():
        if motifs:
            config = get_class_config(class_id)
            print(f"\nClass {class_id} ({config['name']}): {len(motifs)} motifs")
            for motif in motifs[:3]:  # Show first 3
                print(f"  Position {motif['start']}-{motif['end']}: {motif['classification']} (score: {motif['score']})")