"""
NBDFinder All Motifs Orchestrator (Refactored)

This module orchestrates the parallel detection of all 11 motif classes in the NBDFinder system,
including the new A-philic DNA detection (Class 9) with comprehensive integration.

Updated Class System:
1. Curved DNA - Intrinsic DNA curvature patterns
2. Slipped DNA - Direct repeats and STR sequences
3. Cruciform DNA - Inverted repeat structures  
4. R-loop - RNA-DNA hybrid formation sites
5. Triplex - Triple-helix DNA structures
6. G-Quadruplex - G4 and variant formations
7. i-Motif - C-rich quadruplex structures
8. Z-DNA - Left-handed DNA conformations
9. A-philic DNA - A-tract-favoring protein binding sites (NEW)
10. Hybrid - Multi-class overlapping regions (updated from Class 9)
11. Non-B DNA Clusters - Hotspot regions (updated from Class 10)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

# Import motif detection modules
from motifs.a_philic_dna import find_a_philic_dna
from classification_config import NBDFinderConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBDFinderOrchestrator:
    """
    Main orchestrator for parallel NBDFinder motif detection across all 11 classes
    """
    
    def __init__(self, config: Optional[NBDFinderConfig] = None):
        """
        Initialize orchestrator with configuration
        
        Args:
            config: NBDFinder configuration object (optional)
        """
        self.config = config or NBDFinderConfig()
        
        # Define motif detection functions for each class
        # Note: Most are placeholder implementations except A-philic DNA (Class 9)
        self.detector_functions = {
            1: self._detect_curved_dna,
            2: self._detect_slipped_dna,
            3: self._detect_cruciform_dna,
            4: self._detect_r_loop,
            5: self._detect_triplex,
            6: self._detect_g_quadruplex,
            7: self._detect_i_motif,
            8: self._detect_z_dna,
            9: self._detect_a_philic_dna,  # Fully implemented
            10: self._detect_hybrid,
            11: self._detect_non_b_clusters
        }
        
        # A-philic specific quality thresholding parameters
        self.a_philic_quality_params = {
            'min_score': 10.0,
            'score_weight': 0.4,
            'length_weight': 0.3,
            'confidence_weight': 0.3
        }
    
    def _detect_a_philic_dna(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """
        Detect A-philic DNA motifs with quality thresholding
        
        Args:
            sequence: DNA sequence to analyze
            sequence_id: Sequence identifier
            
        Returns:
            List of A-philic DNA motifs meeting quality criteria
        """
        logger.info(f"Detecting A-philic DNA motifs for sequence {sequence_id}")
        
        # Use the implemented A-philic DNA finder
        motifs = find_a_philic_dna(sequence, sequence_id)
        
        # Apply A-philic specific quality filters
        filtered_motifs = []
        for motif in motifs:
            if self._passes_a_philic_quality_filter(motif):
                # Add orchestrator metadata
                motif['detector'] = 'NBDFinder_A_philic'
                motif['version'] = '1.0'
                motif['detection_time'] = time.time()
                filtered_motifs.append(motif)
        
        logger.info(f"A-philic DNA: {len(filtered_motifs)} motifs passed quality filter")
        return filtered_motifs
    
    def _passes_a_philic_quality_filter(self, motif: Dict[str, Any]) -> bool:
        """
        Check if A-philic motif passes quality thresholding
        
        Args:
            motif: A-philic motif dictionary
            
        Returns:
            True if motif passes quality criteria
        """
        # Primary score filter - use motif's own scoring, not orchestrator min_score
        if motif.get('Score', 0) < 1.5:  # Use lower threshold for tetranucleotide scores
            return False
        
        # Length filter (must be within config constraints)  
        length = motif.get('Length', 0)
        min_len, max_len = self.config.get_motif_limits(9)
        # Allow longer sequences for A-philic motifs since they can form extended regions
        if length < min_len or length > 200:  # Increase max length for A-tracts
            return False
        
        # Calculate composite quality score
        score_component = min(motif.get('Score', 0) / 20.0, 1.0)  # Normalize to max 20
        length_component = min(length / max_len, 1.0)
        confidence_component = 1.0 if motif.get('confidence') == 'high' else 0.7
        
        composite_quality = (
            score_component * self.a_philic_quality_params['score_weight'] +
            length_component * self.a_philic_quality_params['length_weight'] +  
            confidence_component * self.a_philic_quality_params['confidence_weight']
        )
        
        # Require composite quality > 0.6
        return composite_quality > 0.6
    
    # Placeholder implementations for other motif classes
    # These would contain actual detection algorithms in a complete implementation
    
    def _detect_curved_dna(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Curved DNA detection (Class 1)"""
        logger.info(f"Detecting Curved DNA motifs for sequence {sequence_id}")
        # Placeholder - would implement curvature analysis
        return []
    
    def _detect_slipped_dna(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Slipped DNA detection (Class 2)"""
        logger.info(f"Detecting Slipped DNA motifs for sequence {sequence_id}")
        # Placeholder - would implement repeat detection
        return []
    
    def _detect_cruciform_dna(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Cruciform DNA detection (Class 3)"""
        logger.info(f"Detecting Cruciform DNA motifs for sequence {sequence_id}")
        # Placeholder - would implement inverted repeat analysis
        return []
    
    def _detect_r_loop(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for R-loop detection (Class 4)"""
        logger.info(f"Detecting R-loop motifs for sequence {sequence_id}")
        # Placeholder - would implement R-loop propensity analysis
        return []
    
    def _detect_triplex(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Triplex detection (Class 5)"""
        logger.info(f"Detecting Triplex motifs for sequence {sequence_id}")
        # Placeholder - would implement triplex formation analysis
        return []
    
    def _detect_g_quadruplex(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for G-Quadruplex detection (Class 6)"""
        logger.info(f"Detecting G-Quadruplex motifs for sequence {sequence_id}")
        # Placeholder - would implement G4 scoring
        return []
    
    def _detect_i_motif(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for i-Motif detection (Class 7)"""
        logger.info(f"Detecting i-Motif motifs for sequence {sequence_id}")
        # Placeholder - would implement i-motif scoring
        return []
    
    def _detect_z_dna(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Z-DNA detection (Class 8)"""
        logger.info(f"Detecting Z-DNA motifs for sequence {sequence_id}")
        # Placeholder - would implement Z-DNA propensity analysis
        return []
    
    def _detect_hybrid(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Hybrid motif detection (Class 10)"""
        logger.info(f"Detecting Hybrid motifs for sequence {sequence_id}")
        # Placeholder - would implement overlap analysis
        return []
    
    def _detect_non_b_clusters(self, sequence: str, sequence_id: str) -> List[Dict[str, Any]]:
        """Placeholder for Non-B DNA Cluster detection (Class 11)"""
        logger.info(f"Detecting Non-B DNA Cluster motifs for sequence {sequence_id}")
        # Placeholder - would implement cluster density analysis
        return []
    
    def detect_all_motifs(self, sequence: str, sequence_id: str = "unknown", 
                         parallel: bool = True, 
                         selected_classes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Detect all motif types in parallel across the 11-class system
        
        Args:
            sequence: DNA sequence to analyze
            sequence_id: Identifier for the sequence
            parallel: Whether to use parallel processing (default: True)
            selected_classes: List of specific classes to detect (default: all)
            
        Returns:
            Dictionary containing detection results for all classes
        """
        logger.info(f"Starting NBDFinder analysis for sequence {sequence_id}")
        logger.info(f"Sequence length: {len(sequence)} bp")
        
        if not sequence or len(sequence) < self.config.global_params['min_sequence_length']:
            logger.warning(f"Sequence too short for analysis: {len(sequence)} bp")
            return self._empty_results()
        
        # Determine which classes to run
        classes_to_run = selected_classes or list(range(1, 12))
        
        results = {
            'sequence_id': sequence_id,
            'sequence_length': len(sequence),
            'analysis_time': time.time(),
            'classes_analyzed': classes_to_run,
            'motifs_by_class': {},
            'summary_stats': {}
        }
        
        if parallel and len(classes_to_run) > 1:
            # Parallel processing
            results['motifs_by_class'] = self._detect_parallel(
                sequence, sequence_id, classes_to_run
            )
        else:
            # Sequential processing
            results['motifs_by_class'] = self._detect_sequential(
                sequence, sequence_id, classes_to_run
            )
        
        # Generate summary statistics
        results['summary_stats'] = self._generate_summary_stats(results['motifs_by_class'])
        
        logger.info(f"NBDFinder analysis complete. Total motifs: {results['summary_stats']['total_motifs']}")
        return results
    
    def _detect_parallel(self, sequence: str, sequence_id: str, 
                        classes_to_run: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Run motif detection in parallel"""
        motifs_by_class = {}
        
        with ThreadPoolExecutor(max_workers=min(len(classes_to_run), 8)) as executor:
            # Submit detection tasks
            future_to_class = {
                executor.submit(
                    self.detector_functions[cls], sequence, sequence_id
                ): cls for cls in classes_to_run
            }
            
            # Collect results
            for future in as_completed(future_to_class):
                cls = future_to_class[future]
                try:
                    motifs = future.result()
                    motifs_by_class[cls] = motifs
                    logger.info(f"Class {cls} ({self.config.get_class_name(cls)}): {len(motifs)} motifs")
                except Exception as e:
                    logger.error(f"Error detecting Class {cls}: {e}")
                    motifs_by_class[cls] = []
        
        return motifs_by_class
    
    def _detect_sequential(self, sequence: str, sequence_id: str,
                          classes_to_run: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Run motif detection sequentially"""
        motifs_by_class = {}
        
        for cls in classes_to_run:
            try:
                motifs = self.detector_functions[cls](sequence, sequence_id)
                motifs_by_class[cls] = motifs
                logger.info(f"Class {cls} ({self.config.get_class_name(cls)}): {len(motifs)} motifs")
            except Exception as e:
                logger.error(f"Error detecting Class {cls}: {e}")
                motifs_by_class[cls] = []
        
        return motifs_by_class
    
    def _generate_summary_stats(self, motifs_by_class: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate summary statistics for detection results"""
        total_motifs = sum(len(motifs) for motifs in motifs_by_class.values())
        
        stats = {
            'total_motifs': total_motifs,
            'motifs_per_class': {cls: len(motifs) for cls, motifs in motifs_by_class.items()},
            'class_names': {cls: self.config.get_class_name(cls) for cls in motifs_by_class.keys()}
        }
        
        # A-philic specific statistics
        if 9 in motifs_by_class:
            a_philic_motifs = motifs_by_class[9]
            high_conf = len([m for m in a_philic_motifs if m.get('confidence') == 'high'])
            moderate_conf = len([m for m in a_philic_motifs if m.get('confidence') == 'moderate'])
            
            stats['a_philic_breakdown'] = {
                'high_confidence': high_conf,
                'moderate_confidence': moderate_conf,
                'avg_score': np.mean([m.get('Score', 0) for m in a_philic_motifs]) if a_philic_motifs else 0
            }
        
        return stats
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'sequence_id': 'unknown',
            'sequence_length': 0,
            'analysis_time': time.time(),
            'classes_analyzed': [],
            'motifs_by_class': {},
            'summary_stats': {'total_motifs': 0, 'motifs_per_class': {}, 'class_names': {}}
        }


def analyze_sequence_nbd_finder(sequence: str, sequence_id: str = "unknown",
                               parallel: bool = True,
                               selected_classes: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Main entry point for NBDFinder sequence analysis
    
    Args:
        sequence: DNA sequence to analyze
        sequence_id: Identifier for the sequence
        parallel: Whether to use parallel processing
        selected_classes: Specific classes to analyze (default: all)
        
    Returns:
        Complete NBDFinder analysis results
    """
    orchestrator = NBDFinderOrchestrator()
    return orchestrator.detect_all_motifs(sequence, sequence_id, parallel, selected_classes)


if __name__ == "__main__":
    # Test orchestrator with A-philic DNA detection
    print("NBDFinder Orchestrator Test")
    print("=" * 50)
    
    # Test sequence with A-philic motifs
    test_sequence = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAATTTTTTTTTTTTTTGGGCCCAAAAAATTTTTGGGCCCAAAAAATTTTTGGGCCCAAA"
    
    # Test just A-philic DNA detection (Class 9)
    print("Testing A-philic DNA detection (Class 9):")
    results = analyze_sequence_nbd_finder(
        test_sequence, 
        "test_sequence",
        parallel=False,
        selected_classes=[9]
    )
    
    print(f"Sequence length: {results['sequence_length']} bp")
    print(f"Classes analyzed: {results['classes_analyzed']}")
    print(f"Total motifs: {results['summary_stats']['total_motifs']}")
    
    if 9 in results['motifs_by_class']:
        a_philic_motifs = results['motifs_by_class'][9]
        print(f"A-philic motifs found: {len(a_philic_motifs)}")
        
        if 'a_philic_breakdown' in results['summary_stats']:
            breakdown = results['summary_stats']['a_philic_breakdown']
            print(f"  High confidence: {breakdown['high_confidence']}")
            print(f"  Moderate confidence: {breakdown['moderate_confidence']}")
            print(f"  Average score: {breakdown['avg_score']:.1f}")
    
    print(f"\nSample A-philic motifs:")
    if 9 in results['motifs_by_class']:
        for i, motif in enumerate(results['motifs_by_class'][9][:5]):
            print(f"  {i+1}. {motif['Subclass']}: {motif['Start']}-{motif['End']} "
                  f"(score: {motif['Score']}, length: {motif['Length']}bp)")
    
    print(f"\nTesting full 11-class system (A-philic only implemented):")
    full_results = analyze_sequence_nbd_finder(test_sequence, "full_test", parallel=True)
    print(f"Classes with motifs: {[cls for cls, motifs in full_results['motifs_by_class'].items() if motifs]}")
    print(f"Total motifs across all classes: {full_results['summary_stats']['total_motifs']}")