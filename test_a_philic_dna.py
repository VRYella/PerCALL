#!/usr/bin/env python3
"""
Tests for A-philic DNA detection functionality
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from motifs.a_philic_dna import detect_a_philic_motifs, classify_window, build_contrib_array
    from classification_config import get_class_config, NBD_CLASSES
    from all_motifs_refactored import NBDMotifOrchestrator
except ImportError:
    # If imports fail, we'll skip these tests
    pytest.skip("NBDFinder modules not available", allow_module_level=True)


class TestAPhilicDNA:
    """Test cases for A-philic DNA detection"""
    
    def test_a_philic_detection_basic(self):
        """Test basic A-philic DNA detection"""
        # Sequence with high-scoring tetranucleotides
        sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
        
        results = detect_a_philic_motifs(sequence, min_len=10, max_len=20)
        
        assert isinstance(results, list)
        assert len(results) > 0  # Should detect A-philic motifs
        
        # Check result structure
        for result in results:
            assert 'start' in result
            assert 'end' in result
            assert 'classification' in result
            assert 'confidence' in result
            assert 'sum_log2' in result
            assert result['classification'] in ['A_high_confidence', 'A_moderate']
    
    def test_a_philic_detection_empty_sequence(self):
        """Test A-philic detection with empty sequence"""
        sequence = ""
        
        results = detect_a_philic_motifs(sequence)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_a_philic_detection_short_sequence(self):
        """Test A-philic detection with very short sequence"""
        sequence = "ATGC"  # Too short for meaningful analysis
        
        results = detect_a_philic_motifs(sequence, min_len=10)
        
        assert isinstance(results, list)
        assert len(results) == 0  # Too short to detect anything
    
    def test_a_philic_classification(self):
        """Test A-philic window classification"""
        # Test high confidence classification - need score >= 16.0 and strong_count >= 6
        high_score = 18.0
        n_tets = 8
        strong_count = 6
        
        result = classify_window(high_score, n_tets, strong_count)
        assert result == "A_high_confidence"
        
        # Test moderate classification - need score >= 8.0 and strong_count >= 5
        mod_score = 10.0
        n_tets = 8
        strong_count = 5
        
        result = classify_window(mod_score, n_tets, strong_count)
        assert result == "A_moderate"
        
        # Test not A-philic - low score and/or insufficient strong count
        low_score = 6.0
        n_tets = 8
        strong_count = 2
        
        result = classify_window(low_score, n_tets, strong_count)
        assert result == "not_A"
    
    def test_build_contrib_array(self):
        """Test tetranucleotide contribution array building"""
        sequence = "AGGGCCCC"
        tet_log2 = {"AGGG": 3.7004, "CCCC": 2.5361}
        
        contrib = build_contrib_array(sequence, tet_log2)
        
        assert len(contrib) == len(sequence)
        assert contrib[0] == 3.7004  # AGGG
        assert contrib[4] == 2.5361  # CCCC


class TestClassificationConfig:
    """Test cases for classification configuration"""
    
    def test_nbd_classes_count(self):
        """Test that we have all 11 NBD classes"""
        assert len(NBD_CLASSES) == 11
        assert all(i in NBD_CLASSES for i in range(1, 12))
    
    def test_a_philic_config(self):
        """Test A-philic DNA (Class 9) configuration"""
        config = get_class_config(9)
        
        assert config is not None
        assert config['name'] == 'A-philic DNA'
        assert config['method'] == 'tetranucleotide_log2_odds'
        assert config['color'] == '#E6B8F7'
        assert config['s_min'] == 10
        assert config['s_max'] == 50
        assert 'thresholds' in config
    
    def test_all_classes_have_required_fields(self):
        """Test that all classes have required configuration fields"""
        required_fields = ['name', 'description', 'color', 's_min', 's_max', 'method']
        
        for class_id, config in NBD_CLASSES.items():
            for field in required_fields:
                assert field in config, f"Class {class_id} missing {field}"


class TestNBDMotifOrchestrator:
    """Test cases for NBDMotifOrchestrator"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = NBDMotifOrchestrator(max_workers=2)
        
        assert hasattr(orchestrator, 'max_workers')
        assert hasattr(orchestrator, 'classes')
        assert orchestrator.max_workers == 2
    
    def test_a_philic_detection_through_orchestrator(self):
        """Test A-philic DNA detection through orchestrator"""
        orchestrator = NBDMotifOrchestrator(max_workers=2)
        sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
        
        results = orchestrator.detect_all_motifs(sequence, classes_to_run=[9])
        
        assert isinstance(results, dict)
        assert 9 in results  # A-philic DNA class
        assert isinstance(results[9], list)
        assert len(results[9]) > 0  # Should detect A-philic motifs
    
    def test_quality_filtering(self):
        """Test quality filtering functionality"""
        orchestrator = NBDMotifOrchestrator(max_workers=2)
        sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
        
        all_motifs = orchestrator.detect_all_motifs(sequence, classes_to_run=[9])
        filtered_motifs = orchestrator.apply_quality_filters(all_motifs)
        
        assert isinstance(filtered_motifs, dict)
        assert 9 in filtered_motifs
        # Filtered results should be <= original results
        assert len(filtered_motifs[9]) <= len(all_motifs[9])
    
    def test_summary_stats(self):
        """Test summary statistics generation"""
        orchestrator = NBDMotifOrchestrator(max_workers=2)
        sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
        
        results = orchestrator.detect_all_motifs(sequence, classes_to_run=[9])
        filtered_results = orchestrator.apply_quality_filters(results)
        summary = orchestrator.get_summary_stats(filtered_results)
        
        assert isinstance(summary, dict)
        assert 'total_motifs' in summary
        assert 'classes_detected' in summary
        assert 'class_counts' in summary
        assert summary['total_motifs'] >= 0
        assert summary['classes_detected'] >= 0


# Sequence validation tests
def test_a_philic_sequence_validation():
    """Test A-philic DNA detection with various sequence types"""
    
    # Sequence with high-scoring tetranucleotides (AGGG, CCCC)
    high_score_seq = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
    results_high = detect_a_philic_motifs(high_score_seq)
    
    # Random sequence - should have fewer high-scoring patterns
    random_seq = "ATCGATCGATCGATCGATCGATCGATCGATCG"
    results_random = detect_a_philic_motifs(random_seq)
    
    # High-scoring sequence should have more detections
    assert len(results_high) >= len(results_random)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])