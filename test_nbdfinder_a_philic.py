"""
Comprehensive Tests for NBDFinder A-philic DNA Detection System

This test suite validates the A-philic DNA motif detection system, including:
- Scanner functionality and tetranucleotide scoring
- Configuration system validation
- Orchestrator integration testing
- UI compatibility testing
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from motifs.a_philic_dna import (
    find_a_philic_dna, 
    APhilicDNAScanner, 
    validate_a_philic_detection,
    TEST_SEQUENCE
)
from classification_config import NBDFinderConfig, get_motif_limits
from all_motifs_refactored import analyze_sequence_nbd_finder, NBDFinderOrchestrator


class TestAPhilicDNAScanner:
    """Test cases for A-philic DNA scanner functionality"""
    
    def test_scanner_initialization(self):
        """Test A-philic DNA scanner initialization"""
        scanner = APhilicDNAScanner()
        
        # Check tetranucleotide scores are loaded
        assert len(scanner.tetranucleotide_scores) == 18
        assert 'AAAA' in scanner.tetranucleotide_scores
        assert scanner.tetranucleotide_scores['AAAA'] == 3.2
        
        # Check thresholds
        assert scanner.high_confidence_threshold == 2.0
        assert scanner.moderate_threshold == 1.5
        
        # Check window parameters
        assert scanner.min_window_size == 10
        assert scanner.max_window_size == 20
    
    def test_tetranucleotide_scoring(self):
        """Test tetranucleotide log2 odds scoring"""
        scanner = APhilicDNAScanner()
        
        # Test high-scoring A-rich sequence
        a_rich_seq = "AAAAAAAAATTTTTTTTT"
        score = scanner._calculate_tetranucleotide_score(a_rich_seq)
        assert score > 2.0, f"A-rich sequence should score high, got {score}"
        
        # Test low-scoring random sequence
        random_seq = "GCTAGCTAGCTAGCTAG"
        score = scanner._calculate_tetranucleotide_score(random_seq)
        assert score < 1.0, f"Random sequence should score low, got {score}"
        
        # Test empty sequence
        score = scanner._calculate_tetranucleotide_score("")
        assert score == 0.0
        
        # Test short sequence (< 4 bp)
        score = scanner._calculate_tetranucleotide_score("ATG")
        assert score == 0.0
    
    def test_window_scanning(self):
        """Test sliding window scanning functionality"""
        scanner = APhilicDNAScanner()
        
        # Test with A-philic sequence
        test_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTT"
        motifs = scanner._scan_windows(test_seq, 15)
        
        assert len(motifs) > 0, "Should find motifs in A-rich sequence"
        
        # Check motif structure
        for motif in motifs:
            assert 'start' in motif
            assert 'end' in motif
            assert 'score' in motif
            assert 'sequence' in motif
            assert motif['score'] >= scanner.moderate_threshold
    
    def test_motif_merging(self):
        """Test overlapping motif merging"""
        scanner = APhilicDNAScanner()
        
        # Create overlapping motifs
        motifs = [
            {'start': 0, 'end': 15, 'score': 2.5, 'sequence': 'AAAAAAAAAAAAAAA'},
            {'start': 10, 'end': 25, 'score': 2.3, 'sequence': 'AAAAAATTTTTTTT'},
            {'start': 30, 'end': 45, 'score': 2.1, 'sequence': 'TTTTTTTTTTTTTT'}
        ]
        
        merged = scanner._merge_overlapping_motifs(motifs)
        
        # Should merge first two overlapping motifs
        assert len(merged) == 2, f"Expected 2 merged motifs, got {len(merged)}"
        assert merged[0]['start'] == 0
        assert merged[0]['end'] == 25
        assert merged[0]['score'] == 2.5  # Max score
    
    def test_motif_classification(self):
        """Test motif classification into high/moderate confidence"""
        scanner = APhilicDNAScanner()
        
        motifs = [
            {'start': 0, 'end': 15, 'score': 2.5},  # High confidence
            {'start': 20, 'end': 35, 'score': 1.8}  # Moderate confidence
        ]
        
        classified = scanner._classify_motifs(motifs)
        
        assert len(classified) == 2
        assert classified[0]['Subclass'] == 'High Confidence A-philic'
        assert classified[0]['confidence'] == 'high'
        assert classified[0]['Class'] == 9
        
        assert classified[1]['Subclass'] == 'Moderate A-philic'
        assert classified[1]['confidence'] == 'moderate'
        assert classified[1]['Class'] == 9
    
    def test_find_a_philic_dna_function(self):
        """Test main A-philic DNA finding function"""
        # Test with A-rich sequence
        test_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAAA"
        motifs = find_a_philic_dna(test_seq, "test_seq")
        
        assert len(motifs) > 0, "Should find A-philic motifs"
        
        # Check motif structure
        for motif in motifs:
            assert 'Class' in motif
            assert motif['Class'] == 9
            assert 'Subclass' in motif
            assert 'Start' in motif
            assert 'End' in motif
            assert 'Score' in motif
            assert 'sequence_id' in motif
            assert motif['sequence_id'] == "test_seq"
        
        # Test with empty sequence
        empty_motifs = find_a_philic_dna("", "empty")
        assert len(empty_motifs) == 0
        
        # Test with short sequence
        short_motifs = find_a_philic_dna("ATGC", "short")
        assert len(short_motifs) == 0
    
    def test_validation_function(self):
        """Test A-philic DNA validation function"""
        stats = validate_a_philic_detection(TEST_SEQUENCE)
        
        assert 'total_motifs' in stats
        assert 'high_confidence_count' in stats
        assert 'moderate_confidence_count' in stats
        assert 'score_range' in stats
        assert 'avg_length' in stats
        
        # Should find some motifs in test sequence
        assert stats['total_motifs'] > 0


class TestNBDFinderConfiguration:
    """Test cases for NBDFinder configuration system"""
    
    def test_config_initialization(self):
        """Test configuration system initialization"""
        config = NBDFinderConfig()
        
        # Check A-philic constraints
        assert config.a_philic_constraints['S_min'] == 10
        assert config.a_philic_constraints['S_max'] == 50
        assert config.a_philic_constraints['min_score'] == 10.0
        assert config.a_philic_constraints['scoring_method'] == 'tetranucleotide_log2_odds'
        
        # Check all 11 classes are defined
        assert len(config.motif_classes) == 11
        assert 9 in config.motif_classes  # A-philic DNA
        
        # Check A-philic specific class
        a_philic_config = config.motif_classes[9]
        assert a_philic_config['name'] == 'A_philic_DNA'
        assert a_philic_config['S_min'] == 10
        assert a_philic_config['S_max'] == 50
        assert a_philic_config['scoring_method'] == 'tetranucleotide_log2_odds'
    
    def test_get_motif_limits(self):
        """Test motif limits retrieval"""
        config = NBDFinderConfig()
        
        # Test A-philic DNA (Class 9)
        limits = config.get_motif_limits(9)
        assert limits == (10, 50)
        
        # Test global function
        global_limits = get_motif_limits(9)
        assert global_limits == (10, 50)
        
        # Test invalid class
        with pytest.raises(ValueError):
            config.get_motif_limits(12)
    
    def test_config_methods(self):
        """Test configuration getter methods"""
        config = NBDFinderConfig()
        
        # Test get_motif_config
        a_philic_config = config.get_motif_config(9)
        assert a_philic_config['name'] == 'A_philic_DNA'
        
        # Test get_scoring_method
        scoring = config.get_scoring_method(9)
        assert scoring == 'tetranucleotide_log2_odds'
        
        # Test get_min_score
        min_score = config.get_min_score(9)
        assert min_score == 10.0
        
        # Test get_class_name
        name = config.get_class_name(9)
        assert name == 'A_philic_DNA'
        
        # Test get_all_class_names
        all_names = config.get_all_class_names()
        assert len(all_names) == 11
        assert all_names[9] == 'A_philic_DNA'
    
    def test_a_philic_parameters(self):
        """Test A-philic specific parameters"""
        config = NBDFinderConfig()
        
        params = config.get_a_philic_parameters()
        
        assert params['S_min'] == 10
        assert params['S_max'] == 50
        assert params['min_score'] == 10.0
        assert params['scoring_method'] == 'tetranucleotide_log2_odds'
        assert params['class_number'] == 9
        assert params['tetranucleotide_threshold'] == 2.0
        assert len(params['window_sizes']) == 11  # 10-20 inclusive
        assert len(params['classification_levels']) == 2
        
        # Check references
        assert 'references' in params
        refs = params['references']
        assert 'Vinogradov (2003)' in refs['tetranucleotide_methodology']
        assert 'Bolshoy et al. (1991)' in refs['a_tract_properties']
        assert 'Rohs et al. (2009)' in refs['protein_dna_interactions']
    
    def test_motif_validation(self):
        """Test motif result validation"""
        config = NBDFinderConfig()
        
        # Valid A-philic motif
        valid_motif = {
            'Class': 9,
            'Length': 20,
            'Score': 15.0,
            'Start': 0,
            'End': 20
        }
        assert config.validate_motif_result(valid_motif) == True
        
        # Invalid class
        invalid_class = {'Class': 12, 'Length': 20, 'Score': 15.0}
        assert config.validate_motif_result(invalid_class) == False
        
        # Invalid length (too short)
        invalid_length = {'Class': 9, 'Length': 5, 'Score': 15.0}
        assert config.validate_motif_result(invalid_length) == False
        
        # Invalid score (too low for A-philic DNA tetranucleotide scoring)
        invalid_score = {'Class': 9, 'Length': 20, 'Score': 1.0}  # Below 1.5 threshold
        assert config.validate_motif_result(invalid_score) == False


class TestNBDFinderOrchestrator:
    """Test cases for NBDFinder orchestrator integration"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = NBDFinderOrchestrator()
        
        assert orchestrator.config is not None
        assert len(orchestrator.detector_functions) == 11
        assert 9 in orchestrator.detector_functions  # A-philic DNA
        
        # Check A-philic quality parameters
        assert 'min_score' in orchestrator.a_philic_quality_params
        assert orchestrator.a_philic_quality_params['min_score'] == 10.0
    
    def test_a_philic_detection_integration(self):
        """Test A-philic DNA detection through orchestrator"""
        orchestrator = NBDFinderOrchestrator()
        
        # Test A-philic detection
        test_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAAA"
        motifs = orchestrator._detect_a_philic_dna(test_seq, "test")
        
        # Should apply quality filtering
        for motif in motifs:
            assert 'detector' in motif
            assert motif['detector'] == 'NBDFinder_A_philic'
            assert 'version' in motif
            assert 'detection_time' in motif
    
    def test_quality_filtering(self):
        """Test A-philic quality filtering"""
        orchestrator = NBDFinderOrchestrator()
        
        # High quality motif
        high_quality = {
            'Score': 15.0,
            'Length': 25,
            'confidence': 'high',
            'Start': 0,
            'End': 25
        }
        assert orchestrator._passes_a_philic_quality_filter(high_quality) == True
        
        # Low quality motif (low score)
        low_quality = {
            'Score': 1.0,
            'Length': 25,
            'confidence': 'high',
            'Start': 0,
            'End': 25
        }
        assert orchestrator._passes_a_philic_quality_filter(low_quality) == False
    
    def test_full_analysis(self):
        """Test complete NBDFinder analysis"""
        # Test A-philic only analysis
        test_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAAA"
        results = analyze_sequence_nbd_finder(
            test_seq, 
            "test_analysis",
            parallel=False,
            selected_classes=[9]
        )
        
        # Check result structure
        assert 'sequence_id' in results
        assert results['sequence_id'] == "test_analysis"
        assert 'sequence_length' in results
        assert 'classes_analyzed' in results
        assert 9 in results['classes_analyzed']
        assert 'motifs_by_class' in results
        assert 'summary_stats' in results
        
        # Check A-philic specific results
        if 9 in results['motifs_by_class']:
            a_philic_motifs = results['motifs_by_class'][9]
            for motif in a_philic_motifs:
                assert motif['Class'] == 9
                assert 'Subclass' in motif
    
    def test_parallel_vs_sequential(self):
        """Test parallel vs sequential processing"""
        test_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAAA"
        
        # Sequential analysis
        seq_results = analyze_sequence_nbd_finder(
            test_seq, "sequential", parallel=False, selected_classes=[9]
        )
        
        # Parallel analysis
        par_results = analyze_sequence_nbd_finder(
            test_seq, "parallel", parallel=True, selected_classes=[9]
        )
        
        # Results should be the same
        assert seq_results['summary_stats']['total_motifs'] == par_results['summary_stats']['total_motifs']


class TestIntegration:
    """Integration tests for the complete A-philic DNA system"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end A-philic DNA detection workflow"""
        # 1. Use configuration system
        config = NBDFinderConfig()
        a_philic_params = config.get_a_philic_parameters()
        
        # 2. Run A-philic detection
        test_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAAA"
        motifs = find_a_philic_dna(test_seq, "integration_test")
        
        # 3. Validate against configuration
        for motif in motifs:
            assert config.validate_motif_result(motif)
        
        # 4. Run through orchestrator
        full_results = analyze_sequence_nbd_finder(
            test_seq, "full_integration", selected_classes=[9]
        )
        
        # 5. Verify consistency
        orchestrator_motifs = full_results['motifs_by_class'].get(9, [])
        
        # Should have similar number of motifs (quality filtering may reduce)
        assert len(orchestrator_motifs) <= len(motifs)
    
    def test_scientific_validation(self):
        """Test against expected scientific results"""
        # Test with known A-tract sequences
        a_tract_seq = "AAAAAATTTTTTAAAAAATTTTTTAAAAAATTTTTT"
        motifs = find_a_philic_dna(a_tract_seq, "a_tract_test")
        
        assert len(motifs) > 0, "Should detect A-tracts"
        
        # Check high confidence classification
        high_conf_motifs = [m for m in motifs if m.get('confidence') == 'high']
        assert len(high_conf_motifs) > 0, "Should have high confidence A-philic motifs"
        
        # Check tetranucleotide scores
        for motif in motifs:
            assert motif['Score'] >= 1.5, "A-tract should have good tetranucleotide score"
    
    def test_literature_references(self):
        """Test that scientific references are properly included"""
        config = NBDFinderConfig()
        refs = config.a_philic_references
        
        # Check required references are present
        assert 'tetranucleotide_methodology' in refs
        assert 'a_tract_properties' in refs
        assert 'protein_dna_interactions' in refs
        
        # Check reference content
        assert 'Vinogradov (2003)' in refs['tetranucleotide_methodology']
        assert 'Bolshoy et al. (1991)' in refs['a_tract_properties']
        assert 'Rohs et al. (2009)' in refs['protein_dna_interactions']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])