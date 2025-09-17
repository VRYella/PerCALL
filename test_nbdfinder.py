#!/usr/bin/env python3
"""
Test script for A-philic DNA detection and NBDFinder functionality
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_a_philic_detection():
    """Test A-philic DNA detection functionality"""
    print("Testing A-philic DNA detection...")
    
    from motifs.a_philic_dna import detect_a_philic_motifs
    
    # Test sequence with high-scoring tetranucleotides
    test_sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
    
    results = detect_a_philic_motifs(test_sequence, min_len=10, max_len=20)
    
    print(f"✅ Detected {len(results)} A-philic motifs")
    
    if results:
        for i, result in enumerate(results[:3]):  # Show first 3
            print(f"  {i+1}. Position {result['start']}-{result['end']}: {result['classification']} (score: {result['sum_log2']})")
    
    return len(results) > 0


def test_orchestrator():
    """Test NBDMotifOrchestrator functionality"""
    print("Testing NBDMotifOrchestrator...")
    
    from all_motifs_refactored import NBDMotifOrchestrator
    
    orchestrator = NBDMotifOrchestrator(max_workers=2)
    
    test_sequence = "AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGG"
    
    results = orchestrator.detect_all_motifs(test_sequence, classes_to_run=[9])
    filtered_results = orchestrator.apply_quality_filters(results)
    summary = orchestrator.get_summary_stats(filtered_results)
    
    print(f"✅ Orchestrator detected {summary['total_motifs']} motifs")
    
    return summary['total_motifs'] > 0


def test_classification_config():
    """Test classification configuration"""
    print("Testing classification configuration...")
    
    from classification_config import NBD_CLASSES, get_class_config, get_class_colors
    
    print(f"✅ Configuration loaded with {len(NBD_CLASSES)} classes")
    
    # Test A-philic DNA config
    a_philic_config = get_class_config(9)
    if a_philic_config:
        print(f"  A-philic DNA (Class 9): {a_philic_config['name']}")
        print(f"  Method: {a_philic_config['method']}")
        print(f"  Color: {a_philic_config['color']}")
    
    return len(NBD_CLASSES) == 11


def test_integration():
    """Test the integrated NBDStructureDetector"""
    print("Testing NBDStructureDetector integration...")
    
    # Test without streamlit imports
    try:
        # Import the required modules directly
        from classification_config import NBD_CLASSES, get_class_colors, get_class_names
        from all_motifs_refactored import NBDMotifOrchestrator
        
        class TestNBDStructureDetector:
            """Test version of NBDStructureDetector without Streamlit dependencies"""
            
            def __init__(self):
                self.orchestrator = NBDMotifOrchestrator(max_workers=4)
                self.class_configs = NBD_CLASSES
                self.class_colors = get_class_colors()
                self.class_names = get_class_names()
            
            def detect_structures(self, sequence: str, classes_to_run=None):
                if classes_to_run is None:
                    classes_to_run = [9]
                
                all_motifs = self.orchestrator.detect_all_motifs(
                    sequence, classes_to_run=classes_to_run, enable_parallel=True
                )
                
                filtered_motifs = self.orchestrator.apply_quality_filters(all_motifs)
                summary = self.orchestrator.get_summary_stats(filtered_motifs)
                
                return {
                    'sequence': sequence,
                    'sequence_length': len(sequence),
                    'classes_analyzed': classes_to_run,
                    'detected_motifs': filtered_motifs,
                    'summary_stats': summary
                }
        
        # Test the detector
        detector = TestNBDStructureDetector()
        test_seq = 'AAAAAAGGGGGGGGGCCCCTGGGGGCCCAAGGGATGCTAGCGATCGATCGTAGCGATC'
        results = detector.detect_structures(test_seq, classes_to_run=[9])
        
        print(f"✅ Integration test successful: {results['summary_stats']['total_motifs']} motifs detected")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧬 NBDFinder Test Suite")
    print("=" * 50)
    
    tests = [
        test_a_philic_detection,
        test_orchestrator, 
        test_classification_config,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}\n")
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! NBDFinder is ready.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")