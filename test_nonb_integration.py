"""
Test script for Non-B DNA motif integration with CisPerplexity
"""
import sys
from streamlit_app import PromoterPredictor
from nonb_motif_detector import detect_all_nonb_motifs


def test_nonb_module():
    """Test the Non-B DNA motif detector module independently"""
    print("=" * 60)
    print("TEST 1: Non-B DNA Module (Independent)")
    print("=" * 60)
    
    # Test sequence with various Non-B DNA structures
    test_seq = "GGGGGGGGGGCCCCCCCCCTAGGGGGGGGGCCCCCCCCCTAAAAAATTTTTTAGCGCGCGCGCG"
    results = detect_all_nonb_motifs(test_seq)
    
    total_motifs = sum(len(m) for m in results.values())
    print(f"✓ Detected {total_motifs} motifs")
    
    for class_name, motifs in results.items():
        if motifs:
            print(f"  - {class_name}: {len(motifs)} motifs")
            for motif in motifs[:2]:  # Show first 2 motifs per class
                print(f"    • {motif.get('Subclass', 'N/A')} at {motif['Start']}-{motif['End']} (Score: {motif.get('Score', 'N/A')})")
    
    print()
    return True


def test_cisperplexity_integration():
    """Test the full CisPerplexity integration"""
    print("=" * 60)
    print("TEST 2: CisPerplexity Integration")
    print("=" * 60)
    
    predictor = PromoterPredictor()
    
    # Human TATA-box promoter example with Non-B structures
    test_sequence = """
    GCGCGCGCATATAAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCG
    TAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTATAGCGTAGCGTAGCGTAGCG
    GGGGGGGGGGCCCCCCCCCTAGGGGGGGGGCCCCCCCCCTGGGGGGGGGCCCC
    TAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTA
    GCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAGCGTAG
    AAAAAAAATTTTTTTAAAAAAAATTTTTTTAAAAAAAATTTTTT
    """.replace('\n', '').replace(' ', '')
    
    results = predictor.predict_promoters(test_sequence)
    
    print(f"✓ Sequence analyzed: {results['sequence_length']} bp")
    print(f"✓ Predicted promoters: {len(results['predicted_promoters'])}")
    print(f"✓ Promoter motifs: {sum(len(m) for m in results['motif_matches'].values())}")
    
    if results.get('nonb_motifs'):
        total_nonb = sum(len(m) for m in results['nonb_motifs'].values())
        print(f"✓ Non-B DNA motifs: {total_nonb}")
        
        print("\nNon-B DNA Structures Detected:")
        for class_name, motifs in results['nonb_motifs'].items():
            if motifs:
                avg_score = sum(m.get('Score', 0) for m in motifs) / len(motifs)
                print(f"  - {class_name}: {len(motifs)} motifs (avg score: {avg_score:.2f})")
    
    # Check promoter predictions
    if results['predicted_promoters']:
        print("\nTop Promoter Predictions:")
        for i, promoter in enumerate(results['predicted_promoters'][:3], 1):
            print(f"  {i}. Region {promoter['start']}-{promoter['end']}")
            print(f"     Confidence: {promoter['confidence']:.2%}")
            print(f"     Method: {promoter['method']}")
            print(f"     Motifs: {promoter['motif_count']}")
    
    print()
    return True


def test_data_structure():
    """Test that the data structure is correct"""
    print("=" * 60)
    print("TEST 3: Data Structure Validation")
    print("=" * 60)
    
    predictor = PromoterPredictor()
    test_seq = "GGGGGGGGGGCCCCCCCCCTAGGGGGGGGGCCCCCCCCCT" * 3
    results = predictor.predict_promoters(test_seq)
    
    # Check required fields
    required_fields = [
        'sequence_length', 'window_size', 'perplexity_window',
        'perplexity', 'gc_content', 'motif_matches', 'nonb_motifs',
        'predicted_promoters'
    ]
    
    for field in required_fields:
        if field not in results:
            print(f"✗ Missing field: {field}")
            return False
        print(f"✓ Field present: {field}")
    
    # Check Non-B DNA motif structure
    if results.get('nonb_motifs'):
        print("\n✓ Non-B DNA motif classes:")
        for class_name in results['nonb_motifs'].keys():
            print(f"  - {class_name}")
    
    print()
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TESTING NON-B DNA INTEGRATION WITH CISPERPLEXITY")
    print("=" * 60 + "\n")
    
    tests = [
        test_nonb_module,
        test_cisperplexity_integration,
        test_data_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED with exception:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
