#!/usr/bin/env python3
"""
Test Script for Genome Analyzer

This script tests the genome analyzer with the provided genome files
and validates the output.
"""

import subprocess
import sys
from pathlib import Path
import json

def test_genome_analyzer():
    """Test genome_analyzer.py with a small genome"""
    print("="*80)
    print("Testing Genome Analyzer")
    print("="*80)
    print()
    
    # Test with smallest genome
    genome_file = "Candidatus Carsonella ruddii.fna"
    
    if not Path(genome_file).exists():
        print(f"ERROR: Test genome file not found: {genome_file}")
        return False
    
    print(f"Testing with: {genome_file}")
    print()
    
    # Run analyzer
    cmd = [
        "python", "genome_analyzer.py",
        genome_file,
        "--num-regions", "5",
        "--analysis-window", "100"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Check if output files were created
        output_dir = Path("eda_reports")
        if not output_dir.exists():
            print("ERROR: Output directory not created")
            return False
        
        # Find the latest JSON file
        json_files = sorted(output_dir.glob("*_analysis_*.json"))
        if not json_files:
            print("ERROR: No JSON output files found")
            return False
        
        latest_json = json_files[-1]
        print(f"\nValidating output: {latest_json}")
        
        # Load and validate JSON
        with open(latest_json, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        required_keys = [
            'genome_file', 'sequence_length', 'perplexity_window',
            'analysis_window', 'num_regions_found', 'regions'
        ]
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"ERROR: Missing keys in output: {missing_keys}")
            return False
        
        print(f"✓ Output JSON structure is valid")
        print(f"✓ Found {data['num_regions_found']} regions")
        print(f"✓ Sequence length: {data['sequence_length']:,} bp")
        
        # Check regions have motif data
        if data['regions']:
            region = data['regions'][0]
            if 'nonb_motifs' in region:
                print(f"✓ Non-B DNA motifs detected in regions")
            else:
                print("WARNING: No motif data in regions")
        
        print()
        print("="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_algorithm_correctness():
    """Test Kadane's algorithm implementation"""
    print("\n" + "="*80)
    print("Testing Kadane's Algorithm Implementation")
    print("="*80)
    print()
    
    try:
        from genome_analyzer import KadaneRegionFinder
        import numpy as np
        
        # Test case 1: Simple array with clear minimum
        test_array = np.array([5.0, 3.0, 2.0, 1.0, 2.0, 3.0, 5.0])
        finder = KadaneRegionFinder()
        
        regions = finder.find_low_perplexity_regions(
            test_array,
            analysis_window_size=3,
            num_regions=1,
            percentile_threshold=50.0
        )
        
        if not regions:
            print("ERROR: No regions found in test array")
            return False
        
        print(f"✓ Found {len(regions)} region(s) in test array")
        print(f"  Region: start={regions[0]['start']}, end={regions[0]['end']}, mean={regions[0]['mean_perplexity']:.4f}")
        
        # Verify the region is in the low-value area
        if regions[0]['mean_perplexity'] < 3.0:
            print(f"✓ Region correctly identified low perplexity area")
        else:
            print(f"WARNING: Region may not be optimal")
        
        print()
        print("="*80)
        print("✓ ALGORITHM TESTS PASSED")
        print("="*80)
        return True
        
    except ImportError as e:
        print(f"ERROR: Could not import genome_analyzer: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("GENOME ANALYZER TEST SUITE")
    print("="*80)
    print()
    
    # Test 1: Algorithm correctness
    test1_passed = test_algorithm_correctness()
    
    # Test 2: Full integration test
    test2_passed = test_genome_analyzer()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Algorithm Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Integration Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("="*80)
    
    return 0 if (test1_passed and test2_passed) else 1

if __name__ == "__main__":
    exit(main())
