#!/usr/bin/env python3
"""
Genome Analysis Script: Low Perplexity Region Detection with Non-B DNA Motif Finder

This script implements:
1. Kadane's subarray algorithm to find low perplexity regions (window size 100, tunable)
2. Non-B DNA motif detection in identified low perplexity regions
3. Exploratory Data Analysis (EDA) report generation

Author: CisPerplexity Project
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import existing modules
from nonb_motif_detector import detect_all_nonb_motifs


class PerplexityCalculator:
    """Calculate dinucleotide perplexity for DNA sequences"""
    
    @staticmethod
    def calculate_perplexity(sequence: str, window_size: int = 10) -> np.ndarray:
        """
        Calculate dinucleotide perplexity for a DNA sequence using sliding windows
        
        Args:
            sequence: DNA sequence string
            window_size: Size of sliding window for perplexity calculation
            
        Returns:
            Array of perplexity values for each window position
        """
        seq_len = len(sequence)
        if seq_len < window_size:
            return np.array([])
        
        num_windows = seq_len - window_size + 1
        perplexities = np.zeros(num_windows)
        
        # Generate all dinucleotides in the sequence
        dinucleotides = [sequence[i:i + 2] for i in range(seq_len - 1)]
        
        for i in range(num_windows):
            # Get dinucleotides in current window
            window_dinucleotides = dinucleotides[i:i + window_size - 1]
            
            # Count dinucleotide frequencies
            dinucleotide_counts = Counter(window_dinucleotides)
            total_dinucleotides = sum(dinucleotide_counts.values())
            
            if total_dinucleotides == 0:
                perplexities[i] = 0
                continue
            
            # Calculate probabilities
            probabilities = np.array(list(dinucleotide_counts.values())) / total_dinucleotides
            
            # Calculate entropy: H = -Σ(p × log₂(p))
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Calculate perplexity: 2^H
            perplexities[i] = 2 ** entropy
        
        return perplexities


class KadaneRegionFinder:
    """
    Implementation of Kadane's Algorithm for finding low perplexity regions
    Uses modified Kadane's algorithm to find minimum average subarray
    """
    
    @staticmethod
    def find_low_perplexity_regions(
        perplexities: np.ndarray, 
        analysis_window_size: int = 100,
        num_regions: int = 10,
        percentile_threshold: float = 25.0
    ) -> List[Dict]:
        """
        Find low perplexity regions using Kadane's algorithm
        
        Args:
            perplexities: Array of perplexity values
            analysis_window_size: Window size for region detection (default: 100)
            num_regions: Maximum number of regions to find
            percentile_threshold: Percentile threshold for low perplexity (default: 25.0)
            
        Returns:
            List of dictionaries containing region information
        """
        if len(perplexities) < analysis_window_size:
            return []
        
        # Calculate threshold
        threshold = np.percentile(perplexities, percentile_threshold)
        
        regions = []
        excluded_indices = set()
        
        # Convert to float array
        work_arr = perplexities.astype(np.float64)
        max_val = np.max(work_arr) * 10
        
        for region_idx in range(num_regions):
            # Create modified array excluding already found regions
            modified_arr = work_arr.copy()
            for idx in excluded_indices:
                if idx < len(modified_arr):
                    modified_arr[idx] = max_val
            
            # Find region with minimum average perplexity using sliding window
            best_start = 0
            best_end = analysis_window_size - 1
            min_avg = float('inf')
            
            for start in range(len(modified_arr) - analysis_window_size + 1):
                end = start + analysis_window_size - 1
                window_values = modified_arr[start:end + 1]
                avg = np.mean(window_values)
                
                if avg < min_avg:
                    min_avg = avg
                    best_start = start
                    best_end = end
            
            # Stop if no more valid regions (all values too high)
            if min_avg >= max_val or min_avg > threshold:
                break
            
            # Check for overlap with existing regions
            has_overlap = False
            for existing_region in regions:
                overlap_start = max(best_start, existing_region['start'])
                overlap_end = min(best_end, existing_region['end'])
                
                if overlap_start <= overlap_end:
                    overlap_length = overlap_end - overlap_start + 1
                    if overlap_length / analysis_window_size > 0.5:
                        has_overlap = True
                        break
            
            if not has_overlap:
                region_info = {
                    'region_id': len(regions) + 1,
                    'start': best_start,
                    'end': best_end,
                    'length': best_end - best_start + 1,
                    'mean_perplexity': min_avg,
                    'min_perplexity': np.min(work_arr[best_start:best_end + 1]),
                    'max_perplexity': np.max(work_arr[best_start:best_end + 1]),
                    'std_perplexity': np.std(work_arr[best_start:best_end + 1])
                }
                regions.append(region_info)
                
                # Mark indices as excluded
                for i in range(best_start, best_end + 1):
                    excluded_indices.add(i)
            
            # Stop if too much of the array is excluded
            if len(excluded_indices) >= len(perplexities) * 0.8:
                break
        
        return regions


class GenomeAnalyzer:
    """Main analyzer for genome sequences"""
    
    def __init__(
        self, 
        perplexity_window: int = 10,
        analysis_window: int = 100,
        num_regions: int = 10,
        percentile_threshold: float = 25.0
    ):
        """
        Initialize genome analyzer
        
        Args:
            perplexity_window: Window size for perplexity calculation (default: 10)
            analysis_window: Window size for Kadane's region detection (default: 100)
            num_regions: Number of low perplexity regions to find (default: 10)
            percentile_threshold: Percentile threshold for low perplexity (default: 25.0)
        """
        self.perplexity_window = perplexity_window
        self.analysis_window = analysis_window
        self.num_regions = num_regions
        self.percentile_threshold = percentile_threshold
        
        self.perplexity_calc = PerplexityCalculator()
        self.region_finder = KadaneRegionFinder()
    
    def parse_fasta(self, fasta_file: str) -> Tuple[str, str]:
        """
        Parse FASTA file
        
        Args:
            fasta_file: Path to FASTA file
            
        Returns:
            Tuple of (header, sequence)
        """
        header = ""
        sequence = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    header = line[1:]
                else:
                    sequence.append(line.upper())
        
        return header, ''.join(sequence)
    
    def analyze_genome(self, genome_file: str) -> Dict:
        """
        Analyze genome file for low perplexity regions and non-B DNA motifs
        
        Args:
            genome_file: Path to genome file (.fna or .fasta)
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing genome: {genome_file}")
        print(f"{'='*80}\n")
        
        # Parse genome
        header, sequence = self.parse_fasta(genome_file)
        seq_length = len(sequence)
        
        print(f"Genome: {header}")
        print(f"Sequence length: {seq_length:,} bp\n")
        
        # Calculate perplexities
        print(f"Calculating perplexities (window={self.perplexity_window})...")
        perplexities = self.perplexity_calc.calculate_perplexity(sequence, self.perplexity_window)
        print(f"Calculated {len(perplexities):,} perplexity values\n")
        
        # Find low perplexity regions using Kadane's algorithm
        print(f"Finding low perplexity regions using Kadane's algorithm (window={self.analysis_window})...")
        regions = self.region_finder.find_low_perplexity_regions(
            perplexities,
            analysis_window_size=self.analysis_window,
            num_regions=self.num_regions,
            percentile_threshold=self.percentile_threshold
        )
        print(f"Found {len(regions)} low perplexity regions\n")
        
        # Detect non-B DNA motifs in each region
        print("Detecting non-B DNA motifs in low perplexity regions...")
        for region in regions:
            # Extract sequence for this region
            region_seq = sequence[region['start']:region['end'] + 1]
            
            # Detect non-B DNA motifs
            motifs = detect_all_nonb_motifs(region_seq)
            
            # Add motif information to region
            region['sequence_length'] = len(region_seq)
            region['gc_content'] = self._calculate_gc_content(region_seq)
            region['nonb_motifs'] = motifs
            region['total_motifs'] = sum(len(motifs.get(motif_class, [])) for motif_class in motifs)
            
            print(f"  Region {region['region_id']}: {region['total_motifs']} non-B DNA motifs detected")
        
        print("\nAnalysis complete!\n")
        
        # Prepare results
        results = {
            'genome_file': genome_file,
            'genome_header': header,
            'sequence_length': seq_length,
            'perplexity_window': self.perplexity_window,
            'analysis_window': self.analysis_window,
            'num_regions_found': len(regions),
            'percentile_threshold': self.percentile_threshold,
            'perplexity_stats': {
                'mean': float(np.mean(perplexities)),
                'median': float(np.median(perplexities)),
                'std': float(np.std(perplexities)),
                'min': float(np.min(perplexities)),
                'max': float(np.max(perplexities))
            },
            'regions': regions,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content percentage"""
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence) * 100) if len(sequence) > 0 else 0.0
    
    def generate_eda_report(self, results: Dict, output_dir: str = "eda_reports") -> str:
        """
        Generate Exploratory Data Analysis (EDA) report
        
        Args:
            results: Analysis results dictionary
            output_dir: Directory to save report files
            
        Returns:
            Path to generated report
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        genome_name = Path(results['genome_file']).stem
        
        print(f"\n{'='*80}")
        print(f"Generating EDA Report")
        print(f"{'='*80}\n")
        
        # 1. Save detailed JSON results
        json_file = output_path / f"{genome_name}_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved detailed results: {json_file}")
        
        # 2. Generate summary statistics
        summary_file = output_path / f"{genome_name}_summary_{timestamp}.txt"
        self._generate_summary_report(results, summary_file)
        print(f"✓ Saved summary report: {summary_file}")
        
        # 3. Generate CSV with region details
        csv_file = output_path / f"{genome_name}_regions_{timestamp}.csv"
        self._generate_regions_csv(results, csv_file)
        print(f"✓ Saved regions CSV: {csv_file}")
        
        # 4. Generate visualizations
        viz_file = output_path / f"{genome_name}_visualizations_{timestamp}.png"
        self._generate_visualizations(results, viz_file)
        print(f"✓ Saved visualizations: {viz_file}")
        
        # 5. Generate non-B DNA motif summary
        motif_file = output_path / f"{genome_name}_motifs_{timestamp}.csv"
        self._generate_motif_summary(results, motif_file)
        print(f"✓ Saved motif summary: {motif_file}")
        
        print(f"\n{'='*80}")
        print(f"EDA Report Generation Complete!")
        print(f"Reports saved in: {output_path}")
        print(f"{'='*80}\n")
        
        return str(summary_file)
    
    def _generate_summary_report(self, results: Dict, output_file: Path):
        """Generate text summary report"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GENOME PERPLEXITY ANALYSIS - EXPLORATORY DATA ANALYSIS (EDA) REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {results['analysis_timestamp']}\n")
            f.write(f"Genome File: {results['genome_file']}\n")
            f.write(f"Genome: {results['genome_header']}\n")
            f.write(f"Sequence Length: {results['sequence_length']:,} bp\n\n")
            
            f.write("-"*80 + "\n")
            f.write("ANALYSIS PARAMETERS\n")
            f.write("-"*80 + "\n")
            f.write(f"Perplexity Window Size: {results['perplexity_window']} bp\n")
            f.write(f"Analysis Window Size (Kadane): {results['analysis_window']} bp\n")
            f.write(f"Percentile Threshold: {results['percentile_threshold']}%\n")
            f.write(f"Regions Requested: {self.num_regions}\n")
            f.write(f"Regions Found: {results['num_regions_found']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PERPLEXITY STATISTICS (GENOME-WIDE)\n")
            f.write("-"*80 + "\n")
            stats = results['perplexity_stats']
            f.write(f"Mean Perplexity: {stats['mean']:.4f}\n")
            f.write(f"Median Perplexity: {stats['median']:.4f}\n")
            f.write(f"Std Dev: {stats['std']:.4f}\n")
            f.write(f"Min Perplexity: {stats['min']:.4f}\n")
            f.write(f"Max Perplexity: {stats['max']:.4f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("LOW PERPLEXITY REGIONS (KADANE'S ALGORITHM)\n")
            f.write("-"*80 + "\n\n")
            
            for region in results['regions']:
                f.write(f"Region {region['region_id']}:\n")
                f.write(f"  Position: {region['start']:,} - {region['end']:,} bp\n")
                f.write(f"  Length: {region['length']:,} bp\n")
                f.write(f"  GC Content: {region['gc_content']:.2f}%\n")
                f.write(f"  Mean Perplexity: {region['mean_perplexity']:.4f}\n")
                f.write(f"  Min Perplexity: {region['min_perplexity']:.4f}\n")
                f.write(f"  Max Perplexity: {region['max_perplexity']:.4f}\n")
                f.write(f"  Std Dev: {region['std_perplexity']:.4f}\n")
                f.write(f"  Total Non-B DNA Motifs: {region['total_motifs']}\n")
                
                # List motif classes found
                if region['total_motifs'] > 0:
                    f.write(f"  Motif Classes Detected:\n")
                    for motif_class, motifs in region['nonb_motifs'].items():
                        if len(motifs) > 0:
                            f.write(f"    - {motif_class}: {len(motifs)} motif(s)\n")
                
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
    
    def _generate_regions_csv(self, results: Dict, output_file: Path):
        """Generate CSV with region details"""
        rows = []
        for region in results['regions']:
            row = {
                'region_id': region['region_id'],
                'start': region['start'],
                'end': region['end'],
                'length': region['length'],
                'gc_content': region['gc_content'],
                'mean_perplexity': region['mean_perplexity'],
                'min_perplexity': region['min_perplexity'],
                'max_perplexity': region['max_perplexity'],
                'std_perplexity': region['std_perplexity'],
                'total_motifs': region['total_motifs']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def _generate_visualizations(self, results: Dict, output_file: Path):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Region Perplexity Distribution
        ax1 = axes[0, 0]
        mean_perplexities = [r['mean_perplexity'] for r in results['regions']]
        region_ids = [r['region_id'] for r in results['regions']]
        
        ax1.bar(region_ids, mean_perplexities, color='steelblue', edgecolor='navy')
        ax1.axhline(y=results['perplexity_stats']['mean'], color='red', 
                    linestyle='--', label='Genome Mean')
        ax1.set_xlabel('Region ID', fontsize=12)
        ax1.set_ylabel('Mean Perplexity', fontsize=12)
        ax1.set_title('Mean Perplexity by Low Perplexity Region', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Non-B DNA Motif Distribution
        ax2 = axes[0, 1]
        total_motifs = [r['total_motifs'] for r in results['regions']]
        
        ax2.bar(region_ids, total_motifs, color='coral', edgecolor='darkred')
        ax2.set_xlabel('Region ID', fontsize=12)
        ax2.set_ylabel('Number of Non-B DNA Motifs', fontsize=12)
        ax2.set_title('Non-B DNA Motifs Detected per Region', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: GC Content Distribution
        ax3 = axes[1, 0]
        gc_contents = [r['gc_content'] for r in results['regions']]
        
        ax3.bar(region_ids, gc_contents, color='seagreen', edgecolor='darkgreen')
        ax3.set_xlabel('Region ID', fontsize=12)
        ax3.set_ylabel('GC Content (%)', fontsize=12)
        ax3.set_title('GC Content in Low Perplexity Regions', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Region Length Distribution
        ax4 = axes[1, 1]
        lengths = [r['length'] for r in results['regions']]
        
        ax4.bar(region_ids, lengths, color='mediumpurple', edgecolor='indigo')
        ax4.set_xlabel('Region ID', fontsize=12)
        ax4.set_ylabel('Region Length (bp)', fontsize=12)
        ax4.set_title('Length of Low Perplexity Regions', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Genome Analysis: {Path(results["genome_file"]).stem}', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_motif_summary(self, results: Dict, output_file: Path):
        """Generate detailed motif summary CSV"""
        rows = []
        
        for region in results['regions']:
            region_id = region['region_id']
            
            for motif_class, motifs in region['nonb_motifs'].items():
                for motif in motifs:
                    row = {
                        'region_id': region_id,
                        'region_start': region['start'],
                        'region_end': region['end'],
                        'motif_class': motif_class,
                        'motif_start': motif.get('start', 'N/A'),
                        'motif_end': motif.get('end', 'N/A'),
                        'motif_length': motif.get('length', 'N/A'),
                        'motif_subclass': motif.get('subclass', 'N/A'),
                        'motif_score': motif.get('score', 'N/A')
                    }
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
        else:
            # Create empty file with headers
            with open(output_file, 'w') as f:
                f.write("region_id,region_start,region_end,motif_class,motif_start,motif_end,motif_length,motif_subclass,motif_score\n")


def main():
    """Main entry point for genome analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genome Analysis: Find low perplexity regions and detect non-B DNA motifs"
    )
    parser.add_argument(
        'genome_file',
        type=str,
        help='Path to genome file (.fna or .fasta)'
    )
    parser.add_argument(
        '--perplexity-window',
        type=int,
        default=10,
        help='Window size for perplexity calculation (default: 10)'
    )
    parser.add_argument(
        '--analysis-window',
        type=int,
        default=100,
        help='Window size for Kadane region detection (default: 100)'
    )
    parser.add_argument(
        '--num-regions',
        type=int,
        default=10,
        help='Number of low perplexity regions to find (default: 10)'
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=25.0,
        help='Percentile threshold for low perplexity (default: 25.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eda_reports',
        help='Output directory for EDA reports (default: eda_reports)'
    )
    
    args = parser.parse_args()
    
    # Validate genome file exists
    if not Path(args.genome_file).exists():
        print(f"Error: Genome file not found: {args.genome_file}")
        return 1
    
    # Create analyzer
    analyzer = GenomeAnalyzer(
        perplexity_window=args.perplexity_window,
        analysis_window=args.analysis_window,
        num_regions=args.num_regions,
        percentile_threshold=args.percentile
    )
    
    # Run analysis
    results = analyzer.analyze_genome(args.genome_file)
    
    # Generate EDA report
    analyzer.generate_eda_report(results, output_dir=args.output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())
