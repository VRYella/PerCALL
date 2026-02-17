#!/usr/bin/env python3
"""
Genome Analysis Helper

Lists available genomes and suggests optimal parameters for analysis.
"""

import os
from pathlib import Path
import argparse

def get_genome_files():
    """Find all genome files in current directory"""
    fna_files = list(Path('.').glob('*.fna'))
    fasta_files = list(Path('.').glob('*.fasta'))
    return sorted(fna_files + fasta_files)

def get_file_size(filepath):
    """Get file size in bytes"""
    return os.path.getsize(filepath)

def suggest_parameters(size_bytes):
    """Suggest analysis parameters based on genome size"""
    size_mb = size_bytes / (1024 * 1024)
    
    if size_mb < 0.5:
        return {
            'category': 'Very Small',
            'num_regions': 5,
            'analysis_window': 100,
            'percentile': 25.0,
            'expected_time': '< 30 seconds'
        }
    elif size_mb < 1.5:
        return {
            'category': 'Small',
            'num_regions': 10,
            'analysis_window': 100,
            'percentile': 25.0,
            'expected_time': '1-2 minutes'
        }
    elif size_mb < 3.0:
        return {
            'category': 'Medium',
            'num_regions': 10,
            'analysis_window': 100,
            'percentile': 25.0,
            'expected_time': '2-5 minutes'
        }
    elif size_mb < 6.0:
        return {
            'category': 'Large',
            'num_regions': 15,
            'analysis_window': 100,
            'percentile': 30.0,
            'expected_time': '5-10 minutes'
        }
    else:
        return {
            'category': 'Very Large',
            'num_regions': 20,
            'analysis_window': 150,
            'percentile': 30.0,
            'expected_time': '10+ minutes'
        }

def format_size(size_bytes):
    """Format size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def main():
    parser = argparse.ArgumentParser(
        description="List available genomes and suggest analysis parameters"
    )
    parser.add_argument(
        '--generate-commands',
        action='store_true',
        help='Generate ready-to-run commands'
    )
    parser.add_argument(
        '--genome',
        type=str,
        help='Show details for specific genome'
    )
    
    args = parser.parse_args()
    
    genomes = get_genome_files()
    
    if not genomes:
        print("No genome files (.fna or .fasta) found in current directory")
        return 1
    
    if args.genome:
        # Show details for specific genome
        genome_path = Path(args.genome)
        if not genome_path.exists():
            print(f"Error: Genome file not found: {args.genome}")
            return 1
        
        size = get_file_size(genome_path)
        params = suggest_parameters(size)
        
        print("="*80)
        print(f"Genome Analysis Recommendations: {genome_path.name}")
        print("="*80)
        print(f"\nFile: {genome_path}")
        print(f"Size: {format_size(size)}")
        print(f"Category: {params['category']}")
        print(f"Expected Time: {params['expected_time']}")
        print(f"\nSuggested Parameters:")
        print(f"  --num-regions {params['num_regions']}")
        print(f"  --analysis-window {params['analysis_window']}")
        print(f"  --percentile {params['percentile']}")
        print(f"\nRecommended Command:")
        print(f"  python genome_analyzer.py \"{genome_path}\" \\")
        print(f"      --num-regions {params['num_regions']} \\")
        print(f"      --analysis-window {params['analysis_window']} \\")
        print(f"      --percentile {params['percentile']}")
        print()
        
    else:
        # List all genomes
        print("="*80)
        print("Available Genomes for Analysis")
        print("="*80)
        print()
        
        print(f"Found {len(genomes)} genome file(s):\n")
        
        # Create table
        headers = ["#", "Genome", "Size", "Category", "Est. Time", "Suggested Regions"]
        
        # Calculate column widths
        col_widths = [3, 40, 12, 12, 15, 17]
        
        # Print header
        header_line = "  ".join(
            h.ljust(w) for h, w in zip(headers, col_widths)
        )
        print(header_line)
        print("-" * len(header_line))
        
        # Print genomes
        for idx, genome in enumerate(genomes, 1):
            size = get_file_size(genome)
            params = suggest_parameters(size)
            
            row = [
                str(idx),
                genome.name[:38] + ".." if len(genome.name) > 40 else genome.name,
                format_size(size),
                params['category'],
                params['expected_time'],
                str(params['num_regions'])
            ]
            
            row_line = "  ".join(
                val.ljust(w) for val, w in zip(row, col_widths)
            )
            print(row_line)
        
        print()
        print("="*80)
        print("Usage Examples:")
        print("="*80)
        print()
        print("Get recommendations for a specific genome:")
        print(f"  python {Path(__file__).name} --genome {genomes[0].name}")
        print()
        print("Analyze a genome:")
        print(f"  python genome_analyzer.py \"{genomes[0].name}\"")
        print()
        print("Batch process all genomes:")
        print("  python batch_genome_analysis.py")
        print()
        
        if args.generate_commands:
            print("="*80)
            print("Generated Commands (copy and paste):")
            print("="*80)
            print()
            
            for genome in genomes:
                size = get_file_size(genome)
                params = suggest_parameters(size)
                
                print(f"# {genome.name} ({format_size(size)})")
                print(f"python genome_analyzer.py \"{genome}\" \\")
                print(f"    --num-regions {params['num_regions']} \\")
                print(f"    --analysis-window {params['analysis_window']} \\")
                print(f"    --percentile {params['percentile']}")
                print()
    
    return 0

if __name__ == "__main__":
    exit(main())
