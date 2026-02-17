#!/usr/bin/env python3
"""
Batch Genome Analysis Script

Processes multiple genome files and generates comparative EDA reports.
This demonstrates the genome_analyzer.py functionality across different organisms.
"""

import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Genome files to analyze
GENOMES = [
    {
        "file": "Candidatus Carsonella ruddii.fna",
        "name": "C. Carsonella ruddii",
        "num_regions": 5,
        "window": 100
    },
    {
        "file": "Buchnera aphidicola.fna", 
        "name": "B. aphidicola",
        "num_regions": 5,
        "window": 100
    },
    {
        "file": "hpylori.fna",
        "name": "H. pylori",
        "num_regions": 10,
        "window": 100
    },
    {
        "file": "Streptococcus pneumoniae.fna",
        "name": "S. pneumoniae",
        "num_regions": 10,
        "window": 100
    }
]


def run_analysis(genome_info):
    """Run genome_analyzer.py for a single genome"""
    genome_file = genome_info["file"]
    num_regions = genome_info["num_regions"]
    window = genome_info["window"]
    
    print(f"\n{'='*80}")
    print(f"Processing: {genome_info['name']}")
    print(f"File: {genome_file}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python", "genome_analyzer.py",
        genome_file,
        "--num-regions", str(num_regions),
        "--analysis-window", str(window)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Error processing {genome_file}:")
        print(e.stderr)
        return False, str(e)
    except FileNotFoundError:
        print(f"Error: Genome file not found: {genome_file}")
        return False, "File not found"


def generate_comparative_report(output_dir="eda_reports"):
    """Generate a comparative report across all analyzed genomes"""
    print(f"\n{'='*80}")
    print("Generating Comparative Report")
    print(f"{'='*80}\n")
    
    reports_path = Path(output_dir)
    
    # Find all JSON files
    json_files = sorted(reports_path.glob("*_analysis_*.json"))
    
    if not json_files:
        print("No analysis files found!")
        return
    
    # Collect data from all analyses
    comparative_data = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Extract summary information
            summary = {
                "genome": Path(data["genome_file"]).stem,
                "sequence_length": data["sequence_length"],
                "num_regions": data["num_regions_found"],
                "mean_perplexity": data["perplexity_stats"]["mean"],
                "median_perplexity": data["perplexity_stats"]["median"],
                "std_perplexity": data["perplexity_stats"]["std"],
                "total_motifs": sum(r["total_motifs"] for r in data["regions"]),
                "avg_motifs_per_region": sum(r["total_motifs"] for r in data["regions"]) / len(data["regions"]) if data["regions"] else 0,
                "avg_gc_content": sum(r["gc_content"] for r in data["regions"]) / len(data["regions"]) if data["regions"] else 0,
                "avg_region_perplexity": sum(r["mean_perplexity"] for r in data["regions"]) / len(data["regions"]) if data["regions"] else 0
            }
            comparative_data.append(summary)
    
    # Create comparative DataFrame
    df = pd.DataFrame(comparative_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = reports_path / f"comparative_analysis_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved comparative report: {output_file}\n")
    
    # Display summary table
    print("="*80)
    print("COMPARATIVE GENOME ANALYSIS SUMMARY")
    print("="*80)
    print()
    print(df.to_string(index=False))
    print()
    print("="*80)
    
    # Save text report
    text_file = reports_path / f"comparative_summary_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPARATIVE GENOME ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
        f.write(f"Number of Genomes Analyzed: {len(comparative_data)}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
    
    print(f"✓ Saved text summary: {text_file}\n")


def main():
    """Main batch processing function"""
    print("="*80)
    print("BATCH GENOME ANALYSIS")
    print("="*80)
    print(f"Processing {len(GENOMES)} genome files...")
    print()
    
    results = []
    for genome_info in GENOMES:
        success, error = run_analysis(genome_info)
        results.append({
            "genome": genome_info["name"],
            "file": genome_info["file"],
            "success": success,
            "error": error
        })
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print()
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"Total genomes: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed genomes:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['genome']}: {r['error']}")
        print()
    
    # Generate comparative report
    if successful > 0:
        generate_comparative_report()
    
    print("="*80)
    print("BATCH ANALYSIS COMPLETE")
    print("="*80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
