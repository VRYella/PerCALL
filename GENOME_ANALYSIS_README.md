# Genome Analysis: Low Perplexity Regions with Non-B DNA Motif Detection

## Overview

This module implements a comprehensive genome analysis pipeline that:

1. **Identifies low perplexity regions** in genomic sequences using Kadane's subarray algorithm
2. **Detects non-B DNA motifs** in these regions using advanced pattern matching
3. **Generates exploratory data analysis (EDA) reports** with visualizations and statistics

## Key Features

### 🧬 Kadane's Algorithm for Low Perplexity Detection
- Uses modified Kadane's maximum subarray algorithm to find optimal low perplexity regions
- Tunable window size (default: 100 bp, configurable)
- Finds multiple non-overlapping regions
- More precise than simple threshold-based methods

### 🔬 Non-B DNA Motif Detection
- Detects 11 classes of Non-B DNA structures:
  - A-philic DNA
  - Curved DNA
  - G-Quadruplex
  - i-Motif
  - Z-DNA
  - Slipped DNA
  - Cruciform
  - Triplex
  - R-loop
  - and more...
- No scoring required - just detection
- Uses pattern matching methods from the NonBDNAFinder integration

### 📊 Comprehensive EDA Reports
Generated reports include:
- **Summary Report** (TXT): Human-readable analysis summary
- **Detailed Results** (JSON): Complete structured data
- **Region Details** (CSV): Tabular data for regions
- **Motif Summary** (CSV): Detailed motif information
- **Visualizations** (PNG): Charts and graphs

## Installation

Dependencies are already included in the main `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python genome_analyzer.py <genome_file>
```

### Example: Analyze E. coli genome

```bash
python genome_analyzer.py ecoli.fna
```

### With Custom Parameters

```bash
python genome_analyzer.py ecoli.fna \
    --analysis-window 150 \
    --num-regions 20 \
    --perplexity-window 15 \
    --percentile 30 \
    --output-dir my_reports
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `genome_file` | str | Required | Path to genome file (.fna or .fasta) |
| `--perplexity-window` | int | 10 | Window size for perplexity calculation |
| `--analysis-window` | int | 100 | Window size for Kadane's region detection |
| `--num-regions` | int | 10 | Number of low perplexity regions to find |
| `--percentile` | float | 25.0 | Percentile threshold for low perplexity |
| `--output-dir` | str | `eda_reports` | Output directory for reports |

## Example Analyses

### Small Genome (Candidatus Carsonella ruddii)
```bash
python genome_analyzer.py "Candidatus Carsonella ruddii.fna" --num-regions 5
```

### Medium Genome (H. pylori)
```bash
python genome_analyzer.py hpylori.fna --num-regions 10
```

### Large Genome (E. coli)
```bash
python genome_analyzer.py ecoli.fna --num-regions 15 --analysis-window 150
```

### Very Large Genome (S. cerevisiae)
```bash
python genome_analyzer.py Scer.fna --num-regions 20 --analysis-window 200
```

## Algorithm Details

### Perplexity Calculation

Perplexity is calculated using dinucleotide frequencies:

1. **Sliding Window**: Uses configurable window (default: 10 bp)
2. **Dinucleotide Counting**: Counts all dinucleotides in window
3. **Entropy Calculation**: H = -Σ(p × log₂(p))
4. **Perplexity**: 2^H

### Kadane's Algorithm Adaptation

The classic Kadane's algorithm finds the maximum sum subarray. We adapt it to find:

1. **Minimum Average Subarray**: Regions with lowest average perplexity
2. **Fixed Window Size**: Enforces analysis window size constraint
3. **Multiple Regions**: Finds top N non-overlapping regions
4. **Threshold Filtering**: Only includes regions below percentile threshold

### Non-B DNA Motif Detection

Uses the integrated NonBDNAFinder module:
- Pattern-based detection for each motif class
- No scoring required (binary detection)
- Fast pattern matching algorithms
- Comprehensive motif library

## Output Files

After analysis, you'll find these files in the output directory:

### 1. Summary Report (TXT)
Human-readable report with:
- Genome information
- Analysis parameters
- Perplexity statistics
- Detailed region information
- Motif counts by class

### 2. Detailed Results (JSON)
Complete structured data including:
- All analysis parameters
- Full region details
- All detected motifs with metadata
- Timestamp information

### 3. Regions CSV
Tabular data for easy processing:
- Region ID, start, end, length
- GC content
- Perplexity statistics (mean, min, max, std)
- Total motif count

### 4. Motifs CSV
Detailed motif information:
- Region ID and location
- Motif class and subclass
- Motif position within region
- Motif length
- Additional metadata

### 5. Visualizations (PNG)
Four-panel figure showing:
- **Panel 1**: Mean perplexity by region
- **Panel 2**: Non-B DNA motifs detected per region
- **Panel 3**: GC content distribution
- **Panel 4**: Region length distribution

## Example Output

```
================================================================================
Analyzing genome: hpylori.fna
================================================================================

Genome: NZ_CP024072.1 Helicobacter pylori strain 7.13_R1b chromosome
Sequence length: 1,674,010 bp

Calculating perplexities (window=10)...
Calculated 1,674,001 perplexity values

Finding low perplexity regions using Kadane's algorithm (window=100)...
Found 10 low perplexity regions

Detecting non-B DNA motifs in low perplexity regions...
  Region 1: 10 non-B DNA motifs detected
  Region 2: 4 non-B DNA motifs detected
  Region 3: 3 non-B DNA motifs detected
  ...

Analysis complete!

================================================================================
Generating EDA Report
================================================================================

✓ Saved detailed results: eda_reports/hpylori_analysis_20260217_055309.json
✓ Saved summary report: eda_reports/hpylori_summary_20260217_055309.txt
✓ Saved regions CSV: eda_reports/hpylori_regions_20260217_055309.csv
✓ Saved visualizations: eda_reports/hpylori_visualizations_20260217_055309.png
✓ Saved motif summary: eda_reports/hpylori_motifs_20260217_055309.csv

================================================================================
EDA Report Generation Complete!
Reports saved in: eda_reports
================================================================================
```

## Scientific Background

### Why Low Perplexity Regions?

Low perplexity regions in genomes often correspond to:
- **Regulatory regions**: Promoters, enhancers
- **Structural variations**: Non-B DNA formations
- **Functional elements**: Binding sites, replication origins
- **Evolutionary constraints**: Conserved sequences

### Non-B DNA Structures

Non-B DNA structures play important roles in:
- Gene regulation and transcription
- DNA replication and repair
- Chromatin structure and organization
- Genomic instability and disease
- Evolutionary hotspots

## Performance

### Time Complexity
- **Perplexity Calculation**: O(n) where n = sequence length
- **Kadane's Algorithm**: O(n × w) where w = analysis window
- **Motif Detection**: Depends on motif patterns and region size

### Memory Usage
- Efficient numpy operations
- Streaming FASTA parsing
- Reasonable for genomes up to ~10 MB

### Expected Run Times
- Small genomes (<1 MB): < 1 minute
- Medium genomes (1-5 MB): 1-5 minutes
- Large genomes (5-15 MB): 5-15 minutes

## Integration with CisPerplexity

This genome analyzer is part of the CisPerplexity project and integrates seamlessly with:
- Existing non-B DNA detection modules
- PerplexityCalculator implementations
- Structural feature encoders
- Visualization utilities

## Future Enhancements

Potential improvements:
- [ ] Parallel processing for large genomes
- [ ] Hyperscan acceleration for motif detection
- [ ] Additional statistical tests
- [ ] Interactive HTML reports
- [ ] Batch processing mode for multiple genomes
- [ ] Comparative analysis across genomes

## Troubleshooting

### Large Genome Takes Too Long
- Reduce `--num-regions` to find fewer regions
- Increase `--percentile` threshold
- Consider analyzing chromosome by chromosome

### Memory Issues
- Process smaller segments of genome
- Increase system memory
- Use streaming mode (future enhancement)

### No Motifs Detected
- Verify genome file format
- Check if regions are too small
- Try adjusting `--analysis-window` size

## References

1. CisPerplexity: Promoter Prediction Using Dinucleotide Perplexity Analysis
2. NonBDNAFinder: Comprehensive Non-B DNA Structure Detection
3. Kadane's Algorithm for Maximum Subarray Sum

## Citation

If you use this genome analyzer in your research, please cite:

```
CisPerplexity: An Integrated Computational Tool for Promoter Prediction Using 
Dinucleotide Perplexity Analysis and Kadane's Algorithm. (2024)
```

## Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: See main README.md and KADANE_IMPLEMENTATION.md
- **Examples**: Genome files included in repository

---

**Built with ❤️ as part of the CisPerplexity project**
