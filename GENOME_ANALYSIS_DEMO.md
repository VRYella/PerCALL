# Genome Analysis Demonstration

This document demonstrates the complete workflow for analyzing genomes to find low perplexity regions and detect non-B DNA motifs.

## Quick Start Examples

### Example 1: Analyze a Single Genome

```bash
# Basic analysis with default parameters
python genome_analyzer.py ecoli.fna

# With custom parameters
python genome_analyzer.py ecoli.fna \
    --perplexity-window 15 \
    --analysis-window 150 \
    --num-regions 20 \
    --percentile 30 \
    --output-dir my_analysis
```

### Example 2: Batch Process Multiple Genomes

```bash
# Process all genomes in the batch list
python batch_genome_analysis.py
```

### Example 3: Run Tests

```bash
# Validate installation and functionality
python test_genome_analyzer.py
```

## Sample Output

### Console Output

```
================================================================================
Analyzing genome: hpylori.fna
================================================================================

Genome: NZ_CP024072.1 Helicobacter pylori strain 7.13_R1b chromosome, complete genome
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
```

### Summary Report (Text)

```
================================================================================
GENOME PERPLEXITY ANALYSIS - EXPLORATORY DATA ANALYSIS (EDA) REPORT
================================================================================

Analysis Date: 2026-02-17T05:53:09.701779
Genome File: hpylori.fna
Genome: NZ_CP024072.1 Helicobacter pylori strain 7.13_R1b chromosome
Sequence Length: 1,674,010 bp

--------------------------------------------------------------------------------
ANALYSIS PARAMETERS
--------------------------------------------------------------------------------
Perplexity Window Size: 10 bp
Analysis Window Size (Kadane): 100 bp
Percentile Threshold: 25.0%
Regions Requested: 10
Regions Found: 10

--------------------------------------------------------------------------------
PERPLEXITY STATISTICS (GENOME-WIDE)
--------------------------------------------------------------------------------
Mean Perplexity: 6.1144
Median Perplexity: 6.2403
Std Dev: 1.4216
Min Perplexity: 1.0000
Max Perplexity: 9.0000

--------------------------------------------------------------------------------
LOW PERPLEXITY REGIONS (KADANE'S ALGORITHM)
--------------------------------------------------------------------------------

Region 1:
  Position: 213,173 - 213,272 bp
  Length: 100 bp
  GC Content: 25.00%
  Mean Perplexity: 3.8583
  Min Perplexity: 1.4174
  Max Perplexity: 7.7152
  Std Dev: 1.2990
  Total Non-B DNA Motifs: 10
  Motif Classes Detected:
    - Curved_DNA: 3 motif(s)
    - Slipped DNA: 6 motif(s)
    - Triplex: 1 motif(s)

...
```

### Region Details (CSV)

```csv
region_id,start,end,length,gc_content,mean_perplexity,min_perplexity,max_perplexity,std_perplexity,total_motifs
1,213173,213272,100,25.0,3.8583,1.4174,7.7152,1.2990,10
2,630999,631098,100,25.0,3.9578,1.9813,6.6138,1.1652,4
3,1528961,1529060,100,19.0,3.9704,1.4174,7.7152,1.2058,3
...
```

### Comparative Analysis (Batch Processing)

```
================================================================================
COMPARATIVE GENOME ANALYSIS SUMMARY
================================================================================

                      genome  sequence_length  num_regions  mean_perplexity  total_motifs  avg_motifs_per_region
         Buchnera aphidicola           452078            5         5.028521            79                   15.8
Candidatus Carsonella ruddii           174014            5         4.967822            48                    9.6
    Streptococcus pneumoniae          2110968           10         6.444041            61                    6.1
                     hpylori          1674010           10         6.114398            60                    6.0
```

## Biological Interpretation

### Low Perplexity Regions

Low perplexity regions in genomes often indicate:
- **Regulatory regions**: Promoters, enhancers, and other control elements
- **Structural constraints**: Regions with conserved sequence patterns
- **Functional elements**: DNA-protein binding sites
- **Evolutionary pressure**: Conserved sequences under selection

### Non-B DNA Motifs

The detected non-B DNA structures have important biological functions:

1. **Slipped DNA** (most common)
   - Short tandem repeats (STRs) and microsatellites
   - Associated with genomic instability
   - Important for genetic diversity

2. **Curved DNA**
   - A-tract mediated bending
   - Important for DNA packaging
   - Affects chromatin structure

3. **G-Quadruplex**
   - Found in telomeres and promoters
   - Role in gene regulation
   - Potential therapeutic targets

4. **Cruciform**
   - Inverted repeats forming hairpin structures
   - Important for recombination
   - DNA damage sites

5. **Triplex**
   - Three-stranded structures
   - Gene regulation and silencing
   - Found in regulatory regions

## Analysis Workflow

```
Input: Genome File (.fna/.fasta)
  ↓
Step 1: Parse FASTA
  ↓
Step 2: Calculate Dinucleotide Perplexity
  - Sliding window (default: 10 bp)
  - Entropy calculation
  - Perplexity = 2^H
  ↓
Step 3: Apply Kadane's Algorithm
  - Find minimum average subarrays
  - Window size (default: 100 bp)
  - Multiple non-overlapping regions
  ↓
Step 4: Filter by Threshold
  - Percentile-based (default: 25%)
  - Keep only low perplexity regions
  ↓
Step 5: Detect Non-B DNA Motifs
  - Pattern matching in each region
  - 11 classes of structures
  - Binary detection (no scoring)
  ↓
Step 6: Generate EDA Reports
  - Summary statistics
  - Visualizations
  - Export to multiple formats
  ↓
Output: Comprehensive Analysis Reports
```

## Performance Benchmarks

Approximate processing times (on standard hardware):

| Genome Size | Processing Time | Regions Found | Motifs Detected |
|-------------|----------------|---------------|-----------------|
| 174 KB      | ~15 seconds    | 5             | 48              |
| 452 KB      | ~25 seconds    | 5             | 79              |
| 1.6 MB      | ~2 minutes     | 10            | 60              |
| 2.1 MB      | ~3 minutes     | 10            | 61              |
| 4.6 MB (E. coli) | ~8 minutes | 15            | ~120*           |

*Estimated based on smaller genomes

## Tips for Best Results

1. **Window Size Selection**
   - Smaller windows (50-100 bp): More precise, shorter regions
   - Larger windows (150-200 bp): Broader regions, more context
   - Default (100 bp) works well for most analyses

2. **Number of Regions**
   - Start with 10-15 regions for initial exploration
   - Increase for comprehensive genome-wide analysis
   - Consider genome size when choosing

3. **Percentile Threshold**
   - Lower (15-20%): More stringent, fewer regions
   - Higher (30-40%): More regions, less stringent
   - Default (25%) provides good balance

4. **Batch Processing**
   - Process multiple related genomes together
   - Use comparative analysis to identify patterns
   - Look for conserved low perplexity regions across species

## Citation

If you use this genome analysis tool in your research, please cite:

```
CisPerplexity: An Integrated Computational Tool for Promoter Prediction Using 
Dinucleotide Perplexity Analysis and Kadane's Algorithm
https://github.com/VRYella/CisPerplexity
```

## Support and Issues

For questions, bug reports, or feature requests:
- GitHub Issues: https://github.com/VRYella/CisPerplexity/issues
- Documentation: See GENOME_ANALYSIS_README.md
- Examples: Run test_genome_analyzer.py for working examples

---

**Last Updated**: February 2026
