# REGPLEX v13

## Scientific Overview
REGPLEX identifies **Low Perplexity Regions (LPRs)** in DNA by quantifying local dinucleotide sequence complexity and contrasting candidate segments against immediate genomic background. The method is deterministic, training-free, and species-independent.

## Background
Many genomic intervals contain sustained local sequence simplification relative to nearby context. REGPLEX operationalizes this as a signal-processing problem using information-theoretic complexity and bounded optimization.

## Methodology
The implementation uses one primary signal (dinucleotide perplexity), one smoothing stage (Savitzky–Golay), one contrast metric (Perplexity Depression Score; PDS), and one constrained segmentation strategy (bounded Kadane optimization), followed by region expansion, merging, and ranking.

## Algorithm Workflow
DNA  
↓  
Dinucleotide Perplexity (17 nt)  
↓  
Savitzky–Golay Smoothing  
↓  
Perplexity Depression Score (PDS)  
↓  
Bounded Kadane Optimization  
↓  
Region Expansion  
↓  
Region Merging  
↓  
Low Perplexity Region Ranking  
↓  
Optional Motif Annotation  
↓  
Downloads

## Mathematical Description
For a candidate window with upstream and downstream flanks:

\[
\mathrm{PDS} = \frac{\overline{U} + \overline{D}}{2} - \overline{R}
\]

where \(\overline{U}\), \(\overline{R}\), and \(\overline{D}\) are mean smoothed perplexity in upstream, region, and downstream windows. A position is retained only if both flanks exceed the candidate mean.

Region ranking uses:

\[
\mathrm{RegionScore} = \mathrm{PDSMean} \times \mathrm{Persistence} \times \log(\mathrm{Length}) \times \frac{1}{\mathrm{Variance}+\varepsilon}
\]

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
### Streamlit application
```bash
streamlit run app.py
```

### Command line
```bash
python regplex_core.py examples/ecoli.fasta --out regplex_regions.csv
```

## Parameters
- `--sg-window`: Savitzky–Golay window length (default 21)
- `--sg-order`: Savitzky–Golay polynomial order (default 3)
- `--flank-size`: flank window size for PDS (default 100)
- `--spacer-size`: spacer between flank and candidate (default 50)
- `--min-candidate`: minimum candidate size for adaptive PDS geometry (default 50)
- `--max-candidate`: maximum candidate size for adaptive PDS geometry (default 1000)
- `--min-region`: minimum accepted LPR length (default 100)
- `--max-region`: maximum bounded-Kadane core length (default 1000)
- `--merge-gap`: merge gap between adjacent regions (default 100)

## Outputs
Primary table columns:
- `Region_ID`
- `Start`, `End`, `Length`
- `MeanPerplexity`, `MinPerplexity`, `MaxPerplexity`
- `UpstreamMean`, `RegionMean`, `DownstreamMean`
- `PDSMean`, `PDSMax`
- `Prominence`, `Persistence`, `Variance`
- `GC%`
- `RegionScore`, `Rank`
- `MotifCount`, `Motifs`, `Sequence`

## Population Analysis

REGPLEX supports two complementary analytical workflows:

### 1. Single-Sequence Analysis
Detect and rank Low Perplexity Regions in an individual genomic sequence.  
Run as shown in Quick Start above.

### 2. Population-Level Analysis
Identify **conserved Low Perplexity Regions** across a set of aligned genomic sequences.

**Activation:** Population Analysis Mode is activated automatically when every sequence in the input FASTA has **identical length** and there are ≥2 sequences. No user intervention is required.

**Supported input types:**
- Aligned promoter sequences (e.g. from multiple species or strains)
- Multiple orthologous promoters
- Multiple strains of the same organism
- Multiple genomes aligned to a common reference
- Ortholog comparisons from comparative genomics databases

**Population statistics computed:**

| Statistic | Description |
|---|---|
| Mean Perplexity Profile | Per-position mean ± SD of dinucleotide perplexity |
| Mean PDS Profile | Per-position mean ± SD of Perplexity Depression Score |
| LPR Frequency | Fraction of sequences with an LPR at each position |
| Region Boundary Density | Histogram of LPR start and end positions |
| Mean Region Score | Per-position mean RegionScore of overlapping LPRs |
| Mean GC Profile | GC% estimated from overlapping region data |

**Consensus LPRs:**  
A consensus LPR is a contiguous run of positions where the LPR frequency ≥ the minimum support threshold (default: 50%).  
Adjacent runs within the merge gap are joined.

Output fields per consensus region:

| Column | Description |
|---|---|
| Region_ID | CLPR_XXXX identifier |
| Consensus_Start | Start position in signal coordinates |
| Consensus_End | End position |
| Length | Consensus region length (bp) |
| Support | Fraction of sequences with overlapping LPR |
| Mean_PDS | Mean Perplexity Depression Score |
| Mean_RegionScore | Mean Region Score |
| Mean_Perplexity | Mean dinucleotide perplexity |
| GC% | Estimated GC content |

**Occurrence matrix:**  
Binary Sequence × ConsensusLPR matrix (1 = LPR overlaps; 0 = absent). Downloadable as CSV/Excel.

**Motif enrichment:**  
For each consensus region, reports the fraction of sequences where a motif was found in any overlapping LPR — enabling motif conservation analysis across the population.

**Streamlit:**  
The **Population Analysis** tab appears automatically in the navigation bar when all sequences share equal length.  
Click "🧬 Population Example" on the Home or Analysis pages to load the bundled 10-sequence synthetic promoter demo.

**Notebook:**  
Section 15 of `REGPLEX_Local.ipynb` demonstrates the full population workflow with publication-quality figures.

**Python API:**
```python
from population_analysis import (
    is_population_mode,
    compute_population_stats,
    compute_consensus_lprs,
    build_occurrence_matrix,
    population_summary_table,
    compute_motif_frequencies,
)

# After running analyze_sequence on each record:
if is_population_mode(results):
    stats     = compute_population_stats(results)
    consensus = compute_consensus_lprs(stats, results=results, min_support=0.5)
    summary   = population_summary_table(consensus)
    occ       = build_occurrence_matrix(results, consensus)
```

**Performance:**  
All population statistics use NumPy vectorized operations.  
Target: 1000 sequences × 6000 bp processed in seconds on a standard workstation.



## Interpretation
REGPLEX ranks algorithmic candidates by local sequence-complexity depression. Scores are descriptive and should be interpreted as prioritization metrics, not biological proof.

## Examples
Example FASTA files are available in `examples/`:
- `ecoli.fasta` — single E. coli sequence for single-sequence analysis
- `population_example.fasta` — 10 synthetic 2000 bp promoter sequences (equal length) for population analysis demo
- `sample_output.csv` — example output table

## Motif Annotation
Motif annotation is optional and executed after LPR detection. Motifs can be provided as regex or IUPAC patterns and are counted only within detected region sequences.

## Performance
The pipeline is NumPy-vectorized for signal computation and uses linear-time bounded segmentation inside contiguous positive-PDS runs.

## Limitations
- REGPLEX detects sequence-complexity depressions, not direct regulatory function.
- Biological interpretation requires external validation.
- Ambiguous (`N`) windows are excluded from perplexity estimation.

## Citation
If you use REGPLEX, cite this repository and associated manuscript package.

## License
MIT License.

## References
1. Shannon CE. *A mathematical theory of communication*. 1948.
2. Bentley JL. *Programming Pearls: algorithm design techniques*. 1984.
3. Harris CR et al. *Array programming with NumPy*. 2020.
4. Virtanen P et al. *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. 2020.
