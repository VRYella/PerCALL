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

## Examples
Example FASTA files are available in `examples/`. A synchronized example output table is provided at `examples/sample_output.csv`.

## Interpretation
REGPLEX ranks algorithmic candidates by local sequence-complexity depression. Scores are descriptive and should be interpreted as prioritization metrics, not biological proof.

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
