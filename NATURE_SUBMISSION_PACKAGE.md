# REGPLEX v13 Nature Submission Package

## Abstract
REGPLEX is a training-free method for identifying **Low Perplexity Regions (LPRs)** in DNA sequences using information-theoretic dinucleotide complexity. The workflow computes dinucleotide perplexity (17 nt), applies Savitzky–Golay smoothing, derives a Perplexity Depression Score (PDS) by bilateral local-context contrast, performs bounded Kadane optimization, expands and merges candidate regions, and ranks final intervals with a deterministic region score. The method is species-independent and fully explainable at each computational step.

## Introduction
Sequence complexity varies across local genomic contexts. REGPLEX formalizes this variation as a regional depression in dinucleotide perplexity relative to neighboring windows. The framework is designed for algorithmic detection of candidate intervals without model training, feature learning, or species-specific calibration.

## Methods
### Study design
All sequences are analyzed independently after FASTA sanitization. Input is converted to uppercase DNA, uracil is mapped to thymine, and only `A/C/G/T/N` symbols are retained.

### Algorithm workflow
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

### Dinucleotide complexity signal
REGPLEX computes dinucleotide perplexity from 17-nt windows (16 transitions). For each window:

\[
H = -\sum_i p_i\log_2 p_i,\qquad P = 2^H
\]

Windows containing ambiguous nucleotides are marked missing.

### Smoothing
A single Savitzky–Golay pass (default window 21, order 3) is applied to improve local continuity while preserving profile shape.

### Perplexity Depression Score (PDS)
For each candidate center, means are computed from upstream flank, candidate region, and downstream flank with fixed flank/spacer geometry:

\[
\mathrm{PDS} = \frac{\overline{U}+\overline{D}}{2} - \overline{R}
\]

Positions are retained only when both flanks exceed candidate mean, enforcing bilateral local-context support.

### Region detection and ranking
Positive-PDS runs are segmented with bounded Kadane optimization (default 100–1000 bp), then expanded under PDS support and merged across short gaps (default ≤100 bp). Region ranking uses:

\[
\mathrm{RegionScore}=\mathrm{PDSMean}\times\mathrm{Persistence}\times\log(\mathrm{Length})\times\frac{1}{\mathrm{Variance}+\varepsilon}
\]

`Rank=1` denotes highest score.

### Optional motif annotation
Motif patterns (regex or IUPAC) are compiled and matched within detected region sequences only. Output includes total motif count and per-pattern summary.

## Results Reporting
REGPLEX reports algorithmic candidate intervals with coordinates, perplexity statistics, local context means, PDS statistics, prominence, persistence, variance, GC fraction, region score, rank, and optional motif summaries. These outputs prioritize regions for downstream study.

## Discussion
REGPLEX provides an explainable and training-free approach for sequence-based region detection using local information-theoretic context. The method is intended for computational prioritization and does not by itself establish biochemical activity or causal regulatory function.

## Figure Legend Guidance
Workflow figures should depict the exact implemented pipeline: dinucleotide perplexity → SG smoothing → PDS → bounded Kadane → expansion → merging → ranking → optional motif annotation.

## Supplementary Notes
- Deterministic execution for fixed inputs and parameters.
- No training stage or fitted model during inference.
- Biological validation remains external to this algorithmic workflow.

## Code Availability
Repository: <https://github.com/VRYella/PerCALL>

Primary modules:
- `regplex_core.py` (analysis pipeline)
- `motif_engine.py` (motif parsing/annotation)
- `visualization.py` (figures)
- `app.py` (Streamlit interface)
- `styles.css` (UI styling)

## Data Availability
Example FASTA files are distributed under `examples/`. Reproducible outputs can be generated with:

```bash
streamlit run app.py
python regplex_core.py examples/ecoli.fasta --out regplex_regions.csv
```

## References
1. Shannon CE. A mathematical theory of communication. *Bell System Technical Journal*. 1948.
2. Bentley JL. Programming Pearls: algorithm design techniques. *Communications of the ACM*. 1984.
3. Harris CR et al. Array programming with NumPy. *Nature*. 2020.
4. Virtanen P et al. SciPy 1.0. *Nature Methods*. 2020.
