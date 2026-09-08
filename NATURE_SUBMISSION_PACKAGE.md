# REGPLEX Nature-Style Submission Package

## Abstract
REGPLEX is a training-free method for identifying **candidate regulatory regions** in DNA sequences using information-theoretic dinucleotide complexity and optional motif annotation. The workflow computes dinucleotide perplexity, applies Savitzky–Golay smoothing, derives a Perplexity Depression Score (PDS) by local-context contrast, detects bounded regions, and annotates them against a curated text-backed motif library spanning promoter-associated and non-B-DNA-like sequence patterns. The method is species-independent, explainable, and designed for transparent prioritization rather than definitive functional labeling.

## Introduction
Sequence complexity varies across local genomic contexts. REGPLEX formalizes this variation as a regional depression in dinucleotide perplexity relative to neighboring windows. The framework is designed for algorithmic detection of candidate intervals without model training, feature learning, or species-specific calibration.

## Methods
### Study design
All sequences are analyzed independently after FASTA sanitization. Inputs may originate from pasted sequence text, uploaded or disk-backed FASTA/text files, or live NCBI nucleotide accession retrieval. Input is converted to uppercase DNA, uracil is mapped to thymine, and only `A/C/G/T/N` symbols are retained.

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
Motif Annotation from `regulatory_motifs.txt`  
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

### Motif annotation
Motif patterns are loaded from a plain-text library where each line is either a raw pattern or a tab-delimited `name<TAB>pattern` entry. IUPAC-only patterns are expanded to regular expressions, while explicit regex patterns are compiled directly. Region-level output reports total motif burden and per-pattern counts.

## Results Reporting
REGPLEX reports algorithmic candidate intervals with coordinates, perplexity statistics, local context means, PDS statistics, persistence, rank, motif count, and motif summaries. These outputs prioritize regions for downstream study, targeted wet-lab follow-up, and comparative motif-aware screening.

## Software validation and testing
The repository contains an automated pytest suite covering entropy calculation, perplexity profiling, background estimation, depression scoring, interval detection, prediction bounding, and the supervised benchmarking split logic. The current release additionally verifies motif parsing, sequence-source loading from text and disk, and motif annotation on detected intervals.

## Limitations
REGPLEX is not a replacement for a curated external regulatory database, ChIP-seq evidence, transcriptomic validation, or structure-probing assays. Motif hits are sequence-level annotations only; they do not prove occupancy, promoter activity, chromatin state, or stable non-B structure formation.

## Discussion
REGPLEX provides an explainable and training-free approach for sequence-based region detection using local information-theoretic context. The method is intended for computational prioritization and does not by itself establish biochemical activity or causal regulatory function.

## Figure Legend Guidance
Workflow figures should depict the exact implemented pipeline: input acquisition → dinucleotide perplexity → SG smoothing → PDS → bounded region detection → ranking → motif annotation → export.

## Supplementary Notes
- Deterministic execution for fixed inputs and parameters.
- No training stage or fitted model during inference.
- Biological validation remains external to this algorithmic workflow.

## Code Availability
Repository: <https://github.com/VRYella/REGPLEX>

Primary modules:
- `regplex_core.py` (analysis pipeline)
- `src/motifs.py` (motif parsing/annotation)
- `visualization.py` (figures)
- `app.py` (Streamlit interface)
- `styles.css` (UI styling)

## Data Availability
Example FASTA files are distributed under `examples/`. Reproducible outputs can be generated with:

```bash
streamlit run app.py
python regplex_core.py examples/ecoli.fasta --motifs-file regulatory_motifs.txt --out regplex_regions.csv
```

## References
1. Shannon CE. A mathematical theory of communication. *Bell System Technical Journal*. 1948.
2. Bentley JL. Programming Pearls: algorithm design techniques. *Communications of the ACM*. 1984.
3. Harris CR et al. Array programming with NumPy. *Nature*. 2020.
4. Virtanen P et al. SciPy 1.0. *Nature Methods*. 2020.
