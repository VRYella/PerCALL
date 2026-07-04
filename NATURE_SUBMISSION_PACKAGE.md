# REGPLEX v9 Nature Submission Package

## Methods

### Study design
REGPLEX v9 is a training-free computational framework for identifying candidate regulatory regions directly from DNA sequence. The method is designed to detect **Perplexity Valleys (PVs)**, defined as continuous sequence intervals in which local sequence uncertainty is consistently lower than flanking genomic context across multiple observation scales. No species-specific training data, parameter fitting, or external annotations are required for inference.

### Sequence preprocessing
Input sequences are parsed from FASTA text and converted to uppercase DNA. Uracil is converted to thymine, and only `A`, `C`, `G`, `T`, and `N` symbols are retained during sequence sanitization. Empty records are excluded. Each retained sequence is analyzed independently.

### Dinucleotide perplexity signal
The primary signal is a local dinucleotide perplexity profile (**P1**). For each sequence, REGPLEX computes 10-mer windows by default and derives dinucleotide frequencies from adjacent bases within each window. For a given window, Shannon entropy is calculated from the observed dinucleotide probability distribution and converted to perplexity:

\[
H = -\sum_i p_i \log_2 p_i,\qquad P1 = 2^H
\]

Windows containing ambiguous nucleotides are assigned `NaN` to avoid introducing artificial certainty.

### Multi-scale landscape construction
REGPLEX computes P1 once per sequence and reuses it across all downstream scales. The default observation scales are generated around a base scale of 100 bp, yielding 25, 50, 100, 200, and 400 bp windows. At each scale, a smoothed landscape is derived from the P1 signal using either a rolling mean (default) or rolling median. This design separates signal generation from scale-specific contextualization and avoids repeated entropy computation.

### Local Perplexity Contrast
At each observation scale \(s\), REGPLEX computes a three-window **Local Perplexity Contrast (LPC)** profile. The flank and spacer geometry is:

- upstream window = \(s\)
- spacer = \(s/2\)
- candidate window = `clamp(s, min_candidate, max_candidate)`
- downstream window = \(s\)

The local contrasts are defined as:

\[
LPC_{up} = \overline{U} - \overline{C}, \qquad
LPC_{down} = \overline{D} - \overline{C}
\]

where \(\overline{U}\), \(\overline{C}\), and \(\overline{D}\) are the upstream, candidate, and downstream means of the scale-specific landscape. Positions are rejected if either flank does not exceed the candidate mean. The final per-position LPC value is:

\[
LPC = \min(LPC_{up}, LPC_{down})
\]

This conservative definition requires bilateral support and suppresses single-sided troughs.

### Robust normalization and consensus aggregation
Each per-scale LPC profile is normalized independently using a robust z-score:

\[
z = \frac{x - \mathrm{median}(x)}{\mathrm{MAD}(x) \times 1.4826}
\]

The **Consensus LPC** is then obtained by taking the position-wise `nanmedian` across all normalized scale profiles. The resulting track highlights intervals that remain valley-like across multiple spatial resolutions rather than at a single arbitrary window size.

### Valley detection and boundary expansion
REGPLEX identifies candidate valley cores from positive Consensus LPC regions using bounded Kadane-style segmentation, with an upper bound on segment length. Core intervals are then expanded to natural boundaries defined by the contiguous positive Consensus LPC neighborhood. Nearby intervals are merged when separated by less than the configured merge gap. Final valleys are filtered by minimum and maximum valley length constraints.

### Valley-level summary metrics
For each reported valley, REGPLEX computes:

- genomic coordinates and interval length
- mean, minimum, and maximum P1
- variance, standard deviation, and coefficient of variation of P1
- mean flanking background P1
- mean and maximum Consensus LPC
- area under the positive Consensus LPC profile
- GC content
- sequence string

Three composite quality metrics are also reported:

- **Persistence**: fraction of valley positions with positive Consensus LPC
- **ScaleSupport**: fraction of observation scales with positive mean raw LPC over the valley
- **ValleyScore**:  
  \[
  \mathrm{MeanLPC} \times \mathrm{Persistence} \times \mathrm{ScaleSupport} \times \log(\mathrm{Length}) \times \frac{1}{\mathrm{Variance} + \varepsilon}
  \]

An internal normalized valley score is additionally reported for ranking.

### Motif annotation
Optional motif annotation is performed after valley detection. Users may provide one motif per line as either a regular expression or an IUPAC DNA motif. IUPAC patterns are translated to regular expressions before compilation. Matching is performed only against valley sequences, and the output records both a total motif count and a semicolon-delimited per-motif hit summary.

### Software implementation
REGPLEX v9 is implemented in Python and distributed with both command-line and Streamlit interfaces. The principal repository modules are:

- `regplex_core.py` — sequence analysis engine
- `motif_engine.py` — motif parsing and annotation
- `visualization.py` — visualization routines
- `app.py` — Streamlit application
- `styles.css` — custom user-interface stylesheet

### Tool information
The current repository declares the following software dependencies:

- Python 3
- NumPy `>=1.20.0`
- pandas `>=1.3.0`
- openpyxl `>=3.0.0`
- Plotly `>=5.0.0`
- Streamlit `>=1.25.0`

### Execution
The documented entry points are:

```bash
streamlit run app.py
python regplex_core.py examples/ecoli.fasta --out regplex_valleys.csv
```

## Reporting Summary

### Research design
This work presents a deterministic computational method for sequence-based discovery of candidate regulatory-like regions. The framework is methodological and does not rely on randomized training, manual curation during inference, or organism-specific model calibration.

### Input data
REGPLEX accepts nucleotide sequences in FASTA-compatible text. Example files bundled with the repository include bacterial and human promoter-oriented inputs located under `/home/runner/work/PerCALL/PerCALL/examples/`.

### Inclusion and exclusion criteria
All non-empty FASTA records are included after sanitization. Non-DNA characters are removed, uracil is converted to thymine, and windows containing ambiguous bases are excluded from local perplexity calculation by assignment to missing values.

### Outcome definition
The primary output is a ranked set of Perplexity Valleys representing candidate regulatory intervals. Outputs are available as tabular, interval, sequence, and structured formats.

### Reproducibility
The workflow is deterministic for a fixed input sequence and parameter set. No stochastic optimization, bootstrapping, or random seed configuration is required.

### Statistical considerations
The method uses information-theoretic sequence complexity, robust median/MAD normalization, bounded segmentation, and rule-based aggregation across scales. Reported scores are descriptive algorithmic metrics rather than inferential p-values.

### Visualization and reporting outputs
The application provides plots for the P1 profile, multi-scale landscapes, scale-support heatmap, consensus LPC, Kadane segments, ranked valleys, motif architecture, and workflow overview. Exported outputs include CSV, XLSX, BED, GFF, GFF3, FASTA, and JSON.

### Limitations relevant to reporting
The framework identifies candidate regions by intrinsic sequence-structure signatures and does not, by itself, establish biochemical activity, chromatin state, transcription factor occupancy, or causal regulatory function. Such interpretation requires external experimental or comparative validation.

## Code Availability
The REGPLEX v9 source code is available in the GitHub repository `VRYella/PerCALL`: <https://github.com/VRYella/PerCALL>. The main executable interfaces are the Streamlit application (`streamlit run app.py`) and the command-line pipeline (`python regplex_core.py <fasta> --out <csv>`). The implementation files used for analysis and visualization are located at `/home/runner/work/PerCALL/PerCALL/app.py`, `/home/runner/work/PerCALL/PerCALL/regplex_core.py`, `/home/runner/work/PerCALL/PerCALL/motif_engine.py`, and `/home/runner/work/PerCALL/PerCALL/visualization.py`.

## Data Availability
No newly generated large-scale datasets are bundled with this methods package. Example input files distributed with the repository for demonstration and smoke testing are available at `/home/runner/work/PerCALL/PerCALL/examples/ecoli.fasta` and `/home/runner/work/PerCALL/PerCALL/examples/human_promoters.fasta`. Output tables can be reproduced locally by running the documented command-line or Streamlit workflows on these example inputs or on user-supplied FASTA files.

## References
1. Shannon CE. A mathematical theory of communication. *Bell System Technical Journal*. 1948;27:379-423, 623-656.
2. Bentley JL. Programming Pearls: algorithm design techniques. *Communications of the ACM*. 1984;27(9):865-873.
3. Rousseeuw PJ, Croux C. Alternatives to the median absolute deviation. *Journal of the American Statistical Association*. 1993;88(424):1273-1283.
4. Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. *Nature*. 2020;585:357-362.
5. McKinney W. Data structures for statistical computing in Python. In: *Proceedings of the 9th Python in Science Conference*. 2010:56-61.
6. Plotly Technologies Inc. Plotly Python Graphing Library. Available at: <https://plotly.com/python/>.
7. Streamlit Inc. Streamlit documentation. Available at: <https://docs.streamlit.io/>.
8. openpyxl contributors. openpyxl documentation. Available at: <https://openpyxl.readthedocs.io/>.
