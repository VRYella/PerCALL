# REGPLEX v13

**A training-free computational framework for identifying extended genomic perplexity valleys.**

REGPLEX identifies regions where local sequence complexity is significantly lower than the surrounding genomic background, using dinucleotide perplexity, three-window contrast analysis, bounded Kadane optimization, and interpretable valley ranking.

---

## Scientific Hypothesis

> Identify extended low-perplexity genomic valleys relative to their local genomic background using a training-free information-theoretic framework.

REGPLEX is not a promoter predictor, motif scanner, or classifier. It is a general-purpose framework for detecting extended regions of anomalously low sequence complexity.

---

## Algorithm

```
DNA sequence
    │
    ▼
Dinucleotide Perplexity  (17 nt sliding window)
    │
    ▼
Savitzky–Golay Smoothing  (window 21 bp, polynomial order 3)
    │
    ▼
Perplexity Depression Score (PDS)
    Three-window contrast:
    ┌────────────┐  spacer  ┌──────────────────┐  spacer  ┌────────────┐
    │  Upstream  │  50 bp   │    Candidate      │  50 bp   │ Downstream │
    │  100 bp    │          │    (adaptive)     │          │  100 bp    │
    └────────────┘          └──────────────────┘          └────────────┘
    PDS = ((UpstreamMean + DownstreamMean) / 2) − CandidateMean
    │
    ▼
Bounded Kadane Detection  (100–1000 bp)
    │
    ▼
Valley Expansion  (expand while PDS > 0 or PDS > 20% of peak)
    │
    ▼
Valley Merging  (merge if gap ≤ 100 bp)
    │
    ▼
Biological Filter  (UpstreamMean > CandidateMean AND DownstreamMean > CandidateMean)
    │
    ▼
ValleyScore Ranking
    │
    ▼
Optional Motif Annotation
    │
    ▼
Downloads (CSV / Excel / BED / GFF3 / FASTA / JSON)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `scipy`, `pandas`, `openpyxl`, `streamlit`, `plotly`

---

## Usage

### Streamlit web application

```bash
streamlit run app.py
```

### Command-line interface

```bash
python regplex_core.py examples/ecoli.fasta --out results.csv
```

### Python API

```python
from regplex_core import parse_fasta, analyze_sequence

records = parse_fasta(open("examples/ecoli.fasta").read())
header, seq = records[0]
result = analyze_sequence(header, seq)

for valley in result.domains:
    print(valley["ID"], valley["Start"], valley["End"], valley["ValleyScore"])
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `perplexity_window` | 17 | Dinucleotide perplexity sliding window (nt) |
| `sg_window` | 21 | Savitzky–Golay smoothing window (bp, odd) |
| `sg_order` | 3 | Savitzky–Golay polynomial order |
| `flank_size` | 100 | Upstream/downstream flank window (bp) |
| `spacer_size` | 50 | Gap between flank and candidate windows (bp) |
| `min_candidate` | 50 | Minimum candidate window size (bp) |
| `max_candidate` | 1000 | Maximum candidate window size (bp) |
| `min_valley_length` | 100 | Minimum accepted valley length (bp) |
| `max_valley_length` | 1000 | Maximum accepted valley length (bp) |
| `merge_gap` | 100 | Maximum gap between valleys to merge (bp) |

---

## Input

- FASTA format, single or multi-record
- DNA sequences (A, C, G, T, and IUPAC ambiguity codes)
- No minimum or maximum length enforced

---

## Output Columns

| Column | Description |
|--------|-------------|
| `ID` | Unique valley identifier (`PV_000001`, …) |
| `Start` | 0-based start position |
| `End` | End position (exclusive) |
| `Length` | Valley length (bp) |
| `MeanPerplexity` | Mean dinucleotide perplexity within valley |
| `MinPerplexity` | Minimum perplexity within valley |
| `MaxPerplexity` | Maximum perplexity within valley |
| `UpstreamMean` | Mean perplexity in upstream flank |
| `CandidateMean` | Mean perplexity in candidate region |
| `DownstreamMean` | Mean perplexity in downstream flank |
| `UpstreamDifference` | UpstreamMean − CandidateMean |
| `DownstreamDifference` | DownstreamMean − CandidateMean |
| `PDSMean` | Mean Perplexity Depression Score |
| `PDSMax` | Maximum Perplexity Depression Score |
| `Prominence` | Peak perplexity contrast |
| `Persistence` | Fraction of valley positions with PDS > 0 |
| `AreaUnderValley` | Integral of perplexity depression |
| `Variance` | Perplexity variance within valley |
| `GC%` | GC content of valley sequence |
| `MotifCount` | Number of motif matches |
| `Motifs` | Matched motif patterns |
| `Sequence` | Valley nucleotide sequence |
| `ValleyScore` | Composite interpretable score |
| `ValleyScoreNormalized` | ValleyScore normalized 0–1 |
| `Rank` | Rank by ValleyScore |

---

## ValleyScore Formula

```
Contrast     = PDSMean
LengthFactor = log(Length)
Stability    = 1 / (Variance + 1e-9)

ValleyScore  = Contrast × Persistence × LengthFactor × Stability
```

Normalized 0–1 within each sequence. No arbitrary weights.

---

## Motif Annotation

REGPLEX supports optional motif annotation of detected valleys.

Accepted formats:

- **IUPAC codes** — e.g. `TATAWAWR`, `GCNNNNNGC`
- **Regular expressions** — e.g. `GGGN{1,7}GGG`, `(CAG){5,}`

Motifs are scanned only within detected valleys — never genome-wide. They do not influence detection.

---

## File Structure

```
regplex_core.py      # Core algorithm: perplexity, PDS, Kadane, metrics
visualization.py     # Plotly figures: perplexity, PDS, three-window, ranking
motif_engine.py      # IUPAC/regex motif scanning
app.py               # Streamlit web application
examples/            # Example FASTA files
README.md
```

---

## Design Principles

- **Training-free** — no reference genomes, no labelled data, no species-specific parameters
- **Single signal** — dinucleotide perplexity only; no mono- or tri-perplexity
- **Single pass** — one smoothing operation, no multi-scale layers
- **Interpretable** — every component of ValleyScore is directly interpretable
- **No mode switching** — identical algorithm for all inputs

---

## Citation

If you use REGPLEX in your research, please cite the repository:

```
REGPLEX v13 — https://github.com/VRYella/PerCALL
```
