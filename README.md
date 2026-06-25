# REGPLEX

**Perplexity Valley Discovery from DNA Sequence Alone**

REGPLEX is a training-free, species-independent sequence analysis framework that discovers **Perplexity Valleys**: genomic regions where sequence uncertainty is locally and symmetrically depressed relative to the surrounding landscape.

## Core pipeline

DNA sequence  
↓  
10-mer dinucleotide perplexity (**P1**)  
↓  
Sliding smooth landscape (**P2**)  
↓  
Three-window valley analysis  
↓  
Perplexity Valley Score (**PVS = Contrast × Symmetry**)  
↓  
Bounded Kadane optimization  
↓  
Valley domains  
↓  
Confidence ranking  
↓  
Optional motif annotation

## Design principles

- Training free
- Species independent
- Annotation independent
- Motif independent for discovery
- Explainable
- Vectorized with NumPy and prefix sums
- Practical for genome-scale scans

## Algorithm summary

### 1. First-order perplexity (P1)
REGPLEX computes 10-mer dinucleotide perplexity along the input DNA sequence.

### 2. Smooth landscape (P2)
The P1 signal is stabilized with a sliding window mean to produce a smooth uncertainty landscape.

### 3. Three-window valley model
For each genomic position REGPLEX compares:

- upstream context window
- spacer
- candidate window (fixed or adaptive 50–300 bp)
- spacer
- downstream context window

**First decision**: upstream must be elevated relative to the candidate (upstream mean > candidate mean). Positions that fail this test are rejected immediately.

Fixed-center mode uses one candidate width. Adaptive mode scans a user-defined candidate-width interval using prefix sums and keeps the strongest local valley score.

### 4. Perplexity Valley Score (PVS)

For each position:

- `UpstreamDifference = UpstreamMean - CandidateMean`  (first decision, must be > 0)
- `DownstreamDifference = DownstreamMean - CandidateMean`
- `Contrast = ((UpstreamMean + DownstreamMean) / 2) - CandidateMean`
- `Symmetry = 1 - |UpstreamMean - DownstreamMean| / (UpstreamMean + DownstreamMean)`
- `PVS = Contrast × Symmetry`

High PVS indicates a deep, symmetric valley. Near-zero indicates background. Negative indicates not a valley.

### 5. Domain recovery
REGPLEX runs bounded minimum-mean Kadane on the PVS track to recover sustained positive-valley domains.

### 6. Confidence ranking
Each domain is ranked with a normalized confidence score:

```
Confidence ∝ mean(PVS) × Persistence × log(Length) × 1/(Variance+ε)
```

Since `PVS = Contrast × Symmetry`, this is equivalent to `Contrast × Symmetry × Persistence × log(Length) × 1/(Variance+ε)` normalized to 0–1.

## Installation

```bash
git clone https://github.com/VRYella/PerCALL.git
cd PerCALL
pip install -r requirements.txt
```

## Usage

### Streamlit app

```bash
streamlit run app.py
```

### Command line

```bash
python regplex_core.py examples/ecoli.fasta --out regplex_domains.csv
```

## Outputs

REGPLEX exports all predicted valleys as:

- CSV
- TSV
- Excel (.xlsx)
- BED
- GFF
- GFF3
- FASTA
- JSON

## Reported columns

Each predicted domain includes:

- Domain ID
- Sequence ID
- Start / End / Length
- Mean P1
- Minimum P1
- Maximum P1
- Mean smoothed P1 (P2)
- Mean PVS
- Maximum PVS
- Area Under Valley
- Upstream mean
- Candidate mean
- Downstream mean
- Upstream difference
- Downstream difference
- Combined valley score
- Symmetry
- Variance / SD / CV
- GC content
- Persistence
- Confidence (0–1 normalized)
- Candidate window
- Motif count / motifs
- Domain sequence

## Visualizations

Plotly figures provided by the app:

1. Raw Perplexity Profile
2. Smoothed Landscape
3. Three-Window Valley Illustration
4. Genome-wide Valley Score (PVS)
5. Kadane Domain Selection
6. Domain Ranking
7. Motif Architecture
8. Complete Algorithm Workflow

## Motif annotation

Motif annotation is optional and is never used for discovery. Supply one IUPAC motif or regular expression per line in the app. REGPLEX scans only predicted domains.
