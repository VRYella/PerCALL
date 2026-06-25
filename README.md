# REGPLEX

**Perplexity Valley Discovery from DNA Sequence Alone**

REGPLEX is a training-free, species-independent sequence analysis framework that discovers **Perplexity Valleys**: genomic regions where sequence uncertainty is locally depressed relative to the surrounding landscape.

## Core pipeline

DNA sequence  
↓  
10-mer dinucleotide perplexity (**P1**)  
↓  
second-order perplexity landscape (**P2**)  
↓  
three-window valley comparison  
↓  
Perplexity Valley Score (**PVS**)  
↓  
bounded minimum-mean Kadane optimization  
↓  
valley domains  
↓  
confidence ranking  
↓  
optional motif annotation

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

### 2. Second-order perplexity (P2)
The P1 signal is stabilized with a sliding second-order window. REGPLEX averages local first-order entropy and converts it back to perplexity to produce a smooth uncertainty landscape.

### 3. Three-window valley model
For each genomic position REGPLEX compares:

- upstream context window
- spacer
- candidate window
- spacer
- downstream context window

Fixed-center mode uses one candidate width. Adaptive mode scans a user-defined candidate-width interval and keeps the strongest local valley score obtained with cumulative-sum-based window means.

### 4. Perplexity Valley Score (PVS)
For each comparison:

- `UpstreamDifference = UpstreamMean - CandidateMean`
- `DownstreamDifference = DownstreamMean - CandidateMean`
- `CombinedValleyScore = ((UpstreamMean + DownstreamMean) / 2) - CandidateMean`

Positive PVS indicates a local uncertainty valley.

### 5. Domain recovery
REGPLEX runs the existing bounded minimum-mean Kadane routine on the PVS track to recover sustained positive-valley domains across the full sequence.

### 6. Confidence ranking
Each domain is ranked with a normalized confidence score built from:

- contrast = mean PVS
- persistence = fraction of positions with positive PVS
- length = log(domain length)
- stability = `1 / (variance(P1) + ε)`

## Installation

```bash
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
- Mean P2
- Mean PVS
- Maximum PVS
- Minimum P1
- Upstream mean
- Candidate mean
- Downstream mean
- Upstream difference
- Downstream difference
- Combined valley score
- Variance / SD / CV
- GC content
- Persistence
- Confidence
- Candidate window
- Motif count / motifs
- Domain sequence

## Visualizations

Plotly figures provided by the app:

1. Raw Perplexity Profile
2. Second-Order Landscape
3. Three-Window Valley Illustration
4. Genome-wide Valley Score
5. Kadane Domain Selection
6. Domain Ranking
7. Motif Architecture
8. Complete Algorithm Workflow

## Motif annotation

Motif annotation is optional and is never used for discovery. Supply one IUPAC motif or regular expression per line in the app. REGPLEX scans only predicted domains.
