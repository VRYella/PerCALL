# REGPLEX

**Perplexity Valley discovery from DNA sequence alone**

REGPLEX is a training-free, species-independent, annotation-independent framework for discovering **Perplexity Valleys (PVs)**: sustained regions where sequence uncertainty is reduced relative to immediate genomic context.

## Core pipeline

DNA sequence  
↓  
10-mer dinucleotide perplexity (**P1**)  
↓  
Perplexity Landscape (sliding mean/median)  
↓  
Local Perplexity Contrast (**LPC**)  
↓  
Adaptive candidate optimization (50–300 bp default)  
↓  
Bounded Kadane valley recovery  
↓  
Valley quality metrics + score  
↓  
Optional motif annotation (IUPAC/regex; valleys only)

## Scientific definitions

For each position, REGPLEX compares upstream/candidate/downstream windows with spacer regions:

- `LPC_up = UpstreamMean - CandidateMean`
- `LPC_down = DownstreamMean - CandidateMean`
- Reject if `LPC_up <= 0` or `LPC_down <= 0`
- `LPC = min(LPC_up, LPC_down)`

Additional interpretation metrics:

- `Symmetry = 1 - |UpstreamMean - DownstreamMean| / (UpstreamMean + DownstreamMean + ε)`
- `Persistence = fraction of valley positions with P1 below local flank mean`
- `ValleyScore = MeanLPC × Persistence × log(Length)`

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
python regplex_core.py examples/ecoli.fasta --out regplex_valleys.csv
```

## Outputs

Downloads include:

- CSV
- Excel (.xlsx)
- BED
- GFF
- GFF3
- FASTA
- JSON

## Reported columns

- ID
- Sequence_ID
- Start / End / Length
- MeanP1 / MinimumP1 / MaximumP1
- Variance / SD / CV
- MeanLPC / MaximumLPC
- AreaUnderValley
- UpstreamMean / CandidateMean / DownstreamMean
- LPC_up / LPC_down
- Persistence
- Symmetry
- ValleyScore / ValleyScoreNormalized
- GC%
- MotifCount / Motifs
- Sequence

## Visualizations (Plotly)

1. Perplexity Landscape
2. Adaptive Valley Detection
3. Three Window Illustration
4. Local Perplexity Contrast Profile
5. Kadane Optimization
6. Valley Ranking
7. Motif Architecture
8. Complete Workflow

## Repository layout

- `app.py`
- `regplex_core.py`
- `motif_engine.py`
- `visualization.py`
- `README.md`
- `REGPLEX_Local.ipynb`
- `examples/`
