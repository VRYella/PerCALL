# REGPLEX v9

**Multi-scale Perplexity Valley Framework for DNA Regulatory Region Discovery**

REGPLEX v9 is a training-free, species-independent framework for discovering
**Perplexity Valleys (PVs)**: continuous genomic regions where sequence
uncertainty is consistently lower than the surrounding DNA *across multiple
observation scales*.

## Core pipeline

```
DNA sequence
↓
10-mer dinucleotide perplexity (P1) — computed exactly once
↓
Multi-scale Landscapes (default: 25 / 50 / 100 / 200 / 400 bp)
↓
Per-scale Three-window LPC (upstream = downstream = scale, spacer = scale÷2)
↓
Robust z-score normalisation of each LPC profile
↓
Consensus LPC (nanmedian across normalised profiles)
↓
Bounded Kadane valley core detection + natural boundary expansion
↓
Valley merging
↓
Quality metrics + valley score
↓
Optional motif annotation (IUPAC/regex; valleys only)
```

## Scientific definitions

At each observation scale *s*, for every genomic position:

- `upstream = s`, `spacer = s÷2`, `candidate = max(s, 50)`, `downstream = s`
- `LPC_up  = UpstreamMean  − CandidateMean`
- `LPC_down = DownstreamMean − CandidateMean`
- Reject if `LPC_up ≤ 0` or `LPC_down ≤ 0`
- `LPC = min(LPC_up, LPC_down)`

Each LPC profile is independently normalised by robust z-score:

```
z = (x − median) / (MAD × 1.4826)
```

The **Consensus LPC** is the position-wise nanmedian of all normalised profiles.

Valley quality metrics:

- `Persistence   = fraction of valley positions with ConsensusLPC > 0`
- `ScaleSupport  = fraction of scales with mean raw LPC > 0 inside the valley`
- `ValleyScore   = MeanLPC × Persistence × ScaleSupport × log(Length) × 1/(Variance+ε)`

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
python regplex_core.py examples/ecoli.fasta --base-scale 200 --landscape-method median
```

## Outputs

Downloads include: CSV · Excel (.xlsx) · BED · GFF · GFF3 · FASTA · JSON

## Reported columns

| Column | Description |
|---|---|
| ID | Valley identifier (PV_XXXXXX) |
| Sequence_ID | Source sequence name |
| Start / End / Length | Nucleotide coordinates (0-based) |
| MeanP1 / MinimumP1 / MaximumP1 | P1 statistics inside the valley |
| Variance / SD / CV | P1 variability |
| MeanBackground | Mean P1 in flanking regions |
| MeanLPC / MaximumLPC | Consensus LPC statistics |
| AreaUnderValley | Sum of positive ConsensusLPC |
| Persistence | Fraction of valley with ConsensusLPC > 0 |
| ScaleSupport | Fraction of scales supporting the valley |
| NScales | Total number of observation scales used |
| ValleyScore / ValleyScoreNormalized | Raw and normalised valley scores |
| GC% | GC content of the valley sequence |
| MotifCount / Motifs | Motif annotation results |
| Sequence | DNA sequence of the valley |

## Visualizations (Plotly, SVG/PNG/PDF export)

1. Perplexity Profile (P1)
2. Multi-scale Perplexity Landscapes
3. Scale Support Heatmap
4. Consensus LPC
5. Kadane Segments
6. Valley Ranking
7. Motif Architecture
8. Complete Workflow (Sankey)

## REGPLEX v9 interface

- Full-width, no-sidebar layout with fixed top navigation
- Pages: Home · Analysis · Results · Motifs · About
- New parameter controls: Base scale · Custom scales · Merge gap · Min/Max candidate

## Repository layout

- `app.py` — Streamlit application
- `regplex_core.py` — Multi-scale analysis engine
- `motif_engine.py` — IUPAC/regex motif scanner
- `visualization.py` — Plotly figure library
- `styles.css` — UI design system
- `REGPLEX_Local.ipynb` — Jupyter notebook interface
- `examples/` — Example FASTA files
