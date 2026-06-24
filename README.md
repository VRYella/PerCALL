# REGPLEX

**Regulatory Architecture Discovery Through Perplexity Depression**

## Scientific Motivation

Functional genomic regions can show localized collapse in sequence uncertainty relative to surrounding DNA.
REGPLEX quantifies this as **Perplexity Depression Index (PDI)** and discovers sustained uncertainty-collapse domains without training data, species assumptions, or external annotations.

## Algorithm

DNA  
↓  
10-mer Perplexity (P1)  
↓  
Perplexity Depression Index (PDI)  
↓  
Bounded Minimum-Mean Kadane  
↓  
Domains  
↓  
Optional Motif Annotation

## Installation

```bash
git clone https://github.com/VRYella/PerCALL.git
cd PerCALL
pip install -r requirements.txt
```

## Usage

### Streamlit

```bash
streamlit run app.py
```

### Notebook

Open `REGPLEX_Local.ipynb` and run cells top-to-bottom.

### Command Line

```bash
python regplex_core.py examples/ecoli.fasta --out regplex_domains.csv
```

## Outputs

REGPLEX exports:

- CSV
- TSV
- Excel (.xlsx)
- BED
- GFF
- GFF3
- FASTA
- JSON

## Figures

The app provides Plotly publication figures:

1. P1 profile
2. PDI profile
3. Domain map
4. Domain ranking
5. Domain statistics
6. Motif architecture
7. Algorithm illustration

Add screenshots in this section for manuscript submission.

## Citation

> Citation placeholder: add manuscript DOI and software citation after publication.
