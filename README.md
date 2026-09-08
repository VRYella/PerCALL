# REGPLEX

REGPLEX predicts **candidate regulatory regions** by detecting persistent local depressions in DNA dinucleotide perplexity and annotating the resulting intervals with curated regulatory motif patterns.

## Core method

```text
DNA sequence
    ↓
16 dinucleotide probabilities
    ↓
Shannon entropy
    ↓
DNA perplexity = 2^H
    ↓
local background comparison
    ↓
perplexity depression (PDS)
    ↓
persistent candidate-region detection
    ↓
transparent ranking
```

For each window:

- \(H = -\sum_i p_i \log_2 p_i\)
- \(\text{PPL} = 2^H\)

REGPLEX compares local perplexity to surrounding flanks:

- \(\text{PDS}(x) = P_{background}(x) - PPL(x)\)

Positive PDS means local sequence is less perplexing than its local background.

## Scientific interpretation

REGPLEX does **not** claim that low perplexity uniquely defines promoters. It prioritizes **candidate regulatory regions** that should be validated against biological datasets.

Each region is reported with explainable quantities:

- mean perplexity
- local background perplexity
- mean and max PDS
- persistence (bp)
- length
- rank

## Project structure

```text
src/
  preprocessing/   # FASTA parsing, input-source loading, and sequence validation
  perplexity/      # entropy and perplexity profile generation
  prediction/      # background, PDS, region detection, ranking
  motifs.py        # motif-library parsing and region annotation
  visualization/   # profile and region plotting helpers
  output/          # CSV/BED/GFF/FASTA exporters
  models/          # dataclasses for config/profile/results

app/
  app.py
  pages/
    home.py
    analyze.py
    results.py
    visualization.py
    download.py
    help.py

tests/
  test_entropy.py
  test_perplexity.py
  test_background.py
  test_depression.py
  test_region_detection.py
  test_prediction.py
```

regulatory_motifs.txt  # bundled regulatory + non-B DNA motif library

## Run

```bash
pip install -r requirements.txt  # installs numpy, scipy, pandas, plotly, streamlit, openpyxl
streamlit run app.py
```

The **Analyze** page accepts:

- pasted FASTA or raw DNA
- uploaded FASTA/text files
- indexed local FASTA/text files from the workspace or `/tmp`
- NCBI nucleotide accessions

Motifs are read from `regulatory_motifs.txt` by default, and you can append custom tab-delimited `name<TAB>pattern` lines from the UI.

## CLI

```bash
python regplex_core.py examples/ecoli.fasta --out regplex_regions.csv
```

With motif annotation:

```bash
python regplex_core.py examples/ecoli.fasta \
  --motifs-file regulatory_motifs.txt \
  --out regplex_regions.csv
```

With NCBI accession input:

```bash
python regplex_core.py --accession NC_000913.3 --out regplex_regions.csv
```

## Motif library

The bundled library includes named promoter-associated and structure-associated patterns such as:

- TATA box
- CAAT box
- GC box
- BRE upstream/downstream elements
- Initiator-like patterns
- DPE-like patterns
- CpG-rich seeds
- G-quadruplex-like patterns
- repeat-associated non-B DNA signatures

Each detected region reports:

- `Motif_Count`
- `Motifs` as a semicolon-separated `name:count` summary

## Testing

Run the existing test suite with:

```bash
pytest
```

## Supervised fine-tuning pipeline (human + E. coli)

`supervised_finetune.py` adds a leakage-safe supervised benchmark workflow with:

- real labeled promoter/non-promoter FASTA datasets for **human** and **E. coli**
- canonical deduplication (sequence + reverse-complement grouping) before splitting
- stratified **train/val/test** splitting by species and class
- hyperparameter tuning on validation split
- metric reporting: **precision, recall, F1, MCC** + confusion counts

Run:

```bash
python supervised_finetune.py \
  --cache-dir /tmp/regplex_supervised_data \
  --output-json /tmp/regplex_supervised_report.json
```

Dataset provenance used by the pipeline:

- `human_non_tata.fa` (positive) and `human_nonprom_big.fa` (negative)
- `Ecoli_prom.fa` (positive) and `Ecoli_non_prom.fa` (negative)
- source repository: `nmach22/Promoter-Classification`
- original biological sources documented there include EPDnew (human promoters) and RegulonDB (E. coli promoters)
