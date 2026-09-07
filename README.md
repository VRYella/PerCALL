# PerCALL

PerCALL predicts **candidate regulatory regions** by detecting persistent local depressions in DNA dinucleotide perplexity.

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

PerCALL compares local perplexity to surrounding flanks:

- \(\text{PDS}(x) = P_{background}(x) - PPL(x)\)

Positive PDS means local sequence is less perplexing than its local background.

## Scientific interpretation

PerCALL does **not** claim that low perplexity uniquely defines promoters. It predicts **candidate regulatory regions** that should be validated against biological datasets.

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
  preprocessing/   # FASTA parsing and sequence validation
  perplexity/      # entropy and perplexity profile generation
  prediction/      # background, PDS, region detection, ranking
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

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI

```bash
python regplex_core.py examples/ecoli.fasta --out percall_regions.csv
```
