# REGPLEX v13

**Training-Free Perplexity Valley Detector**

REGPLEX identifies extended low-perplexity genomic valleys relative to their local genomic background using a training-free information-theoretic framework.

---

## Scientific Hypothesis

> "Identify extended low-perplexity genomic valleys relative to their local genomic background using a training-free information-theoretic framework."

REGPLEX is **not** a promoter predictor, not a motif predictor, and not a classifier.  
No training data. No species-specific parameters. No machine learning.

---

## Algorithm Pipeline

```
DNA
 ↓
Dinucleotide Perplexity  (window = 17 nt)
 ↓
Savitzky-Golay Smoothing  (window=21, order=3)
 ↓
Perplexity Depression Score (PDS)
 ↓
Bounded Kadane Valley Detection  (100–1000 bp)
 ↓
Valley Expansion
 ↓
Valley Merging  (gap ≤ 100 bp)
 ↓
Valley Ranking  (ValleyScore)
 ↓
Optional Motif Annotation
 ↓
Downloads
```

### Step 1 — Dinucleotide Perplexity

Only dinucleotide perplexity is used. Window = 17 nt (16 dinucleotide transitions).  
This provides a stable local estimate while remaining highly local.

### Step 2 — Savitzky-Golay Smoothing

Applied **exactly once** (window=21, order=3). Never stacked.

### Step 3 — Perplexity Depression Score

Three-window contrast for each position:

```
[upstream 100 bp] [spacer 50 bp] [candidate] [spacer 50 bp] [downstream 100 bp]
```

```
PDS = ((UpstreamMean + DownstreamMean) / 2) − CandidateMean
```

Rejected if either flank mean ≤ candidate mean.  
Positive PDS = genuine depression relative to local background.

### Step 4 — Bounded Kadane

Applied only to the PDS profile. Identifies all positive-PDS valleys of length 100–1000 bp.

### Step 5 — Valley Expansion

Each Kadane core is expanded while:
- `PDS > 0` OR
- `PDS > 20% of peak PDS`

### Step 6 — Valley Merging

Adjacent valleys with gap ≤ 100 bp are merged into a single biological domain.

### Step 7 — ValleyScore

```
ValleyScore = PDSMean × Persistence × log(Length) × Stability
Stability   = 1 / (Variance + 1e-9)
```

Normalised to [0, 1]. Ranked by ValleyScore.

---

## Output Columns

| Column | Description |
|--------|-------------|
| ID | Valley identifier (PV_XXXXXX) |
| Start / End | Nucleotide coordinates |
| Length | Valley length (bp) |
| MeanPerplexity | Mean di-perplexity in valley |
| MinPerplexity / MaxPerplexity | Range |
| UpstreamMean | Upstream flank perplexity |
| CandidateMean | Valley perplexity |
| DownstreamMean | Downstream flank perplexity |
| UpstreamDifference | UpstreamMean − CandidateMean |
| DownstreamDifference | DownstreamMean − CandidateMean |
| PDSMean / PDSMax | PDS statistics |
| Prominence | Background − valley minimum |
| Persistence | Fraction of valley positions with PDS > 0 |
| AreaUnderValley | Integral of positive PDS |
| Variance | PDS variance (lower = more stable) |
| GC% | GC content |
| MotifCount / Motifs | Optional motif annotation |
| ValleyScore | Primary ranking score |
| ValleyScoreNormalized | Normalised 0–1 |
| Rank | 1 = best |

---

## Usage

### Streamlit Web App

```bash
streamlit run app.py
```

### Command Line

```bash
python regplex_core.py <fasta> --out valleys.csv
```

Options:

```
--sg-window       Savitzky-Golay window (default 21)
--sg-order        Savitzky-Golay order (default 3)
--flank-size      PDS flank window (default 100 bp)
--spacer-size     PDS spacer (default 50 bp)
--min-candidate   PDS min candidate (default 50 bp)
--max-candidate   PDS max candidate (default 1000 bp)
--min-valley      Minimum valley length (default 100 bp)
--max-valley      Maximum valley length (default 1000 bp)
--merge-gap       Valley merge gap (default 100 bp)
```

---

## Requirements

```
numpy>=1.20.0,<3.0.0
openpyxl>=3.0.0,<4.0.0
pandas>=1.3.0,<4.0.0
plotly>=5.0.0,<7.0.0
scipy>=1.7.0,<2.0.0
streamlit>=1.40.0,<2.0.0
```

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `regplex_core.py` | Core algorithm (perplexity, PDS, Kadane, metrics) |
| `motif_engine.py` | IUPAC/regex motif scanner |
| `visualization.py` | Plotly figures (white scientific theme) |
| `examples/` | Example FASTA files |

---

## License

MIT License
