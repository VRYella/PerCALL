<div align="center">

# 🧬 REGPLEX

**Regulatory Domain Discovery Using DNA Sequence Perplexity**

*Predict regulatory domains in DNA sequences using information theory, Kadane's algorithm, and Non-B DNA structure detection — all from a sleek scientific web interface.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## ✨ What is REGPLEX?

REGPLEX analyses DNA sequences and **discovers regulatory domains** by combining three complementary signals:

| Signal | Method | Rationale |
|--------|--------|-----------|
| 📐 **Dinucleotide perplexity** | Sliding-window entropy | Promoters have lower compositional complexity |
| 🔍 **Non-B DNA motifs** | Regex scanning (8 classes) | Structural motifs are enriched at regulatory loci |
| 🏆 **Kadane's algorithm** | Modified minimum-subarray | Globally optimal low-perplexity region extraction |

The result is a ranked list of predicted regulatory domains with confidence scores, interactive visualisations, and one-click data export.

---

## 🗂 Repository Layout

```
REGPLEX/
├── streamlit_app.py        ← main web application (single entry-point)
├── core/
│   ├── motifs.py           ← Non-B DNA regex patterns & scanning
│   ├── perplexity.py       ← sliding-window entropy calculation
│   ├── plotting.py         ← Plotly chart builders
│   ├── region_caller.py    ← Kadane-based region detection
│   └── report.py           ← PDF report generation (fpdf2)
├── example_data/           ← 11 curated bacterial/fungal genomes
├── output/                 ← analysis results land here
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .streamlit/config.toml
```

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/VRYella/PerCALL.git
cd PerCALL
docker compose up
```

Open **http://localhost:8501** in your browser.

### Option B — Local Python

```bash
git clone https://github.com/VRYella/PerCALL.git
cd PerCALL
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🖥 Interface Overview

REGPLEX is a **five-page horizontal-navbar app** — no sidebar, no clutter.

| Tab | Purpose |
|-----|---------|
| 🏠 **Home** | Hero overview, feature cards, workflow guide |
| 🔬 **Analysis** | Sequence input, parameter controls, run analysis |
| 📊 **Results** | Interactive Plotly charts, region table, motif hits |
| 🧪 **Examples** | 11 curated example genomes with expected outputs |
| 📄 **Reports & Exports** | CSV / JSON / PDF downloads |

---

## 🔬 Algorithm Details

### 1 · Dinucleotide Perplexity

For a window of length *w* centred at position *i*, REGPLEX calculates the **Shannon perplexity** over the 16 possible dinucleotide frequencies:

```
H  = -Σ p(xy) · log₂ p(xy)
PP = 2^H        (perplexity, range 1–16)
```

Low perplexity → skewed dinucleotide usage → promoter-like composition.

### 2 · Kadane-based Region Caller

Rather than applying a fixed threshold, REGPLEX uses a **modified minimum-subarray (Kadane's) algorithm** on the perplexity signal to find the contiguous windows with the globally lowest cumulative perplexity — delivering results that adapt to each sequence.

### 3 · Non-B DNA Motif Scanning

Eight classes of Non-B DNA structures are scanned with compiled regexes:

| Key | Structure | Regex basis |
|-----|-----------|-------------|
| `G4` | G-Quadruplex | 4 × G≥3 runs with 1–7 nt loops |
| `iMotif` | i-Motif | 4 × C≥3 runs |
| `ZDNA` | Z-DNA | Alternating CG / GC / CA / TG ≥4 units |
| `eGZDNA` | Expanded G/Z-DNA | CGG / GGC trinucleotide repeats ≥4 |
| `Triplex` | Triplex-forming | AG mirror repeats ≥10 bp |
| `STR` | Short Tandem Repeat | 1–6 bp unit × ≥4 copies |
| `DirectRepeat` | Direct Repeat | 4–10 bp unit with ≤10 bp gap |
| `PolyAT` | PolyA/T | Homopolymeric A or T run ≥7 bp |

---

## 📈 Analysis Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Perplexity window | 10 bp | 5–50 bp | Grain of entropy calculation |
| Analysis window | 100 bp | 50–500 bp | Size of reported candidate regions |
| Percentile threshold | 25th | 10–50 | Stringency of region selection |

---

## 📦 Example Data

The `example_data/` directory ships with 11 sequences spanning diverse organisms:

- *Escherichia coli* K-12 (`ecoli.fna`)  
- *Mycobacterium tuberculosis* H37Rv (`mtb.fasta`)  
- *Helicobacter pylori* (`hpylori.fna`)  
- *Staphylococcus aureus* (`saureus.fna`)  
- *Streptococcus pneumoniae* (`Streptococcus pneumoniae.fna`)  
- *Saccharomyces cerevisiae* (`Scer.fna`)  
- *Buchnera aphidicola* · *Candidatus Carsonella ruddii* · *Cellulomonas shaoxiangyii* · *Miltoncostaea marina*  
- Generic FASTA demo (`example.fasta`)

Load any of these from the **Examples** tab with one click.

---

## 📤 Export Formats

| Format | Content |
|--------|---------|
| **JSON** | Full session — regions, scores, motifs, parameters |
| **CSV** | Tabular region results (position, score, motifs) |
| **PDF** | Publication-ready report with figures and statistics |

---

## 🛠 Development

```bash
# Install dev dependencies
pip install -r requirements.txt pytest flake8

# Lint (strict — syntax + undefined names only)
flake8 streamlit_app.py --count --select=E9,F63,F7,F82 --show-source

# Run app in headless mode for smoke-test
streamlit run streamlit_app.py --server.headless true --server.port 8501
```

CI runs on Python 3.8 · 3.9 · 3.10 · 3.11 via GitHub Actions (see `.github/workflows/ci-cd.yml`).

---

## 📚 Citation

If REGPLEX contributes to your research, please cite:

```bibtex
@software{regplex2025,
  title   = {{REGPLEX}: Regulatory Domain Discovery Using DNA Sequence Perplexity},
  author  = {Yella, Venkata Rajesh},
  year    = {2024},
  url     = {https://github.com/VRYella/PerCALL},
  license = {MIT}
}
```

---

## 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ using <a href="https://streamlit.io">Streamlit</a> · <a href="https://numpy.org">NumPy</a> · <a href="https://plotly.com">Plotly</a>
</div>
