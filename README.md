# REGPLEX

<p align="center">
  <img src="regplexlogo.png" alt="REGPLEX logo" width="340" />
</p>

<p align="center"><strong>Hierarchical Perplexity Ensemble for Explainable Regulatory Valley Discovery</strong></p>

## Scientific Motivation

REGPLEX v10 models genomic complexity as three independent information-theoretic observers:

- **Mononucleotide perplexity**: composition and GC/AT structure
- **Dinucleotide perplexity**: local dependency and structural organization (primary layer)
- **Trinucleotide perplexity**: higher-order sequence grammar

The method is training-free and uses no learned model weights.

## Algorithm

1. Compute mono/di/tri perplexity **once**.
2. Build multi-scale landscapes (default: 25, 50, 100, 200, 400 bp) using rolling mean or median.
3. At each scale, compute three-window local perplexity contrast (upstream/spacer/candidate/spacer/downstream).
4. Normalize each scale LPC within each layer (robust z-score or percentile), then median-combine to layer consensus.
5. Ensemble layer consensuses to final **ConsensusLPC** (median or trimmed mean).
6. Run bounded Kadane core detection once on ConsensusLPC, expand boundaries while signal stays positive, merge overlaps.
7. Report valley support and intermediate metrics for full explainability.

## Illustrated Workflow

<svg width="980" height="160" viewBox="0 0 980 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="REGPLEX workflow">
  <title>REGPLEX workflow diagram</title>
  <desc>Workflow: DNA to mono, di and tri perplexity; multi-scale analysis; layer consensus; ensemble ConsensusLPC; Kadane-driven valleys.</desc>
  <defs>
    <style>
      .n{fill:#fff;stroke:#1E3A8A;stroke-width:1.5;rx:10}
      .t{font:600 12px Arial,sans-serif;fill:#0f172a}
      .a{stroke:#0F766E;stroke-width:2;marker-end:url(#m)}
    </style>
    <marker id="m" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#0F766E"/></marker>
  </defs>
  <rect class="n" x="20" y="50" width="110" height="48"/><text class="t" x="44" y="79">DNA</text>
  <rect class="n" x="160" y="16" width="120" height="34"/><text class="t" x="174" y="37">Mono</text>
  <rect class="n" x="160" y="62" width="120" height="34"/><text class="t" x="178" y="83">Di</text>
  <rect class="n" x="160" y="108" width="120" height="34"/><text class="t" x="175" y="129">Tri</text>
  <rect class="n" x="310" y="50" width="140" height="48"/><text class="t" x="326" y="79">Multi-scale</text>
  <rect class="n" x="480" y="50" width="150" height="48"/><text class="t" x="507" y="79">Layer consensus</text>
  <rect class="n" x="660" y="50" width="130" height="48"/><text class="t" x="679" y="79">ConsensusLPC</text>
  <rect class="n" x="820" y="50" width="140" height="48"/><text class="t" x="840" y="79">Kadane → Valleys</text>
  <line class="a" x1="130" y1="74" x2="160" y2="74"/>
  <line class="a" x1="280" y1="74" x2="310" y2="74"/>
  <line class="a" x1="450" y1="74" x2="480" y2="74"/>
  <line class="a" x1="630" y1="74" x2="660" y2="74"/>
  <line class="a" x1="790" y1="74" x2="820" y2="74"/>
</svg>

## Installation

```bash
git clone https://github.com/VRYella/PerCALL.git
cd PerCALL
pip install -r requirements.txt
```

## Quick Start

### Web interface

```bash
streamlit run app.py
```

The Motif section includes built-in non-B DNA and promoter motif boxes by default, and you can add extra custom Regex/IUPAC motifs for additional valley annotation.

### CLI

```bash
python regplex_core.py examples/ecoli.fasta --out regplex_v10_valleys.csv
python regplex_core.py examples/ecoli.fasta --scales 25,50,100,200,400 --landscape-method median --normalization-method robust_z --ensemble-method median
```

## Notebook

- `REGPLEX_Local.ipynb` contains a 15-section scientific walkthrough aligned to v10.

## Examples

- `examples/ecoli.fasta`
- `examples/human_promoters.fasta`

## Outputs

Core valley fields include:

- Coordinates and length
- `Contrast`, `Persistence`, `Area`
- `MonoSupport`, `DiSupport`, `TriSupport`
- `ScaleSupport`, `LayerSupport`, `OverallSupport`
- `ValleyScore`, `ValleyScoreNormalized`
- motif annotations and sequence extraction

Formats: CSV, Excel, BED, GFF/GFF3, FASTA, JSON.

## Figures

The UI generates publication-ready Plotly figures:

1. Layer perplexity profiles (mono/di/tri)
2. Per-layer multi-scale landscapes
3. Layer consensus LPC
4. Ensemble ConsensusLPC
5. Scale/layer support heatmap
6. Kadane core with expanded valleys
7. Valley ranking
8. Motif architecture
9. Workflow diagram

## Performance

- Fully NumPy/Pandas-based computation
- Prefix-sum accelerated rolling means
- Single-pass layer signal computation
- No repeated raw perplexity calculations

## Citation

If you use REGPLEX in research, cite this repository and include algorithm version (`v10`) in methods.

## License

MIT

## Contributing

Contributions are welcome. Please open an issue with:

- biological question
- expected behavior
- reproducible FASTA example

## Screenshots

Run the Streamlit app and capture:

- Home hero + workflow
- Analysis controls
- Results tabs (ConsensusLPC, support, ranking)
- About scientific workflow page
