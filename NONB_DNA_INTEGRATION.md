# Non-B DNA Integration Guide

## Overview

CisPerplexity now includes comprehensive Non-B DNA structure detection capabilities integrated from the NonBDNAFinder repository. This integration allows users to identify not only traditional promoter motifs but also various Non-B DNA structures that may play regulatory roles.

## What are Non-B DNA Structures?

Non-B DNA structures are alternative DNA conformations that differ from the standard right-handed B-form double helix. These structures can form under specific sequence and environmental conditions and are important for:

- **Gene Regulation**: Affecting transcription factor binding and gene expression
- **DNA Packaging**: Influencing chromatin structure and nucleosome positioning
- **Genomic Stability**: Involved in recombination, replication, and DNA repair
- **Disease Association**: Linked to mutational hotspots and genetic instability

## Detected Non-B DNA Classes

### 1. A-philic DNA
**Description**: Regions with propensity to adopt A-form DNA conformation

**Biological Significance**: A-form DNA is more compact and has different groove geometry, affecting protein-DNA interactions

**Subclasses**:
- High Confidence A-form
- Moderate A-form Preference
- Weak A-form Signature

**Detection Method**: Structure-informed propensity scoring based on experimental crystal structure data

### 2. Curved DNA
**Description**: DNA bending caused by phased A-tracts

**Biological Significance**: Affects nucleosome positioning, transcription factor binding, and DNA packaging

**Subclasses**:
- High Quality Local Curvature
- Strong Local Curvature
- Local Curvature
- High Quality Global Curvature
- Strong Global Curvature
- Global Curvature
- Strong Directional Curvature
- Directional Curvature

**Detection Method**: Phased A-tract analysis with quality scoring

### 3. G-Quadruplex (G4)
**Description**: Four-stranded structures formed by guanine-rich sequences

**Biological Significance**: Found in telomeres, promoters, and oncogene regulatory regions; involved in transcription regulation

**Subclasses**:
- Telomeric G4
- Canonical G4
- Extended G4

**Detection Method**: Pattern matching with thermodynamic scoring

### 4. i-Motif
**Description**: Four-stranded structures formed by cytosine-rich sequences (complementary to G-quadruplexes)

**Biological Significance**: pH-dependent structures involved in gene regulation

**Subclasses**:
- Canonical i-Motif
- HuR AC Motif

**Detection Method**: C-tract content analysis with loop entropy scoring

### 5. Z-DNA
**Description**: Left-handed helix formed in alternating purine-pyrimidine sequences

**Biological Significance**: Associated with transcription activation, genetic instability, and immune responses

**Subclasses**:
- Canonical Z-DNA

**Detection Method**: Pattern matching for alternating purine-pyrimidine sequences

### 6. Slipped DNA
**Description**: Structures formed by direct repeats, including microsatellites

**Biological Significance**: Associated with DNA replication errors and trinucleotide repeat expansion diseases

**Subclasses**:
- Direct Repeat
- Short Tandem Repeats (STRs)

**Detection Method**: Direct repeat detection with thermodynamic stability scoring

### 7. Cruciform
**Description**: Four-way junction structures formed by inverted repeats (palindromes)

**Biological Significance**: Sites of recombination, replication fork stalling, and chromosome fragility

**Subclasses**:
- Inverted Repeat
- Hairpin structures

**Detection Method**: Palindrome detection with stem-loop stability analysis

### 8. Triplex DNA
**Description**: Three-stranded DNA structures formed by mirror repeats

**Biological Significance**: Involved in gene regulation and recombination

**Subclasses**:
- Mirror Repeat
- Sticky DNA

**Detection Method**: Mirror repeat detection with purity scoring

## Using Non-B DNA Detection

### In the Web Interface

1. **Input your sequence** using any of the available methods (paste, upload FASTA, or use examples)

2. **Run the analysis** by clicking "Analyze Sequence"

3. **View Non-B DNA results** in the dedicated "Non-B DNA Structures" section, which includes:
   - Summary metrics (total motifs, classes detected, average scores)
   - Expandable sections for each motif class
   - Distribution visualizations
   - Genomic position overview

### Programmatic Usage

```python
from nonb_motif_detector import detect_all_nonb_motifs

# Detect all Non-B DNA structures
sequence = "GGGGGGGGGGCCCCCCCCCTAGGGGGGGGGCCCCCCCCCT"
nonb_motifs = detect_all_nonb_motifs(sequence)

# Access results by class
for motif_class, motifs in nonb_motifs.items():
    print(f"{motif_class}: {len(motifs)} detected")
    for motif in motifs:
        print(f"  Position: {motif['Start']}-{motif['End']}")
        print(f"  Score: {motif['Score']}")
        print(f"  Subclass: {motif['Subclass']}")
```

### Integrated Analysis

```python
from streamlit_app import PromoterPredictor

# Create predictor with Non-B DNA detection enabled
predictor = PromoterPredictor()

# Run full analysis
results = predictor.predict_promoters(sequence)

# Access Non-B DNA results
nonb_motifs = results['nonb_motifs']
print(f"Total Non-B structures: {sum(len(m) for m in nonb_motifs.values())}")
```

## Interpreting Results

### Score Interpretation

Non-B DNA motifs are scored on a scale typically ranging from 1.0 to 3.0:

- **2.5 - 3.0**: Very high confidence - strong structural propensity
- **2.0 - 2.5**: High confidence - clear structural signature
- **1.5 - 2.0**: Moderate confidence - detectable structure
- **1.0 - 1.5**: Low confidence - marginal propensity

### Biological Context

When interpreting Non-B DNA structures in promoter regions:

1. **Regulatory Potential**: G-quadruplexes and Z-DNA in promoters often correlate with transcriptional regulation
2. **Structural Constraints**: Curved DNA may affect nucleosome positioning and accessibility
3. **Evolutionary Conservation**: Conserved Non-B structures suggest functional importance
4. **Disease Association**: Slipped DNA and cruciforms may indicate fragile sites

## Output Formats

Non-B DNA detection results are included in all output formats:

- **JSON**: Complete structured data with all motif details
- **Interactive Visualizations**: 
  - Class distribution bar charts
  - Position-based scatter plots
  - Genomic distribution timeline
- **Expandable Tables**: Per-class detailed motif listings

## Performance Considerations

- **Sequence Length**: Detection works efficiently on sequences up to 10,000+ bp
- **Computational Cost**: Non-B DNA detection adds minimal overhead (~10-20% increase)
- **Memory Usage**: Scales linearly with sequence length

## References

### Scientific Background

1. **A-philic DNA**: Vinogradov 2003 (Bioinformatics), Bolshoy et al. 1991 (PNAS)
2. **Curved DNA**: Crothers & Drak 1992 (Nature), Goodsell & Dickerson 1994 (PNAS)
3. **G-Quadruplex**: Parkinson 2002, Huppert & Balasubramanian 2005, Neidle 2019
4. **i-Motif**: Zeraati et al. 2018 (Nature Chemistry)
5. **Z-DNA**: Ho et al. 1986, Wang et al. 2010, Herbert 2019 (Nature Communications)
6. **Non-B DB**: Cer et al. 2013 (Nucleic Acids Research)

### Source Integration

This integration is based on the NonBDNAFinder repository:
- GitHub: [VRYella/NonBDNAFinder](https://github.com/VRYella/NonBDNAFinder)
- Author: Dr. Venkata Rajesh Yella

## Troubleshooting

### Common Issues

1. **No Non-B structures detected**: 
   - Check sequence length (minimum ~50 bp recommended)
   - Some structures are rare and may not be present

2. **High number of detections**:
   - Certain structures like cruciforms are common in repetitive sequences
   - Use score filtering to focus on high-confidence predictions

3. **Performance issues**:
   - For very long sequences (>50,000 bp), consider chunking
   - Disable Non-B detection if not needed

## Future Enhancements

Planned improvements include:

- Additional Non-B DNA classes (R-loops with full RLFS model)
- Thermodynamic scoring for all structures
- Hybrid motif detection (overlapping structures)
- Batch processing for multiple sequences
- Species-specific pattern libraries

## Contact

For questions or issues related to Non-B DNA detection:
- Open an issue on GitHub
- Contact: yvrajesh_bt@kluniversity.in
