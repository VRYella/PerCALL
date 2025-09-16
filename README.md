# NBDFinder: Advanced Non-B DNA Motif Detection System

A comprehensive computational tool for detecting and analyzing non-B DNA structures across 11 distinct motif classes, including the newly integrated **A-philic DNA** detection system with tetranucleotide log2 odds scoring.

## 🌟 Key Features

### 🧬 11-Class Motif Detection System
- **Class 1**: Curved DNA - Intrinsic DNA curvature patterns
- **Class 2**: Slipped DNA - Direct repeats and STR sequences  
- **Class 3**: Cruciform DNA - Inverted repeat structures
- **Class 4**: R-loop - RNA-DNA hybrid formation sites
- **Class 5**: Triplex - Triple-helix DNA structures
- **Class 6**: G-Quadruplex - G4 and variant formations
- **Class 7**: i-Motif - C-rich quadruplex structures
- **Class 8**: Z-DNA - Left-handed DNA conformations
- **Class 9**: 🆕 **A-philic DNA** - A-tract-favoring protein binding sites
- **Class 10**: Hybrid - Multi-class overlapping regions
- **Class 11**: Non-B DNA Clusters - Hotspot regions

### 🆕 A-philic DNA Detection (Class 9)

A-philic DNA represents sequences with high affinity for A-tract formation and protein-DNA interactions. Our implementation uses advanced tetranucleotide log2 odds scoring for precise detection.

**Key Features:**
- **Tetranucleotide Scoring**: Uses 18 key tetranucleotides with literature-based weights
- **Dual Classification**: Distinguishes High Confidence (≥2.0 log2 odds) and Moderate A-philic motifs
- **Optimized Windows**: 10-20bp sliding windows for comprehensive analysis
- **Scientific Validation**: Based on established computational biology methods

**Scientific References:**
- Vinogradov (2003) Bioinformatics - Tetranucleotide analysis methodology
- Bolshoy et al. (1991) PNAS - A-tract structural properties
- Rohs et al. (2009) Nature - Protein-DNA interaction patterns

### 📊 Professional Web Interface
- **Navigation System**: Clean interface with Analysis, About, Examples, and Advanced sections
- **Multiple Input Methods**: Paste sequences, upload FASTA files, or use curated example sequences
- **Real-time Analysis**: Live motif detection with progress indicators
- **Interactive Visualizations**: Professional Plotly charts showing motif distribution and A-philic scoring
- **Enhanced Results Display**: Detailed metrics, confidence scores, and downloadable analysis reports
- **A-philic DNA Visualization**: Custom color scheme (#E6B8F7) and specialized analysis views

### 🔬 Analysis Components
1. **A-philic DNA Scanner**: Tetranucleotide log2 odds scoring with dual classification system
2. **Parallel Orchestrator**: Concurrent detection across all 11 motif classes  
3. **Configuration System**: Class-specific parameters and quality thresholding
4. **Quality Filtering**: A-philic-specific filters (min_score: 10.0 for orchestrator, 1.5 for tetranucleotide scores)
5. **Standardized Output**: NBDFinder-compatible result format with comprehensive metadata

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/VRYella/CisPerplexity.git
cd CisPerplexity

# Install dependencies
pip install -r requirements.txt

# Run the NBDFinder web application
streamlit run app.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t nbdfinder .
docker run -p 8501:8501 nbdfinder
```

### Command Line Usage
```python
# Basic A-philic DNA detection
from motifs.a_philic_dna import find_a_philic_dna

sequence = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAA"
motifs = find_a_philic_dna(sequence, "my_sequence")

for motif in motifs:
    print(f"{motif['Subclass']}: {motif['Start']}-{motif['End']} "
          f"(score: {motif['Score']}, length: {motif['Length']}bp)")

# Full NBDFinder analysis
from all_motifs_refactored import analyze_sequence_nbd_finder

results = analyze_sequence_nbd_finder(sequence, "my_sequence")
print(f"Total motifs detected: {results['summary_stats']['total_motifs']}")
print(f"A-philic motifs: {len(results['motifs_by_class'].get(9, []))}")
```

### Local Installation

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the Application:**
```bash
streamlit run streamlit_app.py
```

3. **Access the Interface:**
   Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### Basic Analysis
1. **Navigate to Analysis**: Use the sidebar navigation to access the analysis interface
2. **Select Motif Classes**: Choose which of the 11 motif classes to analyze (default: A-philic DNA)
3. **Select Input Method**: Choose between pasting sequence, uploading FASTA, or using examples
4. **Configure Parameters**: Enable parallel processing and A-philic DNA specific options
5. **Run Analysis**: Click the "🔬 Analyze Sequence" button
6. **Review Results**: Examine detected motifs, visualizations, and A-philic DNA analysis
7. **Download Data**: Export results in JSON format for further analysis

### Advanced Features
- **Class-Specific Analysis**: Focus on specific motif classes or run comprehensive 11-class detection
- **A-philic DNA Options**: Specialized visualization and analysis for tetranucleotide scoring
- **Example Library**: Explore curated examples with A-philic rich sequences
- **Parallel Processing**: Enable concurrent detection across all motif classes
- **Quality Thresholding**: Configurable score filters for each motif class

## 🔧 Algorithm Details

### A-philic DNA Detection (Class 9)
- **Method**: Tetranucleotide log2 odds scoring using 18 key tetranucleotides
- **Scoring**: Literature-based weights from Vinogradov (2003) methodology
- **Classification**: Dual system - High Confidence (≥2.0 log2 odds) and Moderate (1.5-2.0)
- **Windows**: 10-20bp sliding windows with configurable parameters
- **Thresholding**: Strong tetranucleotide filtering for robust detection

### NBDFinder Architecture
- **Parallel Processing**: Concurrent detection across all 11 motif classes
- **Orchestrator**: Centralized coordination with class-specific quality filters
- **Configuration**: Modular parameter system for each motif class
- **Output Format**: Standardized NBDFinder-compatible results with comprehensive metadata

### 11-Class Detection System
Each motif class has specific detection algorithms and parameters:
1. **Curved DNA**: Curvature analysis using conformational properties
2. **Slipped DNA**: Direct repeat and STR sequence detection
3. **Cruciform DNA**: Inverted repeat structure analysis
4. **R-loop**: RNA-DNA hybrid formation propensity
5. **Triplex**: Triple-helix DNA structure detection
6. **G-Quadruplex**: G4 formation scoring and variant detection
7. **i-Motif**: C-rich quadruplex structure analysis
8. **Z-DNA**: Left-handed DNA conformation propensity
9. **A-philic DNA**: Tetranucleotide scoring for A-tract affinity
10. **Hybrid**: Multi-class overlap analysis
11. **Non-B DNA Clusters**: Hotspot region density analysis

## 📁 File Structure

```
CisPerplexity/
├── app.py                        # Main NBDFinder Streamlit application
├── streamlit_app.py              # Legacy CisPerplexity application (preserved)
├── all_motifs_refactored.py      # NBDFinder orchestrator with 11-class system
├── classification_config.py      # Configuration system for all motif classes
├── motifs/                       # Motif detection modules
│   ├── __init__.py
│   └── a_philic_dna.py          # A-philic DNA scanner (Class 9)
├── selected_encoding_dict.json   # Structural feature dictionary
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container configuration
├── docker-compose.yml           # Docker Compose setup
├── test_nbdfinder_a_philic.py   # Comprehensive A-philic DNA tests
├── test_cisperplexity.py        # Legacy CisPerplexity tests
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── README.md                    # This file
└── KADANE_IMPLEMENTATION.md     # Legacy algorithm documentation
```

## 🧠 Scientific Background

### The CisPerplexity Hypothesis
Promoter regions exhibit distinct sequence characteristics:
- **Lower Perplexity**: Due to regulatory constraints and specific composition patterns
- **Structural Properties**: Unique conformational features affecting DNA-protein interactions
- **Motif Signatures**: Conserved regulatory sequences for transcription factor binding

### Performance Characteristics
- **Time Complexity**: O(n²) for region finding, O(n) for perplexity calculation
- **Memory Efficiency**: Optimized numpy operations
- **Scalability**: Handles sequences up to 10,000+ bp efficiently
- **Accuracy**: High confidence predictions with multi-factor validation

## 🛠️ Configuration

### Analysis Parameters
- **Motif Classes**: Select from 11 distinct non-B DNA motif classes
- **Parallel Processing**: Enable concurrent detection across all classes  
- **A-philic DNA Options**: Specialized parameters for tetranucleotide scoring
- **Quality Thresholding**: Class-specific score filters and length constraints

### A-philic DNA Configuration
- **Length Constraints**: S_min: 10bp, S_max: 50bp (extendable to 200bp for A-tracts)
- **Scoring Method**: tetranucleotide_log2_odds with 18 key tetranucleotides
- **Thresholds**: High confidence ≥2.0, Moderate confidence 1.5-2.0 log2 odds
- **Window Sizes**: 10-20bp sliding windows for comprehensive analysis

### Input Requirements
- **Sequence Types**: DNA sequences (A, T, G, C only)
- **Length**: Minimum 10 bp for A-philic DNA, recommended <10,000 bp for full analysis
- **Formats**: Plain text, FASTA files

## 📊 Output Features
- **Motif Classification**: 11-class system with standardized NBDFinder format
- **A-philic DNA Analysis**: Dual classification with tetranucleotide scoring details
- **Interactive Visualizations**: Motif distribution plots with class-specific color coding (#E6B8F7 for A-philic DNA)
- **Comprehensive Metrics**: Total motifs, active classes, confidence breakdowns
- **Quality Filtering**: Class-specific score validation and length constraints
- **Downloadable Results**: JSON format with complete analysis metadata

## 🔬 Research Background

This tool implements the NBDFinder system for comprehensive non-B DNA motif detection, featuring the newly integrated A-philic DNA analysis system. The A-philic DNA detection is based on established tetranucleotide analysis methods and A-tract structural properties research.

### Scientific Foundation
- **A-philic DNA**: Based on tetranucleotide log2 odds scoring methodology
- **Tetranucleotide Analysis**: Implements Vinogradov (2003) computational methods
- **A-tract Properties**: Incorporates findings from Bolshoy et al. (1991) on A-tract structural characteristics
- **Protein-DNA Interactions**: Utilizes insights from Rohs et al. (2009) on sequence-dependent protein binding

## 📚 Citation

If you use NBDFinder or the A-philic DNA detection system in your research, please cite:

```
NBDFinder: A Comprehensive Non-B DNA Motif Detection System with 
Integrated A-philic DNA Analysis Using Tetranucleotide Log2 Odds Scoring. (2024)
```

### Key References
- Vinogradov, A.E. (2003). DNA helix: the importance of being GC-rich. *Bioinformatics*, 19(16), 2049-2052.
- Bolshoy, A., et al. (1991). Curved DNA without A-A: experimental estimation of all 16 DNA wedge angles. *PNAS*, 88(6), 2312-2316.
- Rohs, R., et al. (2009). The role of DNA shape in protein-DNA recognition. *Nature*, 461(7268), 1248-1253.

## 🤝 Contributing

We welcome contributions to the NBDFinder system! Please see our contributing guidelines and feel free to submit issues and enhancement requests.

### Development Areas
- **Additional Motif Classes**: Implement detection algorithms for Classes 1-8 and 10-11
- **Enhanced A-philic DNA**: Expand tetranucleotide scoring or add additional A-tract features
- **Performance Optimization**: Improve parallel processing and memory efficiency
- **Visualization**: Enhance motif distribution plots and add new analysis views
- **Documentation**: Expand scientific documentation and usage examples

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/VRYella/CisPerplexity/issues)
- **Documentation**: See detailed algorithm information in the About section of the web app
- **Examples**: Use the built-in Examples section for tutorials and A-philic DNA use cases
- **Testing**: Run comprehensive tests with `pytest test_nbdfinder_a_philic.py -v`

**Built with ❤️ using Streamlit, NumPy, Plotly, and modern bioinformatics tools for advanced non-B DNA structure detection.**