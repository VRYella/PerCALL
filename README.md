# CisPerplexity: Advanced Promoter Prediction Tool

A professional Streamlit web application for predicting promoter regions in DNA sequences using an integrated approach that combines dinucleotide perplexity analysis, structural features, and promoter motif detection.

## 🌟 Key Features

### 🧬 Integrated Algorithm
- **Dinucleotide Perplexity Analysis**: Calculates perplexity using sliding windows to identify low-complexity regions (promoters typically have lower perplexity)
- **Kadane's Algorithm Integration**: Advanced optimization for finding optimal low-perplexity regions
- **Structural Features**: Encodes sequences using conformational and physicochemical properties from comprehensive structural dictionary
- **Motif Detection**: Identifies 17+ known promoter motifs including TATA-box, Initiator elements, and other regulatory sequences

### 📊 Professional Web Interface
- **Navigation System**: Clean interface with Analysis, About, Examples, and Advanced sections
- **Multiple Input Methods**: Paste sequences, upload FASTA files, or use curated example sequences
- **Real-time Analysis**: Live perplexity calculation and promoter prediction with progress indicators
- **Interactive Visualizations**: Professional Plotly charts showing perplexity, GC content, and motif frequency
- **Enhanced Results Display**: Detailed metrics, confidence scores, and downloadable analysis reports
- **Responsive Design**: Works seamlessly across desktop and mobile devices

### 🔬 Analysis Components
1. **Perplexity Calculator**: Sliding window dinucleotide perplexity using entropy calculations
2. **Structural Feature Encoder**: DNA property encoding using comprehensive 150+ parameter dictionary
3. **Motif Detector**: Pattern matching for 17+ known promoter motifs
4. **Kadane's Algorithm**: Optimal low-perplexity region detection using modified maximum subarray algorithm
5. **Confidence Scoring**: Multi-factor confidence assessment based on perplexity, motifs, and region characteristics

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/VRYella/CisPerplexity.git
cd CisPerplexity

# Run with Docker Compose
docker-compose up

# Or build and run manually
docker build -t cisperplexity .
docker run -p 8501:8501 cisperplexity
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
2. **Select Input Method**: Choose between pasting sequence, uploading FASTA, or using examples
3. **Adjust Parameters**: Configure analysis and perplexity window sizes in the sidebar
4. **Run Analysis**: Click the "🔬 Analyze Sequence" button
5. **Review Results**: Examine predicted promoter regions, visualizations, and detected motifs
6. **Download Data**: Export results in JSON format for further analysis

### Advanced Features
- **Parameter Tuning**: Adjust window sizes and thresholds for different sequence types
- **Example Library**: Explore curated examples with expected results
- **Comprehensive Documentation**: Access detailed algorithm explanations in the About section

## 🔧 Algorithm Details

### Dinucleotide Perplexity Analysis
- **Method**: Sliding window entropy calculation
- **Formula**: Perplexity = 2^H, where H = -Σ(p × log₂(p))
- **Innovation**: Kadane's Algorithm adaptation for optimal region detection
- **Configuration**: Separate windows for calculation (5-50 bp) and analysis (50-500 bp)

### Structural Features
The encoding dictionary includes 150+ parameters:
- **Conformational**: Twist, Roll, Slide, Tilt, Wedge, Major Groove Depth, etc.
- **Physicochemical**: Free energy, Melting temperature, Stiffness, Bendability
- **Letter-based**: GC content, Purine content, Keto content, etc.

### Promoter Motifs
Detects patterns for:
- **Core Elements**: TATA-box, Initiator elements (Human/Drosophila)
- **Regulatory Elements**: BREu, BREd, TCT, DPE, MTE
- **Structural Motifs**: G-quadruplex, i-motif formations
- **Transcription Factor Sites**: Sp1, CAAT-box, and others

### Kadane's Algorithm Integration
- **Purpose**: Finds globally optimal low-perplexity regions
- **Advantage**: Superior to threshold-based methods
- **Implementation**: Modified for minimum sum detection
- **Output**: Multiple non-overlapping candidate regions with confidence scores

## 📁 File Structure

```
CisPerplexity/
├── streamlit_app.py              # Main Streamlit application
├── selected_encoding_dict.json   # Structural feature dictionary
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container configuration
├── docker-compose.yml           # Docker Compose setup
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── README.md                    # This file
└── KADANE_IMPLEMENTATION.md     # Detailed algorithm documentation
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
- **Perplexity Window**: 5-50 bp (default: 10 bp)
- **Analysis Window**: 50-500 bp (default: 100 bp)  
- **Threshold**: 10-50th percentile (default: 25th percentile)

### Input Requirements
- **Sequence Types**: DNA sequences (A, T, G, C only)
- **Length**: Minimum 50 bp, recommended <10,000 bp
- **Formats**: Plain text, FASTA files

## 📊 Output Features
- **Predicted Regions**: Start/end positions with confidence scores
- **Method Attribution**: Kadane's algorithm vs. threshold-based detection
- **Structural Analysis**: Mean and standard deviation for each property
- **Motif Detection**: Region-specific and sequence-wide motif analysis
- **Interactive Visualizations**: Perplexity plots, GC content analysis, motif frequency charts
- **Downloadable Results**: JSON format for further analysis

## 🔬 Research Background

This tool implements algorithms from the CisPerplexity research project, which explores the relationship between DNA dinucleotide perplexity and promoter regions. The hypothesis is that promoter regions exhibit lower perplexity compared to neighboring genomic regions due to their regulatory constraints and sequence composition.

## 📚 Citation

If you use CisPerplexity in your research, please cite:

```
CisPerplexity: An Integrated Computational Tool for Promoter Prediction Using 
Dinucleotide Perplexity Analysis and Kadane's Algorithm. (2024)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues and enhancement requests.

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/VRYella/CisPerplexity/issues)
- **Documentation**: See `KADANE_IMPLEMENTATION.md` for detailed algorithm information
- **Examples**: Use the built-in Examples section for tutorials and use cases

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Streamlit, NumPy, Plotly, and modern bioinformatics tools.**