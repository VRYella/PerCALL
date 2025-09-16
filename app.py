"""
NBDFinder Streamlit App - Enhanced with A-philic DNA Detection

This is the enhanced Streamlit web application for the NBDFinder system,
featuring the complete 11-class motif detection system including A-philic DNA (Class 9).

Features:
- Visual integration with A-philic DNA motif visualization (#E6B8F7 color scheme)
- Updated documentation for 11-class system
- Comprehensive A-philic DNA explanation with scientific references
- Integration with NBDFinder orchestrator and configuration system
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import NBDFinder modules
from all_motifs_refactored import analyze_sequence_nbd_finder
from classification_config import NBDFinderConfig
from motifs.a_philic_dna import find_a_philic_dna

# Page configuration
st.set_page_config(
    page_title="NBDFinder: Non-B DNA Motif Detection System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with A-philic DNA color scheme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .a-philic-highlight {
        background-color: #E6B8F7;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #B87ED3;
    }
    .motif-class-9 {
        color: #E6B8F7;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class NBDFinderApp:
    """Main NBDFinder Streamlit application class"""
    
    def __init__(self):
        """Initialize the NBDFinder application"""
        self.config = NBDFinderConfig()
        
        # Define motif color scheme including A-philic DNA
        self.motif_colors = {
            1: '#FF6B6B',    # Curved DNA - Red
            2: '#4ECDC4',    # Slipped DNA - Teal  
            3: '#45B7D1',    # Cruciform DNA - Blue
            4: '#96CEB4',    # R-loop - Green
            5: '#FFEAA7',    # Triplex - Yellow
            6: '#DDA0DD',    # G-Quadruplex - Plum
            7: '#98D8C8',    # i-Motif - Mint
            8: '#F7DC6F',    # Z-DNA - Light Yellow
            9: '#E6B8F7',    # A-philic DNA - Purple (NEW)
            10: '#AED6F1',   # Hybrid - Light Blue  
            11: '#F8C471'    # Non-B DNA Clusters - Orange
        }
        
        # Updated motif order and descriptions for 11-class system
        self.motif_order = [
            (1, 'Curved DNA', 'Intrinsic DNA curvature patterns'),
            (2, 'Slipped DNA', 'Direct repeats and STR sequences'),
            (3, 'Cruciform DNA', 'Inverted repeat structures'),
            (4, 'R-loop', 'RNA-DNA hybrid formation sites'),
            (5, 'Triplex', 'Triple-helix DNA structures'),
            (6, 'G-Quadruplex', 'G4 and variant formations'),
            (7, 'i-Motif', 'C-rich quadruplex structures'),
            (8, 'Z-DNA', 'Left-handed DNA conformations'),
            (9, 'A-philic DNA', 'A-tract-favoring protein binding sites'),
            (10, 'Hybrid', 'Multi-class overlapping regions'),
            (11, 'Non-B DNA Clusters', 'Hotspot regions')
        ]
    
    def render_sidebar(self):
        """Render the sidebar with navigation and parameters"""
        st.sidebar.markdown("## 🧬 NBDFinder System")
        st.sidebar.markdown("**11-Class Non-B DNA Motif Detection**")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Analysis", "About", "Examples", "Advanced"]
        )
        
        if page == "Analysis":
            self.render_analysis_page()
        elif page == "About":
            self.render_about_page()
        elif page == "Examples":
            self.render_examples_page()
        elif page == "Advanced":
            self.render_advanced_page()
    
    def render_analysis_page(self):
        """Render the main analysis page"""
        st.markdown('<h1 class="main-header">🧬 NBDFinder: Non-B DNA Motif Detection</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Advanced 11-Class Motif Detection System
        
        NBDFinder detects 11 distinct classes of non-B DNA motifs, including the newly integrated 
        **A-philic DNA** detection system with tetranucleotide log2 odds scoring.
        """)
        
        # Analysis parameters in sidebar
        st.sidebar.markdown("### Analysis Parameters")
        
        # Motif class selection
        all_classes = [cls for cls, _, _ in self.motif_order]
        selected_classes = st.sidebar.multiselect(
            "Select Motif Classes",
            all_classes,
            default=[9],  # Default to A-philic DNA
            format_func=lambda x: f"Class {x}: {self.config.get_class_name(x)}"
        )
        
        # Processing options
        parallel_processing = st.sidebar.checkbox("Parallel Processing", value=True)
        
        # A-philic DNA specific options
        if 9 in selected_classes:
            st.sidebar.markdown("### A-philic DNA Options")
            show_a_philic_details = st.sidebar.checkbox("Show A-philic Details", value=True)
        
        # Input methods
        st.markdown('<h2 class="subheader">📝 Input Sequence</h2>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Choose input method:",
            ["Paste sequence", "Upload FASTA file", "Use example"]
        )
        
        sequence = self.get_sequence_input(input_method)
        
        if sequence and st.button("🔬 Analyze Sequence", type="primary"):
            self.run_analysis(sequence, selected_classes, parallel_processing)
    
    def get_sequence_input(self, method: str) -> str:
        """Get sequence input based on selected method"""
        sequence = ""
        
        if method == "Paste sequence":
            sequence = st.text_area(
                "Enter DNA sequence (A, T, G, C only):",
                height=150,
                placeholder="ATGCATGCATGC..."
            )
        
        elif method == "Upload FASTA file":
            uploaded_file = st.file_uploader("Choose a FASTA file", type=['fasta', 'fa', 'txt'])
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                # Simple FASTA parser
                lines = content.split('\n')
                sequence = ''.join([line for line in lines if not line.startswith('>')])
        
        elif method == "Use example":
            example_choice = st.selectbox(
                "Select example sequence:",
                ["A-philic rich sequence", "G-quadruplex rich", "Mixed motifs"]
            )
            
            examples = {
                "A-philic rich sequence": 
                    "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAATAAAATAAAATAAAAAAGGGCCCAAAAAATTTTTGGGCCCAAA",
                "G-quadruplex rich": 
                    "GGGAGGGAGGGAGGGCCCGGGCCCGGGCCCTATATATACCCGGGCCCGGGCCCGGGAGGGAGGGA",
                "Mixed motifs":
                    "AAAAATTTTTGGGAGGGAGGGAGGGCCCTATATATATATGCGCGCGCAAAAAATTTTTTCCCCGGGG"
            }
            
            sequence = examples.get(example_choice, "")
            st.text_area("Example sequence:", value=sequence, height=100, disabled=True)
        
        return sequence.upper().replace(' ', '').replace('\n', '')
    
    def run_analysis(self, sequence: str, selected_classes: List[int], parallel: bool):
        """Run NBDFinder analysis and display results"""
        
        # Validate sequence
        if not sequence:
            st.error("Please provide a DNA sequence.")
            return
        
        if len(sequence) < 10:
            st.error("Sequence must be at least 10 base pairs long.")
            return
        
        # Clean sequence
        clean_sequence = re.sub(r'[^ATGC]', '', sequence)
        if len(clean_sequence) != len(sequence):
            st.warning(f"Removed {len(sequence) - len(clean_sequence)} non-DNA characters.")
            sequence = clean_sequence
        
        # Run analysis
        with st.spinner("Analyzing sequence for Non-B DNA motifs..."):
            results = analyze_sequence_nbd_finder(
                sequence,
                "user_sequence", 
                parallel=parallel,
                selected_classes=selected_classes if selected_classes else None
            )
        
        # Display results
        self.display_results(results, sequence)
    
    def display_results(self, results: Dict[str, Any], sequence: str):
        """Display analysis results with enhanced visualization"""
        
        st.markdown('<h2 class="subheader">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sequence Length", f"{results['sequence_length']} bp")
        
        with col2:
            st.metric("Total Motifs", results['summary_stats']['total_motifs'])
        
        with col3:
            classes_with_motifs = len([cls for cls, motifs in results['motifs_by_class'].items() if motifs])
            st.metric("Active Classes", f"{classes_with_motifs}/11")
        
        with col4:
            if 'a_philic_breakdown' in results['summary_stats']:
                a_philic_total = (results['summary_stats']['a_philic_breakdown']['high_confidence'] + 
                                results['summary_stats']['a_philic_breakdown']['moderate_confidence'])
                st.metric("A-philic Motifs", a_philic_total)
        
        # Motif visualization
        if results['summary_stats']['total_motifs'] > 0:
            self.plot_motif_distribution(results, sequence)
            
            # A-philic DNA specific visualization
            if 9 in results['motifs_by_class'] and results['motifs_by_class'][9]:
                self.display_a_philic_details(results['motifs_by_class'][9])
            
            # Detailed motif table
            self.display_motif_table(results)
        
        else:
            st.info("No motifs detected with current parameters.")
        
        # Download results
        self.add_download_option(results)
    
    def plot_motif_distribution(self, results: Dict[str, Any], sequence: str):
        """Plot motif distribution along the sequence"""
        
        st.markdown('<h3 class="subheader">🎯 Motif Distribution</h3>', unsafe_allow_html=True)
        
        # Create visualization of motifs along sequence
        fig = go.Figure()
        
        # Add sequence track
        positions = list(range(len(sequence)))
        fig.add_trace(go.Scatter(
            x=positions,
            y=[0] * len(sequence),
            mode='lines',
            name='Sequence',
            line=dict(color='lightgray', width=2),
            showlegend=False
        ))
        
        # Add motif tracks
        y_offset = 0.1
        track_height = 0.15
        
        for class_num in sorted(results['motifs_by_class'].keys()):
            motifs = results['motifs_by_class'][class_num]
            if not motifs:
                continue
            
            class_name = self.config.get_class_name(class_num)
            color = self.motif_colors.get(class_num, '#808080')
            
            for motif in motifs:
                # Draw motif as rectangle
                fig.add_shape(
                    type="rect",
                    x0=motif['Start'],
                    x1=motif['End'],
                    y0=y_offset,
                    y1=y_offset + track_height,
                    fillcolor=color,
                    opacity=0.7,
                    line=dict(color=color, width=2)
                )
                
                # Add motif label
                fig.add_annotation(
                    x=(motif['Start'] + motif['End']) / 2,
                    y=y_offset + track_height/2,
                    text=f"C{class_num}",
                    showarrow=False,
                    font=dict(size=8, color='white'),
                    bgcolor=color
                )
            
            y_offset += track_height + 0.05
        
        fig.update_layout(
            title="Motif Distribution Along Sequence",
            xaxis_title="Position (bp)",
            yaxis_title="Motif Classes",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_a_philic_details(self, a_philic_motifs: List[Dict[str, Any]]):
        """Display detailed A-philic DNA analysis"""
        
        st.markdown('<h3 class="subheader motif-class-9">🧬 A-philic DNA Analysis</h3>', 
                   unsafe_allow_html=True)
        
        # A-philic summary
        high_conf = len([m for m in a_philic_motifs if m.get('confidence') == 'high'])
        moderate_conf = len([m for m in a_philic_motifs if m.get('confidence') == 'moderate'])
        avg_score = np.mean([m.get('Score', 0) for m in a_philic_motifs])
        
        st.markdown(f"""
        <div class="a-philic-highlight">
        <strong>A-philic DNA Detection Results</strong><br>
        • Total motifs: {len(a_philic_motifs)}<br>
        • High confidence: {high_conf}<br>
        • Moderate confidence: {moderate_conf}<br>
        • Average tetranucleotide score: {avg_score:.1f}<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Score distribution plot
        if a_philic_motifs:
            scores = [m.get('Score', 0) for m in a_philic_motifs]
            fig = px.histogram(
                x=scores,
                title="A-philic DNA Score Distribution",
                labels={'x': 'Tetranucleotide Log2 Odds Score', 'y': 'Count'},
                color_discrete_sequence=['#E6B8F7']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_motif_table(self, results: Dict[str, Any]):
        """Display detailed motif results table"""
        
        st.markdown('<h3 class="subheader">📋 Detailed Motif Results</h3>', unsafe_allow_html=True)
        
        # Prepare data for table
        table_data = []
        for class_num, motifs in results['motifs_by_class'].items():
            class_name = self.config.get_class_name(class_num)
            
            for motif in motifs:
                table_data.append({
                    'Class': class_num,
                    'Motif Type': class_name,
                    'Start': motif.get('Start', motif.get('start', 0)),
                    'End': motif.get('End', motif.get('end', 0)),
                    'Length': motif.get('Length', motif.get('length', 0)),
                    'Score': motif.get('Score', motif.get('score', 0)),
                    'Subclass': motif.get('Subclass', 'N/A'),
                    'Confidence': motif.get('confidence', 'N/A')
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            
            # Color-code by class
            def highlight_class(row):
                color = self.motif_colors.get(row['Class'], '#FFFFFF')
                return [f'background-color: {color}; opacity: 0.3'] * len(row)
            
            styled_df = df.style.apply(highlight_class, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No motifs to display.")
    
    def add_download_option(self, results: Dict[str, Any]):
        """Add download option for results"""
        
        st.markdown('<h3 class="subheader">💾 Download Results</h3>', unsafe_allow_html=True)
        
        # Convert results to JSON
        json_data = json.dumps(results, indent=2, default=str)
        
        st.download_button(
            label="📄 Download Results (JSON)",
            data=json_data,
            file_name="nbdfinder_results.json",
            mime="application/json"
        )
    
    def render_about_page(self):
        """Render the about page with system documentation"""
        
        st.markdown('<h1 class="main-header">📖 About NBDFinder</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Advanced Non-B DNA Motif Detection System
        
        NBDFinder is a comprehensive computational tool for detecting and analyzing 
        non-B DNA structures across 11 distinct motif classes, including the newly 
        integrated A-philic DNA detection system.
        
        ### 🧬 11-Class Motif System
        """)
        
        # Display motif classes with descriptions
        for class_num, name, description in self.motif_order:
            color = self.motif_colors[class_num]
            highlight_class = "a-philic-highlight" if class_num == 9 else "metric-container"
            
            st.markdown(f"""
            <div class="{highlight_class}">
            <strong style="color: {color}">Class {class_num}: {name}</strong><br>
            {description}
            </div>
            """, unsafe_allow_html=True)
        
        # A-philic DNA detailed explanation
        st.markdown("""
        ### 🆕 A-philic DNA Detection (Class 9)
        
        A-philic DNA represents sequences with high affinity for A-tract formation and 
        protein-DNA interactions. Our implementation uses advanced tetranucleotide 
        log2 odds scoring for precise detection.
        
        **Key Features:**
        - **Tetranucleotide Scoring**: Uses 18 key tetranucleotides with literature-based weights
        - **Dual Classification**: Distinguishes High Confidence (≥2.0 log2 odds) and Moderate A-philic motifs
        - **Optimized Windows**: 10-20bp sliding windows for comprehensive analysis
        - **Scientific Validation**: Based on established computational biology methods
        
        **Scientific References:**
        - Vinogradov (2003) Bioinformatics - Tetranucleotide analysis methodology
        - Bolshoy et al. (1991) PNAS - A-tract structural properties
        - Rohs et al. (2009) Nature - Protein-DNA interaction patterns
        """)
        
        st.markdown("""
        ### 🔧 Technical Implementation
        
        - **Parallel Processing**: Concurrent detection across all motif classes
        - **Quality Thresholding**: Class-specific scoring and filtering
        - **Standardized Output**: NBDFinder-compatible result format
        - **Scalable Architecture**: Modular design for easy extension
        """)
    
    def render_examples_page(self):
        """Render examples page with sample analyses"""
        
        st.markdown('<h1 class="main-header">💡 Examples</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Sample Analyses and Use Cases
        
        Explore NBDFinder capabilities with curated examples:
        """)
        
        # Example 1: A-philic DNA rich sequence
        st.markdown("### Example 1: A-philic DNA Rich Sequence")
        
        example1_seq = "AAAAAAAAAAAAAATTTTTTTTTTTTTTAAAAAAAAAAAAAATTTTTTTTTTTTTTGGGCCCAAAAAATTTTTGGGCCCAAA"
        
        st.code(example1_seq)
        
        if st.button("Analyze A-philic Example", key="ex1"):
            with st.spinner("Analyzing..."):
                results = analyze_sequence_nbd_finder(example1_seq, "example1", selected_classes=[9])
                st.write(f"**Results**: {results['summary_stats']['total_motifs']} A-philic motifs detected")
                
                if 9 in results['motifs_by_class']:
                    for i, motif in enumerate(results['motifs_by_class'][9]):
                        st.write(f"  {i+1}. {motif['Subclass']}: {motif['Start']}-{motif['End']} "
                               f"(score: {motif['Score']}, length: {motif['Length']}bp)")
        
        # Example 2: Mixed motif analysis
        st.markdown("### Example 2: Multi-Class Analysis")
        
        example2_seq = "GGGAGGGAGGGAGGGCCCAAAAATTTTTTGGGCCCTATATATACCCGGGAAAAAATTTTTTCCCCGGGG"
        
        st.code(example2_seq)
        
        if st.button("Analyze Mixed Example", key="ex2"):
            with st.spinner("Analyzing..."):
                results = analyze_sequence_nbd_finder(example2_seq, "example2")
                st.write(f"**Results**: {results['summary_stats']['total_motifs']} total motifs detected")
                
                for class_num, motifs in results['motifs_by_class'].items():
                    if motifs:
                        class_name = self.config.get_class_name(class_num)
                        st.write(f"  Class {class_num} ({class_name}): {len(motifs)} motifs")
    
    def render_advanced_page(self):
        """Render advanced features and configuration"""
        
        st.markdown('<h1 class="main-header">⚙️ Advanced Features</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## System Configuration and Advanced Options
        
        ### Class-Specific Parameters
        """)
        
        # Display configuration for each class
        config_data = []
        for class_num in range(1, 12):
            config = self.config.get_motif_config(class_num)
            config_data.append({
                'Class': class_num,
                'Name': config['name'],
                'Min Length (bp)': config['S_min'],
                'Max Length (bp)': config['S_max'],
                'Min Score': config['min_score'],
                'Scoring Method': config['scoring_method']
            })
        
        df = pd.DataFrame(config_data)
        st.dataframe(df, use_container_width=True)
        
        # A-philic specific parameters
        st.markdown("### A-philic DNA Parameters")
        
        a_philic_params = self.config.get_a_philic_parameters()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Length Constraints": {
                    "S_min": a_philic_params['S_min'],
                    "S_max": a_philic_params['S_max']
                },
                "Scoring": {
                    "Method": a_philic_params['scoring_method'],
                    "Threshold": a_philic_params['tetranucleotide_threshold'],
                    "Min Score": a_philic_params['min_score']
                }
            })
        
        with col2:
            st.json({
                "Window Sizes": a_philic_params['window_sizes'],
                "Classification Levels": a_philic_params['classification_levels'],
                "Class Number": a_philic_params['class_number']
            })
        
        st.markdown("""
        ### Performance Options
        
        - **Parallel Processing**: Enable concurrent detection across motif classes
        - **Quality Filtering**: Apply class-specific score thresholds
        - **Memory Optimization**: Efficient processing for large sequences
        
        ### API Usage
        
        For programmatic access:
        
        ```python
        from all_motifs_refactored import analyze_sequence_nbd_finder
        
        # Analyze sequence with all classes
        results = analyze_sequence_nbd_finder(sequence, "my_sequence")
        
        # Analyze specific classes only
        results = analyze_sequence_nbd_finder(
            sequence, 
            "my_sequence", 
            selected_classes=[9]  # A-philic DNA only
        )
        ```
        """)


def main():
    """Main application entry point"""
    app = NBDFinderApp()
    app.render_sidebar()


if __name__ == "__main__":
    main()