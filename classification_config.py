"""
NBDFinder Classification Configuration

This module provides configuration parameters for all motif classes in the NBDFinder system,
including the new A-philic DNA detection (Class 9) with tetranucleotide log2 odds scoring.

Updated Class System (11 classes):
1. Curved DNA - Intrinsic DNA curvature patterns
2. Slipped DNA - Direct repeats and STR sequences  
3. Cruciform DNA - Inverted repeat structures
4. R-loop - RNA-DNA hybrid formation sites
5. Triplex - Triple-helix DNA structures
6. G-Quadruplex - G4 and variant formations
7. i-Motif - C-rich quadruplex structures
8. Z-DNA - Left-handed DNA conformations
9. A-philic DNA - A-tract-favoring protein binding sites (NEW)
10. Hybrid - Multi-class overlapping regions
11. Non-B DNA Clusters - Hotspot regions
"""

from typing import Dict, Any, Tuple


class NBDFinderConfig:
    """Configuration class for NBDFinder motif detection system"""
    
    def __init__(self):
        """Initialize configuration parameters for all motif classes"""
        
        # A-philic DNA length constraints (Class 9)
        self.a_philic_constraints = {
            'S_min': 10,  # Minimum sequence length (bp)
            'S_max': 50,  # Maximum sequence length (bp)
            'min_score': 10.0,  # Minimum quality score for filtering
            'scoring_method': 'tetranucleotide_log2_odds'
        }
        
        # Complete motif class configuration
        self.motif_classes = {
            1: {
                'name': 'Curved_DNA',
                'description': 'Intrinsic DNA curvature patterns',
                'S_min': 15,
                'S_max': 100,
                'min_score': 5.0,
                'scoring_method': 'curvature_analysis'
            },
            2: {
                'name': 'Slipped_DNA', 
                'description': 'Direct repeats and STR sequences',
                'S_min': 8,
                'S_max': 200,
                'min_score': 3.0,
                'scoring_method': 'repeat_detection'
            },
            3: {
                'name': 'Cruciform_DNA',
                'description': 'Inverted repeat structures',
                'S_min': 12,
                'S_max': 150,
                'min_score': 4.0,
                'scoring_method': 'inverted_repeat_analysis'
            },
            4: {
                'name': 'R_loop',
                'description': 'RNA-DNA hybrid formation sites',
                'S_min': 20,
                'S_max': 300,
                'min_score': 6.0,
                'scoring_method': 'rloop_propensity'
            },
            5: {
                'name': 'Triplex',
                'description': 'Triple-helix DNA structures',
                'S_min': 15,
                'S_max': 100,
                'min_score': 7.0,
                'scoring_method': 'triplex_formation'
            },
            6: {
                'name': 'G_Quadruplex',
                'description': 'G4 and variant formations',
                'S_min': 16,
                'S_max': 80,
                'min_score': 8.0,
                'scoring_method': 'g4_score'
            },
            7: {
                'name': 'i_Motif',
                'description': 'C-rich quadruplex structures',
                'S_min': 12,
                'S_max': 60,
                'min_score': 6.5,
                'scoring_method': 'i_motif_score'
            },
            8: {
                'name': 'Z_DNA',
                'description': 'Left-handed DNA conformations',
                'S_min': 8,
                'S_max': 40,
                'min_score': 5.5,
                'scoring_method': 'z_dna_propensity'
            },
            9: {
                'name': 'A_philic_DNA',
                'description': 'A-tract-favoring protein binding sites',
                'S_min': self.a_philic_constraints['S_min'],
                'S_max': self.a_philic_constraints['S_max'],
                'min_score': self.a_philic_constraints['min_score'],
                'scoring_method': self.a_philic_constraints['scoring_method']
            },
            10: {
                'name': 'Hybrid',
                'description': 'Multi-class overlapping regions',
                'S_min': 10,
                'S_max': 500,
                'min_score': 2.0,
                'scoring_method': 'overlap_analysis'
            },
            11: {
                'name': 'Non_B_DNA_Clusters',
                'description': 'Hotspot regions',
                'S_min': 50,
                'S_max': 1000,
                'min_score': 3.0,
                'scoring_method': 'cluster_density'
            }
        }
        
        # Scientific references for A-philic DNA methodology
        self.a_philic_references = {
            'tetranucleotide_methodology': 'Vinogradov (2003) Bioinformatics',
            'a_tract_properties': 'Bolshoy et al. (1991) PNAS',
            'protein_dna_interactions': 'Rohs et al. (2009) Nature'
        }
        
        # Global NBDFinder parameters
        self.global_params = {
            'max_sequence_length': 10000,
            'min_sequence_length': 10,
            'overlap_threshold': 0.5,
            'quality_filter_enabled': True
        }
    
    def get_motif_limits(self, motif_class: int) -> Tuple[int, int]:
        """
        Get sequence length limits for a specific motif class
        
        Args:
            motif_class: Integer motif class (1-11)
            
        Returns:
            Tuple of (S_min, S_max) length constraints
            
        Raises:
            ValueError: If motif_class is not in valid range
        """
        if motif_class not in self.motif_classes:
            raise ValueError(f"Invalid motif class: {motif_class}. Must be 1-11.")
        
        config = self.motif_classes[motif_class]
        return (config['S_min'], config['S_max'])
    
    def get_motif_config(self, motif_class: int) -> Dict[str, Any]:
        """
        Get complete configuration for a specific motif class
        
        Args:
            motif_class: Integer motif class (1-11)
            
        Returns:
            Dictionary containing all configuration parameters
        """
        if motif_class not in self.motif_classes:
            raise ValueError(f"Invalid motif class: {motif_class}. Must be 1-11.")
        
        return self.motif_classes[motif_class].copy()
    
    def get_scoring_method(self, motif_class: int) -> str:
        """
        Get scoring method for a specific motif class
        
        Args:
            motif_class: Integer motif class (1-11)
            
        Returns:
            String identifying the scoring method
        """
        if motif_class not in self.motif_classes:
            raise ValueError(f"Invalid motif class: {motif_class}. Must be 1-11.")
        
        return self.motif_classes[motif_class]['scoring_method']
    
    def get_min_score(self, motif_class: int) -> float:
        """
        Get minimum score threshold for a specific motif class
        
        Args:
            motif_class: Integer motif class (1-11)
            
        Returns:
            Minimum score threshold for quality filtering
        """
        if motif_class not in self.motif_classes:
            raise ValueError(f"Invalid motif class: {motif_class}. Must be 1-11.")
        
        return self.motif_classes[motif_class]['min_score']
    
    def get_class_name(self, motif_class: int) -> str:
        """
        Get name for a specific motif class
        
        Args:
            motif_class: Integer motif class (1-11)
            
        Returns:
            String name of the motif class
        """
        if motif_class not in self.motif_classes:
            raise ValueError(f"Invalid motif class: {motif_class}. Must be 1-11.")
        
        return self.motif_classes[motif_class]['name']
    
    def get_all_class_names(self) -> Dict[int, str]:
        """
        Get all motif class names
        
        Returns:
            Dictionary mapping class numbers to names
        """
        return {cls: config['name'] for cls, config in self.motif_classes.items()}
    
    def get_a_philic_parameters(self) -> Dict[str, Any]:
        """
        Get specific parameters for A-philic DNA detection
        
        Returns:
            Dictionary with A-philic DNA specific parameters
        """
        return {
            **self.a_philic_constraints,
            'references': self.a_philic_references,
            'class_number': 9,
            'tetranucleotide_threshold': 2.0,
            'window_sizes': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'classification_levels': ['High Confidence A-philic', 'Moderate A-philic']
        }
    
    def validate_motif_result(self, motif_result: Dict[str, Any]) -> bool:
        """
        Validate a motif detection result against class constraints
        
        Args:
            motif_result: Dictionary containing motif detection result
            
        Returns:
            True if result meets class constraints, False otherwise
        """
        if 'Class' not in motif_result:
            return False
        
        motif_class = motif_result['Class']
        if motif_class not in self.motif_classes:
            return False
        
        config = self.motif_classes[motif_class]
        
        # Check length constraints
        if 'Length' in motif_result:
            length = motif_result['Length']
            if length < config['S_min'] or length > config['S_max']:
                return False
        
        # Check score threshold
        if 'Score' in motif_result:
            score = motif_result['Score']
            if score < config['min_score']:
                return False
        
        return True


# Global configuration instance
config = NBDFinderConfig()


def get_motif_limits(motif_class: int) -> Tuple[int, int]:
    """
    Global function to get motif length limits
    Updated to handle A-philic DNA motifs (Class 9)
    
    Args:
        motif_class: Integer motif class (1-11)
        
    Returns:
        Tuple of (S_min, S_max) length constraints
    """
    return config.get_motif_limits(motif_class)


if __name__ == "__main__":
    # Test configuration system
    print("NBDFinder Configuration System Test")
    print("=" * 50)
    
    config = NBDFinderConfig()
    
    print("All motif classes:")
    for cls, name in config.get_all_class_names().items():
        limits = config.get_motif_limits(cls)
        min_score = config.get_min_score(cls)
        scoring = config.get_scoring_method(cls)
        print(f"  {cls}. {name}: {limits[0]}-{limits[1]}bp, "
              f"min_score={min_score}, method={scoring}")
    
    print(f"\nA-philic DNA (Class 9) parameters:")
    a_philic_params = config.get_a_philic_parameters()
    for key, value in a_philic_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nTesting A-philic DNA constraints:")
    a_philic_limits = get_motif_limits(9)
    print(f"  Length limits: {a_philic_limits[0]}-{a_philic_limits[1]} bp")
    print(f"  Minimum score: {config.get_min_score(9)}")
    print(f"  Scoring method: {config.get_scoring_method(9)}")