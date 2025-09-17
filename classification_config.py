#!/usr/bin/env python3
"""
NBDFinder Classification Configuration System
Complete 11-class system for non-B DNA structure detection
"""

# Complete 11-class NBDFinder system
NBD_CLASSES = {
    1: {
        'name': 'Curved DNA',
        'description': 'Intrinsic DNA curvature patterns',
        'color': '#FF6B6B',
        's_min': 10,
        's_max': 100,
        'method': 'curvature_analysis'
    },
    2: {
        'name': 'Slipped DNA',
        'description': 'Direct repeats and STR sequences',
        'color': '#4ECDC4',
        's_min': 10,
        's_max': 200,
        'method': 'repeat_analysis'
    },
    3: {
        'name': 'Cruciform DNA',
        'description': 'Inverted repeat structures',
        'color': '#45B7D1',
        's_min': 10,
        's_max': 100,
        'method': 'inverted_repeat_analysis'
    },
    4: {
        'name': 'R-loop',
        'description': 'RNA-DNA hybrid formation sites',
        'color': '#96CEB4',
        's_min': 15,
        's_max': 150,
        'method': 'rloop_analysis'
    },
    5: {
        'name': 'Triplex',
        'description': 'Triple-helix DNA structures',
        'color': '#FFEAA7',
        's_min': 12,
        's_max': 80,
        'method': 'triplex_analysis'
    },
    6: {
        'name': 'G-Quadruplex',
        'description': 'G4 and variant formations',
        'color': '#DDA0DD',
        's_min': 15,
        's_max': 50,
        'method': 'g4_analysis'
    },
    7: {
        'name': 'i-Motif',
        'description': 'C-rich quadruplex structures',
        'color': '#F8BBD9',
        's_min': 12,
        's_max': 40,
        'method': 'imotif_analysis'
    },
    8: {
        'name': 'Z-DNA',
        'description': 'Left-handed DNA conformations',
        'color': '#C4E17F',
        's_min': 8,
        's_max': 60,
        'method': 'zdna_analysis'
    },
    9: {
        'name': 'A-philic DNA',
        'description': 'A-tract formation and protein-DNA interactions',
        'color': '#E6B8F7',
        's_min': 10,
        's_max': 50,  # extendable to 200bp for A-tracts
        'method': 'tetranucleotide_log2_odds',
        'extended_max': 200,
        'thresholds': {
            'high_confidence': 2.0,
            'moderate': 1.0,
            'strong_tetranuc': 2.0,
            'fraction_high': 5/7,
            'fraction_mod': 4/7
        }
    },
    10: {
        'name': 'Hybrid',
        'description': 'Multi-class overlapping regions',
        'color': '#FFA07A',
        's_min': 10,
        's_max': 100,
        'method': 'overlap_analysis'
    },
    11: {
        'name': 'Non-B DNA Clusters',
        'description': 'Hotspot regions',
        'color': '#20B2AA',
        's_min': 20,
        's_max': 500,
        'method': 'cluster_analysis'
    }
}

# Scoring methods configuration
SCORING_METHODS = {
    'tetranucleotide_log2_odds': {
        'description': 'A-philic DNA scoring using tetranucleotide log2 odds ratios',
        'parameters': {
            'window_min': 10,
            'window_max': 20,
            'step_size': 1,
            'scoring_table': 'TET_LOG2'
        }
    },
    'curvature_analysis': {
        'description': 'DNA curvature prediction using bend angles',
        'parameters': {
            'bend_threshold': 10.0,
            'window_size': 11
        }
    },
    'repeat_analysis': {
        'description': 'Direct repeat and STR detection',
        'parameters': {
            'min_repeat_length': 2,
            'max_repeat_length': 10,
            'min_copies': 3
        }
    },
    'inverted_repeat_analysis': {
        'description': 'Inverted repeat detection for cruciforms',
        'parameters': {
            'min_stem_length': 6,
            'max_loop_length': 20,
            'mismatch_tolerance': 1
        }
    },
    'rloop_analysis': {
        'description': 'R-loop formation prediction',
        'parameters': {
            'gc_skew_threshold': 0.2,
            'window_size': 50
        }
    },
    'triplex_analysis': {
        'description': 'Triple helix structure prediction',
        'parameters': {
            'purine_threshold': 0.7,
            'min_length': 12
        }
    },
    'g4_analysis': {
        'description': 'G-quadruplex detection',
        'parameters': {
            'g_runs': 4,
            'min_g_length': 3,
            'max_loop_length': 7
        }
    },
    'imotif_analysis': {
        'description': 'i-Motif detection',
        'parameters': {
            'c_runs': 4,
            'min_c_length': 3,
            'max_loop_length': 15
        }
    },
    'zdna_analysis': {
        'description': 'Z-DNA conformation prediction',
        'parameters': {
            'alternating_threshold': 0.8,
            'gc_content_min': 0.6
        }
    },
    'overlap_analysis': {
        'description': 'Multi-class overlap detection',
        'parameters': {
            'overlap_threshold': 0.5,
            'min_classes': 2
        }
    },
    'cluster_analysis': {
        'description': 'Non-B DNA hotspot clustering',
        'parameters': {
            'cluster_distance': 100,
            'min_motifs': 3
        }
    }
}

# Quality filtering parameters
QUALITY_FILTERS = {
    'default': {
        'min_score': 0.5,
        'min_length': 8,
        'max_overlap': 0.8
    },
    'a_philic': {
        'min_score': 1.0,
        'min_length': 10,
        'max_overlap': 0.5,
        'require_strong_tetranuc': True
    },
    'g4': {
        'min_score': 0.7,
        'min_length': 15,
        'max_overlap': 0.3
    },
    'cruciform': {
        'min_score': 0.6,
        'min_length': 12,
        'max_overlap': 0.4
    }
}

# Parallel processing configuration
PARALLEL_CONFIG = {
    'max_workers': 4,
    'chunk_size': 1000,
    'classes_to_run': list(range(1, 12)),  # All 11 classes by default
    'enable_hybrid_detection': True,
    'enable_clustering': True
}


def get_class_config(class_id):
    """Get configuration for a specific NBD class."""
    return NBD_CLASSES.get(class_id, None)


def get_scoring_config(method_name):
    """Get configuration for a specific scoring method."""
    return SCORING_METHODS.get(method_name, None)


def get_quality_filter(filter_name):
    """Get quality filter configuration."""
    return QUALITY_FILTERS.get(filter_name, QUALITY_FILTERS['default'])


def validate_class_constraints(class_id, start, end):
    """Validate that a motif meets class-specific constraints."""
    config = get_class_config(class_id)
    if not config:
        return False
    
    length = end - start
    if length < config['s_min'] or length > config['s_max']:
        return False
    
    return True


def get_extended_constraints(class_id):
    """Get extended constraints for classes that support them."""
    config = get_class_config(class_id)
    if not config:
        return None
    
    extended = {
        's_min': config['s_min'],
        's_max': config['s_max']
    }
    
    if 'extended_max' in config:
        extended['extended_max'] = config['extended_max']
    
    return extended


def get_class_colors():
    """Get color mapping for all NBD classes."""
    return {class_id: config['color'] for class_id, config in NBD_CLASSES.items()}


def get_class_names():
    """Get name mapping for all NBD classes."""
    return {class_id: config['name'] for class_id, config in NBD_CLASSES.items()}


if __name__ == "__main__":
    # Test configuration
    print("NBDFinder Classification System - 11 Classes")
    for class_id, config in NBD_CLASSES.items():
        print(f"Class {class_id}: {config['name']} - {config['description']}")
    
    print(f"\nA-philic DNA (Class 9) configuration:")
    a_philic_config = get_class_config(9)
    print(f"  Length constraints: {a_philic_config['s_min']}-{a_philic_config['s_max']} bp")
    print(f"  Extended max: {a_philic_config.get('extended_max', 'N/A')} bp")
    print(f"  Method: {a_philic_config['method']}")
    print(f"  Color: {a_philic_config['color']}")