"""
Detectors Module - Non-B DNA Motif Detectors
============================================

This module contains all specialized motif detector classes:
- BaseMotifDetector: Abstract base class for all detectors
- CurvedDNADetector: A-tract mediated DNA curvature detection
- ZDNADetector: Z-DNA and left-handed helix detection
- APhilicDetector: A-philic DNA tetranucleotide analysis
- SlippedDNADetector: Direct repeats and STR detection
- CruciformDetector: Palindromic inverted repeat detection
- RLoopDetector: R-loop formation site detection
- TriplexDetector: Triplex and mirror repeat detection
- GQuadruplexDetector: G4 and G-quadruplex variants
- IMotifDetector: i-Motif and AC-motif detection
"""

__version__ = "2025.1"

# Import base detector
from .base import BaseMotifDetector

# Import all detector implementations
from .curved_dna import CurvedDNADetector
from .z_dna import ZDNADetector
from .a_philic import APhilicDetector
from .slipped_dna import SlippedDNADetector
from .cruciform import CruciformDetector
from .r_loop import RLoopDetector
from .triplex import TriplexDetector
from .g_quadruplex import GQuadruplexDetector
from .i_motif import IMotifDetector

__all__ = [
    'BaseMotifDetector',
    'CurvedDNADetector',
    'ZDNADetector',
    'APhilicDetector',
    'SlippedDNADetector',
    'CruciformDetector',
    'RLoopDetector',
    'TriplexDetector',
    'GQuadruplexDetector',
    'IMotifDetector',
]
