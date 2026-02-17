"""Pattern Registry Module
=======================

Functions for loading and managing pattern registries and Hyperscan databases.
Provides centralized registry management for all detector classes.

Extracted from utilities.py for modular architecture.
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Pattern registry version
PATTERN_REGISTRY_VERSION = "2024.2"

# Cache for consolidated registry
_CONSOLIDATED_REGISTRY = None

# Check for optional dependencies
_HYPERSCAN_AVAILABLE = False
_HYPERSCAN_VERSION = None
_HYPERSCAN_ERROR = None

try:
    import hyperscan
    _HYPERSCAN_AVAILABLE = True
    try:
        _HYPERSCAN_VERSION = hyperscan.__version__
    except AttributeError:
        _HYPERSCAN_VERSION = 'unknown'
    logger.info(f"Hyperscan loaded successfully (version: {_HYPERSCAN_VERSION})")
except ImportError as e:
    hyperscan = None
    _HYPERSCAN_ERROR = f"Hyperscan Python bindings not installed: {e}"
    logger.info(f"Hyperscan not available - using pure Python fallback. {_HYPERSCAN_ERROR}")
except Exception as e:
    hyperscan = None
    _HYPERSCAN_ERROR = f"Hyperscan initialization failed: {e}"
    logger.warning(f"Hyperscan not available - using pure Python fallback. {_HYPERSCAN_ERROR}")

# pandas is optional for Excel registry loading
_PANDAS_AVAILABLE = False
_PANDAS_WARNING_LOGGED = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None

def _load_consolidated_registry_from_excel():
    """Load the consolidated registry from Excel file and cache it
    
    Returns:
        dict: Registry data loaded from pattern_registry2.xlsx
    """
    global _PANDAS_WARNING_LOGGED
    
    if not _PANDAS_AVAILABLE:
        if not _PANDAS_WARNING_LOGGED:
            logger.warning("pandas not available - Excel pattern loading disabled. Install with: pip install pandas openpyxl")
            _PANDAS_WARNING_LOGGED = True
        return None
    
    # Load pattern_registry2.xlsx from current directory
    excel_path = "pattern_registry2.xlsx"
    if not os.path.isfile(excel_path):
        logger.error(f"Pattern registry file not found: {excel_path}")
        return None
    
    try:
        # Read all sheets from Excel
        excel_data = pd.read_excel(excel_path, sheet_name=None)  # Load all sheets
        
        # Build consolidated registry structure
        registries = {}
        for sheet_name, df in excel_data.items():
            if sheet_name == 'Summary':
                continue  # Skip summary sheet
            
            # Convert DataFrame to list of dictionaries
            patterns = df.to_dict('records')
            
            # Clean up NaN values
            for pattern in patterns:
                for key, value in list(pattern.items()):
                    if pd.isna(value):
                        pattern[key] = None
            
            # Create registry structure
            registries[sheet_name] = {
                'class': sheet_name,
                'n_patterns': len(patterns),
                'patterns': patterns
            }
        
        registry_data = {
            'version': f'{PATTERN_REGISTRY_VERSION}-excel',
            'source': excel_path,
            'registries': registries,
            'total_patterns': sum(r['n_patterns'] for r in registries.values()),
            'total_classes': len(registries)
        }
        
        logger.info(f"Loaded consolidated registry from {excel_path} ({registry_data['total_patterns']} patterns)")
        return registry_data
        
    except Exception as e:
        logger.error(f"Failed to load Excel registry from {excel_path}: {e}")
        return None


def _load_consolidated_registry():
    """Load the consolidated registry file once and cache it
    
    Loads from Excel file (pattern_registry2.xlsx with updated normalized scores).
    
    Returns:
        dict: Consolidated registry data
    """
    global _CONSOLIDATED_REGISTRY
    if _CONSOLIDATED_REGISTRY is not None:
        return _CONSOLIDATED_REGISTRY
    
    # Load from Excel
    _CONSOLIDATED_REGISTRY = _load_consolidated_registry_from_excel()
    if _CONSOLIDATED_REGISTRY is not None:
        return _CONSOLIDATED_REGISTRY
    
    # If Excel loading failed, raise an error
    raise FileNotFoundError(
        "Failed to load pattern registry from pattern_registry2.xlsx. "
        "Please ensure the Excel file exists and is properly formatted."
    )


def _load_registry(registry_dir: str, class_name: str):
    """Load registry from consolidated Excel file"""
    # Load from consolidated registry
    consolidated = _load_consolidated_registry()
    if consolidated and "registries" in consolidated:
        if class_name in consolidated["registries"]:
            logger.debug(f"Loading {class_name} from consolidated registry")
            return consolidated["registries"][class_name]
    
    raise FileNotFoundError(f"No registry found for {class_name} in pattern registry Excel file")


def clear_pattern_registry_cache():
    """
    Clear the cached pattern registry data.
    
    This is mainly useful for testing or when you need to reload
    the pattern data after modifying the Excel or JSON files.
    
    Example:
        >>> from utilities import clear_pattern_registry_cache, _load_consolidated_registry
        >>> clear_pattern_registry_cache()
        >>> registry = _load_consolidated_registry()  # Reloads from file
    """
    global _CONSOLIDATED_REGISTRY
    _CONSOLIDATED_REGISTRY = None
    logger.debug("Pattern registry cache cleared")


def load_db_for_class(class_name: str, registry_dir: str = "registry"):
    """
    Returns (db, id_to_pattern, id_to_score).
    db is a hyperscan.Database instance (or None if hyperscan not available).
    id_to_pattern and id_to_score are dicts mapping integer id -> pattern/tenmer / score.
    
    Handles both 10-mer patterns (with 'tenmer' key) and regex patterns (with 'pattern' key).
    """
    reg = _load_registry(registry_dir, class_name)
    patterns = reg.get("patterns", [])
    
    # Handle both 10-mer registries (tenmer) and regex registries (pattern)
    id_to_pattern = {}
    for p in patterns:
        pattern_id = int(p["id"])
        # Try 'tenmer' first (for 10-mer patterns like ZDNA, APhilic)
        if "tenmer" in p:
            id_to_pattern[pattern_id] = p["tenmer"]
        # Fall back to 'pattern' (for regex patterns like G4, IMotif, etc.)
        elif "pattern" in p:
            id_to_pattern[pattern_id] = p["pattern"]
        else:
            logger.warning(f"Pattern {pattern_id} in {class_name} has neither 'tenmer' nor 'pattern' key")
    
    id_to_score = {int(p["id"]): float(p.get("score", 0.0)) for p in patterns}

    db = None
    hsdb_path = os.path.join(registry_dir, f"{class_name}.hsdb")
    if _HYPERSCAN_AVAILABLE:
        try:
            db = hyperscan.Database()
            if os.path.isfile(hsdb_path):
                # Try to deserialize pre-compiled database
                try:
                    with open(hsdb_path, "rb") as fh:
                        raw = fh.read()
                    db.deserialize(raw)
                    logger.info(f"Loaded serialized Hyperscan DB for {class_name} from {hsdb_path}")
                except Exception as deserialize_error:
                    logger.warning(f"Failed to deserialize {hsdb_path}: {deserialize_error}. Recompiling from patterns...")
                    # Fallback: compile from patterns
                    try:
                        expressions = [id_to_pattern[i].encode("ascii") for i in sorted(id_to_pattern.keys())]
                        ids = sorted(id_to_pattern.keys())
                        db.compile(expressions=expressions, ids=ids, elements=len(expressions))
                        logger.info(f"Successfully compiled Hyperscan DB for {class_name} from patterns")
                    except Exception as compile_error:
                        logger.error(f"Failed to compile Hyperscan DB for {class_name}: {compile_error}")
                        db = None
            else:
                # No serialized DB; compile from patterns
                logger.debug(f"No serialized DB found at {hsdb_path}, compiling from patterns...")
                try:
                    expressions = [id_to_pattern[i].encode("ascii") for i in sorted(id_to_pattern.keys())]
                    ids = sorted(id_to_pattern.keys())
                    db.compile(expressions=expressions, ids=ids, elements=len(expressions))
                    logger.info(f"Compiled Hyperscan DB for {class_name} from patterns in {registry_dir}")
                except Exception as compile_error:
                    logger.error(f"Failed to compile Hyperscan DB for {class_name}: {compile_error}")
                    db = None
        except Exception as e:
            logger.error(f"Hyperscan database operations failed for {class_name}: {e}")
            db = None
    else:
        logger.debug(f"Hyperscan not installed; returning None DB for {class_name} (use pure-Python matcher).")

    return db, id_to_pattern, id_to_score
"""
Helper to load regex pattern registries and compile them with Hyperscan.

This module provides utilities for loading and using regex-based pattern registries
(CurvedDNA, G4, IMotif) similar to how the 10-mer registries work for A-philic and Z-DNA.

API:
    load_registry_for_class(class_name: str, registry_dir: str) 
        -> (db, id_to_pattern, id_to_subclass, id_to_score)
      - db: hyperscan.Database instance (or None if hyperscan not available)
      - id_to_pattern: dict mapping id -> regex pattern (string)
      - id_to_subclass: dict mapping id -> subclass name (string)
      - id_to_score: dict mapping id -> score (float)
"""


def load_registry_for_class(class_name: str, registry_dir: str = "registry"):
    """
    Load regex pattern registry and compile with Hyperscan.
    
    Returns (db, id_to_pattern, id_to_subclass, id_to_score).
    - db is a hyperscan.Database instance (or None if hyperscan not available).
    - id_to_pattern: dict mapping integer id -> regex pattern string
    - id_to_subclass: dict mapping integer id -> subclass name
    - id_to_score: dict mapping integer id -> score value
    """
    # Load registry data
    reg = _load_registry(registry_dir, class_name)
    patterns = reg.get("patterns", [])
    
    # Extract mappings
    id_to_pattern = {int(p["id"]): p["pattern"] for p in patterns}
    id_to_subclass = {int(p["id"]): p.get("subclass", "unknown") for p in patterns}
    id_to_score = {int(p["id"]): float(p.get("score", 0.0)) for p in patterns}
    
    # Compile Hyperscan database if available
    db = None
    if _HYPERSCAN_AVAILABLE:
        try:
            # Try to load pre-compiled DB first
            hsdb_path = os.path.join(registry_dir, f"{class_name}.hsdb")
            if os.path.isfile(hsdb_path):
                try:
                    db = hyperscan.Database()
                    with open(hsdb_path, "rb") as fh:
                        raw = fh.read()
                    db.deserialize(raw)
                    logger.info(f"Loaded serialized Hyperscan DB for {class_name}")
                except Exception as deserialize_error:
                    logger.warning(f"Failed to deserialize {class_name}.hsdb: {deserialize_error}. Recompiling...")
                    db = None
            
            # If no pre-compiled DB or deserialization failed, compile from patterns
            if db is None:
                logger.debug(f"Compiling Hyperscan DB for {class_name} from {len(id_to_pattern)} patterns...")
                try:
                    expressions = []
                    ids = []
                    flags = []
                    
                    for i in sorted(id_to_pattern.keys()):
                        expressions.append(id_to_pattern[i].encode("ascii"))
                        ids.append(i)
                        # Use CASELESS and DOTALL flags for DNA matching
                        flags.append(hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_DOTALL)
                    
                    db = hyperscan.Database()
                    db.compile(
                        expressions=expressions,
                        ids=ids,
                        elements=len(expressions),
                        flags=flags
                    )
                    logger.info(f"Compiled Hyperscan DB for {class_name} from {len(expressions)} patterns")
                except Exception as compile_error:
                    logger.error(f"Hyperscan compilation failed for {class_name}: {compile_error}")
                    db = None
                
        except Exception as e:
            logger.error(f"Hyperscan operations failed for {class_name}: {e}")
            db = None
    else:
        logger.debug(f"Hyperscan not available for {class_name}; use pure-Python matching")
    
    return db, id_to_pattern, id_to_subclass, id_to_score


# Cache compiled databases to avoid recompilation
_REGISTRY_CACHE = {}


def get_cached_registry(class_name: str, registry_dir: str = "registry"):
    """
    Get cached registry or load and compile it.
    Returns (db, id_to_pattern, id_to_subclass, id_to_score).
    """
    cache_key = f"{registry_dir}/{class_name}"
    
    if cache_key not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[cache_key] = load_registry_for_class(class_name, registry_dir)
    
    return _REGISTRY_CACHE[cache_key]


def scan_with_registry(class_name: str, sequence: str, registry_dir: str = "registry"):
    """
    Scan a sequence using a registry's compiled Hyperscan DB.
    
    Returns list of (start, end, pattern_id, subclass) matches.
    Falls back to pure-Python regex matching if Hyperscan not available.
    
    Improved error handling and logging.
    """
    db, id_to_pattern, id_to_subclass, id_to_score = get_cached_registry(class_name, registry_dir)
    matches = []
    
    if db is not None and _HYPERSCAN_AVAILABLE:
        # Use Hyperscan for fast matching
        def on_match(pattern_id, start, end, flags, context):
            subclass = id_to_subclass.get(pattern_id, "unknown")
            matches.append((start, end, pattern_id, subclass))
        
        try:
            db.scan(sequence.encode(), match_event_handler=on_match)
            logger.debug(f"Hyperscan scan completed for {class_name}: {len(matches)} matches")
            # Successfully completed Hyperscan scan - sort and return results
            matches.sort(key=lambda x: x[0])
            return matches
        except Exception as e:
            logger.error(f"Hyperscan scan failed for {class_name}: {e}. Falling back to regex.")
            # Clear matches and fall through to regex matching below
            matches = []
    
    # Fallback to pure-Python regex matching (only if Hyperscan not available or failed)
    logger.debug(f"Using pure-Python regex matching for {class_name}")
    import re
    for pattern_id in sorted(id_to_pattern.keys()):
        pattern = id_to_pattern[pattern_id]
        subclass = id_to_subclass[pattern_id]
        
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            for match in compiled_pattern.finditer(sequence):
                matches.append((match.start(), match.end(), pattern_id, subclass))
        except re.error as e:
            logger.warning(f"Invalid regex pattern {pattern_id} in {class_name}: {e}")
            continue
    
    # Sort by start position
    matches.sort(key=lambda x: x[0])
    return matches
"""
Non-B DNA Motif Patterns Registry
=================================

Comprehensive pattern library with scoring algorithms for all 11 motif classes.
Optimized for scientific accuracy and Hyperscan compatibility.

PATTERN CLASSIFICATION TABLE:
=============================
Class | Patterns | Subclasses | Scoring Method            | References
------|----------|------------|---------------------------|------------------
  1   |    15    |     2      | A-tract curvature         | Olson 1998
  2   |    37    |     2      | Repeat instability        | Wells 2005  
  3   |    89    |     1      | Palindrome stability      | Lilley 2000
  4   |    11    |     1      | R-loop formation          | Skourti 2019
  5   |    14    |     2      | Triplex potential         | Frank-Kamenetskii 1995
  6   |    16    |     7      | G4Hunter algorithm        | Bedrat 2016
  7   |    12    |     3      | i-motif pH stability      | Zeraati 2018
  8   |    13    |     2      | Z-DNA transition          | Ho 1986
  9   |     8    |     1      | A-philic propensity       | Gorin 1995
 10   |  Dynamic |  Dynamic   | Overlap analysis          | This work
 11   |  Dynamic |  Dynamic   | Clustering algorithm      | This work

Total: 207+ patterns across 22+ subclasses
"""

# Note: Hyperscan availability is checked once at module initialization (above)
# to avoid redundant imports and maintain consistent state across the module.

# =============================================================================
# COMPREHENSIVE PATTERN REGISTRY
# =============================================================================

class PatternRegistry:
    """Complete registry of all Non-B DNA patterns with metadata"""
    
    # Class 1: Curved DNA - A-tract mediated DNA bending (optimized)
    CURVED_DNA_PATTERNS = {
        'a_tracts': [
            # (pattern, pattern_id, name, subclass, min_len, scoring_method, confidence, biological_significance, reference)
            (r'A{4,15}', 'CRV_1_1', 'A-tract', 'Local Curvature', 4, 'curvature_score', 0.95, 'DNA bending', 'Crothers 1992'),
            (r'T{4,15}', 'CRV_1_2', 'T-tract', 'Local Curvature', 4, 'curvature_score', 0.95, 'DNA bending', 'Crothers 1992'),
            (r'(?:A{3,8}T{1,5}){2,}', 'CRV_1_3', 'AT-rich tract', 'Local Curvature', 8, 'curvature_score', 0.85, 'Sequence-dependent bending', 'Hagerman 1986'),
        ],
        'phased_a_tracts': [
            (r'(?:A{3,8}.{8,12}){3,}A{3,8}', 'CRV_1_4', 'Phased A-tracts', 'Global Curvature', 20, 'phasing_score', 0.90, 'Macroscopic curvature', 'Koo 1986'),
            (r'(?:T{3,8}.{8,12}){3,}T{3,8}', 'CRV_1_5', 'Phased T-tracts', 'Global Curvature', 20, 'phasing_score', 0.90, 'Macroscopic curvature', 'Koo 1986'),
        ]
    }
    
    # Class 2: Slipped DNA - Tandem repeat-induced slippage
    SLIPPED_DNA_PATTERNS = {
        'short_tandem_repeats': [
            (r'([ATGC])\1{9,}', 'SLP_2_1', 'Mononucleotide repeat', 'STR', 10, 'instability_score', 0.98, 'Replication slippage', 'Schlötterer 2000'),
            (r'([ATGC]{2})\1{4,}', 'SLP_2_2', 'Dinucleotide repeat', 'STR', 10, 'instability_score', 0.95, 'Microsatellite instability', 'Weber 1989'),
            (r'([ATGC]{3})\1{3,}', 'SLP_2_3', 'Trinucleotide repeat', 'STR', 12, 'instability_score', 0.92, 'Expansion diseases', 'Ashley 1993'),
            (r'([ATGC]{4})\1{2,}', 'SLP_2_4', 'Tetranucleotide repeat', 'STR', 12, 'instability_score', 0.85, 'Genetic polymorphisms', 'Edwards 1991'),
            (r'(CA)\1{4,}', 'SLP_2_5', 'CA repeat', 'STR', 10, 'instability_score', 0.95, 'Common microsatellite', 'Weber 1989'),
            (r'(CGG)\1{3,}', 'SLP_2_6', 'CGG repeat', 'STR', 12, 'instability_score', 0.90, 'Fragile X syndrome', 'Verkerk 1991'),
        ],
        'direct_repeats': [
            (r'([ATGC]{5,20})(?:[ATGC]{0,100})\1', 'SLP_2_7', 'Direct repeat', 'Direct Repeat', 10, 'repeat_score', 0.80, 'Recombination hotspots', 'Jeffreys 1985'),
            (r'([ATGC]{10,50})(?:[ATGC]{0,200})\1', 'SLP_2_8', 'Long direct repeat', 'Direct Repeat', 20, 'repeat_score', 0.75, 'Genomic instability', 'Lupski 1998'),
        ]
    }
    
    # Class 3: Cruciform DNA - Inverted repeat-induced four-way junctions  
    CRUCIFORM_PATTERNS = {
        'inverted_repeats': [
            (r'([ATGC]{6,20})[ATGC]{0,50}', 'CRU_3_1', 'Potential palindrome', 'Inverted Repeats', 12, 'cruciform_stability', 0.95, 'DNA secondary structure', 'Lilley 2000'),
            (r'([ATGC]{8,15})[ATGC]{2,20}([ATGC]{8,15})', 'CRU_3_2', 'Inverted repeat candidate', 'Inverted Repeats', 16, 'cruciform_stability', 0.80, 'Secondary structure prone', 'Pearson 1996'),
            (r'([ATGC]{4,10})[ATGC]{0,10}([ATGC]{4,10})', 'CRU_3_3', 'Short inverted repeat', 'Inverted Repeats', 8, 'cruciform_stability', 0.70, 'Local secondary structure', 'Sinden 1994'),
        ]
    }
    
    # Class 4: R-loop - RNA-DNA hybrid structures
    R_LOOP_PATTERNS = {
        'r_loop_formation_sites': [
            (r'[GC]{10,}[AT]{2,10}[GC]{10,}', 'RLP_4_1', 'GC-rich R-loop site', 'R-loop formation sites', 20, 'r_loop_potential', 0.85, 'Transcription-replication conflicts', 'Aguilera 2012'),
            (r'G{5,}[ATGC]{10,100}C{5,}', 'RLP_4_2', 'G-C rich region', 'R-loop formation sites', 20, 'r_loop_potential', 0.80, 'R-loop prone regions', 'Ginno 2012'),
            (r'[GC]{6,}[AT]{1,5}[GC]{6,}', 'RLP_4_3', 'GC-AT pattern', 'R-loop formation sites', 15, 'r_loop_potential', 0.75, 'Transcriptional pausing', 'Skourti-Stathaki 2011'),
        ],
        'qmrlfs_model_1': [
            (r'G{3,}[ATCGU]{1,10}?G{3,}(?:[ATCGU]{1,10}?G{3,}){1,}?', 'QmRLFS_4_1', 'QmRLFS Model 1', 'QmRLFS-m1', 25, 'qmrlfs_score', 0.90, 'RIZ detection with 3+ G tracts', 'Jenjaroenpun 2016'),
        ],
        'qmrlfs_model_2': [
            (r'G{4,}(?:[ATCGU]{1,10}?G{4,}){1,}?', 'QmRLFS_4_2', 'QmRLFS Model 2', 'QmRLFS-m2', 30, 'qmrlfs_score', 0.95, 'RIZ detection with 4+ G tracts', 'Jenjaroenpun 2016'),
        ]
    }
    
    # Class 5: Triplex DNA - Three-stranded structures (optimized)
    TRIPLEX_PATTERNS = {
        'triplex_forming_sequences': [
            (r'[GA]{15,}', 'TRX_5_1', 'Homopurine tract', 'Triplex', 15, 'triplex_potential', 0.90, 'H-DNA formation', 'Frank-Kamenetskii 1995'),
            (r'[CT]{15,}', 'TRX_5_2', 'Homopyrimidine tract', 'Triplex', 15, 'triplex_potential', 0.90, 'H-DNA formation', 'Frank-Kamenetskii 1995'),
            (r'(?:GA){6,}[GA]*(?:TC){6,}', 'TRX_5_3', 'Mirror repeat', 'Triplex', 24, 'triplex_potential', 0.85, 'Intermolecular triplex', 'Beal 1996'),
            (r'(?:GAA){4,}', 'TRX_5_4', 'GAA repeat', 'Sticky DNA', 12, 'sticky_dna_score', 0.95, 'Disease-associated repeats', 'Sakamoto 1999'),
            (r'(?:TTC){4,}', 'TRX_5_5', 'TTC repeat', 'Sticky DNA', 12, 'sticky_dna_score', 0.95, 'Disease-associated repeats', 'Sakamoto 1999'),
        ]
    }
    
    # Class 6: G-Quadruplex Family - Updated 2024 (8 subclasses with priorities)
    G_QUADRUPLEX_PATTERNS = {
        'telomeric_g4': [
            (r'(?:TTAGGG){4,}', 'G4_TEL', 'Telomeric G4', 'Telomeric G4', 24, 'g4hunter_score', 0.95, 'Human telomeric G4 structure', 'Parkinson et al., Nature 2002'),
        ],
        'stacked_canonical_g4s': [
            (r'(?:(?:G{3,}[ATGC]{1,7}){3}G{3,}){2,}', 'G4_STK_CAN', 'Stacked canonical G4s', 'Stacked canonical G4s', 30, 'g4hunter_score', 0.90, 'Structural polymorphism & stacking', 'Phan et al., NAR 2007'),
        ],
        'stacked_g4s_linker': [
            (r'(?:(?:G{3,}[ATGC]{1,7}){3}G{3,})(?:[ATGC]{0,20}(?:(?:G{3,}[ATGC]{1,7}){3}G{3,})){1,}', 'G4_STK_LNK', 'Stacked G4s with linker', 'Stacked G4s with linker', 30, 'g4hunter_score', 0.85, 'Clustered G4s in chromatin', 'Hänsel-Hertsch et al., Nat Genet 2016'),
        ],
        'canonical_g4': [
            (r'G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}', 'G4_CAN', 'Canonical intramolecular G4', 'Canonical intramolecular G4', 15, 'g4hunter_score', 0.80, 'Canonical G4 structure', 'Huppert & Balasubramanian, NAR 2005'),
        ],
        'extended_loop_g4': [
            (r'G{3,}[ATGC]{1,12}G{3,}[ATGC]{1,12}G{3,}[ATGC]{1,12}G{3,}', 'G4_EXT', 'Extended-loop canonical', 'Extended-loop canonical', 15, 'g4hunter_score', 0.70, 'G4-seq reveals long-loop G4s', 'Chambers et al., Nat Biotechnol 2015'),
        ],
        'higher_order_g4': [
            (r'(?:G{3,}[ATGC]{1,7}){7,}', 'G4_HIGH', 'Higher-order G4 array/G4-wire', 'Higher-order G4 array/G4-wire', 49, 'g4hunter_score', 0.65, 'Higher-order G4s', 'Wong & Wu, Biochimie 2003'),
        ],
        'g_triplex': [
            (r'G{3,}[ATGC]{1,7}G{3,}[ATGC]{1,7}G{3,}', 'G4_TRX', 'Intramolecular G-triplex', 'Intramolecular G-triplex', 12, 'g_triplex_score', 0.45, 'G-triplex structures', 'Lim & Phan, JACS 2013'),
        ],
        'weak_pqs': [
            (r'G{2,}[ATGC]{1,7}G{2,}[ATGC]{1,7}G{2,}[ATGC]{1,7}G{2,}', 'G4_WEAK', 'Two-tetrad weak PQS', 'Two-tetrad weak PQS', 11, 'g4hunter_score', 0.25, 'QGRS Mapper weak PQS', 'Kikin et al., NAR 2006'),
        ]
    }
    
    # Class 7: i-Motif Family - C-rich structures (3 subclasses)
    I_MOTIF_PATTERNS = {
        'canonical_imotif': [
            (r'C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}', 'IM_7_1', 'Canonical i-motif', 'Canonical i-motif', 15, 'imotif_score', 0.95, 'pH-dependent C-rich structure', 'Gehring 1993'),
            (r'C{4,}[ATGC]{1,5}C{4,}[ATGC]{1,5}C{4,}[ATGC]{1,5}C{4,}', 'IM_7_2', 'High-density i-motif', 'Canonical i-motif', 16, 'imotif_score', 0.98, 'Stable i-motif', 'Leroy 1993'),
        ],
        'relaxed_imotif': [
            (r'C{2,}[ATGC]{1,12}C{2,}[ATGC]{1,12}C{2,}[ATGC]{1,12}C{2,}', 'IM_7_3', 'Relaxed i-motif', 'Relaxed i-motif', 12, 'imotif_score', 0.80, 'Potential i-motif structures', 'Mergny 1995'),
            (r'C{3,}[ATGC]{8,15}C{3,}[ATGC]{1,7}C{3,}[ATGC]{1,7}C{3,}', 'IM_7_4', 'Long-loop i-motif', 'Relaxed i-motif', 18, 'imotif_score', 0.75, 'Alternative i-motif topology', 'Phan 2002'),
        ],
        'ac_motif': [
            (r'(?:AC){4,}|(?:CA){4,}', 'IM_7_5', 'AC-motif', 'AC-motif', 8, 'ac_motif_score', 0.85, 'AC alternating motif', 'Kang 2009'),
            (r'(?:A{2,3}C{2,3}){3,}', 'IM_7_6', 'Extended AC-motif', 'AC-motif', 12, 'ac_motif_score', 0.80, 'Variable AC motif', 'Zhou 2010'),
            (r'(?:ACG){3,}|(?:GCA){3,}', 'IM_7_7', 'ACG-motif', 'AC-motif', 9, 'ac_motif_score', 0.75, 'Trinucleotide AC motif', 'Liu 2012'),
        ]
    }
    
    # Class 8: Z-DNA - Left-handed double helix (2 subclasses)
    Z_DNA_PATTERNS = {
        'z_dna_canonical': [
            (r'(?:CG){4,}|(?:GC){4,}', 'ZDN_8_1', 'CG alternating', 'Z-DNA', 8, 'z_dna_score', 0.95, 'Classical Z-DNA sequence', 'Rich 1984'),
            (r'(?:CA){4,}(?:TG){4,}|(?:TG){4,}(?:CA){4,}', 'ZDN_8_2', 'CA-TG alternating', 'Z-DNA', 16, 'z_dna_score', 0.85, 'Alternative Z-DNA', 'Nordheim 1981'),
            (r'(?:AT){6,}|(?:TA){6,}', 'ZDN_8_3', 'AT alternating', 'Z-DNA', 12, 'z_dna_score', 0.75, 'AT-rich Z-DNA', 'Ellison 1986'),
        ],
        'egz_dna': [
            (r'[CG]{8,}', 'ZDN_8_4', 'CG-rich region', 'eGZ (Extruded-G) DNA', 8, 'egz_score', 0.90, 'Extruded-G Z-DNA', 'Herbert 1997'),
            (r'G{4,}C{4,}|C{4,}G{4,}', 'ZDN_8_5', 'GC clusters', 'eGZ (Extruded-G) DNA', 8, 'egz_score', 0.85, 'Clustered GC Z-DNA', 'Liu 1999'),
        ]
    }
    
    # Class 9: A-philic DNA - A-rich structural motifs
    A_PHILIC_PATTERNS = {
        'a_philic_tracts': [
            (r'A{6,}', 'APH_9_1', 'Poly-A tract', 'A-philic DNA', 6, 'a_philic_score', 0.95, 'A-rich structural element', 'Gorin 1995'),
            (r'(?:A{3,}[AT]){3,}', 'APH_9_2', 'A-rich region', 'A-philic DNA', 12, 'a_philic_score', 0.85, 'Mixed A-rich motif', 'Nelson 1987'),
            (r'(?:AA[AT]){4,}', 'APH_9_3', 'AAT/AAA motif', 'A-philic DNA', 12, 'a_philic_score', 0.80, 'Structured A-rich element', 'Crothers 1990'),
        ]
    }

    @classmethod
    def get_all_patterns(cls) -> Dict[str, Dict[str, List[Tuple]]]:
        """Return complete pattern registry"""
        return {
            'curved_dna': cls.CURVED_DNA_PATTERNS,
            'slipped_dna': cls.SLIPPED_DNA_PATTERNS,  
            'cruciform': cls.CRUCIFORM_PATTERNS,
            'r_loop': cls.R_LOOP_PATTERNS,
            'triplex': cls.TRIPLEX_PATTERNS,
            'g_quadruplex': cls.G_QUADRUPLEX_PATTERNS,
            'i_motif': cls.I_MOTIF_PATTERNS,
            'z_dna': cls.Z_DNA_PATTERNS,
            'a_philic': cls.A_PHILIC_PATTERNS
        }
    
    @classmethod
    def get_pattern_count(cls) -> Dict[str, int]:
        """Get pattern count statistics"""
        all_patterns = cls.get_all_patterns()
        counts = {}
        total = 0
        
        for motif_class, pattern_groups in all_patterns.items():
            class_count = sum(len(patterns) for patterns in pattern_groups.values())
            counts[motif_class] = class_count
            total += class_count
        
        counts['total'] = total
        return counts
    
    @classmethod 
    def get_subclass_mapping(cls) -> Dict[str, List[str]]:
        """Get mapping of classes to subclasses"""
        mapping = {
            'curved_dna': ['Global curvature', 'Local Curvature'],
            'slipped_dna': ['Direct Repeat', 'STR'],
            'cruciform': ['Inverted Repeats'],
            'r_loop': ['R-loop formation sites', 'QmRLFS-m1', 'QmRLFS-m2'],
            'triplex': ['Triplex', 'Sticky DNA'],
            'g_quadruplex': ['Telomeric G4', 'Stacked canonical G4s', 'Stacked G4s with linker', 
                           'Canonical intramolecular G4', 'Extended-loop canonical', 
                           'Higher-order G4 array/G4-wire', 'Intramolecular G-triplex', 
                           'Two-tetrad weak PQS'],
            'i_motif': ['Canonical i-motif', 'Relaxed i-motif', 'AC-motif'],
            'z_dna': ['Z-DNA', 'eGZ (Extruded-G) DNA'],
            'a_philic': ['A-philic DNA'],
            'hybrid': ['Dynamic overlaps'],
            'cluster': ['Dynamic clusters']
        }
        return mapping

# =============================================================================
# SCIENTIFIC SCORING ALGORITHMS
# =============================================================================

class MotifScoring:
    """Comprehensive scoring algorithms for all motif classes"""
    
    @staticmethod
    def g4hunter_score(sequence: str, window_size: int = 25) -> float:
        """
        G4Hunter algorithm for G-quadruplex scoring (Bedrat et al., 2016)
        
        Args:
            sequence: DNA sequence
            window_size: Sliding window size
            
        Returns:
            G4Hunter score (normalized)
        """
        if len(sequence) < window_size:
            window_size = len(sequence)
        
        scores = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            score = 0
            
            for base in window:
                if base == 'G':
                    score += 1
                elif base == 'C':
                    score -= 1
            
            scores.append(score)
        
        if not scores:
            return 0.0
        
        max_score = max(abs(s) for s in scores)
        return max_score / window_size
    
    @staticmethod
    def imotif_score(sequence: str) -> float:
        """
        i-motif scoring based on C-tract analysis (Zeraati et al., 2018)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            i-motif formation score
        """
        if len(sequence) < 12:
            return 0.0
        
        # Count C-tracts
        c_tracts = re.findall(r'C{2,}', sequence)
        if len(c_tracts) < 3:
            return 0.0
        
        # Calculate score based on C-tract density and length
        total_c_length = sum(len(tract) for tract in c_tracts)
        c_density = total_c_length / len(sequence)
        tract_bonus = len(c_tracts) / 4  # Bonus for multiple tracts
        
        return min(c_density + tract_bonus * 0.2, 1.0)
    
    @staticmethod
    def z_dna_score(sequence: str) -> float:
        """
        Z-DNA scoring based on alternating purine-pyrimidine content (Ho et al., 1986)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Z-DNA formation probability
        """
        if len(sequence) < 6:
            return 0.0
        
        alternating_score = 0
        total_pairs = 0
        
        for i in range(len(sequence) - 1):
            curr_base = sequence[i]
            next_base = sequence[i + 1]
            total_pairs += 1
            
            # Score alternating purine-pyrimidine pattern
            if ((curr_base in 'AG' and next_base in 'CT') or 
                (curr_base in 'CT' and next_base in 'AG')):
                alternating_score += 1
                
                # Bonus for CG steps (classical Z-DNA)
                if (curr_base == 'C' and next_base == 'G') or (curr_base == 'G' and next_base == 'C'):
                    alternating_score += 0.5
        
        return alternating_score / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod
    def curvature_score(sequence: str) -> float:
        """
        DNA curvature scoring based on A-tract analysis (Olson et al., 1998)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Intrinsic curvature score
        """
        if len(sequence) < 4:
            return 0.0
        
        # Find A/T tracts
        a_tracts = re.findall(r'A{3,}', sequence)
        t_tracts = re.findall(r'T{3,}', sequence)
        
        # Calculate curvature based on tract length and frequency
        curvature = 0
        for tract in a_tracts + t_tracts:
            curvature += len(tract) ** 1.5  # Non-linear length dependency
        
        return min(curvature / len(sequence), 1.0)
    
    @staticmethod
    def triplex_potential(sequence: str) -> float:
        """
        Triplex formation potential (Frank-Kamenetskii & Mirkin, 1995)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Triplex formation score
        """
        if len(sequence) < 15:
            return 0.0
        
        # Calculate homopurine and homopyrimidine content
        purine_runs = re.findall(r'[AG]{5,}', sequence)
        pyrimidine_runs = re.findall(r'[CT]{5,}', sequence)
        
        # Score based on run lengths and purity
        purine_score = sum(len(run) ** 1.2 for run in purine_runs)
        pyrimidine_score = sum(len(run) ** 1.2 for run in pyrimidine_runs)
        
        max_score = max(purine_score, pyrimidine_score)
        return min(max_score / (len(sequence) ** 1.2), 1.0)
    
    @staticmethod
    def r_loop_potential(sequence: str) -> float:
        """
        R-loop formation potential (Aguilera & García-Muse, 2012)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            R-loop formation score
        """
        if len(sequence) < 20:
            return 0.0
        
        # GC skew calculation
        gc_skew = 0
        for base in sequence:
            if base == 'G':
                gc_skew += 1
            elif base == 'C':
                gc_skew -= 1
        
        # Normalize by sequence length
        gc_skew = abs(gc_skew) / len(sequence)
        
        # GC content (R-loops prefer GC-rich regions)
        gc_content = len(re.findall(r'[GC]', sequence)) / len(sequence)
        
        return min(gc_skew * 0.6 + gc_content * 0.4, 1.0)
    
    @staticmethod
    def qmrlfs_score(sequence: str) -> float:
        """
        QmRLFS-based R-loop formation scoring (Jenjaroenpun & Wongsurawat, 2016)
        
        Implements the QmRLFS algorithm for R-loop forming sequence detection
        based on RIZ (R-loop Initiating Zone) and REZ (R-loop Extending Zone)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            QmRLFS score (0.0-1.0)
        """
        try:
            from qmrlfs_finder import QmRLFSDetector
            
            if len(sequence) < 20:
                return 0.0
            
            # Use quick mode for scoring to avoid performance issues
            detector = QmRLFSDetector(quick_mode=True)
            results = detector.analyze_sequence(sequence, analyze_both_strands=False)
            
            if not results:
                return 0.0
            
            # Return the highest scoring RLFS
            max_score = max(result["qmrlfs_score"] for result in results)
            return max_score
            
        except ImportError:
            # Fallback to simple scoring if QmRLFS module not available
            return MotifScoring.r_loop_potential(sequence)
    
    @staticmethod
    def instability_score(sequence: str) -> float:
        """
        Repeat instability scoring for slipped DNA structures
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Instability score based on repeat characteristics
        """
        if len(sequence) < 6:
            return 0.0
        
        # Find repeating units
        max_instability = 0
        
        for unit_length in range(1, min(7, len(sequence) // 3)):
            for i in range(len(sequence) - unit_length * 2):
                unit = sequence[i:i + unit_length]
                count = 1
                
                # Count consecutive repeats
                pos = i + unit_length
                while pos + unit_length <= len(sequence) and sequence[pos:pos + unit_length] == unit:
                    count += 1
                    pos += unit_length
                
                if count >= 3:  # At least 3 repeats
                    instability = count * (unit_length ** 0.5)
                    max_instability = max(max_instability, instability)
        
        return min(max_instability / 10, 1.0)
    
    @staticmethod
    def cruciform_stability(sequence: str) -> float:
        """
        Cruciform stability based on palindrome characteristics
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Stability score for cruciform formation
        """
        if len(sequence) < 8:
            return 0.0
        
        def reverse_complement(seq):
            complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
            return ''.join(complement.get(base, base) for base in reversed(seq))
        
        max_palindrome = 0
        
        # Look for palindromic regions
        for i in range(len(sequence)):
            for j in range(i + 8, len(sequence) + 1):
                subseq = sequence[i:j]
                if subseq == reverse_complement(subseq):
                    palindrome_length = len(subseq)
                    stability = palindrome_length ** 1.5 / len(sequence)
                    max_palindrome = max(max_palindrome, stability)
        
        return min(max_palindrome, 1.0)
    
    @staticmethod
    def a_philic_score(sequence: str) -> float:
        """
        A-philic DNA scoring based on A-tract characteristics
        
        Args:
            sequence: DNA sequence
            
        Returns:
            A-philic propensity score
        """
        if len(sequence) < 6:
            return 0.0
        
        a_content = len(re.findall(r'A', sequence)) / len(sequence)
        a_tracts = re.findall(r'A{3,}', sequence)
        
        # Score based on A content and tract formation
        tract_bonus = sum(len(tract) ** 1.2 for tract in a_tracts) / len(sequence)
        
        return min(a_content * 0.7 + tract_bonus * 0.3, 1.0)

# =============================================================================
# HYPERSCAN INTEGRATION & OPTIMIZATION
# =============================================================================

class HyperscanManager:
    """Hyperscan database management for high-performance pattern matching"""
    
    def __init__(self):
        self.compiled_db = None
        self.pattern_info = {}
        self.hyperscan_available = _HYPERSCAN_AVAILABLE
    
    def compile_database(self, patterns: List[Tuple[str, str]]) -> bool:
        """
        Compile Hyperscan database from patterns
        
        Args:
            patterns: List of (pattern, identifier) tuples
            
        Returns:
            True if compilation successful
        """
        if not self.hyperscan_available:
            return False
        
        try:
            # Prepare patterns for Hyperscan
            hyperscan_patterns = []
            for i, (pattern, pattern_id) in enumerate(patterns):
                hyperscan_patterns.append((pattern.encode(), i, hyperscan.HS_FLAG_CASELESS))
                self.pattern_info[i] = pattern_id
            
            # Compile database
            self.compiled_db = hyperscan.hs_compile_multi(hyperscan_patterns)
            return True
            
        except Exception as e:
            print(f"Hyperscan compilation failed: {e}")
            return False
    
    def scan_sequence(self, sequence: str) -> List[Tuple[int, int, str]]:
        """
        Scan sequence with compiled Hyperscan database
        
        Args:
            sequence: DNA sequence to scan
            
        Returns:
            List of (start, end, pattern_id) matches
        """
        if not self.compiled_db:
            return []
        
        matches = []
        
        def match_handler(pattern_id: int, start: int, end: int, flags: int, context=None):
            pattern_info = self.pattern_info.get(pattern_id, f'pattern_{pattern_id}')
            matches.append((start, end, pattern_info))
        
        try:
            hyperscan.hs_scan(self.compiled_db, sequence.encode(), match_handler, None)
        except Exception as e:
            print(f"Hyperscan scanning failed: {e}")
        
        return matches

# =============================================================================
# PATTERN VALIDATION & TESTING
# =============================================================================

def validate_all_patterns() -> Dict[str, Any]:
    """
    Validate all patterns for correctness and Hyperscan compatibility
    
    Returns:
        Validation results dictionary
    """
    results = {
        'total_patterns': 0,
        'valid_patterns': 0,
        'invalid_patterns': [],
        'hyperscan_compatible': 0,
        'pattern_counts': {},
        'validation_passed': True
    }
    
    all_patterns = PatternRegistry.get_all_patterns()
    
    for motif_class, pattern_groups in all_patterns.items():
        class_count = 0
        for pattern_group, patterns in pattern_groups.items():
            for pattern_tuple in patterns:
                if len(pattern_tuple) >= 9:  # Full tuple
                    pattern = pattern_tuple[0]
                    pattern_id = pattern_tuple[1]
                    
                    results['total_patterns'] += 1
                    class_count += 1
                    
                    # Test regex compilation
                    try:
                        re.compile(pattern, re.IGNORECASE)
                        results['valid_patterns'] += 1
                        
                        # Test Hyperscan compatibility
                        if not any(incompatible in pattern for incompatible in ['\\b', '\\B', '(?=', '(?!', '(?<=', '(?<!', '\\1', '\\2']):
                            results['hyperscan_compatible'] += 1
                        
                    except re.error as e:
                        results['invalid_patterns'].append((motif_class, pattern_id, str(e)))
                        results['validation_passed'] = False
        
        results['pattern_counts'][motif_class] = class_count
    
    return results

def run_pattern_tests() -> bool:
    """Run comprehensive pattern tests"""
    print("Running Non-B DNA Pattern Validation...")
    
    # Test sequences for each class
    test_sequences = {
        'curved_dna': 'AAAAAAAATTTTTTTAAAAATTTT',  # A/T tracts
        'slipped_dna': 'CACACACACACACACACACA',     # CA repeats
        'cruciform': 'ATGCATGCATGCATGC',          # Palindrome
        'r_loop': 'GGGCCCGGGATGCCCGGG',           # GC-rich R-loop site
        'triplex': 'GAAAGAAAGAAAGAAAGAAA',        # Homopurine tract
        'g_quadruplex': 'GGGTTAGGGTTAGGGTTAGGG',   # Canonical G4
        'i_motif': 'CCCTAACCCTAACCCTAACCC',       # Canonical i-motif
        'z_dna': 'CGCGCGCGCGCGCGCGCG',            # CG alternating
        'a_philic': 'AAAAAAAAAAAAAAAAA'           # Poly-A
    }
    
    validation_results = validate_all_patterns()
    print(f"Pattern validation: {validation_results['valid_patterns']}/{validation_results['total_patterns']} patterns valid")
    print(f"Hyperscan compatible: {validation_results['hyperscan_compatible']}/{validation_results['total_patterns']} patterns")
    
    # Test pattern matching
    all_patterns = PatternRegistry.get_all_patterns()
    # Instantiate scoring system to validate initialization
    scoring = MotifScoring()
    
    test_results = {}
    for motif_class, test_seq in test_sequences.items():
        if motif_class in all_patterns:
            matches_found = 0
            
            for pattern_group, patterns in all_patterns[motif_class].items():
                for pattern_tuple in patterns:
                    if len(pattern_tuple) >= 3:
                        pattern = pattern_tuple[0]
                        try:
                            compiled_pattern = re.compile(pattern, re.IGNORECASE)
                            if compiled_pattern.search(test_seq):
                                matches_found += 1
                                break
                        except:
                            continue
                            
                if matches_found > 0:
                    break
            
            test_results[motif_class] = matches_found > 0
            print(f"[OK] {motif_class}: {'PASS' if matches_found > 0 else 'FAIL'}")
    
    all_passed = all(test_results.values()) and validation_results['validation_passed']
    print(f"\nOverall validation: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

# =============================================================================
# HYPERSCAN REGISTRY LOADER INTEGRATION
# =============================================================================

try:
    # from .load_hsdb import load_db_for_class
    _LOAD_HSDB_AVAILABLE = True
except ImportError:
    try:
        # Fallback: direct import
        import sys
        import os as _os
        sys.path.insert(0, _os.path.dirname(__file__))
        from load_hsdb import load_db_for_class
        _LOAD_HSDB_AVAILABLE = True
    except ImportError:
        _LOAD_HSDB_AVAILABLE = False
        load_db_for_class = None

# In-memory cache to avoid repeated compiles/deserializes
_HS_DB_CACHE = {}
_REGISTRY_CACHE = {}


def get_pattern_registry(class_name: str, registry_dir: str = "registry"):
    """
    Returns parsed registry dict (as saved by generator): contains 'patterns' list etc.
    Caches the result.
    """
    if not _LOAD_HSDB_AVAILABLE:
        raise ImportError("load_hsdb module not available")
    
    key = f"{registry_dir}/{class_name}"
    if key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[key]
    
    # Read registry directly from file
    import pickle
    import json
    pkl_path = os.path.join(registry_dir, f"{class_name}_registry.pkl")
    json_path = os.path.join(registry_dir, f"{class_name}_registry.json")
    
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as fh:
            full = pickle.load(fh)
    elif os.path.isfile(json_path):
        with open(json_path, "r") as fh:
            full = json.load(fh)
    else:
        raise FileNotFoundError(f"No registry found for {class_name} in {registry_dir}")
    
    _REGISTRY_CACHE[key] = full
    return full


def get_hs_db_for_class(class_name: str, registry_dir: str = "registry"):
    """
    Return (db, id_to_pattern, id_to_score). Caches DB in process memory.
    db is a hyperscan.Database instance (or None if hyperscan not available).
    id_to_pattern: dict mapping id -> pattern string (tenmer for 10-mer, regex for others)
    """
    if not _LOAD_HSDB_AVAILABLE:
        raise ImportError("load_hsdb module not available")
    
    key = f"{registry_dir}/{class_name}"
    if key in _HS_DB_CACHE:
        return _HS_DB_CACHE[key]
    
    db, id_to_pattern, id_to_score = load_db_for_class(class_name, registry_dir)
    _HS_DB_CACHE[key] = (db, id_to_pattern, id_to_score)
    return db, id_to_pattern, id_to_score


# =============================================================================
# PATTERN STATISTICS & INFORMATION
# =============================================================================
