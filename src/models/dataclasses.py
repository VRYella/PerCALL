from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PerplexityConfig:
    perplexity_window: int = 17
    step_size: int = 1
    smoothing_window: int = 21
    smoothing_poly_order: int = 3
    flank_size: int = 100
    min_region_length: int = 100
    max_region_length: int = 1000
    min_perplexity_depression: float = 0.25
    min_persistence_bp: int = 80
    merge_distance: int = 100


@dataclass(frozen=True)
class PerplexityProfile:
    positions: np.ndarray
    raw_perplexity: np.ndarray
    smoothed_perplexity: np.ndarray


@dataclass
class CandidateRegion:
    start: int
    end: int
    length: int
    mean_perplexity: float
    background_perplexity: float
    mean_pds: float
    max_pds: float
    persistence: int
    rank: int = 0
    multiscale_support: tuple[int, int] | None = None
    motif_count: int = 0
    motifs: str = ""


@dataclass
class PredictionResult:
    sequence_id: str
    sequence_length: int
    profile: PerplexityProfile
    background: np.ndarray
    pds: np.ndarray
    candidate_regions: list[CandidateRegion] = field(default_factory=list)
