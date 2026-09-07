from __future__ import annotations

import numpy as np

from src.models.dataclasses import CandidateRegion, PerplexityConfig, PredictionResult
from src.perplexity.profile import calculate_perplexity_profile
from src.prediction.background import estimate_local_background
from src.prediction.depression import calculate_perplexity_depression
from src.prediction.ranking import rank_regions
from src.prediction.regions import detect_candidate_intervals, merge_intervals
from src.preprocessing.sequence import clean_sequence


def predict_regulatory_regions(sequence_id: str, sequence: str, config: PerplexityConfig) -> PredictionResult:
    clean = clean_sequence(sequence)
    profile = calculate_perplexity_profile(clean, config)
    background = estimate_local_background(profile.smoothed_perplexity, config.flank_size)
    pds = calculate_perplexity_depression(profile.smoothed_perplexity, background, normalize=False)

    intervals = detect_candidate_intervals(
        pds=pds,
        min_pds=config.min_perplexity_depression,
        min_region_length=config.min_region_length,
        min_persistence_bp=config.min_persistence_bp,
        step_size=config.step_size,
    )
    intervals = [iv for iv in intervals if (iv[1] - iv[0] + 1) * config.step_size <= config.max_region_length]
    intervals = merge_intervals(intervals, config.merge_distance, config.step_size)
    intervals = [iv for iv in intervals if (iv[1] - iv[0] + 1) * config.step_size <= config.max_region_length]

    regions: list[CandidateRegion] = []
    for start_w, end_w in intervals:
        sl = slice(start_w, end_w + 1)
        pds_slice = pds[sl]
        perplex_slice = profile.smoothed_perplexity[sl]
        bg_slice = background[sl]

        finite_pds = pds_slice[np.isfinite(pds_slice)]
        finite_perp = perplex_slice[np.isfinite(perplex_slice)]
        finite_bg = bg_slice[np.isfinite(bg_slice)]

        start_bp = int(start_w * config.step_size)
        end_bp = int(end_w * config.step_size + config.perplexity_window)
        persistence = int(np.sum(pds_slice >= config.min_perplexity_depression) * config.step_size)

        regions.append(CandidateRegion(
            start=start_bp,
            end=min(end_bp, len(clean)),
            length=max(0, min(end_bp, len(clean)) - start_bp),
            mean_perplexity=float(np.mean(finite_perp)) if finite_perp.size else float("nan"),
            background_perplexity=float(np.mean(finite_bg)) if finite_bg.size else float("nan"),
            mean_pds=float(np.mean(finite_pds)) if finite_pds.size else 0.0,
            max_pds=float(np.max(finite_pds)) if finite_pds.size else 0.0,
            persistence=persistence,
        ))

    ranked = rank_regions(regions)
    return PredictionResult(
        sequence_id=sequence_id,
        sequence_length=len(clean),
        profile=profile,
        background=background,
        pds=pds,
        candidate_regions=ranked,
    )
