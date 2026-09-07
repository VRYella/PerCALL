from __future__ import annotations

import numpy as np

from src.models.dataclasses import CandidateRegion, PerplexityConfig, PredictionResult
from src.perplexity.profile import calculate_perplexity_profile
from src.prediction.background import estimate_local_background
from src.prediction.depression import calculate_perplexity_depression
from src.prediction.ranking import rank_regions
from src.prediction.regions import detect_candidate_intervals, merge_intervals
from src.preprocessing.sequence import clean_sequence


def _interval_span_bp(start_w: int, end_w: int, config: PerplexityConfig) -> int:
    n_windows = end_w - start_w + 1
    if n_windows <= 0:
        return 0
    return int(config.perplexity_window + (n_windows - 1) * config.step_size)




def _longest_contiguous_run(mask: np.ndarray) -> int:
    max_run = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


def _bound_interval_by_max_pds(
    interval: tuple[int, int],
    pds: np.ndarray,
    config: PerplexityConfig,
) -> tuple[int, int] | None:
    start_w, end_w = interval
    span_bp = _interval_span_bp(start_w, end_w, config)
    if span_bp <= config.max_region_length:
        return interval

    max_windows = max(1, (config.max_region_length - config.perplexity_window) // config.step_size + 1)
    if max_windows <= 0:
        return None

    best: tuple[int, int] | None = None
    best_score = -np.inf
    for left in range(start_w, end_w - max_windows + 2):
        right = left + max_windows - 1
        seg = pds[left:right + 1]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            continue
        score = float(np.mean(finite))
        if score > best_score:
            best_score = score
            best = (left, right)
    return best


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
        window_size=config.perplexity_window,
    )
    intervals = merge_intervals(intervals, config.merge_distance, config.step_size)

    bounded: list[tuple[int, int]] = []
    for interval in intervals:
        bounded_interval = _bound_interval_by_max_pds(interval, pds, config)
        if bounded_interval is not None:
            bounded.append(bounded_interval)

    regions: list[CandidateRegion] = []
    for start_w, end_w in bounded:
        sl = slice(start_w, end_w + 1)
        pds_slice = pds[sl]
        perplex_slice = profile.smoothed_perplexity[sl]
        bg_slice = background[sl]

        finite_pds = pds_slice[np.isfinite(pds_slice)]
        finite_perp = perplex_slice[np.isfinite(perplex_slice)]
        finite_bg = bg_slice[np.isfinite(bg_slice)]

        span_bp = _interval_span_bp(start_w, end_w, config)
        start_bp = int(start_w * config.step_size)
        end_bp_inclusive = min(len(clean) - 1, start_bp + span_bp - 1)

        persistence_mask = pds_slice >= config.min_perplexity_depression
        persistence_windows = _longest_contiguous_run(persistence_mask)
        persistence_bp = int(config.perplexity_window + (persistence_windows - 1) * config.step_size) if persistence_windows else 0

        regions.append(CandidateRegion(
            start=start_bp,
            end=end_bp_inclusive,
            length=max(0, end_bp_inclusive - start_bp + 1),
            mean_perplexity=float(np.mean(finite_perp)) if finite_perp.size else float("nan"),
            background_perplexity=float(np.mean(finite_bg)) if finite_bg.size else float("nan"),
            mean_pds=float(np.mean(finite_pds)) if finite_pds.size else 0.0,
            max_pds=float(np.max(finite_pds)) if finite_pds.size else 0.0,
            persistence=persistence_bp,
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
