from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from src.models.dataclasses import PerplexityConfig, PerplexityProfile
from src.perplexity.dinucleotide import encode_dinucleotides, has_ambiguous_base
from src.perplexity.entropy import perplexity_from_entropy, shannon_entropy


def calculate_perplexity_profile(sequence: str, config: PerplexityConfig) -> PerplexityProfile:
    if len(sequence) < config.perplexity_window:
        empty = np.array([], dtype=np.float32)
        return PerplexityProfile(positions=np.array([], dtype=np.int64), raw_perplexity=empty, smoothed_perplexity=empty)

    ambiguous = has_ambiguous_base(sequence, config.perplexity_window, config.step_size)
    dinucleotide_indices = encode_dinucleotides(sequence, config.perplexity_window, config.step_size)

    counts = np.zeros((len(dinucleotide_indices), 16), dtype=np.int16)
    valid_rows = np.flatnonzero(~ambiguous)
    if valid_rows.size:
        np.add.at(counts, (valid_rows[:, None], dinucleotide_indices[valid_rows]), 1)

    probs = counts / max(config.perplexity_window - 1, 1)
    entropy = shannon_entropy(probs)
    raw = perplexity_from_entropy(entropy).astype(np.float32)
    raw[ambiguous] = np.nan

    smoothed = smooth_profile(raw, config.smoothing_window, config.smoothing_poly_order)
    positions = np.arange(len(raw), dtype=np.int64) * config.step_size
    return PerplexityProfile(positions=positions, raw_perplexity=raw, smoothed_perplexity=smoothed)


def smooth_profile(values: np.ndarray, window: int, poly_order: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values.copy()
    wl = min(window, n)
    if wl % 2 == 0:
        wl -= 1
    po = min(poly_order, wl - 1)
    if wl < 3 or po < 1:
        return values.astype(np.float32)

    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()

    work = values.astype(np.float64)
    if not np.all(finite):
        xp = np.flatnonzero(finite)
        fp = values[finite].astype(np.float64)
        work = np.interp(np.arange(n, dtype=np.float64), xp, fp)

    smoothed = savgol_filter(work, wl, po).astype(np.float32)
    smoothed[~finite] = np.nan
    return smoothed
