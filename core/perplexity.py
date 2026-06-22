"""
core/perplexity.py
──────────────────
Dinucleotide perplexity calculation and local-baseline residual estimation.

The algorithm is a vectorised O(n) sliding-window implementation:
  1. Map each nucleotide to an integer index (A=0, C=1, G=2, T=3; N→masked).
  2. For every window of length *window*, tally the 16 possible dinucleotide
     counts using stride-trick views.
  3. Compute the Shannon entropy H = -Σ p·log₂(p) over those counts and
     exponentiate to obtain perplexity = 2^H.
  4. Any window containing an N is set to NaN.

The local residual is perplexity minus a rolling median-window mean, which
adapts to GC-content drift along the sequence without hard thresholding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Lookup table: ASCII byte → dinucleotide index (0–3 for ACGT, 4 for N/other)
_MAP = np.full(256, 4, dtype=np.uint8)
for _b, _i in zip("ACGT", range(4)):
    _MAP[ord(_b)] = _i


def compute_perplexity(seq: str, window: int = 10) -> np.ndarray:
    """
    Return a float32 array of sliding-window dinucleotide perplexity values.

    Length = len(seq) - window + 1.  Windows that contain any N are NaN.

    Parameters
    ----------
    seq : str
        DNA sequence (uppercase A/C/G/T/N).
    window : int
        Sliding window size (default 10).

    Returns
    -------
    np.ndarray, dtype=float32, shape=(len(seq)-window+1,)
    """
    if len(seq) < window:
        return np.full(max(0, len(seq) - window + 1), np.nan, dtype=np.float32)

    x = _MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = x == 4
    x_c = np.where(has_n, 0, x)

    w = sliding_window_view(x_c, window)          # shape (n, window)
    wn = sliding_window_view(has_n, window)        # shape (n, window)

    # Dinucleotide indices for consecutive pairs within each window
    a, b_ = w[:, :-1], w[:, 1:]
    din = (a * 4 + b_).astype(np.int16)            # values 0–15

    n = len(w)
    counts = np.zeros((n, 16), dtype=np.int16)
    np.add.at(counts, (np.arange(n)[:, None], din), 1)

    # Probabilities: if count==0 treat as p=1 so that -p·log2(p) contributes 0
    # Zero counts: set p=1 so that -p·log₂(p) contributes 0 (no information from absent pairs)
    p = np.where(counts == 0, 1.0, counts / (window - 1)).astype(np.float32)
    H = -np.sum(p * np.log2(p), axis=1)
    perp = (2.0 ** H).astype(np.float32)
    perp[wn.any(axis=1)] = np.nan
    return perp


def local_residual(perp: np.ndarray, baseline_win: int = 200) -> np.ndarray:
    """
    Compute perplexity residual against a centred rolling baseline.

    ``min_periods=baseline_win`` is intentional: positions near sequence edges
    where a full-width baseline cannot be estimated are masked to NaN, preventing
    artefactual troughs caused by noisy edge estimates.

    Parameters
    ----------
    perp : np.ndarray
        Perplexity array (may contain NaNs).
    baseline_win : int
        Rolling window width for the local baseline (default 200).

    Returns
    -------
    np.ndarray of same shape as *perp*.
    """
    bl = (
        pd.Series(perp)
        .rolling(baseline_win, center=True, min_periods=baseline_win)
        .mean()
        .values
    )
    return perp - bl
