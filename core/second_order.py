"""
core/second_order.py
────────────────────
Second-order perplexity (P2) computation.

PURPOSE
───────
Quantify the *stability* of the perplexity landscape by measuring how
diverse (spread) the P1 values are within a sliding window.

ALGORITHM
─────────
1. Pre-bin every P1 value into ``n_bins`` fixed-width bins spanning the
   canonical P1 range [1, 9].
2. For each window of length ``window`` (default 100 P1 positions), derive
   the bin-count histogram using cumulative sums — a single vectorised
   O(n) pass per bin (only ``n_bins`` such passes, so O(n × n_bins)).
3. Compute Shannon entropy  H = −Σ p · log₂(p)  over the normalised
   histogram.
4. Second-order perplexity  P2 = 2ᴴ.

The returned array has the **same length as the input P1 array**: positions
where the full window cannot be formed (the first and last ``⌊window/2⌋``
positions) are set to NaN.  Windows containing > 20 % NaN P1 values are
also set to NaN.

PERFORMANCE
───────────
No Python loops over sequence positions.  The inner ``n_bins``-iteration
loop runs exactly 16 times regardless of sequence length.  For a human
chromosome (~250 M P1 positions) the total work is ≈ 16 × 250 M = 4 B
element operations — well within NumPy's vectorised throughput.
"""

from __future__ import annotations

import numpy as np


def compute_second_order_perplexity(
    p1: np.ndarray,
    window: int = 100,
    n_bins: int = 16,
) -> np.ndarray:
    """
    Compute second-order perplexity P2 from a first-order perplexity profile.

    Parameters
    ----------
    p1 : np.ndarray
        First-order perplexity array (float32, may contain NaN).
        Typical values lie in [1, 9].
    window : int
        Sliding window size in P1 positions (default 100).
    n_bins : int
        Number of histogram bins for entropy estimation (default 16).

    Returns
    -------
    np.ndarray, dtype=float32, same length as *p1*.
        NaN where the full window is unavailable or too sparse.

    Interpretation
    --------------
    Low P2  → stable perplexity landscape (narrow P1 distribution in window).
    High P2 → chaotic perplexity landscape (broad P1 distribution in window).
    """
    n = len(p1)
    result = np.full(n, np.nan, dtype=np.float32)

    n_windows = n - window + 1
    if n_windows <= 0:
        return result

    valid = np.isfinite(p1)

    # ── Fixed bin edges spanning the canonical P1 range ──────────────────
    _P1_MIN = 1.0
    _P1_MAX = 9.0 + 1e-6          # open right end so max value falls inside
    edges_inner = np.linspace(_P1_MIN, _P1_MAX, n_bins + 1)[1:-1]  # n_bins-1 thresholds

    p1_clipped = np.clip(p1, _P1_MIN, _P1_MAX - 1e-9)

    # Bin index for every position: 0 … n_bins-1 for valid; -1 sentinel for NaN
    binned = np.digitize(p1_clipped, edges_inner).astype(np.int32)   # 0 … n_bins-1
    binned[~valid] = -1

    # ── Cumulative bin counts (n_bins passes, each O(n)) ─────────────────
    # cum_b[b, i] = number of P1 values in bin b among positions 0 … i-1
    cum = np.empty((n_bins, n + 1), dtype=np.int32)
    cum[:, 0] = 0
    for b in range(n_bins):
        cum[b, 1:] = np.cumsum(binned == b)

    # Window counts: shape (n_bins, n_windows)
    win_counts = (cum[:, window:] - cum[:, :n_windows]).astype(np.float32)

    # Number of valid P1 values in each window
    total_valid = win_counts.sum(axis=0)   # shape (n_windows,)

    # Mask windows that are too sparse (< 80 % valid)
    sparse = total_valid < (0.8 * window)

    # ── Shannon entropy ───────────────────────────────────────────────────
    safe_total = np.where(total_valid > 0, total_valid, 1.0)
    p = win_counts / safe_total[np.newaxis, :]        # (n_bins, n_windows)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log2(p), 0.0)
    H = -np.sum(p * log_p, axis=0).astype(np.float32)  # (n_windows,)

    p2_core = (2.0 ** H).astype(np.float32)
    p2_core[sparse] = np.nan

    # ── Pad to original P1 length ─────────────────────────────────────────
    pad_left = (window - 1) // 2
    result[pad_left: pad_left + n_windows] = p2_core
    return result


def second_order_residual(
    p2: np.ndarray,
    baseline_win: int = 200,
) -> np.ndarray:
    """
    Compute P2 residual using the same centred rolling-baseline strategy
    as ``core.perplexity.local_residual``.

    Parameters
    ----------
    p2 : np.ndarray
        Second-order perplexity array (may contain NaN).
    baseline_win : int
        Rolling window width for the local baseline (default 200).

    Returns
    -------
    np.ndarray of same shape as *p2*.
    """
    import pandas as pd  # local import to avoid top-level dependency at module load

    bl = (
        pd.Series(p2)
        .rolling(baseline_win, center=True, min_periods=baseline_win)
        .mean()
        .values
    )
    return p2 - bl
