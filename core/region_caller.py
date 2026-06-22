"""
core/region_caller.py
─────────────────────
Bounded minimum-mean contiguous-subarray region caller (Kadane family, O(n)).

ALGORITHM OVERVIEW
──────────────────
We want the contiguous stretch of the perplexity-residual signal with the
**minimum mean** value, subject to min_len ≤ length ≤ max_len.  This is the
"bounded-length minimum-average subarray" problem, solved in O(n) via a
monotonic deque over the prefix-sum array — the provably-optimal technique
for this problem class.

Key detail: minimum-SUM (plain Kadane) is NOT used here because residuals
hover near zero almost everywhere — an unbounded minimum-sum search would
greedily swallow the entire sequence.  Minimum-MEAN with both a floor
(min_len) and a ceiling (max_len) is the correct, well-posed formulation.

NaN segments (from N-containing windows) are treated as hard breaks: each
finite run is processed independently so a single N cannot bridge two
unrelated regions.

After each region is found, its span is masked to NaN and the scan repeats
to discover the next best non-overlapping region (up to top_k).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .motifs import MOTIF_PATTERNS


def min_mean_subarray_bounded(
    arr: np.ndarray,
    min_len: int,
    max_len: int,
) -> tuple[Any, Any, float]:
    """
    Return (start, end_inclusive, mean) of the contiguous run with the
    smallest MEAN value, subject to min_len ≤ length ≤ max_len.

    Returns (None, None, inf) if the array is shorter than min_len.
    """
    n = len(arr)
    if n < min_len:
        return None, None, np.inf

    cums = np.empty(n + 1, dtype=np.float64)
    cums[0] = 0.0
    cums[1:] = np.cumsum(arr)

    best_mean: float = np.inf
    best_i = best_j = -1
    dq: deque[int] = deque()

    for j in range(n):
        i_min = j - max_len + 1
        i_max = j - min_len + 1
        if i_max < 0:
            continue
        # Maintain deque as indices with monotonically increasing cums[i]
        while dq and cums[dq[-1]] >= cums[i_max]:
            dq.pop()
        dq.append(i_max)
        while dq and dq[0] < i_min:
            dq.popleft()
        if dq:
            i = dq[0]
            length = j - i + 1
            mean_val = (cums[j + 1] - cums[i]) / length
            if mean_val < best_mean:
                best_mean = mean_val
                best_i, best_j = i, j

    return best_i, best_j, best_mean


def find_regions(
    residual: np.ndarray,
    seq: str,
    *,
    min_len: int = 50,
    max_len: int = 300,
    top_k: int = 5,
    score_cutoff: float = -0.05,
    window: int = 10,
    active_motifs: set[str] | None = None,
) -> list[dict]:
    """
    Iteratively find up to *top_k* non-overlapping low-perplexity regions.

    Parameters
    ----------
    residual : np.ndarray
        Perplexity-residual signal (may contain NaNs).
    seq : str
        Corresponding DNA sequence (used for GC% and motif annotation).
        Must be long enough that ``seq[gs : ge + window]`` is valid.
    min_len : int
        Minimum region length in bp (default 50).
    max_len : int
        Maximum region length in bp (default 300).
    top_k : int
        Maximum number of regions to report (default 5).
    score_cutoff : float
        Regions with mean residual ≥ this value are not reported (default -0.05).
    window : int
        Perplexity window size, used to extend the sequence slice for motif
        annotation (default 10).
    active_motifs : set[str] | None
        Set of motif keys to annotate.  None or empty set disables annotation.

    Returns
    -------
    list of dict, each with keys:
        rank, start, end, width, trough, mean_residual, gc_pct, motifs
    """
    arr = residual.copy().astype(np.float64)
    n = len(arr)
    nan_mask = np.isnan(arr)
    regions: list[dict] = []

    for rank in range(1, top_k + 1):
        best_overall: tuple[float, Any, Any] = (np.inf, None, None)
        i = 0
        while i < n:
            if nan_mask[i]:
                i += 1
                continue
            j = i
            while j < n and not nan_mask[j]:
                j += 1
            seg = arr[i:j]
            if len(seg) >= min_len:
                s, e, m = min_mean_subarray_bounded(seg, min_len, max_len)
                if s is not None and m < best_overall[0]:
                    best_overall = (m, i + s, i + e)
            i = j

        m, gs, ge = best_overall
        if gs is None or m >= score_cutoff:
            break

        width = ge - gs + 1
        trough_i = gs + int(np.nanargmin(arr[gs : ge + 1]))
        seq_end = min(ge + window, len(seq))
        reg_seq = seq[gs:seq_end]

        gc_pct = round(
            100.0 * (reg_seq.count("G") + reg_seq.count("C")) / max(len(reg_seq), 1),
            2,
        )

        # Motif annotation
        motif_hits: list[str] = []
        if active_motifs:
            for name, pat in MOTIF_PATTERNS.items():
                if name in active_motifs and pat.search(reg_seq):
                    motif_hits.append(name)

        regions.append(
            {
                "rank": rank,
                "start": int(gs),
                "end": int(ge),
                "width": width,
                "trough": int(trough_i),
                "mean_residual": round(float(m), 5),
                "gc_pct": gc_pct,
                "motifs": ";".join(motif_hits),
            }
        )

        # Mask region out so it is not re-picked in subsequent iterations
        arr[gs : ge + 1] = np.nan
        nan_mask[gs : ge + 1] = True

    return regions
