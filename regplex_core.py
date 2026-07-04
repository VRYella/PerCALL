from __future__ import annotations

import argparse
import io
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---------------------------
# Tunable scientific defaults
# ---------------------------
PERPLEXITY_WINDOW = 10
DEFAULT_SCALES = [25, 50, 100, 200, 400]
LANDSCAPE_METHOD = "mean"  # mean | median
NORMALIZATION_METHOD = "robust_z"  # robust_z | percentile
ENSEMBLE_METHOD = "median"  # median | trimmed_mean
DEFAULT_MODE = "promoter"   # promoter | genome | ensemble
CORE_WINDOW_UPSTREAM = 500
CORE_WINDOW_DOWNSTREAM = 200
MIN_CANDIDATE = 50      # Kadane refinement minimum length (bp)
MAX_CANDIDATE = 1000    # Kadane refinement maximum length (bp)
MIN_DOMAIN = 50         # Final valley minimum length (bp)
MAX_DOMAIN = 1000       # Final valley maximum length (bp)
MERGE_GAP = 25          # Gap for post-NMS valley merging (bp)
SG_WINDOW_LENGTH = 21   # Savitzky-Golay window length (must be odd)
SG_POLY_ORDER = 3       # Savitzky-Golay polynomial order
PERSISTENCE_THRESHOLD = 0.80  # Hard persistence filter (fraction of positions with ConsensusLPC > 0)
NMS_OVERLAP_THRESHOLD = 0.50  # Non-maximum suppression overlap fraction
TOP_N_DISPLAY = 20      # Default number of top valleys to display
FLANK_WINDOW = 100      # Flanking window size for prominence computation (bp)
EPSILON = 1e-9

# Legacy VPI constants retained for compatibility with optional visualization helpers.
MAX_INTERNAL_GAP = 20
VPI_CANDIDATE_THRESHOLD = 0.6
VPI_EXPAND_THRESHOLD = 0.3

LAYERS = ("mono", "di", "tri")
_LAYER_LABELS = {"mono": "Mono", "di": "Di", "tri": "Tri"}
OPERATING_MODES = ("promoter", "genome", "ensemble")

_IUPAC_DNA = set("ACGTN")
_MAP = np.full(256, 4, dtype=np.uint8)
for base, idx in zip("ACGT", range(4)):
    _MAP[ord(base)] = idx


@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    mono: np.ndarray
    di: np.ndarray
    tri: np.ndarray
    smoothed_mono: np.ndarray
    smoothed_di: np.ndarray
    smoothed_tri: np.ndarray
    landscapes: dict[str, dict[int, np.ndarray]]
    lpc_profiles: dict[str, dict[int, np.ndarray]]
    layer_consensus: dict[str, np.ndarray]
    consensus_lpc: np.ndarray
    candidates: list[dict]
    domains: list[dict]
    params: dict
    kadane_core: tuple[int | None, int | None]


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = "".join(chunks).upper().replace("U", "T")
                seq = "".join(c for c in seq if c in _IUPAC_DNA)
                if seq:
                    records.append((header, seq))
            header = line[1:].strip() or "sequence"
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        seq = "".join(chunks).upper().replace("U", "T")
        seq = "".join(c for c in seq if c in _IUPAC_DNA)
        if seq:
            records.append((header, seq))
    return records


def _prefix(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(arr).astype(np.float64)
    values = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    values2 = np.where(np.isfinite(arr), arr.astype(np.float64) ** 2, 0.0)
    csum = np.empty(len(arr) + 1, dtype=np.float64)
    csum2 = np.empty(len(arr) + 1, dtype=np.float64)
    ccount = np.empty(len(arr) + 1, dtype=np.float64)
    csum[0] = csum2[0] = ccount[0] = 0.0
    np.cumsum(values, out=csum[1:])
    np.cumsum(values2, out=csum2[1:])
    np.cumsum(valid, out=ccount[1:])
    return csum, csum2, ccount


def _window_means_from_prefix(
    csum: np.ndarray,
    ccount: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    out = np.full(starts.shape, np.nan, dtype=np.float64)
    limit = len(csum) - 1
    valid = (starts >= 0) & (ends <= limit) & (ends > starts)
    if not np.any(valid):
        return out
    s = starts[valid]
    e = ends[valid]
    counts = ccount[e] - ccount[s]
    pos = counts > 0
    if not np.any(pos):
        return out
    idx = np.flatnonzero(valid)[pos]
    out[idx] = (csum[e[pos]] - csum[s[pos]]) / counts[pos]
    return out


def _window_medians(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window < 1 or n < window:
        return out
    views = np.lib.stride_tricks.sliding_window_view(arr, window)
    med = np.nanmedian(views, axis=1)
    offset = window // 2
    out[offset: offset + len(med)] = med
    return out


def _compute_entropy_perplexity(indices: np.ndarray, n_states: int) -> np.ndarray:
    """Return window-wise perplexity from discrete state indices.

    indices: (n_windows, n_events_per_window) encoded symbols per window.
    n_states: size of the symbol alphabet (4 mono, 16 di, 64 tri).
    """
    if len(indices) == 0:
        return np.array([], dtype=np.float32)
    counts = np.zeros((len(indices), n_states), dtype=np.int16)
    np.add.at(counts, (np.arange(len(indices))[:, None], indices), 1)
    denom = np.maximum(indices.shape[1], 1)
    p = np.where(counts == 0, 1.0, counts / denom).astype(np.float32)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return (2.0 ** entropy).astype(np.float32)


def compute_mono_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    if len(seq) < window:
        return np.array([], dtype=np.float32)
    arr = _MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = arr == 4
    masked = np.where(has_n, 0, arr)
    windows = np.lib.stride_tricks.sliding_window_view(masked, window)
    windows_n = np.lib.stride_tricks.sliding_window_view(has_n, window)
    p = _compute_entropy_perplexity(windows.astype(np.int16), 4)
    p[windows_n.any(axis=1)] = np.nan
    return p


def compute_di_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    if len(seq) < window:
        return np.array([], dtype=np.float32)
    arr = _MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = arr == 4
    masked = np.where(has_n, 0, arr)
    windows = np.lib.stride_tricks.sliding_window_view(masked, window)
    windows_n = np.lib.stride_tricks.sliding_window_view(has_n, window)
    a, b = windows[:, :-1], windows[:, 1:]
    din = (a * 4 + b).astype(np.int16)
    p = _compute_entropy_perplexity(din, 16)
    p[windows_n.any(axis=1)] = np.nan
    return p


def compute_tri_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    if len(seq) < window:
        return np.array([], dtype=np.float32)
    arr = _MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = arr == 4
    masked = np.where(has_n, 0, arr)
    windows = np.lib.stride_tricks.sliding_window_view(masked, window)
    windows_n = np.lib.stride_tricks.sliding_window_view(has_n, window)
    a, b, c = windows[:, :-2], windows[:, 1:-1], windows[:, 2:]
    tri = (a * 16 + b * 4 + c).astype(np.int16)
    p = _compute_entropy_perplexity(tri, 64)
    p[windows_n.any(axis=1)] = np.nan
    return p


def smooth_perplexity(
    arr: np.ndarray,
    window_length: int = SG_WINDOW_LENGTH,
    poly_order: int = SG_POLY_ORDER,
) -> np.ndarray:
    """Apply a single Savitzky-Golay pass to a perplexity profile.

    Preserves valley position, depth and boundary shape while removing local
    noise.  NaN positions are reconstructed via linear interpolation before
    smoothing, then restored after.  If the array is too short for the
    requested window, the original array is returned unchanged.
    """
    n = len(arr)
    if n == 0:
        return arr.copy()

    # Ensure window_length is odd and does not exceed the array length.
    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1
    # polynomial order must be strictly less than window length
    po = min(poly_order, wl - 1)
    if wl < 3 or po < 1:
        return arr.astype(np.float32)

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return arr.copy()

    # Interpolate NaN so savgol_filter receives a complete array.
    work = arr.astype(np.float64)
    if not np.all(finite_mask):
        xp = np.flatnonzero(finite_mask)
        fp = arr[finite_mask].astype(np.float64)
        work = np.interp(np.arange(n, dtype=np.float64), xp, fp)

    smoothed = savgol_filter(work, wl, po).astype(np.float32)
    smoothed[~finite_mask] = np.nan
    return smoothed


def build_landscapes(arr: np.ndarray, scales: list[int], method: str = LANDSCAPE_METHOD) -> dict[int, np.ndarray]:
    if len(arr) == 0:
        return {s: np.array([], dtype=np.float32) for s in scales}
    method = (method or "mean").lower()
    landscapes: dict[int, np.ndarray] = {}
    if method == "median":
        for s in scales:
            landscapes[s] = _window_medians(arr.astype(np.float64), s).astype(np.float32)
    else:
        csum, _, ccount = _prefix(arr)
        pos = np.arange(len(arr), dtype=np.int64)
        for s in scales:
            starts = pos - (s // 2)
            ends = starts + s
            landscapes[s] = _window_means_from_prefix(csum, ccount, starts, ends).astype(np.float32)
    return landscapes


def _compute_scale_lpc(
    landscape: np.ndarray,
    scale: int,
    min_candidate: int = MIN_CANDIDATE,
    max_candidate: int = MAX_CANDIDATE,
) -> np.ndarray:
    candidate_w = min(max(scale, min_candidate), max_candidate)
    upstream_w = scale
    spacer = scale // 2
    downstream_w = scale

    n = len(landscape)
    if n == 0:
        return np.array([], dtype=np.float32)

    centers = np.arange(n, dtype=np.int64)
    cand_s = centers - (candidate_w // 2)
    cand_e = cand_s + candidate_w
    up_s = cand_s - spacer - upstream_w
    up_e = cand_s - spacer
    dn_s = cand_e + spacer
    dn_e = dn_s + downstream_w

    csum, _, ccount = _prefix(landscape)
    up_mean = _window_means_from_prefix(csum, ccount, up_s, up_e)
    cand_mean = _window_means_from_prefix(csum, ccount, cand_s, cand_e)
    dn_mean = _window_means_from_prefix(csum, ccount, dn_s, dn_e)

    lpc_up = up_mean - cand_mean
    lpc_down = dn_mean - cand_mean
    lpc = np.minimum(lpc_up, lpc_down)

    valid = (
        np.isfinite(up_mean)
        & np.isfinite(cand_mean)
        & np.isfinite(dn_mean)
        & (lpc_up > 0)
        & (lpc_down > 0)
    )
    lpc[~valid] = np.nan
    return lpc.astype(np.float32)


def _robust_normalize(arr: np.ndarray) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float32)
    idx = np.isfinite(arr)
    vals = arr[idx].astype(np.float64)
    if vals.size < 2:
        if vals.size == 1:
            out[idx] = 0.0
        return out
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = mad * 1.4826
    if sigma < EPSILON:
        out[idx] = np.where(vals > med, 1.0, 0.0).astype(np.float32)
        return out
    out[idx] = ((vals - med) / sigma).astype(np.float32)
    return out


def _percentile_scale(arr: np.ndarray) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float32)
    idx = np.isfinite(arr)
    vals = arr[idx].astype(np.float64)
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(vals.size, dtype=np.float64)
    if vals.size == 1:
        out[idx] = 0.5
    else:
        out[idx] = (ranks / (vals.size - 1)).astype(np.float32)
    return out


def _normalize(arr: np.ndarray, method: str) -> np.ndarray:
    """Normalize one LPC profile with percentile or robust z-score scaling."""
    return _percentile_scale(arr) if method == "percentile" else _robust_normalize(arr)


def build_layer_consensus(lpc_profiles: dict[int, np.ndarray], normalization: str = NORMALIZATION_METHOD) -> np.ndarray:
    profiles = list(lpc_profiles.values())
    if not profiles:
        return np.array([], dtype=np.float32)
    n = len(profiles[0])
    stacked = np.full((len(profiles), n), np.nan, dtype=np.float64)
    method = (normalization or NORMALIZATION_METHOD).lower()
    for i, lpc in enumerate(profiles):
        stacked[i, : len(lpc)] = _normalize(lpc, method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stacked, axis=0).astype(np.float32)


def build_ensemble_consensus(layer_consensus: dict[str, np.ndarray], method: str = ENSEMBLE_METHOD) -> np.ndarray:
    profiles = [layer_consensus[layer] for layer in LAYERS if layer in layer_consensus]
    if not profiles:
        return np.array([], dtype=np.float32)
    n = len(profiles[0])
    stacked = np.full((len(profiles), n), np.nan, dtype=np.float64)
    for i, arr in enumerate(profiles):
        stacked[i, : len(arr)] = arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "trimmed_mean":
            sorted_vals = np.sort(stacked, axis=0)
            trim = max(0, int(math.floor(len(profiles) * 0.2)))
            if trim * 2 < len(profiles):
                sorted_vals = sorted_vals[trim: len(profiles) - trim]
            return np.nanmean(sorted_vals, axis=0).astype(np.float32)
        return np.nanmedian(stacked, axis=0).astype(np.float32)


def bounded_min_mean(arr: np.ndarray, min_len: int, max_len: int) -> tuple[int | None, int | None, float]:
    n = len(arr)
    if n < min_len:
        return None, None, np.inf
    pref = np.empty(n + 1, dtype=np.float64)
    pref[0] = 0.0
    pref[1:] = np.cumsum(arr)
    best_mean = np.inf
    best_i = best_j = None
    dq: deque[int] = deque()
    for j in range(n):
        i_min = j - max_len + 1
        i_max = j - min_len + 1
        if i_max < 0:
            continue
        while dq and pref[dq[-1]] >= pref[i_max]:
            dq.pop()
        dq.append(i_max)
        while dq and dq[0] < i_min:
            dq.popleft()
        if dq:
            i = dq[0]
            mean = (pref[j + 1] - pref[i]) / (j - i + 1)
            if mean < best_mean:
                best_mean = mean
                best_i, best_j = i, j
    return best_i, best_j, float(best_mean)


def _expand_valley(consensus_lpc: np.ndarray, core_start: int, core_end: int) -> tuple[int, int]:
    n = len(consensus_lpc)
    left = core_start
    while left > 0 and np.isfinite(consensus_lpc[left - 1]) and consensus_lpc[left - 1] > 0:
        left -= 1
    right = core_end
    while right < n - 1 and np.isfinite(consensus_lpc[right + 1]) and consensus_lpc[right + 1] > 0:
        right += 1
    return left, right


def _merge_valleys(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        ps, pe = merged[-1]
        if start <= pe + gap:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged


def _split_positive_runs(consensus_lpc: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    positive = np.isfinite(consensus_lpc) & (consensus_lpc > 0)
    i = 0
    n = len(consensus_lpc)
    while i < n:
        if not positive[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and positive[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


# ---------------------------------------------------------------------------
# v12 VPI (Valley Persistence Index) engine
# ---------------------------------------------------------------------------

def _compute_vpi(lpc_profiles: dict[str, dict[int, np.ndarray]], n: int) -> np.ndarray:
    """Compute Valley Persistence Index for each signal position.

    VPI[i] = (number of scale×layer combinations with LPC[i] > 0) / (total combinations).
    A VPI of 1.0 means all scales and layers support a valley at that position.

    Args:
        lpc_profiles: nested dict {layer_name: {scale: lpc_array}}.
        n: signal length (number of positions to evaluate).
    Returns:
        np.ndarray of shape (n,) with dtype float32, values in [0, 1].
    """
    support = np.zeros(n, dtype=np.float64)
    total = 0
    for layer in LAYERS:
        for lpc in lpc_profiles.get(layer, {}).values():
            end = min(len(lpc), n)
            finite_pos = np.isfinite(lpc[:end]) & (lpc[:end] > 0)
            support[:end] += finite_pos.astype(np.float64)
            total += 1
    if total == 0:
        return np.zeros(n, dtype=np.float32)
    return (support / total).astype(np.float32)


def _fill_gaps_vectorized(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill internal gaps shorter than max_gap in a boolean mask (fully vectorized).

    Only fills gaps that are bounded by True on both sides; boundary gaps are
    never filled.  Uses cumulative-sum index marking — no loops over positions.

    Args:
        mask: boolean array of shape (n,).
        max_gap: maximum number of False positions to bridge.
    Returns:
        np.ndarray of shape (n,) dtype bool with short internal gaps filled.
    """
    if max_gap <= 0 or not np.any(mask):
        return mask.astype(bool).copy()
    n = len(mask)
    filled = mask.astype(bool).copy()
    # Detect False-run boundaries by padding with True on each end.
    padded = np.concatenate([[True], filled, [True]])
    delta = np.diff(padded.view(np.int8))          # length = n+1
    false_starts = np.flatnonzero(delta < 0)       # first position of each False run (orig coords)
    false_ends = np.flatnonzero(delta > 0) - 1     # last position of each False run (orig coords)
    if false_starts.size == 0:
        return filled
    # Only fill runs that are (a) internal and (b) short enough.
    is_internal = (false_starts > 0) & (false_ends < n - 1)
    gap_lengths = false_ends - false_starts + 1
    to_fill = is_internal & (gap_lengths <= max_gap)
    fs = false_starts[to_fill]
    fe = false_ends[to_fill]
    if fs.size > 0:
        marker = np.zeros(n + 1, dtype=np.int32)
        np.add.at(marker, fs, 1)
        np.add.at(marker, fe + 1, -1)
        filled |= np.cumsum(marker[:n]).astype(bool)
    return filled


def _generate_vpi_candidates(
    vpi: np.ndarray,
    max_gap: int = MAX_INTERNAL_GAP,
    threshold: float = VPI_CANDIDATE_THRESHOLD,
) -> list[tuple[int, int]]:
    """Generate candidate valleys from VPI >= threshold runs with gap tolerance.

    Gaps shorter than max_gap inside a high-VPI region are bridged before
    extracting connected components.  No per-nucleotide loops.
    """
    n = len(vpi)
    if n == 0:
        return []
    mask = vpi >= threshold
    if not np.any(mask):
        return []
    filled = _fill_gaps_vectorized(mask, max_gap)
    padded = np.concatenate([[False], filled, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes > 0)
    ends = np.flatnonzero(changes < 0) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _expand_valley_vpi(
    vpi: np.ndarray,
    consensus_lpc: np.ndarray,
    core_start: int,
    core_end: int,
    expand_threshold: float = VPI_EXPAND_THRESHOLD,
) -> tuple[int, int]:
    """Expand valley boundaries while VPI > expand_threshold OR ConsensusLPC > 0.

    Stops at the first position where both conditions fail simultaneously.
    Fully vectorized using array index operations.

    Args:
        vpi: VPI array of shape (n,) dtype float32.
        consensus_lpc: ConsensusLPC array of shape (n,) dtype float32.
        core_start: first position of the high-VPI core (inclusive).
        core_end: last position of the high-VPI core (inclusive).
        expand_threshold: VPI threshold below which expansion stops (default 0.3).
    Returns:
        Tuple (expanded_start, expanded_end) with final boundary positions.
    """
    n = len(vpi)
    expand_cond = (vpi > expand_threshold) | (np.isfinite(consensus_lpc) & (consensus_lpc > 0))

    # Left expansion: find first False scanning backwards from core_start-1
    if core_start == 0:
        left = 0
    else:
        rev = expand_cond[:core_start][::-1]
        false_in_rev = np.flatnonzero(~rev)
        left = 0 if false_in_rev.size == 0 else (core_start - int(false_in_rev[0]))

    # Right expansion: find first False scanning forward from core_end+1
    if core_end >= n - 1:
        right = n - 1
    else:
        fwd = expand_cond[core_end + 1:]
        false_positions = np.flatnonzero(~fwd)
        right = n - 1 if false_positions.size == 0 else (core_end + int(false_positions[0]))

    return left, right


def _compute_vpi_persistence(
    vpi: np.ndarray,
    start: int,
    end: int,
    threshold: float = VPI_CANDIDATE_THRESHOLD,
) -> float:
    """Fraction of positions in [start, end] with VPI >= threshold."""
    seg = vpi[start: end + 1]
    if seg.size == 0:
        return 0.0
    return float(np.mean(seg >= threshold))


def _layer_support_counts(lpc_profiles: dict[str, dict[int, np.ndarray]], start: int, end: int) -> tuple[dict[str, tuple[int, int]], int, int]:
    layer_support: dict[str, tuple[int, int]] = {}
    total_support = 0
    total_possible = 0
    for layer in LAYERS:
        profiles = lpc_profiles.get(layer, {})
        layer_total = len(profiles)
        layer_hits = 0
        for lpc in profiles.values():
            seg = lpc[start: end + 1]
            finite = seg[np.isfinite(seg)]
            if finite.size > 0 and float(np.mean(finite)) > 0:
                layer_hits += 1
        layer_support[layer] = (layer_hits, layer_total)
        total_support += layer_hits
        total_possible += layer_total
    return layer_support, total_support, total_possible


def _nucleotide_bounds(signal_start: int, signal_end: int, seq_len: int, p_window: int) -> tuple[int, int]:
    start = max(0, int(signal_start))
    end = min(seq_len - 1, int(signal_end + p_window - 1))
    return start, end


def _parse_optional_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "null", "na", "off"}:
            return None
    return int(value)


def _resolve_operating_mode(mode: str | None) -> str:
    resolved = (mode or DEFAULT_MODE).strip().lower()
    if resolved not in OPERATING_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from {', '.join(OPERATING_MODES)}.")
    return resolved


def _count_di_scale_support(di_lpc_profiles: dict[int, np.ndarray], start: int, end: int) -> tuple[int, int]:
    total = len(di_lpc_profiles)
    hits = 0
    for lpc in di_lpc_profiles.values():
        seg = lpc[start: end + 1]
        finite = seg[np.isfinite(seg)]
        if finite.size > 0 and float(np.mean(finite)) > 0:
            hits += 1
    return hits, total


def _passes_positional_prior(
    start: int,
    end: int,
    seq_len: int,
    p_window: int,
    reference_point: int | None,
    core_window_upstream: int | None,
    core_window_downstream: int | None,
) -> bool:
    if core_window_upstream is None or core_window_downstream is None or reference_point is None:
        return True
    start_nt, end_nt = _nucleotide_bounds(start, end, seq_len, p_window)
    midpoint = (start_nt + end_nt) / 2.0
    rel = midpoint - float(reference_point)
    return (-float(core_window_upstream)) <= rel <= float(core_window_downstream)


# ---------------------------------------------------------------------------
# Candidate generation (Step 4)
# ---------------------------------------------------------------------------

def _generate_candidate_valleys(consensus_lpc: np.ndarray) -> list[tuple[int, int]]:
    """Return all contiguous positive runs in consensus_lpc as raw candidates.

    No size filtering is applied here; Kadane refinement handles size bounds.
    """
    candidates: list[tuple[int, int]] = []
    positive = np.isfinite(consensus_lpc) & (consensus_lpc > 0)
    n = len(consensus_lpc)
    i = 0
    while i < n:
        if not positive[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and positive[j + 1]:
            j += 1
        candidates.append((i, j))
        i = j + 1
    return candidates


# ---------------------------------------------------------------------------
# Kadane refinement (Step 5)
# ---------------------------------------------------------------------------

def _refine_with_kadane(
    consensus_lpc: np.ndarray,
    cand_start: int,
    cand_end: int,
    min_len: int,
    max_len: int,
) -> tuple[int | None, int | None]:
    """Find the strongest contiguous core inside a candidate valley.

    Returns refined (start, end) with length in [min_len, max_len], or
    (None, None) if no valid core exists.
    """
    seg = -consensus_lpc[cand_start: cand_end + 1].astype(np.float64)
    seg_len = len(seg)
    if seg_len < min_len:
        return None, None
    s, e, mv = bounded_min_mean(seg, min_len=min_len, max_len=min(max_len, seg_len))
    if s is None or not np.isfinite(mv):
        # Fall back: use the whole candidate capped to max_len
        s, e = 0, min(seg_len - 1, max_len - 1)
        if e - s + 1 < min_len:
            return None, None
    return cand_start + s, cand_start + e


# ---------------------------------------------------------------------------
# Persistence filter (Step 6)
# ---------------------------------------------------------------------------

def _compute_persistence(consensus_lpc: np.ndarray, start: int, end: int) -> float:
    """Fraction of positions inside [start, end] with ConsensusLPC > 0."""
    seg = consensus_lpc[start: end + 1]
    finite = seg[np.isfinite(seg)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite > 0))


# ---------------------------------------------------------------------------
# Prominence (Step 7)
# ---------------------------------------------------------------------------

def _compute_prominence(
    signal: np.ndarray,
    start: int,
    end: int,
    flank_window: int = FLANK_WINDOW,
) -> float:
    """Compute prominence using a perplexity signal.

    Prominence = mean(flanks) - min(valley).
    Deep valleys (low perplexity relative to surrounding sequence) score high.
    """
    n = len(signal)
    left_flank = signal[max(0, start - flank_window): start]
    right_flank = signal[end + 1: min(n, end + 1 + flank_window)]

    flanks = np.concatenate([left_flank, right_flank])
    flanks_f = flanks[np.isfinite(flanks)]
    background = float(np.mean(flanks_f)) if flanks_f.size else float("nan")

    valley_seg = signal[start: end + 1]
    valley_f = valley_seg[np.isfinite(valley_seg)]
    valley_min = float(np.min(valley_f)) if valley_f.size else float("nan")

    if not (np.isfinite(background) and np.isfinite(valley_min)):
        return 0.0
    return max(0.0, background - valley_min)


# ---------------------------------------------------------------------------
# Non-maximum suppression (Step 9)
# ---------------------------------------------------------------------------

def _overlap_fraction(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Symmetric overlap fraction: intersection / min(length_a, length_b)."""
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_start > inter_end:
        return 0.0
    inter_len = inter_end - inter_start + 1
    min_len = min(a_end - a_start + 1, b_end - b_start + 1)
    return inter_len / max(min_len, 1)


def _apply_nms(
    candidates: list[dict],
    overlap_threshold: float = NMS_OVERLAP_THRESHOLD,
) -> list[dict]:
    """Non-maximum suppression on a list of scored candidate dicts.

    Each dict must have keys: Signal_Start, Signal_End, ValleyScore_raw.
    Returns a list without mutually overlapping duplicates, keeping the
    highest-scoring candidate within each overlapping group.
    """
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda d: d["ValleyScore_raw"], reverse=True)
    kept: list[dict] = []
    for cand in sorted_cands:
        cs, ce = cand["Signal_Start"], cand["Signal_End"]
        suppressed = False
        for kand in kept:
            ks, ke = kand["Signal_Start"], kand["Signal_End"]
            if _overlap_fraction(cs, ce, ks, ke) >= overlap_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Merge (Step 10)
# ---------------------------------------------------------------------------

def _merge_valleys(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        ps, pe = merged[-1]
        if start <= pe + gap:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Layer support helpers
# ---------------------------------------------------------------------------

def _layer_support_counts(
    lpc_profiles: dict[str, dict[int, np.ndarray]], start: int, end: int
) -> tuple[dict[str, tuple[int, int]], int, int]:
    layer_support: dict[str, tuple[int, int]] = {}
    total_support = 0
    total_possible = 0
    for layer in LAYERS:
        profiles = lpc_profiles.get(layer, {})
        layer_total = len(profiles)
        layer_hits = 0
        for lpc in profiles.values():
            seg = lpc[start: end + 1]
            finite = seg[np.isfinite(seg)]
            if finite.size > 0 and float(np.mean(finite)) > 0:
                layer_hits += 1
        layer_support[layer] = (layer_hits, layer_total)
        total_support += layer_hits
        total_possible += layer_total
    return layer_support, total_support, total_possible


# ---------------------------------------------------------------------------
# Candidate metrics (Step 4)
# ---------------------------------------------------------------------------

def candidate_statistics(
    start: int,
    end: int,
    support_lpc_profiles: dict[str, dict[int, np.ndarray]],
    scoring_lpc_profiles: dict[int, np.ndarray],
    layer_consensus: dict[str, np.ndarray],
    consensus_lpc: np.ndarray,
) -> dict:
    sl = slice(start, end + 1)
    cons_seg = consensus_lpc[sl]
    finite_cons = cons_seg[np.isfinite(cons_seg)]
    peak_lpc = float(np.max(finite_cons)) if finite_cons.size else 0.0
    persistence = float(np.mean(finite_cons > 0)) if finite_cons.size else 0.0
    area = float(np.sum(np.maximum(finite_cons, 0.0))) if finite_cons.size else 0.0

    layer_support, support_hits, support_total = _layer_support_counts(support_lpc_profiles, start, end)
    di_hits, di_total = _count_di_scale_support(scoring_lpc_profiles, start, end)
    scale_support_fraction = (di_hits / di_total) if di_total else 0.0

    layer_hits = 0
    for layer in LAYERS:
        arr = layer_consensus.get(layer)
        if arr is None:
            continue
        seg = arr[sl]
        finite = seg[np.isfinite(seg)]
        if finite.size and float(np.mean(finite)) > 0:
            layer_hits += 1
    layer_support_fraction = layer_hits / len(LAYERS)

    return {
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Signal_Length": int(end - start + 1),
        "PeakLPC": peak_lpc,
        "Area": area,
        "Persistence": persistence,
        "ScaleSupport": f"{di_hits}/{di_total}",
        "ScaleSupportFraction": float(scale_support_fraction),
        "LayerSupport": f"{layer_hits}/{len(LAYERS)}",
        "LayerSupportFraction": float(layer_support_fraction),
        "MonoSupport": f"{layer_support['mono'][0]}/{layer_support['mono'][1]}",
        "DiSupport": f"{layer_support['di'][0]}/{layer_support['di'][1]}",
        "TriSupport": f"{layer_support['tri'][0]}/{layer_support['tri'][1]}",
        # OverallSupport retains all-layer support for interpretability.
        "OverallSupport": f"{support_hits}/{support_total}",
    }


# ---------------------------------------------------------------------------
# Valley statistics (Step 13 columns)
# ---------------------------------------------------------------------------

def valley_statistics(
    valley_index: int,
    start: int,
    end: int,
    mono: np.ndarray,
    di: np.ndarray,
    tri: np.ndarray,
    support_lpc_profiles: dict[str, dict[int, np.ndarray]],
    scoring_lpc_profiles: dict[int, np.ndarray],
    support_layer_consensus: dict[str, np.ndarray],
    consensus_lpc: np.ndarray,
    scoring_signal: np.ndarray,
    seq: str,
    p_window: int,
    kadane_core: tuple[int | None, int | None],
    prominence: float = 0.0,
    valley_kadane_core: tuple[int | None, int | None] = (None, None),
) -> dict:
    sl = slice(start, end + 1)
    mono_seg = mono[sl]
    di_seg = di[sl]
    tri_seg = tri[sl]
    cons_seg = consensus_lpc[sl]

    def _fmean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def _fmin(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.min(finite)) if finite.size else float("nan")

    def _fmax(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.max(finite)) if finite.size else float("nan")

    mono_mean = _fmean(mono_seg)
    di_mean = _fmean(di_seg)
    tri_mean = _fmean(tri_seg)

    # Mean/Min/Max perplexity on the scoring signal (Di in default modes)
    avg_seg = scoring_signal[sl]
    mean_perplexity = _fmean(avg_seg)
    min_perplexity = _fmin(avg_seg)
    max_perplexity = _fmax(avg_seg)

    finite_cons = cons_seg[np.isfinite(cons_seg)]
    mean_lpc = float(np.mean(finite_cons)) if finite_cons.size else 0.0
    max_lpc = float(np.max(finite_cons)) if finite_cons.size else 0.0
    persistence = float(np.mean(finite_cons > 0)) if finite_cons.size else 0.0
    area = float(np.sum(np.maximum(finite_cons, 0.0))) if finite_cons.size else 0.0
    variance = float(np.var(finite_cons)) if finite_cons.size > 1 else 0.0

    layer_support, support_hits, support_total = _layer_support_counts(support_lpc_profiles, start, end)
    di_hits, di_total = _count_di_scale_support(scoring_lpc_profiles, start, end)
    scale_support_fraction = (di_hits / di_total) if di_total else 0.0

    layer_hits = 0
    for layer in LAYERS:
        arr = support_layer_consensus.get(layer)
        if arr is None:
            continue
        seg = arr[sl]
        finite = seg[np.isfinite(seg)]
        if finite.size and float(np.mean(finite)) > 0:
            layer_hits += 1
    layer_support_fraction = layer_hits / len(LAYERS)

    length_signal = end - start + 1
    # Raw score (di-derived support in v12 default modes).
    log_length = math.log(max(length_signal, 1))
    valley_score_raw = (
        mean_lpc
        * persistence
        * scale_support_fraction
        * area
        * log_length
        * (1.0 / (variance + EPSILON))
    )

    start_nt, end_nt = _nucleotide_bounds(start, end, len(seq), p_window)
    sequence = seq[start_nt: end_nt + 1]
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    k_start, k_end = kadane_core
    core_overlap = bool(
        k_start is not None and k_end is not None
        and not (end < k_start or start > k_end)
    )

    # Per-valley Kadane core (for visualization within the final merged valley)
    vk_start, vk_end = valley_kadane_core

    return {
        "ID": f"PV_{valley_index:06d}",
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Signal_Length": int(length_signal),
        "Start": int(start_nt),
        "End": int(end_nt),
        "Length": int(end_nt - start_nt + 1),
        # Perplexity
        "MeanPerplexity": mean_perplexity,
        "MinPerplexity": min_perplexity,
        "MaxPerplexity": max_perplexity,
        "MeanMono": mono_mean,
        "MeanDi": di_mean,
        "MeanTri": tri_mean,
        # LPC / signal quality
        "MeanLPC": float(mean_lpc),
        "MaxLPC": float(max_lpc),
        "Contrast": float(mean_lpc),        # legacy alias kept for visualizations
        "ConsensusPeak": float(max_lpc),    # legacy alias
        "Prominence": float(prominence),
        "Persistence": float(persistence),
        "Area": float(area),
        "Variance": float(variance),
        # Support
        "MonoSupport": f"{layer_support['mono'][0]}/{layer_support['mono'][1]}",
        "DiSupport": f"{layer_support['di'][0]}/{layer_support['di'][1]}",
        "TriSupport": f"{layer_support['tri'][0]}/{layer_support['tri'][1]}",
        "ScaleSupport": f"{di_hits}/{di_total}",
        "ScaleSupportFraction": float(scale_support_fraction),
        "LayerSupport": f"{layer_hits}/{len(LAYERS)}",
        "LayerSupportFraction": float(layer_support_fraction),
        "OverallSupport": f"{support_hits}/{support_total}",
        # Scoring
        "ValleyScore_raw": float(valley_score_raw),
        "GC%": float(gc),
        "KadaneCoreOverlap": core_overlap,
        # Per-valley Kadane core (visualization only)
        "KadaneCoreStart": int(vk_start) if vk_start is not None else None,
        "KadaneCoreEnd": int(vk_end) if vk_end is not None else None,
        "Sequence": sequence,
    }


def normalize_valley_score(domains: list[dict]) -> list[dict]:
    if not domains:
        return domains
    raw = np.array([max(0.0, float(d.get("ValleyScore_raw", 0.0))) for d in domains], dtype=np.float64)
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    norm = np.where(raw > 0, 1.0, 0.0) if span < EPSILON else (raw - lo) / span
    for i, d in enumerate(domains):
        d["ValleyScore"] = float(raw[i])
        d["ValleyScoreNormalized"] = float(norm[i])
        d.pop("ValleyScore_raw", None)
    # Assign rank (1 = best)
    ranked = sorted(range(len(domains)), key=lambda idx: domains[idx]["ValleyScore"], reverse=True)
    for rank, idx in enumerate(ranked, 1):
        domains[idx]["Rank"] = rank
    return domains


# ---------------------------------------------------------------------------
# Main domain-finding pipeline
# ---------------------------------------------------------------------------

def find_domains(
    mono: np.ndarray,
    di: np.ndarray,
    tri: np.ndarray,
    support_lpc_profiles: dict[str, dict[int, np.ndarray]],
    support_layer_consensus: dict[str, np.ndarray],
    consensus_lpc: np.ndarray,
    scoring_lpc_profiles: dict[int, np.ndarray],
    scoring_signal: np.ndarray,
    seq: str,
    min_domain: int = MIN_DOMAIN,
    max_domain: int = MAX_DOMAIN,
    merge_gap: int = MERGE_GAP,
    persistence_threshold: float = PERSISTENCE_THRESHOLD,
    nms_overlap: float = NMS_OVERLAP_THRESHOLD,
    p_window: int = PERPLEXITY_WINDOW,
    core_window_upstream: int | None = None,
    core_window_downstream: int | None = None,
    reference_point: int | None = None,
) -> tuple[list[dict], list[dict], tuple[int | None, int | None]]:
    """Candidate → refinement → ranking pipeline.

    ConsensusLPC candidates → Kadane refinement → persistence filter →
    adaptive prominence filter → score → NMS → merge → final valleys.
    """
    n = len(consensus_lpc)
    if n == 0:
        return [], [], (None, None)

    prominence_signal = np.full(n, np.nan, dtype=np.float32)
    if len(scoring_signal) > 0:
        signal_end = min(n, len(scoring_signal))
        prominence_signal[:signal_end] = scoring_signal[:signal_end].astype(np.float32)

    # Step 4: generate all candidate valleys from positive ConsensusLPC runs.
    raw_candidate_intervals = _generate_candidate_valleys(consensus_lpc)
    raw_candidates = [
        candidate_statistics(
            s,
            e,
            support_lpc_profiles=support_lpc_profiles,
            scoring_lpc_profiles=scoring_lpc_profiles,
            layer_consensus=support_layer_consensus,
            consensus_lpc=consensus_lpc,
        )
        for s, e in raw_candidate_intervals
    ]

    # Step 5: refine each candidate with bounded Kadane.
    refined: list[tuple[int, int, int | None, int | None]] = []
    global_best_score: float = np.inf
    global_kadane_core: tuple[int | None, int | None] = (None, None)

    for cand_start, cand_end in raw_candidate_intervals:
        ks, ke = _refine_with_kadane(consensus_lpc, cand_start, cand_end, min_domain, max_domain)
        if ks is not None and ke is not None:
            seg_neg = -consensus_lpc[ks: ke + 1].astype(np.float64)
            finite_neg = seg_neg[np.isfinite(seg_neg)]
            if finite_neg.size > 0:
                mn = float(np.mean(finite_neg))
                if mn < global_best_score:
                    global_best_score = mn
                    global_kadane_core = (ks, ke)
            refined.append((ks, ke, ks, ke))

    if not refined:
        return [], raw_candidates, global_kadane_core

    # Step 6: hard persistence filter.
    after_persistence: list[tuple[int, int, int | None, int | None]] = [
        (s, e, ks, ke)
        for s, e, ks, ke in refined
        if _compute_persistence(consensus_lpc, s, e) >= persistence_threshold
    ]
    if not after_persistence:
        return [], raw_candidates, global_kadane_core

    # Step 7: adaptive prominence filter.
    prominences = [
        _compute_prominence(prominence_signal, s, e)
        for s, e, ks, ke in after_persistence
    ]
    prom_arr = np.array(prominences, dtype=np.float64)
    adaptive_threshold = float(np.nanpercentile(prom_arr, 75)) if prom_arr.size > 1 else 0.0

    after_prominence: list[tuple[int, int, int | None, int | None, float]] = [
        (s, e, ks, ke, prom)
        for (s, e, ks, ke), prom in zip(after_persistence, prominences)
        if prom >= adaptive_threshold
    ]
    if not after_prominence:
        return [], raw_candidates, global_kadane_core

    # Steps 8–11: score each candidate, suppress overlaps, then merge.
    scored_dicts: list[dict] = []
    for s, e, ks, ke, prom in after_prominence:
        d = valley_statistics(
            valley_index=0,
            start=s,
            end=e,
            mono=mono,
            di=di,
            tri=tri,
            support_lpc_profiles=support_lpc_profiles,
            scoring_lpc_profiles=scoring_lpc_profiles,
            support_layer_consensus=support_layer_consensus,
            consensus_lpc=consensus_lpc,
            scoring_signal=prominence_signal,
            seq=seq,
            p_window=p_window,
            kadane_core=global_kadane_core,
            prominence=prom,
            valley_kadane_core=(ks, ke),
        )
        scored_dicts.append(d)

    after_nms = _apply_nms(scored_dicts, overlap_threshold=nms_overlap)
    nms_intervals = [(d["Signal_Start"], d["Signal_End"]) for d in after_nms]
    merged_intervals = _merge_valleys(nms_intervals, merge_gap)

    domains: list[dict] = []
    for idx, (s, e) in enumerate(merged_intervals, 1):
        if not _passes_positional_prior(
            s,
            e,
            seq_len=len(seq),
            p_window=p_window,
            reference_point=reference_point,
            core_window_upstream=core_window_upstream,
            core_window_downstream=core_window_downstream,
        ):
            continue
        if e - s + 1 < min_domain:
            continue
        ks = ke = None
        if e - s + 1 > max_domain:
            ks, ke = _refine_with_kadane(consensus_lpc, s, e, min_domain, max_domain)
            if ks is None or ke is None:
                continue
            s, e = ks, ke
        if _compute_persistence(consensus_lpc, s, e) < persistence_threshold:
            continue
        prom = _compute_prominence(prominence_signal, s, e)
        if prom < adaptive_threshold:
            continue
        if ks is None or ke is None:
            ks, ke = _refine_with_kadane(consensus_lpc, s, e, min_domain, max_domain)
        d = valley_statistics(
            valley_index=idx,
            start=s,
            end=e,
            mono=mono,
            di=di,
            tri=tri,
            support_lpc_profiles=support_lpc_profiles,
            scoring_lpc_profiles=scoring_lpc_profiles,
            support_layer_consensus=support_layer_consensus,
            consensus_lpc=consensus_lpc,
            scoring_signal=prominence_signal,
            seq=seq,
            p_window=p_window,
            kadane_core=global_kadane_core,
            prominence=prom,
            valley_kadane_core=(ks, ke),
        )
        domains.append(d)

    # Sort by genomic position before scoring normalization.
    domains.sort(key=lambda d: d["Signal_Start"])
    return normalize_valley_score(domains), raw_candidates, global_kadane_core


def _resolve_scales(scales: list[int] | None) -> list[int]:
    if scales:
        return sorted(set(max(10, int(s)) for s in scales))
    return DEFAULT_SCALES.copy()


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    """REGPLEX v12 analysis pipeline.

    Default behavior is di-centric: promoter/genome modes use Dinucleotide
    ConsensusLPC for detection and keep Mono/Tri as support annotations.
    Ensemble mode is retained as an exploratory multi-layer option.

    Steps
    -----
    1. Mono / Di / Tri perplexity (single pass, unchanged from v10).
    2. Savitzky-Golay smoothing on each perplexity profile (applied once).
    3. Multi-scale landscapes from smoothed profiles.
    4. Three-window Local Perplexity Contrast (LPC) at each scale.
    5. Layer consensus (median across scales, normalized).
    6. Operating-mode consensus: Di-only (promoter/genome) or ensemble (experimental).
    7. Candidate valleys from positive ConsensusLPC runs.
    8–14. Kadane refinement → ConsensusLPC persistence → adaptive prominence →
          score → NMS → merge → final ranking.
    """
    params = {
        "perplexity_window": int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        "scales": kwargs.get("scales", None),
        "landscape_method": kwargs.get("landscape_method", LANDSCAPE_METHOD),
        "normalization_method": kwargs.get("normalization_method", NORMALIZATION_METHOD),
        "ensemble_method": kwargs.get("ensemble_method", ENSEMBLE_METHOD),
        "min_candidate": int(kwargs.get("min_candidate", MIN_CANDIDATE)),
        "max_candidate": int(kwargs.get("max_candidate", MAX_CANDIDATE)),
        "min_domain": int(kwargs.get("min_domain", MIN_DOMAIN)),
        "max_domain": int(kwargs.get("max_domain", MAX_DOMAIN)),
        "merge_gap": int(kwargs.get("merge_gap", MERGE_GAP)),
        "sg_window_length": int(kwargs.get("sg_window_length", SG_WINDOW_LENGTH)),
        "sg_poly_order": int(kwargs.get("sg_poly_order", SG_POLY_ORDER)),
        "persistence_threshold": float(kwargs.get("persistence_threshold", PERSISTENCE_THRESHOLD)),
        "nms_overlap": float(kwargs.get("nms_overlap", NMS_OVERLAP_THRESHOLD)),
        "mode": kwargs.get("mode", DEFAULT_MODE),
        "core_window_upstream": _parse_optional_int(kwargs.get("core_window_upstream", None)),
        "core_window_downstream": _parse_optional_int(kwargs.get("core_window_downstream", None)),
        "reference_point": _parse_optional_int(kwargs.get("reference_point", 0)),
    }
    mode = _resolve_operating_mode(params["mode"])
    params["resolved_mode"] = mode
    scales = _resolve_scales(params["scales"])
    params["resolved_scales"] = scales

    # Step 1: one-pass layer perplexity computations (unchanged from v10).
    mono = compute_mono_perplexity(seq, params["perplexity_window"])
    di = compute_di_perplexity(seq, params["perplexity_window"])
    tri = compute_tri_perplexity(seq, params["perplexity_window"])

    # Step 2: Savitzky-Golay smoothing — applied once per profile.
    smoothed_mono = smooth_perplexity(mono, params["sg_window_length"], params["sg_poly_order"])
    smoothed_di = smooth_perplexity(di, params["sg_window_length"], params["sg_poly_order"])
    smoothed_tri = smooth_perplexity(tri, params["sg_window_length"], params["sg_poly_order"])

    # Steps 3–6: multi-scale landscapes from smoothed profiles, LPC, consensus.
    smoothed_signals = {"mono": smoothed_mono, "di": smoothed_di, "tri": smoothed_tri}
    landscapes: dict[str, dict[int, np.ndarray]] = {}
    lpc_profiles: dict[str, dict[int, np.ndarray]] = {}
    layer_consensus: dict[str, np.ndarray] = {}
    for layer in LAYERS:
        landscapes[layer] = build_landscapes(
            smoothed_signals[layer], scales, params["landscape_method"]
        )
        lpc_profiles[layer] = {
            s: _compute_scale_lpc(
                landscapes[layer][s],
                s,
                min_candidate=params["min_candidate"],
                max_candidate=params["max_candidate"],
            )
            for s in scales
        }
        layer_consensus[layer] = build_layer_consensus(
            lpc_profiles[layer],
            normalization=params["normalization_method"],
        )

    # Step 6: final ConsensusLPC determined by operating mode.
    if mode == "ensemble":
        consensus_lpc = build_ensemble_consensus(layer_consensus, method=params["ensemble_method"])
        core_upstream = params["core_window_upstream"]
        core_downstream = params["core_window_downstream"]
    else:
        consensus_lpc = layer_consensus["di"].copy()
        if mode == "promoter":
            core_upstream = (
                CORE_WINDOW_UPSTREAM
                if "core_window_upstream" not in kwargs
                else params["core_window_upstream"]
            )
            core_downstream = (
                CORE_WINDOW_DOWNSTREAM
                if "core_window_downstream" not in kwargs
                else params["core_window_downstream"]
            )
        else:  # genome
            core_upstream = None
            core_downstream = None
    params["resolved_core_window_upstream"] = core_upstream
    params["resolved_core_window_downstream"] = core_downstream

    # Steps 7–14: detection pipeline.
    domains, raw_candidates, kadane_core = find_domains(
        mono=mono,
        di=di,
        tri=tri,
        support_lpc_profiles=lpc_profiles,
        support_layer_consensus=layer_consensus,
        consensus_lpc=consensus_lpc,
        scoring_lpc_profiles=lpc_profiles["di"],
        scoring_signal=di,
        seq=seq,
        min_domain=params["min_domain"],
        max_domain=params["max_domain"],
        merge_gap=params["merge_gap"],
        persistence_threshold=params["persistence_threshold"],
        nms_overlap=params["nms_overlap"],
        p_window=params["perplexity_window"],
        core_window_upstream=core_upstream,
        core_window_downstream=core_downstream,
        reference_point=params["reference_point"],
    )
    for d in domains:
        d["Sequence_ID"] = sequence_id

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        mono=mono,
        di=di,
        tri=tri,
        smoothed_mono=smoothed_mono,
        smoothed_di=smoothed_di,
        smoothed_tri=smoothed_tri,
        landscapes=landscapes,
        lpc_profiles=lpc_profiles,
        layer_consensus=layer_consensus,
        consensus_lpc=consensus_lpc,
        candidates=raw_candidates,
        domains=domains,
        params=params,
        kadane_core=kadane_core,
    )


def domains_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        rows.extend(result.domains)
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt == "csv":
        return df.to_csv(index=False).encode()
    if fmt == "tsv":
        return df.to_csv(index=False, sep="\t").encode()
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buf.getvalue()
    if fmt == "json":
        return df.to_json(orient="records", indent=2).encode()
    raise ValueError(f"Unsupported format: {fmt}")


def export_bed(df: pd.DataFrame) -> bytes:
    bed = df[["Sequence_ID", "Start", "End", "ID", "ValleyScore"]].copy()
    bed["Start"] = bed["Start"].astype(int)
    bed["End"] = bed["End"].astype(int) + 1
    return bed.to_csv(index=False, sep="\t", header=False).encode()


def export_fasta(df: pd.DataFrame) -> bytes:
    lines: list[str] = []
    for _, row in df.iterrows():
        lines.append(
            f">{row['ID']}|{row['Sequence_ID']}:{row['Start']}-{row['End']}|ValleyScore={row['ValleyScore']:.4f}"
        )
        lines.append(str(row["Sequence"]))
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def export_gff(df: pd.DataFrame, gff3: bool = False) -> bytes:
    rows: list[str] = []
    for _, row in df.iterrows():
        attr = (
            f"ID={row['ID']};"
            f"ValleyScore={row['ValleyScore']:.4f};"
            f"MeanLPC={row.get('MeanLPC', row.get('Contrast', 0.0)):.4f};"
            f"Persistence={row['Persistence']:.4f};"
            f"Prominence={row.get('Prominence', 0.0):.4f};"
            f"Area={row['Area']:.4f};"
            f"ScaleSupport={row['ScaleSupport']};"
            f"LayerSupport={row['LayerSupport']};"
            f"Motifs={row.get('Motifs', '')}"
        )
        rows.append(
            "\t".join(
                [
                    str(row["Sequence_ID"]),
                    "REGPLEX",
                    "perplexity_valley",
                    str(int(row["Start"]) + 1),
                    str(int(row["End"]) + 1),
                    f"{float(row['ValleyScore']):.4f}",
                    ".",
                    ".",
                    attr,
                ]
            )
        )
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()


def cli() -> None:
    def _cli_optional_int(value: str) -> int | None:
        return _parse_optional_int(value)

    parser = argparse.ArgumentParser(description="REGPLEX v12 — di-centric perplexity valley detector")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_valleys.csv", help="Output CSV path")
    parser.add_argument(
        "--scales",
        default=",".join(str(s) for s in DEFAULT_SCALES),
        help="Comma-separated scales (bp), e.g. 25,50,100,200,400",
    )
    parser.add_argument(
        "--landscape-method",
        default=LANDSCAPE_METHOD,
        choices=["mean", "median"],
        help="Landscape smoothing method",
    )
    parser.add_argument(
        "--normalization-method",
        default=NORMALIZATION_METHOD,
        choices=["robust_z", "percentile"],
    )
    parser.add_argument(
        "--ensemble-method",
        default=ENSEMBLE_METHOD,
        choices=["median", "trimmed_mean"],
        help="Layer aggregation method used only in ensemble mode",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=list(OPERATING_MODES),
        help="Operating mode: promoter (default), genome, ensemble (experimental)",
    )
    parser.add_argument(
        "--core-window-upstream",
        type=_cli_optional_int,
        default=CORE_WINDOW_UPSTREAM,
        help="Promoter prior upstream window in bp (use 'none' to disable)",
    )
    parser.add_argument(
        "--core-window-downstream",
        type=_cli_optional_int,
        default=CORE_WINDOW_DOWNSTREAM,
        help="Promoter prior downstream window in bp (use 'none' to disable)",
    )
    parser.add_argument(
        "--reference-point",
        type=_cli_optional_int,
        default=0,
        help="Reference point for positional prior (default 0)",
    )
    parser.add_argument("--sg-window", type=int, default=SG_WINDOW_LENGTH, help="Savitzky-Golay window length (odd)")
    parser.add_argument("--sg-order", type=int, default=SG_POLY_ORDER, help="Savitzky-Golay polynomial order")
    parser.add_argument("--persistence", type=float, default=PERSISTENCE_THRESHOLD, help="Persistence threshold (0–1)")
    parser.add_argument("--nms-overlap", type=float, default=NMS_OVERLAP_THRESHOLD, help="NMS overlap fraction (0–1)")
    parser.add_argument("--min-domain", type=int, default=MIN_DOMAIN, help="Minimum valley length (bp)")
    parser.add_argument("--max-domain", type=int, default=MAX_DOMAIN, help="Maximum valley length (bp)")
    parser.add_argument("--merge-gap", type=int, default=MERGE_GAP, help="Gap for post-NMS merging (bp)")
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        fasta_text = handle.read()

    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    records = parse_fasta(fasta_text)
    results = [
        analyze_sequence(
            header,
            sequence,
            scales=scales,
            landscape_method=args.landscape_method,
            normalization_method=args.normalization_method,
            ensemble_method=args.ensemble_method,
            mode=args.mode,
            core_window_upstream=args.core_window_upstream,
            core_window_downstream=args.core_window_downstream,
            reference_point=args.reference_point,
            sg_window_length=args.sg_window,
            sg_poly_order=args.sg_order,
            persistence_threshold=args.persistence,
            nms_overlap=args.nms_overlap,
            min_domain=args.min_domain,
            max_domain=args.max_domain,
            merge_gap=args.merge_gap,
        )
        for header, sequence in records
    ]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} valleys to {args.out}")


if __name__ == "__main__":
    cli()
