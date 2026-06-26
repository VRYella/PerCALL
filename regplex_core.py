from __future__ import annotations

import argparse
import io
import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------
# Tunable scientific defaults
# ---------------------------
PERPLEXITY_WINDOW = 10
DEFAULT_SCALES = [25, 50, 100, 200, 400]
LANDSCAPE_METHOD = "mean"  # mean | median
NORMALIZATION_METHOD = "robust_z"  # robust_z | percentile
ENSEMBLE_METHOD = "median"  # median | trimmed_mean
MIN_CANDIDATE = 50
MAX_CANDIDATE = 1000
MIN_DOMAIN = 50
MAX_DOMAIN = 1000
MERGE_GAP = 25
EPSILON = 1e-9

LAYERS = ("mono", "di", "tri")
_LAYER_LABELS = {"mono": "Mono", "di": "Di", "tri": "Tri"}

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
    landscapes: dict[str, dict[int, np.ndarray]]
    lpc_profiles: dict[str, dict[int, np.ndarray]]
    layer_consensus: dict[str, np.ndarray]
    consensus_lpc: np.ndarray
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


def valley_statistics(
    valley_index: int,
    start: int,
    end: int,
    mono: np.ndarray,
    di: np.ndarray,
    tri: np.ndarray,
    lpc_profiles: dict[str, dict[int, np.ndarray]],
    layer_consensus: dict[str, np.ndarray],
    consensus_lpc: np.ndarray,
    seq: str,
    p_window: int,
    kadane_core: tuple[int | None, int | None],
) -> dict:
    sl = slice(start, end + 1)
    mono_seg = mono[sl]
    di_seg = di[sl]
    tri_seg = tri[sl]
    cons_seg = consensus_lpc[sl]

    def _fmean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if finite.size else float("nan")

    mono_mean = _fmean(mono_seg)
    di_mean = _fmean(di_seg)
    tri_mean = _fmean(tri_seg)

    finite_cons = cons_seg[np.isfinite(cons_seg)]
    contrast = float(np.mean(finite_cons)) if finite_cons.size else 0.0
    persistence = float(np.mean(finite_cons > 0)) if finite_cons.size else 0.0
    area = float(np.sum(np.maximum(finite_cons, 0.0))) if finite_cons.size else 0.0

    layer_support, support_hits, support_total = _layer_support_counts(lpc_profiles, start, end)
    scale_support_fraction = (support_hits / support_total) if support_total else 0.0

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

    length_signal = end - start + 1
    # ValleyScore terms:
    # - contrast: mean ConsensusLPC inside the valley (depth of local complexity collapse)
    # - persistence: fraction of valley positions with ConsensusLPC > 0
    # - area: integrated positive ConsensusLPC mass across the valley
    # - scale_support_fraction / layer_support_fraction: cross-scale and cross-layer agreement
    # +1 on area keeps very small-but-supported valleys from collapsing to zero score.
    valley_score_raw = contrast * persistence * (area + 1.0) * scale_support_fraction * (layer_support_fraction + EPSILON)

    start_nt, end_nt = _nucleotide_bounds(start, end, len(seq), p_window)
    sequence = seq[start_nt: end_nt + 1]
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    k_start, k_end = kadane_core
    core_overlap = bool(k_start is not None and k_end is not None and not (end < k_start or start > k_end))

    return {
        "ID": f"PV_{valley_index:06d}",
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Signal_Length": int(length_signal),
        "Start": int(start_nt),
        "End": int(end_nt),
        "Length": int(end_nt - start_nt + 1),
        "MeanMono": mono_mean,
        "MeanDi": di_mean,
        "MeanTri": tri_mean,
        "Contrast": float(contrast),
        "Persistence": float(persistence),
        "Area": float(area),
        "MonoSupport": f"{layer_support['mono'][0]}/{layer_support['mono'][1]}",
        "DiSupport": f"{layer_support['di'][0]}/{layer_support['di'][1]}",
        "TriSupport": f"{layer_support['tri'][0]}/{layer_support['tri'][1]}",
        "ScaleSupport": f"{support_hits}/{support_total}",
        "ScaleSupportFraction": float(scale_support_fraction),
        "LayerSupport": f"{layer_hits}/{len(LAYERS)}",
        "LayerSupportFraction": float(layer_support_fraction),
        "OverallSupport": f"{support_hits}/{support_total}",
        "ValleyScore_raw": float(valley_score_raw),
        "ConsensusPeak": float(np.max(finite_cons)) if finite_cons.size else 0.0,
        "GC%": float(gc),
        "KadaneCoreOverlap": core_overlap,
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
    return domains


def find_domains(
    mono: np.ndarray,
    di: np.ndarray,
    tri: np.ndarray,
    lpc_profiles: dict[str, dict[int, np.ndarray]],
    layer_consensus: dict[str, np.ndarray],
    consensus_lpc: np.ndarray,
    seq: str,
    min_domain: int = MIN_DOMAIN,
    max_domain: int = MAX_DOMAIN,
    merge_gap: int = MERGE_GAP,
    p_window: int = PERPLEXITY_WINDOW,
) -> tuple[list[dict], tuple[int | None, int | None]]:
    n = len(consensus_lpc)
    if n == 0:
        return [], (None, None)

    # Step 6: bounded Kadane-like core detection executed once on ConsensusLPC.
    work = -consensus_lpc.astype(np.float64)
    work[~np.isfinite(work)] = np.nan
    finite = np.isfinite(work)
    if not np.any(finite):
        return [], (None, None)

    core_start = core_end = None
    best_score = np.inf
    i = 0
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < n and finite[j]:
            j += 1
        seg = work[i:j]
        if len(seg) >= min_domain:
            s, e, mv = bounded_min_mean(seg, min_len=min_domain, max_len=max_domain)
            if s is not None and np.isfinite(mv) and mv < best_score:
                best_score = mv
                core_start, core_end = i + s, i + e
        i = j

    intervals: list[tuple[int, int]] = []
    if core_start is not None and core_end is not None and best_score < 0:
        intervals.append(_expand_valley(consensus_lpc, core_start, core_end))

    intervals.extend(_split_positive_runs(consensus_lpc))
    merged = [(s, e) for s, e in _merge_valleys(intervals, merge_gap) if e - s + 1 >= min_domain]

    domains: list[dict] = []
    for idx, (s, e) in enumerate(merged, 1):
        if e - s + 1 > max_domain:
            e = s + max_domain - 1
        domains.append(
            valley_statistics(
                valley_index=idx,
                start=s,
                end=e,
                mono=mono,
                di=di,
                tri=tri,
                lpc_profiles=lpc_profiles,
                layer_consensus=layer_consensus,
                consensus_lpc=consensus_lpc,
                seq=seq,
                p_window=p_window,
                kadane_core=(core_start, core_end),
            )
        )
    domains.sort(key=lambda d: d["Signal_Start"])
    return normalize_valley_score(domains), (core_start, core_end)


def _resolve_scales(scales: list[int] | None) -> list[int]:
    if scales:
        return sorted(set(max(10, int(s)) for s in scales))
    return DEFAULT_SCALES.copy()


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
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
    }
    scales = _resolve_scales(params["scales"])
    params["resolved_scales"] = scales

    # Step 1: one-pass layer computations.
    mono = compute_mono_perplexity(seq, params["perplexity_window"])
    di = compute_di_perplexity(seq, params["perplexity_window"])
    tri = compute_tri_perplexity(seq, params["perplexity_window"])

    # Step 2–4: multi-scale and per-layer consensus.
    layer_signals = {"mono": mono, "di": di, "tri": tri}
    landscapes: dict[str, dict[int, np.ndarray]] = {}
    lpc_profiles: dict[str, dict[int, np.ndarray]] = {}
    layer_consensus: dict[str, np.ndarray] = {}
    for layer in LAYERS:
        landscapes[layer] = build_landscapes(layer_signals[layer], scales, params["landscape_method"])
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

    # Step 5: final ensemble consensus.
    consensus_lpc = build_ensemble_consensus(layer_consensus, method=params["ensemble_method"])

    # Step 6–8: one-pass core detection + expanded valley reporting.
    domains, kadane_core = find_domains(
        mono=mono,
        di=di,
        tri=tri,
        lpc_profiles=lpc_profiles,
        layer_consensus=layer_consensus,
        consensus_lpc=consensus_lpc,
        seq=seq,
        min_domain=params["min_domain"],
        max_domain=params["max_domain"],
        merge_gap=params["merge_gap"],
        p_window=params["perplexity_window"],
    )
    for d in domains:
        d["Sequence_ID"] = sequence_id

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        mono=mono,
        di=di,
        tri=tri,
        landscapes=landscapes,
        lpc_profiles=lpc_profiles,
        layer_consensus=layer_consensus,
        consensus_lpc=consensus_lpc,
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
            f"Contrast={row['Contrast']:.4f};"
            f"Persistence={row['Persistence']:.4f};"
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
    parser = argparse.ArgumentParser(description="REGPLEX v10 — hierarchical perplexity ensemble")
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
    )
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
        )
        for header, sequence in records
    ]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} valleys to {args.out}")


if __name__ == "__main__":
    cli()
