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
from scipy.signal import savgol_filter

# ---------------------------
# Tunable scientific defaults
# ---------------------------
PERPLEXITY_WINDOW = 17      # 17 nt → 16 dinucleotide transitions
SG_WINDOW_LENGTH = 21       # Savitzky-Golay window length (odd)
SG_POLY_ORDER = 3           # Savitzky-Golay polynomial order
FLANK_SIZE = 100            # Upstream / downstream context window (bp)
SPACER_SIZE = 50            # Gap between candidate and flanks (bp)
MIN_CANDIDATE = 50          # PDS candidate window size (bp)
MAX_CANDIDATE = 1000        # Maximum candidate window size (bp)
MIN_VALLEY_LENGTH = 100     # Minimum final valley length (bp)
MAX_VALLEY_LENGTH = 1000    # Maximum final valley length (bp)
MERGE_GAP = 100             # Merge adjacent valleys within this gap (bp)
TOP_N_DISPLAY = 20          # Default top-N valleys to display
EPSILON = 1e-9

_IUPAC_DNA = set("ACGTN")
_MAP = np.full(256, 4, dtype=np.uint8)
for _base, _idx in zip("ACGT", range(4)):
    _MAP[ord(_base)] = _idx


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    di: np.ndarray          # raw dinucleotide perplexity
    smoothed_di: np.ndarray # Savitzky-Golay smoothed perplexity
    pds: np.ndarray         # Perplexity Depression Score profile
    domains: list[dict]
    params: dict


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Vectorized prefix-sum helpers
# ---------------------------------------------------------------------------

def _prefix(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(arr).astype(np.float64)
    values = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    csum = np.empty(len(arr) + 1, dtype=np.float64)
    ccount = np.empty(len(arr) + 1, dtype=np.float64)
    csum[0] = ccount[0] = 0.0
    np.cumsum(values, out=csum[1:])
    np.cumsum(valid, out=ccount[1:])
    return csum, ccount


def _window_means(
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


# ---------------------------------------------------------------------------
# Step 1 – Dinucleotide perplexity
# ---------------------------------------------------------------------------

def _compute_entropy_perplexity(indices: np.ndarray, n_states: int) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=np.float32)
    counts = np.zeros((len(indices), n_states), dtype=np.int16)
    np.add.at(counts, (np.arange(len(indices))[:, None], indices), 1)
    denom = np.maximum(indices.shape[1], 1)
    p = np.where(counts == 0, 1.0, counts / denom).astype(np.float32)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return (2.0 ** entropy).astype(np.float32)


def compute_di_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    """Compute dinucleotide perplexity with a sliding window.

    Returns an array of length max(0, len(seq) - window + 1).
    Each value is the Shannon perplexity of the 16 dinucleotide types
    observed within that window.
    """
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


# ---------------------------------------------------------------------------
# Step 2 – Savitzky-Golay smoothing
# ---------------------------------------------------------------------------

def smooth_perplexity(
    arr: np.ndarray,
    window_length: int = SG_WINDOW_LENGTH,
    poly_order: int = SG_POLY_ORDER,
) -> np.ndarray:
    """Apply a single Savitzky-Golay pass to a perplexity profile.

    Applied ONCE only.  NaN positions are reconstructed via linear
    interpolation before smoothing, then restored after.
    """
    n = len(arr)
    if n == 0:
        return arr.copy()
    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1
    po = min(poly_order, wl - 1)
    if wl < 3 or po < 1:
        return arr.astype(np.float32)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return arr.copy()
    work = arr.astype(np.float64)
    if not np.all(finite_mask):
        xp = np.flatnonzero(finite_mask)
        fp = arr[finite_mask].astype(np.float64)
        work = np.interp(np.arange(n, dtype=np.float64), xp, fp)
    smoothed = savgol_filter(work, wl, po).astype(np.float32)
    smoothed[~finite_mask] = np.nan
    return smoothed


# ---------------------------------------------------------------------------
# Step 3 – Perplexity Depression Score
# ---------------------------------------------------------------------------

def compute_pds(
    smoothed: np.ndarray,
    flank_size: int = FLANK_SIZE,
    spacer_size: int = SPACER_SIZE,
    candidate_size: int = MIN_CANDIDATE,
) -> np.ndarray:
    """Compute genome-wide Perplexity Depression Score (PDS).

    For every signal position i, the three-window layout is:

        [upstream: flank_size bp]
        [spacer: spacer_size bp]
        [candidate: candidate_size bp, centered at i]
        [spacer: spacer_size bp]
        [downstream: flank_size bp]

    PDS[i] = ((UpstreamMean + DownstreamMean) / 2) - CandidateMean

    Positions where any window lies outside the signal array receive NaN.
    The biological filter (UpstreamMean > CandidateMean and
    DownstreamMean > CandidateMean) is applied at valley level in
    find_valleys(), not here.
    """
    n = len(smoothed)
    if n == 0:
        return np.array([], dtype=np.float32)

    centers = np.arange(n, dtype=np.int64)
    half_cand = candidate_size // 2

    cand_s = centers - half_cand
    cand_e = cand_s + candidate_size

    up_s = cand_s - spacer_size - flank_size
    up_e = cand_s - spacer_size

    dn_s = cand_e + spacer_size
    dn_e = dn_s + flank_size

    csum, ccount = _prefix(smoothed)
    up_mean = _window_means(csum, ccount, up_s, up_e)
    cand_mean = _window_means(csum, ccount, cand_s, cand_e)
    dn_mean = _window_means(csum, ccount, dn_s, dn_e)

    contrast = ((up_mean + dn_mean) / 2.0) - cand_mean

    # Set NaN where any window is invalid (out-of-bounds or all-NaN).
    # The biological filter (UpstreamMean > CandidateMean, DownstreamMean > CandidateMean)
    # is applied at the valley level in find_valleys(), not per-position.
    valid = np.isfinite(up_mean) & np.isfinite(cand_mean) & np.isfinite(dn_mean)
    pds = np.where(valid, contrast, np.nan)
    return pds.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 – Bounded Kadane on PDS
# ---------------------------------------------------------------------------

def _bounded_kadane_max(arr: np.ndarray, min_len: int, max_len: int) -> tuple[int | None, int | None]:
    """Find sub-array of arr with maximum mean, length in [min_len, max_len].

    Uses a monotone deque in O(n).  Returns (start, end) indices inclusive,
    or (None, None) if no valid sub-array exists.
    """
    n = len(arr)
    if n < min_len:
        return None, None
    # Replace NaN with 0 (neutral contribution)
    clean = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    pref = np.empty(n + 1, dtype=np.float64)
    pref[0] = 0.0
    pref[1:] = np.cumsum(clean)

    best_mean = -np.inf
    best_i = best_j = None
    dq: deque[int] = deque()

    for j in range(n):
        i_min = j - max_len + 1
        i_max = j - min_len + 1
        if i_max < 0:
            continue
        # Maintain deque with minimum prefix sums (front = argmin pref) so that
        # pref[j+1] - pref[dq[0]] is maximised → maximum mean sub-array.
        # Pop from back while the new candidate has a smaller (or equal) prefix sum.
        while dq and pref[dq[-1]] >= pref[i_max]:
            dq.pop()
        dq.append(i_max)
        while dq and dq[0] < i_min:
            dq.popleft()
        if dq:
            i = dq[0]
            mean = (pref[j + 1] - pref[i]) / (j - i + 1)
            if mean > best_mean:
                best_mean = mean
                best_i, best_j = i, j

    return best_i, best_j


def _find_positive_runs(pds: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return all contiguous runs where PDS is finite and positive."""
    runs: list[tuple[int, int]] = []
    positive = np.isfinite(pds) & (pds > 0)
    n = len(pds)
    i = 0
    while i < n:
        if not positive[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and positive[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j))
        i = j + 1
    return runs


# ---------------------------------------------------------------------------
# Step 5 – Valley expansion
# ---------------------------------------------------------------------------

def _expand_valley(pds: np.ndarray, core_start: int, core_end: int) -> tuple[int, int]:
    """Expand valley boundaries while PDS > 20% of peak PDS.

    This is O(n) and gives stable boundaries relative to the signal peak.
    """
    n = len(pds)
    seg = pds[core_start: core_end + 1]
    finite_seg = seg[np.isfinite(seg)]
    peak_pds = float(np.max(finite_seg)) if finite_seg.size else 0.0
    threshold = 0.2 * peak_pds

    left = core_start
    while left > 0 and np.isfinite(pds[left - 1]) and pds[left - 1] >= threshold:
        left -= 1

    right = core_end
    while right < n - 1 and np.isfinite(pds[right + 1]) and pds[right + 1] >= threshold:
        right += 1

    return left, right


# ---------------------------------------------------------------------------
# Step 6 – Valley merging
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
# Helpers for valley metrics
# ---------------------------------------------------------------------------

def _nucleotide_bounds(signal_start: int, signal_end: int, seq_len: int, p_window: int) -> tuple[int, int]:
    start = max(0, int(signal_start))
    end = min(seq_len - 1, int(signal_end + p_window - 1))
    return start, end


def _fmean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _fmin(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.min(finite)) if finite.size else float("nan")


def _fmax(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float("nan")


# ---------------------------------------------------------------------------
# Step 7 – Valley metrics
# ---------------------------------------------------------------------------

def _valley_metrics(
    valley_index: int,
    start: int,
    end: int,
    di: np.ndarray,
    smoothed_di: np.ndarray,
    pds: np.ndarray,
    seq: str,
    p_window: int,
    flank_size: int,
    spacer_size: int,
) -> dict:
    sl = slice(start, end + 1)
    n = len(smoothed_di)

    # Raw perplexity statistics
    di_seg = di[sl] if end < len(di) else np.array([], dtype=np.float32)
    mean_perplexity = _fmean(di_seg)
    min_perplexity = _fmin(di_seg)
    max_perplexity = _fmax(di_seg)

    # Three-window context from smoothed perplexity
    up_s = max(0, start - spacer_size - flank_size)
    up_e = max(0, start - spacer_size)
    dn_s = min(n, end + spacer_size)
    dn_e = min(n, end + spacer_size + flank_size)

    upstream_mean = _fmean(smoothed_di[up_s:up_e]) if up_e > up_s else float("nan")
    candidate_mean = _fmean(smoothed_di[sl])
    downstream_mean = _fmean(smoothed_di[dn_s:dn_e]) if dn_e > dn_s else float("nan")

    upstream_diff = (upstream_mean - candidate_mean) if (
        np.isfinite(upstream_mean) and np.isfinite(candidate_mean)
    ) else float("nan")
    downstream_diff = (downstream_mean - candidate_mean) if (
        np.isfinite(downstream_mean) and np.isfinite(candidate_mean)
    ) else float("nan")

    # PDS statistics
    pds_seg = pds[sl] if end < len(pds) else np.array([], dtype=np.float32)
    pds_finite = pds_seg[np.isfinite(pds_seg)]
    pds_mean = float(np.mean(pds_finite)) if pds_finite.size else 0.0
    pds_max = float(np.max(pds_finite)) if pds_finite.size else 0.0

    # Prominence: ((UpstreamMean + DownstreamMean) / 2) - min(valley)
    flanks_mean = float("nan")
    if np.isfinite(upstream_mean) and np.isfinite(downstream_mean):
        flanks_mean = (upstream_mean + downstream_mean) / 2.0
    elif np.isfinite(upstream_mean):
        flanks_mean = upstream_mean
    elif np.isfinite(downstream_mean):
        flanks_mean = downstream_mean
    prominence = max(0.0, flanks_mean - min_perplexity) if (
        np.isfinite(flanks_mean) and np.isfinite(min_perplexity)
    ) else 0.0

    # Persistence: fraction of positions with PDS > 0
    persistence = float(np.mean(pds_finite > 0)) if pds_finite.size else 0.0

    # Area under valley (sum of positive PDS)
    area = float(np.sum(np.maximum(pds_finite, 0.0))) if pds_finite.size else 0.0

    # Variance of PDS
    variance = float(np.var(pds_finite)) if pds_finite.size > 1 else 0.0

    # Nucleotide-space coordinates and sequence
    length_signal = end - start + 1
    start_nt, end_nt = _nucleotide_bounds(start, end, len(seq), p_window)
    sequence = seq[start_nt: end_nt + 1]
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    # Score (filled in later by normalize_valley_score)
    log_length = math.log(max(length_signal, 1))
    stability = 1.0 / (variance + EPSILON)
    valley_score_raw = pds_mean * persistence * log_length * stability

    return {
        "ID": f"PV_{valley_index:06d}",
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Start": int(start_nt),
        "End": int(end_nt),
        "Length": int(end_nt - start_nt + 1),
        "MeanPerplexity": mean_perplexity,
        "MinPerplexity": min_perplexity,
        "MaxPerplexity": max_perplexity,
        "UpstreamMean": upstream_mean,
        "CandidateMean": candidate_mean,
        "DownstreamMean": downstream_mean,
        "UpstreamDifference": upstream_diff,
        "DownstreamDifference": downstream_diff,
        "PDSMean": pds_mean,
        "PDSMax": pds_max,
        "Prominence": prominence,
        "Persistence": persistence,
        "AreaUnderValley": area,
        "Variance": variance,
        "GC%": gc,
        "MotifCount": 0,
        "Motifs": "",
        "Sequence": sequence,
        "ValleyScore_raw": float(valley_score_raw),
    }


# ---------------------------------------------------------------------------
# Step 8 – Valley scoring and normalization
# ---------------------------------------------------------------------------

def normalize_valley_score(domains: list[dict]) -> list[dict]:
    """Normalize ValleyScore to [0, 1] and assign rank."""
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
    ranked = sorted(range(len(domains)), key=lambda idx: domains[idx]["ValleyScore"], reverse=True)
    for rank, idx in enumerate(ranked, 1):
        domains[idx]["Rank"] = rank
    return domains


# ---------------------------------------------------------------------------
# Main valley-finding pipeline
# ---------------------------------------------------------------------------

def find_valleys(
    pds: np.ndarray,
    smoothed_di: np.ndarray,
    di: np.ndarray,
    seq: str,
    min_len: int = MIN_VALLEY_LENGTH,
    max_len: int = MAX_VALLEY_LENGTH,
    merge_gap: int = MERGE_GAP,
    flank_size: int = FLANK_SIZE,
    spacer_size: int = SPACER_SIZE,
    p_window: int = PERPLEXITY_WINDOW,
) -> list[dict]:
    """Full REGPLEX v13 valley detection pipeline.

    1. Find positive PDS runs (≥ min_len).
    2. Apply bounded Kadane to find best core within [min_len, max_len].
    3. Expand core while PDS ≥ 20% of peak.
    4. Merge valleys within merge_gap.
    5. Trim merged valleys > max_len with Kadane.
    6. Score and rank.
    """
    if len(pds) == 0:
        return []

    # Step 1: positive PDS runs
    runs = _find_positive_runs(pds, min_len)
    if not runs:
        return []

    # Steps 2–3: Kadane core + expansion for each run
    valley_intervals: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        seg = pds[run_start: run_end + 1]
        run_len = run_end - run_start + 1

        if run_len <= max_len:
            # Run fits within bounds: use it directly (no Kadane needed)
            exp_start, exp_end = _expand_valley(pds, run_start, run_end)
        else:
            # Run exceeds max: find densest core with Kadane
            ks, ke = _bounded_kadane_max(seg, min_len, max_len)
            if ks is None:
                continue
            ks += run_start
            ke += run_start
            exp_start, exp_end = _expand_valley(pds, ks, ke)

        if exp_end - exp_start + 1 >= min_len:
            valley_intervals.append((exp_start, exp_end))

    if not valley_intervals:
        return []

    # Step 4: merge
    merged = _merge_valleys(valley_intervals, merge_gap)

    # Step 5: trim merged valleys that exceed max_len
    final_intervals: list[tuple[int, int]] = []
    for s, e in merged:
        length = e - s + 1
        if length < min_len:
            continue
        if length > max_len:
            seg = pds[s: e + 1]
            ks, ke = _bounded_kadane_max(seg, min_len, max_len)
            if ks is None:
                continue
            s, e = s + ks, s + ke
            if e - s + 1 < min_len:
                continue
        final_intervals.append((s, e))

    if not final_intervals:
        return []

    # Step 6: compute metrics and apply valley-level biological filter
    domains: list[dict] = []
    for idx, (s, e) in enumerate(sorted(final_intervals), 1):
        d = _valley_metrics(idx, s, e, di, smoothed_di, pds, seq, p_window, flank_size, spacer_size)
        # Biological filter: both upstream and downstream must exceed the candidate mean
        up_m = d.get("UpstreamMean", float("nan"))
        dn_m = d.get("DownstreamMean", float("nan"))
        cand_m = d.get("CandidateMean", float("nan"))
        if np.isfinite(up_m) and np.isfinite(dn_m) and np.isfinite(cand_m):
            if up_m <= cand_m or dn_m <= cand_m:
                continue
        domains.append(d)

    if not domains:
        return []

    # Re-index IDs after filtering
    for idx, d in enumerate(domains, 1):
        d["ID"] = f"PV_{idx:06d}"

    return normalize_valley_score(domains)


# ---------------------------------------------------------------------------
# Top-level analysis entry point
# ---------------------------------------------------------------------------

def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    """REGPLEX v13 analysis pipeline.

    Steps
    -----
    1.  Dinucleotide perplexity (17 nt window, 16 di-transitions).
    2.  Savitzky-Golay smoothing (applied once).
    3.  Perplexity Depression Score (PDS) — three-window contrast.
    4.  Bounded Kadane detection on PDS.
    5.  Valley expansion.
    6.  Valley merging (gap ≤ 100 bp).
    7.  Valley metrics and scoring.
    """
    params = {
        "perplexity_window": int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        "sg_window_length": int(kwargs.get("sg_window_length", SG_WINDOW_LENGTH)),
        "sg_poly_order": int(kwargs.get("sg_poly_order", SG_POLY_ORDER)),
        "flank_size": int(kwargs.get("flank_size", FLANK_SIZE)),
        "spacer_size": int(kwargs.get("spacer_size", SPACER_SIZE)),
        "min_candidate": int(kwargs.get("min_candidate", MIN_CANDIDATE)),
        "min_valley_length": int(kwargs.get("min_valley_length", MIN_VALLEY_LENGTH)),
        "max_valley_length": int(kwargs.get("max_valley_length", MAX_VALLEY_LENGTH)),
        "merge_gap": int(kwargs.get("merge_gap", MERGE_GAP)),
    }

    di = compute_di_perplexity(seq, params["perplexity_window"])
    smoothed_di = smooth_perplexity(di, params["sg_window_length"], params["sg_poly_order"])
    pds = compute_pds(
        smoothed_di,
        flank_size=params["flank_size"],
        spacer_size=params["spacer_size"],
        candidate_size=params["min_candidate"],
    )

    domains = find_valleys(
        pds=pds,
        smoothed_di=smoothed_di,
        di=di,
        seq=seq,
        min_len=params["min_valley_length"],
        max_len=params["max_valley_length"],
        merge_gap=params["merge_gap"],
        flank_size=params["flank_size"],
        spacer_size=params["spacer_size"],
        p_window=params["perplexity_window"],
    )

    for d in domains:
        d["Sequence_ID"] = sequence_id

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        di=di,
        smoothed_di=smoothed_di,
        pds=pds,
        domains=domains,
        params=params,
    )


# ---------------------------------------------------------------------------
# Tabular output helpers
# ---------------------------------------------------------------------------

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
            f">{row['ID']}|{row['Sequence_ID']}:{row['Start']}-{row['End']}"
            f"|ValleyScore={row['ValleyScore']:.4f}"
        )
        lines.append(str(row["Sequence"]))
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def export_gff(df: pd.DataFrame, gff3: bool = False) -> bytes:
    rows: list[str] = []
    for _, row in df.iterrows():
        attr = (
            f"ID={row['ID']};"
            f"ValleyScore={row['ValleyScore']:.4f};"
            f"PDSMean={row.get('PDSMean', 0.0):.4f};"
            f"Persistence={row.get('Persistence', 0.0):.4f};"
            f"Prominence={row.get('Prominence', 0.0):.4f};"
            f"AreaUnderValley={row.get('AreaUnderValley', 0.0):.4f};"
            f"Motifs={row.get('Motifs', '')}"
        )
        rows.append(
            "\t".join([
                str(row["Sequence_ID"]),
                "REGPLEX",
                "perplexity_valley",
                str(int(row["Start"]) + 1),
                str(int(row["End"]) + 1),
                f"{float(row['ValleyScore']):.4f}",
                ".",
                ".",
                attr,
            ])
        )
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser(
        description="REGPLEX v13 — training-free genomic perplexity valley detector"
    )
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_valleys.csv", help="Output CSV path")
    parser.add_argument("--perplexity-window", type=int, default=PERPLEXITY_WINDOW,
                        help=f"Di-perplexity window size (default {PERPLEXITY_WINDOW})")
    parser.add_argument("--sg-window", type=int, default=SG_WINDOW_LENGTH,
                        help=f"Savitzky-Golay window length, odd (default {SG_WINDOW_LENGTH})")
    parser.add_argument("--sg-order", type=int, default=SG_POLY_ORDER,
                        help=f"Savitzky-Golay polynomial order (default {SG_POLY_ORDER})")
    parser.add_argument("--flank-size", type=int, default=FLANK_SIZE,
                        help=f"Upstream/downstream context window bp (default {FLANK_SIZE})")
    parser.add_argument("--spacer-size", type=int, default=SPACER_SIZE,
                        help=f"Spacer between candidate and flanks bp (default {SPACER_SIZE})")
    parser.add_argument("--min-valley", type=int, default=MIN_VALLEY_LENGTH,
                        help=f"Minimum valley length bp (default {MIN_VALLEY_LENGTH})")
    parser.add_argument("--max-valley", type=int, default=MAX_VALLEY_LENGTH,
                        help=f"Maximum valley length bp (default {MAX_VALLEY_LENGTH})")
    parser.add_argument("--merge-gap", type=int, default=MERGE_GAP,
                        help=f"Merge gap bp (default {MERGE_GAP})")
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        fasta_text = handle.read()

    records = parse_fasta(fasta_text)
    results = [
        analyze_sequence(
            header,
            sequence,
            perplexity_window=args.perplexity_window,
            sg_window_length=args.sg_window,
            sg_poly_order=args.sg_order,
            flank_size=args.flank_size,
            spacer_size=args.spacer_size,
            min_valley_length=args.min_valley,
            max_valley_length=args.max_valley,
            merge_gap=args.merge_gap,
        )
        for header, sequence in records
    ]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} valleys to {args.out}")


if __name__ == "__main__":
    cli()
