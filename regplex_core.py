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

# ─── Tunable scientific defaults ────────────────────────────────────────────
PERPLEXITY_WINDOW = 17    # 17 nt → 16 dinucleotide transitions (stable local estimate)
SG_WINDOW_LENGTH  = 21    # Savitzky-Golay window length (must be odd)
SG_POLY_ORDER     = 3     # Savitzky-Golay polynomial order
FLANK_SIZE        = 100   # PDS upstream/downstream flank window (bp)
SPACER_SIZE       = 50    # PDS spacer between flank and candidate (bp)
MIN_CANDIDATE     = 50    # Minimum PDS candidate window (bp)
MAX_CANDIDATE     = 1000  # Maximum PDS candidate window (bp)
MIN_VALLEY_LENGTH = 100   # Minimum accepted valley length (bp)
MAX_VALLEY_LENGTH = 1000  # Maximum accepted valley length (bp)
MERGE_GAP         = 100   # Merge adjacent valleys within this distance (bp)
TOP_N_DISPLAY     = 20    # Default top-N display limit
EPSILON           = 1e-9

_IUPAC_DNA = set("ACGTN")
_MAP = np.full(256, 4, dtype=np.uint8)
for _base, _idx in zip("ACGT", range(4)):
    _MAP[ord(_base)] = _idx


# ─── Data structure ─────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    di: np.ndarray
    smoothed_di: np.ndarray
    pds: np.ndarray
    domains: list[dict]
    params: dict


# ─── FASTA parser ───────────────────────────────────────────────────────────

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


# ─── Prefix-sum utilities ───────────────────────────────────────────────────

def _prefix(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (cumulative sum, cumulative valid count) arrays of length n+1."""
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
    """Vectorised window-mean computation from prefix arrays."""
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


# ─── Step 1 – Dinucleotide perplexity ───────────────────────────────────────

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

    With window=17 there are 16 dinucleotide transitions, giving a stable
    local estimate while remaining highly local.  This is the ONLY primary
    signal used in REGPLEX v13.
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


# ─── Step 2 – Savitzky-Golay smoothing ──────────────────────────────────────

def smooth_perplexity(
    arr: np.ndarray,
    window_length: int = SG_WINDOW_LENGTH,
    poly_order: int = SG_POLY_ORDER,
) -> np.ndarray:
    """Apply a single Savitzky-Golay pass.  Never called more than once."""
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


# ─── Step 3 – Perplexity Depression Score ───────────────────────────────────

def compute_pds(
    smoothed_di: np.ndarray,
    flank_size: int = FLANK_SIZE,
    spacer_size: int = SPACER_SIZE,
    min_candidate: int = MIN_CANDIDATE,
    max_candidate: int = MAX_CANDIDATE,
) -> np.ndarray:
    """Compute Perplexity Depression Score (PDS) for every genomic position.

    Three-window layout around each position:

        [upstream flank] [spacer] [candidate] [spacer] [downstream flank]
           flank_size     spacer    cand_w     spacer     flank_size

    The candidate window width is set to the geometric mean of the allowed
    bounds, making it adaptive to the configured detection range.

    PDS  =  ((UpstreamMean + DownstreamMean) / 2)  −  CandidateMean

    Biological filters applied before assigning PDS:
      • UpstreamMean   > CandidateMean  (upstream background is higher)
      • DownstreamMean > CandidateMean  (downstream background is higher)

    Positions failing either filter are set to NaN.
    Positive PDS values indicate genuine depressions relative to local
    background; negative values indicate local peaks or uniform regions.
    """
    n = len(smoothed_di)
    if n == 0:
        return np.array([], dtype=np.float32)

    # Adaptive candidate width: geometric mean of bounds (≈ 224 bp for 50–1000)
    cand_w = int(round(math.sqrt(min_candidate * max_candidate)))
    cand_w = min(max(cand_w, min_candidate), max_candidate)

    csum, ccount = _prefix(smoothed_di)
    centers = np.arange(n, dtype=np.int64)

    half_cand = cand_w // 2
    cand_s = centers - half_cand
    cand_e = cand_s + cand_w

    up_e = cand_s - spacer_size
    up_s = up_e - flank_size

    dn_s = cand_e + spacer_size
    dn_e = dn_s + flank_size

    up_mean   = _window_means(csum, ccount, up_s, up_e)
    cand_mean = _window_means(csum, ccount, cand_s, cand_e)
    dn_mean   = _window_means(csum, ccount, dn_s, dn_e)

    contrast = ((up_mean + dn_mean) / 2.0) - cand_mean

    valid = (
        np.isfinite(up_mean)
        & np.isfinite(cand_mean)
        & np.isfinite(dn_mean)
        & (up_mean > cand_mean)
        & (dn_mean > cand_mean)
    )
    return np.where(valid, contrast, np.nan).astype(np.float32)


# ─── Step 4 – Bounded Kadane valley detection ───────────────────────────────

def _bounded_max_mean(
    arr: np.ndarray,
    min_len: int,
    max_len: int,
) -> tuple[int | None, int | None, float]:
    """Find the contiguous subarray with maximum mean and length in [min_len, max_len].

    Uses a monotone-deque O(n) algorithm: for each end position j, the deque
    tracks candidate start positions with minimum prefix sum, which maximises
    the sum — and therefore the mean — of the selected segment.
    """
    n = len(arr)
    if n < min_len:
        return None, None, -np.inf
    pref = np.empty(n + 1, dtype=np.float64)
    pref[0] = 0.0
    pref[1:] = np.cumsum(arr)
    best_mean = -np.inf
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
            if mean > best_mean:
                best_mean = mean
                best_i, best_j = i, j
    return best_i, best_j, float(best_mean)


def _find_all_pds_valleys(
    pds: np.ndarray,
    min_valley_length: int,
    max_valley_length: int,
) -> list[tuple[int, int]]:
    """Return valley cores from all contiguous positive-PDS runs.

    For each run of length >= min_valley_length:
    - If the run fits within max_valley_length, use the whole run as a core.
    - Otherwise, apply bounded max-mean Kadane to extract the strongest
      sub-core of length in [min_valley_length, max_valley_length].

    Returns a list of (core_start, core_end) index tuples.  All detected
    valleys are returned — not only the best one.
    """
    n = len(pds)
    if n < min_valley_length:
        return []

    positive = np.isfinite(pds) & (pds > 0)

    # Extract all contiguous positive runs
    padded = np.concatenate([[False], positive, [False]])
    delta = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(delta > 0)
    ends   = np.flatnonzero(delta < 0) - 1

    cores: list[tuple[int, int]] = []
    for run_s, run_e in zip(starts, ends):
        run_len = run_e - run_s + 1
        if run_len < min_valley_length:
            continue
        if run_len <= max_valley_length:
            cores.append((int(run_s), int(run_e)))
        else:
            seg = pds[run_s: run_e + 1].astype(np.float64)
            s, e, mv = _bounded_max_mean(seg, min_valley_length, max_valley_length)
            if s is not None and np.isfinite(mv):
                cores.append((int(run_s + s), int(run_s + e)))
    return cores


# ─── Step 5 – Valley expansion ──────────────────────────────────────────────

def _expand_valley_pds(
    pds: np.ndarray,
    core_start: int,
    core_end: int,
) -> tuple[int, int]:
    """Expand valley boundaries while PDS > 0 OR PDS > 20 % of peak PDS.

    Vectorised O(n) implementation: builds a boolean include-mask, then
    scans left/right from the core boundaries for the first excluded position.
    """
    n = len(pds)
    core_finite = pds[core_start: core_end + 1]
    core_finite = core_finite[np.isfinite(core_finite)]
    peak_pds = float(np.max(core_finite)) if core_finite.size else 0.0
    threshold = 0.2 * peak_pds  # 20 % of peak

    include = np.isfinite(pds) & ((pds > 0) | (pds > threshold))

    # Left expansion
    if core_start == 0:
        left = 0
    else:
        rev = include[:core_start][::-1]
        false_in_rev = np.flatnonzero(~rev)
        left = 0 if false_in_rev.size == 0 else int(core_start - false_in_rev[0])

    # Right expansion
    if core_end >= n - 1:
        right = n - 1
    else:
        fwd = include[core_end + 1:]
        false_fwd = np.flatnonzero(~fwd)
        right = n - 1 if false_fwd.size == 0 else int(core_end + false_fwd[0])

    return left, right


# ─── Step 6 – Valley merging ─────────────────────────────────────────────────

def _merge_valleys(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Merge adjacent valleys whose gap is ≤ gap bp."""
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


# ─── Step 7 – Valley metrics ─────────────────────────────────────────────────

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
    """Compute all metrics for a single valley."""
    sl = slice(start, end + 1)
    n = len(smoothed_di)

    def _fmean(a: np.ndarray) -> float:
        f = a[np.isfinite(a)]
        return float(np.mean(f)) if f.size else float("nan")

    def _fmin(a: np.ndarray) -> float:
        f = a[np.isfinite(a)]
        return float(np.min(f)) if f.size else float("nan")

    def _fmax(a: np.ndarray) -> float:
        f = a[np.isfinite(a)]
        return float(np.max(f)) if f.size else float("nan")

    # Perplexity stats (raw di)
    di_seg = di[sl]
    mean_perplexity = _fmean(di_seg)
    min_perplexity  = _fmin(di_seg)
    max_perplexity  = _fmax(di_seg)

    # PDS stats
    pds_seg    = pds[sl]
    pds_finite = pds_seg[np.isfinite(pds_seg)]
    pds_mean   = float(np.mean(pds_finite))   if pds_finite.size else 0.0
    pds_max    = float(np.max(pds_finite))    if pds_finite.size else 0.0
    persistence = float(np.mean(pds_finite > 0)) if pds_finite.size else 0.0
    area        = float(np.sum(np.maximum(pds_finite, 0.0))) if pds_finite.size else 0.0
    variance    = float(np.var(pds_finite))   if pds_finite.size > 1 else 0.0

    # Prominence: background mean − valley minimum
    bg_up_s  = max(0, start - spacer_size - flank_size)
    bg_up_e  = max(0, start - spacer_size)
    bg_dn_s  = min(n, end + spacer_size + 1)
    bg_dn_e  = min(n, end + spacer_size + 1 + flank_size)
    bg_parts = []
    if bg_up_e > bg_up_s:
        bg_parts.append(smoothed_di[bg_up_s: bg_up_e])
    if bg_dn_e > bg_dn_s:
        bg_parts.append(smoothed_di[bg_dn_s: bg_dn_e])
    if bg_parts:
        bg_all = np.concatenate(bg_parts)
        bg_finite = bg_all[np.isfinite(bg_all)]
        background = float(np.mean(bg_finite)) if bg_finite.size else float("nan")
    else:
        background = float("nan")
    val_seg_s = smoothed_di[max(0, start): end + 1]
    val_min   = _fmin(val_seg_s)
    prominence = max(0.0, background - val_min) if np.isfinite(background) and np.isfinite(val_min) else 0.0

    # Three-window means (using smoothed profile, valley boundaries as candidate)
    up_mean   = _fmean(smoothed_di[bg_up_s: bg_up_e]) if bg_up_e > bg_up_s else float("nan")
    cand_mean = _fmean(smoothed_di[max(0, start): end + 1])
    dn_mean   = _fmean(smoothed_di[bg_dn_s: bg_dn_e]) if bg_dn_e > bg_dn_s else float("nan")
    upstream_diff   = up_mean - cand_mean   if np.isfinite(up_mean)   and np.isfinite(cand_mean) else float("nan")
    downstream_diff = dn_mean - cand_mean   if np.isfinite(dn_mean)   and np.isfinite(cand_mean) else float("nan")

    # Nucleotide-coordinate mapping (signal positions → sequence positions)
    start_nt = max(0, int(start))
    end_nt   = min(len(seq) - 1, int(end + p_window - 1))
    sequence = seq[start_nt: end_nt + 1]
    length   = end_nt - start_nt + 1
    gc       = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    # ValleyScore (raw; normalised later)
    log_length       = math.log(max(end - start + 1, 1))
    stability        = 1.0 / (variance + EPSILON)
    valley_score_raw = pds_mean * persistence * log_length * stability

    return {
        "ID":                 f"PV_{valley_index:06d}",
        "Signal_Start":       int(start),
        "Signal_End":         int(end),
        "Start":              int(start_nt),
        "End":                int(end_nt),
        "Length":             int(length),
        "MeanPerplexity":     mean_perplexity,
        "MinPerplexity":      min_perplexity,
        "MaxPerplexity":      max_perplexity,
        "UpstreamMean":       up_mean,
        "CandidateMean":      cand_mean,
        "DownstreamMean":     dn_mean,
        "UpstreamDifference": upstream_diff,
        "DownstreamDifference": downstream_diff,
        "PDSMean":            pds_mean,
        "PDSMax":             pds_max,
        "Prominence":         prominence,
        "Persistence":        persistence,
        "AreaUnderValley":    area,
        "Variance":           variance,
        "GC%":                gc,
        "MotifCount":         0,
        "Motifs":             "",
        "Sequence":           sequence,
        "ValleyScore_raw":    valley_score_raw,
    }


# ─── Step 8 – Valley score normalisation & ranking ──────────────────────────

def normalize_valley_score(domains: list[dict]) -> list[dict]:
    """Normalise ValleyScore to [0, 1] and assign genomic rank (1 = best)."""
    if not domains:
        return domains
    raw = np.array([max(0.0, float(d.get("ValleyScore_raw", 0.0))) for d in domains], dtype=np.float64)
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    norm = np.where(raw > 0, 1.0, 0.0) if span < EPSILON else (raw - lo) / span
    for i, d in enumerate(domains):
        d["ValleyScore"]           = float(raw[i])
        d["ValleyScoreNormalized"] = float(norm[i])
        d.pop("ValleyScore_raw", None)
    ranked = sorted(range(len(domains)), key=lambda idx: domains[idx]["ValleyScore"], reverse=True)
    for rank, idx in enumerate(ranked, 1):
        domains[idx]["Rank"] = rank
    return domains


# ─── Main pipeline ───────────────────────────────────────────────────────────

def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    """REGPLEX v13 analysis pipeline.

    Steps
    -----
    1. Dinucleotide perplexity (window = 17 nt).
    2. Savitzky-Golay smoothing — applied exactly once.
    3. Perplexity Depression Score (PDS): three-window background contrast.
    4. Bounded Kadane valley detection on the PDS profile.
    5. Valley expansion while PDS > 0 OR PDS > 20 % of peak PDS.
    6. Valley merging (gap ≤ 100 bp).
    7. Valley metrics, ValleyScore, normalisation, and ranking.
    """
    params = {
        "perplexity_window": int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        "sg_window_length":  int(kwargs.get("sg_window_length",  SG_WINDOW_LENGTH)),
        "sg_poly_order":     int(kwargs.get("sg_poly_order",     SG_POLY_ORDER)),
        "flank_size":        int(kwargs.get("flank_size",        FLANK_SIZE)),
        "spacer_size":       int(kwargs.get("spacer_size",       SPACER_SIZE)),
        "min_candidate":     int(kwargs.get("min_candidate",     MIN_CANDIDATE)),
        "max_candidate":     int(kwargs.get("max_candidate",     MAX_CANDIDATE)),
        "min_valley_length": int(kwargs.get("min_valley_length", MIN_VALLEY_LENGTH)),
        "max_valley_length": int(kwargs.get("max_valley_length", MAX_VALLEY_LENGTH)),
        "merge_gap":         int(kwargs.get("merge_gap",         MERGE_GAP)),
    }

    # Step 1: dinucleotide perplexity
    di = compute_di_perplexity(seq, params["perplexity_window"])

    # Step 2: Savitzky-Golay smoothing (single pass)
    smoothed_di = smooth_perplexity(di, params["sg_window_length"], params["sg_poly_order"])

    # Step 3: Perplexity Depression Score
    pds = compute_pds(
        smoothed_di,
        flank_size=params["flank_size"],
        spacer_size=params["spacer_size"],
        min_candidate=params["min_candidate"],
        max_candidate=params["max_candidate"],
    )

    # Step 4: Bounded Kadane — find all positive-PDS valley cores
    cores = _find_all_pds_valleys(
        pds,
        params["min_valley_length"],
        params["max_valley_length"],
    )

    # Step 5: Expand each core while PDS > 0 or > 20 % of peak
    expanded = [_expand_valley_pds(pds, cs, ce) for cs, ce in cores]

    # Step 6: Merge adjacent expanded valleys
    merged = _merge_valleys(expanded, params["merge_gap"])
    # Re-filter by minimum length after merging (expansion may have shortened edges)
    merged = [(s, e) for s, e in merged if e - s + 1 >= params["min_valley_length"]]

    # Step 7: Compute metrics and scoring
    domains: list[dict] = []
    for idx, (s, e) in enumerate(merged, 1):
        d = _valley_metrics(
            valley_index=idx,
            start=s,
            end=e,
            di=di,
            smoothed_di=smoothed_di,
            pds=pds,
            seq=seq,
            p_window=params["perplexity_window"],
            flank_size=params["flank_size"],
            spacer_size=params["spacer_size"],
        )
        d["Sequence_ID"] = sequence_id
        domains.append(d)

    domains.sort(key=lambda d: d["Signal_Start"])
    domains = normalize_valley_score(domains)

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        di=di,
        smoothed_di=smoothed_di,
        pds=pds,
        domains=domains,
        params=params,
    )


# ─── Output helpers ──────────────────────────────────────────────────────────

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
    bed["End"]   = (bed["End"].astype(int) + 1)
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
            f"Persistence={row['Persistence']:.4f};"
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


# ─── CLI ─────────────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(
        description="REGPLEX v13 — training-free perplexity valley detector"
    )
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_valleys.csv", help="Output CSV path")
    parser.add_argument("--sg-window",    type=int,   default=SG_WINDOW_LENGTH,  help="Savitzky-Golay window (odd)")
    parser.add_argument("--sg-order",     type=int,   default=SG_POLY_ORDER,     help="Savitzky-Golay polynomial order")
    parser.add_argument("--flank-size",   type=int,   default=FLANK_SIZE,        help="PDS flank window (bp)")
    parser.add_argument("--spacer-size",  type=int,   default=SPACER_SIZE,       help="PDS spacer (bp)")
    parser.add_argument("--min-candidate",type=int,   default=MIN_CANDIDATE,     help="Min PDS candidate window (bp)")
    parser.add_argument("--max-candidate",type=int,   default=MAX_CANDIDATE,     help="Max PDS candidate window (bp)")
    parser.add_argument("--min-valley",   type=int,   default=MIN_VALLEY_LENGTH, help="Min valley length (bp)")
    parser.add_argument("--max-valley",   type=int,   default=MAX_VALLEY_LENGTH, help="Max valley length (bp)")
    parser.add_argument("--merge-gap",    type=int,   default=MERGE_GAP,         help="Valley merge gap (bp)")
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        fasta_text = handle.read()

    records = parse_fasta(fasta_text)
    results = [
        analyze_sequence(
            header,
            sequence,
            sg_window_length=args.sg_window,
            sg_poly_order=args.sg_order,
            flank_size=args.flank_size,
            spacer_size=args.spacer_size,
            min_candidate=args.min_candidate,
            max_candidate=args.max_candidate,
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
