from __future__ import annotations

import argparse
import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ==========================================================================
# 1. Global Parameters
# ==========================================================================
PERPLEXITY_WINDOW = 17
SG_WINDOW_LENGTH = 21
SG_POLY_ORDER = 3
FLANK_SIZE = 100
SPACER_SIZE = 50
MIN_CANDIDATE = 50
MAX_CANDIDATE = 1000
MIN_REGION_LENGTH = 100
MAX_REGION_LENGTH = 1000
MERGE_GAP = 100
TOP_N_DISPLAY = 20
EXPANSION_THRESHOLD_FRACTION = 0.2
EPSILON = 1e-9

_IUPAC_DNA = set("ACGTN")
_MAP = np.full(256, 4, dtype=np.uint8)
for _base, _idx in zip("ACGT", range(4)):
    _MAP[ord(_base)] = _idx


@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    di: np.ndarray
    smoothed_di: np.ndarray
    pds: np.ndarray
    regions: list[dict]
    params: dict


# ==========================================================================
# Utility Functions
# ==========================================================================
def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Purpose: Parse FASTA text into sanitized DNA records.

    Parameters:
    - text: FASTA-formatted string.

    Returns:
    - List of (header, sequence) tuples with A/C/G/T/N symbols only.

    Computational complexity:
    - O(n) time and O(n) space for input size n.
    """
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


def _prefix(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Purpose: Build prefix sums and valid-value counts for NaN-aware means.

    Parameters:
    - arr: Input numeric array.

    Returns:
    - Tuple (prefix_sum, prefix_count), each of length len(arr)+1.

    Computational complexity:
    - O(n) time and O(n) space.
    """
    valid = np.isfinite(arr).astype(np.float64)
    values = np.where(np.isfinite(arr), arr.astype(np.float64), 0.0)
    csum = np.empty(len(arr) + 1, dtype=np.float64)
    ccount = np.empty(len(arr) + 1, dtype=np.float64)
    csum[0] = ccount[0] = 0.0
    np.cumsum(values, out=csum[1:])
    np.cumsum(valid, out=ccount[1:])
    return csum, ccount


def _window_means(csum: np.ndarray, ccount: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Purpose: Compute vectorized NaN-aware means for many windows.

    Parameters:
    - csum: Prefix sum array.
    - ccount: Prefix valid-count array.
    - starts: Window starts.
    - ends: Window ends (exclusive).

    Returns:
    - Array of window means with NaN for invalid windows.

    Computational complexity:
    - O(k) time and O(k) space for k windows.
    """
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


# ==========================================================================
# 2. Dinucleotide Perplexity
# ==========================================================================
def _compute_entropy_perplexity(indices: np.ndarray, n_states: int) -> np.ndarray:
    """Purpose: Convert per-window state indices into entropy-derived perplexity.

    Parameters:
    - indices: 2D integer array of encoded state observations per window.
    - n_states: Number of categorical states.

    Returns:
    - Per-window perplexity values.

    Computational complexity:
    - O(w*s) time and O(w*s) space for w windows and s states.
    """
    if len(indices) == 0:
        return np.array([], dtype=np.float32)
    counts = np.zeros((len(indices), n_states), dtype=np.int16)
    np.add.at(counts, (np.arange(len(indices))[:, None], indices), 1)
    denom = np.maximum(indices.shape[1], 1)
    p = np.where(counts == 0, 1.0, counts / denom).astype(np.float32)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return (2.0 ** entropy).astype(np.float32)


def compute_di_perplexity(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    """Purpose: Compute local dinucleotide perplexity from DNA sequence.

    Parameters:
    - seq: DNA sequence.
    - window: Sliding window length (default 17 nt).

    Returns:
    - Dinucleotide perplexity profile as float32 array.

    Computational complexity:
    - O(n) time and O(n) space for sequence length n.
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


# ==========================================================================
# 3. Savitzky–Golay Smoothing
# ==========================================================================
def smooth_perplexity(arr: np.ndarray, window_length: int = SG_WINDOW_LENGTH, poly_order: int = SG_POLY_ORDER) -> np.ndarray:
    """Purpose: Apply one Savitzky–Golay smoothing pass to perplexity signal.

    Parameters:
    - arr: Input perplexity profile.
    - window_length: Smoothing window (odd).
    - poly_order: Polynomial order.

    Returns:
    - Smoothed perplexity profile.

    Computational complexity:
    - O(n) time and O(n) space.
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


# ==========================================================================
# 4. Perplexity Depression Score
# ==========================================================================
def compute_pds(
    smoothed_di: np.ndarray,
    flank_size: int = FLANK_SIZE,
    spacer_size: int = SPACER_SIZE,
    min_candidate: int = MIN_CANDIDATE,
    max_candidate: int = MAX_CANDIDATE,
) -> np.ndarray:
    """Purpose: Compute PDS using bilateral local-background contrast.

    Parameters:
    - smoothed_di: Smoothed perplexity profile.
    - flank_size: Upstream/downstream flank length.
    - spacer_size: Spacer between flank and candidate.
    - min_candidate: Minimum candidate length.
    - max_candidate: Maximum candidate length.

    Returns:
    - PDS profile (NaN where bilateral contrast criteria fail).

    Computational complexity:
    - O(n) time and O(n) space.
    """
    n = len(smoothed_di)
    if n == 0:
        return np.array([], dtype=np.float32)

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

    up_mean = _window_means(csum, ccount, up_s, up_e)
    reg_mean = _window_means(csum, ccount, cand_s, cand_e)
    dn_mean = _window_means(csum, ccount, dn_s, dn_e)

    contrast = ((up_mean + dn_mean) / 2.0) - reg_mean
    valid = (
        np.isfinite(up_mean)
        & np.isfinite(reg_mean)
        & np.isfinite(dn_mean)
        & (up_mean > reg_mean)
        & (dn_mean > reg_mean)
    )
    return np.where(valid, contrast, np.nan).astype(np.float32)


# ==========================================================================
# 5. Region Detection
# ==========================================================================
def _bounded_max_mean(arr: np.ndarray, min_len: int, max_len: int) -> tuple[int | None, int | None, float]:
    """Purpose: Find max-mean segment under length bounds within one run.

    Parameters:
    - arr: Positive-PDS run values.
    - min_len: Minimum segment length.
    - max_len: Maximum segment length.

    Returns:
    - (start, end, max_mean) for best segment; None values when unavailable.

    Computational complexity:
    - O(n) time and O(n) space.
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


def _find_all_pds_regions(pds: np.ndarray, min_region_length: int, max_region_length: int) -> list[tuple[int, int]]:
    """Purpose: Detect positive-PDS core regions using bounded Kadane segmentation.

    Parameters:
    - pds: Perplexity Depression Score profile.
    - min_region_length: Minimum core length.
    - max_region_length: Maximum core length.

    Returns:
    - List of (core_start, core_end) index tuples.

    Computational complexity:
    - O(n) time and O(n) space.
    """
    n = len(pds)
    if n < min_region_length:
        return []

    positive = np.isfinite(pds) & (pds > 0)
    padded = np.concatenate([[False], positive, [False]])
    delta = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(delta > 0)
    ends = np.flatnonzero(delta < 0) - 1

    cores: list[tuple[int, int]] = []
    for run_s, run_e in zip(starts, ends):
        run_len = run_e - run_s + 1
        if run_len < min_region_length:
            continue
        if run_len <= max_region_length:
            cores.append((int(run_s), int(run_e)))
        else:
            seg = pds[run_s: run_e + 1].astype(np.float64)
            s, e, mv = _bounded_max_mean(seg, min_region_length, max_region_length)
            if s is not None and np.isfinite(mv):
                cores.append((int(run_s + s), int(run_s + e)))
    return cores


# ==========================================================================
# 6. Region Expansion
# ==========================================================================
def _expand_region_pds(pds: np.ndarray, core_start: int, core_end: int) -> tuple[int, int]:
    """Purpose: Expand a detected core while support remains above expansion rule.

    Parameters:
    - pds: PDS profile.
    - core_start: Core start index.
    - core_end: Core end index.

    Returns:
    - Expanded (start, end) indices.

    Computational complexity:
    - O(n) time and O(n) space.
    """
    n = len(pds)
    core_finite = pds[core_start: core_end + 1]
    core_finite = core_finite[np.isfinite(core_finite)]
    peak_pds = float(np.max(core_finite)) if core_finite.size else 0.0
    threshold = EXPANSION_THRESHOLD_FRACTION * peak_pds

    include = np.isfinite(pds) & ((pds > 0) | (pds > threshold))

    if core_start == 0:
        left = 0
    else:
        rev = include[:core_start][::-1]
        false_in_rev = np.flatnonzero(~rev)
        left = 0 if false_in_rev.size == 0 else int(core_start - false_in_rev[0])

    if core_end >= n - 1:
        right = n - 1
    else:
        fwd = include[core_end + 1:]
        false_fwd = np.flatnonzero(~fwd)
        right = n - 1 if false_fwd.size == 0 else int(core_end + false_fwd[0])

    return left, right


# ==========================================================================
# 7. Region Merging
# ==========================================================================
def _merge_regions(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Purpose: Merge adjacent region intervals within configured genomic gap.

    Parameters:
    - intervals: Region intervals.
    - gap: Maximum merge distance.

    Returns:
    - Merged and sorted region intervals.

    Computational complexity:
    - O(k log k) time and O(k) space for k intervals.
    """
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


# ==========================================================================
# 8. Region Ranking
# ==========================================================================
def _region_metrics(
    region_index: int,
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
    """Purpose: Compute summary statistics and raw ranking score for one region.

    Parameters:
    - region_index: Sequential region index.
    - start: Region start in signal coordinates.
    - end: Region end in signal coordinates.
    - di: Raw dinucleotide perplexity profile.
    - smoothed_di: Smoothed perplexity profile.
    - pds: PDS profile.
    - seq: Input DNA sequence.
    - p_window: Perplexity window length.
    - flank_size: Flank length for local context.
    - spacer_size: Spacer length for local context.

    Returns:
    - Region metrics dictionary.

    Computational complexity:
    - O(m) time and O(m) space for region span m.
    """
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

    di_seg = di[sl]
    mean_perplexity = _fmean(di_seg)
    min_perplexity = _fmin(di_seg)
    max_perplexity = _fmax(di_seg)

    pds_seg = pds[sl]
    pds_finite = pds_seg[np.isfinite(pds_seg)]
    pds_mean = float(np.mean(pds_finite)) if pds_finite.size else 0.0
    pds_max = float(np.max(pds_finite)) if pds_finite.size else 0.0
    persistence = float(np.mean(pds_finite > 0)) if pds_finite.size else 0.0
    variance = float(np.var(pds_finite)) if pds_finite.size > 1 else 0.0

    bg_up_s = max(0, start - spacer_size - flank_size)
    bg_up_e = max(0, start - spacer_size)
    bg_dn_s = min(n, end + spacer_size + 1)
    bg_dn_e = min(n, end + spacer_size + 1 + flank_size)

    bg_parts = []
    if bg_up_e > bg_up_s:
        bg_parts.append(smoothed_di[bg_up_s:bg_up_e])
    if bg_dn_e > bg_dn_s:
        bg_parts.append(smoothed_di[bg_dn_s:bg_dn_e])

    if bg_parts:
        bg_all = np.concatenate(bg_parts)
        bg_finite = bg_all[np.isfinite(bg_all)]
        background = float(np.mean(bg_finite)) if bg_finite.size else float("nan")
    else:
        background = float("nan")

    reg_seg = smoothed_di[max(0, start): end + 1]
    reg_mean = _fmean(reg_seg)
    reg_min = _fmin(reg_seg)
    prominence = max(0.0, background - reg_min) if np.isfinite(background) and np.isfinite(reg_min) else 0.0

    up_mean = _fmean(smoothed_di[bg_up_s:bg_up_e]) if bg_up_e > bg_up_s else float("nan")
    dn_mean = _fmean(smoothed_di[bg_dn_s:bg_dn_e]) if bg_dn_e > bg_dn_s else float("nan")

    start_nt = max(0, int(start))
    end_nt = min(len(seq) - 1, int(end + p_window - 1))
    sequence = seq[start_nt: end_nt + 1]
    length = end_nt - start_nt + 1
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    log_length = math.log(max(end - start + 1, 1))
    stability = 1.0 / (variance + EPSILON)
    region_score_raw = pds_mean * persistence * log_length * stability

    return {
        "Region_ID": f"LPR_{region_index:06d}",
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Start": int(start_nt),
        "End": int(end_nt),
        "Length": int(length),
        "MeanPerplexity": mean_perplexity,
        "MinPerplexity": min_perplexity,
        "MaxPerplexity": max_perplexity,
        "UpstreamMean": up_mean,
        "RegionMean": reg_mean,
        "DownstreamMean": dn_mean,
        "PDSMean": pds_mean,
        "PDSMax": pds_max,
        "Prominence": prominence,
        "Persistence": persistence,
        "Variance": variance,
        "GC%": gc,
        "MotifCount": 0,
        "Motifs": "",
        "Sequence": sequence,
        "RegionScore_raw": region_score_raw,
    }


def normalize_region_score(regions: list[dict]) -> list[dict]:
    """Purpose: Convert raw region scores to non-negative ranked values.

    Parameters:
    - regions: Region metric dictionaries.

    Returns:
    - Same list with RegionScore and Rank fields added.

    Computational complexity:
    - O(k log k) time and O(k) space for k regions.
    """
    if not regions:
        return regions

    raw = np.array([max(0.0, float(r.get("RegionScore_raw", 0.0))) for r in regions], dtype=np.float64)
    for i, region in enumerate(regions):
        region["RegionScore"] = float(raw[i])
        region.pop("RegionScore_raw", None)

    ranked = sorted(range(len(regions)), key=lambda idx: regions[idx]["RegionScore"], reverse=True)
    for rank, idx in enumerate(ranked, 1):
        regions[idx]["Rank"] = rank

    return regions


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    """Purpose: Run complete REGPLEX low-perplexity region workflow for one sequence.

    Parameters:
    - sequence_id: Sequence identifier.
    - seq: DNA sequence.
    - kwargs: Optional pipeline parameter overrides.

    Returns:
    - AnalysisResult containing signals, detected regions, and parameters.

    Computational complexity:
    - O(n) average signal processing plus O(k log k) interval merge/rank.
    """
    params = {
        "perplexity_window": int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        "sg_window_length": int(kwargs.get("sg_window_length", SG_WINDOW_LENGTH)),
        "sg_poly_order": int(kwargs.get("sg_poly_order", SG_POLY_ORDER)),
        "flank_size": int(kwargs.get("flank_size", FLANK_SIZE)),
        "spacer_size": int(kwargs.get("spacer_size", SPACER_SIZE)),
        "min_candidate": int(kwargs.get("min_candidate", MIN_CANDIDATE)),
        "max_candidate": int(kwargs.get("max_candidate", MAX_CANDIDATE)),
        "min_region_length": int(kwargs.get("min_region_length", MIN_REGION_LENGTH)),
        "max_region_length": int(kwargs.get("max_region_length", MAX_REGION_LENGTH)),
        "merge_gap": int(kwargs.get("merge_gap", MERGE_GAP)),
    }

    di = compute_di_perplexity(seq, params["perplexity_window"])
    smoothed_di = smooth_perplexity(di, params["sg_window_length"], params["sg_poly_order"])
    pds = compute_pds(
        smoothed_di,
        flank_size=params["flank_size"],
        spacer_size=params["spacer_size"],
        min_candidate=params["min_candidate"],
        max_candidate=params["max_candidate"],
    )

    cores = _find_all_pds_regions(pds, params["min_region_length"], params["max_region_length"])
    expanded = [_expand_region_pds(pds, cs, ce) for cs, ce in cores]
    merged = _merge_regions(expanded, params["merge_gap"])
    merged = [(s, e) for s, e in merged if e - s + 1 >= params["min_region_length"]]

    regions: list[dict] = []
    for idx, (s, e) in enumerate(merged, 1):
        region = _region_metrics(
            region_index=idx,
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
        region["Sequence_ID"] = sequence_id
        regions.append(region)

    regions.sort(key=lambda r: r["Signal_Start"])
    regions = normalize_region_score(regions)

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        di=di,
        smoothed_di=smoothed_di,
        pds=pds,
        regions=regions,
        params=params,
    )


# ==========================================================================
# 9. Export Utilities
# ==========================================================================
def regions_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    """Purpose: Convert sequence analysis results into one tabular dataframe.

    Parameters:
    - results: Iterable of AnalysisResult objects.

    Returns:
    - Combined pandas DataFrame of all detected regions.

    Computational complexity:
    - O(r) time and O(r) space for total region count r.
    """
    rows: list[dict] = []
    for result in results:
        rows.extend(result.regions)
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, fmt: str) -> bytes:
    """Purpose: Export tabular region results to standard serialized formats.

    Parameters:
    - df: Region dataframe.
    - fmt: One of csv/tsv/xlsx/json.

    Returns:
    - Encoded bytes for the requested format.

    Computational complexity:
    - O(r*c) time and O(r*c) space for r rows and c columns.
    """
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
    """Purpose: Export detected regions in BED interval format.

    Parameters:
    - df: Region dataframe.

    Returns:
    - BED content bytes.

    Computational complexity:
    - O(r) time and O(r) space.
    """
    bed = df[["Sequence_ID", "Start", "End", "Region_ID", "RegionScore"]].copy()
    bed["Start"] = bed["Start"].astype(int)
    bed["End"] = bed["End"].astype(int) + 1
    return bed.to_csv(index=False, sep="\t", header=False).encode()


def export_fasta(df: pd.DataFrame) -> bytes:
    """Purpose: Export detected region sequences as FASTA entries.

    Parameters:
    - df: Region dataframe.

    Returns:
    - FASTA content bytes.

    Computational complexity:
    - O(total_sequence_length) time and space.
    """
    lines: list[str] = []
    for _, row in df.iterrows():
        lines.append(
            f">{row['Region_ID']}|{row['Sequence_ID']}:{row['Start']}-{row['End']}"
            f"|RegionScore={row['RegionScore']:.4f}"
        )
        lines.append(str(row["Sequence"]))
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def export_gff(df: pd.DataFrame, gff3: bool = False) -> bytes:
    """Purpose: Export detected regions in GFF/GFF3 annotation format.

    Parameters:
    - df: Region dataframe.
    - gff3: Whether to include GFF3 header.

    Returns:
    - GFF or GFF3 content bytes.

    Computational complexity:
    - O(r) time and O(r) space.
    """
    rows: list[str] = []
    for _, row in df.iterrows():
        attr = (
            f"ID={row['Region_ID']};"
            f"RegionScore={row['RegionScore']:.4f};"
            f"PDSMean={row.get('PDSMean', 0.0):.4f};"
            f"Persistence={row['Persistence']:.4f};"
            f"Prominence={row.get('Prominence', 0.0):.4f};"
            f"Motifs={row.get('Motifs', '')}"
        )
        rows.append(
            "\t".join([
                str(row["Sequence_ID"]),
                "REGPLEX",
                "low_perplexity_region",
                str(int(row["Start"]) + 1),
                str(int(row["End"]) + 1),
                f"{float(row['RegionScore']):.4f}",
                ".",
                ".",
                attr,
            ])
        )
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()


def cli() -> None:
    """Purpose: Run command-line REGPLEX analysis and write CSV output.

    Parameters:
    - None (arguments parsed from command line).

    Returns:
    - None.

    Computational complexity:
    - O(total_input_length) dominated by per-sequence analysis.
    """
    parser = argparse.ArgumentParser(description="REGPLEX — low perplexity region detector")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_regions.csv", help="Output CSV path")
    parser.add_argument("--sg-window", type=int, default=SG_WINDOW_LENGTH, help="Savitzky-Golay window (odd)")
    parser.add_argument("--sg-order", type=int, default=SG_POLY_ORDER, help="Savitzky-Golay polynomial order")
    parser.add_argument("--flank-size", type=int, default=FLANK_SIZE, help="PDS flank window (bp)")
    parser.add_argument("--spacer-size", type=int, default=SPACER_SIZE, help="PDS spacer (bp)")
    parser.add_argument("--min-candidate", type=int, default=MIN_CANDIDATE, help="Min PDS candidate window (bp)")
    parser.add_argument("--max-candidate", type=int, default=MAX_CANDIDATE, help="Max PDS candidate window (bp)")
    parser.add_argument("--min-region", type=int, default=MIN_REGION_LENGTH, help="Min region length (bp)")
    parser.add_argument("--max-region", type=int, default=MAX_REGION_LENGTH, help="Max region length (bp)")
    parser.add_argument("--merge-gap", type=int, default=MERGE_GAP, help="Region merge gap (bp)")
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
            min_region_length=args.min_region,
            max_region_length=args.max_region,
            merge_gap=args.merge_gap,
        )
        for header, sequence in records
    ]
    df = regions_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} low-perplexity regions to {args.out}")


if __name__ == "__main__":
    cli()
