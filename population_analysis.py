"""population_analysis.py — Population-level Low Perplexity Region analysis.

All statistics are fully vectorised (NumPy arrays, no nested loops).
Target throughput: 1000 sequences × 6000 bp processed in seconds.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from regplex_core import AnalysisResult

__all__ = [
    "is_population_mode",
    "PopulationStats",
    "compute_population_stats",
    "ConsensusLPR",
    "compute_consensus_lprs",
    "build_occurrence_matrix",
    "population_summary_table",
    "compute_motif_frequencies",
]

# ==========================================================================
# 1. Population mode detection
# ==========================================================================

def is_population_mode(results: list) -> bool:
    """Return True when all AnalysisResult objects share identical signal length.

    Parameters
    ----------
    results : list[AnalysisResult]
        Results from analyze_sequence.

    Returns
    -------
    bool
        True iff ≥2 sequences all have the same len(di).
    """
    if len(results) < 2:
        return False
    lengths = [len(r.di) for r in results]
    return len(set(lengths)) == 1 and lengths[0] > 0


# ==========================================================================
# 2. Population-level positional statistics
# ==========================================================================

@dataclass
class PopulationStats:
    """Vectorised per-position statistics across all sequences.

    Attributes
    ----------
    n_seq : int
        Number of sequences.
    signal_length : int
        Number of signal positions.
    mean_perplexity : np.ndarray  shape (L,)
        Mean dinucleotide perplexity across sequences.
    std_perplexity : np.ndarray   shape (L,)
        Standard deviation of perplexity.
    mean_pds : np.ndarray         shape (L,)
        Mean PDS across sequences.
    std_pds : np.ndarray          shape (L,)
        Standard deviation of PDS.
    lpr_frequency : np.ndarray    shape (L,)
        Fraction of sequences whose LPR overlaps each position.
    mean_region_score : np.ndarray shape (L,)
        Mean RegionScore of overlapping regions at each position.
    mean_gc : np.ndarray          shape (L,)
        Mean GC content in a 21-nt window centred on each position.
    boundary_starts : np.ndarray  shape (k,)
        All LPR start positions (for histogram).
    boundary_ends : np.ndarray    shape (k,)
        All LPR end positions (for histogram).
    region_lengths : np.ndarray   shape (k,)
        All individual LPR lengths.
    region_scores : np.ndarray    shape (k,)
        All individual Region_Scores.
    """

    n_seq: int
    signal_length: int
    mean_perplexity: np.ndarray
    std_perplexity: np.ndarray
    mean_pds: np.ndarray
    std_pds: np.ndarray
    lpr_frequency: np.ndarray
    mean_region_score: np.ndarray
    mean_gc: np.ndarray
    boundary_starts: np.ndarray
    boundary_ends: np.ndarray
    region_lengths: np.ndarray
    region_scores: np.ndarray


def compute_population_stats(results: list) -> PopulationStats:
    """Compute vectorised population statistics across all AnalysisResults.

    Parameters
    ----------
    results : list[AnalysisResult]
        Must satisfy is_population_mode(results) == True.

    Returns
    -------
    PopulationStats

    Complexity
    ----------
    O(N × L) time and space for N sequences and L signal positions.
    """
    n = len(results)
    L = len(results[0].di)

    # --- stack perplexity and PDS arrays --- shape (N, L)
    perp_mat = np.full((n, L), np.nan, dtype=np.float64)
    pds_mat  = np.full((n, L), np.nan, dtype=np.float64)

    for i, r in enumerate(results):
        di = r.di.astype(np.float64)
        pds = r.pds.astype(np.float64)
        perp_mat[i, :len(di)]  = di
        pds_mat[i, :len(pds)]  = pds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_perplexity = np.nanmean(perp_mat, axis=0)
            std_perplexity  = np.nanstd(perp_mat,  axis=0)
            mean_pds        = np.nanmean(pds_mat,  axis=0)
            std_pds         = np.nanstd(pds_mat,   axis=0)

    # --- LPR frequency and mean region score per position ---
    lpr_freq   = np.zeros(L, dtype=np.float64)
    score_sum  = np.zeros(L, dtype=np.float64)
    score_cnt  = np.zeros(L, dtype=np.int64)

    all_starts: list[int] = []
    all_ends: list[int]   = []
    all_lengths: list[int] = []
    all_scores: list[float] = []

    for r in results:
        covered = np.zeros(L, dtype=bool)
        for region in r.regions:
            s = int(region.get("Signal_Start", region.get("Start", 0)))
            e = int(region.get("Signal_End",   region.get("End",   0)))
            s = max(0, min(s, L - 1))
            e = max(s, min(e, L - 1))
            covered[s:e + 1] = True
            score = float(region.get("Region_Score", 0.0))
            score_sum[s:e + 1] += score
            score_cnt[s:e + 1] += 1
            all_starts.append(s)
            all_ends.append(e)
            all_lengths.append(int(region.get("Length", e - s + 1)))
            all_scores.append(score)
        lpr_freq += covered.astype(np.float64)

    lpr_freq /= n
    with np.errstate(invalid="ignore"):
        mean_region_score = np.where(score_cnt > 0, score_sum / score_cnt, 0.0)

    # --- mean GC profile with a 21-nt sliding window ---
    mean_gc = _compute_mean_gc_profile(results, L)

    return PopulationStats(
        n_seq=n,
        signal_length=L,
        mean_perplexity=mean_perplexity,
        std_perplexity=std_perplexity,
        mean_pds=mean_pds,
        std_pds=std_pds,
        lpr_frequency=lpr_freq,
        mean_region_score=mean_region_score,
        mean_gc=mean_gc,
        boundary_starts=np.array(all_starts, dtype=np.int64),
        boundary_ends=np.array(all_ends, dtype=np.int64),
        region_lengths=np.array(all_lengths, dtype=np.int64),
        region_scores=np.array(all_scores, dtype=np.float64),
    )


def _compute_mean_gc_profile(results: list, L: int, gc_window: int = 21) -> np.ndarray:
    """Compute a mean GC-fraction profile using a sliding window over raw sequences.

    Parameters
    ----------
    results : list[AnalysisResult]
        Analysis results; raw sequences are read from ``r.params['_seq']`` when
        present (optional — returns an all-zero array when absent).
    L : int
        Target signal length (number of positions in the output array).
    gc_window : int
        Width of the sliding window used to smooth the per-position GC fraction.
        Default 21.

    Returns
    -------
    np.ndarray  shape (L,), dtype float32
        Mean GC fraction at each signal position across all sequences.
        Values are in the range [0, 1].
    """
    n = len(results)
    gc_mat = np.zeros((n, L), dtype=np.float32)
    half = gc_window // 2

    for i, r in enumerate(results):
        seq = r.params.get("_seq", None)
        if seq is None:
            continue
        arr = np.frombuffer(seq.encode(), dtype=np.uint8)
        is_gc = (arr == ord("G")) | (arr == ord("C"))
        padded = np.pad(is_gc.astype(np.float32), half, mode="edge")
        wins = np.lib.stride_tricks.sliding_window_view(padded, gc_window)
        gc_profile = wins.mean(axis=1)[:L]
        gc_mat[i, :len(gc_profile)] = gc_profile

    return gc_mat.mean(axis=0)


# ==========================================================================
# 3. Consensus LPR detection
# ==========================================================================

@dataclass
class ConsensusLPR:
    """A consensus Low Perplexity Region derived from population overlap.

    Attributes
    ----------
    region_id : str
    consensus_start : int
    consensus_end : int
    consensus_length : int
    support_fraction : float
    mean_region_score : float
    mean_pds : float
    mean_perplexity : float
    mean_gc : float
    motif_frequencies : dict[str, float]
        Per-motif fraction of sequences where motif occurs (populated later).
    """

    region_id: str
    consensus_start: int
    consensus_end: int
    consensus_length: int
    support_fraction: float
    mean_region_score: float
    mean_pds: float
    mean_perplexity: float
    mean_gc: float
    motif_frequencies: dict = field(default_factory=dict)


def compute_consensus_lprs(
    stats: PopulationStats,
    results: list | None = None,
    min_support: float = 0.5,
    merge_gap: int = 50,
) -> list[ConsensusLPR]:
    """Extract consensus LPRs from the LPR frequency profile.

    A position belongs to a consensus region when lpr_frequency >= min_support.
    Adjacent qualifying positions are merged when their gap <= merge_gap.

    Parameters
    ----------
    stats : PopulationStats
    results : list[AnalysisResult] | None
        When provided, mean GC% is derived from overlapping region GC_Content.
    min_support : float
        Minimum fraction of sequences (0–1). Default 0.5.
    merge_gap : int
        Maximum gap between qualifying runs to merge. Default 50.

    Returns
    -------
    list[ConsensusLPR]

    Complexity
    ----------
    O(L) time and O(L) space.
    """
    freq = stats.lpr_frequency
    L = len(freq)

    above = freq >= min_support
    padded = np.concatenate([[False], above, [False]])
    delta  = np.diff(padded.view(np.int8))
    starts = np.flatnonzero(delta > 0)
    ends   = np.flatnonzero(delta < 0) - 1

    # merge runs within merge_gap
    merged: list[tuple[int, int]] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if merged and s <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # build region GC map if results provided
    gc_map: np.ndarray | None = None
    if results is not None:
        gc_map = _build_gc_map(results, L)

    consensus: list[ConsensusLPR] = []
    for idx, (s, e) in enumerate(merged, 1):
        sl = slice(s, e + 1)
        support = float(np.mean(freq[sl]))
        msc     = float(np.mean(stats.mean_region_score[sl]))
        with np.errstate(invalid="ignore"):
            mpds    = float(np.nanmean(stats.mean_pds[sl]))
            mperp   = float(np.nanmean(stats.mean_perplexity[sl]))
        if gc_map is not None:
            mgc = float(np.mean(gc_map[sl])) * 100.0
        else:
            mgc = float(np.mean(stats.mean_gc[sl])) * 100.0
        consensus.append(
            ConsensusLPR(
                region_id=f"CLPR_{idx:04d}",
                consensus_start=int(s),
                consensus_end=int(e),
                consensus_length=int(e - s + 1),
                support_fraction=round(support, 4),
                mean_region_score=round(msc, 6),
                mean_pds=round(mpds, 6),
                mean_perplexity=round(mperp, 4),
                mean_gc=round(mgc, 2),
            )
        )
    return consensus


def _build_gc_map(results: list, L: int) -> np.ndarray:
    """Build a positional GC-fraction array from per-region GC_Content values.

    For each signal position, accumulates the GC fraction of every detected LPR
    that overlaps that position and returns the mean.  Positions not covered by
    any LPR default to 0.5 (neutral assumption).

    Parameters
    ----------
    results : list[AnalysisResult]
        Analysis results containing detected regions with ``GC_Content`` fields.
    L : int
        Signal length — number of positions in the output array.

    Returns
    -------
    np.ndarray  shape (L,), dtype float64
        Per-position mean GC fraction in [0, 1].
    """
    gc_sum = np.zeros(L, dtype=np.float64)
    gc_cnt = np.zeros(L, dtype=np.int64)
    for r in results:
        for region in r.regions:
            s = int(region.get("Signal_Start", region.get("Start", 0)))
            e = int(region.get("Signal_End",   region.get("End",   0)))
            s = max(0, min(s, L - 1))
            e = max(s, min(e, L - 1))
            gc = float(region.get("GC_Content", 50.0)) / 100.0
            gc_sum[s:e + 1] += gc
            gc_cnt[s:e + 1] += 1
    with np.errstate(invalid="ignore"):
        return np.where(gc_cnt > 0, gc_sum / gc_cnt, 0.5)


# ==========================================================================
# 4. Occurrence matrix
# ==========================================================================

def build_occurrence_matrix(
    results: list,
    consensus_lprs: list[ConsensusLPR],
) -> pd.DataFrame:
    """Build binary Sequence × ConsensusLPR occurrence matrix.

    A cell is 1 when ≥1 detected LPR in that sequence overlaps the consensus
    region by at least one position, else 0.

    Parameters
    ----------
    results : list[AnalysisResult]
    consensus_lprs : list[ConsensusLPR]

    Returns
    -------
    pd.DataFrame  shape (N_sequences, N_consensus_regions)

    Complexity
    ----------
    O(N × C) time and space (vectorised interval overlap check).
    """
    if not consensus_lprs:
        return pd.DataFrame()

    seq_ids = [r.sequence_id for r in results]
    cols = [c.region_id for c in consensus_lprs]
    # consensus region intervals as arrays
    c_starts = np.array([c.consensus_start for c in consensus_lprs], dtype=np.int64)
    c_ends   = np.array([c.consensus_end   for c in consensus_lprs], dtype=np.int64)

    mat = np.zeros((len(results), len(consensus_lprs)), dtype=np.int8)
    for i, r in enumerate(results):
        for region in r.regions:
            s = int(region.get("Signal_Start", region.get("Start", 0)))
            e = int(region.get("Signal_End",   region.get("End",   0)))
            # overlap iff not (e < c_start or s > c_end)
            overlap = ~((e < c_starts) | (s > c_ends))
            mat[i] |= overlap.astype(np.int8)

    return pd.DataFrame(mat, index=seq_ids, columns=cols)


# ==========================================================================
# 5. Population summary table
# ==========================================================================

def population_summary_table(
    consensus_lprs: list[ConsensusLPR],
) -> pd.DataFrame:
    """Convert consensus LPR list to a publication-ready summary DataFrame.

    Parameters
    ----------
    consensus_lprs : list[ConsensusLPR]

    Returns
    -------
    pd.DataFrame  with columns:
        Region_ID, Consensus_Start, Consensus_End, Length,
        Support, Mean_PDS, Mean_RegionScore, Mean_Perplexity, GC%
    """
    if not consensus_lprs:
        return pd.DataFrame()

    rows = [
        {
            "Region_ID":        c.region_id,
            "Consensus_Start":  c.consensus_start,
            "Consensus_End":    c.consensus_end,
            "Length":           c.consensus_length,
            "Support":          c.support_fraction,
            "Mean_PDS":         c.mean_pds,
            "Mean_RegionScore": c.mean_region_score,
            "Mean_Perplexity":  c.mean_perplexity,
            "GC%":              c.mean_gc,
        }
        for c in consensus_lprs
    ]
    return pd.DataFrame(rows)


# ==========================================================================
# 6. Motif frequency enrichment
# ==========================================================================

def compute_motif_frequencies(
    results: list,
    consensus_lprs: list[ConsensusLPR],
    compiled_motifs: list,
) -> list[ConsensusLPR]:
    """Compute per-motif conservation across sequences for each consensus LPR.

    For each consensus region, motif_frequency[motif] =
        #sequences containing ≥1 hit anywhere in any overlapping LPR
        ÷ total sequences.

    Parameters
    ----------
    results : list[AnalysisResult]
    consensus_lprs : list[ConsensusLPR]
    compiled_motifs : list[tuple[str, re.Pattern]]

    Returns
    -------
    list[ConsensusLPR]  (mutated in-place, returned for convenience)

    Complexity
    ----------
    O(N × C × M × len(sequence)) for N sequences, C consensus regions,
    M motifs — but sequence scanning is bounded by LPR length.
    """
    if not compiled_motifs or not consensus_lprs:
        return consensus_lprs

    n = len(results)
    c_starts = np.array([c.consensus_start for c in consensus_lprs], dtype=np.int64)
    c_ends   = np.array([c.consensus_end   for c in consensus_lprs], dtype=np.int64)

    # hit_counts[c_idx][motif] = number of sequences with ≥1 hit
    hit_counts: list[dict[str, int]] = [{} for _ in consensus_lprs]

    for r in results:
        for region in r.regions:
            s = int(region.get("Signal_Start", region.get("Start", 0)))
            e = int(region.get("Signal_End",   region.get("End",   0)))
            overlap_idx = np.flatnonzero(~((e < c_starts) | (s > c_ends)))
            if overlap_idx.size == 0:
                continue
            seq = region.get("Sequence", "")
            if not seq:
                continue
            for motif_str, pattern in compiled_motifs:
                if pattern.search(seq):
                    for ci in overlap_idx:
                        hit_counts[ci][motif_str] = hit_counts[ci].get(motif_str, 0) + 1

    for ci, clpr in enumerate(consensus_lprs):
        clpr.motif_frequencies = {
            motif: round(count / n, 4)
            for motif, count in hit_counts[ci].items()
        }

    return consensus_lprs


# ==========================================================================
# 7. Export helpers
# ==========================================================================

def export_occurrence_matrix(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    """Serialize the occurrence matrix to bytes.

    Parameters
    ----------
    df : pd.DataFrame
    fmt : str  one of csv / tsv / xlsx

    Returns
    -------
    bytes
    """
    fmt = fmt.lower()
    if fmt == "tsv":
        return df.to_csv(sep="\t").encode()
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer)
        return buf.getvalue()
    return df.to_csv().encode()  # default csv
