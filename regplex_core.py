from __future__ import annotations

import argparse
import io
import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Tunable scientific defaults
# ---------------------------
PERPLEXITY_WINDOW = 10       # P1 k-mer window
BASE_SCALE = 100             # centre of auto-generated observation scales
LANDSCAPE_METHOD = "mean"    # mean | median
MIN_CANDIDATE = 50           # minimum candidate window (bp)
MAX_CANDIDATE = 1000         # maximum candidate window (bp)
MIN_DOMAIN = 50              # minimum reported valley length (bp)
MAX_DOMAIN = 1000            # Kadane maximum segment length (bp)
MERGE_GAP = 25               # merge valleys closer than this (bp)
BACKGROUND_FLANK = 200       # fallback flank size for background P1 estimate
EPSILON = 1e-9

_IUPAC_DNA = set("ACGTN")
_MAP = np.full(256, 4, dtype=np.uint8)
for base, idx in zip("ACGT", range(4)):
    _MAP[ord(base)] = idx


@dataclass
class AnalysisResult:
    sequence_id: str
    length: int
    p1: np.ndarray
    landscapes: dict           # {scale_bp: np.ndarray}
    lpc_profiles: dict         # {scale_bp: np.ndarray}  raw per-scale LPC
    consensus_lpc: np.ndarray  # normalised median across scales
    domains: list[dict]
    params: dict


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


def compute_p1(seq: str, window: int = PERPLEXITY_WINDOW) -> np.ndarray:
    if len(seq) < window:
        return np.array([], dtype=np.float32)
    arr = _MAP[np.frombuffer(seq.encode(), dtype=np.uint8)]
    has_n = arr == 4
    masked = np.where(has_n, 0, arr)
    windows = np.lib.stride_tricks.sliding_window_view(masked, window)
    windows_n = np.lib.stride_tricks.sliding_window_view(has_n, window)
    a, b = windows[:, :-1], windows[:, 1:]
    din = (a * 4 + b).astype(np.int16)
    counts = np.zeros((len(windows), 16), dtype=np.int16)
    np.add.at(counts, (np.arange(len(windows))[:, None], din), 1)
    p = np.where(counts == 0, 1.0, counts / (window - 1)).astype(np.float32)
    entropy = -np.sum(p * np.log2(p), axis=1)
    perplexity = (2.0 ** entropy).astype(np.float32)
    perplexity[windows_n.any(axis=1)] = np.nan
    return perplexity


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


def _window_means_from_prefix(csum: np.ndarray, ccount: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.full(starts.shape, np.nan, dtype=np.float64)
    limit = len(csum) - 1
    valid = (starts >= 0) & (ends <= limit) & (ends > starts)
    if not np.any(valid):
        return out
    s = starts[valid]
    e = ends[valid]
    counts = ccount[e] - ccount[s]
    positive = counts > 0
    if not np.any(positive):
        return out
    idx = np.flatnonzero(valid)[positive]
    out[idx] = (csum[e[positive]] - csum[s[positive]]) / counts[positive]
    return out


def _window_medians(arr: np.ndarray, window: int) -> np.ndarray:
    """Return center-aligned sliding medians with NaN-padded edge positions."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or window < 1 or n < window:
        return out
    views = np.lib.stride_tricks.sliding_window_view(arr, window)
    med = np.nanmedian(views, axis=1)
    offset = window // 2
    out[offset:offset + len(med)] = med
    return out


def _scales_from_base(base: int) -> list[int]:
    """Return 5 observation scales centred on *base*."""
    b = max(20, int(base))
    seen: set[int] = set()
    result: list[int] = []
    for factor in (0.25, 0.5, 1, 2, 4):
        v = max(10, int(round(b * factor)))
        if v not in seen:
            seen.add(v)
            result.append(v)
    return sorted(result)


def build_landscapes(
    p1: np.ndarray,
    scales: list[int],
    method: str = LANDSCAPE_METHOD,
) -> dict:
    """Compute rolling-mean/median landscapes at every scale from P1."""
    if len(p1) == 0:
        return {s: np.array([], dtype=np.float32) for s in scales}
    method = (method or "mean").lower()
    landscapes: dict[int, np.ndarray] = {}
    if method == "median":
        for s in scales:
            landscapes[s] = _window_medians(
                p1.astype(np.float64), s
            ).astype(np.float32)
    else:
        # Share one prefix-sum pass across all scales.
        csum, _, ccount = _prefix(p1)
        positions = np.arange(len(p1), dtype=np.int64)
        for s in scales:
            starts = positions - (s // 2)
            ends = starts + s
            landscapes[s] = _window_means_from_prefix(
                csum, ccount, starts, ends
            ).astype(np.float32)
    return landscapes


def _compute_scale_lpc(
    landscape: np.ndarray,
    scale: int,
    min_candidate: int = MIN_CANDIDATE,
    max_candidate: int = MAX_CANDIDATE,
) -> np.ndarray:
    """Three-window LPC at one observation scale.

    upstream = scale, spacer = scale//2,
    candidate = clamp(scale, min_candidate, max_candidate),
    downstream = scale.

    LPC = min(upstream_mean − candidate_mean,
              downstream_mean − candidate_mean).
    Positions where either flank does not exceed the candidate → NaN.
    """
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
    """Per-profile robust z-score: (x − median) / (MAD × 1.4826)."""
    out = np.full(len(arr), np.nan, dtype=np.float32)
    idx = np.isfinite(arr)
    vals = arr[idx].astype(np.float64)
    if vals.size < 2:
        # A single value equals the median by definition; its z-score is 0.
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


def build_consensus_lpc(lpc_profiles: dict) -> np.ndarray:
    """Normalize each scale's LPC independently, then take the nanmedian."""
    profiles = list(lpc_profiles.values())
    if not profiles:
        return np.array([], dtype=np.float32)
    n = len(profiles[0])
    stacked = np.full((len(profiles), n), np.nan, dtype=np.float64)
    for i, lpc in enumerate(profiles):
        stacked[i, : len(lpc)] = _robust_normalize(lpc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stacked, axis=0).astype(np.float32)


def bounded_min_mean(arr: np.ndarray, min_len: int, max_len: int) -> tuple[Optional[int], Optional[int], float]:
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


def _nucleotide_bounds(signal_start: int, signal_end: int, seq_len: int, p1_window: int) -> tuple[int, int]:
    start = max(0, int(signal_start))
    end = min(seq_len - 1, int(signal_end + p1_window - 1))
    return start, end


def _nan_stats(arr: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return mean/min/max/variance/sd/cv for finite values; all-NaN if empty."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    var = float(np.var(finite))
    sd = float(np.sqrt(var))
    cv = float(sd / (mean + EPSILON)) if np.isfinite(sd) and np.isfinite(mean) else float("nan")
    return mean, min_v, max_v, var, sd, cv


def _expand_valley(
    consensus_lpc: np.ndarray, core_start: int, core_end: int
) -> tuple[int, int]:
    """Grow a valley core outward while ConsensusLPC > 0."""
    n = len(consensus_lpc)
    left = core_start
    while (
        left > 0
        and np.isfinite(consensus_lpc[left - 1])
        and consensus_lpc[left - 1] > 0
    ):
        left -= 1
    right = core_end
    while (
        right < n - 1
        and np.isfinite(consensus_lpc[right + 1])
        and consensus_lpc[right + 1] > 0
    ):
        right += 1
    return left, right


def _merge_valleys(
    intervals: list[tuple[int, int]], gap: int
) -> list[tuple[int, int]]:
    """Merge intervals whose start-to-end gap is ≤ *gap* bp."""
    if not intervals:
        return []
    merged: list[tuple[int, int]] = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e + gap:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


def _scale_support(
    lpc_profiles: dict, start: int, end: int
) -> tuple[float, int]:
    """Fraction of scales whose mean raw LPC in [start, end] > 0."""
    n_scales = len(lpc_profiles)
    if n_scales == 0:
        return 0.0, 0
    supporting = 0
    for lpc in lpc_profiles.values():
        seg = lpc[start: end + 1]
        finite = seg[np.isfinite(seg)]
        if finite.size > 0 and float(np.mean(finite)) > 0:
            supporting += 1
    return float(supporting) / n_scales, n_scales


def valley_statistics(
    valley_index: int,
    start: int,
    end: int,
    p1: np.ndarray,
    lpc_profiles: dict,
    consensus_lpc: np.ndarray,
    seq: str,
    p1_window: int,
    scales: list[int],
) -> dict:
    """Compute all quality metrics for one valley."""
    sl = slice(start, end + 1)
    domain_p1 = p1[sl]
    domain_lpc = consensus_lpc[sl]

    mean_p1, min_p1, max_p1, variance, sd, cv = _nan_stats(domain_p1)

    finite_lpc = domain_lpc[np.isfinite(domain_lpc)]
    mean_lpc = float(np.mean(finite_lpc)) if finite_lpc.size > 0 else float("nan")
    max_lpc = float(np.max(finite_lpc)) if finite_lpc.size > 0 else float("nan")
    auc = float(np.sum(np.maximum(finite_lpc, 0.0)))

    # Persistence: fraction of positions with ConsensusLPC > 0
    persistence = (
        float(np.mean(finite_lpc > 0)) if finite_lpc.size > 0 else 0.0
    )

    # Scale support
    scale_support, n_scales = _scale_support(lpc_profiles, start, end)

    # Background: mean P1 in flanking regions
    flank = max(scales) if scales else BACKGROUND_FLANK
    bg = np.concatenate([
        p1[max(0, start - flank): start],
        p1[end + 1: min(len(p1), end + 1 + flank)],
    ])
    bg_finite = bg[np.isfinite(bg)]
    mean_background = (
        float(np.mean(bg_finite)) if bg_finite.size > 0 else float("nan")
    )

    signal_length = end - start + 1
    start_nt, end_nt = _nucleotide_bounds(start, end, len(seq), p1_window)
    sequence = seq[start_nt: end_nt + 1]
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    safe_lpc = max(mean_lpc, 0.0) if np.isfinite(mean_lpc) else 0.0
    safe_var = float(variance) if np.isfinite(variance) else 0.0
    valley_score_raw = (
        safe_lpc
        * persistence
        * scale_support
        * math.log(max(signal_length, 2))
        * (1.0 / (safe_var + EPSILON))
    )

    return {
        "ID": f"PV_{idx:06d}",
        "Signal_Start": int(start),
        "Signal_End": int(end),
        "Signal_Length": int(signal_length),
        "Start": int(start_nt),
        "End": int(end_nt),
        "Length": int(end_nt - start_nt + 1),
        "MeanP1": mean_p1,
        "MinimumP1": min_p1,
        "MaximumP1": max_p1,
        "Variance": variance,
        "SD": sd,
        "CV": cv,
        "MeanBackground": mean_background,
        "MeanLPC": mean_lpc,
        "MaximumLPC": max_lpc,
        "AreaUnderValley": auc,
        "Persistence": persistence,
        "ScaleSupport": scale_support,
        "NScales": n_scales,
        "ValleyScore_raw": valley_score_raw,
        "GC%": gc,
        "Sequence": sequence,
    }


def normalize_valley_score(domains: list[dict]) -> list[dict]:
    if not domains:
        return domains
    raw = np.array(
        [max(0.0, float(d.get("ValleyScore_raw", 0.0))) for d in domains],
        dtype=np.float64,
    )
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    norm = np.where(raw > 0, 1.0, 0.0) if span < EPSILON else (raw - lo) / span
    for i, domain in enumerate(domains):
        domain["ValleyScore"] = float(raw[i])
        domain["ValleyScoreNormalized"] = float(norm[i])
        domain.pop("ValleyScore_raw", None)
    return domains


def find_domains(
    p1: np.ndarray,
    lpc_profiles: dict,
    consensus_lpc: np.ndarray,
    seq: str,
    scales: list[int],
    min_domain: int = MIN_DOMAIN,
    max_domain: int = MAX_DOMAIN,
    merge_gap: int = MERGE_GAP,
    p1_window: int = PERPLEXITY_WINDOW,
) -> list[dict]:
    """Detect perplexity valleys: Kadane → expand → merge → statistics."""
    work = -consensus_lpc.copy().astype(np.float64)
    nan_mask = ~np.isfinite(work)
    n = len(work)
    cores: list[tuple[int, int]] = []

    while True:
        best: tuple[float, Optional[int], Optional[int]] = (np.inf, None, None)
        i = 0
        while i < n:
            if nan_mask[i]:
                i += 1
                continue
            j = i
            while j < n and not nan_mask[j]:
                j += 1
            seg = work[i:j]
            if len(seg) >= min_domain:
                s, e, mv = bounded_min_mean(seg, min_domain, max_domain)
                if s is not None and mv < best[0]:
                    best = (mv, i + s, i + e)
            i = j

        score, cs, ce = best
        if cs is None or not np.isfinite(score) or score >= 0:
            break

        # Enforce minimum core length with bounds safety.
        # Clamp ce first, then pull cs back if still too short.
        if ce - cs + 1 < min_domain:
            mid = (cs + ce) // 2
            cs = max(0, mid - min_domain // 2)
            ce = min(n - 1, cs + min_domain - 1)
            cs = max(0, ce - min_domain + 1)  # re-clamp if ce hit the boundary

        exp_s, exp_e = _expand_valley(consensus_lpc, cs, ce)
        cores.append((exp_s, exp_e))
        work[cs: ce + 1] = np.nan
        nan_mask[cs: ce + 1] = True

    if not cores:
        return []

    merged = [
        (s, e)
        for s, e in _merge_valleys(cores, merge_gap)
        if e - s + 1 >= min_domain
    ]
    if not merged:
        return []

    domains: list[dict] = []
    for idx, (start, end) in enumerate(merged, 1):
        domains.append(
            valley_statistics(
                idx, start, end,
                p1, lpc_profiles, consensus_lpc,
                seq, p1_window, scales,
            )
        )
    domains.sort(key=lambda d: d["Signal_Start"])
    return normalize_valley_score(domains)


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    params = {
        "perplexity_window": int(
            kwargs.get("perplexity_window", PERPLEXITY_WINDOW)
        ),
        "base_scale": int(kwargs.get("base_scale", BASE_SCALE)),
        "scales": kwargs.get("scales", None),
        "landscape_method": kwargs.get("landscape_method", LANDSCAPE_METHOD),
        "min_candidate": int(kwargs.get("min_candidate", MIN_CANDIDATE)),
        "max_candidate": int(kwargs.get("max_candidate", MAX_CANDIDATE)),
        "min_domain": int(kwargs.get("min_domain", MIN_DOMAIN)),
        "max_domain": int(kwargs.get("max_domain", MAX_DOMAIN)),
        "merge_gap": int(kwargs.get("merge_gap", MERGE_GAP)),
    }

    if params["scales"]:
        scales = sorted(set(max(10, int(s)) for s in params["scales"]))
    else:
        scales = _scales_from_base(params["base_scale"])
    params["resolved_scales"] = scales

    # Step 1 — P1 computed exactly once
    p1 = compute_p1(seq, params["perplexity_window"])

    # Step 2 — Multi-scale landscapes from the same P1
    landscapes = build_landscapes(p1, scales, params["landscape_method"])

    # Steps 3–4 — Three-window LPC at each scale independently
    lpc_profiles: dict[int, np.ndarray] = {}
    for s in scales:
        lpc_profiles[s] = _compute_scale_lpc(
            landscapes[s], s,
            params["min_candidate"],
            params["max_candidate"],
        )

    # Step 5 — Consensus LPC
    consensus_lpc = build_consensus_lpc(lpc_profiles)

    # Steps 6–8 — Kadane, expansion, merging
    domains = find_domains(
        p1, lpc_profiles, consensus_lpc, seq, scales,
        min_domain=params["min_domain"],
        max_domain=params["max_domain"],
        merge_gap=params["merge_gap"],
        p1_window=params["perplexity_window"],
    )
    for domain in domains:
        domain["Sequence_ID"] = sequence_id

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        p1=p1,
        landscapes=landscapes,
        lpc_profiles=lpc_profiles,
        consensus_lpc=consensus_lpc,
        domains=domains,
        params=params,
    )


def domains_dataframe(results: Iterable[AnalysisResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for result in results:
        for domain in result.domains:
            rows.append({
                "ID": domain["ID"],
                "Sequence_ID": domain["Sequence_ID"],
                "Start": domain["Start"],
                "End": domain["End"],
                "Length": domain["Length"],
                "MeanP1": domain["MeanP1"],
                "MinimumP1": domain["MinimumP1"],
                "MaximumP1": domain["MaximumP1"],
                "Variance": domain["Variance"],
                "SD": domain["SD"],
                "CV": domain["CV"],
                "MeanBackground": domain["MeanBackground"],
                "MeanLPC": domain["MeanLPC"],
                "MaximumLPC": domain["MaximumLPC"],
                "AreaUnderValley": domain["AreaUnderValley"],
                "Persistence": domain["Persistence"],
                "ScaleSupport": domain["ScaleSupport"],
                "NScales": domain["NScales"],
                "ValleyScore": domain["ValleyScore"],
                "ValleyScoreNormalized": domain["ValleyScoreNormalized"],
                "GC%": domain["GC%"],
                "MotifCount": domain.get("MotifCount", 0),
                "Motifs": domain.get("Motifs", ""),
                "Sequence": domain["Sequence"],
                "Signal_Start": domain["Signal_Start"],
                "Signal_End": domain["Signal_End"],
                "Signal_Length": domain["Signal_Length"],
            })
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
            f"MeanP1={row['MeanP1']:.4f};"
            f"MaximumLPC={row['MaximumLPC']:.4f};"
            f"Persistence={row['Persistence']:.4f};"
            f"ScaleSupport={row['ScaleSupport']:.4f};"
            f"AreaUnderValley={row['AreaUnderValley']:.4f};"
            f"Motifs={row['Motifs']}"
        )
        rows.append("\t".join([
            str(row["Sequence_ID"]),
            "REGPLEX",
            "perplexity_valley",
            str(int(row["Start"]) + 1),
            str(int(row["End"]) + 1),
            f"{float(row['ValleyScore']):.4f}",
            ".",
            ".",
            attr,
        ]))
    prefix = "##gff-version 3\n" if gff3 else ""
    return (prefix + "\n".join(rows) + ("\n" if rows else "")).encode()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="REGPLEX v9 — multi-scale perplexity valley analysis"
    )
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument(
        "--out", default="regplex_valleys.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--base-scale", type=int, default=BASE_SCALE,
        help="Base observation scale (bp); auto-generates 5 scales",
    )
    parser.add_argument(
        "--landscape-method", default=LANDSCAPE_METHOD,
        choices=["mean", "median"],
        help="Landscape smoothing method",
    )
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        fasta_text = handle.read()

    records = parse_fasta(fasta_text)
    results = [
        analyze_sequence(
            header, sequence,
            base_scale=args.base_scale,
            landscape_method=args.landscape_method,
        )
        for header, sequence in records
    ]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} valleys to {args.out}")


if __name__ == "__main__":
    cli()
