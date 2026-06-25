from __future__ import annotations

import argparse
import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Tunable scientific defaults
# ---------------------------
PERPLEXITY_WINDOW = 10
LANDSCAPE_WINDOW = 100
LANDSCAPE_METHOD = "mean"  # mean | median
UPSTREAM_WINDOW = 100
DOWNSTREAM_WINDOW = 100
SPACER = 50
ADAPTIVE_MIN_WINDOW = 50
ADAPTIVE_MAX_WINDOW = 300
MIN_DOMAIN = 50
MAX_DOMAIN = 1000
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
    landscape: np.ndarray
    lpc: np.ndarray
    upstream_mean: np.ndarray
    candidate_mean: np.ndarray
    downstream_mean: np.ndarray
    lpc_up: np.ndarray
    lpc_down: np.ndarray
    flank_mean: np.ndarray
    candidate_window: np.ndarray
    candidate_start: np.ndarray
    candidate_end: np.ndarray
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


def compute_landscape(p1: np.ndarray, window: int = LANDSCAPE_WINDOW, method: str = LANDSCAPE_METHOD) -> np.ndarray:
    if len(p1) == 0:
        return np.array([], dtype=np.float32)
    method = (method or "mean").lower()
    if method == "median":
        return _window_medians(p1.astype(np.float64), window).astype(np.float32)
    csum, _, ccount = _prefix(p1)
    positions = np.arange(len(p1), dtype=np.int64)
    starts = positions - (window // 2)
    ends = starts + window
    return _window_means_from_prefix(csum, ccount, starts, ends).astype(np.float32)


def _lpc_for_candidate_window(
    landscape: np.ndarray,
    upstream_window: int,
    spacer: int,
    candidate_window: int,
    downstream_window: int,
) -> dict[str, np.ndarray]:
    n = len(landscape)
    centers = np.arange(n, dtype=np.int64)

    candidate_start = centers - (candidate_window // 2)
    candidate_end = candidate_start + candidate_window
    upstream_end = candidate_start - spacer
    upstream_start = upstream_end - upstream_window
    downstream_start = candidate_end + spacer
    downstream_end = downstream_start + downstream_window

    csum, _, ccount = _prefix(landscape)
    upstream_mean = _window_means_from_prefix(csum, ccount, upstream_start, upstream_end)
    candidate_mean = _window_means_from_prefix(csum, ccount, candidate_start, candidate_end)
    downstream_mean = _window_means_from_prefix(csum, ccount, downstream_start, downstream_end)

    lpc_up = upstream_mean - candidate_mean
    lpc_down = downstream_mean - candidate_mean
    lpc = np.minimum(lpc_up, lpc_down)

    valid = (
        np.isfinite(upstream_mean)
        & np.isfinite(candidate_mean)
        & np.isfinite(downstream_mean)
        & (lpc_up > 0)
        & (lpc_down > 0)
        & np.isfinite(lpc)
    )

    flank_mean = (upstream_mean + downstream_mean) / 2.0
    for arr in (upstream_mean, candidate_mean, downstream_mean, lpc_up, lpc_down, lpc, flank_mean):
        arr[~valid] = np.nan

    candidate_start = candidate_start.astype(np.int32)
    candidate_end = (candidate_end - 1).astype(np.int32)
    candidate_start[~valid] = -1
    candidate_end[~valid] = -1
    candidate_window_track = np.zeros(n, dtype=np.int32)
    candidate_window_track[valid] = candidate_window

    return {
        "lpc": lpc.astype(np.float32),
        "upstream_mean": upstream_mean.astype(np.float32),
        "candidate_mean": candidate_mean.astype(np.float32),
        "downstream_mean": downstream_mean.astype(np.float32),
        "lpc_up": lpc_up.astype(np.float32),
        "lpc_down": lpc_down.astype(np.float32),
        "flank_mean": flank_mean.astype(np.float32),
        "candidate_window": candidate_window_track,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
    }


def compute_lpc_profile(
    landscape: np.ndarray,
    mode: str = "adaptive",
    upstream_window: int = UPSTREAM_WINDOW,
    spacer: int = SPACER,
    candidate_window: int = ADAPTIVE_MIN_WINDOW,
    downstream_window: int = DOWNSTREAM_WINDOW,
    adaptive_min_window: int = ADAPTIVE_MIN_WINDOW,
    adaptive_max_window: int = ADAPTIVE_MAX_WINDOW,
) -> dict[str, np.ndarray]:
    n = len(landscape)
    if mode == "fixed":
        return _lpc_for_candidate_window(landscape, upstream_window, spacer, candidate_window, downstream_window)

    min_window = max(1, int(adaptive_min_window))
    max_window = max(min_window, int(adaptive_max_window))

    best_profile = {
        "lpc": np.full(n, np.nan, dtype=np.float32),
        "upstream_mean": np.full(n, np.nan, dtype=np.float32),
        "candidate_mean": np.full(n, np.nan, dtype=np.float32),
        "downstream_mean": np.full(n, np.nan, dtype=np.float32),
        "lpc_up": np.full(n, np.nan, dtype=np.float32),
        "lpc_down": np.full(n, np.nan, dtype=np.float32),
        "flank_mean": np.full(n, np.nan, dtype=np.float32),
        "candidate_window": np.zeros(n, dtype=np.int32),
        "candidate_start": np.full(n, -1, dtype=np.int32),
        "candidate_end": np.full(n, -1, dtype=np.int32),
    }

    for current_window in range(min_window, max_window + 1):
        current = _lpc_for_candidate_window(landscape, upstream_window, spacer, current_window, downstream_window)
        better = np.isfinite(current["lpc"]) & (
            ~np.isfinite(best_profile["lpc"]) | (current["lpc"] > best_profile["lpc"])
        )
        if not np.any(better):
            continue
        for key in best_profile:
            best_profile[key][better] = current[key][better]

    return best_profile


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


def valley_statistics(
    idx: int,
    start: int,
    end: int,
    p1: np.ndarray,
    landscape: np.ndarray,
    profile: dict[str, np.ndarray],
    seq: str,
    p1_window: int,
) -> dict:
    signal_slice = slice(start, end + 1)
    domain_p1 = p1[signal_slice]
    domain_lpc = profile["lpc"][signal_slice]

    mean_p1, min_p1, max_p1, variance, sd, cv = _nan_stats(domain_p1)
    mean_lpc = float(np.nanmean(domain_lpc)) if np.isfinite(domain_lpc).any() else float("nan")
    max_lpc = float(np.nanmax(domain_lpc)) if np.isfinite(domain_lpc).any() else float("nan")

    upstream_mean = float(np.nanmean(profile["upstream_mean"][signal_slice]))
    candidate_mean = float(np.nanmean(profile["candidate_mean"][signal_slice]))
    downstream_mean = float(np.nanmean(profile["downstream_mean"][signal_slice]))
    lpc_up = float(np.nanmean(profile["lpc_up"][signal_slice]))
    lpc_down = float(np.nanmean(profile["lpc_down"][signal_slice]))

    symmetry = 1.0 - abs(upstream_mean - downstream_mean) / (upstream_mean + downstream_mean + EPSILON)
    symmetry = float(np.clip(symmetry, 0.0, 1.0)) if np.isfinite(symmetry) else float("nan")

    flank_local = profile["flank_mean"][signal_slice]
    finite_mask = np.isfinite(domain_p1) & np.isfinite(flank_local)
    if np.any(finite_mask):
        below = np.sum(domain_p1[finite_mask] < flank_local[finite_mask])
        persistence = float(below / np.sum(finite_mask))
        auc = float(np.nansum(np.maximum(flank_local[finite_mask] - domain_p1[finite_mask], 0.0)))
    else:
        persistence = 0.0
        auc = 0.0

    signal_length = end - start + 1
    start_nt, end_nt = _nucleotide_bounds(start, end, len(seq), p1_window)
    sequence = seq[start_nt:end_nt + 1]
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)

    lpc_strength = max(mean_lpc, 0.0) if np.isfinite(mean_lpc) else 0.0
    valley_score_raw = lpc_strength * persistence * math.log(max(signal_length, 2))

    candidate_windows = profile["candidate_window"][signal_slice]
    selected_window = int(np.nanmedian(candidate_windows[candidate_windows > 0])) if np.any(candidate_windows > 0) else 0

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
        "MeanLPC": mean_lpc,
        "MaximumLPC": max_lpc,
        "AreaUnderValley": auc,
        "UpstreamMean": upstream_mean,
        "CandidateMean": candidate_mean,
        "DownstreamMean": downstream_mean,
        "LPC_up": lpc_up,
        "LPC_down": lpc_down,
        "Persistence": persistence,
        "Symmetry": symmetry,
        "ValleyScore_raw": valley_score_raw,
        "GC%": gc,
        "CandidateWindow": selected_window,
        "Sequence": sequence,
    }


def normalize_valley_score(domains: list[dict]) -> list[dict]:
    if not domains:
        return domains
    raw = np.array([max(0.0, float(d.get("ValleyScore_raw", 0.0))) for d in domains], dtype=np.float64)
    lo = float(raw.min())
    hi = float(raw.max())
    span = hi - lo
    if span < EPSILON:
        norm = np.where(raw > 0, 1.0, 0.0)
    else:
        norm = (raw - lo) / span
    for idx, domain in enumerate(domains):
        domain["ValleyScore"] = float(raw[idx])
        domain["ValleyScoreNormalized"] = float(norm[idx])
        domain.pop("ValleyScore_raw", None)
    return domains


def find_domains(
    p1: np.ndarray,
    landscape: np.ndarray,
    profile: dict[str, np.ndarray],
    seq: str,
    min_domain: int = MIN_DOMAIN,
    max_domain: int = MAX_DOMAIN,
    p1_window: int = PERPLEXITY_WINDOW,
) -> list[dict]:
    signal = profile["lpc"].astype(np.float64)
    work = -signal  # bounded minimum-mean Kadane on inverted LPC
    nan_mask = ~np.isfinite(work)
    n = len(work)
    domains: list[dict] = []
    k = 1

    while True:
        best = (np.inf, None, None)
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
                s, e, mean_value = bounded_min_mean(seg, min_domain, max_domain)
                if s is not None and mean_value < best[0]:
                    best = (mean_value, i + s, i + e)
            i = j

        score, start, end = best
        if start is None or not np.isfinite(score) or score >= 0:
            break

        domains.append(
            valley_statistics(
                k,
                start,
                end,
                p1=p1,
                landscape=landscape,
                profile=profile,
                seq=seq,
                p1_window=p1_window,
            )
        )
        k += 1
        work[start:end + 1] = np.nan
        nan_mask[start:end + 1] = True

    domains.sort(key=lambda domain: domain["Signal_Start"])
    return normalize_valley_score(domains)


def analyze_sequence(sequence_id: str, seq: str, **kwargs) -> AnalysisResult:
    params = {
        "perplexity_window": int(kwargs.get("perplexity_window", PERPLEXITY_WINDOW)),
        "landscape_window": int(kwargs.get("landscape_window", LANDSCAPE_WINDOW)),
        "landscape_method": kwargs.get("landscape_method", LANDSCAPE_METHOD),
        "candidate_mode": kwargs.get("candidate_mode", "adaptive"),
        "candidate_window": int(kwargs.get("candidate_window", ADAPTIVE_MIN_WINDOW)),
        "upstream_window": int(kwargs.get("upstream_window", UPSTREAM_WINDOW)),
        "downstream_window": int(kwargs.get("downstream_window", DOWNSTREAM_WINDOW)),
        "spacer": int(kwargs.get("spacer", SPACER)),
        "adaptive_min_window": int(kwargs.get("adaptive_min_window", ADAPTIVE_MIN_WINDOW)),
        "adaptive_max_window": int(kwargs.get("adaptive_max_window", ADAPTIVE_MAX_WINDOW)),
        "min_domain": int(kwargs.get("min_domain", MIN_DOMAIN)),
        "max_domain": int(kwargs.get("max_domain", MAX_DOMAIN)),
    }

    p1 = compute_p1(seq, params["perplexity_window"])
    landscape = compute_landscape(p1, params["landscape_window"], params["landscape_method"])
    profile = compute_lpc_profile(
        landscape,
        mode=params["candidate_mode"],
        upstream_window=params["upstream_window"],
        spacer=params["spacer"],
        candidate_window=params["candidate_window"],
        downstream_window=params["downstream_window"],
        adaptive_min_window=params["adaptive_min_window"],
        adaptive_max_window=params["adaptive_max_window"],
    )
    domains = find_domains(
        p1,
        landscape,
        profile,
        seq,
        min_domain=params["min_domain"],
        max_domain=params["max_domain"],
        p1_window=params["perplexity_window"],
    )
    for domain in domains:
        domain["Sequence_ID"] = sequence_id

    return AnalysisResult(
        sequence_id=sequence_id,
        length=len(seq),
        p1=p1,
        landscape=landscape,
        lpc=profile["lpc"],
        upstream_mean=profile["upstream_mean"],
        candidate_mean=profile["candidate_mean"],
        downstream_mean=profile["downstream_mean"],
        lpc_up=profile["lpc_up"],
        lpc_down=profile["lpc_down"],
        flank_mean=profile["flank_mean"],
        candidate_window=profile["candidate_window"],
        candidate_start=profile["candidate_start"],
        candidate_end=profile["candidate_end"],
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
                "MeanLPC": domain["MeanLPC"],
                "MaximumLPC": domain["MaximumLPC"],
                "AreaUnderValley": domain["AreaUnderValley"],
                "UpstreamMean": domain["UpstreamMean"],
                "CandidateMean": domain["CandidateMean"],
                "DownstreamMean": domain["DownstreamMean"],
                "LPC_up": domain["LPC_up"],
                "LPC_down": domain["LPC_down"],
                "Persistence": domain["Persistence"],
                "Symmetry": domain["Symmetry"],
                "ValleyScore": domain["ValleyScore"],
                "ValleyScoreNormalized": domain["ValleyScoreNormalized"],
                "GC%": domain["GC%"],
                "MotifCount": domain.get("MotifCount", 0),
                "Motifs": domain.get("Motifs", ""),
                "Sequence": domain["Sequence"],
                "Signal_Start": domain["Signal_Start"],
                "Signal_End": domain["Signal_End"],
                "Signal_Length": domain["Signal_Length"],
                "CandidateWindow": domain["CandidateWindow"],
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
            f"ID={row['ID']};ValleyScore={row['ValleyScore']:.4f};"
            f"MeanP1={row['MeanP1']:.4f};MaximumLPC={row['MaximumLPC']:.4f};"
            f"Persistence={row['Persistence']:.4f};Symmetry={row['Symmetry']:.4f};"
            f"AreaUnderValley={row['AreaUnderValley']:.4f};Motifs={row['Motifs']}"
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
    parser = argparse.ArgumentParser(description="Run REGPLEX perplexity valley analysis from FASTA")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--out", default="regplex_valleys.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.fasta, encoding="utf-8") as handle:
        fasta_text = handle.read()

    records = parse_fasta(fasta_text)
    results = [analyze_sequence(header, sequence) for header, sequence in records]
    df = domains_dataframe(results)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} valleys to {args.out}")


if __name__ == "__main__":
    cli()
